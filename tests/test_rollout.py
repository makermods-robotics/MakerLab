# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for makermodslab.rollout — request schema, pure helpers, and the
non-subprocess branches of the start/stop/status handlers.

handle_start_inference's happy path spawns a real subprocess and a stdout-
pumping thread; covering it would require mocking subprocess.Popen, threading,
and setup_follower_calibration_file. We test only the early-return mutex
branches here — the parts that matter for safety."""

from __future__ import annotations

import io
import threading

import pytest

from makermodslab.eval_protocol import (
    CMD_EPISODE,
    CMD_QUIT,
    CMD_STOP,
    REASON_DURATION,
    REASON_STOPPED,
)


@pytest.fixture(autouse=True)
def _reset_rollout_globals(monkeypatch: pytest.MonkeyPatch):
    """Reset rollout's module-level state around each test so a leaking
    `inference_active=True` from one case can't poison the next."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "inference_active", False)
    monkeypatch.setattr(rollout, "_inference_proc", None)
    monkeypatch.setattr(rollout, "_inference_started_at", None)
    monkeypatch.setattr(rollout, "_inference_rollout_started_at", None)
    monkeypatch.setattr(rollout, "_inference_meta", {})
    monkeypatch.setattr(rollout, "_inference_cancel", None)
    monkeypatch.setattr(rollout, "_last_result", None)
    monkeypatch.setattr(rollout, "_inference_startup_thread", None)
    monkeypatch.setattr(rollout, "_eval_session", None)


class _SyncThread:
    """A ``threading.Thread`` stand-in whose ``.start()`` runs the target inline.

    The start handler now hands the heavy work (download → preflight → spawn) to
    a background ``threading.Thread``; patching it with this lets a test drive
    that worker — and the stdout-pump thread it in turn spawns — deterministically
    in the calling thread, no real threads or sleeps. Only the keyword call shape
    the code uses (``Thread(target=..., args=..., name=..., daemon=...)``) is
    supported."""

    def __init__(self, target=None, args=(), kwargs=None, name=None, daemon=None) -> None:
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self) -> None:
        if self._target is not None:
            self._target(*self._args, **self._kwargs)


class _EmptyStdout:
    """A subprocess ``stdout`` that is immediately at EOF, so the stdout pump's
    ``iter(proc.stdout.readline, b"")`` loop exits at once when a test runs it
    synchronously."""

    def readline(self) -> bytes:
        return b""


def test_inference_request_rejects_missing_required_fields() -> None:
    from pydantic import ValidationError

    from makermodslab.rollout import InferenceRequest

    with pytest.raises(ValidationError):
        InferenceRequest()


def test_inference_request_has_expected_defaults() -> None:
    from makermodslab.rollout import InferenceRequest

    req = InferenceRequest(
        follower_port="/dev/ttyUSB0",
        follower_config="robot_a",
        policy_ref="user/repo@checkpoints/000050",
    )
    assert req.task == ""
    assert req.cameras == {}
    assert req.duration_s == 60


def test_inference_request_bimanual_fields_default_to_single() -> None:
    """A request that omits the bimanual block is single-arm — the right-arm
    fields are inert and `mode` defaults to 'single'."""
    from makermodslab.rollout import InferenceRequest

    req = InferenceRequest(
        follower_port="/dev/ttyUSB0",
        follower_config="robot_a",
        policy_ref="user/repo@checkpoints/000050",
    )
    assert req.mode == "single"
    assert req.right_follower_port == ""
    assert req.right_follower_config == ""
    assert req.robot_name == ""
    assert req.checkpoint_state_dim is None


def test_inference_request_accepts_bimanual_block() -> None:
    from makermodslab.rollout import InferenceRequest

    req = InferenceRequest(
        follower_port="/dev/left",
        follower_config="left_cal",
        policy_ref="user/repo@checkpoints/000050",
        mode="bimanual",
        right_follower_port="/dev/right",
        right_follower_config="right_cal",
        robot_name="dual_arm",
        checkpoint_state_dim=12,
    )
    assert req.mode == "bimanual"
    assert req.right_follower_port == "/dev/right"
    assert req.right_follower_config == "right_cal"
    assert req.robot_name == "dual_arm"
    assert req.checkpoint_state_dim == 12


def test_inference_request_defaults_to_sync_engine() -> None:
    """Absent an explicit choice the request pins lerobot's own default, so
    adding the A/B knob can't change what an existing caller gets."""
    from makermodslab.rollout import InferenceRequest

    req = InferenceRequest(
        follower_port="/dev/ttyUSB0",
        follower_config="robot_a",
        policy_ref="user/repo@checkpoints/000050",
    )
    assert req.inference_engine == "sync"


def test_inference_request_rejects_unknown_engine() -> None:
    """The field is a Literal, so a typo is a 422 at the API edge rather than a
    draccus parse crash inside the rollout subprocess."""
    from pydantic import ValidationError

    from makermodslab.rollout import InferenceRequest

    with pytest.raises(ValidationError):
        InferenceRequest(
            follower_port="/dev/ttyUSB0",
            follower_config="robot_a",
            policy_ref="user/repo@checkpoints/000050",
            inference_engine="async",
        )


# ---------------------------------------------------------------------------
# _arm_count_mismatch — the pre-spawn checkpoint/robot arm-count guard
# ---------------------------------------------------------------------------


def test_arm_count_mismatch_none_when_state_dim_unknown() -> None:
    """A checkpoint with no observation.state (state_dim None) can't be judged
    cheaply — defer to the subprocess's own shape check."""
    from makermodslab.rollout import _arm_count_mismatch

    assert _arm_count_mismatch("single", None) is None
    assert _arm_count_mismatch("bimanual", None) is None


def test_arm_count_mismatch_none_when_single_matches_single() -> None:
    from makermodslab.rollout import _arm_count_mismatch

    assert _arm_count_mismatch("single", 6) is None


def test_arm_count_mismatch_none_when_bimanual_matches_bimanual() -> None:
    from makermodslab.rollout import _arm_count_mismatch

    assert _arm_count_mismatch("bimanual", 12) is None


def test_arm_count_mismatch_flags_bimanual_checkpoint_on_single_robot() -> None:
    from makermodslab.rollout import _arm_count_mismatch

    msg = _arm_count_mismatch("single", 12)
    assert msg is not None
    assert "bimanual" in msg
    assert "single-arm" in msg


def test_arm_count_mismatch_flags_single_checkpoint_on_bimanual_robot() -> None:
    from makermodslab.rollout import _arm_count_mismatch

    msg = _arm_count_mismatch("bimanual", 6)
    assert msg is not None
    assert "single-arm" in msg
    assert "bimanual" in msg


def test_arm_count_mismatch_none_for_unrecognised_width() -> None:
    """A width that's neither a single arm nor a clean multiple is left to the
    subprocess rather than guessed at (e.g. 7 = 6 + an extra sensor dim)."""
    from makermodslab.rollout import _arm_count_mismatch

    assert _arm_count_mismatch("single", 7) is None
    assert _arm_count_mismatch("bimanual", 7) is None


def test_detect_device_returns_cpu_when_neither_cuda_nor_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from makermodslab.rollout import _detect_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert _detect_device() == "cpu"


def test_detect_device_prefers_cuda_over_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from makermodslab.rollout import _detect_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert _detect_device() == "cuda"


def test_detect_device_falls_back_to_mps_when_no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from makermodslab.rollout import _detect_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert _detect_device() == "mps"


def test_detect_device_returns_cpu_when_torch_probe_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """The function wraps both probes in a broad try/except — if torch is
    broken at runtime we still need a sensible fallback."""
    import torch

    from makermodslab.rollout import _detect_device

    def _boom() -> bool:
        raise RuntimeError("simulated torch.cuda failure")

    monkeypatch.setattr(torch.cuda, "is_available", _boom)
    assert _detect_device() == "cpu"


def test_resolve_policy_path_returns_local_dir_unchanged(tmp_path) -> None:
    from makermodslab.rollout import _resolve_policy_path

    pretrained = tmp_path / "pretrained_model"
    pretrained.mkdir()
    assert _resolve_policy_path(str(pretrained)) == str(pretrained)


def test_resolve_policy_path_raises_on_unparsable_ref() -> None:
    from makermodslab.rollout import _resolve_policy_path

    with pytest.raises(ValueError, match="Unrecognised policy ref"):
        _resolve_policy_path("not-a-real-ref-no-at-sign")


def test_resolve_policy_path_resolves_hub_ref(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Hub refs ('user/repo@checkpoints/000050') must be passed through
    snapshot_download and joined to the standard checkpoints/<step>/pretrained_model
    layout."""
    from makermodslab.rollout import _resolve_policy_path

    fake_root = tmp_path / "snapshot"
    fake_root.mkdir()
    seen_kwargs: dict = {}

    def fake_snapshot_download(**kwargs):
        seen_kwargs.update(kwargs)
        return str(fake_root)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    result = _resolve_policy_path("user/my-repo@checkpoints/000050")

    assert seen_kwargs["repo_id"] == "user/my-repo"
    assert seen_kwargs["repo_type"] == "model"
    assert seen_kwargs["allow_patterns"] == ["checkpoints/000050/pretrained_model/*"]
    assert result == str(fake_root / "checkpoints" / "000050" / "pretrained_model")


def test_resolve_policy_path_resolves_hub_root_ref(monkeypatch, tmp_path) -> None:
    """A flat-model ref ('user/repo@root') downloads the repo root and returns
    it — but excludes the checkpoints/ and training_state/ sub-trees (neither is
    needed to run inference, both can be multi-GB) so only the root pretrained
    files are pulled."""
    from makermodslab.rollout import _resolve_policy_path

    fake_root = tmp_path / "snapshot"
    fake_root.mkdir()
    seen = {}

    def fake_snapshot_download(**kwargs):
        seen.update(kwargs)
        return str(fake_root)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    result = _resolve_policy_path("user/repo@root")
    assert seen["repo_id"] == "user/repo"
    # Byte-scoping: no allow_patterns (the whole root IS the model), but the
    # heavy sibling sub-trees are ignored.
    assert "allow_patterns" not in seen
    assert seen["ignore_patterns"] == ["checkpoints/**", "training_state/**"]
    assert result == str(fake_root)


def test_format_cameras_arg_empty_yields_empty_braces() -> None:
    from makermodslab.rollout import _format_cameras_arg

    assert _format_cameras_arg({}) == "{}"


def test_format_cameras_arg_renames_camera_index_to_index_or_path() -> None:
    """lerobot's CLI expects `index_or_path`, but the frontend posts
    `camera_index`. The rename is the whole point of this helper."""
    from makermodslab.rollout import _format_cameras_arg

    result = _format_cameras_arg(
        {"front": {"type": "opencv", "camera_index": 0, "width": 640, "height": 480, "fps": 30}}
    )
    assert "index_or_path: 0" in result
    assert "camera_index" not in result
    assert result.startswith("{front: {")
    assert result.endswith("}}")


def test_format_cameras_arg_omits_none_values() -> None:
    from makermodslab.rollout import _format_cameras_arg

    result = _format_cameras_arg({"front": {"camera_index": 0, "fps": None}})
    assert "fps" not in result
    assert "index_or_path: 0" in result


def test_format_cameras_arg_handles_multiple_cameras() -> None:
    from makermodslab.rollout import _format_cameras_arg

    result = _format_cameras_arg(
        {
            "front": {"camera_index": 0, "fps": 30},
            "wrist": {"camera_index": 1, "fps": 30},
        }
    )
    assert "front: {" in result
    assert "wrist: {" in result


def test_handle_stop_inference_when_idle_returns_409() -> None:
    from makermodslab.rollout import handle_stop_inference

    result = handle_stop_inference()
    assert result["success"] is False
    assert result["status_code"] == 409


def test_handle_inference_status_when_idle_returns_dict_with_expected_keys() -> None:
    from makermodslab.rollout import handle_inference_status

    result = handle_inference_status()
    assert isinstance(result, dict)
    assert result["inference_active"] is False
    assert result["phase"] is None
    for key in ("started_at", "rollout_started_at", "elapsed_s", "rollout_elapsed_s"):
        assert key in result


def _stub_request():
    from makermodslab.rollout import InferenceRequest

    return InferenceRequest(
        follower_port="/dev/ttyUSB0",
        follower_config="robot_a",
        policy_ref="user/repo@checkpoints/000050",
    )


def test_handle_start_inference_blocked_when_teleoperation_active(monkeypatch) -> None:
    """If teleop owns the bus, inference must refuse rather than race for
    the serial port."""
    from makermodslab.rollout import handle_start_inference

    monkeypatch.setattr("makermodslab.teleoperate.teleoperation_active", True)
    result = handle_start_inference(_stub_request())
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "Teleoperation" in result["message"]


def test_handle_start_inference_blocked_when_recording_active(monkeypatch) -> None:
    from makermodslab.rollout import handle_start_inference

    monkeypatch.setattr("makermodslab.record.recording_active", True)
    result = handle_start_inference(_stub_request())
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "Recording" in result["message"]


def test_handle_start_inference_blocked_when_already_active(monkeypatch) -> None:
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "inference_active", True)
    result = rollout.handle_start_inference(_stub_request())
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "already active" in result["message"]


def test_handle_start_inference_blocked_when_calibration_active(monkeypatch) -> None:
    """Inference must refuse to start while manual calibration owns the same
    serial bus, rather than opening a second connection on a live port."""
    from makermodslab.rollout import handle_start_inference

    monkeypatch.setattr("makermodslab.calibrate.calibration_manager.status.calibration_active", True)
    result = handle_start_inference(_stub_request())
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "Calibration" in result["message"]


def test_handle_start_inference_blocked_when_auto_calibration_active(monkeypatch) -> None:
    from makermodslab.rollout import handle_start_inference

    monkeypatch.setattr("makermodslab.auto_calibrate.auto_calibration_manager.status.active", True)
    result = handle_start_inference(_stub_request())
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "Auto-calibration" in result["message"]


def test_handle_start_inference_blocked_when_wiggle_active(monkeypatch) -> None:
    from makermodslab.rollout import handle_start_inference

    monkeypatch.setattr("makermodslab.wiggle.wiggle_active", True)
    result = handle_start_inference(_stub_request())
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "wiggle" in result["message"].lower()


def test_handle_start_inference_pins_return_to_initial_position(monkeypatch, tmp_path) -> None:
    """The stop dialog promises the follower eases back to its start pose on
    teardown. That behaviour is lerobot's `return_to_initial_position`, which
    defaults to True today — but we pin it explicitly so an upstream default
    flip can't silently break the promise. Capture the rollout command and
    assert the flag is present.

    This is the one command-construction test: it stubs out the subprocess and
    every hardware-touching preflight so nothing real is started, runs the
    background startup worker synchronously (via the _SyncThread stub), and
    redirects HOME so the worker's log file lands in tmp rather than the real
    cache — we only inspect the argv handed to Popen. The resolve stub takes the
    `report` kwarg the worker now passes for download progress."""
    from makermodslab import rollout

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(rollout, "setup_follower_calibration_file", lambda cfg: cfg)
    monkeypatch.setattr(rollout, "_preflight_arm_identity", lambda *a, **k: [])
    monkeypatch.setattr(rollout, "_preflight_motor_registers", lambda *a, **k: [])
    monkeypatch.setattr(
        rollout, "_resolve_policy_path", lambda ref, report=None: str(tmp_path / "pretrained_model")
    )
    monkeypatch.setattr(rollout, "_detect_device", lambda: "cpu")

    captured: dict = {}

    class _FakeProc:
        pid = 4321

        def __init__(self, cmd, **kwargs):
            captured["cmd"] = cmd
            # A stdin for the newline-seeding block, a stdout the pump can drain.
            self.stdin = io.BytesIO()
            self.stdout = _EmptyStdout()

        def poll(self):
            return None

    monkeypatch.setattr(rollout.subprocess, "Popen", _FakeProc)
    # Run the startup worker (and its stdout pump) inline.
    monkeypatch.setattr(rollout.threading, "Thread", _SyncThread)

    result = rollout.handle_start_inference(_stub_request())
    assert result["success"] is True, result

    cmd = captured["cmd"]
    assert "--return_to_initial_position=true" in cmd
    # Sanity: the core rollout invocation is intact around our pinned flag.
    assert "lerobot.scripts.lerobot_rollout" in cmd
    assert "--strategy.type=base" in cmd


def test_rollout_cli_args_emits_sync_engine_by_default() -> None:
    """`inference` is a draccus ChoiceRegistry field defaulting to sync
    upstream; we name it explicitly so an upstream flip can't silently change
    which engine drives the arm."""
    from makermodslab.rollout import _rollout_cli_args

    args = _rollout_cli_args(_stub_request(), "/tmp/pretrained_model", [])
    assert "--inference.type=sync" in args


def test_rollout_cli_args_forwards_the_engine_to_both_front_ends() -> None:
    """_rollout_cli_args is shared by the single-episode command and the eval
    runner — guard against the eval path drifting away from the rollout path."""
    from makermodslab.rollout import (
        InferenceRequest,
        _build_eval_runner_cmd,
        _build_rollout_cmd,
    )

    request = InferenceRequest(
        follower_port="/dev/ttyUSB0",
        follower_config="robot_a",
        policy_ref="user/repo@checkpoints/000050",
        inference_engine="rtc",
        eval_episodes=5,
    )
    single = _build_rollout_cmd(request, "/tmp/pretrained_model", [])
    evalrun = _build_eval_runner_cmd(request, "/tmp/pretrained_model", [])
    assert "--inference.type=rtc" in single
    assert "--inference.type=rtc" in evalrun
    assert "--inference.type=sync" not in single
    assert "--inference.type=sync" not in evalrun


# ---------------------------------------------------------------------------
# --robot.* arg construction — single vs bimanual (pure, no I/O)
# ---------------------------------------------------------------------------


def _bimanual_request():
    from makermodslab.rollout import InferenceRequest

    return InferenceRequest(
        follower_port="/dev/left",
        follower_config="left_cal",
        policy_ref="user/repo@checkpoints/000050",
        mode="bimanual",
        right_follower_port="/dev/right",
        right_follower_config="right_cal",
        robot_name="dual_arm",
    )


def test_single_robot_args_uses_so101_follower_type() -> None:
    from makermodslab.rollout import _single_robot_args

    args = _single_robot_args(_stub_request(), "robot_a")
    assert "--robot.type=so101_follower" in args
    assert "--robot.port=/dev/ttyUSB0" in args
    assert "--robot.id=robot_a" in args
    # No cameras on the stub request → no --robot.cameras arg.
    assert not any(a.startswith("--robot.cameras=") for a in args)


def test_single_robot_args_appends_cameras_when_present() -> None:
    from makermodslab.rollout import InferenceRequest, _single_robot_args

    req = InferenceRequest(
        follower_port="/dev/ttyUSB0",
        follower_config="robot_a",
        policy_ref="user/repo@checkpoints/000050",
        cameras={"front": {"type": "opencv", "camera_index": 0, "width": 640, "height": 480}},
    )
    args = _single_robot_args(req, "robot_a")
    cam_arg = next(a for a in args if a.startswith("--robot.cameras="))
    assert "front:" in cam_arg
    assert "index_or_path: 0" in cam_arg


def test_bimanual_robot_args_uses_bi_so_follower_with_both_ports() -> None:
    from makermodslab.rollout import _bimanual_robot_args

    args = _bimanual_robot_args(_bimanual_request(), "dual_arm", "/staging/follower")
    assert "--robot.type=bi_so_follower" in args
    assert "--robot.id=dual_arm" in args
    assert "--robot.calibration_dir=/staging/follower" in args
    assert "--robot.left_arm_config.port=/dev/left" in args
    assert "--robot.right_arm_config.port=/dev/right" in args


def test_bimanual_robot_args_puts_cameras_on_left_arm_only() -> None:
    from makermodslab.rollout import InferenceRequest, _bimanual_robot_args

    req = InferenceRequest(
        follower_port="/dev/left",
        follower_config="left_cal",
        policy_ref="user/repo@checkpoints/000050",
        mode="bimanual",
        right_follower_port="/dev/right",
        right_follower_config="right_cal",
        cameras={"front": {"type": "opencv", "camera_index": 0, "width": 640, "height": 480}},
    )
    args = _bimanual_robot_args(req, "dual_arm", "/staging/follower")
    assert any(a.startswith("--robot.left_arm_config.cameras=") for a in args)
    assert not any(a.startswith("--robot.right_arm_config.cameras=") for a in args)


def test_build_rollout_cmd_wraps_robot_args_with_shared_flags() -> None:
    from makermodslab.rollout import _build_rollout_cmd

    robot_args = ["--robot.type=so101_follower", "--robot.port=/dev/ttyUSB0"]
    cmd = _build_rollout_cmd(_stub_request(), "/local/pretrained_model", robot_args)
    assert "lerobot.scripts.lerobot_rollout" in cmd
    assert "--strategy.type=base" in cmd
    assert "--policy.path=/local/pretrained_model" in cmd
    assert "--robot.type=so101_follower" in cmd
    assert "--return_to_initial_position=true" in cmd
    assert "--duration=60" in cmd


# ---------------------------------------------------------------------------
# handle_start_inference — the arm-count 409 guard (fires before any port opens)
# ---------------------------------------------------------------------------


def test_handle_start_inference_rejects_bimanual_checkpoint_on_single_robot() -> None:
    """A bimanual checkpoint on a single-arm robot returns 409 without opening
    any port or spawning a subprocess."""
    from makermodslab.rollout import InferenceRequest, handle_start_inference

    req = InferenceRequest(
        follower_port="/dev/ttyUSB0",
        follower_config="robot_a",
        policy_ref="user/repo@checkpoints/000050",
        mode="single",
        checkpoint_state_dim=12,
    )
    result = handle_start_inference(req)
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "bimanual" in result["message"]


def test_handle_start_inference_rejects_single_checkpoint_on_bimanual_robot() -> None:
    from makermodslab.rollout import InferenceRequest, handle_start_inference

    req = InferenceRequest(
        follower_port="/dev/left",
        follower_config="left_cal",
        policy_ref="user/repo@checkpoints/000050",
        mode="bimanual",
        right_follower_port="/dev/right",
        right_follower_config="right_cal",
        checkpoint_state_dim=6,
    )
    result = handle_start_inference(req)
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "single-arm" in result["message"]


def test_handle_start_inference_arm_count_guard_releases_slot() -> None:
    """A rejected start must leave inference_active False so the next request
    isn't wedged behind a phantom session."""
    from makermodslab import rollout

    req = rollout.InferenceRequest(
        follower_port="/dev/ttyUSB0",
        follower_config="robot_a",
        policy_ref="user/repo@checkpoints/000050",
        mode="single",
        checkpoint_state_dim=12,
    )
    rollout.handle_start_inference(req)
    assert rollout.inference_active is False


def test_handle_start_inference_bimanual_builds_bi_so_follower_command(monkeypatch, tmp_path) -> None:
    """End-to-end (no hardware): a bimanual request stages the two follower
    calibrations and hands Popen a `bi_so_follower` argv with both ports and
    two stdin newlines (one prompt per sub-arm's connect()).

    Mirrors the pin-test's stub pattern: subprocess, the two preflights, and the
    staging helper are all replaced so nothing real runs; the startup worker (and
    its stdout pump) run inline via _SyncThread and HOME is redirected so the log
    file lands in tmp."""
    from makermodslab import rollout

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(rollout, "bimanual_base_id", lambda name: "dual_arm")
    monkeypatch.setattr(
        rollout,
        "stage_bimanual_follower_calibrations",
        lambda *a, **k: ("/staging/follower", "dual_arm"),
    )
    monkeypatch.setattr(rollout, "_preflight_arm_identity", lambda *a, **k: [])
    monkeypatch.setattr(rollout, "_preflight_motor_registers", lambda *a, **k: [])
    monkeypatch.setattr(
        rollout, "_resolve_policy_path", lambda ref, report=None: str(tmp_path / "pretrained_model")
    )
    monkeypatch.setattr(rollout, "_detect_device", lambda: "cpu")

    captured: dict = {}

    class _FakeStdin:
        def __init__(self) -> None:
            self.written = b""

        def write(self, data: bytes) -> None:
            self.written += data

        def flush(self) -> None:
            pass

        def close(self) -> None:
            pass

    class _FakeProc:
        pid = 9999

        def __init__(self, cmd, **kwargs):
            captured["cmd"] = cmd
            self.stdin = _FakeStdin()
            self.stdout = _EmptyStdout()
            captured["stdin"] = self.stdin

        def poll(self):
            return None

    monkeypatch.setattr(rollout.subprocess, "Popen", _FakeProc)
    monkeypatch.setattr(rollout.threading, "Thread", _SyncThread)

    result = rollout.handle_start_inference(_bimanual_request())
    assert result["success"] is True, result

    cmd = captured["cmd"]
    assert "--robot.type=bi_so_follower" in cmd
    assert "--robot.left_arm_config.port=/dev/left" in cmd
    assert "--robot.right_arm_config.port=/dev/right" in cmd
    assert "--robot.calibration_dir=/staging/follower" in cmd
    # Two sub-arms → two seeded newlines (single-arm seeds only one).
    assert captured["stdin"].written == b"\n\n"


# ---------------------------------------------------------------------------
# Startup phase model — the "which substep am I in" status (download / subprocess
# fully MOCKED; no real inference, no hardware, no port opened).
# ---------------------------------------------------------------------------


def test_resolve_policy_path_sets_downloading_model_phase(monkeypatch, tmp_path) -> None:
    """During the Hub snapshot_download, an active session's phase must read
    `downloading_model` so the UI can name that (multi-second) wait."""
    from makermodslab import rollout

    # Seed a live meta the way handle_start_inference does before the download.
    monkeypatch.setattr(rollout, "_inference_meta", {"phase": rollout.PHASE_STARTING})

    seen_phase: dict = {}

    def fake_snapshot_download(**kwargs):
        # Capture the phase *at the moment of download*, not after.
        seen_phase["phase"] = rollout._inference_meta.get("phase")
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    rollout._resolve_policy_path("user/repo@root")

    assert seen_phase["phase"] == rollout.PHASE_DOWNLOADING_MODEL


def test_resolve_policy_path_local_dir_leaves_phase_untouched(monkeypatch, tmp_path) -> None:
    """A local checkpoint dir needs no download, so it must NOT flip the phase
    to downloading_model."""
    from makermodslab import rollout

    pretrained = tmp_path / "pretrained_model"
    pretrained.mkdir()
    monkeypatch.setattr(rollout, "_inference_meta", {"phase": rollout.PHASE_STARTING})

    rollout._resolve_policy_path(str(pretrained))

    assert rollout._inference_meta["phase"] == rollout.PHASE_STARTING


def test_set_phase_noops_without_active_session(monkeypatch) -> None:
    """A late stdout line arriving after teardown (empty meta) can't resurrect
    a phase on an empty dict."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "_inference_meta", {})
    rollout._set_phase(rollout.PHASE_CONNECTING)
    assert rollout._inference_meta == {}


class _LineFeeder:
    def __init__(self, lines: list[bytes]) -> None:
        self._it = iter(lines + [b""])

    def readline(self) -> bytes:
        return next(self._it)


class _NullLog:
    def write(self, *a) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


def test_pump_stdout_advances_phases_through_setup(monkeypatch) -> None:
    """The stdout pump walks loading_policy → connecting → running off the
    stable lerobot setup lines, then pins running at the rollout marker."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "_inference_meta", {"phase": rollout.PHASE_STARTING})
    monkeypatch.setattr(rollout, "_inference_started_at", 0.0)
    monkeypatch.setattr(rollout, "_inference_rollout_started_at", None)

    phases_seen: list[str] = []

    real_set_phase = rollout._set_phase

    def recording_set_phase(phase: str) -> None:
        real_set_phase(phase)
        phases_seen.append(phase)

    monkeypatch.setattr(rollout, "_set_phase", recording_set_phase)

    class _Proc:
        stdout = _LineFeeder(
            [
                b"INFO Loading policy from 'user/repo'...\n",
                b"INFO Policy loaded: type=act, device=cpu\n",
                b"INFO Connecting robot (so101_follower)...\n",
                b"INFO Robot connected: so101_follower\n",
                b"INFO Rollout setup complete, starting rollout...\n",
                b"INFO step 0\n",
            ]
        )

    rollout._pump_stdout(_Proc(), _NullLog())

    assert phases_seen == [
        rollout.PHASE_LOADING_POLICY,
        rollout.PHASE_CONNECTING,
        rollout.PHASE_RUNNING,
    ]
    assert rollout._inference_meta["phase"] == rollout.PHASE_RUNNING
    # The marker also stamped the rollout-start time.
    assert rollout._inference_rollout_started_at is not None


def test_pump_stdout_does_not_regress_phase_after_marker(monkeypatch) -> None:
    """A setup-looking line AFTER the rollout marker must not drag a running
    session back to `connecting`."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "_inference_meta", {"phase": rollout.PHASE_STARTING})
    monkeypatch.setattr(rollout, "_inference_started_at", 0.0)
    monkeypatch.setattr(rollout, "_inference_rollout_started_at", None)

    class _Proc:
        stdout = _LineFeeder(
            [
                b"INFO Rollout setup complete, starting rollout...\n",
                b"INFO Connecting robot (stray later mention)...\n",
            ]
        )

    rollout._pump_stdout(_Proc(), _NullLog())
    assert rollout._inference_meta["phase"] == rollout.PHASE_RUNNING


def test_start_inference_seeds_starting_phase(monkeypatch) -> None:
    """The start handler seeds a `starting` phase synchronously before handing
    off to the background worker, so the very first status poll can already name
    the wait. Here the worker Thread is a no-op — modelling the instant after the
    POST returns, before the worker has run — so the phase stays `starting`."""
    from makermodslab import rollout

    # A no-op Thread: the background startup worker is never actually run, so the
    # meta shows the state the POST left behind.
    monkeypatch.setattr(
        rollout.threading, "Thread", lambda *a, **k: type("_T", (), {"start": lambda self: None})()
    )

    result = rollout.handle_start_inference(_stub_request())
    assert result["success"] is True, result
    assert rollout._inference_meta["phase"] == rollout.PHASE_STARTING

    status = rollout.handle_inference_status()
    assert status["phase"] == rollout.PHASE_STARTING


def test_stop_inference_sets_stopping_phase(monkeypatch) -> None:
    """A stop request stamps `stopping` on the meta before terminate/wait, so a
    racing status poll doesn't report a stale `running`."""
    from makermodslab import rollout

    phase_at_terminate: dict = {}

    class _FakeProc:
        def terminate(self):
            phase_at_terminate["phase"] = rollout._inference_meta.get("phase")

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(rollout, "_inference_proc", _FakeProc())
    monkeypatch.setattr(rollout, "_inference_meta", {"phase": rollout.PHASE_RUNNING})

    result = rollout.handle_stop_inference()
    assert result["success"] is True
    assert phase_at_terminate["phase"] == rollout.PHASE_STOPPING


def test_status_finalisation_reports_stopped_on_clean_exit(monkeypatch) -> None:
    """A subprocess that exited rc=0 finalises to the terminal `stopped` phase."""
    from makermodslab import rollout

    class _ExitedProc:
        returncode = 0

        def poll(self):
            return 0

    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(rollout, "_inference_proc", _ExitedProc())
    monkeypatch.setattr(rollout, "_inference_started_at", 0.0)
    monkeypatch.setattr(rollout, "_inference_meta", {"phase": rollout.PHASE_RUNNING})

    result = rollout.handle_inference_status()
    assert result["exited"] is True
    assert result["phase"] == rollout.PHASE_STOPPED


def test_status_finalisation_reports_error_on_nonzero_exit(monkeypatch) -> None:
    """A non-zero exit code finalises to the terminal `error` phase."""
    from makermodslab import rollout

    class _CrashedProc:
        returncode = 1

        def poll(self):
            return 1

    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(rollout, "_inference_proc", _CrashedProc())
    monkeypatch.setattr(rollout, "_inference_started_at", 0.0)
    monkeypatch.setattr(rollout, "_inference_meta", {"phase": rollout.PHASE_CONNECTING})

    result = rollout.handle_inference_status()
    assert result["exited"] is True
    assert result["phase"] == rollout.PHASE_ERROR


def test_terminal_status_is_idempotent_across_polls(monkeypatch) -> None:
    """The terminal payload must survive repeated polls, not report-once.

    Several surfaces poll /inference-status concurrently (session dialog +
    Deploy panel); with a consume-once payload, whichever poll lands first
    after the subprocess dies swallows the outcome/error/hint and the dialog
    misreports a crash as a clean finish. A new start clears the stored
    result so the next session's first poll reflects THAT session."""
    from makermodslab import rollout

    class _CrashedProc:
        returncode = 1

        def poll(self):
            return 1

    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(rollout, "_inference_proc", _CrashedProc())
    monkeypatch.setattr(rollout, "_inference_started_at", 0.0)
    monkeypatch.setattr(rollout, "_inference_meta", {"phase": rollout.PHASE_RUNNING})

    first = rollout.handle_inference_status()  # finalises the exit
    second = rollout.handle_inference_status()  # a second poller must see the same
    assert first["exited"] is True and first["outcome"] == "failed"
    assert second == first

    # A new start supersedes the stored result.
    monkeypatch.setattr(
        rollout.threading, "Thread", lambda *a, **k: type("_T", (), {"start": lambda self: None})()
    )
    assert rollout.handle_start_inference(_stub_request())["success"] is True
    status = rollout.handle_inference_status()
    assert status["inference_active"] is True
    assert "outcome" not in status


# ---------------------------------------------------------------------------
# I1: a stopped startup worker has no tracked thread handle, so it keeps
# touching hardware after stop() and a new session can't tell it's still
# running (unlike teleoperate.py's `teleoperation_thread`, which the start
# guard checks with `.is_alive()` before claiming the slot).
# ---------------------------------------------------------------------------


def test_stopped_startup_worker_blocks_a_new_session_from_starting(monkeypatch) -> None:
    """`_inference_cancel` only aborts the worker at coarse boundaries (before
    entering `_prepare_robot`, after it returns) — nothing checks the cancel
    flag WHILE `_prepare_robot` itself runs, and nothing tracks a handle to the
    worker thread. So a stop pressed mid-preflight lets the orphaned worker
    keep opening the bus / writing motor registers, and — because there is no
    handle to ask "is that worker still alive" — a brand-new start goes ahead
    and races it for the same serial port.

    Real thread, real Event: the worker blocks inside a stubbed
    `_prepare_robot` (standing in for the hardware-touching preflight) until
    released, letting the test control the exact interleaving stop() must
    protect against."""
    from makermodslab import rollout

    entered_preflight = threading.Event()
    release_preflight = threading.Event()

    def _blocking_prepare_robot(request):
        entered_preflight.set()
        assert release_preflight.wait(timeout=5), "test setup: preflight release never signalled"
        return [], []

    monkeypatch.setattr(rollout, "_prepare_robot", _blocking_prepare_robot)
    monkeypatch.setattr(rollout, "_resolve_policy_path", lambda ref, report=None: ref)

    created_threads: list[threading.Thread] = []
    real_thread = threading.Thread

    def _tracking_thread(*args, **kwargs):
        t = real_thread(*args, **kwargs)
        created_threads.append(t)
        return t

    monkeypatch.setattr(rollout.threading, "Thread", _tracking_thread)

    try:
        first = rollout.handle_start_inference(_stub_request())
        assert first["success"] is True
        assert entered_preflight.wait(timeout=5), "worker never reached _prepare_robot"

        # Stop while the worker is INSIDE _prepare_robot — hardware is already
        # being touched, and the worker has no way to be interrupted mid-call.
        stopped = rollout.handle_stop_inference()
        assert stopped["success"] is True
        assert rollout.inference_active is False

        # The orphaned worker from the stopped session is still alive (stuck in
        # _prepare_robot) and still driving hardware. A new session must be
        # refused rather than being allowed to open the same serial port out
        # from under it.
        second = rollout.handle_start_inference(_stub_request())
        assert second["success"] is False, (
            "a new session was allowed to start while a stopped session's "
            "startup worker was still alive and touching hardware"
        )
        assert second["status_code"] == 409
    finally:
        release_preflight.set()
        for t in created_threads:
            t.join(timeout=5)


# ---------------------------------------------------------------------------
# I6: I1 added the is_alive() guard that refuses a NEW session while a
# stopped session's startup worker is still alive, but gave the operator no
# way to see that from /inference-status (idle looks identical either way)
# and no way to force/confirm it from a second stop-inference call (unlike
# teleoperate.py's second-stop, which joins the worker with a timeout and
# reports honestly). These tests cover that gap.
# ---------------------------------------------------------------------------


class _FakeStartupWorker:
    """Thread double for the orphaned inference-startup worker.

    ``dies_on_join`` controls whether ``join()`` simulates the worker actually
    finishing (mirrors teleoperate.py's test double) or simulates a worker
    still stuck inside the unjoinable ``_prepare_robot`` call (stays alive no
    matter how long the caller waits)."""

    def __init__(self, dies_on_join: bool) -> None:
        self._alive = True
        self._dies_on_join = dies_on_join
        self.joined_with_timeout: float | None = None

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:
        self.joined_with_timeout = timeout
        if self._dies_on_join:
            self._alive = False


def test_inference_status_reports_shutting_down_when_startup_worker_orphaned(monkeypatch) -> None:
    """A stopped session whose startup worker hasn't exited yet must be
    visible on /inference-status — not indistinguishable from true idle."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "_inference_startup_thread", _FakeStartupWorker(dies_on_join=False))

    status = rollout.handle_inference_status()

    assert status["shutting_down"] is True


def test_inference_status_not_shutting_down_when_truly_idle() -> None:
    from makermodslab.rollout import handle_inference_status

    status = handle_inference_status()

    assert status["shutting_down"] is False


def test_second_stop_while_startup_worker_alive_joins_with_timeout_and_reports(monkeypatch) -> None:
    """Pressing Stop again while the orphaned startup worker is still alive
    must actually wait (bounded) for it, not just repeat a blanket 409 —
    mirrors teleoperate.py's second-stop-during-grace behavior, adapted for a
    worker with no cooperative cancellation checkpoint to force through."""
    from makermodslab import rollout

    worker = _FakeStartupWorker(dies_on_join=False)
    monkeypatch.setattr(rollout, "_inference_startup_thread", worker)

    result = rollout.handle_stop_inference()

    assert result["success"] is True
    assert worker.joined_with_timeout is not None, "second stop must join() the orphaned worker"
    assert result["shutting_down"] is True
    assert "shutting down" in result["message"].lower()


def test_second_stop_while_startup_worker_exits_during_join_reports_finished(monkeypatch) -> None:
    from makermodslab import rollout

    worker = _FakeStartupWorker(dies_on_join=True)
    monkeypatch.setattr(rollout, "_inference_startup_thread", worker)

    result = rollout.handle_stop_inference()

    assert result["success"] is True
    assert result.get("shutting_down") is not True
    assert "finished" in result["message"].lower()


def test_fail_startup_result_is_idempotent_across_polls(monkeypatch) -> None:
    """A pre-subprocess failure (download/preflight) persists the same way."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(rollout, "_inference_meta", {"policy_ref": "user/repo@root"})
    rollout._fail_startup("Failed to download the model: boom")

    first = rollout.handle_inference_status()
    second = rollout.handle_inference_status()
    assert first["exited"] is True and first["outcome"] == "failed"
    assert first["error"] == "Failed to download the model: boom"
    assert first["policy_ref"] == "user/repo@root"
    assert second == first


def test_classify_outcome_ok_warns_and_fails() -> None:
    from makermodslab.rollout import _classify_outcome

    # rc 0/None => the run was fine.
    assert _classify_outcome(0, True, "overload") == "ok"
    assert _classify_outcome(None, True, None) == "ok"
    # Non-zero AFTER the rollout started, with a torque-disable/overload on
    # shutdown => the skill ran; only cleanup tripped.
    assert _classify_outcome(1, True, "Motor 6 overload, torque_enable failed") == "ran_with_warning"
    # Never started, or an unrelated error => a real failure.
    assert _classify_outcome(1, False, "overload") == "failed"
    assert _classify_outcome(1, True, "could not connect to the arm") == "failed"
    # A connection lost mid-run (cable bumped while the policy is driving) is a
    # real failure, not a shutdown/cleanup warning — connection-loss markers are
    # deliberately excluded from the cleanup set.
    assert _classify_outcome(1, True, "DeviceNotConnectedError: follower is not connected") == "failed"


def test_friendly_hint_maps_common_failures() -> None:
    from makermodslab.utils.errors import friendly_hint

    assert "gripper" in (friendly_hint("Motor overload detected") or "").lower()
    assert "connect" in (friendly_hint("Failed to connect to the follower") or "").lower()
    assert friendly_hint("some unrecognised traceback") is None
    assert friendly_hint(None) is None


def test_friendly_hint_servo_bus_error_is_not_a_download_failure() -> None:
    """A servo that stops answering must read as an ARM problem.

    lerobot's motors bus raises every serial failure as `ConnectionError`, and
    the type name itself contains "connect" — keying the Hub-download hint on
    that token labelled arm-side startup crashes "couldn't download the model"
    (observed 2026-08-03: a `Failed to write 'Lock' ... [TxRxResult] Incorrect
    status packet!` at robot.connect() reported as a failed model download)."""
    from makermodslab.utils.errors import friendly_hint

    for text in (
        "ConnectionError: Failed to write 'Lock' on id_=3 with '1' after 1 tries. "
        "[TxRxResult] Incorrect status packet!",
        "Failed to start inference: ConnectionError: Failed to sync read 'Present_Position' "
        "on ids=[1, 2, 3] after 1 tries. [TxRxResult] There is no status packet!",
    ):
        hint = friendly_hint(text) or ""
        assert "motor" in hint.lower()
        assert "download" not in hint.lower()


def test_friendly_hint_still_names_real_download_failures() -> None:
    """The other side of the tightening: a genuine fetch failure keeps its Hub
    hint. Download-step failures reach here with rollout's own
    "Failed to download the model: …" prefix (see _inference_startup_thread)."""
    from makermodslab.utils.errors import friendly_hint

    network = friendly_hint(
        "Failed to download the model: (MaxRetryError(\"HTTPSConnectionPool(host='huggingface.co', "
        'port=443): Max retries exceeded with url: /api/models/user/repo"))'
    )
    assert network is not None and "download the model" in network.lower()
    # No hub host in the text at all — rollout's prefix is what identifies it.
    offline = friendly_hint(
        "Failed to download the model: An error happened while trying to locate the file on the Hub "
        "and we cannot find the requested files in the local cache. Please check your connection."
    )
    assert offline is not None and "download the model" in offline.lower()
    missing = friendly_hint(
        "Failed to download the model: RepositoryNotFoundError: 404 Client Error. "
        "Repository Not Found for url: https://huggingface.co/api/models/user/repo"
    )
    assert missing is not None and "hub" in missing.lower()
    full = friendly_hint("Failed to download the model: OSError: [Errno 28] No space left on device")
    assert full is not None and "disk space" in full.lower()


def test_extract_error_from_log_pulls_exception_tail(tmp_path) -> None:
    from makermodslab.rollout import _extract_error_from_log

    log = tmp_path / "rollout.log"
    log.write_text(
        "INFO starting rollout\n"
        "Traceback (most recent call last):\n"
        '  File "x.py", line 1\n'
        "RuntimeError: gripper overload during shutdown\n",
        encoding="utf-8",
    )
    out = _extract_error_from_log(str(log))
    assert out is not None and "RuntimeError: gripper overload during shutdown" in out
    assert _extract_error_from_log(None) is None
    assert _extract_error_from_log(str(tmp_path / "missing.log")) is None


def test_inference_in_use_path_none_when_idle() -> None:
    """No active inference -> no in-use path (delete guards stay open)."""
    from makermodslab import rollout

    assert rollout.inference_in_use_path() is None


def test_inference_in_use_path_returns_resolved_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """While a session is active, the accessor exposes the RESOLVED local
    checkpoint dir captured at start (not the possibly-hub policy_ref)."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(
        rollout,
        "_inference_meta",
        {"policy_ref": "user/repo@root", "policy_path": "/tmp/ckpt/pretrained_model"},
    )
    assert rollout.inference_in_use_path() == "/tmp/ckpt/pretrained_model"


# ---------------------------------------------------------------------------
# Navigate-first startup: the POST returns immediately and the heavy work
# (download → preflight → spawn) runs in the background worker. All of these
# fully MOCK snapshot_download / the subprocess — no network, no hardware.
# ---------------------------------------------------------------------------


def test_start_inference_returns_immediately_without_downloading(monkeypatch) -> None:
    """The whole point of the rework: the POST must not block on the Hub
    download. With the worker Thread stubbed to a no-op, the handler still
    returns success and claims the session — and snapshot_download is never
    touched on the request thread (it would raise here if it were)."""
    from makermodslab import rollout

    def _boom(**kwargs):
        raise AssertionError("snapshot_download must not run on the request thread")

    monkeypatch.setattr("huggingface_hub.snapshot_download", _boom)
    monkeypatch.setattr(
        rollout.threading, "Thread", lambda *a, **k: type("_T", (), {"start": lambda self: None})()
    )

    result = rollout.handle_start_inference(_stub_request())
    assert result["success"] is True
    assert rollout.inference_active is True
    # Visible from the very first status poll, before the worker has run.
    assert rollout._inference_meta["phase"] == rollout.PHASE_STARTING


def test_download_progress_reported_into_status(monkeypatch) -> None:
    """While a Hub checkpoint downloads, snapshot_download's byte updates flow
    through the progress tqdm into the meta, and /inference-status exposes them
    as download_bytes_done / _total / _percent. The total can arrive after some
    bytes (metadata discovery), which is exactly the refresh()-then-update()
    order huggingface_hub uses on the shared bytes bar."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(rollout, "_inference_started_at", 0.0)
    monkeypatch.setattr(
        rollout,
        "_inference_meta",
        {"phase": rollout.PHASE_STARTING, "policy_ref": "user/repo@checkpoints/000050"},
    )

    def fake_snapshot_download(**kwargs):
        # huggingface_hub instantiates the shared bytes bar (unit="B"); a file's
        # size becoming known grows total via refresh(), chunks arrive via
        # update(n).
        cls = kwargs["tqdm_class"]
        bar = cls(total=None, unit="B")
        bar.total = 1000
        bar.refresh()
        bar.update(250)
        return "/tmp/snap"

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    rollout._resolve_policy_path("user/repo@checkpoints/000050", report=rollout._report_download_progress)

    assert rollout._inference_meta["phase"] == rollout.PHASE_DOWNLOADING_MODEL
    status = rollout.handle_inference_status()
    assert status["download_bytes_done"] == 250
    assert status["download_bytes_total"] == 1000
    assert status["download_percent"] == 25.0


def test_download_percent_is_none_until_total_known(monkeypatch) -> None:
    """Before any file size is known the total is None, so download_percent is
    None too → the UI shows an indeterminate bar rather than a bogus 0/0%."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(
        rollout, "_inference_meta", {"phase": rollout.PHASE_STARTING, "policy_ref": "user/repo@root"}
    )

    def fake_snapshot_download(**kwargs):
        cls = kwargs["tqdm_class"]
        bar = cls(total=None, unit="B")
        bar.update(128)  # bytes trickling in before any total is known
        return "/tmp/snap"

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    rollout._resolve_policy_path("user/repo@root", report=rollout._report_download_progress)

    status = rollout.handle_inference_status()
    assert status["download_bytes_done"] == 128
    assert status["download_bytes_total"] is None
    assert status["download_percent"] is None


def test_startup_download_failure_reports_failed_and_hint_without_spawn(monkeypatch) -> None:
    """A Hub download that raises (offline / 404 / disk full) is finalised as a
    `failed` outcome carrying the error text + a friendly hint — and no arm
    preflight runs and no subprocess spawns."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(
        rollout,
        "_inference_meta",
        {"phase": rollout.PHASE_STARTING, "policy_ref": "user/repo@checkpoints/000050"},
    )

    def _raise(ref, report=None):
        raise RuntimeError("Repository Not Found for url: https://huggingface.co/api/models/x")

    monkeypatch.setattr(rollout, "_resolve_policy_path", _raise)

    def _no_prepare(*a, **k):
        raise AssertionError("preflight must not run after a download failure")

    def _no_popen(*a, **k):
        raise AssertionError("no subprocess may spawn after a download failure")

    monkeypatch.setattr(rollout, "_prepare_robot", _no_prepare)
    monkeypatch.setattr(rollout.subprocess, "Popen", _no_popen)

    rollout._run_inference_startup(_stub_request(), threading.Event())

    assert rollout.inference_active is False
    status = rollout.handle_inference_status()
    assert status["exited"] is True
    assert status["outcome"] == "failed"
    assert status["phase"] == rollout.PHASE_ERROR
    assert "download" in (status["error"] or "").lower()
    # friendly_hint recognises the Hub-not-found token and adds a hint.
    assert status["hint"] is not None and "Hub" in status["hint"]


def test_stop_during_download_leaves_clean_idle_without_spawn(monkeypatch) -> None:
    """Pressing Stop while the model is still downloading tears the session down
    to a clean idle: the worker abandons after the download returns, never
    opening the bus (_prepare_robot) or spawning a subprocess. Models the real
    ordering — stop() with no subprocess yet flips the session idle and sets the
    cancel event; the in-flight download still finishes into the cache."""
    from makermodslab import rollout

    cancel = threading.Event()
    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(rollout, "_inference_cancel", cancel)
    monkeypatch.setattr(rollout, "_inference_proc", None)
    monkeypatch.setattr(
        rollout,
        "_inference_meta",
        {"phase": rollout.PHASE_DOWNLOADING_MODEL, "policy_ref": "user/repo@checkpoints/000050"},
    )

    def _resolve_then_stop(ref, report=None):
        rollout.handle_stop_inference()
        return "/tmp/snap/pretrained_model"

    def _no_prepare(*a, **k):
        raise AssertionError("no bus may be opened after a stop during download")

    def _no_popen(*a, **k):
        raise AssertionError("no subprocess may spawn after a stop during download")

    monkeypatch.setattr(rollout, "_resolve_policy_path", _resolve_then_stop)
    monkeypatch.setattr(rollout, "_prepare_robot", _no_prepare)
    monkeypatch.setattr(rollout.subprocess, "Popen", _no_popen)

    rollout._run_inference_startup(_stub_request(), cancel)

    assert rollout.inference_active is False
    assert rollout._inference_proc is None
    assert rollout._inference_meta == {}
    assert rollout.handle_inference_status()["inference_active"] is False


def test_run_inference_startup_local_ref_skips_download_phase(monkeypatch, tmp_path) -> None:
    """A local checkpoint dir needs no download: the worker resolves it instantly,
    never enters the downloading_model phase, and proceeds straight to preflight
    + spawn."""
    from makermodslab import rollout

    monkeypatch.setenv("HOME", str(tmp_path))
    pretrained = tmp_path / "pretrained_model"
    pretrained.mkdir()

    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(
        rollout, "_inference_meta", {"phase": rollout.PHASE_STARTING, "policy_ref": str(pretrained)}
    )

    phases: list[str] = []
    real_set_phase = rollout._set_phase

    def _rec(phase: str) -> None:
        phases.append(phase)
        real_set_phase(phase)

    monkeypatch.setattr(rollout, "_set_phase", _rec)
    monkeypatch.setattr(rollout, "_prepare_robot", lambda req: (["--robot.type=so101_follower"], []))
    monkeypatch.setattr(rollout, "_detect_device", lambda: "cpu")

    class _FakeProc:
        pid = 1

        def __init__(self, cmd, **kwargs):
            self.stdin = io.BytesIO()
            self.stdout = _EmptyStdout()

        def poll(self):
            return None

    monkeypatch.setattr(rollout.subprocess, "Popen", _FakeProc)
    monkeypatch.setattr(rollout.threading, "Thread", _SyncThread)

    req = rollout.InferenceRequest(follower_port="/dev/x", follower_config="c", policy_ref=str(pretrained))
    rollout._run_inference_startup(req, threading.Event())

    assert rollout.PHASE_DOWNLOADING_MODEL not in phases
    assert rollout._inference_proc is not None


# ---------------------------------------------------------------------------
# Multi-episode EVALUATION mode
#
# Pure helpers (clamping, accuracy math, verdict classification, protocol
# parsing), the request schema, the status-payload shape, the orchestrator state
# machine driven over a fake runner pipe, and the idle/mutex branches of the two
# new endpoints. Nothing here spawns a process or touches hardware: the eval
# runner is never executed, only stood in for. Per CLAUDE.md the subprocess
# happy path stays untested — what IS tested is the bookkeeping either side of
# it, which is where the verdicts and the crash containment live.
# ---------------------------------------------------------------------------


class _ExitedProc:
    """A `subprocess.Popen` stand-in that has already exited with `rc`."""

    def __init__(self, rc: int = 0) -> None:
        self.returncode = rc
        self.pid = 4242
        # A dead process has no usable command pipe — writing to it is exactly
        # how the orchestrator discovers the runner is gone.
        self.stdin = None

    def poll(self) -> int:
        return self.returncode


class _CommandPipe:
    """A subprocess `stdin` that records the command lines written to it."""

    def __init__(self, sink: list[str]) -> None:
        self._sink = sink

    def write(self, data: bytes) -> None:
        self._sink.append(data.decode().strip())

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class _FakeRunner:
    """A live `makermodslab.eval_runner` subprocess stand-in.

    Unlike `_ExitedProc` it is ALIVE (`poll()` → None) and stays alive across
    episodes, which is the property the redesign turns on: a live process no
    longer implies a live episode, so the tests drive episode boundaries through
    the protocol handlers rather than by pretending a process exited. Every
    command the orchestrator sends lands in `.commands`."""

    def __init__(self, rc: int | None = None) -> None:
        self.pid = 4242
        self.returncode = rc
        self.commands: list[str] = []
        self.stdin = _CommandPipe(self.commands)

    def poll(self) -> int | None:
        return self.returncode


def _eval_request(episodes: int = 3):
    from makermodslab.rollout import InferenceRequest

    return InferenceRequest(
        follower_port="/dev/ttyUSB0",
        follower_config="robot_a",
        policy_ref="user/repo@checkpoints/000050",
        eval_episodes=episodes,
    )


def _arm_eval_session(
    monkeypatch,
    rollout,
    episodes: int = 3,
    *,
    running: bool = True,
    proc=None,
):
    """Put the module into a mid-eval state with a live runner.

    `running` is whether an EPISODE is in flight, which is now independent of
    whether the runner process is up — pass `proc=None` for the (recoverable)
    state left behind by a runner that died."""
    session = rollout._EvalSession(request=_eval_request(episodes), episodes_total=episodes)
    session.policy_path = "/tmp/policy"
    session.robot_args = ["--robot.type=so101_follower"]
    session.episode_running = running
    monkeypatch.setattr(rollout, "_eval_session", session)
    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(rollout, "_inference_started_at", 1000.0)
    monkeypatch.setattr(rollout, "_inference_rollout_started_at", 1005.0 if running else None)
    monkeypatch.setattr(
        rollout,
        "_inference_meta",
        {
            "phase": rollout.PHASE_RUNNING if running else rollout.PHASE_RESETTING,
            "policy_ref": "user/repo@checkpoints/000050",
            "duration_s": 60,
        },
    )
    monkeypatch.setattr(rollout, "_inference_proc", _FakeRunner() if (proc is None and running) else proc)
    return session


def test_eval_episodes_defaults_to_one() -> None:
    """The historical single-rollout request is unchanged: no eval fields set."""
    from makermodslab.rollout import InferenceRequest

    req = InferenceRequest(
        follower_port="/dev/ttyUSB0",
        follower_config="robot_a",
        policy_ref="user/repo@checkpoints/000050",
    )
    assert req.eval_episodes == 1


def test_inference_request_accepts_eval_episodes() -> None:
    assert _eval_request(10).eval_episodes == 10


def test_clamp_eval_episodes_bounds_and_fallbacks() -> None:
    from makermodslab.rollout import MAX_EVAL_EPISODES, clamp_eval_episodes

    assert clamp_eval_episodes(1) == 1
    assert clamp_eval_episodes(10) == 10
    assert clamp_eval_episodes(0) == 1
    assert clamp_eval_episodes(-5) == 1
    assert clamp_eval_episodes(MAX_EVAL_EPISODES) == MAX_EVAL_EPISODES
    assert clamp_eval_episodes(10_000) == MAX_EVAL_EPISODES
    # Junk degrades to a single episode rather than raising out of the POST.
    assert clamp_eval_episodes(None) == 1
    assert clamp_eval_episodes("nope") == 1


def test_eval_accuracy_scores_successes_over_scored_episodes() -> None:
    from makermodslab.rollout import eval_accuracy

    assert eval_accuracy(["success", "success", "failure", "failure"]) == 0.5
    assert eval_accuracy(["success"]) == 1.0
    assert eval_accuracy(["failure", "failure"]) == 0.0


def test_eval_accuracy_excludes_errored_episodes_from_the_denominator() -> None:
    """A serial glitch must not poison the number: an errored episode counts
    neither for nor against the policy."""
    from makermodslab.rollout import eval_accuracy

    # 1 success, 1 failure, 2 crashes -> 1/2, not 1/4.
    assert eval_accuracy(["success", "failure", "error", "error"]) == 0.5


def test_eval_accuracy_is_none_when_nothing_scoreable() -> None:
    from makermodslab.rollout import eval_accuracy

    assert eval_accuracy([]) is None
    assert eval_accuracy(["error", "error"]) is None


def test_classify_episode_early_stop_is_a_success_whatever_the_exit_code() -> None:
    """We terminated the subprocess ourselves, so its exit code is our own
    SIGTERM and says nothing about the run."""
    from makermodslab.rollout import EPISODE_SUCCESS, classify_episode

    assert classify_episode(-15, True, True, None) == EPISODE_SUCCESS
    assert classify_episode(0, True, True, None) == EPISODE_SUCCESS
    assert classify_episode(1, True, True, "RuntimeError: boom") == EPISODE_SUCCESS


def test_classify_episode_clean_timeout_is_a_failure() -> None:
    from makermodslab.rollout import EPISODE_FAILURE, classify_episode

    assert classify_episode(0, False, True, None) == EPISODE_FAILURE


def test_classify_episode_cleanup_warning_is_a_failure_not_an_error() -> None:
    """The rollout ran its full duration and only teardown was noisy — the
    episode legitimately timed out, so it's a failure, not a crash."""
    from makermodslab.rollout import EPISODE_FAILURE, classify_episode

    # "overload" is one of errors.CLEANUP_MARKERS — a gripper still holding an
    # object when torque is released at teardown.
    verdict = classify_episode(1, False, True, "RuntimeError: motor 6 overload during shutdown")
    assert verdict == EPISODE_FAILURE


def test_classify_episode_crash_is_an_error() -> None:
    from makermodslab.rollout import EPISODE_ERROR, classify_episode

    assert classify_episode(1, False, False, "DeviceNotConnectedError: bus is gone") == EPISODE_ERROR


def test_eval_fields_are_null_shaped_for_a_single_episode_run() -> None:
    """The status shape stays stable: a plain run reports eval_mode False with
    null companions rather than omitting the keys."""
    from makermodslab.rollout import _eval_fields

    fields = _eval_fields(None)
    assert fields["eval_mode"] is False
    assert fields["episode_index"] is None
    assert fields["episodes_total"] is None
    assert fields["episode_results"] is None
    assert fields["accuracy"] is None


def test_idle_status_reports_single_episode_shape() -> None:
    from makermodslab.rollout import handle_inference_status

    result = handle_inference_status()
    assert result["eval_mode"] is False
    assert result["episodes_total"] is None
    assert result["accuracy"] is None


def test_start_inference_seeds_the_eval_session(monkeypatch) -> None:
    from makermodslab import rollout

    monkeypatch.setattr(rollout.threading, "Thread", _SyncThread)
    monkeypatch.setattr(rollout, "_run_inference_startup", lambda *a, **k: None)
    monkeypatch.setattr(rollout.camera_preview_manager, "stop_all", lambda: None)
    monkeypatch.setattr(rollout, "_policy_ref_is_valid", lambda ref: True)

    result = rollout.handle_start_inference(_eval_request(5))
    assert result["success"] is True
    assert rollout._eval_session is not None
    assert rollout._eval_session.episodes_total == 5
    assert rollout._eval_session.episode_index == 1
    status = rollout.handle_inference_status()
    assert status["eval_mode"] is True
    assert status["episodes_total"] == 5
    assert status["episode_index"] == 1
    assert status["episode_results"] == []
    assert status["accuracy"] is None


def test_start_inference_with_one_episode_leaves_eval_session_none(monkeypatch) -> None:
    """eval_episodes=1 must be bit-for-bit the historical flow."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout.threading, "Thread", _SyncThread)
    monkeypatch.setattr(rollout, "_run_inference_startup", lambda *a, **k: None)
    monkeypatch.setattr(rollout.camera_preview_manager, "stop_all", lambda: None)
    monkeypatch.setattr(rollout, "_policy_ref_is_valid", lambda ref: True)

    rollout.handle_start_inference(_eval_request(1))
    assert rollout._eval_session is None
    assert rollout.handle_inference_status()["eval_mode"] is False


def test_start_inference_clamps_the_requested_episode_count(monkeypatch) -> None:
    from makermodslab import rollout

    monkeypatch.setattr(rollout.threading, "Thread", _SyncThread)
    monkeypatch.setattr(rollout, "_run_inference_startup", lambda *a, **k: None)
    monkeypatch.setattr(rollout.camera_preview_manager, "stop_all", lambda: None)
    monkeypatch.setattr(rollout, "_policy_ref_is_valid", lambda ref: True)

    rollout.handle_start_inference(_eval_request(10_000))
    assert rollout._eval_session.episodes_total == rollout.MAX_EVAL_EPISODES


def test_start_inference_guard_failure_clears_the_eval_session(monkeypatch) -> None:
    """A rejected start must not leave eval bookkeeping behind for the next run."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "_policy_ref_is_valid", lambda ref: False)
    result = rollout.handle_start_inference(_eval_request(4))
    assert result["success"] is False
    assert rollout._eval_session is None
    assert rollout.inference_active is False


def test_episode_timeout_parks_the_session_in_the_reset_phase(monkeypatch) -> None:
    """An episode that runs out its duration scores a FAILURE and keeps the
    session — and its hold on the inference slot, the cameras AND the loaded
    policy — alive for the reset."""
    from makermodslab import rollout

    _arm_eval_session(monkeypatch, rollout, episodes=3)
    runner = rollout._inference_proc
    rollout._on_episode_ended(REASON_DURATION)
    status = rollout.handle_inference_status()

    assert status["phase"] == rollout.PHASE_RESETTING
    assert status["inference_active"] is True
    assert status["episode_results"] == ["failure"]
    assert status["episode_index"] == 2
    assert status["accuracy"] is None
    # The slot stays claimed through the reset — recording/teleop stay blocked.
    assert rollout.inference_active is True
    # And so does the runner: the whole point is that the next episode does not
    # re-pay the policy load. It is only told to QUIT once the session is over.
    assert rollout._inference_proc is runner
    assert runner.commands == []


def test_early_stop_scores_the_episode_a_success(monkeypatch) -> None:
    """The success button asks the runner to end the episode — no signal, no
    kill — and the verdict lands when the runner reports the end."""
    from makermodslab import rollout

    session = _arm_eval_session(monkeypatch, rollout, episodes=3)
    runner = rollout._inference_proc

    assert rollout.handle_stop_episode()["success"] is True
    assert runner.commands == [CMD_STOP]
    assert session.stop_requested is True
    # Still running until the runner says otherwise — nothing was scored yet.
    assert session.results == []

    rollout._on_episode_ended(REASON_STOPPED)
    status = rollout.handle_inference_status()

    assert status["episode_results"] == ["success"]
    assert status["phase"] == rollout.PHASE_RESETTING
    # The flag is one-shot: the next episode starts unstopped.
    assert session.stop_requested is False


def test_episode_end_reason_stopped_scores_a_success_without_the_flag(monkeypatch) -> None:
    """The reason IS the STOP we sent, so it stands on its own — a lost flag
    can't turn the user's success into a timeout."""
    from makermodslab import rollout

    _arm_eval_session(monkeypatch, rollout, episodes=3)
    rollout._on_episode_ended(REASON_STOPPED)
    assert rollout.handle_inference_status()["episode_results"] == ["success"]


def test_episode_end_with_no_episode_in_flight_is_ignored(monkeypatch) -> None:
    """A duplicate/late end line must not append a phantom verdict."""
    from makermodslab import rollout

    session = _arm_eval_session(monkeypatch, rollout, episodes=3, running=False, proc=_FakeRunner())
    rollout._on_episode_ended(REASON_DURATION)
    assert session.results == []


def test_crashed_runner_parks_the_episode_with_the_error_visible(monkeypatch) -> None:
    """A crash is neither success nor failure: the in-flight episode is scored
    `error` and the session parks in the reset phase with the error on show, so
    the user can continue (paying one reload) or abort."""
    from makermodslab import rollout

    monkeypatch.setattr(
        rollout,
        "_extract_error_from_log",
        lambda p: "DeviceNotConnectedError: could not connect to the follower bus",
    )
    _arm_eval_session(monkeypatch, rollout, episodes=3)
    runner = rollout._inference_proc
    monkeypatch.setattr(rollout, "_inference_rollout_started_at", None)

    rollout._handle_runner_exit(runner, 1)

    status = rollout.handle_inference_status()
    assert status["episode_results"] == ["error"]
    assert status["phase"] == rollout.PHASE_RESETTING
    assert status["inference_active"] is True
    assert "DeviceNotConnectedError" in status["error"]
    # The existing error taxonomy still applies to an episode-level crash.
    assert status["hint"]
    # The dead runner is dropped, which is what makes the continue respawn.
    assert rollout._inference_proc is None


def test_crashed_runner_prefers_its_own_error_line(monkeypatch) -> None:
    """The runner's ERROR event is the exception itself; log mining is a
    heuristic over a traceback, so the event wins when both exist."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "_extract_error_from_log", lambda p: "mined from the log tail")
    _arm_eval_session(monkeypatch, rollout, episodes=3)
    runner = rollout._inference_proc

    rollout._on_runner_error("RuntimeError: the gripper stalled")
    rollout._handle_runner_exit(runner, 1)

    assert rollout.handle_inference_status()["error"] == "RuntimeError: the gripper stalled"


def test_runner_death_during_a_reset_keeps_the_session_recoverable(monkeypatch) -> None:
    """Nothing to score — but the user has to learn that continuing now costs a
    reload, and the tally must survive."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "_extract_error_from_log", lambda p: "OSError: the port vanished")
    session = _arm_eval_session(monkeypatch, rollout, episodes=4, running=False, proc=_FakeRunner())
    session.results.extend(["success", "failure"])
    runner = rollout._inference_proc

    rollout._handle_runner_exit(runner, 1)
    status = rollout.handle_inference_status()

    assert status["episode_results"] == ["success", "failure"]  # no phantom verdict
    assert status["phase"] == rollout.PHASE_RESETTING
    assert status["inference_active"] is True
    assert "OSError" in status["error"]
    assert rollout._inference_proc is None


def test_expected_runner_exit_after_quit_is_not_scored(monkeypatch) -> None:
    """An abort asks the runner to quit; that exit must not be read as a crash
    and score the episode the abort deliberately left unscored."""
    from makermodslab import rollout

    session = _arm_eval_session(monkeypatch, rollout, episodes=3)
    runner = rollout._inference_proc
    session.quitting = True

    rollout._handle_runner_exit(runner, 0)
    assert session.results == []


def test_last_episode_finishes_the_session_with_accuracy(monkeypatch) -> None:
    from makermodslab import rollout

    session = _arm_eval_session(monkeypatch, rollout, episodes=3)
    runner = rollout._inference_proc
    session.results.extend(["success", "success"])
    rollout._on_episode_ended(REASON_DURATION)
    status = rollout.handle_inference_status()

    assert status["phase"] == rollout.PHASE_FINISHED
    assert status["inference_active"] is False
    assert status["exited"] is True
    assert status["episode_results"] == ["success", "success", "failure"]
    assert status["episodes_total"] == 3
    assert status["episode_index"] == 3
    assert status["accuracy"] == pytest.approx(2 / 3, rel=1e-3)
    # The slot is released for the next session.
    assert rollout.inference_active is False
    assert rollout._eval_session is None
    # And the runner is sent home rather than left holding the bus and cameras.
    assert runner.commands == [CMD_QUIT]


def test_finished_eval_payload_is_idempotent_across_polls(monkeypatch) -> None:
    """Several surfaces poll concurrently; the summary must survive every one."""
    from makermodslab import rollout

    session = _arm_eval_session(monkeypatch, rollout, episodes=2)
    session.results.append("success")
    rollout._on_episode_ended(REASON_DURATION)
    first = rollout.handle_inference_status()
    second = rollout.handle_inference_status()
    third = rollout.handle_inference_status()

    assert first["accuracy"] == second["accuracy"] == third["accuracy"] == 0.5
    assert second["episode_results"] == third["episode_results"] == ["success", "failure"]
    assert third["phase"] == rollout.PHASE_FINISHED


def test_accuracy_excludes_errors_in_the_finished_payload(monkeypatch) -> None:
    from makermodslab import rollout

    session = _arm_eval_session(monkeypatch, rollout, episodes=3)
    session.results.extend(["success", "error"])
    rollout._on_episode_ended(REASON_DURATION)
    status = rollout.handle_inference_status()

    assert status["episode_results"] == ["success", "error", "failure"]
    # 1 success out of 2 scored episodes, not out of 3.
    assert status["accuracy"] == 0.5


def test_stop_episode_when_idle_returns_409() -> None:
    from makermodslab.rollout import handle_stop_episode

    result = handle_stop_episode()
    assert result["success"] is False
    assert result["status_code"] == 409


def test_stop_episode_refuses_outside_eval_mode(monkeypatch) -> None:
    """A single-episode run has no tally to record a success into."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(rollout, "_inference_proc", _ExitedProc(0))
    result = rollout.handle_stop_episode()
    assert result["success"] is False
    assert result["status_code"] == 409


def test_stop_episode_refuses_while_parked_in_a_reset(monkeypatch) -> None:
    """The runner is still ALIVE between episodes, so a live process is no
    longer proof an episode is running — `episode_running` is."""
    from makermodslab import rollout

    _arm_eval_session(monkeypatch, rollout, episodes=3, running=False, proc=_FakeRunner())
    result = rollout.handle_stop_episode()
    assert result["success"] is False
    assert result["status_code"] == 409


def test_stop_episode_reports_a_dead_runner_instead_of_claiming_success(monkeypatch) -> None:
    """A success can't be recorded against a runner that isn't there to end the
    episode — the crash path will score it an error instead."""
    from makermodslab import rollout

    _arm_eval_session(monkeypatch, rollout, episodes=3, proc=_ExitedProc(1))
    result = rollout.handle_stop_episode()
    assert result["success"] is False
    assert result["status_code"] == 409


def test_next_episode_when_idle_returns_409() -> None:
    from makermodslab.rollout import handle_next_episode

    result = handle_next_episode()
    assert result["success"] is False
    assert result["status_code"] == 409


def test_next_episode_refuses_while_an_episode_is_still_running(monkeypatch) -> None:
    from makermodslab import rollout

    _arm_eval_session(monkeypatch, rollout, episodes=3)
    # Phase is `running`, not `resetting` — there is nothing to continue from.
    result = rollout.handle_next_episode()
    assert result["success"] is False
    assert result["status_code"] == 409


def test_next_episode_on_a_live_runner_costs_one_command(monkeypatch) -> None:
    """The headline of the redesign: continuing does NOT spawn anything. The
    policy is still resident and the bus and cameras are still open, so the
    whole cost is one line on the runner's stdin."""
    from makermodslab import rollout

    session = _arm_eval_session(monkeypatch, rollout, episodes=3, running=False, proc=_FakeRunner())
    runner = rollout._inference_proc
    session.results.append("failure")
    session.error = "old crash"
    session.hint = "old hint"

    def _explode(*a, **k):
        raise AssertionError("continuing must not spawn a process")

    monkeypatch.setattr(rollout, "_launch_eval_runner", _explode)
    monkeypatch.setattr(rollout, "_launch_rollout_subprocess", _explode)

    result = rollout.handle_next_episode()
    assert result["success"] is True
    assert runner.commands == [CMD_EPISODE]
    assert rollout._inference_proc is runner
    # The previous episode's crash banner is cleared on continue.
    assert session.error is None
    assert session.hint is None
    assert rollout._inference_meta["phase"] == rollout.PHASE_STARTING
    # Both timers restart so the dialog clocks the EPISODE, not the session.
    assert rollout._inference_rollout_started_at is None


def test_next_episode_respawns_a_dead_runner_and_carries_the_tally(monkeypatch) -> None:
    """Crash containment: a runner that died costs ONE reload, not the session.
    The resolved policy path and preflighted `--robot.*` args are reused
    verbatim — no re-download, no second arm-identity pass — and the tally so
    far is carried forward."""
    from makermodslab import rollout

    session = _arm_eval_session(monkeypatch, rollout, episodes=3, running=False, proc=None)
    session.results.append("error")
    monkeypatch.setattr(rollout.threading, "Thread", _SyncThread)

    launched = {}

    def _fake_launch(request, policy_path, robot_args):
        launched["policy_path"] = policy_path
        launched["robot_args"] = robot_args
        proc = _FakeRunner()
        proc.stdout = _EmptyStdout()
        return proc, io.StringIO(), __import__("pathlib").Path("/tmp/ep2.log")

    monkeypatch.setattr(rollout, "_launch_eval_runner", _fake_launch)
    # The pump would otherwise reap the fake proc and fire crash containment.
    monkeypatch.setattr(rollout, "_handle_runner_exit", lambda proc, rc: None)

    result = rollout.handle_next_episode()
    assert result["success"] is True
    assert launched["policy_path"] == "/tmp/policy"
    assert launched["robot_args"] == ["--robot.type=so101_follower"]
    assert session.results == ["error"]
    # The episode is PENDING, not started: the respawned runner has to finish
    # loading first, and its READY is what issues the command.
    assert session.episode_pending is True
    assert rollout._inference_meta["log_path"] == "/tmp/ep2.log"


def test_runner_ready_issues_the_pending_episode(monkeypatch) -> None:
    from makermodslab import rollout

    session = _arm_eval_session(monkeypatch, rollout, episodes=3, running=False, proc=_FakeRunner())
    runner = rollout._inference_proc
    session.episode_pending = True

    rollout._on_runner_ready()
    assert runner.commands == [CMD_EPISODE]
    assert rollout._inference_meta["phase"] == rollout.PHASE_STARTING


def test_runner_ready_stays_idle_when_no_episode_is_pending(monkeypatch) -> None:
    """A READY from a respawn the user hasn't continued from — or one that lands
    after an abort — must not put the arm in motion."""
    from makermodslab import rollout

    session = _arm_eval_session(monkeypatch, rollout, episodes=3, running=False, proc=_FakeRunner())
    runner = rollout._inference_proc
    session.episode_pending = False
    rollout._on_runner_ready()
    assert runner.commands == []

    session.episode_pending = True
    session.quitting = True
    rollout._on_runner_ready()
    assert runner.commands == []


def test_episode_started_flips_the_phase_to_running(monkeypatch) -> None:
    from makermodslab import rollout

    session = _arm_eval_session(monkeypatch, rollout, episodes=3, running=False, proc=_FakeRunner())
    session.episode_pending = True

    rollout._on_episode_started()
    assert session.episode_running is True
    assert session.episode_pending is False
    assert rollout._inference_meta["phase"] == rollout.PHASE_RUNNING
    assert rollout._inference_rollout_started_at is not None


def test_stop_inference_quits_the_runner_instead_of_signalling_it(monkeypatch) -> None:
    """Abort mid-episode: the runner is asked to wind down so the follower still
    eases home, and the in-flight episode stays deliberately unscored."""
    from makermodslab import rollout

    session = _arm_eval_session(monkeypatch, rollout, episodes=5)
    session.results.extend(["success", "failure"])
    runner = rollout._inference_proc
    quit_calls = []
    monkeypatch.setattr(rollout, "_quit_runner", lambda proc: quit_calls.append(proc))

    result = rollout.handle_stop_inference()
    assert result["success"] is True
    assert quit_calls == [runner]
    assert session.quitting is True

    status = rollout.handle_inference_status()
    assert status["phase"] == rollout.PHASE_ABORTED
    assert status["episode_results"] == ["success", "failure"]  # the cut episode isn't scored
    assert status["accuracy"] is None


def test_stop_inference_aborts_an_eval_parked_in_a_reset(monkeypatch) -> None:
    """Abort reports the partial tally and deliberately claims NO accuracy."""
    from makermodslab import rollout

    session = _arm_eval_session(monkeypatch, rollout, episodes=5, running=False, proc=None)
    session.results.extend(["success", "failure"])

    result = rollout.handle_stop_inference()
    assert result["success"] is True

    status = rollout.handle_inference_status()
    assert status["phase"] == rollout.PHASE_ABORTED
    assert status["inference_active"] is False
    assert status["episode_results"] == ["success", "failure"]
    assert status["episodes_total"] == 5
    assert status["accuracy"] is None
    assert rollout._eval_session is None


def test_stop_inference_still_ends_a_single_run_the_old_way(monkeypatch) -> None:
    """No eval session -> the historical idle-with-no-payload teardown."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "inference_active", True)
    monkeypatch.setattr(rollout, "_inference_meta", {"phase": rollout.PHASE_DOWNLOADING_MODEL})

    result = rollout.handle_stop_inference()
    assert result["success"] is True
    assert rollout.inference_active is False
    assert rollout._last_result is None


# ---------------------------------------------------------------------------
# Eval-runner line protocol + argv (makermodslab/eval_protocol.py)
#
# Pure string handling, no process anywhere near it. The runner itself is NEVER
# executed by the suite — it drives real servos.
# ---------------------------------------------------------------------------


def test_protocol_round_trips_an_event_and_its_payload() -> None:
    from makermodslab.eval_protocol import format_event, parse_event

    line = format_event("EPISODE_ENDED", "reason=duration")
    assert parse_event(line) == ("EPISODE_ENDED", "reason=duration")
    assert parse_event(format_event("READY")) == ("READY", "")


def test_protocol_collapses_a_multiline_payload_onto_one_line() -> None:
    """A traceback in an ERROR payload must not split one event across several
    lines — the reader is line-oriented and would read the tail as junk."""
    from makermodslab.eval_protocol import format_event, parse_event

    line = format_event("ERROR", "RuntimeError: boom\n  File 'x.py', line 3\n")
    assert "\n" not in line
    event, payload = parse_event(line)
    assert event == "ERROR"
    assert payload.startswith("RuntimeError: boom")


def test_protocol_ignores_ordinary_log_lines() -> None:
    from makermodslab.eval_protocol import parse_event

    assert parse_event("INFO 2026-07-31 Connecting robot (so101_follower)...\n") is None
    assert parse_event("") is None


def test_protocol_finds_an_event_appended_to_a_log_line() -> None:
    """The runner's logging handler shares the pipe; a record flushed without
    its newline must not swallow the event behind it."""
    from makermodslab.eval_protocol import parse_event

    assert parse_event("INFO some log MAKERMODSLAB-EVAL EPISODE_STARTED") == ("EPISODE_STARTED", "")


def test_protocol_reason_parsing_is_conservative() -> None:
    """An unrecognised (or renamed) reason yields "" rather than a guess: the
    orchestrator must not score an episode off a reason it doesn't understand."""
    from makermodslab.eval_protocol import parse_episode_end_reason

    assert parse_episode_end_reason("reason=stopped") == "stopped"
    assert parse_episode_end_reason("elapsed=30 reason=duration") == "duration"
    assert parse_episode_end_reason("") == ""
    assert parse_episode_end_reason("something-else") == ""


def test_unknown_episode_end_reason_scores_a_failure_not_a_success(monkeypatch) -> None:
    """Degrade the way the pre-runner code did — an episode that ended without
    the user calling it a success is a failure — rather than inventing a verdict."""
    from makermodslab import rollout

    _arm_eval_session(monkeypatch, rollout, episodes=3)
    rollout._on_episode_ended("")
    assert rollout.handle_inference_status()["episode_results"] == ["failure"]


def test_eval_runner_and_rollout_argv_share_every_flag() -> None:
    """One flag list, two entry points. A flag added for the single-episode path
    must never be missing from the eval runner's."""
    from makermodslab.rollout import _build_eval_runner_cmd, _build_rollout_cmd

    request = _eval_request(4)
    args = ("/local/pretrained_model", ["--robot.type=so101_follower", "--robot.port=/dev/ttyUSB0"])
    rollout_cmd = _build_rollout_cmd(request, *args)
    runner_cmd = _build_eval_runner_cmd(request, *args)

    assert rollout_cmd[1:3] == ["-m", "lerobot.scripts.lerobot_rollout"]
    assert runner_cmd[1:3] == ["-m", "makermodslab.eval_runner"]
    assert rollout_cmd[3:] == runner_cmd[3:]
    assert "--return_to_initial_position=true" in runner_cmd
    assert "--strategy.type=base" in runner_cmd


def test_eval_start_spawns_the_runner_with_stdin_left_open(monkeypatch, tmp_path) -> None:
    """Eval mode gets ONE long-lived runner whose stdin is the command channel;
    the single-episode path still gets `lerobot-rollout` with stdin closed."""
    from makermodslab import rollout

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(rollout, "_preflight_arm_identity", lambda *a, **k: [])
    monkeypatch.setattr(rollout, "_preflight_motor_registers", lambda *a, **k: [])
    monkeypatch.setattr(rollout, "setup_follower_calibration_file", lambda name: name)
    monkeypatch.setattr(rollout, "_resolve_policy_path", lambda ref, report=None: "/local/model")
    monkeypatch.setattr(rollout, "_detect_device", lambda: "cpu")
    monkeypatch.setattr(rollout, "_policy_ref_is_valid", lambda ref: True)
    monkeypatch.setattr(rollout.camera_preview_manager, "stop_all", lambda: None)
    monkeypatch.setattr(rollout.threading, "Thread", _SyncThread)
    monkeypatch.setattr(rollout, "_pump_runner_stdout", lambda proc, log: None)

    captured: dict = {}

    class _FakeStdin:
        def __init__(self) -> None:
            self.written = b""
            self.closed = False

        def write(self, data: bytes) -> None:
            self.written += data

        def flush(self) -> None:
            pass

        def close(self) -> None:
            self.closed = True

    class _FakeProc:
        pid = 9999

        def __init__(self, cmd, **kwargs):
            captured["cmd"] = cmd
            self.stdin = _FakeStdin()
            self.stdout = _EmptyStdout()
            captured["stdin"] = self.stdin

        def poll(self):
            return None

    monkeypatch.setattr(rollout.subprocess, "Popen", _FakeProc)

    assert rollout.handle_start_inference(_eval_request(6))["success"] is True
    assert captured["cmd"][1:3] == ["-m", "makermodslab.eval_runner"]
    # The command channel has to stay open — closing it is what a one-shot
    # rollout does, and it would make every later EPISODE unsendable.
    assert captured["stdin"].closed is False
    assert captured["stdin"].written == b"\n"
    # Episode 1 is pending: it is issued when the runner reports READY, after
    # the one-time policy load.
    assert rollout._eval_session.episode_pending is True
    assert rollout._eval_session.episode_running is False


def test_single_episode_start_still_spawns_lerobot_rollout(monkeypatch, tmp_path) -> None:
    """`eval_episodes == 1` is untouched by the redesign: same module, and stdin
    closed straight after the calibration seed."""
    from makermodslab import rollout

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(rollout, "_preflight_arm_identity", lambda *a, **k: [])
    monkeypatch.setattr(rollout, "_preflight_motor_registers", lambda *a, **k: [])
    monkeypatch.setattr(rollout, "setup_follower_calibration_file", lambda name: name)
    monkeypatch.setattr(rollout, "_resolve_policy_path", lambda ref, report=None: "/local/model")
    monkeypatch.setattr(rollout, "_detect_device", lambda: "cpu")
    monkeypatch.setattr(rollout, "_policy_ref_is_valid", lambda ref: True)
    monkeypatch.setattr(rollout.camera_preview_manager, "stop_all", lambda: None)
    monkeypatch.setattr(rollout.threading, "Thread", _SyncThread)

    captured: dict = {}

    class _FakeStdin:
        def __init__(self) -> None:
            self.written = b""
            self.closed = False

        def write(self, data: bytes) -> None:
            self.written += data

        def flush(self) -> None:
            pass

        def close(self) -> None:
            self.closed = True

    class _FakeProc:
        pid = 9999

        def __init__(self, cmd, **kwargs):
            captured["cmd"] = cmd
            self.stdin = _FakeStdin()
            self.stdout = _EmptyStdout()
            captured["stdin"] = self.stdin

        def poll(self):
            return None

    monkeypatch.setattr(rollout.subprocess, "Popen", _FakeProc)

    assert rollout.handle_start_inference(_eval_request(1))["success"] is True
    assert captured["cmd"][1:3] == ["-m", "lerobot.scripts.lerobot_rollout"]
    assert captured["stdin"].closed is True
    assert rollout._eval_session is None


def test_runner_death_before_the_first_episode_fails_the_session(monkeypatch) -> None:
    """A bad policy path / missing camera / busy bus kills the runner before any
    episode starts. That is a startup failure, not an evaluation with one bad
    episode — parking in a reset would offer a continue that can only fail the
    same way."""
    from makermodslab import rollout

    monkeypatch.setattr(rollout, "_extract_error_from_log", lambda p: "FileNotFoundError: no such checkpoint")
    session = _arm_eval_session(monkeypatch, rollout, episodes=4, running=False, proc=_FakeRunner())
    session.episode_pending = True
    runner = rollout._inference_proc

    rollout._handle_runner_exit(runner, 1)

    status = rollout.handle_inference_status()
    assert status["phase"] == rollout.PHASE_ERROR
    assert status["outcome"] == "failed"
    assert status["inference_active"] is False
    assert "FileNotFoundError" in status["error"]
    assert rollout._eval_session is None
    # Idempotent, like every other terminal payload.
    assert rollout.handle_inference_status()["phase"] == rollout.PHASE_ERROR
