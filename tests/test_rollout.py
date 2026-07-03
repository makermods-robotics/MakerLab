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
"""Tests for lelab.rollout — request schema, pure helpers, and the
non-subprocess branches of the start/stop/status handlers.

handle_start_inference's happy path spawns a real subprocess and a stdout-
pumping thread; covering it would require mocking subprocess.Popen, threading,
and setup_follower_calibration_file. We test only the early-return mutex
branches here — the parts that matter for safety."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_rollout_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset rollout's module-level state around each test so a leaking
    `inference_active=True` from one case can't poison the next."""
    from lelab import rollout

    monkeypatch.setattr(rollout, "inference_active", False)
    monkeypatch.setattr(rollout, "_inference_proc", None)
    monkeypatch.setattr(rollout, "_inference_started_at", None)
    monkeypatch.setattr(rollout, "_inference_rollout_started_at", None)
    monkeypatch.setattr(rollout, "_inference_meta", {})


def test_inference_request_rejects_missing_required_fields() -> None:
    from pydantic import ValidationError

    from lelab.rollout import InferenceRequest

    with pytest.raises(ValidationError):
        InferenceRequest()


def test_inference_request_has_expected_defaults() -> None:
    from lelab.rollout import InferenceRequest

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
    from lelab.rollout import InferenceRequest

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
    from lelab.rollout import InferenceRequest

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


# ---------------------------------------------------------------------------
# _arm_count_mismatch — the pre-spawn checkpoint/robot arm-count guard
# ---------------------------------------------------------------------------


def test_arm_count_mismatch_none_when_state_dim_unknown() -> None:
    """A checkpoint with no observation.state (state_dim None) can't be judged
    cheaply — defer to the subprocess's own shape check."""
    from lelab.rollout import _arm_count_mismatch

    assert _arm_count_mismatch("single", None) is None
    assert _arm_count_mismatch("bimanual", None) is None


def test_arm_count_mismatch_none_when_single_matches_single() -> None:
    from lelab.rollout import _arm_count_mismatch

    assert _arm_count_mismatch("single", 6) is None


def test_arm_count_mismatch_none_when_bimanual_matches_bimanual() -> None:
    from lelab.rollout import _arm_count_mismatch

    assert _arm_count_mismatch("bimanual", 12) is None


def test_arm_count_mismatch_flags_bimanual_checkpoint_on_single_robot() -> None:
    from lelab.rollout import _arm_count_mismatch

    msg = _arm_count_mismatch("single", 12)
    assert msg is not None
    assert "bimanual" in msg
    assert "single-arm" in msg


def test_arm_count_mismatch_flags_single_checkpoint_on_bimanual_robot() -> None:
    from lelab.rollout import _arm_count_mismatch

    msg = _arm_count_mismatch("bimanual", 6)
    assert msg is not None
    assert "single-arm" in msg
    assert "bimanual" in msg


def test_arm_count_mismatch_none_for_unrecognised_width() -> None:
    """A width that's neither a single arm nor a clean multiple is left to the
    subprocess rather than guessed at (e.g. 7 = 6 + an extra sensor dim)."""
    from lelab.rollout import _arm_count_mismatch

    assert _arm_count_mismatch("single", 7) is None
    assert _arm_count_mismatch("bimanual", 7) is None


def test_detect_device_returns_cpu_when_neither_cuda_nor_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from lelab.rollout import _detect_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert _detect_device() == "cpu"


def test_detect_device_prefers_cuda_over_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from lelab.rollout import _detect_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert _detect_device() == "cuda"


def test_detect_device_falls_back_to_mps_when_no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from lelab.rollout import _detect_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert _detect_device() == "mps"


def test_detect_device_returns_cpu_when_torch_probe_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """The function wraps both probes in a broad try/except — if torch is
    broken at runtime we still need a sensible fallback."""
    import torch

    from lelab.rollout import _detect_device

    def _boom() -> bool:
        raise RuntimeError("simulated torch.cuda failure")

    monkeypatch.setattr(torch.cuda, "is_available", _boom)
    assert _detect_device() == "cpu"


def test_resolve_policy_path_returns_local_dir_unchanged(tmp_path) -> None:
    from lelab.rollout import _resolve_policy_path

    pretrained = tmp_path / "pretrained_model"
    pretrained.mkdir()
    assert _resolve_policy_path(str(pretrained)) == str(pretrained)


def test_resolve_policy_path_raises_on_unparsable_ref() -> None:
    from lelab.rollout import _resolve_policy_path

    with pytest.raises(ValueError, match="Unrecognised policy ref"):
        _resolve_policy_path("not-a-real-ref-no-at-sign")


def test_resolve_policy_path_resolves_hub_ref(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Hub refs ('user/repo@checkpoints/000050') must be passed through
    snapshot_download and joined to the standard checkpoints/<step>/pretrained_model
    layout."""
    from lelab.rollout import _resolve_policy_path

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
    """A flat-model ref ('user/repo@root') downloads the whole repo and
    returns its root."""
    from lelab.rollout import _resolve_policy_path

    fake_root = tmp_path / "snapshot"
    fake_root.mkdir()
    seen = {}

    def fake_snapshot_download(**kwargs):
        seen.update(kwargs)
        return str(fake_root)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    result = _resolve_policy_path("user/repo@root")
    assert seen["repo_id"] == "user/repo"
    assert "allow_patterns" not in seen
    assert result == str(fake_root)


def test_format_cameras_arg_empty_yields_empty_braces() -> None:
    from lelab.rollout import _format_cameras_arg

    assert _format_cameras_arg({}) == "{}"


def test_format_cameras_arg_renames_camera_index_to_index_or_path() -> None:
    """lerobot's CLI expects `index_or_path`, but the frontend posts
    `camera_index`. The rename is the whole point of this helper."""
    from lelab.rollout import _format_cameras_arg

    result = _format_cameras_arg(
        {"front": {"type": "opencv", "camera_index": 0, "width": 640, "height": 480, "fps": 30}}
    )
    assert "index_or_path: 0" in result
    assert "camera_index" not in result
    assert result.startswith("{front: {")
    assert result.endswith("}}")


def test_format_cameras_arg_omits_none_values() -> None:
    from lelab.rollout import _format_cameras_arg

    result = _format_cameras_arg({"front": {"camera_index": 0, "fps": None}})
    assert "fps" not in result
    assert "index_or_path: 0" in result


def test_format_cameras_arg_handles_multiple_cameras() -> None:
    from lelab.rollout import _format_cameras_arg

    result = _format_cameras_arg(
        {
            "front": {"camera_index": 0, "fps": 30},
            "wrist": {"camera_index": 1, "fps": 30},
        }
    )
    assert "front: {" in result
    assert "wrist: {" in result


def test_handle_stop_inference_when_idle_returns_409() -> None:
    from lelab.rollout import handle_stop_inference

    result = handle_stop_inference()
    assert result["success"] is False
    assert result["status_code"] == 409


def test_handle_inference_status_when_idle_returns_dict_with_expected_keys() -> None:
    from lelab.rollout import handle_inference_status

    result = handle_inference_status()
    assert isinstance(result, dict)
    assert result["inference_active"] is False
    for key in ("started_at", "rollout_started_at", "elapsed_s", "rollout_elapsed_s"):
        assert key in result


def _stub_request():
    from lelab.rollout import InferenceRequest

    return InferenceRequest(
        follower_port="/dev/ttyUSB0",
        follower_config="robot_a",
        policy_ref="user/repo@checkpoints/000050",
    )


def test_handle_start_inference_blocked_when_teleoperation_active(monkeypatch) -> None:
    """If teleop owns the bus, inference must refuse rather than race for
    the serial port."""
    from lelab.rollout import handle_start_inference

    monkeypatch.setattr("lelab.teleoperate.teleoperation_active", True)
    result = handle_start_inference(_stub_request())
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "Teleoperation" in result["message"]


def test_handle_start_inference_blocked_when_recording_active(monkeypatch) -> None:
    from lelab.rollout import handle_start_inference

    monkeypatch.setattr("lelab.record.recording_active", True)
    result = handle_start_inference(_stub_request())
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "Recording" in result["message"]


def test_handle_start_inference_blocked_when_already_active(monkeypatch) -> None:
    from lelab import rollout

    monkeypatch.setattr(rollout, "inference_active", True)
    result = rollout.handle_start_inference(_stub_request())
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "already active" in result["message"]


def test_handle_start_inference_pins_return_to_initial_position(monkeypatch, tmp_path) -> None:
    """The stop dialog promises the follower eases back to its start pose on
    teardown. That behaviour is lerobot's `return_to_initial_position`, which
    defaults to True today — but we pin it explicitly so an upstream default
    flip can't silently break the promise. Capture the rollout command and
    assert the flag is present.

    This is the one command-construction test: it stubs out the subprocess,
    the stdout pump, and every hardware-touching preflight so nothing real is
    started — we only inspect the argv handed to Popen."""
    from lelab import rollout

    monkeypatch.setattr(rollout, "setup_follower_calibration_file", lambda cfg: cfg)
    monkeypatch.setattr(rollout, "_preflight_arm_identity", lambda *a, **k: [])
    monkeypatch.setattr(rollout, "_preflight_motor_power", lambda *a, **k: [])
    monkeypatch.setattr(rollout, "_resolve_policy_path", lambda ref: str(tmp_path / "pretrained_model"))
    monkeypatch.setattr(rollout, "_detect_device", lambda: "cpu")

    captured: dict = {}

    class _FakeProc:
        pid = 4321
        stdin = None

        def __init__(self, cmd, **kwargs):
            captured["cmd"] = cmd
            # Provide a stdin object so the newline-seeding block runs cleanly.
            import io

            self.stdin = io.BytesIO()

    monkeypatch.setattr(rollout.subprocess, "Popen", _FakeProc)
    # Don't spawn the real stdout-pump thread.
    monkeypatch.setattr(
        rollout.threading, "Thread", lambda *a, **k: type("_T", (), {"start": lambda self: None})()
    )

    result = rollout.handle_start_inference(_stub_request())
    assert result["success"] is True, result

    cmd = captured["cmd"]
    assert "--return_to_initial_position=true" in cmd
    # Sanity: the core rollout invocation is intact around our pinned flag.
    assert "lerobot.scripts.lerobot_rollout" in cmd
    assert "--strategy.type=base" in cmd


# ---------------------------------------------------------------------------
# --robot.* arg construction — single vs bimanual (pure, no I/O)
# ---------------------------------------------------------------------------


def _bimanual_request():
    from lelab.rollout import InferenceRequest

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
    from lelab.rollout import _single_robot_args

    args = _single_robot_args(_stub_request(), "robot_a")
    assert "--robot.type=so101_follower" in args
    assert "--robot.port=/dev/ttyUSB0" in args
    assert "--robot.id=robot_a" in args
    # No cameras on the stub request → no --robot.cameras arg.
    assert not any(a.startswith("--robot.cameras=") for a in args)


def test_single_robot_args_appends_cameras_when_present() -> None:
    from lelab.rollout import InferenceRequest, _single_robot_args

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
    from lelab.rollout import _bimanual_robot_args

    args = _bimanual_robot_args(_bimanual_request(), "dual_arm", "/staging/follower")
    assert "--robot.type=bi_so_follower" in args
    assert "--robot.id=dual_arm" in args
    assert "--robot.calibration_dir=/staging/follower" in args
    assert "--robot.left_arm_config.port=/dev/left" in args
    assert "--robot.right_arm_config.port=/dev/right" in args


def test_bimanual_robot_args_puts_cameras_on_left_arm_only() -> None:
    from lelab.rollout import InferenceRequest, _bimanual_robot_args

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
    from lelab.rollout import _build_rollout_cmd

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
    from lelab.rollout import InferenceRequest, handle_start_inference

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
    from lelab.rollout import InferenceRequest, handle_start_inference

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
    from lelab import rollout

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

    Mirrors the pin-test's stub pattern: subprocess, the stdout pump, the two
    preflights, and the staging helper are all replaced so nothing real runs."""
    from lelab import rollout

    monkeypatch.setattr(rollout, "bimanual_base_id", lambda name: "dual_arm")
    monkeypatch.setattr(
        rollout,
        "stage_bimanual_follower_calibrations",
        lambda *a, **k: ("/staging/follower", "dual_arm"),
    )
    monkeypatch.setattr(rollout, "_preflight_arm_identity", lambda *a, **k: [])
    monkeypatch.setattr(rollout, "_preflight_motor_power", lambda *a, **k: [])
    monkeypatch.setattr(rollout, "_resolve_policy_path", lambda ref: str(tmp_path / "pretrained_model"))
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
            captured["stdin"] = self.stdin

    monkeypatch.setattr(rollout.subprocess, "Popen", _FakeProc)
    monkeypatch.setattr(
        rollout.threading, "Thread", lambda *a, **k: type("_T", (), {"start": lambda self: None})()
    )

    result = rollout.handle_start_inference(_bimanual_request())
    assert result["success"] is True, result

    cmd = captured["cmd"]
    assert "--robot.type=bi_so_follower" in cmd
    assert "--robot.left_arm_config.port=/dev/left" in cmd
    assert "--robot.right_arm_config.port=/dev/right" in cmd
    assert "--robot.calibration_dir=/staging/follower" in cmd
    # Two sub-arms → two seeded newlines (single-arm seeds only one).
    assert captured["stdin"].written == b"\n\n"
