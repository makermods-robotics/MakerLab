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
"""Tests for lelab.record — request schemas and handler entry points."""

from __future__ import annotations

import pytest


def test_recording_request_rejects_missing_required_fields() -> None:
    from pydantic import ValidationError

    from lelab.record import RecordingRequest

    with pytest.raises(ValidationError):
        RecordingRequest()


def test_recording_status_handler_exposes_state_fields() -> None:
    from lelab.record import handle_recording_status

    result = handle_recording_status()
    assert isinstance(result, dict)
    # Pinning the exact keys so a rename in handle_recording_status surfaces here.
    assert "recording_active" in result
    assert "current_phase" in result
    assert "session_ended" in result
    assert "available_controls" in result


def test_handle_stop_recording_when_idle_returns_dict(tmp_lerobot_home) -> None:
    from lelab.record import handle_stop_recording

    result = handle_stop_recording()
    assert isinstance(result, dict)


class _FakeWorker:
    """Thread double: reports alive until joined."""

    def __init__(self, alive: bool = True) -> None:
        self._alive = alive
        self.joined = False

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:
        self.joined = True
        self._alive = False


def test_stop_recording_during_release_grace_releases_now(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second Stop while the session-end cleanup is holding torque for the
    release grace must cut the hold short (release immediately).
    """
    import threading

    import lelab.record as record

    release_now = threading.Event()
    monkeypatch.setattr(record, "releasing", True)
    monkeypatch.setattr(record, "_release_now", release_now)

    result = record.handle_stop_recording()

    assert result["success"] is True
    assert release_now.is_set()
    assert "releasing" in result["message"].lower()


def test_stop_recording_mentions_release_grace(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first stop must tell the user about the post-session torque hold."""
    import lelab.record as record

    monkeypatch.setattr(record, "releasing", False)
    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "recording_events", {"stop_recording": False, "exit_early": False})

    result = record.handle_stop_recording()

    assert result["success"] is True
    assert "holds its pose" in result["message"]


def test_record_finish_pending_release_cuts_grace_short(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    import lelab.record as record

    worker = _FakeWorker()
    release_now = threading.Event()
    monkeypatch.setattr(record, "recording_thread", worker)
    monkeypatch.setattr(record, "releasing", True)
    monkeypatch.setattr(record, "_release_now", release_now)

    assert record.finish_pending_release() is True
    assert release_now.is_set()
    assert worker.joined is True


def test_record_finish_pending_release_leaves_live_session_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    import lelab.record as record

    worker = _FakeWorker()
    release_now = threading.Event()
    monkeypatch.setattr(record, "recording_thread", worker)
    monkeypatch.setattr(record, "releasing", False)
    monkeypatch.setattr(record, "_release_now", release_now)

    assert record.finish_pending_release() is False
    assert not release_now.is_set()
    assert worker.joined is False


def test_record_finish_pending_release_noop_when_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lelab.record as record

    monkeypatch.setattr(record, "recording_thread", None)
    assert record.finish_pending_release() is True


def test_recording_status_reports_releasing(monkeypatch: pytest.MonkeyPatch) -> None:
    import lelab.record as record

    monkeypatch.setattr(record, "releasing", True)
    status = record.handle_recording_status()
    assert status["releasing"] is True


def test_create_record_config_pins_dshow_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """On Windows, recording must use the DSHOW backend so a camera_index opens
    the same device /available-cameras enumerated (via pygrabber, DSHOW order).
    """
    import lelab.record as record
    from lerobot.cameras.configs import Cv2Backends

    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setattr(record, "setup_calibration_files", lambda leader, follower: ("leader", "follower"))

    request = record.RecordingRequest(
        leader_port="COM_LEADER",
        follower_port="COM_FOLLOWER",
        leader_config="leader",
        follower_config="follower",
        dataset_repo_id="user/dataset",
        single_task="pick up the cube",
        cameras={"wrist": {"type": "opencv", "camera_index": 0, "width": 640, "height": 480, "fps": 30}},
    )

    config = record.create_record_config(request)
    assert config.robot.cameras["wrist"].backend == Cv2Backends.DSHOW


def test_create_record_config_builds_biso_for_bimanual(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bimanual request produces a lerobot BiSO leader+follower pair config."""
    import lelab.record as record
    from lerobot.robots.bi_so_follower import BiSOFollowerConfig
    from lerobot.teleoperators.bi_so_leader import BiSOLeaderConfig

    monkeypatch.setattr(record, "setup_calibration_files", lambda leader, follower: (leader, follower))

    # Configs follow lerobot's "<base>_left"/"<base>_right" convention.
    request = record.RecordingRequest(
        leader_port="/dev/ll", follower_port="/dev/lf",
        leader_config="mybot_left", follower_config="mybot_left",
        mode="bimanual",
        right_leader_port="/dev/rl", right_follower_port="/dev/rf",
        right_leader_config="mybot_right", right_follower_config="mybot_right",
        dataset_repo_id="user/dataset", single_task="pick up the cube",
    )

    config = record.create_record_config(request)
    assert isinstance(config.robot, BiSOFollowerConfig)
    assert isinstance(config.teleop, BiSOLeaderConfig)
    # BiSO id is the convention base so lerobot auto-loads "<base>_left/right.json".
    assert config.robot.id == "mybot"
    assert config.teleop.id == "mybot"
    assert config.robot.right_arm_config.port == "/dev/rf"


def test_build_camera_configs_uses_default_backend_when_unset() -> None:
    from lelab.record import _build_camera_configs
    from lerobot.cameras.configs import Cv2Backends

    cameras = {"cam": {"type": "opencv", "camera_index": 0, "width": 640, "height": 480, "fps": 30}}
    configs = _build_camera_configs(cameras, Cv2Backends.AVFOUNDATION)

    assert configs["cam"].backend == Cv2Backends.AVFOUNDATION
    assert configs["cam"].fourcc is None
    assert configs["cam"].index_or_path == 0


def test_build_camera_configs_passes_fourcc_through() -> None:
    from lelab.record import _build_camera_configs
    from lerobot.cameras.configs import Cv2Backends

    cameras = {"cam": {"type": "opencv", "camera_index": 0, "fourcc": "MJPG"}}
    configs = _build_camera_configs(cameras, Cv2Backends.ANY)

    assert configs["cam"].fourcc == "MJPG"


def test_build_camera_configs_explicit_backend_overrides_default() -> None:
    from lelab.record import _build_camera_configs
    from lerobot.cameras.configs import Cv2Backends

    cameras = {"cam": {"type": "opencv", "camera_index": 0, "backend": "V4L2"}}
    configs = _build_camera_configs(cameras, Cv2Backends.AVFOUNDATION)

    assert configs["cam"].backend == Cv2Backends.V4L2


def test_build_camera_configs_invalid_backend_raises() -> None:
    from lelab.record import _build_camera_configs
    from lerobot.cameras.configs import Cv2Backends

    cameras = {"cam": {"type": "opencv", "camera_index": 0, "backend": "NOPE"}}
    with pytest.raises(KeyError):
        _build_camera_configs(cameras, Cv2Backends.ANY)


def test_build_camera_configs_skips_non_opencv_type() -> None:
    from lelab.record import _build_camera_configs
    from lerobot.cameras.configs import Cv2Backends

    cameras = {"cam": {"type": "realsense", "camera_index": 0}}
    configs = _build_camera_configs(cameras, Cv2Backends.ANY)

    assert configs == {}


def test_record_start_clears_stale_release_state_from_previous_double_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cross-session leak regression (mirrors the teleoperation test): a stale
    _release_now from a previous session's double-stop must be cleared under
    the state lock when a new recording claims the active flag — otherwise
    every later release grace is cut short instantly until a server restart."""
    import threading

    import lelab.record as record
    import lelab.teleoperate as teleop

    stale = threading.Event()
    stale.set()
    monkeypatch.setattr(record, "recording_active", False)
    monkeypatch.setattr(record, "recording_thread", None)
    monkeypatch.setattr(record, "_release_now", stale)
    monkeypatch.setattr(record, "releasing", True)
    # Teleop side idle so the cross-module pending-release check no-ops.
    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(teleop, "teleoperation_thread", None)

    # Fail fast AFTER the locked reset, before any hardware is touched.
    def _boom(request):
        raise RuntimeError("stop before hardware")

    monkeypatch.setattr(record, "create_record_config", _boom)

    result = record.handle_start_recording(
        record.RecordingRequest(
            leader_port="COM_LEADER",
            follower_port="COM_FOLLOWER",
            leader_config="leader",
            follower_config="follower",
            dataset_repo_id="tester/dataset",
            single_task="pick",
        )
    )

    # The start fails, but the per-session reset already ran under the lock.
    assert result["success"] is False
    assert not stale.is_set()
    assert record.releasing is False
    assert record.recording_active is False
