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


def test_stop_recording_mentions_rest_pose_return(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first stop must tell the user the arm returns to its starting
    position, then goes limp — no timed hold anymore (same as teleop)."""
    import lelab.record as record

    monkeypatch.setattr(record, "releasing", False)
    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "recording_events", {"stop_recording": False, "exit_early": False})

    result = record.handle_stop_recording()

    assert result["success"] is True
    assert "returns to its starting position" in result["message"]
    assert "holds its pose" not in result["message"]  # the timed hold is gone
    assert "Stop again" in result["message"]


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
    """During the post-stop return the status must say the arm is still
    energized and going home (releasing) rather than pretending the session
    is fully over."""
    import lelab.record as record

    monkeypatch.setattr(record, "releasing", True)
    status = record.handle_recording_status()
    assert status["releasing"] is True
    assert "returning the arm" in status["message"].lower()


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
    """A bimanual request stages the four arbitrarily-named library configs and
    builds a BiSO pair pointed at the per-device staging dirs."""
    import lelab.record as record
    from lerobot.robots.bi_so_follower import BiSOFollowerConfig
    from lerobot.teleoperators.bi_so_leader import BiSOLeaderConfig

    staged: dict = {}

    def _fake_stage(base, leader_left, leader_right, follower_left, follower_right):
        staged.update(
            base=base,
            leader=(leader_left, leader_right),
            follower=(follower_left, follower_right),
        )
        return (f"/staging/{base}/leader", f"/staging/{base}/follower", base)

    monkeypatch.setattr(record, "stage_bimanual_calibrations", _fake_stage)

    # Config names are ARBITRARY — no "<base>_left/right" convention required.
    request = record.RecordingRequest(
        leader_port="/dev/ll",
        follower_port="/dev/lf",
        leader_config="alice",
        follower_config="bob",
        mode="bimanual",
        right_leader_port="/dev/rl",
        right_follower_port="/dev/rf",
        right_leader_config="carol",
        right_follower_config="dave",
        robot_name="mybot",
        dataset_repo_id="user/dataset",
        single_task="pick up the cube",
    )

    config = record.create_record_config(request)
    assert isinstance(config.robot, BiSOFollowerConfig)
    assert isinstance(config.teleop, BiSOLeaderConfig)
    # BiSO id + calibration_dir come from the staging helper (base = robot name).
    assert config.robot.id == "mybot"
    assert config.teleop.id == "mybot"
    assert str(config.robot.calibration_dir) == "/staging/mybot/follower"
    assert str(config.teleop.calibration_dir) == "/staging/mybot/leader"
    assert config.robot.right_arm_config.port == "/dev/rf"
    # Helper received the four library stems, grouped per device.
    assert staged["base"] == "mybot"
    assert staged["leader"] == ("alice", "carol")
    assert staged["follower"] == ("bob", "dave")


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


def _make_dataset_dir(cache, repo_id: str, total_episodes: int):
    """Create a minimal on-disk LeRobot dataset dir (meta/info.json) under the
    tmp cache root, plus a fake video file so 'removed' is observable."""
    import json
    from pathlib import Path

    target = Path(cache) / repo_id
    (target / "meta").mkdir(parents=True, exist_ok=True)
    (target / "meta" / "info.json").write_text(json.dumps({"total_episodes": total_episodes}))
    (target / "videos").mkdir(parents=True, exist_ok=True)
    (target / "videos" / "ep.mp4").write_bytes(b"\x00" * 1024)
    return target


def test_discard_empty_dataset_removes_zero_episode_dir(tmp_lerobot_home) -> None:
    """A non-resume session that saved zero episodes has its directory removed."""
    import lelab.record as record

    target = _make_dataset_dir(tmp_lerobot_home, "tester/big_20260703_120000", total_episodes=0)
    assert target.exists()

    removed = record._discard_empty_dataset("tester/big_20260703_120000", resume=False)

    assert removed is True
    assert not target.exists()


def test_discard_empty_dataset_keeps_nonempty_dir(tmp_lerobot_home) -> None:
    """A directory that recorded >=1 episode is never removed."""
    import lelab.record as record

    target = _make_dataset_dir(tmp_lerobot_home, "tester/good_20260703_120000", total_episodes=3)

    removed = record._discard_empty_dataset("tester/good_20260703_120000", resume=False)

    assert removed is False
    assert target.exists()


def test_discard_empty_dataset_never_touches_resume_session(tmp_lerobot_home) -> None:
    """A resume/append session writes into a pre-existing dataset — even at zero
    NEW episodes on disk, the directory must never be removed."""
    import lelab.record as record

    target = _make_dataset_dir(tmp_lerobot_home, "tester/preexisting", total_episodes=0)

    removed = record._discard_empty_dataset("tester/preexisting", resume=True)

    assert removed is False
    assert target.exists()


def test_discard_empty_dataset_rejects_path_traversal(tmp_lerobot_home) -> None:
    """A repo_id escaping the cache root is refused (no deletion outside cache)."""
    import lelab.record as record

    removed = record._discard_empty_dataset("../../etc", resume=False)
    assert removed is False


def test_discard_empty_dataset_invalidates_hub_status(tmp_lerobot_home) -> None:
    """Removing an empty dataset drops any cached Hub-existence probe for it."""
    import lelab.datasets as datasets
    import lelab.record as record

    _make_dataset_dir(tmp_lerobot_home, "tester/probed_20260703", total_episodes=0)
    # Seed a cached probe answer for the repo id.
    with datasets._HUB_STATUS_LOCK:
        datasets._HUB_STATUS_CACHE["tester/probed_20260703"] = "local_only"

    assert record._discard_empty_dataset("tester/probed_20260703", resume=False) is True

    with datasets._HUB_STATUS_LOCK:
        assert "tester/probed_20260703" not in datasets._HUB_STATUS_CACHE


def test_recording_status_reports_discarded_empty_at_session_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the ended session discarded its empty dataset, the status payload
    tells the frontend honestly that nothing was kept."""
    import lelab.record as record

    monkeypatch.setattr(record, "recording_active", False)
    monkeypatch.setattr(record, "current_phase", "completed")
    monkeypatch.setattr(record, "last_session_discarded_empty", True)

    status = record.handle_recording_status()

    assert status["session_ended"] is True
    assert status["discarded_empty"] is True


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


# ---------------------------------------------------------------------------
# Rest-pose return on session end (mirrors the teleop stop-path integration).
# record_with_web_events captures each follower's pose at session start and, on
# a NORMAL end, drives it back before releasing torque — same helpers as teleop
# (lelab.rest_pose, lelab.teleoperate._return_followers_to_rest), so the shared
# return logic itself is covered in tests/test_teleoperate.py. These tests pin
# record's own finally-block wiring: normal end returns then releases, a
# double-stop skips the return, an error skips it, the pose is captured per
# follower, and the gripper is excluded.
# ---------------------------------------------------------------------------


class _RecReturnBus:
    """Follower bus double: serves capture_rest_pose and records nothing else
    (the return itself is spied via _return_followers_to_rest)."""

    _MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    def __init__(self, positions: dict[str, int] | None = None, port: str = "COM_FOLLOWER") -> None:
        self.port = port
        self.motors = dict.fromkeys(self._MOTORS)
        self.positions = dict.fromkeys(self._MOTORS, 1000) if positions is None else dict(positions)

    def sync_read(self, reg: str, normalize: bool = True) -> dict:
        assert reg == "Present_Position" and normalize is False
        return dict(self.positions)


class _RecRobot:
    """Follower robot double exposing one .bus for _device_buses/capture."""

    def __init__(self, bus: _RecReturnBus) -> None:
        self.bus = bus
        self.disconnected = False

    def disconnect(self) -> None:
        self.disconnected = True


def _run_record_session(
    monkeypatch: pytest.MonkeyPatch,
    robot: _RecRobot,
    *,
    stop_events: dict | None = None,
    raise_in_loop: bool = False,
    preset_release_now: bool = False,
):
    """Drive record_with_web_events with every lerobot dependency mocked so no
    real hardware, dataset, or record_loop runs. Returns the spy call log for
    _return_followers_to_rest (the rest-pose return) and the robot.

    The loop runs a single episode: record_loop sets `stop_recording` (via the
    supplied events) so the session ends normally after one save, unless
    `raise_in_loop` makes record_loop raise (the error path)."""
    import lelab.record as record

    return_calls: list[tuple] = []

    def _spy_return(rest_poses, abort_event):
        return_calls.append((list(rest_poses), abort_event))

    monkeypatch.setattr(record, "_return_followers_to_rest", _spy_return)
    monkeypatch.setattr(record, "force_disable_torque", lambda device, label="": [])
    monkeypatch.setattr(record, "apply_motor_power", lambda *a, **k: [])
    monkeypatch.setattr(record, "clear_goal_velocity", lambda *a, **k: [])
    monkeypatch.setattr(record, "verify_devices", lambda *a, **k: [])

    if preset_release_now:
        record._release_now.set()
    else:
        record._release_now.clear()

    # lerobot symbols resolved at call time inside record_with_web_events.
    monkeypatch.setattr("lerobot.robots.make_robot_from_config", lambda cfg: robot, raising=False)
    monkeypatch.setattr(
        "lerobot.teleoperators.make_teleoperator_from_config", lambda cfg: None, raising=False
    )
    monkeypatch.setattr(
        "lerobot.processor.make_default_processors", lambda: (None, None, None), raising=False
    )
    monkeypatch.setattr(
        "lerobot.utils.feature_utils.hw_to_dataset_features", lambda *a, **k: {}, raising=False
    )
    monkeypatch.setattr("lerobot.utils.utils.log_say", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(
        "lerobot.common.control_utils.sanity_check_dataset_name", lambda *a, **k: None, raising=False
    )

    def _fake_record_loop(*args, **kwargs):
        if raise_in_loop:
            raise RuntimeError("bus died mid-episode")
        events = kwargs.get("events")
        if events is not None and kwargs.get("dataset") is not None:
            events.update(stop_events or {"stop_recording": True, "_exit_early_triggered": True})

    monkeypatch.setattr("lerobot.scripts.lerobot_record.record_loop", _fake_record_loop, raising=False)

    dataset_calls: list[str] = []

    class _FakeDataset:
        num_episodes = 1
        num_frames = 1
        fps = 30
        features = {"action": None}
        meta = type("M", (), {"robot_type": "so101"})()

        @staticmethod
        def create(*args, **kwargs):
            return _FakeDataset()

        def save_episode(self) -> None:
            dataset_calls.append("save_episode")

        def clear_episode_buffer(self) -> None:
            dataset_calls.append("clear_episode_buffer")

    monkeypatch.setattr("lerobot.datasets.LeRobotDataset", _FakeDataset, raising=False)

    # robot.connect(calibrate=False) is called on the double.
    robot.connect = lambda **kwargs: None  # type: ignore[attr-defined]
    robot.name = "so101"  # type: ignore[attr-defined]
    robot.cameras = {}  # type: ignore[attr-defined]
    robot.action_features = {}  # type: ignore[attr-defined]
    robot.observation_features = {}  # type: ignore[attr-defined]
    robot.calibration = {}  # type: ignore[attr-defined]

    cfg = record.create_record_config(
        record.RecordingRequest(
            leader_port="COM_LEADER",
            follower_port="COM_FOLLOWER",
            leader_config="leader",
            follower_config="follower",
            dataset_repo_id="tester/ds",
            single_task="pick",
            num_episodes=1,
            video=False,
        )
    )
    # No teleop device: keep the return path follower-only and simple.
    cfg.teleop = None

    web_events = {"exit_early": False, "stop_recording": False, "rerecord_episode": False}
    error: Exception | None = None
    try:
        record.record_with_web_events(cfg, web_events)
    except Exception as e:  # the error-path test expects this
        error = e
    return return_calls, robot, error, dataset_calls


def test_record_normal_end_returns_then_releases(monkeypatch: pytest.MonkeyPatch, tmp_lerobot_home) -> None:
    """A normal session end drives the follower back to its captured start pose
    (once), then disconnects — same as teleop's stop, no timed hold."""
    import lelab.record as record

    monkeypatch.setattr(record, "setup_calibration_files", lambda leader, follower: ("leader", "follower"))
    bus = _RecReturnBus(positions=dict.fromkeys(_RecReturnBus._MOTORS, 1500))
    robot = _RecRobot(bus)

    return_calls, robot, error, _dataset_calls = _run_record_session(monkeypatch, robot)

    assert error is None
    assert len(return_calls) == 1  # the return ran exactly once
    assert robot.disconnected is True
    assert record.releasing is False  # reset in the finally


def test_record_captures_pose_per_follower_excluding_gripper(
    monkeypatch: pytest.MonkeyPatch, tmp_lerobot_home
) -> None:
    """The captured pose is the follower's raw ticks with the gripper removed
    (it may be holding an object at stop time)."""
    import lelab.record as record

    monkeypatch.setattr(record, "setup_calibration_files", lambda leader, follower: ("leader", "follower"))
    positions = {
        "shoulder_pan": 1111,
        "shoulder_lift": 2222,
        "elbow_flex": 3333,
        "wrist_flex": 4444,
        "wrist_roll": 5555,
        "gripper": 9999,
    }
    robot = _RecRobot(_RecReturnBus(positions=positions))

    return_calls, _robot, error, _dataset_calls = _run_record_session(monkeypatch, robot)

    assert error is None
    (rest_poses, _abort) = return_calls[0]
    assert len(rest_poses) == 1  # one follower bus
    captured_bus, captured_pose = rest_poses[0]
    assert captured_bus is robot.bus
    assert "gripper" not in captured_pose  # excluded — may be holding an object
    assert captured_pose == {k: v for k, v in positions.items() if k != "gripper"}


def test_record_double_stop_skips_the_return(monkeypatch: pytest.MonkeyPatch, tmp_lerobot_home) -> None:
    """A second stop (release-now) set before the session-end cleanup runs must
    skip the return and release immediately, mirroring teleop."""
    import lelab.record as record

    monkeypatch.setattr(record, "setup_calibration_files", lambda leader, follower: ("leader", "follower"))
    robot = _RecRobot(_RecReturnBus())

    return_calls, robot, error, _dataset_calls = _run_record_session(
        monkeypatch, robot, preset_release_now=True
    )

    assert error is None
    assert return_calls == []  # release-now skipped the return
    assert robot.disconnected is True


def test_record_error_path_skips_return_and_releases(
    monkeypatch: pytest.MonkeyPatch, tmp_lerobot_home
) -> None:
    """An exception in the loop (dead bus) skips the return entirely — the bus
    may be gone, so release ASAP — but still disconnects."""
    import lelab.record as record

    monkeypatch.setattr(record, "setup_calibration_files", lambda leader, follower: ("leader", "follower"))
    robot = _RecRobot(_RecReturnBus())

    return_calls, robot, error, _dataset_calls = _run_record_session(monkeypatch, robot, raise_in_loop=True)

    assert isinstance(error, RuntimeError)
    assert return_calls == []  # error path never returns to rest
    assert robot.disconnected is True


def test_stop_during_recording_phase_discards_episode_no_reset(
    monkeypatch: pytest.MonkeyPatch, tmp_lerobot_home
) -> None:
    """Stop pressed mid-episode (stop_recording set, but NOT _exit_early_triggered
    — exactly what handle_stop_recording produces) must discard the in-progress
    episode (clear_episode_buffer, never save_episode) and end the session
    immediately, with no reset detour, then return to rest and disconnect once."""
    import lelab.record as record

    monkeypatch.setattr(record, "setup_calibration_files", lambda leader, follower: ("leader", "follower"))
    robot = _RecRobot(_RecReturnBus())

    # No _exit_early_triggered: the pre-fix classification would have called this
    # a timeout, flipped rerecord on, and run a reset phase before honoring stop.
    return_calls, robot, error, dataset_calls = _run_record_session(
        monkeypatch, robot, stop_events={"stop_recording": True}
    )

    assert error is None
    assert dataset_calls == ["clear_episode_buffer"]  # discarded, never saved
    assert record.current_phase == "completed"  # not "resetting" — no reset detour
    assert len(return_calls) == 1  # rest-pose return ran once
    assert robot.disconnected is True


def test_stop_wins_over_skip_when_both_set_in_same_episode(
    monkeypatch: pytest.MonkeyPatch, tmp_lerobot_home
) -> None:
    """When stop_recording AND _exit_early_triggered land in the same episode,
    stop wins: the short-circuit is checked FIRST, so the episode is discarded,
    not saved. (Stop is a deliberate 'end now, drop this take' action.)"""
    import lelab.record as record

    monkeypatch.setattr(record, "setup_calibration_files", lambda leader, follower: ("leader", "follower"))
    robot = _RecRobot(_RecReturnBus())

    return_calls, robot, error, dataset_calls = _run_record_session(
        monkeypatch, robot, stop_events={"stop_recording": True, "_exit_early_triggered": True}
    )

    assert error is None
    assert dataset_calls == ["clear_episode_buffer"]  # stop precedence: discard, not save
    assert record.current_phase == "completed"


# ---------------------------------------------------------------------------
# UploadManager — background dataset upload (start → running → done | error).
# The push runs in a worker thread; tests mock LeRobotDataset so no real Hub
# call happens, then join the thread before asserting on the final state.
# ---------------------------------------------------------------------------


def _fake_dataset(num_episodes: int = 3, push=None):
    from unittest.mock import MagicMock

    ds = MagicMock(name="LeRobotDataset")
    ds.num_episodes = num_episodes
    if push is not None:
        ds.push_to_hub = push
    return ds


def _join_upload(mgr, timeout: float = 5.0) -> None:
    thread = mgr._thread
    if thread is not None:
        thread.join(timeout=timeout)


def test_upload_manager_start_runs_and_completes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A start pushes in a worker thread and lands in state "done" with the
    dataset_url, invalidating the cached hub status."""
    from lelab.record import UploadManager, UploadRequest

    ds = _fake_dataset()
    monkeypatch.setattr("lerobot.datasets.LeRobotDataset", lambda repo_id: ds)
    invalidated: list[str] = []
    monkeypatch.setattr("lelab.record.invalidate_hub_status", invalidated.append)

    mgr = UploadManager()
    # Nothing else is writing this dataset — _dataset_in_use must return None.
    monkeypatch.setattr("lelab.datasets._dataset_in_use", lambda repo_id: None)

    result = mgr.start(UploadRequest(dataset_repo_id="tester/ds", tags=["x"], private=True))
    assert result == {"started": True, "repo_id": "tester/ds", "message": "Upload started"}

    _join_upload(mgr)
    status = mgr.get_status()
    assert status["state"] == "done"
    assert status["repo_id"] == "tester/ds"
    assert status["dataset_url"] == "https://huggingface.co/datasets/tester/ds"
    assert invalidated == ["tester/ds"]
    # push_to_hub got the lelab-tagged tags + private flag.
    ds.push_to_hub.assert_called_once()
    kwargs = ds.push_to_hub.call_args.kwargs
    assert kwargs["private"] is True
    assert "x" in kwargs["tags"]


def test_upload_manager_error_maps_auth_friendly(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 401 during push lands in state "error" with the friendly login message
    and the docs_url, not a raw traceback string."""
    from lelab.record import UploadManager, UploadRequest

    def _raise_401(**kwargs):
        raise RuntimeError("401 Client Error: you must be authenticated")

    ds = _fake_dataset(push=_raise_401)
    monkeypatch.setattr("lerobot.datasets.LeRobotDataset", lambda repo_id: ds)
    monkeypatch.setattr("lelab.datasets._dataset_in_use", lambda repo_id: None)

    mgr = UploadManager()
    mgr.start(UploadRequest(dataset_repo_id="tester/ds"))
    _join_upload(mgr)

    status = mgr.get_status()
    assert status["state"] == "error"
    assert "hf auth login" in status["message"]
    assert status["docs_url"].startswith("https://huggingface.co/docs")


def test_upload_manager_error_generic_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-auth failure surfaces its message without a docs_url."""
    from lelab.record import UploadManager, UploadRequest

    def _boom(**kwargs):
        raise RuntimeError("disk exploded")

    ds = _fake_dataset(push=_boom)
    monkeypatch.setattr("lerobot.datasets.LeRobotDataset", lambda repo_id: ds)
    monkeypatch.setattr("lelab.datasets._dataset_in_use", lambda repo_id: None)

    mgr = UploadManager()
    mgr.start(UploadRequest(dataset_repo_id="tester/ds"))
    _join_upload(mgr)

    status = mgr.get_status()
    assert status["state"] == "error"
    assert "disk exploded" in status["message"]
    assert "docs_url" not in status


def test_upload_manager_rejects_concurrent_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """A second start while one is running is refused (409-mapped by the route),
    naming the repo already uploading; the running upload is untouched."""
    from lelab.record import UploadManager, UploadRequest

    mgr = UploadManager()
    monkeypatch.setattr("lelab.datasets._dataset_in_use", lambda repo_id: None)
    # Pretend an upload is already running for another repo (don't spawn one).
    mgr.state = "running"
    mgr.repo_id = "tester/first"

    result = mgr.start(UploadRequest(dataset_repo_id="tester/second"))
    assert result["started"] is False
    assert "already running" in result["message"]
    assert "tester/first" in result["message"]
    # State unchanged — the second start didn't clobber the running upload.
    assert mgr.repo_id == "tester/first"


def test_upload_manager_refuses_busy_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    """A start is refused when the dataset is being written by another op —
    _dataset_in_use returns a reason, and no worker thread is spawned."""
    from lelab.record import UploadManager, UploadRequest

    monkeypatch.setattr(
        "lelab.datasets._dataset_in_use",
        lambda repo_id: "A recording session is writing to this dataset. Stop it before renaming.",
    )
    mgr = UploadManager()
    result = mgr.start(UploadRequest(dataset_repo_id="tester/ds"))
    assert result["started"] is False
    assert "recording session" in result["message"]
    assert mgr.state == "idle"
    assert mgr._thread is None


def test_upload_status_idle_shape() -> None:
    from lelab.record import UploadManager

    status = UploadManager().get_status()
    assert status["state"] == "idle"
    assert status["repo_id"] is None
    assert status["dataset_url"] is None
    assert "docs_url" not in status


def test_delete_dataset_refused_mid_upload(tmp_lerobot_home, monkeypatch: pytest.MonkeyPatch) -> None:
    """Deleting a dataset that's being pushed to the Hub is refused, and the
    directory is left on disk."""
    import json

    import lelab.record as record
    from lelab.record import DatasetInfoRequest, handle_delete_dataset

    repo_id = "tester/uploading"
    meta = tmp_lerobot_home / repo_id / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({"total_episodes": 2}))

    monkeypatch.setattr(record.upload_manager, "state", "running")
    monkeypatch.setattr(record.upload_manager, "repo_id", repo_id)

    result = handle_delete_dataset(DatasetInfoRequest(dataset_repo_id=repo_id))
    assert result["success"] is False
    assert "uploaded" in result["message"].lower()
    assert (tmp_lerobot_home / repo_id).exists()
