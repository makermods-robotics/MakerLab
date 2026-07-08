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


def test_recording_status_surfaces_preparing_substeps(monkeypatch) -> None:
    """record_with_web_events refines the coarse "preparing" window into named
    substeps ("connecting_robot", "connecting_teleop") by writing current_phase.
    The status handler must pass those through verbatim so the UI can name the
    substep — verified here without touching hardware by driving the module
    global the worker sets."""
    from lelab import record

    for substep in ("connecting_robot", "connecting_teleop"):
        monkeypatch.setattr(record, "current_phase", substep)
        # An active session with no config still surfaces current_phase.
        result = record.handle_recording_status()
        assert result["current_phase"] == substep
        # A preparing substep is not a completed/errored session.
        assert result["session_ended"] is False


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
    monkeypatch.setattr(
        "lelab.utils.robot_factory.setup_calibration_files",
        lambda leader, follower: ("leader", "follower"),
    )

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

    monkeypatch.setattr("lelab.utils.robot_factory.stage_bimanual_calibrations", _fake_stage)

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


# ---------------------------------------------------------------------------
# unique_id -> cv2 index re-resolution at record start (survives hotplug reshuffle)
# ---------------------------------------------------------------------------


def test_resolve_camera_index_passthrough_without_unique_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """No unique_id on the entry -> the stored index is used unchanged and NO
    enumeration is triggered (index-only configs keep their current behavior)."""
    from lelab import camera_enumeration
    from lelab.record import _resolve_camera_index

    def _boom():
        raise AssertionError("enumeration must not run when no unique_id is present")

    monkeypatch.setattr(camera_enumeration, "list_cameras", _boom)
    index, enumerated = _resolve_camera_index("wrist", {"camera_index": 3}, None)
    assert index == 3
    assert enumerated is None


def test_resolve_camera_index_remaps_by_unique_id(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    """When the device moved to a new cv2 index since the config was saved, the
    unique_id match resolves the CURRENT index and the remap is logged."""
    import logging

    from lelab.record import _resolve_camera_index

    enumerated = [
        {"index": 0, "name": "Built-in", "unique_id": "builtin-000"},
        {"index": 1, "name": "Arm cam", "unique_id": "uvc-7749-e450209"},
    ]
    with caplog.at_level(logging.WARNING):
        index, out = _resolve_camera_index(
            "wrist", {"camera_index": 0, "unique_id": "uvc-7749-e450209"}, enumerated
        )
    assert index == 1  # remapped from stored 0 -> current 1
    assert out is enumerated  # reused the passed-in enumeration, no re-enumerate
    assert any("REMAP" in r.message for r in caplog.records)


def test_resolve_camera_index_missing_device_raises_legible_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A unique_id that matches no connected device is a hard, legible error
    (raised before any hardware is touched)."""
    from lelab import camera_enumeration
    from lelab.record import _resolve_camera_index

    monkeypatch.setattr(camera_enumeration, "list_cameras", lambda: [{"index": 0, "unique_id": "other"}])
    with pytest.raises(camera_enumeration.CameraNotConnectedError) as exc:
        _resolve_camera_index("wrist", {"camera_index": 0, "unique_id": "uvc-7749-e450209"}, None)
    assert "not connected" in str(exc.value)
    assert "e450209" in str(exc.value)  # the id tail is surfaced to the user


def test_build_camera_configs_resolves_index_from_unique_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end through _build_camera_configs: the OpenCVCameraConfig opens the
    RE-RESOLVED index, not the stale stored one."""
    from lelab import camera_enumeration
    from lelab.record import _build_camera_configs
    from lerobot.cameras.configs import Cv2Backends

    monkeypatch.setattr(
        camera_enumeration,
        "list_cameras",
        lambda: [
            {"index": 0, "name": "Built-in", "unique_id": "builtin-000"},
            {"index": 2, "name": "Arm cam", "unique_id": "uvc-7749-e450209"},
        ],
    )
    cameras = {"wrist": {"type": "opencv", "camera_index": 0, "unique_id": "uvc-7749-e450209"}}
    configs = _build_camera_configs(cameras, Cv2Backends.AVFOUNDATION)
    assert configs["wrist"].index_or_path == 2


def test_build_camera_configs_rejects_two_cameras_on_the_same_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two roles resolving to the SAME cv2 index is a legible pre-hardware error
    naming both cameras — covers the mixed legacy+unique_id drift case: a legacy
    'front' pinned to stored index 0 collides with a 'wrist' whose unique_id
    re-resolves to 0."""
    from lelab import camera_enumeration
    from lelab.record import _build_camera_configs
    from lerobot.cameras.configs import Cv2Backends

    # 'wrist' (unique_id) re-resolves to index 0, where legacy 'front' already sits.
    monkeypatch.setattr(
        camera_enumeration,
        "list_cameras",
        lambda: [{"index": 0, "name": "USB Camera", "unique_id": "uvc-7749-wrist"}],
    )
    cameras = {
        "front": {"type": "opencv", "camera_index": 0},  # legacy, no unique_id
        "wrist": {"type": "opencv", "camera_index": 9, "unique_id": "uvc-7749-wrist"},
    }
    with pytest.raises(ValueError) as exc:
        _build_camera_configs(cameras, Cv2Backends.AVFOUNDATION)
    msg = str(exc.value)
    assert "front" in msg and "wrist" in msg  # both roles named
    assert "same physical camera" in msg
    assert "re-select" in msg


def test_build_camera_configs_allows_distinct_resolved_indexes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The duplicate guard must not trip when the two cameras resolve to
    DIFFERENT indexes (legacy 'front' at 0, 'wrist' unique_id at 1)."""
    from lelab import camera_enumeration
    from lelab.record import _build_camera_configs
    from lerobot.cameras.configs import Cv2Backends

    monkeypatch.setattr(
        camera_enumeration,
        "list_cameras",
        lambda: [{"index": 1, "name": "USB Camera", "unique_id": "uvc-7749-wrist"}],
    )
    cameras = {
        "front": {"type": "opencv", "camera_index": 0},
        "wrist": {"type": "opencv", "camera_index": 9, "unique_id": "uvc-7749-wrist"},
    }
    configs = _build_camera_configs(cameras, Cv2Backends.AVFOUNDATION)
    assert configs["front"].index_or_path == 0
    assert configs["wrist"].index_or_path == 1


def test_create_record_config_errors_before_hardware_on_missing_camera(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A saved camera whose unique_id is no longer connected fails the record
    start cleanly, before any robot/camera construction (create_record_config
    builds the camera configs first)."""
    from lelab import camera_enumeration
    from lelab.record import RecordingRequest, create_record_config

    monkeypatch.setattr(camera_enumeration, "list_cameras", lambda: [])  # nothing connected
    request = RecordingRequest(
        leader_port="COM_LEADER",
        follower_port="COM_FOLLOWER",
        leader_config="leader",
        follower_config="follower",
        dataset_repo_id="tester/dataset",
        single_task="pick",
        cameras={"wrist": {"type": "opencv", "camera_index": 0, "unique_id": "uvc-7749-e450209"}},
    )
    with pytest.raises(camera_enumeration.CameraNotConnectedError):
        create_record_config(request)


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


# ---------------------------------------------------------------------------
# Quit-without-saving (discard) — _discard_session_dataset + the stop handler.
# The resume-protection test comes FIRST and is the load-bearing data-safety
# guarantee: a quit must never delete a pre-existing (resume) dataset.
# ---------------------------------------------------------------------------


def test_discard_session_dataset_never_touches_resume_session(tmp_lerobot_home) -> None:
    """A QUIT on a RESUME session must NEVER delete the pre-existing dataset —
    lerobot already committed its earlier episodes, so they must survive. The
    resume guard is checked first; this is the load-bearing safety property."""
    import lelab.record as record

    target = _make_dataset_dir(tmp_lerobot_home, "tester/preexisting", total_episodes=5)

    removed = record._discard_session_dataset("tester/preexisting", resume=True)

    assert removed is False
    assert target.exists()  # every already-saved episode is intact


def test_discard_session_dataset_removes_fresh_dir_with_episodes(tmp_lerobot_home) -> None:
    """A QUIT on a FRESH session removes the whole stamped directory even when
    episodes were saved earlier THIS session — quit discards everything the
    session created (unlike _discard_empty_dataset, which keeps a non-empty dir)."""
    import lelab.record as record

    target = _make_dataset_dir(tmp_lerobot_home, "tester/quit_20260708_120000", total_episodes=3)
    assert target.exists()

    removed = record._discard_session_dataset("tester/quit_20260708_120000", resume=False)

    assert removed is True
    assert not target.exists()


def test_discard_session_dataset_rejects_path_traversal(tmp_lerobot_home) -> None:
    """A repo_id escaping the cache root is refused — no deletion outside cache."""
    import lelab.record as record

    removed = record._discard_session_dataset("../../etc", resume=False)
    assert removed is False


def test_discard_session_dataset_invalidates_hub_status(tmp_lerobot_home) -> None:
    """Discarding a quit session drops any cached Hub-existence probe for it."""
    import lelab.datasets as datasets
    import lelab.record as record

    _make_dataset_dir(tmp_lerobot_home, "tester/quit_probed", total_episodes=2)
    with datasets._HUB_STATUS_LOCK:
        datasets._HUB_STATUS_CACHE["tester/quit_probed"] = "local_only"

    assert record._discard_session_dataset("tester/quit_probed", resume=False) is True

    with datasets._HUB_STATUS_LOCK:
        assert "tester/quit_probed" not in datasets._HUB_STATUS_CACHE


def test_handle_stop_recording_discard_arms_flag_and_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Quit stop (discard=True) on a live session arms discard_requested, sets
    the same stop events a Done stop does, and echoes discard in the response."""
    import lelab.record as record

    events = {"stop_recording": False, "exit_early": False}
    monkeypatch.setattr(record, "releasing", False)
    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "recording_events", events)
    monkeypatch.setattr(record, "discard_requested", False)

    result = record.handle_stop_recording(discard=True)

    assert result["success"] is True
    assert result["discard"] is True
    assert record.discard_requested is True
    assert events["stop_recording"] is True
    assert events["exit_early"] is True
    assert "without saving" in result["message"].lower()


def test_handle_stop_recording_discard_ignored_when_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    """A discard stop against no active session is refused and never arms the
    discard flag — an idle/mutex miss can't schedule a dataset deletion."""
    import lelab.record as record

    monkeypatch.setattr(record, "releasing", False)
    monkeypatch.setattr(record, "recording_active", False)
    monkeypatch.setattr(record, "recording_events", None)
    monkeypatch.setattr(record, "discard_requested", False)

    result = record.handle_stop_recording(discard=True)

    assert result["success"] is False
    assert record.discard_requested is False


def test_worker_quit_discards_fresh_dataset_with_saved_episodes(
    monkeypatch: pytest.MonkeyPatch, tmp_lerobot_home
) -> None:
    """End-to-end through the real worker: a fresh session whose user quit
    (discard_requested set) has its whole stamped directory removed in the
    finally block, even with episodes saved, and reports discarded_empty."""
    import lelab.record as record

    def _work_then_quit(cfg, events, **kwargs):
        # Create the dataset dir at the stamped repo id the session recorded into,
        # then simulate a completed loop with saved episodes and a user quit.
        repo_id = record.recording_config.dataset_repo_id
        _make_dataset_dir(tmp_lerobot_home, repo_id, total_episodes=2)
        record.current_phase = "completed"
        record.saved_episodes = 2
        record.discard_requested = True  # handle_stop_recording(discard=True) would set this

    try:
        status = _start_session_with_fake_work(monkeypatch, _work_then_quit)

        assert status["session_ended"] is True
        assert status["discarded_empty"] is True
        assert not (tmp_lerobot_home / status["dataset_repo_id"]).exists()
    finally:
        record.discard_requested = False  # don't leak the armed flag into later tests


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
    repo_id: str = "tester/ds",
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
            dataset_repo_id=repo_id,
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


def test_record_accepts_bare_repo_id(monkeypatch: pytest.MonkeyPatch, tmp_lerobot_home) -> None:
    """A bare dataset name (no HF login → no `user/` namespace) records fine —
    lerobot's sanity_check_dataset_name would crash on it, so we don't call it."""

    monkeypatch.setattr(
        "lelab.utils.robot_factory.setup_calibration_files",
        lambda leader, follower: ("leader", "follower"),
    )
    bus = _RecReturnBus(positions=dict.fromkeys(_RecReturnBus._MOTORS, 1500))
    robot = _RecRobot(bus)

    _, robot, error, _ = _run_record_session(monkeypatch, robot, repo_id="bare_local_name")

    assert error is None


def test_record_refuses_eval_prefixed_repo_id(monkeypatch: pytest.MonkeyPatch, tmp_lerobot_home) -> None:
    """eval_ names are reserved for policy-evaluation recordings (rollout flow),
    with or without a namespace."""

    monkeypatch.setattr(
        "lelab.utils.robot_factory.setup_calibration_files",
        lambda leader, follower: ("leader", "follower"),
    )
    for repo_id in ("eval_ds", "tester/eval_ds"):
        bus = _RecReturnBus(positions=dict.fromkeys(_RecReturnBus._MOTORS, 1500))
        robot = _RecRobot(bus)
        _, robot, error, _ = _run_record_session(monkeypatch, robot, repo_id=repo_id)
        assert isinstance(error, ValueError), repo_id
        assert "eval_" in str(error)


def test_record_normal_end_returns_then_releases(monkeypatch: pytest.MonkeyPatch, tmp_lerobot_home) -> None:
    """A normal session end drives the follower back to its captured start pose
    (once), then disconnects — same as teleop's stop, no timed hold."""
    import lelab.record as record

    monkeypatch.setattr(
        "lelab.utils.robot_factory.setup_calibration_files",
        lambda leader, follower: ("leader", "follower"),
    )
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

    monkeypatch.setattr(
        "lelab.utils.robot_factory.setup_calibration_files",
        lambda leader, follower: ("leader", "follower"),
    )
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

    monkeypatch.setattr(
        "lelab.utils.robot_factory.setup_calibration_files",
        lambda leader, follower: ("leader", "follower"),
    )
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

    monkeypatch.setattr(
        "lelab.utils.robot_factory.setup_calibration_files",
        lambda leader, follower: ("leader", "follower"),
    )
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

    monkeypatch.setattr(
        "lelab.utils.robot_factory.setup_calibration_files",
        lambda leader, follower: ("leader", "follower"),
    )
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

    monkeypatch.setattr(
        "lelab.utils.robot_factory.setup_calibration_files",
        lambda leader, follower: ("leader", "follower"),
    )
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


def test_delete_dataset_refused_mid_recording(tmp_lerobot_home, monkeypatch: pytest.MonkeyPatch) -> None:
    """Deleting a dataset an active recording session is writing is refused —
    the delete guard now runs the full _dataset_in_use check, not just the
    upload one."""
    import json
    from unittest.mock import MagicMock

    import lelab.record as record
    from lelab.record import DatasetInfoRequest, handle_delete_dataset

    repo_id = "tester/recording_ds"
    meta = tmp_lerobot_home / repo_id / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({"total_episodes": 1}))

    cfg = MagicMock()
    cfg.dataset_repo_id = repo_id
    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "recording_config", cfg)

    result = handle_delete_dataset(DatasetInfoRequest(dataset_repo_id=repo_id))
    assert result["success"] is False
    assert "recording" in result["message"].lower()
    assert (tmp_lerobot_home / repo_id).exists()


def test_delete_dataset_refused_mid_merge(tmp_lerobot_home, monkeypatch: pytest.MonkeyPatch) -> None:
    """Deleting the output dataset of a running merge is refused."""
    import json

    from lelab import merge
    from lelab.record import DatasetInfoRequest, handle_delete_dataset

    repo_id = "tester/merging_ds"
    meta = tmp_lerobot_home / repo_id / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({"total_episodes": 1}))

    monkeypatch.setattr(merge.merge_manager, "state", "running")
    monkeypatch.setattr(merge.merge_manager, "output_repo_id", repo_id)

    result = handle_delete_dataset(DatasetInfoRequest(dataset_repo_id=repo_id))
    assert result["success"] is False
    assert "merge" in result["message"].lower()
    assert (tmp_lerobot_home / repo_id).exists()


def test_delete_dataset_refused_mid_local_training(tmp_lerobot_home, monkeypatch: pytest.MonkeyPatch) -> None:
    """Deleting a dataset a running local training job reads is refused."""
    import json
    from unittest.mock import MagicMock

    from lelab import jobs
    from lelab.record import DatasetInfoRequest, handle_delete_dataset

    repo_id = "tester/training_ds"
    meta = tmp_lerobot_home / repo_id / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({"total_episodes": 1}))

    job = MagicMock()
    job.state = "running"
    job.runner = "local"
    job.config.dataset_repo_id = repo_id
    monkeypatch.setattr(jobs.job_registry, "list", lambda limit=200: [job])

    result = handle_delete_dataset(DatasetInfoRequest(dataset_repo_id=repo_id))
    assert result["success"] is False
    assert "training" in result["message"].lower()
    assert (tmp_lerobot_home / repo_id).exists()


def test_delete_refusal_wording_is_action_neutral(tmp_lerobot_home, monkeypatch: pytest.MonkeyPatch) -> None:
    """The shared in-use guard's refusals are action-neutral ("Stop it first."),
    not rename-specific — they now surface from delete too."""
    import json
    from unittest.mock import MagicMock

    import lelab.record as record
    from lelab.record import DatasetInfoRequest, handle_delete_dataset

    repo_id = "tester/neutral_ds"
    meta = tmp_lerobot_home / repo_id / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({"total_episodes": 1}))

    cfg = MagicMock()
    cfg.dataset_repo_id = repo_id
    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "recording_config", cfg)

    result = handle_delete_dataset(DatasetInfoRequest(dataset_repo_id=repo_id))
    assert result["success"] is False
    assert "renaming" not in result["message"]
    assert result["message"].endswith("Stop it first.")


def test_start_recording_resume_skips_timestamp_stamp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resume must append to the EXISTING directory: the repo_id is used
    verbatim (no '_<timestamp>' suffix), unlike a fresh session which stamps
    one. Regression-guards the `if not request.resume` skip."""
    import re

    import lelab.record as record
    import lelab.rollout as rollout
    import lelab.teleoperate as teleop

    monkeypatch.setattr(record, "recording_active", False)
    monkeypatch.setattr(record, "recording_thread", None)
    monkeypatch.setattr(record, "releasing", False)
    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(teleop, "teleoperation_thread", None)
    monkeypatch.setattr(rollout, "inference_active", False)

    # Fail fast AFTER the stamp point (create_record_config runs right after),
    # before any hardware is touched.
    def _boom(request):
        raise RuntimeError("stop before hardware")

    monkeypatch.setattr(record, "create_record_config", _boom)

    def _start(resume: bool):
        req = record.RecordingRequest(
            leader_port="COM_LEADER",
            follower_port="COM_FOLLOWER",
            leader_config="leader",
            follower_config="follower",
            dataset_repo_id="tester/existing_ds",
            single_task="pick",
            resume=resume,
        )
        result = record.handle_start_recording(req)
        assert result["success"] is False  # the _boom stub stopped the start
        return req.dataset_repo_id

    # Resume: the id is untouched. Fresh: a "_YYYYMMDD_HHMMSS" stamp lands.
    assert _start(resume=True) == "tester/existing_ds"
    monkeypatch.setattr(record, "recording_active", False)  # release the claim
    assert re.fullmatch(r"tester/existing_ds_\d{8}_\d{6}", _start(resume=False))


# ---------------------------------------------------------------------------
# Session error taxonomy — outcome / error / hint (in-process twin of the
# rollout exited payload). The worker's catch site holds the actual exception,
# so the error text is formatted from the object (no log forensics); the
# outcome is classified by catch-site phase: an exception AFTER the recording
# loop finished (phase already "completed" — episodes saved) is only noisy
# teardown, a warning; any earlier phase is a real failure.
# ---------------------------------------------------------------------------


def test_classify_outcome_three_ways() -> None:
    """The pure classifier behind both record and teleop catch sites."""
    from lelab.utils.errors import classify_outcome

    # No error: the session was fine, wherever it stood.
    assert classify_outcome(work_completed=True, error_text=None) == "ok"
    assert classify_outcome(work_completed=False, error_text=None) == "ok"
    # The saved-episodes-then-teardown-overload case: the loop finished its
    # real work, then disabling torque on a loaded gripper raised. Data is
    # safe — a warning, NOT a failed session.
    assert (
        classify_outcome(True, "RuntimeError: Overload detected on gripper (torque_enable failed)")
        == "ran_with_warning"
    )
    # The mid-episode-failure case: same-looking error text, but the work was
    # cut short — catch-site phase (not text markers) decides: failed.
    assert classify_outcome(False, "RuntimeError: Overload detected on gripper") == "failed"
    assert classify_outcome(False, "ConnectionError: could not connect to the arm") == "failed"


def test_format_exception_type_message_and_truncation() -> None:
    from lelab.utils.errors import format_exception

    assert format_exception(RuntimeError("boom")) == "RuntimeError: boom"
    out = format_exception(RuntimeError("x" * 2000))
    assert out.startswith("RuntimeError: ")
    assert out.endswith("…")
    assert len(out) <= 501  # 500-char cap + ellipsis


def test_recording_status_carries_outcome_error_hint_at_session_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The session-end status payload exposes the taxonomy fields, with the
    hint derived from the error text via friendly_hint."""
    import lelab.record as record

    monkeypatch.setattr(record, "recording_active", False)
    monkeypatch.setattr(record, "current_phase", "completed")
    monkeypatch.setattr(record, "last_session_outcome", "ran_with_warning")
    monkeypatch.setattr(
        record,
        "last_session_error",
        "RuntimeError: Overload detected on gripper (torque_enable failed)",
    )

    status = record.handle_recording_status()

    assert status["session_ended"] is True
    assert status["outcome"] == "ran_with_warning"
    assert "Overload" in status["error"]
    assert "motor overloaded" in status["hint"].lower()


def test_recording_status_omits_outcome_fields_while_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The taxonomy describes an ENDED session only — a live session's status
    carries none of the three fields (mirrors discarded_empty)."""
    import lelab.record as record

    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "current_phase", "recording")

    status = record.handle_recording_status()

    assert status["session_ended"] is False
    for key in ("outcome", "error", "hint"):
        assert key not in status


def _start_session_with_fake_work(monkeypatch: pytest.MonkeyPatch, fake_work):
    """Drive handle_start_recording with record_with_web_events replaced by
    `fake_work`, so the REAL worker thread runs the real catch site. Returns
    after joining the worker. All feature mutexes idle; no hardware touched."""
    import lelab.record as record
    import lelab.rollout as rollout
    import lelab.teleoperate as teleop

    monkeypatch.setattr(record, "recording_active", False)
    monkeypatch.setattr(record, "recording_thread", None)
    monkeypatch.setattr(record, "releasing", False)
    monkeypatch.setattr(record, "current_phase", "preparing")
    monkeypatch.setattr(record, "last_session_outcome", None)
    monkeypatch.setattr(record, "last_session_error", None)
    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(teleop, "teleoperation_thread", None)
    monkeypatch.setattr(rollout, "inference_active", False)
    monkeypatch.setattr(record, "create_record_config", lambda request: None)
    monkeypatch.setattr(record, "record_with_web_events", fake_work)

    result = record.handle_start_recording(
        record.RecordingRequest(
            leader_port="COM_LEADER",
            follower_port="COM_FOLLOWER",
            leader_config="leader",
            follower_config="follower",
            dataset_repo_id="tester/taxonomy_ds",
            single_task="pick",
        )
    )
    assert result["success"] is True
    record.recording_thread.join(timeout=5.0)
    assert not record.recording_thread.is_alive()
    return record.handle_recording_status()


def test_worker_classifies_teardown_failure_as_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_lerobot_home
) -> None:
    """THE headline case: a session whose recording loop finished (episodes
    saved, phase "completed") but whose teardown overloaded the gripper must
    end ran_with_warning with phase "completed" — NOT a failed session."""
    import lelab.record as record

    def _work_then_teardown_boom(cfg, events, **kwargs):
        # The loop finished its real work before cleanup raised.
        record.current_phase = "completed"
        record.saved_episodes = 3
        raise RuntimeError("Overload detected on gripper while disabling torque (torque_enable)")

    status = _start_session_with_fake_work(monkeypatch, _work_then_teardown_boom)

    assert status["session_ended"] is True
    assert status["current_phase"] == "completed"  # not "error"
    assert status["outcome"] == "ran_with_warning"
    assert "Overload" in status["error"]
    assert "motor overloaded" in status["hint"].lower()


def test_worker_classifies_midsession_failure_as_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_lerobot_home
) -> None:
    """An exception mid-episode (loop still in "recording") is a real failure:
    phase "error", outcome "failed", with the camera hint mapped."""

    import lelab.record as record

    def _boom_mid_episode(cfg, events, **kwargs):
        record.current_phase = "recording"
        raise RuntimeError("Camera cam0: frame is too old (age 3.2s)")

    status = _start_session_with_fake_work(monkeypatch, _boom_mid_episode)

    assert status["session_ended"] is True
    assert status["current_phase"] == "error"
    assert status["outcome"] == "failed"
    assert "frame is too old" in status["error"]
    assert "camera" in status["hint"].lower()


def test_worker_reports_ok_outcome_on_clean_end(monkeypatch: pytest.MonkeyPatch, tmp_lerobot_home) -> None:
    """A session that ends without raising reports outcome "ok" (no error, no
    hint) so the frontend's normal navigate-to-upload path is untouched."""
    import lelab.record as record

    class _FakeDataset:
        num_episodes = 2

    def _clean_work(cfg, events, **kwargs):
        record.current_phase = "completed"
        record.saved_episodes = 2
        return _FakeDataset()

    status = _start_session_with_fake_work(monkeypatch, _clean_work)

    assert status["session_ended"] is True
    assert status["outcome"] == "ok"
    assert status["error"] is None
    assert status["hint"] is None
