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
"""Tests for lelab.calibrate — manager initial state, request schema, and the
post-recording centering guard."""

from __future__ import annotations

from lelab.calibrate import final_motor_ranges, find_off_center_joints


def test_final_motor_ranges_forces_wrist_roll_full_turn() -> None:
    # Wrist_roll's swept sliver is discarded for the full turn (matching
    # upstream lerobot); other joints keep their recorded ranges.
    mins = {"shoulder_pan": 900, "wrist_roll": 2000}
    maxes = {"shoulder_pan": 3200, "wrist_roll": 2010}
    assert final_motor_ranges(mins, maxes) == {
        "shoulder_pan": (900, 3200),
        "wrist_roll": (0, 4095),
    }


def test_final_motor_ranges_forces_full_turn_even_if_unmoved() -> None:
    # Not moving wrist_roll at all is the documented procedure.
    mins = {"wrist_roll": 2047}
    maxes = {"wrist_roll": 2047}
    assert final_motor_ranges(mins, maxes) == {"wrist_roll": (0, 4095)}


def test_calibration_status_defaults_to_idle() -> None:
    from lelab.calibrate import CalibrationStatus

    status = CalibrationStatus()
    assert status.calibration_active is False
    assert status.status == "idle"
    assert status.device_type is None
    assert status.error is None
    assert status.step == 0


def test_calibration_request_dataclass_round_trip() -> None:
    from lelab.calibrate import CalibrationRequest

    req = CalibrationRequest(
        device_type="teleop",
        port="/dev/ttyUSB0",
        config_file="my_calib",
    )
    assert req.device_type == "teleop"
    assert req.port == "/dev/ttyUSB0"
    assert req.config_file == "my_calib"
    assert req.robot_name is None


def test_calibration_manager_starts_idle() -> None:
    from lelab.calibrate import CalibrationManager

    mgr = CalibrationManager()
    assert mgr.status.calibration_active is False
    assert mgr.status.status == "idle"
    assert mgr.device is None
    assert mgr.calibration_thread is None


def test_calibration_manager_get_status_when_idle_returns_status_object() -> None:
    from lelab.calibrate import CalibrationManager, CalibrationStatus

    mgr = CalibrationManager()
    s = mgr.get_status()
    assert isinstance(s, CalibrationStatus)
    assert s.status == "idle"


def test_calibration_manager_rejects_double_start_via_message() -> None:
    """When calibration_active is True, start_calibration returns success=False."""
    from lelab.calibrate import CalibrationManager, CalibrationRequest

    mgr = CalibrationManager()
    mgr.status.calibration_active = True  # simulate already running

    result = mgr.start_calibration(
        CalibrationRequest(device_type="teleop", port="/dev/null", config_file="x")
    )
    assert result.get("success") is False
    assert "already" in result.get("message", "").lower()


def test_start_calibration_refuses_existing_config_without_overwrite(tmp_lerobot_home) -> None:
    """Completing calibration saves <config_file>.json; if that name already
    exists, start must refuse (code=name_taken) unless overwrite=True — so no
    file is silently clobbered, and no hardware is touched."""
    from pathlib import Path

    from lelab.calibrate import CalibrationManager, CalibrationRequest
    from lelab.utils import config as cfg

    (Path(cfg.LEADER_CONFIG_PATH) / "taken.json").write_text("{}")

    mgr = CalibrationManager()
    result = mgr.start_calibration(
        CalibrationRequest(device_type="teleop", port="/dev/null", config_file="taken")
    )
    assert result.get("success") is False
    assert result.get("code") == "name_taken"
    # The guard returns before activating or spawning the worker thread.
    assert mgr.status.calibration_active is False
    assert mgr.calibration_thread is None


def test_find_off_center_joints_passes_centered_ranges() -> None:
    """Ranges whose midpoints sit on the raw-tick center (2047) all pass."""
    ranges = {
        "shoulder_pan": (1047, 3047),  # midpoint exactly 2047
        "shoulder_lift": (1500, 2600),  # midpoint 2050, well within tolerance
        "elbow_flex": (1000, 3000),
        "wrist_flex": (1200, 2900),
    }
    assert find_off_center_joints(ranges) == []


def test_find_off_center_joints_names_the_skewed_joint() -> None:
    """A range lying almost entirely to one side of 2047 is flagged by name."""
    ranges = {
        "shoulder_pan": (1047, 3047),  # centered, passes
        "shoulder_lift": (2000, 3600),  # midpoint 2800, 753 off vs 320 allowed
    }
    assert find_off_center_joints(ranges) == ["shoulder_lift"]


def test_find_off_center_joints_exempts_gripper_and_wrist_roll() -> None:
    """Gripper is legitimately homed closed, and wrist_roll is a full-turn
    motor upstream — both skip the check no matter how skewed their range is."""
    ranges = {
        "gripper": (2000, 3500),  # midpoint 2750, would fail if checked
        "wrist_roll": (2500, 4000),  # midpoint 3250, would fail if checked
    }
    assert find_off_center_joints(ranges) == []


def test_find_off_center_joints_tolerance_boundary() -> None:
    """Deviation equal to 20% of the range width passes; one tick more fails."""
    # Width 2000 -> 400 ticks allowed. Midpoint 2447 deviates by exactly 400.
    assert find_off_center_joints({"elbow_flex": (1447, 3447)}) == []
    # Midpoint 2448 deviates by 401 — just over the line.
    assert find_off_center_joints({"elbow_flex": (1448, 3448)}) == ["elbow_flex"]


# --- Batch (multi-arm) calibration ------------------------------------------


def _batch_arm(device_type="teleop", arm="left", config_file="c", port="/dev/null"):
    from lelab.calibrate import CalibrationBatchArm

    return CalibrationBatchArm(device_type=device_type, port=port, config_file=config_file, arm=arm)


def test_batch_request_coerces_dict_arms() -> None:
    """FastAPI hands nested models as dicts; __post_init__ coerces them into
    CalibrationBatchArm instances."""
    from lelab.calibrate import CalibrationBatchArm, CalibrationBatchRequest

    req = CalibrationBatchRequest(
        robot_name="r",
        arms=[{"device_type": "teleop", "port": "/dev/null", "config_file": "c", "arm": "left"}],
    )
    assert len(req.arms) == 1
    assert isinstance(req.arms[0], CalibrationBatchArm)
    assert req.arms[0].device_type == "teleop"
    assert req.overwrite is False


def test_validate_batch_arms_rejects_empty_list() -> None:
    from lelab.calibrate import validate_batch_arms

    assert validate_batch_arms([]) is not None


def test_validate_batch_arms_rejects_more_than_four() -> None:
    from lelab.calibrate import validate_batch_arms

    arms = [
        _batch_arm("teleop", "left", "a"),
        _batch_arm("teleop", "right", "b"),
        _batch_arm("robot", "left", "c"),
        _batch_arm("robot", "right", "d"),
        _batch_arm("teleop", "left", "e"),  # 5th
    ]
    assert validate_batch_arms(arms) is not None


def test_validate_batch_arms_rejects_duplicate_slot() -> None:
    """Two arms targeting the same (device_type, arm) slot is invalid."""
    from lelab.calibrate import validate_batch_arms

    arms = [_batch_arm("teleop", "left", "a"), _batch_arm("teleop", "left", "b")]
    reason = validate_batch_arms(arms)
    assert reason is not None
    assert "slot" in reason.lower()


def test_validate_batch_arms_rejects_same_side_name_collision() -> None:
    """Two same-side (both leader) arms sharing a config name is invalid."""
    from lelab.calibrate import validate_batch_arms

    arms = [_batch_arm("teleop", "left", "shared"), _batch_arm("teleop", "right", "shared")]
    reason = validate_batch_arms(arms)
    assert reason is not None
    assert "shared" in reason


def test_validate_batch_arms_allows_cross_side_same_name() -> None:
    """A leader and a follower sharing a name is fine — different dirs."""
    from lelab.calibrate import validate_batch_arms

    arms = [_batch_arm("teleop", "left", "same"), _batch_arm("robot", "left", "same")]
    assert validate_batch_arms(arms) is None


def test_validate_batch_arms_accepts_four_distinct_slots() -> None:
    from lelab.calibrate import validate_batch_arms

    arms = [
        _batch_arm("teleop", "left", "a"),
        _batch_arm("teleop", "right", "b"),
        _batch_arm("robot", "left", "c"),
        _batch_arm("robot", "right", "d"),
    ]
    assert validate_batch_arms(arms) is None


def test_batch_rejected_while_single_calibration_active() -> None:
    """The batch mutex refuses to start while a single calibration is running."""
    from lelab.calibrate import CalibrationBatchRequest, CalibrationManager

    mgr = CalibrationManager()
    mgr.status.calibration_active = True  # simulate single-arm running

    result = mgr.start_calibration_batch(CalibrationBatchRequest(robot_name="r", arms=[_batch_arm()]))
    assert result.get("success") is False
    assert "already" in result.get("message", "").lower()


def test_single_calibration_rejected_while_batch_active() -> None:
    """Conversely, a single start is refused while a batch is running (both flip
    calibration_active, so the existing single-arm mutex covers this too)."""
    from lelab.calibrate import CalibrationManager, CalibrationRequest

    mgr = CalibrationManager()
    mgr.status.batch_active = True
    mgr.status.calibration_active = True

    result = mgr.start_calibration(
        CalibrationRequest(device_type="teleop", port="/dev/null", config_file="x")
    )
    assert result.get("success") is False
    assert "already" in result.get("message", "").lower()


def test_batch_precheck_reports_name_taken_for_offending_arm(tmp_lerobot_home) -> None:
    """Every arm's overwrite collision is checked UP FRONT, before any hardware
    moves. A taken name yields code=name_taken naming the offending arm, and no
    driver thread is spawned."""
    from pathlib import Path

    from lelab.calibrate import CalibrationBatchRequest, CalibrationManager
    from lelab.utils import config as cfg

    # The SECOND arm's follower name is taken; the first (leader) is free.
    (Path(cfg.FOLLOWER_CONFIG_PATH) / "taken.json").write_text("{}")

    mgr = CalibrationManager()
    result = mgr.start_calibration_batch(
        CalibrationBatchRequest(
            robot_name="r",
            arms=[
                _batch_arm("teleop", "left", "free"),
                _batch_arm("robot", "left", "taken"),
            ],
        )
    )
    assert result.get("success") is False
    assert result.get("code") == "name_taken"
    assert result.get("arm") == {"device_type": "robot", "arm": "left"}
    # Fail fast: no activation, no batch thread.
    assert mgr.status.calibration_active is False
    assert mgr.status.batch_active is False
    assert mgr.batch_thread is None


def test_batch_precheck_bypassed_with_overwrite(tmp_lerobot_home, monkeypatch) -> None:
    """With overwrite=True the up-front name-taken pre-check is skipped, so the
    batch proceeds to spawn its driver (hardware is stubbed out via
    _run_single_arm so no real device is touched)."""
    from pathlib import Path

    from lelab.calibrate import CalibrationBatchRequest, CalibrationManager
    from lelab.utils import config as cfg

    (Path(cfg.LEADER_CONFIG_PATH) / "taken.json").write_text("{}")

    mgr = CalibrationManager()
    # Stub the per-arm runner so the driver thread doesn't touch hardware; make
    # it return immediately as "completed".
    monkeypatch.setattr(mgr, "_run_single_arm", lambda request: True)

    result = mgr.start_calibration_batch(
        CalibrationBatchRequest(
            robot_name="r",
            arms=[_batch_arm("teleop", "left", "taken")],
            overwrite=True,
        )
    )
    assert result.get("success") is True
    # Let the (trivial) driver thread finish.
    if mgr.batch_thread is not None:
        mgr.batch_thread.join(timeout=2.0)
    assert mgr.status.batch_active is False
    assert mgr.status.status == "completed"


def test_batch_sequences_arms_and_advances_index(tmp_lerobot_home, monkeypatch) -> None:
    """The driver calibrates each arm in order, advancing batch_index and
    recording completions, then reports completed. Hardware is stubbed."""
    from lelab.calibrate import CalibrationBatchRequest, CalibrationManager

    mgr = CalibrationManager()
    seen_indices: list[int] = []

    def fake_run(request):
        # Capture the batch_index the driver set before each arm.
        seen_indices.append(mgr.status.batch_index)
        return True

    monkeypatch.setattr(mgr, "_run_single_arm", fake_run)

    result = mgr.start_calibration_batch(
        CalibrationBatchRequest(
            robot_name="r",
            arms=[
                _batch_arm("teleop", "left", "a"),
                _batch_arm("robot", "left", "b"),
            ],
        )
    )
    assert result.get("success") is True
    mgr.batch_thread.join(timeout=2.0)

    assert seen_indices == [0, 1]
    assert mgr.status.status == "completed"
    assert mgr.status.batch_active is False
    assert mgr.status.calibration_active is False
    assert mgr.status.batch_completed == ["teleop left", "robot left"]
    assert mgr.status.batch_failed_arm is None


def test_batch_stops_on_arm_error_and_names_it(tmp_lerobot_home, monkeypatch) -> None:
    """When an arm raises, the batch STOPS: later arms are skipped, the failing
    arm is named, and earlier arms stay recorded as completed."""
    from lelab.calibrate import CalibrationBatchRequest, CalibrationManager

    mgr = CalibrationManager()
    calls: list[str] = []

    def fake_run(request):
        calls.append(f"{request.device_type} {request.arm}")
        if request.device_type == "robot":
            raise RuntimeError("connect failed")
        return True

    monkeypatch.setattr(mgr, "_run_single_arm", fake_run)

    result = mgr.start_calibration_batch(
        CalibrationBatchRequest(
            robot_name="r",
            arms=[
                _batch_arm("teleop", "left", "a"),  # completes
                _batch_arm("robot", "left", "b"),  # fails
                _batch_arm("robot", "right", "c"),  # never reached
            ],
        )
    )
    assert result.get("success") is True
    mgr.batch_thread.join(timeout=2.0)

    # Third arm never ran.
    assert calls == ["teleop left", "robot left"]
    assert mgr.status.status == "error"
    assert mgr.status.batch_failed_arm == "robot left"
    assert mgr.status.batch_completed == ["teleop left"]
    assert mgr.status.batch_active is False
    assert mgr.status.calibration_active is False
