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
"""Tests for makerlab.calibrate — manager initial state, request schema, and the
post-recording centering guard."""

from __future__ import annotations

from makerlab.calibrate import final_motor_ranges, find_off_center_joints


def test_final_motor_ranges_forces_wrist_roll_full_turn() -> None:
    # Wrist_roll's swept sliver is discarded for the full turn (matching
    # upstream lerobot); other joints keep their recorded ranges.
    mins = {"shoulder_pan": 900, "wrist_roll": 2000}
    maxes = {"shoulder_pan": 3200, "wrist_roll": 2010}
    assert final_motor_ranges(mins, maxes) == {
        "shoulder_pan": (900, 3200),
        "wrist_roll": (0, 4095),
    }


def test_calibration_request_dataclass_round_trip() -> None:
    from makerlab.calibrate import CalibrationRequest

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
    from makerlab.calibrate import CalibrationManager, CalibrationStatus

    mgr = CalibrationManager()
    status = mgr.get_status()

    assert isinstance(status, CalibrationStatus)
    assert status is mgr.status
    assert status.calibration_active is False
    assert status.status == "idle"
    assert status.device_type is None
    assert status.error is None
    assert status.step == 0
    assert mgr.device is None
    assert mgr.calibration_thread is None


def test_calibration_manager_rejects_double_start_via_message() -> None:
    """When calibration_active is True, start_calibration returns success=False."""
    from makerlab.calibrate import CalibrationManager, CalibrationRequest

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

    from makerlab.calibrate import CalibrationManager, CalibrationRequest
    from makerlab.utils import config as cfg

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
