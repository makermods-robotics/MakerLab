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


def test_final_motor_ranges_forces_full_turn_even_if_unmoved() -> None:
    # Not moving wrist_roll at all is the documented procedure.
    mins = {"wrist_roll": 2047}
    maxes = {"wrist_roll": 2047}
    assert final_motor_ranges(mins, maxes) == {"wrist_roll": (0, 4095)}


def test_calibration_status_defaults_to_idle() -> None:
    from makerlab.calibrate import CalibrationStatus

    status = CalibrationStatus()
    assert status.calibration_active is False
    assert status.status == "idle"
    assert status.device_type is None
    assert status.error is None
    assert status.step == 0


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
    from makerlab.calibrate import CalibrationManager

    mgr = CalibrationManager()
    assert mgr.status.calibration_active is False
    assert mgr.status.status == "idle"
    assert mgr.device is None
    assert mgr.calibration_thread is None


def test_calibration_manager_get_status_when_idle_returns_status_object() -> None:
    from makerlab.calibrate import CalibrationManager, CalibrationStatus

    mgr = CalibrationManager()
    s = mgr.get_status()
    assert isinstance(s, CalibrationStatus)
    assert s.status == "idle"


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


# ---------------------------------------------------------------------------
# Servo snapshot/rollback: a canceled/errored calibration must restore the
# EEPROM registers it mutated (Homing_Offset + range limits) so the arm's
# persistent state does not diverge from its untouched calibration file.
# ---------------------------------------------------------------------------

_MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


class _FakeBus:
    """Minimal FeetechMotorsBus double: register store + a write log."""

    def __init__(self, registers: dict[str, dict[str, int]]) -> None:
        self.motors = dict.fromkeys(_MOTORS)
        self.registers = registers  # {reg: {motor: value}}
        self.writes: list[tuple[str, str, int]] = []
        self.read_fail: set[tuple[str, str]] = set()
        self.write_fail: set[tuple[str, str]] = set()

    def read(self, reg: str, motor: str, normalize: bool = True) -> int:
        if (reg, motor) in self.read_fail:
            raise ConnectionError("no response")
        return self.registers[reg][motor]

    def write(self, reg: str, motor: str, value: int, normalize: bool = True) -> None:
        if (reg, motor) in self.write_fail:
            raise ConnectionError("write NAK")
        self.writes.append((reg, motor, value))
        self.registers.setdefault(reg, {})[motor] = value


class _FakeDevice:
    def __init__(self, bus: _FakeBus) -> None:
        self.bus = bus
        self.is_connected = True

    def disconnect(self) -> None:
        self.is_connected = False


def _all_regs(value_of) -> dict[str, dict[str, int]]:
    """Build a {register: {motor: value}} store from a per-register value fn."""
    from makerlab.calibrate import _SNAPSHOT_REGISTERS

    return {reg: {m: value_of(reg, m) for m in _MOTORS} for reg in _SNAPSHOT_REGISTERS}


def _make_manager_with_snapshot(registers):
    """A CalibrationManager wired to a fake device, snapshot already captured."""
    from makerlab.calibrate import CalibrationManager

    bus = _FakeBus(registers)
    mgr = CalibrationManager()
    mgr.device = _FakeDevice(bus)
    mgr._snapshot_servo_state()
    return mgr, bus


def test_snapshot_captures_all_mutated_registers() -> None:
    from makerlab.calibrate import _SNAPSHOT_REGISTERS

    mgr, _ = _make_manager_with_snapshot(_all_regs(lambda reg, m: 111))
    assert set(mgr._servo_snapshot) == set(_MOTORS)
    for motor in _MOTORS:
        assert set(mgr._servo_snapshot[motor]) == set(_SNAPSHOT_REGISTERS)


def test_cancel_restores_snapshotted_servo_state() -> None:
    """After a mutation (as the homing step would do), a cancel restores every
    snapshotted register to its pre-session value."""
    orig = _all_regs(lambda reg, m: {"Homing_Offset": 500, "Min_Position_Limit": 0, "Max_Position_Limit": 4095}[reg])
    mgr, bus = _make_manager_with_snapshot(orig)

    # Simulate the homing step rewriting Homing_Offset and reset_calibration
    # rewriting the limits on every motor.
    for m in _MOTORS:
        bus.write("Homing_Offset", m, 1234)
        bus.write("Min_Position_Limit", m, 0)
        bus.write("Max_Position_Limit", m, 4095)

    mgr._cleanup_and_finish("Calibration cancelled", status="idle")

    # Registers are back to their original snapshot values.
    for m in _MOTORS:
        assert bus.registers["Homing_Offset"][m] == 500
    # Snapshot discarded (no double restore) and no rollback warning surfaced.
    assert mgr._servo_snapshot is None
    assert mgr.status.warning is None


def test_restore_failure_surfaces_warning() -> None:
    """When a restore write fails, the status carries an explicit inconsistency
    warning naming the offending motor/register."""
    mgr, bus = _make_manager_with_snapshot(_all_regs(lambda reg, m: 500))
    bus.write_fail = {("Homing_Offset", "elbow_flex")}

    mgr._cleanup_and_finish("Calibration cancelled", status="idle")

    assert mgr.status.warning is not None
    assert "may be inconsistent" in mgr.status.warning
    assert "elbow_flex.Homing_Offset" in mgr.status.warning


def test_success_discards_snapshot_so_no_restore_runs() -> None:
    """Once the snapshot is discarded (a successful save), cleanup must not write
    any register back — the freshly-saved calibration is the truth."""
    mgr, bus = _make_manager_with_snapshot(_all_regs(lambda reg, m: 500))

    # A successful save discards the snapshot.
    mgr._servo_snapshot = None
    writes_before = len(bus.writes)

    mgr._cleanup_and_finish("Calibration completed", status="completed")

    assert len(bus.writes) == writes_before  # no restore writes
    assert mgr.status.warning is None
