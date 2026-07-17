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
"""Tests for makerlab.motor_power — percent→register mapping, the mocked-bus
apply path (mirrors the FakeRobot/FakeBus patterns in tests/mocks.py), and
the one-shot supply-voltage read (hardware mocked, like tests/test_wiggle.py)."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient


class _FakeBus:
    """Minimal Feetech bus stand-in: records writes, can fail per motor."""

    def __init__(self, motors: list[str], fail: tuple[str, ...] = (), port: str = "/dev/fake") -> None:
        self.motors = dict.fromkeys(motors)
        self.port = port
        self.writes: list[tuple[str, str, Any, bool, int]] = []
        self._fail = set(fail)

    def write(
        self, data_name: str, motor: str, value: Any, *, normalize: bool = True, num_retry: int = 0
    ) -> None:
        if motor in self._fail:
            raise RuntimeError("write failed")
        self.writes.append((data_name, motor, value, normalize, num_retry))


class _FakeArm:
    def __init__(self, bus: _FakeBus) -> None:
        self.bus = bus


class _FakeBimanual:
    def __init__(self, left: _FakeBus, right: _FakeBus) -> None:
        self.left_arm = _FakeArm(left)
        self.right_arm = _FakeArm(right)


def test_torque_limit_from_percent_scales_and_clamps() -> None:
    from makerlab.motor_power import torque_limit_from_percent

    assert torque_limit_from_percent(100) == 1000
    assert torque_limit_from_percent(55) == 550
    assert torque_limit_from_percent(10) == 100
    # Clamped inputs: below range, above range, junk.
    assert torque_limit_from_percent(5) == 100
    assert torque_limit_from_percent(1000) == 1000
    # Junk/missing (None) falls back to DEFAULT_MOTOR_POWER (38 = the auto-cal
    # torque level, not full power) → 380. This is a fixed constant in
    # utils.config, not a persisted/per-machine read.
    assert torque_limit_from_percent(None) == 380


def test_apply_motor_power_writes_ram_register_to_every_motor() -> None:
    from makerlab.motor_power import apply_motor_power

    bus = _FakeBus(["shoulder_pan", "elbow_flex", "gripper"])
    warnings = apply_motor_power(_FakeArm(bus), 40)

    assert warnings == []
    assert len(bus.writes) == 3
    for data_name, _motor, value, normalize, num_retry in bus.writes:
        # RAM register only — never the EEPROM "Max_Torque_Limit".
        assert data_name == "Torque_Limit"
        assert value == 400
        assert normalize is False
        assert num_retry > 0


def test_apply_motor_power_writes_even_at_full_power() -> None:
    """100% still writes 1000: restores full power when a gentler previous
    session set the RAM register and the arm was never power-cycled."""
    from makerlab.motor_power import apply_motor_power

    bus = _FakeBus(["gripper"])
    apply_motor_power(_FakeArm(bus), 100)
    assert bus.writes == [("Torque_Limit", "gripper", 1000, False, 2)]


def test_apply_motor_power_failure_warns_but_does_not_abort() -> None:
    """One bad motor must not stop the others, and the failure surfaces as a
    warning message (degraded to previous/full power) — never an exception."""
    from makerlab.motor_power import apply_motor_power

    bus = _FakeBus(["shoulder_pan", "elbow_flex", "gripper"], fail=("elbow_flex",))
    warnings = apply_motor_power(_FakeArm(bus), 50)

    written = [w[1] for w in bus.writes]
    assert written == ["shoulder_pan", "gripper"]
    assert len(warnings) == 1
    assert "elbow_flex" in warnings[0]
    assert "50%" in warnings[0]
    assert "/dev/fake" in warnings[0]


def test_apply_motor_power_covers_both_bimanual_arms() -> None:
    from makerlab.motor_power import apply_motor_power

    left = _FakeBus(["gripper"], port="/dev/left")
    right = _FakeBus(["gripper"], port="/dev/right")
    warnings = apply_motor_power(_FakeBimanual(left, right), 30)

    assert warnings == []
    assert left.writes == [("Torque_Limit", "gripper", 300, False, 2)]
    assert right.writes == [("Torque_Limit", "gripper", 300, False, 2)]


def test_apply_motor_power_handles_none_device() -> None:
    from makerlab.motor_power import apply_motor_power

    assert apply_motor_power(None, 50) == []


# ---------------------------------------------------------------------------
# clear_goal_velocity: reset the RAM speed cap (Goal_Velocity=0) at session
# start so a leftover cap (auto-cal fold/unfold=1000, rest-pose return=400)
# can't throttle the next session. Mirrors apply_motor_power's shape.
# ---------------------------------------------------------------------------


def test_clear_goal_velocity_zeroes_every_follower_motor() -> None:
    from makerlab.motor_power import clear_goal_velocity

    bus = _FakeBus(["shoulder_pan", "elbow_flex", "gripper"])
    warnings = clear_goal_velocity(_FakeArm(bus))

    assert warnings == []
    assert len(bus.writes) == 3
    for data_name, _motor, value, normalize, num_retry in bus.writes:
        assert data_name == "Goal_Velocity"
        assert value == 0  # 0 = uncapped servo default
        assert normalize is False
        assert num_retry > 0
    assert {w[1] for w in bus.writes} == {"shoulder_pan", "elbow_flex", "gripper"}


def test_clear_goal_velocity_covers_both_bimanual_arms() -> None:
    from makerlab.motor_power import clear_goal_velocity

    left = _FakeBus(["gripper"], port="/dev/left")
    right = _FakeBus(["gripper"], port="/dev/right")
    warnings = clear_goal_velocity(_FakeBimanual(left, right))

    assert warnings == []
    assert left.writes == [("Goal_Velocity", "gripper", 0, False, 2)]
    assert right.writes == [("Goal_Velocity", "gripper", 0, False, 2)]


def test_clear_goal_velocity_failure_warns_but_does_not_abort() -> None:
    """One bad motor must not stop the others, and the failure surfaces as a
    warning message (the motor keeps its leftover cap) — never an exception, so
    the session start still proceeds."""
    from makerlab.motor_power import clear_goal_velocity

    bus = _FakeBus(["shoulder_pan", "elbow_flex", "gripper"], fail=("elbow_flex",))
    warnings = clear_goal_velocity(_FakeArm(bus))

    written = [w[1] for w in bus.writes]
    assert written == ["shoulder_pan", "gripper"]  # the good motors were still cleared
    assert len(warnings) == 1
    assert "elbow_flex" in warnings[0]
    assert "Goal_Velocity" in warnings[0]
    assert "/dev/fake" in warnings[0]


def test_clear_goal_velocity_handles_none_device() -> None:
    from makerlab.motor_power import clear_goal_velocity

    assert clear_goal_velocity(None) == []


def test_request_models_default_to_full_power() -> None:
    from makerlab.record import RecordingRequest
    from makerlab.rollout import InferenceRequest
    from makerlab.teleoperate import TeleoperateRequest

    teleop = TeleoperateRequest(
        leader_port="/dev/l", follower_port="/dev/f", leader_config="L", follower_config="F"
    )
    record = RecordingRequest(
        leader_port="/dev/l",
        follower_port="/dev/f",
        leader_config="L",
        follower_config="F",
        dataset_repo_id="user/data",
        single_task="task",
    )
    inference = InferenceRequest(follower_port="/dev/f", follower_config="F", policy_ref="/tmp/x")
    assert teleop.motor_power == 100
    assert record.motor_power == 100
    assert inference.motor_power == 100


def test_voltage_from_raw_scales_tenths_of_a_volt() -> None:
    """Present_Voltage is in 0.1 V units (121 raw = 12.1 V)."""
    from makerlab.motor_power import voltage_from_raw

    assert voltage_from_raw(121) == 12.1
    assert voltage_from_raw(0) == 0.0


async def test_read_supply_voltage_rejects_empty_port() -> None:
    from makerlab.motor_power import read_supply_voltage

    result = await read_supply_voltage("   ")
    assert result == {"success": False, "message": "No port provided."}


def test_supply_voltage_endpoint_success(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # Stub the blocking hardware call so the happy path runs without a device.
    monkeypatch.setattr("makerlab.motor_power._read_voltage_sync", lambda port: 12.1)

    response = client.get("/supply-voltage", params={"port": "/dev/fake"})
    assert response.status_code == 200
    assert response.json() == {"success": True, "voltage": 12.1}


def test_supply_voltage_endpoint_reports_hardware_failure(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(port: str) -> float:
        raise RuntimeError("no device on port")

    monkeypatch.setattr("makerlab.motor_power._read_voltage_sync", boom)

    response = client.get("/supply-voltage", params={"port": "/dev/fake"})
    # Logical failures stay HTTP 200 with success=False (like other handlers).
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert "no device on port" in body["message"]
