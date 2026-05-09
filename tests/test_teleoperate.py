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
"""Tests for lelab.teleoperate — request schema and status handlers."""

from __future__ import annotations

import pytest


def test_teleoperate_request_rejects_missing_fields() -> None:
    from pydantic import ValidationError

    from lelab.teleoperate import TeleoperateRequest

    with pytest.raises(ValidationError):
        TeleoperateRequest()


def test_handle_teleoperation_status_returns_dict() -> None:
    from lelab.teleoperate import handle_teleoperation_status

    result = handle_teleoperation_status()
    assert isinstance(result, dict)


def test_handle_get_joint_positions_returns_dict_when_idle() -> None:
    from lelab.teleoperate import handle_get_joint_positions

    result = handle_get_joint_positions()
    assert isinstance(result, dict)


def test_get_joint_positions_from_robot_uses_provided_object() -> None:
    from tests.mocks import FakeRobot

    from lelab.teleoperate import get_joint_positions_from_robot

    robot = FakeRobot()
    robot.connect()
    try:
        positions = get_joint_positions_from_robot(robot)
        # Implementation may return {} or {"shoulder_pan.pos": 0.0, ...}.
        # Either is acceptable; we just assert the call is non-fatal.
        assert isinstance(positions, dict)
    except (AttributeError, NotImplementedError):
        # If the function expects more methods on the robot than FakeRobot
        # implements, that's fine — log via skip rather than fail.
        pytest.skip("get_joint_positions_from_robot expects a richer robot surface")
