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
"""Tests for lelab.auto_calibrate — subprocess manager (process mocked)."""

from __future__ import annotations

import pytest


def test_auto_calibration_rejects_bad_device() -> None:
    import lelab.auto_calibrate as ac

    mgr = ac.AutoCalibrationManager()
    result = mgr.start(ac.AutoCalibrationRequest(device_type="bogus", port="/dev/x", config_file="c"))
    assert result["success"] is False


def test_auto_calibration_rejects_empty_port() -> None:
    import lelab.auto_calibrate as ac

    mgr = ac.AutoCalibrationManager()
    result = mgr.start(ac.AutoCalibrationRequest(device_type="robot", port="", config_file="c"))
    assert result["success"] is False


def test_auto_calibration_status_idle() -> None:
    import lelab.auto_calibrate as ac

    status = ac.AutoCalibrationManager().get_status()
    assert status["status"] == "idle"
    assert status["active"] is False
    assert status["logs"] == []


def test_auto_calibration_launches_captures_logs_and_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful subprocess run captures its stdout and ends 'completed'."""
    import lelab.auto_calibrate as ac

    class FakeProc:
        def __init__(self) -> None:
            self.stdout = iter(["Stage 0: init\n", "calibration done\n"])

        def wait(self) -> int:
            return 0

        def terminate(self) -> None:
            pass

    monkeypatch.setattr(ac.subprocess, "Popen", lambda *a, **k: FakeProc())

    mgr = ac.AutoCalibrationManager()
    # No robot_name -> no record write-back, so no filesystem needed.
    result = mgr.start(ac.AutoCalibrationRequest(device_type="robot", port="/dev/x", config_file="my_arm"))
    assert result["success"] is True

    if mgr._thread is not None:
        mgr._thread.join(timeout=2)

    status = mgr.get_status()
    assert status["status"] == "completed"
    assert status["active"] is False
    assert any("Stage 0" in line for line in status["logs"])
