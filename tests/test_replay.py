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
"""Tests for makermodslab.replay — mutex, validation, and idle/status branches.

Per CLAUDE.md's testing policy, the worker thread's happy path (connect,
ease-in, real-time playback loop) is deliberately NOT unit-tested here —
only request schemas, pure helpers, and mutex/idle branches are."""

from __future__ import annotations

import threading

import pytest


@pytest.fixture(autouse=True)
def _reset_replay_globals(monkeypatch: pytest.MonkeyPatch):
    """Reset replay's module-level state around each test, mirroring
    tests/test_rollout.py's _reset_rollout_globals."""
    from makermodslab import replay

    monkeypatch.setattr(replay, "replay_active", False)
    monkeypatch.setattr(replay, "replay_thread", None)
    monkeypatch.setattr(replay, "_replay_meta", {})
    monkeypatch.setattr(replay, "_replay_started_at", None)


def _stub_request():
    from makermodslab.replay import ReplayRequest

    return ReplayRequest(
        repo_id="alice/pick",
        episode_index=0,
        follower_port="/dev/ttyUSB0",
        follower_config="robot_a",
    )


def test_handle_start_replay_blocked_when_teleoperation_active(monkeypatch) -> None:
    from makermodslab.replay import handle_start_replay

    monkeypatch.setattr("makermodslab.teleoperate.teleoperation_active", True)
    result = handle_start_replay(_stub_request())
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "Teleoperation" in result["message"]


def test_handle_start_replay_blocked_when_recording_active(monkeypatch) -> None:
    from makermodslab.replay import handle_start_replay

    monkeypatch.setattr("makermodslab.record.recording_active", True)
    result = handle_start_replay(_stub_request())
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "Recording" in result["message"]


def test_handle_start_replay_blocked_when_inference_active(monkeypatch) -> None:
    from makermodslab.replay import handle_start_replay

    monkeypatch.setattr("makermodslab.rollout.inference_active", True)
    result = handle_start_replay(_stub_request())
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "Inference" in result["message"]


def test_handle_start_replay_blocked_when_calibration_active(monkeypatch) -> None:
    from makermodslab.replay import handle_start_replay

    monkeypatch.setattr("makermodslab.calibrate.calibration_manager.status.calibration_active", True)
    result = handle_start_replay(_stub_request())
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "Calibration" in result["message"]


def test_handle_start_replay_blocked_when_auto_calibration_active(monkeypatch) -> None:
    from makermodslab.replay import handle_start_replay

    monkeypatch.setattr("makermodslab.auto_calibrate.auto_calibration_manager.status.active", True)
    result = handle_start_replay(_stub_request())
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "Auto-calibration" in result["message"]


def test_handle_start_replay_blocked_when_wiggle_active(monkeypatch) -> None:
    from makermodslab.replay import handle_start_replay

    monkeypatch.setattr("makermodslab.wiggle.wiggle_active", True)
    result = handle_start_replay(_stub_request())
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "wiggle" in result["message"].lower()


def test_handle_start_replay_blocked_when_already_active(monkeypatch) -> None:
    from makermodslab import replay

    monkeypatch.setattr(replay, "replay_active", True)
    result = replay.handle_start_replay(_stub_request())
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "already active" in result["message"].lower()


def test_handle_start_replay_blocked_while_previous_worker_still_alive(monkeypatch) -> None:
    """I-series regression shape: replay_active already False but the
    previous worker hasn't actually exited yet must still refuse a new
    start, not race it for the same serial port."""
    from makermodslab import replay

    alive_worker = threading.Thread(target=lambda: None)
    alive_worker._started = threading.Event()  # type: ignore[attr-defined]
    alive_worker.start()
    alive_worker.join()
    # Simulate "still alive" without a real long-running thread: patch is_alive.
    monkeypatch.setattr(alive_worker, "is_alive", lambda: True)
    monkeypatch.setattr(replay, "replay_thread", alive_worker)

    result = replay.handle_start_replay(_stub_request())
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "still" in result["message"].lower()


def test_handle_start_replay_rejects_bimanual_robot(monkeypatch, tmp_path) -> None:
    from makermodslab.replay import handle_start_replay

    monkeypatch.setattr(
        "makermodslab.replay._load_robot_record",
        lambda name: {"mode": "bimanual", "follower_port": "/dev/x", "follower_config": "c"},
    )
    result = handle_start_replay(_stub_request())
    assert result["success"] is False
    assert result["status_code"] == 400
    assert "bimanual" in result["message"].lower()


def test_handle_start_replay_rejects_action_name_mismatch(monkeypatch) -> None:
    from makermodslab.replay import handle_start_replay

    monkeypatch.setattr(
        "makermodslab.replay.get_episode_action_series",
        lambda repo_id, episode_index: {
            "action_names": ["not_a_real_joint.pos"],
            "timestamps": [0.0],
            "values": [[1.0]],
        },
    )

    class _FakeRobot:
        action_features = {"shoulder_pan.pos": float}

    monkeypatch.setattr("makermodslab.replay._connect_follower", lambda request: (_FakeRobot(), []))

    result = handle_start_replay(_stub_request())
    assert result["success"] is False
    assert result["status_code"] == 400
    assert "match" in result["message"].lower() or "joint" in result["message"].lower()


def test_handle_start_replay_returns_400_when_episode_has_no_action_data(monkeypatch) -> None:
    from makermodslab.replay import handle_start_replay

    monkeypatch.setattr("makermodslab.replay.get_episode_action_series", lambda repo_id, episode_index: None)

    result = handle_start_replay(_stub_request())
    assert result["success"] is False
    assert result["status_code"] == 400


def test_handle_replay_status_idle() -> None:
    from makermodslab.replay import handle_replay_status

    status = handle_replay_status()
    assert status["replay_active"] is False
    assert status["phase"] == "idle"


def test_handle_stop_replay_when_idle_is_a_noop() -> None:
    from makermodslab.replay import handle_stop_replay

    result = handle_stop_replay()
    assert result["success"] is False
    assert result["status_code"] == 409
    assert "No replay" in result["message"]
