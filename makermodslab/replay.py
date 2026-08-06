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

"""Physical episode replay: drives the connected follower through a recorded
dataset episode's `action` column, open-loop, real-time paced, no policy and
no cameras involved.

Mirrors `teleoperate.py` in shape — in-process worker thread (there's no
model/vision to isolate, unlike rollout.py's subprocess), single global
session, mutex with every other feature that owns the same serial bus (see
CLAUDE.md's "State model & mutual exclusion"). Connects the follower
*synchronously* (mirrors handle_start_teleoperation, not
handle_start_inference's background-connect — replay's connect is fast, no
multi-minute Hub model download to hide behind a background worker), so a
connection failure is reported straight back to the caller; only the
ease-in + playback loop runs in the background thread.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from pydantic import BaseModel

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from .arm_identity import verify_devices
from .datasets import get_episode_action_series
from .motor_power import clear_goal_velocity, reset_torque_limit
from .rest_pose import RETURN_CEILING_S, capture_rest_pose, return_to_rest_pose
from .teleoperate import _cleanup_after_setup_failure, force_disable_torque
from .utils.config import get_robot_record, setup_follower_calibration_file

logger = logging.getLogger(__name__)

# v1 is single-arm only — a bimanual dataset/robot mismatch is a confusing
# error otherwise (left_/right_-prefixed action keys read as "every joint is
# missing"), and bimanual replay needs its own two-bus ease-in/playback shape
# this module doesn't implement yet.
_SINGLE_ARM_MODE = "single"

# Ease-in tolerance, in the SAME normalized units as robot.send_action() (not
# raw ticks — see rest_pose.py's normalize=True path). Deliberately smaller
# than RETURN_ARRIVE_TOLERANCE's raw-ticks value since it's compared in a
# different unit space; validate on real hardware before trusting this
# default (see the plan's manual hardware verification step) — arm joints
# and the gripper share this one tolerance despite having different
# normalized ranges (~-100..100 vs 0..100 by default), same simplification
# RETURN_ARRIVE_TOLERANCE already makes across differently-geared joints.
EASE_ARRIVE_TOLERANCE = 2.0

# Bound on how long stop_and_wait() waits for the worker to actually finish
# releasing the arm before forcing an immediate release — mirrors
# teleoperate.py's _STOP_AND_WAIT_TIMEOUT_S shape (RETURN_CEILING_S is
# rest_pose.py's own poll-loop ceiling).
_STOP_AND_WAIT_TIMEOUT_S = RETURN_CEILING_S + 5.0


class ReplayRequest(BaseModel):
    repo_id: str
    episode_index: int
    follower_port: str
    follower_config: str
    # Robot record name, used only to look up the record's `mode` for the
    # bimanual-rejection guard — the port/config above are what actually
    # drive the connection.
    robot_name: str = ""
    skip_identity_check: bool = False


replay_active: bool = False
replay_thread: threading.Thread | None = None
_state_lock = threading.Lock()
_replay_started_at: float | None = None
# {phase, episode_index, elapsed_s, duration_s, error, hint} — see
# handle_replay_status. phase is one of: idle | easing_in | playing |
# stopping | done | error.
_replay_meta: dict[str, Any] = {}


def _load_robot_record(robot_name: str) -> dict[str, Any] | None:
    """Thin wrapper so tests can monkeypatch the record lookup without
    touching disk — get_robot_record already does the real lookup."""
    return get_robot_record(robot_name)


def handle_start_replay(request: ReplayRequest, websocket_manager=None) -> dict[str, Any]:
    """Validate, connect the follower synchronously, and spawn the
    ease-in + playback worker. Returns a dict the route layer turns into a
    JSON response or HTTPException (status_code key present on every
    failure), mirroring rollout.py's InferenceRequest response shape."""
    global replay_active, replay_thread, _replay_started_at, _replay_meta

    from . import (
        auto_calibrate as _auto_calibrate,
        calibrate as _calibrate,
        record as _record,
        rollout as _rollout,
        teleoperate as _teleoperate,
        wiggle as _wiggle,
    )

    with _state_lock:
        if _teleoperate.teleoperation_active:
            return {
                "success": False,
                "status_code": 409,
                "message": "Teleoperation is currently active. Stop it first.",
            }
        if _record.recording_active:
            return {
                "success": False,
                "status_code": 409,
                "message": "Recording is currently active. Stop it first.",
            }
        if _rollout.inference_active:
            return {
                "success": False,
                "status_code": 409,
                "message": "Inference is currently active. Stop it first.",
            }
        if _calibrate.calibration_is_active():
            return {
                "success": False,
                "status_code": 409,
                "message": "Calibration is currently active. Stop it first.",
            }
        if _auto_calibrate.auto_calibration_is_active():
            return {
                "success": False,
                "status_code": 409,
                "message": "Auto-calibration is currently active. Stop it first.",
            }
        if _wiggle.wiggle_active:
            return {
                "success": False,
                "status_code": 409,
                "message": "A gripper wiggle is currently in progress. Wait for it to finish.",
            }
        if replay_active:
            return {
                "success": False,
                "status_code": 409,
                "message": "Replay is already active. Stop it first.",
            }
        if replay_thread is not None and replay_thread.is_alive():
            return {
                "success": False,
                "status_code": 409,
                "message": "The previous replay session is still shutting down. Try again in a few seconds.",
            }

        record = _load_robot_record(request.robot_name)
        if record is not None and record.get("mode") == "bimanual":
            return {
                "success": False,
                "status_code": 400,
                "message": "Bimanual replay isn't supported yet — select a single-arm robot.",
            }

        action_series = get_episode_action_series(request.repo_id, request.episode_index)
        if action_series is None:
            return {
                "success": False,
                "status_code": 400,
                "message": "Could not read this episode's recorded actions — it may not be downloaded locally yet.",
            }

        try:
            robot, identity_warnings = _connect_follower(request)
        except Exception as e:
            return {"success": False, "status_code": 500, "message": str(e)}

        if set(action_series["action_names"]) != set(robot.action_features.keys()):
            _cleanup_after_setup_failure(robot, None, "follower arm", "leader arm")
            return {
                "success": False,
                "status_code": 400,
                "message": "This dataset's joints don't match the connected robot.",
            }

        replay_active = True
        _replay_started_at = time.time()
        _replay_meta = {
            "phase": "easing_in",
            "episode_index": request.episode_index,
            "duration_s": action_series["timestamps"][-1] if action_series["timestamps"] else 0.0,
            "error": None,
            "hint": None,
        }

    worker = threading.Thread(
        target=_replay_worker,
        args=(robot, action_series, websocket_manager),
        name="replay-worker",
        daemon=True,
    )
    replay_thread = worker
    worker.start()

    response: dict[str, Any] = {"success": True, "message": "Replay starting"}
    if identity_warnings:
        response["warning"] = " ".join(identity_warnings)
    return response


def _connect_follower(request: ReplayRequest) -> tuple[SO101Follower, list[str]]:
    """Connect and configure the follower for replay — mirrors
    teleoperate.py's single-arm connect sequence (write_calibration →
    configure → reset_torque_limit → clear_goal_velocity), follower-only.
    Raises on a connection or hard identity-mismatch failure; the caller
    (handle_start_replay) is responsible for cleanup on that path."""
    follower_id = setup_follower_calibration_file(request.follower_config)
    robot = SO101Follower(SO101FollowerConfig(port=request.follower_port, id=follower_id))
    try:
        robot.bus.connect()
    except Exception as e:
        raise RuntimeError(
            f"Could not connect to the follower arm on {request.follower_port}. "
            "Make sure it's plugged in and powered on, then try again."
        ) from e

    identity_warnings = verify_devices(((robot, "follower"),), skip=request.skip_identity_check)

    robot.bus.write_calibration(robot.calibration)
    robot.configure()
    identity_warnings += reset_torque_limit(robot, "follower arm")
    identity_warnings += clear_goal_velocity(robot, "follower arm")
    return robot, identity_warnings


def _replay_worker(robot: SO101Follower, action_series: dict[str, Any], websocket_manager) -> None:
    """Ease the arm to the episode's first frame, then stream send_action()
    calls at the episode's recorded pace, broadcasting live joint feedback
    over the existing joint-data websocket every frame. On stop or
    completion, gently return to the pose captured at the start and release
    torque — mirrors teleoperation_worker's graceful-stop shape exactly."""
    global replay_active

    action_names = action_series["action_names"]
    timestamps = action_series["timestamps"]
    frames = action_series["values"]
    stop_event = threading.Event()

    def _stop_check() -> bool:
        return not replay_active or stop_event.is_set()

    start_pose = capture_rest_pose(robot.bus, normalize=False)

    try:
        if frames:
            frame0 = dict(zip(action_names, frames[0], strict=True))
            arrived, reason = return_to_rest_pose(
                robot.bus,
                frame0,
                abort_event=stop_event,
                label="follower arm",
                normalize=True,
                tolerance=EASE_ARRIVE_TOLERANCE,
            )
            if not arrived and reason != "cut-short":
                with _state_lock:
                    _replay_meta["phase"] = "error"
                    _replay_meta["error"] = f"Could not reach the episode's starting position ({reason})"
                    _replay_meta["hint"] = (
                        "The arm may be posed too far from this episode's start. "
                        "Try again, or reposition it closer first."
                    )
                return
            if reason == "cut-short" or _stop_check():
                return

            with _state_lock:
                _replay_meta["phase"] = "playing"

            t0 = time.monotonic()
            for ts, values in zip(timestamps, frames, strict=True):
                if _stop_check():
                    break
                target_t = t0 + ts
                while True:
                    now = time.monotonic()
                    if now >= target_t or _stop_check():
                        break
                    time.sleep(min(0.01, target_t - now))
                if _stop_check():
                    break

                action = dict(zip(action_names, values, strict=True))
                robot.send_action(action)

                with _state_lock:
                    _replay_meta["elapsed_s"] = time.monotonic() - t0

                if websocket_manager is not None and getattr(websocket_manager, "active_connections", None):
                    try:
                        observation = robot.get_observation()
                        websocket_manager.broadcast_joint_data_sync(
                            {"type": "replay_joint_update", "joints": observation, "timestamp": time.time()}
                        )
                    except Exception as e:
                        logger.warning(f"Could not broadcast replay joint feedback: {e}")

        with _state_lock:
            _replay_meta["phase"] = "stopping"
        return_to_rest_pose(robot.bus, start_pose, label="follower arm")
    except Exception as e:
        logger.error(f"Replay worker error: {e}")
        with _state_lock:
            _replay_meta["phase"] = "error"
            _replay_meta["error"] = str(e)
    finally:
        force_disable_torque(robot, "follower arm")
        try:
            robot.bus.disconnect(disable_torque=False)
        except Exception as e:
            logger.warning(f"Could not disconnect the follower after replay: {e}")
        with _state_lock:
            replay_active = False
            if _replay_meta.get("phase") not in ("error",):
                _replay_meta["phase"] = "done"


def handle_replay_status() -> dict[str, Any]:
    with _state_lock:
        elapsed_s = time.time() - _replay_started_at if _replay_started_at else 0.0
        return {
            "replay_active": replay_active,
            "phase": _replay_meta.get("phase", "idle"),
            "episode_index": _replay_meta.get("episode_index"),
            "elapsed_s": elapsed_s,
            "duration_s": _replay_meta.get("duration_s"),
            "error": _replay_meta.get("error"),
            "hint": _replay_meta.get("hint"),
        }


def handle_stop_replay() -> dict[str, Any]:
    global replay_active
    with _state_lock:
        if not replay_active:
            return {"success": False, "status_code": 409, "message": "No replay is active"}
        replay_active = False
        _replay_meta["phase"] = "stopping"
    return {"success": True, "message": "Replay stopping"}


def stop_and_wait(timeout: float = _STOP_AND_WAIT_TIMEOUT_S) -> None:
    """Stop replay and block until the worker has actually released the arm
    — for the FastAPI shutdown hook, which has no UI to poll and no "press
    Stop again" gesture available. Mirrors teleoperate.py's stop_and_wait.
    A no-op when idle."""
    if replay_active:
        handle_stop_replay()
    worker = replay_thread
    if worker is None or not worker.is_alive():
        return
    worker.join(timeout=timeout)
    if worker.is_alive():
        logger.warning("Replay worker did not finish releasing within %.0fs", timeout)
