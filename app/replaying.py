import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import math

import numpy as np
from pydantic import BaseModel

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from . import dataset_browser

logger = logging.getLogger(__name__)

# Map dataset action motor names → URDF joint names.
# Keep in sync with app/teleoperating.py:get_joint_positions_from_robot.
_MOTOR_TO_URDF_JOINT = {
    "shoulder_pan": "Rotation",
    "shoulder_lift": "Pitch",
    "elbow_flex": "Elbow",
    "wrist_flex": "Wrist_Pitch",
    "wrist_roll": "Wrist_Roll",
    "gripper": "Jaw",
}
_DEG_TO_RAD = math.pi / 180.0


class StartReplayRequest(BaseModel):
    repo_id: str
    episode: int


class ReplayControlRequest(BaseModel):
    action: str  # "pause" | "resume" | "seek" | "set_speed"
    value: float | int | None = None


@dataclass
class ReplayState:
    active: bool = False
    repo_id: str | None = None
    episode: int | None = None
    frame: int = 0
    total_frames: int = 0
    fps: float = 30.0
    speed: float = 1.0
    paused: bool = False
    joint_names: list[str] = field(default_factory=list)


_state_lock = threading.Lock()
_state = ReplayState()
_actions: np.ndarray | None = None  # (T, J)
_stop_event = threading.Event()
_ticker_thread: threading.Thread | None = None


def _map_action_names_to_urdf(names: list[str]) -> list[str | None]:
    """Strip a trailing ``.pos`` suffix and map SO-101 motor names to URDF joint names.

    Returns one entry per input position. Unknown motors map to ``None`` so the
    ticker can skip them without changing column ordering.
    """
    out: list[str | None] = []
    for n in names:
        motor = n[:-4] if n.endswith(".pos") else n
        out.append(_MOTOR_TO_URDF_JOINT.get(motor))
    return out


def _ticker_loop(manager) -> None:
    global _state, _actions
    while not _stop_event.is_set():
        with _state_lock:
            paused = _state.paused
            frame = _state.frame
            total = _state.total_frames
            fps = _state.fps
            speed = _state.speed
            joint_names = _state.joint_names
            actions = _actions

        if paused or frame >= total:
            if frame >= total and not paused:
                with _state_lock:
                    _state.paused = True
                # Broadcast a terminal tick so the frontend can update its UI.
                if total > 0:
                    last_frame = total - 1
                    last_row = actions[last_frame] if actions is not None else None
                    last_joints = (
                        {
                            name: float(last_row[i]) * _DEG_TO_RAD
                            for i, name in enumerate(joint_names)
                            if name is not None
                        }
                        if last_row is not None else {}
                    )
                    manager.broadcast_joint_data_sync({
                        "type": "joint_update",
                        "joints": last_joints,
                        "timestamp": last_frame / fps,
                        "frame": last_frame,
                        "ended": True,
                    })
            time.sleep(0.05)
            continue

        if actions is None:
            time.sleep(0.05)
            continue

        row = actions[frame]
        joints = {
            name: float(row[i]) * _DEG_TO_RAD
            for i, name in enumerate(joint_names)
            if name is not None
        }
        manager.broadcast_joint_data_sync({
            "type": "joint_update",
            "joints": joints,
            "timestamp": frame / fps,
            "frame": frame,
        })

        with _state_lock:
            _state.frame = min(_state.frame + 1, _state.total_frames)

        # Sleep in slices so seek/stop responsiveness stays high at low speeds.
        target = 1.0 / max(fps * speed, 0.01)
        slept = 0.0
        while slept < target and not _stop_event.is_set():
            chunk = min(0.05, target - slept)
            time.sleep(chunk)
            slept += chunk


def handle_start_replay(req: StartReplayRequest, manager) -> dict[str, Any]:
    global _state, _actions, _ticker_thread

    # Concurrency guard: refuse if teleop or recording is active.
    from .teleoperating import teleoperation_active
    from .recording import recording_active
    if teleoperation_active or recording_active:
        return {"success": False, "message": "Stop teleoperation or recording first."}

    with _state_lock:
        if _state.active:
            return {"success": False, "message": "Replay already active. Stop it first."}

    try:
        assets = dataset_browser.get_replay_assets(req.repo_id, req.episode)
    except Exception as e:
        logger.exception("get_replay_assets failed")
        return {"success": False, "message": f"Could not resolve dataset assets: {e}"}

    try:
        ds = LeRobotDataset(req.repo_id, episodes=[req.episode], download_videos=False)
    except Exception as e:
        logger.exception("LeRobotDataset load failed")
        return {"success": False, "message": f"Failed to load episode: {e}"}

    try:
        action_col = ds.hf_dataset["action"]
    except Exception:
        # Older LeRobotDataset attribute layout
        action_col = [ds[i]["action"] for i in range(len(ds))]
    actions_np = np.asarray([np.asarray(a, dtype=np.float32) for a in action_col], dtype=np.float32)

    joint_names = _map_action_names_to_urdf(assets["joint_names"])
    if not any(n is not None for n in joint_names):
        return {"success": False, "message": "Dataset has no SO-101 motor names — incompatible with this viewer."}

    _stop_event.clear()
    with _state_lock:
        _state = ReplayState(
            active=True,
            repo_id=req.repo_id,
            episode=req.episode,
            frame=0,
            total_frames=int(actions_np.shape[0]),
            fps=float(assets["fps"]),
            speed=1.0,
            paused=False,
            joint_names=joint_names,
        )
    _actions = actions_np

    _ticker_thread = threading.Thread(target=_ticker_loop, args=(manager,), daemon=True)
    _ticker_thread.start()

    return {
        "success": True,
        "joint_names": [n for n in joint_names if n is not None],
        "cameras": assets["cameras"],
        "fps": float(assets["fps"]),
        "num_frames": int(actions_np.shape[0]),
    }


def handle_replay_control(req: ReplayControlRequest) -> dict[str, Any]:
    with _state_lock:
        if not _state.active:
            return {"success": False, "message": "No active replay session."}

        if req.action == "pause":
            _state.paused = True
        elif req.action == "resume":
            if _state.frame >= _state.total_frames:
                _state.frame = 0
            _state.paused = False
        elif req.action == "seek":
            if req.value is None:
                return {"success": False, "message": "seek requires a value (frame index)."}
            target = max(0, min(int(req.value), max(_state.total_frames - 1, 0)))
            _state.frame = target
        elif req.action == "set_speed":
            if req.value is None:
                return {"success": False, "message": "set_speed requires a value."}
            _state.speed = max(0.25, min(float(req.value), 16.0))
        else:
            return {"success": False, "message": f"Unknown action: {req.action}"}

    return {"success": True}


def handle_stop_replay() -> dict[str, Any]:
    global _state, _actions, _ticker_thread

    _stop_event.set()
    thread = _ticker_thread
    if thread is not None:
        thread.join(timeout=1.5)
    _ticker_thread = None

    with _state_lock:
        _state = ReplayState()
    _actions = None
    return {"success": True}


def handle_replay_status() -> dict[str, Any]:
    with _state_lock:
        return {
            "active": _state.active,
            "repo_id": _state.repo_id,
            "episode": _state.episode,
            "frame": _state.frame,
            "total_frames": _state.total_frames,
            "fps": _state.fps,
            "speed": _state.speed,
            "paused": _state.paused,
        }


def cleanup() -> None:
    handle_stop_replay()
