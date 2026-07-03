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

import json
import logging
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.datasets import LeRobotDataset
from lerobot.robots.bi_so_follower import BiSOFollowerConfig
from lerobot.robots.so_follower import SO101FollowerConfig

# Import the main record functionality to reuse it
from lerobot.scripts.lerobot_record import RecordConfig
from lerobot.teleoperators.bi_so_leader import BiSOLeaderConfig
from lerobot.teleoperators.so_leader import SO101LeaderConfig

from .arm_identity import ArmIdentityError, verify_devices
from .camera_preview import camera_preview_manager
from .datasets import _lerobot_cache_root, invalidate_hub_status
from .motor_power import apply_motor_power, clear_goal_velocity
from .rest_pose import capture_rest_pose
from .teleoperate import _device_buses, _return_followers_to_rest, force_disable_torque
from .utils.config import (
    bimanual_base_id,
    setup_calibration_files,
    stage_bimanual_calibrations,
    validate_dataset_repo_id,
    with_lelab_tag,
)

logger = logging.getLogger(__name__)

# Global variables for recording state
recording_active = False
recording_thread: threading.Thread | None = None
recording_events = None  # Events dict for controlling recording session
recording_config = None  # Store recording configuration
recording_start_time = None  # Track when recording started
session_end_elapsed_seconds = None  # Final session duration after the run ends
current_episode = 1  # Track current episode number
saved_episodes = 0  # Track how many episodes have been saved
current_phase = "preparing"  # Track current phase: "preparing", "recording", "resetting", "completed"
phase_start_time = None  # Track when current phase started
# True when the most recent session saved zero episodes and its (freshly
# created) dataset directory was discarded. Surfaced in the session-end status
# so the frontend can tell the user nothing was kept (see Upload.tsx).
last_session_discarded_empty = False
# Warn-but-allow arm-identity findings from the current session's guard (see
# lelab/arm_identity.py). The guard runs inside the recording worker (after the
# start response has already been sent), so the messages are surfaced through
# the /recording-status payload instead of the start response.
identity_warnings: list[str] = []
# Guards the start path so two concurrent POST /start-recording calls cannot
# both pass the active-flag check.
_state_lock = threading.Lock()

# True while the session's cleanup is driving the follower(s) back to their
# session-start pose (and on through the release): the recording loop is over
# but the arms are still energized (holding position, then moving home) and the
# serial ports are still held. Surfaced in the status payload so the UI isn't
# lying about the arm's state. No timed hold anymore — a normal stop returns the
# follower to where the session started, then releases (same as teleop; see
# lelab/rest_pose.py).
releasing = False
# Cuts the post-stop return short: set by a second stop request ("release now")
# or by a new start request that needs the serial ports. Cleared on start.
_release_now = threading.Event()


def finish_pending_release(timeout: float = 10.0) -> bool:
    """Cut a pending torque-release grace short and wait for its cleanup.

    Called by start paths (recording and teleoperation) so a start arriving
    during the grace window releases the arms and frees the serial ports
    immediately instead of failing port-busy for the rest of the grace.
    Returns True when no recording worker is running afterwards; False when a
    live session is still recording or the worker did not exit in time.
    """
    worker = recording_thread
    if worker is None or not worker.is_alive():
        return True
    if not releasing:
        # A live recording session, not a pending release — the caller's
        # mutex check will report "already active".
        return False
    _release_now.set()
    worker.join(timeout=timeout)
    return not worker.is_alive()


class RecordingRequest(BaseModel):
    leader_port: str
    follower_port: str
    leader_config: str
    follower_config: str
    # Bimanual: the primary pair above is the LEFT arm; these add the right arm.
    mode: str = "single"
    right_leader_port: str = ""
    right_follower_port: str = ""
    right_leader_config: str = ""
    right_follower_config: str = ""
    # Robot record name — used only as the BiSO staging base id (bimanual). It
    # decides the on-disk staging dir, not which calibration drives which arm.
    # Blank/invalid falls back to DEFAULT_BIMANUAL_BASE.
    robot_name: str = ""
    dataset_repo_id: str
    single_task: str
    num_episodes: int = 5
    episode_time_s: int = 30
    reset_time_s: int = 10
    fps: int = 30
    video: bool = True
    push_to_hub: bool = False
    tags: list[str] = []
    private: bool = False
    resume: bool = False
    streaming_encoding: bool = True
    cameras: dict = {}
    test_mode: bool = False  # Skip robot connection for testing
    # Escape hatch for the arm-identity guard (see lelab/arm_identity.py):
    # when true, record even if the connected arms don't match their calibrations.
    skip_identity_check: bool = False
    # Follower torque as a percentage of full power (see lelab/motor_power.py).
    # Applied to follower motors only; clamped server-side to 10-100.
    motor_power: int = 100


class UploadRequest(BaseModel):
    dataset_repo_id: str
    tags: list[str] = []
    private: bool = False


class DatasetInfoRequest(BaseModel):
    dataset_repo_id: str


def _platform_backend():
    """Pin the OpenCV backend per-platform so the index→camera mapping matches
    what the /available-cameras thumbnails were captured with. cv2.CAP_ANY can
    pick different backends across calls on macOS, silently reordering cameras
    between the modal preview and the recording."""
    import platform

    from lerobot.cameras.configs import Cv2Backends

    system = platform.system()
    if system == "Darwin":
        return Cv2Backends.AVFOUNDATION
    if system == "Linux":
        return Cv2Backends.V4L2
    if system == "Windows":
        # DirectShow, matching the order /available-cameras enumerates (via
        # pygrabber) so a camera_index always opens the previewed device.
        return Cv2Backends.DSHOW
    return Cv2Backends.ANY


def _build_camera_configs(cameras: dict, default_backend) -> dict:
    """Convert the frontend camera dict into OpenCVCameraConfig objects.

    `backend` (a Cv2Backends name) and `fourcc` (a 4-char code) are optional per
    camera; when omitted they fall back to `default_backend` and auto-detect.
    """
    from lerobot.cameras.configs import Cv2Backends
    from lerobot.cameras.opencv import OpenCVCameraConfig

    camera_configs: dict = {}
    for camera_name, camera_data in cameras.items():
        if camera_data.get("type") != "opencv":
            logger.warning(
                f"⚠️ CAMERA CONFIG: Unsupported camera type '{camera_data.get('type')}' for {camera_name}"
            )
            continue

        backend_name = camera_data.get("backend")
        backend = Cv2Backends[backend_name] if backend_name else default_backend
        fourcc = camera_data.get("fourcc") or None

        camera_configs[camera_name] = OpenCVCameraConfig(
            index_or_path=camera_data.get("camera_index", 0),
            backend=backend,
            fps=camera_data.get("fps"),
            width=camera_data.get("width"),
            height=camera_data.get("height"),
            fourcc=fourcc,
        )
        logger.info(
            f"✅ CAMERA CONFIG: {camera_name} -> OpenCVCameraConfig("
            f"index={camera_data.get('camera_index')}, backend={backend.name}, "
            f"{camera_data.get('width')}x{camera_data.get('height')}@{camera_data.get('fps')}fps, "
            f"fourcc={fourcc})"
        )
    return camera_configs


def create_record_config(request: RecordingRequest) -> RecordConfig:
    """Create a RecordConfig from the recording request"""
    # Convert the frontend camera dict into OpenCVCameraConfig objects. Backend
    # defaults to the platform pin unless the request overrides it per camera.
    camera_configs = _build_camera_configs(request.cameras, _platform_backend())

    if request.mode == "bimanual":
        # Build a lerobot BiSO leader+follower pair. lerobot loads each sub-arm's
        # calibration as "<base>_left/right.json" from a single calibration_dir,
        # with no way to point left/right at differently named library files. So
        # stage the four arbitrarily-named library calibrations into per-device
        # dirs as "<base>_left/right.json" and point BiSO at those (otherwise the
        # sub-arms have no calibration and connect() would try to interactively
        # recalibrate — which hangs the record thread). Cameras go on the left
        # follower arm (exposed prefixed "left_*"). The staging copy fails fast
        # with a clear per-slot error if any library file is missing.
        base = bimanual_base_id(request.robot_name)
        leader_staging, follower_staging, _ = stage_bimanual_calibrations(
            base,
            request.leader_config,
            request.right_leader_config,
            request.follower_config,
            request.right_follower_config,
        )
        robot_config = BiSOFollowerConfig(
            id=base,
            calibration_dir=Path(follower_staging),
            left_arm_config=SO101FollowerConfig(port=request.follower_port, cameras=camera_configs),
            right_arm_config=SO101FollowerConfig(port=request.right_follower_port),
        )
        teleop_config = BiSOLeaderConfig(
            id=base,
            calibration_dir=Path(leader_staging),
            left_arm_config=SO101LeaderConfig(port=request.leader_port),
            right_arm_config=SO101LeaderConfig(port=request.right_leader_port),
        )
    else:
        # Setup calibration files
        leader_config_name, follower_config_name = setup_calibration_files(
            request.leader_config, request.follower_config
        )

        # Create robot config
        robot_config = SO101FollowerConfig(
            port=request.follower_port,
            id=follower_config_name,
            cameras=camera_configs,
        )

        # Create teleop config
        teleop_config = SO101LeaderConfig(
            port=request.leader_port,
            id=leader_config_name,
        )

    # Create dataset config
    dataset_config = DatasetRecordConfig(
        repo_id=request.dataset_repo_id,
        single_task=request.single_task,
        num_episodes=request.num_episodes,
        episode_time_s=request.episode_time_s,
        reset_time_s=request.reset_time_s,
        fps=request.fps,
        video=request.video,
        push_to_hub=request.push_to_hub,
        # Upstream typing: tags is `list[str] | None`. None when push is off
        # keeps the lerobot default.
        tags=with_lelab_tag(request.tags) if request.push_to_hub else None,
        private=request.private,
        streaming_encoding=request.streaming_encoding,
    )

    # Create the main record config
    record_config = RecordConfig(
        robot=robot_config,
        teleop=teleop_config,
        dataset=dataset_config,
        resume=request.resume,
        display_data=False,  # Don't display data in API mode
        play_sounds=False,  # Don't play sounds in API mode
    )

    return record_config


def handle_start_recording(request: RecordingRequest) -> dict[str, Any]:
    """Handle start recording request by using the existing record() function"""
    global \
        recording_active, \
        releasing, \
        recording_thread, \
        recording_events, \
        recording_config, \
        recording_start_time, \
        session_end_elapsed_seconds, \
        current_episode, \
        saved_episodes, \
        current_phase, \
        phase_start_time, \
        last_session_discarded_empty, \
        identity_warnings

    from . import rollout as _rollout, teleoperate as _teleoperate

    # Claim the active flag under the lock so two concurrent starts can't both
    # pass the precondition check.
    logger.info(
        "Recording start requested: dataset=%r, task=%r, resume=%s, mode=%s",
        request.dataset_repo_id,
        request.single_task,
        request.resume,
        getattr(request, "mode", "single"),
    )

    # A previous session (recording or teleop) may still be holding torque for
    # its release grace — cut it short so this start doesn't fail on a busy
    # serial port. Best effort: failures are surfaced by the checks below.
    finish_pending_release()
    _teleoperate.finish_pending_release()

    with _state_lock:
        if recording_active:
            return {
                "success": False,
                "message": (
                    "The previous session is still releasing the arms. Try again in a few seconds."
                    if releasing
                    else "Recording is already active"
                ),
            }
        if _teleoperate.teleoperation_active:
            return {"success": False, "message": "Teleoperation is currently active. Stop it first."}
        if _rollout.inference_active:
            return {"success": False, "message": "Inference is currently active. Stop it first."}
        # Refuse a malformed dataset name up front (before claiming the flag or
        # touching hardware). Rejecting beats silent sanitization: "whoo/" used to
        # smuggle in a namespace and land the dataset at "user/whoo/".
        name_ok, name_reason = validate_dataset_repo_id(request.dataset_repo_id)
        if not name_ok:
            logger.warning(
                "Rejected recording start: invalid dataset name %r (%s)",
                request.dataset_repo_id,
                name_reason,
            )
            return {"success": False, "message": name_reason}
        # Per-session state reset, under the same lock that claims the active
        # flag: a stale _release_now from a previous session's double-stop
        # would otherwise cut EVERY later release grace short until the server
        # restarts (regression-tested in tests/test_record.py).
        recording_active = True
        releasing = False
        _release_now.clear()
        recording_thread = None
        recording_events = None
        recording_config = None
        recording_start_time = None
        session_end_elapsed_seconds = None
        current_episode = 1
        saved_episodes = 0
        current_phase = "preparing"
        phase_start_time = None
        last_session_discarded_empty = False
        identity_warnings = []

    try:
        # Backend camera previews (GET /camera-preview/{index}) hold the cv2
        # devices this session is about to open — recording always wins, so
        # force-release them now, before any robot/camera construction and
        # before the worker's 2s browser-stream release sleep.
        camera_preview_manager.stop_all()

        # The name is already validated (validate_dataset_repo_id in the lock), so
        # no sanitization is needed here. Stamp the repo_id with a timestamp
        # (matches lerobot-record CLI behavior) so each session lands in a unique
        # directory and the frontend gets the final id back in the response.
        if not request.resume and request.dataset_repo_id:
            request.dataset_repo_id = f"{request.dataset_repo_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Starting recording for dataset: {request.dataset_repo_id}")
        logger.info(f"Task: {request.single_task}")

        recording_config = request
        recording_events = {
            "exit_early": False,  # Right arrow key -> "Skip to next episode" button
            "stop_recording": False,  # ESC key -> "Stop recording" button
            "rerecord_episode": False,  # Left arrow key -> "Re-record episode" button
        }

        record_config = create_record_config(request)

        def recording_worker():
            global \
                recording_active, \
                recording_start_time, \
                session_end_elapsed_seconds, \
                current_phase, \
                phase_start_time, \
                current_episode, \
                saved_episodes, \
                last_session_discarded_empty
            recording_start_time = time.time()
            current_episode = 1
            saved_episodes = 0

            try:
                logger.info(
                    "Recording session started: dataset=%s task=%r episodes=%d",
                    request.dataset_repo_id,
                    request.single_task,
                    request.num_episodes,
                )

                # Give the frontend's camera streams time to release the
                # underlying devices before lerobot tries to open them.
                if request.cameras:
                    logger.info(
                        "Waiting for camera resources to be released (cameras: %s)",
                        list(request.cameras.keys()),
                    )
                    time.sleep(2.0)

                # Bimanual: the sub-arm ids are BiSO staging aliases, so give the
                # identity guard the real library stems (in arm-iteration order:
                # left follower, right follower, left leader, right leader).
                identity_config_names = (
                    [
                        request.follower_config,
                        request.right_follower_config,
                        request.leader_config,
                        request.right_leader_config,
                    ]
                    if request.mode == "bimanual"
                    else None
                )
                dataset = record_with_web_events(
                    record_config,
                    recording_events,
                    skip_identity_check=request.skip_identity_check,
                    motor_power=request.motor_power,
                    identity_config_names=identity_config_names,
                )
                logger.info(f"Recording completed successfully. Dataset has {dataset.num_episodes} episodes")
            except Exception:
                logger.exception("Recording session failed")
                current_phase = "error"
                if recording_start_time:
                    session_end_elapsed_seconds = int(time.time() - recording_start_time)
            finally:
                if current_phase != "error":
                    current_phase = "completed"
                if recording_start_time:
                    session_end_elapsed_seconds = int(time.time() - recording_start_time)

                # Discard a dataset this session created but never wrote an
                # episode into (interrupted/failed session, or every take
                # re-recorded away). Ordered here — after record_with_web_events
                # has returned/raised, which released torque and disconnected in
                # its own finally — so cleanup never blocks the hardware release.
                # Best-effort: _discard_empty_dataset swallows its own errors and
                # never re-raises, so the original error path is preserved.
                # Guarded on the in-memory counter AND, inside the helper, the
                # on-disk episode count; resume sessions are never touched.
                if saved_episodes == 0:
                    last_session_discarded_empty = _discard_empty_dataset(
                        request.dataset_repo_id, request.resume
                    )

                recording_active = False
                recording_start_time = None
                phase_start_time = None
                current_episode = 1
                saved_episodes = 0
                logger.info("Recording session ended")

        recording_thread = threading.Thread(target=recording_worker, name="recording-worker", daemon=True)
        recording_thread.start()

        return {
            "success": True,
            "message": "Recording started successfully",
            "dataset_id": request.dataset_repo_id,
            "num_episodes": request.num_episodes,
        }

    except Exception as e:
        recording_active = False
        logger.error(f"Failed to start recording: {e}")
        return {"success": False, "message": f"Failed to start recording: {str(e)}"}


def handle_stop_recording() -> dict[str, Any]:
    """Handle stop recording request - replaces ESC key.

    A second stop while the session-end cleanup is holding torque for the
    release grace cuts the hold short ("release now").
    """
    global current_phase, phase_start_time

    if releasing:
        _release_now.set()
        logger.info("Second stop during the release grace — releasing the arms now")
        return {
            "success": True,
            "message": "Releasing the arms now",
            "session_ending": True,
        }

    if not recording_active or recording_events is None:
        return {"success": False, "message": "No recording session is active"}

    recording_events["stop_recording"] = True
    recording_events["exit_early"] = True
    current_phase = "stopping"
    phase_start_time = None
    logger.info("Stop recording triggered from web interface")
    return {
        "success": True,
        "message": (
            "Recording stop requested. When the session ends, the arm returns to its starting "
            "position, then goes limp — press Stop again to release it immediately."
        ),
        "session_ending": True,
    }


def handle_exit_early() -> dict[str, Any]:
    """Handle exit early request - replaces right arrow key"""
    if not recording_active or recording_events is None:
        return {"success": False, "message": "No recording session is active"}
    recording_events["exit_early"] = True
    # Tracking flag that record_loop won't reset, so the worker can tell
    # "user pressed skip" from "control_time_s elapsed naturally".
    recording_events["_exit_early_triggered"] = True
    logger.info("Exit early triggered (current phase: %s)", current_phase)
    phase_name = "recording phase" if current_phase == "recording" else "reset phase"
    return {
        "success": True,
        "message": f"Exit early triggered successfully for {phase_name}",
        "current_phase": current_phase,
        "events_state": dict(recording_events),
    }


def handle_rerecord_episode() -> dict[str, Any]:
    """Handle rerecord episode request - replaces left arrow key"""
    if not recording_active or recording_events is None:
        return {"success": False, "message": "No recording session is active"}
    recording_events["rerecord_episode"] = True
    recording_events["exit_early"] = True
    logger.info("Re-record episode triggered")
    return {
        "success": True,
        "message": "Re-record episode requested successfully",
        "events_state": dict(recording_events),
    }


def handle_recording_status() -> dict[str, Any]:
    """Handle recording status request"""
    # If recording is not active and phase is completed or error, indicate session has ended
    session_ended = not recording_active and current_phase in ["completed", "error"]

    # Log when session has ended to help debug frontend polling
    if session_ended:
        if current_phase == "error":
            logger.info(
                "📡 RECORDING STATUS REQUEST: Session failed with error - frontend should stop polling"
            )
            print("📡 STATUS CHANGE: Frontend is still polling after session error - should stop now")
        else:
            logger.info("📡 RECORDING STATUS REQUEST: Session has ended - frontend should stop polling")
            print("📡 STATUS CHANGE: Frontend is still polling after session end - should stop now")

    status = {
        "recording_active": recording_active,
        "current_phase": current_phase,  # "preparing", "recording", "resetting", "completed"
        "session_ended": session_ended,  # New field to indicate session completion
        # True during the post-session rest-pose return: the recording loop is
        # over but the arm is still energized and driving back to its
        # session-start pose before torque is released.
        "releasing": releasing,
        "available_controls": {
            "stop_recording": recording_active,  # ESC key replacement
            "exit_early": recording_active,  # Right arrow key replacement
            "rerecord_episode": recording_active
            and current_phase == "recording",  # Only during recording phase
        },
        "message": "Recording session failed with error - check logs"
        if current_phase == "error"
        else (
            "Returning the arm to its rest position…"
            if releasing
            else (
                "Recording session has ended - stop polling"
                if session_ended
                else "Recording status retrieved successfully"
            )
        ),
    }

    # Always echo the stamped dataset id whenever a config exists, so the frontend
    # can read the actual on-disk repo_id (post stamp) for upload navigation.
    if recording_config:
        status["dataset_repo_id"] = recording_config.dataset_repo_id

    # When the session has ended, tell the frontend honestly whether anything
    # was kept. A session that saved zero episodes had its (freshly created)
    # dataset directory discarded — the post-recording page shows a "nothing was
    # saved" variant and does NOT link the (now-gone) repo id.
    if session_ended:
        status["discarded_empty"] = last_session_discarded_empty

    # Warn-but-allow arm-identity findings (the guard runs in the worker, after
    # the start response) — the frontend shows these as a non-blocking toast.
    if identity_warnings:
        status["warning"] = " ".join(identity_warnings)

    # Add episode information if recording is active
    if recording_active and recording_config:
        status["current_episode"] = current_episode
        status["total_episodes"] = recording_config.num_episodes
        status["saved_episodes"] = saved_episodes  # Track completed episodes

        # Add session start time if available
        if recording_start_time:
            status["session_start_time"] = recording_start_time
            status["session_elapsed_seconds"] = int(time.time() - recording_start_time)

        # Add phase timing information
        if phase_start_time:
            status["phase_start_time"] = phase_start_time
            status["phase_elapsed_seconds"] = int(time.time() - phase_start_time)

            # Add phase time limits
            if current_phase == "recording":
                status["phase_time_limit_s"] = recording_config.episode_time_s
            elif current_phase == "resetting":
                status["phase_time_limit_s"] = recording_config.reset_time_s
    elif session_end_elapsed_seconds is not None:
        status["session_elapsed_seconds"] = session_end_elapsed_seconds

    return status


def handle_delete_dataset(request: DatasetInfoRequest) -> dict[str, Any]:
    """Remove a recorded dataset's directory from local disk."""
    from pathlib import Path

    from lerobot.utils.constants import HF_LEROBOT_HOME

    repo_id = request.dataset_repo_id
    root = Path(HF_LEROBOT_HOME).resolve()
    target = (root / repo_id).resolve()

    # Reject path traversal: target must stay strictly inside HF_LEROBOT_HOME.
    if target == root or root not in target.parents:
        return {"success": False, "message": "Invalid dataset path"}

    # Don't yank the directory out from under an in-flight push to the Hub.
    if upload_manager.state == "running" and upload_manager.repo_id == repo_id:
        return {
            "success": False,
            "message": "This dataset is being uploaded to the Hub right now. Wait for it to finish.",
        }

    if not target.exists():
        return {"success": False, "message": f"Dataset not found on disk: {repo_id}"}

    try:
        shutil.rmtree(target)
    except Exception as e:
        logger.error(f"Failed to delete dataset {repo_id}: {e}")
        return {"success": False, "message": f"Failed to delete dataset: {e}"}

    logger.info(f"Deleted dataset directory {target}")
    return {"success": True, "message": f"Deleted {repo_id}"}


def _discard_empty_dataset(repo_id: str, resume: bool) -> bool:
    """Remove the directory of a session that saved zero episodes.

    Interrupted/failed recording sessions used to leave 0-episode datasets on
    disk — hundreds of MB of video, invisible in the picker (empties are
    hidden) yet still consuming space. When a session ends having saved no
    episodes, delete the directory THIS session created.

    Guards (all must hold before anything is removed):
      * ``resume`` is False — a resume/append session writes into a
        pre-existing dataset, so we must NEVER delete it even at zero *new*
        episodes. Only non-resume sessions stamp a fresh timestamped directory
        (see handle_start_recording), so only those are ours to discard.
      * The directory reports zero episodes — confirmed against ``meta/info.json``'s
        ``total_episodes`` (the same signal the picker uses), not just the
        in-memory counter.
      * The path stays strictly inside the LeRobot cache root (traversal guard,
        mirroring handle_delete_dataset).

    Best-effort: any failure is logged as a warning and swallowed — this runs
    during session-end cleanup and must never mask an original error or block
    the hardware release. Returns True iff the directory was removed.
    """
    if resume:
        # Append-into-existing: the dataset predates this session. Never delete.
        return False
    if not repo_id:
        return False

    root = _lerobot_cache_root().resolve()
    try:
        target = (root / repo_id).resolve()
    except OSError:
        return False

    # Reject path traversal: target must stay strictly inside the cache root.
    if target == root or root not in target.parents:
        return False
    if not target.is_dir():
        return False

    # Confirm zero episodes against the on-disk metadata, not just the counter.
    info_path = target / "meta" / "info.json"
    try:
        info = json.loads(info_path.read_text())
    except (OSError, ValueError):
        # No readable info.json — the dataset was never created far enough to
        # hold episodes. Treat as empty and clean it up.
        info = {}
    if info.get("total_episodes"):
        return False

    try:
        shutil.rmtree(target)
    except Exception as e:
        logger.warning(f"Failed to remove empty dataset {repo_id}: {e}")
        return False

    # Invalidate the cached Hub-existence probe (cheap correctness — the
    # repo no longer exists here).
    invalidate_hub_status(repo_id)

    logger.info(f"Removed empty dataset {repo_id} — no episodes were saved.")
    return True


def _upload_auth_error(exc: Exception) -> dict[str, str] | None:
    """If ``exc`` is a Hub auth failure, return the friendly {message, docs_url}
    the frontend shows for it; else None. Kept separate so the sync worker and
    any future caller map the 401 identically."""
    err_text = str(exc).lower()
    looks_like_auth = any(
        m in err_text
        for m in ("401", "you must be authenticated", "authentication required", "huggingfacehub_token")
    )
    if looks_like_auth:
        return {
            "message": (
                "You're not logged into the Hugging Face Hub. Run `hf auth login` in your "
                "terminal, then retry."
            ),
            "docs_url": "https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication",
        }
    return None


class UploadManager:
    """Runs one dataset upload at a time in a background thread.

    ``push_to_hub`` copies 100+ MB of video/parquet over the network and takes
    minutes, so we run it off the request thread (same start/poll shape as
    MergeManager) rather than block the browser on a multi-minute HTTP request
    that a navigation-away would abort mid-push. One upload at a time: a second
    concurrent start for any repo is refused (409-mapped by the route). The
    per-repo status lets the info card / picker row poll "is *my* dataset
    uploading?" and survive navigation.
    """

    def __init__(self) -> None:
        self.state: str = "idle"  # "idle" | "running" | "done" | "error"
        self.repo_id: str | None = None
        self.message: str | None = None
        self.dataset_url: str | None = None
        self.docs_url: str | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self, request: UploadRequest) -> dict[str, Any]:
        repo_id = request.dataset_repo_id
        with self._lock:
            if self.state == "running":
                return {
                    "started": False,
                    "repo_id": self.repo_id,
                    "message": f"An upload is already running for {self.repo_id}",
                }
            # Refuse a dataset another operation is actively writing — pushing a
            # half-written directory would ship a corrupt dataset. Reuses the
            # rename busy-guard (recording / merge / local training); lazy import
            # to avoid the datasets<->record cycle documented in _dataset_in_use.
            from .datasets import _dataset_in_use

            in_use = _dataset_in_use(repo_id)
            if in_use is not None:
                return {"started": False, "repo_id": repo_id, "message": in_use}
            self.state = "running"
            self.repo_id = repo_id
            self.message = f"Uploading {repo_id} to the Hub…"
            self.dataset_url = None
            self.docs_url = None

        self._thread = threading.Thread(
            target=self._worker, args=(request,), name="upload-worker", daemon=True
        )
        self._thread.start()
        return {"started": True, "repo_id": repo_id, "message": "Upload started"}

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            status = {
                "state": self.state,
                "repo_id": self.repo_id,
                "message": self.message,
                "dataset_url": self.dataset_url,
            }
            if self.docs_url is not None:
                status["docs_url"] = self.docs_url
            return status

    def _worker(self, request: UploadRequest) -> None:
        from lerobot.datasets import LeRobotDataset

        repo_id = request.dataset_repo_id
        try:
            logger.info(f"Loading dataset {repo_id} for upload")
            dataset = LeRobotDataset(repo_id)
            logger.info(f"Dataset loaded with {dataset.num_episodes} episodes")

            tags = with_lelab_tag(request.tags)
            logger.info(f"Uploading to HuggingFace Hub with tags: {tags}, private: {request.private}")
            dataset.push_to_hub(tags=tags, private=request.private)
            logger.info(f"Dataset {repo_id} uploaded successfully to HuggingFace Hub")

            # The dataset now exists on the Hub; drop any cached "local_only"
            # answer so the info card's next hub-status check flips to "On Hub".
            invalidate_hub_status(repo_id)

            with self._lock:
                self.state = "done"
                self.message = f"Dataset {repo_id} uploaded successfully to the Hugging Face Hub"
                self.dataset_url = f"https://huggingface.co/datasets/{repo_id}"
                self.docs_url = None
        except Exception as e:
            logger.error(f"Error uploading dataset {repo_id}: {e}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            auth = _upload_auth_error(e)
            with self._lock:
                self.state = "error"
                self.dataset_url = None
                if auth is not None:
                    self.message = auth["message"]
                    self.docs_url = auth["docs_url"]
                else:
                    self.message = f"Failed to upload dataset: {e}"
                    self.docs_url = None


upload_manager = UploadManager()


def handle_upload_dataset(request: UploadRequest) -> dict[str, Any]:
    """Start a background upload of a local dataset to the Hub.

    Returns immediately with ``{started, repo_id, message}`` — the actual push
    runs in a worker thread; poll /upload-status for progress. ``started`` is
    False when an upload is already running or the dataset is busy being
    written (recording / merge / training)."""
    return upload_manager.start(request)


def handle_upload_status() -> dict[str, Any]:
    """Current upload state (idle | running | done | error) + repo_id, message,
    and dataset_url once done."""
    return upload_manager.get_status()


def record_with_web_events(
    cfg: RecordConfig,
    web_events: dict,
    skip_identity_check: bool = False,
    motor_power: int = 100,
    identity_config_names: list[str] | None = None,
) -> LeRobotDataset:
    """
    Implement recording with phase tracking - exactly mirrors original record() function behavior

    `identity_config_names` (bimanual only) are the real library calibration stems
    — [left_follower, right_follower, left_leader, right_leader] — so the arm
    identity guard compares against the library instead of the BiSO staging alias
    ids ("<base>_left"/"<base>_right"). None for single-arm (id is the real stem).
    """
    import time

    from lerobot.common.control_utils import (
        sanity_check_dataset_robot_compatibility,
    )
    from lerobot.datasets import LeRobotDataset
    from lerobot.processor import make_default_processors
    from lerobot.robots import make_robot_from_config
    from lerobot.scripts.lerobot_record import record_loop
    from lerobot.teleoperators import make_teleoperator_from_config
    from lerobot.utils.feature_utils import hw_to_dataset_features
    from lerobot.utils.utils import log_say

    global current_phase, phase_start_time, current_episode, saved_episodes, releasing
    global identity_warnings

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    action_features = hw_to_dataset_features(robot.action_features, "action", cfg.dataset.video)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", cfg.dataset.video)
    dataset_features = {**action_features, **obs_features}

    if cfg.resume:
        num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0
        dataset = LeRobotDataset.resume(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            vcodec=cfg.dataset.vcodec,
            streaming_encoding=cfg.dataset.streaming_encoding,
            encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            encoder_threads=cfg.dataset.encoder_threads,
            image_writer_processes=cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras
            if num_cameras > 0
            else 0,
        )
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
    else:
        # lerobot's sanity_check_dataset_name requires a namespaced "user/name"
        # id and crashes on the bare names we allow for local recording without
        # an HF login. Inline its only rule that applies here (policy is None):
        # an eval_ prefix is reserved for policy-evaluation recordings.
        if cfg.dataset.repo_id.rsplit("/", 1)[-1].startswith("eval_"):
            raise ValueError(
                f"Dataset name '{cfg.dataset.repo_id}' begins with 'eval_', which is reserved "
                "for policy-evaluation datasets recorded through the rollout flow."
            )
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.dataset.fps,
            root=cfg.dataset.root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=cfg.dataset.video,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            vcodec=cfg.dataset.vcodec,
            streaming_encoding=cfg.dataset.streaming_encoding,
            encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            encoder_threads=cfg.dataset.encoder_threads,
        )

    # 🔧 ROBOT CONNECTION: Connect with enhanced error handling for camera conflicts
    try:
        logger.info("🔧 ROBOT CONNECTION: Attempting to connect robot...")
        # Calibration is already on disk (loaded via the configs above), so never
        # let connect() drop into interactive recalibration — that would hang the
        # headless record thread (the "stuck on preparing session" symptom).
        robot.connect(calibrate=False)
        logger.info("✅ ROBOT CONNECTION: Robot connected successfully")
    except Exception as e:
        logger.error(f"❌ ROBOT CONNECTION: Failed to connect robot: {e}")
        # If robot connection fails due to camera conflict, provide clear error
        if "camera" in str(e).lower() or "device" in str(e).lower() or "busy" in str(e).lower():
            logger.error("💡 ROBOT CONNECTION: Camera connection failure - likely camera resource conflict")
            logger.error(
                "💡 ROBOT CONNECTION: Make sure frontend camera streams are released before recording"
            )
        raise

    if teleop is not None:
        try:
            logger.info("🔧 TELEOP CONNECTION: Attempting to connect teleoperator...")
            # calibrate=False for the same reason as the robot connect above —
            # and critically for the identity guard below: with the default
            # calibrate=True, an EEPROM/file mismatch (exactly the swapped-port
            # case) drops into SOLeader.calibrate(), which writes the wrong
            # JSON into the servos' EEPROM (or hangs on input() in this
            # headless thread). The calibration write happens explicitly in
            # _write_calibration below, after the guard has read the EEPROM.
            teleop.connect(calibrate=False)
            logger.info("✅ TELEOP CONNECTION: Teleoperator connected successfully")
        except Exception as e:
            logger.error(f"❌ TELEOP CONNECTION: Failed to connect teleoperator: {e}")
            raise

    # Arm-identity guard: read-only check that each connected arm matches its
    # assigned calibration, BEFORE _write_calibration below can stamp a wrong
    # file into a swapped arm's EEPROM and before any action is sent. On a hard
    # mismatch, release the arms (torque was never enabled) and let the worker's
    # error path surface the message via the recording status.
    try:
        identity_warnings = verify_devices(
            ((robot, "follower"), (teleop, "leader")),
            skip=skip_identity_check,
            config_names=identity_config_names,
        )
    except ArmIdentityError:
        robot.disconnect()
        if teleop is not None:
            teleop.disconnect()
        raise

    # Ensure calibration is properly loaded and applied to the devices
    logger.info("Applying calibration to devices")

    # Write calibration to motors' memory (similar to teleoperation code). A
    # single-arm device has its own .bus/.calibration; a bimanual BiSO device
    # exposes left_arm/right_arm sub-arms instead, so write each of those.
    def _write_calibration(device, label: str) -> None:
        if device is None:
            return
        sub_arms = [
            a
            for a in (getattr(device, "left_arm", None), getattr(device, "right_arm", None))
            if a is not None
        ]
        targets = sub_arms if sub_arms else [device]
        wrote = False
        for target in targets:
            if hasattr(target, "bus") and getattr(target, "calibration", None) is not None:
                try:
                    target.bus.write_calibration(target.calibration)
                    wrote = True
                except Exception as e:
                    logger.error(f"Error writing {label} calibration: {e}")
        if wrote:
            logger.info(f"{label.capitalize()} calibration applied successfully")
        else:
            logger.warning(
                f"{label.capitalize()} bus or calibration not available - calibration may not be applied"
            )

    _write_calibration(robot, "robot")
    _write_calibration(teleop, "teleop")

    # Session motor power (RAM Torque_Limit) — the follower only, never the
    # human-held leader. robot.connect() above already ran configure(), so
    # nothing overwrites this before the recording loop; a failed write
    # degrades to full power (logged inside) and must not abort the session.
    apply_motor_power(robot, motor_power, "follower arm")
    # Clear any leftover Goal_Velocity speed cap a previous arm-driving feature
    # stamped in RAM (auto-cal fold/unfold=1000, rest-pose return=400); the
    # follower only, never the human-held leader. See lelab/motor_power.py.
    clear_goal_velocity(robot, "follower arm")

    # Capture the follower's rest pose now — after connect/configure/identity
    # guard, before the recording loop moves anything — so a normal stop can
    # drive it back to where the user left it (same as teleop; see
    # lelab/rest_pose.py). Followers only (a bimanual BiSO robot exposes two
    # follower buses), NEVER the human-held leader. The gripper is excluded: at
    # stop time it may be holding an object, and returning it to its (likely
    # open) starting width would drop the object mid-return.
    follower_rest_poses = [
        (bus, {m: v for m, v in capture_rest_pose(bus).items() if m != "gripper"})
        for bus in _device_buses(robot)
    ]

    # Start with episode 1 - but track it properly
    current_episode = 1
    saved_episodes = 0  # Track how many episodes we've actually saved

    # Set once the session ends on a user/planned path (stop button, all
    # episodes recorded) — those get the torque-release grace below. An
    # exception (camera death, unplugged bus, ...) leaves it False so the
    # release is attempted immediately: the bus may already be unreachable.
    ended_normally = False

    try:
        while saved_episodes < cfg.dataset.num_episodes:
            # RECORDING PHASE - with dataset (matches original record.py exactly)
            current_phase = "recording"
            phase_start_time = time.time()
            logger.info(f"Starting recording phase for episode {current_episode}")
            logger.info(f"Events state at start of recording phase: {web_events}")
            print(
                f"🎬 STATUS CHANGE: Starting recording phase for episode {current_episode}/{cfg.dataset.num_episodes}"
            )

            log_say(f"Recording episode {current_episode}", cfg.play_sounds)

            # Add a tracking flag that won't be reset by record_loop
            web_events["_exit_early_triggered"] = False
            logger.info(f"Recording phase - calling record_loop with events: {web_events}")

            record_loop(
                robot=robot,
                events=web_events,
                fps=cfg.dataset.fps,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                teleop=teleop,
                dataset=dataset,
                control_time_s=cfg.dataset.episode_time_s,
                single_task=cfg.dataset.single_task,
                display_data=cfg.display_data,
            )

            logger.info(f"Recording phase completed - events state: {web_events}")

            # Stop pressed mid-episode: discard the in-progress (incomplete)
            # episode and end the session immediately — no save, no reset phase,
            # no re-record. Checked BEFORE the exit-early/timeout classification
            # below, because handle_stop_recording sets exit_early (so record_loop
            # returns early) but deliberately does NOT set _exit_early_triggered.
            # Without this short-circuit the early return would be misclassified
            # as a timeout, flip rerecord_episode on, and drag the user through a
            # full reset phase before honoring the stop. Previously saved episodes
            # stay saved; the empty-dataset discard path handles the case where
            # this was episode 1 and nothing was ever saved.
            if web_events["stop_recording"]:
                logger.info(
                    "🛑 STOP RECORDING requested during recording phase - "
                    "discarding incomplete episode and ending session"
                )
                print(
                    "🛑 STATUS CHANGE: Stopped by user during recording - "
                    "incomplete episode discarded, ending session"
                )
                dataset.clear_episode_buffer()
                break

            # Check if exit_early was triggered (use our tracking flag)
            recording_interrupted_by_exit_early = web_events.get("_exit_early_triggered", False)
            if recording_interrupted_by_exit_early:
                logger.info("🟡 RECORDING PHASE INTERRUPTED BY EXIT_EARLY - proceeding to save episode")
                print(
                    f"🟡 STATUS CHANGE: Recording phase interrupted by user - episode {current_episode} data collected"
                )
                # Reset our tracking flag
                web_events["_exit_early_triggered"] = False
            else:
                # Recording completed due to timeout - trigger re-record behavior
                logger.info("⏰ RECORDING PHASE COMPLETED DUE TO TIMEOUT - triggering re-record")
                print(
                    f"⏰ STATUS CHANGE: Recording timeout reached for episode {current_episode} - re-recording"
                )
                web_events["rerecord_episode"] = True

            # Handle rerecord logic first (before saving)
            if web_events["rerecord_episode"]:
                log_say("Re-record episode", cfg.play_sounds)
                print(
                    f"🔄 STATUS CHANGE: Re-recording episode {current_episode} (episode number stays the same)"
                )
                web_events["rerecord_episode"] = False
                web_events["exit_early"] = False
                dataset.clear_episode_buffer()

                # Go through reset phase before re-recording (don't increment episode counters)
                # RESET PHASE - without dataset (matches original record.py exactly)
                current_phase = "resetting"
                phase_start_time = time.time()
                logger.info(f"Starting reset phase for re-record of episode {current_episode}")
                logger.info(f"Events state at start of reset phase: {web_events}")
                print(f"🔄 STATUS CHANGE: Starting reset phase for episode {current_episode}")

                log_say("Reset the environment", cfg.play_sounds)

                # Reset exit_early flag at the start of each phase
                web_events["exit_early"] = False
                logger.info(f"Reset phase - calling record_loop with events: {web_events}")

                record_loop(
                    robot=robot,
                    events=web_events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    # NOTE: NO dataset parameter here - matches LeRobot CLI exactly
                    # This means NO recording happens during reset phase
                    control_time_s=cfg.dataset.reset_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                )

                logger.info(f"Reset phase completed - events state: {web_events}")

                # Check if reset was interrupted by exit_early
                if web_events["exit_early"]:
                    logger.info("🟡 RESET PHASE INTERRUPTED BY EXIT_EARLY during re-record")
                    print("🟡 STATUS CHANGE: Reset phase interrupted by user during re-record")
                    web_events["exit_early"] = False

                # Check if stop recording was requested during re-record reset phase
                if web_events["stop_recording"]:
                    logger.info("🛑 STOP RECORDING requested during re-record reset phase - ending session")
                    print(
                        "🛑 STATUS CHANGE: Stop recording requested during re-record reset - ending session"
                    )
                    break

                # Don't increment current_episode or saved_episodes - we're re-recording the same episode
                continue

            # Save episode immediately after recording phase (matches expected flow)
            logger.info(f"💾 Saving episode {current_episode}...")
            print(f"💾 STATUS CHANGE: Saving episode {current_episode}")
            dataset.save_episode()
            logger.info(f"✅ Episode {current_episode} saved successfully")
            print(f"✅ STATUS CHANGE: Episode {current_episode} saved successfully")

            # Increment episode counters after successful save
            saved_episodes += 1
            current_episode += 1

            # Check if we should stop recording
            if web_events["stop_recording"]:
                print("🛑 STATUS CHANGE: Recording manually stopped by user")
                break

            # Check if we've completed all episodes
            if saved_episodes >= cfg.dataset.num_episodes:
                break

            # Execute reset phase to prepare for next episode
            # Skip reset for the last episode that was just saved
            if saved_episodes < cfg.dataset.num_episodes:
                # RESET PHASE - without dataset (matches original record.py exactly)
                current_phase = "resetting"
                phase_start_time = time.time()
                logger.info(f"Starting reset phase for next episode {current_episode}")
                logger.info(f"Events state at start of reset phase: {web_events}")
                print(f"🔄 STATUS CHANGE: Starting reset phase for episode {current_episode}")

                log_say("Reset the environment", cfg.play_sounds)

                # Reset exit_early flag at the start of each phase
                web_events["exit_early"] = False
                logger.info(f"Reset phase - calling record_loop with events: {web_events}")

                record_loop(
                    robot=robot,
                    events=web_events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    # NOTE: NO dataset parameter here - matches LeRobot CLI exactly
                    # This means NO recording happens during reset phase
                    control_time_s=cfg.dataset.reset_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                )

                logger.info(f"Reset phase completed - events state: {web_events}")

                # Check if reset was interrupted by exit_early
                if web_events["exit_early"]:
                    logger.info("🟡 RESET PHASE INTERRUPTED BY EXIT_EARLY - proceeding to next episode")
                    print("🟡 STATUS CHANGE: Reset phase interrupted by user - proceeding to next episode")
                    web_events["exit_early"] = False

                # Check if stop recording was requested during reset phase
                if web_events["stop_recording"]:
                    logger.info("🛑 STOP RECORDING requested during reset phase - ending session")
                    print("🛑 STATUS CHANGE: Stop recording requested during reset - ending session")
                    break

        # Recording completed
        current_phase = "completed"
        phase_start_time = None
        print("🏁 STATUS CHANGE: Recording session completed - all episodes finished")
        log_say("Stop recording", cfg.play_sounds, blocking=True)
        ended_normally = True

    finally:
        try:
            if ended_normally and not _release_now.is_set():
                # User-initiated stop / planned session end: no timed hold — the
                # servos hold their last goal on their own in position mode — so
                # drive the follower(s) straight back to their session-start
                # pose, then release (same behavior as the teleop / auto-cal
                # stop). A second stop (release-now) skips/aborts the return;
                # error exits skip this — the bus may be gone, release ASAP.
                releasing = True
                _return_followers_to_rest(follower_rest_poses, _release_now)
            # Belt and braces: disable torque explicitly before disconnect, so a
            # failure inside disconnect() can't leave an arm energized (rigid).
            # force_disable_torque logs any failure at ERROR level with the port.
            force_disable_torque(robot, "robot")
            force_disable_torque(teleop, "teleop")
            robot.disconnect()
            if teleop:
                teleop.disconnect()
        finally:
            releasing = False

    if cfg.dataset.push_to_hub:
        dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

    log_say("Exiting", cfg.play_sounds)
    return dataset
