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

import asyncio
import contextlib
import glob
import io
import json
import logging
import os
import queue
import re
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub.errors import HfHubHTTPError
from pydantic import BaseModel
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response
from starlette.types import Scope

from lerobot.policies.factory import make_policy_config

# Module objects (not from-imports) for the camera-preview mutex checks:
# record/teleoperate REBIND their *_active globals, so only live attribute
# access sees the current value — a from-import would freeze the startup value.
from . import datasets as dataset_browser, record as record_state, teleoperate as teleoperate_state

# Import our custom calibration functionality
from .auto_calibrate import AutoCalibrationRequest, auto_calibration_manager
from .calibrate import CalibrationRequest, calibration_manager
from .camera_preview import CameraOpenError, camera_preview_manager
from .identify import identify_arm_by_motion
from .jobs import (
    DatasetNotOnHubError,
    JobAlreadyRunningError,
    JobNotFoundError,
    JobNotRunningError,
    JobTarget,
    _list_local_checkpoints,
    job_registry,
)
from .merge import MergeRequest, handle_merge_status, handle_start_merge
from .motor_power import read_supply_voltage

# Import our custom recording functionality
from .record import (
    DatasetInfoRequest,
    RecordingRequest,
    UploadRequest,
    handle_delete_dataset,
    handle_exit_early,
    handle_recording_status,
    handle_rerecord_episode,
    handle_start_recording,
    handle_stop_recording,
    handle_upload_dataset,
    handle_upload_status,
)
from .rollout import (
    InferenceRequest,
    handle_inference_status,
    handle_start_inference,
    handle_stop_inference,
)

# Import our custom teleoperation functionality
from .teleoperate import (
    TeleoperateRequest,
    handle_get_joint_positions,
    handle_start_teleoperation,
    handle_stop_teleoperation,
    handle_teleoperation_status,
)

# Training is now job-based; see app/jobs.py.
from .train import TrainingRequest
from .update import handle_run_update, handle_update_check
from .utils import config
from .utils.config import (
    FOLLOWER_CONFIG_PATH,
    LEADER_CONFIG_PATH,
    add_dismissed_hub_job,
    clear_config_references,
    config_slot_conflict,
    delete_robot_record,
    detect_port_after_disconnect,
    find_available_ports,
    find_robot_port,
    get_default_robot_port,
    get_dismissed_hub_jobs,
    get_robot_record,
    get_saved_robot_port,
    is_robot_record_clean,
    is_valid_robot_name,
    list_robot_records,
    port_slot_conflict,
    prune_dismissed_hub_jobs,
    rename_calibration_config,
    rename_robot_record,
    save_imported_calibration,
    save_robot_port,
    save_robot_record,
)
from .utils.hf_auth import (
    cached_whoami,
    handle_hf_auth_status,
    handle_hf_login,
    hf_hub_offline,
    shared_hf_api,
)
from .utils.system import (
    handle_get_cuda_status,
    handle_get_policy_extra,
    handle_get_training_extra,
    handle_get_wandb_extra,
    handle_install_policy_extra,
    handle_install_policy_extra_status,
    handle_install_training_extra,
    handle_install_training_extra_status,
    handle_install_wandb_extra,
    handle_install_wandb_extra_status,
    warn_if_cuda_mismatch,
)
from .wiggle import wiggle_gripper

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# High-frequency read-only status polls (~2 Hz each from the frontend) that
# drown the uvicorn access log and bury real warnings (a torque warning was
# once lost in this noise). Successful GETs to these EXACT paths (query string
# ignored) are dropped from the access log; non-GETs, other paths (including
# subpaths like /jobs/{id}/logs), and error responses still log.
_QUIET_STATUS_POLL_PATHS = {
    "/auto-calibration-status",
    "/calibration-status",
    "/teleoperation-status",
    "/recording-status",
    "/joint-positions",
    "/jobs",
}


class _StatusPollAccessFilter(logging.Filter):
    """Drop uvicorn.access records for successful high-frequency status polls.

    uvicorn.access records carry args = (client_addr, method, full_path,
    http_version, status_code); anything else passes through untouched.
    Only affects the access log — app-level loggers are not filtered.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if not isinstance(args, tuple) or len(args) != 5:
            return True
        _, method, full_path, _, status_code = args
        if method != "GET" or not isinstance(status_code, int):
            return True
        # Errors (and redirects-gone-wrong) must still log.
        if status_code >= 400:
            return True
        path = str(full_path).split("?", 1)[0]
        return path not in _QUIET_STATUS_POLL_PATHS


logging.getLogger("uvicorn.access").addFilter(_StatusPollAccessFilter())


class StartTrainingBody(BaseModel):
    """Wrapping body for POST /jobs/training. Adds optional target spec."""

    config: TrainingRequest
    target: JobTarget | None = None

    @classmethod
    def from_legacy(cls, raw: dict) -> "StartTrainingBody":
        """Accept the old request shape (TrainingRequest fields at top level)
        as well as the new shape ({config: ..., target: ...}).
        """
        if "config" in raw and isinstance(raw["config"], dict):
            return cls.model_validate(raw)
        # Legacy: top-level training fields, no target.
        return cls(config=TrainingRequest.model_validate(raw))


# Cache for HF Jobs hardware flavors (5-minute TTL)
_flavors_cache: dict = {"data": None, "fetched_at": 0.0}
_FLAVOR_CACHE_TTL_SECONDS = 300.0


app = FastAPI()

# In dev mode the React app runs on :8080 while the API runs on :8000; in
# prod they share an origin and CORS is unnecessary. allow_credentials with
# a wildcard origin is rejected by browsers, so we drop it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"

# Get the path to the lerobot root directory (3 levels up from this script)
LEROBOT_PATH = str(Path(__file__).parent.parent.parent.parent)
logger.info(f"LeRobot path: {LEROBOT_PATH}")


class ConnectionManager:
    def __init__(self):
        # Each websocket is bound to the asyncio loop that accepted it; sends
        # from the broadcast worker thread must be marshaled onto that loop.
        self.active_connections: dict[WebSocket, asyncio.AbstractEventLoop] = {}
        self.broadcast_queue = queue.Queue()
        self.broadcast_thread = None
        self.is_running = False
        # Guards `active_connections` since the broadcast worker thread also
        # mutates it on send failure.
        self._connections_lock = threading.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        with self._connections_lock:
            self.active_connections[websocket] = asyncio.get_running_loop()
            count = len(self.active_connections)
        logger.info(f"WebSocket connected. Total connections: {count}")

        if not self.is_running:
            self.start_broadcast_thread()

    def disconnect(self, websocket: WebSocket):
        """Remove a connection and stop the worker if none remain.

        Only called from request-handler context (the endpoint's cleanup and
        server shutdown), never from the broadcast worker — the worker uses
        _drop_connection so it can't end up joining its own thread.
        """
        with self._connections_lock:
            if self.active_connections.pop(websocket, None) is not None:
                count = len(self.active_connections)
                logger.info(f"WebSocket disconnected. Total connections: {count}")
            else:
                count = len(self.active_connections)

        if count == 0 and self.is_running:
            self.stop_broadcast_thread()

    def _drop_connection(self, websocket: WebSocket):
        """Forget a connection whose send failed, without stopping the worker.

        The endpoint's receive loop notices the disconnect independently and
        its cleanup calls disconnect(), which is where thread stop happens.
        """
        with self._connections_lock:
            if self.active_connections.pop(websocket, None) is not None:
                count = len(self.active_connections)
                logger.info(f"Dropped unreachable WebSocket. Total connections: {count}")

    def start_broadcast_thread(self):
        """Start the background thread for broadcasting data"""
        if self.is_running:
            return

        self.is_running = True
        self.broadcast_thread = threading.Thread(target=self._broadcast_worker, daemon=True)
        self.broadcast_thread.start()
        logger.info("📡 Broadcast thread started")

    def stop_broadcast_thread(self):
        """Signal the worker thread to stop. Never joins.

        Joining here is unsafe in both directions: from the uvicorn event
        loop it can stall the loop while the worker waits on a send it
        scheduled onto that same loop, and from the worker itself it would
        be a self-join. The daemon worker notices the cleared flag (or a
        newer thread replacing it) within its 0.1 s queue timeout and exits.
        """
        self.is_running = False
        self.broadcast_thread = None
        logger.info("📡 Broadcast thread stop requested")

    def _broadcast_worker(self):
        """Background worker thread for broadcasting WebSocket data"""
        me = threading.current_thread()
        # The identity check makes a rapid stop→start cycle safe: if a new
        # worker has been started, this one exits even though is_running is
        # True again.
        while self.is_running and self.broadcast_thread is me:
            try:
                # Get data from queue with timeout
                data = self.broadcast_queue.get(timeout=0.1)
                if data is None:  # Poison pill to stop
                    break

                self._send_to_all_connections(data)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in broadcast worker: {e}")

        logger.info("📡 Broadcast thread stopped")

    def _send_to_all_connections(self, data: dict[str, Any]):
        """Send data to all active connections, each on its own event loop.

        Runs on the broadcast worker thread: every send is submitted to the
        loop that accepted the websocket via run_coroutine_threadsafe (a
        websocket's ASGI channel is not usable from any other loop). All
        sends are submitted before any is waited on so one slow client
        doesn't delay the others.
        """
        with self._connections_lock:
            connections = list(self.active_connections.items())
        if not connections:
            return

        pending = []
        for connection, loop in connections:
            try:
                future = asyncio.run_coroutine_threadsafe(connection.send_json(data), loop)
            except Exception as e:  # loop closed or shutting down
                logger.error(f"Error scheduling send to WebSocket: {e}")
                self._drop_connection(connection)
            else:
                pending.append((connection, future))

        for connection, future in pending:
            try:
                future.result(timeout=1.0)
            except Exception as e:
                logger.error(f"Error sending data to WebSocket: {e}")
                future.cancel()
                self._drop_connection(connection)

    def broadcast_joint_data_sync(self, data: dict[str, Any]):
        """Thread-safe method to queue data for broadcasting"""
        if self.is_running and self.active_connections:
            try:
                self.broadcast_queue.put_nowait(data)
            except queue.Full:
                logger.warning("Broadcast queue is full, dropping data")

    def notify_jobs_changed(self) -> None:
        """Push a 'jobs_changed' event to all WS clients so they refetch.

        Called from JobRegistry on submit / watchdog finalisation / delete.
        Skipped silently if no clients are connected — the frontend does an
        initial fetch on mount, so a missed broadcast is self-healing.
        """
        if self.is_running and self.active_connections:
            with contextlib.suppress(queue.Full):
                self.broadcast_queue.put_nowait({"type": "jobs_changed", "timestamp": time.time()})

    def notify_job_progress(self, snapshots: list[dict]) -> None:
        """Push a 'job_progress' event with per-running-job snapshots.

        Fired from the JobRegistry watchdog (~1Hz) while jobs are running so
        the dashboard's progress bar updates live without refetching /jobs
        (let alone /jobs/hub, which hits the HF API on every call).
        """
        if self.is_running and self.active_connections:
            with contextlib.suppress(queue.Full):
                self.broadcast_queue.put_nowait(
                    {"type": "job_progress", "jobs": snapshots, "timestamp": time.time()}
                )


manager = ConnectionManager()
job_registry.set_on_change(manager.notify_jobs_changed)
job_registry.set_on_progress(manager.notify_job_progress)


@app.get("/get-configs")
def get_configs():
    # Get all available calibration configs as STEMS (no .json) — the canonical
    # user-facing name. The .json is only the on-disk filename.
    leader_configs = [
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(LEADER_CONFIG_PATH, "*.json"))
    ]
    follower_configs = [
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(FOLLOWER_CONFIG_PATH, "*.json"))
    ]

    return {"leader_configs": leader_configs, "follower_configs": follower_configs}


# Frontend policy_type -> lerobot registry name. In this lerobot pin the names
# match 1:1 (pi0_fast registers as "pi0_fast", not the older "pi0fast").
# reward_classifier is NOT a policy in this pin: it registers under the
# separate RewardModelConfig registry (lerobot/rewards/), so make_policy_config
# raises for it and it reports available=False below. Keep in sync with
# POLICY_TYPE_OPTIONS in frontend/src/components/training/types.ts.
_POLICY_TYPE_TO_LEROBOT = {
    "act": "act",
    "diffusion": "diffusion",
    "pi0": "pi0",
    "smolvla": "smolvla",
    "tdmpc": "tdmpc",
    "vqbet": "vqbet",
    "pi0_fast": "pi0_fast",
    "sac": "sac",
    "reward_classifier": "reward_classifier",
}

# Optimizer preset class name -> frontend optimizer_type value.
_OPTIMIZER_CLASS_TO_NAME = {
    "adamw": "adamw",
    "adam": "adam",
    "multiadam": "multi_adam",
    "sgd": "sgd",
}


def _optimizer_name_from_preset(preset) -> str:
    """Derive the optimizer_type value from the preset config class name.

    e.g. AdamWConfig -> "adamw", MultiAdamConfig -> "multi_adam". Falls back to
    the lowercased class name (with a trailing "config" stripped) for unknown
    types so we never crash on an optimizer we haven't mapped.
    """
    name = type(preset).__name__.lower()
    if name.endswith("config"):
        name = name[: -len("config")]
    return _OPTIMIZER_CLASS_TO_NAME.get(name, name)


@app.get("/policy-optimizer-defaults")
def get_policy_optimizer_defaults():
    """Return each policy's optimizer preset (lr / weight_decay / grad_clip_norm
    + optimizer type) so the training UI can show the real "policy default"
    instead of a generic placeholder.

    Every frontend policy_type is included. `available` says whether this
    lerobot pin can construct the policy config at all — false means a training
    run with that type is doomed at policy construction, so the UI disables the
    button (e.g. reward_classifier, which isn't a policy in this pin). Policies
    whose config exists but whose optimizer preset can't be read stay available
    with a null entry in `defaults`.
    """
    defaults: dict[str, Any] = {}
    available: dict[str, bool] = {}
    for frontend_name, lerobot_name in _POLICY_TYPE_TO_LEROBOT.items():
        try:
            config = make_policy_config(lerobot_name)
        except Exception as e:
            logger.warning(
                "Policy %r (lerobot %r) is unavailable in this lerobot install: %s",
                frontend_name,
                lerobot_name,
                e,
            )
            available[frontend_name] = False
            defaults[frontend_name] = None
            continue
        available[frontend_name] = True
        try:
            preset = config.get_optimizer_preset()
            defaults[frontend_name] = {
                "optimizer": _optimizer_name_from_preset(preset),
                "lr": preset.lr,
                "weight_decay": preset.weight_decay,
                "grad_clip_norm": preset.grad_clip_norm,
            }
        except Exception as e:
            logger.warning(
                "No optimizer preset for policy %r (lerobot %r): %s",
                frontend_name,
                lerobot_name,
                e,
            )
            defaults[frontend_name] = None

    return {"defaults": defaults, "available": available}


@app.post("/move-arm")
def teleoperate_arm(request: TeleoperateRequest):
    """Start teleoperation of the robot arm"""
    return handle_start_teleoperation(request, manager)


@app.post("/stop-teleoperation")
def stop_teleoperation():
    """Stop the current teleoperation session"""
    return handle_stop_teleoperation()


@app.get("/teleoperation-status")
def teleoperation_status():
    """Get the current teleoperation status"""
    return handle_teleoperation_status()


@app.get("/joint-positions")
def get_joint_positions():
    """Get current robot joint positions"""
    return handle_get_joint_positions()


@app.post("/start-inference")
def start_inference(request: InferenceRequest):
    result = handle_start_inference(request)
    if not result.get("success"):
        raise HTTPException(
            status_code=result.get("status_code", 500),
            detail=result.get("message", "Failed to start inference"),
        )
    return result


@app.post("/stop-inference")
def stop_inference():
    result = handle_stop_inference()
    if not result.get("success"):
        raise HTTPException(
            status_code=result.get("status_code", 500),
            detail=result.get("message", "Failed to stop inference"),
        )
    return result


@app.get("/inference-status")
def inference_status():
    return handle_inference_status()


@app.get("/health")
def health_check():
    """Simple health check endpoint to verify server is running"""
    return {"status": "ok", "message": "FastAPI server is running"}


@app.get("/hf-auth-status")
def hf_auth_status():
    """Check whether the local HF CLI is authenticated and return user info."""
    return handle_hf_auth_status()


class HfLoginBody(BaseModel):
    token: str


@app.post("/hf-auth/login")
def hf_auth_login(body: HfLoginBody):
    """Persist a pasted HF token (validated against whoami) for this user."""
    try:
        return handle_hf_login(body.token)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@app.get("/datasets")
def datasets_list():
    """List datasets available to the user — Hub-owned + local cache.

    Each entry carries a `source` field: "local", "hub", or "both".
    """
    return dataset_browser.list_all_datasets()


@app.get("/datasets/info")
def datasets_info(repo_id: str):
    """Detail card for one locally-cached dataset (episodes, cameras, tasks,
    size on disk). repo_id is a query param because repo ids contain '/'."""
    info = dataset_browser.get_local_dataset_info(repo_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{repo_id}' not found in the local cache")
    return info


@app.get("/datasets/hub-status")
def datasets_hub_status(repo_id: str):
    """Whether a dataset repo with this id exists on the Hub.

    Fetched lazily by the info card (separate from /datasets/info) so it never
    blocks the card render. Degrades to status "unknown" offline/unauthenticated
    — see get_hub_status. repo_id is a query param because repo ids contain '/'.
    """
    return dataset_browser.get_hub_status(repo_id)


class DatasetRenameBody(BaseModel):
    repo_id: str
    new_name: str


@app.post("/datasets/rename")
def datasets_rename(body: DatasetRenameBody):
    """Rename a locally-cached dataset by moving its directory.

    `new_name` is the NAME PART ONLY — the namespace prefix stays fixed, so
    `ns/old` renamed to `new` becomes `ns/new`. Refuses (409) if the dataset is
    being recorded, merged, or trained on locally. Returns the new repo_id.
    """
    try:
        new_repo_id = dataset_browser.rename_local_dataset(body.repo_id, body.new_name)
    except dataset_browser.DatasetRenameError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.message) from exc
    return {"success": True, "repo_id": new_repo_id}


@app.post("/datasets/merge")
def datasets_merge(request: MergeRequest):
    """Aggregate 2+ datasets into a new local dataset in the background."""
    return handle_start_merge(request)


@app.get("/datasets/merge/status")
def datasets_merge_status():
    """Current merge state + drained log lines (idle | running | done | error)."""
    return handle_merge_status()


@app.get("/ws-test")
def websocket_test():
    """Test endpoint to verify WebSocket support"""
    return {"websocket_endpoint": "/ws/joint-data", "status": "available"}


@app.websocket("/ws/joint-data")
async def websocket_endpoint(websocket: WebSocket):
    logger.info("🔗 New WebSocket connection attempt")
    try:
        await manager.connect(websocket)
        logger.info("✅ WebSocket connection established")

        while True:
            # Keep the connection alive and wait for messages
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                # Handle any incoming messages if needed
                logger.debug(f"Received WebSocket message: {data}")
            except TimeoutError:
                # No message received, continue
                pass
            except WebSocketDisconnect:
                logger.info("🔌 WebSocket client disconnected")
                break

            # Small delay to prevent excessive CPU usage
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        logger.info("🔌 WebSocket disconnected normally")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)
        logger.info("🧹 WebSocket connection cleaned up")


@app.post("/start-recording")
def start_recording(request: RecordingRequest):
    """Start a dataset recording session"""
    return handle_start_recording(request)


@app.post("/stop-recording")
def stop_recording():
    """Stop the current recording session"""
    return handle_stop_recording()


@app.get("/recording-status")
def recording_status():
    """Get the current recording status"""
    return handle_recording_status()


@app.post("/recording-exit-early")
def recording_exit_early():
    """Skip to next episode (replaces right arrow key)"""
    return handle_exit_early()


@app.post("/recording-rerecord-episode")
def recording_rerecord_episode():
    """Re-record current episode (replaces left arrow key)"""
    return handle_rerecord_episode()


@app.post("/upload-dataset")
def upload_dataset(request: UploadRequest):
    """Start a background upload of a local dataset to the Hub.

    Returns immediately with {started, repo_id, message}; poll /upload-status
    for progress. 409 when an upload is already running (frontend maps it to a
    "an upload is already running" toast)."""
    result = handle_upload_dataset(request)
    if not result.get("started"):
        raise HTTPException(status_code=409, detail=result.get("message", "Upload could not be started"))
    return result


@app.get("/upload-status")
def upload_status():
    """Current upload state + repo_id, message, and dataset_url once done."""
    return handle_upload_status()


@app.post("/delete-dataset")
def delete_dataset(request: DatasetInfoRequest):
    """Remove a recorded dataset directory from local disk."""
    return handle_delete_dataset(request)


# ============================================================================
# JOB ENDPOINTS
# ============================================================================


@app.post("/jobs/training", status_code=201)
async def create_training_job(req: Request):
    raw = await req.json()
    body = StartTrainingBody.from_legacy(raw)
    cfg = body.config
    # Soft warning (not a block): lerobot saves/logs on `step % freq == 0`, so a
    # frequency larger than the total step count means the action never fires —
    # no checkpoint gets saved / no metrics logged. Almost always a config
    # mistake, but we still let the run proceed.
    if cfg.steps:
        if cfg.save_freq > cfg.steps:
            logger.warning(
                "save_freq (%d) exceeds steps (%d) — no checkpoint will be saved.",
                cfg.save_freq,
                cfg.steps,
            )
        if cfg.log_freq > cfg.steps:
            logger.warning(
                "log_freq (%d) exceeds steps (%d) — no metrics will be logged.",
                cfg.log_freq,
                cfg.steps,
            )
    # Hard block (not a warning): when resuming, the total step count must be
    # strictly above the checkpoint's step — lerobot requires --steps be raised
    # above the resumed checkpoint, and steps == checkpoint would train nothing.
    if cfg.resume_from_step is not None and cfg.steps <= cfg.resume_from_step:
        logger.warning(
            "Rejecting resume: steps (%d) <= checkpoint step (%d).",
            cfg.steps,
            cfg.resume_from_step,
        )
        raise HTTPException(
            status_code=400,
            detail=(
                f"Total steps ({cfg.steps}) must be greater than the checkpoint's "
                f"step ({cfg.resume_from_step}) to continue training."
            ),
        )
    try:
        record = job_registry.start(body.config, body.target)
    except JobAlreadyRunningError as exc:
        raise HTTPException(status_code=409, detail=f"Job already running: {exc}") from exc
    except DatasetNotOnHubError as exc:
        # Cloud run on a local-only dataset. 409: the caller must upload the
        # dataset first (the browser flow does this automatically before
        # submitting, so this fires for non-UI callers).
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        # e.g. "flavor is required when runner is hf_cloud"
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return record


class ImportModelRequest(BaseModel):
    source: str
    name: str | None = None


@app.post("/jobs/import", status_code=201)
def import_model(body: ImportModelRequest):
    """Register an external model (local dir or HF repo) as a pseudo-job.

    Importing an already-registered source is idempotent: the registry
    returns the EXISTING record (id and display alias preserved), and the
    response carries `already_imported: true` with a 200 (not 201) so the
    frontend can say "already imported" instead of pretending a new entry
    was created."""
    try:
        existing = job_registry.find_imported(body.source)
        record = job_registry.register_imported(body.source, body.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if existing is not None and existing.id == record.id:
        payload = record.model_dump(mode="json")
        payload["already_imported"] = True
        return JSONResponse(status_code=200, content=payload)
    return record


@app.get("/jobs")
def list_jobs(limit: int = 10):
    return {"jobs": job_registry.list(limit=limit)}


# A lelab cloud-training run repo is named "<policy>_<namespace>_<dataset>_<ts>"
# where the trailing "_YYYY-MM-DD_HH-MM-SS" is stamped by _generate_job_id()
# (jobs.py). We match on that timestamp suffix rather than the policy prefix so
# the pattern stays policy-agnostic as new policy types are added. Used to pull
# lelab's OWN empty/untagged run repos into the /jobs/hub listing without also
# surfacing a user's unrelated personal models.
_RUN_REPO_RE = re.compile(r"_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")

# Hub job stages still doing work. Mirrors HUB_ACTIVE_STAGES in the frontend
# (jobsApi.ts); a dismissed id in one of these stages is NOT hidden from the
# listing, so a live run can never be dismissed out of sight.
_HUB_ACTIVE_STAGES = {"RUNNING", "QUEUED", "SCHEDULING"}


def _hub_job_stage(ji) -> str:
    """Uppercased status stage of a huggingface_hub JobInfo ('' when absent)."""
    return (ji.status.stage or "").upper() if ji.status else ""


@app.get("/jobs/hub")
def list_hub_jobs():
    """List the user's HF Cloud compute Jobs and their uploaded LeRobot model
    repos on huggingface.co.

    Returns 200 with empty lists when no token is configured so the frontend
    can render an unauthenticated empty state without surfacing an error.

    Declared before `/jobs/{job_id}` so FastAPI's first-match routing doesn't
    treat "hub" as a job id.
    """
    info = cached_whoami()
    if info is None:
        return {"authenticated": False, "jobs": [], "models": []}
    api = shared_hf_api()

    authors: list[str] = []
    if info.get("name"):
        authors.append(info["name"])
    for o in info.get("orgs", []) or []:
        if isinstance(o, dict) and o.get("name"):
            authors.append(o["name"])

    jobs_permission = True
    jobs_listed = True
    try:
        # list_jobs() returns a lazy pagination generator — materialize it here
        # so any HTTP error (e.g. 403 when the token lacks the job.read scope)
        # is raised and caught inside this try, not later while building the
        # response, which would escape as an unhandled 500.
        jobs = list(api.list_jobs())
    except Exception as exc:
        logger.warning("list_jobs failed: %s", exc)
        jobs = []
        jobs_listed = False
        # A 401/403 means the token is valid but lacks the job.read scope —
        # surface that to the frontend so it can show a hint instead of a
        # silently-empty list. Other failures are treated as transient.
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            jobs_permission = False

    # Drop jobs the user dismissed from the UI — but only in a terminal stage:
    # an id whose job is still active stays visible, so a live run can't be
    # dismissed out of sight. Ids that have fallen out of the Hub listing are
    # pruned so the file doesn't grow forever; skipped when list_jobs() failed,
    # otherwise a transient outage would forget every dismissal.
    dismissed = get_dismissed_hub_jobs()
    if jobs_listed:
        prune_dismissed_hub_jobs({ji.id for ji in jobs})
    if dismissed:
        jobs = [ji for ji in jobs if ji.id not in dismissed or _hub_job_stage(ji) in _HUB_ACTIVE_STAGES]

    seen_models: set[str] = set()
    models: list[dict] = []

    def _add(m) -> None:
        if m.id in seen_models:
            return
        seen_models.add(m.id)
        models.append(
            {
                "repo_id": m.id,
                "last_modified": m.last_modified.isoformat() if m.last_modified else None,
                "private": bool(getattr(m, "private", False)),
            }
        )

    for author in authors:
        # Two passes, unioned + deduped by _add():
        #
        # 1. The `lerobot` library tag — lowercase, which is what LeRobot's
        #    push_to_hub actually stamps (the old `filter="LeRobot"` was both the
        #    wrong case AND excluded any repo without a tag, so it returned
        #    NOTHING here — hiding even a successfully-pushed run).
        # 2. An UNFILTERED author listing restricted to lelab run-repo names
        #    (the "_<timestamp>" suffix). This is what pulls in the empty repos
        #    a crashed cloud run pre-creates but never populates (no commit, no
        #    tags) — the orphans the untracked-cleanup path exists to delete.
        #    Restricting to the run-repo naming keeps a user's unrelated personal
        #    models out of the list; those are theirs, not lelab's to surface.
        #
        # expand=["lastModified", ...] is requested because the default listing
        # returns last_modified=None, which would collapse the sort key.
        try:
            for m in api.list_models(
                author=author, filter="lerobot", limit=200, expand=["lastModified", "private"]
            ):
                _add(m)
        except Exception as exc:
            logger.warning("list_models(%s, tag=lerobot) failed: %s", author, exc)
        try:
            for m in api.list_models(author=author, limit=200, expand=["lastModified", "private"]):
                if _RUN_REPO_RE.search(m.id.split("/", 1)[-1]):
                    _add(m)
        except Exception as exc:
            logger.warning("list_models(%s, unfiltered) failed: %s", author, exc)
    models.sort(key=lambda m: m["last_modified"] or "", reverse=True)

    return {
        "authenticated": True,
        "jobs_permission": jobs_permission,
        "jobs": [
            {
                "id": ji.id,
                "created_at": ji.created_at.isoformat() if ji.created_at else None,
                "docker_image": ji.docker_image,
                "space_id": ji.space_id,
                "flavor": ji.flavor,
                "status": ({"stage": ji.status.stage, "message": ji.status.message} if ji.status else None),
                "owner": ji.owner.name if ji.owner else None,
                "url": ji.url,
            }
            for ji in jobs
        ],
        "models": models,
    }


@app.delete("/jobs/hub/models/{repo_id:path}")
def delete_hub_model(repo_id: str):
    """Permanently delete a model repo from the Hugging Face Hub.

    Scoped to model repos under the authenticated user's own namespace — used
    to clean up orphaned repos (e.g. an empty repo left behind by a crashed
    cloud run). This destroys weights on the Hub; it is not a local-record
    deletion.

    Semantics:
    - A missing repo (404 from the Hub) is treated as already-gone success,
      mirroring the idempotent robot-delete convention.
    - Repos NOT under the caller's own username are refused up front with a
      clear message (the Hub would 403 anyway; fail fast).
    - Auth/permission failures (401/403) surface the friendly "token needs
      write access" message.

    The `/jobs/hub` listing is not cached backend-side — it re-queries the Hub
    on every call — so the frontend just needs to re-fetch after this returns.
    """
    info = cached_whoami()
    username = info.get("name") if info else None
    if not username:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated. Add a Hugging Face token with write access first.",
        )

    # Only allow deleting repos the caller owns (namespace == their username).
    # An org-owned repo (username/... mismatch) is refused rather than 403ing.
    namespace = repo_id.split("/", 1)[0] if "/" in repo_id else ""
    if namespace != username:
        raise HTTPException(
            status_code=403,
            detail=(
                f"Refusing to delete {repo_id!r}: it is not under your namespace "
                f"({username!r}). You can only delete your own model repos."
            ),
        )

    api = shared_hf_api()
    try:
        # missing_ok=True: a repo that's already gone (404) is a no-op success,
        # so re-issuing the delete is idempotent.
        api.delete_repo(repo_id, repo_type="model", missing_ok=True)
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            raise HTTPException(
                status_code=403,
                detail=(
                    "Your Hugging Face token can't delete this repo. It needs "
                    "write access to your namespace — re-log in with a write token."
                ),
            ) from exc
        logger.warning("delete_repo(%s) failed: %s", repo_id, exc)
        raise HTTPException(status_code=502, detail=f"Hub delete failed: {exc}") from exc

    return {"status": "success", "repo_id": repo_id}


@app.post("/jobs/hub/jobs/{job_id}/dismiss")
def dismiss_hub_job(job_id: str):
    """Hide a Hub job from the /jobs/hub listing.

    The HF Jobs API has no delete — a finished job stays in list_jobs()
    indefinitely — so "removing" a dead untracked job from the UI is a local,
    persisted hide (utils/config.DISMISSED_HUB_JOBS_FILE), not a Hub mutation.
    The listing keeps showing a dismissed id while its stage is still active
    (RUNNING/QUEUED/SCHEDULING); it disappears once the job reaches a terminal
    stage. Ids that later drop out of the Hub listing are pruned automatically.
    """
    if not add_dismissed_hub_job(job_id):
        raise HTTPException(status_code=400, detail="Job id can't be empty.")
    return {"status": "success", "job_id": job_id.strip()}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    try:
        return job_registry.get(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc


@app.get("/jobs/{job_id}/logs")
def get_job_logs(job_id: str):
    try:
        logs = job_registry.drain_logs(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc
    return {"logs": logs}


@app.get("/jobs/{job_id}/log-file")
def get_job_log_file(job_id: str):
    """Return the entire on-disk log file for a job. Drains the live queue too
    so the next /logs poll returns only lines that arrived after this call."""
    try:
        logs = job_registry.read_persisted_logs(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc
    # Best-effort drain so the frontend doesn't double-display.
    with contextlib.suppress(JobNotFoundError):
        job_registry.drain_logs(job_id)
    return {"logs": logs}


@app.get("/jobs/{job_id}/metrics-history")
def get_job_metrics_history(job_id: str):
    """Return the per-step loss/lr/grad-norm series reconstructed from the
    job's log.jsonl. Used to seed the monitoring charts so curves persist
    across page reloads, navigation, and lelab restarts."""
    try:
        points = job_registry.read_metrics_history(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc
    return {"points": points}


@app.get("/jobs/{job_id}/checkpoints")
def get_job_checkpoints(job_id: str):
    """List the checkpoints saved for this job, ascending by step."""
    try:
        return {"checkpoints": job_registry.list_checkpoints(job_id)}
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc


@app.get("/jobs/{job_id}/checkpoints/{step}/policy-config")
def get_checkpoint_policy_config(job_id: str, step: int):
    """Return the UX-relevant slice of a checkpoint's pretrained_model config:
    policy_type, image_features (per-camera height/width), requires_task, and
    the flat state_dim/action_dim (6 = single arm, 12 = bimanual) the inference
    modal uses to flag a single-arm/bimanual mismatch."""
    try:
        return job_registry.get_policy_config_summary(job_id, step)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/jobs/{job_id}/checkpoints/{step}/download")
def download_checkpoint(job_id: str, step: int):
    """Stream a zip of a local checkpoint's `pretrained_model/` directory.

    This bundles the portable, importable model (config.json + weights +
    pre/post-processors) — NOT the large `training_state/` optimizer dir.
    Hub-hosted models are downloadable from their HF page, so only local runs
    are supported here.
    """
    try:
        record = job_registry.get(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc

    if record.runner != "local":
        raise HTTPException(
            status_code=400,
            detail="Only local checkpoints can be downloaded; Hub models are available on their HF page.",
        )

    # The pretrained_model dir comes from _list_local_checkpoints (which resolves
    # it under record.output_dir/checkpoints/<step>), not from user input, so
    # path traversal isn't a concern. Match on the int step, never a raw path.
    checkpoint = next((c for c in _list_local_checkpoints(record.output_dir) if c.step == step), None)
    if checkpoint is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} has no checkpoint at step {step}")

    pretrained_dir = Path(checkpoint.ref)

    buffer = io.BytesIO()
    # safetensors weights are already incompressible, so DEFLATE would burn CPU
    # for ~no gain; store uncompressed.
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as zf:
        for path in sorted(pretrained_dir.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(pretrained_dir).as_posix())
    buffer.seek(0)

    # Build a filesystem-safe filename from the job's display alias (falling
    # back to its name) + step, then to the job id if sanitising leaves
    # nothing usable.
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", record.display_name or record.name).strip("_")
    if not safe_name:
        safe_name = job_id
    filename = f"{safe_name}_step_{step}.zip"

    logger.info("Downloading checkpoint for job %s at step %d", job_id, step)

    return Response(
        content=buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


class RenameJobBody(BaseModel):
    new_name: str


@app.post("/jobs/{job_id}/rename")
def rename_job(job_id: str, body: RenameJobBody):
    """Set a job's display alias (shown in place of the auto-generated name).

    Metadata-only: never moves the output directory or rewrites the run id /
    hub repo id — those are the job's immutable identity (resume lineage,
    imported-model dedup, and remote HF/W&B names key off them). Validation
    (trim, reject empty, is_valid-style character guard) lives in
    JobRegistry.rename; unlike calibration/robot renames, aliases are
    display-only and need not be unique.
    """
    try:
        return job_registry.rename(job_id, body.new_name)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/jobs/{job_id}/stop")
def stop_job(job_id: str):
    try:
        return job_registry.stop(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc
    except JobNotRunningError as exc:
        raise HTTPException(status_code=409, detail=f"Job {job_id!r} is not running") from exc


@app.delete("/jobs/{job_id}", status_code=204)
def delete_job(job_id: str):
    try:
        record = job_registry.get(job_id)
        job_registry.delete(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc
    except JobNotRunningError as exc:
        raise HTTPException(status_code=409, detail=f"Job {job_id!r} is running; stop it first") from exc
    # Deleting a tracked cloud run removes the local record, but its Hub job
    # would resurface in /jobs/hub as an untracked card on the next poll (the
    # HF Jobs API has no delete). Mark it dismissed so the removal sticks.
    if record.hf_job_id:
        add_dismissed_hub_job(record.hf_job_id)


@app.get("/jobs/runners/hardware")
def get_runners_hardware():
    """Return HF Jobs flavor catalog + auth state for the TargetCard.

    Both the flavors list and the whoami result are cached in-process to
    keep this endpoint cheap (it can be re-fetched whenever auth state
    changes). The whoami cache is invalidated on login.
    """
    # Offline mode disables every Hub write, so the cloud-training flow can't
    # upload a local-only dataset. Surface it here (same fetch TargetCard uses)
    # so the UI can keep Start disabled and explain why for those datasets.
    offline = hf_hub_offline()
    info = cached_whoami()
    if info is None or not info.get("name"):
        return {"authenticated": False, "username": None, "flavors": [], "offline": offline}
    username: str = info["name"]
    api = shared_hf_api()

    now = time.time()
    if _flavors_cache["data"] is None or now - _flavors_cache["fetched_at"] > _FLAVOR_CACHE_TTL_SECONDS:
        try:
            hw_list = api.list_jobs_hardware()
        except Exception as exc:
            logger.warning("list_jobs_hardware failed: %s", exc)
            return {"authenticated": True, "username": username, "flavors": [], "offline": offline}
        _flavors_cache["data"] = [
            {
                "name": h.name,
                "pretty_name": h.pretty_name,
                "cpu": h.cpu,
                "ram": h.ram,
                "accelerator": h.accelerator,
                "unit_cost_usd": h.unit_cost_usd,
                "unit_label": h.unit_label,
            }
            for h in hw_list
        ]
        _flavors_cache["fetched_at"] = now

    return {
        "authenticated": True,
        "username": username,
        "flavors": _flavors_cache["data"],
        "offline": offline,
    }


# ============================================================================
# SYSTEM ENDPOINTS
# ============================================================================


@app.get("/system/cuda-status")
def get_cuda_status():
    """Report whether an NVIDIA GPU is present but PyTorch is CPU-only (issue #30)."""
    return handle_get_cuda_status()


@app.get("/system/training-extra")
def get_training_extra():
    """Return whether the LeRobot training extra (accelerate) is importable."""
    return handle_get_training_extra()


@app.post("/system/training-extra/install")
def install_training_extra():
    """Spawn `pip install accelerate` as a background subprocess. No-op if already running."""
    return handle_install_training_extra()


@app.get("/system/training-extra/install-status")
def install_training_extra_status():
    """Return current install state plus any pending log lines (drained on read)."""
    return handle_install_training_extra_status()


@app.get("/system/wandb-extra")
def get_wandb_extra():
    """Return whether the `wandb` package is importable in this lelab process."""
    return handle_get_wandb_extra()


@app.post("/system/wandb-extra/install")
def install_wandb_extra():
    """Spawn `pip install wandb` as a background subprocess. No-op if already running."""
    return handle_install_wandb_extra()


@app.get("/system/wandb-extra/install-status")
def install_wandb_extra_status():
    """Return current wandb install state plus any pending log lines (drained on read)."""
    return handle_install_wandb_extra_status()


@app.get("/system/policy-extra/{policy_type}")
def get_policy_extra(policy_type: str):
    """Whether the optional LeRobot extra a policy needs (e.g. transformers for
    smolvla/pi0, diffusers for diffusion) is importable. Core policies report available."""
    return handle_get_policy_extra(policy_type)


@app.post("/system/policy-extra/{policy_type}/install")
def install_policy_extra(policy_type: str):
    """Spawn `pip install lerobot[<extra>]` for the policy's extra in the background."""
    return handle_install_policy_extra(policy_type)


@app.get("/system/policy-extra/{policy_type}/install-status")
def install_policy_extra_status(policy_type: str):
    """Return the policy extra's install state plus any pending log lines (drained on read)."""
    return handle_install_policy_extra_status(policy_type)


@app.get("/system/update-check")
def update_check():
    """Report whether a newer LeLab commit exists on GitHub (cached, silent on failure)."""
    return handle_update_check()


@app.post("/system/update")
def run_update():
    """Run the pip upgrade in-process; the user must restart lelab afterwards."""
    return handle_run_update()


# Replay is rendered by the embedded lerobot/visualize_dataset Space; no backend routes needed.


# ============================================================================
# Calibration endpoints
@app.post("/start-calibration")
def start_calibration(request: CalibrationRequest):
    """Start calibration process"""
    return calibration_manager.start_calibration(request)


@app.post("/stop-calibration")
def stop_calibration():
    """Stop calibration process"""
    return calibration_manager.stop_calibration_process()


@app.get("/calibration-status")
def calibration_status():
    """Get current calibration status"""
    from dataclasses import asdict

    status = calibration_manager.get_status()
    return asdict(status)


@app.post("/complete-calibration-step")
def complete_calibration_step():
    """Complete the current calibration step"""
    return calibration_manager.complete_step()


# --- Auto-calibration (drives the arm under torque; runs the vendored script) ---


@app.post("/start-auto-calibration")
def start_auto_calibration(request: AutoCalibrationRequest):
    """Start auto-calibration as a subprocess. The arm moves on its own."""
    return auto_calibration_manager.start(request)


@app.post("/stop-auto-calibration")
def stop_auto_calibration():
    """Stop a running auto-calibration."""
    return auto_calibration_manager.stop()


@app.get("/auto-calibration-status")
def auto_calibration_status():
    """Current auto-calibration state + streamed log lines."""
    return auto_calibration_manager.get_status()


@app.get("/calibration-configs/{device_type}")
def get_calibration_configs(device_type: str):
    """Get all calibration config files for a specific device type"""
    try:
        if device_type == "robot":
            config_path = FOLLOWER_CONFIG_PATH
        elif device_type == "teleop":
            config_path = LEADER_CONFIG_PATH
        else:
            return {"success": False, "message": "Invalid device type"}

        # Get all JSON files in the config directory
        configs = []
        if os.path.exists(config_path):
            for file in os.listdir(config_path):
                if file.endswith(".json"):
                    config_name = os.path.splitext(file)[0]
                    file_path = os.path.join(config_path, file)
                    file_size = os.path.getsize(file_path)
                    modified_time = os.path.getmtime(file_path)

                    configs.append(
                        {
                            "name": config_name,
                            "filename": file,
                            "size": file_size,
                            "modified": modified_time,
                        }
                    )

        return {"success": True, "configs": configs, "device_type": device_type}

    except Exception as e:
        logger.error(f"Error getting calibration configs: {e}")
        return {"success": False, "message": str(e)}


@app.delete("/calibration-configs/{device_type}/{config_name}")
def delete_calibration_config(device_type: str, config_name: str):
    """Delete a calibration config file"""
    try:
        if device_type == "robot":
            config_path = FOLLOWER_CONFIG_PATH
        elif device_type == "teleop":
            config_path = LEADER_CONFIG_PATH
        else:
            return {"success": False, "message": "Invalid device type"}

        # config_name is interpolated into a filename, so reject path-traversal
        # characters (/, \, ..) before touching the filesystem. Defense-in-depth:
        # FastAPI path params already block a literal "/", but not "\" or "..".
        # Reuses the same guard already applied to robot-record deletes.
        if not is_valid_robot_name(config_name):
            return {"success": False, "message": "Invalid configuration name"}

        # Construct the file path
        filename = f"{config_name}.json"
        file_path = os.path.join(config_path, filename)

        # Check if file exists
        if not os.path.exists(file_path):
            return {"success": False, "message": "Configuration file not found"}

        # Delete the file. This dir IS the location lerobot reads calibrations
        # from (setup_calibration_files' source == target), so removing the file
        # removes the only copy — nothing stale can silently keep working.
        os.remove(file_path)
        logger.info(f"Deleted calibration config: {file_path}")

        # Unassign every robot record that still pointed at this config, so
        # those arms return to the "needs calibration" state instead of
        # dangling on a missing file. The response lists them so the UI can
        # refresh the affected robots.
        unassigned = clear_config_references(device_type, config_name)
        if unassigned:
            robots = ", ".join(u["robot"] for u in unassigned)
            message = (
                f"Configuration '{config_name}' deleted. Robot(s) {robots} now need calibration before use."
            )
        else:
            message = f"Configuration '{config_name}' deleted successfully"

        return {
            "success": True,
            "message": message,
            "unassigned": unassigned,
        }

    except Exception as e:
        logger.error(f"Error deleting calibration config: {e}")
        return {"success": False, "message": str(e)}


@app.get("/calibration-configs/{device_type}/{config_name}/download")
def download_calibration_config(device_type: str, config_name: str):
    """
    Download one arm's calibration as a raw lerobot calibration JSON file.

    The file IS lerobot's own calibration file (no LeLab wrapper), so it's
    drop-in: shareable, hand-copyable, and re-importable anywhere. The arm's
    side/name are supplied by the caller on re-import, not stored in the file.
    """
    if device_type == "robot":
        config_path = FOLLOWER_CONFIG_PATH
    elif device_type == "teleop":
        config_path = LEADER_CONFIG_PATH
    else:
        return JSONResponse(status_code=400, content={"success": False, "message": "Invalid device type"})

    # config_name is interpolated into a filename, so reject path-traversal
    # characters before touching the filesystem (same guard as delete).
    if not is_valid_robot_name(config_name):
        return JSONResponse(
            status_code=400, content={"success": False, "message": "Invalid configuration name"}
        )

    # Robot records store config names WITH the .json extension while this
    # resource is otherwise stem-based; accept either form so callers that pass
    # `robot.leader_config` ("so101.json") don't resolve to "so101.json.json".
    if config_name.endswith(".json"):
        config_name = config_name[: -len(".json")]

    file_path = os.path.join(config_path, f"{config_name}.json")
    if not os.path.exists(file_path):
        return JSONResponse(
            status_code=404, content={"success": False, "message": "Configuration file not found"}
        )

    try:
        with open(file_path, "rb") as f:
            data = f.read()
    except OSError as e:
        logger.error(f"Error reading calibration config {file_path}: {e}")
        return JSONResponse(status_code=500, content={"success": False, "message": str(e)})

    return Response(
        content=data,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{config_name}.json"'},
    )


@app.post("/calibration-configs/{device_type}/upload")
def upload_calibration_config(device_type: str, body: dict):
    """
    Import a calibration into a side's config dir. Body: {"name": "...",
    "data": {<raw lerobot calibration>}}. The data is shape-validated; an
    existing name is never overwritten (409 → caller renames).
    """
    name = (body or {}).get("name", "")
    data = (body or {}).get("data")
    if not isinstance(name, str):
        return JSONResponse(status_code=400, content={"success": False, "message": "name must be a string"})

    ok, reason, saved = save_imported_calibration(device_type, name, data)
    if ok:
        return {"success": True, "name": saved}

    if reason == "invalid_device":
        return JSONResponse(status_code=400, content={"success": False, "message": "Invalid device type"})
    if reason == "invalid_name":
        return JSONResponse(
            status_code=400, content={"success": False, "message": "Invalid configuration name"}
        )
    if reason == "name_taken":
        return JSONResponse(
            status_code=409,
            content={
                "success": False,
                "message": f"A config named '{saved}' already exists. Choose a different name.",
            },
        )
    if reason.startswith("invalid_data:"):
        return JSONResponse(status_code=400, content={"success": False, "message": reason.split(":", 1)[1]})
    return JSONResponse(status_code=500, content={"success": False, "message": "Import failed"})


@app.post("/calibration-configs/{device_type}/{config_name}/rename")
def rename_calibration_config_endpoint(device_type: str, config_name: str, body: dict):
    """
    Rename a calibration config file. Body: {"new_name": "..."}. Never
    overwrites; robot records referencing the old name are repointed.
    """
    new_name = (body or {}).get("new_name", "")
    if not isinstance(new_name, str):
        return JSONResponse(
            status_code=400, content={"success": False, "message": "new_name must be a string"}
        )

    ok, reason = rename_calibration_config(device_type, config_name, new_name)
    if ok:
        return {"success": True, "name": new_name.strip().removesuffix(".json")}

    status_code, message = {
        "invalid_device": (400, "Invalid device type"),
        "invalid_name": (400, "Invalid configuration name"),
        "not_found": (404, "Configuration file not found"),
        "name_taken": (409, "A config with that name already exists. Choose a different name."),
    }.get(reason, (500, "Rename failed"))
    return JSONResponse(status_code=status_code, content={"success": False, "message": message})


# ============================================================================
# PORT DETECTION ENDPOINTS
# ============================================================================


@app.get("/available-ports")
def get_available_ports():
    """Get all available serial ports"""
    try:
        ports = find_available_ports()
        return {"status": "success", "ports": ports}
    except Exception as e:
        logger.error(f"Error getting available ports: {e}")
        return {"status": "error", "message": str(e)}


class WiggleRequest(BaseModel):
    port: str


@app.post("/wiggle")
async def wiggle(request: WiggleRequest):
    """Wiggle the gripper on a port so the user can see which arm it is."""
    return await wiggle_gripper(request.port)


class IdentifyArmRequest(BaseModel):
    # Candidate ports to watch; empty/omitted = all detected arm ports.
    ports: list[str] | None = None


@app.post("/identify-arm")
async def identify_arm(request: IdentifyArmRequest):
    """The inverse of /wiggle: the user swings an arm's base (shoulder pan) by
    hand and we report which port saw the motion. Read-only — no motor writes."""
    return await identify_arm_by_motion(request.ports)


@app.get("/supply-voltage")
async def supply_voltage(port: str = ""):
    """One-shot, read-only supply-voltage reading (Present_Voltage) from the arm
    on `port`. Connects, reads, and releases the port immediately — never holds
    it — so calibration/teleoperation can grab the port right after."""
    return await read_supply_voltage(port)


# Runs in a fresh Python — see _avfoundation_cameras_in_cv2_order for why.
# Mirrors OpenCV's macOS enumeration: video + muxed devices sorted by
# uniqueID (cap_avfoundation_mac.mm), so the returned index matches what
# cv2.VideoCapture will open.
_AVF_ENUM_SCRIPT = """
import json, objc
from Foundation import NSBundle
bundle = NSBundle.bundleWithPath_("/System/Library/Frameworks/AVFoundation.framework")
bundle.load()
types = []
for name in (
    "AVCaptureDeviceTypeBuiltInWideAngleCamera",
    "AVCaptureDeviceTypeExternalUnknown",   # macOS < 14
    "AVCaptureDeviceTypeExternal",          # macOS >= 14
    "AVCaptureDeviceTypeContinuityCamera",  # macOS >= 14
    # AVCaptureDeviceTypeDeskViewCamera (Continuity Desk View, macOS >= 13) is
    # deliberately omitted: AVFoundation enumerates it, but OpenCV's index-based
    # cv2.VideoCapture cannot open it ("out device of bound"), so listing it
    # produced permanent /camera-preview 503s and a retry-looping frontend tile.
    # Excluding it here also keeps the reported index aligned with cv2's own
    # ordering, which doesn't count Desk View.
):
    loaded = {}
    try:
        objc.loadBundleVariables(bundle, loaded, [(name, b"@")])
    except objc.error:
        continue
    if loaded.get(name) is not None:
        types.append(loaded[name])
cls = objc.lookUpClass("AVCaptureDeviceDiscoverySession")
devs = []
for mt in ("vide", "muxx"):
    devs.extend(cls.discoverySessionWithDeviceTypes_mediaType_position_(types, mt, 0).devices() or [])
devs.sort(key=lambda d: d.uniqueID())
print(json.dumps([
    {"index": i, "name": str(d.localizedName()), "unique_id": str(d.uniqueID())}
    for i, d in enumerate(devs)
]))
"""


def _avfoundation_cameras_in_cv2_order() -> list[dict[str, Any]]:
    """Enumerate macOS cameras in a fresh Python subprocess.

    AVFoundation's in-process device cache doesn't refresh on USB
    hotplug. Both the deprecated ``+devicesWithMediaType:`` and a
    long-lived ``AVCaptureDeviceDiscoverySession`` go stale, because
    device-connection notifications are delivered via
    ``NSNotificationCenter`` on a thread that needs an active
    ``NSRunLoop`` — uvicorn workers don't run one. A fresh subprocess
    re-initializes AVFoundation, which reads IOKit's live device state
    at startup.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", _AVF_ENUM_SCRIPT],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning("AVFoundation enumeration subprocess failed: %s", e)
        return []
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        logger.warning("AVFoundation enumeration returned invalid JSON: %s", e)
        return []


def _generic_cv2_cameras(backend) -> list[dict[str, Any]]:
    """Last-resort enumeration: probe cv2 indices with placeholder names."""
    import cv2

    cameras: list[dict[str, Any]] = []
    for i in range(10):
        cap = cv2.VideoCapture(i, backend)
        opened = cap.isOpened()
        cap.release()
        if opened:
            cameras.append({"index": i, "name": f"Camera {i}", "available": True})
    return cameras


def _windows_cameras() -> list[dict[str, Any]]:
    """Enumerate Windows cameras with their real DirectShow names.

    pygrabber lists DirectShow video devices in the same order cv2's DSHOW
    backend indexes them (which recording is pinned to), so the returned index
    matches what ``cv2.VideoCapture(i, CAP_DSHOW)`` opens. The real names let the
    frontend match each index to the browser's ``MediaDeviceInfo.label`` for the
    live preview. Falls back to generic names if pygrabber is unavailable.
    """
    try:
        from pygrabber.dshow_graph import FilterGraph

        names = FilterGraph().get_input_devices()
    except Exception as e:  # ImportError, or a COM/DirectShow failure
        logger.warning("pygrabber unavailable; using generic camera names: %s", e)
        import cv2

        return _generic_cv2_cameras(cv2.CAP_DSHOW)
    return [{"index": i, "name": name, "available": True} for i, name in enumerate(names)]


def _v4l2_camera_name(index: int) -> str | None:
    """Real camera name for /dev/video{index} from sysfs (Linux, no deps)."""
    try:
        with open(f"/sys/class/video4linux/video{index}/name", encoding="utf-8") as f:
            return f.read().strip() or None
    except OSError:
        return None


def _linux_cameras() -> list[dict[str, Any]]:
    """Enumerate Linux cameras, naming each from sysfs (no extra deps)."""
    import cv2

    cameras: list[dict[str, Any]] = []
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        opened = cap.isOpened()
        cap.release()
        if not opened:
            continue
        cameras.append({"index": i, "name": _v4l2_camera_name(i) or f"Camera {i}", "available": True})
    return cameras


@app.get("/available-cameras")
def get_available_cameras():
    """List cameras with the same index ordering cv2 will use to record.

    Each platform enumerates in the order its cv2 backend indexes devices, and
    pairs each index with the device's real name so the frontend can match it to
    the browser's ``MediaDeviceInfo.label`` for the live preview:
      - macOS: AVFoundation ``localizedName`` (via a PyObjC subprocess);
      - Windows: DirectShow FriendlyName (via pygrabber; recording pinned DSHOW);
      - Linux: the v4l2 device name from sysfs.
    Without real names the frontend can't match a camera and shows "No browser
    match" with an empty device_id (issues #12, #16).
    """
    try:
        import platform

        system = platform.system()

        if system == "Darwin":
            cameras = _avfoundation_cameras_in_cv2_order()
            for cam in cameras:
                cam["available"] = True
            return {"status": "success", "cameras": cameras}
        if system == "Windows":
            return {"status": "success", "cameras": _windows_cameras()}
        if system == "Linux":
            return {"status": "success", "cameras": _linux_cameras()}

        import cv2

        return {"status": "success", "cameras": _generic_cv2_cameras(cv2.CAP_ANY)}
    except ImportError:
        logger.warning("OpenCV not available for camera detection")
        return {"status": "success", "cameras": []}
    except Exception as e:
        logger.error(f"Error detecting cameras: {e}")
        return {"status": "error", "message": str(e), "cameras": []}


@app.get("/camera-preview/{index}")
def camera_preview_stream(index: int):
    """MJPEG preview stream of a camera attached to the *server* machine.

    Fallback for headless deployments (e.g. a Jetson on the LAN): the browser's
    getUserMedia can't see the server's cameras, so the preview tiles render
    ``<img src="/camera-preview/{index}">`` instead. The capture is shared and
    refcounted per index (see lelab/camera_preview.py); recording and
    teleoperation always win — their start paths force-release every preview.

    Returns 409 while recording or teleoperation is active (they own the cv2
    devices) and 503 when the camera can't be opened.
    """
    if record_state.recording_active:
        raise HTTPException(
            status_code=409,
            detail="Recording is active — the cameras are in use. Stop recording to preview them.",
        )
    if teleoperate_state.teleoperation_active:
        raise HTTPException(
            status_code=409,
            detail="Teleoperation is active — stop it to preview the cameras.",
        )
    try:
        stream = camera_preview_manager.open_stream(index)
    except CameraOpenError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return StreamingResponse(stream, media_type="multipart/x-mixed-replace; boundary=frame")


RobotSideLiteral = Literal["leader", "follower"]


class PortDetectionBody(BaseModel):
    robot_type: RobotSideLiteral = "follower"


class PortDisconnectBody(BaseModel):
    ports_before: list[str]


class SaveRobotPortBody(BaseModel):
    robot_type: RobotSideLiteral
    port: str


class SaveRobotConfigBody(BaseModel):
    robot_type: RobotSideLiteral
    config_name: str


@app.post("/start-port-detection")
def start_port_detection(body: PortDetectionBody):
    """Snapshot available ports so the follow-up /detect-port-after-disconnect
    call can diff them."""
    result = find_robot_port(body.robot_type)
    return {"status": "success", "data": result}


@app.post("/detect-port-after-disconnect")
def detect_port_after_disconnect_endpoint(body: PortDisconnectBody):
    """Block up to 15s waiting for one port from `ports_before` to disappear."""
    try:
        detected_port = detect_port_after_disconnect(body.ports_before)
    except OSError as exc:
        raise HTTPException(status_code=408, detail=str(exc)) from exc
    return {"status": "success", "port": detected_port}


@app.post("/save-robot-port")
def save_robot_port_endpoint(body: SaveRobotPortBody):
    """Save a robot port for future use"""
    save_robot_port(body.robot_type, body.port)
    return {"status": "success", "message": f"Port {body.port} saved for {body.robot_type}"}


@app.get("/robot-port/{robot_type}")
def get_robot_port(robot_type: RobotSideLiteral):
    """Get the saved port for a robot type"""
    saved_port = get_saved_robot_port(robot_type)
    default_port = get_default_robot_port(robot_type)
    return {"status": "success", "saved_port": saved_port, "default_port": default_port}


@app.post("/save-robot-config")
def save_robot_config_endpoint(body: SaveRobotConfigBody):
    """Save a robot configuration for future use"""
    if not config.save_robot_config(body.robot_type, body.config_name):
        raise HTTPException(status_code=500, detail="Failed to save configuration")
    return {"status": "success", "message": f"Configuration saved for {body.robot_type}"}


@app.get("/robot-config/{robot_type}")
def get_robot_config(robot_type: RobotSideLiteral, available_configs: str = ""):
    """Get the saved configuration for a robot type"""
    available_configs_list = [c.strip() for c in available_configs.split(",") if c.strip()]
    saved_config = config.get_saved_robot_config(robot_type)
    default_config = config.get_default_robot_config(robot_type, available_configs_list)
    return {"status": "success", "saved_config": saved_config, "default_config": default_config}


# ============================================================================
# Robot config records (named robots)


def _record_with_clean(record: dict) -> dict:
    """Attach `is_clean` to a record for API responses."""
    return {**record, "is_clean": is_robot_record_clean(record)}


@app.get("/robots")
def get_robots():
    """List all saved robot records."""
    try:
        records = [_record_with_clean(r) for r in list_robot_records()]
        return {"status": "success", "robots": records}
    except Exception as e:
        logger.error(f"Error listing robots: {e}")
        return {"status": "error", "message": str(e), "robots": []}


@app.get("/robots/{name}")
def get_robot(name: str):
    """Get a single robot record by name."""
    if not is_valid_robot_name(name):
        return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid robot name"})
    record = get_robot_record(name)
    if record is None:
        return JSONResponse(status_code=404, content={"status": "error", "message": "Robot not found"})
    return {"status": "success", "robot": _record_with_clean(record)}


@app.post("/robots/{name}")
def upsert_robot(name: str, data: dict, create: bool = False):
    """
    Upsert a robot record.

    - `?create=true` is the "Add Robot" path: returns 409 if a record with that
      name already exists; otherwise creates with empty fields then merges body.
    - Without `?create=true` is the "patch" path (e.g., calibration write-back):
      merges body into existing record. If no record exists, no-ops and returns
      success — see deletion-during-calibration edge case in the spec.
    """
    if not is_valid_robot_name(name):
        return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid robot name"})

    body = data or {}
    existing = get_robot_record(name) or {}

    # Mode is fixed at creation. A bimanual rig is a different machine (different
    # robot_type on datasets, forced _left/_right calibration naming, different
    # arms/cameras), and allowing a live toggle was a recurring stale-state bug
    # source. On the patch path (no ?create=true) reject any body `mode` that
    # differs from the stored value; a same-value echo stays a no-op. On create
    # the mode in the body is what establishes it.
    if (
        not create
        and existing
        and body.get("mode") in ("single", "bimanual")
        and body["mode"] != existing.get("mode", "single")
    ):
        return JSONResponse(
            status_code=409,
            content={
                "status": "error",
                "message": "Mode is fixed at creation — create a new robot for a bimanual (or single-arm) setup.",
            },
        )

    # Effective mode for the slot/port conflict checks below. Because mode can't
    # change on an existing record, this is the stored mode for patches and the
    # body mode for creates (defaulting to single).
    effective_mode = (
        body["mode"]
        if create and body.get("mode") in ("single", "bimanual")
        else existing.get("mode", "single")
    )

    # Reject assigning the same calibration to both same-side arms of a bimanual
    # robot — that would point two physical arms at one calibration. Only checked
    # when the request actually touches a config slot, so unrelated edits
    # (cameras, ports) aren't blocked even on a pre-existing conflict.
    config_fields = ("leader_config", "follower_config", "right_leader_config", "right_follower_config")
    if any(f in body for f in config_fields):
        prospective = {"mode": effective_mode}
        for f in config_fields:
            prospective[f] = body[f] if isinstance(body.get(f), str) else existing.get(f, "")
        side = config_slot_conflict(prospective)
        if side:
            return JSONResponse(
                status_code=409,
                content={
                    "status": "error",
                    "message": f"That {side} config is already assigned to the other {side} arm. "
                    "Each physical arm needs its own calibration — pick a different config.",
                },
            )

    # Reject assigning one serial port to more than one arm — each physical arm
    # is its own USB device. Checked when the request touches a port.
    port_field_names = ("leader_port", "follower_port", "right_leader_port", "right_follower_port")
    if any(f in body for f in port_field_names):
        prospective = {"mode": effective_mode}
        for f in port_field_names:
            prospective[f] = body[f] if isinstance(body.get(f), str) else existing.get(f, "")
        dup_port = port_slot_conflict(prospective)
        if dup_port:
            return JSONResponse(
                status_code=409,
                content={
                    "status": "error",
                    "message": f"Port {dup_port} is already assigned to another arm of this robot. "
                    "Each arm needs its own serial port.",
                },
            )

    try:
        if create:
            if get_robot_record(name) is not None:
                return JSONResponse(
                    status_code=409,
                    content={"status": "error", "message": "A robot with this name already exists"},
                )
            save_robot_record(name, data or {}, allow_create=True)
        else:
            save_robot_record(name, data or {}, allow_create=False)
        record = get_robot_record(name)
        if record is None:
            return {"status": "success", "robot": None}
        return {"status": "success", "robot": _record_with_clean(record)}
    except Exception as e:
        logger.error(f"Error upserting robot {name}: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/robots/{name}/rename")
def rename_robot(name: str, data: dict):
    """
    Rename a robot record. Body: {"new_name": "..."}. Calibration files are not
    affected (they're keyed by config name, not robot name).
    """
    new_name = (data or {}).get("new_name", "")
    if not isinstance(new_name, str):
        return JSONResponse(
            status_code=400, content={"status": "error", "message": "new_name must be a string"}
        )
    new_name = new_name.strip()

    ok, reason = rename_robot_record(name, new_name)
    if ok:
        record = get_robot_record(new_name)
        return {"status": "success", "robot": _record_with_clean(record) if record else None}

    status_code, message = {
        "invalid_name": (400, "Invalid robot name"),
        "not_found": (404, "Robot not found"),
        "name_taken": (409, "A robot with that name already exists"),
    }.get(reason, (500, "Rename failed"))
    return JSONResponse(status_code=status_code, content={"status": "error", "message": message})


@app.delete("/robots/{name}")
def delete_robot(name: str):
    """Delete a robot record."""
    if not is_valid_robot_name(name):
        return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid robot name"})
    if delete_robot_record(name):
        return {"status": "success"}
    return JSONResponse(status_code=404, content={"status": "error", "message": "Robot not found"})


@app.on_event("startup")
def startup_event():
    """One-time startup diagnostics surfaced in the server terminal."""
    warn_if_cuda_mismatch()


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources when FastAPI shuts down"""
    logger.info("🔄 FastAPI shutting down, cleaning up...")

    # Stop any active recording - handled by recording module cleanup

    if manager:
        manager.stop_broadcast_thread()
    logger.info("✅ Cleanup completed")


def _accepts_html(accept: str) -> bool:
    """Whether an Accept header explicitly wants text/html (quality > 0).

    Browser navigations list `text/html` with a positive quality value, so
    they get the SPA shell. A `text/html;q=0` entry is an explicit refusal and
    must not count — a plain substring check would wrongly treat it as a yes.
    `*/*` (curl, XHR, API clients) is deliberately not treated as wanting HTML.
    """
    for part in accept.split(","):
        media_type, _, params = part.strip().partition(";")
        if media_type.strip().lower() != "text/html":
            continue
        quality = 1.0
        for param in params.split(";"):
            key, _, value = param.partition("=")
            if key.strip().lower() == "q":
                try:
                    quality = float(value)
                except ValueError:
                    quality = 0.0
        return quality > 0
    return False


class SPAStaticFiles(StaticFiles):
    """StaticFiles that serves index.html for unknown client-side routes.

    The frontend is a single-page app: routes like /recording and /calibration
    exist only in the browser's router, not as files on disk. A hard reload or
    deep link to one of those URLs asks the server for a file that isn't there;
    plain StaticFiles answers 404 ({"detail":"Not Found"}), so the page breaks.

    Here we fall back to index.html on 404 so the SPA boots and its router
    renders the route. Only requests that accept HTML (i.e. browser navigations)
    get the fallback — API typos, XHR, and curl still receive a JSON 404.
    """

    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404 and _accepts_html(Headers(scope=scope).get("accept", "")):
                return await super().get_response("index.html", scope)
            raise


# Serve the built frontend at /. Must be mounted last so API routes win.
if FRONTEND_DIST.exists():
    app.mount("/", SPAStaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
else:
    logger.warning(
        f"frontend/dist not found at {FRONTEND_DIST}; run `npm run build` in frontend/ or use `makerlab --dev`."
    )
