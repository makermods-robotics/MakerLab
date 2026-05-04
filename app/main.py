from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
import glob
import asyncio
from typing import List, Dict, Any
import threading
import queue
from pathlib import Path
from . import config

# Import our custom recording functionality
from .recording import (
    RecordingRequest,
    UploadRequest,
    DatasetInfoRequest,
    handle_start_recording,
    handle_stop_recording,
    handle_exit_early,
    handle_rerecord_episode,
    handle_recording_status,
    handle_upload_dataset,
    handle_get_dataset_info,
)

# Import our custom teleoperation functionality
from .teleoperating import (
    TeleoperateRequest,
    handle_start_teleoperation,
    handle_stop_teleoperation,
    handle_teleoperation_status,
    handle_get_joint_positions,
)

# Import our custom calibration functionality
from .calibrating import CalibrationRequest, calibration_manager

# Training is now job-based; see app/jobs.py.
from .training import TrainingRequest
from .jobs import (
    job_registry,
    JobAlreadyRunningError,
    JobNotFoundError,
    JobNotRunningError,
)

from .system import (
    handle_get_training_extra,
    handle_install_training_extra,
    handle_install_training_extra_status,
)

from .hf_auth import handle_hf_auth_status
from . import dataset_browser


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for WebSocket connections
connected_websockets: List[WebSocket] = []


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"

# Get the path to the lerobot root directory (3 levels up from this script)
LEROBOT_PATH = str(Path(__file__).parent.parent.parent.parent)
logger.info(f"LeRobot path: {LEROBOT_PATH}")

# Import shared configuration constants
from .config import (
    CALIBRATION_BASE_PATH_TELEOP,
    CALIBRATION_BASE_PATH_ROBOTS,
    LEADER_CONFIG_PATH,
    FOLLOWER_CONFIG_PATH,
    find_available_ports,
    find_robot_port,
    detect_port_after_disconnect,
    save_robot_port,
    get_saved_robot_port,
    get_default_robot_port,
    save_robot_record,
    get_robot_record,
    list_robot_records,
    delete_robot_record,
    is_robot_record_clean,
    is_valid_robot_name,
)


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.broadcast_queue = queue.Queue()
        self.broadcast_thread = None
        self.is_running = False

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(
            f"WebSocket connected. Total connections: {len(self.active_connections)}"
        )

        # Start broadcast thread if not running
        if not self.is_running:
            self.start_broadcast_thread()

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(
                f"WebSocket disconnected. Total connections: {len(self.active_connections)}"
            )

        # Stop broadcast thread if no connections
        if not self.active_connections and self.is_running:
            self.stop_broadcast_thread()

    def start_broadcast_thread(self):
        """Start the background thread for broadcasting data"""
        if self.is_running:
            return

        self.is_running = True
        self.broadcast_thread = threading.Thread(
            target=self._broadcast_worker, daemon=True
        )
        self.broadcast_thread.start()
        logger.info("📡 Broadcast thread started")

    def stop_broadcast_thread(self):
        """Stop the background thread"""
        self.is_running = False
        if self.broadcast_thread:
            self.broadcast_thread.join(timeout=1.0)
            logger.info("📡 Broadcast thread stopped")

    def _broadcast_worker(self):
        """Background worker thread for broadcasting WebSocket data"""
        import asyncio

        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            while self.is_running:
                try:
                    # Get data from queue with timeout
                    data = self.broadcast_queue.get(timeout=0.1)
                    if data is None:  # Poison pill to stop
                        break

                    # Broadcast to all connections
                    if self.active_connections:
                        loop.run_until_complete(self._send_to_all_connections(data))

                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error in broadcast worker: {e}")

        finally:
            loop.close()

    async def _send_to_all_connections(self, data: Dict[str, Any]):
        """Send data to all active WebSocket connections"""
        if not self.active_connections:
            return

        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except Exception as e:
                logger.error(f"Error sending data to WebSocket: {e}")
                disconnected.append(connection)

        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)

    def broadcast_joint_data_sync(self, data: Dict[str, Any]):
        """Thread-safe method to queue data for broadcasting"""
        if self.is_running and self.active_connections:
            try:
                self.broadcast_queue.put_nowait(data)
            except queue.Full:
                logger.warning("Broadcast queue is full, dropping data")


manager = ConnectionManager()


@app.get("/get-configs")
def get_configs():
    # Get all available calibration configs
    leader_configs = [
        os.path.basename(f)
        for f in glob.glob(os.path.join(LEADER_CONFIG_PATH, "*.json"))
    ]
    follower_configs = [
        os.path.basename(f)
        for f in glob.glob(os.path.join(FOLLOWER_CONFIG_PATH, "*.json"))
    ]

    return {"leader_configs": leader_configs, "follower_configs": follower_configs}


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


@app.get("/health")
def health_check():
    """Simple health check endpoint to verify server is running"""
    return {"status": "ok", "message": "FastAPI server is running"}


@app.get("/hf-auth-status")
def hf_auth_status():
    """Check whether the local HF CLI is authenticated and return user info."""
    return handle_hf_auth_status()


@app.get("/datasets")
def datasets_list():
    """List datasets the logged-in HF user owns or shares with their orgs."""
    return dataset_browser.list_user_datasets()


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
            except asyncio.TimeoutError:
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
    return handle_start_recording(request, manager)


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
    """Upload dataset to HuggingFace Hub"""
    return handle_upload_dataset(request)


@app.post("/dataset-info")
def get_dataset_info(request: DatasetInfoRequest):
    """Get information about a saved dataset"""
    return handle_get_dataset_info(request)


# ============================================================================
# JOB ENDPOINTS
# ============================================================================


@app.post("/jobs/training", status_code=201)
def create_training_job(request: TrainingRequest):
    try:
        record = job_registry.start(request)
    except JobAlreadyRunningError as exc:
        raise HTTPException(status_code=409, detail=f"A training job is already running: {exc}")
    return record


@app.get("/jobs")
def list_jobs(limit: int = 10):
    return {"jobs": job_registry.list(limit=limit)}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    try:
        return job_registry.get(job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")


@app.get("/jobs/{job_id}/logs")
def get_job_logs(job_id: str):
    try:
        logs = job_registry.drain_logs(job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return {"logs": logs}


@app.post("/jobs/{job_id}/stop")
def stop_job(job_id: str):
    try:
        return job_registry.stop(job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    except JobNotRunningError:
        raise HTTPException(status_code=409, detail=f"Job {job_id!r} is not running")


@app.delete("/jobs/{job_id}", status_code=204)
def delete_job(job_id: str):
    try:
        job_registry.delete(job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    except JobNotRunningError:
        raise HTTPException(status_code=409, detail=f"Job {job_id!r} is running; stop it first")


# ============================================================================
# SYSTEM ENDPOINTS
# ============================================================================


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

        # Construct the file path
        filename = f"{config_name}.json"
        file_path = os.path.join(config_path, filename)

        # Check if file exists
        if not os.path.exists(file_path):
            return {"success": False, "message": "Configuration file not found"}

        # Delete the file
        os.remove(file_path)
        logger.info(f"Deleted calibration config: {file_path}")

        return {
            "success": True,
            "message": f"Configuration '{config_name}' deleted successfully",
        }

    except Exception as e:
        logger.error(f"Error deleting calibration config: {e}")
        return {"success": False, "message": str(e)}


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


def _list_avfoundation_cameras() -> Dict[int, str]:
    """Return {index: name} for AVFoundation video devices on macOS via ffmpeg.

    The order matches OpenCV's AVFoundation backend, so the names returned
    here can be used to label the cv2.VideoCapture indices reliably.
    """
    import subprocess
    import re
    try:
        result = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning(f"ffmpeg AVFoundation listing unavailable: {e}")
        return {}

    names: Dict[int, str] = {}
    in_video_section = False
    for line in result.stderr.splitlines():
        if "AVFoundation video devices" in line:
            in_video_section = True
            continue
        if "AVFoundation audio devices" in line:
            break
        if not in_video_section:
            continue
        m = re.search(r"\[(\d+)\]\s+(.+?)\s*$", line)
        if m:
            names[int(m.group(1))] = m.group(2).strip()
    return names


@app.get("/available-cameras")
def get_available_cameras():
    """Get all available cameras with a JPEG thumbnail per OpenCV index.

    The thumbnail is the source of truth for camera identity: the frontend
    shows it so the user picks an index by what the *backend* actually sees,
    avoiding the mismatch between browser enumeration order and OpenCV's
    AVFoundation/V4L2 indexing.
    """
    try:
        import cv2
        import base64
        import platform
        cameras = []

        # Pin the backend so the indices we hand back match what the recording
        # session will see. cv2.CAP_ANY can otherwise pick a different backend
        # on subsequent calls (notably macOS), silently reordering cameras.
        system = platform.system()
        if system == "Darwin":
            backend = cv2.CAP_AVFOUNDATION
        elif system == "Linux":
            backend = cv2.CAP_V4L2
        else:
            backend = cv2.CAP_ANY

        avf_names = _list_avfoundation_cameras() if system == "Darwin" else {}

        for i in range(10):
            cap = cv2.VideoCapture(i, backend)
            if not cap.isOpened():
                cap.release()
                continue

            # First read often returns a blank/garbage frame on warmup; try a few.
            frame = None
            for _ in range(5):
                ret, candidate = cap.read()
                if ret and candidate is not None:
                    frame = candidate
                    break

            if frame is None:
                cap.release()
                continue

            entry = {
                "index": i,
                "name": avf_names.get(i, f"Camera {i}"),
                "available": True,
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": int(cap.get(cv2.CAP_PROP_FPS)),
            }

            try:
                small = cv2.resize(frame, (160, 120))
                ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    encoded = base64.b64encode(buf.tobytes()).decode("ascii")
                    entry["thumbnail"] = f"data:image/jpeg;base64,{encoded}"
            except Exception as thumb_err:
                logger.warning(f"Thumbnail capture failed for camera {i}: {thumb_err}")

            cameras.append(entry)
            cap.release()

        return {"status": "success", "cameras": cameras}
    except ImportError:
        # OpenCV not available, return empty list
        logger.warning("OpenCV not available for camera detection")
        return {"status": "success", "cameras": []}
    except Exception as e:
        logger.error(f"Error detecting cameras: {e}")
        return {"status": "error", "message": str(e), "cameras": []}


@app.post("/start-port-detection")
def start_port_detection(data: dict):
    """Start port detection process for a robot"""
    try:
        robot_type = data.get("robot_type", "robot")
        result = find_robot_port(robot_type)
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Error starting port detection: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/detect-port-after-disconnect")
def detect_port_after_disconnect_endpoint(data: dict):
    """Detect port after disconnection"""
    try:
        ports_before = data.get("ports_before", [])
        detected_port = detect_port_after_disconnect(ports_before)
        return {"status": "success", "port": detected_port}
    except Exception as e:
        logger.error(f"Error detecting port: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/save-robot-port")
def save_robot_port_endpoint(data: dict):
    """Save a robot port for future use"""
    try:
        robot_type = data.get("robot_type")
        port = data.get("port")
        
        if not robot_type or not port:
            return {"status": "error", "message": "robot_type and port are required"}
        
        save_robot_port(robot_type, port)
        return {"status": "success", "message": f"Port {port} saved for {robot_type}"}
    except Exception as e:
        logger.error(f"Error saving robot port: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/robot-port/{robot_type}")
def get_robot_port(robot_type: str):
    """Get the saved port for a robot type"""
    try:
        saved_port = get_saved_robot_port(robot_type)
        default_port = get_default_robot_port(robot_type)
        return {
            "status": "success", 
            "saved_port": saved_port,
            "default_port": default_port
        }
    except Exception as e:
        logger.error(f"Error getting robot port: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/save-robot-config")
def save_robot_config_endpoint(data: dict):
    """Save a robot configuration for future use"""
    try:
        robot_type = data.get("robot_type")
        config_name = data.get("config_name")
        
        if not robot_type or not config_name:
            return {"status": "error", "message": "Missing robot_type or config_name"}
            
        success = config.save_robot_config(robot_type, config_name)
        
        if success:
            return {"status": "success", "message": f"Configuration saved for {robot_type}"}
        else:
            return {"status": "error", "message": "Failed to save configuration"}
            
    except Exception as e:
        logger.error(f"Error saving robot configuration: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/robot-config/{robot_type}")
def get_robot_config(robot_type: str, available_configs: str = ""):
    """Get the saved configuration for a robot type"""
    try:
        # Parse available configs from query parameter
        available_configs_list = []
        if available_configs:
            available_configs_list = [cfg.strip() for cfg in available_configs.split(",") if cfg.strip()]
        
        saved_config = config.get_saved_robot_config(robot_type)
        default_config = config.get_default_robot_config(robot_type, available_configs_list)
        
        return {
            "status": "success", 
            "saved_config": saved_config,
            "default_config": default_config
        }
    except Exception as e:
        logger.error(f"Error getting robot configuration: {e}")
        return {"status": "error", "message": str(e)}


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
    try:
        if create:
            if get_robot_record(name) is not None:
                return JSONResponse(status_code=409, content={"status": "error", "message": "A robot with this name already exists"})
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


@app.delete("/robots/{name}")
def delete_robot(name: str):
    """Delete a robot record."""
    if not is_valid_robot_name(name):
        return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid robot name"})
    if delete_robot_record(name):
        return {"status": "success"}
    return JSONResponse(status_code=404, content={"status": "error", "message": "Robot not found"})


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources when FastAPI shuts down"""
    logger.info("🔄 FastAPI shutting down, cleaning up...")

    # Stop any active recording - handled by recording module cleanup

    if manager:
        manager.stop_broadcast_thread()
    logger.info("✅ Cleanup completed")


# Serve the built frontend at /. Must be mounted last so API routes win.
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
else:
    logger.warning(
        f"frontend/dist not found at {FRONTEND_DIST}; "
        "run `npm run build` in frontend/ or use `lelab --dev`."
    )
