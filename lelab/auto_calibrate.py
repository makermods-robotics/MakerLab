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
"""
Auto-calibration: run the vendored Feetech auto-calibration script as a
subprocess (it drives the arm under torque to find each joint's range), stream
its logs, and on success copy the file to LeLab's expected path + write the
config back into the robot record.

The vendored script (lelab/vendor/feetech_autocal) always saves to
``.../calibration/robots/<robot_type>/<id>.json``. For a follower that IS LeLab's
follower dir; for a leader we copy it to ``.../teleoperators/so_leader``.
"""

import logging
import os
import shutil
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass

from pydantic import BaseModel

from .utils.config import (
    CALIBRATION_BASE_PATH_ROBOTS,
    LEADER_CONFIG_PATH,
    save_robot_record,
)

logger = logging.getLogger(__name__)

_MAX_LOG_LINES = 1000
_SCRIPT_MODULE = "lelab.vendor.feetech_autocal.auto_calibrate_script"


class AutoCalibrationRequest(BaseModel):
    device_type: str  # "teleop" (leader) or "robot" (follower)
    port: str
    config_file: str
    robot_name: str | None = None
    arm: str = "left"  # "left" (also single) or "right"


@dataclass
class AutoCalibrationStatus:
    active: bool = False
    status: str = "idle"  # idle | running | completed | failed | stopped
    message: str = ""
    error: str | None = None


def _stem(name: str) -> str:
    return name[: -len(".json")] if name.endswith(".json") else name


class AutoCalibrationManager:
    """Runs the auto-calibration subprocess and tracks its state + logs."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._logs: deque[str] = deque(maxlen=_MAX_LOG_LINES)
        self._request: AutoCalibrationRequest | None = None
        self.status = AutoCalibrationStatus()

    def start(self, request: AutoCalibrationRequest) -> dict:
        with self._lock:
            if self.status.active:
                return {"success": False, "message": "Auto-calibration is already running"}

            if request.device_type not in ("teleop", "robot"):
                return {"success": False, "message": "Invalid device type"}
            if not request.port:
                return {"success": False, "message": "No port provided"}

            config_stem = _stem(request.config_file)
            robot_type = "so_follower" if request.device_type == "robot" else "so_leader"
            command = [
                sys.executable,
                "-m",
                _SCRIPT_MODULE,
                "--port",
                request.port,
                "--save",
                "--robot-id",
                config_stem,
                "--robot-type",
                robot_type,
            ]

            self._logs.clear()
            self._request = request
            try:
                self._proc = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
            except Exception as e:
                logger.error(f"Failed to launch auto-calibration: {e}")
                self.status = AutoCalibrationStatus(active=False, status="failed", error=str(e))
                return {"success": False, "message": str(e)}

            self.status = AutoCalibrationStatus(active=True, status="running", message="Auto-calibration running…")
            self._thread = threading.Thread(target=self._run, name="auto-calibration", daemon=True)
            self._thread.start()
            return {"success": True, "message": "Auto-calibration started"}

    def _run(self) -> None:
        proc = self._proc
        assert proc is not None
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                self._logs.append(line.rstrip("\n"))
        except Exception as e:
            logger.warning(f"Auto-calibration log reader error: {e}")
        code = proc.wait()

        with self._lock:
            if code == 0:
                try:
                    self._finalize_success()
                    self.status = AutoCalibrationStatus(
                        active=False, status="completed", message="Auto-calibration complete"
                    )
                except Exception as e:
                    logger.error(f"Auto-calibration post-processing failed: {e}")
                    self.status = AutoCalibrationStatus(active=False, status="failed", error=str(e))
            elif self.status.status == "stopped":
                # Already marked stopped by stop(); keep it.
                self.status = AutoCalibrationStatus(active=False, status="stopped", message="Auto-calibration stopped")
            else:
                self.status = AutoCalibrationStatus(
                    active=False, status="failed", error=f"Auto-calibration exited with code {code}"
                )
            self._proc = None

    def _finalize_success(self) -> None:
        """Copy the leader file to LeLab's path (if needed) and write the config
        back into the robot record for the calibrated side + arm."""
        request = self._request
        if request is None:
            return
        config_stem = _stem(request.config_file)

        # The script saves leaders under robots/so_leader; LeLab reads leaders
        # from teleoperators/so_leader. Copy it over.
        if request.device_type == "teleop":
            src = os.path.join(CALIBRATION_BASE_PATH_ROBOTS, "so_leader", f"{config_stem}.json")
            dst = os.path.join(LEADER_CONFIG_PATH, f"{config_stem}.json")
            if os.path.exists(src):
                os.makedirs(LEADER_CONFIG_PATH, exist_ok=True)
                shutil.copy2(src, dst)
                logger.info(f"Copied auto-calibrated leader config to {dst}")

        # Write port + config back into the robot record (arm-aware), mirroring
        # the manual calibration write-back.
        if request.robot_name:
            is_right = request.arm == "right"
            if request.device_type == "teleop":
                port_field = "right_leader_port" if is_right else "leader_port"
                config_field = "right_leader_config" if is_right else "leader_config"
            else:
                port_field = "right_follower_port" if is_right else "follower_port"
                config_field = "right_follower_config" if is_right else "follower_config"
            try:
                save_robot_record(
                    request.robot_name,
                    {port_field: request.port, config_field: config_stem},
                    allow_create=False,
                )
            except Exception as e:
                logger.warning(f"Auto-calibration write-back failed for {request.robot_name}: {e}")

    def stop(self) -> dict:
        with self._lock:
            if not self.status.active or self._proc is None:
                return {"success": False, "message": "No auto-calibration is running"}
            self.status.status = "stopped"
            try:
                self._proc.terminate()
            except Exception as e:
                logger.warning(f"Error terminating auto-calibration: {e}")
        return {"success": True, "message": "Stopping auto-calibration"}

    def get_status(self) -> dict:
        with self._lock:
            return {
                "active": self.status.active,
                "status": self.status.status,
                "message": self.status.message,
                "error": self.status.error,
                "logs": list(self._logs),
            }


auto_calibration_manager = AutoCalibrationManager()
