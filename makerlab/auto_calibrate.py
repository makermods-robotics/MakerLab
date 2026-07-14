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
its logs, and on success copy the file to MakerLab's expected path + write the
config back into the robot record.

The vendored script (makerlab/vendor/feetech_autocal) always saves to
``.../calibration/robots/<robot_type>/<id>.json``. For a follower that IS MakerLab's
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

from lerobot.motors.feetech import FeetechMotorsBus

from .torque import force_disable_bus_torque
from .utils.config import (
    CALIBRATION_BASE_PATH_ROBOTS,
    LEADER_CONFIG_PATH,
    save_robot_record,
)
from .utils.subprocess_env import utf8_child_env
from .vendor.feetech_autocal.calibration_defaults import SO_FOLLOWER_MOTORS

logger = logging.getLogger(__name__)

_MAX_LOG_LINES = 1000
_SCRIPT_MODULE = "makerlab.vendor.feetech_autocal.auto_calibrate_script"

# Stop escalation timings. The script's own SIGTERM cleanup (KeyboardInterrupt
# -> _graceful_stop -> safe_disable_all) first freezes the arm (~1s), then
# drives it back to its starting pose — progress-based, giving up on stall,
# with RETURN_TO_REST_BUDGET_S = 10s as an absolute ceiling — or holds the
# freeze (STOP_HOLD_S = 3s) before the 6-motor release + disconnect (~1-2s,
# more on a bus that's mid overload). Worst case ~15-16s, so 18s of grace
# before concluding the script is wedged (e.g. blocked in a C-level serial
# call, where the raised KeyboardInterrupt never materializes) and killing
# it. Keep this above the vendored script's summed stop budget
# (makerlab/vendor/feetech_autocal/auto_calibrate_script.py) so a SIGKILL can
# never land on a healthy graceful stop mid-motion.
_STOP_GRACE_S = 18.0
_STOP_KILL_WAIT_S = 5.0
_READER_JOIN_S = 5.0


class AutoCalibrationRequest(BaseModel):
    device_type: str  # "teleop" (leader) or "robot" (follower)
    port: str
    config_file: str
    robot_name: str | None = None
    arm: str = "left"  # "left" (also single) or "right"


@dataclass
class AutoCalibrationStatus:
    active: bool = False
    status: str = "idle"  # idle | running | stopping | completed | failed | stopped
    message: str = ""
    error: str | None = None


def _stem(name: str) -> str:
    return name[: -len(".json")] if name.endswith(".json") else name


def _subprocess_output_path(device_type: str, config_stem: str) -> str:
    """The file the vendored subprocess writes with --save.

    It always saves under ``.../calibration/robots/<robot_type>/<id>.json``:
    for a follower that IS MakerLab's real library dir; for a leader it's a scratch
    location (robots/so_leader) that _finalize_success later copies into the real
    leader library (teleoperators/so_leader). Used to remove a stray file left by
    a run that didn't finish cleanly, so a failed/aborted/dropped auto-calibration
    never leaves a phantom library entry (follower) or scratch file (leader)."""
    robot_type = "so_follower" if device_type == "robot" else "so_leader"
    return os.path.join(CALIBRATION_BASE_PATH_ROBOTS, robot_type, f"{config_stem}.json")


def _remove_stray_calibration_file(device_type: str, config_stem: str) -> None:
    """Best-effort removal of the subprocess's --save output on a non-success
    run. The subprocess only writes it at the natural end of a fully-successful
    calibration (an interrupt raises before its save block), so on the normal
    failed/stopped path there is nothing to remove; this is the belt-and-braces
    guard for the case where the process wrote the file and then exited non-zero
    (or post-processing failed), which would otherwise leave a phantom entry."""
    path = _subprocess_output_path(device_type, config_stem)
    try:
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"Removed stray auto-calibration file from a non-successful run: {path}")
    except OSError as e:
        logger.warning(f"Could not remove stray auto-calibration file {path}: {e}")


def _release_arm_torque(port: str) -> list[str]:
    """Fallback torque release, run from THIS process after the calibration
    subprocess is dead (so the serial port is free again).

    The script's own SIGTERM handler releases torque on a clean stop, but a
    killed or wedged script leaves the arm energized (rigid) — so reconnect a
    fresh bus to the port and disable torque motor by motor, with retries.
    Returns problem descriptions; empty means every motor was released.

    Deliberately INSTANT — no freeze/return-to-start like the script's own
    graceful stop: this is the emergency path after the child died or was
    killed, the bus state is unknown, and the priority is de-energizing the
    arm, not landing it nicely.
    """
    try:
        bus = FeetechMotorsBus(port=port, motors=SO_FOLLOWER_MOTORS.copy())
        # No handshake: a mid-calibration motor can be in overload and slow to
        # answer pings, but still accept the Torque_Enable=0 writes below.
        bus.connect(handshake=False)
    except Exception as e:
        message = (
            f"TORQUE MAY STILL BE ENABLED on {port} — could not reconnect to release the arm ({e}). "
            "The arm can stay rigid; unplug its power to release it."
        )
        logger.error(message)
        return [message]
    try:
        return force_disable_bus_torque(bus, "auto-calibration arm")
    finally:
        try:
            bus.disconnect(disable_torque=False)
        except Exception as e:
            logger.warning(f"Auto-calibration fallback release: disconnect failed: {e}")


class AutoCalibrationManager:
    """Runs the auto-calibration subprocess and tracks its state + logs."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop_thread: threading.Thread | None = None
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
            self._stop_thread = None
            # See utils/subprocess_env.py for why PYTHONIOENCODING is forced.
            # encoding="utf-8" below then makes the parent decode the pipe the
            # same way the child was told to encode it. tests/test_auto_calibrate.py
            # fakes Popen, so it can't exercise this — if you touch these
            # kwargs, re-verify with a real subprocess on Windows.
            child_env = utf8_child_env()
            try:
                self._proc = subprocess.Popen(
                    command,
                    # DEVNULL, not inherited: with the server launched from a
                    # terminal the child would see a TTY on stdin and run
                    # interactively — its "press Enter" prompts would then
                    # block forever on a terminal nobody is answering.
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    env=child_env,
                )
            except Exception as e:
                logger.error(f"Failed to launch auto-calibration: {e}")
                self.status = AutoCalibrationStatus(active=False, status="failed", error=str(e))
                return {"success": False, "message": str(e)}

            self.status = AutoCalibrationStatus(
                active=True, status="running", message="Auto-calibration running…"
            )
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

        request = self._request
        with self._lock:
            if code == 0:
                try:
                    self._finalize_success()
                    self.status = AutoCalibrationStatus(
                        active=False, status="completed", message="Auto-calibration complete"
                    )
                except Exception as e:
                    logger.error(f"Auto-calibration post-processing failed: {e}")
                    # Post-processing failed after the subprocess wrote its file
                    # — don't leave that file behind as a phantom under a failed
                    # status. (The record write-back never assigns the name on
                    # this path either: it runs inside _finalize_success, which
                    # aborted.)
                    if request is not None:
                        _remove_stray_calibration_file(request.device_type, _stem(request.config_file))
                    self.status = AutoCalibrationStatus(active=False, status="failed", error=str(e))
            elif self.status.status in ("stopping", "stopped"):
                # Stop path: _stop_worker owns the terminal status — it still
                # has to run the fallback torque release after the process is
                # gone, and only then flips the status to stopped/failed. It
                # also removes any stray output file. Leave "stopping"
                # (active=True) so the UI keeps polling.
                pass
            else:
                # Plain failure (nonzero exit, connection dropped): the
                # subprocess writes its file only on a fully-successful run, so
                # normally there's nothing here — but remove it defensively in
                # case the process wrote then died, so no phantom entry remains.
                if request is not None:
                    _remove_stray_calibration_file(request.device_type, _stem(request.config_file))
                self.status = AutoCalibrationStatus(
                    active=False, status="failed", error=f"Auto-calibration exited with code {code}"
                )
            self._proc = None

    def _finalize_success(self) -> None:
        """Copy the leader file to MakerLab's path (if needed) and write the config
        back into the robot record for the calibrated side + arm."""
        request = self._request
        if request is None:
            return
        config_stem = _stem(request.config_file)

        # The script saves leaders under robots/so_leader; MakerLab reads leaders
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
        """Request a stop. Escalation runs in a worker thread so this returns
        immediately; the status is guaranteed to reach a terminal state
        (stopped/failed) even if the subprocess has to be killed."""
        with self._lock:
            if not self.status.active or self._proc is None:
                return {"success": False, "message": "No auto-calibration is running"}
            if self.status.status == "stopping":
                return {"success": False, "message": "Auto-calibration is already stopping"}
            self.status.status = "stopping"
            # The script's graceful stop freezes the arm, drives it back to its
            # starting pose, and only then releases torque — up to ~10s. Say so,
            # or the wait reads as an unresponsive Stop button.
            self.status.message = "Stopping — returning the arm to its starting position…"
            proc = self._proc
            port = self._request.port if self._request is not None else ""
            self._stop_thread = threading.Thread(
                target=self._stop_worker, args=(proc, port), name="auto-calibration-stop", daemon=True
            )
            self._stop_thread.start()
        return {"success": True, "message": "Stopping auto-calibration"}

    def _stop_worker(self, proc: subprocess.Popen, port: str) -> None:
        """Escalating stop: SIGTERM → grace period for the script's own torque
        release → SIGKILL → direct fallback torque release from this process.

        SIGTERM makes the script raise KeyboardInterrupt and release torque
        itself, but if its main thread is wedged in a C-level serial call the
        exception never materializes and the process won't die — observed on
        hardware. SIGKILL can't be caught, so after it the arm is assumed
        energized and we release torque directly over the (now free) port.
        Always ends on a terminal status so the UI can never freeze mid-stop.
        """
        killed = False
        problems: list[str] = []
        try:
            try:
                proc.terminate()
            except Exception as e:
                logger.warning(f"Error terminating auto-calibration: {e}")
            try:
                proc.wait(timeout=_STOP_GRACE_S)
            except subprocess.TimeoutExpired:
                killed = True
                logger.warning(
                    f"Auto-calibration did not exit within {_STOP_GRACE_S}s of SIGTERM; killing it"
                )
                self._logs.append(f"Stop: process did not exit within {_STOP_GRACE_S:.0f}s; killing it.")
                try:
                    proc.kill()
                except Exception as e:
                    logger.warning(f"Error killing auto-calibration: {e}")
                try:
                    proc.wait(timeout=_STOP_KILL_WAIT_S)
                except subprocess.TimeoutExpired:
                    # Unkillable = stuck in an uninterruptible kernel call; the
                    # port is still held, so the release below will fail loudly.
                    logger.error("Auto-calibration process survived SIGKILL")

            # Let the reader thread drain the last logs and observe the exit.
            reader = self._thread
            if reader is not None:
                reader.join(timeout=_READER_JOIN_S)

            # Belt and braces: release torque from here even when the script
            # exited cleanly — a killed or wedged script leaves torque enabled,
            # and re-disabling already-released motors is harmless.
            problems = _release_arm_torque(port)
            if killed and not problems:
                self._logs.append("Process killed; torque released directly over the port.")
        except Exception as e:
            logger.error(f"Auto-calibration stop worker failed: {e}")
            problems.append(
                f"TORQUE MAY STILL BE ENABLED on {port} — the stop sequence failed ({e}). "
                "The arm can stay rigid; unplug its power to release it."
            )
        finally:
            with self._lock:
                if self.status.status == "completed":
                    # The run finished during the grace period; keep the result
                    # (and its saved file — this was a success, not a stop).
                    pass
                else:
                    # A stopped run must leave no phantom library entry. The
                    # subprocess only writes its file at the natural end of a
                    # fully-successful run, so a stop mid-calibration normally
                    # wrote nothing — remove it defensively in case a stop
                    # landed just after the save block.
                    request = self._request
                    if request is not None:
                        _remove_stray_calibration_file(request.device_type, _stem(request.config_file))
                    if problems:
                        self.status = AutoCalibrationStatus(
                            active=False,
                            status="failed",
                            message="Auto-calibration stopped",
                            error=" ".join(problems),
                        )
                    else:
                        self.status = AutoCalibrationStatus(
                            active=False, status="stopped", message="Auto-calibration stopped"
                        )
                self._proc = None

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
