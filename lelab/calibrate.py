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
Calibration module for the web interface.

This module provides calibration functionality similar to the CLI calibrate.py,
but adapted for the web interface with step-by-step guidance.
"""

import logging
import os
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Literal

from lerobot.motors import MotorCalibration
from lerobot.motors.feetech import OperatingMode
from lerobot.robots import (
    Robot,
    make_robot_from_config,
)
from lerobot.teleoperators import (
    Teleoperator,
    make_teleoperator_from_config,
)
from lerobot.utils.utils import init_logging

from .utils.config import calibration_dir_for_device, save_robot_record

logger = logging.getLogger(__name__)

# Raw-tick center of the 12-bit (0-4095) Feetech encoder. Step 1 homing offsets
# the start pose to read ~2047, so a calibration that really started from the
# documented middle position records a range whose midpoint sits near this value.
ENCODER_MID_TICK = 2047

# Allowed deviation of a recorded range's midpoint from ENCODER_MID_TICK, as a
# fraction of the recorded range width.
CENTERING_TOLERANCE = 0.2

# Joints exempt from the centering check: users legitimately home the gripper
# closed (~580 ticks of midpoint deviation on real data), and wrist_roll is a
# full-turn motor upstream (lerobot's CLI calibration forces its range to
# 0-4095 instead of recording it), so its recorded midpoint carries no meaning.
CENTERING_EXEMPT_MOTORS = frozenset({"gripper", "wrist_roll"})

# wrist_roll is a continuous full-turn joint. Official lerobot-calibrate
# behavior (lerobot/robots/so_follower/so_follower.py::calibrate, same in
# so_leader.py): the user is told to move all joints EXCEPT wrist_roll, and its
# range is unconditionally hardcoded to the full turn — a continuous joint has
# no min/max, only the homing offset from the middle pose. Sweeping it crosses
# the encoder wrap (a ~4096 single-frame jump), so it is exempt from the
# discontinuity check and any recorded range is discarded.
FULL_TURN_MOTORS = frozenset({"wrist_roll"})
FULL_TURN_RANGE = (0, 4095)


def final_motor_ranges(mins: dict[str, int], maxes: dict[str, int]) -> dict[str, tuple[int, int]]:
    """Recorded (min, max) per motor, with full-turn joints forced to 0-4095."""
    return {
        motor: (FULL_TURN_RANGE if motor in FULL_TURN_MOTORS else (mins[motor], maxes[motor]))
        for motor in mins
    }


def find_off_center_joints(ranges: dict[str, tuple[float, float]]) -> list[str]:
    """Return the joints whose recorded range is not centered on the start pose.

    `ranges` maps motor name to (range_min, range_max) in raw ticks. A joint
    passes when |ENCODER_MID_TICK - (range_min + range_max) / 2| is at most
    CENTERING_TOLERANCE of the range width; a larger deviation means the joint
    started near one of its limits rather than mid-range, so the homing offsets
    captured in step 1 would skew the saved calibration.
    """
    offending = []
    for motor, (range_min, range_max) in ranges.items():
        if motor in CENTERING_EXEMPT_MOTORS:
            continue
        midpoint = (range_min + range_max) / 2
        if abs(ENCODER_MID_TICK - midpoint) > CENTERING_TOLERANCE * (range_max - range_min):
            offending.append(motor)
    return offending


class CalibrationCenteringError(Exception):
    """Raised when the recorded ranges show calibration didn't start mid-pose.

    Detected after range recording, before anything is saved: if a joint's
    recorded range is heavily skewed to one side of the raw-tick center, the
    arm was not in the documented middle position when calibration began.
    """


class CalibrationDiscontinuityError(Exception):
    """Raised when a motor position reading jumps across the encoder wrap-around.

    The Feetech encoder is 12-bit (0-4095); if calibration starts with a joint
    near a boundary, moving it past 0 or 4095 produces a single-frame delta of
    ~4096. The user-side fix is to start with all joints in the middle of their
    range, as documented in the SO-101 docs.
    """


@dataclass
class CalibrationStatus:
    """Status information for calibration process"""

    calibration_active: bool = False
    status: str = "idle"  # "idle", "connecting", "recording", "completed", "error", "stopping"
    device_type: str | None = None
    error: str | None = None
    message: str = ""
    step: int = 0  # Current calibration step
    total_steps: int = 1  # Total number of calibration steps
    current_positions: dict[str, float] = None
    recorded_ranges: dict[str, dict[str, float]] = None  # {motor: {min: val, max: val, current: val}}
    # Batch (multi-arm) calibration overlay. `batch_active` is True while a
    # sequential batch driver is running; the per-arm fields above (status,
    # step, recorded_ranges, ...) always describe the CURRENT arm, so a client
    # reads both surfaces at once: which arm of N is active AND its step.
    batch_active: bool = False
    batch_total: int = 0  # Number of arms in the running batch
    batch_index: int = 0  # 0-based index of the arm currently being calibrated
    # Describes the current arm: {device_type, arm, port, config_file}.
    batch_current: dict[str, str] | None = None
    # Names of arms (device_type + arm) that finished successfully in this batch.
    batch_completed: list[str] = None
    # When a batch aborts on an arm's error, names the failing arm.
    batch_failed_arm: str | None = None


@dataclass
class CalibrationRequest:
    """Request parameters for starting calibration"""

    device_type: Literal["robot", "teleop"]
    port: str
    config_file: str
    robot_name: str | None = None  # When set, write port + config back into the robot record on success
    overwrite: bool = False  # Must be explicitly true to replace an existing config file of the same name
    arm: Literal["left", "right"] = (
        "left"  # Which arm of a bimanual robot; "left" is also the single-arm pair
    )


@dataclass
class CalibrationBatchArm:
    """One arm's target inside a batch calibration request.

    Carries the same per-arm fields a single CalibrationRequest needs (minus
    robot_name/overwrite, which live on the parent batch and apply to all arms).
    """

    device_type: Literal["robot", "teleop"]
    port: str
    config_file: str
    arm: Literal["left", "right"] = "left"


@dataclass
class CalibrationBatchRequest:
    """Request to calibrate a chosen SUBSET of arms in one guided sequence.

    The arms are calibrated one at a time, reusing the single-arm flow; the
    existing /complete-calibration-step endpoint advances whichever arm is
    currently active. `robot_name`/`overwrite` apply to every arm.
    """

    robot_name: str
    arms: list[CalibrationBatchArm]
    overwrite: bool = False

    def __post_init__(self) -> None:
        # FastAPI hands us plain dicts for nested models; coerce so downstream
        # code (and validation) always sees CalibrationBatchArm instances.
        self.arms = [a if isinstance(a, CalibrationBatchArm) else CalibrationBatchArm(**a) for a in self.arms]


def validate_batch_arms(arms: list[CalibrationBatchArm]) -> str | None:
    """Return a human-readable reason the arm list is invalid, or None if OK.

    Rejects: empty list, more than 4 arms, two arms targeting the same
    (device_type, arm) slot, and two same-side arms sharing a config name
    (mirrors config_slot_conflict: one physical arm's calibration on two arms).
    """
    if not arms:
        return "Select at least one arm to calibrate."
    if len(arms) > 4:
        return "A batch can calibrate at most 4 arms."

    slots: set[tuple[str, str]] = set()
    for a in arms:
        slot = (a.device_type, a.arm)
        if slot in slots:
            return f"Duplicate arm slot selected: {a.device_type} {a.arm}."
        slots.add(slot)

    # Same-side config-name collision (both leaders or both followers sharing a
    # name = one physical arm's calibration on two arms).
    for device_type in ("teleop", "robot"):
        side = [a for a in arms if a.device_type == device_type]
        stems = [a.config_file[:-5] if a.config_file.endswith(".json") else a.config_file for a in side]
        dupes = {s for s in stems if stems.count(s) > 1}
        if dupes:
            label = "leader" if device_type == "teleop" else "follower"
            return (
                f"Two {label} arms share the calibration name "
                f"'{next(iter(dupes))}'. Give each arm a distinct name."
            )
    return None


class CalibrationManager:
    """Manages calibration process for the web interface"""

    def __init__(self):
        self.status = CalibrationStatus()
        self.device: Robot | Teleoperator | None = None
        self.calibration_thread: threading.Thread | None = None
        self.stop_calibration = False
        self._status_lock = threading.Lock()
        self._step_complete = threading.Event()
        self._recording_active = False
        self._start_positions = {}
        self._mins = {}
        self._maxes = {}
        self._homing_offsets = {}
        self._current_request: CalibrationRequest | None = None
        # Batch driver thread + cancellation flag (separate from stop_calibration,
        # which the per-arm logic already watches). Set to abort remaining arms.
        self.batch_thread: threading.Thread | None = None
        self._batch_abort = False

        # Initialize logging
        init_logging()

    def get_status(self) -> CalibrationStatus:
        """Get current calibration status"""
        with self._status_lock:
            # Update current positions if we're recording and device is connected
            if self.status.status == "recording" and self.device and self.device.is_connected:
                try:
                    # Try reading positions with quick retry on port contention
                    positions = None
                    for attempt in range(2):  # Quick retry for status updates
                        try:
                            positions = self.device.bus.sync_read("Present_Position", normalize=False)
                            break
                        except Exception as read_error:
                            if "Port is in use" in str(read_error) and attempt < 1:
                                time.sleep(0.005)  # Very short delay
                                continue
                            else:
                                raise read_error

                    if positions:
                        # Update recorded ranges
                        if not self.status.recorded_ranges:
                            self.status.recorded_ranges = {}

                        for motor, pos in positions.items():
                            # Filter out invalid readings (0, negative, or extreme values)
                            if pos <= 0 or pos >= 5000:
                                continue  # Skip invalid readings

                            if motor in FULL_TURN_MOTORS:
                                # Report the range that will actually be saved
                                # (forced full turn), not the swept sliver —
                                # the live marker still tracks `current`.
                                full_min, full_max = FULL_TURN_RANGE
                                self.status.recorded_ranges[motor] = {
                                    "min": full_min,
                                    "max": full_max,
                                    "current": pos,
                                }
                            elif motor not in self.status.recorded_ranges:
                                self.status.recorded_ranges[motor] = {"min": pos, "max": pos, "current": pos}
                            else:
                                self.status.recorded_ranges[motor]["current"] = pos
                                self.status.recorded_ranges[motor]["min"] = min(
                                    self.status.recorded_ranges[motor]["min"], pos
                                )
                                self.status.recorded_ranges[motor]["max"] = max(
                                    self.status.recorded_ranges[motor]["max"], pos
                                )
                except Exception as e:
                    # Reduce log spam by using debug level for expected port contention
                    if "Port is in use" in str(e):
                        logger.debug(f"Port busy during position read: {e}")
                    else:
                        logger.warning(f"Failed to read positions: {e}")

            return self.status

    def _update_status(self, **kwargs):
        """Update calibration status thread-safely"""
        with self._status_lock:
            for key, value in kwargs.items():
                if hasattr(self.status, key):
                    setattr(self.status, key, value)

    def _existing_config_name(self, device_type: str, config_file: str) -> str | None:
        """Return the config STEM if a calibration file of that name already
        exists for the given side, else None. Used by the overwrite guards to
        refuse silently clobbering an existing calibration."""
        config_dir = calibration_dir_for_device(device_type)
        if config_dir is None:
            return None
        stem = config_file[:-5] if config_file.endswith(".json") else config_file
        if os.path.exists(os.path.join(config_dir, f"{stem}.json")):
            return stem
        return None

    def start_calibration(self, request: CalibrationRequest) -> dict[str, Any]:
        """Start calibration process"""
        try:
            if self.status.calibration_active:
                return {"success": False, "message": "Calibration already active"}

            # Refuse to silently overwrite an existing config file. Completing a
            # calibration saves "<config_file>.json"; if that name is taken, the
            # caller must pass overwrite=True (after confirming) or pick another
            # name. Lets the frontend warn before any data is clobbered.
            if not request.overwrite:
                stem = self._existing_config_name(request.device_type, request.config_file)
                if stem is not None:
                    return {
                        "success": False,
                        "code": "name_taken",
                        "message": f"A calibration named '{stem}' already exists. Overwrite it or choose a different name.",
                    }

            # Reset status and clear any previous calibration data
            self._start_positions = {}
            self._mins = {}
            self._maxes = {}
            self._homing_offsets = {}

            self._update_status(
                calibration_active=True,
                status="connecting",
                device_type=request.device_type,
                error=None,
                message=f"Starting calibration for {request.device_type}",
                step=0,
                current_positions=None,
                recorded_ranges=None,
            )
            self._current_request = request

            # Start calibration in a separate thread
            self.calibration_thread = threading.Thread(
                target=self._calibration_worker, args=(request,), daemon=True
            )
            self.stop_calibration = False
            self._step_complete.clear()
            self.calibration_thread.start()

            return {"success": True, "message": "Calibration started"}

        except Exception as e:
            logger.error(f"Error starting calibration: {e}")
            self._update_status(
                calibration_active=False, status="error", error=str(e), message="Failed to start calibration"
            )
            return {"success": False, "message": str(e)}

    def start_calibration_batch(self, request: CalibrationBatchRequest) -> dict[str, Any]:
        """Start a sequential batch calibration over a chosen subset of arms.

        Rejects if a single or batch calibration is already active (mutex). The
        arm list is validated, then EVERY arm's overwrite collision is checked
        up front so the batch fails fast before any hardware moves. One driver
        thread then calibrates each arm in turn, reusing the single-arm flow;
        the existing /complete-calibration-step endpoint advances whichever arm
        is currently active.
        """
        try:
            if self.status.calibration_active or self.status.batch_active:
                return {"success": False, "message": "Calibration already active"}

            reason = validate_batch_arms(request.arms)
            if reason is not None:
                return {"success": False, "message": reason}

            # Fail fast: pre-check EVERY arm's overwrite collision before any
            # hardware moves. Report which arm is taken (same shape/code the
            # single-arm guard uses) so the frontend can prompt per-arm.
            if not request.overwrite:
                for arm in request.arms:
                    stem = self._existing_config_name(arm.device_type, arm.config_file)
                    if stem is not None:
                        return {
                            "success": False,
                            "code": "name_taken",
                            "arm": {"device_type": arm.device_type, "arm": arm.arm},
                            "message": (
                                f"A calibration named '{stem}' already exists for "
                                f"{arm.device_type} {arm.arm}. Overwrite it or choose a different name."
                            ),
                        }

            # Reset per-arm + batch status for a clean start.
            self._start_positions = {}
            self._mins = {}
            self._maxes = {}
            self._homing_offsets = {}

            self._update_status(
                calibration_active=True,
                status="connecting",
                device_type=request.arms[0].device_type,
                error=None,
                message="Starting batch calibration",
                step=0,
                current_positions=None,
                recorded_ranges=None,
                batch_active=True,
                batch_total=len(request.arms),
                batch_index=0,
                batch_current={
                    "device_type": request.arms[0].device_type,
                    "arm": request.arms[0].arm,
                    "port": request.arms[0].port,
                    "config_file": request.arms[0].config_file,
                },
                batch_completed=[],
                batch_failed_arm=None,
            )

            self.stop_calibration = False
            self._batch_abort = False
            self._step_complete.clear()
            self.batch_thread = threading.Thread(target=self._batch_worker, args=(request,), daemon=True)
            self.batch_thread.start()

            return {"success": True, "message": "Batch calibration started"}

        except Exception as e:
            logger.error(f"Error starting batch calibration: {e}")
            self._update_status(
                calibration_active=False,
                batch_active=False,
                status="error",
                error=str(e),
                message="Failed to start batch calibration",
            )
            return {"success": False, "message": str(e)}

    def _batch_worker(self, request: CalibrationBatchRequest):
        """Driver thread: calibrate each arm sequentially, reusing the single-arm
        flow. Advances batch_index after each arm completes. On any arm error the
        batch STOPS and records which arm failed — earlier arms stay calibrated
        (partial completion is acceptable; downstream staging handles it)."""
        completed_labels: list[str] = []
        try:
            for index, arm in enumerate(request.arms):
                if self._batch_abort or self.stop_calibration:
                    logger.info("Batch calibration aborted before arm %d", index)
                    break

                label = f"{arm.device_type} {arm.arm}"
                self._update_status(
                    device_type=arm.device_type,
                    batch_index=index,
                    batch_current={
                        "device_type": arm.device_type,
                        "arm": arm.arm,
                        "port": arm.port,
                        "config_file": arm.config_file,
                    },
                    status="connecting",
                    step=0,
                    error=None,
                    recorded_ranges=None,
                    message=f"Arm {index + 1} of {len(request.arms)}: {label}",
                )

                per_arm = CalibrationRequest(
                    device_type=arm.device_type,
                    port=arm.port,
                    config_file=arm.config_file,
                    robot_name=request.robot_name,
                    overwrite=request.overwrite,
                    arm=arm.arm,
                )

                # The step event is reset per-arm inside _run_single_arm, and
                # /complete-calibration-step keeps advancing THIS arm's steps.
                try:
                    finished = self._run_single_arm(per_arm)
                except (CalibrationCenteringError, CalibrationDiscontinuityError) as e:
                    logger.error(f"Batch arm {label} aborted: {e}")
                    self._update_status(
                        error=str(e),
                        batch_failed_arm=label,
                        batch_completed=list(completed_labels),
                    )
                    self._cleanup_and_finish(f"Batch stopped — {label}: {e}", status="error")
                    return
                except Exception as e:
                    logger.error(f"Batch arm {label} failed: {e}")
                    logger.error(traceback.format_exc())
                    self._update_status(batch_failed_arm=label, batch_completed=list(completed_labels))
                    self._cleanup_and_finish(f"Batch stopped — {label} failed: {e}", status="error")
                    return

                if not finished:
                    # A stop/abort was requested mid-arm.
                    logger.info(f"Batch cancelled during arm {label}")
                    self._update_status(batch_completed=list(completed_labels))
                    self._cleanup_and_finish("Batch calibration cancelled")
                    return

                completed_labels.append(label)
                self._update_status(batch_completed=list(completed_labels))

            # All requested arms done (or aborted between arms).
            if self._batch_abort or self.stop_calibration:
                self._update_status(batch_completed=list(completed_labels))
                self._cleanup_and_finish("Batch calibration cancelled")
            else:
                self._update_status(batch_completed=list(completed_labels))
                self._cleanup_and_finish("Batch calibration completed successfully", status="completed")

        except Exception as e:
            logger.error(f"Batch calibration driver error: {e}")
            logger.error(traceback.format_exc())
            self._update_status(batch_completed=list(completed_labels))
            self._cleanup_and_finish(f"Batch calibration failed: {e}", status="error")
        finally:
            logger.info("Batch calibration driver thread finishing")
            self._update_status(batch_active=False)
            if self.status.calibration_active:
                logger.warning("Batch driver ending but calibration still marked active - forcing cleanup")
                self._cleanup_and_finish("Batch calibration stopped", status="idle")

    def complete_step(self) -> dict[str, Any]:
        """Complete the current calibration step"""
        try:
            if not self.status.calibration_active:
                return {"success": False, "message": "No calibration active"}

            if self.status.status == "recording":
                # Complete recording step
                self._recording_active = False
                self._step_complete.set()
                return {"success": True, "message": "Range recording completed"}

            else:
                return {"success": False, "message": f"Cannot complete step in status: {self.status.status}"}

        except Exception as e:
            logger.error(f"Error completing step: {e}")
            return {"success": False, "message": str(e)}

    def stop_calibration_process(self) -> dict[str, Any]:
        """Stop calibration process"""
        try:
            if not self.status.calibration_active:
                return {"success": False, "message": "No calibration active"}

            logger.info("Stopping calibration process...")
            # _batch_abort skips any remaining arms; stop_calibration stops the
            # current arm's step loop. Both are honored by the batch driver.
            self.stop_calibration = True
            self._batch_abort = True
            self._recording_active = False
            self._step_complete.set()  # Unblock any waiting step

            self._update_status(status="stopping", message="Stopping calibration...")

            # Wait for whichever thread is running (single-arm worker or batch driver).
            for thread in (self.calibration_thread, self.batch_thread):
                if thread and thread.is_alive():
                    thread.join(timeout=5.0)
                    if thread.is_alive():
                        logger.warning("Calibration thread did not finish within timeout, forcing cleanup")

            # Force cleanup and finish
            self._cleanup_and_finish("Calibration stopped", status="idle")

            logger.info("Calibration stop completed")
            return {"success": True, "message": "Calibration stopped"}

        except Exception as e:
            logger.error(f"Error stopping calibration: {e}")
            # Force cleanup on error too
            self._cleanup_and_finish("Calibration stopped with error", status="error")
            return {"success": False, "message": str(e)}

    def _run_single_arm(self, request: CalibrationRequest) -> bool:
        """Run ONE arm's full calibration synchronously: connect → Step 1 homing
        (waits on _step_complete) → Step 2 range recording → save + record
        write-back. Returns True when the arm completed, False if a stop was
        requested mid-flow. Raises on hardware/validation errors (the caller
        decides how to surface them). Does NOT touch calibration_active or run
        the final cleanup — the caller (single-arm worker or batch driver) owns
        lifecycle so the same logic serves both. Always disconnects this arm's
        device before returning/raising.

        Per-arm state (_mins/_maxes/_homing_offsets/_start_positions) and the
        _step_complete event are reset here so a batch's later arm never
        inherits the previous arm's data.
        """
        # Fresh per-arm state so a batch's second arm starts clean.
        self._start_positions = {}
        self._mins = {}
        self._maxes = {}
        self._homing_offsets = {}
        self._step_complete.clear()
        self._current_request = request

        try:
            logger.info(f"Starting calibration worker for {request.device_type}")

            # Create device configuration
            if request.device_type == "robot":
                from lerobot.robots.so_follower import SO101FollowerConfig

                config = SO101FollowerConfig(port=request.port, id=request.config_file)
            elif request.device_type == "teleop":
                from lerobot.teleoperators.so_leader import SO101LeaderConfig

                config = SO101LeaderConfig(port=request.port, id=request.config_file)
            else:
                raise ValueError(f"Unknown device type: {request.device_type}")

            self._update_status(status="connecting", message="Connecting to device...")

            # Create and connect device
            if request.device_type == "robot":
                self.device = make_robot_from_config(config)
            else:
                self.device = make_teleoperator_from_config(config)

            logger.info("Connecting to device...")
            self.device.connect(calibrate=False)

            if self.stop_calibration:
                logger.info("Calibration stopped after device connection")
                return False

            # Start Step 1: Homing
            self._step_homing()

            if self.stop_calibration:
                logger.info("Calibration stopped after homing step")
                return False

            # Start Step 2: Range Recording
            self._step_range_recording()

            if self.stop_calibration:
                logger.info("Calibration stopped after recording step")
                return False

            # Complete calibration (save + record write-back)
            self._complete_calibration()

            logger.info("Calibration completed successfully")
            return True
        finally:
            # Each arm releases its serial port before the next arm connects.
            self._cleanup_device()
            self._recording_active = False

    def _calibration_worker(self, request: CalibrationRequest):
        """Worker thread for a single-arm calibration."""
        try:
            completed = self._run_single_arm(request)
            if not completed:
                self._cleanup_and_finish("Calibration cancelled")
                return
            self._cleanup_and_finish("Calibration completed successfully", status="completed")

        except (CalibrationCenteringError, CalibrationDiscontinuityError) as e:
            logger.error(f"Calibration aborted: {e}")
            self._update_status(error=str(e))
            self._cleanup_and_finish(str(e), status="error")
        except Exception as e:
            logger.error(f"Calibration error: {e}")
            logger.error(traceback.format_exc())
            # Ensure cleanup happens even on error
            self._cleanup_and_finish(f"Calibration failed: {e}", status="error")
        finally:
            # Ensure we always clean up and reset the active flag
            logger.info("Calibration worker thread finishing")
            if self.status.calibration_active:
                logger.warning(
                    "Worker thread ending but calibration still marked as active - forcing cleanup"
                )
                self._cleanup_and_finish("Calibration stopped", status="idle")

    def _step_homing(self):
        """Auto-capture homing offsets from the device's current position."""
        logger.info("Setting homing offsets from current position")

        # Disable torque to allow manual movement during recording
        self.device.bus.disable_torque()
        for motor in self.device.bus.motors:
            self.device.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        self.device.bus.reset_calibration()
        actual_positions = self.device.bus.sync_read("Present_Position", normalize=False)
        logger.info(f"Current positions for homing: {actual_positions}")

        self._homing_offsets = self.device.bus._get_half_turn_homings(actual_positions)
        logger.info(f"Calculated homing offsets: {self._homing_offsets}")

        for motor, offset in self._homing_offsets.items():
            self.device.bus.write("Homing_Offset", motor, offset)

    def _step_range_recording(self):
        """Record range of motion as the user moves all joints."""
        logger.info("Starting range recording step")

        # Initialize range tracking with retry and validation
        self._start_positions = {}
        for attempt in range(5):  # Try multiple times to get valid initial positions
            try:
                positions = self.device.bus.sync_read("Present_Position", normalize=False)
                # Validate initial positions
                valid_positions = {}
                for motor, pos in positions.items():
                    if pos > 0 and pos < 5000:  # Valid range
                        valid_positions[motor] = pos

                if len(valid_positions) == len(positions):  # All positions are valid
                    self._start_positions = valid_positions
                    break
                else:
                    logger.warning(f"Attempt {attempt + 1}: Got invalid initial positions, retrying...")
                    time.sleep(0.1)
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}: Failed to read initial positions: {e}")
                time.sleep(0.1)

        if not self._start_positions:
            raise RuntimeError("Could not get valid initial positions after multiple attempts")

        logger.info(f"Starting positions for range recording: {self._start_positions}")

        self._mins = self._start_positions.copy()
        self._maxes = self._start_positions.copy()
        logger.info(f"Initialized mins: {self._mins}")
        logger.info(f"Initialized maxes: {self._maxes}")

        self._update_status(
            status="recording",
            step=1,
            message="Move every joint EXCEPT the wrist roll through its FULL range of motion - from minimum to maximum. Leave the wrist roll near the middle: it rotates continuously and its range is set automatically.",
            recorded_ranges={
                motor: (
                    {"min": FULL_TURN_RANGE[0], "max": FULL_TURN_RANGE[1], "current": pos}
                    if motor in FULL_TURN_MOTORS
                    else {"min": pos, "max": pos, "current": pos}
                )
                for motor, pos in self._start_positions.items()
            },
        )

        self._recording_active = True
        prev_positions: dict[str, int] = dict(self._start_positions)

        # Record positions until user completes step
        while not self._step_complete.is_set() and not self.stop_calibration:
            try:
                # Try reading positions with retry on port contention
                positions = None
                for attempt in range(3):  # Try up to 3 times
                    try:
                        positions = self.device.bus.sync_read("Present_Position", normalize=False)
                        break  # Success, exit retry loop
                    except Exception as read_error:
                        if "Port is in use" in str(read_error) and attempt < 2:
                            time.sleep(0.01)  # Short delay before retry
                            continue
                        else:
                            raise read_error  # Re-raise if not port contention or final attempt

                if positions:
                    # Validate the readings - filter out invalid/zero values
                    valid_positions = {}
                    for motor, pos in positions.items():
                        # Filter out clearly invalid readings (0, negative, or extreme values)
                        if pos > 0 and pos < 5000:  # Reasonable range for motor positions
                            valid_positions[motor] = pos
                        else:
                            logger.debug(f"Filtered invalid position for {motor}: {pos}")

                    # Only update if we have valid readings
                    if valid_positions:
                        for motor, pos in valid_positions.items():
                            # Full-turn joints legitimately cross the encoder
                            # wrap when rolled — no discontinuity to detect.
                            if (
                                motor not in FULL_TURN_MOTORS
                                and motor in prev_positions
                                and abs(pos - prev_positions[motor]) > 2000
                            ):
                                raise CalibrationDiscontinuityError(
                                    "Motor discontinuity detected. Make sure to start "
                                    "the calibration with the robot in a middle position "
                                    "- all joints in the middle of their ranges."
                                )
                            prev_positions[motor] = pos
                            if motor in self._mins:
                                self._mins[motor] = min(self._mins[motor], pos)
                                self._maxes[motor] = max(self._maxes[motor], pos)

                time.sleep(0.05)  # 20Hz update rate
            except CalibrationDiscontinuityError:
                raise
            except Exception as e:
                if "Port is in use" in str(e):
                    logger.debug(f"Port busy during position read: {e}")
                else:
                    logger.warning(f"Error reading positions during recording: {e}")
                # Increase sleep time on error to reduce port contention
                time.sleep(0.2)

        if self.stop_calibration:
            logger.info("Range recording step cancelled due to stop request")
            return

        # Log the final recorded ranges for debugging
        logger.info("Final recorded ranges:")
        for motor in self._mins:
            logger.info(
                f"  {motor}: min={self._mins[motor]}, max={self._maxes[motor]}, range={self._maxes[motor] - self._mins[motor]}"
            )

        # Validate ranges. Full-turn joints are exempt: their recorded sweep is
        # discarded for the forced 0-4095 range, and NOT moving them is the
        # documented procedure.
        same_min_max = [
            motor
            for motor in self._mins
            if motor not in FULL_TURN_MOTORS and self._mins[motor] == self._maxes[motor]
        ]
        if same_min_max:
            raise ValueError(f"Some motors have the same min and max values: {same_min_max}")

        # Check for insufficient range movement (less than 100 motor steps)
        insufficient_range = []
        for motor in self._mins:
            if motor in FULL_TURN_MOTORS:
                continue
            range_diff = self._maxes[motor] - self._mins[motor]
            if range_diff < 100:  # Less than 100 motor steps seems insufficient
                insufficient_range.append(f"{motor}: {range_diff}")

        if insufficient_range:
            logger.warning(
                f"Some motors may not have been moved through sufficient range: {insufficient_range}"
            )
            logger.warning("Consider moving all joints through their full range of motion during calibration")

        self._step_complete.clear()
        logger.info("Range recording step completed")

    def _complete_calibration(self):
        """Complete the calibration and save results"""
        logger.info("Completing calibration...")

        # Centering guard: fail before anything is written if the recorded
        # ranges show the arm didn't start from the middle pose (see
        # find_off_center_joints). The worker's error path cleans up, so no
        # half-written calibration file is left behind.
        off_center = find_off_center_joints(
            {motor: (self._mins[motor], self._maxes[motor]) for motor in self._mins}
        )
        if off_center:
            raise CalibrationCenteringError(
                f"Start pose wasn't the middle position: {', '.join(off_center)}. "
                "Re-run calibration starting from the middle pose."
            )

        # Log motor information for debugging
        logger.info("Motor configuration:")
        for motor, m in self.device.bus.motors.items():
            logger.info(f"  {motor}: ID={m.id}, Model={m.model}")

        # Create calibration dict. Full-turn joints get the forced 0-4095 range
        # (matching upstream lerobot), not whatever sliver was swept.
        ranges = final_motor_ranges(self._mins, self._maxes)
        calibration = {}
        for motor, m in self.device.bus.motors.items():
            range_min, range_max = ranges[motor]
            calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=self._homing_offsets[motor],
                range_min=range_min,
                range_max=range_max,
            )
            logger.info(
                f"Calibration for {motor}: "
                f"ID={m.id}, "
                f"homing_offset={self._homing_offsets[motor]}, "
                f"range_min={range_min}, "
                f"range_max={range_max}"
            )

        # Write and save calibration
        self.device.calibration = calibration
        self.device.bus.write_calibration(calibration)
        self.device._save_calibration()

        logger.info(f"Calibration saved to {self.device.calibration_fpath}")

        # Robot-record write-back: if this calibration was launched from a tile,
        # update the robot's port + config field for the side that was just calibrated.
        request = self._current_request
        if request is not None and request.robot_name:
            # Store the config as a STEM (no .json) — that's the canonical
            # user-facing name and the id lerobot uses; the extension is only the
            # on-disk filename. (Records used to store "<name>.json"; reads now
            # normalize old ones, so this stays consistent.)
            config_stem = (
                request.config_file[:-5] if request.config_file.endswith(".json") else request.config_file
            )
            # Pick the record fields for this side AND arm. For a bimanual robot
            # the right arm writes the right_* fields; "left" is also the single
            # robot's only pair.
            is_right = request.arm == "right"
            if request.device_type == "teleop":
                port_field = "right_leader_port" if is_right else "leader_port"
                config_field = "right_leader_config" if is_right else "leader_config"
            else:
                port_field = "right_follower_port" if is_right else "follower_port"
                config_field = "right_follower_config" if is_right else "follower_config"
            patch = {port_field: request.port, config_field: config_stem}
            try:
                save_robot_record(request.robot_name, patch, allow_create=False)
            except Exception as e:
                logger.warning(f"Robot-record write-back failed for {request.robot_name}: {e}")

    def _cleanup_and_finish(self, message: str, status: str = "completed"):
        """Clean up and finish calibration.

        Called only at the END of a session (single-arm worker, or the batch
        driver after the last/failed/cancelled arm) — never between a batch's
        arms — so clearing batch_active here always reflects a finished batch.
        """
        self._cleanup_device()
        self._recording_active = False
        self._update_status(calibration_active=False, batch_active=False, status=status, message=message)

    def _cleanup_device(self):
        """Clean up device connection"""
        try:
            if self.device:
                logger.info("Disconnecting device...")
                self.device.disconnect()
                self.device = None
        except Exception as e:
            logger.error(f"Error disconnecting device: {e}")


# Global calibration manager instance
calibration_manager = CalibrationManager()
