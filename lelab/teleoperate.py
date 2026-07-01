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

import logging
import math
import threading
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from lerobot.robots.bi_so_follower import BiSOFollower, BiSOFollowerConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.bi_so_leader import BiSOLeader, BiSOLeaderConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

from .utils.config import (
    FOLLOWER_CONFIG_PATH,
    LEADER_CONFIG_PATH,
    bimanual_base,
    setup_calibration_files,
)

logger = logging.getLogger(__name__)

# sts3215 motor resolution; lerobot's DEGREES normalization uses (resolution - 1).
_STS3215_MAX_RES = 4095

# General URDF mapping. Rather than chase a per-joint "zero tick" (fragile — the
# URDF's zero pose isn't reachable/meaningful for every joint), affinely map each
# motor's *calibrated* travel [range_min, range_max] onto its URDF joint limits
# [lower, upper]. Both ends are known per arm: the motor range comes from
# calibration, the limits from so101_new_calib.urdf. So the on-screen model
# tracks the real arm across its full range, with no clamping and no per-arm tick
# constants. The only per-joint fact that can't be derived is `sign`: whether
# motor-increasing maps to URDF-increasing (+1) or -decreasing (-1). Limits are
# in radians (URDF units); the gripper is RANGE_0_100 (0-100), not degrees.
_SO101_URDF_JOINTS = {
    # motor_name: (urdf_joint, lower_rad, upper_rad, sign)
    "shoulder_pan": ("Rotation", -1.91986, 1.91986, +1),
    "shoulder_lift": ("Pitch", -1.74533, 1.74533, +1),
    "elbow_flex": ("Elbow", -1.74533, 1.57080, +1),
    "wrist_flex": ("Wrist_Pitch", -1.65806, 1.65806, +1),
    "wrist_roll": ("Wrist_Roll", -2.79253, 2.79253, +1),
    "gripper": ("Jaw", -0.174533, 1.74533, +1),
}


def _motor_fraction(motor_name: str, value: float, cal) -> float | None:
    """Position of ``value`` within the motor's calibrated travel, as a 0..1 fraction.

    Returns ``None`` when a body (DEGREES) joint has no calibration to define its
    range, so the caller can fall back instead of guessing.
    """
    if motor_name == "gripper":
        # RANGE_0_100 norm mode: the observation already is a 0-100 percentage.
        return value / 100.0
    if cal is None:
        return None
    # DEGREES norm mode: value = (ticks - mid) * 360 / max_res, i.e. symmetric
    # about 0 across the calibrated range, so 0.5 sits at the midpoint.
    full_range_deg = (cal.range_max - cal.range_min) * 360.0 / _STS3215_MAX_RES
    if full_range_deg <= 0:
        return None
    return 0.5 + value / full_range_deg

# Global variables for teleoperation state
teleoperation_active = False
teleoperation_thread: threading.Thread | None = None
current_robot = None
current_teleop = None
# Guards the start path; the worker owns disconnect so stop() does not race.
_state_lock = threading.Lock()


class TeleoperateRequest(BaseModel):
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


def get_joint_positions_from_robot(robot, prefix: str = "", calibration=None) -> dict[str, float]:
    """
    Extract current joint positions from the robot and convert to URDF joint format.

    Args:
        robot: The robot instance (SO101Follower, or a BiSO* arm).
        prefix: Motor-key prefix in the observation. "" for single-arm; for a
            bimanual BiSO robot pass "left_"/"right_" to pull one arm.
        calibration: Calibration dict to use for the URDF correction. Defaults to
            ``robot.calibration``; for a BiSO robot pass the sub-arm's calibration.

    Returns:
        Dictionary mapping URDF joint names to radian values
    """
    try:
        observation = robot.get_observation()
        calibration = calibration if calibration is not None else (getattr(robot, "calibration", None) or {})

        joint_positions: dict[str, float] = {}
        debug_rows = []
        for motor_name, (urdf_joint_name, lower, upper, sign) in _SO101_URDF_JOINTS.items():
            # Single-arm uses "<motor>.pos"; a bimanual BiSO robot prefixes both
            # arms ("left_<motor>.pos"/"right_<motor>.pos"). Callers pass the
            # prefix for bimanual; when unset, fall back to the left arm so a
            # single-arm caller still gets something from a BiSO robot.
            motor_key = f"{prefix}{motor_name}.pos"
            if motor_key not in observation and not prefix and f"left_{motor_name}.pos" in observation:
                motor_key = f"left_{motor_name}.pos"
            if motor_key not in observation:
                logger.warning(f"Motor {motor_key} not found in observation")
                joint_positions[urdf_joint_name] = 0.0
                continue

            value = observation[motor_key]
            frac = _motor_fraction(motor_name, value, calibration.get(motor_name))
            if frac is None:
                # No calibration for a DEGREES joint — render the raw normalized
                # value as degrees so we still show *something* (uncalibrated).
                urdf_rad = value * math.pi / 180.0
            else:
                # Affinely map the calibrated range onto the URDF limits; the
                # clamp guards the rare case of driving slightly past calibration.
                frac = min(1.0, max(0.0, frac))
                if sign < 0:
                    frac = 1.0 - frac
                urdf_rad = lower + frac * (upper - lower)

            joint_positions[urdf_joint_name] = urdf_rad
            debug_rows.append(
                f"{motor_name:14s} raw={value:+8.2f} → {urdf_joint_name:11s} = {urdf_rad * 180.0 / math.pi:+8.2f}°"
            )

        # Throttled debug print (~once per second at 20 Hz broadcast).
        now = time.time()
        if now - getattr(get_joint_positions_from_robot, "_last_log", 0) > 1.0:
            get_joint_positions_from_robot._last_log = now
            logger.info("[joint-debug]\n  " + "\n  ".join(debug_rows))

        return joint_positions

    except Exception as e:
        logger.error(f"Error getting joint positions: {e}")
        return {urdf[0]: 0.0 for urdf in _SO101_URDF_JOINTS.values()}


def _safe_disconnect(device) -> None:
    """Disconnect a robot/teleop device, swallowing (but logging) any error.

    Used on the connection-failure cleanup path so one device's failure can't
    leave the other holding its serial port open.
    """
    if device is None:
        return
    try:
        device.disconnect()
    except Exception as e:
        logger.warning(f"Error disconnecting device during cleanup: {e}")


def _connect_bimanual(request: TeleoperateRequest):
    """Build, connect, and configure a bimanual leader+follower pair.

    Each side is a lerobot BiSO* device wrapping two SO101 arms (left = the
    primary leader/follower pair, right = the right_* pair). lerobot loads each
    sub-arm's calibration as "<base>_left/right.json" from the side's dir, so we
    set the BiSO id to that base and let lerobot load it. Returns
    (robot, teleop_device) connected, or raises after disconnecting any device.
    """
    # Validate the four files exist and follow lerobot's "<base>_left/right" naming.
    setup_calibration_files(request.leader_config, request.follower_config)
    setup_calibration_files(request.right_leader_config, request.right_follower_config)
    follower_base = bimanual_base(request.follower_config, request.right_follower_config, "follower")
    leader_base = bimanual_base(request.leader_config, request.right_leader_config, "leader")

    robot = BiSOFollower(
        BiSOFollowerConfig(
            id=follower_base,
            calibration_dir=Path(FOLLOWER_CONFIG_PATH),
            left_arm_config=SO101FollowerConfig(port=request.follower_port),
            right_arm_config=SO101FollowerConfig(port=request.right_follower_port),
        )
    )
    teleop_device = BiSOLeader(
        BiSOLeaderConfig(
            id=leader_base,
            calibration_dir=Path(LEADER_CONFIG_PATH),
            left_arm_config=SO101LeaderConfig(port=request.leader_port),
            right_arm_config=SO101LeaderConfig(port=request.right_leader_port),
        )
    )

    try:
        # Connect each of the four buses, naming the one that fails.
        for arm, label, port in (
            (robot.left_arm, "left follower", request.follower_port),
            (robot.right_arm, "right follower", request.right_follower_port),
            (teleop_device.left_arm, "left leader", request.leader_port),
            (teleop_device.right_arm, "right leader", request.right_leader_port),
        ):
            try:
                arm.bus.connect()
            except Exception as e:
                raise RuntimeError(
                    f"Could not connect to the {label} arm on {port}. "
                    "Make sure it's plugged in and powered on, then try again."
                ) from e

        # Each sub-arm auto-loaded its calibration in __init__ (id=<base>_side);
        # register it on the bus, then cameras + configure both sides.
        for arm in (robot.left_arm, robot.right_arm, teleop_device.left_arm, teleop_device.right_arm):
            arm.bus.write_calibration(arm.calibration)
        for cam in robot.cameras.values():
            cam.connect()
        robot.configure()
        teleop_device.configure()
        logger.info("Successfully connected to both bimanual arms")
        return robot, teleop_device
    except Exception:
        _safe_disconnect(robot)
        _safe_disconnect(teleop_device)
        raise


def handle_start_teleoperation(request: TeleoperateRequest, websocket_manager=None) -> dict[str, Any]:
    """Handle start teleoperation request.

    Connects to both arms *synchronously* so that a connection failure (arm
    unplugged, port busy, power off) is reported back to the caller, rather than
    dying silently in the worker thread while the API has already claimed
    success. Only the teleoperation loop runs in the background thread.
    """
    global teleoperation_active, teleoperation_thread, current_robot, current_teleop

    from . import record as _record, rollout as _rollout

    with _state_lock:
        if teleoperation_active:
            return {"success": False, "message": "Teleoperation is already active"}
        if _record.recording_active:
            return {"success": False, "message": "Recording is currently active. Stop it first."}
        if _rollout.inference_active:
            return {"success": False, "message": "Inference is currently active. Stop it first."}
        teleoperation_active = True

    robot = None
    teleop_device = None
    try:
        logger.info(
            f"Starting teleoperation with leader port: {request.leader_port}, follower port: {request.follower_port}"
        )

        if request.mode == "bimanual":
            robot, teleop_device = _connect_bimanual(request)
        else:
            # Setup calibration files
            leader_config_name, follower_config_name = setup_calibration_files(
                request.leader_config, request.follower_config
            )

            # Create robot and teleop configs
            robot_config = SO101FollowerConfig(
                port=request.follower_port,
                id=follower_config_name,
            )

            teleop_config = SO101LeaderConfig(
                port=request.leader_port,
                id=leader_config_name,
            )

            # Connect synchronously. If either device fails to connect, clean up the
            # other (so its serial port is released) and report the error — do NOT
            # leave the caller thinking teleoperation started.
            logger.info("Initializing robot and teleop device...")
            robot = SO101Follower(robot_config)
            teleop_device = SO101Leader(teleop_config)

            # Connect each arm separately so the error names which one failed and
            # tells the user what to do, instead of a generic "failed to start".
            logger.info("Connecting to follower arm...")
            try:
                robot.bus.connect()
            except Exception as e:
                raise RuntimeError(
                    f"Could not connect to the follower arm on {request.follower_port}. "
                    "Make sure it's plugged in and powered on, then try again."
                ) from e

            logger.info("Connecting to leader arm...")
            try:
                teleop_device.bus.connect()
            except Exception as e:
                raise RuntimeError(
                    f"Could not connect to the leader arm on {request.leader_port}. "
                    "Make sure it's plugged in and powered on, then try again."
                ) from e

            # Write calibration to motors' memory
            logger.info("Writing calibration to motors...")
            robot.bus.write_calibration(robot.calibration)
            teleop_device.bus.write_calibration(teleop_device.calibration)

            # Connect cameras and configure motors
            logger.info("Connecting cameras and configuring motors...")
            for cam in robot.cameras.values():
                cam.connect()
            robot.configure()
            teleop_device.configure()
            logger.info("Successfully connected to both devices")

        current_robot = robot
        current_teleop = teleop_device

        # Stream the arms in the background; the worker owns disconnect so stop()
        # does not race the serial bus from the request thread.
        # A bimanual BiSO robot exposes left_arm/right_arm; broadcast both arms'
        # joints so the frontend can drive two 3D viewers.
        is_bimanual = hasattr(robot, "left_arm") and hasattr(robot, "right_arm")

        def teleoperation_worker():
            global teleoperation_active, current_robot, current_teleop

            logger.info("Starting teleoperation loop...")
            try:
                last_broadcast_time = 0
                broadcast_interval = 0.05  # 20 FPS

                while teleoperation_active:
                    action = teleop_device.get_action()
                    robot.send_action(action)

                    current_time = time.time()
                    if current_time - last_broadcast_time >= broadcast_interval:
                        try:
                            if is_bimanual:
                                joint_positions = get_joint_positions_from_robot(
                                    robot, prefix="left_", calibration=robot.left_arm.calibration
                                )
                            else:
                                joint_positions = get_joint_positions_from_robot(robot)
                            joint_data = {
                                "type": "joint_update",
                                "joints": joint_positions,
                                "timestamp": current_time,
                            }
                            if is_bimanual:
                                joint_data["joints_right"] = get_joint_positions_from_robot(
                                    robot, prefix="right_", calibration=robot.right_arm.calibration
                                )
                            if websocket_manager and websocket_manager.active_connections:
                                websocket_manager.broadcast_joint_data_sync(joint_data)
                            last_broadcast_time = current_time
                        except Exception as e:
                            logger.error(f"Error broadcasting joint data: {e}")

                    time.sleep(0.001)
            except Exception as e:
                logger.error(f"Error during teleoperation loop: {e}")
            finally:
                _safe_disconnect(robot)
                _safe_disconnect(teleop_device)
                logger.info("Teleoperation stopped")
                teleoperation_active = False
                current_robot = None
                current_teleop = None

        teleoperation_thread = threading.Thread(
            target=teleoperation_worker, name="teleoperation-worker", daemon=True
        )
        teleoperation_thread.start()

        return {
            "success": True,
            "message": "Teleoperation started successfully",
            "leader_port": request.leader_port,
            "follower_port": request.follower_port,
        }

    except Exception as e:
        # Connection (or setup) failed before the loop started: release any
        # device that did open, reset state, and surface the error.
        _safe_disconnect(robot)
        _safe_disconnect(teleop_device)
        teleoperation_active = False
        current_robot = None
        current_teleop = None
        logger.error(f"Failed to start teleoperation: {e}")
        # str(e) is already a user-facing message for the connection failures
        # raised above; the toast title supplies the "error starting" context.
        return {"success": False, "message": str(e)}


def handle_stop_teleoperation() -> dict[str, Any]:
    """Handle stop teleoperation request.

    Signals the worker via `teleoperation_active = False` and waits for it to
    exit. The worker owns the disconnect call, so this avoids racing the
    serial bus from the request thread.
    """
    global teleoperation_active, teleoperation_thread

    if not teleoperation_active:
        return {"success": False, "message": "No teleoperation session is active"}

    logger.info("Stop teleoperation triggered from web interface")
    teleoperation_active = False

    worker = teleoperation_thread
    if worker is not None and worker.is_alive():
        worker.join(timeout=5.0)
        if worker.is_alive():
            logger.warning("Teleoperation worker did not exit within 5s")
    teleoperation_thread = None

    return {"success": True, "message": "Teleoperation stopped successfully"}


def handle_teleoperation_status() -> dict[str, Any]:
    """Handle teleoperation status request"""
    return {
        "teleoperation_active": teleoperation_active,
        "available_controls": {
            "stop_teleoperation": teleoperation_active,
        },
        "message": "Teleoperation status retrieved successfully",
    }


def handle_get_joint_positions() -> dict[str, Any]:
    """Handle get current robot joint positions request"""
    global current_robot

    if not teleoperation_active or current_robot is None:
        return {"success": False, "message": "No active teleoperation session"}

    try:
        joint_positions = get_joint_positions_from_robot(current_robot)
        return {"success": True, "joint_positions": joint_positions, "timestamp": time.time()}
    except Exception as e:
        logger.error(f"Error getting joint positions: {e}")
        return {"success": False, "message": f"Failed to get joint positions: {str(e)}"}
