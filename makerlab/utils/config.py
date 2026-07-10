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
import os
import platform
import re
import shutil
import time
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

RobotSide = Literal["leader", "follower"]

# Define the calibration config paths (shared between features)
CALIBRATION_BASE_PATH_TELEOP = os.path.expanduser("~/.cache/huggingface/lerobot/calibration/teleoperators")
CALIBRATION_BASE_PATH_ROBOTS = os.path.expanduser("~/.cache/huggingface/lerobot/calibration/robots")
LEADER_CONFIG_PATH = os.path.join(CALIBRATION_BASE_PATH_TELEOP, "so_leader")
FOLLOWER_CONFIG_PATH = os.path.join(CALIBRATION_BASE_PATH_ROBOTS, "so_follower")

# Define port storage path
PORT_CONFIG_PATH = os.path.expanduser("~/.cache/huggingface/lerobot/ports")
LEADER_PORT_FILE = os.path.join(PORT_CONFIG_PATH, "leader_port.txt")
FOLLOWER_PORT_FILE = os.path.join(PORT_CONFIG_PATH, "follower_port.txt")

# Define configuration storage path
CONFIG_STORAGE_PATH = os.path.expanduser("~/.cache/huggingface/lerobot/saved_configs")
LEADER_CONFIG_FILE = os.path.join(CONFIG_STORAGE_PATH, "leader_config.txt")
FOLLOWER_CONFIG_FILE = os.path.join(CONFIG_STORAGE_PATH, "follower_config.txt")

# Robot config records (per-robot JSON metadata)
ROBOTS_PATH = os.path.expanduser("~/.cache/huggingface/lerobot/robots")

# Staging root for bimanual (BiSO) sessions. lerobot's BiSO devices take ONE
# calibration_dir + ONE base id and load each sub-arm as "<base>_left.json" /
# "<base>_right.json" — there is no way to point left/right at differently named
# library files. To free bimanual calibration from that naming constraint, we
# alias: the user-facing library keeps arbitrary names, and at session start we
# COPY the four selected library files into per-device staging dirs under this
# root as "<base>_left.json"/"<base>_right.json" for lerobot to load. The copy is
# unconditional every session (see stage_bimanual_calibrations) so a recalibrated
# library file always refreshes its stale staging alias.
MAKERLAB_BISO_STAGING_PATH = os.path.expanduser("~/.cache/huggingface/lerobot/makerlab_biso")

# Fallback base id when a bimanual start request carries no robot name (older
# frontends). Filesystem-safe and stable; a single unnamed bimanual robot reuses
# the same staging dir harmlessly since the copy is unconditional.
DEFAULT_BIMANUAL_BASE = "bimanual"

# Hub-job ids the user dismissed from the jobs UI (JSON list of strings). The
# HF Jobs API has no delete — a finished job stays in list_jobs() indefinitely
# — so hiding a dead run from the untracked list must be persisted locally.
DISMISSED_HUB_JOBS_FILE = os.path.expanduser("~/.cache/huggingface/lerobot/dismissed_hub_jobs.json")

# Tag stamped on every dataset pushed to the Hub from MakerLab, so we can later
# query the Hub for MakerLab-produced datasets and compute usage metrics.
MAKERLAB_TAG = "MakerLab"


def with_makerlab_tag(tags: list[str] | None) -> list[str]:
    """Return `tags` with MAKERLAB_TAG appended (deduped, order preserved)."""
    out = list(tags or [])
    if MAKERLAB_TAG not in out:
        out.append(MAKERLAB_TAG)
    return out


def _atomic_write_text(path: str, content: str) -> None:
    """Write to <path>.tmp then os.replace, so a crash mid-write never leaves
    a half-written file on disk."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        f.write(content)
    os.replace(tmp, path)


def _port_file_for(robot_type: RobotSide) -> str:
    if robot_type == "leader":
        return LEADER_PORT_FILE
    if robot_type == "follower":
        return FOLLOWER_PORT_FILE
    raise ValueError(f"robot_type must be 'leader' or 'follower', got {robot_type!r}")


def _config_file_for(robot_type: RobotSide) -> str:
    rt = robot_type.lower() if isinstance(robot_type, str) else robot_type
    if rt == "leader":
        return LEADER_CONFIG_FILE
    if rt == "follower":
        return FOLLOWER_CONFIG_FILE
    raise ValueError(f"robot_type must be 'leader' or 'follower', got {robot_type!r}")


def _require_assigned_config(config: str, side: str) -> None:
    """Fail with a legible message when an arm has no calibration assigned.

    An empty name would otherwise resolve to the calibration *directory* and
    crash shutil.copy2 with an opaque IsADirectoryError. This happens when a
    robot record's config field was cleared (e.g. its calibration config was
    deleted) and a start request is issued anyway.
    """
    if not (config or "").strip():
        raise FileNotFoundError(
            f"The {side} arm has no calibration assigned. Calibrate it "
            "(or assign a saved calibration config) before starting."
        )


def setup_calibration_files(leader_config: str, follower_config: str):
    """Setup calibration files in the correct locations for teleoperation and recording"""
    _require_assigned_config(leader_config, "leader")
    _require_assigned_config(follower_config, "follower")
    # Extract config names from file paths (remove .json extension)
    leader_config_name = os.path.splitext(leader_config)[0]
    follower_config_name = os.path.splitext(follower_config)[0]

    # Log the full paths to check if files exist
    leader_config_full_path = os.path.join(LEADER_CONFIG_PATH, leader_config)
    follower_config_full_path = os.path.join(FOLLOWER_CONFIG_PATH, follower_config)

    logger.info("Checking calibration files:")
    logger.info(f"Leader config path: {leader_config_full_path}")
    logger.info(f"Follower config path: {follower_config_full_path}")
    logger.info(f"Leader config exists: {os.path.exists(leader_config_full_path)}")
    logger.info(f"Follower config exists: {os.path.exists(follower_config_full_path)}")

    # Create calibration directories if they don't exist
    leader_calibration_dir = LEADER_CONFIG_PATH
    follower_calibration_dir = FOLLOWER_CONFIG_PATH
    os.makedirs(leader_calibration_dir, exist_ok=True)
    os.makedirs(follower_calibration_dir, exist_ok=True)

    # Copy calibration files to the correct locations if they're not already there
    leader_target_path = os.path.join(leader_calibration_dir, f"{leader_config_name}.json")
    follower_target_path = os.path.join(follower_calibration_dir, f"{follower_config_name}.json")

    if not os.path.exists(leader_target_path):
        if os.path.exists(leader_config_full_path):
            shutil.copy2(leader_config_full_path, leader_target_path)
            logger.info(f"Copied leader calibration to {leader_target_path}")
        else:
            raise FileNotFoundError(f"Leader calibration file not found: {leader_config_full_path}")
    else:
        logger.info(f"Leader calibration already exists at {leader_target_path}")

    if not os.path.exists(follower_target_path):
        if os.path.exists(follower_config_full_path):
            shutil.copy2(follower_config_full_path, follower_target_path)
            logger.info(f"Copied follower calibration to {follower_target_path}")
        else:
            raise FileNotFoundError(f"Follower calibration file not found: {follower_config_full_path}")
    else:
        logger.info(f"Follower calibration already exists at {follower_target_path}")

    return leader_config_name, follower_config_name


def setup_follower_calibration_file(follower_config: str):
    """Setup follower calibration file in the correct location for replay functionality"""
    _require_assigned_config(follower_config, "follower")
    # Extract config name from file path (remove .json extension)
    follower_config_name = os.path.splitext(follower_config)[0]

    # Log the full path to check if file exists
    follower_config_full_path = os.path.join(FOLLOWER_CONFIG_PATH, follower_config)

    logger.info("Checking follower calibration file:")
    logger.info(f"Follower config path: {follower_config_full_path}")
    logger.info(f"Follower config exists: {os.path.exists(follower_config_full_path)}")

    # Create calibration directory if it doesn't exist
    follower_calibration_dir = FOLLOWER_CONFIG_PATH
    os.makedirs(follower_calibration_dir, exist_ok=True)

    # Copy calibration file to the correct location if it's not already there
    follower_target_path = os.path.join(follower_calibration_dir, f"{follower_config_name}.json")

    if not os.path.exists(follower_target_path):
        if os.path.exists(follower_config_full_path):
            shutil.copy2(follower_config_full_path, follower_target_path)
            logger.info(f"Copied follower calibration to {follower_target_path}")
        else:
            raise FileNotFoundError(f"Follower calibration file not found: {follower_config_full_path}")
    else:
        logger.info(f"Follower calibration already exists at {follower_target_path}")

    return follower_config_name


def find_available_ports():
    """Find all available serial ports on the system"""
    try:
        from serial.tools import list_ports  # Part of pyserial library
    except ImportError as exc:
        raise ImportError("pyserial library is required. Install it with: pip install pyserial") from exc

    if platform.system() == "Windows":
        # List COM ports using pyserial
        ports = [port.device for port in list_ports.comports()]
    else:
        # Linux/macOS: globbing all of /dev/tty* returns dozens of pseudo-ttys
        # and Bluetooth/debug devices. Restrict to USB-serial adapters — the only
        # thing an SO-101 arm shows up as — and keep the tty.* naming the rest of
        # the code (and saved robot records) use.
        #   macOS:  /dev/tty.usbmodem*  /dev/tty.usbserial*
        #   Linux:  /dev/ttyUSB*        /dev/ttyACM*
        patterns = ("tty.usbmodem*", "tty.usbserial*", "ttyUSB*", "ttyACM*")
        ports = [str(path) for pattern in patterns for path in Path("/dev").glob(pattern)]
    return sorted(ports)


def find_robot_port(robot_type="robot"):
    """
    Find the port for a robot by detecting the difference when disconnecting/reconnecting

    Args:
        robot_type (str): Type of robot ("leader" or "follower" or generic "robot")

    Returns:
        str: The detected port
    """
    logger.info(f"Finding port for {robot_type}")

    # Get initial ports
    ports_before = find_available_ports()
    logger.info(f"Ports before disconnecting: {ports_before}")

    # This function returns the port detection logic, but the actual user interaction
    # should be handled by the frontend
    return {"ports_before": ports_before, "robot_type": robot_type}


def detect_port_after_disconnect(ports_before, timeout_s: float = 15.0, poll_interval_s: float = 0.3):
    """
    Wait for the user to unplug the robot and detect which port disappeared.

    Polls the available ports until exactly one entry from ``ports_before`` vanishes,
    or until ``timeout_s`` elapses. Polling avoids racing the user — they may need
    several seconds to physically pull the USB cable.

    Args:
        ports_before (list): List of ports before disconnection
        timeout_s (float): Maximum seconds to wait for a port to disappear
        poll_interval_s (float): Seconds between checks

    Returns:
        str: The detected port

    Raises:
        OSError: If the timeout elapses with no change, or more than one port disappears.
    """
    before_set = set(ports_before)
    deadline = time.monotonic() + timeout_s
    last_diff: list = []

    while time.monotonic() < deadline:
        ports_after = find_available_ports()
        ports_diff = list(before_set - set(ports_after))
        last_diff = ports_diff

        if len(ports_diff) == 1:
            port = ports_diff[0]
            logger.info(f"Detected port: {port}")
            return port
        if len(ports_diff) > 1:
            raise OSError(f"Could not detect the port. More than one port disappeared: {ports_diff}.")

        time.sleep(poll_interval_s)

    logger.info(f"Timed out waiting for unplug. Final diff: {last_diff}")
    raise OSError(
        "Timed out waiting for the robot to be unplugged. Please try again and unplug the USB cable when prompted."
    )


def save_robot_port(robot_type: RobotSide, port: str) -> None:
    """Persist the robot port for `robot_type` ('leader' or 'follower')."""
    port_file = _port_file_for(robot_type)
    _atomic_write_text(port_file, port)
    logger.info(f"Saved {robot_type} port: {port}")


def get_saved_robot_port(robot_type: RobotSide) -> str | None:
    """Return the saved port for `robot_type`, or None if no file exists."""
    port_file = _port_file_for(robot_type)
    if not os.path.exists(port_file):
        logger.info(f"No saved port found for {robot_type}")
        return None
    with open(port_file) as f:
        port = f.read().strip()
    logger.info(f"Retrieved saved {robot_type} port: {port}")
    return port


def get_default_robot_port(robot_type: RobotSide) -> str:
    """Saved port if present, else a platform-typical default."""
    saved_port = get_saved_robot_port(robot_type)
    if saved_port:
        return saved_port
    if platform.system() == "Windows":
        return "COM3"
    return "/dev/ttyUSB0"


def save_robot_config(robot_type: RobotSide, config_name: str) -> bool:
    try:
        config_file_path = _config_file_for(robot_type)
    except ValueError as e:
        logger.error(str(e))
        return False
    try:
        _atomic_write_text(config_file_path, config_name.strip())
    except Exception as e:
        logger.error(f"Error saving {robot_type} configuration: {e}")
        return False
    logger.info(f"Saved {robot_type} configuration: {config_name}")
    return True


def get_saved_robot_config(robot_type: RobotSide) -> str | None:
    try:
        config_file_path = _config_file_for(robot_type)
    except ValueError as e:
        logger.error(str(e))
        return None
    if not os.path.exists(config_file_path):
        logger.info(f"No saved {robot_type} configuration found")
        return None
    try:
        with open(config_file_path) as f:
            config_name = f.read().strip()
    except OSError as e:
        logger.error(f"Error reading saved {robot_type} configuration: {e}")
        return None
    if not config_name:
        return None
    logger.info(f"Found saved {robot_type} configuration: {config_name}")
    return config_name


def get_default_robot_config(robot_type: str, available_configs: list):
    """Get the default configuration for a robot, checking saved configs first"""
    saved_config = get_saved_robot_config(robot_type)
    if saved_config and saved_config in available_configs:
        return saved_config

    # Return first available config as fallback
    if available_configs:
        return available_configs[0]

    return None


# ---------------------------------------------------------------------------
# Robot record helpers
# ---------------------------------------------------------------------------

# Characters disallowed in a robot name (filesystem safety)
_INVALID_NAME_CHARS = ("/", "\\", "..")

# The primary leader/follower pair. In bimanual mode this is the LEFT arm pair;
# in single mode it's the only pair. Reusing these keeps existing records valid.
_SINGLE_CONFIG_FIELDS = ("leader_port", "follower_port", "leader_config", "follower_config")
# The RIGHT arm pair — populated only when mode == "bimanual".
_BIMANUAL_CONFIG_FIELDS = (
    "right_leader_port",
    "right_follower_port",
    "right_leader_config",
    "right_follower_config",
)
_ROBOT_STRING_FIELDS = _SINGLE_CONFIG_FIELDS + _BIMANUAL_CONFIG_FIELDS
_ROBOT_LIST_FIELDS = ("cameras",)

# Follower session torque cap, written RAW to the Feetech RAM Torque_Limit
# register (see makerlab/motor_power.py) — 0 = no torque, 1000 = full torque.
# Default 380 matches auto-calibration's DEFAULT_TORQUE_LIMIT: a gentler cap than
# stock so collisions and pose snaps are softer out of the box.
MAX_TORQUE_LIMIT_MIN = 0
MAX_TORQUE_LIMIT_MAX = 1000
DEFAULT_MAX_TORQUE_LIMIT = 380

# Legacy records stored motor_power as a percentage (10-100); the RAM register is
# scaled in 0.1% units, so the equivalent raw value is percent × 10.
_LEGACY_MOTOR_POWER_TO_RAW = 10


def clamp_max_torque_limit(value: object) -> int:
    """Coerce a max_torque_limit value to a safe integer in [0, 1000].

    Anything non-numeric (including bool, a subclass of int) falls back to the
    DEFAULT rather than raising, so a corrupted record can never block a session
    start.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return DEFAULT_MAX_TORQUE_LIMIT
    return max(MAX_TORQUE_LIMIT_MIN, min(MAX_TORQUE_LIMIT_MAX, int(value)))


# Config-name fields whose stored value may carry a ".json" extension to strip.
_CONFIG_NAME_FIELDS = ("leader_config", "follower_config", "right_leader_config", "right_follower_config")
_VALID_MODES = ("single", "bimanual")
_DEFAULT_MODE = "single"


def _robot_record_path(name: str) -> str:
    return os.path.join(ROBOTS_PATH, f"{name}.json")


def is_valid_robot_name(name: str) -> bool:
    """Check that a robot name is safe to use as a filename."""
    if not name or not isinstance(name, str):
        return False
    if name.strip() != name:
        return False
    return not any(bad in name for bad in _INVALID_NAME_CHARS)


def _empty_record(name: str) -> dict:
    record: dict = {"name": name, "mode": _DEFAULT_MODE, "max_torque_limit": DEFAULT_MAX_TORQUE_LIMIT}
    for field in _ROBOT_STRING_FIELDS:
        record[field] = ""
    for field in _ROBOT_LIST_FIELDS:
        record[field] = []
    return record


def get_robot_record(name: str) -> dict | None:
    """Return the robot record by name, or None if missing."""
    path = _robot_record_path(name)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to read robot record {name}: {e}")
        return None
    # Ensure all expected fields exist (forward/back compat)
    record = _empty_record(name)
    record.update({k: v for k, v in data.items() if k in record})
    record["name"] = name
    # Canonical config names are STEMS (no .json). Older records stored the
    # filename with the extension — normalize on read so every consumer sees the
    # same form. The on-disk file keeps its .json.
    for field in _CONFIG_NAME_FIELDS:
        value = record.get(field, "")
        if isinstance(value, str) and value.endswith(".json"):
            record[field] = value[: -len(".json")]
    # Guard against an unknown mode on disk.
    if record.get("mode") not in _VALID_MODES:
        record["mode"] = _DEFAULT_MODE
    # Legacy records carry motor_power (a 10-100 percentage) but no
    # max_torque_limit. Migrate on read — max_torque_limit = clamp(percent × 10)
    # — so the new field is seeded from the old cap. motor_power is not a field of
    # _empty_record, so it never survives into the returned/API record.
    if "max_torque_limit" not in data:
        legacy = data.get("motor_power")
        if isinstance(legacy, (int, float)) and not isinstance(legacy, bool):
            record["max_torque_limit"] = legacy * _LEGACY_MOTOR_POWER_TO_RAW
    # An out-of-range or corrupted value on disk (or a migrated one) is clamped so
    # every consumer sees a safe [0, 1000] integer.
    record["max_torque_limit"] = clamp_max_torque_limit(record.get("max_torque_limit"))
    return record


def list_robot_records() -> list[dict]:
    """Return all robot records on disk."""
    if not os.path.exists(ROBOTS_PATH):
        return []
    records = []
    for filename in sorted(os.listdir(ROBOTS_PATH)):
        if not filename.endswith(".json"):
            continue
        name = os.path.splitext(filename)[0]
        record = get_robot_record(name)
        if record is not None:
            records.append(record)
    return records


def save_robot_record(name: str, data: dict, allow_create: bool = True) -> bool:
    """
    Upsert a robot record. Merges `data` into the existing record, preserving
    fields not provided. Returns True if a write occurred, False if no-oped.

    - If the record exists: merge and write.
    - If the record does not exist and `allow_create` is True: create with empty
      fields then merge.
    - If the record does not exist and `allow_create` is False: log and no-op.
    """
    if not is_valid_robot_name(name):
        logger.error(f"Invalid robot name: {name!r}")
        return False

    os.makedirs(ROBOTS_PATH, exist_ok=True)
    existing = get_robot_record(name)
    if existing is None and not allow_create:
        logger.info(f"save_robot_record no-op: {name} does not exist (allow_create=False)")
        return False

    record = existing if existing is not None else _empty_record(name)
    for field in _ROBOT_STRING_FIELDS:
        if field in data and isinstance(data[field], str):
            record[field] = data[field]
    for field in _ROBOT_LIST_FIELDS:
        if field in data and isinstance(data[field], list):
            record[field] = data[field]
    # Same known-typed-fields-only merge as above: a numeric max_torque_limit is
    # clamped to the safe range, anything else is ignored (keeps existing).
    value = data.get("max_torque_limit")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        record["max_torque_limit"] = clamp_max_torque_limit(value)
    if data.get("mode") in _VALID_MODES:
        record["mode"] = data["mode"]
    record.setdefault("mode", _DEFAULT_MODE)
    record["name"] = name

    path = _robot_record_path(name)
    _atomic_write_text(path, json.dumps(record, indent=2))
    logger.info(f"Saved robot record {name}: {record}")
    return True


def delete_robot_record(name: str) -> bool:
    """Delete a robot record. Returns True if a file was removed."""
    if not is_valid_robot_name(name):
        return False
    path = _robot_record_path(name)
    if not os.path.exists(path):
        return False
    os.remove(path)
    logger.info(f"Deleted robot record {name}")
    return True


def rename_robot_record(old_name: str, new_name: str) -> tuple[bool, str]:
    """
    Rename a robot record file. Returns (ok, reason).

    `reason` is a machine-readable code on failure: "invalid_name" (either name
    fails validation), "not_found" (no record under old_name), or "name_taken"
    (a record already exists under new_name). On success reason is "".

    Renaming the *robot* record never touches calibration files: those live under
    config-name paths (leader_config / follower_config), independent of the robot
    record's name. A no-op rename (old == new) succeeds.
    """
    if not is_valid_robot_name(old_name) or not is_valid_robot_name(new_name):
        return False, "invalid_name"

    record = get_robot_record(old_name)
    if record is None:
        return False, "not_found"

    if old_name == new_name:
        return True, ""

    if os.path.exists(_robot_record_path(new_name)):
        return False, "name_taken"

    record["name"] = new_name
    _atomic_write_text(_robot_record_path(new_name), json.dumps(record, indent=2))
    os.remove(_robot_record_path(old_name))
    logger.info(f"Renamed robot record {old_name} -> {new_name}")
    return True, ""


def is_robot_record_clean(record: dict) -> bool:
    """
    A record is 'clean' when every operational field for its mode is populated AND
    every referenced calibration file exists on disk. Cameras are optional.

    - single   : the leader/follower pair (4 fields, 2 calibration files).
    - bimanual : that pair (= left arm) plus the right pair (8 fields, 4 files).
    """
    if not record:
        return False

    # Config fields are stems; the file on disk is "<stem>.json". Tolerate a
    # stored value that still carries the extension (defensive).
    def _file_for(base: str, name: str) -> str:
        stem = name[: -len(".json")] if name.endswith(".json") else name
        return os.path.join(base, f"{stem}.json")

    bimanual = record.get("mode") == "bimanual"
    required_fields = _SINGLE_CONFIG_FIELDS + (_BIMANUAL_CONFIG_FIELDS if bimanual else ())
    for field in required_fields:
        value = record.get(field, "")
        if not isinstance(value, str) or not value.strip():
            return False

    config_files = [
        _file_for(LEADER_CONFIG_PATH, record["leader_config"]),
        _file_for(FOLLOWER_CONFIG_PATH, record["follower_config"]),
    ]
    if bimanual:
        config_files += [
            _file_for(LEADER_CONFIG_PATH, record["right_leader_config"]),
            _file_for(FOLLOWER_CONFIG_PATH, record["right_follower_config"]),
        ]
    return all(os.path.exists(p) for p in config_files)


def config_slot_conflict(record: dict) -> str | None:
    """
    Detect when a bimanual record points two same-side arms at the SAME config.

    The two leader slots share the so_leader dir and the two follower slots share
    so_follower, so an identical config name on both = one physical arm's
    calibration on two arms (at least one is wrong). Returns "leader"/"follower"
    for the offending side, or None. Single mode (one slot per side) never
    conflicts. A leader and follower sharing a name is fine — different dirs.
    """
    if record.get("mode") != "bimanual":
        return None
    leader = record.get("leader_config", "")
    if leader and leader == record.get("right_leader_config", ""):
        return "leader"
    follower = record.get("follower_config", "")
    if follower and follower == record.get("right_follower_config", ""):
        return "follower"
    return None


# Port fields per mode. Unlike configs (which may legitimately share a name
# across leader/follower dirs), a serial PORT is one physical USB device, so
# every arm's port must be distinct — across BOTH sides.
_SINGLE_PORT_FIELDS = ("leader_port", "follower_port")
_BIMANUAL_PORT_FIELDS = ("right_leader_port", "right_follower_port")


def _stage_one_side(
    library_dir: str, staging_dir: str, base: str, left_stem: str, right_stem: str, side: str
):
    """Copy one device's two library calibrations into its staging dir as
    "<base>_left.json"/"<base>_right.json". Overwrites unconditionally so a
    recalibrated library file always refreshes the staging alias. Raises a
    clear, user-facing FileNotFoundError naming the slot and file when a
    referenced library file is missing (before lerobot's connect() can fall into
    interactive recalibration, which hangs the headless thread)."""
    os.makedirs(staging_dir, exist_ok=True)
    for arm, stem in (("left", left_stem), ("right", right_stem)):
        slot = f"{arm} {side}"
        _require_assigned_config(stem, slot)
        stem = stem[: -len(".json")] if stem.endswith(".json") else stem
        src = os.path.join(library_dir, f"{stem}.json")
        if not os.path.exists(src):
            raise FileNotFoundError(
                f"The {slot} arm's calibration file '{stem}.json' was not found in "
                f"{library_dir}. Calibrate that arm (or assign a saved calibration) "
                "before starting."
            )
        dst = os.path.join(staging_dir, f"{base}_{arm}.json")
        shutil.copy2(src, dst)
        logger.info(f"Staged {slot} calibration {src} -> {dst}")


def bimanual_base_id(robot_name: str | None) -> str:
    """Filesystem-safe, stable BiSO staging base id from a robot record name.

    The robot name (already validated by is_valid_robot_name when set via the
    record API) is ideal — one staging dir per robot. Blank or unsafe names fall
    back to DEFAULT_BIMANUAL_BASE so a start request can never produce an unsafe
    path or hang; the copy is unconditional so reuse of the fallback dir is safe.
    """
    name = (robot_name or "").strip()
    if name and is_valid_robot_name(name):
        return name
    return DEFAULT_BIMANUAL_BASE


# BiSO staging dir layout, shared so leader/follower stagers agree on paths.
# lerobot's BiSO devices load each sub-arm's calibration as "<base>_left.json"
# /"<base>_right.json" from a single calibration_dir, so MakerLab's arbitrary
# library names can't be pointed at left/right directly — hence per-device
# staging dirs under MAKERLAB_BISO_STAGING_PATH/<base>/{leader,follower}/.
def _bimanual_leader_staging_dir(base: str) -> str:
    return os.path.join(MAKERLAB_BISO_STAGING_PATH, base, "leader")


def _bimanual_follower_staging_dir(base: str) -> str:
    return os.path.join(MAKERLAB_BISO_STAGING_PATH, base, "follower")


def stage_bimanual_calibrations(
    base: str,
    leader_left: str,
    leader_right: str,
    follower_left: str,
    follower_right: str,
) -> tuple[str, str, str]:
    """Stage the four arbitrarily-named library calibrations for a BiSO session.

    Copies the four selected library files into per-device staging dirs named to
    match lerobot's "<base>_left/right.json" convention, and returns the leader
    staging dir, the follower staging dir, and the base id for building
    BiSO*Config(id=base, calibration_dir=<staging dir>). The copy OVERWRITES
    unconditionally every call so a recalibrated library file refreshes its stale
    staging alias. Any missing library file fails fast with a clear per-slot
    error BEFORE lerobot's connect() (an absent calibration makes lerobot fall
    into interactive recalibration, which hangs the headless thread).

    Both sides are staged; use stage_bimanual_follower_calibrations for flows
    (inference) that drive followers only.
    """
    leader_staging = _bimanual_leader_staging_dir(base)
    follower_staging = _bimanual_follower_staging_dir(base)
    _stage_one_side(LEADER_CONFIG_PATH, leader_staging, base, leader_left, leader_right, "leader")
    _stage_one_side(FOLLOWER_CONFIG_PATH, follower_staging, base, follower_left, follower_right, "follower")
    return leader_staging, follower_staging, base


def stage_bimanual_follower_calibrations(
    base: str,
    follower_left: str,
    follower_right: str,
) -> tuple[str, str]:
    """Stage only the two follower calibrations for a follower-only BiSO session.

    Inference has no leader arms, so staging (and thus requiring) leader library
    files would fail spuriously — the leader library dir need not even contain a
    file matching the follower's name. Produces the identical follower staging
    layout as stage_bimanual_calibrations (same dir, same "<base>_left/right.json"
    names, same overwrite/fail-fast semantics) and returns (follower_staging_dir,
    base).
    """
    follower_staging = _bimanual_follower_staging_dir(base)
    _stage_one_side(FOLLOWER_CONFIG_PATH, follower_staging, base, follower_left, follower_right, "follower")
    return follower_staging, base


def port_slot_conflict(record: dict) -> str | None:
    """
    Return a serial port assigned to more than one arm of this robot, or None.

    Two physical arms can't share a port, so all of a robot's ports must differ —
    leader vs follower in single mode, and all four in bimanual mode. Empty ports
    are ignored (not yet set).
    """
    fields = _SINGLE_PORT_FIELDS + (_BIMANUAL_PORT_FIELDS if record.get("mode") == "bimanual" else ())
    seen: set[str] = set()
    for field in fields:
        port = record.get(field, "")
        if not isinstance(port, str) or not port.strip():
            continue
        if port in seen:
            return port
        seen.add(port)
    return None


# ---------------------------------------------------------------------------
# Dismissed hub jobs
# ---------------------------------------------------------------------------


def get_dismissed_hub_jobs() -> set[str]:
    """Return the set of hub-job ids the user dismissed from the jobs UI.

    A missing or corrupted file yields the empty set — dismissal is cosmetic,
    so it must never block the hub listing.
    """
    if not os.path.exists(DISMISSED_HUB_JOBS_FILE):
        return set()
    try:
        with open(DISMISSED_HUB_JOBS_FILE) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to read dismissed hub jobs: {e}")
        return set()
    if not isinstance(data, list):
        return set()
    return {j for j in data if isinstance(j, str) and j.strip()}


def add_dismissed_hub_job(job_id: str) -> bool:
    """Persist a hub-job id as dismissed. Returns False for a blank id.

    Idempotent — re-dismissing an already-dismissed id is a no-op success.
    """
    job_id = (job_id or "").strip()
    if not job_id:
        return False
    dismissed = get_dismissed_hub_jobs()
    if job_id not in dismissed:
        dismissed.add(job_id)
        _atomic_write_text(DISMISSED_HUB_JOBS_FILE, json.dumps(sorted(dismissed), indent=2))
        logger.info(f"Dismissed hub job {job_id}")
    return True


def prune_dismissed_hub_jobs(live_job_ids: set[str]) -> None:
    """Drop dismissed ids that no longer appear in the Hub listing, so the file
    only tracks ids that still need hiding.

    Call only with a listing that actually succeeded — pruning against a failed
    (empty) fetch would forget every dismissal.
    """
    dismissed = get_dismissed_hub_jobs()
    kept = dismissed & live_job_ids
    if kept != dismissed:
        _atomic_write_text(DISMISSED_HUB_JOBS_FILE, json.dumps(sorted(kept), indent=2))


# ---------------------------------------------------------------------------
# Calibration config import
# ---------------------------------------------------------------------------

# A lerobot motor calibration entry has exactly these integer fields.
_CALIBRATION_MOTOR_FIELDS = ("id", "drive_mode", "homing_offset", "range_min", "range_max")


def calibration_dir_for_device(device_type: str) -> str | None:
    """Map an API device_type ("teleop"/"robot") to its calibration dir, or None."""
    if device_type == "robot":
        return FOLLOWER_CONFIG_PATH
    if device_type == "teleop":
        return LEADER_CONFIG_PATH
    return None


# A dataset id is either a bare "name" or "namespace/name" (exactly one slash).
# Each segment is an HF-style path component: 1-96 chars of [A-Za-z0-9._-] that
# starts and ends with an alphanumeric. We REJECT bad names (rather than silently
# sanitize) so e.g. "whoo/" fails loudly at the source instead of smuggling in a
# namespace and landing the dataset in a surprising path like "user/whoo/".
_DATASET_SEGMENT_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?$")


def validate_dataset_name(name: object) -> tuple[bool, str]:
    """Validate ONE dataset repo-id segment (the user-typed name, or a namespace).

    Returns (ok, human_readable_reason).
    """
    if not isinstance(name, str) or not name.strip():
        return False, "Dataset name can't be empty."
    if name != name.strip():
        return False, "Dataset name can't have leading or trailing spaces."
    if "/" in name or "\\" in name:
        return False, "Dataset name can't contain slashes."
    if name in (".", ".."):
        return False, "Dataset name can't be '.' or '..'."
    if len(name) > 96:
        return False, "Dataset name is too long (max 96 characters)."
    if not _DATASET_SEGMENT_RE.match(name):
        return False, (
            "Dataset name may only use letters, digits, '.', '_' and '-', and must "
            "start and end with a letter or digit."
        )
    return True, ""


def validate_dataset_repo_id(repo_id: object) -> tuple[bool, str]:
    """Validate a full dataset id: a bare name, or 'namespace/name' (one slash).

    Returns (ok, human_readable_reason). Used by both recording and merge so a bad
    name is refused at the point of creation, not silently rewritten.
    """
    if not isinstance(repo_id, str) or not repo_id.strip():
        return False, "Dataset name can't be empty."
    parts = repo_id.split("/")
    if len(parts) > 2:
        return False, "Dataset name may contain at most one '/' (namespace/name)."
    if len(parts) == 2:
        ns_ok, ns_reason = validate_dataset_name(parts[0])
        if not ns_ok:
            return False, ns_reason.replace("Dataset name", "Namespace")
        return validate_dataset_name(parts[1])
    return validate_dataset_name(parts[0])


def validate_calibration_data(data: object) -> tuple[bool, str]:
    """
    Check that `data` looks like a lerobot motor calibration: a non-empty dict of
    motor_name -> {id, drive_mode, homing_offset, range_min, range_max} with
    integer values. Returns (ok, human_readable_reason). Validating here means a
    bad import fails loudly at upload instead of later inside teleop/calibration.
    """
    if not isinstance(data, dict) or not data:
        return False, "Calibration must be a non-empty object of motors."
    for motor, fields in data.items():
        if not isinstance(fields, dict):
            return False, f"Motor '{motor}' must be an object."
        for key in _CALIBRATION_MOTOR_FIELDS:
            if key not in fields:
                return False, f"Motor '{motor}' is missing '{key}'."
            value = fields[key]
            # bool is a subclass of int; a JSON true/false here is not valid.
            if not isinstance(value, int) or isinstance(value, bool):
                return False, f"Motor '{motor}' field '{key}' must be an integer."
    return True, ""


def save_imported_calibration(device_type: str, name: str, data: object) -> tuple[bool, str, str]:
    """
    Validate and persist an uploaded calibration as <name>.json under the side's
    config dir. Never overwrites an existing file. Returns (ok, reason, name)
    where `name` is the normalized config name (extension stripped). Reason codes:
    "invalid_device", "invalid_name", "invalid_data:<msg>", "name_taken", "".
    """
    config_path = calibration_dir_for_device(device_type)
    if config_path is None:
        return False, "invalid_device", ""

    name = name.strip()
    # Accept either a stem or a "<name>.json" filename (records carry the ext).
    if name.endswith(".json"):
        name = name[: -len(".json")]
    if not is_valid_robot_name(name):
        return False, "invalid_name", name

    ok, msg = validate_calibration_data(data)
    if not ok:
        return False, f"invalid_data:{msg}", name

    os.makedirs(config_path, exist_ok=True)
    file_path = os.path.join(config_path, f"{name}.json")
    if os.path.exists(file_path):
        return False, "name_taken", name

    _atomic_write_text(file_path, json.dumps(data, indent=2))
    logger.info(f"Imported calibration {device_type}/{name}")
    return True, "", name


def rename_calibration_config(device_type: str, old_name: str, new_name: str) -> tuple[bool, str]:
    """
    Rename a calibration config file within a side's dir. Never overwrites an
    existing target. Robot records that referenced the old name (on this side)
    are repointed to the new name so they stay valid. Returns (ok, reason):
    "invalid_device", "invalid_name", "not_found", "name_taken", "".
    """
    config_path = calibration_dir_for_device(device_type)
    if config_path is None:
        return False, "invalid_device"

    old_stem = old_name[: -len(".json")] if old_name.endswith(".json") else old_name
    new_stem = new_name.strip()
    if new_stem.endswith(".json"):
        new_stem = new_stem[: -len(".json")]
    if not is_valid_robot_name(old_stem) or not is_valid_robot_name(new_stem):
        return False, "invalid_name"

    old_path = os.path.join(config_path, f"{old_stem}.json")
    if not os.path.exists(old_path):
        return False, "not_found"
    if old_stem == new_stem:
        return True, ""  # no-op

    new_path = os.path.join(config_path, f"{new_stem}.json")
    if os.path.exists(new_path):
        return False, "name_taken"

    os.rename(old_path, new_path)

    # Repoint any robot records that used the old config on this side — both the
    # primary/left slot and the bimanual right slot live in the same dir.
    fields = (
        ("leader_config", "right_leader_config")
        if device_type == "teleop"
        else ("follower_config", "right_follower_config")
    )
    for rec in list_robot_records():
        patch = {f: new_stem for f in fields if rec.get(f) == old_stem}
        if patch:
            save_robot_record(rec["name"], patch, allow_create=False)

    logger.info(f"Renamed calibration {device_type}/{old_stem} -> {new_stem}")
    return True, ""


def clear_config_references(device_type: str, config_name: str) -> list[dict]:
    """Blank every robot-record field (on this side) that references this
    calibration config, across ALL robot records — both the primary/left slot
    and the bimanual right slot, regardless of mode. A stale right_* reference
    in a single-mode record is cleared too: it points at a file that no longer
    exists, so leaving it would resurface a dangling name on a mode switch.

    Called when a calibration config is deleted: instead of refusing the
    delete, the referencing arms are unassigned and return to the "needs
    calibration" state (is_robot_record_clean → False, and teleop/record refuse
    to start with a clear message until the arm is recalibrated or reassigned).

    Returns [{"robot": <name>, "fields": [<cleared fields>]}] for each record
    modified, so callers can tell the user which arms now need calibration.
    """
    fields = (
        ("leader_config", "right_leader_config")
        if device_type == "teleop"
        else ("follower_config", "right_follower_config")
    )
    stem = config_name.removesuffix(".json")
    cleared: list[dict] = []
    for rec in list_robot_records():
        hit = [f for f in fields if rec.get(f) == stem]
        if hit:
            save_robot_record(rec["name"], dict.fromkeys(hit, ""), allow_create=False)
            cleared.append({"robot": rec["name"], "fields": hit})
    return cleared
