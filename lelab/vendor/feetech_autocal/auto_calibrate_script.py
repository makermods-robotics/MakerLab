#!/usr/bin/env python

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
Feetech STS servo auto-calibration (with unfold).

Full flow (single command):
  Stage 0  Initialization: stop all servos, Lock=1, configure PID/acceleration, enable torque
  Stage 2  Unfold joints 2-4 (skippable via --unfold-angle 0)
  Stage 3  Calibrate servos 2-6 (5 -> 6 -> 4 -> 3 -> 2)
  Stage 4  Calibrate servo 1 shoulder_pan last and return to mid
  Stage 5  Wait for user confirmation, then release torque

Usage examples:

  lerobot-auto-calibrate-feetech --port COM3
  lerobot-auto-calibrate-feetech --port COM3 --save
  lerobot-auto-calibrate-feetech --port COM3 --unfold-angle 0
  lerobot-auto-calibrate-feetech --port COM3 --save --robot-id default
  lerobot-auto-calibrate-feetech --port COM3 --unfold-only   # debug arm-unfold only (Stage 0 + Stage 2)
"""

import argparse
import signal
import sys
import time
from collections.abc import Callable

import draccus
from lerobot.motors import MotorCalibration
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION

# Vendored auto-calibration: upstream FeetechMotorsBus + the calibration mixin.
from .bus import COMM_ERR, AutoCalBus as FeetechMotorsBus
from .calibration_defaults import (
    CALIBRATE_FIRST,
    CALIBRATE_REST,
    DEFAULT_ACCELERATION,
    FULL_TURN,
    DEFAULT_D_COEFFICIENT,
    DEFAULT_I_COEFFICIENT,
    DEFAULT_MAX_TORQUE,
    DEFAULT_P_COEFFICIENT,
    DEFAULT_POS_SPEED,
    DEFAULT_TIMEOUT,
    DEFAULT_TORQUE_LIMIT,
    DEFAULT_UNFOLD_ANGLE,
    DEFAULT_UNFOLD_TIMEOUT,
    DEFAULT_VELOCITY_LIMIT,
    HOMING_OFFSET_MAX_MAG,
    MOTOR_NAMES,
    SO_FOLLOWER_MOTORS,
    STS_HALF_TURN_RAW,
    UNFOLD_ORDER,
    motor_label,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feetech servo auto-calibration (with unfold): full flow in one command."
    )
    parser.add_argument(
        "--port", type=str, required=True,
        help="Serial port path, e.g. COM3 or /dev/ttyUSB0",
    )
    parser.add_argument(
        "--motor", type=str, choices=MOTOR_NAMES, default=None,
        help="Test only this servo (skip unfold); if not specified, test all 6 servos in order",
    )

    cal = parser.add_argument_group("Calibration parameters")
    cal.add_argument(
        "--velocity-limit", type=int, default=DEFAULT_VELOCITY_LIMIT,
        help=f"Calibration limit-probing speed (constant-speed mode Goal_Velocity), default {DEFAULT_VELOCITY_LIMIT}",
    )
    cal.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT,
        help=f"Calibration single-direction limit wait timeout (seconds), default {DEFAULT_TIMEOUT}",
    )
    unfold = parser.add_argument_group("Unfold parameters")
    unfold.add_argument(
        "--unfold-only", action="store_true",
        help="Run only the arm unfold (Stage 0 init + Stage 2 unfold), no calibration; for debugging unfold logic",
    )
    unfold.add_argument(
        "--unfold-angle", type=float, default=DEFAULT_UNFOLD_ANGLE,
        help=f"Unfold angle (degrees); set to 0 to skip unfold. Default {DEFAULT_UNFOLD_ANGLE}",
    )
    unfold.add_argument(
        "--unfold-timeout", type=float, default=DEFAULT_UNFOLD_TIMEOUT,
        help=f"Per-motion timeout for unfold (seconds), default {DEFAULT_UNFOLD_TIMEOUT}",
    )
    out = parser.add_argument_group("Output (same path and format as manual calibration)")
    out.add_argument(
        "--save", action="store_true",
        help="Write calibration data to servo EEPROM and save to the same local path as manual calibration (draccus format)",
    )
    out.add_argument(
        "--robot-id", type=str, default="default",
        help="Robot id used in the saved filename; path is .../calibration/robots/<robot_type>/<robot_id>.json. Must match config.id used when starting the arm.",
    )
    out.add_argument(
        "--robot-type", type=str, default="so_follower",
        choices=["so_follower", "so_leader"],
        help="Robot type for calibration file path: 'so_follower' (default) or 'so_leader'",
    )

    return parser.parse_args()


# ====================== Unfold-related ======================

def _unfold_joints(
    bus: FeetechMotorsBus,
    unfold_angle: float,
    unfold_timeout: float,
    unfold_directions: dict[str, str | None] | None = None,
) -> None:
    """Unfold joints 2-4 to avoid mechanical interference during calibration.
    If unfold_directions is provided, record each joint's unfold direction."""
    print(f"\n{'='*20} Stage 2: Unfold joints 2-4 ({unfold_angle}°) {'='*20}")
    for motor in UNFOLD_ORDER:
        direction, _ = bus.unfold_single_joint(motor, unfold_angle, unfold_timeout)
        if unfold_directions is not None and direction is not None:
            unfold_directions[motor] = direction
    print("\n  Unfold complete; joints 2-4 are lifted. Per-joint unfold directions:")
    if unfold_directions is not None:
        for motor in UNFOLD_ORDER:
            direction = unfold_directions.get(motor, "unknown")
            print(f"    {motor_label(motor)}: unfold direction = {direction}")


def _fold_arm(
    bus: FeetechMotorsBus,
    all_mins: dict[str, int],
    all_maxes: dict[str, int],
    all_unfold_directions: dict[str, str | None],
    *,
    motors: list[str] | None = None,
    unfold: bool = False,
    unfold_per_motor: dict[str, bool] | None = None,
) -> None:
    """Fold the specified joints, or fully unfold them. Multiple servos move simultaneously.

    Fold (unfold=False): forward-unfold -> fold target = range_max; reverse -> range_min;
    gripper fixed at range_min.
    Unfold (unfold=True): targets are reversed; forward -> range_min, reverse -> range_max;
    gripper fixed at range_max.
    motors: list of servos to move; if None or empty, use the default order
    (shoulder_lift -> elbow_flex -> wrist_flex -> gripper).
    unfold_per_motor: optional, per-joint fold(False)/unfold(True); joints not listed use 'unfold'.
    If None, all use 'unfold'.
    """
    default_order = ["shoulder_lift", "elbow_flex", "wrist_flex", "gripper"]
    fold_order = default_order if not motors else motors
    title = "Fold/Unfold arm" if unfold_per_motor else (("Unfold" if unfold else "Fold") + " arm")
    print(f"\n{'='*20} {title} (simultaneous) {'='*20}")
    values: dict[str, tuple[int, int, int]] = {}

    for motor in fold_order:
        if motor not in all_mins or motor not in all_maxes:
            continue
        per_unfold = unfold_per_motor.get(motor, unfold) if unfold_per_motor is not None else unfold
        direction = all_unfold_directions.get(motor)
        # First compute fold-end and unfold-end, then pick one based on per_unfold
        if motor == "gripper":
            fold_end = all_mins[motor]
            unfold_end = all_maxes[motor]
        else:
            fold_end = all_maxes[motor] if direction == "reverse" else all_mins[motor]
            unfold_end = all_mins[motor] if direction == "reverse" else all_maxes[motor]
        target = unfold_end if per_unfold else fold_end
        label = "range_max" if target == all_maxes[motor] else "range_min"
        if motor == "gripper":
            label += "(gripper)" if per_unfold else "(gripper forward)"
        action_m = "Unfold" if per_unfold else "Fold"
        values[motor] = (target, DEFAULT_POS_SPEED, DEFAULT_ACCELERATION)
        bus.write("Operating_Mode", motor, 0)  # servo mode
        try:
            pos = bus.read("Present_Position", motor, normalize=False)
            print(f"  {motor_label(motor)} current pos={pos}, {action_m} to {label}={target}.")
        except COMM_ERR:
            print(f"  {motor_label(motor)} failed to read current pos, {action_m} to {label}={target}.")
    if not values:
        action = "Unfold" if unfold else "Fold"
        print(f"  No valid servos, skipping {action}.\n")
        return
    bus.sync_write_pos_ex(values)
    time.sleep(0.3)
    # Poll until all stopped
    timeout_s = 10.0
    poll_s = 0.05
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        try:
            if all(bus.read("Moving", m, normalize=False) == 0 for m in values):
                break
        except COMM_ERR:
            pass
        time.sleep(poll_s)
    done_label = "Fold/Unfold" if unfold_per_motor else ("Unfold" if unfold else "Fold")
    for m in values:
        try:
            pos = bus.read("Present_Position", m, normalize=False)
            print(f"  {motor_label(m)} after {done_label}: end pos={pos}, arrived")
        except COMM_ERR:
            print(f"  {motor_label(m)} after {done_label}: failed to read pos, arrived")
    print(f"  {done_label} complete.\n")


def _move_arm_by_angle(
    bus: FeetechMotorsBus,
    all_unfold_directions: dict[str, str | None],
    angle_deg: float,
    *,
    fold: bool = False,
    motors: list[str] | None = None,
    all_mins: dict[str, int] | None = None,
    all_maxes: dict[str, int] | None = None,
) -> None:
    """Move the given joints by the specified degrees relative to current position, in either the
    unfold or the fold direction. Does not probe direction; relies on all_unfold_directions.

    Direction matches _fold_arm: forward-unfold -> position increases for unfold, decreases for fold;
    reverse-unfold -> the opposite.
    fold: False = unfold direction, True = fold direction.
    motors: list of servos to move; if None or empty, uses the default order
    (shoulder_lift -> elbow_flex -> wrist_flex).
    all_mins / all_maxes: optional; if provided, the target position is clamped to them.
    """
    default_order = ["shoulder_lift", "elbow_flex", "wrist_flex"]
    move_order = default_order if not motors else motors
    angle_steps = int(angle_deg / 360.0 * FULL_TURN)
    direction_label = "fold" if fold else "unfold"
    print(f"\n{'='*20} Relative {direction_label} {angle_deg:.1f}° from current pos {'='*20}")
    for motor in move_order:
        if all_mins is not None and all_maxes is not None and (motor not in all_mins or motor not in all_maxes):
            continue
        try:
            present = bus.read("Present_Position", motor, normalize=False)
        except COMM_ERR:
            print(f"  Warning: {motor_label(motor)} failed to read current pos, skipping")
            continue
        direction = all_unfold_directions.get(motor)
        # Matches _fold_arm: forward -> unfold increases pos, fold decreases; reverse -> opposite
        if fold:
            target = present - angle_steps if direction == "forward" else present + angle_steps
        else:
            target = present + angle_steps if direction == "forward" else present - angle_steps
        if all_mins is not None and all_maxes is not None and motor in all_mins and motor in all_maxes:
            target = max(all_mins[motor], min(all_maxes[motor], target))
        print(f"  {motor_label(motor)} {direction_label} {angle_deg:.1f}°: pos {present} -> {target}")
        ok = bus.write_pos_ex_and_wait(
            motor, target, DEFAULT_POS_SPEED, DEFAULT_ACCELERATION,
            timeout_s=DEFAULT_UNFOLD_TIMEOUT, poll_interval_s=0.05,
        )
        if not ok:
            print(f"  Warning: {motor_label(motor)} motion timeout, holding current position")
        else:
            print(f"  {motor_label(motor)} arrived")
    print(f"  {direction_label} complete.\n")


# ====================== Calibration-related ======================


def _record_reference_position(
    bus: FeetechMotorsBus,
    motor_name: str,
    out: dict[str, int],
) -> None:
    """Read the servo's current reference position (Present_Position + Homing_Offset) % FULL_TURN
    and write it to out[motor_name]; on read failure, leave out unchanged."""
    try:
        pr = bus.read("Present_Position", motor_name, normalize=False)
        ho = bus.read("Homing_Offset", motor_name, normalize=False)
        out[motor_name] = (pr + ho) % FULL_TURN
    except COMM_ERR:
        pass


def _calibrate_motors(
    bus: FeetechMotorsBus,
    motor_names: list[str],
    *,
    velocity_limit: int = DEFAULT_VELOCITY_LIMIT,
    timeout_s: float = DEFAULT_TIMEOUT,
    ccw_first: bool = False,
    unfold_directions: dict[str, str | None] | None = None,
    reference_positions: dict[str, int] | None = None,
) -> dict[str, tuple[int, int, int]]:
    """Calibrate a group of servos (uniformly via measure_ranges_of_motion_multi, then write back
    and return to mid). Returns {motor_name: (range_min, range_max, mid_raw)}.
    If unfold_directions is provided and both servos 2 and 3 are calibrated together: both move
    together; servo 2 fully folds, servo 3 fully unfolds along its unfold direction. Otherwise
    follow the original return-to-mid logic.
    If reference_positions is provided: for those servos, use the reference position to pick the
    arc during calibration, skipping limit back-off and re-read."""
    if not motor_names:
        return {}
    raw_results = bus.measure_ranges_of_motion_multi(
        motor_names,
        velocity_limit=velocity_limit,
        timeout_s=timeout_s,
        ccw_first=ccw_first,
        reference_positions=reference_positions,
    )
    print("Preparing to write registers")
    result: dict[str, tuple[int, int, int]] = {}
    for m in motor_names:
        rmin, rmax, mid_raw, _raw_min_meas, _raw_max_meas, homing_offset = raw_results[m]
        print(f"  {motor_label(m)}: post-offset range_min={rmin}, range_max={rmax}, mid={mid_raw}, Homing_Offset register={homing_offset}")
        time.sleep(0.05)
        try:
            ho_before = bus.read("Homing_Offset", m, normalize=False)
            min_before = bus.read("Min_Position_Limit", m, normalize=False)
            max_before = bus.read("Max_Position_Limit", m, normalize=False)
            print(f"  {motor_label(m)} pre-write: Min_Position_Limit={min_before}, Max_Position_Limit={max_before}, Homing_Offset={ho_before}")
        except COMM_ERR:
            print(f"  {motor_label(m)} pre-write: register read failed")
        bus.safe_write("Homing_Offset", m, homing_offset, normalize=False)
        bus.safe_write_position_limits(m, rmin, rmax)
        time.sleep(0.1)
        try:
            ho_after = bus.read("Homing_Offset", m, normalize=False)
            min_after = bus.read("Min_Position_Limit", m, normalize=False)
            max_after = bus.read("Max_Position_Limit", m, normalize=False)
            print(f"  {motor_label(m)} post-write: Min_Position_Limit={min_after}, Max_Position_Limit={max_after}, Homing_Offset={ho_after}")
        except COMM_ERR:
            print(f"  {motor_label(m)} post-write: register read failed")
        time.sleep(0.1)
        do_2_3_together = (
            unfold_directions is not None
            and "shoulder_lift" in motor_names
            and "elbow_flex" in motor_names
        )
        if m == "wrist_roll":
            pass
        elif do_2_3_together and m in ("shoulder_lift", "elbow_flex"):
            # Servos 2 and 3 lock torque
            pass
        else:
            bus.go_to_mid(m)
        result[m] = (rmin, rmax, mid_raw)

    return result


# ====================== Connect and init (shared) ======================

def _connect_and_clear(port: str) -> FeetechMotorsBus:
    """Create the bus, clear residual Overload, then connect for real. Raises on failure."""
    bus = FeetechMotorsBus(port=port, motors=SO_FOLLOWER_MOTORS.copy())
    bus.connect(handshake=False)
    print("Clearing residual servo state...")
    all_zero = {m: 0 for m in MOTOR_NAMES}
    for _ in range(3):
        try:
            bus.sync_write("Goal_Velocity", all_zero)
        except COMM_ERR:
            pass
        try:
            bus.sync_write("Torque_Enable", all_zero)
        except COMM_ERR:
            pass
        time.sleep(0.2)
    bus.disconnect(disable_torque=False)
    time.sleep(0.2)
    bus.connect()
    print("All servos ready.")
    return bus


def _run_with_bus(
    port: str,
    interactive: bool,
    body: Callable[[FeetechMotorsBus], None],
) -> int:
    """After connecting the bus, run body(bus); uniformly handle connect failures, KeyboardInterrupt,
    Exception, and disconnect. Returns 0 on success, 1 on error, 130 on user interrupt."""
    try:
        bus = _connect_and_clear(port)
    except Exception as e:
        print(f"Connect failed: {e}", file=sys.stderr)
        return 1
    try:
        body(bus)
    except KeyboardInterrupt:
        print("\nUser interrupt; releasing all servos...")
        bus.safe_disable_all()
        return 130
    except Exception as e:
        print(f"Exception: {e}", file=sys.stderr)
        bus.safe_disable_all()
        if interactive:
            try:
                input("Press Enter to exit...")
            except EOFError:
                pass
        return 1
    finally:
        bus.disconnect()
    return 0


# Stage 0 init: per-register write -> read -> compare, table-driven; special items
# (limits, Torque_Enable) handled separately
INIT_CHECKS = [
    ("Lock", 1),
    ("Return_Delay_Time", 0),
    ("Operating_Mode", 0),
    ("Max_Torque_Limit", DEFAULT_MAX_TORQUE),
    ("Torque_Limit", DEFAULT_TORQUE_LIMIT),
    ("Acceleration", DEFAULT_ACCELERATION),
    ("P_Coefficient", DEFAULT_P_COEFFICIENT),
    ("I_Coefficient", DEFAULT_I_COEFFICIENT),
    ("D_Coefficient", DEFAULT_D_COEFFICIENT),
    ("Homing_Offset", 0),
]


def _run_init(bus: FeetechMotorsBus, *, interactive: bool = True) -> None:
    """Stage 0: Lock=1, PID, limits, Homing_Offset, enable torque. On parameter anomaly,
    if interactive, wait for Enter."""
    print(f"\n{'='*20} Stage 0: Initialization {'='*20}")
    for m in MOTOR_NAMES:
        print(f"Configuring servo: {motor_label(m)}")
        try:
            bus.write("Torque_Enable", m, 0)
            time.sleep(0.05)
        except COMM_ERR:
            pass
        param_set_ok = True
        try:
            for reg, expected in INIT_CHECKS:
                bus.write(reg, m, expected, normalize=(reg != "Homing_Offset"))
                time.sleep(0.01)
                got = bus.read(reg, m, normalize=False)
                if got != expected:
                    print(f"  [Warning] {reg} setting failed on {m}: set value={expected}, read value={got}")
                    param_set_ok = False
            # Limits: separate write/read/compare
            bus.write_position_limits(m, 0, 4095)
            time.sleep(0.05)
            limits = bus.read_position_limits(m)
            if limits != (0, 4095):
                print(f"  [Warning] Position_Limits setting failed on {m}: set value=(0, 4095), read value={limits}")
                param_set_ok = False
            time.sleep(0.2)
            # Finally enable torque
            bus.write("Torque_Enable", m, 1)
            time.sleep(0.05)
            te_read = bus.read("Torque_Enable", m, normalize=False)
            if te_read != 1:
                print(f"  [Warning] Torque_Enable enable failed on {m}: set value=1, read value={te_read}")
                param_set_ok = False
            time.sleep(0.1)
        except Exception as e:
            print(f"  [Exception] Error setting parameters on {m}: {e}")
            param_set_ok = False
        if not param_set_ok and interactive:
            try:
                input("  [Warning] Parameter setting/verification has anomalies; check wiring and power, press Enter to force-continue...")
            except Exception:
                pass
    print(
        f"Initialized and torque enabled (P={DEFAULT_P_COEFFICIENT}, "
        f"Acc={DEFAULT_ACCELERATION}, Torque={DEFAULT_TORQUE_LIMIT})."
    )


# ====================== Public entry points (full calibration / unfold-only / single servo) ======================


def _apply_calibration_results(
    results: dict[str, tuple[int, int, int]],
    all_mins: dict[str, int],
    all_maxes: dict[str, int],
    all_mids: dict[str, int],
    motor_list: list[str],
) -> None:
    """Apply _calibrate_motors return values to all_mins / all_maxes / all_mids."""
    for m in motor_list:
        all_mins[m], all_maxes[m], all_mids[m] = results[m]


def run_full_calibration(
    port: str,
    *,
    save: bool = False,
    robot_id: str = "default",
    robot_type: str = "so_follower",
    velocity_limit: int = DEFAULT_VELOCITY_LIMIT,
    timeout_s: float = DEFAULT_TIMEOUT,
    unfold_timeout_s: float = DEFAULT_UNFOLD_TIMEOUT,
    interactive: bool = True,
) -> int:
    """Full calibration flow: init -> servos 2-6 (with arm-lift to avoid obstruction) ->
    servo 1 shoulder_pan calibrated last -> fold.
    If save is True: write to servo EEPROM and save to the same path/format as manual calibration
    (draccus, loaded by the arm at startup).

    For CLI or teleop programs to call. Returns 0 on success, 1 on error, 130 on user interrupt.
    """

    def body(bus: FeetechMotorsBus) -> None:
        all_mins: dict[str, int] = {}
        all_maxes: dict[str, int] = {}
        all_mids: dict[str, int] = {}
        all_unfold_directions: dict[str, str | None] = {}
        all_reference_positions: dict[str, int] = {}
        _run_init(bus, interactive=interactive)
        # Lift servo 4 by 80 degrees
        direction, _ = bus.unfold_single_joint("wrist_flex", 80, unfold_timeout_s)
        if direction is not None:
            all_unfold_directions["wrist_flex"] = direction
        time.sleep(0.1)
        # Lift servos 2 and 3 and record reference positions
        # (Present_Position + Homing_Offset; used to pick the arc during calibration)
        direction, _ = bus.unfold_single_joint("shoulder_lift", 15, unfold_timeout_s)
        if direction is not None:
            all_unfold_directions["shoulder_lift"] = direction
        _record_reference_position(bus, "shoulder_lift", all_reference_positions)
        direction, _ = bus.unfold_single_joint("elbow_flex", 30, unfold_timeout_s)
        if direction is not None:
            all_unfold_directions["elbow_flex"] = direction
        _record_reference_position(bus, "elbow_flex", all_reference_positions)
        time.sleep(0.1)
        # Fold: retract shoulder_lift and elbow_flex
        for m in ["shoulder_lift", "elbow_flex"]:
            bus.go_to_mid(m)
            time.sleep(0.1)
        # Use multi-servo calibration for servos 2 and 3; the first rotation direction is the
        # opposite of each servo's lift direction.
        # forward-lifted -> CCW first; reverse-lifted -> CW first; default is CCW first when not recorded.
        ccw_first_2_3 = {
            "shoulder_lift": all_unfold_directions.get("shoulder_lift") != "reverse",
            "elbow_flex": all_unfold_directions.get("elbow_flex") != "reverse",
        }
        print(f"\n{'='*20} Calibrating servos 2 and 3 (multi-servo, opposite of lift direction) {'='*20}")
        results_2_3 = _calibrate_motors(
            bus, ["shoulder_lift", "elbow_flex"],
            velocity_limit=velocity_limit,
            timeout_s=timeout_s,
            ccw_first=ccw_first_2_3,
            unfold_directions=all_unfold_directions,
            reference_positions=all_reference_positions,
        )
        _apply_calibration_results(results_2_3, all_mins, all_maxes, all_mids, ["shoulder_lift", "elbow_flex"])
        _fold_arm(bus, all_mins, all_maxes, all_unfold_directions, motors=["shoulder_lift", "elbow_flex"])

        time.sleep(0.1)
        # Stage 3: Calibrate the remaining servos 4, 5, 6 (multi-servo simultaneous, with arm lift to avoid obstruction)
        print(f"\n{'='*20} Stage 3: Calibrate servos 4-6 (multi-servo simultaneous) {'='*20}")
        _move_arm_by_angle(bus, all_unfold_directions, 80, fold=False, motors=["elbow_flex"], all_mins=all_mins, all_maxes=all_maxes)
        CALIBRATE_REST_REMAINING = ["wrist_roll", "gripper", "wrist_flex"]
        results_rest = _calibrate_motors(
            bus, CALIBRATE_REST_REMAINING,
            velocity_limit=velocity_limit,
            timeout_s=timeout_s,
            reference_positions=all_reference_positions,
        )
        _apply_calibration_results(results_rest, all_mins, all_maxes, all_mids, CALIBRATE_REST_REMAINING)
        time.sleep(0.1)
        # Fold servo 3, fully unfold servo 4 (executed together in one call)
        _fold_arm(bus, all_mins, all_maxes, all_unfold_directions,
            motors=["elbow_flex", "wrist_flex","gripper"],
            unfold_per_motor={"elbow_flex": False, "wrist_flex": True, "gripper": False})
        # Stage 4: Calibrate servo 1 shoulder_pan last
        print(f"\n{'='*20} Stage 4: Calibrate {motor_label('shoulder_pan')} (servo 1) and return to mid {'='*20}")
        results_pan = _calibrate_motors(
            bus, ["shoulder_pan"], velocity_limit=velocity_limit, timeout_s=timeout_s
        )
        _apply_calibration_results(results_pan, all_mins, all_maxes, all_mids, ["shoulder_pan"])
        time.sleep(0.1)
        motors_calibrated = CALIBRATE_REST + CALIBRATE_FIRST
        print(f"\n{'='*20} Calibration results {'='*20}")
        for name in motors_calibrated:
            offset = all_mids[name] - STS_HALF_TURN_RAW
            print(
                f"  {motor_label(name)}: min={all_mins[name]}, max={all_maxes[name]}, "
                f"mid={all_mids[name]}, offset={offset}"
            )


        _fold_arm(bus, all_mins, all_maxes, all_unfold_directions)
           # Before persistence: unlock EEPROM (Lock=0) and restore all servos to servo mode (Operating_Mode=0)
        for name in bus.motors:
            bus.write("Lock", name, 0)
            time.sleep(0.01)
            bus.write("Operating_Mode", name, 0)
            time.sleep(0.01)
        time.sleep(1)
        if interactive:
            bus.safe_disable_all()
            print("\nCalibration complete.")

        if save:
            print(f"\n{'='*20} Persistence (same scheme as manual calibration) {'='*20}")
            bus.safe_disable_all()
            cal = {}
            for name in motors_calibrated:
                m = SO_FOLLOWER_MOTORS[name]
                offset = all_mids[name] - STS_HALF_TURN_RAW
                offset = max(-HOMING_OFFSET_MAX_MAG, min(HOMING_OFFSET_MAX_MAG, offset))
                cal[name] = MotorCalibration(
                    id=m.id,
                    drive_mode=0,
                    homing_offset=offset,
                    range_min=all_mins[name],
                    range_max=all_maxes[name],
                )
            bus.write_calibration(cal, cache=True)
            print("Wrote calibration to servo EEPROM.")
            # Same path and format as manual calibration; loaded by the arm at startup.
            calibration_fpath = HF_LEROBOT_CALIBRATION / "robots" / robot_type / f"{robot_id}.json"
            calibration_fpath.parent.mkdir(parents=True, exist_ok=True)
            with open(calibration_fpath, "w") as f, draccus.config_type("json"):
                draccus.dump(cal, f, indent=4)
            print(f"Wrote calibration to: {calibration_fpath}")
        print("Releasing all servos...")
        bus.safe_disable_all()

    return _run_with_bus(port, interactive, body)


def unfold_joints(
    port: str,
    angle_deg: float,
    *,
    timeout_s: float = DEFAULT_UNFOLD_TIMEOUT,
    interactive: bool = True,
) -> int:
    """Run only Stage 0 init + unfold joints 2-4 to the given angle. For debugging the unfold.
    Returns 0/1/130."""

    def body(bus: FeetechMotorsBus) -> None:
        _run_init(bus, interactive=interactive)
        all_unfold_directions: dict[str, str | None] = {}
        if angle_deg > 0:
            _unfold_joints(bus, angle_deg, timeout_s, all_unfold_directions)
            print("  Unfold complete; unfold directions:")
            for motor, direction in all_unfold_directions.items():
                print(f"    {motor_label(motor)}: {direction}")
            if interactive:
                input("  Press Enter to release torque and exit...")
        else:
            print("  Unfold angle is 0, skipping unfold.")
            if interactive:
                input("  Press Enter to release torque and exit...")
        bus.safe_disable_all()

    return _run_with_bus(port, interactive, body)


def calibrate_single_motor(
    port: str,
    motor_name: str,
    *,
    velocity_limit: int = DEFAULT_VELOCITY_LIMIT,
    timeout_s: float = DEFAULT_TIMEOUT,
    interactive: bool = True,
) -> int:
    """Run only Stage 0 + calibrate the given servo, no fold, no save. For testing.
    Returns 0/1/130."""

    def body(bus: FeetechMotorsBus) -> None:
        _run_init(bus, interactive=interactive)
        print(f"\n{'='*20} Calibrating {motor_label(motor_name)} {'='*20}")
        _calibrate_motors(bus, [motor_name], velocity_limit=velocity_limit, timeout_s=timeout_s)
        time.sleep(0.1)
        if interactive:
            input("  Calibration complete. Press Enter to release torque and exit...")
        bus.safe_disable_all()

    return _run_with_bus(port, interactive, body)


# ====================== CLI entry ======================

def _handle_sigterm(signum, frame) -> None:
    """LeLab's Stop button terminates this subprocess with SIGTERM; raise
    KeyboardInterrupt so _run_with_bus releases torque (safe_disable_all)
    instead of dying with the arm still energized."""
    raise KeyboardInterrupt


def main() -> int:
    """CLI: based on arguments, invoke full calibration, unfold-only, or single-servo calibration."""
    signal.signal(signal.SIGTERM, _handle_sigterm)
    args = parse_args()
    # LeLab runs this as a subprocess (no TTY). Without a TTY the "press Enter"
    # prompts have no one to answer them, so run non-interactively — every
    # input() is guarded by `interactive` and torque still releases afterwards.
    interactive = sys.stdin.isatty()
    print(f"Serial port: {args.port}")
    if getattr(args, "unfold_only", False):
        print("Arm unfold only (--unfold-only): Stage 0 init + Stage 2 unfold, no calibration")
        print(f"Unfold angle: {args.unfold_angle}°")
        return unfold_joints(
            args.port,
            args.unfold_angle,
            timeout_s=args.unfold_timeout,
            interactive=interactive,
        )
    if args.motor is not None:
        print(f"Single-servo mode: {args.motor}")
        return calibrate_single_motor(
            args.port,
            args.motor,
            velocity_limit=args.velocity_limit,
            timeout_s=args.timeout,
            interactive=interactive,
        )
    print(f"Full calibration: {CALIBRATE_FIRST + CALIBRATE_REST}")
    return run_full_calibration(
        args.port,
        save=args.save,
        robot_id=args.robot_id,
        robot_type=args.robot_type,
        velocity_limit=args.velocity_limit,
        timeout_s=args.timeout,
        unfold_timeout_s=args.unfold_timeout,
        interactive=interactive,
    )


if __name__ == "__main__":
    sys.exit(main())
