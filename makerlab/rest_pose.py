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
"""Return an arm to a captured rest pose before releasing torque.

Stop paths that cut torque instantly drop the arm under gravity. The STS3215
has no torque-control mode for true gravity compensation, so the graceful
alternative is: capture the pose the arm started the session in, and on a
normal stop drive it back there — progress-based (a slow return is never cut
short, only a stuck one gives up), gently, and abortable — then release.

MIRROR NOTE: the vendored auto-calibration script carries its own twin of this
logic (makerlab/vendor/feetech_autocal/auto_calibrate_script.py, "Graceful stop"
section). It runs as a standalone subprocess and cannot import makerlab cleanly,
so the two are kept mirrored by hand — change one, check the other. The
vendored twin additionally re-expresses targets against Homing_Offset (the
calibration rewrites offsets mid-run); THIS module stores raw Present_Position
ticks, which is valid precisely because teleoperation never touches
Homing_Offset mid-session.

Register facts (pinned lerobot, lerobot/motors/feetech/tables.py):
``Goal_Position`` (42), ``Goal_Velocity`` (46, the profile speed in position
mode), ``Present_Position`` (56, read-only), ``Moving`` (66, read-only).
"""

import contextlib
import logging
import threading
import time

logger = logging.getLogger(__name__)

# Gentle return speed (ticks/s-ish register units; teleop snaps run at full).
RETURN_POS_SPEED = 400
RETURN_SETTLE_S = 0.3  # let motion start before polling
RETURN_POLL_S = 0.05
# Stall detection: the return may be slow, never stuck. Give up only when the
# aggregate distance-to-target fails to shrink by more than
# RETURN_STALL_MIN_PROGRESS ticks for a full RETURN_STALL_WINDOW_S window.
RETURN_STALL_WINDOW_S = 1.5
RETURN_STALL_MIN_PROGRESS = 10  # ticks, summed over all motors
# Absolute ceiling, not a working limit: termination is progress-based, so
# this must never matter in practice — it only bounds a pathological loop.
RETURN_CEILING_S = 10.0
# Arrived when every motor is within this many ticks of its target (matches
# the auto-calibration mixin's POSITION_TOLERANCE).
RETURN_ARRIVE_TOLERANCE = 20


def capture_rest_pose(bus) -> dict[str, int]:
    """Raw Present_Position (ticks) of every motor on one bus, or {} on failure.

    Call at session start, before any motion. Raw ticks are directly
    replayable as Goal_Position later because nothing in a teleop/record
    session rewrites Homing_Offset (see module docstring). Never raises —
    a session must not fail to start over an optional nicety.
    """
    try:
        return {m: int(v) for m, v in bus.sync_read("Present_Position", normalize=False).items()}
    except Exception as e:
        port = getattr(bus, "port", None) or "unknown port"
        logger.warning(f"Could not capture the rest pose on {port}: {e}")
        return {}


def return_to_rest_pose(
    bus,
    rest_pose: dict[str, int],
    abort_event: threading.Event | None = None,
    label: str = "arm",
) -> tuple[bool, str]:
    """Drive one bus's motors back to ``rest_pose``, then report how it went.

    Returns ``(arrived, reason)`` with reason starting with one of
    ``returned`` (with per-motor |final - target| deltas), ``settled`` (every
    motor stopped — Moving == 0 — but short of target: a latched or weak
    motor, reported with its deltas), ``stalled``, ``ceiling``, ``cut-short``
    (abort_event set — a second stop or a new session start), ``no-pose``, or
    ``comm-error: ...``. Success requires every motor within
    RETURN_ARRIVE_TOLERANCE ticks of its target — Moving == 0 alone is NOT
    success, precisely because a weak motor can sit motionless far from its
    target and that must not be reported as "returned".

    Torque must still be enabled (call BEFORE force_disable_torque). Targets
    are written as-is: they were captured from this same session's calibration,
    so they are inherently within the arm's limits — no re-expression or
    clamping needed (unlike the auto-calibration twin). Never raises.

    On EVERY exit path the gentle RETURN_POS_SPEED profile cap written into the
    motors' RAM Goal_Velocity is reset to 0 (uncapped) before returning, so it
    can't linger and throttle the next session — the RAM-persistent speed-cap
    hazard that makerlab/motor_power.clear_goal_velocity also guards at session
    start (see _restore_goal_velocity).
    """
    motors = getattr(bus, "motors", None) or {}
    targets = {m: v for m, v in rest_pose.items() if m in motors}
    if not targets:
        logger.info(f"Rest-pose return skipped for the {label}: no captured pose")
        return False, "no-pose"
    try:
        for motor in targets:
            bus.write("Goal_Velocity", motor, RETURN_POS_SPEED, normalize=False)
        bus.sync_write("Goal_Position", targets, normalize=False)
    except Exception as e:
        logger.warning(f"Rest-pose return failed to start for the {label}: {e}")
        # The gentle RETURN_POS_SPEED cap may already be stamped on some motors;
        # clear it so it can't throttle the next session (see _restore_goal_velocity).
        _restore_goal_velocity(bus, targets, label)
        return False, f"comm-error: {e}"

    try:
        return _run_return_loop(bus, targets, abort_event)
    finally:
        # Belt and braces: whatever the outcome (returned / settled / stalled /
        # ceiling / cut-short), we wrote the gentle RETURN_POS_SPEED cap into the
        # motors' RAM Goal_Velocity above. Reset it to 0 before the caller
        # releases torque, so a slow-return speed cap can't linger and throttle
        # the next teleop/record/inference session on this power-up. Best-effort.
        _restore_goal_velocity(bus, targets, label)


def _restore_goal_velocity(bus, targets: dict[str, int], label: str = "arm") -> None:
    """Reset Goal_Velocity to 0 (uncapped) on the motors the return just drove.

    The return writes a gentle RETURN_POS_SPEED profile cap into each motor's
    RAM Goal_Velocity; that value is RAM-persistent across sessions (only a
    power cycle resets it), so leaving it stamped would silently throttle the
    next session's follower moves (the same leftover-cap mechanism the vendored
    auto-cal fold/unfold=1000 and this module's return=400 both create — see
    makerlab/motor_power.clear_goal_velocity, the primary session-start guard).
    Best-effort: a failed reset is logged and ignored — the release must run.
    """
    try:
        bus.sync_write("Goal_Velocity", dict.fromkeys(targets, 0), normalize=False)
    except Exception as e:
        logger.warning(f"Could not reset the return speed cap (Goal_Velocity) for the {label}: {e}")


def _run_return_loop(
    bus,
    targets: dict[str, int],
    abort_event: threading.Event | None,
) -> tuple[bool, str]:
    """The poll-until-arrived loop of return_to_rest_pose (see its docstring).

    Split out so return_to_rest_pose can wrap every exit path in a finally that
    resets the Goal_Velocity cap without duplicating the reset at each return.
    """
    time.sleep(RETURN_SETTLE_S)
    t0 = time.monotonic()
    best_dist: int | None = None
    last_progress_t = t0
    remaining = -1
    distances: dict[str, int] = {}
    while time.monotonic() - t0 < RETURN_CEILING_S:
        if abort_event is not None and abort_event.is_set():
            return False, "cut-short"
        try:
            positions = bus.sync_read("Present_Position", normalize=False)
        except Exception:
            positions = {}
        now = time.monotonic()
        if positions:
            distances = {m: abs(int(positions[m]) - t) for m, t in targets.items() if m in positions}
            remaining = sum(distances.values())
            if distances and all(d <= RETURN_ARRIVE_TOLERANCE for d in distances.values()):
                return True, f"returned: max delta {max(distances.values())} ticks ({_deltas(distances)})"
            if best_dist is None or remaining < best_dist - RETURN_STALL_MIN_PROGRESS:
                best_dist = remaining
                last_progress_t = now
        if now - last_progress_t > RETURN_STALL_WINDOW_S:
            detail = f"remaining {remaining} ticks ({_deltas(distances)})"
            # Distinguish a motor still fighting (stalled) from one that gave
            # up and sits motionless short of target (settled: a latched or
            # weak motor) — a one-shot Moving read, only on this exit path.
            with contextlib.suppress(Exception):
                moving = bus.sync_read("Moving", normalize=False)
                if all(moving.get(m, 1) == 0 for m in targets):
                    return False, f"settled short of target after {now - t0:.1f}s, {detail}"
            return False, f"stalled after {now - t0:.1f}s, {detail}"
        time.sleep(RETURN_POLL_S)
    return False, f"ceiling ({RETURN_CEILING_S:.0f}s), remaining {remaining} ticks"


def _deltas(distances: dict[str, int]) -> str:
    """Compact per-motor |final - target| report, e.g. 'shoulder_pan=4, ...'."""
    return ", ".join(f"{m}={d}" for m, d in distances.items()) or "no readable motors"
