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
Wiggle-to-find-port: drive the gripper on a given serial port a few times so the
user can see which physical arm is on that port. Uses raw motor positions, so no
calibration is required. Only upstream lerobot APIs are used.
"""

import asyncio
import logging
import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

logger = logging.getLogger(__name__)

# ~200 encoder steps (of 4096) is a small but clearly visible movement.
_WIGGLE_OFFSET = 200
_WIGGLE_REPEATS = 3
_WIGGLE_TIMEOUT_S = 15.0


def _wiggle_gripper_sync(port: str) -> None:
    """Connect to the gripper (motor id 6) on `port` and wiggle it in place.

    Reads the current raw position, then moves +/- _WIGGLE_OFFSET a few times and
    returns to the start. Blocking; run in a worker thread.
    """
    bus = FeetechMotorsBus(
        port=port,
        motors={"gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100)},
    )
    try:
        bus.connect()
        current = bus.sync_read("Present_Position", "gripper", normalize=False)["gripper"]

        if current - _WIGGLE_OFFSET < 0 or current + _WIGGLE_OFFSET > 4095:
            raise ValueError(
                f"Gripper position ({current}) is too close to the edge of its range. "
                "Power-cycle the arm and try again."
            )

        for _ in range(_WIGGLE_REPEATS):
            bus.write("Goal_Position", "gripper", current + _WIGGLE_OFFSET, normalize=False)
            time.sleep(0.3)
            bus.write("Goal_Position", "gripper", current - _WIGGLE_OFFSET, normalize=False)
            time.sleep(0.3)

        # Return to where we started.
        bus.write("Goal_Position", "gripper", current, normalize=False)
        time.sleep(0.3)
    finally:
        bus.disconnect()


async def wiggle_gripper(port: str) -> dict:
    """
    Run the wiggle in a worker thread with a timeout. Returns a result dict
    ({"success": bool, "message": str}) — logical failures (port busy, arm off)
    are reported, not raised, so the endpoint stays HTTP 200 like the rest of the
    feature handlers.
    """
    if not port or not port.strip():
        return {"success": False, "message": "No port provided."}
    try:
        await asyncio.wait_for(
            asyncio.to_thread(_wiggle_gripper_sync, port.strip()),
            timeout=_WIGGLE_TIMEOUT_S,
        )
        return {"success": True, "message": f"Wiggled the gripper on {port}."}
    except asyncio.TimeoutError:
        return {
            "success": False,
            "message": "Wiggle timed out after 15s — is the arm powered on and the port correct?",
        }
    except Exception as e:
        logger.exception("Wiggle failed")
        return {"success": False, "message": f"Failed to wiggle the gripper: {e}"}
