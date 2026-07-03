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

"""Inference mode: drives the SO-101 follower with a trained policy.

Mirrors `app/teleoperating.py` in shape — single global session, mutex
with teleoperation/recording (the follower's serial bus can only be
opened once), `lerobot.scripts.lerobot_rollout` running as a subprocess
for clean cancellation. Hub-checkpoint refs are resolved to a local dir
via huggingface_hub.snapshot_download before we spawn the subprocess.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from .arm_identity import ArmIdentityError, ArmSlot, verify_devices
from .motor_power import apply_motor_power, clear_goal_velocity
from .utils.config import list_robot_records, setup_follower_calibration_file

logger = logging.getLogger(__name__)


class InferenceRequest(BaseModel):
    follower_port: str
    follower_config: str
    policy_ref: str  # opaque ref returned by /jobs/{id}/checkpoints
    task: str = ""
    cameras: dict[str, dict[str, Any]] = {}
    duration_s: int = 60
    # Escape hatch for the arm-identity guard (see lelab/arm_identity.py):
    # when true, run even if the connected arm doesn't match its calibration.
    skip_identity_check: bool = False
    # Follower torque as a percentage of full power (see lelab/motor_power.py).
    # Clamped server-side to 10-100; written before the subprocess starts.
    motor_power: int = 100


inference_active: bool = False
_inference_proc: subprocess.Popen | None = None
_inference_started_at: float | None = None
_inference_rollout_started_at: float | None = None
_inference_meta: dict[str, Any] = {}
# Guards mutations to the globals above; held only for the short critical
# sections in start/stop/status.
_state_lock = threading.Lock()
_HUB_REF_RE = re.compile(r"^(?P<repo>[^@]+)@checkpoints/(?P<step_dir>\d+)$")
_HUB_ROOT_REF_RE = re.compile(r"^(?P<repo>[^@]+)@root$")
# lerobot prints this once per run, the moment its main control loop is
# about to take over from the setup phase. We watch stdout for it so the
# UI can present a "rollout time" separate from the multi-second policy
# load + bus connect + camera connect setup overhead.
_ROLLOUT_START_MARKER = "Rollout setup complete"


def _pump_stdout(proc: subprocess.Popen, log_handle) -> None:
    """Tee the subprocess's stdout to the log file and watch for the
    rollout-start marker."""
    global _inference_rollout_started_at
    try:
        for raw in iter(proc.stdout.readline, b""):
            try:
                line = raw.decode("utf-8", errors="replace")
            except Exception:
                continue
            try:
                log_handle.write(line)
                log_handle.flush()
            except Exception:
                pass
            if _inference_rollout_started_at is None and _ROLLOUT_START_MARKER in line:
                _inference_rollout_started_at = time.time()
                logger.info(
                    "Inference rollout main loop started after %.1fs of setup",
                    _inference_rollout_started_at - (_inference_started_at or _inference_rollout_started_at),
                )
    except Exception as exc:
        logger.exception("Inference stdout pump failed: %s", exc)
    finally:
        with contextlib.suppress(Exception):
            log_handle.close()


def _detect_device() -> str:
    """cuda → mps → cpu, picked once at start time."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _resolve_policy_path(policy_ref: str) -> str:
    """Turn a checkpoints API ref into a local path that lerobot accepts.

    Local refs are already absolute paths to a pretrained_model dir.
    Hub refs look like 'user/repo@checkpoints/<step_dir>' where
    <step_dir> is lerobot's zero-padded directory name (e.g. 000050) — we
    forward it verbatim into snapshot_download's allow_patterns and the
    resolved local path.
    A 'user/repo@root' ref means the whole repo IS the pretrained_model
    (no checkpoints sub-tree); the full repo is downloaded via
    snapshot_download and its root is returned directly."""
    if Path(policy_ref).is_dir():
        return policy_ref
    from huggingface_hub import snapshot_download

    m = _HUB_REF_RE.match(policy_ref)
    if m:
        repo_id, step_dir = m.group("repo"), m.group("step_dir")
        local_root = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=[f"checkpoints/{step_dir}/pretrained_model/*"],
        )
        return str(Path(local_root) / "checkpoints" / step_dir / "pretrained_model")
    m = _HUB_ROOT_REF_RE.match(policy_ref)
    if m:
        return snapshot_download(repo_id=m.group("repo"), repo_type="model")
    raise ValueError(f"Unrecognised policy ref: {policy_ref!r}")


def _counterpart_leader_slots(follower_id: str) -> list[ArmSlot]:
    """Leader config(s) paired with this follower config in saved robot records.

    Inference only connects the follower, so the guard can't derive the
    counterpart slot from the session itself (the way teleop/record do). Look
    it up: any robot record whose follower slot is `follower_id` names the
    leader config that belongs on the OTHER port — if the connected arm's
    EEPROM fingerprint matches that config, the ports are swapped (hard block
    instead of a generic warning)."""
    slots: list[ArmSlot] = []
    seen: set[tuple[str, str]] = set()
    for record in list_robot_records():
        for follower_field, leader_field, label in (
            ("follower_config", "leader_config", "leader"),
            ("right_follower_config", "right_leader_config", "right leader"),
        ):
            leader_name = record.get(leader_field) or ""
            if record.get(follower_field) == follower_id and leader_name and (label, leader_name) not in seen:
                seen.add((label, leader_name))
                slots.append(ArmSlot(label, "leader", leader_name))
    return slots


def _preflight_arm_identity(port: str, follower_id: str) -> list[str]:
    """Read-only identity check of the follower arm before the rollout
    subprocess starts.

    The subprocess itself can't be guarded (its stdin is pre-seeded with a
    newline, which auto-confirms lerobot's "use the calibration file" prompt
    and stamps the file into EEPROM on mismatch), so the check happens here:
    connect the bare bus, verify, and release the port for the subprocess to
    reopen. Raises ArmIdentityError on a hard mismatch; returns the
    warn-but-allow messages otherwise."""
    robot = SO101Follower(SO101FollowerConfig(port=port, id=follower_id))
    robot.bus.connect()
    try:
        return verify_devices(((robot, "follower"),), extra_slots=_counterpart_leader_slots(follower_id))
    finally:
        # Reads only: torque was never enabled, so skip the torque-disable
        # write and just close the port.
        robot.bus.disconnect(disable_torque=False)


def _preflight_motor_power(port: str, follower_id: str, percent: int) -> list[str]:
    """Prime the follower's RAM motor registers before the rollout subprocess
    starts.

    The subprocess itself can't be instrumented, but Torque_Limit and
    Goal_Velocity are both RAM registers: they survive closing the serial port
    (only a power cycle resets them), and the subprocess's connect()/configure()
    never writes them — so setting them here and releasing the port is enough
    for the whole rollout. Two priming steps:
      - apply_motor_power: the session torque cap (percent → Torque_Limit).
      - clear_goal_velocity: reset any leftover speed cap a previous
        arm-driving feature stamped (auto-cal fold/unfold=1000, rest-pose
        return=400), which would otherwise throttle the whole rollout.
    Never raises: a failure degrades to the previous register value (logged)
    and returns warning messages instead of aborting the start."""
    robot = SO101Follower(SO101FollowerConfig(port=port, id=follower_id))
    try:
        robot.bus.connect()
        try:
            return apply_motor_power(robot, percent, "follower arm") + clear_goal_velocity(
                robot, "follower arm"
            )
        finally:
            # Torque was never enabled here; just release the port for the
            # subprocess to reopen.
            robot.bus.disconnect(disable_torque=False)
    except Exception as exc:
        message = (
            f"Could not set motor power to {percent}% on {port}: {exc}. "
            "The arm runs at its previous limit (full power after a power-up)."
        )
        logger.warning(message)
        return [message]


def _format_cameras_arg(cameras: dict[str, dict[str, Any]]) -> str:
    """Convert {name: {type, camera_index, width, height, fps}} into
    lerobot's CLI dict syntax. The frontend key `camera_index` is
    remapped to lerobot's `index_or_path`."""
    parts = []
    for name, cfg in cameras.items():
        remapped = {
            ("index_or_path" if k == "camera_index" else k): v for k, v in cfg.items() if v is not None
        }
        body = ", ".join(f"{k}: {v}" for k, v in remapped.items())
        parts.append(f"{name}: {{{body}}}")
    return "{" + ", ".join(parts) + "}"


def handle_start_inference(request: InferenceRequest) -> dict[str, Any]:
    """Start a one-shot rollout subprocess. Returns a dict — the route
    layer turns it into a JSON response or HTTPException as appropriate."""
    global inference_active, _inference_proc, _inference_started_at
    global _inference_rollout_started_at, _inference_meta

    # Mutex with teleop and recording: all three drive the same serial bus.
    from . import record as _record, teleoperate as _teleoperate

    with _state_lock:
        if _teleoperate.teleoperation_active:
            return {
                "success": False,
                "status_code": 409,
                "message": "Teleoperation is currently active. Stop it first.",
            }
        if _record.recording_active:
            return {
                "success": False,
                "status_code": 409,
                "message": "Recording is currently active. Stop it first.",
            }
        if inference_active:
            return {
                "success": False,
                "status_code": 409,
                "message": "Inference is already active. Stop it first.",
            }
        # Claim the slot now so a concurrent caller losing the race sees us.
        inference_active = True

    try:
        # `setup_follower_calibration_file` returns the basename without the
        # .json extension. We need that stripped form for `--robot.id`,
        # because lerobot appends `.json` itself when constructing
        # `calibration_dir / f"{id}.json"`.
        follower_id = setup_follower_calibration_file(request.follower_config)

        # Arm-identity guard: refuse before the subprocess can move (or stamp
        # the wrong calibration into) an arm that doesn't match its file.
        identity_warnings: list[str] = []
        if request.skip_identity_check:
            logger.warning("Arm identity check SKIPPED by request (skip_identity_check=true)")
        else:
            identity_warnings = _preflight_arm_identity(request.follower_port, follower_id)

        # Always written (even at 100%) so a gentler previous session can't
        # linger when the arm was never power-cycled.
        identity_warnings += _preflight_motor_power(request.follower_port, follower_id, request.motor_power)

        policy_path = _resolve_policy_path(request.policy_ref)

        cmd = [
            sys.executable,
            "-m",
            "lerobot.scripts.lerobot_rollout",
            "--strategy.type=base",
            f"--policy.path={policy_path}",
            f"--policy.device={_detect_device()}",
            "--robot.type=so101_follower",
            f"--robot.port={request.follower_port}",
            f"--robot.id={follower_id}",
            f"--task={request.task}",
            f"--duration={request.duration_s}",
            # Pin the teardown behaviour the stop dialog promises ("eases the
            # follower back to its start pose, then goes limp"). lerobot's
            # RolloutConfig.return_to_initial_position defaults to True today,
            # but relying on that default means an upstream flip would silently
            # break the promise — the arm would stay wherever the policy left
            # it. Set it explicitly so the contract is ours, not upstream's.
            "--return_to_initial_position=true",
        ]
        if request.cameras:
            cmd.append(f"--robot.cameras={_format_cameras_arg(request.cameras)}")

        log_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / "inference_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{int(time.time())}.log"
        log_handle = log_path.open("w", buffering=1)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # Feed a single newline into stdin so SOFollower.calibrate()'s
        # `input("Press ENTER to use the calibration file ...")` returns "" and
        # writes the existing calibration to the motors instead of hanging
        # forever waiting for an interactive operator. Subsequent input()
        # calls in the recalibration path get EOF and raise — which is fine,
        # because we never want to enter that path from the UI.
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        try:
            assert proc.stdin is not None
            proc.stdin.write(b"\n")
            proc.stdin.flush()
            proc.stdin.close()
        except Exception as exc:
            logger.warning("Failed to seed stdin for inference subprocess: %s", exc)
        threading.Thread(
            target=_pump_stdout,
            args=(proc, log_handle),
            name="inference-stdout-pump",
            daemon=True,
        ).start()
    except ArmIdentityError as exc:
        # The connected arm doesn't match its assigned calibration; the message
        # is already user-facing. Subprocess never started — release the slot.
        with _state_lock:
            inference_active = False
        return {"success": False, "status_code": 409, "message": str(exc)}
    except Exception as exc:
        logger.exception("Failed to start inference")
        # Subprocess never started — release the slot.
        with _state_lock:
            inference_active = False
        return {"success": False, "status_code": 500, "message": f"Failed to start inference: {exc}"}

    with _state_lock:
        _inference_proc = proc
        _inference_started_at = time.time()
        _inference_rollout_started_at = None
        _inference_meta = {
            "policy_ref": request.policy_ref,
            "duration_s": request.duration_s,
            "log_path": str(log_path),
        }
    logger.info("Inference started: pid=%s policy=%s", proc.pid, policy_path)
    response = {"success": True, "message": "Inference started", "log_path": str(log_path)}
    if identity_warnings:
        response["warning"] = " ".join(identity_warnings)
    return response


def handle_stop_inference() -> dict[str, Any]:
    global inference_active, _inference_proc, _inference_started_at
    global _inference_rollout_started_at, _inference_meta

    with _state_lock:
        if not inference_active or _inference_proc is None:
            return {"success": False, "status_code": 409, "message": "No inference is active"}
        proc = _inference_proc

    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Inference did not exit in 5s; killing")
            proc.kill()
            proc.wait()
    except Exception as exc:
        logger.exception("Stop inference: %s", exc)

    with _state_lock:
        inference_active = False
        _inference_proc = None
        _inference_started_at = None
        _inference_rollout_started_at = None
        _inference_meta = {}
    return {"success": True, "message": "Inference stopped"}


def handle_inference_status() -> dict[str, Any]:
    global inference_active, _inference_proc, _inference_started_at
    global _inference_rollout_started_at, _inference_meta

    # Finalise state lazily if the subprocess died on its own.
    with _state_lock:
        proc = _inference_proc
        if proc is not None and proc.poll() is not None:
            rc = proc.returncode
            logger.info("Inference subprocess exited rc=%s", rc)
            finished_meta = _inference_meta
            finished_started = _inference_started_at
            finished_rollout_started = _inference_rollout_started_at
            inference_active = False
            _inference_proc = None
            _inference_started_at = None
            _inference_rollout_started_at = None
            _inference_meta = {}
            return {
                "inference_active": False,
                "exited": True,
                "exit_code": rc,
                "policy_ref": finished_meta.get("policy_ref"),
                "duration_s": finished_meta.get("duration_s"),
                "log_path": finished_meta.get("log_path"),
                "started_at": finished_started,
                "rollout_started_at": finished_rollout_started,
                "rollout_elapsed_s": 0,
                "elapsed_s": 0,
            }
        elapsed = (time.time() - _inference_started_at) if _inference_started_at else 0
        rollout_elapsed = time.time() - _inference_rollout_started_at if _inference_rollout_started_at else 0
        return {
            "inference_active": inference_active,
            "started_at": _inference_started_at,
            "rollout_started_at": _inference_rollout_started_at,
            "elapsed_s": elapsed,
            "rollout_elapsed_s": rollout_elapsed,
            "duration_s": _inference_meta.get("duration_s"),
            "policy_ref": _inference_meta.get("policy_ref"),
            "log_path": _inference_meta.get("log_path"),
        }
