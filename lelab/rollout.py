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
from .utils.config import (
    bimanual_base_id,
    list_robot_records,
    setup_follower_calibration_file,
    stage_bimanual_follower_calibrations,
)

logger = logging.getLogger(__name__)

# Flat proprioceptive state width of a single SO-101 follower arm (one dim per
# joint). A bimanual checkpoint trains on two arms → twice this. The frontend
# forwards the checkpoint's state_dim (from /policy-config) so the server can
# reject an arm-count mismatch BEFORE spawning the rollout subprocess, instead
# of letting the shape mismatch crash deep inside it.
_SINGLE_ARM_STATE_DIM = 6


class InferenceRequest(BaseModel):
    follower_port: str
    follower_config: str
    policy_ref: str  # opaque ref returned by /jobs/{id}/checkpoints
    task: str = ""
    cameras: dict[str, dict[str, Any]] = {}
    duration_s: int = 60
    # Bimanual: the follower_port/follower_config above is the LEFT arm; these
    # add the RIGHT arm. Inference has no leader arms — only the two followers
    # are driven — so there is no right_leader_* here (cf. record/teleop).
    mode: str = "single"
    right_follower_port: str = ""
    right_follower_config: str = ""
    # Robot record name — used only as the BiSO staging base id (bimanual). It
    # decides the on-disk staging dir, not which calibration drives which arm.
    # Blank/invalid falls back to DEFAULT_BIMANUAL_BASE.
    robot_name: str = ""
    # Flat state width of the selected checkpoint (6 = single SO-101 arm, 12 =
    # bimanual), forwarded from /policy-config so the server can reject an
    # arm-count mismatch pre-spawn. None when the checkpoint omits the feature —
    # the guard then defers to the rollout subprocess's own shape check.
    checkpoint_state_dim: int | None = None
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


def _arm_count_mismatch(mode: str, checkpoint_state_dim: int | None) -> str | None:
    """Explain a checkpoint/robot arm-count mismatch, or None when they agree.

    An SO-101 follower has 6 state dims; a bimanual robot drives two arms (12
    dims). A checkpoint trained on one arm-count crashes on the other deep in
    the rollout subprocess (a raw shape mismatch, no explanation). Reject it
    up front with a legible message when the checkpoint exposes enough to tell.

    `checkpoint_state_dim` is None when the checkpoint omits observation.state
    (e.g. a vision-only policy) — then we can't tell cheaply, so return None and
    let the subprocess's own shape check speak (reported in the modal via the
    existing post-mortem path). A dim that's neither 6 nor a clean multiple is
    also left to the subprocess rather than guessed at here.
    """
    if checkpoint_state_dim is None:
        return None
    robot_is_bimanual = mode == "bimanual"
    # The checkpoint is bimanual iff its state is (a multiple of) two arms wide.
    if checkpoint_state_dim <= _SINGLE_ARM_STATE_DIM:
        checkpoint_is_bimanual = False
    elif checkpoint_state_dim % _SINGLE_ARM_STATE_DIM == 0:
        checkpoint_is_bimanual = checkpoint_state_dim // _SINGLE_ARM_STATE_DIM >= 2
    else:
        # An odd width we don't recognise — don't block on a guess.
        return None
    if robot_is_bimanual == checkpoint_is_bimanual:
        return None
    if checkpoint_is_bimanual:
        return (
            f"This checkpoint was trained on a bimanual robot "
            f"({checkpoint_state_dim}-dim state, 2 arms), but the selected robot is "
            "single-arm. Select a bimanual robot to run this policy."
        )
    return (
        f"This checkpoint was trained on a single-arm robot "
        f"({checkpoint_state_dim}-dim state), but the selected robot is bimanual. "
        "Select a single-arm robot to run this policy."
    )


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


def _preflight_arm_identity(port: str, follower_id: str, config_name: str | None = None) -> list[str]:
    """Read-only identity check of ONE follower arm before the rollout
    subprocess starts.

    The subprocess itself can't be guarded (its stdin is pre-seeded with a
    newline, which auto-confirms lerobot's "use the calibration file" prompt
    and stamps the file into EEPROM on mismatch), so the check happens here:
    connect the bare bus, verify, and release the port for the subprocess to
    reopen. Raises ArmIdentityError on a hard mismatch; returns the
    warn-but-allow messages otherwise.

    `follower_id` names the calibration the arm loads and is what identifies the
    slot by default. For a bimanual staging alias id ("<base>_left"), pass the
    real library stem as `config_name` so the guard compares against the library
    entry rather than the alias (mirrors verify_devices' config_names in
    record/teleop). Bimanual runs each follower bus through this separately —
    each opens and releases its own port — so the two are never open at once."""
    robot = SO101Follower(SO101FollowerConfig(port=port, id=follower_id))
    robot.bus.connect()
    try:
        return verify_devices(
            ((robot, "follower"),),
            extra_slots=_counterpart_leader_slots(config_name or follower_id),
            config_names=[config_name] if config_name is not None else None,
        )
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


def _build_rollout_cmd(request: InferenceRequest, policy_path: str, robot_args: list[str]) -> list[str]:
    """Assemble the full `lerobot-rollout` argv from the robot-specific args.

    `robot_args` is the `--robot.*` block built per mode (single vs bimanual);
    everything else — strategy, policy, task, duration, and the teardown pin —
    is identical across modes and lives here so both paths stay in sync."""
    cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_rollout",
        "--strategy.type=base",
        f"--policy.path={policy_path}",
        f"--policy.device={_detect_device()}",
        *robot_args,
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
    return cmd


def _single_robot_args(request: InferenceRequest, follower_id: str) -> list[str]:
    """`--robot.*` args for a single SO-101 follower."""
    args = [
        "--robot.type=so101_follower",
        f"--robot.port={request.follower_port}",
        f"--robot.id={follower_id}",
    ]
    if request.cameras:
        args.append(f"--robot.cameras={_format_cameras_arg(request.cameras)}")
    return args


def _bimanual_robot_args(request: InferenceRequest, base: str, follower_staging: str) -> list[str]:
    """`--robot.*` args for a bimanual BiSO follower.

    lerobot's BiSOFollowerConfig wraps two SOFollowerConfig sub-arms
    (left_arm_config / right_arm_config) sharing ONE calibration_dir + base id,
    loading each sub-arm's calibration as "<base>_left.json"/"<base>_right.json".
    `follower_staging` is the per-session dir the two library calibrations were
    staged into under that convention (see stage_bimanual_follower_calibrations).
    Cameras
    go on the LEFT arm (BiSO re-exposes them prefixed "left_*"); the right arm is
    camera-free, matching the record/teleop bimanual shape."""
    args = [
        "--robot.type=bi_so_follower",
        f"--robot.id={base}",
        f"--robot.calibration_dir={follower_staging}",
        f"--robot.left_arm_config.port={request.follower_port}",
        f"--robot.right_arm_config.port={request.right_follower_port}",
    ]
    if request.cameras:
        args.append(f"--robot.left_arm_config.cameras={_format_cameras_arg(request.cameras)}")
    return args


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

    # Arm-count guard: reject a single-arm checkpoint on a bimanual robot (and
    # vice versa) BEFORE opening any port or spawning the subprocess, where the
    # shape mismatch would otherwise crash unexplained. Best-effort — defers to
    # the subprocess when the checkpoint doesn't expose observation.state.
    mismatch = _arm_count_mismatch(request.mode, request.checkpoint_state_dim)
    if mismatch is not None:
        with _state_lock:
            inference_active = False
        return {"success": False, "status_code": 409, "message": mismatch}

    is_bimanual = request.mode == "bimanual"
    try:
        if is_bimanual:
            # BiSO loads each sub-arm's calibration as "<base>_left/right.json"
            # from one dir, with no way to point left/right at differently named
            # library files. Stage the two arbitrarily-named follower library
            # calibrations into that convention and point BiSO at the staging
            # dir. Inference has NO leader arms, so stage the follower side only
            # — staging the leader side would require leader library files that
            # this flow never uses (and usually don't exist under the follower's
            # names). The copy fails fast with a clear per-slot error if a
            # library file is missing.
            base = bimanual_base_id(request.robot_name)
            follower_staging, _ = stage_bimanual_follower_calibrations(
                base,
                request.follower_config,
                request.right_follower_config,
            )
            # Sub-arm ids are the BiSO staging aliases ("<base>_left/right"), so
            # the identity guard compares against the real library stems.
            left_id, right_id = f"{base}_left", f"{base}_right"

            identity_warnings = []
            if request.skip_identity_check:
                logger.warning("Arm identity check SKIPPED by request (skip_identity_check=true)")
            else:
                # Each bus opens/verifies/releases sequentially — never both at
                # once — mirroring the single-arm preflight.
                identity_warnings += _preflight_arm_identity(
                    request.follower_port, left_id, config_name=request.follower_config
                )
                identity_warnings += _preflight_arm_identity(
                    request.right_follower_port, right_id, config_name=request.right_follower_config
                )
            # Motor power on both buses, sequentially (each opens its own port).
            identity_warnings += _preflight_motor_power(request.follower_port, left_id, request.motor_power)
            identity_warnings += _preflight_motor_power(
                request.right_follower_port, right_id, request.motor_power
            )

            robot_args = _bimanual_robot_args(request, base, follower_staging)
        else:
            # `setup_follower_calibration_file` returns the basename without the
            # .json extension. We need that stripped form for `--robot.id`,
            # because lerobot appends `.json` itself when constructing
            # `calibration_dir / f"{id}.json"`.
            follower_id = setup_follower_calibration_file(request.follower_config)

            # Arm-identity guard: refuse before the subprocess can move (or stamp
            # the wrong calibration into) an arm that doesn't match its file.
            identity_warnings = []
            if request.skip_identity_check:
                logger.warning("Arm identity check SKIPPED by request (skip_identity_check=true)")
            else:
                identity_warnings = _preflight_arm_identity(request.follower_port, follower_id)

            # Always written (even at 100%) so a gentler previous session can't
            # linger when the arm was never power-cycled.
            identity_warnings += _preflight_motor_power(
                request.follower_port, follower_id, request.motor_power
            )

            robot_args = _single_robot_args(request, follower_id)

        policy_path = _resolve_policy_path(request.policy_ref)
        cmd = _build_rollout_cmd(request, policy_path, robot_args)

        log_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / "inference_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{int(time.time())}.log"
        log_handle = log_path.open("w", buffering=1)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # Feed a newline into stdin PER follower arm so SOFollower.calibrate()'s
        # `input("Press ENTER to use the calibration file ...")` returns "" and
        # writes the existing calibration to the motors instead of hanging
        # forever waiting for an interactive operator. A BiSO follower connects
        # its two sub-arms sequentially (left then right), each of which can fire
        # that prompt once — so seed two newlines for bimanual, one for single.
        # Any prompt that doesn't fire just leaves an unread newline (harmless);
        # subsequent input() calls in the recalibration path get EOF and raise —
        # fine, because we never want to enter that path from the UI.
        stdin_seed = b"\n\n" if is_bimanual else b"\n"
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        try:
            assert proc.stdin is not None
            proc.stdin.write(stdin_seed)
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
