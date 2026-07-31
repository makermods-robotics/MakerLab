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
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from pydantic import BaseModel

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from .arm_identity import ArmIdentityError, ArmSlot, verify_devices
from .camera_preview import camera_preview_manager
from .jobs import download_hub_checkpoint_ref, make_snapshot_progress_tqdm
from .motor_power import clear_goal_velocity, reset_torque_limit
from .record import _DEFAULT_FOURCC
from .utils.config import (
    bimanual_base_id,
    list_robot_records,
    setup_follower_calibration_file,
    stage_bimanual_follower_calibrations,
)
from .utils.errors import friendly_hint, is_cleanup_error

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
    # Escape hatch for the arm-identity guard (see makermodslab/arm_identity.py):
    # when true, run even if the connected arm doesn't match its calibration.
    skip_identity_check: bool = False
    # Multi-episode EVALUATION mode. 1 (the default) is exactly the historical
    # single-rollout flow — no episode bookkeeping, no reset phase, no accuracy.
    # >1 walks N sequential rollout subprocesses inside ONE session (one model
    # download, one arm preflight, one camera handover), scoring each episode.
    # Clamped server-side to [1, MAX_EVAL_EPISODES] — see clamp_eval_episodes.
    eval_episodes: int = 1


inference_active: bool = False
_inference_proc: subprocess.Popen | None = None
_inference_started_at: float | None = None
_inference_rollout_started_at: float | None = None
_inference_meta: dict[str, Any] = {}
# The finished (exited) status payload of the most recent run, kept until the
# NEXT start claims the slot. Terminal outcomes must be idempotent, not
# consume-once: several surfaces poll /inference-status concurrently (the
# session dialog at 1 Hz, the Deploy panel at 0.5 Hz), and with a
# report-once-then-clear scheme whichever poll lands first after the subprocess
# dies swallows the outcome/error/hint — the dialog then sees a bare idle
# status and misreports a crash as a clean finish.
_last_result: dict[str, Any] | None = None
# Set for the CURRENT session at claim time; the background startup worker
# captures its own reference and stop() sets it. It's the only way to abandon a
# start that's still in its pre-subprocess window (Hub download / arm preflight),
# where there's no process to terminate. A fresh Event per session means an
# orphaned worker from a stopped session sees its (set) event and bails, while a
# new session gets a clean one. None while idle.
_inference_cancel: threading.Event | None = None
# Handle to the background startup worker (_run_inference_startup) for the
# CURRENT/most-recently-started session. `_inference_cancel` only aborts the
# worker at coarse boundaries (before it opens the bus in _prepare_robot,
# and again after _prepare_robot returns) — nothing interrupts it WHILE
# _prepare_robot is actually touching hardware, so a stop can leave the
# worker alive and still driving the bus for a few more seconds. Tracking the
# thread (mirrors teleoperate.py's `teleoperation_thread`, added for the same
# reason in T3) lets handle_start_inference refuse a new session while that
# orphaned worker is still alive, instead of racing it for the same serial
# port. None once the worker has exited or before any session has started.
_inference_startup_thread: threading.Thread | None = None
# Multi-episode evaluation bookkeeping for the CURRENT session lives in
# `_eval_session`, declared just below the _EvalSession dataclass.
# Guards mutations to the globals above (and _eval_session); held only for the
# short critical sections in start/stop/status.
_state_lock = threading.Lock()
# Bound on how long a second stop-inference call waits for an orphaned startup
# worker (see _inference_startup_thread) to exit before giving up and
# reporting it's still alive. Mirrors teleoperate.py's second-stop join
# timeout; unlike that one, this can't force the worker out mid-call (no
# cooperative cancellation checkpoint inside _prepare_robot), so it's a
# bounded wait-and-report rather than a true force-release.
_STARTUP_STOP_JOIN_TIMEOUT_S = 5.0
# The two hub-ref shapes /jobs/{id}/checkpoints hands out. Kept here for the
# cheap no-network shape check below; the DOWNLOAD they imply lives once, in
# jobs.download_hub_checkpoint_ref, so inference and fine-tune resolve a ref
# identically.
_HUB_REF_RE = re.compile(r"^(?P<repo>[^@]+)@checkpoints/(?P<step_dir>\d+)$")
_HUB_ROOT_REF_RE = re.compile(r"^(?P<repo>[^@]+)@root$")
# lerobot prints this once per run, the moment its main control loop is
# about to take over from the setup phase. We watch stdout for it so the
# UI can present a "rollout time" separate from the multi-second policy
# load + bus connect + camera connect setup overhead.
_ROLLOUT_START_MARKER = "Rollout setup complete"

# Structured "which substep am I in" for the startup sequence, surfaced in the
# /inference-status payload so the UI can name the wait ("Downloading model…",
# "Connecting to arm…") instead of a single opaque spinner. Ordered:
#   downloading_model — snapshot_download of a Hub checkpoint (server thread,
#       BEFORE the subprocess spawns). Skipped for a local checkpoint dir.
#   starting          — subprocess spawned, before any recognised setup line.
#   loading_policy    — lerobot's context.py "Loading policy from ..." emitted.
#   connecting        — lerobot's "Connecting robot ..." emitted (the bus- and
#       camera-connect window; both open inside robot.connect()).
#   running           — the rollout main loop has taken over (marker seen).
#   stopping/stopped/error — terminal, set by stop/status finalisation.
# There is no `downloading_dataset` phase: the base-strategy rollout command we
# build passes no --dataset, so build_rollout_context never sets up (or
# downloads) a dataset. We omit the phase rather than invent one that never
# fires.
#
# EVAL-ONLY phases (eval_episodes > 1). The startup phases above repeat per
# episode (each episode is its own subprocess), plus:
#   resetting — an episode ended, the tally was updated, and the session is
#       parked waiting for the user to rearrange the scene and POST
#       /inference-next-episode. Also where a CRASHED episode parks, with
#       `error`/`hint` populated so the user can continue or abort.
#   finished  — every episode ran; terminal, carries `accuracy`.
#   aborted   — /stop-inference ended the session early; terminal, partial
#       tally, NO accuracy claimed.
PHASE_DOWNLOADING_MODEL = "downloading_model"
PHASE_STARTING = "starting"
PHASE_LOADING_POLICY = "loading_policy"
PHASE_CONNECTING = "connecting"
PHASE_RUNNING = "running"
PHASE_STOPPING = "stopping"
PHASE_STOPPED = "stopped"
PHASE_ERROR = "error"
PHASE_RESETTING = "resetting"
PHASE_FINISHED = "finished"
PHASE_ABORTED = "aborted"

# Per-episode verdicts, in the order the UI tallies them.
#   success — the user pressed "task succeeded" and we terminated the episode.
#   failure — the episode ran out its --duration without the user calling it.
#   error   — the episode crashed (a serial glitch, a camera drop, a policy
#       blow-up). NEITHER success nor failure: deliberately excluded from the
#       accuracy denominator so one hardware hiccup can't poison a 20-episode
#       number.
EPISODE_SUCCESS = "success"
EPISODE_FAILURE = "failure"
EPISODE_ERROR = "error"

# Upper bound on a single eval session. 200 episodes × a 60s duration is
# already a >3h bench session; anything past this is a typo, not a plan.
MAX_EVAL_EPISODES = 200


def clamp_eval_episodes(value: int | None) -> int:
    """Coerce a requested episode count into [1, MAX_EVAL_EPISODES].

    Clamps rather than rejects: a nonsensical count (0, -5, 10_000) is a UI slip,
    and silently running one episode / the cap is friendlier than a 422 that
    loses the whole configured launch. A non-integer or None falls back to 1."""
    try:
        n = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 1
    return max(1, min(MAX_EVAL_EPISODES, n))


def eval_accuracy(results: Sequence[str]) -> float | None:
    """successes / (successes + failures) over the recorded episode verdicts.

    Crashed episodes (EPISODE_ERROR) are excluded from BOTH numerator and
    denominator — see EPISODE_ERROR. Returns None when nothing scoreable has
    happened yet (no episodes, or every episode crashed), so the UI shows
    "no accuracy" instead of a misleading 0%."""
    scored = [r for r in results if r in (EPISODE_SUCCESS, EPISODE_FAILURE)]
    if not scored:
        return None
    return round(sum(1 for r in scored if r == EPISODE_SUCCESS) / len(scored), 4)


def classify_episode(
    rc: int | None,
    stop_requested: bool,
    rollout_started: bool,
    error_text: str | None,
) -> str:
    """Turn one episode's subprocess exit into a verdict.

    `stop_requested` (the user pressed "task succeeded — stop episode") wins
    outright: we terminated the subprocess ourselves, so its exit code is our own
    SIGTERM and says nothing about the run.

    Otherwise the episode ended on its own. Reuse `_classify_outcome`:
      ok / ran_with_warning → the rollout ran its full --duration (a noisy
          torque-disable on teardown is not a failed episode) → FAILURE, i.e.
          the policy never got the task done in the time allowed.
      failed → the episode crashed → ERROR, excluded from the accuracy."""
    if stop_requested:
        return EPISODE_SUCCESS
    if _classify_outcome(rc, rollout_started, error_text) == "failed":
        return EPISODE_ERROR
    return EPISODE_FAILURE


@dataclass
class _EvalSession:
    """Bookkeeping for ONE multi-episode evaluation session.

    Lives for the whole session (across N subprocesses) — unlike
    `_inference_meta`, which is per-episode and cleared at each subprocess exit.
    Mutated only under `_state_lock`. None whenever the session is single-episode
    or idle, which is what every eval-only endpoint gates on."""

    request: InferenceRequest
    episodes_total: int
    # Resolved ONCE by the startup worker and reused verbatim for every episode:
    # the model is downloaded once and the arm preflight runs once per session.
    policy_path: str | None = None
    robot_args: list[str] = field(default_factory=list)
    # Verdicts in episode order; len(results) is how many episodes have finished,
    # so the CURRENT episode is 1-based index len(results) + 1.
    results: list[str] = field(default_factory=list)
    # Set by /inference-episode-stop just before it terminates the subprocess, so
    # the exit finalisation scores the episode a success instead of reading our
    # own SIGTERM as a crash. Cleared as each episode is scored.
    stop_requested: bool = False
    # A crashed episode's mined error + plain-language hint, surfaced on the
    # resetting payload and cleared when the user continues.
    error: str | None = None
    hint: str | None = None

    @property
    def episode_index(self) -> int:
        """1-based index of the episode currently running (or about to run).

        Clamped to episodes_total so the final payload reads "10 / 10" rather
        than "11 / 10"."""
        return min(len(self.results) + 1, self.episodes_total)


# Evaluation bookkeeping for the CURRENT session, or None when the session is a
# plain single rollout (eval_episodes <= 1) or nothing is running. Every
# eval-only endpoint gates on this being non-None, which is what keeps the
# single-episode flow bit-for-bit unchanged. Mutated under `_state_lock`.
_eval_session: _EvalSession | None = None


def _eval_fields(
    ev: _EvalSession | None,
    *,
    accuracy: float | None = None,
) -> dict[str, Any]:
    """The eval block of an /inference-status payload.

    Emitted on EVERY payload so the shape is stable for the frontend: a
    single-episode run reports `eval_mode: False` with null/empty companions
    rather than omitting the keys. `accuracy` is passed in (not derived) because
    it is claimed ONLY on a session that ran to completion — an aborted session
    reports its partial tally with accuracy None."""
    if ev is None:
        return {
            "eval_mode": False,
            "episode_index": None,
            "episodes_total": None,
            "episode_results": None,
            "accuracy": None,
        }
    return {
        "eval_mode": True,
        "episode_index": ev.episode_index,
        "episodes_total": ev.episodes_total,
        "episode_results": list(ev.results),
        "accuracy": accuracy,
    }


# Stable lerobot setup log fragments (lerobot/rollout/context.py) that mark the
# transition into a finer sub-phase. Watched in _pump_stdout. These are plain
# logger.info messages, not a documented contract — if an upstream bump renames
# them the phase just stays at its previous (coarser but still correct) value,
# so a drift degrades gracefully rather than crashing.
_PHASE_MARKERS: tuple[tuple[str, str], ...] = (
    ("Loading policy from", PHASE_LOADING_POLICY),
    ("Connecting robot", PHASE_CONNECTING),
)


def _set_phase(phase: str) -> None:
    """Record the current startup sub-phase on the shared inference meta.

    Guarded by _state_lock (short critical section). A no-op when no session is
    active — a late stdout line arriving after teardown can't resurrect a
    phase on an empty meta dict."""
    with _state_lock:
        if _inference_meta:
            _inference_meta["phase"] = phase


def _pump_stdout(proc: subprocess.Popen, log_handle) -> None:
    """Tee the subprocess's stdout to the log file, advance the startup
    sub-phase off recognised lerobot setup lines, and watch for the
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
            # Advance to a finer setup sub-phase on the first matching line.
            # Cheap substring checks; only fires before the rollout marker, so
            # a later line mentioning "Connecting robot" can't drag a running
            # session backwards.
            if _inference_rollout_started_at is None:
                for fragment, phase in _PHASE_MARKERS:
                    if fragment in line:
                        _set_phase(phase)
                        break
            if _inference_rollout_started_at is None and _ROLLOUT_START_MARKER in line:
                _inference_rollout_started_at = time.time()
                _set_phase(PHASE_RUNNING)
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


def _report_download_progress(bytes_done: int, bytes_total: int | None) -> None:
    """Record Hub-download byte progress on the live inference meta.

    Fed from the snapshot_download tqdm hook (make_snapshot_progress_tqdm), may
    fire from any thread. A no-op once the meta is gone (a stopped/failed session
    cleared it) so a late tqdm callback can't resurrect a dead session. ``percent``
    is None while the total is still unknown → the UI shows an indeterminate bar."""
    with _state_lock:
        if not _inference_meta:
            return
        _inference_meta["download_bytes_done"] = bytes_done
        _inference_meta["download_bytes_total"] = bytes_total
        _inference_meta["download_percent"] = (
            round(bytes_done / bytes_total * 100, 1) if bytes_total else None
        )


def _policy_ref_is_valid(policy_ref: str) -> bool:
    """Cheap shape check for a policy ref (one is_dir stat, no network) so a
    malformed ref is rejected synchronously in the POST — surfacing in the modal
    as a 4xx — instead of failing later on the inference page."""
    return (
        bool(_HUB_REF_RE.match(policy_ref))
        or bool(_HUB_ROOT_REF_RE.match(policy_ref))
        or Path(policy_ref).is_dir()
    )


def _resolve_policy_path(policy_ref: str, report: Callable[[int, int | None], None] | None = None) -> str:
    """Turn a checkpoints API ref into a local path that lerobot accepts.

    Local refs are already absolute paths to a pretrained_model dir.
    Hub refs look like 'user/repo@checkpoints/<step_dir>' where
    <step_dir> is lerobot's zero-padded directory name (e.g. 000050) — we
    forward it verbatim into snapshot_download's allow_patterns and the
    resolved local path.
    A 'user/repo@root' ref means the whole repo IS the pretrained_model
    (no checkpoints sub-tree); the full repo is downloaded via
    snapshot_download and its root is returned directly.

    When ``report`` is given, snapshot_download streams byte progress through it
    (see make_snapshot_progress_tqdm) so the inference page can show a real
    download bar. Local refs never download, so they never report and never flip
    the phase.

    The download itself (which patterns each ref shape pulls, and what path it
    yields) lives in jobs.download_hub_checkpoint_ref, shared with the fine-tune
    path so a ref resolves to the same weights whoever asks. This wrapper owns
    only what is inference-specific: the local-dir short-circuit, the
    downloading_model phase, and the progress hook."""
    if Path(policy_ref).is_dir():
        # A local checkpoint — nothing to fetch, so no downloading_model phase.
        return policy_ref
    if not _policy_ref_is_valid(policy_ref):
        raise ValueError(f"Unrecognised policy ref: {policy_ref!r}")

    # A Hub ref: the download may pull hundreds of MB and take minutes.
    # Announce it (downloading_model phase) so the UI names the wait, and feed
    # byte progress through the tqdm hook when a reporter is supplied. Set only on
    # the download paths (not the local branch above), and only when a session is
    # live (_set_phase no-ops otherwise), so this helper stays safe to call from
    # the unit tests.
    _set_phase(PHASE_DOWNLOADING_MODEL)
    tqdm_class = make_snapshot_progress_tqdm(report) if report is not None else None
    return download_hub_checkpoint_ref(policy_ref, tqdm_class=tqdm_class)


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


@contextmanager
def _open_follower(port: str, follower_id: str):
    """Open a bare follower bus on `port`, yield the connected robot, and
    release the port read-only on exit.

    Both rollout preflights connect one follower, do read-only work, then must
    free the port for the subprocess to reopen. Torque is never enabled here,
    so the release skips the torque-disable write (``disconnect(
    disable_torque=False)``) — a plain port close. The disconnect runs on any
    exit path (success or exception)."""
    robot = SO101Follower(SO101FollowerConfig(port=port, id=follower_id))
    robot.bus.connect()
    try:
        yield robot
    finally:
        robot.bus.disconnect(disable_torque=False)


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
    with _open_follower(port, follower_id) as robot:
        return verify_devices(
            ((robot, "follower"),),
            extra_slots=_counterpart_leader_slots(config_name or follower_id),
            config_names=[config_name] if config_name is not None else None,
        )


def _preflight_motor_registers(port: str, follower_id: str) -> list[str]:
    """Prime the follower's RAM motor registers before the rollout subprocess
    starts.

    The subprocess itself can't be instrumented, but Torque_Limit and
    Goal_Velocity are both RAM registers: they survive closing the serial port
    (only a power cycle resets them), and the subprocess's connect()/configure()
    never writes them — so setting them here and releasing the port is enough
    for the whole rollout. Two priming steps:
      - reset_torque_limit: restore stock torque (a previous auto-calibration's
        working torque would otherwise cap the whole rollout).
      - clear_goal_velocity: reset any leftover speed cap a previous
        arm-driving feature stamped (auto-cal fold/unfold=1000, rest-pose
        return=400), which would otherwise throttle the whole rollout.
    Never raises: a failure degrades to the previous register value (logged)
    and returns warning messages instead of aborting the start."""
    try:
        with _open_follower(port, follower_id) as robot:
            return reset_torque_limit(robot, "follower arm") + clear_goal_velocity(
                robot, "follower arm"
            )
    except Exception as exc:
        message = (
            f"Could not reset the motor registers on {port}: {exc}. "
            "The arm runs at its previous torque/speed limits for this rollout."
        )
        logger.warning(message)
        return [message]


def _format_cameras_arg(cameras: dict[str, dict[str, Any]]) -> str:
    """Convert {name: {type, camera_index, width, height, fps}} into
    lerobot's CLI dict syntax. The frontend key `camera_index` is
    remapped to lerobot's `index_or_path`.

    Like recording (`record._build_camera_configs`), opencv cameras default to
    MJPG when the request doesn't pin a fourcc: without it, Linux/V4L2
    negotiates raw YUYV and a 3-camera rig exhausts the USB bus at STREAMON —
    the third camera fails during inference only, since recording already
    defaults to MJPG. An explicit fourcc from the UI still wins.
    """
    parts = []
    for name, cfg in cameras.items():
        remapped = {
            ("index_or_path" if k == "camera_index" else k): v for k, v in cfg.items() if v is not None
        }
        if cfg.get("type") == "opencv" and not cfg.get("fourcc"):
            remapped["fourcc"] = _DEFAULT_FOURCC
        body = ", ".join(f"{k}: {v}" for k, v in remapped.items())
        parts.append(f"{name}: {{{body}}}")
    return "{" + ", ".join(parts) + "}"


# Exception lines at the tail of a Python traceback look like
# "RuntimeError: ..." or "lerobot.errors.DeviceNotConnectedError: ...".
_EXC_LINE_RE = re.compile(r"^[A-Za-z_][\w.]*(?:Error|Exception|Interrupt|Timeout|Failure)\b")


def _read_log_tail_lines(log_path: str | None) -> list[str] | None:
    """Decode the last ~64 KB of a log file into text lines (the window's oldest
    line first, newest last).

    Only the tail is read, so a multi-MB verbose log is never materialized in
    full — the shared basis for both the error-mining in _extract_error_from_log
    and the log-tail endpoint in handle_inference_log. Returns None for a missing
    path or an unreadable file (OSError); an empty list for an empty file."""
    if not log_path:
        return None
    try:
        with open(log_path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - 64 * 1024))
            data = fh.read()
    except OSError:
        return None
    return data.decode("utf-8", errors="replace").splitlines()


def _extract_error_from_log(log_path: str | None) -> str | None:
    """Pull the meaningful error out of a failed rollout's log so the UI can
    show it directly instead of telling the user to open a file in the cache.

    Subprocess forensics: we only have the log, so we mine the tail for the
    last traceback exception line + its message body. (Recording/teleop run
    in-process and will hand the caught exception's text straight to
    friendly_hint/is_cleanup_error instead — this step is rollout-only.)"""
    lines = _read_log_tail_lines(log_path)
    if lines is None:
        return None
    tail = lines[-50:]
    # Prefer the last exception line + everything after it (the message body).
    exc_idx = next((i for i in range(len(tail) - 1, -1, -1) if _EXC_LINE_RE.match(tail[i])), None)
    if exc_idx is not None:
        snippet = "\n".join(tail[exc_idx:]).strip()
    else:
        non_empty = [ln for ln in tail if ln.strip()]
        snippet = "\n".join(non_empty[-6:]).strip()
    snippet = re.sub(r"\n\s*\n+", "\n", snippet)
    if len(snippet) > 500:
        snippet = snippet[:500].rstrip() + "…"
    return snippet or None


def _classify_outcome(rc: int | None, rollout_started: bool, error_text: str | None) -> str:
    """ok | ran_with_warning | failed.

    A non-zero exit *after* the rollout main loop started, where the error is a
    torque-disable/overload on shutdown, means the skill ran but a motor (usually
    the loaded gripper) complained during cleanup — that's a warning, not a
    failure, so the UI shouldn't call a working run "failed". A mid-run
    disconnect (or a non-zero exit before the loop began) stays a real failure —
    is_cleanup_error deliberately excludes connection-loss markers."""
    if not rc:
        return "ok"
    if rollout_started and is_cleanup_error(error_text):
        return "ran_with_warning"
    return "failed"


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


def _prepare_robot(request: InferenceRequest) -> tuple[list[str], list[str]]:
    """Stage calibrations, run the arm-identity + motor-power preflights, and
    build the `--robot.*` argv for the rollout subprocess.

    This is the robot-TOUCHING part of startup: it opens and releases the
    follower serial bus (read-only identity check + RAM torque-limit priming).
    It runs in the background startup worker AFTER the model download, so a stop
    pressed during the (long) download never reaches here — no bus is opened and
    no register is written. Raises ArmIdentityError on a hard arm mismatch;
    returns (robot_args, warn-but-allow messages)."""
    is_bimanual = request.mode == "bimanual"
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

        identity_warnings: list[str] = []
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
        # Register reset on both buses, sequentially (each opens its own port).
        identity_warnings += _preflight_motor_registers(request.follower_port, left_id)
        identity_warnings += _preflight_motor_registers(request.right_follower_port, right_id)

        return _bimanual_robot_args(request, base, follower_staging), identity_warnings

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

    # Always reset so a previous auto-calibration's torque cap can't linger
    # when the arm was never power-cycled.
    identity_warnings += _preflight_motor_registers(request.follower_port, follower_id)

    return _single_robot_args(request, follower_id), identity_warnings


def _fail_startup(error: str) -> None:
    """Record a background-startup failure (download or preflight — before any
    subprocess exists) as the terminal `_last_result` payload, reusing the exact
    outcome/error/hint contract the subprocess-exit path already exposes so the
    inference page surfaces it the same way (and keeps surfacing it on every
    poll until the next run starts).

    A no-op when a stop already tore the session down (inference_active False):
    the stop wins, and a download that raised while being abandoned must not
    resurrect a phantom failure.

    Session-level, not episode-level: this is the download/preflight/first-spawn
    window, so in eval mode there is nothing to score yet — the eval session is
    dropped and the failure is reported the same way a single run's would be."""
    global inference_active, _inference_proc, _inference_started_at
    global _inference_rollout_started_at, _inference_meta, _last_result, _eval_session
    with _state_lock:
        if not inference_active:
            return
        policy_ref = _inference_meta.get("policy_ref")
        finished_eval = _eval_session
        inference_active = False
        _inference_proc = None
        _inference_started_at = None
        _inference_rollout_started_at = None
        _inference_meta = {}
        _eval_session = None
        _last_result = {
            "inference_active": False,
            "exited": True,
            "exit_code": None,
            "outcome": "failed",
            "error": error,
            "hint": friendly_hint(error),
            "phase": PHASE_ERROR,
            "policy_ref": policy_ref,
            "duration_s": None,
            "log_path": None,
            "started_at": None,
            "rollout_started_at": None,
            "rollout_elapsed_s": 0,
            "elapsed_s": 0,
            **_eval_fields(finished_eval),
        }


def _launch_rollout_subprocess(
    request: InferenceRequest,
    policy_path: str,
    robot_args: list[str],
) -> tuple[subprocess.Popen, IO[str], Path]:
    """Build the argv, open a fresh log file, spawn ONE rollout subprocess and
    seed its stdin.

    The pure "start a rollout process" step, factored out of the startup worker
    so eval mode's second-and-later episodes can respawn without re-downloading
    the model or re-preflighting the arm. Returns (proc, log_handle, log_path);
    the caller owns committing them to the module state and starting the stdout
    pump. Raises on spawn failure (after closing the log handle) — the caller
    decides how to report it, since a first-episode failure fails the session
    while a later one fails just that episode."""
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
    stdin_seed = b"\n\n" if request.mode == "bimanual" else b"\n"
    try:
        proc = subprocess.Popen(
            _build_rollout_cmd(request, policy_path, robot_args),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
    except Exception:
        with contextlib.suppress(Exception):
            log_handle.close()
        raise
    try:
        assert proc.stdin is not None
        proc.stdin.write(stdin_seed)
        proc.stdin.flush()
        proc.stdin.close()
    except Exception as exc:
        logger.warning("Failed to seed stdin for inference subprocess: %s", exc)
    return proc, log_handle, log_path


def _run_inference_startup(request: InferenceRequest, cancel_event: threading.Event) -> None:
    """Background startup sequence for one rollout: download the model (with byte
    progress), preflight the arm, then spawn the rollout subprocess.

    Runs off the request thread so POST /start-inference returns immediately and
    the UI lands on the inference page while the (possibly multi-minute) Hub
    download runs there with a progress bar. Ordered download → preflight → spawn
    so a stop pressed DURING the download never opens the serial bus or spawns a
    subprocess ("no robot touched"). snapshot_download can't be interrupted
    mid-flight, so a stop during the download abandons this worker: the download
    finishes into the HF cache (cached for next time) and the worker bails at the
    next cancel check without preflighting or spawning. Terminal download/
    preflight failures flow through _fail_startup into the shared outcome/error/
    hint status machinery."""
    global _inference_proc, _inference_rollout_started_at, _inference_meta

    # 1. Resolve/download the policy. A Hub ref streams byte progress into the
    #    meta; a local dir returns instantly (no downloading_model phase, no
    #    robot touched yet).
    try:
        policy_path = _resolve_policy_path(request.policy_ref, report=_report_download_progress)
    except Exception as exc:
        logger.exception("Inference model download failed")
        _fail_startup(f"Failed to download the model: {exc}")
        return
    # Stop during the download → abandon (stop already set the state idle).
    if cancel_event.is_set():
        logger.info("Inference startup abandoned during model download (stop requested)")
        return

    # 2. Preflight + stage the arm (opens the serial bus). This is the first
    #    robot-touching step, deliberately AFTER the download.
    try:
        robot_args, identity_warnings = _prepare_robot(request)
    except ArmIdentityError as exc:
        # The connected arm doesn't match its assigned calibration; the message
        # is already user-facing.
        _fail_startup(str(exc))
        return
    except Exception as exc:
        logger.exception("Failed to prepare robot for inference")
        _fail_startup(f"Failed to start inference: {exc}")
        return
    if cancel_event.is_set():
        logger.info("Inference startup abandoned after preflight (stop requested)")
        return

    # 3. Spawn the rollout subprocess (episode 1 of 1, or of N in eval mode).
    #    In eval mode, cache what the later episodes reuse verbatim — the model
    #    is downloaded once and the arm preflight runs once per SESSION, not per
    #    episode (which is what makes N episodes cheap to chain).
    with _state_lock:
        if _eval_session is not None:
            _eval_session.policy_path = policy_path
            _eval_session.robot_args = list(robot_args)

    try:
        proc, log_handle, log_path = _launch_rollout_subprocess(request, policy_path, robot_args)
    except Exception as exc:
        logger.exception("Failed to spawn rollout subprocess")
        _fail_startup(f"Failed to start inference: {exc}")
        return

    # Commit the subprocess under the lock, re-checking the cancel flag: a stop
    # that raced the spawn must NOT leave a live subprocess driving the arm.
    with _state_lock:
        abandoned = cancel_event.is_set() or not inference_active
        if not abandoned:
            _inference_proc = proc
            _inference_rollout_started_at = None
            # Carry forward any phase the not-yet-started pump could set later;
            # the download phase is behind us, so `starting` is the floor.
            carried_phase = _inference_meta.get("phase") or PHASE_STARTING
            if carried_phase == PHASE_DOWNLOADING_MODEL:
                carried_phase = PHASE_STARTING
            meta: dict[str, Any] = {
                "policy_ref": request.policy_ref,
                # The RESOLVED local checkpoint dir (policy_ref can be a Hub ref,
                # fragile for path comparisons) — read by inference_in_use_path so
                # models.delete_local_model can refuse deleting it mid-run.
                "policy_path": policy_path,
                "duration_s": request.duration_s,
                "log_path": str(log_path),
                "phase": carried_phase,
            }
            # Warn-but-allow arm-identity findings, surfaced once via the status
            # payload now that the POST returned before the preflight ran.
            if identity_warnings:
                meta["warning"] = " ".join(identity_warnings)
            _inference_meta = meta

    if abandoned:
        # Stopped during/just after the spawn — kill the subprocess we just
        # started and leave the (already idle) state alone.
        logger.info("Inference startup abandoned after spawn (stop requested); killing subprocess")
        with contextlib.suppress(Exception):
            proc.terminate()
        with contextlib.suppress(Exception):
            proc.wait(timeout=5)
        with contextlib.suppress(Exception):
            log_handle.close()
        return

    # Start the stdout pump only after committing, so it never advances the phase
    # of a subprocess we might have abandoned above.
    threading.Thread(
        target=_pump_stdout,
        args=(proc, log_handle),
        name="inference-stdout-pump",
        daemon=True,
    ).start()
    logger.info("Inference started: pid=%s policy=%s", proc.pid, policy_path)


def handle_start_inference(request: InferenceRequest) -> dict[str, Any]:
    """Validate the request cheaply and hand the heavy startup (model download →
    arm preflight → subprocess spawn) to a background worker, returning
    immediately.

    Returns a dict — the route layer turns it into a JSON response or
    HTTPException as appropriate. Only cheap, synchronous checks stay here
    (mutex, arm-count guard, policy-ref shape) so a 4xx still surfaces in the
    launch modal; the multi-minute Hub download moves off the request thread so
    the UI lands on the inference page and shows download progress there."""
    global inference_active, _inference_started_at, _inference_meta, _inference_cancel
    global _last_result, _inference_startup_thread, _eval_session

    # Mutex with every other feature that drives the same serial bus (see
    # CLAUDE.md's "State model & mutual exclusion").
    from . import (
        auto_calibrate as _auto_calibrate,
        calibrate as _calibrate,
        record as _record,
        teleoperate as _teleoperate,
        wiggle as _wiggle,
    )

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
        if _inference_startup_thread is not None and _inference_startup_thread.is_alive():
            # A previous session was stopped while its startup worker was
            # inside _prepare_robot (already touching hardware) or still
            # unwinding just after — inference_active is already False, but
            # the worker itself hasn't exited yet. Starting a new session now
            # would open the same serial port out from under it. Refuse until
            # it's actually gone.
            return {
                "success": False,
                "status_code": 409,
                "message": "The previous session is still shutting down. Try again in a few seconds.",
            }
        if _calibrate.calibration_is_active():
            return {
                "success": False,
                "status_code": 409,
                "message": "Calibration is currently active. Stop it first.",
            }
        if _auto_calibrate.auto_calibration_is_active():
            return {
                "success": False,
                "status_code": 409,
                "message": "Auto-calibration is currently active. Stop it first.",
            }
        if _wiggle.wiggle_active:
            return {
                "success": False,
                "status_code": 409,
                "message": "A gripper wiggle is currently in progress. Wait for it to finish.",
            }
        # Claim the slot now so a concurrent caller losing the race sees us, and
        # seed the meta + timer so the phase is visible from the very first
        # status poll (the download runs on the inference page — the UI must be
        # able to name that wait before the subprocess even exists). A fresh
        # cancel Event lets stop() abandon the pre-subprocess window.
        inference_active = True
        _inference_started_at = time.time()
        _inference_cancel = threading.Event()
        cancel_event = _inference_cancel
        _inference_meta = {"phase": PHASE_STARTING, "policy_ref": request.policy_ref}
        # A new run supersedes the previous run's terminal payload — status
        # polls must reflect THIS session from the first tick.
        _last_result = None
        # Eval mode is decided once, here, and clamped: episodes > 1 seeds the
        # session bookkeeping so the very first status poll already reports
        # "episode 1 / N". A count of 1 leaves `_eval_session` None, which is
        # what keeps the historical single-rollout flow untouched.
        episodes = clamp_eval_episodes(request.eval_episodes)
        _eval_session = _EvalSession(request=request, episodes_total=episodes) if episodes > 1 else None

    def _release_slot() -> None:
        global inference_active, _inference_started_at, _inference_cancel, _inference_meta
        global _eval_session
        with _state_lock:
            inference_active = False
            _inference_started_at = None
            _inference_cancel = None
            _inference_meta = {}
            _eval_session = None

    # Arm-count guard: reject a single-arm checkpoint on a bimanual robot (and
    # vice versa) BEFORE spawning the worker, where the shape mismatch would
    # otherwise crash unexplained. Cheap (no I/O) — defers to the subprocess when
    # the checkpoint doesn't expose observation.state.
    mismatch = _arm_count_mismatch(request.mode, request.checkpoint_state_dim)
    if mismatch is not None:
        _release_slot()
        return {"success": False, "status_code": 409, "message": mismatch}

    # Cheap policy-ref shape check so a malformed ref 4xxs in the modal instead
    # of failing later on the inference page (one is_dir stat, no network).
    if not _policy_ref_is_valid(request.policy_ref):
        _release_slot()
        return {
            "success": False,
            "status_code": 400,
            "message": f"Unrecognised policy ref: {request.policy_ref!r}",
        }

    # Backend camera previews hold the cv2 devices the rollout subprocess is about
    # to open. Released here — after the cheap guards above, so a rejected request
    # doesn't needlessly kill the modal's previews, and while `inference_active`
    # is already True so /camera-preview 409s instead of re-acquiring a device.
    camera_preview_manager.stop_all()

    # Everything heavy (download, preflight, spawn) runs off the request thread.
    # Tracked so a later start can tell whether a stopped session's worker is
    # still alive (see the is_alive() guard above) instead of racing it.
    worker = threading.Thread(
        target=_run_inference_startup,
        args=(request, cancel_event),
        name="inference-startup",
        daemon=True,
    )
    _inference_startup_thread = worker
    worker.start()
    return {"success": True, "message": "Inference starting"}


def inference_in_use_path() -> str | None:
    """The RESOLVED local policy path the running inference is reading, or None
    when no inference is active.

    The meta's ``policy_ref`` can be a Hub ref (``user/repo@root``), which is
    fragile for path comparisons — this is the local directory
    ``_resolve_policy_path`` returned, captured at start. Guarded by
    _state_lock (short critical section). Consumed by ``models._model_in_use``
    so deleting a checkpoint a live inference is reading is refused."""
    with _state_lock:
        if not inference_active:
            return None
        return _inference_meta.get("policy_path")


def _go_idle_locked() -> None:
    """Drop every per-session global back to the idle shape.

    Caller must hold `_state_lock`. Does NOT touch `_last_result` — whether a
    teardown leaves a terminal payload behind is the caller's decision."""
    global inference_active, _inference_proc, _inference_started_at
    global _inference_rollout_started_at, _inference_meta, _eval_session
    inference_active = False
    _inference_proc = None
    _inference_started_at = None
    _inference_rollout_started_at = None
    _inference_meta = {}
    _eval_session = None


def _abort_eval_locked(ev: _EvalSession) -> None:
    """End an eval session early and leave the ABORTED terminal payload behind.

    Caller must hold `_state_lock`. Partial tally, and deliberately NO accuracy:
    a session the user cut short says nothing about the policy's success rate
    over N episodes, so claiming one would be a lie. The episode that was running
    when the abort landed is simply not scored."""
    global _last_result
    finished_meta = _inference_meta
    finished_started = _inference_started_at
    _go_idle_locked()
    _last_result = {
        "inference_active": False,
        "exited": True,
        "exit_code": None,
        "outcome": "ok",
        "error": ev.error,
        "hint": ev.hint,
        "phase": PHASE_ABORTED,
        "policy_ref": finished_meta.get("policy_ref"),
        "duration_s": finished_meta.get("duration_s"),
        "log_path": finished_meta.get("log_path"),
        "started_at": finished_started,
        "rollout_started_at": None,
        "rollout_elapsed_s": 0,
        "elapsed_s": 0,
        **_eval_fields(ev),
    }


def handle_stop_inference() -> dict[str, Any]:
    """Abort the WHOLE session — single run or eval.

    In eval mode this is the session-level stop, not the per-episode one: it
    ends the run wherever it is (mid-episode or parked in a reset) and reports
    the partial tally under the `aborted` phase. Ending only the current episode
    while keeping the session alive is `handle_stop_episode`."""
    with _state_lock:
        session_active = inference_active
        orphaned_worker = _inference_startup_thread if not session_active else None

    if not session_active:
        if orphaned_worker is None or not orphaned_worker.is_alive():
            return {"success": False, "status_code": 409, "message": "No inference is active"}
        # A previous stop already fired (inference_active is False), but that
        # session's startup worker is still stuck inside _prepare_robot with no
        # way to be interrupted mid-call. This is the "press Stop again" gesture
        # (mirrors teleoperate.py's second stop): bounded-wait for it and report
        # honestly, instead of repeating a blanket "nothing to stop" that hides
        # a worker still touching the serial bus. Joined outside _state_lock so
        # a slow/stuck worker can't stall other requests (status polls, a
        # concurrent start) for the whole timeout.
        orphaned_worker.join(timeout=_STARTUP_STOP_JOIN_TIMEOUT_S)
        if orphaned_worker.is_alive():
            return {
                "success": True,
                "shutting_down": True,
                "message": (
                    "The previous session is still shutting down "
                    f"(waited {_STARTUP_STOP_JOIN_TIMEOUT_S:.0f}s more). Try again shortly."
                ),
            }
        return {"success": True, "message": "The previous session has now finished shutting down."}

    with _state_lock:
        # Signal the background startup worker to abandon: this is the only way
        # to stop during the pre-subprocess window (Hub download / arm
        # preflight), where there's no process to terminate.
        if _inference_cancel is not None:
            _inference_cancel.set()
        proc = _inference_proc
        ev = _eval_session
        # Surface the stop as its own phase so a status poll racing the
        # terminate/wait below sees "stopping" rather than a stale "running".
        if _inference_meta:
            _inference_meta["phase"] = PHASE_STOPPING

        if proc is None:
            # Stop pressed with no live subprocess: either before the first one
            # spawned (during the model download / arm preflight — no policy has
            # driven the robot, and the orphaned startup worker bails at its next
            # cancel check), or, in eval mode, while parked in a reset between
            # episodes. Either way there's nothing to terminate.
            if ev is not None:
                _abort_eval_locked(ev)
                return {"success": True, "message": "Evaluation aborted"}
            _go_idle_locked()
            return {"success": True, "message": "Inference stopped"}

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
        # Re-read: a status poll could have finalised the exit (and, in eval
        # mode, even finished the session) while we were outside the lock.
        ev = _eval_session
        if ev is not None:
            _abort_eval_locked(ev)
            return {"success": True, "message": "Evaluation aborted"}
        _go_idle_locked()
    return {"success": True, "message": "Inference stopped"}


def handle_stop_episode() -> dict[str, Any]:
    """End the CURRENT eval episode early and score it a SUCCESS.

    This is the "the robot did the task — next" button. It terminates the
    episode's subprocess exactly the way `handle_stop_inference` does (so the
    follower still eases back to its start pose via
    `--return_to_initial_position`) but leaves the SESSION standing: the exit
    finalisation in `handle_inference_status` then parks it in the reset phase
    (or finishes it, if that was the last episode).

    Eval-only by design: a single-episode run has no tally to record a success
    into, so it gets a 409 rather than a silent no-op."""
    with _state_lock:
        if not inference_active or _eval_session is None:
            return {
                "success": False,
                "status_code": 409,
                "message": "No evaluation episode is running",
            }
        proc = _inference_proc
        if proc is None:
            # Either still starting up, or already parked in a reset — there is
            # no running episode to call a success.
            return {
                "success": False,
                "status_code": 409,
                "message": "No evaluation episode is running",
            }
        # Set BEFORE terminating so the flag is always visible to whichever
        # thread finalises the exit. If a status poll beats us to the exit (the
        # episode hit its duration in this exact window) it clears
        # `_inference_proc` under this same lock, and the guard above turns the
        # next call into a 409 — the two orders can't both score the episode.
        _eval_session.stop_requested = True

    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Eval episode did not exit in 5s; killing")
            proc.kill()
            proc.wait()
    except Exception as exc:
        logger.exception("Stop eval episode: %s", exc)
    # The verdict is recorded in ONE place — the exit finalisation in
    # handle_inference_status — so a poll that raced us can't double-score.
    return {"success": True, "message": "Episode recorded as a success"}


def handle_next_episode() -> dict[str, Any]:
    """Leave the reset phase and spawn the next eval episode.

    The reset between episodes is explicitly user-ended (no auto-timer, unlike
    recording's): rearranging a bench scene has no reason to be rushed. Reuses
    the session's already-resolved policy path and already-preflighted
    `--robot.*` args, so continuing costs one subprocess spawn — no re-download,
    no second arm-identity pass, and the cameras stay owned by the session
    throughout."""
    global _inference_proc, _inference_started_at, _inference_rollout_started_at, _inference_meta

    with _state_lock:
        ev = _eval_session
        if not inference_active or ev is None:
            return {"success": False, "status_code": 409, "message": "No evaluation is active"}
        if _inference_meta.get("phase") != PHASE_RESETTING:
            return {
                "success": False,
                "status_code": 409,
                "message": "The evaluation is not waiting for a reset",
            }
        if ev.policy_path is None:
            # Only reachable if the startup worker never got far enough to cache
            # the resolved path, which also means no episode ever ran.
            return {
                "success": False,
                "status_code": 409,
                "message": "The evaluation session has no prepared policy to run",
            }
        request, policy_path, robot_args = ev.request, ev.policy_path, list(ev.robot_args)
        carried_ref = _inference_meta.get("policy_ref")
        carried_warning = _inference_meta.get("warning")

    try:
        proc, log_handle, log_path = _launch_rollout_subprocess(request, policy_path, robot_args)
    except Exception as exc:
        logger.exception("Failed to spawn the next eval episode")
        # Spawning is the cheap part; a failure here is a session-level problem
        # (a broken interpreter/env), not something the next reset can fix.
        _fail_startup(f"Failed to start the next episode: {exc}")
        return {"success": False, "status_code": 500, "message": f"Failed to start the next episode: {exc}"}

    with _state_lock:
        if not inference_active or _eval_session is None:
            # Aborted while we were spawning — kill what we just started rather
            # than leave a policy driving the arm for a dead session.
            logger.info("Next eval episode abandoned right after spawn; killing subprocess")
            with contextlib.suppress(Exception):
                proc.terminate()
            with contextlib.suppress(Exception):
                proc.wait(timeout=5)
            with contextlib.suppress(Exception):
                log_handle.close()
            return {"success": False, "status_code": 409, "message": "No evaluation is active"}
        _inference_proc = proc
        # Both timers restart per episode: `elapsed_s` is this episode's setup
        # time and `rollout_elapsed_s` its rollout time, so the dialog's clock
        # and the frontend's past-duration safety net both measure the EPISODE,
        # not the (much longer) session.
        _inference_started_at = time.time()
        _inference_rollout_started_at = None
        # Clear the previous episode's crash banner — the user chose to continue.
        _eval_session.error = None
        _eval_session.hint = None
        meta: dict[str, Any] = {
            "policy_ref": carried_ref or request.policy_ref,
            "policy_path": policy_path,
            "duration_s": request.duration_s,
            "log_path": str(log_path),
            "phase": PHASE_STARTING,
        }
        if carried_warning:
            meta["warning"] = carried_warning
        _inference_meta = meta
        episode_index = _eval_session.episode_index

    threading.Thread(
        target=_pump_stdout,
        args=(proc, log_handle),
        name="inference-stdout-pump",
        daemon=True,
    ).start()
    logger.info("Eval episode %s started: pid=%s", episode_index, proc.pid)
    return {"success": True, "message": f"Episode {episode_index} starting"}


# Tail cap for the inference-log endpoint: last N lines, bounded so a very long
# run's log can never be shipped to the browser in full.
_INFERENCE_LOG_MAX_LINES = 500


def _resolve_inference_log_path() -> Path | None:
    """Path of the current (or most-recent) run's inference log, or None.

    Prefers the active session's `_inference_meta["log_path"]`; when no session
    is active (or its meta lacks a path), falls back to the newest `*.log` under
    the inference_logs dir so a just-finished run's log is still viewable."""
    with _state_lock:
        meta_path = _inference_meta.get("log_path")
    if meta_path:
        p = Path(meta_path)
        if p.is_file():
            return p
    log_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / "inference_logs"
    try:
        logs = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime)
    except OSError:
        return None
    return logs[-1] if logs else None


def handle_inference_log(max_lines: int = _INFERENCE_LOG_MAX_LINES) -> dict[str, Any]:
    """Return the tail of the active/most-recent inference log.

    Read-only and bounded: at most `max_lines` trailing lines. Never raises —
    a missing/unreadable log yields empty text, so the route stays 200 even
    before a run has produced any output."""
    path = _resolve_inference_log_path()
    if path is None:
        return {"logs": "", "log_path": None}
    # Bounded read: only the last ~64 KB is decoded (shared with the error-mining
    # path), which holds every line a rollout log this size produces. A
    # missing/unreadable file yields None -> empty text, keeping the route 200.
    lines = _read_log_tail_lines(str(path))
    if lines is None:
        return {"logs": "", "log_path": str(path)}
    tail = lines[-max_lines:] if max_lines > 0 else lines
    return {"logs": "\n".join(tail), "log_path": str(path)}


def _finalise_eval_episode_locked(rc: int | None, ev: _EvalSession) -> dict[str, Any]:
    """Score one finished eval episode and either park or finish the session.

    Caller must hold `_state_lock`. This is the SINGLE place an episode verdict
    is recorded — `handle_stop_episode` only sets the flag and terminates, so a
    status poll racing the stop can't double-score.

    Not finishing the session means keeping `inference_active` True through the
    reset: the session still owns the inference slot (recording/teleop stay
    blocked) and still owns the cameras (previews stay 409'd), which is exactly
    what lets the next episode spawn straight into a ready rig."""
    global _inference_proc, _inference_rollout_started_at, _inference_meta, _last_result

    finished_meta = _inference_meta
    finished_started = _inference_started_at
    rollout_started = _inference_rollout_started_at is not None
    # Mine the log only when the exit was non-zero; a stop we asked for exits
    # non-zero via our own SIGTERM, so the snippet is discarded unless the
    # verdict actually turns out to be a crash.
    error = _extract_error_from_log(finished_meta.get("log_path")) if rc else None
    verdict = classify_episode(rc, ev.stop_requested, rollout_started, error)
    # Read the index BEFORE appending — the property is derived from the result
    # count, so afterwards it names the NEXT episode.
    logger.info(
        "Eval episode %s/%s exited rc=%s -> %s",
        ev.episode_index,
        ev.episodes_total,
        rc,
        verdict,
    )
    ev.results.append(verdict)
    ev.stop_requested = False
    if verdict == EPISODE_ERROR:
        ev.error = error
        ev.hint = friendly_hint(error)
    else:
        ev.error = None
        ev.hint = None

    _inference_proc = None
    _inference_rollout_started_at = None

    if len(ev.results) < ev.episodes_total:
        # More to go: park in the reset phase and wait for the user to rearrange
        # the scene and POST /inference-next-episode. No auto-timer.
        _inference_meta = {**finished_meta, "phase": PHASE_RESETTING}
        return {
            **_eval_fields(ev),
            "inference_active": True,
            "started_at": finished_started,
            "rollout_started_at": None,
            "elapsed_s": 0,
            "rollout_elapsed_s": 0,
            "duration_s": finished_meta.get("duration_s"),
            "policy_ref": finished_meta.get("policy_ref"),
            "log_path": finished_meta.get("log_path"),
            "phase": PHASE_RESETTING,
            "download_bytes_done": None,
            "download_bytes_total": None,
            "download_percent": None,
            "warning": finished_meta.get("warning"),
            # Populated only for a CRASHED episode — the reset screen doubles as
            # the "this one broke, continue or abort?" screen.
            "error": ev.error,
            "hint": ev.hint,
        }

    # Last episode: the session is done. Claim the accuracy and release the slot.
    _go_idle_locked()
    _last_result = {
        "inference_active": False,
        "exited": True,
        "exit_code": rc,
        "outcome": "ok",
        "error": ev.error,
        "hint": ev.hint,
        "phase": PHASE_FINISHED,
        "policy_ref": finished_meta.get("policy_ref"),
        "duration_s": finished_meta.get("duration_s"),
        "log_path": finished_meta.get("log_path"),
        "started_at": finished_started,
        "rollout_started_at": None,
        "rollout_elapsed_s": 0,
        "elapsed_s": 0,
        **_eval_fields(ev, accuracy=eval_accuracy(ev.results)),
    }
    logger.info("Evaluation finished: %s accuracy=%s", ev.results, _last_result["accuracy"])
    return dict(_last_result)


def handle_inference_status() -> dict[str, Any]:
    global inference_active, _inference_proc, _inference_started_at
    global _inference_rollout_started_at, _inference_meta, _last_result

    # Finalise state lazily if the subprocess died on its own.
    with _state_lock:
        proc = _inference_proc
        # True only while idle: a previous session's startup worker (see
        # _inference_startup_thread) is still alive after its stop already
        # fired, so a poller isn't looking at a status indistinguishable from
        # true idle while the worker still holds the serial bus.
        shutting_down = (
            not inference_active
            and _inference_startup_thread is not None
            and _inference_startup_thread.is_alive()
        )
        # Idle with a recorded terminal result (a subprocess exit finalised
        # below, or a download/preflight failure from _fail_startup): keep
        # returning that payload verbatim until the next start clears it.
        # Idempotence matters — several surfaces poll this endpoint
        # concurrently, and a consume-once payload lets one poller swallow the
        # error the user needed to see (see _last_result's declaration).
        if proc is None and not inference_active and _last_result is not None:
            return {**_last_result, "shutting_down": shutting_down}
        if proc is not None and proc.poll() is not None:
            rc = proc.returncode
            if _eval_session is not None:
                # Eval mode: this exit ends an EPISODE, which only ends the
                # SESSION if it was the last one.
                return _finalise_eval_episode_locked(rc, _eval_session)
            logger.info("Inference subprocess exited rc=%s", rc)
            finished_meta = _inference_meta
            finished_started = _inference_started_at
            finished_rollout_started = _inference_rollout_started_at
            # Terminal phase: a clean exit (rc 0, including a stop we asked for)
            # is `stopped`; any non-zero code is `error`. The prior phase in
            # `finished_meta` (e.g. "stopping" from a stop request) is
            # superseded — the subprocess has actually gone now.
            terminal_phase = PHASE_STOPPED if rc == 0 else PHASE_ERROR
            inference_active = False
            _inference_proc = None
            _inference_started_at = None
            _inference_rollout_started_at = None
            _inference_meta = {}
            # On a non-zero exit, mine the real error out of the log so the UI
            # can show it directly (hint + snippet) instead of sending the user
            # digging through the cache. `outcome` further distinguishes a true
            # failure from a run that worked but tripped a noisy shutdown/cleanup
            # warning (see _classify_outcome) so the false-failure isn't reported
            # as a hard error.
            error = _extract_error_from_log(finished_meta.get("log_path")) if rc else None
            outcome = _classify_outcome(rc, finished_rollout_started is not None, error)
            _last_result = {
                "inference_active": False,
                "exited": True,
                "exit_code": rc,
                "outcome": outcome,
                "error": error,
                "hint": friendly_hint(error),
                "phase": terminal_phase,
                "policy_ref": finished_meta.get("policy_ref"),
                "duration_s": finished_meta.get("duration_s"),
                "log_path": finished_meta.get("log_path"),
                "started_at": finished_started,
                "rollout_started_at": finished_rollout_started,
                "rollout_elapsed_s": 0,
                "elapsed_s": 0,
                **_eval_fields(None),
            }
            return {**_last_result, "shutting_down": shutting_down}
        elapsed = (time.time() - _inference_started_at) if _inference_started_at else 0
        rollout_elapsed = time.time() - _inference_rollout_started_at if _inference_rollout_started_at else 0
        return {
            **_eval_fields(_eval_session),
            # A crashed episode parks the eval session in the reset phase with
            # its mined error still on show, so the user can decide to continue
            # or abort. Null on every other live payload.
            "error": _eval_session.error if _eval_session else None,
            "hint": _eval_session.hint if _eval_session else None,
            "inference_active": inference_active,
            "started_at": _inference_started_at,
            "rollout_started_at": _inference_rollout_started_at,
            "elapsed_s": elapsed,
            "rollout_elapsed_s": rollout_elapsed,
            "duration_s": _inference_meta.get("duration_s"),
            "policy_ref": _inference_meta.get("policy_ref"),
            "log_path": _inference_meta.get("log_path"),
            # None when idle (no session has seeded a meta yet); the frontend
            # treats an absent phase as "no active startup to narrate".
            "phase": _inference_meta.get("phase"),
            # Byte progress of the Hub model download, populated only during the
            # downloading_model phase (all None outside it / for a local
            # checkpoint). download_percent is None while the total is still
            # unknown → the UI shows an indeterminate bar.
            "download_bytes_done": _inference_meta.get("download_bytes_done"),
            "download_bytes_total": _inference_meta.get("download_bytes_total"),
            "download_percent": _inference_meta.get("download_percent"),
            # Warn-but-allow arm-identity finding, surfaced once the run is up
            # (the preflight now runs in the background, after the POST returned).
            "warning": _inference_meta.get("warning"),
            "shutting_down": shutting_down,
        }
