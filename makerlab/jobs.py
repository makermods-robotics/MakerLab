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

"""Job lifecycle and registry for trainings (and, in future, other long-running
work). One JobRunner instance owns one subprocess; the JobRegistry owns the
overall state, including history persisted to disk under outputs/train/."""

from __future__ import annotations

import builtins
import contextlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Literal, Protocol, runtime_checkable

import psutil
from pydantic import BaseModel

from .train import TrainingRequest
from .utils.config import is_valid_robot_name
from .utils.hf_auth import shared_hf_api
from .utils.subprocess_env import utf8_child_env

logger = logging.getLogger(__name__)


JobState = Literal["running", "done", "failed", "interrupted"]


class JobTarget(BaseModel):
    """Where a job should run. `local` ⇒ LocalJobRunner. `hf_cloud` requires
    a non-empty `flavor` from HfApi.list_jobs_hardware()."""

    runner: Literal["local", "hf_cloud"] = "local"
    flavor: str | None = None


class TrainingMetrics(BaseModel):
    current_step: int = 0
    total_steps: int = 0
    current_loss: float | None = None
    current_lr: float | None = None
    grad_norm: float | None = None
    eta_seconds: float | None = None


class LogLine(BaseModel):
    timestamp: float
    message: str


class JobRecord(BaseModel):
    id: str
    name: str
    # User-editable display alias set via JobRegistry.rename. Display-only:
    # the immutable identity (id / output_dir / hf_repo_id) never changes on
    # rename, so resume lineage, imported-model dedup, and remote HF/W&B
    # names stay intact. None ⇒ the UI falls back to `name`.
    display_name: str | None = None
    state: JobState
    config: TrainingRequest
    output_dir: str
    started_at: float
    ended_at: float | None = None
    exit_code: int | None = None
    error_message: str | None = None
    metrics: TrainingMetrics = TrainingMetrics()
    runner: Literal["local", "hf_cloud", "imported"] = "local"
    # PID of the detached subprocess (local runner only); survives uvicorn
    # --reload so a fresh registry can re-attach by tailing the log file.
    process_pid: int | None = None
    # HF Jobs identifiers (hf_cloud runner only)
    hf_job_id: str | None = None
    hf_flavor: str | None = None
    hf_repo_id: str | None = None
    hf_job_url: str | None = None
    # Captured from training stdout the first time wandb prints the run URL.
    wandb_run_url: str | None = None
    # Number of checkpoints currently visible (local: filesystem; cloud:
    # Hub repo). Filled in by JobRegistry.list/get; persisted as zero.
    checkpoint_count: int = 0


class JobCheckpoint(BaseModel):
    """One checkpoint produced by a training job.

    `ref` is opaque to the frontend; the inference handler resolves it back
    to a usable `--policy.path` value (a local dir for both sources, after
    snapshot_download for hub refs)."""

    step: int
    source: Literal["local", "hub"]
    ref: str


class MetricsHistoryPoint(BaseModel):
    """One (step, metrics) sample reconstructed from a job's log.jsonl.

    Used by GET /jobs/{id}/metrics-history to seed the monitoring charts.
    A point is emitted for each log line that carried a `step: ... loss: ...`
    payload (the log-freq lines from lerobot). Tqdm progress lines are
    skipped — they carry step + ETA but no loss/lr/grdn."""

    step: int
    loss: float | None = None
    lr: float | None = None
    grad_norm: float | None = None


def _process_isolation_kwargs() -> dict[str, object]:
    """Popen kwargs that keep a spawned child alive through signals sent to
    this process (uvicorn --reload restarting its worker on a .py file
    change, or a user hitting Ctrl+C on the server's console) so training
    survives and the next worker re-attaches via TailingJobRunner using
    job.json's pid.

    start_new_session=True (setsid) is how POSIX does this, but it is
    silently discarded on Windows: CPython's Windows _execute_child names
    the parameter unused_start_new_session and never passes it to
    CreateProcess, so it provides zero isolation there — confirmed by
    reading that source directly, not inferred. The Windows equivalent is
    creationflags=CREATE_NEW_PROCESS_GROUP, which Microsoft's docs state
    makes a process ignore a console-wide Ctrl+C.

    NOTE: unlike the POSIX path, this hasn't been verified by observing a live
    Ctrl+C actually fail to cascade on Windows (that requires a real
    interactive console/session, which wasn't available when this was
    written). Re-verify manually on a real Windows terminal before relying on
    it under load.
    """
    if platform.system() == "Windows":
        # getattr, not a direct attribute reference: CREATE_NEW_PROCESS_GROUP
        # only exists on the Windows build of the subprocess module
        # (CPython docs mark it "Availability: Windows"). A direct reference
        # would raise AttributeError under a mocked-platform unit test
        # running on our POSIX CI, even though this branch never actually
        # executes there in production (platform.system() isn't mocked
        # there). 0x00000200 is the documented win32 constant value.
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        return {"creationflags": creationflags}
    return {"start_new_session": True}


def _pid_alive(pid: int) -> bool:
    """Return True if a process with this PID exists. Cross-platform via psutil,
    which handles PID liveness correctly on both POSIX and Windows."""
    return psutil.pid_exists(pid)


@runtime_checkable
class JobRunner(Protocol):
    """Backend interface for running one job. LocalJobRunner is the only impl
    today; remote runners (SSH, Slurm) drop in here later. @runtime_checkable
    lets `isinstance(r, JobRunner)` work in tests / sanity checks."""

    def start(self, job_id: str, config: TrainingRequest, output_dir: str) -> None: ...
    def stop(self) -> None: ...
    def is_running(self) -> bool: ...
    def returncode(self) -> int | None: ...
    def stream_log_lines(self) -> list[LogLine]: ...
    def wandb_run_url(self) -> str | None: ...


# tqdm progress: "Training:   1%|▏         | 125/10000 [02:02<2:36:10,  1.05step/s]"
_TQDM_RE = re.compile(r"Training:\s*\d+%[^|]*\|[^|]*\|\s*(\d+)/(\d+)\s*\[(?:[\d:]+)<([\d:]+)")

# Wandb prints something like "wandb: 🚀 View run at https://wandb.ai/<entity>/<project>/runs/<id>"
# when it boots. We capture the first URL of that shape we see.
_WANDB_URL_RE = re.compile(r"https://wandb\.ai/[^\s/]+/[^\s/]+/runs/[A-Za-z0-9]+")


def extract_wandb_run_url(line: str) -> str | None:
    match = _WANDB_URL_RE.search(line)
    return match.group(0) if match else None


def _parse_duration(s: str) -> float | None:
    """Parse tqdm's HH:MM:SS or MM:SS into seconds. Returns None on '?'."""
    parts = s.split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except ValueError:
        return None
    return None


def parse_metrics_into(line: str, metrics: TrainingMetrics, resume_total: int | None = None) -> None:
    """Update `metrics` in-place from one stdout line.

    Two complementary sources:
      * tqdm progress for current_step + total_steps + ETA (~1s cadence).
      * 'INFO ... step:N smpl:... loss:X grdn:Y lr:Z ...' for loss/lr/grdn
        (only at log_freq cadence, default every 250 steps).

    `resume_total` is the run's full step target for a *resumed* run (None for a
    fresh run). On resume lerobot's tqdm bar counts only the remaining window
    (0 → steps−checkpoint), so the raw bar understates the true global step; we
    rebase it to `checkpoint + bar = resume_total − remaining_total + bar` so
    the UI shows e.g. 150/200 instead of 50/100. The `step:N` log line already
    carries the true global step, so it needs no rebasing.
    """
    try:
        tqdm_match = _TQDM_RE.search(line)
        if tqdm_match:
            try:
                tqdm_step = int(tqdm_match.group(1))
                total = int(tqdm_match.group(2))
                if resume_total is not None and total > 0:
                    metrics.current_step = resume_total - total + tqdm_step
                    metrics.total_steps = resume_total
                else:
                    metrics.current_step = tqdm_step
                    if total > 0:
                        metrics.total_steps = total
                eta = _parse_duration(tqdm_match.group(3))
                if eta is not None:
                    metrics.eta_seconds = eta
            except (ValueError, IndexError):
                pass

        if "step:" in line and "loss:" in line:
            with contextlib.suppress(ValueError):
                metrics.current_step = int(line.split("step:")[1].split()[0].replace(",", ""))
            with contextlib.suppress(ValueError):
                metrics.current_loss = float(line.split("loss:")[1].split()[0])
            if "lr:" in line:
                with contextlib.suppress(ValueError):
                    metrics.current_lr = float(line.split("lr:")[1].split()[0])
            if "grdn:" in line:
                with contextlib.suppress(ValueError):
                    metrics.grad_norm = float(line.split("grdn:")[1].split()[0])

    except Exception as exc:
        logger.debug("Error parsing log line %r: %s", line, exc)


def _resume_total_steps(config: TrainingRequest) -> int | None:
    """The full step target to rebase a resumed run's tqdm bar against (see
    parse_metrics_into). None for a fresh run — its bar is already global."""
    return config.steps if config.resume else None


def _read_log_metrics(path: Path, resume_total: int | None) -> builtins.list[MetricsHistoryPoint]:
    """Parse one job's log.jsonl into (step, loss, lr, grad_norm) points.

    Feed every line through ONE accumulator rather than a fresh one per line.
    lerobot formats the log-line step with format_big_number, so at >=1000 steps
    its token becomes "1K"/"2K" and int() can't parse it; a fresh-per-line parse
    would leave current_step at 0 and silently drop every point past step 1000.
    Carrying state keeps the exact integer step from the interleaved tqdm lines
    for the loss lines that follow.
    """
    if not path.exists():
        return []
    points: list[MetricsHistoryPoint] = []
    acc = TrainingMetrics()
    with path.open() as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                log_line = LogLine.model_validate_json(raw)
            except Exception:
                continue  # skip malformed line, same as read_persisted_logs
            msg = log_line.message
            parse_metrics_into(msg, acc, resume_total)
            # Only the log-freq lines carry loss/lr; tqdm lines just advance the
            # step. Emit a point only when a loss value is present so we don't
            # add a flat point per tqdm tick.
            if "loss:" not in msg or acc.current_step <= 0 or acc.current_loss is None:
                continue
            point = MetricsHistoryPoint(
                step=acc.current_step,
                loss=acc.current_loss,
                lr=acc.current_lr,
                grad_norm=acc.grad_norm,
            )
            # Dedupe by step: overwrite on consecutive same-step lines.
            if points and points[-1].step == point.step:
                points[-1] = point
            else:
                points.append(point)
    return points


class LocalJobRunner:
    """Run a training as a local subprocess.

    The runner is single-shot: instantiate a fresh one per job. Lifetime of
    the underlying subprocess is bounded by this object's existence in memory.
    """

    def __init__(
        self,
        metrics: TrainingMetrics,
        log_file_path: Path | None = None,
    ) -> None:
        self._metrics = metrics
        self._process: subprocess.Popen | None = None
        self._log_queue: Queue[LogLine] = Queue()
        self._monitor_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._log_file_path = log_file_path
        self._log_file = None  # type: ignore[assignment]
        self._wandb_run_url: str | None = None
        self._resume_total: int | None = None

    def start(
        self,
        job_id: str,
        config: TrainingRequest,
        output_dir: str,
    ) -> None:
        if self._process is not None:
            raise RuntimeError("LocalJobRunner already started")

        self._resume_total = _resume_total_steps(config)

        # Build the command via the helper that lives in train.py.
        from .train import build_training_command  # avoid import cycle at module load

        cmd = build_training_command(config, output_dir, sys.executable)
        logger.info("Starting job %s: %s", job_id, " ".join(cmd))

        # Open the persistent log sink (one JSON line per stdout line). Held
        # open for the subprocess's lifetime so we don't reopen per write.
        # encoding="utf-8": see the PYTHONIOENCODING note below — without it, a
        # non-ASCII line decoded from the child's stdout can fail to write here
        # on Windows too (default file encoding is the ANSI codepage).
        if self._log_file_path is not None:
            self._log_file_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = self._log_file_path.open("a", buffering=1, encoding="utf-8")

        # PYTHONUNBUFFERED makes the child's stdout flush per line. Without it
        # block-buffering hides log lines from our parser for many seconds.
        # See utils/subprocess_env.py for why PYTHONIOENCODING is forced.
        child_env = utf8_child_env(PYTHONUNBUFFERED="1")

        # See _process_isolation_kwargs() for why this isn't a plain
        # start_new_session=True.
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=child_env,
            **_process_isolation_kwargs(),
        )

        self._monitor_thread = threading.Thread(
            target=self._pump_stdout, name=f"job-{job_id}-stdout", daemon=True
        )
        self._monitor_thread.start()

    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    def stop(self) -> None:
        if self._process is None or self._process.poll() is not None:
            return
        self._stop_event.set()
        try:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Subprocess did not terminate in 10s, killing")
                self._process.kill()
                self._process.wait()
        except Exception as exc:
            logger.exception("Error stopping subprocess: %s", exc)

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def returncode(self) -> int | None:
        if self._process is None:
            return None
        return self._process.poll()

    def stream_log_lines(self) -> list[LogLine]:
        """Drain whatever has accumulated since the last call."""
        out: list[LogLine] = []
        try:
            while True:
                out.append(self._log_queue.get_nowait())
        except Empty:
            pass
        return out

    def wandb_run_url(self) -> str | None:
        return self._wandb_run_url

    # -- internals --

    def _pump_stdout(self) -> None:
        assert self._process is not None
        try:
            for line in iter(self._process.stdout.readline, ""):
                if self._stop_event.is_set():
                    break
                stripped = line.rstrip()
                if not stripped:
                    continue
                parse_metrics_into(stripped, self._metrics, self._resume_total)
                if self._wandb_run_url is None:
                    url = extract_wandb_run_url(stripped)
                    if url is not None:
                        self._wandb_run_url = url
                log_line = LogLine(timestamp=time.time(), message=stripped)
                if self._log_file is not None:
                    try:
                        self._log_file.write(log_line.model_dump_json() + "\n")
                    except Exception as exc:  # pragma: no cover — best-effort persist
                        logger.exception("Error writing to log file: %s", exc)
                # Cap queue so a chatty subprocess can't grow memory unbounded.
                if self._log_queue.qsize() >= 1000:
                    with contextlib.suppress(Empty):
                        self._log_queue.get_nowait()
                self._log_queue.put(log_line)
        except Exception as exc:
            logger.exception("Error reading subprocess stdout: %s", exc)
        finally:
            if self._log_file is not None:
                with contextlib.suppress(Exception):
                    self._log_file.close()
                self._log_file = None


class TailingJobRunner:
    """Re-attaches to a detached subprocess after a uvicorn reload.

    We can't recover the original Popen object across processes, so we don't
    own stdout. Instead we tail the persisted log file and watch the pid.
    Implements the JobRunner Protocol so JobRegistry can use it interchangeably
    with LocalJobRunner.
    """

    def __init__(
        self,
        metrics: TrainingMetrics,
        log_file_path: Path,
        pid: int,
        resume_total: int | None = None,
    ) -> None:
        self._metrics = metrics
        self._log_file_path = log_file_path
        self._pid = pid
        self._resume_total = resume_total
        self._log_queue: Queue[LogLine] = Queue()
        self._tail_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # Replay everything that's already on disk so the parser catches up
        # on metrics, then tail from the current EOF.
        self._tail_offset = 0
        self._wandb_run_url: str | None = None

    def start(self, job_id: str, config: TrainingRequest, output_dir: str) -> None:
        # Required by JobRunner Protocol but irrelevant here; the subprocess
        # we're tailing was started by a previous uvicorn worker.
        raise RuntimeError("TailingJobRunner reattaches to an existing pid; use start_tailing() instead")

    def start_tailing(self) -> None:
        if self._tail_thread is not None:
            return
        self._tail_thread = threading.Thread(
            target=self._tail_loop, name=f"job-tail-{self._pid}", daemon=True
        )
        self._tail_thread.start()

    def stop(self) -> None:
        with contextlib.suppress(psutil.NoSuchProcess):
            psutil.Process(self._pid).terminate()
        self._stop_event.set()

    def is_running(self) -> bool:
        return _pid_alive(self._pid)

    def returncode(self) -> int | None:
        # We can't reap a process from another session, so we don't know the
        # actual exit code. Return 0 once the pid is gone — the watchdog
        # finalises as "done" rather than "failed", which is the better
        # default for a detached training that completed normally.
        if _pid_alive(self._pid):
            return None
        return 0

    def stream_log_lines(self) -> list[LogLine]:
        out: list[LogLine] = []
        try:
            while True:
                out.append(self._log_queue.get_nowait())
        except Empty:
            pass
        return out

    def pid(self) -> int | None:
        return self._pid

    def wandb_run_url(self) -> str | None:
        return self._wandb_run_url

    # -- internals --

    def _tail_loop(self) -> None:
        """Read lines as they arrive in log_file_path. Exits when pid dies
        AND there are no more new lines to read."""
        try:
            while not self._stop_event.is_set():
                if not self._log_file_path.exists():
                    if not _pid_alive(self._pid):
                        return
                    self._stop_event.wait(0.5)
                    continue
                with self._log_file_path.open() as f:
                    f.seek(self._tail_offset)
                    while not self._stop_event.is_set():
                        raw = f.readline()
                        if not raw:
                            self._tail_offset = f.tell()
                            if not _pid_alive(self._pid):
                                return
                            self._stop_event.wait(0.5)
                            continue
                        try:
                            log_line = LogLine.model_validate_json(raw.strip())
                        except Exception:
                            continue
                        parse_metrics_into(log_line.message, self._metrics, self._resume_total)
                        if self._wandb_run_url is None:
                            url = extract_wandb_run_url(log_line.message)
                            if url is not None:
                                self._wandb_run_url = url
                        if self._log_queue.qsize() >= 1000:
                            with contextlib.suppress(Empty):
                                self._log_queue.get_nowait()
                        self._log_queue.put(log_line)
        except Exception as exc:
            logger.exception("Tailing loop error: %s", exc)


_PERSIST_THROTTLE_SECONDS = 1.0


def _list_local_checkpoints(output_dir: str) -> list[JobCheckpoint]:
    """Scan an output dir for valid checkpoint subdirectories.

    A directory under <output_dir>/checkpoints/ is a valid checkpoint iff
    its name parses to an int and it contains pretrained_model/config.json.
    """
    root = Path(output_dir) / "checkpoints"
    if not root.is_dir():
        return []
    out: list[JobCheckpoint] = []
    for entry in root.iterdir():
        if entry.is_symlink() or not entry.is_dir():
            continue
        try:
            step = int(entry.name)
        except ValueError:
            continue
        config_json = entry / "pretrained_model" / "config.json"
        if not config_json.is_file():
            continue
        out.append(
            JobCheckpoint(
                step=step,
                source="local",
                ref=str((entry / "pretrained_model").resolve()),
            )
        )
    out.sort(key=lambda c: c.step)
    return out


# lerobot writes this per-checkpoint config inside pretrained_model/; resuming
# needs it as --config_path so lerobot can reconstruct the run.
_TRAIN_CONFIG_NAME = "train_config.json"


# A Hub checkpoint's training_state/ is what makes it resumable (optimizer +
# step). The cloud wrapper uploads the whole checkpoints/<step>/ entry, so both
# subtrees land in the repo; this file is the cheapest existence probe.
_HUB_TRAINING_STATE_FILE = "training_state/training_step.json"


def _resolve_cloud_resume(source: JobRecord, step: int | None) -> tuple[str, str]:
    """Return (repo_id, step_dir) identifying the Hub checkpoint a cloud run
    should resume from (`step` = None ⇒ the latest available on the Hub).

    The cloud container downloads checkpoints/<step_dir>/ (both pretrained_model/
    and training_state/) from `repo_id` and hands lerobot the reconstructed
    output-dir layout, so resume restores the optimizer and step counter — true
    resume, not a weights-only re-init.

    Raises ValueError (→ HTTP 400) with a user-facing message when the source
    can't be resumed on the cloud: not a cloud run, no output repo, no
    checkpoints at all (the run died before its first save), an unknown step, or
    a checkpoint whose training_state/ never made it to the Hub.
    """
    if source.runner != "hf_cloud":
        raise ValueError(
            "This resume path is for cloud runs; local runs resume from their on-disk checkpoint instead."
        )
    if not source.hf_repo_id:
        raise ValueError(f"Cloud run {source.id!r} has no output repo on the Hub to resume from.")
    api = shared_hf_api()
    checkpoints = _list_hub_checkpoints(api, source.hf_repo_id)
    if not checkpoints:
        raise ValueError(
            f"Cloud run {source.id!r} left no checkpoints on the Hub — nothing to "
            "resume from (the run died before its first save)."
        )
    if step is None:
        chosen = checkpoints[-1]  # step-sorted; take the latest
    else:
        chosen = next((c for c in checkpoints if c.step == step), None)
        if chosen is None:
            raise ValueError(f"Cloud run {source.id!r} has no checkpoint at step {step}.")
    # chosen.ref is 'repo@checkpoints/<step_dir>'; recover the zero-padded dir.
    m = _HUB_CKPT_REF_RE.match(chosen.ref)
    if not m:
        raise ValueError(f"Unexpected checkpoint ref for cloud run {source.id!r}: {chosen.ref!r}")
    step_dir = m.group("step_dir")
    try:
        files = set(api.list_repo_files(source.hf_repo_id, repo_type="model"))
    except Exception as exc:
        raise ValueError(
            f"Could not read cloud run {source.id!r}'s repo to verify the "
            f"checkpoint at step {chosen.step}: {exc}"
        ) from exc
    if f"checkpoints/{step_dir}/{_HUB_TRAINING_STATE_FILE}" not in files:
        raise ValueError(
            f"Checkpoint at step {chosen.step} has no optimizer/step state "
            "(training_state/) on the Hub, so it can't be resumed."
        )
    return source.hf_repo_id, step_dir


def _resolve_resume_config_path(source: JobRecord, step: int | None) -> str:
    """Return the train_config.json path lerobot needs to resume `source` from
    `step` (or its latest checkpoint if step is None).

    Raises ValueError (→ HTTP 400) with a user-facing message when the source
    can't be resumed: not a local run, no checkpoints, unknown step, or a
    weights-only checkpoint missing the training_state/ (optimizer + step)
    needed to continue.
    """
    if source.runner != "local":
        raise ValueError(
            "Only local training runs can be resumed — lerobot doesn't support resuming from the Hub."
        )
    checkpoints = _list_local_checkpoints(source.output_dir)
    if not checkpoints:
        raise ValueError(f"Run {source.id!r} has no saved checkpoints to resume from.")
    if step is None:
        chosen = checkpoints[-1]  # list is step-sorted; take the latest
    else:
        chosen = next((c for c in checkpoints if c.step == step), None)
        if chosen is None:
            raise ValueError(f"Run {source.id!r} has no checkpoint at step {step}.")
    # chosen.ref is <output_dir>/checkpoints/<step>/pretrained_model
    pretrained_dir = Path(chosen.ref)
    train_config = pretrained_dir / _TRAIN_CONFIG_NAME
    training_state = pretrained_dir.parent / "training_state"
    if not train_config.is_file():
        raise ValueError(
            f"Checkpoint at step {chosen.step} is missing {_TRAIN_CONFIG_NAME}, so it can't be resumed."
        )
    if not training_state.is_dir():
        raise ValueError(
            f"Checkpoint at step {chosen.step} has no optimizer/step state "
            "(training_state/), so it can't be resumed. Weights-only models "
            "(e.g. imported) can only start a fresh run."
        )
    return str(train_config.resolve())


def _resolve_finetune_pretrained_path(source: JobRecord, step: int | None) -> str:
    """Return a `--policy.pretrained_path` value that initializes a FRESH run's
    weights from `source`'s checkpoint at `step` (or its latest if step is None).

    Unlike resume, this does NOT require training_state/ — weights-only is the
    whole point of fine-tuning. lerobot's PreTrainedConfig.pretrained_path loads
    the policy weights (and processors) from a local pretrained_model dir or a
    Hub repo on a non-resume run.

    Handles every source shape via its own checkpoint listing:
      * imported local / normal local run → a `local` ref that is the absolute
        pretrained_model dir; returned directly (e.g. a flat imported dir
        becomes a step-0 checkpoint whose ref is the dir itself).
      * imported hub / cloud run → a `hub` ref ('repo@checkpoints/<step>' or
        'repo@root'); we return the plain repo id (the root model). lerobot's
        pretrained_path takes a repo id, not a sub-path, so a specific hub
        sub-step can't be targeted — see the limitation note below.

    Raises ValueError (→ HTTP 400) with a user-facing message when the source
    has no usable checkpoint.
    """
    if source.runner == "imported":
        if source.hf_repo_id:
            checkpoints = _list_imported_hub(shared_hf_api(), source.hf_repo_id)
        else:
            checkpoints = _list_imported_local(source.output_dir)
    elif source.runner == "local":
        checkpoints = _list_local_checkpoints(source.output_dir)
    else:  # hf_cloud
        checkpoints = _list_hub_checkpoints(shared_hf_api(), source.hf_repo_id)

    if not checkpoints:
        raise ValueError(f"Source {source.id!r} has no usable checkpoint to fine-tune from.")
    if step is None:
        chosen = checkpoints[-1]  # step-sorted; take the latest
    else:
        chosen = next((c for c in checkpoints if c.step == step), None)
        if chosen is None:
            raise ValueError(f"Source {source.id!r} has no checkpoint at step {step}.")

    if chosen.source == "local":
        # chosen.ref is the absolute pretrained_model dir lerobot loads directly.
        return chosen.ref
    # Hub ref: 'repo@checkpoints/<step_dir>' or 'repo@root'. lerobot's
    # pretrained_path accepts a repo id (root model), not a sub-step path, so we
    # hand back the repo portion. Fine-tuning from a specific hub sub-step isn't
    # supported here — it uses whatever weights live at the repo root.
    repo_id = chosen.ref.split("@", 1)[0]
    return repo_id


_CLOUD_CKPT_TTL_SECONDS = 30.0
_CKPT_PATH_RE = re.compile(r"^checkpoints/(\d+)/pretrained_model/config\.json$")


def _hub_checkpoints_from_files(files, repo_id: str) -> list[JobCheckpoint]:
    """Parse a repo file listing into checkpoints. The ref preserves the
    original zero-padded directory name (e.g. 000050); JobCheckpoint.step is
    the int form for sorting and UI display."""
    seen: dict[int, JobCheckpoint] = {}
    for path in files:
        m = _CKPT_PATH_RE.match(path)
        if not m:
            continue
        step_dir = m.group(1)
        step = int(step_dir)
        seen[step] = JobCheckpoint(
            step=step,
            source="hub",
            ref=f"{repo_id}@checkpoints/{step_dir}",
        )
    out = list(seen.values())
    out.sort(key=lambda c: c.step)
    return out


def _list_imported_local(path: str) -> list[JobCheckpoint]:
    """Auto-detect the layout of an imported local directory.

    A checkpoints/<step>/pretrained_model tree → reuse _list_local_checkpoints.
    Otherwise, if the dir itself is a pretrained_model (config.json present) →
    a single step-0 checkpoint. Neither → empty (source moved/unusable)."""
    tree = _list_local_checkpoints(path)
    if tree:
        return tree
    if (Path(path) / "config.json").is_file():
        return [JobCheckpoint(step=0, source="local", ref=str(Path(path).resolve()))]
    return []


def _list_imported_hub(api, repo_id: str) -> list[JobCheckpoint]:
    """Auto-detect the layout of an imported Hub model repo.

    A checkpoints/<step>/pretrained_model tree → reuse the tree parse.
    Otherwise, a root config.json → a single step-0 checkpoint with a
    'repo@root' ref (the whole repo is the pretrained_model)."""
    try:
        files = api.list_repo_files(repo_id, repo_type="model")
    except Exception:
        return []
    tree = _hub_checkpoints_from_files(files, repo_id)
    if tree:
        return tree
    if "config.json" in files:
        return [JobCheckpoint(step=0, source="hub", ref=f"{repo_id}@root")]
    return []


def _list_hub_checkpoints(api, repo_id: str) -> list[JobCheckpoint]:
    """List checkpoints by introspecting the model repo file tree."""
    try:
        files = api.list_repo_files(repo_id, repo_type="model")
    except Exception:
        # Repo may not exist yet (training just started, sidecar hasn't
        # uploaded anything). Treat as no checkpoints.
        return []
    return _hub_checkpoints_from_files(files, repo_id)


_LANGUAGE_CONDITIONED_POLICY_TYPES = {"smolvla", "pi0", "pi0_fast", "pi05"}


_HUB_CKPT_REF_RE = re.compile(r"^(?P<repo>[^@]+)@checkpoints/(?P<step_dir>\d+)$")
_HUB_ROOT_REF_RE = re.compile(r"^(?P<repo>[^@]+)@root$")


def _read_checkpoint_config(ckpt: JobCheckpoint) -> dict[str, object]:
    """Load the pretrained_model/config.json for one checkpoint.

    Keyed on the checkpoint's own source/ref shape so it works for training
    jobs and imports alike:
      * local  → ckpt.ref is the absolute pretrained_model dir.
      * hub    → 'repo@checkpoints/<step_dir>' (a tree) or 'repo@root' (a flat
                 model repo); both resolve via hf_hub_download.
    """
    if ckpt.source == "local":
        with open(Path(ckpt.ref) / "config.json") as f:
            return json.load(f)
    from huggingface_hub import hf_hub_download

    m = _HUB_CKPT_REF_RE.match(ckpt.ref)
    if m:
        repo_id = m.group("repo")
        filename = f"checkpoints/{m.group('step_dir')}/pretrained_model/config.json"
    else:
        m = _HUB_ROOT_REF_RE.match(ckpt.ref)
        if not m:
            raise ValueError(f"Bad hub ref: {ckpt.ref!r}")
        repo_id = m.group("repo")
        filename = "config.json"
    local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")
    with open(local_path) as f:
        return json.load(f)


def _flat_feature_dim(feat: object) -> int | None:
    """Flat width of a policy feature (e.g. observation.state, action).

    Checkpoint config features carry a `shape` list; for the proprioceptive
    state and action these are 1-D — `[6]` for a single SO-101 arm, `[12]` for
    a bimanual (two-arm) checkpoint. Returns the single dim, or None when the
    feature is absent or not 1-D (nothing downstream should guess in that
    case)."""
    if not isinstance(feat, dict):
        return None
    shape = feat.get("shape")
    if not isinstance(shape, (list, tuple)) or len(shape) != 1:
        return None
    try:
        return int(shape[0])
    except (TypeError, ValueError):
        return None


def _generate_job_id(policy_type: str, dataset_repo_id: str) -> str:
    """Build a sortable, collision-free job id from policy type and dataset slug."""
    from .train import _SLUG_RE

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dataset_slug = _SLUG_RE.sub("_", dataset_repo_id).strip("_") or "dataset"
    return f"{policy_type}_{dataset_slug}_{timestamp}"


# Accepted in place of a bare repo id when importing from the Hub — users
# paste the model page URL as often as the id.
_HUB_URL_PREFIXES = (
    "https://huggingface.co/",
    "http://huggingface.co/",
    "https://hf.co/",
    "http://hf.co/",
    "huggingface.co/",
    "hf.co/",
)


def _normalize_import_source(source: str) -> str:
    """Boundary normalization for import sources, applied before both storing
    and comparing: trim whitespace, strip a pasted Hub URL prefix down to the
    bare repo id, and drop trailing slashes. Local absolute paths start with
    '/' so the URL prefixes never match them."""
    src = source.strip()
    lowered = src.lower()
    for prefix in _HUB_URL_PREFIXES:
        if lowered.startswith(prefix):
            src = src[len(prefix) :]
            break
    return src.rstrip("/")


def _paths_are_same_dir(a: str, b: str) -> bool:
    """True when two path strings refer to the same directory on disk.

    os.path.samefile compares device+inode, so it survives spellings that a
    string compare misses — most importantly case variants on the (default)
    case-insensitive macOS filesystem: a real duplicate pair was registered as
    '/Users/mokuroh54/…' and '/Users/Mokuroh54/…' because Path.resolve()
    preserves the typed case. Falls back to exact string equality when either
    path can't be stat'ed (e.g. the recorded source has since moved)."""
    if a == b:
        return True
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def _job_dir(output_root: Path, job_id: str) -> Path:
    return output_root / job_id


def _job_log_path(output_root: Path, job_id: str) -> Path:
    return _job_dir(output_root, job_id) / "log.jsonl"


def _job_meta_path(output_root: Path, job_id: str) -> Path:
    return _job_dir(output_root, job_id) / "job.json"


class JobAlreadyRunningError(Exception):
    """Raised when start() is called while another local job is running."""


class JobNotFoundError(Exception):
    """Raised when a lookup hits an unknown id."""


class JobNotRunningError(Exception):
    """Raised when stop() is called on a non-running job."""


class DatasetNotOnHubError(Exception):
    """Raised by JobRegistry.start when a cloud (hf_cloud) run is requested on a
    dataset that isn't on the Hub. HF Jobs pods resolve the dataset by repo_id
    from the Hub — they can't see this machine's local cache — so a local-only
    dataset would make the remote job fail. The UI's upload-then-train flow
    makes this unreachable from the browser; this guard exists for non-UI
    callers (and as belt-and-braces) so they get a clear 409 instead of a
    remote crash. `repo_id` is the offending dataset."""

    def __init__(self, repo_id: str) -> None:
        self.repo_id = repo_id
        super().__init__(
            f"Dataset '{repo_id}' is not on the Hugging Face Hub. Cloud training "
            "runs from the Hub, so upload the dataset first (or record/select one "
            "that's already on the Hub)."
        )


class JobRegistry:
    """Owns the registry of training jobs and their persistence.

    On instantiation, scans outputs/train/ for existing job.json files. For
    each record marked 'running': local jobs reattach if the pid is alive
    (else 'interrupted'); hf_cloud jobs always reattach and let the tail loop
    drive finalisation.
    """

    def __init__(self, output_root: Path) -> None:
        self._output_root = output_root.resolve()
        self._output_root.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._records: dict[str, JobRecord] = {}
        self._runners: dict[str, JobRunner] = {}
        self._last_persist_at: dict[str, float] = {}

        self._stop_watchdog = threading.Event()
        self._watchdog_thread: threading.Thread | None = None

        # repo_id -> (expires_at_epoch, checkpoint list)
        self._cloud_ckpt_cache: dict[str, tuple[float, list[JobCheckpoint]]] = {}

        # Fired (best-effort) on every state change: new job, stop initiated,
        # watchdog finalisation, delete. Server wires this to a WebSocket
        # broadcast so the frontend can refetch on-event instead of polling.
        self._on_change: Callable[[], None] | None = None

        # Fired from the watchdog at ~1Hz with a compact snapshot of every
        # running job (id, state, metrics, wandb url, checkpoint count) so
        # the dashboard keeps the progress bar live without refetching /jobs.
        self._on_progress: Callable[[builtins.list[dict]], None] | None = None

        self._migrate_legacy_cwd_jobs()
        self._load_from_disk()
        self._dedupe_imported_records()
        self._start_watchdog()

    def _migrate_legacy_cwd_jobs(self) -> None:
        """One-shot migration from cwd-relative `outputs/train/` to the new
        absolute root.

        Older makerlab versions wrote job dirs to `<cwd>/outputs/train/`, which
        meant history disappeared when you launched from a different cwd. We
        now anchor to ~/.cache/.../outputs/train. On first boot under the new
        layout, move any pre-existing cwd-relative job dirs over and rewrite
        each job.json's `output_dir` field to the new absolute path.

        Idempotent: skipped if (a) the new root is the legacy one itself
        (MAKERLAB_OUTPUT_ROOT=outputs/train still wins for tests), or (b) the
        legacy dir is absent / already empty.
        """
        legacy_root = (Path.cwd() / "outputs" / "train").resolve()
        if legacy_root == self._output_root or not legacy_root.is_dir():
            return

        legacy_dirs = [p for p in legacy_root.iterdir() if p.is_dir()]
        if not legacy_dirs:
            return

        logger.info(
            "Migrating %d legacy job dirs from %s to %s",
            len(legacy_dirs),
            legacy_root,
            self._output_root,
        )
        for src in legacy_dirs:
            dst = self._output_root / src.name
            if dst.exists():
                logger.warning("Migration: %s already exists at destination; skipping", src.name)
                continue
            try:
                shutil.move(str(src), str(dst))
            except Exception as exc:
                logger.warning("Migration: failed to move %s: %s", src.name, exc)
                continue
            self._rewrite_output_dir_in_meta(dst)

        # If the legacy dir is now empty, remove it so subsequent boots skip
        # the scan. A leftover non-dir file keeps it around — that's fine.
        with contextlib.suppress(OSError):
            legacy_root.rmdir()

    def _rewrite_output_dir_in_meta(self, job_dir: Path) -> None:
        """Repoint `output_dir` in a migrated job.json to its new absolute
        path. Pre-migration records stored `outputs/train/<id>/run` which
        no longer resolves once cwd has moved."""
        meta = job_dir / "job.json"
        if not meta.is_file():
            return
        try:
            data = json.loads(meta.read_text())
        except Exception as exc:
            logger.warning("Migration: could not parse %s: %s", meta, exc)
            return
        data["output_dir"] = str(job_dir / "run")
        tmp = meta.with_suffix(meta.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, meta)

    def set_on_change(self, callback: Callable[[], None] | None) -> None:
        """Register a single observer fired when registry state changes."""
        self._on_change = callback

    def set_on_progress(self, callback: Callable[[builtins.list[dict]], None] | None) -> None:
        """Register an observer fired each watchdog tick with one dict per
        running job. Quiet when no job runs: a tick with no running jobs
        produces no callback."""
        self._on_progress = callback

    def _notify_change(self) -> None:
        cb = self._on_change
        if cb is None:
            return
        try:
            cb()
        except Exception as exc:
            logger.exception("JobRegistry on_change callback failed: %s", exc)

    def _notify_progress(self, snapshots: builtins.list[dict]) -> None:
        cb = self._on_progress
        if cb is None or not snapshots:
            return
        try:
            cb(snapshots)
        except Exception as exc:
            logger.exception("JobRegistry on_progress callback failed: %s", exc)

    # -- public API --

    def list(self, limit: int = 10) -> builtins.list[JobRecord]:
        with self._lock:
            records = list(self._records.values())
        records.sort(key=lambda r: r.started_at, reverse=True)
        records = records[:limit]
        for r in records:
            r.checkpoint_count = self._count_checkpoints(r)
        return records

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            record = self._records.get(job_id)
        if record is None:
            raise JobNotFoundError(job_id)
        record.checkpoint_count = self._count_checkpoints(record)
        return record

    def start(self, config: TrainingRequest, target: JobTarget | None = None) -> JobRecord:
        from .runners.hf_cloud import HfCloudJobRunner  # lazy import to avoid circular import

        target = target or JobTarget()
        if target.runner == "hf_cloud" and not target.flavor:
            raise ValueError("flavor is required when runner is hf_cloud")

        # Cloud preflight (belt-and-braces): the HF Jobs pod resolves the
        # dataset by repo_id from the Hub and can't see this machine's local
        # cache, so a local-only dataset would fail the remote job. Reject up
        # front with a clear error instead of submitting a doomed job. Only a
        # definitive "local_only" blocks; "unknown" (offline / transient
        # transport error) is left to the existing _ensure_dataset_on_hub
        # fallback so a network blip doesn't wrongly refuse a Hub dataset. The
        # browser flow uploads-then-trains before ever reaching here, so this
        # path is primarily for non-UI callers.
        if target.runner == "hf_cloud":
            from .datasets import get_hub_status

            if get_hub_status(config.dataset_repo_id).get("status") == "local_only":
                raise DatasetNotOnHubError(config.dataset_repo_id)

        with self._lock:
            # Local trainings are bounded by this machine's GPU/USB resources,
            # so at most one runs at a time. Cloud trainings each get their
            # own remote container, so any number can be in flight in parallel.
            if target.runner == "local":
                for r in self._records.values():
                    if r.state == "running" and r.runner == "local":
                        raise JobAlreadyRunningError(r.id)

            # Resume and fine-tune are distinct and mutually exclusive: resume
            # continues optimizer+step from a checkpoint (needs training_state);
            # fine-tune starts a FRESH run whose weights are init'd from a
            # checkpoint (weights-only is fine). Reject the nonsensical combo up
            # front rather than letting one silently win.
            if config.resume and config.finetune_from_job_id:
                raise ValueError(
                    "A run can't both resume and fine-tune. Resume continues an "
                    "existing run's optimizer/step; fine-tune starts a fresh run "
                    "from a checkpoint's weights."
                )

            # Fine-tune: turn the selected source run + step into the
            # pretrained_path lerobot loads weights from. A fresh run (resume
            # stays false); no training_state required. Resolved under the lock
            # and before the record so a bad selection fails with no orphan.
            if config.finetune_from_job_id:
                source = self._records.get(config.finetune_from_job_id)
                if source is None:
                    raise ValueError(f"Fine-tune source {config.finetune_from_job_id!r} not found.")
                config.policy_pretrained_path = _resolve_finetune_pretrained_path(
                    source, config.finetune_from_step
                )

            # Resume: turn the selected source run + step into the config_path
            # lerobot needs. Do this under the lock (source lookup) and before
            # creating the record so a bad selection fails cleanly with no
            # orphaned job.
            if config.resume:
                if config.resume_from_job_id:
                    source = self._records.get(config.resume_from_job_id)
                    if source is None:
                        raise ValueError(f"Resume source {config.resume_from_job_id!r} not found.")
                    if source.runner == "hf_cloud":
                        # An HF Job is immutable once ended: resuming a cloud run
                        # launches a NEW cloud job that continues from the parent's
                        # Hub checkpoint. Record the source repo + step dir; the
                        # HfCloudJobRunner turns them into an in-container download
                        # + reconstruct + --config_path. The dataset-on-Hub guard
                        # (target.runner == hf_cloud above) still applies, so a
                        # run whose dataset vanished fails the same way a fresh
                        # cloud run would.
                        repo_id, step_dir = _resolve_cloud_resume(source, config.resume_from_step)
                        config.resume_from_hub_repo = repo_id
                        config.resume_from_hub_step = step_dir
                    else:
                        config.config_path = _resolve_resume_config_path(source, config.resume_from_step)
                elif not config.config_path:
                    raise ValueError(
                        "Resume is on but no source checkpoint was selected. Use "
                        '"Continue" on a completed local run rather than toggling '
                        "resume manually."
                    )

            job_id = self._unique_job_id(config.policy_type, config.dataset_repo_id)
            job_dir = _job_dir(self._output_root, job_id)
            lerobot_output_dir = str(job_dir / "run")
            name = (
                config.job_name.strip()
                if (config.job_name and config.job_name.strip())
                else f"{config.policy_type.upper()} · {config.dataset_repo_id}"
            )
            record = JobRecord(
                id=job_id,
                name=name,
                state="running",
                config=config,
                output_dir=lerobot_output_dir,
                started_at=time.time(),
                runner=target.runner,
                hf_flavor=target.flavor,
            )

            job_dir.mkdir(parents=True, exist_ok=True)
            self._records[job_id] = record
            self._persist(record, force=True)

            log_path = _job_log_path(self._output_root, job_id)
            if target.runner == "local":
                runner = LocalJobRunner(record.metrics, log_file_path=log_path)
            else:
                runner = HfCloudJobRunner(record.metrics, log_path, target.flavor)

            try:
                runner.start(job_id, config, lerobot_output_dir)
            except Exception as exc:
                logger.exception("Failed to start runner for job %s", job_id)
                record.state = "failed"
                record.ended_at = time.time()
                record.error_message = f"Failed to start runner: {exc}"
                self._persist(record, force=True)
                raise

            # Capture runner-specific identifiers.
            if target.runner == "local":
                record.process_pid = runner.pid()
            else:
                record.hf_job_id = runner.hf_job_id()
                record.hf_job_url = runner.hf_job_url()
                # config was mutated by HfCloudJobRunner.start to set
                # policy_repo_id; mirror it onto the record for the UI.
                record.hf_repo_id = config.policy_repo_id

            self._persist(record, force=True)
            self._runners[job_id] = runner
        self._notify_change()
        return record

    def _unique_job_id(self, policy_type: str, dataset_repo_id: str) -> str:
        """_generate_job_id with a collision guard. The generated id embeds a
        second-granularity timestamp, so two jobs created within the same
        second would otherwise share an id and silently overwrite each other
        in the registry (and on disk). Suffix -2, -3, … until unused."""
        base = _generate_job_id(policy_type, dataset_repo_id)
        job_id = base
        n = 2
        while job_id in self._records or _job_dir(self._output_root, job_id).exists():
            job_id = f"{base}-{n}"
            n += 1
        return job_id

    def find_imported(self, source: str) -> JobRecord | None:
        """Return the already-registered imported record for `source`, if any.

        `source` is normalized first (whitespace, pasted Hub URLs, trailing
        slashes — see _normalize_import_source). Identity per import kind:
          * local dir → filesystem identity of the resolved path vs the stored
            output_dir (_paths_are_same_dir: samefile, so case variants on a
            case-insensitive filesystem and moved-cwd spellings still match —
            a plain string compare demonstrably missed a real duplicate pair);
          * hub repo → hf_repo_id compared CASE-INSENSITIVELY. Reversal of the
            earlier exact-match choice: HF repo ids are practically unique
            case-insensitively (the Hub redirects across casings), and the
            failure mode of exact matching is silent duplicates.
        """
        src = _normalize_import_source(source)
        if not src:
            return None
        local_path = Path(src).expanduser()
        local_key = str(local_path.resolve()) if local_path.is_dir() else None
        with self._lock:
            for r in self._records.values():
                if r.runner != "imported":
                    continue
                if local_key is not None:
                    if not r.hf_repo_id and r.output_dir and _paths_are_same_dir(r.output_dir, local_key):
                        return r
                elif (r.hf_repo_id or "").lower() == src.lower():
                    return r
        return None

    def register_imported(self, source: str, name: str | None = None) -> JobRecord:
        """Register an externally-trained model as a pointer-only pseudo-job.

        `source` is either an existing local directory (its path is stored in
        output_dir) or, failing that, a Hugging Face repo id (stored in
        hf_repo_id). The source must expose at least one checkpoint under the
        auto-detect rules, else ValueError. Nothing is copied; delete only
        removes the pointer.

        Idempotent per source: importing an already-registered path/repo
        returns the EXISTING record (its id and display alias untouched)
        instead of creating a second entry — see find_imported for the
        identity keys. The source is normalized first (whitespace, pasted Hub
        URLs, trailing slashes), and the normalized form is what gets stored."""
        src = _normalize_import_source(source)
        if not src:
            raise ValueError("source is required")

        existing = self.find_imported(src)
        if existing is not None:
            return existing

        local_path = Path(src).expanduser()
        if local_path.is_dir():
            resolved = str(local_path.resolve())
            ckpts = _list_imported_local(resolved)
            output_dir, hf_repo_id = resolved, None
            label = local_path.name or resolved
        else:
            ckpts = _list_imported_hub(shared_hf_api(), src)
            output_dir, hf_repo_id = "", src
            label = src

        if not ckpts:
            raise ValueError(
                f"No usable model at {src!r}. For a local path, expected a "
                "pretrained_model (config.json) or a checkpoints/<step>/"
                "pretrained_model tree. For a Hugging Face repo, the repo may "
                "not exist, be private without auth, or lack a model config."
            )

        # Best-effort policy type for the display name; inference reads the
        # real config from the checkpoint, so a wrong guess here is harmless.
        policy_type = "model"
        with contextlib.suppress(Exception):
            policy_type = str(_read_checkpoint_config(ckpts[-1]).get("type") or "model")

        with self._lock:
            job_id = self._unique_job_id(policy_type, "imported")
            record = JobRecord(
                id=job_id,
                name=name or f"Imported · {label}",
                state="done",
                config=TrainingRequest(dataset_repo_id="(imported)", policy_type=policy_type),
                output_dir=output_dir,
                started_at=time.time(),
                ended_at=time.time(),
                runner="imported",
                hf_repo_id=hf_repo_id,
            )
            self._records[job_id] = record
            self._persist(record, force=True)
        self._notify_change()
        return record

    def rename(self, job_id: str, new_name: str) -> JobRecord:
        """Set a job's display alias. Metadata-only by design: the immutable
        identity (run id, output_dir, hub repo id) is never touched, so resume
        lineage (charts stitch across runs by id), live training/inference
        reads, imported-model hub identity (dedup on re-import), and remote
        HF Jobs / W&B names all keep working. The UI shows the alias and falls
        back to `name` when unset.

        Aliases are display-only, so uniqueness is NOT enforced (unlike
        calibration/robot renames, where the name is a file key). The same
        is_valid-style guard is applied for consistency (rejects path-ish
        characters); trimmed; empty ⇒ ValueError (→ HTTP 400)."""
        name = new_name.strip()
        if not name:
            raise ValueError("Display name cannot be empty.")
        if not is_valid_robot_name(name):
            raise ValueError("Invalid display name.")
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                raise JobNotFoundError(job_id)
            record.display_name = name
            self._persist(record, force=True)
        self._notify_change()
        return record

    def stop(self, job_id: str) -> JobRecord:
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                raise JobNotFoundError(job_id)
            runner = self._runners.get(job_id)
        if record.state != "running" or runner is None:
            raise JobNotRunningError(job_id)
        runner.stop()
        # The watchdog will finalise the record (state, ended_at, exit_code).
        # Wait briefly so the caller sees the new state in the response.
        for _ in range(20):
            time.sleep(0.1)
            with self._lock:
                if record.state != "running":
                    return record
        return record

    def drain_logs(self, job_id: str) -> builtins.list[LogLine]:
        with self._lock:
            if job_id not in self._records:
                raise JobNotFoundError(job_id)
            runner = self._runners.get(job_id)
        if runner is None:
            return []
        return runner.stream_log_lines()

    def read_persisted_logs(self, job_id: str) -> builtins.list[LogLine]:
        """Read all log lines that have been written to disk for this job.

        Used by the frontend on Monitoring-page mount to seed the log panel
        with history (e.g. after navigating away and back, or after a makerlab
        restart marked the job 'interrupted').
        """
        with self._lock:
            if job_id not in self._records:
                raise JobNotFoundError(job_id)
        path = _job_log_path(self._output_root, job_id)
        if not path.exists():
            return []
        out: list[LogLine] = []
        with path.open() as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    out.append(LogLine.model_validate_json(raw))
                except Exception:
                    continue  # skip a malformed line rather than 500ing
        return out

    def read_metrics_history(self, job_id: str) -> builtins.list[MetricsHistoryPoint]:
        """Reconstruct the per-step loss/lr/grad-norm series from log.jsonl.

        Walks the resume lineage (job -> resume source -> …, oldest first) and
        concatenates each run's points, so a resumed run's curve is continuous
        across the whole training rather than starting at the resume step. Stops
        at a missing ancestor (a deleted source) — the curve just starts later.

        Used by the frontend on Monitoring-page mount to seed the curves so they
        survive page reloads, navigation, and makerlab restarts. Re-parses on every
        call; cache later if a slow file ever shows up.
        """
        with self._lock:
            if job_id not in self._records:
                raise JobNotFoundError(job_id)
            chain: list[JobRecord] = []
            seen: set[str] = set()
            cur: JobRecord | None = self._records[job_id]
            while cur is not None and cur.id not in seen:
                chain.append(cur)
                seen.add(cur.id)
                parent_id = cur.config.resume_from_job_id
                cur = self._records.get(parent_id) if parent_id else None
        chain.reverse()  # oldest (root) first so steps ascend across the chain

        # Concatenate each run's points; dedupe by step (later run wins) in case
        # a resume boundary overlaps, then sort for a clean ascending curve.
        by_step: dict[int, MetricsHistoryPoint] = {}
        for record in chain:
            log_path = _job_log_path(self._output_root, record.id)
            for point in _read_log_metrics(log_path, _resume_total_steps(record.config)):
                by_step[point.step] = point
        return sorted(by_step.values(), key=lambda p: p.step)

    def _checkpoints_for(self, record: JobRecord) -> builtins.list[JobCheckpoint]:
        if record.runner == "imported":
            if record.hf_repo_id:
                return self._list_cloud_cached(record.hf_repo_id, _list_imported_hub)
            return _list_imported_local(record.output_dir)
        if record.runner == "local":
            return _list_local_checkpoints(record.output_dir)
        return self._list_cloud_cached(record.hf_repo_id)

    def list_checkpoints(self, job_id: str) -> builtins.list[JobCheckpoint]:
        """Return checkpoints saved for this job, ascending by step.

        Local jobs scan <output_dir>/checkpoints/. Cloud jobs introspect the
        Hub repo (30s TTL cache). Imported jobs auto-detect single-model vs
        checkpoints-tree from their local path or Hub repo id."""
        with self._lock:
            record = self._records.get(job_id)
        if record is None:
            raise JobNotFoundError(job_id)
        return self._checkpoints_for(record)

    def _list_cloud_cached(
        self, repo_id: str | None, fetch=_list_hub_checkpoints
    ) -> builtins.list[JobCheckpoint]:
        """30s-TTL cache over a hub checkpoint listing. `fetch(api, repo_id)`
        defaults to the training-job tree scan; imported hub models pass
        `_list_imported_hub` so they share the same cache + rate-limit budget."""
        if not repo_id:
            return []
        now = time.time()
        cached = self._cloud_ckpt_cache.get(repo_id)
        if cached is not None and cached[0] > now:
            return cached[1]
        result = fetch(shared_hf_api(), repo_id)
        self._cloud_ckpt_cache[repo_id] = (now + _CLOUD_CKPT_TTL_SECONDS, result)
        return result

    def _count_checkpoints(self, record: JobRecord) -> int:
        return len(self._checkpoints_for(record))

    def get_policy_config_summary(self, job_id: str, step: int) -> dict[str, object]:
        """Read the checkpoint's pretrained_model/config.json and return only
        the UX-relevant slice: policy type, expected camera names + their
        height/width, and whether the policy needs a --task string."""
        with self._lock:
            record = self._records.get(job_id)
        if record is None:
            raise JobNotFoundError(job_id)
        ckpts = self.list_checkpoints(job_id)
        match = next((c for c in ckpts if c.step == step), None)
        if match is None:
            raise FileNotFoundError(f"No checkpoint at step {step} for job {record.id}")
        cfg = _read_checkpoint_config(match)
        policy_type = cfg.get("type")
        input_features = cfg.get("input_features") or {}
        image_features: dict[str, dict[str, int]] = {}
        for full_name, feat in input_features.items():
            if feat.get("type") != "VISUAL":
                continue
            shape = feat.get("shape") or []
            if len(shape) != 3:
                continue
            _channels, height, width = shape
            # The policy keys are 'observation.images.<name>'; the rollout CLI
            # takes just the suffix.
            name = full_name.split(".")[-1]
            image_features[name] = {"height": int(height), "width": int(width)}
        return {
            "policy_type": policy_type,
            "image_features": image_features,
            "requires_task": policy_type in _LANGUAGE_CONDITIONED_POLICY_TYPES,
            # Flat proprioceptive state / action widths. For an SO-101 arm this
            # is 6 (one per joint); a bimanual-trained checkpoint carries 12
            # (two arms). The inference modal compares this against the selected
            # robot's arm count to explain a single-arm/bimanual mismatch before
            # the user hits Start. None when the checkpoint omits the feature.
            "state_dim": _flat_feature_dim(input_features.get("observation.state")),
            "action_dim": _flat_feature_dim((cfg.get("output_features") or {}).get("action")),
        }

    def delete(self, job_id: str) -> None:
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                raise JobNotFoundError(job_id)
            if record.state == "running":
                raise JobNotRunningError(job_id)
            self._records.pop(job_id, None)
            self._runners.pop(job_id, None)
            self._last_persist_at.pop(job_id, None)
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree(_job_dir(self._output_root, job_id))
        self._notify_change()

    def shutdown(self) -> None:
        """For tests / orderly process exit. Not wired to FastAPI lifespan today."""
        self._stop_watchdog.set()

    # -- internals --

    def _load_from_disk(self) -> None:
        for job_dir in self._output_root.glob("*/"):
            meta = job_dir / "job.json"
            if not meta.exists():
                continue
            try:
                data = json.loads(meta.read_text())
                record = JobRecord.model_validate(data)
            except Exception as exc:
                logger.warning("Skipping malformed job.json at %s: %s", meta, exc)
                continue
            if record.state == "running":
                if record.runner == "local":
                    pid = record.process_pid
                    if pid is not None and _pid_alive(pid):
                        logger.info(
                            "Re-attaching to detached local job %s (pid %d)",
                            record.id,
                            pid,
                        )
                        runner = TailingJobRunner(
                            record.metrics,
                            _job_log_path(self._output_root, record.id),
                            pid,
                            _resume_total_steps(record.config),
                        )
                        runner.start_tailing()
                        self._runners[record.id] = runner
                    else:
                        record.state = "interrupted"
                        if record.ended_at is None:
                            record.ended_at = time.time()
                        self._write_meta(record)
                elif record.runner == "hf_cloud" and record.hf_job_id and record.hf_flavor:
                    # Always reattach; the status poller is the source of truth
                    # for terminal state. If the HF job already finished, the
                    # next inspect_job call resolves the final stage and the
                    # watchdog finalises the record. A transient HF API hiccup
                    # at startup no longer strands the record as "interrupted".
                    logger.info(
                        "Re-attaching to HF Cloud job %s (hf_job_id=%s)",
                        record.id,
                        record.hf_job_id,
                    )
                    from .runners.hf_cloud import HfCloudJobRunner

                    runner = HfCloudJobRunner(
                        record.metrics,
                        _job_log_path(self._output_root, record.id),
                        record.hf_flavor,
                    )
                    runner.reattach(record.hf_job_id)
                    self._runners[record.id] = runner
                else:
                    # Malformed running record — mark interrupted.
                    record.state = "interrupted"
                    if record.ended_at is None:
                        record.ended_at = time.time()
                    self._write_meta(record)
            self._records[record.id] = record

    def _dedupe_imported_records(self) -> None:
        """One-time collapse of duplicate imported pointers left behind before
        dedup-at-registration existed (same local path or hub repo id
        registered more than once).

        Runs at boot, after _load_from_disk and before the watchdog starts
        (single-threaded, so no lock needed). Per identity group: keep the
        OLDEST record; if the keeper has no alias, migrate the newest
        duplicate's display_name onto it. Duplicates are dropped from the
        in-memory map; their job dir is deleted ONLY when it contains nothing
        but job.json — anything else (weights, logs, leftovers) means the
        files stay put and we just log. Conservative by design: malformed
        pointers (no identity) are left alone entirely."""
        groups: dict[tuple[str, str], list[JobRecord]] = {}
        for r in self._records.values():
            if r.runner != "imported":
                continue
            if r.hf_repo_id:
                # Case-insensitive: HF repo ids are practically unique
                # case-insensitively (same reversal as find_imported).
                key = ("hub", r.hf_repo_id.lower())
            elif r.output_dir:
                # Filesystem identity (device:inode) so spellings that differ
                # only by case on a case-insensitive filesystem — the real
                # duplicate pair — group together. Unstat-able paths (source
                # moved/deleted) fall back to the raw string: conservative,
                # they only group with byte-identical spellings.
                try:
                    st = os.stat(r.output_dir)
                    key = ("local", f"{st.st_dev}:{st.st_ino}")
                except OSError:
                    key = ("local", r.output_dir)
            else:
                continue  # no identity — when in doubt, keep it
            groups.setdefault(key, []).append(r)

        for (kind, _ident), records in groups.items():
            if len(records) < 2:
                continue
            records.sort(key=lambda r: r.started_at)
            keeper, dupes = records[0], records[1:]
            if keeper.display_name is None:
                # Newest aliased duplicate wins — it's the user's latest word.
                for dup in reversed(dupes):
                    if dup.display_name:
                        keeper.display_name = dup.display_name
                        self._write_meta(keeper)
                        break
            for dup in dupes:
                self._records.pop(dup.id, None)
                dup_dir = _job_dir(self._output_root, dup.id)
                removed = False
                if dup_dir.is_dir():
                    try:
                        only_meta = [p.name for p in dup_dir.iterdir()] == ["job.json"]
                    except OSError:
                        only_meta = False
                    if only_meta:
                        shutil.rmtree(dup_dir, ignore_errors=True)
                        removed = True
                    else:
                        logger.info(
                            "Duplicate imported model %s: leaving %s in place (contains more than job.json).",
                            dup.id,
                            dup_dir,
                        )
                logger.info(
                    "Collapsed duplicate imported model %s into %s (same %s %r)%s",
                    dup.id,
                    keeper.id,
                    kind,
                    keeper.hf_repo_id or keeper.output_dir,
                    "" if removed else " — pointer dropped from the registry only",
                )

    def _start_watchdog(self) -> None:
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, name="job-registry-watchdog", daemon=True
        )
        self._watchdog_thread.start()

    def _watchdog_loop(self) -> None:
        while not self._stop_watchdog.is_set():
            try:
                self._tick()
            except Exception as exc:
                logger.exception("Watchdog tick failed: %s", exc)
            self._stop_watchdog.wait(1.0)

    def _tick(self) -> None:
        with self._lock:
            running_ids = [jid for jid, r in self._records.items() if r.state == "running"]

        progress_snapshots: builtins.list[dict] = []

        for jid in running_ids:
            with self._lock:
                runner = self._runners.get(jid)
                record = self._records.get(jid)
            if runner is None or record is None:
                continue
            if runner.is_running():
                # Pull the wandb run URL once it appears in stdout.
                if record.wandb_run_url is None:
                    url = runner.wandb_run_url()
                    if url is not None:
                        with self._lock:
                            record.wandb_run_url = url
                        self._persist(record, force=True)
                # Persist metric snapshot at most once per second.
                self._persist(record, force=False)
                progress_snapshots.append(
                    {
                        "id": record.id,
                        "state": record.state,
                        "metrics": record.metrics.model_dump(),
                        "wandb_run_url": record.wandb_run_url,
                        "checkpoint_count": self._count_checkpoints(record),
                    }
                )
                continue

            # Subprocess exited since the last tick. Finalise.
            rc = runner.returncode()
            with self._lock:
                if record.wandb_run_url is None:
                    record.wandb_run_url = runner.wandb_run_url()
                record.state = "done" if rc == 0 else "failed"
                record.ended_at = time.time()
                record.exit_code = rc
                if rc != 0 and record.error_message is None:
                    # Prefer a runner-supplied reason (e.g. HF Jobs'
                    # 'Job timeout') over the synthetic exit-code message.
                    reason = None
                    get_message = getattr(runner, "terminal_message", None)
                    if callable(get_message):
                        try:
                            reason = get_message()
                        except Exception:
                            reason = None
                    record.error_message = reason or f"Subprocess exited with code {rc}"
                self._runners.pop(jid, None)
            self._persist(record, force=True)
            self._notify_change()

        self._notify_progress(progress_snapshots)

    def _persist(self, record: JobRecord, force: bool) -> None:
        now = time.time()
        last = self._last_persist_at.get(record.id, 0.0)
        if not force and (now - last) < _PERSIST_THROTTLE_SECONDS:
            return
        self._last_persist_at[record.id] = now
        self._write_meta(record)

    def _write_meta(self, record: JobRecord) -> None:
        path = _job_meta_path(self._output_root, record.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write so a crash mid-write never strands a half-written file
        # that would skip the job on next _load_from_disk.
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(record.model_dump_json(indent=2))
        os.replace(tmp, path)


# Module-level singleton. Anchored to ~/.cache so history survives launches
# from different cwds. JobRegistry.__init__ migrates legacy `<cwd>/outputs/train/`
# job dirs into this root on first boot. MAKERLAB_OUTPUT_ROOT overrides for tests.
_DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get("MAKERLAB_OUTPUT_ROOT")
    or (Path.home() / ".cache" / "huggingface" / "lerobot" / "outputs" / "train")
).expanduser()
job_registry = JobRegistry(_DEFAULT_OUTPUT_ROOT)

__all__ = [
    "JobState",
    "JobTarget",
    "TrainingMetrics",
    "LogLine",
    "JobRecord",
    "JobCheckpoint",
    "MetricsHistoryPoint",
    "JobRunner",
    "LocalJobRunner",
    "JobRegistry",
    "JobAlreadyRunningError",
    "JobNotFoundError",
    "JobNotRunningError",
    "job_registry",
    "parse_metrics_into",
]
