"""Job lifecycle and registry for trainings (and, in future, other long-running
work). One JobRunner instance owns one subprocess; the JobRegistry owns the
overall state, including history persisted to disk under outputs/train/."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import threading
import time
from queue import Empty, Queue
from typing import List, Literal, Optional, Protocol, runtime_checkable

from pydantic import BaseModel

from .training import TrainingRequest

logger = logging.getLogger(__name__)


JobState = Literal["running", "done", "failed", "interrupted"]


class TrainingMetrics(BaseModel):
    current_step: int = 0
    total_steps: int = 0
    current_loss: Optional[float] = None
    current_lr: Optional[float] = None
    grad_norm: Optional[float] = None
    eta_seconds: Optional[float] = None


class LogLine(BaseModel):
    timestamp: float
    message: str


class JobRecord(BaseModel):
    id: str
    name: str
    state: JobState
    config: TrainingRequest
    output_dir: str
    started_at: float
    ended_at: Optional[float] = None
    exit_code: Optional[int] = None
    error_message: Optional[str] = None
    metrics: TrainingMetrics = TrainingMetrics()
    runner: Literal["local"] = "local"


@runtime_checkable
class JobRunner(Protocol):
    """Backend interface for running one job. LocalJobRunner is the only impl
    today; remote runners (SSH, Slurm) drop in here later. @runtime_checkable
    lets `isinstance(r, JobRunner)` work in tests / sanity checks."""

    def start(self, job_id: str, config: TrainingRequest, output_dir: str) -> None: ...
    def stop(self) -> None: ...
    def is_running(self) -> bool: ...
    def returncode(self) -> Optional[int]: ...
    def stream_log_lines(self) -> List[LogLine]: ...


# tqdm progress: "Training:   1%|▏         | 125/10000 [02:02<2:36:10,  1.05step/s]"
_TQDM_RE = re.compile(
    r"Training:\s*\d+%[^|]*\|[^|]*\|\s*(\d+)/(\d+)\s*\[(?:[\d:]+)<([\d:]+)"
)


def _parse_duration(s: str) -> Optional[float]:
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


def parse_metrics_into(line: str, metrics: TrainingMetrics) -> None:
    """Update `metrics` in-place from one stdout line.

    Two complementary sources:
      * tqdm progress for current_step + total_steps + ETA (~1s cadence).
      * 'INFO ... step:N smpl:... loss:X grdn:Y lr:Z ...' for loss/lr/grdn
        (only at log_freq cadence, default every 250 steps).
    """
    try:
        tqdm_match = _TQDM_RE.search(line)
        if tqdm_match:
            try:
                metrics.current_step = int(tqdm_match.group(1))
                total = int(tqdm_match.group(2))
                if total > 0:
                    metrics.total_steps = total
                eta = _parse_duration(tqdm_match.group(3))
                if eta is not None:
                    metrics.eta_seconds = eta
            except (ValueError, IndexError):
                pass

        if "step:" in line and "loss:" in line:
            try:
                metrics.current_step = int(line.split("step:")[1].split()[0].replace(",", ""))
            except ValueError:
                pass
            try:
                metrics.current_loss = float(line.split("loss:")[1].split()[0])
            except ValueError:
                pass
            if "lr:" in line:
                try:
                    metrics.current_lr = float(line.split("lr:")[1].split()[0])
                except ValueError:
                    pass
            if "grdn:" in line:
                try:
                    metrics.grad_norm = float(line.split("grdn:")[1].split()[0])
                except ValueError:
                    pass

    except Exception as exc:
        logger.debug("Error parsing log line %r: %s", line, exc)


class LocalJobRunner:
    """Run a training as a local subprocess.

    The runner is single-shot: instantiate a fresh one per job. Lifetime of
    the underlying subprocess is bounded by this object's existence in memory.
    """

    def __init__(self, metrics: TrainingMetrics) -> None:
        self._metrics = metrics
        self._process: Optional[subprocess.Popen] = None
        self._log_queue: "Queue[LogLine]" = Queue()
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(
        self,
        job_id: str,
        config: TrainingRequest,
        output_dir: str,
    ) -> None:
        if self._process is not None:
            raise RuntimeError("LocalJobRunner already started")

        # Build the command via the helper that lives in training.py.
        from .training import build_training_command  # avoid import cycle at module load
        cmd = build_training_command(config, output_dir)
        logger.info("Starting job %s: %s", job_id, " ".join(cmd))

        # PYTHONUNBUFFERED makes the child's stdout flush per line. Without it
        # block-buffering hides log lines from our parser for many seconds.
        child_env = os.environ.copy()
        child_env["PYTHONUNBUFFERED"] = "1"

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env=child_env,
        )

        self._monitor_thread = threading.Thread(
            target=self._pump_stdout, name=f"job-{job_id}-stdout", daemon=True
        )
        self._monitor_thread.start()

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

    def returncode(self) -> Optional[int]:
        if self._process is None:
            return None
        return self._process.poll()

    def stream_log_lines(self) -> List[LogLine]:
        """Drain whatever has accumulated since the last call."""
        out: List[LogLine] = []
        try:
            while True:
                out.append(self._log_queue.get_nowait())
        except Empty:
            pass
        return out

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
                parse_metrics_into(stripped, self._metrics)
                # Cap queue so a chatty subprocess can't grow memory unbounded.
                if self._log_queue.qsize() >= 1000:
                    try:
                        self._log_queue.get_nowait()
                    except Empty:
                        pass
                self._log_queue.put(LogLine(timestamp=time.time(), message=stripped))
        except Exception as exc:
            logger.exception("Error reading subprocess stdout: %s", exc)


# Re-exported here so callers don't need to know they came from training.py.
# Filled in by Task 2 (LocalJobRunner) and Task 3 (JobRegistry).
__all__ = [
    "JobState",
    "TrainingMetrics",
    "LogLine",
    "JobRecord",
    "JobRunner",
    "parse_metrics_into",
]
