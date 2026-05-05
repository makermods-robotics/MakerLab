"""HF Jobs runner — runs a training as an HF Jobs job on HuggingFace's GPUs.

Uses huggingface/lerobot-gpu:latest as the runtime image (lerobot pre-installed).
Tails logs via HfApi.fetch_job_logs and reuses the existing parse_metrics_into
parser since stdout format is identical to a local lerobot run.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import List, Optional

from huggingface_hub import HfApi, get_token

from ..jobs import LogLine, TrainingMetrics, parse_metrics_into
from ..training import TrainingRequest, build_training_command

logger = logging.getLogger(__name__)

LEROBOT_IMAGE = "huggingface/lerobot-gpu:latest"


class HfCloudJobRunner:
    """Run a training as an HF Jobs job. Single-shot — instantiate per job."""

    def __init__(
        self,
        metrics: TrainingMetrics,
        log_file_path: Path,
        flavor: str,
    ) -> None:
        self._metrics = metrics
        self._log_file_path = log_file_path
        self._flavor = flavor
        self._api = HfApi()
        self._hf_job_id: Optional[str] = None
        self._log_queue: "Queue[LogLine]" = Queue()
        self._tail_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._log_file = None  # type: ignore[assignment]
        # Cached terminal status once the job ends; None while live.
        self._terminal_status: Optional[str] = None

    def start(self, job_id: str, config: TrainingRequest, output_dir: str) -> None:
        if self._hf_job_id is not None:
            raise RuntimeError("HfCloudJobRunner already started")

        token = get_token()
        if not token:
            raise RuntimeError(
                "HF token not found. Run 'hf auth login' before launching cloud jobs."
            )

        whoami = self._api.whoami()
        username = whoami.get("name") if isinstance(whoami, dict) else None
        if not username:
            raise RuntimeError("Could not resolve HF username from whoami()")

        # Mutate the config so build_training_command emits the right flags.
        # The mutated config is what gets persisted in JobRecord.config, so
        # the historical record reflects what actually ran.
        config.policy_push_to_hub = True
        # job_id is already a unique slug like "act_dataset_2026-05-04_10-22-03".
        config.policy_repo_id = f"{username}/{job_id}"

        argv = build_training_command(config, output_dir)
        logger.info("Submitting HF Cloud job %s on %s: %s",
                    job_id, self._flavor, " ".join(argv))

        # Open the persistent log sink — same shape as LocalJobRunner.
        self._log_file_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_file_path.open("a", buffering=1)

        # HF_TOKEN goes via `secrets` (not `env`) so it doesn't show up in
        # the job's environment variable inspection / logs.
        job = self._api.run_job(
            image=LEROBOT_IMAGE,
            command=argv,
            flavor=self._flavor,
            secrets={"HF_TOKEN": token},
        )
        self._hf_job_id = job.id

        self._tail_thread = threading.Thread(
            target=self._tail_loop, name=f"hf-job-{job_id}-logs", daemon=True
        )
        self._tail_thread.start()

    def reattach(self, hf_job_id: str) -> None:
        """Take over an existing HF job after a process restart.

        Skips submission; just opens the log file in append mode and starts
        the log-tailing thread. The watchdog will finalise based on inspect_job.
        """
        if self._hf_job_id is not None:
            raise RuntimeError("HfCloudJobRunner already started")
        self._hf_job_id = hf_job_id
        self._log_file_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_file_path.open("a", buffering=1)
        self._tail_thread = threading.Thread(
            target=self._tail_loop, name=f"hf-job-{hf_job_id}-logs-reattach", daemon=True
        )
        self._tail_thread.start()

    def _tail_loop(self) -> None:
        """Consume HfApi.fetch_job_logs until it returns. Tee each line to
        the log file and the in-memory queue, and update metrics inline.

        On disconnect, retry up to 3 times with exponential backoff. After
        that, exit the loop; the registry watchdog will catch the eventual
        terminal state via inspect_job.
        """
        assert self._hf_job_id is not None
        try:
            retries = 0
            while not self._stop_event.is_set():
                try:
                    for raw in self._api.fetch_job_logs(job_id=self._hf_job_id, follow=True):
                        if self._stop_event.is_set():
                            return
                        stripped = raw.rstrip()
                        if not stripped:
                            continue
                        parse_metrics_into(stripped, self._metrics)
                        log_line = LogLine(timestamp=time.time(), message=stripped)
                        if self._log_file is not None:
                            try:
                                self._log_file.write(log_line.model_dump_json() + "\n")
                            except Exception as exc:  # pragma: no cover
                                logger.exception("Error writing HF log: %s", exc)
                        if self._log_queue.qsize() >= 1000:
                            try:
                                self._log_queue.get_nowait()
                            except Empty:
                                pass
                        self._log_queue.put(log_line)
                    # Generator returned cleanly — job ended.
                    return
                except Exception as exc:
                    retries += 1
                    if retries > 3:
                        logger.warning(
                            "HF log tail gave up after 3 retries for job %s: %s",
                            self._hf_job_id, exc,
                        )
                        return
                    logger.info("HF log tail disconnected (retry %d/3): %s",
                                retries, exc)
                    self._stop_event.wait(2 ** retries)
        finally:
            if self._log_file is not None:
                try:
                    self._log_file.close()
                except Exception:
                    pass
                self._log_file = None

    def stop(self) -> None:
        if self._hf_job_id is None:
            return
        self._stop_event.set()
        try:
            self._api.cancel_job(job_id=self._hf_job_id)
        except Exception as exc:
            # Already-completed jobs may 404; that's fine. Watchdog will
            # finalise on its next tick.
            logger.info("cancel_job(%s) ignored: %s", self._hf_job_id, exc)

    def is_running(self) -> bool:
        if self._hf_job_id is None:
            return False
        try:
            info = self._api.inspect_job(job_id=self._hf_job_id)
        except Exception as exc:
            logger.warning("inspect_job failed for %s: %s", self._hf_job_id, exc)
            return False
        # Status values from huggingface_hub: QUEUED, RUNNING, COMPLETED,
        # FAILED, CANCELLED. We treat queued/running as alive.
        status = getattr(info, "status", None)
        status_str = str(status).upper() if status is not None else ""
        if status_str in {"QUEUED", "RUNNING"}:
            return True
        # Cache the terminal status so returncode() can map it without
        # re-calling the API.
        self._terminal_status = status_str
        return False

    def returncode(self) -> Optional[int]:
        if self._hf_job_id is None:
            return None
        # If we haven't yet observed the terminal status, ask now.
        if self._terminal_status is None and self.is_running():
            return None
        if self._terminal_status is None:
            return None
        return 0 if self._terminal_status == "COMPLETED" else 1

    def stream_log_lines(self) -> List[LogLine]:
        out: List[LogLine] = []
        try:
            while True:
                out.append(self._log_queue.get_nowait())
        except Empty:
            pass
        return out

    def hf_job_id(self) -> Optional[str]:
        return self._hf_job_id
