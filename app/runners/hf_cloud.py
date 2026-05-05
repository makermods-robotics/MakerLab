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

# Inlined sidecar uploader for HF Jobs. Spawns the lerobot trainer as a
# subprocess and concurrently uploads new <output_dir>/checkpoints/<step>/
# directories to the Hub model repo, so the lelab UI can list them while
# training is in progress.
#
# Sent verbatim as the value of `python -c '...'`. Anything after `--` in
# the command argv is forwarded to the trainer.
WRAPPER_SOURCE = r'''
import os, re, sys, threading, subprocess
from pathlib import Path
from huggingface_hub import HfApi

argv = sys.argv[1:]
if "--" not in argv:
    print("[wrapper] missing -- separator", flush=True)
    sys.exit(2)
sep = argv.index("--")
trainer_argv = argv[sep + 1:]


def _arg(name):
    """Return the value of --name=foo or --name foo from trainer_argv."""
    for i, tok in enumerate(trainer_argv):
        if tok == name and i + 1 < len(trainer_argv):
            return trainer_argv[i + 1]
        if tok.startswith(name + "="):
            return tok.split("=", 1)[1]
    return None


output_dir = _arg("--output_dir")
repo_id = _arg("--policy.repo_id")
if not output_dir or not repo_id:
    print(f"[wrapper] need --output_dir and --policy.repo_id; got {output_dir} / {repo_id}", flush=True)
    sys.exit(2)

api = HfApi()
seen = set()
stop_event = threading.Event()


def _scan_and_upload():
    root = Path(output_dir) / "checkpoints"
    if not root.is_dir():
        return
    # Snapshot before iterating so deletions during the walk do not raise.
    entries = sorted(p for p in root.iterdir() if p.is_dir() and not p.is_symlink())
    for entry in entries:
        if not re.fullmatch(r"\d+", entry.name):
            continue
        config_json = entry / "pretrained_model" / "config.json"
        if not config_json.is_file():
            continue
        if entry.name in seen:
            continue
        try:
            api.upload_folder(
                folder_path=str(entry),
                repo_id=repo_id,
                path_in_repo=f"checkpoints/{entry.name}",
                commit_message=f"checkpoint {entry.name}",
            )
            seen.add(entry.name)
            print(f"[wrapper] uploaded checkpoint {entry.name}", flush=True)
        except Exception as exc:
            print(f"[wrapper] upload failed for {entry.name}: {exc}", flush=True)


def _watch():
    while not stop_event.is_set():
        try:
            _scan_and_upload()
        except Exception as exc:
            print(f"[wrapper] scan error: {exc}", flush=True)
        stop_event.wait(15)


watch_thread = threading.Thread(target=_watch, name="ckpt-watcher", daemon=True)
watch_thread.start()

print(f"[wrapper] launching trainer: {' '.join(trainer_argv)}", flush=True)
proc = subprocess.Popen(list(trainer_argv), env=os.environ.copy())
try:
    rc = proc.wait()
finally:
    stop_event.set()
    # One final pass picks up any checkpoint saved in the last 15s window.
    try:
        _scan_and_upload()
    except Exception as exc:
        print(f"[wrapper] final scan error: {exc}", flush=True)

print(f"[wrapper] trainer exited with rc={rc}", flush=True)
sys.exit(rc)
'''

# HF Jobs' platform default timeout has killed legitimate runs that pushed
# the model successfully but were still uploading auxiliary files. 2h covers
# our typical ACT/SmolVLA runs on t4-small with comfortable headroom.
HF_JOB_TIMEOUT = "2h"


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
        self._hf_job_url: Optional[str] = None
        self._log_queue: "Queue[LogLine]" = Queue()
        self._tail_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._log_file = None  # type: ignore[assignment]
        # Cached terminal status once the job ends; None while live.
        self._terminal_status: Optional[str] = None
        # Status.message at the terminal tick (e.g. "Job timeout"), so the
        # registry can surface it to the UI instead of a synthetic exit code.
        self._terminal_message: Optional[str] = None

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

        trainer_argv = build_training_command(config, output_dir)
        # The wrapper expects `python -c WRAPPER_SOURCE -- <trainer argv>`.
        # `python -c` consumes the first non-option argument as the script,
        # so we prepend a "--" sentinel of our own.
        wrapped_command = ["python", "-c", WRAPPER_SOURCE, "--", *trainer_argv]
        logger.info(
            "Submitting HF Cloud job %s on %s (wrapped trainer): %s",
            job_id, self._flavor, " ".join(trainer_argv),
        )

        self._log_file_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_file_path.open("a", buffering=1)

        # HF_TOKEN goes via `secrets` (not `env`) so it doesn't show up in
        # the job's environment variable inspection / logs.
        job = self._api.run_job(
            image=LEROBOT_IMAGE,
            command=wrapped_command,
            flavor=self._flavor,
            secrets={"HF_TOKEN": token},
            timeout=HF_JOB_TIMEOUT,
        )
        self._hf_job_id = job.id
        self._hf_job_url = getattr(job, "url", None)

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
        # info.status is a JobStatus dataclass; the actual stage string
        # ("RUNNING", "COMPLETED", "ERROR", transient values like
        # "QUEUED"/"SCHEDULING", …) lives on .stage. Documented terminal
        # values: COMPLETED, CANCELED, ERROR, DELETED. We also accept
        # CANCELLED/FAILED in case the API surfaces alternative spellings.
        # Anything else — including unknown future states — is treated as
        # alive so we don't prematurely finalise a healthy job.
        status_obj = getattr(info, "status", None)
        stage = getattr(status_obj, "stage", None) if status_obj is not None else None
        stage_str = str(stage).upper() if stage is not None else ""
        terminal = {"COMPLETED", "CANCELED", "CANCELLED", "ERROR", "FAILED", "DELETED"}
        if stage_str in terminal:
            self._terminal_status = stage_str
            message = getattr(status_obj, "message", None)
            if message:
                self._terminal_message = str(message)
            return False
        return True

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

    def hf_job_url(self) -> Optional[str]:
        return self._hf_job_url

    def terminal_message(self) -> Optional[str]:
        """Status.message captured when the job reached a terminal stage.

        Set by the most recent is_running() call that observed a terminal
        stage. Used by the registry watchdog to surface platform reasons
        like 'Job timeout' rather than a synthetic 'exit code 1'.
        """
        return self._terminal_message
