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

"""Dataset merging: wrap lerobot's ``aggregate_datasets`` as a background job.

Aggregation copies every episode's parquet + video files and recomputes stats,
so it can take minutes for large datasets. We run it in a subprocess (same
shape as training/pip-install) and stream its stdout for a live progress log,
rather than blocking a server thread on CPU-bound work.

The subprocess entry is ``python -m lelab.merge <output_repo_id> <src> <src>…``.
"""

import argparse
import contextlib
import logging
import queue
import subprocess
import sys
import threading
import time
from typing import Any

from pydantic import BaseModel

from lerobot.datasets.aggregate import aggregate_datasets

logger = logging.getLogger(__name__)


class MergeRequest(BaseModel):
    source_repo_ids: list[str]
    output_repo_id: str


class MergeManager:
    """Runs one dataset merge at a time as a tracked subprocess."""

    def __init__(self) -> None:
        self.state: str = "idle"  # "idle" | "running" | "done" | "error"
        self.error: str | None = None
        self.output_repo_id: str | None = None
        self.process: subprocess.Popen | None = None
        self.log_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self, request: MergeRequest) -> dict[str, Any]:
        sources = [s for s in request.source_repo_ids if s.strip()]
        output = request.output_repo_id.strip()
        with self._lock:
            if self.state == "running":
                return {"started": False, "message": "A merge is already in progress"}
            if len(sources) < 2:
                return {"started": False, "message": "Select at least two datasets to merge"}
            if not output:
                return {"started": False, "message": "An output dataset name is required"}
            if output in sources:
                return {"started": False, "message": "Output name must differ from the sources"}
            self.state = "running"
            self.error = None
            self.output_repo_id = output
            self._drain_queue()

        cmd = [sys.executable, "-m", "lelab.merge", output, *sources]
        logger.info("Starting dataset merge: %s", " ".join(cmd))
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )
        except Exception as exc:
            logger.exception("Failed to spawn merge subprocess")
            with self._lock:
                self.state = "error"
                self.error = f"Failed to spawn merge: {exc}"
            return {"started": False, "message": str(exc)}

        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()
        return {"started": True, "message": "Merge started"}

    def get_status(self) -> dict[str, Any]:
        logs: list[dict[str, Any]] = []
        with contextlib.suppress(queue.Empty):
            while True:
                logs.append(self.log_queue.get_nowait())
        return {
            "state": self.state,
            "error": self.error,
            "output_repo_id": self.output_repo_id,
            "logs": logs,
        }

    def _monitor(self) -> None:
        assert self.process is not None
        try:
            for line in iter(self.process.stdout.readline, ""):
                if not line:
                    break
                self._enqueue(line.rstrip())
        except Exception as exc:  # pragma: no cover — best-effort streaming
            logger.exception("Error reading merge output")
            self._enqueue(f"[merge] error reading output: {exc}")
        self.process.wait()
        return_code = self.process.returncode
        with self._lock:
            if return_code == 0:
                self.state = "done"
                self.error = None
            else:
                self.state = "error"
                self.error = f"Merge exited with code {return_code}"

    def _enqueue(self, message: str) -> None:
        # Cap the queue so a chatty subprocess can't grow memory unbounded.
        if self.log_queue.qsize() >= 1000:
            with contextlib.suppress(queue.Empty):
                self.log_queue.get_nowait()
        self.log_queue.put({"timestamp": time.time(), "message": message})

    def _drain_queue(self) -> None:
        with contextlib.suppress(queue.Empty):
            while True:
                self.log_queue.get_nowait()


merge_manager = MergeManager()


def handle_start_merge(request: MergeRequest) -> dict[str, Any]:
    return merge_manager.start(request)


def handle_merge_status() -> dict[str, Any]:
    return merge_manager.get_status()


def _run_cli(argv: list[str] | None = None) -> int:
    """Subprocess entry: aggregate the source datasets into the output repo."""
    parser = argparse.ArgumentParser(description="Merge LeRobot datasets")
    parser.add_argument("output_repo_id")
    parser.add_argument("source_repo_ids", nargs="+")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    print(f"Merging {len(args.source_repo_ids)} datasets → {args.output_repo_id}", flush=True)
    aggregate_datasets(
        repo_ids=args.source_repo_ids,
        aggr_repo_id=args.output_repo_id,
    )
    print(f"Done. Created {args.output_repo_id}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(_run_cli())
