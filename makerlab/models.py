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
"""Listing of the user's models for the Model Library UI: Hugging Face Hub
repos plus models that exist on this machine (local training runs, imported
local directories), each tagged with a `source` so the frontend can show
which are local and which are on the Hub."""

import logging
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from huggingface_hub.errors import HfHubHTTPError

from .utils.hf_auth import cached_whoami, shared_hf_api

logger = logging.getLogger(__name__)

# A makerlab cloud-training run repo is named "<policy>_<namespace>_<dataset>_<ts>"
# where the trailing "_YYYY-MM-DD_HH-MM-SS" is stamped by _generate_job_id()
# (jobs.py). Same pattern as server.py's /jobs/hub handler — match on the
# timestamp suffix so it stays policy-agnostic. Here it only annotates each
# entry with a `cloud_run` flag; unlike /jobs/hub it does NOT narrow the listing.
_RUN_REPO_RE = re.compile(r"_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")


def list_user_models() -> list[dict[str, Any]]:
    """All model repos owned by the user and their orgs, annotated for the
    library filters.

    Unlike list_user_datasets, this applies NO narrowing filter — every repo the
    author owns is returned. Each entry carries `lerobot` (has the lowercase
    "lerobot" library tag) and `cloud_run` (name matches a makerlab run-repo
    timestamp) flags so the frontend can filter without a second round-trip.
    """
    info = cached_whoami()
    if info is None:
        return []

    authors = [info["name"]] + [o["name"] for o in info.get("orgs", [])]
    api = shared_hf_api()
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for author in authors:
        try:
            for m in api.list_models(
                author=author, limit=200, expand=["lastModified", "private", "tags", "createdAt"]
            ):
                if m.id in seen:
                    continue
                seen.add(m.id)
                tags = getattr(m, "tags", None) or []
                name = m.id.split("/", 1)[-1]
                created = getattr(m, "created_at", None)
                out.append(
                    {
                        "repo_id": m.id,
                        "last_modified": m.last_modified.isoformat() if m.last_modified else None,
                        "created_at": created.isoformat() if created else None,
                        "private": bool(getattr(m, "private", False)),
                        "lerobot": "lerobot" in tags,
                        "cloud_run": bool(_RUN_REPO_RE.search(name)),
                        "source": "hub",
                    }
                )
        except HfHubHTTPError as e:
            logger.warning(f"list_models({author}) failed: {e}")

    out.sort(key=_recency_key, reverse=True)
    return out


def _recency_key(entry: dict[str, Any]) -> str:
    """Most-recently-ADDED ordering: created_at when known, else last_modified.
    ISO strings sort lexically."""
    return entry.get("created_at") or entry.get("last_modified") or ""


def _epoch_iso(ts: float | None) -> str | None:
    if not ts:
        return None
    return datetime.fromtimestamp(ts, tz=UTC).isoformat()


def list_all_models(records: Iterable[Any]) -> list[dict[str, Any]]:
    """Hub repos plus models that exist on this machine, sorted newest-added
    first.

    `records` are JobRecords (duck-typed to avoid a jobs import cycle). Local =
    a job whose model artifacts are on disk: an imported local directory, or a
    local training run with at least one checkpoint. Hub imports (pointer
    records carrying hf_repo_id) are NOT local — their files live on the Hub
    and the hub listing already carries the repo; those hub entries get the
    pointer's `job_id` instead, so run/fine-tune skips the redundant re-import.
    Local entries carry `job_id` for the same reason.
    """
    out = list_user_models()
    tracked = {
        r.hf_repo_id: r.id for r in records if r.runner == "imported" and r.hf_repo_id
    }
    for entry in out:
        job_id = tracked.get(entry["repo_id"])
        if job_id:
            entry["job_id"] = job_id
    for r in records:
        if r.runner == "imported":
            if r.hf_repo_id or not r.output_dir:
                continue
        elif r.runner == "local":
            # A still-training run isn't a browsable model yet (and offering
            # fine-tune on it would collide with the active session) — it
            # already shows prominently as a job card.
            if r.checkpoint_count < 1 or r.state == "running":
                continue
        else:
            continue
        out.append(
            {
                "repo_id": r.display_name or r.name,
                "job_id": r.id,
                "last_modified": _epoch_iso(r.ended_at or r.started_at),
                "created_at": _epoch_iso(r.started_at),
                # Not on the Hub at all — "private" in the only sense that
                # matters to the filters (never publicly visible).
                "private": True,
                "lerobot": False,
                "cloud_run": False,
                "source": "local",
            }
        )
    out.sort(key=_recency_key, reverse=True)
    return out
