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

import json
import logging
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub.errors import HfHubHTTPError

from .utils.config import validate_dataset_name
from .utils.hf_auth import cached_whoami, shared_hf_api

logger = logging.getLogger(__name__)

CAMERA_FEATURE_PREFIX = "observation.images."

# In-process cache of Hub existence checks, keyed by repo_id. /whoami-v2 and
# repo-existence lookups hit the network, so the info card fetches this lazily
# and we memoize the "on Hub" answer for the process lifetime. A successful
# upload invalidates the entry (see invalidate_hub_status), so the card can
# flip Local only -> On Hub without waiting for a cache expiry. "unknown" (the
# offline/unauthenticated/error degrade) is never cached, so connectivity
# returning is picked up on the next check.
_HUB_STATUS_CACHE: dict[str, str] = {}
_HUB_STATUS_LOCK = threading.Lock()


def invalidate_hub_status(repo_id: str) -> None:
    """Drop the cached Hub-existence answer for `repo_id`. Called after a
    successful upload so the next /datasets/hub-status re-checks (and sees
    the freshly pushed repo)."""
    with _HUB_STATUS_LOCK:
        _HUB_STATUS_CACHE.pop(repo_id, None)


def get_hub_status(repo_id: str) -> dict[str, Any]:
    """Whether a dataset repo with this id exists on the Hub.

    Returns ``{"repo_id": ..., "status": "on_hub" | "local_only" | "unknown",
    "url": <hub url> | None}``. Never raises: offline, unauthenticated, or any
    transport error degrades to ``"unknown"`` (no error spam — the card just
    hides the badge). Definitive answers (exists / doesn't) are memoized per
    repo_id for the process lifetime; ``"unknown"`` is not cached so transient
    failures self-heal on the next check.
    """
    url = f"https://huggingface.co/datasets/{repo_id}"

    with _HUB_STATUS_LOCK:
        cached = _HUB_STATUS_CACHE.get(repo_id)
    if cached is not None:
        return {"repo_id": repo_id, "status": cached, "url": url if cached == "on_hub" else None}

    api = shared_hf_api()
    try:
        exists = api.repo_exists(repo_id, repo_type="dataset")
    except Exception as exc:
        # Offline / rate-limited / any other transport error: degrade to
        # "unknown" without caching so it re-checks once connectivity returns.
        logger.info("hub-status repo_exists(%s) failed: %s", repo_id, exc)
        return {"repo_id": repo_id, "status": "unknown", "url": None}

    status = "on_hub" if exists else "local_only"
    with _HUB_STATUS_LOCK:
        _HUB_STATUS_CACHE[repo_id] = status
    return {"repo_id": repo_id, "status": status, "url": url if exists else None}


def _lerobot_cache_root() -> Path:
    return Path(os.environ.get("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")).expanduser()


def _is_dataset_dir(path: Path) -> bool:
    """A directory is a LeRobot dataset iff <dir>/meta/info.json exists."""
    try:
        return (path / "meta" / "info.json").is_file()
    except OSError:
        return False


def _dataset_has_episodes(path: Path) -> bool:
    """True if the dataset recorded at least one episode. An empty dataset
    (0 episodes — e.g. a recording aborted before saving) has no task/data
    files and only breaks downstream steps like training and merging, so we
    hide it from the listing rather than let it be selected."""
    try:
        info = json.loads((path / "meta" / "info.json").read_text())
    except (OSError, ValueError):
        return False
    return bool(info.get("total_episodes"))


def _dir_mtime_iso(path: Path) -> str | None:
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts, tz=UTC).isoformat()
    except OSError:
        return None


def _dir_birthtime_iso(path: Path) -> str | None:
    """Directory creation time (macOS st_birthtime), falling back to mtime on
    filesystems that don't record it. Feeds the newest-added-first ordering."""
    try:
        st = path.stat()
        ts = getattr(st, "st_birthtime", None) or st.st_mtime
        return datetime.fromtimestamp(ts, tz=UTC).isoformat()
    except OSError:
        return None


def list_local_datasets() -> list[dict[str, Any]]:
    """Scan the LeRobot cache for local datasets (dirs containing meta/info.json).

    Walks one level deep: a top-level dataset dir is recorded as "<name>"; if a
    top-level dir is not itself a dataset, each subdir that is a dataset is
    recorded as "<top>/<sub>". Does not descend further.
    """
    root = _lerobot_cache_root()
    if not root.is_dir():
        return []

    out: list[dict[str, Any]] = []
    try:
        top_entries = list(root.iterdir())
    except OSError as e:
        logger.warning(f"Could not read LeRobot cache root {root}: {e}")
        return []

    for top in top_entries:
        try:
            if not top.is_dir():
                continue
        except OSError:
            continue

        if _is_dataset_dir(top):
            # It IS a dataset (empty or not) — record it only if non-empty, but
            # don't descend into its subdirs either way.
            if _dataset_has_episodes(top):
                out.append(
                    {
                        "repo_id": top.name,
                        "last_modified": _dir_mtime_iso(top),
                        "created_at": _dir_birthtime_iso(top),
                        "private": False,
                    }
                )
            continue

        # Not a dataset itself — descend one level.
        try:
            sub_entries = list(top.iterdir())
        except OSError:
            continue
        for sub in sub_entries:
            try:
                if not sub.is_dir():
                    continue
            except OSError:
                continue
            if _is_dataset_dir(sub) and _dataset_has_episodes(sub):
                out.append(
                    {
                        "repo_id": f"{top.name}/{sub.name}",
                        "last_modified": _dir_mtime_iso(sub),
                        "created_at": _dir_birthtime_iso(sub),
                        "private": False,
                    }
                )

    out.sort(key=_recency_key, reverse=True)
    return out


def _read_task_strings(meta_dir: Path) -> list[str]:
    """Task strings for a dataset, ordered by task_index.

    v3.0 datasets keep them in ``meta/tasks.parquet`` (columns ``task_index``
    and ``task``; pandas stores ``task`` as the frame index, but pyarrow
    surfaces both as plain columns). Older v2.x datasets use
    ``meta/tasks.jsonl`` with one ``{"task_index": …, "task": …}`` object per
    line. Unreadable/absent task metadata degrades to an empty list — the info
    card can render without it.
    """
    parquet_path = meta_dir / "tasks.parquet"
    if parquet_path.is_file():
        try:
            table = pq.read_table(parquet_path).to_pydict()
            rows = sorted(zip(table.get("task_index", []), table.get("task", []), strict=True))
            return [str(task) for _, task in rows]
        except Exception as e:
            logger.warning(f"Could not read {parquet_path}: {e}")
            return []

    jsonl_path = meta_dir / "tasks.jsonl"
    if jsonl_path.is_file():
        try:
            rows = []
            for line in jsonl_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rows.append((obj.get("task_index", 0), str(obj.get("task", ""))))
            rows.sort()
            return [task for _, task in rows if task]
        except (OSError, ValueError) as e:
            logger.warning(f"Could not read {jsonl_path}: {e}")

    return []


def _count_task_episodes(meta_dir: Path) -> dict[str, int]:
    """Episodes per task string, from the per-episode ``tasks`` column.

    Each episode lists the task strings it uses, and an episode counts once
    per distinct task. Read directly from the metadata files —
    v3.0 keeps episode rows in ``meta/episodes/chunk-*/file-*.parquet`` (only
    the ``tasks`` column is loaded, not the wide per-episode stats), v2.x in
    ``meta/episodes.jsonl`` — so the endpoint stays a cheap file read instead
    of a full ``LeRobotDataset`` load. Unreadable/absent episode metadata
    degrades to an empty dict (counts render as 0).
    """
    counts: dict[str, int] = {}

    episodes_dir = meta_dir / "episodes"
    if episodes_dir.is_dir():
        for parquet_path in sorted(episodes_dir.glob("**/*.parquet")):
            try:
                table = pq.read_table(parquet_path, columns=["tasks"])
            except Exception as e:
                logger.warning(f"Could not read {parquet_path}: {e}")
                continue
            for episode_tasks in table.column("tasks").to_pylist():
                for task in set(episode_tasks or []):
                    counts[str(task)] = counts.get(str(task), 0) + 1
        return counts

    jsonl_path = meta_dir / "episodes.jsonl"
    if jsonl_path.is_file():
        try:
            for line in jsonl_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                for task in set(obj.get("tasks") or []):
                    counts[str(task)] = counts.get(str(task), 0) + 1
        except (OSError, ValueError) as e:
            logger.warning(f"Could not read {jsonl_path}: {e}")

    return counts


def _dir_size_bytes(path: Path) -> int:
    """Total size of all files under `path`. Unreadable files are skipped."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            try:
                total += os.stat(os.path.join(dirpath, name)).st_size
            except OSError:
                continue
    return total


def get_local_dataset_info(repo_id: str) -> dict[str, Any] | None:
    """Detail view of one locally-cached dataset, for the selection info card.

    Reads ``meta/info.json`` + task metadata and walks the directory for its
    size on disk — per-dataset on demand, so the cheap ``/datasets`` listing
    stays cheap. Returns None if `repo_id` escapes the cache root or isn't a
    local dataset (e.g. it only exists on the Hub).
    """
    root = _lerobot_cache_root().resolve()
    try:
        path = (root / repo_id).resolve()
    except OSError:
        return None
    # Reject path traversal: the dataset dir must stay strictly inside the cache.
    if path == root or root not in path.parents:
        return None
    if not _is_dataset_dir(path):
        return None

    try:
        info = json.loads((path / "meta" / "info.json").read_text())
    except (OSError, ValueError):
        return None

    features = info.get("features") or {}
    cameras = [key[len(CAMERA_FEATURE_PREFIX) :] for key in features if key.startswith(CAMERA_FEATURE_PREFIX)]

    task_counts = _count_task_episodes(path / "meta")
    tasks = [
        {"task": task, "num_episodes": task_counts.get(task, 0)} for task in _read_task_strings(path / "meta")
    ]

    return {
        "repo_id": repo_id,
        "total_episodes": int(info.get("total_episodes") or 0),
        "total_frames": int(info.get("total_frames") or 0),
        "fps": info.get("fps"),
        "robot_type": info.get("robot_type"),
        "cameras": cameras,
        "tasks": tasks,
        "size_bytes": _dir_size_bytes(path),
    }


class DatasetRenameError(Exception):
    """Raised by rename_local_dataset when the rename can't proceed. `status`
    is the HTTP status the route should return (400 invalid, 404 not found,
    409 conflict/busy); `message` is the user-facing reason."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _dataset_in_use(repo_id: str) -> str | None:
    """If `repo_id`'s directory is in use by a running operation, return a
    legible reason to refuse a rename; else None.

    Checks the four ways a dataset dir can be actively read/written:
      * recording — the active session's (timestamp-stamped) repo id, OR the
        base name the user typed (recording stamps ``name`` → ``name_<ts>``,
        so a rename of the base while a session writes ``name_<ts>`` would
        pull the directory out from under it);
      * merge — the output dataset currently being aggregated;
      * upload — the dataset currently being pushed to the Hub (renaming or
        deleting the directory mid-push would corrupt the upload);
      * local training — any running local job whose config trains on it.

    Read-only imports; each module owns its own state. Kept deliberately
    simple: a false "in use" is safer than yanking a directory mid-write.
    """
    # Recording: record.py owns recording_active + recording_config.
    from . import record as _record

    if _record.recording_active and _record.recording_config is not None:
        active_id = getattr(_record.recording_config, "dataset_repo_id", None)
        # The session stamps a timestamp onto the base name (name -> name_<ts>),
        # so match either the stamped id or a rename of the still-writing base.
        if active_id and (active_id == repo_id or active_id.startswith(f"{repo_id}_")):
            return "A recording session is writing to this dataset. Stop it before renaming."

    # Upload: record.py owns an UploadManager singleton (state + repo_id). Same
    # lazy import (datasets<->record cycle) as recording above.
    if _record.upload_manager.state == "running" and _record.upload_manager.repo_id == repo_id:
        return "This dataset is being uploaded to the Hub right now. Wait for it to finish."

    # Merge: merge.py exposes a MergeManager singleton with state + output id.
    from . import merge as _merge

    mgr = _merge.merge_manager
    if mgr.state == "running" and mgr.output_repo_id == repo_id:
        return "A merge is producing this dataset right now. Wait for it to finish before renaming."

    # Local training: a running local job whose config trains on this dataset.
    from .jobs import job_registry

    for record in job_registry.list(limit=200):
        if (
            record.state == "running"
            and record.runner == "local"
            and record.config.dataset_repo_id == repo_id
        ):
            return "A local training run is using this dataset. Stop it before renaming."

    return None


def rename_local_dataset(repo_id: str, new_name: str) -> str:
    """Rename a locally-cached dataset by moving its directory.

    A dataset's repo id *is* its path under the cache root, so a rename is a
    directory move. `new_name` is the NAME PART ONLY — the namespace prefix is
    fixed, so ``ns/old`` renamed to ``new`` becomes ``ns/new`` and a bare
    ``old`` becomes ``new``. Returns the new repo id.

    Raises DatasetRenameError (with an HTTP status + message) on: a bad
    new_name, a source that isn't a local dataset, a target that already
    exists, or the dataset being actively used (recording / merge / local
    training). Invalidates the cached Hub-existence answer for BOTH ids so the
    info card re-checks after the move.
    """
    ok, reason = validate_dataset_name(new_name)
    if not ok:
        raise DatasetRenameError(400, reason)

    root = _lerobot_cache_root().resolve()
    try:
        src = (root / repo_id).resolve()
    except OSError:
        raise DatasetRenameError(400, "Invalid dataset path") from None
    # Reject path traversal: the source must stay strictly inside the cache.
    if src == root or root not in src.parents:
        raise DatasetRenameError(400, "Invalid dataset path")
    if not _is_dataset_dir(src):
        raise DatasetRenameError(404, f"Dataset '{repo_id}' not found in the local cache")

    # The namespace prefix is fixed — swap only the final path segment.
    namespace = repo_id.rsplit("/", 1)[0] if "/" in repo_id else None
    new_repo_id = f"{namespace}/{new_name}" if namespace else new_name
    if new_repo_id == repo_id:
        return repo_id  # no-op

    dst = src.parent / new_name
    if dst.exists():
        raise DatasetRenameError(409, f"A dataset named '{new_repo_id}' already exists.")

    in_use = _dataset_in_use(repo_id)
    if in_use is not None:
        raise DatasetRenameError(409, in_use)

    try:
        os.rename(src, dst)
    except OSError as exc:
        logger.error("Failed to rename dataset %s -> %s: %s", repo_id, new_repo_id, exc)
        raise DatasetRenameError(500, f"Failed to rename dataset: {exc}") from exc

    # The old id no longer exists and the new id now does — drop both cached
    # Hub-existence answers so the next hub-status check re-queries.
    invalidate_hub_status(repo_id)
    invalidate_hub_status(new_repo_id)

    logger.info("Renamed dataset directory %s -> %s", src, dst)
    return new_repo_id


def list_user_datasets() -> list[dict[str, Any]]:
    info = cached_whoami()
    if info is None:
        return []

    authors = [info["name"]] + [o["name"] for o in info.get("orgs", [])]
    api = shared_hf_api()
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for author in authors:
        try:
            for ds in api.list_datasets(
                author=author,
                filter="LeRobot",
                limit=200,
                expand=["lastModified", "private", "createdAt"],
            ):
                if ds.id in seen:
                    continue
                seen.add(ds.id)
                created = getattr(ds, "created_at", None)
                out.append(
                    {
                        "repo_id": ds.id,
                        "last_modified": ds.last_modified.isoformat() if ds.last_modified else None,
                        "created_at": created.isoformat() if created else None,
                        "private": bool(getattr(ds, "private", False)),
                    }
                )
        except HfHubHTTPError as e:
            logger.warning(f"list_datasets({author}) failed: {e}")

    out.sort(key=_recency_key, reverse=True)
    return out


def _recency_key(entry: dict[str, Any]) -> str:
    """Most-recently-ADDED ordering: created_at when known, else last_modified.
    ISO strings sort lexically."""
    return entry.get("created_at") or entry.get("last_modified") or ""


def list_all_datasets() -> list[dict[str, Any]]:
    """Merged listing: Hub datasets + local cache, with `source` field, sorted
    newest-added first.

    A repo_id present in both lists is collapsed to one entry with
    source="both", last_modified set to the more recent of the two, and
    created_at to the older (when the dataset first existed anywhere).
    """
    hub = list_user_datasets()
    local = list_local_datasets()

    merged: dict[str, dict[str, Any]] = {}
    for item in hub:
        merged[item["repo_id"]] = {**item, "source": "hub"}
    for item in local:
        rid = item["repo_id"]
        if rid in merged:
            existing = merged[rid]
            existing["source"] = "both"
            # Keep the newer timestamp; ISO strings sort lexically.
            a = existing.get("last_modified") or ""
            b = item.get("last_modified") or ""
            existing["last_modified"] = max(a, b) or None
            created = [c for c in (existing.get("created_at"), item.get("created_at")) if c]
            existing["created_at"] = min(created) if created else None
        else:
            merged[rid] = {**item, "source": "local"}

    out = list(merged.values())
    out.sort(key=_recency_key, reverse=True)
    return out
