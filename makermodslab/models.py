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

"""Trained-model browser: the datasets.py of policies.

Mirrors `makermodslab/datasets.py` for MODELS instead of datasets:

  * LOCAL models are the final checkpoint of each COMPLETED local training run,
    read straight from the job registry (`makermodslab/jobs.py`) — we never re-track
    jobs here, only read them. A run's final checkpoint is the highest-step
    `checkpoints/<step>/pretrained_model/` dir; its `train_config.json` supplies
    the policy type, base dataset repo_id, and step target.
  * HUB models are the user's LeRobot policy repos on the Hub. Listed with the
    same per-author fan-out + resilience the datasets listing uses (a GFW-killed
    TLS handshake / timeout for one author degrades to the others rather than
    500ing), reusing `datasets._fan_out_hub_authors`.

`list_all_models` merges the two into one listing with a `source` of
"local" / "hub" / "both", exactly like `list_all_datasets`. Upload pushes a
local checkpoint to the Hub as a public, MakerModsLab-tagged model repo; delete removes
a local run's output dir (strictly sandboxed under outputs/train/).
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from huggingface_hub import metadata_update, snapshot_download

from .datasets import (
    DownloadManager,
    _dir_mtime_iso,
    _fan_out_hub_authors,
    _lerobot_cache_root,
)
from .jobs import (
    JobRecord,
    _list_local_checkpoints,
    _read_checkpoint_config,
    job_registry,
)
from .utils.config import (
    get_hidden_models,
    get_saved_custom_models,
    validate_dataset_repo_id,
    with_makermodslab_tag,
)
from .utils.hf_auth import cached_whoami, hf_hub_offline, shared_hf_api
from .utils.naming import (
    KNOWN_POLICY_TYPES,
    POLICY_TYPES_BY_LENGTH as _POLICY_TYPES_BY_LENGTH,
    RUN_REPO_TIMESTAMP_RE,
    dedupe_display_names,
    imported_name_suffixes,
    iso_time_suffixes,
)

logger = logging.getLogger(__name__)

# Cap on the local-run scan. The registry keeps full history; a browser never
# needs thousands of entries, and this bounds the per-run checkpoint stat work.
_LOCAL_MODEL_SCAN_LIMIT = 500

# A repo qualifies as a MakerMods Lab-relevant model if it carries the `lerobot` library
# tag (what push_to_hub stamps) OR its name matches the MakerMods Lab run-repo pattern
# (a "_<timestamp>" suffix). Same union as server.py's _list_author_models.
# The pattern itself now lives in utils/naming.py, which also derives titles from
# it — one definition, so a repo that reads as generated to one reads so to both.
_RUN_REPO_RE = RUN_REPO_TIMESTAMP_RE


def _hub_policy_type(tags: list[str] | None, repo_name: str) -> str | None:
    """Infer a Hub model repo's policy type from data the listing already has —
    zero extra network.

    Two signals, in order:
      * TAGS — lerobot-native pushes stamp the type as a tag
        (lerobot/policies/pretrained.py push_model_to_hub → generate_model_card
        with ``tags={"robotics", "lerobot", model_type}``);
      * NAME PREFIX — MakerMods Lab UI uploads historically don't tag the type (fixed
        by upload_local_model's stamping going forward) but embed it as the
        repo name's prefix (jobs._generate_job_id → ``{policy_type}_{dataset}_
        {ts}``). Matched longest-first so pi0 never shadows pi0_fast.

    Returns None when neither signal matches (the UI simply omits the label)."""
    for tag in tags or []:
        if tag in KNOWN_POLICY_TYPES:
            return tag
    for policy_type in _POLICY_TYPES_BY_LENGTH:
        if repo_name.startswith(policy_type + "_"):
            return policy_type
    return None


# Short-TTL cache of the merged /models listing, mirroring datasets.py's
# _listing_cache. Startup + navigation re-hit this endpoint in quick succession;
# without a cache each load re-fans-out to the (slow/flaky) Hub. A mutation the
# user just performed (upload/delete) invalidates the cache so it reflects
# immediately. TTL is measured with time.monotonic() (app runtime, immune to
# wall-clock jumps).
_LISTING_CACHE_TTL_S = 45.0
_listing_cache_lock = threading.Lock()
_listing_cache: dict[str, Any] | None = None  # {"at": monotonic, "value": [...]}


def invalidate_model_listing_cache() -> None:
    """Drop the cached /models listing so the next call re-fetches. Called after
    any mutation that changes the listing (model upload / delete) so a change the
    user just made shows up immediately instead of after the TTL. Mirrors
    datasets.invalidate_dataset_listing_cache."""
    global _listing_cache
    with _listing_cache_lock:
        _listing_cache = None


class ModelError(Exception):
    """Raised when a model mutation (upload/delete) can't proceed. `status` is
    the HTTP status the route should return (400 offline/invalid, 403 no write
    permission, 404 not found, 409 busy, 502 other Hub failure); `message` is
    the user-facing reason; `docs_url` (optional) links auth docs for a login
    failure."""

    def __init__(self, status: int, message: str, docs_url: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.docs_url = docs_url


# ---------------------------------------------------------------------------
# Local models — the final checkpoint of each completed local training run.
# ---------------------------------------------------------------------------


def _final_checkpoint_dir(record: JobRecord) -> Path | None:
    """The pretrained_model dir of a completed local run's final (highest-step)
    checkpoint, or None if the run saved no checkpoint.

    Reuses jobs._list_local_checkpoints (which validates each
    checkpoints/<step>/pretrained_model/config.json), so a half-saved checkpoint
    dir is never returned. The list is step-sorted; the last entry is the final
    checkpoint. `ref` is the absolute pretrained_model dir."""
    checkpoints = _list_local_checkpoints(record.output_dir)
    if not checkpoints:
        return None
    return Path(checkpoints[-1].ref)


def _read_train_config(pretrained_dir: Path) -> dict[str, Any]:
    """Load a checkpoint's train_config.json (policy type / dataset / steps).

    lerobot writes this alongside config.json inside pretrained_model/. Absent or
    unreadable metadata degrades to an empty dict — the caller renders without
    it rather than dropping the model from the listing."""
    path = pretrained_dir / "train_config.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        logger.info("Could not read %s: %s", path, exc)
        return {}


def _local_model_summary(record: JobRecord, pretrained_dir: Path) -> dict[str, Any]:
    """Build the listing row for one completed local run's final checkpoint.

    policy_type / dataset / steps come from train_config.json where present, and
    fall back to the registry record (config.policy_type, config.dataset_repo_id)
    so a checkpoint missing its train_config still renders a usable row. The
    checkpoint's actual step (its dir name) is the authoritative `steps` value;
    train_config's `steps` is the run's target and is used only as a fallback."""
    train_config = _read_train_config(pretrained_dir)

    policy = train_config.get("policy") or {}
    policy_type = policy.get("type") or record.config.policy_type

    dataset = train_config.get("dataset") or {}
    dataset_repo_id = dataset.get("repo_id") or record.config.dataset_repo_id or None

    # Prefer the checkpoint's real step (its dir name) over the config target.
    try:
        steps: int | None = int(pretrained_dir.parent.name)
    except (ValueError, TypeError):
        raw_steps = train_config.get("steps")
        steps = int(raw_steps) if isinstance(raw_steps, int) else None

    return {
        "id": record.id,
        "name": record.display_name or record.name,
        "policy_type": policy_type,
        "dataset": dataset_repo_id,
        "steps": steps,
        "path": str(pretrained_dir),
        "last_modified": _epoch_iso(record.ended_at or record.started_at),
        "hf_repo_id": record.hf_repo_id or None,
        "source": "local",
    }


def _epoch_iso(ts: float | None) -> str | None:
    from datetime import UTC, datetime

    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(ts, tz=UTC).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def list_local_models() -> list[dict[str, Any]]:
    """Every COMPLETED local training run that produced a final checkpoint.

    Reads the job registry (never mutates it). A run qualifies iff it is a
    `local` runner in state "done" AND has at least one valid on-disk checkpoint
    — running / failed / interrupted runs and checkpoint-less runs are skipped
    (a failed run may have died before its first save). Each row carries the
    policy type, base dataset repo_id, step count, and the local checkpoint path,
    read from the final checkpoint's train_config.json.

    Newest first (by the run's end/start time)."""
    out: list[dict[str, Any]] = []
    for record in job_registry.list(limit=_LOCAL_MODEL_SCAN_LIMIT):
        if record.runner != "local" or record.state != "done":
            continue
        pretrained_dir = _final_checkpoint_dir(record)
        if pretrained_dir is None:
            continue
        out.append(_local_model_summary(record, pretrained_dir))

    out.sort(key=lambda m: m["last_modified"] or "", reverse=True)
    return out


def _find_local_record(model_id: str) -> JobRecord | None:
    """The completed local-run registry record for `model_id`, or None.

    None when the id is unknown, isn't a local run, isn't done, or left no
    checkpoint — the same qualification `list_local_models` applies, so callers
    (info / upload / delete) agree on what a "local model" is."""
    from .jobs import JobNotFoundError

    try:
        record = job_registry.get(model_id)
    except JobNotFoundError:
        return None
    if record.runner != "local" or record.state != "done":
        return None
    if _final_checkpoint_dir(record) is None:
        return None
    return record


# ---------------------------------------------------------------------------
# Downloaded / imported local models — checkpoints in the local models dir.
# ---------------------------------------------------------------------------
#
# The second kind of "local" model, alongside completed training runs: a
# checkpoint COPIED onto this machine, either downloaded from the Hub
# (model_download_manager) or imported from a folder on disk
# (import_local_model). They live under a stable local models dir so the
# /models listing can flip a Hub repo to "both" and inference can run offline —
# the models mirror of the datasets' FLAT download layout.


def _local_models_root() -> Path:
    """The stable directory downloaded/imported model checkpoints land in:
    ``<lerobot_home>/makermodslab_models/<repo_id>/``. Kept under the same home as the
    datasets' flat layout (which uses the home root directly — models get a
    dedicated subdir so checkpoint dirs are never confused with dataset dirs).

    Existing installs used ``lelab_models``. Move that directory on first use
    so locally downloaded checkpoints remain available after the package
    rebrand. If the move is not possible (for example, a read-only cache), keep
    using the legacy directory rather than forcing a new Hub download.
    """
    cache_root = _lerobot_cache_root()
    root = cache_root / "makermodslab_models"
    legacy_root = cache_root / "lelab_models"
    if not root.exists() and legacy_root.exists():
        try:
            legacy_root.rename(root)
        except OSError as exc:
            logger.warning("Could not migrate local model cache from %s to %s: %s", legacy_root, root, exc)
            return legacy_root
    return root


def _resolve_pretrained_dir(path: Path) -> Path | None:
    """The pretrained_model dir inside a downloaded/imported checkpoint dir, or
    None when `path` isn't a usable policy checkpoint.

    Recognizes the same two layouts jobs._list_imported_local does (and in the
    same order): a ``checkpoints/<step>/pretrained_model`` tree (the final,
    highest-step checkpoint wins), else the dir itself when it has a root
    ``config.json`` (the shape ``upload_local_model`` pushes). The returned dir
    is EXACTLY what rollout._resolve_policy_path consumes verbatim for a local
    ref, so `path` rows built from it are directly usable for inference.
    """
    try:
        checkpoints = _list_local_checkpoints(str(path))
        if checkpoints:
            return Path(checkpoints[-1].ref)
        if (path / "config.json").is_file():
            return path
    except OSError:
        return None
    return None


def _downloaded_model_dir(repo_id: str) -> Path | None:
    """The models-root dir for `repo_id` IF it holds a usable checkpoint, else
    None. Traversal-guarded: a repo_id that resolves outside the models root
    (e.g. '../evil') is refused."""
    root = _local_models_root().resolve()
    try:
        path = (root / repo_id).resolve()
    except OSError:
        return None
    if path == root or root not in path.parents:
        return None
    if not path.is_dir() or _resolve_pretrained_dir(path) is None:
        return None
    return path


def is_model_available_locally(repo_id: str) -> bool:
    """True when `repo_id` has a usable checkpoint in the local models dir, so
    inference could run on it without a Hub download. Filesystem-only (no
    network) — safe to call offline. The datasets mirror is
    is_dataset_available_locally."""
    return _downloaded_model_dir(repo_id) is not None


def _downloaded_model_summary(repo_id: str, model_dir: Path) -> dict[str, Any]:
    """Listing row for one downloaded/imported model checkpoint.

    Shaped like a local-run row (id/name = repo_id since there's no run record):
    policy type from the checkpoint's config.json, dataset/steps from its
    train_config.json where present (upload_folder pushes the whole
    pretrained_model dir, so downloaded MakerMods Lab-made models carry it; a tree
    checkpoint's step dir is authoritative for steps). `path` is the resolved
    pretrained dir — directly consumable by rollout._resolve_policy_path.
    hf_repo_id stays None here: the scan can't know whether the dir came from
    the Hub or a disk import; the listing merge / pin fold fills it in when the
    repo is confirmed on the Hub."""
    pretrained = _resolve_pretrained_dir(model_dir)
    assert pretrained is not None  # callers guarantee it (dir already probed)

    policy_type = None
    try:
        policy_type = json.loads((pretrained / "config.json").read_text()).get("type")
    except (OSError, ValueError) as exc:
        logger.info("Could not read %s: %s", pretrained / "config.json", exc)

    train_config = _read_train_config(pretrained)
    dataset = (train_config.get("dataset") or {}).get("repo_id")

    steps: int | None = None
    if pretrained != model_dir:
        # Tree layout: checkpoints/<step>/pretrained_model — the dir name is the
        # actual step.
        try:
            steps = int(pretrained.parent.name)
        except (TypeError, ValueError):
            steps = None
    if steps is None:
        raw_steps = train_config.get("steps")
        steps = int(raw_steps) if isinstance(raw_steps, int) else None

    return {
        "id": repo_id,
        "name": repo_id,
        "policy_type": policy_type,
        "dataset": dataset,
        "steps": steps,
        "path": str(pretrained),
        "last_modified": _dir_mtime_iso(model_dir),
        "hf_repo_id": None,
        "source": "local",
    }


def list_downloaded_models() -> list[dict[str, Any]]:
    """Scan the local models dir for downloaded/imported checkpoints.

    Walks one level deep, mirroring datasets.list_local_datasets: a top-level
    dir that is itself a checkpoint is recorded as "<name>"; otherwise each
    checkpoint subdir is recorded as "<top>/<sub>" (namespace layout). Unusable
    dirs (no config.json / checkpoints tree — e.g. a foreign download) are
    skipped."""
    root = _local_models_root()
    if not root.is_dir():
        return []

    try:
        top_entries = list(root.iterdir())
    except OSError as exc:
        logger.warning("Could not read local models root %s: %s", root, exc)
        return []

    out: list[dict[str, Any]] = []
    for top in top_entries:
        try:
            if not top.is_dir():
                continue
        except OSError:
            continue

        if _resolve_pretrained_dir(top) is not None:
            out.append(_downloaded_model_summary(top.name, top))
            continue

        # Not a checkpoint itself — treat as a namespace dir, descend one level.
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
            if _resolve_pretrained_dir(sub) is not None:
                out.append(_downloaded_model_summary(f"{top.name}/{sub.name}", sub))

    out.sort(key=lambda m: m["last_modified"] or "", reverse=True)
    return out


# ---------------------------------------------------------------------------
# Hub models — the user's LeRobot policy repos on the Hub.
# ---------------------------------------------------------------------------


def _list_author_models(api, author: str) -> list[dict[str, Any]]:
    """All MakerMods Lab-relevant model repos for one author, materialized.

    ONE unfiltered `list_models(author=...)` call, filtered client-side to the
    union of the `lerobot` library tag and the MakerMods Lab run-repo naming — the same
    set (and the same single-call shape) as server.py's `/jobs/hub` listing. The
    generator is materialized HERE, inside the fan-out worker, so the network I/O
    (and any GFW-killed connection) happens under the per-author timeout budget
    rather than lazily later while we iterate."""
    rows: list[dict[str, Any]] = []
    for m in api.list_models(author=author, limit=200, expand=["lastModified", "private", "tags"]):
        tags = getattr(m, "tags", None) or []
        name = m.id.split("/", 1)[-1]
        if "lerobot" in tags or _RUN_REPO_RE.search(name):
            rows.append(
                {
                    "repo_id": m.id,
                    "last_modified": m.last_modified.isoformat() if m.last_modified else None,
                    "private": bool(getattr(m, "private", False)),
                    # The expand already fetched the tags — recognize the policy
                    # type from them (or the name prefix) for free, instead of
                    # discarding them.
                    "policy_type": _hub_policy_type(tags, name),
                }
            )
    return rows


def list_hub_models() -> list[dict[str, Any]]:
    """The user's (and their orgs') LeRobot policy repos on the Hub.

    Mirrors datasets.list_user_datasets: fan out per author concurrently via the
    shared `_fan_out_hub_authors` helper, so one blocked/slow/GFW-killed author
    degrades to "the others' results" instead of a 500. Empty when no token is
    configured. Deduped by repo_id, newest first."""
    info = cached_whoami()
    if info is None:
        return []

    authors = [info["name"]] + [o["name"] for o in info.get("orgs", [])]
    api = shared_hf_api()

    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for rows in _fan_out_hub_authors(authors, lambda author: _list_author_models(api, author)):
        for row in rows:
            if row["repo_id"] in seen:
                continue
            seen.add(row["repo_id"])
            out.append(row)

    out.sort(key=lambda m: m["last_modified"] or "", reverse=True)
    return out


# ---------------------------------------------------------------------------
# Merged listing.
# ---------------------------------------------------------------------------


def _job_run_steps(record: JobRecord) -> int | None:
    """How many steps the run behind a Hub repo actually trained.

    A run that reached "done" trained to its configured target, and
    `config.steps` is that target in GLOBAL steps even for a resumed run —
    prefer it, because the metrics tail can stop short of the end (the last log
    lines are parsed at log_freq granularity, and a terminal transition can drop
    the final one). Otherwise the run stopped early, so the last step its metrics
    observed is the honest number. None when neither is known: the row keeps the
    listing's null rather than invent one.
    """
    if record.state == "done" and record.config.steps:
        return record.config.steps
    if record.metrics.current_step:
        return record.metrics.current_step
    return None


def _job_outranks(candidate: JobRecord, incumbent: JobRecord) -> bool:
    """Ranking used to pick which run identifies a shared output repo: a run
    that reached "done" beats one that didn't (it is the one that published the
    final policy at the repo root), and among equals the later-started run
    wins (the deepest link of a resume chain)."""
    candidate_done = candidate.state == "done"
    incumbent_done = incumbent.state == "done"
    if candidate_done != incumbent_done:
        return candidate_done
    return candidate.started_at > incumbent.started_at


def _jobs_by_hub_repo() -> dict[str, JobRecord]:
    """Tracked runs keyed by the Hub repo they publish to, one run per repo.

    A cloud resume REUSES its parent's output repo (hf_cloud sets
    `policy_repo_id = resume_from_hub_repo`), so an N-deep resume chain shares a
    single repo named after run #1 — and the Hub listing, which keys on repo_id,
    then has no row under the name of the run that actually finished. Collapse
    each repo to the run that best identifies it (see _job_outranks).

    IMPORTED records are skipped: an import REGISTERS an existing repo rather
    than producing it, so its config is a placeholder (dataset_repo_id
    "(imported)", the default step count) and its auto-generated name says less
    than the repo id does. They are also always "done" and usually the newest
    record for their repo, so ranking them would let a placeholder outrank the
    run that actually trained the weights.

    Reads the registry, never mutates it. The scan limit matches
    list_local_models so the two passes share the registry's per-record
    checkpoint-count work (and, for cloud records, its 30s Hub cache) instead of
    doubling it."""
    best: dict[str, JobRecord] = {}
    for record in job_registry.list(limit=_LOCAL_MODEL_SCAN_LIMIT):
        repo_id = record.hf_repo_id
        if not repo_id or record.runner == "imported":
            continue
        incumbent = best.get(repo_id)
        if incumbent is None or _job_outranks(record, incumbent):
            best[repo_id] = record
    return best


def _run_identity_name(record: JobRecord) -> str:
    """A run's human name peeled down to the TASK it learned.

    jobs.start auto-generates an unnamed run's name as
    "{POLICY} · {dataset_repo_id}" — "SMOLVLA · makermods/eraser_place". Both
    halves of that are stated elsewhere on every surface that renders it: the
    row carries `policy_type` in its own field and `dataset` (the full repo id,
    namespace included) in another, so the title line — the widest and the first
    to truncate — spends its pixels on facts printed twice. Peel both:

      "SMOLVLA · makermods/eraser_place"  ->  "eraser_place"

    which converges model-card naming on one philosophy: the task identity,
    policy-free and namespace-free. Imported titles already read this way
    (utils.naming.derive_imported_title), and the retired "Imported · " prefix
    (jobs._LEGACY_IMPORT_NAME_PREFIX) went for the same reason.

    Only the GENERATED shape is peeled, and only when the head is a policy type
    we recognize — a user-typed job_name that happens to contain " · " keeps
    every word, and so does a name whose head isn't a policy. The namespace peel
    is gated behind that same check, so it can never eat a slash out of a name a
    human wrote. A rename (`display_name`) never reaches here; the caller
    prefers it outright.

    Two datasets that share a task name across namespaces (a/pick and b/pick)
    now render one label twice — that is what _dedupe_listing_names exists for,
    and its date suffix separates them.
    """
    head, separator, tail = record.name.partition(" · ")
    if not (separator and tail and head.lower() in KNOWN_POLICY_TYPES):
        return record.name
    # The tail is a dataset repo id: keep the segment after the namespace. A
    # bare name with no slash is already the task.
    _namespace, slash, task = tail.partition("/")
    return task if slash and task else tail


def _enrich_repo_rows_from_jobs(merged: dict[str, dict[str, Any]]) -> None:
    """Give each repo-keyed row the identity of the run that produced it.

    Applies only to rows still named after their repo id — i.e. rows no local
    checkpoint has claimed (a "both" collapse from a local run already put the
    run's own name and detail there, and local always wins). That covers the
    hub-only rows, hub rows flipped to "both" by a downloaded copy, and pinned
    rows the Hub listing didn't return.

    Identity fields are NOT touched: `id`, `repo_id` and `hf_repo_id` route
    every request (info / download / pin / hide / deploy) and a resume chain's
    repo keeps the name of run #1 on the Hub. Only the human-facing `name` is
    replaced, plus the detail fields the repo listing could not know — filled in
    only where they are still null, so a checkpoint-derived value always wins.
    """
    by_repo = _jobs_by_hub_repo()
    if not by_repo:
        return
    for row in merged.values():
        repo_id = row.get("hf_repo_id")
        if not repo_id or row.get("name") != repo_id:
            continue
        record = by_repo.get(repo_id)
        if record is None:
            continue
        row["name"] = record.display_name or _run_identity_name(record)
        if row.get("policy_type") is None:
            row["policy_type"] = record.config.policy_type or None
        if row.get("dataset") is None:
            row["dataset"] = record.config.dataset_repo_id or None
        if row.get("steps") is None:
            row["steps"] = _job_run_steps(record)


def _row_dedupe_suffixes(row: dict[str, Any]) -> list[str]:
    """The disambiguator ladder for one listing row, least → most specific.

    Prefers the run timestamp embedded in the row's own NAME — every repo a
    MakerMods Lab run publishes to carries the `_<date>_<time>` tail its run id
    was generated with — over the Hub's `last_modified`. The two usually agree,
    and where they disagree the name is the one the user means: last_modified
    moves whenever anything is pushed to the repo (a re-push of the same
    weights, a README edit, a later checkpoint upload), so a row could acquire a
    date months after the run it labels and read as simply wrong next to its
    sibling. The name's stamp is when the run STARTED and never moves.

    Falls back to last_modified for a row whose name carries no stamp — a
    community repo, a hand-named upload — which is the only signal there is.
    Both ladders come from utils.naming, so a name-derived suffix and a
    time-derived one are formatted identically and can disambiguate each other.
    """
    source = str(row.get("hf_repo_id") or row.get("id") or "")
    if source and _RUN_REPO_RE.search(source.rsplit("/", 1)[-1]):
        return imported_name_suffixes(source)
    return iso_time_suffixes(row.get("last_modified"))


def _dedupe_listing_names(merged: dict[str, dict[str, Any]]) -> None:
    """Make the listing's display names unique, in place.

    Run names are auto-generated from policy + dataset (jobs.start →
    "SMOLVLA · ns/task", which the enrichment peels down to "task" — see
    _run_identity_name), so retraining one task — a longer run, a different
    seed — produces a SECOND row with a byte-identical name. Naming rows after
    their run (see _enrich_repo_rows_from_jobs) makes that collision visible in
    the picker, where two rows then read as duplicates of each other with
    nothing on either to say which is which.

    Disambiguate least-specific first (see _row_dedupe_suffixes for where the
    date comes from, and dedupe_display_names, which escalates to date + time
    and falls back to an ordinal if even that ties). Runs over EVERY row, not
    just the enriched ones — two local runs of one policy on one dataset collide
    the same way and never went near the enrichment.

    Rows are grouped by (name, policy_type), so an ACT and a SmolVLA of one task
    are NOT suffixed: the Policy row on the card already separates them, and a
    date there would imply the difference between the two is when they ran.
    Identity fields are left alone, on the same rule as the enrichment: only
    `name` is display.
    """
    rows = list(merged.values())
    resolved = dedupe_display_names(
        [(str(row.get("name") or ""), _row_dedupe_suffixes(row)) for row in rows],
        group_keys=[row.get("policy_type") for row in rows],
    )
    for row, name in zip(rows, resolved, strict=True):
        row["name"] = name


def list_all_models() -> list[dict[str, Any]]:
    """Merged listing: local completed runs + Hub policy repos, with a `source`.

    A local run that was also pushed to the Hub (its `hf_repo_id` matches a Hub
    repo) collapses to ONE entry with source="both"; a Hub-only repo is "hub"; a
    local-only run is "local". Mirrors list_all_datasets's merge/dedup/sort and
    its resilience: the Hub half is best-effort (list_hub_models never raises),
    so a Hub outage degrades to local-only rather than crashing.

    A row still named after its repo id is then named after the tracked run that
    produced it (see _enrich_repo_rows_from_jobs) — a repo id names a run only
    by accident, and never names the right one for a cloud resume chain, which
    publishes every link into the FIRST run's repo.

    Cached for up to _LISTING_CACHE_TTL_S so repeated startup/nav loads reuse a
    recent listing; a mutation invalidates the cache (see
    invalidate_model_listing_cache) so it reflects immediately."""
    global _listing_cache

    now = time.monotonic()
    with _listing_cache_lock:
        if _listing_cache is not None and (now - _listing_cache["at"]) < _LISTING_CACHE_TTL_S:
            return _listing_cache["value"]

    local = list_local_models()
    hub = list_hub_models()

    merged: dict[str, dict[str, Any]] = {}
    # Seed with hub repos, keyed by repo_id. `list_hub_models` returns the
    # /jobs/hub row shape (repo_id / last_modified / private), so map each into
    # the full ModelItem shape the frontend expects — a hub-only model has no
    # local checkpoint, so its `id`/`name`/`hf_repo_id` are all the repo id and
    # the checkpoint-derived detail fields (policy_type/dataset/steps/path) are
    # null. Without this mapping hub rows reach the UI with no id/name and crash
    # the picker.
    for item in hub:
        repo_id = item["repo_id"]
        merged[repo_id] = {
            "id": repo_id,
            "name": repo_id,
            # Kept alongside hf_repo_id for symmetry with the datasets listing
            # rows (which key on repo_id) — hub-derived rows carry both.
            "repo_id": repo_id,
            # Recognized from the repo's tags / name by the hub listing (see
            # _hub_policy_type) — no extra network. A local checkpoint's
            # config.json value still wins on a "both" collapse below.
            "policy_type": item.get("policy_type"),
            "dataset": None,
            "steps": None,
            "path": None,
            "last_modified": item.get("last_modified"),
            "hf_repo_id": repo_id,
            # The hub listing already fetches the private flag — surface it so
            # the picker can badge private repos (mirrors DatasetItem.private).
            "private": bool(item.get("private", False)),
            "source": "hub",
        }

    # Local rows key on their run id, EXCEPT when the run's hub_repo_id matches a
    # hub repo already present — then it's the same model on both, collapsed to
    # "both" under the hub repo_id (so the row isn't duplicated).
    for item in local:
        hub_repo = item.get("hf_repo_id")
        if hub_repo and hub_repo in merged:
            existing = merged[hub_repo]
            existing["source"] = "both"
            # The local checkpoint is authoritative for detail: override the
            # hub row's placeholder id/name and fill in the checkpoint fields it
            # couldn't know (per the contract, `id`/`name` follow the local run
            # for a "both" model).
            for key in ("id", "name", "policy_type", "dataset", "steps", "path"):
                if item.get(key) is not None:
                    existing[key] = item[key]
            a = existing.get("last_modified") or ""
            b = item.get("last_modified") or ""
            existing["last_modified"] = max(a, b) or None
        else:
            merged[item["id"]] = {**item, "source": "local"}

    # Downloaded/imported checkpoints (the local models dir), keyed by repo_id.
    # One that matches a hub row flips it to "both" and fills in the checkpoint
    # detail the hub row couldn't know (path/policy_type/steps) — that's the
    # "download to local" listing flip. A checkpoint the Hub doesn't list (a
    # disk import, or a foreign repo when offline) stays a "local" row.
    for item in list_downloaded_models():
        rid = item["id"]
        if rid in merged:
            existing = merged[rid]
            existing["source"] = "both"
            # The on-disk checkpoint is authoritative for detail: its
            # config.json-derived values override the hub row's tag/name-derived
            # ones (same local-wins rule as the run collapse above).
            for key in ("policy_type", "dataset", "steps", "path"):
                if item.get(key) is not None:
                    existing[key] = item[key]
            a = existing.get("last_modified") or ""
            b = item.get("last_modified") or ""
            existing["last_modified"] = max(a, b) or None
        else:
            merged[rid] = item

    # Fold in the user's pinned custom Hub models (the "Add model" chooser's
    # add-from-HF flow), mirroring list_all_datasets' pin fold. A pin already
    # covered by a hub/both row is redundant and skipped. A pinned repo whose
    # checkpoint was downloaded (the scan added a "local" row) is a Hub model
    # with a local copy: flip to "both", stamp the hf_repo_id the scan couldn't
    # know, and keep saved_custom so "remove from list" (unpin) stays available.
    for repo_id in get_saved_custom_models():
        existing = merged.get(repo_id)
        if existing is None:
            merged[repo_id] = {
                "id": repo_id,
                "name": repo_id,
                # A pinned repo the Hub listing didn't return (private, GFW-dropped,
                # or untagged) still has an inferable type from its name prefix
                # (act_… / smolvla_…) — infer it here instead of dropping to None,
                # so the picker shows the policy label. Mirrors the tag/name
                # inference the hub listing does via _list_author_models.
                "policy_type": _hub_policy_type(None, repo_id.split("/", 1)[-1]),
                "dataset": None,
                "steps": None,
                "path": None,
                "last_modified": None,
                "hf_repo_id": repo_id,
                "private": False,
                "source": "hub",
                "saved_custom": True,
            }
        elif existing["source"] == "local":
            existing["source"] = "both"
            existing["hf_repo_id"] = repo_id
            existing["saved_custom"] = True

    # Name the repo-keyed rows after the run that produced them, now that every
    # source has merged in. A tracked run whose output repo is shared with its
    # resume chain (or simply never collapsed into a local row) would otherwise
    # reach the picker as a bare repo id with null steps/dataset — under the
    # name of the FIRST run in the chain, never the one that finished. Runs
    # after the merge and before the hidden filter so it can't resurrect a
    # hidden row, and so local-checkpoint detail always wins (see the helper).
    _enrich_repo_rows_from_jobs(merged)

    # Then separate the rows the naming above left indistinguishable: two runs
    # of one policy on one dataset carry the same auto-generated name, so the
    # picker would show the same label twice. Immediately after the enrichment
    # so it sees the final names, and still before the hidden filter.
    _dedupe_listing_names(merged)

    # Hidden models ("removed from list") are filtered LAST — after the
    # hub/local/downloaded merge and the pin fold — so a hidden id can't
    # resurface via a pin or a local copy. Re-pinning auto-unhides (see
    # /models/custom). Mirrors list_all_datasets.
    hidden = get_hidden_models()
    if hidden:
        merged = {rid: row for rid, row in merged.items() if rid not in hidden}

    out = list(merged.values())
    out.sort(key=lambda m: m.get("last_modified") or "", reverse=True)

    with _listing_cache_lock:
        _listing_cache = {"at": time.monotonic(), "value": out}
    return out


# ---------------------------------------------------------------------------
# Per-model info card.
# ---------------------------------------------------------------------------


def _dir_size_bytes(path: Path) -> int:
    """Total size of all files under `path`. Unreadable files are skipped."""
    import os

    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            try:
                total += os.stat(os.path.join(dirpath, name)).st_size
            except OSError:
                continue
    return total


def get_model_info(id_or_repo: str) -> dict[str, Any] | None:
    """Detail view of one model: policy type, dataset, steps, size, path/repo.

    Resolves `id_or_repo` as a local run id first (a completed local run with a
    checkpoint); if that misses, treats it as a Hub repo id and reads the
    checkpoint config from the Hub. Returns None when neither resolves.

    Local: reads train_config.json + config.json from the final checkpoint and
    walks the dir for its on-disk size. Hub: reads the root config.json via the
    Hub (the network call is the caller's — the info card fetches it lazily) and
    reports no size (the repo isn't on disk)."""
    record = _find_local_record(id_or_repo)
    if record is not None:
        pretrained_dir = _final_checkpoint_dir(record)
        assert pretrained_dir is not None  # _find_local_record guarantees it
        summary = _local_model_summary(record, pretrained_dir)
        summary["size_bytes"] = _dir_size_bytes(pretrained_dir)
        return summary

    # A downloaded/imported checkpoint in the local models dir — filesystem
    # only, so it works offline (that's the point of downloading a model).
    model_dir = _downloaded_model_dir(id_or_repo)
    if model_dir is not None:
        summary = _downloaded_model_summary(id_or_repo, model_dir)
        summary["size_bytes"] = _dir_size_bytes(model_dir)
        return summary

    # Not local at all — try the Hub. Offline ⇒ can't read a hub-only model.
    if hf_hub_offline():
        return None
    return _hub_model_info(id_or_repo)


# In-process cache of per-repo Hub model metadata (the /models/info hub
# branch), mirroring datasets._HUB_STATUS_CACHE conventions: successful answers
# are memoized for the process lifetime; the offline/error degrade is NEVER
# cached, so connectivity returning is picked up on the next check. Invalidated
# alongside the listing cache on the mutations that change a repo (upload,
# download-complete, delete, hide) — see invalidate_model_hub_info.
_MODEL_HUB_INFO_CACHE: dict[str, dict[str, Any]] = {}
_MODEL_HUB_INFO_LOCK = threading.Lock()


def invalidate_model_hub_info(repo_id: str) -> None:
    """Drop the cached Hub metadata for `repo_id`, so the next /models/info
    re-reads it (e.g. after an upload changed the repo's tags/size)."""
    with _MODEL_HUB_INFO_LOCK:
        _MODEL_HUB_INFO_CACHE.pop(repo_id, None)


def _hub_model_probe(repo_id: str) -> dict[str, Any] | None:
    """The pre-existing file-tree probe for a Hub model repo: reads the
    checkpoint config from the Hub (root config.json, or the latest
    checkpoints/<step>/pretrained_model tree) for the policy type + step. Two
    Hub calls (list_repo_files + hf_hub_download), so it's the FALLBACK when the
    single model_info call fails or leaves the policy type unknown. Returns None
    if the repo has no usable model config (or can't be read)."""
    from .jobs import _list_imported_hub

    api = shared_hf_api()
    checkpoints = _list_imported_hub(api, repo_id)
    if not checkpoints:
        return None
    final = checkpoints[-1]
    policy_type = None
    steps: int | None = final.step or None
    try:
        cfg = _read_checkpoint_config(final)
        policy_type = cfg.get("type")
    except Exception as exc:
        logger.info("Could not read hub model config for %s: %s", repo_id, exc)

    return {
        "id": repo_id,
        "name": repo_id.split("/", 1)[-1],
        "policy_type": policy_type,
        "dataset": None,
        "steps": steps,
        "path": None,
        "hf_repo_id": repo_id,
        "size_bytes": None,
        "source": "hub",
    }


def _hub_model_info(repo_id: str) -> dict[str, Any] | None:
    """Detail for a Hub-only model repo, built around ONE
    ``HfApi.model_info(expand=...)`` call: policy type (card model_name → tags →
    name prefix, all against KNOWN_POLICY_TYPES), base dataset (card
    ``datasets``), size on the Hub (``usedStorage``), private flag, and last
    modified. Only when the cheap signals leave the policy type unknown does it
    fall back to the old two-call file-tree probe (_hub_model_probe), which also
    recovers the checkpoint step.

    Degrade-not-crash: if model_info itself fails this falls back to the probe
    (itself best-effort) rather than raising, and only successful answers are
    cached (see _MODEL_HUB_INFO_CACHE) so transient failures self-heal."""
    with _MODEL_HUB_INFO_LOCK:
        cached = _MODEL_HUB_INFO_CACHE.get(repo_id)
    if cached is not None:
        return dict(cached)

    api = shared_hf_api()
    try:
        info = api.model_info(repo_id, expand=["cardData", "tags", "lastModified", "private", "usedStorage"])
    except Exception as exc:
        # Transient / not-found / auth — degrade to the old probe, uncached.
        logger.info("model_info(%s) failed: %s", repo_id, exc)
        return _hub_model_probe(repo_id)

    name = repo_id.split("/", 1)[-1]
    tags = list(getattr(info, "tags", None) or [])
    card = getattr(info, "card_data", None)

    # lerobot's generate_model_card sets model_name to the policy type — the
    # most explicit signal; then the tag / name-prefix inference.
    policy_type = None
    card_model_name = getattr(card, "model_name", None) if card is not None else None
    if card_model_name in KNOWN_POLICY_TYPES:
        policy_type = card_model_name
    if policy_type is None:
        policy_type = _hub_policy_type(tags, name)

    # Base dataset(s) from the card metadata; a list takes its first entry.
    dataset = getattr(card, "datasets", None) if card is not None else None
    if isinstance(dataset, list):
        dataset = dataset[0] if dataset else None
    if not isinstance(dataset, str):
        dataset = None

    last_modified = getattr(info, "last_modified", None)
    used_storage = getattr(info, "used_storage", None)

    row: dict[str, Any] = {
        "id": repo_id,
        "name": name,
        "policy_type": policy_type,
        "dataset": dataset,
        "steps": None,
        "path": None,
        "last_modified": last_modified.isoformat() if last_modified else None,
        "hf_repo_id": repo_id,
        "private": bool(getattr(info, "private", False)),
        "size_bytes": int(used_storage) if isinstance(used_storage, int) else None,
        "source": "hub",
    }

    if row["policy_type"] is None:
        # Last resort: the old config.json download path (also recovers steps).
        probe = _hub_model_probe(repo_id)
        if probe is not None:
            row["policy_type"] = probe["policy_type"]
            row["steps"] = probe["steps"]

    with _MODEL_HUB_INFO_LOCK:
        _MODEL_HUB_INFO_CACHE[repo_id] = dict(row)
    return row


# ---------------------------------------------------------------------------
# Upload a local checkpoint to the Hub as a model repo.
# ---------------------------------------------------------------------------


def _upload_model_error(exc: Exception) -> ModelError:
    """Map a huggingface_hub exception from create_repo/upload_folder to a
    ModelError with a legible message. A 401/auth failure or a 403 permission
    failure becomes a clear "you can't push this" message; anything else
    degrades to a generic Hub-failure 502. Reuses record._upload_auth_error so
    the 401 maps identically to the dataset upload path."""
    from .record import _upload_auth_error

    auth = _upload_auth_error(exc)
    if auth is not None:
        return ModelError(403, auth["message"], docs_url=auth.get("docs_url"))

    err_text = str(exc).lower()
    if "403" in err_text or "forbidden" in err_text or "permission" in err_text:
        return ModelError(
            403,
            "You don't have permission to push this model to the Hub. You can "
            "only upload models to a namespace you can write to.",
        )
    return ModelError(502, f"The Hub rejected the upload: {exc}")


def _default_model_repo_id(record: JobRecord) -> str:
    """The Hub repo id a local run uploads to when the caller names none.

    Namespaced to the logged-in user (models can't push to a bare name), using
    the run id as the repo name so it's stable and legible. cached_whoami is the
    same auth source the listing uses."""
    info = cached_whoami()
    namespace = info["name"] if info else None
    return f"{namespace}/{record.id}" if namespace else record.id


def upload_local_model(model_id: str, repo_id: str | None = None) -> dict[str, Any]:
    """Push a completed local run's final checkpoint to the Hub as a MODEL repo.

    PUBLIC by default and tagged makermods / openbooth / MakerModsLab via with_makermodslab_tag
    (the same funnel datasets/policies use). Steps:
      1. resolve the local run + its final pretrained_model dir (404 if missing);
      2. create_repo(repo_id, repo_type="model", private=False, exist_ok=True);
      3. upload_folder(folder_path=<checkpoint>, repo_id, repo_type="model");
      4. metadata_update(repo_id, {"tags": with_makermodslab_tag(None)}, repo_type=
         "model", overwrite=True).

    Refuses offline (can't mutate the Hub) with a clear error. Auth/permission
    failures map like the dataset upload path. Invalidates the model-listing
    cache so the freshly-pushed repo appears immediately. Returns
    {repo_id, url, tags}.

    NOTE: this runs synchronously (a small policy checkpoint, unlike a
    multi-GB dataset) — the route calls it inline."""
    if hf_hub_offline():
        raise ModelError(400, "The Hub is offline — you can't upload a model right now.")

    record = _find_local_record(model_id)
    if record is None:
        raise ModelError(
            404,
            f"No completed local model with a saved checkpoint found for {model_id!r}.",
        )
    pretrained_dir = _final_checkpoint_dir(record)
    assert pretrained_dir is not None  # _find_local_record guarantees it

    target_repo_id = repo_id or record.hf_repo_id or _default_model_repo_id(record)
    api = shared_hf_api()
    try:
        api.create_repo(target_repo_id, repo_type="model", private=False, exist_ok=True)
        api.upload_folder(
            folder_path=str(pretrained_dir),
            repo_id=target_repo_id,
            repo_type="model",
        )
    except Exception as exc:
        logger.info("Upload of local model %s -> %s failed: %s", model_id, target_repo_id, exc)
        raise _upload_model_error(exc) from exc

    # Stamp the policy type as a tag alongside the org tags, so future MakerMods Lab
    # uploads are self-describing on the Hub (the same tag lerobot-native
    # pushes stamp; _hub_policy_type's name-prefix fallback covers old repos).
    policy_tag = None
    try:
        policy_tag = json.loads((pretrained_dir / "config.json").read_text()).get("type")
    except (OSError, ValueError) as exc:
        logger.info("Could not read %s for tag stamping: %s", pretrained_dir / "config.json", exc)
    final_tags = with_makermodslab_tag([policy_tag] if policy_tag in KNOWN_POLICY_TYPES else None)
    try:
        metadata_update(target_repo_id, {"tags": final_tags}, repo_type="model", overwrite=True)
    except Exception as exc:
        # The weights are already pushed; a tag write failing is non-fatal to the
        # upload, but surface it so the user knows the tags didn't land.
        logger.info("Tagging model %s failed after upload: %s", target_repo_id, exc)
        raise _upload_model_error(exc) from exc

    invalidate_model_listing_cache()
    # The repo's tags/size just changed — drop its cached hub metadata too.
    invalidate_model_hub_info(target_repo_id)
    logger.info("Uploaded local model %s to %s (tags=%s)", model_id, target_repo_id, final_tags)
    return {
        "repo_id": target_repo_id,
        "url": f"https://huggingface.co/{target_repo_id}",
        "tags": final_tags,
    }


# ---------------------------------------------------------------------------
# Delete a local model (its training-run output dir).
# ---------------------------------------------------------------------------


def _model_in_use(target_dir: Path) -> str | None:
    """If a running inference is reading a checkpoint under `target_dir`,
    return a legible reason to refuse deleting it; else None. Shaped like
    datasets._dataset_in_use (a false "in use" is safer than yanking a
    directory a live subprocess is reading).

    Containment on RESOLVED paths: the inference path (the pretrained_model
    dir rollout captured at start) equals the target, or lives anywhere under
    it — a run/downloaded dir contains its checkpoints/<step>/pretrained_model.

    Lazy import: rollout pulls heavy lerobot modules at import time, and the
    models browser must stay cheap to import (no cycle — rollout never imports
    models)."""
    from . import rollout as _rollout

    in_use = _rollout.inference_in_use_path()
    if not in_use:
        return None
    try:
        active = Path(in_use).resolve()
        target = target_dir.resolve()
    except OSError:
        return None
    if active == target or target in active.parents:
        return "This model is being used by a running inference. Stop it first."
    return None


def delete_local_model(model_id: str) -> dict[str, Any]:
    """Delete a local model's LOCAL files — a training run's output dir, or a
    downloaded/imported checkpoint's dir in the local models dir.

    SAFETY: a run dir is resolved strictly under the job registry's
    outputs/train/ root, a downloaded/imported dir strictly under the local
    models root (both traversal-guarded); anything escaping is refused. Never
    touches the Hub — only local files, so deleting the local copy of a "both"
    model leaves the Hub row listed.

    Training runs coordinate with the registry: a RUNNING job's dir is never
    deleted (reuses JobRegistry.delete, which raises for a running job), and
    deleting through the registry also drops the record + persisted job.json.
    Ids that aren't registry records fall through to the downloaded/imported
    probe (_downloaded_model_dir). Invalidates the model-listing cache.

    Returns {deleted: True, id}. Raises ModelError (404 unknown, 409 running,
    400 unsafe path, 502 delete failure)."""
    from .jobs import JobNotFoundError, JobNotRunningError

    try:
        record = job_registry.get(model_id)
    except JobNotFoundError as exc:
        # Not a training run — a downloaded/imported checkpoint in the local
        # models dir? _downloaded_model_dir is traversal-guarded, so anything it
        # returns is strictly inside the models root and safe to remove.
        model_dir = _downloaded_model_dir(model_id)
        if model_dir is None:
            raise ModelError(404, f"Model {model_id!r} not found.") from exc
        in_use = _model_in_use(model_dir)
        if in_use is not None:
            raise ModelError(409, in_use) from None
        try:
            shutil.rmtree(model_dir)
        except OSError as rm_exc:
            logger.error("Failed to delete downloaded model %s: %s", model_id, rm_exc)
            raise ModelError(502, f"Failed to delete model: {rm_exc}") from rm_exc
        invalidate_model_listing_cache()
        # The local copy is gone; the card falls back to the hub view — drop the
        # cached hub metadata so it re-reads fresh.
        invalidate_model_hub_info(model_id)
        logger.info("Deleted downloaded/imported model %s (dir %s)", model_id, model_dir)
        return {"deleted": True, "id": model_id}

    if record.runner != "local":
        raise ModelError(
            400,
            f"Model {model_id!r} is not a local training run, so there's nothing local to delete.",
        )

    # Resolve the run dir and refuse anything outside outputs/train/. The
    # registry owns the (resolved) root; the run dir is <root>/<id>. Guarding on
    # the resolved paths catches a traversal id or a symlinked output_dir.
    root = job_registry._output_root  # resolved in JobRegistry.__init__
    run_dir = (root / record.id).resolve()
    try:
        run_dir.relative_to(root)
    except ValueError as exc:
        raise ModelError(
            400,
            f"Refusing to delete {model_id!r}: its directory resolves outside the training-output root.",
        ) from exc
    if run_dir == root:
        raise ModelError(400, f"Refusing to delete {model_id!r}: resolves to the output root itself.")

    # A COMPLETED run's checkpoint can still be an active inference target —
    # the registry's running-job guard below doesn't cover that.
    in_use = _model_in_use(run_dir)
    if in_use is not None:
        raise ModelError(409, in_use)

    try:
        job_registry.delete(model_id)
    except JobNotRunningError as exc:
        # JobRegistry.delete raises JobNotRunningError when the job IS running
        # (its guard refuses to delete a live run's dir).
        raise ModelError(
            409,
            f"Model {model_id!r} is still training — stop the run before deleting it.",
        ) from exc
    except JobNotFoundError as exc:
        raise ModelError(404, f"Model {model_id!r} not found.") from exc
    except Exception as exc:
        logger.error("Failed to delete local model %s: %s", model_id, exc)
        raise ModelError(502, f"Failed to delete model: {exc}") from exc

    invalidate_model_listing_cache()
    if record.hf_repo_id:
        # A pushed run's hub row survives the local delete — drop its cached
        # hub metadata so the card re-reads fresh.
        invalidate_model_hub_info(record.hf_repo_id)
    logger.info("Deleted local model %s (run dir %s)", model_id, run_dir)
    return {"deleted": True, "id": model_id}


# ---------------------------------------------------------------------------
# Download a Hub model into the local models dir (background, pollable).
# ---------------------------------------------------------------------------


def _fetch_model_snapshot(repo_id: str) -> None:
    """Snapshot a Hub model repo into ``<local models root>/<repo_id>/``.

    After the fetch the dir must resolve to a usable checkpoint
    (_resolve_pretrained_dir) — a repo with no config.json / checkpoints tree is
    not a policy and is rejected (the manager's cleanup then removes it).
    Invalidates the /models listing cache so the source flips to "both"
    immediately."""
    target = _local_models_root() / repo_id
    snapshot_download(repo_id, repo_type="model", local_dir=str(target))
    if _resolve_pretrained_dir(target) is None:
        raise ModelError(
            400,
            f"'{repo_id}' doesn't look like a policy checkpoint — no config.json "
            "or checkpoints/<step>/pretrained_model tree in the repo.",
        )
    invalidate_model_listing_cache()
    # The card flips from the hub view to the local one — drop the cached hub
    # metadata alongside the listing.
    invalidate_model_hub_info(repo_id)


def _cleanup_partial_model(repo_id: str) -> None:
    """Remove a failed download's partial/unusable dir so it isn't mistaken for
    a complete local checkpoint by is_model_available_locally."""
    target = _local_models_root() / repo_id
    if target.exists() and _resolve_pretrained_dir(target) is None:
        shutil.rmtree(target, ignore_errors=True)


# The models twin of datasets.download_manager — same generic state machine
# (one download at a time, start/poll, survives navigation), model-specific
# fetch/cleanup.
model_download_manager = DownloadManager(_fetch_model_snapshot, _cleanup_partial_model)


# ---------------------------------------------------------------------------
# Import a model checkpoint folder already on disk into the local models dir.
# ---------------------------------------------------------------------------


def import_local_model(source_path: str, name: str | None = None) -> dict[str, Any]:
    """Copy a policy checkpoint folder already on the server machine into the
    local models dir, so it appears under "Local" (the models mirror of
    datasets.import_local_dataset — a COPY, never a move or a pointer-register).

    `source_path` points at a checkpoint dir in either shape
    _resolve_pretrained_dir recognizes (root config.json, or a
    checkpoints/<step>/pretrained_model tree). `name` is the target id — a bare
    name or ``namespace/name``, validated with the same segment rules dataset
    names use; defaults to the source folder's basename.

    Raises ModelError on: a missing source (404), a source that isn't a
    checkpoint (400), a bad target name (400), a target escaping the models
    root or overlapping the source (400), or a target that already exists
    (409). Invalidates the listing cache on success. Returns {"repo_id": ...}.

    NOTE: the copy runs SYNCHRONOUSLY (route blocks, frontend shows a spinner),
    same documented tradeoff as the dataset import — checkpoints are typically
    far smaller than datasets, so inline is even more comfortably acceptable.
    """
    try:
        src = Path(source_path).expanduser().resolve()
    except OSError:
        raise ModelError(400, "Invalid source path.") from None
    if not src.is_dir():
        raise ModelError(404, f"No folder found at '{source_path}'.")
    if _resolve_pretrained_dir(src) is None:
        raise ModelError(
            400,
            "That folder isn't a policy checkpoint (no config.json or "
            "checkpoints/<step>/pretrained_model tree inside it).",
        )

    raw = (name or "").strip() or src.name
    ok, reason = validate_dataset_repo_id(raw)
    if not ok:
        # The segment rules are shared with dataset names; reword for models.
        raise ModelError(400, reason.replace("Dataset name", "Model name"))

    root = _local_models_root().resolve()
    dst = (root / raw).resolve()
    # Reject a target that escapes the models root (traversal).
    if dst == root or root not in dst.parents:
        raise ModelError(400, "Invalid target model name.")
    # Refuse overlapping source/target (e.g. importing a dir into itself).
    if dst == src or src in dst.parents or dst in src.parents:
        raise ModelError(400, "The source folder and the import target overlap.")
    if dst.exists():
        raise ModelError(409, f"A model named '{raw}' already exists locally.")

    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(src, dst)
    except OSError as exc:
        logger.error("Failed to import model %s -> %s: %s", src, dst, exc)
        # Remove a partial copy so a failed import leaves no half-written dir.
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        raise ModelError(500, f"Failed to copy the model: {exc}") from exc

    invalidate_model_listing_cache()
    logger.info("Imported model %s -> %s", src, dst)
    return {"repo_id": raw}
