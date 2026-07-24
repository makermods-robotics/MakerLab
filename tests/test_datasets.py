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
"""Tests for makerlab.datasets — local cache walk and merge logic."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient


def _make_dataset(root: Path, repo_id: str, episodes: int = 1) -> None:
    """Create the minimal layout `_is_dataset_dir` recognizes. `episodes`
    defaults to 1 so the dataset isn't filtered out as empty."""
    d = root / repo_id
    (d / "meta").mkdir(parents=True)
    (d / "meta" / "info.json").write_text(json.dumps({"total_episodes": episodes}))


def test_list_local_datasets_empty_when_root_missing(
    tmp_lerobot_home: Path,
) -> None:
    # tmp_lerobot_home creates the cache; remove it so the function sees the
    # "missing root" branch.
    import shutil

    from makerlab.datasets import list_local_datasets

    shutil.rmtree(tmp_lerobot_home)
    assert list_local_datasets() == []


def test_list_local_datasets_finds_top_level_dataset(
    tmp_lerobot_home: Path,
) -> None:
    from makerlab.datasets import list_local_datasets

    _make_dataset(tmp_lerobot_home, "pusht")
    result = list_local_datasets()
    repo_ids = [d["repo_id"] for d in result]
    assert "pusht" in repo_ids


def test_list_local_datasets_finds_nested_user_dataset(
    tmp_lerobot_home: Path,
) -> None:
    from makerlab.datasets import list_local_datasets

    _make_dataset(tmp_lerobot_home, "alice/pusht")
    result = list_local_datasets()
    repo_ids = [d["repo_id"] for d in result]
    assert "alice/pusht" in repo_ids


def test_list_local_datasets_skips_non_dataset_dirs(
    tmp_lerobot_home: Path,
) -> None:
    from makerlab.datasets import list_local_datasets

    (tmp_lerobot_home / "calibration").mkdir(exist_ok=True)
    (tmp_lerobot_home / "ports").mkdir(exist_ok=True)
    _make_dataset(tmp_lerobot_home, "real_dataset")

    result = list_local_datasets()
    repo_ids = [d["repo_id"] for d in result]
    assert "real_dataset" in repo_ids
    assert "calibration" not in repo_ids
    assert "ports" not in repo_ids


def test_list_local_datasets_hides_empty_dataset(
    tmp_lerobot_home: Path,
) -> None:
    """A 0-episode dataset (aborted recording) is hidden so it can't be picked
    for merging/training, where it only errors out."""
    from makerlab.datasets import list_local_datasets

    _make_dataset(tmp_lerobot_home, "has_eps", episodes=3)
    _make_dataset(tmp_lerobot_home, "empty_ds", episodes=0)

    repo_ids = [d["repo_id"] for d in list_local_datasets()]
    assert "has_eps" in repo_ids
    assert "empty_ds" not in repo_ids


def test_list_user_datasets_returns_empty_when_not_logged_in(
    tmp_lerobot_home: Path,
) -> None:
    from makerlab.datasets import list_user_datasets

    with patch("makerlab.datasets.cached_whoami", return_value=None):
        assert list_user_datasets() == []


def test_list_all_datasets_merges_hub_and_local(
    tmp_lerobot_home: Path,
) -> None:
    from makerlab.datasets import list_all_datasets

    _make_dataset(tmp_lerobot_home, "alice/pusht")

    with patch(
        "makerlab.datasets.list_user_datasets",
        return_value=[
            {"repo_id": "alice/pusht", "last_modified": "2026-01-01T00:00:00Z", "private": False},
            {"repo_id": "alice/aloha", "last_modified": "2026-02-01T00:00:00Z", "private": True},
        ],
    ):
        result = list_all_datasets()

    by_id = {d["repo_id"]: d for d in result}
    assert by_id["alice/pusht"]["source"] == "both"
    assert by_id["alice/aloha"]["source"] == "hub"


def _write_info(root: Path, repo_id: str, info: dict[str, Any]) -> Path:
    """Write a dataset dir with the given meta/info.json; returns the dir."""
    d = root / repo_id
    (d / "meta").mkdir(parents=True)
    (d / "meta" / "info.json").write_text(json.dumps(info))
    return d


def test_get_local_dataset_info_returns_full_details(
    tmp_lerobot_home: Path,
) -> None:
    from makerlab.datasets import get_local_dataset_info

    d = _write_info(
        tmp_lerobot_home,
        "alice/pick",
        {
            "total_episodes": 20,
            "total_frames": 16723,
            "fps": 30,
            "robot_type": "so_follower",
            "features": {
                "action": {"dtype": "float32"},
                "observation.state": {"dtype": "float32"},
                "observation.images.wrist": {"dtype": "video"},
                "observation.images.front": {"dtype": "video"},
            },
        },
    )
    # v3.0 task metadata: tasks.parquet with task_index + task columns,
    # deliberately written out of index order to check the sort.
    pq.write_table(
        pa.table({"task_index": [1, 0], "task": ["second task", "first task"]}),
        d / "meta" / "tasks.parquet",
    )
    # v3.0 episode metadata: per-episode `tasks` column split across chunked
    # parquet files — 18 episodes of "first task", 2 of "second task".
    episodes_dir = d / "meta" / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True)
    pq.write_table(
        pa.table({"episode_index": list(range(15)), "tasks": [["first task"]] * 15}),
        episodes_dir / "file-000.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": list(range(15, 20)),
                "tasks": [["first task"]] * 3 + [["second task"]] * 2,
            }
        ),
        episodes_dir / "file-001.parquet",
    )
    (d / "data").mkdir()
    (d / "data" / "file-000.parquet").write_bytes(b"x" * 1234)

    result = get_local_dataset_info("alice/pick")
    assert result is not None
    assert result["total_episodes"] == 20
    assert result["total_frames"] == 16723
    assert result["fps"] == 30
    assert result["robot_type"] == "so_follower"
    assert result["cameras"] == ["wrist", "front"]
    assert result["tasks"] == [
        {"task": "first task", "num_episodes": 18},
        {"task": "second task", "num_episodes": 2},
    ]
    # Directory walk covers data + meta files, so at least the data blob.
    assert result["size_bytes"] >= 1234


def test_get_local_dataset_info_reads_v2_tasks_jsonl(
    tmp_lerobot_home: Path,
) -> None:
    from makerlab.datasets import get_local_dataset_info

    d = _write_info(
        tmp_lerobot_home,
        "old_format",
        {"total_episodes": 2, "total_frames": 100, "fps": 30, "features": {}},
    )
    lines = [
        json.dumps({"task_index": 1, "task": "beta"}),
        json.dumps({"task_index": 0, "task": "alpha"}),
    ]
    (d / "meta" / "tasks.jsonl").write_text("\n".join(lines))
    # v2.x episode metadata: episodes.jsonl with per-episode `tasks` lists.
    ep_lines = [
        json.dumps({"episode_index": 0, "tasks": ["alpha"], "length": 50}),
        json.dumps({"episode_index": 1, "tasks": ["beta"], "length": 50}),
        json.dumps({"episode_index": 2, "tasks": ["beta"], "length": 50}),
    ]
    (d / "meta" / "episodes.jsonl").write_text("\n".join(ep_lines))

    result = get_local_dataset_info("old_format")
    assert result is not None
    assert result["tasks"] == [
        {"task": "alpha", "num_episodes": 1},
        {"task": "beta", "num_episodes": 2},
    ]


def test_get_local_dataset_info_single_task_missing_episode_metadata(
    tmp_lerobot_home: Path,
) -> None:
    """Task strings without episode metadata still render — counts degrade to 0."""
    from makerlab.datasets import get_local_dataset_info

    d = _write_info(
        tmp_lerobot_home,
        "alice/solo",
        {"total_episodes": 5, "total_frames": 500, "fps": 30, "features": {}},
    )
    pq.write_table(
        pa.table({"task_index": [0], "task": ["only task"]}),
        d / "meta" / "tasks.parquet",
    )

    result = get_local_dataset_info("alice/solo")
    assert result is not None
    assert result["tasks"] == [{"task": "only task", "num_episodes": 0}]


def test_get_local_dataset_info_zero_episodes_and_no_cameras(
    tmp_lerobot_home: Path,
) -> None:
    """A 0-episode dataset is hidden from the listing but must still resolve
    here, so the frontend can render its warning badges."""
    from makerlab.datasets import get_local_dataset_info

    _write_info(
        tmp_lerobot_home,
        "alice/aborted",
        {
            "total_episodes": 0,
            "total_frames": 0,
            "fps": 30,
            "robot_type": "so_follower",
            "features": {"action": {"dtype": "float32"}},
        },
    )

    result = get_local_dataset_info("alice/aborted")
    assert result is not None
    assert result["total_episodes"] == 0
    assert result["cameras"] == []
    assert result["tasks"] == []


def test_get_local_dataset_info_missing_dataset_returns_none(
    tmp_lerobot_home: Path,
) -> None:
    from makerlab.datasets import get_local_dataset_info

    assert get_local_dataset_info("nobody/nothing") is None


def test_get_local_dataset_info_rejects_path_traversal(
    tmp_lerobot_home: Path,
) -> None:
    from makerlab.datasets import get_local_dataset_info

    # A dataset-shaped dir OUTSIDE the cache root must not be reachable.
    outside = tmp_lerobot_home.parent / "outside"
    (outside / "meta").mkdir(parents=True)
    (outside / "meta" / "info.json").write_text(json.dumps({"total_episodes": 1}))

    assert get_local_dataset_info("../outside") is None
    assert get_local_dataset_info("..") is None
    assert get_local_dataset_info(".") is None


def test_datasets_info_endpoint(client: TestClient, tmp_lerobot_home: Path) -> None:
    _write_info(
        tmp_lerobot_home,
        "alice/pick",
        {
            "total_episodes": 3,
            "total_frames": 900,
            "fps": 30,
            "robot_type": "so_follower",
            "features": {"observation.images.front": {"dtype": "video"}},
        },
    )

    ok = client.get("/datasets/info", params={"repo_id": "alice/pick"})
    assert ok.status_code == 200
    body = ok.json()
    assert body["total_episodes"] == 3
    assert body["cameras"] == ["front"]
    assert body["size_bytes"] > 0

    missing = client.get("/datasets/info", params={"repo_id": "alice/ghost"})
    assert missing.status_code == 404


def test_video_camera_names_filters_by_dtype() -> None:
    from makerlab.datasets import _video_camera_names

    features = {
        "observation.images.front": {"dtype": "video"},
        "observation.images.raw": {"dtype": "image"},
        "observation.images.wrist": {"dtype": "video"},
        "observation.state": {"dtype": "float32"},
        "action": {"dtype": "float32"},
    }
    assert _video_camera_names(features) == ["front", "wrist"]


# --- Hub sync status --------------------------------------------------------


def _clear_hub_status_cache() -> None:
    from makerlab import datasets as ds

    with ds._HUB_STATUS_LOCK:
        ds._HUB_STATUS_CACHE.clear()


def test_get_hub_status_reports_on_hub_when_repo_exists() -> None:
    from makerlab import datasets as ds

    _clear_hub_status_cache()
    fake_api = MagicMock()
    fake_api.repo_exists.return_value = True
    with patch("makerlab.datasets.shared_hf_api", return_value=fake_api):
        result = ds.get_hub_status("alice/pick")

    assert result["status"] == "on_hub"
    assert result["url"] == "https://huggingface.co/datasets/alice/pick"
    fake_api.repo_exists.assert_called_once_with("alice/pick", repo_type="dataset")


def test_get_hub_status_reports_local_only_when_missing_from_hub_but_local() -> None:
    """Not on the Hub, but a usable local copy exists → "local_only" (offer
    upload)."""
    from makerlab import datasets as ds

    _clear_hub_status_cache()
    fake_api = MagicMock()
    fake_api.repo_exists.return_value = False
    with (
        patch("makerlab.datasets.shared_hf_api", return_value=fake_api),
        patch("makerlab.datasets.is_dataset_available_locally", return_value=True),
    ):
        result = ds.get_hub_status("alice/pick")

    assert result["status"] == "local_only"
    assert result["url"] is None


def test_get_hub_status_reports_absent_when_neither_hub_nor_local() -> None:
    """Neither on the Hub nor in the local cache → "absent", NOT "local_only".

    This is the BUG-3 root cause: a stale pin (e.g. a merge output that was
    deleted/renamed) used to report "local_only", which the info card read as
    "you have it locally" and rendered the contradictory "not downloaded
    locally" + "Local only / Upload" pair. "absent" is also NOT cached (a later
    record/merge can make it appear locally), so a second call re-checks."""
    from makerlab import datasets as ds

    _clear_hub_status_cache()
    fake_api = MagicMock()
    fake_api.repo_exists.return_value = False
    with (
        patch("makerlab.datasets.shared_hf_api", return_value=fake_api),
        patch("makerlab.datasets.is_dataset_available_locally", return_value=False),
    ):
        result = ds.get_hub_status("makermods/sock")
        assert result["status"] == "absent"
        assert result["url"] is None
        # Not cached: a second call re-invokes repo_exists.
        ds.get_hub_status("makermods/sock")
    assert fake_api.repo_exists.call_count == 2


def test_get_hub_status_degrades_to_unknown_offline() -> None:
    """A transport error (offline / rate-limited) degrades to "unknown" and is
    NOT cached, so the next check re-tries once connectivity returns."""
    from makerlab import datasets as ds

    _clear_hub_status_cache()
    fake_api = MagicMock()
    fake_api.repo_exists.side_effect = OSError("no network")
    with patch("makerlab.datasets.shared_hf_api", return_value=fake_api):
        result = ds.get_hub_status("alice/pick")
        assert result["status"] == "unknown"
        assert result["url"] is None
        # Not cached: a second call re-invokes repo_exists.
        ds.get_hub_status("alice/pick")
    assert fake_api.repo_exists.call_count == 2


def test_get_hub_status_caches_definitive_answer() -> None:
    """A definitive answer is memoized: repo_exists runs once across calls."""
    from makerlab import datasets as ds

    _clear_hub_status_cache()
    fake_api = MagicMock()
    fake_api.repo_exists.return_value = True
    with patch("makerlab.datasets.shared_hf_api", return_value=fake_api):
        ds.get_hub_status("alice/pick")
        ds.get_hub_status("alice/pick")
    assert fake_api.repo_exists.call_count == 1


def test_invalidate_hub_status_forces_recheck() -> None:
    """After invalidation (called on successful upload), the next check
    re-queries the Hub — so a "local_only" answer can flip to "on_hub"."""
    from makerlab import datasets as ds

    _clear_hub_status_cache()
    fake_api = MagicMock()
    fake_api.repo_exists.return_value = False
    with (
        patch("makerlab.datasets.shared_hf_api", return_value=fake_api),
        patch("makerlab.datasets.is_dataset_available_locally", return_value=True),
    ):
        assert ds.get_hub_status("alice/pick")["status"] == "local_only"
        # Simulate a successful upload: repo now exists, cache invalidated.
        ds.invalidate_hub_status("alice/pick")
        fake_api.repo_exists.return_value = True
        assert ds.get_hub_status("alice/pick")["status"] == "on_hub"
    assert fake_api.repo_exists.call_count == 2


def test_hub_status_endpoint(client: TestClient) -> None:
    _clear_hub_status_cache()
    fake_api = MagicMock()
    fake_api.repo_exists.return_value = True
    with patch("makerlab.datasets.shared_hf_api", return_value=fake_api):
        resp = client.get("/datasets/hub-status", params={"repo_id": "alice/pick"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["repo_id"] == "alice/pick"
    assert body["status"] == "on_hub"
    assert body["url"] == "https://huggingface.co/datasets/alice/pick"


# --- Rename -----------------------------------------------------------------


def test_rename_local_dataset_moves_directory(tmp_lerobot_home: Path) -> None:
    """Happy path: the directory moves, only the name segment changes, and the
    returned repo id carries the fixed namespace prefix."""
    from makerlab.datasets import rename_local_dataset

    _make_dataset(tmp_lerobot_home, "makermods/old_name", episodes=3)

    new_id = rename_local_dataset("makermods/old_name", "new_name")

    assert new_id == "makermods/new_name"
    assert not (tmp_lerobot_home / "makermods" / "old_name").exists()
    assert (tmp_lerobot_home / "makermods" / "new_name" / "meta" / "info.json").is_file()


def test_rename_endpoint_old_id_404s_new_id_resolves(client: TestClient, tmp_lerobot_home: Path) -> None:
    """End-to-end through the route: after a rename the old id 404s on
    /datasets/info and the new id resolves."""
    _write_info(
        tmp_lerobot_home,
        "makermods/pick",
        {"total_episodes": 3, "total_frames": 900, "fps": 30, "features": {}},
    )

    resp = client.post(
        "/datasets/rename",
        json={"repo_id": "makermods/pick", "new_name": "place"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"success": True, "repo_id": "makermods/place"}

    old = client.get("/datasets/info", params={"repo_id": "makermods/pick"})
    assert old.status_code == 404
    new = client.get("/datasets/info", params={"repo_id": "makermods/place"})
    assert new.status_code == 200
    assert new.json()["total_episodes"] == 3


def test_rename_bare_dataset_keeps_no_namespace(tmp_lerobot_home: Path) -> None:
    """A dataset with no namespace renames to a bare name (no prefix invented)."""
    from makerlab.datasets import rename_local_dataset

    _make_dataset(tmp_lerobot_home, "solo", episodes=1)
    assert rename_local_dataset("solo", "solo2") == "solo2"
    assert (tmp_lerobot_home / "solo2" / "meta" / "info.json").is_file()


def test_rename_same_name_is_noop(tmp_lerobot_home: Path) -> None:
    from makerlab.datasets import rename_local_dataset

    _make_dataset(tmp_lerobot_home, "makermods/keep", episodes=1)
    assert rename_local_dataset("makermods/keep", "keep") == "makermods/keep"


def test_rename_rejects_invalid_name(tmp_lerobot_home: Path) -> None:
    """new_name is validated with the same rules as recording — a slash is a
    name segment, not a namespace, so it's rejected."""
    from makerlab.datasets import DatasetRenameError, rename_local_dataset

    _make_dataset(tmp_lerobot_home, "makermods/src", episodes=1)

    for bad in ["with/slash", "..", " leading", ""]:
        with pytest.raises(DatasetRenameError) as exc:
            rename_local_dataset("makermods/src", bad)
        assert exc.value.status == 400
    # The source was never moved by a rejected rename.
    assert (tmp_lerobot_home / "makermods" / "src").exists()


def test_rename_missing_source_404s(tmp_lerobot_home: Path) -> None:
    from makerlab.datasets import DatasetRenameError, rename_local_dataset

    with pytest.raises(DatasetRenameError) as exc:
        rename_local_dataset("makermods/ghost", "new")
    assert exc.value.status == 404


def test_rename_target_exists_409s(tmp_lerobot_home: Path) -> None:
    from makerlab.datasets import DatasetRenameError, rename_local_dataset

    _make_dataset(tmp_lerobot_home, "makermods/src", episodes=1)
    _make_dataset(tmp_lerobot_home, "makermods/taken", episodes=1)

    with pytest.raises(DatasetRenameError) as exc:
        rename_local_dataset("makermods/src", "taken")
    assert exc.value.status == 409
    # Neither directory was touched.
    assert (tmp_lerobot_home / "makermods" / "src").exists()
    assert (tmp_lerobot_home / "makermods" / "taken").exists()


def test_rename_rejects_path_traversal(tmp_lerobot_home: Path) -> None:
    """A source id escaping the cache root is refused before any move."""
    from makerlab.datasets import DatasetRenameError, rename_local_dataset

    outside = tmp_lerobot_home.parent / "outside"
    (outside / "meta").mkdir(parents=True)
    (outside / "meta" / "info.json").write_text(json.dumps({"total_episodes": 1}))

    for bad in ["../outside", "..", "."]:
        with pytest.raises(DatasetRenameError):
            rename_local_dataset(bad, "new")


def test_rename_busy_guard_recording(tmp_lerobot_home: Path) -> None:
    """A rename is refused (409) while a recording session writes to the id —
    matching either the stamped id or a rename of the still-writing base."""
    from makerlab import record as rec
    from makerlab.datasets import DatasetRenameError, rename_local_dataset

    _make_dataset(tmp_lerobot_home, "makermods/live", episodes=1)

    fake_cfg = MagicMock()
    # Recording stamps a timestamp: name -> name_<ts>.
    fake_cfg.dataset_repo_id = "makermods/live_20260101"
    with (
        patch.object(rec, "recording_active", True),
        patch.object(rec, "recording_config", fake_cfg),
        pytest.raises(DatasetRenameError) as exc,
    ):
        rename_local_dataset("makermods/live", "renamed")
    assert exc.value.status == 409
    assert (tmp_lerobot_home / "makermods" / "live").exists()


def test_rename_busy_guard_merge(tmp_lerobot_home: Path) -> None:
    """A rename is refused while a merge is producing the target id."""
    from makerlab import merge
    from makerlab.datasets import DatasetRenameError, rename_local_dataset

    _make_dataset(tmp_lerobot_home, "makermods/out", episodes=1)

    with (
        patch.object(merge.merge_manager, "state", "running"),
        patch.object(merge.merge_manager, "output_repo_id", "makermods/out"),
        pytest.raises(DatasetRenameError) as exc,
    ):
        rename_local_dataset("makermods/out", "renamed")
    assert exc.value.status == 409


def test_rename_busy_guard_upload(tmp_lerobot_home: Path) -> None:
    """A rename is refused (409) while the dataset is being pushed to the Hub."""
    from makerlab import record as rec
    from makerlab.datasets import DatasetRenameError, rename_local_dataset

    _make_dataset(tmp_lerobot_home, "makermods/uploading", episodes=1)

    with (
        patch.object(rec.upload_manager, "state", "running"),
        patch.object(rec.upload_manager, "repo_id", "makermods/uploading"),
        pytest.raises(DatasetRenameError) as exc,
    ):
        rename_local_dataset("makermods/uploading", "renamed")
    assert exc.value.status == 409
    assert (tmp_lerobot_home / "makermods" / "uploading").exists()


def test_rename_busy_guard_local_training(tmp_lerobot_home: Path) -> None:
    """A rename is refused while a running local job trains on the id."""
    from makerlab.datasets import DatasetRenameError, rename_local_dataset

    _make_dataset(tmp_lerobot_home, "makermods/train_ds", episodes=1)

    # _dataset_in_use imports job_registry from .jobs lazily (datasets<->record
    # cycle), so patch it at its source module.
    from makerlab import jobs

    job = MagicMock()
    job.state = "running"
    job.runner = "local"
    job.config.dataset_repo_id = "makermods/train_ds"
    with (
        patch.object(jobs.job_registry, "list", return_value=[job]),
        pytest.raises(DatasetRenameError) as exc,
    ):
        rename_local_dataset("makermods/train_ds", "renamed")
    assert exc.value.status == 409


def test_rename_invalidates_hub_status_for_both_ids(tmp_lerobot_home: Path) -> None:
    """The cached Hub-existence answer is dropped for BOTH the old and new id,
    so the info card re-checks each after the move."""
    from makerlab import datasets as ds

    _make_dataset(tmp_lerobot_home, "makermods/before", episodes=1)

    with patch("makerlab.datasets.invalidate_hub_status") as inval:
        ds.rename_local_dataset("makermods/before", "after")

    called = {c.args[0] for c in inval.call_args_list}
    assert called == {"makermods/before", "makermods/after"}


# --- Hub visibility / tags editing (post-upload) ----------------------------


def test_set_dataset_visibility_calls_hfapi_with_repo_type() -> None:
    """set_dataset_visibility drives HfApi.update_repo_settings with the
    requested private flag and repo_type="dataset"; result echoes the flag."""
    from makerlab import datasets as ds

    _clear_hub_status_cache()
    fake_api = MagicMock()
    with (
        patch("makerlab.datasets.shared_hf_api", return_value=fake_api),
        patch("makerlab.datasets.hf_hub_offline", return_value=False),
        patch("makerlab.datasets.invalidate_hub_status") as inval,
    ):
        result = ds.set_dataset_visibility("alice/pick", private=True)

    fake_api.update_repo_settings.assert_called_once_with("alice/pick", private=True, repo_type="dataset")
    assert result == {"repo_id": "alice/pick", "private": True}
    inval.assert_called_once_with("alice/pick")


def test_set_dataset_visibility_public_passes_false() -> None:
    from makerlab import datasets as ds

    fake_api = MagicMock()
    with (
        patch("makerlab.datasets.shared_hf_api", return_value=fake_api),
        patch("makerlab.datasets.hf_hub_offline", return_value=False),
        patch("makerlab.datasets.invalidate_hub_status"),
    ):
        result = ds.set_dataset_visibility("alice/pick", private=False)

    fake_api.update_repo_settings.assert_called_once_with("alice/pick", private=False, repo_type="dataset")
    assert result["private"] is False


def test_set_dataset_visibility_rejected_offline() -> None:
    """Offline: no HfApi call, a 400 DatasetHubEditError instead."""
    from makerlab import datasets as ds

    fake_api = MagicMock()
    with (
        patch("makerlab.datasets.shared_hf_api", return_value=fake_api),
        patch("makerlab.datasets.hf_hub_offline", return_value=True),
        pytest.raises(ds.DatasetHubEditError) as exc,
    ):
        ds.set_dataset_visibility("alice/pick", private=True)

    assert exc.value.status == 400
    fake_api.update_repo_settings.assert_not_called()


def test_set_dataset_visibility_maps_permission_error() -> None:
    """A 403/forbidden Hub failure becomes a 403 DatasetHubEditError."""
    from makerlab import datasets as ds

    fake_api = MagicMock()
    fake_api.update_repo_settings.side_effect = Exception("403 Forbidden: no write access")
    with (
        patch("makerlab.datasets.shared_hf_api", return_value=fake_api),
        patch("makerlab.datasets.hf_hub_offline", return_value=False),
        pytest.raises(ds.DatasetHubEditError) as exc,
    ):
        ds.set_dataset_visibility("alice/pick", private=True)

    assert exc.value.status == 403


def test_set_dataset_tags_runs_through_with_makerlab_tag_before_update() -> None:
    """User tags are funnelled through with_makerlab_tag (so makermods/openbooth/
    MakerLab survive) BEFORE metadata_update, which is called with overwrite=True
    and repo_type="dataset". The returned tag list is what was written."""
    from makerlab import datasets as ds
    from makerlab.utils.config import REQUIRED_HUB_TAGS

    _clear_hub_status_cache()
    with (
        patch("makerlab.datasets.hf_hub_offline", return_value=False),
        patch("makerlab.datasets.metadata_update") as meta,
        patch("makerlab.datasets.invalidate_hub_status") as inval,
    ):
        result = ds.set_dataset_tags("alice/pick", ["robotics", "so101"])

    meta.assert_called_once()
    args, kwargs = meta.call_args
    assert args[0] == "alice/pick"
    written = args[1]["tags"]
    assert kwargs["repo_type"] == "dataset"
    assert kwargs["overwrite"] is True
    # User tags come first, org tags are appended and never dropped.
    assert written[:2] == ["robotics", "so101"]
    for required in REQUIRED_HUB_TAGS:
        assert required in written
    assert result["tags"] == written
    inval.assert_called_once_with("alice/pick")


def test_set_dataset_tags_preserves_org_tags_when_user_omits_them() -> None:
    """Even an empty user tag list still writes the required org tags — an edit
    can never strip makermods/openbooth/MakerLab off the card."""
    from makerlab import datasets as ds
    from makerlab.utils.config import REQUIRED_HUB_TAGS

    with (
        patch("makerlab.datasets.hf_hub_offline", return_value=False),
        patch("makerlab.datasets.metadata_update") as meta,
        patch("makerlab.datasets.invalidate_hub_status"),
    ):
        result = ds.set_dataset_tags("alice/pick", [])

    written = meta.call_args.args[1]["tags"]
    assert set(REQUIRED_HUB_TAGS).issubset(set(written))
    assert result["tags"] == written


def test_set_dataset_tags_rejected_offline() -> None:
    from makerlab import datasets as ds

    with (
        patch("makerlab.datasets.hf_hub_offline", return_value=True),
        patch("makerlab.datasets.metadata_update") as meta,
        pytest.raises(ds.DatasetHubEditError) as exc,
    ):
        ds.set_dataset_tags("alice/pick", ["robotics"])

    assert exc.value.status == 400
    meta.assert_not_called()


def test_set_dataset_tags_maps_auth_error() -> None:
    """A 401/auth Hub failure maps to a 403 DatasetHubEditError with docs_url."""
    from makerlab import datasets as ds

    with (
        patch("makerlab.datasets.hf_hub_offline", return_value=False),
        patch(
            "makerlab.datasets.metadata_update",
            side_effect=Exception("401 you must be authenticated"),
        ),
        pytest.raises(ds.DatasetHubEditError) as exc,
    ):
        ds.set_dataset_tags("alice/pick", ["robotics"])

    assert exc.value.status == 403
    assert exc.value.docs_url is not None


def test_get_hub_settings_returns_private_and_tags() -> None:
    from makerlab import datasets as ds

    fake_info = MagicMock()
    fake_info.private = True
    fake_info.tags = ["robotics", "makermods"]
    fake_api = MagicMock()
    fake_api.dataset_info.return_value = fake_info
    with (
        patch("makerlab.datasets.shared_hf_api", return_value=fake_api),
        patch("makerlab.datasets.hf_hub_offline", return_value=False),
    ):
        result = ds.get_hub_settings("alice/pick")

    fake_api.dataset_info.assert_called_once_with("alice/pick")
    assert result == {"repo_id": "alice/pick", "private": True, "tags": ["robotics", "makermods"]}


def test_get_hub_settings_rejected_offline() -> None:
    from makerlab import datasets as ds

    fake_api = MagicMock()
    with (
        patch("makerlab.datasets.shared_hf_api", return_value=fake_api),
        patch("makerlab.datasets.hf_hub_offline", return_value=True),
        pytest.raises(ds.DatasetHubEditError) as exc,
    ):
        ds.get_hub_settings("alice/pick")

    assert exc.value.status == 400
    fake_api.dataset_info.assert_not_called()


def test_visibility_endpoint(client: TestClient) -> None:
    with (
        patch("makerlab.datasets.shared_hf_api", return_value=MagicMock()),
        patch("makerlab.datasets.hf_hub_offline", return_value=False),
        patch("makerlab.datasets.invalidate_hub_status"),
    ):
        resp = client.post("/datasets/visibility", json={"repo_id": "alice/pick", "private": True})
    assert resp.status_code == 200
    assert resp.json() == {"repo_id": "alice/pick", "private": True}


def test_visibility_endpoint_offline_400(client: TestClient) -> None:
    with patch("makerlab.datasets.hf_hub_offline", return_value=True):
        resp = client.post("/datasets/visibility", json={"repo_id": "alice/pick", "private": True})
    assert resp.status_code == 400


def test_tags_endpoint_writes_and_preserves_org_tags(client: TestClient) -> None:
    from makerlab.utils.config import REQUIRED_HUB_TAGS

    with (
        patch("makerlab.datasets.hf_hub_offline", return_value=False),
        patch("makerlab.datasets.metadata_update") as meta,
        patch("makerlab.datasets.invalidate_hub_status"),
    ):
        resp = client.post("/datasets/tags", json={"repo_id": "alice/pick", "tags": ["robotics"]})
    assert resp.status_code == 200
    written = meta.call_args.args[1]["tags"]
    for required in REQUIRED_HUB_TAGS:
        assert required in written
    assert resp.json()["tags"] == written


def test_hub_settings_endpoint(client: TestClient) -> None:
    fake_info = MagicMock()
    fake_info.private = False
    fake_info.tags = ["robotics"]
    fake_api = MagicMock()
    fake_api.dataset_info.return_value = fake_info
    with (
        patch("makerlab.datasets.shared_hf_api", return_value=fake_api),
        patch("makerlab.datasets.hf_hub_offline", return_value=False),
    ):
        resp = client.get("/datasets/hub-settings", params={"repo_id": "alice/pick"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["private"] is False
    assert body["tags"] == ["robotics"]


# ---------------------------------------------------------------------------
# DownloadManager — background Hub-dataset download (start → running → done|error).
# The fetch runs in a worker thread; tests mock snapshot_download so no real Hub
# call happens, then join the thread before asserting on the final state.
# ---------------------------------------------------------------------------


def _join_download(mgr, timeout: float = 5.0) -> None:
    thread = mgr._thread
    if thread is not None:
        thread.join(timeout=timeout)


def _dataset_download_manager():
    """A fresh DownloadManager wired with the dataset fetch/cleanup callables —
    the same wiring as the module singleton, but with clean state per test."""
    from makerlab import datasets as ds

    return ds.DownloadManager(ds._fetch_dataset_snapshot, ds._cleanup_partial_dataset)


def test_download_manager_idle_shape() -> None:
    status = _dataset_download_manager().get_status()
    assert status["state"] == "idle"
    assert status["repo_id"] is None
    assert status["message"] is None
    assert status["error"] is None


def test_download_manager_start_runs_and_completes(
    tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A start fetches in a worker thread into the FLAT cache layout and lands in
    state "done", invalidating the hub status + listing caches so the source
    flips to "both"."""
    from makerlab import datasets as ds

    def _fake_snapshot(repo_id, repo_type, local_dir):  # noqa: ARG001
        # Materialize the flat layout list_local_datasets / is_dataset_available
        # _locally recognize.
        d = Path(local_dir)
        (d / "meta").mkdir(parents=True)
        (d / "meta" / "info.json").write_text(json.dumps({"total_episodes": 2}))

    monkeypatch.setattr(ds, "snapshot_download", _fake_snapshot)
    invalidated: list[str] = []
    monkeypatch.setattr(ds, "invalidate_hub_status", invalidated.append)

    mgr = _dataset_download_manager()
    result = mgr.start("alice/pick")
    assert result == {"started": True, "repo_id": "alice/pick", "message": "Download started"}

    _join_download(mgr)
    status = mgr.get_status()
    assert status["state"] == "done"
    assert status["repo_id"] == "alice/pick"
    assert status["error"] is None
    assert invalidated == ["alice/pick"]
    # The dataset now lives in the flat layout, so it's available locally.
    assert ds.is_dataset_available_locally("alice/pick")


def test_download_manager_error_surfaces_message(
    tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed fetch lands in state "error" with the message and error set, and
    leaves no half-written dataset dir behind."""
    from makerlab import datasets as ds

    def _boom(repo_id, repo_type, local_dir):  # noqa: ARG001
        raise RuntimeError("network exploded")

    monkeypatch.setattr(ds, "snapshot_download", _boom)

    mgr = _dataset_download_manager()
    mgr.start("alice/pick")
    _join_download(mgr)

    status = mgr.get_status()
    assert status["state"] == "error"
    assert "network exploded" in status["message"]
    assert status["error"] == "network exploded"
    assert not (tmp_lerobot_home / "alice" / "pick").exists()


def test_download_manager_rejects_concurrent_start() -> None:
    """A second start while one is running is refused (409-mapped by the route),
    naming the repo already downloading; the running download is untouched."""
    mgr = _dataset_download_manager()
    mgr.state = "running"
    mgr.repo_id = "alice/first"

    result = mgr.start("bob/second")
    assert result["started"] is False
    assert "already running" in result["message"]
    assert "alice/first" in result["message"]
    assert mgr.repo_id == "alice/first"


def test_download_endpoint_rejects_bad_repo_id(client: TestClient) -> None:
    resp = client.post("/datasets/download", json={"repo_id": "not-a-repo-id"})
    assert resp.status_code == 400
    assert isinstance(resp.json()["detail"], str)


def test_download_endpoint_409_when_running(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import makerlab.datasets as ds

    monkeypatch.setattr(ds.download_manager, "state", "running")
    monkeypatch.setattr(ds.download_manager, "repo_id", "alice/busy")
    resp = client.post("/datasets/download", json={"repo_id": "bob/other"})
    assert resp.status_code == 409
    assert "alice/busy" in resp.json()["detail"]


def test_download_status_endpoint_idle(client: TestClient) -> None:
    resp = client.get("/datasets/download-status")
    assert resp.status_code == 200
    assert resp.json()["state"] in {"idle", "running", "done", "error"}


# ---------------------------------------------------------------------------
# import_local_dataset — copy a local LeRobot dataset folder into the cache.
# ---------------------------------------------------------------------------


def _make_source_dataset(root: Path, name: str, episodes: int = 2) -> Path:
    """A LeRobot dataset dir OUTSIDE the cache, to import FROM."""
    d = root / name
    (d / "meta").mkdir(parents=True)
    (d / "meta" / "info.json").write_text(json.dumps({"total_episodes": episodes}))
    (d / "data").mkdir()
    (d / "data" / "chunk.parquet").write_bytes(b"payload")
    return d


def test_import_local_dataset_copies_into_cache(tmp_lerobot_home: Path, tmp_path: Path) -> None:
    from makerlab.datasets import import_local_dataset

    src = _make_source_dataset(tmp_path / "external", "my_ds")
    result = import_local_dataset(str(src))
    assert result == {"repo_id": "my_ds"}

    dst = tmp_lerobot_home / "my_ds"
    assert (dst / "meta" / "info.json").is_file()
    assert (dst / "data" / "chunk.parquet").read_bytes() == b"payload"
    # COPY, not move — the source is left intact.
    assert (src / "meta" / "info.json").is_file()


def test_import_local_dataset_honors_explicit_namespaced_name(tmp_lerobot_home: Path, tmp_path: Path) -> None:
    from makerlab.datasets import import_local_dataset

    src = _make_source_dataset(tmp_path / "external", "raw")
    result = import_local_dataset(str(src), name="team/renamed")
    assert result == {"repo_id": "team/renamed"}
    assert (tmp_lerobot_home / "team" / "renamed" / "meta" / "info.json").is_file()


def test_import_local_dataset_404_missing_folder(tmp_lerobot_home: Path) -> None:
    from makerlab.datasets import DatasetImportError, import_local_dataset

    with pytest.raises(DatasetImportError) as ei:
        import_local_dataset("/definitely/not/here")
    assert ei.value.status == 404


def test_import_local_dataset_400_not_a_dataset(tmp_lerobot_home: Path, tmp_path: Path) -> None:
    from makerlab.datasets import DatasetImportError, import_local_dataset

    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(DatasetImportError) as ei:
        import_local_dataset(str(plain))
    assert ei.value.status == 400


def test_import_local_dataset_400_empty_dataset(tmp_lerobot_home: Path, tmp_path: Path) -> None:
    from makerlab.datasets import DatasetImportError, import_local_dataset

    src = _make_source_dataset(tmp_path / "external", "empty", episodes=0)
    with pytest.raises(DatasetImportError) as ei:
        import_local_dataset(str(src))
    assert ei.value.status == 400


def test_import_local_dataset_400_bad_name(tmp_lerobot_home: Path, tmp_path: Path) -> None:
    from makerlab.datasets import DatasetImportError, import_local_dataset

    src = _make_source_dataset(tmp_path / "external", "raw")
    with pytest.raises(DatasetImportError) as ei:
        import_local_dataset(str(src), name="a/b/c")  # too many slashes
    assert ei.value.status == 400


def test_import_local_dataset_409_target_exists(tmp_lerobot_home: Path, tmp_path: Path) -> None:
    from makerlab.datasets import DatasetImportError, import_local_dataset

    _make_dataset(tmp_lerobot_home, "taken", episodes=1)  # already in the cache
    src = _make_source_dataset(tmp_path / "external", "src")
    with pytest.raises(DatasetImportError) as ei:
        import_local_dataset(str(src), name="taken")
    assert ei.value.status == 409


def test_import_endpoint_success(client: TestClient, tmp_lerobot_home: Path, tmp_path: Path) -> None:
    src = _make_source_dataset(tmp_path / "external", "endpoint_ds")
    resp = client.post("/datasets/import", json={"path": str(src)})
    assert resp.status_code == 200
    assert resp.json() == {"repo_id": "endpoint_ds"}
    assert (tmp_lerobot_home / "endpoint_ds" / "meta" / "info.json").is_file()


def test_import_endpoint_404_missing(client: TestClient, tmp_lerobot_home: Path) -> None:
    resp = client.post("/datasets/import", json={"path": "/no/such/folder"})
    assert resp.status_code == 404
    assert isinstance(resp.json()["detail"], str)


# ---------------------------------------------------------------------------
# Hidden datasets — persistent "remove from list" for hub rows.
# ---------------------------------------------------------------------------


@pytest.fixture
def hidden_datasets_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect SAVED_HIDDEN_DATASETS_FILE into a tmp file so hide tests never
    touch the developer's real ~/.cache."""
    from makerlab.utils import config as cfg

    path = tmp_path / "hidden_datasets.json"
    monkeypatch.setattr(cfg, "SAVED_HIDDEN_DATASETS_FILE", str(path))
    return path


def test_hidden_datasets_round_trip(hidden_datasets_file: Path) -> None:
    from makerlab.utils.config import (
        add_hidden_dataset,
        get_hidden_datasets,
        remove_hidden_dataset,
    )

    assert get_hidden_datasets() == set()
    assert add_hidden_dataset("alice/pick")
    assert add_hidden_dataset("alice/pick")  # idempotent re-hide
    assert add_hidden_dataset("bob/place")
    assert get_hidden_datasets() == {"alice/pick", "bob/place"}

    assert remove_hidden_dataset("alice/pick")
    assert not remove_hidden_dataset("alice/pick")  # already unhidden
    assert get_hidden_datasets() == {"bob/place"}
    assert not add_hidden_dataset("")  # blank refused


def test_hidden_datasets_corrupt_file_degrades_to_empty(hidden_datasets_file: Path) -> None:
    from makerlab.utils.config import get_hidden_datasets

    hidden_datasets_file.write_text("{not json")
    assert get_hidden_datasets() == set()
    hidden_datasets_file.write_text(json.dumps({"not": "a list"}))
    assert get_hidden_datasets() == set()


def test_listing_filters_hidden_hub_row(tmp_lerobot_home: Path) -> None:
    from makerlab.datasets import list_all_datasets

    hub_rows = [{"repo_id": "alice/pick", "last_modified": None, "private": False}]
    with (
        patch("makerlab.datasets.list_user_datasets", return_value=hub_rows),
        patch("makerlab.datasets.get_saved_custom_datasets", return_value=[]),
        patch("makerlab.datasets.get_hidden_datasets", return_value={"alice/pick"}),
    ):
        result = list_all_datasets()
    assert result == []


def test_listing_hidden_filter_runs_after_pin_fold(tmp_lerobot_home: Path) -> None:
    """A hidden id can't resurface via a pin — the filter runs AFTER the pin
    fold, so hidden+pinned stays hidden (until the pin ROUTE auto-unhides)."""
    from makerlab.datasets import list_all_datasets

    with (
        patch("makerlab.datasets.list_user_datasets", return_value=[]),
        patch("makerlab.datasets.get_saved_custom_datasets", return_value=["alice/pick"]),
        patch("makerlab.datasets.get_hidden_datasets", return_value={"alice/pick"}),
    ):
        result = list_all_datasets()
    assert result == []


def test_listing_hidden_filter_covers_local_copy(tmp_lerobot_home: Path) -> None:
    """A hidden id with a local (downloaded) copy stays hidden — the filter
    runs after the hub/local merge too."""
    from makerlab.datasets import list_all_datasets

    _make_dataset(tmp_lerobot_home, "alice/pick", episodes=2)
    with (
        patch("makerlab.datasets.list_user_datasets", return_value=[]),
        patch("makerlab.datasets.get_saved_custom_datasets", return_value=[]),
        patch("makerlab.datasets.get_hidden_datasets", return_value={"alice/pick"}),
    ):
        result = list_all_datasets()
    assert result == []


def test_hide_endpoint_rejects_bad_repo_id(client: TestClient, hidden_datasets_file: Path) -> None:
    resp = client.post("/datasets/hide", json={"repo_id": "not-a-repo-id"})
    assert resp.status_code == 400
    assert isinstance(resp.json()["detail"], str)


def test_hide_unhide_endpoints_round_trip(client: TestClient, hidden_datasets_file: Path) -> None:
    from makerlab.utils.config import get_hidden_datasets

    resp = client.post("/datasets/hide", json={"repo_id": "alice/pick"})
    assert resp.status_code == 200
    assert resp.json() == {"success": True, "repo_id": "alice/pick"}
    assert get_hidden_datasets() == {"alice/pick"}

    resp = client.request("DELETE", "/datasets/hide", json={"repo_id": "alice/pick"})
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    assert get_hidden_datasets() == set()


def test_hide_endpoint_invalidates_listing_cache(
    client: TestClient, tmp_lerobot_home: Path, hidden_datasets_file: Path
) -> None:
    """Hiding must drop the cached listing so the row vanishes immediately
    instead of after the TTL."""
    hub_rows = [{"repo_id": "alice/pick", "last_modified": None, "private": False}]
    with (
        patch("makerlab.datasets.list_user_datasets", return_value=hub_rows),
        patch("makerlab.datasets.get_saved_custom_datasets", return_value=[]),
    ):
        first = client.get("/datasets").json()
        assert [d["repo_id"] for d in first] == ["alice/pick"]

        client.post("/datasets/hide", json={"repo_id": "alice/pick"})
        second = client.get("/datasets").json()
    assert second == []


def test_pin_route_auto_unhides(
    client: TestClient,
    tmp_lerobot_home: Path,
    hidden_datasets_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-adding a hidden dataset via POST /datasets/custom removes it from the
    hidden set — otherwise the pin would land behind the filter and never show."""
    from makerlab.utils import config as cfg
    from makerlab.utils.config import add_hidden_dataset, get_hidden_datasets

    # Keep the pin write in tmp too.
    monkeypatch.setattr(cfg, "SAVED_CUSTOM_DATASETS_FILE", str(tmp_path / "pins.json"))

    add_hidden_dataset("alice/pick")
    assert get_hidden_datasets() == {"alice/pick"}

    resp = client.post("/datasets/custom", json={"repo_id": "alice/pick"})
    assert resp.status_code == 200
    assert get_hidden_datasets() == set()


# ---------------------------------------------------------------------------
# Hub dataset summary — the /datasets/info hub fallback (meta/info.json only).
# ---------------------------------------------------------------------------


def _clear_hub_dataset_info_cache() -> None:
    from makerlab import datasets as ds

    with ds._HUB_DATASET_INFO_LOCK:
        ds._HUB_DATASET_INFO_CACHE.clear()


def _write_hub_meta(tmp_path: Path, payload: dict) -> Path:
    p = tmp_path / "info.json"
    p.write_text(json.dumps(payload))
    return p


def test_get_hub_dataset_info_maps_meta(tmp_path: Path) -> None:
    from makerlab import datasets as ds

    _clear_hub_dataset_info_cache()
    meta = _write_hub_meta(
        tmp_path,
        {
            "total_episodes": 12,
            "total_frames": 3600,
            "fps": 30,
            "robot_type": "so101_follower",
            "features": {
                "observation.images.front": {"dtype": "video"},
                "observation.images.wrist": {"dtype": "video"},
                "observation.state": {"dtype": "float32"},
            },
        },
    )
    with (
        patch("makerlab.datasets.hf_hub_offline", return_value=False),
        patch("makerlab.datasets.hf_hub_download", return_value=str(meta)) as dl,
    ):
        row = ds.get_hub_dataset_info("alice/pick")

    dl.assert_called_once_with("alice/pick", filename="meta/info.json", repo_type="dataset")
    assert row == {
        "repo_id": "alice/pick",
        "total_episodes": 12,
        "total_frames": 3600,
        "fps": 30,
        "robot_type": "so101_follower",
        "cameras": ["front", "wrist"],
        "tasks": [],
        "size_bytes": None,
        "source": "hub",
    }


def test_get_hub_dataset_info_excludes_non_video_camera_features(tmp_path: Path) -> None:
    """A camera-prefixed feature that isn't dtype == "video" (e.g. raw stored
    images) has no mp4 chunk for this app's video pipeline to serve, so it's
    excluded from `cameras` — the same field the Hub listing filter and viewer
    gate both key off."""
    from makerlab import datasets as ds

    _clear_hub_dataset_info_cache()
    meta = _write_hub_meta(
        tmp_path,
        {
            "total_episodes": 4,
            "total_frames": 100,
            "fps": 30,
            "features": {
                "observation.images.front": {"dtype": "image"},
                "observation.state": {"dtype": "float32"},
            },
        },
    )
    with (
        patch("makerlab.datasets.hf_hub_offline", return_value=False),
        patch("makerlab.datasets.hf_hub_download", return_value=str(meta)),
    ):
        row = ds.get_hub_dataset_info("alice/image_only")

    assert row is not None
    assert row["cameras"] == []


def test_get_hub_dataset_info_offline_returns_none() -> None:
    from makerlab import datasets as ds

    _clear_hub_dataset_info_cache()
    with patch("makerlab.datasets.hf_hub_offline", return_value=True):
        assert ds.get_hub_dataset_info("alice/pick") is None


def test_get_hub_dataset_info_error_degrades_and_is_not_cached() -> None:
    from makerlab import datasets as ds

    _clear_hub_dataset_info_cache()
    with (
        patch("makerlab.datasets.hf_hub_offline", return_value=False),
        patch("makerlab.datasets.hf_hub_download", side_effect=RuntimeError("hub down")) as dl,
    ):
        assert ds.get_hub_dataset_info("alice/pick") is None
        assert ds.get_hub_dataset_info("alice/pick") is None
    assert dl.call_count == 2  # the degrade is never cached


def test_get_hub_dataset_info_caches_success(tmp_path: Path) -> None:
    from makerlab import datasets as ds

    _clear_hub_dataset_info_cache()
    meta = _write_hub_meta(tmp_path, {"total_episodes": 1, "total_frames": 30, "fps": 30})
    with (
        patch("makerlab.datasets.hf_hub_offline", return_value=False),
        patch("makerlab.datasets.hf_hub_download", return_value=str(meta)) as dl,
    ):
        ds.get_hub_dataset_info("alice/cached")
        ds.get_hub_dataset_info("alice/cached")
        assert dl.call_count == 1
        ds.invalidate_hub_dataset_info("alice/cached")
        ds.get_hub_dataset_info("alice/cached")
        assert dl.call_count == 2


def test_datasets_info_endpoint_hub_fallback(
    client: TestClient, tmp_lerobot_home: Path, tmp_path: Path
) -> None:
    """A dataset with no local copy gets the hub summary (source: 'hub')
    instead of a 404; a repo with neither still 404s."""
    from makerlab import datasets as ds

    _clear_hub_dataset_info_cache()
    meta = _write_hub_meta(tmp_path, {"total_episodes": 5, "total_frames": 150, "fps": 30})
    with (
        patch("makerlab.datasets.hf_hub_offline", return_value=False),
        patch("makerlab.datasets.hf_hub_download", return_value=str(meta)),
    ):
        resp = client.get("/datasets/info", params={"repo_id": "alice/hub_only"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "hub"
    assert body["total_episodes"] == 5
    assert body["size_bytes"] is None

    _clear_hub_dataset_info_cache()
    with patch("makerlab.datasets.hf_hub_offline", return_value=True):
        resp = client.get("/datasets/info", params={"repo_id": "alice/nowhere"})
    assert resp.status_code == 404
    assert ds is not None  # keep the import referenced


def test_get_local_dataset_info_marks_source_local(tmp_lerobot_home: Path) -> None:
    from makerlab.datasets import get_local_dataset_info

    _make_dataset(tmp_lerobot_home, "alice/local_ds", episodes=2)
    info = get_local_dataset_info("alice/local_ds")
    assert info is not None
    assert info["source"] == "local"


# ---------------------------------------------------------------------------
# _fan_out_hub_authors — the OVERALL fan-out deadline actually bounds a hung
# author. The shared HfApi httpx client has timeout=None, so this budget is the
# ONLY timeout in the stack: a blackholed connection must be abandoned (and
# named in a warning) rather than stalling the caller.
# ---------------------------------------------------------------------------


def test_fan_out_hub_authors_bounds_a_hung_author(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the budget shrunk to 0.2s, a fast author returns while a hung author
    (blocked on an Event never set during the call) is abandoned by the deadline:
    the call returns fast, carries ONLY the fast author's result, and logs a
    warning naming the hung author."""
    from makerlab import datasets as ds

    monkeypatch.setattr(ds, "_HUB_FANOUT_TIMEOUT_S", 0.2)

    # Never set DURING the call; released in the finally so the leaked worker
    # thread exits and pytest terminates cleanly.
    release = threading.Event()

    def call(author: str) -> str:
        if author == "fast":
            return f"result-for-{author}"
        release.wait(timeout=30)  # the hung author
        return "late"

    try:
        start = time.monotonic()
        with caplog.at_level(logging.WARNING):
            result = ds._fan_out_hub_authors(["fast", "hung"], call)
        elapsed = time.monotonic() - start

        # Bounded by the 0.2s budget, not the 30s the hung worker would take.
        assert elapsed < 3.0
        # Only the finished author's result survives; the hung one contributes nothing.
        assert result == ["result-for-fast"]
        # The timeout warning names the author that didn't finish.
        timeout_logs = [r.getMessage() for r in caplog.records if "exceeded" in r.getMessage()]
        assert timeout_logs, "expected a fan-out timeout warning"
        assert any("hung" in msg for msg in timeout_logs)
    finally:
        release.set()


def test_fan_out_hub_authors_no_timeout_when_all_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deadline is a ceiling, not a floor: when every author finishes well
    inside the budget, all results are returned in author order."""
    from makerlab import datasets as ds

    monkeypatch.setattr(ds, "_HUB_FANOUT_TIMEOUT_S", 0.5)
    result = ds._fan_out_hub_authors(["a", "b", "c"], lambda author: author.upper())
    assert result == ["A", "B", "C"]
