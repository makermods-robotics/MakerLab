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
"""Tests for lelab.datasets — local cache walk and merge logic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
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

    from lelab.datasets import list_local_datasets

    shutil.rmtree(tmp_lerobot_home)
    assert list_local_datasets() == []


def test_list_local_datasets_finds_top_level_dataset(
    tmp_lerobot_home: Path,
) -> None:
    from lelab.datasets import list_local_datasets

    _make_dataset(tmp_lerobot_home, "pusht")
    result = list_local_datasets()
    repo_ids = [d["repo_id"] for d in result]
    assert "pusht" in repo_ids


def test_list_local_datasets_finds_nested_user_dataset(
    tmp_lerobot_home: Path,
) -> None:
    from lelab.datasets import list_local_datasets

    _make_dataset(tmp_lerobot_home, "alice/pusht")
    result = list_local_datasets()
    repo_ids = [d["repo_id"] for d in result]
    assert "alice/pusht" in repo_ids


def test_list_local_datasets_skips_non_dataset_dirs(
    tmp_lerobot_home: Path,
) -> None:
    from lelab.datasets import list_local_datasets

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
    from lelab.datasets import list_local_datasets

    _make_dataset(tmp_lerobot_home, "has_eps", episodes=3)
    _make_dataset(tmp_lerobot_home, "empty_ds", episodes=0)

    repo_ids = [d["repo_id"] for d in list_local_datasets()]
    assert "has_eps" in repo_ids
    assert "empty_ds" not in repo_ids


def test_list_user_datasets_returns_empty_when_not_logged_in(
    tmp_lerobot_home: Path,
) -> None:
    from lelab.datasets import list_user_datasets

    with patch("lelab.datasets.cached_whoami", return_value=None):
        assert list_user_datasets() == []


def test_list_all_datasets_merges_hub_and_local(
    tmp_lerobot_home: Path,
) -> None:
    from lelab.datasets import list_all_datasets

    _make_dataset(tmp_lerobot_home, "alice/pusht")

    with patch(
        "lelab.datasets.list_user_datasets",
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
    from lelab.datasets import get_local_dataset_info

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
    from lelab.datasets import get_local_dataset_info

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
    from lelab.datasets import get_local_dataset_info

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
    from lelab.datasets import get_local_dataset_info

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
    from lelab.datasets import get_local_dataset_info

    assert get_local_dataset_info("nobody/nothing") is None


def test_get_local_dataset_info_rejects_path_traversal(
    tmp_lerobot_home: Path,
) -> None:
    from lelab.datasets import get_local_dataset_info

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
