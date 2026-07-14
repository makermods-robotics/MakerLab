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
"""Tests for makerlab.merge — the request guards that run before any subprocess."""

from __future__ import annotations

import json
from pathlib import Path


def _write_info(
    cache: Path,
    repo_id: str,
    *,
    fps: int = 30,
    cameras: tuple[str, ...] = ("front", "wrist"),
    action_shape: tuple[int, ...] = (6,),
) -> None:
    """Write a minimal ``<cache>/<repo_id>/meta/info.json`` for the helper to read."""
    features: dict = {
        "action": {"dtype": "float32", "shape": list(action_shape)},
        "observation.state": {"dtype": "float32", "shape": list(action_shape)},
    }
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": [480, 640, 3],
        }
    meta_dir = cache / repo_id / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "info.json").write_text(
        json.dumps({"fps": fps, "robot_type": "so101_follower", "features": features})
    )


def test_merge_incompatibility_identical_returns_none(tmp_lerobot_home: Path) -> None:
    from makerlab.merge import _merge_incompatibility

    _write_info(tmp_lerobot_home, "a/one", cameras=("front", "wrist"))
    _write_info(tmp_lerobot_home, "a/two", cameras=("front", "wrist"))
    assert _merge_incompatibility(["a/one", "a/two"]) is None


def test_merge_incompatibility_different_cameras(tmp_lerobot_home: Path) -> None:
    from makerlab.merge import _merge_incompatibility

    _write_info(tmp_lerobot_home, "a/one", cameras=("front", "wrist"))
    _write_info(tmp_lerobot_home, "a/two", cameras=("front", "wrist", "side"))
    msg = _merge_incompatibility(["a/one", "a/two"])
    assert msg is not None
    assert "side" in msg
    assert "camera" in msg.lower()


def test_merge_incompatibility_different_fps(tmp_lerobot_home: Path) -> None:
    from makerlab.merge import _merge_incompatibility

    _write_info(tmp_lerobot_home, "a/one", fps=30)
    _write_info(tmp_lerobot_home, "a/two", fps=50)
    msg = _merge_incompatibility(["a/one", "a/two"])
    assert msg is not None
    assert "30" in msg and "50" in msg
    assert "fps" in msg.lower()


def test_merge_incompatibility_different_feature_shape(tmp_lerobot_home: Path) -> None:
    from makerlab.merge import _merge_incompatibility

    _write_info(tmp_lerobot_home, "a/one", action_shape=(6,))
    _write_info(tmp_lerobot_home, "a/two", action_shape=(7,))
    msg = _merge_incompatibility(["a/one", "a/two"])
    assert msg is not None
    assert "action" in msg


def test_merge_incompatibility_skips_when_not_local(tmp_lerobot_home: Path) -> None:
    from makerlab.merge import _merge_incompatibility

    # Only one source is present locally → can't compare → don't block.
    _write_info(tmp_lerobot_home, "a/one", cameras=("front", "wrist", "side"))
    assert _merge_incompatibility(["a/one", "a/hub-only"]) is None


def test_merge_rejects_fewer_than_two_sources() -> None:
    from makerlab.merge import MergeManager, MergeRequest

    mgr = MergeManager()
    res = mgr.start(MergeRequest(source_repo_ids=["a/one"], output_repo_id="a/merged"))
    assert res["started"] is False
    assert "two" in res["message"].lower()
    assert mgr.state == "idle"  # no subprocess spawned


def test_merge_rejects_output_matching_a_source() -> None:
    from makerlab.merge import MergeManager, MergeRequest

    mgr = MergeManager()
    res = mgr.start(MergeRequest(source_repo_ids=["a/one", "a/two"], output_repo_id="a/one"))
    assert res["started"] is False
    assert mgr.state == "idle"


def test_merge_rejects_blank_output() -> None:
    from makerlab.merge import MergeManager, MergeRequest

    mgr = MergeManager()
    res = mgr.start(MergeRequest(source_repo_ids=["a/one", "a/two"], output_repo_id="  "))
    assert res["started"] is False
    assert mgr.state == "idle"


def test_monitor_survives_non_utf8_replacement_char() -> None:
    """A non-UTF-8 byte from the merge subprocess decodes (via
    errors="replace" on the Popen call in MergeManager.start()) to U+FFFD;
    _monitor's `for line in iter(self.process.stdout.readline, ""):` loop
    must process that line without raising. Without errors="replace" a
    strict decode raises UnicodeDecodeError before any line is read, and
    since nothing then drains the pipe, a child still writing to it deadlocks
    in the subsequent process.wait() (see the Popen kwargs comment).

    Calls _monitor() directly on a manually-wired process double rather than
    going through start()/Popen — this module only tests the pre-subprocess
    guards otherwise."""
    from makerlab.merge import MergeManager

    class _FakeStdout:
        """.readline() double matching text-mode Popen.stdout, yielding a
        line with the replacement char then the "" sentinel that ends
        iter(readline, "")."""

        def __init__(self, lines: list[str]) -> None:
            self._lines = iter(lines)

        def readline(self) -> str:
            return next(self._lines, "")

    class _FakeProcess:
        returncode = 0

        def __init__(self, lines: list[str]) -> None:
            self.stdout = _FakeStdout(lines)

        def wait(self) -> None:
            return None

    mgr = MergeManager()
    mgr.process = _FakeProcess(["bad � byte\n"])

    mgr._monitor()  # must not raise

    status = mgr.get_status()
    assert status["state"] == "done"
    assert any("�" in log["message"] for log in status["logs"])


def test_merge_status_shape_when_idle() -> None:
    from makerlab.merge import MergeManager

    status = MergeManager().get_status()
    assert status["state"] == "idle"
    assert status["output_repo_id"] is None
    assert status["logs"] == []


def _write_dataset_tree(cache: Path, repo_id: str, *, total_episodes: int = 1) -> None:
    """Write a fully-populated local dataset (info.json + the required files)."""
    _write_info(cache, repo_id)
    root = cache / repo_id
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 30,
                "robot_type": "so101_follower",
                "total_episodes": total_episodes,
                "features": {
                    "action": {"dtype": "float32", "shape": [6]},
                    "observation.state": {"dtype": "float32", "shape": [6]},
                    "observation.images.front": {"dtype": "video", "shape": [480, 640, 3]},
                    "observation.images.wrist": {"dtype": "video", "shape": [480, 640, 3]},
                },
            }
        )
    )
    (root / "meta" / "tasks.parquet").write_text("")
    data_file = root / "data" / "chunk-000" / "file-000.parquet"
    data_file.parent.mkdir(parents=True, exist_ok=True)
    data_file.write_text("")
    ep_file = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ep_file.parent.mkdir(parents=True, exist_ok=True)
    ep_file.write_text("")


def test_merge_source_problem_valid_returns_none(tmp_lerobot_home: Path) -> None:
    from makerlab.merge import _merge_source_problem

    _write_dataset_tree(tmp_lerobot_home, "a/one")
    _write_dataset_tree(tmp_lerobot_home, "a/two")
    assert _merge_source_problem(["a/one", "a/two"]) is None


def test_merge_source_problem_missing_tasks_parquet(tmp_lerobot_home: Path) -> None:
    from makerlab.merge import _merge_source_problem

    _write_dataset_tree(tmp_lerobot_home, "a/one")
    _write_dataset_tree(tmp_lerobot_home, "a/two")
    # Corrupt: remove the tasks parquet the metadata references.
    (tmp_lerobot_home / "a/two" / "meta" / "tasks.parquet").unlink()
    msg = _merge_source_problem(["a/one", "a/two"])
    assert msg is not None
    assert "a/two" in msg
    assert "tasks.parquet" in msg
    assert "incomplete" in msg.lower() or "corrupt" in msg.lower()


def test_merge_source_problem_missing_data_parquet(tmp_lerobot_home: Path) -> None:
    from makerlab.merge import _merge_source_problem

    _write_dataset_tree(tmp_lerobot_home, "a/one")
    _write_dataset_tree(tmp_lerobot_home, "a/two")
    # Corrupt: total_episodes>0 but no data parquet on disk.
    (tmp_lerobot_home / "a/two" / "data" / "chunk-000" / "file-000.parquet").unlink()
    msg = _merge_source_problem(["a/one", "a/two"])
    assert msg is not None
    assert "a/two" in msg
    assert "incomplete" in msg.lower() or "corrupt" in msg.lower()


def test_merge_source_problem_not_found_on_hub(tmp_lerobot_home: Path, monkeypatch) -> None:
    from unittest.mock import MagicMock

    from huggingface_hub.utils import RepositoryNotFoundError

    from makerlab import merge

    _write_dataset_tree(tmp_lerobot_home, "a/one")

    def _raise_not_found(self, repo_id):
        raise RepositoryNotFoundError(f"404 for {repo_id}", response=MagicMock())

    monkeypatch.setattr(merge.HfApi, "dataset_info", _raise_not_found)
    msg = merge._merge_source_problem(["a/one", "a/hub-only"])
    assert msg is not None
    assert "a/hub-only" in msg
    assert "found" in msg.lower()
    assert "hub" in msg.lower()


def test_merge_source_problem_offline_does_not_block(tmp_lerobot_home: Path, monkeypatch) -> None:
    from makerlab import merge

    _write_dataset_tree(tmp_lerobot_home, "a/one")

    def _raise_network(self, repo_id):
        # Simulate offline / transient connection failure.
        raise OSError("connection failed")

    monkeypatch.setattr(merge.HfApi, "dataset_info", _raise_network)
    # Offline / transient error → must NOT block the merge.
    assert merge._merge_source_problem(["a/one", "a/hub-only"]) is None
