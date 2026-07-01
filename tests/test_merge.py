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
"""Tests for lelab.merge — the request guards that run before any subprocess."""

from __future__ import annotations


def test_merge_rejects_fewer_than_two_sources() -> None:
    from lelab.merge import MergeManager, MergeRequest

    mgr = MergeManager()
    res = mgr.start(MergeRequest(source_repo_ids=["a/one"], output_repo_id="a/merged"))
    assert res["started"] is False
    assert "two" in res["message"].lower()
    assert mgr.state == "idle"  # no subprocess spawned


def test_merge_rejects_output_matching_a_source() -> None:
    from lelab.merge import MergeManager, MergeRequest

    mgr = MergeManager()
    res = mgr.start(
        MergeRequest(source_repo_ids=["a/one", "a/two"], output_repo_id="a/one")
    )
    assert res["started"] is False
    assert mgr.state == "idle"


def test_merge_rejects_blank_output() -> None:
    from lelab.merge import MergeManager, MergeRequest

    mgr = MergeManager()
    res = mgr.start(
        MergeRequest(source_repo_ids=["a/one", "a/two"], output_repo_id="  ")
    )
    assert res["started"] is False
    assert mgr.state == "idle"


def test_merge_status_shape_when_idle() -> None:
    from lelab.merge import MergeManager

    status = MergeManager().get_status()
    assert status["state"] == "idle"
    assert status["output_repo_id"] is None
    assert status["logs"] == []
