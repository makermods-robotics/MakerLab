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
"""Tests for makerlab.models — the user's Hub model-repo listing."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from huggingface_hub.errors import HfHubHTTPError


def _model(repo_id: str, *, last_modified=None, created_at=None, private=False, tags=None):
    """A stand-in for a huggingface_hub ModelInfo, matching the attributes
    list_user_models reads. `tags` left as the sentinel means "attribute is
    None" (Hub returns None when tags aren't expanded/present)."""
    m = MagicMock()
    m.id = repo_id
    m.last_modified = last_modified
    m.created_at = created_at
    m.private = private
    m.tags = tags
    return m


def _job(
    job_id: str,
    *,
    runner="local",
    name="run",
    display_name=None,
    output_dir="/tmp/out",
    hf_repo_id=None,
    checkpoint_count=1,
    state="done",
    started_at=1_700_000_000.0,
    ended_at=None,
):
    """Duck-typed JobRecord carrying only the attributes list_all_models reads."""
    r = MagicMock()
    r.id = job_id
    r.runner = runner
    r.name = name
    r.display_name = display_name
    r.output_dir = output_dir
    r.hf_repo_id = hf_repo_id
    r.checkpoint_count = checkpoint_count
    r.state = state
    r.started_at = started_at
    r.ended_at = ended_at
    return r


def _dt(iso: str) -> datetime:
    return datetime.fromisoformat(iso)


def test_returns_empty_when_not_logged_in() -> None:
    from makerlab.models import list_user_models

    with patch("makerlab.models.cached_whoami", return_value=None):
        assert list_user_models() == []


def test_dedupes_across_user_and_org_authors() -> None:
    """A repo surfaced under both the user and an org author is listed once."""
    from makerlab.models import list_user_models

    fake_api = MagicMock()

    def _list_models(author, **kwargs):
        if author == "alice":
            return [_model("alice/policy_a", last_modified=_dt("2026-01-01T00:00:00"))]
        if author == "acme":
            # The shared repo appears again, plus an org-only one.
            return [
                _model("alice/policy_a", last_modified=_dt("2026-01-01T00:00:00")),
                _model("acme/policy_b", last_modified=_dt("2026-02-01T00:00:00")),
            ]
        return []

    fake_api.list_models.side_effect = _list_models
    whoami = {"name": "alice", "orgs": [{"name": "acme"}]}
    with (
        patch("makerlab.models.cached_whoami", return_value=whoami),
        patch("makerlab.models.shared_hf_api", return_value=fake_api),
    ):
        result = list_user_models()

    repo_ids = [m["repo_id"] for m in result]
    assert repo_ids.count("alice/policy_a") == 1
    assert set(repo_ids) == {"alice/policy_a", "acme/policy_b"}


def test_lerobot_tag_detection_present_absent_and_none() -> None:
    from makerlab.models import list_user_models

    fake_api = MagicMock()
    fake_api.list_models.return_value = [
        _model("alice/tagged", tags=["lerobot", "robotics"]),
        _model("alice/untagged", tags=["robotics"]),
        _model("alice/no_tags", tags=None),
    ]
    whoami = {"name": "alice", "orgs": []}
    with (
        patch("makerlab.models.cached_whoami", return_value=whoami),
        patch("makerlab.models.shared_hf_api", return_value=fake_api),
    ):
        result = list_user_models()

    by_id = {m["repo_id"]: m for m in result}
    assert by_id["alice/tagged"]["lerobot"] is True
    assert by_id["alice/untagged"]["lerobot"] is False
    # tags=None must not raise and must read as not-lerobot.
    assert by_id["alice/no_tags"]["lerobot"] is False


def test_cloud_run_regex_matches_run_repo_name_only() -> None:
    from makerlab.models import list_user_models

    fake_api = MagicMock()
    fake_api.list_models.return_value = [
        _model("alice/act_makermods_pickcube_2026-07-01_12-30-00"),
        _model("alice/my_cool_policy"),
    ]
    whoami = {"name": "alice", "orgs": []}
    with (
        patch("makerlab.models.cached_whoami", return_value=whoami),
        patch("makerlab.models.shared_hf_api", return_value=fake_api),
    ):
        result = list_user_models()

    by_id = {m["repo_id"]: m for m in result}
    assert by_id["alice/act_makermods_pickcube_2026-07-01_12-30-00"]["cloud_run"] is True
    assert by_id["alice/my_cool_policy"]["cloud_run"] is False


def test_sort_by_last_modified_desc_with_none_last() -> None:
    from makerlab.models import list_user_models

    fake_api = MagicMock()
    fake_api.list_models.return_value = [
        _model("alice/older", last_modified=_dt("2026-01-01T00:00:00")),
        _model("alice/no_date", last_modified=None),
        _model("alice/newer", last_modified=_dt("2026-03-01T00:00:00")),
    ]
    whoami = {"name": "alice", "orgs": []}
    with (
        patch("makerlab.models.cached_whoami", return_value=whoami),
        patch("makerlab.models.shared_hf_api", return_value=fake_api),
    ):
        result = list_user_models()

    assert [m["repo_id"] for m in result] == ["alice/newer", "alice/older", "alice/no_date"]
    assert result[-1]["last_modified"] is None


def test_per_author_http_error_swallowed_others_still_list() -> None:
    """A failing author is logged and skipped; the remaining authors list."""
    from makerlab.models import list_user_models

    fake_api = MagicMock()

    def _list_models(author, **kwargs):
        if author == "alice":
            raise HfHubHTTPError("boom", response=MagicMock(status_code=403))
        return [_model("acme/policy_b", last_modified=_dt("2026-02-01T00:00:00"))]

    fake_api.list_models.side_effect = _list_models
    whoami = {"name": "alice", "orgs": [{"name": "acme"}]}
    with (
        patch("makerlab.models.cached_whoami", return_value=whoami),
        patch("makerlab.models.shared_hf_api", return_value=fake_api),
    ):
        result = list_user_models()

    assert [m["repo_id"] for m in result] == ["acme/policy_b"]


def test_entry_shape_and_types() -> None:
    """Each entry carries exactly the frozen contract keys with correct types."""
    from makerlab.models import list_user_models

    fake_api = MagicMock()
    fake_api.list_models.return_value = [
        _model(
            "alice/policy_a",
            last_modified=_dt("2026-01-01T09:00:00"),
            private=True,
            tags=["lerobot"],
        ),
    ]
    whoami = {"name": "alice", "orgs": []}
    with (
        patch("makerlab.models.cached_whoami", return_value=whoami),
        patch("makerlab.models.shared_hf_api", return_value=fake_api),
    ):
        (entry,) = list_user_models()

    assert entry == {
        "repo_id": "alice/policy_a",
        "last_modified": "2026-01-01T09:00:00",
        "created_at": None,
        "private": True,
        "lerobot": True,
        "cloud_run": False,
        "source": "hub",
    }


# --- list_all_models (hub + local merge) -------------------------------------


def _all_models_with(records, hub=()):
    from makerlab.models import list_all_models

    with patch("makerlab.models.list_user_models", return_value=list(hub)):
        return list_all_models(records)


def test_local_run_with_checkpoints_included() -> None:
    result = _all_models_with([_job("job1", runner="local", name="ACT · alice/cubes")])
    assert len(result) == 1
    entry = result[0]
    assert entry["repo_id"] == "ACT · alice/cubes"
    assert entry["job_id"] == "job1"
    assert entry["source"] == "local"
    assert entry["private"] is True
    assert entry["created_at"] is not None


def test_local_run_without_checkpoints_excluded() -> None:
    assert _all_models_with([_job("job1", runner="local", checkpoint_count=0)]) == []


def test_running_local_job_excluded() -> None:
    """A still-training run isn't a browsable model — it lives in the jobs
    list; offering library run/fine-tune on it would collide with the session."""
    assert _all_models_with([_job("job1", runner="local", state="running")]) == []


def test_hub_import_pointer_excluded() -> None:
    """An imported record with hf_repo_id has no local files — the hub listing
    already carries the repo, so it must not appear as a local entry."""
    records = [_job("job1", runner="imported", output_dir="", hf_repo_id="alice/policy_a")]
    assert _all_models_with(records) == []


def test_local_dir_import_included() -> None:
    records = [_job("job1", runner="imported", output_dir="/models/act", name="Imported · act")]
    result = _all_models_with(records)
    assert [m["repo_id"] for m in result] == ["Imported · act"]
    assert result[0]["source"] == "local"


def test_cloud_run_record_excluded() -> None:
    assert _all_models_with([_job("job1", runner="hf_cloud", hf_repo_id="alice/run")]) == []


def test_display_name_wins_over_name() -> None:
    records = [_job("job1", runner="local", name="ACT · raw", display_name="my nice run")]
    assert _all_models_with(records)[0]["repo_id"] == "my nice run"


def test_merged_ordering_newest_added_first() -> None:
    hub = [
        {
            "repo_id": "alice/old_hub",
            "last_modified": "2026-05-01T00:00:00",
            "created_at": "2026-01-01T00:00:00",
            "private": False,
            "lerobot": True,
            "cloud_run": False,
            "source": "hub",
        },
        {
            "repo_id": "alice/no_created",
            "last_modified": "2026-03-01T00:00:00",
            "created_at": None,
            "private": False,
            "lerobot": True,
            "cloud_run": False,
            "source": "hub",
        },
    ]
    # started_at 2026-06-01T00:00:00Z
    local = _job("job1", runner="local", name="newest local", started_at=1_780_272_000.0)
    result = _all_models_with([local], hub=hub)
    assert [m["repo_id"] for m in result] == [
        "newest local",  # created 2026-06
        "alice/no_created",  # falls back to last_modified 2026-03
        "alice/old_hub",  # created 2026-01
    ]


# --- Route -----------------------------------------------------------------


def test_models_route_unauthenticated_still_lists_local(client: TestClient) -> None:
    """Signed out, the route still returns local models — they don't need the
    Hub. (list_user_models itself degrades to empty without a login.)"""
    local_entry = {
        "repo_id": "my local run",
        "job_id": "job1",
        "last_modified": "2026-07-01T00:00:00+00:00",
        "created_at": "2026-07-01T00:00:00+00:00",
        "private": True,
        "lerobot": False,
        "cloud_run": False,
        "source": "local",
    }
    with (
        patch("makerlab.server.cached_whoami", return_value=None),
        patch("makerlab.server.list_all_models", return_value=[local_entry]),
    ):
        resp = client.get("/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["authenticated"] is False
    assert body["models"] == [local_entry]


def test_tracked_hub_import_gets_job_id() -> None:
    """A hub repo already imported as a pointer job keeps source='hub' but
    carries the pointer's job_id so run/fine-tune skip the re-import."""
    hub = [
        {
            "repo_id": "alice/policy_a",
            "last_modified": "2026-01-01T00:00:00",
            "created_at": "2026-01-01T00:00:00",
            "private": False,
            "lerobot": True,
            "cloud_run": False,
            "source": "hub",
        }
    ]
    records = [_job("job9", runner="imported", output_dir="", hf_repo_id="alice/policy_a")]
    (entry,) = _all_models_with(records, hub=hub)
    assert entry["source"] == "hub"
    assert entry["job_id"] == "job9"


def test_models_route_authenticated(client: TestClient) -> None:
    fake_models = [
        {
            "repo_id": "alice/policy_a",
            "last_modified": "2026-01-01T00:00:00",
            "created_at": "2025-12-01T00:00:00",
            "private": False,
            "lerobot": True,
            "cloud_run": False,
            "source": "hub",
        }
    ]
    with (
        patch("makerlab.server.cached_whoami", return_value={"name": "alice", "orgs": []}),
        patch("makerlab.server.list_all_models", return_value=fake_models),
    ):
        resp = client.get("/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["authenticated"] is True
    assert body["models"] == fake_models
