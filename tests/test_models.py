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


def _model(repo_id: str, *, last_modified=None, private=False, tags=None):
    """A stand-in for a huggingface_hub ModelInfo, matching the attributes
    list_user_models reads. `tags` left as the sentinel means "attribute is
    None" (Hub returns None when tags aren't expanded/present)."""
    m = MagicMock()
    m.id = repo_id
    m.last_modified = last_modified
    m.private = private
    m.tags = tags
    return m


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
        "private": True,
        "lerobot": True,
        "cloud_run": False,
    }


# --- Route -----------------------------------------------------------------


def test_models_route_unauthenticated(client: TestClient) -> None:
    with patch("makerlab.server.cached_whoami", return_value=None):
        resp = client.get("/models")
    assert resp.status_code == 200
    assert resp.json() == {"status": "success", "authenticated": False, "models": []}


def test_models_route_authenticated(client: TestClient) -> None:
    fake_models = [
        {
            "repo_id": "alice/policy_a",
            "last_modified": "2026-01-01T00:00:00",
            "private": False,
            "lerobot": True,
            "cloud_run": False,
        }
    ]
    with (
        patch("makerlab.server.cached_whoami", return_value={"name": "alice", "orgs": []}),
        patch("makerlab.server.list_user_models", return_value=fake_models),
    ):
        resp = client.get("/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["authenticated"] is True
    assert body["models"] == fake_models
