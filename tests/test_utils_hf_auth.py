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
"""Tests for lelab.utils.hf_auth — whoami caching and account management."""

from __future__ import annotations

from unittest.mock import patch

import pytest


def test_invalidate_whoami_cache_clears_cached_value() -> None:
    from lelab.utils import hf_auth

    # cached_whoami() delegates to _WHOAMI_API.whoami(cache=True), which stores
    # results in _WHOAMI_API._whoami_cache keyed by token. Patching the whole
    # whoami() method would bypass the cache logic, so we patch _inner_whoami
    # (the actual HTTP call) instead — the real caching code then runs around it.
    # get_token() must return a truthy value so cached_whoami() doesn't short-circuit.
    with (
        patch("lelab.utils.hf_auth.get_token", return_value="hf_fake_token"),
        patch.object(hf_auth._WHOAMI_API, "_inner_whoami", return_value={"name": "alice"}) as spy,
    ):
        # Clear any pre-existing cache entry.
        hf_auth.invalidate_whoami_cache()

        first = hf_auth.cached_whoami()
        second = hf_auth.cached_whoami()
        assert first == {"name": "alice"}
        assert second == {"name": "alice"}
        # Cached: only one upstream call.
        assert spy.call_count == 1

        hf_auth.invalidate_whoami_cache()

        third = hf_auth.cached_whoami()
        assert third == {"name": "alice"}
        # After invalidation, the next call hits whoami again.
        assert spy.call_count == 2


def test_handle_hf_auth_status_returns_dict() -> None:
    from lelab.utils import hf_auth

    # handle_hf_auth_status() calls the module-level whoami() directly.
    with patch("lelab.utils.hf_auth.whoami", return_value={"name": "alice", "orgs": []}):
        hf_auth.invalidate_whoami_cache()
        result = hf_auth.handle_hf_auth_status()
        assert isinstance(result, dict)
        # Real return shape: {"authenticated": bool, "username": ..., "orgs": ..., "login_command": ...}
        assert result["authenticated"] is True
        assert result["username"] == "alice"
        assert "orgs" in result
        assert "login_command" in result


# --- account store: list / switch / logout / login-stores-named ---------------


def test_handle_hf_accounts_lists_stored_and_marks_active() -> None:
    from lelab.utils import hf_auth

    stored = {"alice-token": "hf_alice", "bob-token": "hf_bob"}
    with (
        patch.object(hf_auth, "get_stored_tokens", return_value=stored),
        patch.object(hf_auth, "_get_token_from_environment", return_value=None),
        patch.object(hf_auth, "_get_token_from_file", return_value="hf_bob"),
    ):
        result = hf_auth.handle_hf_accounts()
    assert result["accounts"] == ["alice-token", "bob-token"]  # sorted
    assert result["active"] == "bob-token"
    assert result["env_token"] is False


def test_handle_hf_accounts_reports_env_token_and_no_active() -> None:
    from lelab.utils import hf_auth

    stored = {"alice-token": "hf_alice"}
    with (
        patch.object(hf_auth, "get_stored_tokens", return_value=stored),
        patch.object(hf_auth, "_get_token_from_environment", return_value="hf_env"),
    ):
        result = hf_auth.handle_hf_accounts()
    # When HF_TOKEN pins identity, we don't claim a stored account is active.
    assert result["env_token"] is True
    assert result["active"] is None
    assert result["accounts"] == ["alice-token"]


def test_handle_hf_switch_activates_and_invalidates_cache() -> None:
    from lelab.utils import hf_auth

    stored = {"alice-token": "hf_alice", "bob-token": "hf_bob"}
    with (
        patch.object(hf_auth, "_get_token_from_environment", return_value=None),
        patch.object(hf_auth, "get_stored_tokens", return_value=stored),
        patch.object(hf_auth, "auth_switch") as switch,
        patch.object(hf_auth, "invalidate_whoami_cache") as invalidate,
        patch.object(hf_auth, "handle_hf_auth_status", return_value={"authenticated": True}),
    ):
        result = hf_auth.handle_hf_switch("bob-token")
    switch.assert_called_once_with("bob-token", add_to_git_credential=False)
    invalidate.assert_called_once()
    assert result == {"authenticated": True}


def test_handle_hf_switch_unknown_account_raises() -> None:
    from lelab.utils import hf_auth

    with (
        patch.object(hf_auth, "_get_token_from_environment", return_value=None),
        patch.object(hf_auth, "get_stored_tokens", return_value={"alice-token": "hf_a"}),
        patch.object(hf_auth, "auth_switch") as switch,
        pytest.raises(ValueError),
    ):
        hf_auth.handle_hf_switch("ghost")
    switch.assert_not_called()


def test_handle_hf_switch_refuses_when_env_token_pins_identity() -> None:
    from lelab.utils import hf_auth

    with (
        patch.object(hf_auth, "_get_token_from_environment", return_value="hf_env"),
        patch.object(hf_auth, "auth_switch") as switch,
        pytest.raises(PermissionError),
    ):
        hf_auth.handle_hf_switch("bob-token")
    switch.assert_not_called()


def test_handle_hf_logout_removes_active_and_falls_back_to_next() -> None:
    from lelab.utils import hf_auth

    # Two accounts, alice active. After logout of alice, fall back to bob.
    before = {"alice-token": "hf_alice", "bob-token": "hf_bob"}
    after = {"bob-token": "hf_bob"}
    stored_seq = iter([before, after])
    with (
        patch.object(hf_auth, "_get_token_from_environment", return_value=None),
        patch.object(hf_auth, "_get_token_from_file", return_value="hf_alice"),
        patch.object(hf_auth, "get_stored_tokens", side_effect=lambda: next(stored_seq)),
        patch.object(hf_auth, "hf_logout") as logout,
        patch.object(hf_auth, "auth_switch") as switch,
        patch.object(hf_auth, "invalidate_whoami_cache") as invalidate,
        patch.object(hf_auth, "handle_hf_auth_status", return_value={"authenticated": True}),
    ):
        result = hf_auth.handle_hf_logout()
    logout.assert_called_once_with(token_name="alice-token")
    switch.assert_called_once_with("bob-token", add_to_git_credential=False)
    invalidate.assert_called_once()
    assert result == {"authenticated": True}


def test_handle_hf_logout_no_remaining_ends_signed_out() -> None:
    from lelab.utils import hf_auth

    before = {"alice-token": "hf_alice"}
    after: dict[str, str] = {}
    stored_seq = iter([before, after])
    with (
        patch.object(hf_auth, "_get_token_from_environment", return_value=None),
        patch.object(hf_auth, "_get_token_from_file", return_value="hf_alice"),
        patch.object(hf_auth, "get_stored_tokens", side_effect=lambda: next(stored_seq)),
        patch.object(hf_auth, "hf_logout") as logout,
        patch.object(hf_auth, "auth_switch") as switch,
        patch.object(hf_auth, "invalidate_whoami_cache"),
        patch.object(hf_auth, "handle_hf_auth_status", return_value={"authenticated": False}),
    ):
        result = hf_auth.handle_hf_logout()
    logout.assert_called_once_with(token_name="alice-token")
    switch.assert_not_called()
    assert result == {"authenticated": False}


def test_handle_hf_logout_refuses_when_env_token_pins_identity() -> None:
    from lelab.utils import hf_auth

    with (
        patch.object(hf_auth, "_get_token_from_environment", return_value="hf_env"),
        patch.object(hf_auth, "hf_logout") as logout,
        pytest.raises(PermissionError),
    ):
        hf_auth.handle_hf_logout()
    logout.assert_not_called()


def test_handle_hf_login_stores_named_and_activates() -> None:
    from lelab.utils import hf_auth

    with (
        patch.object(hf_auth, "_get_token_from_environment", return_value=None),
        patch.object(hf_auth, "whoami", return_value={"name": "alice", "orgs": []}),
        patch.object(hf_auth, "hf_login") as login,
        patch.object(hf_auth, "invalidate_whoami_cache") as invalidate,
    ):
        result = hf_auth.handle_hf_login("hf_alice")
    # huggingface_hub.login() is what writes the NAMED store entry (by the
    # token's displayName) and sets it active — we delegate to it.
    login.assert_called_once_with(token="hf_alice", add_to_git_credential=False)
    invalidate.assert_called_once()
    assert result["authenticated"] is True
    assert result["username"] == "alice"


def test_handle_hf_login_refuses_when_env_token_pins_identity() -> None:
    from lelab.utils import hf_auth

    with (
        patch.object(hf_auth, "_get_token_from_environment", return_value="hf_env"),
        patch.object(hf_auth, "hf_login") as login,
        pytest.raises(PermissionError),
    ):
        hf_auth.handle_hf_login("hf_alice")
    login.assert_not_called()
