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

import logging
import types

from huggingface_hub import (
    HfApi,
    auth_switch,
    get_token,
    login as hf_login,
    logout as hf_logout,
    whoami,
)
from huggingface_hub._login import (
    _get_token_from_environment,
    _get_token_from_file,
    get_stored_tokens,
)
from huggingface_hub.errors import HfHubHTTPError, LocalTokenNotFoundError

logger = logging.getLogger(__name__)

LOGIN_COMMAND = "hf auth login"

# Message returned by switch/logout/login when identity is pinned by the
# HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) environment variable. get_token() gives
# the env var priority over the on-disk store, so mutating the store here would
# have no effect on the active identity — a split-brain risk (publishing to the
# wrong namespace). We refuse rather than fight the env var.
ENV_TOKEN_REFUSAL = (
    "Identity is pinned by the HF_TOKEN environment variable; unset it and "
    "restart the server to manage accounts here."
)

# /whoami-v2 is heavily rate-limited (security). Share one HfApi across the
# app so its in-process whoami cache (cache=True) actually hits — otherwise
# polling endpoints like /jobs/hub would burn the rate limit on every tick.
_WHOAMI_API = HfApi()

# HfApi.run_job() internally calls self.whoami(token=token) WITHOUT cache=True
# to resolve the namespace, which burns the /whoami-v2 rate limit on every
# job submission. Bind a shim on this shared instance that defaults cache=True
# so every HfApi method on _WHOAMI_API (run_job, inspect_job's lazy whoami,
# etc.) hits the in-process cache once a token has been validated.
_orig_whoami = HfApi.whoami


def _whoami_default_cache(self, token=None, *, cache=True):
    return _orig_whoami(self, token=token, cache=cache)


_WHOAMI_API.whoami = types.MethodType(_whoami_default_cache, _WHOAMI_API)


def cached_whoami() -> dict | None:
    """Return cached whoami() for the active HF token, or None if no token.

    Swallows transport errors and returns None — callers treat that as
    "unauthenticated" so the UI degrades gracefully instead of 500ing.
    """
    token = get_token()
    if not token:
        return None
    try:
        return _WHOAMI_API.whoami(token=token, cache=True)
    except Exception as exc:
        logger.info("whoami failed: %s", exc)
        return None


def shared_hf_api() -> HfApi:
    """The shared HfApi used for whoami caching. Reuse it for non-whoami
    calls in the same handler so they share connection pooling, but it's
    the whoami cache that matters."""
    return _WHOAMI_API


def invalidate_whoami_cache() -> None:
    """Drop the cached whoami() result. Call after a token rotation so the
    next caller re-validates against the Hub."""
    _WHOAMI_API._whoami_cache.clear()


def handle_hf_auth_status() -> dict:
    try:
        info = whoami()
        return {
            "authenticated": True,
            "username": info["name"],
            "orgs": [o["name"] for o in info.get("orgs", [])],
            "login_command": LOGIN_COMMAND,
        }
    except (LocalTokenNotFoundError, HfHubHTTPError, OSError) as e:
        logger.info(f"HF auth check: not authenticated ({type(e).__name__})")
        return {
            "authenticated": False,
            "username": None,
            "orgs": [],
            "login_command": LOGIN_COMMAND,
        }


def env_token_active() -> bool:
    """True when the active identity comes from the HF_TOKEN (or legacy
    HUGGING_FACE_HUB_TOKEN) environment variable rather than the on-disk store.

    When true, get_token() ignores the named store entirely, so account
    management here would be a no-op at best and split-brain at worst.
    """
    return _get_token_from_environment() is not None


def _active_token_name(stored: dict[str, str] | None = None) -> str | None:
    """Name of the stored token whose value matches the active on-disk token.

    huggingface_hub keys the store by each token's *displayName* (its label in
    HF settings), not by username. The active token is a bare string at
    HF_TOKEN_PATH; we match it back to a store entry by value. Returns None if
    no on-disk token is set or it isn't in the store (e.g. env-var identity).
    """
    if stored is None:
        stored = get_stored_tokens()
    active = _get_token_from_file()
    if not active:
        return None
    for name, tok in stored.items():
        if tok == active:
            return name
    return None


def handle_hf_accounts() -> dict:
    """List stored HF accounts, which one is active, and whether the active
    identity is pinned by the HF_TOKEN env var.

    Note: account names are HF token *displayNames* shared with the `hf` CLI —
    tokens added via `hf auth login` show up here and vice-versa.
    """
    env = env_token_active()
    stored = get_stored_tokens()
    accounts = sorted(stored.keys())
    active = None if env else _active_token_name(stored)
    return {
        "accounts": accounts,
        "active": active,
        "env_token": env,
    }


def handle_hf_switch(name: str) -> dict:
    """Activate a stored token by name, invalidate the whoami cache, and return
    the fresh auth status. Refuses when HF_TOKEN pins the identity.

    Caveat: already-submitted HF Jobs keep the token they were launched with;
    switching only affects new API calls.
    """
    name = (name or "").strip()
    if not name:
        raise ValueError("Account name must not be empty")
    if env_token_active():
        raise PermissionError(ENV_TOKEN_REFUSAL)
    if name not in get_stored_tokens():
        raise ValueError(f"Account '{name}' not found in the local token store")
    auth_switch(name, add_to_git_credential=False)
    invalidate_whoami_cache()
    return handle_hf_auth_status()


def handle_hf_logout() -> dict:
    """Remove the ACTIVE account from the machine-global token store, then fall
    back to the next stored account (activating it) so the user is rarely
    dumped to fully-signed-out when another identity is available. If no other
    account remains, ends signed-out. Refuses when HF_TOKEN pins the identity.

    Caveat: already-submitted HF Jobs keep the token they were launched with;
    logging out only affects new API calls.
    """
    if env_token_active():
        raise PermissionError(ENV_TOKEN_REFUSAL)
    stored = get_stored_tokens()
    active = _active_token_name(stored)
    if active is None:
        # Nothing on disk is active; nothing to remove. Report current status.
        invalidate_whoami_cache()
        return handle_hf_auth_status()
    # Drop the active account from the store (and its active-token file).
    hf_logout(token_name=active)
    # Fall back to the next remaining stored account, if any, so a user with
    # two accounts who logs out of one lands on the other rather than a signed-
    # out screen (the less surprising outcome).
    remaining = sorted(n for n in get_stored_tokens() if n != active)
    if remaining:
        auth_switch(remaining[0], add_to_git_credential=False)
    invalidate_whoami_cache()
    return handle_hf_auth_status()


def handle_hf_login(token: str) -> dict:
    """Validate and persist an HF token pasted from the UI.

    whoami() validates the token; on success, huggingface_hub.login() stores it
    in the named token store (~/.cache/huggingface/stored_tokens) under the
    token's own displayName — shared with `hf auth login` — and marks it the
    active token. Subsequent get_token() calls then pick it up automatically.

    Refuses when HF_TOKEN pins the identity: storing a token there would not
    change the active identity (env var wins), so we surface that instead of
    silently no-op'ing.
    """
    token = (token or "").strip()
    if not token:
        raise ValueError("Token must not be empty")
    if env_token_active():
        raise PermissionError(ENV_TOKEN_REFUSAL)
    try:
        info = whoami(token=token)
    except HfHubHTTPError as exc:
        raise ValueError(f"Invalid token: {exc}") from exc
    hf_login(token=token, add_to_git_credential=False)
    # The cached whoami was keyed by the previous token (if any); drop it so
    # the next caller validates against the new one.
    invalidate_whoami_cache()
    return {
        "authenticated": True,
        "username": info["name"],
        "orgs": [o["name"] for o in info.get("orgs", [])],
        "login_command": LOGIN_COMMAND,
    }
