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
import os
import types

from huggingface_hub import HfApi, get_token, login as hf_login, whoami
from huggingface_hub.errors import HfHubHTTPError, LocalTokenNotFoundError

logger = logging.getLogger(__name__)

LOGIN_COMMAND = "hf auth login"


def hf_hub_offline() -> bool:
    """True when huggingface_hub is running in offline mode.

    Set via HF_HUB_OFFLINE (the canonical hub var) or the legacy
    TRANSFORMERS_OFFLINE. In offline mode every Hub write is disabled, so the
    cloud-training flow can't upload a local-only dataset — the UI uses this to
    keep the Start button disabled and say so, instead of failing on submit.
    A "1"/"true"/"yes"/"on" (any case) counts as enabled; anything else (incl.
    unset / "0") is treated as online.
    """
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        val = os.environ.get(var, "").strip().lower()
        if val in ("1", "true", "yes", "on"):
            return True
    return False


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


def handle_hf_login(token: str) -> dict:
    """Validate and persist an HF token pasted from the UI.

    whoami() validates the token; on success, huggingface_hub.login() writes
    it to ~/.cache/huggingface/token (same as `hf auth login`). Subsequent
    get_token() calls then pick it up automatically.
    """
    token = (token or "").strip()
    if not token:
        raise ValueError("Token must not be empty")
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
