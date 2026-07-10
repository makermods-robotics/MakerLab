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
"""Listing of the user's Hugging Face model repos for the Model Library UI."""

import logging
import re
from typing import Any

from huggingface_hub.errors import HfHubHTTPError

from .utils.hf_auth import cached_whoami, shared_hf_api

logger = logging.getLogger(__name__)

# A makerlab cloud-training run repo is named "<policy>_<namespace>_<dataset>_<ts>"
# where the trailing "_YYYY-MM-DD_HH-MM-SS" is stamped by _generate_job_id()
# (jobs.py). Same pattern as server.py's /jobs/hub handler — match on the
# timestamp suffix so it stays policy-agnostic. Here it only annotates each
# entry with a `cloud_run` flag; unlike /jobs/hub it does NOT narrow the listing.
_RUN_REPO_RE = re.compile(r"_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")


def list_user_models() -> list[dict[str, Any]]:
    """All model repos owned by the user and their orgs, annotated for the
    library filters.

    Unlike list_user_datasets, this applies NO narrowing filter — every repo the
    author owns is returned. Each entry carries `lerobot` (has the lowercase
    "lerobot" library tag) and `cloud_run` (name matches a makerlab run-repo
    timestamp) flags so the frontend can filter without a second round-trip.
    """
    info = cached_whoami()
    if info is None:
        return []

    authors = [info["name"]] + [o["name"] for o in info.get("orgs", [])]
    api = shared_hf_api()
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for author in authors:
        try:
            for m in api.list_models(
                author=author, limit=200, expand=["lastModified", "private", "tags"]
            ):
                if m.id in seen:
                    continue
                seen.add(m.id)
                tags = getattr(m, "tags", None) or []
                name = m.id.split("/", 1)[-1]
                out.append(
                    {
                        "repo_id": m.id,
                        "last_modified": m.last_modified.isoformat() if m.last_modified else None,
                        "private": bool(getattr(m, "private", False)),
                        "lerobot": "lerobot" in tags,
                        "cloud_run": bool(_RUN_REPO_RE.search(name)),
                    }
                )
        except HfHubHTTPError as e:
            logger.warning(f"list_models({author}) failed: {e}")

    out.sort(key=lambda d: d["last_modified"] or "", reverse=True)
    return out
