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
"""Tests for lelab.wiggle — gripper wiggle port finder (hardware mocked)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


async def test_wiggle_gripper_rejects_empty_port() -> None:
    from lelab.wiggle import wiggle_gripper

    result = await wiggle_gripper("   ")
    assert result == {"success": False, "message": "No port provided."}


def test_wiggle_endpoint_success(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # Stub the blocking hardware call so the happy path runs without a device.
    monkeypatch.setattr("lelab.wiggle._wiggle_gripper_sync", lambda port: None)

    response = client.post("/wiggle", json={"port": "/dev/fake"})
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert "/dev/fake" in body["message"]


def test_wiggle_endpoint_reports_hardware_failure(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(port: str) -> None:
        raise RuntimeError("no device on port")

    monkeypatch.setattr("lelab.wiggle._wiggle_gripper_sync", boom)

    response = client.post("/wiggle", json={"port": "/dev/fake"})
    # Logical failures stay HTTP 200 with success=False (like other handlers).
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert "no device on port" in body["message"]
