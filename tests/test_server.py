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
"""Tests for lelab.server — FastAPI app and ConnectionManager."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import lelab.server as server_mod
from lelab.utils import config as cfg

# A browser sends an Accept header that prefers HTML on navigations/hard-reloads.
BROWSER_ACCEPT = {"accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}

REQUIRED_PATHS = {
    "/health",
    "/get-configs",
    "/move-arm",
    "/stop-teleoperation",
    "/teleoperation-status",
    "/joint-positions",
    "/start-recording",
    "/stop-recording",
    "/recording-status",
    "/start-calibration",
    "/stop-calibration",
    "/calibration-status",
    "/datasets",
    "/jobs",
    "/available-ports",
    "/available-cameras",
    "/hf-auth-status",
    "/ws/joint-data",
}


def test_app_exposes_required_endpoints() -> None:
    from lelab.server import app

    paths = {route.path for route in app.routes}
    missing = REQUIRED_PATHS - paths
    assert not missing, f"missing routes: {missing}"


def test_health_endpoint_returns_200(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200


def test_health_endpoint_returns_dict(client: TestClient) -> None:
    response = client.get("/health")
    body = response.json()
    assert isinstance(body, dict)


def test_unknown_route_returns_404(client: TestClient) -> None:
    response = client.get("/this-does-not-exist")
    assert response.status_code == 404


@pytest.mark.parametrize("unsafe_name", ["evil..name", "..config", "back\\door"])
def test_delete_calibration_config_rejects_unsafe_name(client: TestClient, unsafe_name: str) -> None:
    """A config name with path-traversal characters is rejected before any
    filesystem access — distinct from the "not found" path, so the guard is
    proven to fire. The validator also blocks "/" and "\\"."""
    response = client.delete(f"/calibration-configs/teleop/{unsafe_name}")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert "Invalid configuration name" in body["message"]


def test_delete_in_use_calibration_config_unassigns_robots(
    client: TestClient, tmp_lerobot_home, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deleting an in-use config is ALLOWED; the referencing robots are
    unassigned (arm returns to "needs calibration") and reported back."""
    robots_dir = tmp_lerobot_home / "robots"
    robots_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(cfg, "ROBOTS_PATH", str(robots_dir))
    # server.py binds LEADER_CONFIG_PATH at import; repoint it at the tmp dir.
    monkeypatch.setattr(server_mod, "LEADER_CONFIG_PATH", cfg.LEADER_CONFIG_PATH)

    config_file = Path(cfg.LEADER_CONFIG_PATH) / "mycal.json"
    config_file.write_text("{}")
    cfg.save_robot_record("armA", {"mode": "single", "leader_config": "mycal"}, allow_create=True)

    resp = client.delete("/calibration-configs/teleop/mycal")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["unassigned"] == [{"robot": "armA", "fields": ["leader_config"]}]
    assert "armA" in body["message"]
    # The file is gone (this dir IS where lerobot loads calibrations from, so
    # no stale copy can keep working) and the record is unassigned + dirty.
    assert not config_file.exists()
    record = cfg.get_robot_record("armA")
    assert record["leader_config"] == ""
    assert cfg.is_robot_record_clean(record) is False


def test_delete_unused_calibration_config_reports_no_unassignments(
    client: TestClient, tmp_lerobot_home, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server_mod, "LEADER_CONFIG_PATH", cfg.LEADER_CONFIG_PATH)
    config_file = Path(cfg.LEADER_CONFIG_PATH) / "spare.json"
    config_file.write_text("{}")

    resp = client.delete("/calibration-configs/teleop/spare")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["unassigned"] == []
    assert not config_file.exists()


def test_upsert_robot_rejects_same_side_config_conflict(
    client: TestClient, tmp_lerobot_home
) -> None:
    """Assigning one config to both same-side arms of a bimanual robot is a 409."""
    client.post(
        "/robots/bi?create=true",
        json={"mode": "bimanual", "leader_config": "L1", "right_leader_config": "L2"},
    )
    # right leader = left leader -> conflict.
    resp = client.post("/robots/bi", json={"right_leader_config": "L1"})
    assert resp.status_code == 409
    assert "leader" in resp.json()["message"]

    # A non-slot edit (cameras) is never blocked.
    assert client.post("/robots/bi", json={"cameras": []}).status_code == 200


def test_upsert_robot_rejects_shared_port(client: TestClient, tmp_lerobot_home) -> None:
    """Two arms can't share a serial port (each is its own USB device)."""
    client.post("/robots/p?create=true", json={"leader_port": "/dev/a"})
    # follower on the same port as leader -> 409.
    resp = client.post("/robots/p", json={"follower_port": "/dev/a"})
    assert resp.status_code == 409
    assert "/dev/a" in resp.json()["message"]
    # A distinct port is fine.
    assert client.post("/robots/p", json={"follower_port": "/dev/b"}).status_code == 200


def test_upsert_robot_clears_port_with_empty_string(client: TestClient, tmp_lerobot_home) -> None:
    """Posting an empty-string port releases the assignment (disconnect without
    reconnecting), and two cleared ports never count as a shared-port conflict."""
    client.post("/robots/d?create=true", json={"leader_port": "/dev/a", "follower_port": "/dev/b"})

    resp = client.post("/robots/d", json={"leader_port": ""})
    assert resp.status_code == 200
    assert resp.json()["robot"]["leader_port"] == ""

    # Clearing the other arm too must not trip the duplicate-port guard.
    resp = client.post("/robots/d", json={"follower_port": ""})
    assert resp.status_code == 200
    assert resp.json()["robot"]["follower_port"] == ""

    # A cleared port doesn't block re-assigning that port to the other arm.
    assert client.post("/robots/d", json={"leader_port": "/dev/b"}).status_code == 200


@pytest.mark.parametrize("mode", ["single", "bimanual"])
def test_create_robot_accepts_mode(client: TestClient, tmp_lerobot_home, mode: str) -> None:
    """Mode is established at creation for both values."""
    resp = client.post(f"/robots/created_{mode}?create=true", json={"mode": mode})
    assert resp.status_code == 200
    assert resp.json()["robot"]["mode"] == mode


def test_upsert_robot_rejects_mode_change_on_existing_record(client: TestClient, tmp_lerobot_home) -> None:
    """Mode is fixed at creation. A patch that flips the stored mode is a 409;
    creating a new robot is the migration path instead."""
    client.post("/robots/fixed?create=true", json={"mode": "single"})

    resp = client.post("/robots/fixed", json={"mode": "bimanual"})
    assert resp.status_code == 409
    assert "fixed at creation" in resp.json()["message"]
    # The stored mode is untouched by the rejected patch.
    assert client.get("/robots/fixed").json()["robot"]["mode"] == "single"


def test_upsert_robot_allows_same_mode_echo(client: TestClient, tmp_lerobot_home) -> None:
    """Calibration write-backs echo the full record (including its current
    mode); a same-value mode in the body must stay a no-op, not a 409."""
    client.post("/robots/echo?create=true", json={"mode": "bimanual"})

    # Echo the existing mode alongside a real edit — must succeed.
    resp = client.post("/robots/echo", json={"mode": "bimanual", "leader_port": "/dev/a"})
    assert resp.status_code == 200
    robot = resp.json()["robot"]
    assert robot["mode"] == "bimanual"
    assert robot["leader_port"] == "/dev/a"


def _access_record(method: str, path: str, status: int) -> logging.LogRecord:
    """Build a LogRecord shaped like uvicorn.access emits:
    args = (client_addr, method, full_path, http_version, status_code)."""
    return logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg='%s - "%s %s HTTP/%s" %d',
        args=("127.0.0.1:1234", method, path, "1.1", status),
        exc_info=None,
    )


def test_status_poll_access_filter_drops_only_successful_status_gets() -> None:
    """The uvicorn.access filter silences ~2 Hz status polls but keeps errors,
    writes, and every other path."""
    f = server_mod._StatusPollAccessFilter()

    # High-frequency polls with 2xx are dropped (query string ignored).
    assert f.filter(_access_record("GET", "/teleoperation-status", 200)) is False
    assert f.filter(_access_record("GET", "/auto-calibration-status", 200)) is False
    assert f.filter(_access_record("GET", "/jobs?limit=20", 200)) is False

    # Errors on those same paths must still log.
    assert f.filter(_access_record("GET", "/recording-status", 500)) is True
    assert f.filter(_access_record("GET", "/jobs", 404)) is True

    # Writes and non-status paths are untouched.
    assert f.filter(_access_record("POST", "/jobs/training", 201)) is True
    assert f.filter(_access_record("GET", "/health", 200)) is True
    # Subpaths of /jobs (log tails, checkpoints) are NOT silenced.
    assert f.filter(_access_record("GET", "/jobs/abc123/logs", 200)) is True

    # Records that don't look like uvicorn access lines pass through.
    other = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, "plain", None, None)
    assert f.filter(other) is True


def test_policy_optimizer_defaults_reports_availability(client: TestClient) -> None:
    """`available` marks which policy types this lerobot pin can construct.
    act must work everywhere; reward_classifier registers under lerobot's
    rewards registry (not the policy registry) in this pin, so it's out."""
    data = client.get("/policy-optimizer-defaults").json()
    assert set(data["available"]) == set(data["defaults"])
    assert data["available"]["act"] is True
    assert data["defaults"]["act"] is not None
    assert data["available"]["pi0_fast"] is True
    assert data["available"]["reward_classifier"] is False
    assert data["defaults"]["reward_classifier"] is None


@pytest.mark.parametrize("unsafe_name", ["evil..name", "..config", "back\\door"])
def test_download_calibration_config_rejects_unsafe_name(
    client: TestClient, unsafe_name: str
) -> None:
    response = client.get(f"/calibration-configs/teleop/{unsafe_name}/download")
    assert response.status_code == 400
    assert "Invalid configuration name" in response.json()["message"]


def test_download_calibration_config_rejects_bad_device_type(client: TestClient) -> None:
    response = client.get("/calibration-configs/bogus/arm/download")
    assert response.status_code == 400


def test_download_calibration_config_returns_file(
    client: TestClient, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A leader config downloads byte-for-byte as a raw JSON attachment."""
    leader_dir = tmp_path / "leader"
    leader_dir.mkdir()
    (leader_dir / "armA.json").write_text('{"shoulder_pan": {"id": 1}}')
    # server.py binds its own LEADER_CONFIG_PATH at import — patch that one.
    monkeypatch.setattr("lelab.server.LEADER_CONFIG_PATH", str(leader_dir))

    response = client.get("/calibration-configs/teleop/armA/download")
    assert response.status_code == 200
    assert response.headers["content-disposition"] == 'attachment; filename="armA.json"'
    assert response.json() == {"shoulder_pan": {"id": 1}}


def test_download_calibration_config_accepts_dot_json_suffix(
    client: TestClient, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Robot records store config names with the .json extension; passing that
    form must resolve to the file, not "<name>.json.json"."""
    leader_dir = tmp_path / "leader"
    leader_dir.mkdir()
    (leader_dir / "so101.json").write_text('{"shoulder_pan": {"id": 1}}')
    monkeypatch.setattr("lelab.server.LEADER_CONFIG_PATH", str(leader_dir))

    response = client.get("/calibration-configs/teleop/so101.json/download")
    assert response.status_code == 200
    assert response.headers["content-disposition"] == 'attachment; filename="so101.json"'


def test_download_calibration_config_missing_returns_404(
    client: TestClient, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    leader_dir = tmp_path / "leader"
    leader_dir.mkdir()
    monkeypatch.setattr("lelab.server.LEADER_CONFIG_PATH", str(leader_dir))

    response = client.get("/calibration-configs/teleop/nope/download")
    assert response.status_code == 404


_GOOD_CALIBRATION = {
    "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 1927, "range_min": 741, "range_max": 3472},
}


def test_upload_calibration_config_rejects_bad_device_type(client: TestClient) -> None:
    response = client.post(
        "/calibration-configs/bogus/upload", json={"name": "x", "data": _GOOD_CALIBRATION}
    )
    assert response.status_code == 400


def test_upload_calibration_config_rejects_malformed_data(client: TestClient) -> None:
    response = client.post(
        "/calibration-configs/teleop/upload", json={"name": "x", "data": {"m": {"id": 1}}}
    )
    assert response.status_code == 400
    assert "missing" in response.json()["message"]


def test_upload_calibration_config_writes_then_409_on_collision(
    client: TestClient, tmp_lerobot_home
) -> None:
    """First upload writes; a second under the same name is rejected (no overwrite)."""
    first = client.post(
        "/calibration-configs/teleop/upload", json={"name": "armA", "data": _GOOD_CALIBRATION}
    )
    assert first.status_code == 200
    assert first.json()["name"] == "armA"

    second = client.post(
        "/calibration-configs/teleop/upload", json={"name": "armA", "data": _GOOD_CALIBRATION}
    )
    assert second.status_code == 409


def _spa_mounted(client: TestClient) -> bool:
    return any(getattr(route, "name", None) == "frontend" for route in client.app.routes)


def test_spa_deep_link_serves_index_html(client: TestClient) -> None:
    """A browser hard-reload of a client-side route returns the SPA shell, not a 404."""
    if not _spa_mounted(client):
        pytest.skip("frontend/dist not built; SPA not mounted")
    response = client.get("/recording", headers=BROWSER_ACCEPT)
    assert response.status_code == 200
    assert response.text.lstrip().lower().startswith("<!doctype html")


def test_spa_fallback_does_not_mask_api_404(client: TestClient) -> None:
    """Non-HTML clients (XHR, curl, API typos) still get a real 404, not the SPA shell."""
    response = client.get("/recording", headers={"accept": "application/json"})
    assert response.status_code == 404


def test_spa_fallback_respects_explicit_html_refusal(client: TestClient) -> None:
    """`text/html;q=0` is an explicit refusal — it must not get the SPA shell."""
    response = client.get("/recording", headers={"accept": "application/json,text/html;q=0"})
    assert response.status_code == 404


@pytest.mark.parametrize(
    ("accept", "expected"),
    [
        ("text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", True),
        ("text/html", True),
        ("text/html;q=0.5", True),
        ("application/json", False),
        ("*/*", False),
        ("", False),
        ("text/html;q=0", False),
        ("application/json,text/html;q=0", False),
        ("text/html;q=bogus", False),
    ],
)
def test_accepts_html(accept: str, expected: bool) -> None:
    from lelab.server import _accepts_html

    assert _accepts_html(accept) is expected


def test_connection_manager_tracks_connect_and_disconnect() -> None:
    from lelab.server import ConnectionManager

    mgr = ConnectionManager()
    fake_ws = MagicMock()
    fake_ws.accept = AsyncMock()

    import asyncio

    asyncio.run(mgr.connect(fake_ws))
    assert fake_ws in mgr.active_connections

    mgr.disconnect(fake_ws)
    assert fake_ws not in mgr.active_connections


def test_connection_manager_broadcast_sync_does_not_block_without_loop() -> None:
    from lelab.server import ConnectionManager

    mgr = ConnectionManager()
    # Should enqueue without raising even if there are no consumers.
    mgr.broadcast_joint_data_sync({"shoulder_pan.pos": 1.0})


def _install_fake_pygrabber(monkeypatch: pytest.MonkeyPatch, filter_graph_cls) -> None:
    import sys
    import types

    module = types.ModuleType("pygrabber.dshow_graph")
    module.FilterGraph = filter_graph_cls
    monkeypatch.setitem(sys.modules, "pygrabber", types.ModuleType("pygrabber"))
    monkeypatch.setitem(sys.modules, "pygrabber.dshow_graph", module)


def test_windows_cameras_uses_real_directshow_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Windows path returns pygrabber's real device names in index order so
    the frontend can match each camera to its browser deviceId (issues #12/#16).
    """
    from lelab import server

    class _FakeGraph:
        def get_input_devices(self) -> list[str]:
            return ["USB2.0_CAM1", "ASUS FHD webcam"]

    _install_fake_pygrabber(monkeypatch, _FakeGraph)

    assert server._windows_cameras() == [
        {"index": 0, "name": "USB2.0_CAM1", "available": True},
        {"index": 1, "name": "ASUS FHD webcam", "available": True},
    ]


def test_windows_cameras_falls_back_when_pygrabber_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If pygrabber is missing or its COM init fails, enumeration degrades to the
    generic cv2 probe instead of erroring."""
    from lelab import server

    class _BoomGraph:
        def __init__(self) -> None:
            raise RuntimeError("DirectShow/COM unavailable")

    _install_fake_pygrabber(monkeypatch, _BoomGraph)
    sentinel = [{"index": 0, "name": "Camera 0", "available": True}]
    monkeypatch.setattr(server, "_generic_cv2_cameras", lambda backend: sentinel)

    assert server._windows_cameras() == sentinel


def test_v4l2_camera_name_reads_sysfs(monkeypatch: pytest.MonkeyPatch) -> None:
    import io

    from lelab import server

    monkeypatch.setattr("builtins.open", lambda *a, **k: io.StringIO("HD Pro Webcam C920\n"))
    assert server._v4l2_camera_name(0) == "HD Pro Webcam C920"


def test_v4l2_camera_name_returns_none_when_missing() -> None:
    from lelab import server

    # No such sysfs node (also the case on non-Linux): graceful None, not error.
    assert server._v4l2_camera_name(999999) is None


def test_import_model_route_returns_record(client, monkeypatch) -> None:
    from lelab import server

    fake = {
        "id": "act_imported_x",
        "name": "Imported · model",
        "state": "done",
        "config": {"dataset_repo_id": "(imported)", "policy_type": "act"},
        "output_dir": "/tmp/model",
        "started_at": 1.0,
        "ended_at": 1.0,
        "runner": "imported",
        "hf_repo_id": None,
    }
    from lelab.jobs import JobRecord

    # No pre-existing entry for this source → fresh 201 path.
    monkeypatch.setattr(server.job_registry, "find_imported", lambda source: None)
    monkeypatch.setattr(
        server.job_registry,
        "register_imported",
        lambda source, name=None: JobRecord(**fake),
    )
    resp = client.post("/jobs/import", json={"source": "/tmp/model"})
    assert resp.status_code == 201
    assert resp.json()["runner"] == "imported"
    assert "already_imported" not in resp.json()


def test_import_model_route_flags_duplicate_with_200(client, monkeypatch) -> None:
    """Re-importing an already-registered source returns the EXISTING record
    with already_imported=true and a 200 (not 201)."""
    from lelab import server
    from lelab.jobs import JobRecord

    existing = JobRecord(
        id="act_imported_x",
        name="Imported · model",
        display_name="my alias",
        state="done",
        config={"dataset_repo_id": "(imported)", "policy_type": "act"},
        output_dir="/tmp/model",
        started_at=1.0,
        ended_at=1.0,
        runner="imported",
    )
    monkeypatch.setattr(server.job_registry, "find_imported", lambda source: existing)
    monkeypatch.setattr(
        server.job_registry,
        "register_imported",
        lambda source, name=None: existing,
    )
    resp = client.post("/jobs/import", json={"source": "/tmp/model"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["already_imported"] is True
    assert body["id"] == "act_imported_x"
    assert body["display_name"] == "my alias"  # alias preserved on re-import


def test_import_model_route_maps_value_error_to_400(client, monkeypatch) -> None:
    from lelab import server

    def boom(source, name=None):
        raise ValueError("No usable model at '/tmp/x'")

    monkeypatch.setattr(server.job_registry, "find_imported", lambda source: None)
    monkeypatch.setattr(server.job_registry, "register_imported", boom)
    resp = client.post("/jobs/import", json={"source": "/tmp/x"})
    assert resp.status_code == 400
    assert "No usable model" in resp.json()["detail"]


def test_rename_job_route_returns_updated_record(client, monkeypatch) -> None:
    from lelab import server
    from lelab.jobs import JobRecord

    fake = {
        "id": "act_ds_x",
        "name": "ACT · user/ds",
        "display_name": "my run",
        "state": "done",
        "config": {"dataset_repo_id": "user/ds", "policy_type": "act"},
        "output_dir": "/tmp/run",
        "started_at": 1.0,
    }
    seen = {}

    def fake_rename(job_id, new_name):
        seen["args"] = (job_id, new_name)
        return JobRecord(**fake)

    monkeypatch.setattr(server.job_registry, "rename", fake_rename)
    resp = client.post("/jobs/act_ds_x/rename", json={"new_name": "my run"})
    assert resp.status_code == 200
    assert resp.json()["display_name"] == "my run"
    assert seen["args"] == ("act_ds_x", "my run")


def test_rename_job_route_maps_not_found_to_404(client, monkeypatch) -> None:
    from lelab import server
    from lelab.jobs import JobNotFoundError

    def boom(job_id, new_name):
        raise JobNotFoundError(job_id)

    monkeypatch.setattr(server.job_registry, "rename", boom)
    resp = client.post("/jobs/nope/rename", json={"new_name": "x"})
    assert resp.status_code == 404


def test_rename_job_route_maps_value_error_to_400(client, monkeypatch) -> None:
    from lelab import server

    def boom(job_id, new_name):
        raise ValueError("Display name cannot be empty.")

    monkeypatch.setattr(server.job_registry, "rename", boom)
    resp = client.post("/jobs/act_ds_x/rename", json={"new_name": "   "})
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"]
