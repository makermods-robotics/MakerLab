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
"""Tests for lelab.camera_preview — shared MJPEG previews of backend cameras.

Everything runs against a fake cv2.VideoCapture: no real camera is ever opened
(on macOS a real open would pop a permission dialog and stall the run).
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

import lelab.camera_enumeration as camera_enumeration
import lelab.camera_preview as camera_preview
import lelab.record as record
import lelab.server as server_mod
import lelab.teleoperate as teleoperate
from lelab.camera_preview import CameraOpenError, CameraPreviewManager


class FakeVideoCapture:
    """cv2.VideoCapture double: serves synthetic frames, records release()."""

    def __init__(self, index: int, backend: int | None = None) -> None:
        self.index = index
        self.backend = backend
        self.opened = True
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 — cv2's camelCase API
        return self.opened

    def read(self):
        if not self.opened:
            return False, None
        return True, np.zeros((8, 8, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True
        self.opened = False


class FailingVideoCapture(FakeVideoCapture):
    """A capture whose device can't be opened (unplugged / held elsewhere)."""

    def __init__(self, index: int, backend: int | None = None) -> None:
        super().__init__(index, backend)
        self.opened = False


class ScriptedVideoCapture(FakeVideoCapture):
    """A capture with a scripted sequence of read() outcomes.

    ``results`` is a list of bools consumed one per read(); once exhausted,
    read() falls back to ``self.opened``. Lets a test drive a good-then-failing
    device to exercise the consecutive-read-failure streak detector.
    """

    def __init__(self, index: int, backend: int | None = None, results: list[bool] | None = None) -> None:
        super().__init__(index, backend)
        self._results = list(results) if results is not None else []

    def read(self):
        ok = self._results.pop(0) if self._results else self.opened
        if not ok:
            return False, None
        return True, np.zeros((8, 8, 3), dtype=np.uint8)


class BlockingReadCapture(FakeVideoCapture):
    """A capture whose read() wedges until a gate Event is set.

    Models the tonight's-symptom failure: a device physically yanked mid-stream
    (or a wedged AVFoundation) whose ``cap.read()`` never returns. The reader
    thread parks inside read(); nothing but the process can unstick that handle,
    so the manager must recover *around* it (watchdog + abandon), not by waiting
    on it. Tests MUST set the gate in a ``finally`` so the abandoned reader
    thread can unwind and never hangs pytest.
    """

    def __init__(self, index: int, backend: int | None = None, gate: threading.Event | None = None) -> None:
        super().__init__(index, backend)
        self._gate = gate if gate is not None else threading.Event()

    def read(self):
        self._gate.wait()  # blocks until the test releases it
        return False, None


@pytest.fixture
def fake_captures(monkeypatch: pytest.MonkeyPatch) -> list[FakeVideoCapture]:
    """Patch cv2.VideoCapture (as seen by lelab.camera_preview) with a fake
    factory; returns the list of instances it constructed."""
    instances: list[FakeVideoCapture] = []

    def factory(index: int, backend: int | None = None) -> FakeVideoCapture:
        cap = FakeVideoCapture(index, backend)
        instances.append(cap)
        return cap

    monkeypatch.setattr(camera_preview.cv2, "VideoCapture", factory)
    return instances


# ---------------------------------------------------------------------------
# CameraPreviewManager — refcounting, stop_all, generator lifecycle
# ---------------------------------------------------------------------------


def test_two_clients_share_one_capture_last_release_frees_it(
    fake_captures: list[FakeVideoCapture],
) -> None:
    manager = CameraPreviewManager()
    gen_a = manager.open_stream(0)
    gen_b = manager.open_stream(0)

    # Both clients stream frames from ONE underlying device.
    assert b"--frame" in next(gen_a)
    assert b"Content-Type: image/jpeg" in next(gen_b)
    assert len(fake_captures) == 1

    # First client detaching must NOT release the shared capture...
    gen_a.close()
    assert not fake_captures[0].released

    # ...but the last one must, and the registry entry goes with it.
    gen_b.close()
    assert fake_captures[0].released
    assert manager._captures == {}


def test_distinct_indices_get_distinct_captures(fake_captures: list[FakeVideoCapture]) -> None:
    manager = CameraPreviewManager()
    gen_a = manager.open_stream(0)
    gen_b = manager.open_stream(1)
    next(gen_a)
    next(gen_b)
    assert [cap.index for cap in fake_captures] == [0, 1]
    gen_a.close()
    gen_b.close()
    assert all(cap.released for cap in fake_captures)


def test_open_failure_raises_and_leaks_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    instances: list[FakeVideoCapture] = []

    def factory(index: int, backend: int | None = None) -> FakeVideoCapture:
        cap = FailingVideoCapture(index, backend)
        instances.append(cap)
        return cap

    monkeypatch.setattr(camera_preview.cv2, "VideoCapture", factory)
    manager = CameraPreviewManager()

    with pytest.raises(CameraOpenError):
        manager.open_stream(3)

    # The failed capture was released and no registry entry was left behind.
    assert instances[0].released
    assert manager._captures == {}


def test_open_stream_releases_when_never_consumed(fake_captures: list[FakeVideoCapture]) -> None:
    """The refcount must return to zero even when the client disconnects without
    pulling a single frame. open_stream primes the generator (starts it) before
    handing it to Starlette, so its ``finally`` runs on close(). Pre-fix the ref
    was acquired *before* the generator started, and a never-started generator
    skips ``finally`` entirely (GeneratorExit before the first bytecode) — the
    exact hard leak that wedged a camera open until a backend restart."""
    manager = CameraPreviewManager()
    gen = manager.open_stream(0)  # primed: device open, refcount held, one frame buffered
    assert len(fake_captures) == 1
    assert not fake_captures[0].released

    gen.close()  # client vanished before consuming anything
    assert fake_captures[0].released
    assert manager._captures == {}


def test_close_after_partial_consume_returns_refcount_to_zero(
    fake_captures: list[FakeVideoCapture],
) -> None:
    """Closing a live generator mid-stream (browser tab closed / HMR reload)
    must release the device — this is the client-disconnect finalization path."""
    manager = CameraPreviewManager()
    gen = manager.open_stream(0)
    next(gen)  # stream a couple of frames, then the client goes away
    next(gen)
    gen.close()
    assert fake_captures[0].released
    assert manager._captures == {}


def test_generator_exits_when_device_stops_producing(
    monkeypatch: pytest.MonkeyPatch, fake_captures: list[FakeVideoCapture]
) -> None:
    """A device that stops producing frames is torn down after a streak of
    failures (not on a single dropped frame), and the shared capture is
    released so the index is free again."""
    monkeypatch.setattr(camera_preview, "_MAX_READ_FAILURES", 3)
    monkeypatch.setattr(camera_preview, "TARGET_FPS", 1000.0)  # tiny frame interval -> fast test
    manager = CameraPreviewManager()
    gen = manager.open_stream(0)
    next(gen)
    fake_captures[0].opened = False  # camera unplugged mid-stream
    with pytest.raises(StopIteration):
        for _ in range(20):
            next(gen)
    assert fake_captures[0].released
    assert manager._captures == {}


def test_transient_read_failure_does_not_kill_a_healthy_preview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single dropped frame must NOT tear the preview down: below the streak
    threshold the reader retries and keeps streaming."""
    monkeypatch.setattr(camera_preview, "_MAX_READ_FAILURES", 3)
    monkeypatch.setattr(camera_preview, "TARGET_FPS", 1000.0)
    captures: list[ScriptedVideoCapture] = []

    def factory(index: int, backend: int | None = None) -> ScriptedVideoCapture:
        # one hiccup, then OK -> the first frame must still arrive.
        cap = ScriptedVideoCapture(index, backend, results=[False, True, True])
        captures.append(cap)
        return cap

    monkeypatch.setattr(camera_preview.cv2, "VideoCapture", factory)
    manager = CameraPreviewManager()
    gen = manager.open_stream(0)
    assert b"--frame" in next(gen)  # recovered after the single failed read
    assert not captures[0].released
    gen.close()
    assert captures[0].released


def test_read_failure_streak_force_releases_shared_capture_and_reopens_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A persistently-dead capture (the wedge behind a black preview) is
    force-released even though a client still holds a refcount, the stream ends,
    and the next open_stream re-opens a FRESH capture rather than the dead one."""
    monkeypatch.setattr(camera_preview, "_MAX_READ_FAILURES", 3)
    monkeypatch.setattr(camera_preview, "TARGET_FPS", 1000.0)
    captures: list[ScriptedVideoCapture] = []
    scripts = [[True, False, False, False], None]  # first cap dies; second is healthy

    def factory(index: int, backend: int | None = None) -> ScriptedVideoCapture:
        cap = ScriptedVideoCapture(index, backend, results=scripts[len(captures)])
        captures.append(cap)
        return cap

    monkeypatch.setattr(camera_preview.cv2, "VideoCapture", factory)
    manager = CameraPreviewManager()
    gen = manager.open_stream(0)
    assert b"--frame" in next(gen)  # one healthy frame streams first
    with pytest.raises(StopIteration):
        next(gen)  # then 3 consecutive failures -> force-release + end
    assert captures[0].released
    assert manager._captures == {}  # deregistered: not left wedging the index

    gen2 = manager.open_stream(0)
    assert b"--frame" in next(gen2)
    assert len(captures) == 2  # a brand-new capture, not the dead one
    gen2.close()
    assert captures[1].released


# ---------------------------------------------------------------------------
# Reader-thread watchdog: a read() wedged forever (unplugged mid-stream) must
# never clog the manager, the recording claim, or reopen. cv2 lives only on the
# reader thread, so a hung read can pin neither a client refcount nor a lock the
# force-release paths need — the manager recovers around the abandoned handle.
# ---------------------------------------------------------------------------


def test_hung_read_client_ends_within_deadline_and_deregisters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reader wedged inside cap.read() produces no frames. The frame-age
    watchdog must end the client stream within the deadline (not block forever)
    and deregister the capture so it stops wedging the registry."""
    monkeypatch.setattr(camera_preview, "_FRAME_DEADLINE", 0.15)
    monkeypatch.setattr(camera_preview, "_RELEASE_GRACE", 0.1)
    gate = threading.Event()
    captures: list[BlockingReadCapture] = []

    def factory(index: int, backend: int | None = None) -> BlockingReadCapture:
        cap = BlockingReadCapture(index, backend, gate=gate)
        captures.append(cap)
        return cap

    monkeypatch.setattr(camera_preview.cv2, "VideoCapture", factory)
    manager = CameraPreviewManager()
    try:
        gen = manager.open_stream(0)  # opens fine; the reader then wedges in read()
        t0 = time.monotonic()
        with pytest.raises(StopIteration):
            next(gen)  # no frame ever arrives -> the watchdog ends the stream
        elapsed = time.monotonic() - t0
        assert elapsed < 2.0  # ended near the (small) deadline, not hung forever
        assert manager._captures == {}  # watchdog deregistered the wedged capture
    finally:
        gate.set()  # let the abandoned reader thread unwind (never hang pytest)


def test_hung_read_block_for_recording_returns_promptly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE regression test for tonight's symptom. With a reader wedged inside
    cap.read(), block_for_recording must return within its timeout and clear the
    registry — the recording claim can never be clogged by a hung preview read
    (the old design blocked acquiring the per-index lock the hung read held)."""
    gate = threading.Event()
    captures: list[BlockingReadCapture] = []

    def factory(index: int, backend: int | None = None) -> BlockingReadCapture:
        cap = BlockingReadCapture(index, backend, gate=gate)
        captures.append(cap)
        return cap

    monkeypatch.setattr(camera_preview.cv2, "VideoCapture", factory)
    manager = CameraPreviewManager()
    gen = manager.open_stream(0)  # reader opens, then wedges in read(); refcount held
    try:
        t0 = time.monotonic()
        manager.block_for_recording(timeout=0.1)
        elapsed = time.monotonic() - t0
        assert elapsed < 1.0  # returned ~within the timeout, not blocked on the hung read
        assert manager._captures == {}  # the wedged capture was deregistered (abandoned)
        # And the recording latch is in force: a racing open is refused.
        with pytest.raises(CameraOpenError):
            manager.open_stream(0)
    finally:
        gate.set()  # unstick the abandoned reader before finalizing the client
        gen.close()


def test_reopen_after_force_release_retries_a_busy_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A device freed by an abrupt force-release can report busy for a beat. The
    next open must give it ONE short retry before surfacing CameraOpenError, so a
    tile re-requesting right after a recording claim ends recovers smoothly."""
    monkeypatch.setattr(camera_preview, "_REOPEN_RETRY_DELAY", 0.01)
    captures: list[FakeVideoCapture] = []
    # 1st open (initial) succeeds; after force-release the reopen's first attempt
    # is busy, and the single retry succeeds.
    opened_seq = iter([True, False, True])

    def factory(index: int, backend: int | None = None) -> FakeVideoCapture:
        cap = FakeVideoCapture(index, backend)
        cap.opened = next(opened_seq, True)
        captures.append(cap)
        return cap

    monkeypatch.setattr(camera_preview.cv2, "VideoCapture", factory)
    manager = CameraPreviewManager()
    gen = manager.open_stream(0)
    assert b"--frame" in next(gen)
    manager.stop_all(timeout=0.2)
    assert captures[0].released

    gen2 = manager.open_stream(0)  # first reopen attempt busy -> retry -> success
    assert b"--frame" in next(gen2)
    assert len(captures) == 3  # initial + busy attempt + successful retry
    assert captures[1].released  # the busy handle was released before the retry
    gen2.close()
    assert captures[2].released
    gen.close()


def test_held_healthy_indices_tracks_open_captures(fake_captures: list[FakeVideoCapture]) -> None:
    """held_healthy_indices reports open indexes (so enumeration can report a
    manager-held camera the fresh-open probe would miss) and drops them on
    release."""
    manager = CameraPreviewManager()
    assert manager.held_healthy_indices() == []
    gen0 = manager.open_stream(0)
    gen2 = manager.open_stream(2)
    assert manager.held_healthy_indices() == [0, 2]
    gen0.close()
    assert manager.held_healthy_indices() == [2]
    gen2.close()
    assert manager.held_healthy_indices() == []


def test_stop_all_force_releases_a_stalled_client(fake_captures: list[FakeVideoCapture]) -> None:
    """A generator suspended mid-yield (a stalled/dead client) can't detach on
    its own; stop_all must force-release the device after the brief wait, and
    the generator must exit — not re-grab the camera — when it resumes."""
    manager = CameraPreviewManager()
    gen = manager.open_stream(0)
    next(gen)  # suspended at yield, refcount still held

    manager.stop_all(timeout=0.05)

    assert fake_captures[0].released
    assert manager._captures == {}
    # The lagging client's next pull ends the stream (release is a no-op).
    with pytest.raises(StopIteration):
        next(gen)


def test_stop_all_without_streams_is_a_noop(fake_captures: list[FakeVideoCapture]) -> None:
    manager = CameraPreviewManager()
    manager.stop_all(timeout=0.05)
    assert fake_captures == []


def test_stream_after_stop_all_reopens_the_camera(fake_captures: list[FakeVideoCapture]) -> None:
    """stop_all must not poison the index: a later preview gets a fresh capture."""
    manager = CameraPreviewManager()
    gen = manager.open_stream(0)
    next(gen)
    manager.stop_all(timeout=0.05)

    gen2 = manager.open_stream(0)
    assert b"--frame" in next(gen2)
    assert len(fake_captures) == 2
    gen2.close()
    assert fake_captures[1].released
    gen.close()


def test_block_for_recording_releases_and_latches_opens_off(
    fake_captures: list[FakeVideoCapture],
) -> None:
    """block_for_recording force-releases a held capture AND refuses any new open
    until resume_previews — so a preview retrying right after the release can't
    re-acquire the device the recorder is about to grab (the actual_fps=5.0
    starvation race)."""
    manager = CameraPreviewManager()
    gen = manager.open_stream(0)
    next(gen)  # capture held

    manager.block_for_recording(timeout=0.05)
    assert fake_captures[0].released
    assert manager._captures == {}

    # A preview open racing the recording claim is rejected, not served.
    with pytest.raises(CameraOpenError):
        manager.open_stream(0)
    # No stray capture was opened by the rejected attempt.
    assert len(fake_captures) == 1
    gen.close()


def test_resume_previews_lifts_the_recording_latch(fake_captures: list[FakeVideoCapture]) -> None:
    """After resume_previews the manager serves opens again with a fresh capture."""
    manager = CameraPreviewManager()
    manager.block_for_recording(timeout=0.05)
    with pytest.raises(CameraOpenError):
        manager.open_stream(0)

    manager.resume_previews()
    gen = manager.open_stream(0)
    assert b"--frame" in next(gen)
    gen.close()
    assert fake_captures[0].released


# ---------------------------------------------------------------------------
# camera_id addressing — the unique_id is canonical; the cv2 index is resolved
# fresh on every capture open
# ---------------------------------------------------------------------------


def test_preview_by_unique_id_reresolves_index_on_reopen(
    monkeypatch: pytest.MonkeyPatch, fake_captures: list[FakeVideoCapture]
) -> None:
    """THE regression test this redesign exists for: a preview opened by
    unique_id opens the device's CURRENT cv2 index, and a reopen after a
    hotplug reshuffle re-resolves — it follows the physical device to its new
    index instead of streaming whatever slid into the old slot."""
    enumeration = [{"index": 1, "name": "USB Camera", "unique_id": "uvc-7749-wrist"}]
    monkeypatch.setattr(camera_preview.camera_enumeration, "list_cameras", lambda: list(enumeration))

    manager = CameraPreviewManager()
    gen = manager.open_stream("uvc-7749-wrist")
    assert b"--frame" in next(gen)
    assert fake_captures[0].index == 1  # opened at the device's current index

    # Device released (e.g. stop_all before a session), then a USB reshuffle
    # moves the same physical camera to index 2.
    manager.stop_all(timeout=0.05)
    assert fake_captures[0].released
    enumeration[0] = {"index": 2, "name": "USB Camera", "unique_id": "uvc-7749-wrist"}

    gen2 = manager.open_stream("uvc-7749-wrist")
    assert b"--frame" in next(gen2)
    assert fake_captures[1].index == 2  # fresh open re-resolved the NEW index
    gen2.close()
    gen.close()


def test_preview_by_unknown_unique_id_raises_legible_error(
    monkeypatch: pytest.MonkeyPatch, fake_captures: list[FakeVideoCapture]
) -> None:
    """An id matching no connected device is a CameraOpenError (endpoint 503),
    never a silent fallback to some index."""
    monkeypatch.setattr(camera_preview.camera_enumeration, "list_cameras", lambda: [])
    manager = CameraPreviewManager()
    with pytest.raises(CameraOpenError) as exc:
        manager.open_stream("uvc-7749-gone")
    assert "not connected" in str(exc.value)
    assert fake_captures == []  # no capture was ever opened
    assert manager._captures == {}  # the failed acquire leaked nothing


def test_numeric_camera_id_is_the_index_fallback_lane(
    monkeypatch: pytest.MonkeyPatch, fake_captures: list[FakeVideoCapture]
) -> None:
    """A purely-numeric camera_id (or a legacy int) opens that cv2 index
    directly, with NO enumeration — the back-compat lane for platforms and
    saved entries without stable ids."""

    def _boom():
        raise AssertionError("numeric ids must not trigger enumeration")

    monkeypatch.setattr(camera_preview.camera_enumeration, "list_cameras", _boom)
    manager = CameraPreviewManager()
    gen = manager.open_stream("3")
    assert b"--frame" in next(gen)
    assert fake_captures[0].index == 3
    gen.close()


def test_same_unique_id_shares_one_capture(
    monkeypatch: pytest.MonkeyPatch, fake_captures: list[FakeVideoCapture]
) -> None:
    """Two clients of the same camera_id share one capture (registry is keyed
    by id), and the index is resolved once per open — not per client."""
    calls: list[int] = []

    def _list():
        calls.append(1)
        return [{"index": 0, "name": "USB Camera", "unique_id": "uvc-7749-front"}]

    monkeypatch.setattr(camera_preview.camera_enumeration, "list_cameras", _list)
    manager = CameraPreviewManager()
    gen_a = manager.open_stream("uvc-7749-front")
    gen_b = manager.open_stream("uvc-7749-front")
    assert len(fake_captures) == 1  # shared
    assert len(calls) == 1  # resolved once (second client reuses the open capture)
    gen_a.close()
    assert not fake_captures[0].released  # still one client attached
    gen_b.close()
    assert fake_captures[0].released


# ---------------------------------------------------------------------------
# GET /camera-preview/{index} — status codes and exclusivity
# ---------------------------------------------------------------------------


def test_camera_preview_409_while_recording(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(record, "recording_active", True)
    response = client.get("/camera-preview/0")
    assert response.status_code == 409
    assert "Recording" in response.json()["detail"]


def test_camera_preview_allowed_while_teleoperating(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Teleop drives the serial bus and opens no cv2 cameras, so a preview during
    teleop does not contend — it must NOT 409. (The manager is patched to a
    finite stream so the TestClient request completes.)"""

    def finite_stream(index: int):
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\nfake-jpeg\r\n"

    monkeypatch.setattr(teleoperate, "teleoperation_active", True)
    monkeypatch.setattr(server_mod.camera_preview_manager, "open_stream", finite_stream)
    response = client.get("/camera-preview/0")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("multipart/x-mixed-replace")


def test_camera_preview_503_when_camera_cannot_open(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(camera_preview.cv2, "VideoCapture", FailingVideoCapture)
    response = client.get("/camera-preview/9")
    assert response.status_code == 503
    assert "could not be opened" in response.json()["detail"]


def test_camera_preview_streams_multipart_mjpeg(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Numeric fallback lane: 200 + the multipart media type, with the manager
    patched to a FINITE stream so the TestClient request completes (the real
    generator is endless by design; its behavior is covered by the manager
    tests above)."""

    def finite_stream(camera_id: str):
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\nfake-jpeg\r\n"

    monkeypatch.setattr(server_mod.camera_preview_manager, "open_stream", finite_stream)
    response = client.get("/camera-preview/0")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("multipart/x-mixed-replace")
    assert b"--frame" in response.content


def test_camera_preview_routes_by_unique_id(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Id-addressed lane: a non-numeric camera_id path segment reaches the
    manager verbatim (URL-decoded), and streams like the numeric lane."""
    seen: list[str] = []

    def finite_stream(camera_id: str):
        seen.append(camera_id)
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\nfake-jpeg\r\n"

    monkeypatch.setattr(server_mod.camera_preview_manager, "open_stream", finite_stream)
    response = client.get("/camera-preview/0x1220001e450209")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("multipart/x-mixed-replace")
    assert seen == ["0x1220001e450209"]


def test_camera_preview_503_for_disconnected_unique_id(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An id matching no connected camera answers 503 with the legible
    not-connected detail (real manager, enumeration mocked empty)."""
    monkeypatch.setattr(camera_preview.camera_enumeration, "list_cameras", lambda: [])
    response = client.get("/camera-preview/uvc-7749-gone")
    assert response.status_code == 503
    assert "not connected" in response.json()["detail"]


# ---------------------------------------------------------------------------
# GET /available-cameras — enumeration answers for manager-held indexes
# ---------------------------------------------------------------------------


def test_available_cameras_rows_carry_unique_id_primary_key(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each enumerated row passes its unique_id through — the primary key that
    previews (/camera-preview/{camera_id}) and saved configs bind to. The index
    stays present but is informational/fallback only."""
    monkeypatch.setattr(
        camera_enumeration,
        "list_cameras",
        lambda: [
            {"index": 0, "name": "USB Camera", "available": True, "unique_id": "0x1220001e450209"},
            {"index": 1, "name": "USB Camera", "available": True, "unique_id": "0x1300001e450209"},
        ],
    )
    monkeypatch.setattr(server_mod.camera_preview_manager, "held_healthy_indices", lambda: [])

    response = client.get("/available-cameras")
    assert response.status_code == 200
    rows = response.json()["cameras"]
    assert [r["unique_id"] for r in rows] == ["0x1220001e450209", "0x1300001e450209"]
    assert [r["index"] for r in rows] == [0, 1]


def test_available_cameras_includes_a_held_healthy_index(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A camera the preview manager holds open can't be re-opened by the probe
    (device busy) and would silently vanish from the list. The manager knows
    it's alive, so enumeration must add it back."""
    monkeypatch.setattr(
        camera_enumeration,
        "list_cameras",
        lambda: [{"index": 0, "name": "Built-in", "available": True}],
    )
    monkeypatch.setattr(server_mod.camera_preview_manager, "held_healthy_indices", lambda: [1])

    response = client.get("/available-cameras")
    assert response.status_code == 200
    cameras = response.json()["cameras"]
    assert [c["index"] for c in cameras] == [0, 1]  # held index re-added, sorted
    assert cameras[1]["available"] is True


def test_available_cameras_recovers_a_previously_wedged_index(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wedged capture is force-released (so held_healthy_indices no longer
    reports it) and the fresh probe re-enumerates it: it must reappear exactly
    once, not be dropped or duplicated."""
    monkeypatch.setattr(
        camera_enumeration,
        "list_cameras",
        lambda: [
            {"index": 0, "name": "Built-in", "available": True},
            {"index": 1, "name": "Wrist", "available": True},
        ],
    )
    monkeypatch.setattr(server_mod.camera_preview_manager, "held_healthy_indices", lambda: [])

    response = client.get("/available-cameras")
    assert response.status_code == 200
    assert [c["index"] for c in response.json()["cameras"]] == [0, 1]


def test_available_cameras_does_not_duplicate_a_held_and_enumerated_index(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On macOS the probe (a fresh subprocess) still lists a held camera, so a
    healthy held index that IS enumerated must not be added twice."""
    monkeypatch.setattr(
        camera_enumeration,
        "list_cameras",
        lambda: [{"index": 0, "name": "Built-in", "available": True, "unique_id": "abc"}],
    )
    monkeypatch.setattr(server_mod.camera_preview_manager, "held_healthy_indices", lambda: [0])

    response = client.get("/available-cameras")
    assert [c["index"] for c in response.json()["cameras"]] == [0]


# ---------------------------------------------------------------------------
# Exclusivity wiring — recording/teleop start paths stop the previews
# ---------------------------------------------------------------------------


def test_start_recording_stops_camera_previews(monkeypatch: pytest.MonkeyPatch) -> None:
    """handle_start_recording LATCHES the previews off (block_for_recording, not
    plain stop_all) before any robot/camera construction, and lifts the latch
    (resume_previews) when the start fails — create_record_config is made to
    fail right after, so no worker or hardware is ever touched."""
    calls: list[str] = []
    monkeypatch.setattr(
        record.camera_preview_manager, "block_for_recording", lambda: calls.append("block_for_recording")
    )
    monkeypatch.setattr(
        record.camera_preview_manager, "resume_previews", lambda: calls.append("resume_previews")
    )
    monkeypatch.setattr(record, "recording_active", False)
    monkeypatch.setattr(record, "recording_thread", None)
    monkeypatch.setattr(teleoperate, "teleoperation_active", False)
    monkeypatch.setattr(teleoperate, "teleoperation_thread", None)

    def _boom(request):
        raise RuntimeError("stop before hardware")

    monkeypatch.setattr(record, "create_record_config", _boom)

    result = record.handle_start_recording(
        record.RecordingRequest(
            leader_port="COM_LEADER",
            follower_port="COM_FOLLOWER",
            leader_config="leader",
            follower_config="follower",
            dataset_repo_id="tester/dataset",
            single_task="pick",
        )
    )

    assert result["success"] is False
    # Latched off at start, then lifted on the failed-start path.
    assert calls == ["block_for_recording", "resume_previews"]
    assert record.recording_active is False


def test_start_teleoperation_stops_camera_previews(monkeypatch: pytest.MonkeyPatch) -> None:
    """handle_start_teleoperation force-releases the previews before any
    device construction (setup_calibration_files is made to fail right after)."""
    calls: list[str] = []
    monkeypatch.setattr(teleoperate.camera_preview_manager, "stop_all", lambda: calls.append("stop_all"))
    monkeypatch.setattr(teleoperate, "teleoperation_active", False)
    monkeypatch.setattr(teleoperate, "teleoperation_thread", None)
    monkeypatch.setattr(record, "recording_active", False)
    monkeypatch.setattr(record, "recording_thread", None)

    def _boom(leader, follower):
        raise RuntimeError("stop before hardware")

    monkeypatch.setattr("lelab.utils.robot_factory.setup_calibration_files", _boom)

    result = teleoperate.handle_start_teleoperation(
        teleoperate.TeleoperateRequest(
            leader_port="COM_LEADER",
            follower_port="COM_FOLLOWER",
            leader_config="leader",
            follower_config="follower",
        )
    )

    assert result["success"] is False
    assert calls == ["stop_all"]
    assert teleoperate.teleoperation_active is False
