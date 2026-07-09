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

import cv2
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
        # Property config recording, so tests can assert the preview manager
        # forces MJPG/size/fps before the first read (the USB-2 bandwidth fix).
        # ``set`` returns ``set_returns`` — flip it to False to model a device
        # (e.g. a built-in with no MJPG mode) that rejects the format.
        self.set_calls: list[tuple[int, float]] = []
        self.set_returns = True
        self._props: dict[int, float] = {}
        # Snapshot of set_calls taken at the first read(), so a test can verify
        # every property was set BEFORE any frame was pulled.
        self.sets_before_first_read: list[tuple[int, float]] | None = None

    def isOpened(self) -> bool:  # noqa: N802 — cv2's camelCase API
        return self.opened

    def set(self, prop: int, value: float) -> bool:
        self.set_calls.append((prop, value))
        self._props[prop] = value
        return self.set_returns

    def get(self, prop: int) -> float:
        return self._props.get(prop, 0.0)

    def read(self):
        if self.sets_before_first_read is None:
            self.sets_before_first_read = list(self.set_calls)
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


class GatedFrameCapture(FakeVideoCapture):
    """A capture that opens fine but whose FIRST read() blocks until a gate is set.

    Lets a test hold a camera *inside the open funnel*: the reader opens and
    publishes open_ok (so open_stream returns), then parks in read() with no
    frame published yet — so it still owns the turnstile. Once the gate is set,
    read() returns real frames, the first frame publishes, and the reader leaves
    the funnel. Subsequent reads return frames immediately.
    """

    def __init__(self, index: int, backend: int | None = None, gate: threading.Event | None = None) -> None:
        super().__init__(index, backend)
        self._gate = gate if gate is not None else threading.Event()
        self._first = True

    def read(self):
        if self._first:
            self._gate.wait()  # holds the funnel until the test releases it
            self._first = False
        if not self.opened:
            return False, None
        return True, np.zeros((8, 8, 3), dtype=np.uint8)


def _wait_until(predicate, timeout: float, interval: float = 0.01) -> bool:
    """Poll ``predicate`` until true or ``timeout`` lapses. Returns the final
    value. Bounded (never hangs); used to synchronise on internal manager state
    instead of sleeping a fixed duration."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


@pytest.fixture(autouse=True)
def _synchronous_teardown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable the idle linger by default: most tests assert the classic
    release-frees-the-device contract. Linger-specific tests re-enable it."""
    monkeypatch.setattr(camera_preview, "_IDLE_LINGER", 0.0)


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
# Capture format configuration — every preview asks for 640x480@30 before the
# first read. FOURCC is deliberately NOT forced by default (_PREVIEW_FOURCC is
# None): some AVFoundation/UVC stacks open fine after an MJPG request and then
# never deliver frames, so forced compression must be an explicit opt-in.
# ---------------------------------------------------------------------------


def test_open_configures_size_fps_before_first_read(
    fake_captures: list[FakeVideoCapture],
) -> None:
    """A freshly opened preview capture is configured 640x480 / 30fps (size then
    FPS, lerobot's property order) BEFORE any frame is read — and no FOURCC is
    forced by default."""
    manager = CameraPreviewManager()
    gen = manager.open_stream(0)
    next(gen)  # force at least one read so sets_before_first_read is captured
    cap = fake_captures[0]

    assert cap.sets_before_first_read is not None
    props_in_order = [prop for prop, _ in cap.sets_before_first_read]
    assert props_in_order == [
        cv2.CAP_PROP_FRAME_WIDTH,
        cv2.CAP_PROP_FRAME_HEIGHT,
        cv2.CAP_PROP_FPS,
    ]
    values = dict(cap.sets_before_first_read)
    assert values[cv2.CAP_PROP_FRAME_WIDTH] == 640
    assert values[cv2.CAP_PROP_FRAME_HEIGHT] == 480
    assert values[cv2.CAP_PROP_FPS] == 30

    gen.close()
    assert cap.released


def test_open_sets_fourcc_first_when_explicitly_configured(
    monkeypatch: pytest.MonkeyPatch, fake_captures: list[FakeVideoCapture]
) -> None:
    """When a bench opts in to a forced FOURCC, it is set FIRST (it can change
    the available resolution/FPS options), then size, then FPS."""
    monkeypatch.setattr(camera_preview, "_PREVIEW_FOURCC", "MJPG")
    manager = CameraPreviewManager()
    gen = manager.open_stream(0)
    next(gen)
    cap = fake_captures[0]
    assert cap.sets_before_first_read is not None
    props_in_order = [prop for prop, _ in cap.sets_before_first_read]
    assert props_in_order == [
        cv2.CAP_PROP_FOURCC,
        cv2.CAP_PROP_FRAME_WIDTH,
        cv2.CAP_PROP_FRAME_HEIGHT,
        cv2.CAP_PROP_FPS,
    ]
    assert dict(cap.sets_before_first_read)[cv2.CAP_PROP_FOURCC] == cv2.VideoWriter_fourcc(*"MJPG")
    gen.close()
    assert cap.released


def test_open_streams_normally_when_format_set_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A capture whose set() returns False (a built-in camera that rejects the
    format) keeps its auto-negotiated format and streams normally — property-set
    failures must never fail the open."""
    captures: list[FakeVideoCapture] = []

    def factory(index: int, backend: int | None = None) -> FakeVideoCapture:
        cap = FakeVideoCapture(index, backend)
        cap.set_returns = False  # every format set fails, as a real built-in would
        captures.append(cap)
        return cap

    monkeypatch.setattr(camera_preview.cv2, "VideoCapture", factory)
    manager = CameraPreviewManager()
    gen = manager.open_stream(0)
    assert b"--frame" in next(gen)  # streams despite every set() returning False
    # The sets were still attempted (size + fps), before the first read.
    assert captures[0].sets_before_first_read is not None
    assert len(captures[0].sets_before_first_read) == 3
    gen.close()
    assert captures[0].released


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


def test_block_for_recording_default_reason_is_recording(
    fake_captures: list[FakeVideoCapture],
) -> None:
    """The generalised latch defaults to a recording-flavoured reason, so the
    unchanged recording call site (block_for_recording() with no args) still
    refuses a racing open with the recording-worded message."""
    manager = CameraPreviewManager()
    manager.block_for_recording(timeout=0.05)
    with pytest.raises(CameraOpenError) as exc:
        manager.open_stream(0)
    assert "reserved for the active recording session" in str(exc.value)


def test_block_for_recording_custom_reason_surfaces_in_refusal(
    fake_captures: list[FakeVideoCapture],
) -> None:
    """A caller (inference) passing a custom reason has it named in the refusal a
    racing preview open raises, so the tile can say inference — not recording —
    owns the cameras."""
    manager = CameraPreviewManager()
    manager.block_for_recording(timeout=0.05, reason="inference")
    with pytest.raises(CameraOpenError) as exc:
        manager.open_stream(0)
    assert "reserved for the active inference session" in str(exc.value)
    assert "recording" not in str(exc.value)


# ---------------------------------------------------------------------------
# Open funnel (turnstile): cameras are born ONE AT A TIME. Three tiles mounting
# at once must not fire three concurrent cv2 opens into a shared USB-2 hub — a
# reader admits the next open only after the previous camera published its first
# frame or failed conclusively. The funnel is a best-effort serializer: a holder
# wedged in its first read() delays the next open by at most the deadline, never
# forever (fail-open).
# ---------------------------------------------------------------------------


def test_funnel_serializes_two_concurrent_opens(monkeypatch: pytest.MonkeyPatch) -> None:
    """The second camera's cv2 open must NOT begin until the first has published
    its first frame. Camera A is held inside the funnel (opened, but its first
    read() gated), so while it owns the turnstile camera B's reader parks and
    constructs no cv2 capture. Releasing A's first frame lets B open and stream."""
    gate_a = threading.Event()
    captures: list[FakeVideoCapture] = []

    def factory(index: int, backend: int | None = None) -> FakeVideoCapture:
        # First construction (camera 0) is gated; the rest are healthy.
        cap = (
            GatedFrameCapture(index, backend, gate=gate_a)
            if not captures
            else FakeVideoCapture(index, backend)
        )
        captures.append(cap)
        return cap

    monkeypatch.setattr(camera_preview.cv2, "VideoCapture", factory)
    manager = CameraPreviewManager()

    # A opens (open_ok returns) then parks in its gated first read -> owns the
    # funnel without ever publishing a frame.
    gen_a = manager.open_stream(0)
    assert _wait_until(lambda: len(captures) == 1, timeout=2.0)

    # Open B concurrently: its open_stream blocks in _acquire until B's reader
    # clears the funnel, so drive it from a thread.
    b_box: dict[str, object] = {}

    def open_b() -> None:
        try:
            b_box["gen"] = manager.open_stream(1)
        except Exception as exc:  # pragma: no cover - surfaced via the assert below
            b_box["error"] = exc

    tb = threading.Thread(target=open_b, name="open-b")
    tb.start()
    try:
        # B's reader has registered and started (entry in the registry) but must
        # be parked in the turnstile: no second cv2 capture may exist yet.
        assert _wait_until(lambda: "1" in manager._captures, timeout=2.0)
        # Give B ample room to (wrongly) open if the funnel weren't serializing;
        # it must still be blocked, so this stays False.
        assert not _wait_until(lambda: len(captures) >= 2, timeout=0.3)

        # Release A's first frame -> A leaves the funnel -> B may now open.
        gate_a.set()
        assert _wait_until(lambda: "gen" in b_box, timeout=5.0), b_box.get("error")
        tb.join(timeout=5.0)
        assert not tb.is_alive()
        assert len(captures) == 2  # B opened only after A's first frame
        assert [c.index for c in captures] == [0, 1]

        gen_b = b_box["gen"]
        assert b"--frame" in next(gen_a)
        assert b"--frame" in next(gen_b)
        gen_b.close()
    finally:
        gate_a.set()  # ensure A's reader can never stay wedged
        tb.join(timeout=5.0)
        gen_a.close()


def test_funnel_fails_open_when_holder_wedged_in_first_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """A holder wedged inside its first read() can release nothing — so the funnel
    must let the next open proceed once the holder's deadline lapses (usurp). The
    second camera still opens and streams; the delay is bounded by the deadline,
    never forever."""
    monkeypatch.setattr(camera_preview, "_FUNNEL_HOLD_DEADLINE", 0.3)
    gate = threading.Event()  # never set until the finally: A stays wedged
    captures: list[FakeVideoCapture] = []

    def factory(index: int, backend: int | None = None) -> FakeVideoCapture:
        # A wedges forever in read() (holds the funnel); B is healthy.
        cap = (
            BlockingReadCapture(index, backend, gate=gate)
            if not captures
            else FakeVideoCapture(index, backend)
        )
        captures.append(cap)
        return cap

    monkeypatch.setattr(camera_preview.cv2, "VideoCapture", factory)
    manager = CameraPreviewManager()

    gen_a = manager.open_stream(0)  # opens, then wedges in read() holding the funnel
    gen_b = None
    try:
        assert _wait_until(lambda: len(captures) == 1, timeout=2.0)
        t0 = time.monotonic()
        gen_b = manager.open_stream(1)  # blocks until B usurps at the deadline
        elapsed = time.monotonic() - t0
        assert b"--frame" in next(gen_b)  # B streams despite A wedged (fail-open)
        assert len(captures) == 2
        # Delayed by ~the deadline, not forever and not much beyond it.
        assert elapsed < 2.0
    finally:
        gate.set()  # let the abandoned wedged reader unwind (never hang pytest)
        if gen_b is not None:
            gen_b.close()
        gen_a.close()


def test_funnel_failed_first_open_releases_promptly(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed first open (unopenable device) must release the funnel promptly —
    not hold it for the deadline — so the next camera opens right away. The
    default (large) deadline is left in place: B opening fast proves the funnel
    was freed by the failure, not by a deadline usurp."""
    captures: list[FakeVideoCapture] = []

    def factory(index: int, backend: int | None = None) -> FakeVideoCapture:
        # A can't open; B is healthy.
        cap = FailingVideoCapture(index, backend) if not captures else FakeVideoCapture(index, backend)
        captures.append(cap)
        return cap

    monkeypatch.setattr(camera_preview.cv2, "VideoCapture", factory)
    manager = CameraPreviewManager()

    with pytest.raises(CameraOpenError):
        manager.open_stream(0)  # open fails -> reader leaves the funnel before raising

    t0 = time.monotonic()
    gen_b = manager.open_stream(1)  # funnel already free -> immediate, no deadline wait
    assert b"--frame" in next(gen_b)
    elapsed = time.monotonic() - t0
    assert elapsed < 5.0  # far below the default _FUNNEL_HOLD_DEADLINE (~17.5s)
    assert len(captures) == 2
    assert "0" not in manager._captures  # the failed birth left nothing behind
    assert "1" in manager._captures  # only B remains registered
    gen_b.close()


def test_force_release_of_queued_birth_never_opens_the_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reader stopped while queued at the turnstile must NEVER go on to open
    the device. block_for_recording stops it precisely because the session owner
    is about to open that camera itself — an unserialized open/configure/release
    cycle behind the recorder's back is the exact contention the latch exists to
    prevent. The stopped birth takes the no-handle exit: legible cancel error,
    dead, deregistered, release confirmed, and cv2.VideoCapture never called."""
    gate_a = threading.Event()
    captures: list[FakeVideoCapture] = []

    def factory(index: int, backend: int | None = None) -> FakeVideoCapture:
        # A owns the funnel (gated first read); anything after A is healthy.
        cap = (
            GatedFrameCapture(index, backend, gate=gate_a)
            if not captures
            else FakeVideoCapture(index, backend)
        )
        captures.append(cap)
        return cap

    monkeypatch.setattr(camera_preview.cv2, "VideoCapture", factory)
    manager = CameraPreviewManager()

    gen_a = manager.open_stream(0)  # A opens, parks in its gated first read
    b_box: dict[str, object] = {}

    def open_b() -> None:
        try:
            b_box["gen"] = manager.open_stream(1)
        except CameraOpenError as exc:
            b_box["error"] = exc

    tb = threading.Thread(target=open_b, name="open-b")
    tb.start()
    try:
        # B is registered and queued behind A — its birth has not begun.
        assert _wait_until(lambda: "1" in manager._captures, timeout=2.0)
        entry_b = manager._captures["1"]
        assert entry_b.birth_started_ts is None

        # Recording claims the devices while B is still queued (A, wedged in its
        # gated read, is abandoned — that path is covered elsewhere).
        manager.block_for_recording(timeout=0.2)

        # B's client gets a legible failure, B's reader confirms release...
        assert _wait_until(lambda: "error" in b_box or "gen" in b_box, timeout=5.0)
        assert "error" in b_box, "queued birth must fail once stopped, not stream"
        assert entry_b.released.wait(2.0)
        tb.join(timeout=5.0)
        assert not tb.is_alive()
        # ...and — the point — the device at index 1 was NEVER opened.
        assert [c.index for c in captures] == [0]
    finally:
        gate_a.set()  # let A's abandoned reader unwind (never hang pytest)
        manager.resume_previews()
        tb.join(timeout=5.0)
        gen_a.close()


def test_queued_open_is_not_billed_the_holders_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """A client whose camera is queued at the turnstile must not have its
    _OPEN_TIMEOUT consumed by the holder ahead of it: the per-camera budget
    starts when ITS reader owns the turnstile (birth_started_ts); queue time is
    bounded only by the end-to-end backstop. With the old single clock, the
    third tile of a simultaneous mount 503'd while still healthy in the queue."""
    monkeypatch.setattr(camera_preview, "_OPEN_TIMEOUT", 0.3)
    # Generous backstop so it demonstrably isn't what keeps B's client waiting.
    monkeypatch.setattr(camera_preview, "_QUEUED_OPEN_BACKSTOP", 30.0)
    gate_a = threading.Event()
    captures: list[FakeVideoCapture] = []

    def factory(index: int, backend: int | None = None) -> FakeVideoCapture:
        cap = (
            GatedFrameCapture(index, backend, gate=gate_a)
            if not captures
            else FakeVideoCapture(index, backend)
        )
        captures.append(cap)
        return cap

    monkeypatch.setattr(camera_preview.cv2, "VideoCapture", factory)
    manager = CameraPreviewManager()

    gen_a = manager.open_stream(0)  # A owns the funnel (gated first read)
    b_box: dict[str, object] = {}

    def open_b() -> None:
        try:
            b_box["gen"] = manager.open_stream(1)
        except CameraOpenError as exc:
            b_box["error"] = exc

    tb = threading.Thread(target=open_b, name="open-b")
    tb.start()
    try:
        assert _wait_until(lambda: "1" in manager._captures, timeout=2.0)
        # Hold A well past B's whole _OPEN_TIMEOUT. B's client must still be
        # waiting (queue time bills to the backstop), not failed with a 503.
        assert not _wait_until(lambda: ("error" in b_box) or ("gen" in b_box), timeout=0.8)

        # A publishes its first frame -> turnstile passes to B -> B's own open
        # begins now and completes well inside its own fresh budget.
        gate_a.set()
        assert _wait_until(lambda: "gen" in b_box, timeout=5.0), b_box.get("error")
        assert b"--frame" in next(b_box["gen"])
        b_box["gen"].close()
    finally:
        gate_a.set()
        tb.join(timeout=5.0)
        gen_a.close()


def test_frame_dead_birth_fails_fast_and_releases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A capture that opens fine but never delivers a frame (a session born
    into another session's teardown wake) must give up at the FIRST-FRAME
    deadline — releasing the device and freeing the funnel for the next open —
    rather than camping for the (deliberately patient) mid-stream limits.
    The clients' retry loops then re-roll the open, which heals it."""
    monkeypatch.setattr(camera_preview, "_FIRST_FRAME_DEADLINE", 0.2)
    captures: list[FakeVideoCapture] = []

    def factory(index: int, backend: int | None = None) -> FakeVideoCapture:
        cap = FakeVideoCapture(index, backend)
        cap.opened = True
        cap.read = lambda: (False, None)  # opens fine, never a frame
        captures.append(cap)
        return cap

    monkeypatch.setattr(camera_preview.cv2, "VideoCapture", factory)
    manager = CameraPreviewManager()
    gen = manager.open_stream(0)  # open succeeds (open_ok before first read)
    # The stream ends without ever yielding a frame once the deadline trips...
    with pytest.raises(StopIteration):
        next(gen)
    # ...and the device is actually released for the retry to re-roll.
    assert _wait_until(lambda: captures[0].released, timeout=2.0)
    assert "0" not in manager._captures
    kinds = [e["kind"] for e in camera_preview.preview_events()["events"] if e["camera_id"] == "0"]
    assert "first_frame_timeout" in kinds


def test_idle_linger_reuses_live_capture_across_polls(
    monkeypatch: pytest.MonkeyPatch, fake_captures: list[FakeVideoCapture]
) -> None:
    """THE churn fix: with the linger armed, a release followed by a re-acquire
    (a snapshot tile's next poll) reuses the still-running capture — no device
    teardown, no fresh open, so the teardown-then-open turbulence that births
    frame-dead sessions never happens."""
    monkeypatch.setattr(camera_preview, "_IDLE_LINGER", 30.0)
    manager = CameraPreviewManager()
    first = manager.snapshot(0)
    assert first[:2] == b"\xff\xd8"
    assert not fake_captures[0].released  # lingering, not torn down
    assert "0" in manager._captures  # still registered for reuse
    # Next poll (cache expired) must reuse the same live capture.
    monkeypatch.setattr(camera_preview, "_SNAPSHOT_CACHE_TTL", 0.0)
    second = manager.snapshot(0)
    assert second[:2] == b"\xff\xd8"
    assert len(fake_captures) == 1  # no second open, ever
    manager.stop_all(timeout=1.0)  # cleanup


def test_idle_linger_expiry_frees_the_device(
    monkeypatch: pytest.MonkeyPatch, fake_captures: list[FakeVideoCapture]
) -> None:
    """No re-acquire within the linger → deferred teardown actually frees the
    device and deregisters the entry (nothing leaks)."""
    monkeypatch.setattr(camera_preview, "_IDLE_LINGER", 0.15)
    manager = CameraPreviewManager()
    gen = manager.open_stream(0)
    assert b"--frame" in next(gen)
    gen.close()
    assert not fake_captures[0].released  # lingering
    assert _wait_until(lambda: fake_captures[0].released, timeout=3.0)  # expiry fired
    assert manager._captures == {}


def test_block_for_recording_tears_down_lingering_captures(
    monkeypatch: pytest.MonkeyPatch, fake_captures: list[FakeVideoCapture]
) -> None:
    """Recording always wins: a lingering (idle but alive) capture is force-
    released immediately by the session latch — the linger never makes the
    recorder wait."""
    monkeypatch.setattr(camera_preview, "_IDLE_LINGER", 60.0)
    manager = CameraPreviewManager()
    gen = manager.open_stream(0)
    assert b"--frame" in next(gen)
    gen.close()
    assert not fake_captures[0].released  # lingering with a long timer
    manager.block_for_recording(timeout=1.0)
    assert _wait_until(lambda: fake_captures[0].released, timeout=2.0)
    manager.resume_previews()


def test_snapshot_borrows_device_for_one_frame_and_releases(
    fake_captures: list[FakeVideoCapture],
) -> None:
    """The borrow-don't-reserve lane: a snapshot opens the device, takes one
    frame, and (with the linger disabled, as here) RELEASES it — no standing
    capture."""
    manager = CameraPreviewManager()
    jpeg = manager.snapshot(0)
    assert jpeg[:2] == b"\xff\xd8"  # JPEG SOI — a real encoded frame
    # The one-shot client was the only refcount holder: device freed.
    assert _wait_until(lambda: fake_captures[0].released, timeout=2.0)
    assert manager._captures == {}


def test_snapshot_shares_a_live_stream_without_second_open(
    fake_captures: list[FakeVideoCapture],
) -> None:
    """With a stream already running, a snapshot serves that capture's newest
    frame — no second device open, and the stream survives the snapshot."""
    manager = CameraPreviewManager()
    gen = manager.open_stream(0)
    assert b"--frame" in next(gen)
    jpeg = manager.snapshot(0)
    assert jpeg[:2] == b"\xff\xd8"
    assert len(fake_captures) == 1  # shared, not re-opened
    assert not fake_captures[0].released  # the stream still owns the device
    assert b"--frame" in next(gen)  # and still yields frames
    gen.close()


def test_snapshot_cache_coalesces_polls(fake_captures: list[FakeVideoCapture]) -> None:
    """Polls within _SNAPSHOT_CACHE_TTL are served from the cache: N tiles
    refreshing together cost ONE device borrow, not N."""
    manager = CameraPreviewManager()
    first = manager.snapshot(0)
    assert _wait_until(lambda: fake_captures[0].released, timeout=2.0)
    second = manager.snapshot(0)  # within TTL: cache hit, no new open
    assert second == first
    assert len(fake_captures) == 1


def test_snapshot_no_frame_raises_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A camera that opens but never delivers answers a legible CameraOpenError
    within the bounded snapshot wait — not a hang, not an empty 200."""
    monkeypatch.setattr(camera_preview, "_SNAPSHOT_FRAME_TIMEOUT", 0.3)
    gate = threading.Event()

    def factory(index: int, backend: int | None = None) -> FakeVideoCapture:
        return BlockingReadCapture(index, backend, gate=gate)

    monkeypatch.setattr(camera_preview.cv2, "VideoCapture", factory)
    manager = CameraPreviewManager()
    try:
        with pytest.raises(CameraOpenError) as exc:
            manager.snapshot(0)
        assert "no frame" in str(exc.value)
    finally:
        gate.set()  # let the wedged reader unwind (never hang pytest)


def test_preview_events_record_stream_lifecycle(
    fake_captures: list[FakeVideoCapture],
) -> None:
    """The forensic ring buffer captures a stream's full life: acquire → birth →
    open → first frame → release → reader exit. Filter by a distinctive id;
    the buffer is module-global and other tests write to it too."""
    manager = CameraPreviewManager()
    gen = manager.open_stream(7)
    assert b"--frame" in next(gen)
    gen.close()
    assert _wait_until(lambda: fake_captures[0].released, timeout=2.0)
    kinds = [e["kind"] for e in camera_preview.preview_events()["events"] if e["camera_id"] == "7"]
    for expected in ("acquire", "birth_start", "open_ok", "first_frame", "client_release", "reader_released"):
        assert expected in kinds, f"missing {expected} in {kinds}"


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


def test_burst_opens_share_one_enumeration_and_release_invalidates(
    monkeypatch: pytest.MonkeyPatch, fake_captures: list[FakeVideoCapture]
) -> None:
    """Two DIFFERENT cameras opened within a burst share one enumeration (the
    slow subprocess runs once, not once per tile — see _ENUM_CACHE_TTL), but any
    capture release drops the snapshot so the very next open re-resolves against
    live state — reopen-after-reshuffle correctness outranks burst sharing."""
    calls: list[int] = []

    def _list():
        calls.append(1)
        return [
            {"index": 3, "name": "USB Camera", "unique_id": "uvc-7749-front"},
            {"index": 5, "name": "USB Camera", "unique_id": "uvc-7749-wrist"},
        ]

    monkeypatch.setattr(camera_preview.camera_enumeration, "list_cameras", _list)
    manager = CameraPreviewManager()

    gen_front = manager.open_stream("uvc-7749-front")
    gen_wrist = manager.open_stream("uvc-7749-wrist")  # same burst: snapshot hit
    assert [c.index for c in fake_captures] == [3, 5]
    assert len(calls) == 1  # one enumeration served both births

    # Last client of the wrist camera detaches -> its reader releases -> the
    # snapshot is invalidated -> the reopen must re-enumerate, not trust it.
    gen_wrist.close()
    gen_wrist2 = manager.open_stream("uvc-7749-wrist")
    assert len(calls) == 2
    gen_wrist2.close()
    gen_front.close()


# ---------------------------------------------------------------------------
# GET /camera-preview/{index} — status codes and exclusivity
# ---------------------------------------------------------------------------


def test_camera_preview_409_while_recording(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(record, "recording_active", True)
    response = client.get("/camera-preview/0")
    assert response.status_code == 409
    assert "Recording" in response.json()["detail"]


def test_camera_snapshot_409_while_recording(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(record, "recording_active", True)
    response = client.get("/camera-snapshot/0")
    assert response.status_code == 409
    assert "Recording" in response.json()["detail"]


def test_camera_snapshot_returns_jpeg(client: TestClient, fake_captures: list[FakeVideoCapture]) -> None:
    response = client.get("/camera-snapshot/0")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content[:2] == b"\xff\xd8"
    # One-shot (linger disabled in tests): the endpoint borrowed and released.
    assert _wait_until(lambda: fake_captures[0].released, timeout=2.0)


def test_camera_preview_events_endpoint(client: TestClient) -> None:
    response = client.get("/camera-preview-events")
    assert response.status_code == 200
    body = response.json()
    assert "events" in body and "live_reader_threads" in body


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
