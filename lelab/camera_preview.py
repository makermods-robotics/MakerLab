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
"""Backend MJPEG camera previews for headless deployments.

The frontend's preview tiles normally use getUserMedia, which only sees the
*viewing* machine's cameras. When lelab runs on a headless host (e.g. a Jetson)
with the cameras plugged into the server, no browser deviceId ever matches, so
the tiles would show "No camera selected" forever. This module streams the
backend's cv2 cameras as multipart/x-mixed-replace MJPEG (GET
/camera-preview/{camera_id}) so the tiles can fall back to an ``<img>``.

Cameras are addressed by ``camera_id`` — the stable hardware ``unique_id``
where the platform exposes one (see lelab/camera_enumeration.py). The cv2
integer index is a private detail resolved FRESH on every capture open
(camera_enumeration.resolve_index), so a preview keyed by id can never show
whatever device slid into a stale integer slot after a USB hotplug reshuffle —
even mid-session across a release/reopen. A purely-numeric ``camera_id`` is the
back-compat/fallback lane for platforms and saved entries without stable ids:
it is used directly as the cv2 index, unresolved, exactly as before.

Reader-thread architecture (the fix for "a yanked camera clogs the manager").
Every shared capture owns ONE dedicated reader thread that is the *only* code
that touches cv2 (open → read → JPEG-encode → publish the latest frame under a
Condition). Client generators are pure consumers: they wait-with-timeout on
that Condition for a fresher frame and yield it; they never call cv2. Two
consequences make a hung device recoverable where the old single-lock design
wedged:

* A consumer can ALWAYS be finalized — it is never parked inside ``cap.read()``,
  so a disconnect is deliverable at every wait/yield and a refcount can never be
  pinned by a blocked read.
* Force-release paths (:meth:`stop_all`, :meth:`block_for_recording`) and the
  frame-age watchdog operate on the registry (deregister + signal) under locks
  the reader never holds during I/O, so they return within a bounded timeout
  even when the reader is genuinely stuck in ``cap.read()``. A reader that does
  not confirm its own release within the grace window is *abandoned* (its thread
  + handle leak, bounded by the process) rather than released from another
  thread — cv2 segfaults if a capture is released while a thread is inside
  read(), and a replugged device gets a fresh id/entry anyway.
"""

import logging
import platform
import threading
import time

import cv2

from . import camera_enumeration

logger = logging.getLogger(__name__)

# A preview is a thumbnail, not a recording: ~15 fps at JPEG quality 70 keeps
# per-client bandwidth modest without visible stutter.
TARGET_FPS = 15.0
JPEG_QUALITY = 70

# A capture whose reader fails this many reads *in a row* is treated as dead: the
# reader releases its own handle and exits, every client stream ends, and the
# next open_stream re-opens a fresh handle. Sized to ride out brief
# USB/AVFoundation hiccups (~2s at TARGET_FPS) without tearing down a
# momentarily-stuttering but otherwise healthy preview.
_MAX_READ_FAILURES = 30

# Frame-age watchdog: if the reader publishes no new frame within this many
# seconds, a waiting consumer declares the capture DEAD, deregisters it, and
# ends its stream. This is the recovery path for a reader wedged *inside*
# cap.read() (physically unplugged mid-stream, wedged AVFoundation) — the read
# never returns, so no failure streak ever accrues, but frames stop arriving.
# Must comfortably exceed the frame interval so a healthy-but-slow camera is not
# reaped.
_FRAME_DEADLINE = 2.5

# How long a consumer's open (open_stream priming) waits for its reader thread to
# report open success/failure before giving up with a CameraOpenError. Longer
# than the enumeration subprocess timeout (camera_enumeration, ~10s) so a slow
# id→index resolution isn't mistaken for a failure.
_OPEN_TIMEOUT = 15.0

# How long a force-release / last-detach waits for the reader to confirm it
# released its own capture before abandoning it. Bounds every wait on the
# request/recording-start path: an alive reader confirms in well under this; a
# reader wedged in read() never confirms and is abandoned when the grace lapses.
_RELEASE_GRACE = 1.0

# Reopen resilience: a device freed a beat ago (an abrupt force-release for a
# recording claim) can report "busy" for a moment. The first fresh open of a
# just-released id gets ONE short retry after this delay before surfacing
# CameraOpenError — smooths the tile-retry experience without masking a real
# failure. Eligibility is limited to opens within _REOPEN_RETRY_WINDOW seconds of
# the id's last release, so a genuinely-unpluggable device fails fast.
_REOPEN_RETRY_DELAY = 0.5
_REOPEN_RETRY_WINDOW = 2.0

# Same per-platform backend pin as recording (record._platform_backend) and the
# /available-cameras enumeration: CAP_ANY can pick different backends across
# calls on macOS, silently reordering indices, so a preview could show a
# different physical device than the one the recorder will open.
_CV2_BACKEND = {
    "Darwin": cv2.CAP_AVFOUNDATION,
    "Linux": cv2.CAP_V4L2,
    "Windows": cv2.CAP_DSHOW,
}.get(platform.system(), cv2.CAP_ANY)


class CameraOpenError(RuntimeError):
    """The requested camera could not be opened."""


class _SharedCapture:
    """One cv2.VideoCapture + its dedicated reader thread, shared by every client
    of a camera_id.

    The reader thread is the sole owner of the cv2 handle: it opens it, reads and
    JPEG-encodes frames, publishes the latest frame under ``_cond``, and releases
    the handle itself on exit. No other thread ever calls a cv2 method on
    ``cap`` — that is what lets a disconnect always finalize and a force-release
    never block on a hung read.
    """

    def __init__(
        self, manager: "CameraPreviewManager", camera_id: str, retry_open_on_busy: bool = False
    ) -> None:
        self._manager = manager
        self.camera_id = camera_id
        # The cv2 index this capture was opened at — resolved from camera_id by
        # the reader at open time (fresh per open for id-addressed cameras; the
        # literal number for the numeric fallback lane). None until opened.
        # Reported by held_healthy_indices for the enumeration merge.
        self.index: int | None = None
        # Owned exclusively by the reader thread. Other threads only read it for
        # a None/not-None liveness hint (held_healthy_indices); they never call
        # its methods.
        self.cap: cv2.VideoCapture | None = None
        self.refcount = 0
        # One short retry if the fresh open reports busy (see _REOPEN_RETRY_*).
        self.retry_open_on_busy = retry_open_on_busy

        # Frame handoff + lifecycle state, all guarded by ``_cond``. ``_cond`` is
        # held only for these quick handoffs and NEVER during cv2 I/O, so a
        # thread waiting on it can never be blocked behind a wedged read().
        self._cond = threading.Condition()
        self.latest_jpeg: bytes | None = None
        # Monotonically increasing frame counter; a consumer yields whenever it
        # advances past the seq it last delivered (so it always ships the newest
        # frame and never backs up behind a slow client).
        self.frame_seq = 0
        self.last_frame_ts = 0.0
        self.opened = False  # reader finished its open attempt (result in open_ok)
        self.open_ok = False
        self.open_error: CameraOpenError | None = None
        self.dead = False  # reader has exited / capture declared dead
        # Set by stop_all / block_for_recording / the watchdog / last-detach so
        # the reader stops and every consumer generator exits promptly.
        self.stop = threading.Event()
        # Set by the reader once it has released its own capture (or once an open
        # attempt failed without acquiring one). Force-release waits on this with
        # a bounded grace; a reader that never sets it is abandoned.
        self.released = threading.Event()
        self._reader = threading.Thread(target=self._reader_run, name=f"cam-preview-{camera_id}", daemon=True)

    def start(self) -> None:
        self._reader.start()

    # -- reader thread (sole cv2 owner) --------------------------------------

    def _reader_run(self) -> None:
        try:
            index = self._manager._resolve_cv2_index(self.camera_id)
            cap = self._open_cv2(index)
        except CameraOpenError as exc:
            # Open failed: publish the error so the waiting open_stream can raise
            # it (→ 503), mark dead, deregister, and confirm "released" (there is
            # no handle to release). No frame ever streams.
            with self._cond:
                self.open_error = exc
                self.opened = True
                self.dead = True
                self._cond.notify_all()
            self._manager._deregister_if_current(self)
            self.released.set()
            return
        with self._cond:
            self.cap = cap
            self.index = index
            self.open_ok = True
            self.opened = True
            self.last_frame_ts = time.monotonic()
            self._cond.notify_all()
        self._read_loop(cap)

    def _open_cv2(self, index: int) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(index, _CV2_BACKEND)
        if cap.isOpened():
            return cap
        cap.release()
        if self.retry_open_on_busy:
            time.sleep(_REOPEN_RETRY_DELAY)
            cap = cv2.VideoCapture(index, _CV2_BACKEND)
            if cap.isOpened():
                return cap
            cap.release()
        raise CameraOpenError(
            f"Camera {self.camera_id} could not be opened — it may be unplugged or in use "
            "by another application."
        )

    def _read_loop(self, cap: cv2.VideoCapture) -> None:
        interval = 1.0 / TARGET_FPS
        failures = 0
        try:
            while not self.stop.is_set():
                started = time.monotonic()
                ok, frame = cap.read()  # ONLY this thread touches cv2; no lock held
                if ok:
                    failures = 0
                    enc_ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    if enc_ok:
                        data = jpeg.tobytes()
                        with self._cond:
                            self.latest_jpeg = data
                            self.frame_seq += 1
                            self.last_frame_ts = time.monotonic()
                            self._cond.notify_all()
                else:
                    failures += 1
                    if failures >= _MAX_READ_FAILURES:
                        logger.warning(
                            "Camera %s failed %d consecutive reads; reader releasing shared capture",
                            self.camera_id,
                            failures,
                        )
                        break
                # Pace to ~TARGET_FPS. Waiting on the stop event (not a plain
                # sleep) lets stop_all cut the frame interval short.
                remaining = interval - (time.monotonic() - started)
                if remaining > 0 and self.stop.wait(remaining):
                    break
        finally:
            # Alive reader releases its OWN capture (same thread that reads it —
            # never a cross-thread release, so the segfault guard is honored).
            cap.release()
            self._manager._mark_released(self.camera_id)
            with self._cond:
                self.cap = None
                self.dead = True
                self._cond.notify_all()
            self._manager._deregister_if_current(self)
            self.released.set()


class CameraPreviewManager:
    """Refcounted, shared MJPEG streaming of the backend's cv2 cameras.

    One :class:`_SharedCapture` (cv2.VideoCapture + reader thread) per
    ``camera_id`` (the stable hardware unique_id, or a numeric string for the
    index-fallback lane), shared across all connected preview clients; the last
    client detaching stops the reader, which releases the device. The cv2 index
    is resolved from the id at every fresh capture open, so a reopen after a
    hotplug reshuffle finds the device at its NEW index. Recording and
    teleoperation always win: their start paths call :meth:`stop_all` /
    :meth:`block_for_recording`, which deregister every capture and force-release
    it under a bounded timeout — a stalled client, or a reader wedged in
    ``cap.read()``, can never keep the device away from recording/teleop.
    """

    def __init__(self) -> None:
        self._captures: dict[str, _SharedCapture] = {}
        # Guards the registry dict, the refcounts, the block flag, and the
        # last-release timestamps. Never held during device I/O (that lives on
        # the reader thread) nor simultaneously with any entry's _cond, so it can
        # never be blocked behind a hung read and there is no lock-ordering risk.
        self._registry_lock = threading.Lock()
        # Latched True while a recording session owns the cv2 devices (set by
        # block_for_recording, cleared by resume_previews). Checked in _acquire
        # under the SAME registry lock that block_for_recording snapshots under,
        # so a preview open that races the session's force-release is refused
        # rather than re-acquiring the device the recorder is about to grab —
        # the TOCTOU the recording_active 409 gate alone can't close (that flag
        # lives in the record module, so the route's check and the manager's
        # open are not atomic). See block_for_recording.
        self._blocked = False
        # camera_id → monotonic time the reader last released that id's capture.
        # Lets the *next* open give a just-released (possibly still-busy) device
        # one short retry. Guarded by _registry_lock.
        self._last_release: dict[str, float] = {}

    def open_stream(self, camera_id: str | int):
        """Open (or share) the capture for ``camera_id`` and return a frame generator.

        ``camera_id`` is the camera's stable ``unique_id``; a purely-numeric
        value (or an int, for back-compat callers) is the fallback lane and is
        used directly as the cv2 index. Raises :class:`CameraOpenError` when
        the device can't be opened or the id matches no connected camera. The
        generator yields ``multipart/x-mixed-replace`` JPEG parts (boundary
        ``frame``) at ~TARGET_FPS and drops its reference on exit — client
        disconnect, :meth:`stop_all`, or the device dying mid-stream.

        The refcount is acquired *inside* the returned generator's body, and the
        generator is primed here (advanced past a one-shot sentinel yield, so it
        has run :meth:`_acquire` but not yet produced a frame) before it is
        handed to Starlette. That is load-bearing, not cosmetic:

        * Priming runs :meth:`_acquire` synchronously, which blocks until the
          reader thread reports its open result — so an open failure raises
          :class:`CameraOpenError` here and the endpoint can answer 503.
        * Priming guarantees the generator has *started*, so its ``finally`` runs
          when the client disconnects. Starlette streams a sync generator via
          ``iterate_in_threadpool`` and never calls ``.close()`` on it; a
          *never-started* generator skips ``finally`` entirely (CPython raises
          GeneratorExit before the first bytecode), so acquiring the refcount
          before the generator ran — as the old code did — leaked the count and
          wedged the device on any disconnect that landed before the first frame
          (React StrictMode remounts, HMR, fast tab switches). The sentinel
          ``yield`` is consumed here and never reaches Starlette, so the client
          only ever sees real JPEG parts.
        """
        camera_id = str(camera_id)
        gen = self._frames(camera_id)
        try:
            next(gen)  # run _acquire (raises CameraOpenError -> 503) and stop at the sentinel
        except StopIteration as exc:
            # Ended before the sentinel — only possible if _frames changes shape;
            # the generator has already run its finally, so nothing leaks.
            raise CameraOpenError(f"Camera {camera_id} could not be opened.") from exc
        return gen

    def stop_all(self, timeout: float = 1.0) -> None:
        """Stop every preview stream and release every capture.

        Deregisters each capture and signals its reader to stop, then waits up to
        ``timeout`` seconds for each reader to confirm it released its own device.
        A reader that does not confirm — one wedged inside ``cap.read()`` — is
        abandoned rather than released from this thread (the segfault guard
        stands). Because deregistration and signalling touch only the registry
        lock and the per-entry Condition (never a lock a reader holds during
        I/O), this returns within ``timeout`` even against a hung reader.
        """
        with self._registry_lock:
            entries = list(self._captures.values())
        self._stop_and_release(entries, timeout)

    def block_for_recording(self, timeout: float = 1.0) -> None:
        """Stop every preview AND latch new opens off for a recording session.

        The same force-release as :meth:`stop_all`, but it atomically sets the
        block flag in the SAME critical section that snapshots the live captures.
        That closes the window :meth:`stop_all` leaves open: without the latch, a
        preview open that races the release (a retrying ``<img>`` re-requesting
        right after the device is freed) re-acquires the very device the recorder
        is about to open, starving it to actual_fps=5.0. Because the flag is set
        before the snapshot under ``_registry_lock``, any concurrent
        :meth:`_acquire` either registered before us (so it is in the snapshot and
        gets force-released) or observes the flag and is rejected — it can never
        slip a fresh capture in behind us. Cleared by :meth:`resume_previews` when
        the session ends.
        """
        with self._registry_lock:
            self._blocked = True
            entries = list(self._captures.values())
        self._stop_and_release(entries, timeout)

    def resume_previews(self) -> None:
        """Clear the recording block so preview opens are served again.

        Called on session end (including the failed-to-start path). Idempotent:
        clearing an already-clear flag is a no-op, so a stray extra call after a
        session that never blocked is harmless.
        """
        with self._registry_lock:
            self._blocked = False

    def _stop_and_release(self, entries: list[_SharedCapture], timeout: float) -> None:
        """Deregister each capture, signal its reader to stop, then wait a bounded
        grace for each reader to confirm release — abandoning any that doesn't.
        Shared by :meth:`stop_all` and :meth:`block_for_recording`.

        Deregistering first means a lagging client's later :meth:`_release`
        becomes a no-op and the next :meth:`open_stream` starts from a fresh
        entry (fresh reader, fresh id→index resolution). Signalling wakes both the
        reader (to stop) and any consumers parked on the Condition (to end their
        streams). Nothing here touches cv2 or waits on a lock a reader holds
        during read(), so it never blocks on a hung device.
        """
        if not entries:
            return
        for entry in entries:
            entry.stop.set()
            with self._registry_lock:
                if self._captures.get(entry.camera_id) is entry:
                    del self._captures[entry.camera_id]
            with entry._cond:
                entry._cond.notify_all()
        deadline = time.monotonic() + timeout
        for entry in entries:
            remaining = deadline - time.monotonic()
            if remaining > 0:
                entry.released.wait(remaining)
            if not entry.released.is_set():
                logger.warning(
                    "Camera %s reader did not confirm release within %.2fs; abandoning the wedged capture",
                    entry.camera_id,
                    timeout,
                )

    def held_healthy_indices(self) -> list[int]:
        """Indices whose shared capture is currently open and not dead/stopping.

        The enumeration probe (/available-cameras) fresh-opens each cv2 index on
        Linux/Windows/generic; a device this manager already holds is *busy*, so
        the probe's open fails and the camera silently drops from the list even
        though it is alive. Enumeration merges these back in. A capture that has
        gone dead (reader exited, or the watchdog declared it) is deregistered, so
        it is *not* reported — letting the probe re-open it fresh.
        """
        with self._registry_lock:
            entries = list(self._captures.values())
        return sorted(
            e.index
            for e in entries
            if e.open_ok and not e.dead and not e.stop.is_set() and e.cap is not None and e.index is not None
        )

    @staticmethod
    def _resolve_cv2_index(camera_id: str) -> int:
        """The cv2 index to open for ``camera_id`` — resolved NOW, per open.

        Numeric fallback lane: a purely-numeric id is the index itself (legacy
        callers / platforms without stable ids). Otherwise the id is resolved
        against a FRESH enumeration via camera_enumeration.resolve_index (the
        single id→index lookup, shared with record start), so a reopen after a
        hotplug reshuffle lands on the device's new index instead of streaming
        whatever slid into the old slot. An unknown id becomes a legible
        :class:`CameraOpenError` (endpoint → 503).
        """
        if camera_id.isdigit():
            return int(camera_id)
        try:
            index, _ = camera_enumeration.resolve_index(camera_id)
        except camera_enumeration.CameraNotConnectedError as exc:
            raise CameraOpenError(str(exc)) from exc
        return index

    def _mark_released(self, camera_id: str) -> None:
        """Record that a reader just released ``camera_id``'s capture, so the next
        open of the same id is eligible for the busy-retry (see _REOPEN_RETRY_*)."""
        with self._registry_lock:
            self._last_release[camera_id] = time.monotonic()

    def _deregister_if_current(self, entry: _SharedCapture) -> None:
        """Drop ``entry`` from the registry iff it is still the live entry for its
        id. Called by the reader when it dies so a fresh open never latches onto a
        dead capture, and idempotent with the other deregister paths."""
        with self._registry_lock:
            if self._captures.get(entry.camera_id) is entry:
                del self._captures[entry.camera_id]

    def _declare_dead(self, entry: _SharedCapture) -> None:
        """Frame-age watchdog: a consumer saw no fresh frame within the deadline.

        Mark the capture dead, deregister it (so it stops blocking reopen and
        drops out of held_healthy_indices), and wake every consumer and the
        reader. If the reader is merely slow-failing it will release its own
        capture on exit; if it is genuinely stuck in ``cap.read()`` it is
        abandoned — this never calls release across threads.
        """
        entry.stop.set()
        with self._registry_lock:
            if self._captures.get(entry.camera_id) is entry:
                del self._captures[entry.camera_id]
        with entry._cond:
            entry.dead = True
            entry._cond.notify_all()
        logger.warning(
            "Camera %s produced no frame within %.1fs; declaring the capture dead",
            entry.camera_id,
            _FRAME_DEADLINE,
        )

    def _acquire(self, camera_id: str) -> _SharedCapture:
        with self._registry_lock:
            # Refuse to open (or register) a capture while a recording session
            # holds the devices. Same lock/critical-section that
            # block_for_recording sets the flag and snapshots under, so an open
            # either registers BEFORE the block (and is force-released by the
            # snapshot) or is rejected here — never sneaks a fresh capture in
            # after the recorder freed the device. Surfaces as a 503 the tile
            # retries; the route's recording_active check already answers 409
            # for every non-racing open.
            if self._blocked:
                raise CameraOpenError(f"Camera {camera_id} is reserved for the active recording session.")
            entry = self._captures.get(camera_id)
            created = entry is None
            if created:
                # A device released within the retry window may still be busy for
                # a beat; let this fresh entry's reader retry once (see _open_cv2).
                ts = self._last_release.get(camera_id)
                retry = ts is not None and (time.monotonic() - ts) <= _REOPEN_RETRY_WINDOW
                entry = _SharedCapture(self, camera_id, retry_open_on_busy=retry)
                self._captures[camera_id] = entry
            entry.refcount += 1
        if created:
            entry.start()  # launch the reader; it owns the cv2 open
        # Wait (bounded) for the reader to report its open result. This is the
        # synchronous open the endpoint needs for its 503, but it can never block
        # forever: _OPEN_TIMEOUT caps it and a failed/timed-out open self-cleans.
        with entry._cond:
            deadline = time.monotonic() + _OPEN_TIMEOUT
            while not entry.opened:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                entry._cond.wait(remaining)
            open_ok = entry.open_ok
            open_error = entry.open_error
        if not open_ok:
            self._release(entry)
            raise open_error or CameraOpenError(f"Camera {camera_id} could not be opened.")
        return entry

    def _release(self, entry: _SharedCapture) -> None:
        with self._registry_lock:
            entry.refcount -= 1
            if entry.refcount > 0:
                return
            # Last client: deregister before signalling the reader so a new
            # client never latches onto a capture that is being torn down.
            last = self._captures.get(entry.camera_id) is entry
            if last:
                del self._captures[entry.camera_id]
        # Tell the reader to stop and release its own capture, then wait a bounded
        # grace for it to confirm. The consumer never touches cv2, so this thread
        # is never inside read(); a reader wedged in read() is abandoned when the
        # grace lapses rather than joined. Waiting lets a subsequent same-index
        # open find the device actually free instead of transiently busy.
        entry.stop.set()
        with entry._cond:
            entry._cond.notify_all()
        entry.released.wait(_RELEASE_GRACE)

    def _frames(self, camera_id: str):
        """Acquire the shared capture for ``camera_id`` and yield multipart JPEG parts.

        The acquire lives *inside* this generator (paired with the ``finally``)
        so acquire and release balance for every path a started generator can
        take — client disconnect (GeneratorExit on close/GC), :meth:`stop_all`,
        or the device dying mid-stream. :meth:`open_stream` primes the generator
        past the one-shot ``yield`` sentinel below, which both runs the acquire
        synchronously (for the 503) and guarantees the ``finally`` will run. The
        acquire is *outside* the ``try`` because it self-cleans on failure; a
        finally here would double-release.

        The consumer only ever waits on the capture's Condition for a fresher
        frame and yields it — it never calls cv2 — so it can always be finalized,
        and a reader wedged in ``cap.read()`` shows up here as a frame-age timeout
        that trips the watchdog rather than as an unkillable stream.
        """
        entry = self._acquire(camera_id)  # may raise CameraOpenError (endpoint -> 503); self-cleans
        try:
            # Priming sentinel: consumed by open_stream, never sent to the
            # client. Its only job is to leave the generator suspended *inside*
            # this try, so close()/GC on disconnect runs the finally below.
            yield
            last_seq = 0
            while True:
                with entry._cond:
                    wait_start = time.monotonic()
                    while entry.frame_seq == last_seq and not entry.dead and not entry.stop.is_set():
                        remaining = _FRAME_DEADLINE - (time.monotonic() - wait_start)
                        if remaining <= 0:
                            break
                        entry._cond.wait(remaining)
                    new_frame = entry.frame_seq != last_seq and entry.latest_jpeg is not None
                    if new_frame:
                        data = entry.latest_jpeg
                        last_seq = entry.frame_seq
                    ended = entry.dead or entry.stop.is_set()
                if new_frame:
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(data)}\r\n\r\n".encode()
                        + data
                        + b"\r\n"
                    )
                    continue
                if not ended:
                    # No fresh frame within the deadline and the reader hasn't
                    # stopped/died on its own → it is wedged inside read(). Trip
                    # the watchdog: deregister, signal, and end this stream.
                    self._declare_dead(entry)
                break
        finally:
            self._release(entry)


camera_preview_manager = CameraPreviewManager()
