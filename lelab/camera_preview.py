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

# A capture that fails this many reads *in a row* is treated as dead: the shared
# device is force-released so every client stream ends cleanly and the next
# open_stream re-opens a fresh handle. Sized to ride out brief USB/AVFoundation
# hiccups (~2s at TARGET_FPS) without tearing down a momentarily-stuttering but
# otherwise healthy preview. A dead capture that is *not* released is exactly the
# wedge behind black previews + a device that vanishes from /available-cameras.
_MAX_READ_FAILURES = 30

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
    """One refcounted cv2.VideoCapture, shared by every client of a camera_id."""

    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        # The cv2 index this capture was opened at — resolved from camera_id at
        # open time (fresh per open for id-addressed cameras; the literal number
        # for the numeric fallback lane). None until the capture is opened.
        # Reported by held_healthy_indices for the enumeration merge.
        self.index: int | None = None
        self.cap: cv2.VideoCapture | None = None
        self.refcount = 0
        # Consecutive failed cap.read() count (guarded by ``lock``); reset on any
        # successful read. Reaching _MAX_READ_FAILURES force-releases the device.
        self.read_failures = 0
        # Held around every cap.read() AND every cap.release(): cv2 segfaults
        # if a capture is released while another thread is inside read(), so a
        # release must never happen outside this lock.
        self.lock = threading.Lock()
        # Set by stop_all()/force-release so every client generator exits
        # promptly instead of grabbing another frame.
        self.stop = threading.Event()


class CameraPreviewManager:
    """Refcounted, shared MJPEG streaming of the backend's cv2 cameras.

    One cv2.VideoCapture per ``camera_id`` (the stable hardware unique_id, or
    a numeric string for the index-fallback lane), shared across all connected
    preview clients; the last client detaching releases the device. The cv2
    index is resolved from the id at every fresh capture open, so a reopen
    after a hotplug reshuffle finds the device at its NEW index. Recording and
    teleoperation always win: their start paths call :meth:`stop_all` /
    :meth:`block_for_recording`, which tell every client generator to exit and
    force-release any capture a stalled client would otherwise keep holding.
    """

    def __init__(self) -> None:
        self._captures: dict[str, _SharedCapture] = {}
        # Guards the registry dict and the refcounts; never held during device
        # I/O (open/read/release happen under the per-camera lock instead).
        self._registry_lock = threading.Lock()
        # Latched True while a recording session owns the cv2 devices (set by
        # block_for_recording, cleared by resume_previews). Checked in _acquire
        # under the SAME registry lock that block_for_recording snapshots under,
        # so a preview open that races the session's force-release is refused
        # rather than re-acquiring the device the recorder is about to grab —
        # the TOCTOU the recording_active 409 gate alone can't close (that flag
        # lives in the record module, so the route's check and the manager's
        # open are not atomic). See block_for_recording. Guarded by
        # _registry_lock.
        self._blocked = False

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

        * Priming runs :meth:`_acquire` synchronously, so an open failure raises
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

        Sets each capture's stop event so client generators exit on their next
        frame, waits up to ``timeout`` seconds for them to detach, then
        force-releases anything still held — a client stalled mid-yield on a
        dead connection must not keep the device away from recording/teleop.
        The force-release happens under the per-index lock, so it can never
        pull the capture out from under a thread inside cap.read().
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
        """Signal each capture to stop, wait briefly for clients to detach, then
        force-release any that a stalled/dead client is still holding. Shared by
        :meth:`stop_all` and :meth:`block_for_recording`."""
        if not entries:
            return
        for entry in entries:
            entry.stop.set()
        deadline = time.monotonic() + timeout
        for entry in entries:
            while time.monotonic() < deadline:
                with self._registry_lock:
                    detached = self._captures.get(entry.camera_id) is not entry
                if detached:
                    break
                time.sleep(0.02)
            if self._force_release(entry):
                logger.warning(
                    "Force-releasing camera %s preview capture (a client is still attached)",
                    entry.camera_id,
                )

    def held_healthy_indices(self) -> list[int]:
        """Indices whose shared capture is currently open and not force-released.

        The enumeration probe (/available-cameras) fresh-opens each cv2 index on
        Linux/Windows/generic; a device this manager already holds is *busy*, so
        the probe's open fails and the camera silently drops from the list even
        though it is alive. Enumeration merges these back in. A capture that has
        gone dead is force-released (see :meth:`_frames`) and deregistered here,
        so it is *not* reported — letting the probe re-open it fresh.
        """
        with self._registry_lock:
            return sorted(
                e.index for e in self._captures.values() if e.cap is not None and e.index is not None
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
            if entry is None:
                entry = _SharedCapture(camera_id)
                self._captures[camera_id] = entry
            entry.refcount += 1
        try:
            with entry.lock:
                if entry.cap is None:
                    # Fresh open → fresh id→index resolution (see _resolve_cv2_index):
                    # the integer index lives only between here and VideoCapture().
                    index = self._resolve_cv2_index(camera_id)
                    cap = cv2.VideoCapture(index, _CV2_BACKEND)
                    if not cap.isOpened():
                        cap.release()
                        raise CameraOpenError(
                            f"Camera {camera_id} could not be opened — it may be unplugged or in use "
                            "by another application."
                        )
                    entry.cap = cap
                    entry.index = index
                    entry.read_failures = 0
        except Exception:
            self._release(entry)
            raise
        return entry

    def _release(self, entry: _SharedCapture) -> None:
        with self._registry_lock:
            entry.refcount -= 1
            if entry.refcount > 0:
                return
            # Last client: deregister before the (slow) device release so a new
            # client never latches onto a capture that is being torn down.
            if self._captures.get(entry.camera_id) is entry:
                del self._captures[entry.camera_id]
        with entry.lock:
            if entry.cap is not None:
                entry.cap.release()
                entry.cap = None
                logger.info("Released camera %s preview capture (last client detached)", entry.camera_id)

    def _force_release(self, entry: _SharedCapture) -> bool:
        """Deregister and release a capture regardless of its refcount.

        Used by stop_all and by the read-failure streak detector: a device that
        is dead (or that a stalled client is holding open) must be freed even
        though a refcount is still outstanding. Deregistering first means a
        lagging client's later :meth:`_release` becomes a no-op and the next
        :meth:`open_stream` starts from a fresh entry (fresh stop event, fresh
        id→index resolution). The release runs under the per-camera lock, so it
        can never pull the capture out from under a thread inside cap.read().

        Returns True if a capture was actually released.
        """
        entry.stop.set()
        with self._registry_lock:
            if self._captures.get(entry.camera_id) is entry:
                del self._captures[entry.camera_id]
        with entry.lock:
            if entry.cap is not None:
                entry.cap.release()
                entry.cap = None
                return True
        return False

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
        """
        entry = self._acquire(camera_id)  # may raise CameraOpenError (endpoint -> 503); self-cleans
        try:
            interval = 1.0 / TARGET_FPS
            # Priming sentinel: consumed by open_stream, never sent to the
            # client. Its only job is to leave the generator suspended *inside*
            # this try, so close()/GC on disconnect runs the finally below.
            yield
            while not entry.stop.is_set():
                started = time.monotonic()
                with entry.lock:
                    if entry.cap is None:  # force-released by stop_all / streak
                        break
                    ok, frame = entry.cap.read()
                    if ok:
                        entry.read_failures = 0
                        streak_dead = False
                    else:
                        entry.read_failures += 1
                        streak_dead = entry.read_failures >= _MAX_READ_FAILURES
                if not ok:
                    if streak_dead:
                        # Persistent failures: the device is wedged. Release the
                        # SHARED capture so every client (incl. a stalled one
                        # holding a refcount) ends and the next open re-opens
                        # fresh — otherwise a dead handle keeps the device busy
                        # and blanks the preview indefinitely.
                        logger.warning(
                            "Camera %s failed %d consecutive reads; releasing shared capture",
                            camera_id,
                            entry.read_failures,
                        )
                        self._force_release(entry)
                        break
                    # Transient hiccup: pace and retry rather than tearing the
                    # preview down on a single dropped frame.
                    if entry.stop.wait(interval):
                        break
                    continue
                ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if not ok:
                    continue
                data = jpeg.tobytes()
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(data)}\r\n\r\n".encode()
                    + data
                    + b"\r\n"
                )
                # Pace to ~TARGET_FPS. Waiting on the stop event (not a plain
                # sleep) lets stop_all cut the frame interval short.
                remaining = interval - (time.monotonic() - started)
                if remaining > 0 and entry.stop.wait(remaining):
                    break
        finally:
            self._release(entry)


camera_preview_manager = CameraPreviewManager()
