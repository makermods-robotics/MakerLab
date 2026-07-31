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
*viewing* machine's cameras. When makerlab runs on a headless host (e.g. a Jetson)
with the cameras plugged into the server, no browser deviceId ever matches, so
the tiles would show "No camera selected" forever. This module streams the
backend's cv2 cameras as multipart/x-mixed-replace MJPEG (GET
/camera-preview/{index}) so the tiles can fall back to an ``<img>``.
"""

import logging
import platform
import threading
import time

import cv2

logger = logging.getLogger(__name__)

# A preview is a thumbnail, not a recording: ~15 fps at JPEG quality 70 keeps
# per-client bandwidth modest without visible stutter.
TARGET_FPS = 15.0
JPEG_QUALITY = 70

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
    """The camera at the requested index could not be opened."""


class _SharedCapture:
    """One refcounted cv2.VideoCapture, shared by every client of an index."""

    def __init__(self, index: int) -> None:
        self.index = index
        self.cap: cv2.VideoCapture | None = None
        self.refcount = 0
        # Held around every cap.read() AND every cap.release(): cv2 segfaults
        # if a capture is released while another thread is inside read(), so a
        # release must never happen outside this lock.
        self.lock = threading.Lock()
        # Set by stop_all() so every client generator exits promptly instead of
        # grabbing another frame.
        self.stop = threading.Event()


class CameraPreviewManager:
    """Refcounted, shared MJPEG streaming of the backend's cv2 cameras.

    One cv2.VideoCapture per camera index, shared across all connected preview
    clients; the last client detaching releases the device. Recording and
    teleoperation always win: their start paths call :meth:`stop_all`, which
    tells every client generator to exit and force-releases any capture a
    stalled client would otherwise keep holding.
    """

    def __init__(self) -> None:
        self._captures: dict[int, _SharedCapture] = {}
        # Guards the registry dict and the refcounts; never held during device
        # I/O (open/read/release happen under the per-index lock instead).
        self._registry_lock = threading.Lock()

    def open_stream(self, index: int):
        """Open (or share) the capture for ``index`` and return a frame generator.

        Raises :class:`CameraOpenError` when the device can't be opened. The
        generator yields ``multipart/x-mixed-replace`` JPEG parts (boundary
        ``frame``) at ~TARGET_FPS and drops its reference on exit — client
        disconnect, :meth:`stop_all`, or the device dying mid-stream.
        """
        entry = self._acquire(index)
        return self._frames(entry)

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
        if not entries:
            return
        for entry in entries:
            entry.stop.set()
        deadline = time.monotonic() + timeout
        for entry in entries:
            while time.monotonic() < deadline:
                with self._registry_lock:
                    detached = self._captures.get(entry.index) is not entry
                if detached:
                    break
                time.sleep(0.02)
            with entry.lock:
                if entry.cap is not None:
                    logger.warning(
                        "Force-releasing camera %d preview capture (a client is still attached)",
                        entry.index,
                    )
                    entry.cap.release()
                    entry.cap = None
            # Deregister so a lagging client's _release becomes a no-op and a
            # future preview starts from a fresh entry (fresh stop event).
            with self._registry_lock:
                if self._captures.get(entry.index) is entry:
                    del self._captures[entry.index]

    def _acquire(self, index: int) -> _SharedCapture:
        with self._registry_lock:
            entry = self._captures.get(index)
            if entry is None:
                entry = _SharedCapture(index)
                self._captures[index] = entry
            entry.refcount += 1
        try:
            with entry.lock:
                if entry.cap is None:
                    cap = cv2.VideoCapture(index, _CV2_BACKEND)
                    if not cap.isOpened():
                        cap.release()
                        raise CameraOpenError(
                            f"Camera {index} could not be opened — it may be unplugged or in use "
                            "by another application."
                        )
                    entry.cap = cap
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
            if self._captures.get(entry.index) is entry:
                del self._captures[entry.index]
        with entry.lock:
            if entry.cap is not None:
                entry.cap.release()
                entry.cap = None
                logger.info("Released camera %d preview capture (last client detached)", entry.index)

    def _frames(self, entry: _SharedCapture):
        """Yield multipart JPEG parts from a shared capture until stopped."""
        interval = 1.0 / TARGET_FPS
        try:
            while not entry.stop.is_set():
                started = time.monotonic()
                with entry.lock:
                    if entry.cap is None:  # force-released by stop_all
                        break
                    ok, frame = entry.cap.read()
                if not ok:
                    logger.warning("Camera %d stopped producing frames; ending preview", entry.index)
                    break
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
