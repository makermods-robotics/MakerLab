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
integer index is a private detail resolved at every capture open
(camera_enumeration.resolve_index; opens within a short burst share one
enumeration snapshot — see _ENUM_CACHE_TTL), so a preview keyed by id can
never show whatever device slid into a stale integer slot after a USB hotplug
reshuffle — even mid-session across a release/reopen. A purely-numeric ``camera_id`` is the
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
from collections import deque

import cv2

from . import camera_enumeration

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Forensic event log. Direct-cv2 probes (2026-07-09) proved the OS/hardware
# handle everything this module does — 3 concurrent streams @30fps, zero-gap
# reopen churn, stacked same-device captures — yet tiles still intermittently
# black-screen through the manager. So every lifecycle transition is recorded
# here (cheap ring buffer, thread-safe) and served by GET
# /camera-preview-events, so a black tile can be diagnosed from its actual
# state history instead of re-theorized. The leading suspect this exists to
# catch: an ABANDONED reader (wedged in cap.read(), never releasable) leaking
# a live AVCaptureSession that poisons subsequent opens of the same device
# until the process restarts.
_EVENTS: deque = deque(maxlen=500)
_EVENTS_LOCK = threading.Lock()
_EVENTS_T0 = time.monotonic()


def _log_event(kind: str, camera_id: str, **details) -> None:
    with _EVENTS_LOCK:
        _EVENTS.append(
            {
                "t": round(time.monotonic() - _EVENTS_T0, 3),
                "kind": kind,
                "camera_id": camera_id,
                "thread": threading.current_thread().name,
                **details,
            }
        )


def preview_events() -> dict:
    """Snapshot of the event ring + leaked-reader census for the debug endpoint.

    ``live_reader_threads`` lists cam-preview-* threads that are still alive —
    one past the registry's knowledge is a wedged cap.read() holding a zombie
    AVCaptureSession inside this process (the restart-fixes-it smell)."""
    with _EVENTS_LOCK:
        events = list(_EVENTS)
    reader_threads = [
        t.name for t in threading.enumerate() if t.name.startswith("cam-preview-") and t.is_alive()
    ]
    return {"events": events, "live_reader_threads": reader_threads}


# A preview is a thumbnail, not a recording: ~15 fps at JPEG quality 70 keeps
# per-client bandwidth modest without visible stutter.
TARGET_FPS = 15.0
JPEG_QUALITY = 70

# Frame-age watchdog: if the reader publishes no new frame within this many
# seconds, a waiting consumer declares the capture DEAD, deregisters it, and
# ends its stream. This is the recovery path for a reader wedged *inside*
# cap.read() (physically unplugged mid-stream, wedged AVFoundation) — the read
# never returns, so no failure streak ever accrues, but frames stop arriving.
# Must comfortably exceed the frame interval so a healthy-but-slow camera is not
# reaped. One minute is intentionally very patient: on marginal shared USB buses
# a camera can pause during format negotiation, teleop startup, or a brief hub
# hiccup and then recover; declaring it dead too early makes the frontend look
# like the camera "dropped" even though the device is still alive.
_FRAME_DEADLINE = 60.0

# A capture whose reader fails reads *in a row* for roughly the same one-minute
# window is treated as dead: the reader releases its own handle and exits, every
# client stream ends, and the next open_stream re-opens a fresh handle. Keep this
# time-equivalent with _FRAME_DEADLINE so read-failure streaks are not the hidden
# two-second tripwire while the frame-age watchdog is patient.
_MAX_READ_FAILURES = int(_FRAME_DEADLINE * TARGET_FPS)

# How long a consumer's open (open_stream priming) waits for its reader thread to
# report open success/failure before giving up with a CameraOpenError. Longer
# than the enumeration subprocess timeout (camera_enumeration, ~10s) so a slow
# id→index resolution isn't mistaken for a failure.
_OPEN_TIMEOUT = 15.0

# Open funnel (turnstile) — cameras are opened ONE AT A TIME. Live pages mount up
# to three preview tiles at once; without this, all three reader threads race into
# cv2.VideoCapture open + size/fps negotiation simultaneously, and on a shared
# USB-2 hub the concurrent negotiations trip each other — we watched a third camera
# open "successfully", stream zero frames, wedge, and get abandoned. Steady-state
# bandwidth is NOT the concern here: on macOS these UVC cameras stream MJPEG on
# the wire behind AVFoundation's '420v' formats (verified 2026-07-09: 29.8 fps
# at 1280x720, impossible raw on USB-2), a few MB/s per camera — the funnel
# exists solely for the stampede at BIRTH. It lets at most one reader run
# resolve→open→configure→first-frame at a time; a reader leaves the instant it
# publishes its first frame (or fails conclusively), so steady-state reads never
# queue and single-camera opens see an empty funnel and pass immediately.
#
# LOAD-BEARING FAIL-OPEN PROPERTY: this is a BEST-EFFORT serializer, deadline-
# bounded, NEVER a lock held across cv2 I/O. A holder wedged inside its first
# read() cannot release anything, so a waiter proceeds anyway (usurps) once the
# holder overruns this deadline. NO failure mode of one camera may permanently gate
# another — a wedged/slow birth delays the next open by AT MOST this long, never
# forever.
#
# Sized to exceed the slowest *legitimate* admitted birth, so a healthy-but-slow
# open is never usurped into a concurrent open (which would re-create the very
# stampede this exists to prevent): a cold id→index resolve can take up to
# _OPEN_TIMEOUT (it may spawn the enumeration subprocess), then the first frame up
# to _FRAME_DEADLINE after that. Derived from those two constants rather than a new
# magic number. Cost of the wide bound: a genuinely wedged holder gates the *next
# open* for up to this long (the frame-age watchdog still frees any waiting client
# at _FRAME_DEADLINE) — that is the deliberate price of never usurping a healthy
# slow birth.
_FUNNEL_HOLD_DEADLINE = _OPEN_TIMEOUT + _FRAME_DEADLINE

# Absolute backstop on how long a client's open may wait END TO END, funnel
# queue included. The per-camera open budget (_OPEN_TIMEOUT) starts only when
# the reader owns the turnstile and begins its own resolve→open
# (birth_started_ts); while queued behind other births a client waits against
# this backstop instead. Without the split, the third tile of a simultaneous
# three-tile mount was timed out — and 503'd — for queue time that was never
# its own open. Sized for the worst legitimate queue on a three-camera rig:
# two full funnel holds ahead plus this camera's own open.
_QUEUED_OPEN_BACKSTOP = 2 * _FUNNEL_HOLD_DEADLINE + _OPEN_TIMEOUT

# First-frame deadline: a capture that opened fine but has delivered NO frame
# within this window is a frame-dead session (opened into another session's
# asynchronous teardown wake — the bench's recurring failure mode), and every
# observation says it never heals with time, only with a fresh open. The reader
# gives up, releases, and lets the clients' retry loops re-roll — measured
# healthy first frames arrive in 0.1–5s, and re-rolls have healed every
# frame-dead session observed. Deliberately much tighter than the mid-stream
# patience knobs (_FRAME_DEADLINE / _MAX_READ_FAILURES): a dead BIRTH holds the
# open funnel hostage, so it must fail fast; an established stream stalling is
# a different, patient-friendly situation.
_FIRST_FRAME_DEADLINE = 8.0

# One-shot still frames (GET /camera-snapshot/{camera_id}). A standing MJPEG
# preview holds its capture (and historically a USB bandwidth reservation) for
# as long as it streams; snapshots let N tiles coexist by borrowing the device
# for one frame at a time (the open funnel serializes the borrows) or — with
# the idle linger below — by reading the newest frame off an already-running
# capture. _SNAPSHOT_FRAME_TIMEOUT bounds how long one snapshot waits for its
# frame — generous because a cold camera can take several seconds to its first
# frame (observed ~5s on this bench), but far below the streaming watchdog so a
# no-frame camera answers 503 in bounded time. _SNAPSHOT_CACHE_TTL coalesces
# poll bursts (several clients of one camera refreshing together) into one
# read; sized at 1s so the fast interactive surfaces (pickers/teleop polling
# ~1s) see ~1fps rather than a stale image — with the idle linger keeping the
# capture alive, a cache miss costs a latest-frame read, not a device open.
_SNAPSHOT_FRAME_TIMEOUT = 12.0
_SNAPSHOT_CACHE_TTL = 1.0

# Enumeration snapshot shared across a mount burst. Every id-addressed open
# re-resolves id→index; on macOS that spawns the AVFoundation enumeration
# subprocess. Three tiles mounting together used to run it three times BACK TO
# BACK through the funnel. Births within this TTL share one snapshot instead.
# Correctness: a MISS in the snapshot always falls through to one fresh
# enumeration before failing, so a stale snapshot can never wrongly fail a
# resolve. A HIT can hand out a stale index only if the bus reshuffled within
# the TTL *and* the id still exists elsewhere — a window the next open (fresh
# snapshot) self-corrects; at 3s it is far smaller than the reshuffle-churn it
# removes.
_ENUM_CACHE_TTL = 3.0

# Idle linger: when the LAST client releases a capture, keep its reader (and
# the device) alive this long before actually tearing down, so a re-acquire
# reuses the running capture instead of a fresh open. THE churn fix, from the
# forensic log (2026-07-09): fresh AVCaptureSessions started right after other
# sessions' asynchronous teardowns intermittently come up frame-dead
# (open_ok → read failures, no first frame, on hardware direct-probes prove
# healthy), while steady-state streaming never fails. Poll-driven snapshot
# tiles (1-8s cadence) and retrying stream tiles both used to generate exactly
# that teardown-then-open storm; with the linger they ride one continuously
# live capture instead. Recording/inference are unaffected: their
# block_for_recording force-stop tears lingering captures down immediately,
# refcounts notwithstanding. Sized comfortably above the snapshot poll
# interval; 0 disables (synchronous teardown — used by most unit tests).
_IDLE_LINGER = 20.0

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

# Preview capture format. We intentionally do NOT force a FOURCC by default:
# some AVFoundation/UVC stacks open successfully after MJPG is requested, then
# never deliver frames. On machines with real USB-3/10Gbps headroom, letting the
# backend auto-negotiate is safer than making MJPG a hidden failure mode. Saved
# recording/inference camera configs can still request MJPG explicitly when a
# particular bench needs compressed USB-2 bandwidth.
_PREVIEW_FOURCC: str | None = None
_PREVIEW_WIDTH = 640
_PREVIEW_HEIGHT = 480
_PREVIEW_FPS = 30


def _configure_preview_capture(cap: cv2.VideoCapture, camera_id: str) -> None:
    """Configure a freshly opened preview capture BEFORE its first read.

    Preview asks for size/fps but leaves FOURCC auto-negotiated unless
    ``_PREVIEW_FOURCC`` is set. Some backends claim to accept MJPG and then
    deliver no frames, so forced compression must not be the default preview
    behavior.

    Property order mirrors lerobot's OpenCVCamera._configure_capture_settings:
    FOURCC first (it can change the available resolution/FPS options), then frame
    size, then FPS.
    """
    if _PREVIEW_FOURCC:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*_PREVIEW_FOURCC))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, _PREVIEW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _PREVIEW_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, _PREVIEW_FPS)
    if logger.isEnabledFor(logging.DEBUG):
        # Decode the negotiated FOURCC int to 4 chars for legibility (same idiom
        # as lerobot). Handy for bench debugging: shows requested vs. actual.
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        negotiated_fourcc = "".join(chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4))
        logger.debug(
            "Camera %s preview requested %s/%dx%d@%d; negotiated %s/%dx%d@%.0f",
            camera_id,
            _PREVIEW_FOURCC or "AUTO",
            _PREVIEW_WIDTH,
            _PREVIEW_HEIGHT,
            _PREVIEW_FPS,
            negotiated_fourcc,
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            cap.get(cv2.CAP_PROP_FPS),
        )


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
        # Bumped on every acquire and every refcount-0 release; a pending idle-
        # linger expiry captured an older generation and no-ops (see
        # _linger_expire). Guarded by the manager's _registry_lock.
        self.linger_gen = 0
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
        # Monotonic time the reader actually began its birth — owned the
        # turnstile and started resolve→open. None while still queued behind
        # other births. _acquire keys its open-timeout off this so queue time
        # never counts against this camera's own open budget. Guarded by _cond.
        self.birth_started_ts: float | None = None
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
        # Enter the open turnstile: cameras are born one at a time so three tiles
        # mounting at once don't fire three concurrent cv2 opens into a shared
        # USB-2 hub. This covers resolve+open+configure+FIRST published frame;
        # deadline-bounded (see _FUNNEL_HOLD_DEADLINE) so a wedged birth can never
        # permanently gate the next one. Empty funnel = immediate passage.
        self._manager._enter_funnel(self)
        if self.stop.is_set():
            # Stopped while queued at (or just claiming) the turnstile —
            # force-released for a recording/inference claim, or the last
            # client detached before the birth began. NEVER open now: a
            # block_for_recording stop here means the session owner is about
            # to open this very device, and an unserialized open/release
            # cycle behind its back is exactly the contention the funnel and
            # the latch exist to prevent. Take the no-handle exit: hand the
            # turnstile on (no-op if we never claimed it), publish a legible
            # cancel error for any waiter, mark dead, deregister, confirm
            # released.
            self._manager._leave_funnel(self)
            with self._cond:
                self.open_error = CameraOpenError(
                    f"Camera {self.camera_id} preview open was cancelled before it started."
                )
                self.opened = True
                self.dead = True
                self._cond.notify_all()
            self._manager._deregister_if_current(self)
            self.released.set()
            _log_event("birth_cancelled", self.camera_id)
            return
        with self._cond:
            # Birth begins NOW (turnstile owned): stamp the start and wake any
            # client waiting in _acquire so it switches from the queue backstop
            # to this camera's own _OPEN_TIMEOUT budget.
            self.birth_started_ts = time.monotonic()
            self._cond.notify_all()
        _log_event("birth_start", self.camera_id)
        try:
            index = self._manager._resolve_cv2_index(self.camera_id)
            cap = self._open_cv2(index)
        except CameraOpenError as exc:
            # Open failed conclusively: leave the turnstile FIRST (before waking
            # the waiting open_stream) so the next birth can start the instant
            # this one is known-dead, not after the deadline. Then publish the
            # error so the waiting open_stream can raise it (→ 503), mark dead,
            # deregister, and confirm "released" (there is no handle to release).
            # No frame ever streams.
            self._manager._leave_funnel(self)
            with self._cond:
                self.open_error = exc
                self.opened = True
                self.dead = True
                self._cond.notify_all()
            self._manager._deregister_if_current(self)
            self.released.set()
            _log_event("open_failed", self.camera_id, error=str(exc))
            return
        # Configure size/fps BEFORE the first read (and before publishing
        # open_ok). Never fails the open — see _configure_preview_capture.
        _configure_preview_capture(cap, self.camera_id)
        with self._cond:
            self.cap = cap
            self.index = index
            self.open_ok = True
            self.opened = True
            self.last_frame_ts = time.monotonic()
            self._cond.notify_all()
        _log_event("open_ok", self.camera_id, index=index)
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
        first_frame_published = False
        loop_started = time.monotonic()
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
                        if not first_frame_published:
                            # This birth is confirmed streaming: hand the open
                            # turnstile to the next camera. Outside _cond (never
                            # nest the funnel lock under another) and idempotent.
                            first_frame_published = True
                            self._manager._leave_funnel(self)
                            _log_event("first_frame", self.camera_id)
                else:
                    failures += 1
                    if failures == 1:
                        # Log the START of a failure run (not every miss): a
                        # black tile whose events show read_fail_start with no
                        # first_frame is a device delivering nothing — the
                        # exact signature under investigation.
                        _log_event("read_fail_start", self.camera_id, had_first_frame=first_frame_published)
                    if not first_frame_published and time.monotonic() - loop_started > _FIRST_FRAME_DEADLINE:
                        # Frame-dead birth (see _FIRST_FRAME_DEADLINE): fail
                        # fast and free the funnel — the clients' retry loops
                        # re-roll the open, which heals it.
                        logger.warning(
                            "Camera %s delivered no first frame within %.0fs; releasing for a retry",
                            self.camera_id,
                            _FIRST_FRAME_DEADLINE,
                        )
                        _log_event("first_frame_timeout", self.camera_id)
                        break
                    if failures >= _MAX_READ_FAILURES:
                        logger.warning(
                            "Camera %s failed %d consecutive reads; reader releasing shared capture",
                            self.camera_id,
                            failures,
                        )
                        _log_event("read_fail_streak", self.camera_id, failures=failures)
                        break
                # Pace to ~TARGET_FPS. Waiting on the stop event (not a plain
                # sleep) lets stop_all cut the frame interval short.
                remaining = interval - (time.monotonic() - started)
                if remaining > 0 and self.stop.wait(remaining):
                    break
        finally:
            # Leave the turnstile if we exited before ever publishing (immediate
            # death, stop-before-first-frame, or a failure streak before the first
            # frame). Idempotent: a no-op once the first-frame hook above released
            # it, and a no-op if a deadline usurp already handed it on.
            self._manager._leave_funnel(self)
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
            _log_event("reader_released", self.camera_id)


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
        # Latched True while a session that opens the cv2 devices in a separate
        # capture stack owns them (set by block_for_recording, cleared by
        # resume_previews). Checked in _acquire under the SAME registry lock that
        # block_for_recording snapshots under, so a preview open that races the
        # session's force-release is refused rather than re-acquiring the device
        # the owner is about to grab — the TOCTOU the recording_active 409 gate
        # alone can't close (that flag lives in the record module, so the route's
        # check and the manager's open are not atomic). See block_for_recording.
        self._blocked = False
        # What has the devices latched, folded into the refusal message so a tile
        # names the real owner ("recording" or "inference"). Set alongside
        # _blocked under _registry_lock; stale after a resume, but only ever read
        # while _blocked is True.
        self._block_reason = "recording"
        # camera_id → monotonic time the reader last released that id's capture.
        # Lets the *next* open give a just-released (possibly still-busy) device
        # one short retry. Guarded by _registry_lock.
        self._last_release: dict[str, float] = {}
        # Enumeration snapshot shared across a mount burst (see _ENUM_CACHE_TTL).
        # Guarded by _registry_lock — held only for the read/store; the
        # enumeration itself (the slow subprocess) always runs unlocked. Readers
        # take _registry_lock here while holding the funnel; nothing ever takes
        # the funnel while holding _registry_lock, so the lock order is acyclic.
        self._enum_cache: list | None = None
        self._enum_cache_ts = 0.0
        # camera_id → (monotonic time, jpeg bytes) of the last one-shot still
        # (see _SNAPSHOT_CACHE_TTL). Guarded by _registry_lock.
        self._snapshot_cache: dict[str, tuple[float, bytes]] = {}
        # Open funnel (turnstile): the reader thread of at most one capture may be
        # in resolve→open→configure→first-frame at a time, so concurrent births
        # don't stampede a shared USB-2 hub (see _FUNNEL_HOLD_DEADLINE). Its own
        # small lock — held ONLY for this bookkeeping, NEVER across cv2 I/O and
        # never nested under _registry_lock or an entry's _cond, so it is a leaf
        # in the lock order and the never-hold-two-locks story stands. _holder is
        # the reader currently owning the turnstile (None when free); _deadline is
        # the monotonic time past which a waiter usurps a wedged/slow holder.
        self._funnel_lock = threading.Lock()
        self._funnel_cond = threading.Condition(self._funnel_lock)
        self._funnel_holder: _SharedCapture | None = None
        self._funnel_holder_deadline = 0.0

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

    def snapshot(self, camera_id: str | int) -> bytes:
        """One JPEG frame of ``camera_id`` without holding a standing stream.

        The config page's borrow-don't-reserve lane (see _SNAPSHOT_FRAME_TIMEOUT
        above for why): shares a live capture's newest frame when one is already
        streaming (no extra device open); otherwise acquires the capture, waits
        (bounded) for its first frame, and releases it — with the idle linger,
        that release keeps the capture warm for the next poll. Poll bursts
        within _SNAPSHOT_CACHE_TTL are served from the cache, so N tiles
        refreshing together cost one borrow, not N.

        Raises :class:`CameraOpenError` when the camera can't be opened, is
        latched to a recording/inference session, or produces no frame within
        the bounded wait (endpoint → 503, same contract as open_stream).
        """
        camera_id = str(camera_id)
        with self._registry_lock:
            cached = self._snapshot_cache.get(camera_id)
            if cached is not None and (time.monotonic() - cached[0]) <= _SNAPSHOT_CACHE_TTL:
                return cached[1]
        # _acquire shares an already-streaming capture (refcount bump only) or
        # births a fresh one through the funnel; either way _release below drops
        # our count (into the idle linger when we were the only client).
        entry = self._acquire(camera_id)
        try:
            with entry._cond:
                deadline = time.monotonic() + _SNAPSHOT_FRAME_TIMEOUT
                while entry.latest_jpeg is None and not entry.dead and not entry.stop.is_set():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    entry._cond.wait(remaining)
                jpeg = entry.latest_jpeg
        finally:
            self._release(entry)
        if jpeg is None:
            raise CameraOpenError(
                f"Camera {camera_id} produced no frame within {_SNAPSHOT_FRAME_TIMEOUT:.0f}s."
            )
        with self._registry_lock:
            self._snapshot_cache[camera_id] = (time.monotonic(), jpeg)
        return jpeg

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

    def block_for_recording(self, timeout: float = 1.0, reason: str = "recording") -> None:
        """Stop every preview AND latch new opens off for a session that owns the
        cv2 devices in its own capture stack (recording, or an inference rollout
        subprocess).

        The same force-release as :meth:`stop_all`, but it atomically sets the
        block flag in the SAME critical section that snapshots the live captures.
        That closes the window :meth:`stop_all` leaves open: without the latch, a
        preview open that races the release (a retrying ``<img>`` re-requesting
        right after the device is freed) re-acquires the very device the owner
        is about to open, starving the recorder to actual_fps=5.0 or handing the
        rollout subprocess a device that already has a second capture stack on it.
        Because the flag is set before the snapshot under ``_registry_lock``, any
        concurrent :meth:`_acquire` either registered before us (so it is in the
        snapshot and gets force-released) or observes the flag and is rejected —
        it can never slip a fresh capture in behind us. Cleared by
        :meth:`resume_previews` when the session ends.

        ``reason`` names the owner in the refusal a rejected open raises, so an
        inference-latched tile says inference is using the cameras, not recording.
        The name and default are kept so recording's call sites need no change.
        """
        with self._registry_lock:
            self._blocked = True
            self._block_reason = reason
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
            _log_event("force_stop", entry.camera_id)
            entry.stop.set()
            with self._registry_lock:
                if self._captures.get(entry.camera_id) is entry:
                    del self._captures[entry.camera_id]
            with entry._cond:
                entry._cond.notify_all()
        # A reader still parked in the open turnstile (a birth we force-released
        # mid-open) has just had its stop set — wake it so it drops out promptly
        # instead of waiting out the holder deadline. Separate, un-nested lock.
        self._wake_funnel()
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
                # THE zombie-creation moment: this reader's thread and its live
                # AVCaptureSession now leak inside the process, possibly
                # poisoning future opens of this device until restart.
                _log_event("reader_abandoned", entry.camera_id, grace_s=timeout)

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

    def _resolve_cv2_index(self, camera_id: str) -> int:
        """The cv2 index to open for ``camera_id`` — resolved at open time.

        Numeric fallback lane: a purely-numeric id is the index itself (legacy
        callers / platforms without stable ids). Otherwise the id is resolved
        against a live enumeration via camera_enumeration.resolve_index (the
        single id→index lookup, shared with record start), so a reopen after a
        hotplug reshuffle lands on the device's new index instead of streaming
        whatever slid into the old slot. Births within _ENUM_CACHE_TTL of the
        last enumeration share that snapshot (a simultaneous multi-tile mount
        runs the enumeration subprocess once, not once per camera); a MISS
        in the snapshot always falls through to one fresh enumeration before
        failing. An unknown id becomes a legible :class:`CameraOpenError`
        (endpoint → 503).
        """
        if camera_id.isdigit():
            return int(camera_id)
        cams = self._cached_enumeration()
        if cams is not None:
            index = camera_enumeration.find_camera_index(cams, camera_id)
            if index is not None:
                return index
            # Not in the snapshot: never fail from a cache — the device may
            # have (re)appeared since it was taken. Enumerate fresh below.
        try:
            index, enumerated = camera_enumeration.resolve_index(camera_id)
        except camera_enumeration.CameraNotConnectedError as exc:
            raise CameraOpenError(str(exc)) from exc
        self._store_enumeration(enumerated)
        return index

    def _cached_enumeration(self) -> list | None:
        """The shared enumeration snapshot, or None once it has aged past
        _ENUM_CACHE_TTL (or none was ever stored)."""
        with self._registry_lock:
            if self._enum_cache is not None and (time.monotonic() - self._enum_cache_ts) <= _ENUM_CACHE_TTL:
                return self._enum_cache
        return None

    def _store_enumeration(self, enumerated: list) -> None:
        """Store a fresh enumeration for the burst window. An empty result (a
        failed probe) is never cached — the next resolve should try again."""
        if not enumerated:
            return
        with self._registry_lock:
            self._enum_cache = enumerated
            self._enum_cache_ts = time.monotonic()

    def _mark_released(self, camera_id: str) -> None:
        """Record that a reader just released ``camera_id``'s capture, so the next
        open of the same id is eligible for the busy-retry (see _REOPEN_RETRY_*).

        Also drops the shared enumeration snapshot: a capture ending (device
        died, watchdog, force-release) is the signature of bus churn, and the
        very next open MUST re-resolve against live state — reopen-after-
        reshuffle correctness outranks burst sharing. Bursts only ever share a
        snapshot no capture death has cast doubt on."""
        with self._registry_lock:
            self._last_release[camera_id] = time.monotonic()
            self._enum_cache = None

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
            # A wedged capture is bus-churn evidence: the next birth must
            # re-enumerate, not trust a snapshot (see _mark_released).
            self._enum_cache = None
        with entry._cond:
            entry.dead = True
            entry._cond.notify_all()
        logger.warning(
            "Camera %s produced no frame within %.1fs; declaring the capture dead",
            entry.camera_id,
            _FRAME_DEADLINE,
        )
        _log_event("watchdog_declared_dead", entry.camera_id, deadline_s=_FRAME_DEADLINE)

    def _enter_funnel(self, entry: "_SharedCapture") -> None:
        """Admit ``entry``'s reader into the open turnstile, waiting its turn.

        Blocks until the funnel is free OR the current holder overruns its
        deadline — then this reader claims the turnstile and returns to run its
        resolve→open→configure→first-frame. See :data:`_FUNNEL_HOLD_DEADLINE` for
        the load-bearing fail-open contract: this is a best-effort serializer, not
        a lock held across cv2 I/O, so a holder wedged inside its first ``read()``
        can delay this open by at most the deadline (a waiter *usurps* it), never
        forever. The empty-funnel fast path claims the turnstile with no wait, so
        single-camera opens are latency-identical to before the funnel existed.

        Only ``_funnel_lock`` is held here (across the ``wait`` too, which releases
        it) — never an entry's ``_cond`` or ``_registry_lock`` and never any cv2
        call, so the funnel is a leaf in the lock order.
        """
        with self._funnel_cond:
            while True:
                if entry.stop.is_set():
                    # Stopped before we ever opened (force-release/detach during
                    # the birth window): don't claim the turnstile, let the reader
                    # fall through to its release path. A holder that is stopped
                    # mid-open instead clears via the read-loop finally.
                    return
                now = time.monotonic()
                if self._funnel_holder is None:
                    self._funnel_holder = entry
                    self._funnel_holder_deadline = now + _FUNNEL_HOLD_DEADLINE
                    return
                if now >= self._funnel_holder_deadline:
                    # Holder overran its deadline — presumed wedged/slow inside a
                    # blocking cv2 call it cannot itself return from. Usurp so a
                    # hung birth can never gate this open forever (fail-open).
                    logger.warning(
                        "Camera %s waited out the open turnstile: holder %s overran %.1fs; proceeding",
                        entry.camera_id,
                        self._funnel_holder.camera_id,
                        _FUNNEL_HOLD_DEADLINE,
                    )
                    _log_event("funnel_usurped", entry.camera_id, holder=self._funnel_holder.camera_id)
                    self._funnel_holder = entry
                    self._funnel_holder_deadline = now + _FUNNEL_HOLD_DEADLINE
                    return
                self._funnel_cond.wait(self._funnel_holder_deadline - now)

    def _leave_funnel(self, entry: "_SharedCapture") -> None:
        """Release the open turnstile iff ``entry`` still holds it (idempotent).

        Called on EVERY reader exit from the funnel scope: first frame published,
        open failure, alive-failure streak, stop-before-first-frame, watchdog-
        driven read-loop exit. A no-op when ``entry`` was already usurped (its
        deadline lapsed and a waiter took over) or never became holder — so it is
        safe to call redundantly from both the first-frame hook and the read-loop
        ``finally``. An *abandoned* holder (a reader wedged in ``read()`` that never
        returns to run this) is NOT cleared here; the deadline usurp in
        :meth:`_enter_funnel` covers it. The turnstile is never cross-thread
        cleaned, consistent with the never-release-across-threads rule.
        """
        with self._funnel_cond:
            if self._funnel_holder is entry:
                self._funnel_holder = None
                self._funnel_holder_deadline = 0.0
                self._funnel_cond.notify_all()

    def _wake_funnel(self) -> None:
        """Wake readers parked in :meth:`_enter_funnel` so a stop they were just
        handed (force-release / last-detach) is seen without waiting out the
        holder deadline. Cheap and idempotent; touches only ``_funnel_lock``."""
        with self._funnel_cond:
            self._funnel_cond.notify_all()

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
                _log_event("acquire_rejected_latched", camera_id, owner=self._block_reason)
                raise CameraOpenError(
                    f"Camera {camera_id} is reserved for the active {self._block_reason} session."
                )
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
            # Invalidate any pending idle-linger expiry: this capture has a
            # client again, so the deferred teardown must not fire.
            entry.linger_gen += 1
            refcount_now = entry.refcount
        _log_event("acquire" if created else "acquire_shared", camera_id, refcount=refcount_now)
        if created:
            entry.start()  # launch the reader; it owns the cv2 open
        # Wait (bounded) for the reader to report its open result. This is the
        # synchronous open the endpoint needs for its 503, but it can never block
        # forever. Two clocks, so a camera queued at the funnel behind other
        # births is not billed for time that was never its own open: until the
        # reader stamps birth_started_ts (turnstile owned, resolve→open running)
        # the wait is bounded only by the end-to-end _QUEUED_OPEN_BACKSTOP; once
        # the birth starts, by birth_started_ts + _OPEN_TIMEOUT (backstop-capped).
        # A failed/timed-out open self-cleans either way.
        with entry._cond:
            backstop = time.monotonic() + _QUEUED_OPEN_BACKSTOP
            while not entry.opened:
                if entry.birth_started_ts is not None:
                    deadline = min(entry.birth_started_ts + _OPEN_TIMEOUT, backstop)
                else:
                    deadline = backstop
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
        linger = _IDLE_LINGER
        with self._registry_lock:
            # Decrement, the last-client check, and the deregistration stay in
            # ONE critical section (an acquire between them could re-adopt an
            # entry this thread is about to tear down); the event is logged
            # after, from captured values.
            entry.refcount -= 1
            refcount_now = entry.refcount
            if refcount_now <= 0:
                if linger > 0:
                    # Idle linger (see _IDLE_LINGER): keep the capture alive
                    # and REGISTERED so a near-future acquire reuses it instead
                    # of running a fresh open into the teardown turbulence that
                    # intermittently births frame-dead sessions. The generation
                    # stamp lets a later acquire cancel this deferred teardown.
                    entry.linger_gen += 1
                    linger_gen = entry.linger_gen
                else:
                    # Synchronous teardown (linger disabled): deregister before
                    # signalling the reader so a new client never latches onto
                    # a capture that is being torn down.
                    if self._captures.get(entry.camera_id) is entry:
                        del self._captures[entry.camera_id]
        _log_event("client_release", entry.camera_id, refcount=refcount_now)
        if refcount_now > 0:
            return
        if linger > 0:
            _log_event("linger_start", entry.camera_id, linger_s=linger)
            timer = threading.Timer(linger, self._linger_expire, args=(entry, linger_gen))
            timer.daemon = True
            timer.start()
            return
        self._teardown(entry)

    def _linger_expire(self, entry: _SharedCapture, linger_gen: int) -> None:
        """Deferred last-client teardown (idle linger lapsed with no re-acquire).

        A stale generation means a client re-acquired (and possibly re-released,
        arming a NEWER timer) after this timer was armed — this one must no-op or
        it would tear down a capture that has clients again."""
        with self._registry_lock:
            if entry.refcount > 0 or entry.linger_gen != linger_gen:
                return
            if self._captures.get(entry.camera_id) is entry:
                del self._captures[entry.camera_id]
        _log_event("linger_expired", entry.camera_id)
        self._teardown(entry)

    def _teardown(self, entry: _SharedCapture) -> None:
        """Tell the reader to stop and release its own capture, then wait a bounded
        grace for it to confirm. The caller never touches cv2, so this thread is
        never inside read(); a reader wedged in read() is abandoned when the grace
        lapses rather than joined. Waiting lets a subsequent same-index open find
        the device actually free instead of transiently busy."""
        entry.stop.set()
        # Wake a reader parked in the open turnstile (last client vanished before
        # the birth ever produced a frame) so it sees the stop without waiting out
        # the holder deadline. Un-nested leaf lock.
        self._wake_funnel()
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
