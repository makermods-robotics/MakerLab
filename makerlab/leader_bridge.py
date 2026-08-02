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
"""CLIENT-side leader bridge — stream a local leader arm to a remote MakerLab.

This module runs on the operator's laptop, which owns the physical **leader**
arm. The follower and the cameras live on a remote server machine that runs
MakerLab headless (see ``docs/remote-portal/SPEC.md``). The bridge is a Portal
**Operator**: every tick it reads the local leader's pose, publishes it as an
action over LiveKit Portal, and pulls the server's synced observation back —
including the camera frames, which it re-serves locally as MJPEG so the
existing ``BackendCameraStream`` tiles render the remote workspace.

::

    MACBOOK (this machine)                     MAC MINI (server)
    SO101Leader (local calibration)            LiveKitTeleoperator (Portal Robot)
       │ get_action()                             │ get_action() → SO101Follower
       ▼                                          ▼
    Portal Operator ── action ───────────────▶ MakerLab record/teleop loop
       ▲                                          │ get_observation()
       └──────────── state + video ───────────────┘
       │
       └▶ GET /leader-bridge/camera/{name}  (MJPEG re-serve, this module)

Three things here exist because the naive version fails in practice:

1. **Nothing portal-related is imported at module import time.** A machine
   without the ``livekit-portal`` wheel must still run every other MakerLab
   flow, and this module is imported by ``server.py``'s route handlers.
2. **Control is claimed with retry/backoff.** The claim is an RPC to the robot
   participant; over a relayed link it regularly exceeds the default response
   timeout (``rpc error 1501: Connection timeout``). The robot silently ignores
   every action until someone claims control, so a one-shot claim turns into
   "the arm doesn't move" with no error anywhere.
3. **The Portal tuning must match the server byte for byte**
   (``state_reliable=False``, ``slack=2``, ``reuse_stale_frames=True``) — these
   were measured on the real rig (RTT ~180 ms → ~15 ms, drops ~15/s → ~0).
   Portal fingerprints the schema and drops mismatched packets with only a
   WARN, so the resolved schema is logged on start for cross-checking against
   the server's log.
"""

import asyncio
import logging
import os
import re
import socket
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
from fastapi.responses import JSONResponse, StreamingResponse

# The MJPEG re-serve deliberately reuses the backend-preview conventions
# (~15 fps, JPEG quality 70, boundary "frame") so BackendCameraStream behaves
# identically whether it is pointed at a local cv2 camera or at this bridge.
from .camera_preview import JPEG_QUALITY, TARGET_FPS
from .utils.config import (
    LEADER_CONFIG_PATH,
    bimanual_base_id,
    is_valid_robot_name,
)
from .utils.errors import format_exception, friendly_hint

logger = logging.getLogger(__name__)

# --- wire contract (must match the server; see SPEC.md "Wire contract") -----
# Portal session/room name for a robot record.
ROOM_PREFIX = "makerlab-"
DEFAULT_FPS = 30
# Latency tuning, measured on the real two-machine rig. Changing any of these
# on one side only silently degrades the link — keep them in lockstep with the
# server's LiveKitTeleoperatorConfig.
PORTAL_SLACK = 2
PORTAL_STATE_RELIABLE = False
PORTAL_REUSE_STALE_FRAMES = True

# --- control claim ----------------------------------------------------------
# Initial burst right after connect: 6 tries, 2s apart. If the robot side isn't
# in the room yet (the server's teleop/record loop hasn't been started), the
# claim can't succeed at all, so the worker keeps retrying on a slow cadence
# instead of killing the session — the operator can start the bridge first and
# the server loop second, in either order.
CLAIM_ATTEMPTS = 6
CLAIM_RETRY_DELAY_S = 2.0
CLAIM_RETRY_INTERVAL_S = 10.0

# --- timeouts ---------------------------------------------------------------
HTTP_TIMEOUT_S = 10.0
CONNECT_TIMEOUT_S = 30.0
RPC_TIMEOUT_S = 15.0
DISCONNECT_TIMEOUT_S = 5.0
LOOP_JOIN_TIMEOUT_S = 5.0
STOP_JOIN_TIMEOUT_S = 15.0
METRICS_INTERVAL_S = 1.0

# --- MJPEG re-serve ---------------------------------------------------------
# How long the endpoint waits for the first frame of a track before answering
# 503. Kept short on purpose: server.py's route is `async def`, so this wait
# happens on the event loop.
FIRST_FRAME_WAIT_S = 1.0
# Per-iteration wait inside the stream generator (runs in a threadpool, so a
# longer block is fine here) and how long a track may go frame-less before the
# stream ends and the frontend's retry/backoff takes over.
FRAME_WAIT_S = 1.0
STALE_STREAM_S = 5.0

_IDENTITY_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


# ---------------------------------------------------------------------------
# Module state (the repo's per-feature convention: module-level flags + a lock)
# ---------------------------------------------------------------------------

# True from a successful start until the worker's cleanup has been requested.
bridge_active = False
bridge_thread: threading.Thread | None = None
# Terminal error of the current/most recent session ("Type: message"), surfaced
# by /leader-bridge/status together with its plain-language hint. Cleared on
# start.
last_error: str | None = None
# Set when the leader could not be released/disconnected on the way out — i.e.
# the arm may still be energized. Same semantics as teleoperate.py's flag.
last_cleanup_error: str | None = None
# Guards the start path and the active flag; the worker owns teardown so stop()
# never races the serial bus from the request thread.
_state_lock = threading.Lock()
_stop_event = threading.Event()
_session: "_BridgeSession | None" = None


# ---------------------------------------------------------------------------
# Frame store — the Portal→MJPEG hand-off
# ---------------------------------------------------------------------------


class _FrameStore:
    """Latest RGB frame per camera track, with a "wait for a new one" gate.

    The worker thread publishes; any number of MJPEG generators consume. Each
    frame carries a monotonically increasing sequence number so a consumer can
    block until something genuinely new arrives instead of re-encoding the same
    picture at full tilt.
    """

    def __init__(self, names: list[str]) -> None:
        self.names = list(names)
        self._frames: dict[str, tuple[np.ndarray, int]] = {}
        self._seq = 0
        self._closed = False
        self._cond = threading.Condition()

    @property
    def closed(self) -> bool:
        with self._cond:
            return self._closed

    def put(self, name: str, frame: np.ndarray) -> None:
        with self._cond:
            if self._closed:
                return
            self._seq += 1
            self._frames[name] = (frame, self._seq)
            self._cond.notify_all()

    def wait_for(self, name: str, last_seq: int, timeout: float) -> tuple[np.ndarray, int] | None:
        """Return ``(frame, seq)`` for a frame newer than ``last_seq``.

        ``None`` means the timeout expired with nothing new (or the store was
        closed). Does not consume: two consumers of the same track both see
        every frame.
        """
        deadline = time.monotonic() + timeout
        with self._cond:
            while not self._closed:
                entry = self._frames.get(name)
                if entry is not None and entry[1] != last_seq:
                    return entry
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            return None

    def close(self) -> None:
        """Release every waiting consumer so their generators exit promptly."""
        with self._cond:
            self._closed = True
            self._frames.clear()
            self._cond.notify_all()


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class _BridgeSession:
    """Everything one bridge session owns: the leader, the Portal Operator, and
    the asyncio loop the Operator's coroutines run on.

    Portal's Python surface is async (``connect`` / ``disconnect`` /
    ``set_active_operator``) while MakerLab's feature workers are plain
    threads, so — exactly as the lerobot Portal plugins do — the session owns a
    dedicated event loop in a daemon thread and the worker hops onto it with
    ``run_coroutine_threadsafe``. Nothing in this module ever awaits from a
    worker thread.
    """

    def __init__(
        self,
        *,
        server_url: str,
        livekit_url: str,
        room: str,
        identity: str,
        fps: int,
        camera_names: list[str],
        action_keys: list[str],
    ) -> None:
        self.server_url = server_url
        self.livekit_url = livekit_url
        self.room = room
        self.identity = identity
        self.fps = fps
        self.camera_names = list(camera_names)
        # Portal's state/action fields are the bare motor names; lerobot's
        # action dicts use the "<motor>.pos" suffix. Both sides of the wire
        # sort the keys, so the fingerprints match regardless of dict order.
        self.action_keys = list(action_keys)
        self.action_motors = [_strip_pos(key) for key in self.action_keys]
        self.frames = _FrameStore(self.camera_names)

        self.leader: Any | None = None
        self.operator: Any | None = None
        self.connected = False
        self.claimed = False
        self.rtt_ms: float | None = None

        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None

    # -- background asyncio loop ---------------------------------------------

    def start_loop(self) -> None:
        loop = asyncio.new_event_loop()
        started = threading.Event()

        def _runner() -> None:
            asyncio.set_event_loop(loop)
            started.set()
            loop.run_forever()

        self._loop = loop
        self._loop_thread = threading.Thread(target=_runner, name="leader-bridge-portal-loop", daemon=True)
        self._loop_thread.start()
        started.wait()

    def stop_loop(self) -> None:
        if self._loop is None:
            return
        loop = self._loop
        self._loop = None
        thread = self._loop_thread
        self._loop_thread = None
        # Every step is best-effort: this runs on the worker's cleanup path,
        # where a raised exception would skip the leader's torque release.
        try:
            loop.call_soon_threadsafe(loop.stop)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Stopping the portal event loop failed: %s", exc)
        if thread is not None:
            thread.join(timeout=LOOP_JOIN_TIMEOUT_S)
        try:
            loop.close()
        except Exception as exc:  # noqa: BLE001 — teardown must not mask the real error
            logger.warning("Closing the portal event loop failed: %s", exc)

    def run(self, coro, timeout: float):
        """Run a Portal coroutine on the session loop from a worker thread."""
        if self._loop is None:
            raise RuntimeError("the portal event loop is not running")
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)


def _strip_pos(key: str) -> str:
    return key[: -len(".pos")] if key.endswith(".pos") else key


# ---------------------------------------------------------------------------
# Payload parsing / server handshake
# ---------------------------------------------------------------------------


class _BridgeRequestError(ValueError):
    """A bad start payload — reported to the caller as 400, not a crash."""


def _require_str(payload: dict, key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise _BridgeRequestError(f"{label} is required.")
    return value.strip()


def _normalize_server_url(raw: str) -> str:
    """Accept "192.168.1.4:8000", "http://host:8000/" etc. → "http://host:8000"."""
    url = raw.strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    return url


def _client_identity() -> str:
    """Stable per-machine Portal identity: ``makerlab-client-<hostname>``.

    Stable matters: the claim RPC names an identity, and a server-side UI that
    lists operators should see the same name across reconnects.
    """
    host = socket.gethostname().split(".")[0] or "unknown"
    return f"makerlab-client-{_IDENTITY_SAFE.sub('-', host)}"


def _fetch_portal_token(server_url: str, identity: str, room: str) -> tuple[str, str, str]:
    """Ask the server to mint a JWT for this room. Returns (url, room, token).

    The server owns the LiveKit API key/secret; the client never sees it. A
    failure here is the single most likely start failure (wrong host, server
    not running, portal extra not installed there), so the message names the
    URL that was tried.
    """
    endpoint = f"{server_url}/portal/token"
    try:
        response = httpx.post(
            endpoint,
            json={"identity": identity, "room": room},
            timeout=HTTP_TIMEOUT_S,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not reach the MakerLab server at {server_url} ({exc}). "
            "Check the address, and that the server is running with LiveKit enabled."
        ) from exc
    if response.status_code >= 400:
        detail = _response_detail(response)
        raise RuntimeError(f"The server refused to mint a Portal token ({response.status_code}): {detail}")
    try:
        data = response.json()
    except Exception as exc:
        raise RuntimeError(f"{endpoint} did not return JSON — is that really a MakerLab server?") from exc

    token = data.get("token")
    url = data.get("url")
    if not token or not url:
        raise RuntimeError(
            f"The server's token response is missing 'token'/'url' (got keys: {sorted(data)}). "
            "The server may be running an older MakerLab."
        )
    return str(url), str(data.get("room") or room), str(token)


def _response_detail(response) -> str:
    try:
        body = response.json()
    except Exception:  # noqa: BLE001 — a non-JSON error body is still worth showing
        return response.text[:200]
    if isinstance(body, dict):
        for key in ("detail", "message", "error"):
            value = body.get(key)
            if isinstance(value, str) and value:
                return value
    return str(body)[:200]


def _camera_names_from_payload(raw: object) -> list[str] | None:
    """Camera track names from an explicit payload field, if it carries one.

    Tolerates the three shapes the frontend could plausibly send: a list of
    names, a list of CameraConfig-ish dicts, or a name→config dict (what
    recording uses).
    """
    if isinstance(raw, dict):
        return [str(name) for name in raw]
    if isinstance(raw, list):
        names = []
        for entry in raw:
            if isinstance(entry, str) and entry.strip():
                names.append(entry.strip())
            elif isinstance(entry, dict) and str(entry.get("name") or "").strip():
                names.append(str(entry["name"]).strip())
        return names
    return None


def _fetch_camera_names(server_url: str, robot_name: str) -> list[str]:
    """Camera names from the SERVER's robot record — it owns the cameras.

    Video is a convenience, not the control path: a failure here is logged and
    degrades to "no camera tracks", so a bridge can still drive the arm when
    the record can't be read.
    """
    try:
        response = httpx.get(f"{server_url}/robots/{robot_name}", timeout=HTTP_TIMEOUT_S)
        response.raise_for_status()
        record = response.json().get("robot") or {}
    except Exception as exc:  # noqa: BLE001 — video is optional; control is not
        logger.warning(
            "Could not read robot '%s' from %s (%s); starting the bridge with no camera tracks",
            robot_name,
            server_url,
            exc,
        )
        return []
    names = _camera_names_from_payload(record.get("cameras")) or []
    if not names:
        logger.info("Server robot record '%s' declares no cameras", robot_name)
    return names


# ---------------------------------------------------------------------------
# Local leader arm
# ---------------------------------------------------------------------------


def _leader_calibration_stem(leader_config: str, slot: str) -> str:
    """Validate a leader calibration selection and return its stem.

    ``utils/config.setup_calibration_files`` can't be used here: it requires a
    follower config too, and on this machine there is no follower. Its leader
    half reduces to exactly this — the leader library dir *is* the location
    lerobot loads from, so there is nothing to copy, only to check.
    """
    stem = (leader_config or "").strip()
    if not stem:
        raise _BridgeRequestError(
            f"The {slot} arm has no calibration assigned. Calibrate it "
            "(or assign a saved calibration config) before starting the bridge."
        )
    if stem.endswith(".json"):
        stem = stem[: -len(".json")]
    path = os.path.join(LEADER_CONFIG_PATH, f"{stem}.json")
    if not os.path.exists(path):
        raise _BridgeRequestError(
            f"The {slot} arm's calibration file '{stem}.json' was not found in {LEADER_CONFIG_PATH}. "
            "Calibrate that arm (or assign a saved calibration) before starting the bridge."
        )
    return stem


def _stage_bimanual_leader_calibrations(base: str, left_config: str, right_config: str) -> str:
    """Stage only the two LEADER calibrations for a BiSO bridge session.

    The exact twin of ``config.stage_bimanual_follower_calibrations`` (which
    exists for inference, a follower-only flow) for the mirror-image case: this
    machine has leaders and no followers, so requiring follower library files
    would fail spuriously. Reuses config.py's own staging primitives rather
    than reimplementing the "<base>_left/right.json" convention.
    """
    from .utils.config import _bimanual_leader_staging_dir, _stage_one_side

    staging = _bimanual_leader_staging_dir(base)
    _stage_one_side(LEADER_CONFIG_PATH, staging, base, left_config, right_config, "leader")
    return staging


def _build_leader(payload: dict, mode: str, robot_name: str):
    """Construct the local lerobot leader device (not yet connected).

    lerobot is imported here, not at module scope, so this module stays
    importable on a machine where the portal/lerobot stack is partially
    installed — and so ``server.py``'s lazy route import stays cheap.
    """
    from lerobot.teleoperators.bi_so_leader import BiSOLeader, BiSOLeaderConfig
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

    leader_port = _require_str(payload, "leader_port", "The leader port")
    if mode == "bimanual":
        right_port = _require_str(payload, "right_leader_port", "The right leader port")
        left_stem = _leader_calibration_stem(payload.get("leader_config", ""), "left leader")
        right_stem = _leader_calibration_stem(payload.get("right_leader_config", ""), "right leader")
        if leader_port == right_port:
            raise _BridgeRequestError("The two leader arms cannot share one serial port.")
        base = bimanual_base_id(robot_name)
        staging = _stage_bimanual_leader_calibrations(base, left_stem, right_stem)
        return BiSOLeader(
            BiSOLeaderConfig(
                id=base,
                calibration_dir=Path(staging),
                left_arm_config=SO101LeaderConfig(port=leader_port),
                right_arm_config=SO101LeaderConfig(port=right_port),
            )
        )

    stem = _leader_calibration_stem(payload.get("leader_config", ""), "leader")
    return SO101Leader(SO101LeaderConfig(port=leader_port, id=stem))


def _connect_leader(leader, mode: str, payload: dict) -> list[str]:
    """Connect + configure the leader arm(s), naming whichever one failed.

    Mirrors ``teleoperate.py``'s explicit bus-level path rather than calling
    ``leader.connect()``: lerobot's connect() drops into *interactive*
    recalibration when a calibration is missing, which would hang this thread
    with no output. Returns the arm-identity guard's warn-but-allow messages.
    """
    from .arm_identity import verify_devices

    skip_identity = bool(payload.get("skip_identity_check"))
    if mode == "bimanual":
        arms = (
            (leader.left_arm, "left leader", payload.get("leader_port")),
            (leader.right_arm, "right leader", payload.get("right_leader_port")),
        )
        for arm, label, port in arms:
            try:
                arm.bus.connect()
            except Exception as exc:
                raise RuntimeError(
                    f"Could not connect to the {label} arm on {port}. "
                    "Make sure it's plugged in and powered on, then try again."
                ) from exc
        # Read-only identity guard BEFORE write_calibration can stamp a wrong
        # file into a swapped arm's EEPROM. The sub-arm ids are staging
        # aliases, so pass the real library stems in arm-iteration order.
        warnings = verify_devices(
            ((leader, "leader"),),
            skip=skip_identity,
            config_names=[
                str(payload.get("leader_config") or "").removesuffix(".json"),
                str(payload.get("right_leader_config") or "").removesuffix(".json"),
            ],
        )
        for arm, _label, _port in arms:
            arm.bus.write_calibration(arm.calibration)
        leader.configure()
        return warnings

    try:
        leader.bus.connect()
    except Exception as exc:
        raise RuntimeError(
            f"Could not connect to the leader arm on {payload.get('leader_port')}. "
            "Make sure it's plugged in and powered on, then try again."
        ) from exc
    warnings = verify_devices(((leader, "leader"),), skip=skip_identity)
    leader.bus.write_calibration(leader.calibration)
    leader.configure()
    return warnings


def _release_leader(leader) -> str | None:
    """Always-run torque release + disconnect for the leader arm.

    The leader is hand-held, so a stuck-energized arm is both a safety problem
    and unusable. Runs motor-by-motor first (one bad motor must not abort the
    release for the others), then disconnects. Returns a problem description
    when the arm may still be energized.
    """
    if leader is None:
        return None
    # Reuses teleoperation's release helpers rather than duplicating them.
    # Imported here (not at module scope) because teleoperate.py pulls in the
    # whole lerobot stack, and the import itself is guarded: a release that
    # can't even load its helper must still fall back to a plain disconnect,
    # which is what actually de-energizes the arm.
    try:
        from .teleoperate import _safe_disconnect, force_disable_torque
    except Exception as exc:  # noqa: BLE001
        logger.exception("Could not load the torque-release helpers")
        try:
            leader.disconnect()
            return None
        except Exception as disconnect_exc:  # noqa: BLE001
            return (
                f"Could not release the leader arm ({exc}; {disconnect_exc}). TORQUE MAY STILL BE "
                "ENABLED — if the arm stays rigid, unplug its power to release it."
            )

    problems: list[str] = []
    try:
        problems += force_disable_torque(leader, "leader arm")
    except Exception as exc:  # noqa: BLE001 — never let teardown raise past here
        problems.append(f"Disabling leader torque failed: {exc}")
        logger.exception("Disabling leader torque failed")
    disconnect_error = _safe_disconnect(leader, "leader arm")
    if disconnect_error:
        problems.append(disconnect_error)
    return " ".join(problems) if problems else None


# ---------------------------------------------------------------------------
# Portal Operator
# ---------------------------------------------------------------------------


def _connect_operator(session: _BridgeSession, token: str) -> None:
    """Build + connect the Portal Operator with the server-matching tuning."""
    from livekit.portal import DType, Operator, OperatorConfig

    config = OperatorConfig(session.room)
    for name in session.camera_names:
        config.add_video(name)
    typed = [(motor, DType.F64) for motor in session.action_motors]
    if typed:
        # State mirrors the action schema (the follower reports the same
        # joints it is commanded on) — the lerobot convention both Portal
        # plugins assume.
        config.add_state_typed(typed)
        config.add_action_typed(typed)
    config.set_fps(session.fps)
    config.set_slack(PORTAL_SLACK)
    config.set_state_reliable(PORTAL_STATE_RELIABLE)
    config.set_reuse_stale_frames(PORTAL_REUSE_STALE_FRAMES)

    session.start_loop()
    operator = Operator(config)
    session.operator = operator
    # Logged so a schema mismatch (which Portal only WARNs about while
    # silently dropping every packet — i.e. "the arm doesn't move") can be
    # diffed against the server's log line.
    logger.info(
        "Leader bridge schema — room=%s fps=%d action/state=%s video=%s",
        session.room,
        session.fps,
        session.action_motors,
        session.camera_names,
    )
    session.run(operator.connect(session.livekit_url, token), CONNECT_TIMEOUT_S)
    session.connected = True
    logger.info("Leader bridge connected to %s as '%s'", session.livekit_url, session.identity)


def _try_claim_control(session: _BridgeSession) -> bool:
    """One claim attempt. True when this operator now holds control."""
    operator = session.operator
    if operator is None:
        return False
    identity = operator.local_identity()
    if identity is None:
        logger.warning("Cannot claim control: the Portal session has no local identity yet")
        return False
    session.run(operator.set_active_operator(identity), RPC_TIMEOUT_S)
    logger.info("Leader bridge claimed control as '%s'", identity)
    return True


def _claim_control_with_retry(session: _BridgeSession, attempts: int, delay: float) -> bool:
    """Claim control, retrying on relay timeouts.

    The claim is an RPC to the robot participant. Over a relayed link (~0.5-1s
    RTT) it regularly exceeds the default response timeout and raises
    ``rpc error 1501: Connection timeout``; it also cannot succeed at all until
    the server's loop has joined the room. The robot ignores every action until
    someone claims control, so a single attempt is the difference between
    working teleop and a dead-looking arm. Returns False rather than raising —
    the caller keeps the session alive and retries on a slow cadence.
    """
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        if _stop_event.is_set():
            return False
        try:
            if _try_claim_control(session):
                return True
        except Exception as exc:  # noqa: BLE001 — every RPC failure is retryable here
            last_exc = exc
            logger.warning("Control claim attempt %d/%d failed: %s", attempt, attempts, exc)
        if attempt < attempts and _stop_event.wait(delay):
            return False
    if last_exc is not None:
        logger.warning("Could not claim control after %d attempts: %s", attempts, last_exc)
    return False


def _publish_action(session: _BridgeSession, action: dict) -> None:
    values = {
        motor: float(action[key])
        for key, motor in zip(session.action_keys, session.action_motors, strict=False)
        if key in action
    }
    if values and session.operator is not None:
        session.operator.send_action(values)


def _store_frames(session: _BridgeSession, observation, to_numpy) -> None:
    for name in session.camera_names:
        frame = observation.frames.get(name)
        if frame is None:
            continue
        try:
            session.frames.put(name, to_numpy(frame.data, frame.width, frame.height))
        except Exception as exc:  # noqa: BLE001 — a malformed frame must not kill the loop
            logger.debug("Dropping malformed frame for track '%s': %s", name, exc)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _bridge_worker(session: _BridgeSession) -> None:
    """Read the leader, publish actions, pull observations. Owns teardown."""
    global bridge_active, last_error, last_cleanup_error

    error_text: str | None = None
    try:
        # Inside the try so that even an import failure lands in the cleanup
        # path — an early return here would strand the arm and the flags.
        from livekit.portal import frame_bytes_to_numpy_rgb

        session.claimed = _claim_control_with_retry(session, CLAIM_ATTEMPTS, CLAIM_RETRY_DELAY_S)
        if not session.claimed:
            logger.warning(
                "Leader bridge is connected but has not claimed control yet — the follower will "
                "ignore actions until it does. Retrying every %.0fs.",
                CLAIM_RETRY_INTERVAL_S,
            )
        next_claim = time.monotonic() + CLAIM_RETRY_INTERVAL_S
        next_metrics = time.monotonic()
        interval = 1.0 / session.fps

        while bridge_active and not _stop_event.is_set():
            tick = time.monotonic()

            # Send first so control latency never waits on the read-back path.
            action = session.leader.get_action()
            if action:
                _publish_action(session, action)

            observation = session.operator.get_observation() if session.operator else None
            if observation is not None:
                _store_frames(session, observation, frame_bytes_to_numpy_rgb)

            if not session.claimed and tick >= next_claim:
                next_claim = tick + CLAIM_RETRY_INTERVAL_S
                try:
                    session.claimed = _try_claim_control(session)
                except Exception as exc:  # noqa: BLE001 — keep driving; retry next cycle
                    logger.debug("Control re-claim failed: %s", exc)

            if tick >= next_metrics:
                next_metrics = tick + METRICS_INTERVAL_S
                session.rtt_ms = _sample_rtt_ms(session)

            remaining = interval - (time.monotonic() - tick)
            if remaining > 0 and _stop_event.wait(remaining):
                break
    except Exception as exc:
        error_text = format_exception(exc)
        logger.exception("Leader bridge loop failed")
    finally:
        cleanup_error = _shutdown_session(session)
        with _state_lock:
            bridge_active = False
            last_error = error_text
            last_cleanup_error = cleanup_error


def _sample_rtt_ms(session: _BridgeSession) -> float | None:
    """Last round-trip sample in ms, or None before Portal has one."""
    operator = session.operator
    if operator is None:
        return None
    try:
        metrics = operator.metrics()
    except Exception as exc:  # noqa: BLE001 — metrics are diagnostics, never fatal
        logger.debug("Reading portal metrics failed: %s", exc)
        return None
    last = getattr(getattr(metrics, "rtt", None), "rtt_us_last", None)
    return last / 1e3 if last else None


def _shutdown_session(session: _BridgeSession) -> str | None:
    """Tear a session down. The leader release ALWAYS runs; never raises.

    Order matters: stop the wire first so no further actions are published,
    then release the arm. Every step is wrapped — this is the worker's cleanup
    path, where a raised exception would both skip the torque release and
    strand ``bridge_active`` at True until the server restarts.
    """
    problems: list[str] = []
    try:
        session.frames.close()
        session.claimed = False
        operator = session.operator
        if operator is not None:
            try:
                if session.connected:
                    session.run(operator.disconnect(), DISCONNECT_TIMEOUT_S)
            except Exception as exc:  # noqa: BLE001 — teardown is best-effort
                logger.warning("Portal disconnect failed: %s", exc)
            finally:
                session.connected = False
                try:
                    operator.close()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Closing the Portal operator failed: %s", exc)
                session.operator = None
        session.stop_loop()
    except Exception as exc:  # noqa: BLE001 — must still reach the release below
        logger.exception("Portal teardown failed")
        problems.append(f"Portal teardown failed: {exc}")
    finally:
        try:
            leader_problem = _release_leader(session.leader)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Releasing the leader arm failed")
            leader_problem = (
                f"Releasing the leader arm failed: {exc}. TORQUE MAY STILL BE ENABLED — "
                "if the arm stays rigid, unplug its power to release it."
            )
        session.leader = None
    if leader_problem:
        problems.append(leader_problem)
    cleanup_error = " ".join(problems) if problems else None
    if cleanup_error:
        logger.error("Leader bridge cleanup problem: %s", cleanup_error)
    return cleanup_error


# ---------------------------------------------------------------------------
# HTTP handlers
# ---------------------------------------------------------------------------


def handle_leader_bridge_start(payload: dict) -> dict[str, Any]:
    """Start a leader-bridge session (POST /leader-bridge/start).

    Payload: ``{server_url, leader_port, leader_config, mode, robot_name}``
    plus ``right_leader_port`` / ``right_leader_config`` in bimanual mode, and
    the optional ``cameras`` / ``fps`` / ``skip_identity_check``.

    The leader connect and the Portal connect happen *synchronously* so a
    failure (arm unplugged, server unreachable, portal not installed) is
    reported to the caller instead of dying silently in the worker — the same
    contract as ``handle_start_teleoperation``. Only the control loop and the
    control claim (which retries for up to ~12s) run in the background thread.
    """
    global bridge_active, bridge_thread, last_error, last_cleanup_error, _session

    from . import record as _record, rollout as _rollout, teleoperate as _teleop

    with _state_lock:
        if bridge_active:
            return {"success": False, "status_code": 409, "message": "The leader bridge is already running"}
        if bridge_thread is not None and bridge_thread.is_alive():
            return {
                "success": False,
                "status_code": 409,
                "message": "The previous bridge session is still shutting down. Try again in a few seconds.",
            }
        # Mutual exclusion (CLAUDE.md): everything that drives an arm on this
        # machine contends for the leader's serial port.
        if _teleop.teleoperation_active:
            return {
                "success": False,
                "status_code": 409,
                "message": "Teleoperation is currently active. Stop it first.",
            }
        if _record.recording_active:
            return {
                "success": False,
                "status_code": 409,
                "message": "Recording is currently active. Stop it first.",
            }
        if _rollout.inference_active:
            return {
                "success": False,
                "status_code": 409,
                "message": "Inference is currently active. Stop it first.",
            }
        bridge_active = True
        last_error = None
        last_cleanup_error = None
        _stop_event.clear()
        _session = None

    session: _BridgeSession | None = None
    leader = None
    try:
        server_url = _normalize_server_url(_require_str(payload, "server_url", "The server address"))
        robot_name = _require_str(payload, "robot_name", "The robot name")
        if not is_valid_robot_name(robot_name):
            raise _BridgeRequestError(f"Invalid robot name: {robot_name!r}")
        mode = str(payload.get("mode") or "single")
        if mode not in ("single", "bimanual"):
            raise _BridgeRequestError(f"Unknown mode {mode!r} — expected 'single' or 'bimanual'.")
        fps = _coerce_fps(payload.get("fps"))
        room = f"{ROOM_PREFIX}{robot_name}"
        identity = _client_identity()

        camera_names = _camera_names_from_payload(payload.get("cameras"))
        if camera_names is None:
            camera_names = _fetch_camera_names(server_url, robot_name)

        livekit_url, room, token = _fetch_portal_token(server_url, identity, room)

        leader = _build_leader(payload, mode, robot_name)
        identity_warnings = _connect_leader(leader, mode, payload)

        session = _BridgeSession(
            server_url=server_url,
            livekit_url=livekit_url,
            room=room,
            identity=identity,
            fps=fps,
            camera_names=camera_names,
            # lerobot orders action_features per arm; both sides of the wire
            # sort, so the fingerprints agree.
            action_keys=sorted(leader.action_features.keys()),
        )
        session.leader = leader
        _connect_operator(session, token)

        _session = session
        worker = threading.Thread(target=_bridge_worker, args=(session,), name="leader-bridge", daemon=True)
        bridge_thread = worker
        worker.start()

        message = f"Leader bridge started — streaming to '{room}' on {livekit_url}"
        logger.info(message)
        return {
            "success": True,
            "message": message,
            "room": room,
            "url": livekit_url,
            "identity": identity,
            "cameras": camera_names,
            "fps": fps,
            "warnings": identity_warnings,
        }
    except Exception as exc:
        # Nothing is running yet, so clean up whatever was built: the arm must
        # not be left energized and the port must not be left open.
        cleanup_error = None
        if session is not None:
            session.leader = leader
            cleanup_error = _shutdown_session(session)
        elif leader is not None:
            cleanup_error = _release_leader(leader)
        error_text = format_exception(exc)
        with _state_lock:
            bridge_active = False
            bridge_thread = None
            _session = None
            last_error = error_text
            last_cleanup_error = cleanup_error
        if isinstance(exc, _BridgeRequestError):
            logger.warning("Leader bridge start rejected: %s", exc)
            return {"success": False, "status_code": 400, "message": str(exc), "warning": cleanup_error}
        logger.exception("Leader bridge failed to start")
        return {
            "success": False,
            "status_code": 500,
            "message": str(exc) or error_text,
            "hint": friendly_hint(error_text),
            "warning": cleanup_error,
        }


def _coerce_fps(raw: object) -> int:
    """Clamp a requested fps into a sane range; anything unparseable → default."""
    try:
        fps = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return DEFAULT_FPS
    return max(1, min(120, fps))


def handle_leader_bridge_stop() -> dict[str, Any]:
    """Stop the session (POST /leader-bridge/stop).

    Signals the worker, which owns teardown (Portal disconnect, then the
    leader's torque release), and waits for it so a release problem can be
    surfaced instead of silently landing in the status payload later.
    """
    global bridge_active, bridge_thread

    worker = bridge_thread
    if not bridge_active and (worker is None or not worker.is_alive()):
        return {"success": False, "message": "No leader bridge session is active"}

    logger.info("Stop leader bridge triggered from web interface")
    with _state_lock:
        bridge_active = False
    _stop_event.set()

    if worker is not None and worker.is_alive():
        worker.join(timeout=STOP_JOIN_TIMEOUT_S)
        if worker.is_alive():
            logger.warning("Leader bridge worker did not exit within %.0fs", STOP_JOIN_TIMEOUT_S)
            return {
                "success": True,
                "message": "Stop requested, but the bridge worker has not shut down yet",
                "warning": (
                    f"The leader bridge worker did not shut down within {STOP_JOIN_TIMEOUT_S:.0f}s, so the "
                    "leader arm may not have been released. If it stays rigid, unplug its power."
                ),
            }
    bridge_thread = None
    if last_cleanup_error:
        return {
            "success": True,
            "message": "Leader bridge stopped, but releasing the leader arm reported a problem",
            "warning": last_cleanup_error,
        }
    return {"success": True, "message": "Leader bridge stopped"}


def handle_leader_bridge_status() -> dict[str, Any]:
    """Session + link health (GET /leader-bridge/status).

    ``rtt_ms`` is Portal's own round-trip metric, sampled once a second by the
    worker: it is the only honest signal of whether the link is good enough to
    teleoperate, and it is the first thing to look at when the follower feels
    laggy.
    """
    session = _session
    return {
        "active": bridge_active,
        "connected": bool(session is not None and session.connected),
        # False while connected means the follower is ignoring our actions —
        # the claim RPC has not landed yet (see _claim_control_with_retry).
        "claimed": bool(session is not None and session.claimed),
        "rtt_ms": session.rtt_ms if session is not None else None,
        "error": last_error,
        "hint": friendly_hint(last_error),
        # Extras the UI needs to render the session; not part of the contract.
        "room": session.room if session is not None else None,
        "server_url": session.server_url if session is not None else None,
        "cameras": list(session.camera_names) if session is not None else [],
        "fps": session.fps if session is not None else None,
        "warning": last_cleanup_error,
    }


def handle_leader_bridge_camera(name: str):
    """MJPEG re-serve of a Portal camera track (GET /leader-bridge/camera/{name}).

    Same wire format as ``/camera-preview/{index}`` (multipart/x-mixed-replace,
    boundary ``frame``, ~15 fps, JPEG quality 70) so ``BackendCameraStream``
    works against it unchanged — including its failure lifecycle, which reads
    ``detail`` off a non-200 JSON body to explain the dead tile.
    """
    session = _session
    if not bridge_active or session is None or not session.connected:
        return JSONResponse(
            status_code=409,
            content={"detail": "The leader bridge is not running — start it to view the server's cameras."},
        )
    if name not in session.camera_names:
        available = ", ".join(session.camera_names) or "none"
        return JSONResponse(
            status_code=404,
            content={"detail": f"No camera track named '{name}' in this session (available: {available})."},
        )
    # Portal delivers frames only once the server's loop is publishing them.
    # Wait briefly rather than answering 503 the instant a tile mounts — but
    # only briefly: server.py's route is `async def`, so this blocks the loop.
    first = session.frames.wait_for(name, -1, FIRST_FRAME_WAIT_S)
    if first is None:
        return JSONResponse(
            status_code=503,
            content={
                "detail": f"No video has arrived for '{name}' yet — the server may not be "
                "streaming this camera."
            },
        )
    return StreamingResponse(
        _mjpeg_frames(session.frames, name),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


def _mjpeg_frames(store: _FrameStore, name: str):
    """Yield multipart JPEG parts from Portal-received frames until stopped.

    Ends (rather than hanging) when the bridge stops or the track goes quiet,
    so the frontend's retry/backoff takes over instead of showing a frozen
    picture forever.
    """
    interval = 1.0 / TARGET_FPS
    last_seq = -1
    idle_since = time.monotonic()
    while True:
        started = time.monotonic()
        entry = store.wait_for(name, last_seq, FRAME_WAIT_S)
        if entry is None:
            if store.closed or time.monotonic() - idle_since > STALE_STREAM_S:
                logger.info("Ending MJPEG re-serve of '%s' (no frames)", name)
                return
            continue
        frame, last_seq = entry
        idle_since = time.monotonic()
        # Portal delivers packed RGB; cv2 encodes BGR.
        ok, jpeg = cv2.imencode(
            ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        if not ok:
            continue
        data = jpeg.tobytes()
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n"
            + f"Content-Length: {len(data)}\r\n\r\n".encode()
            + data
            + b"\r\n"
        )
        # Pace to ~TARGET_FPS; the wait above already absorbs slower sources.
        remaining = interval - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(remaining)
