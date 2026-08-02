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
"""Server-side wiring for a REMOTE LEADER session (docs/remote-portal/SPEC.md).

This machine keeps the real follower arm and the real cameras; only the
*leader* moves to another machine. So nothing about the follower changes —
what changes is that the teleop device stops being a serial-bus
``SO101Leader`` and becomes a network object: a ``LiveKitTeleoperator``
(from the ``lerobot-teleoperator-livekit`` plugin) that hands out actions
received over LiveKit Portal and publishes this side's observations back to
the operator.

Two pieces live here:

* :class:`RemoteLeaderSession` — the credentials + robot record for one
  session, resolved on the request thread (so a missing/disabled Portal host
  fails the *start response* instead of dying inside a worker thread) and
  used later to build the teleoperator once the local robot object exists.
* :class:`PortalTeeRobot` — a transparent wrapper around the local follower
  that additionally forwards every ``get_observation()`` to Portal. It is a
  *tee*, never a replacement: recording still reads frames straight off
  ``robot.get_observation()`` and writes them to disk uncompressed. The
  wrapper is also why lerobot's ``record_loop`` needs no modification.

Import discipline (SPEC "Hard rules" #1): NOTHING from ``livekit`` /
``livekit.portal`` / ``lerobot_teleoperator_livekit`` may be imported at
module import time — a machine without the Portal packages installed must
keep running MakerLab normally. Every such import is inside a function.
"""

import contextlib
import logging
import time
from typing import Any

from .utils.config import get_robot_record, is_valid_robot_name, leader_is_remote

logger = logging.getLogger(__name__)

# --- Wire contract (must match the client's leader-bridge exactly) ------------
# The room / Portal session name ("makerlab-<robot_name>") is owned by
# portal_host.room_name — the same string the client builds — and read from
# there rather than re-derived here, because a mismatch shows up only as "the
# arm never moves". Portal likewise fingerprints the schema (names + dtypes)
# and silently DROPS mismatched packets with only a WARN, which is why
# _log_resolved_schema below prints the resolved names/order at startup.
#
# LiveKit identity this side joins with. Distinct from any operator identity.
ROBOT_IDENTITY = "makerlab-robot"
# Default publish rate when the caller has no fps of its own (teleoperation;
# recording passes the session's dataset fps).
DEFAULT_PORTAL_FPS = 30

# Portal tuning — MUST match the operator side. Measured on real hardware
# (portal/examples/python/so101/RUNBOOK.local.md): with the defaults instead,
# RTT ran ~180 ms and ~15 state packets/second were dropped; with these it is
# ~15 ms and ~0 drops. Do not "tidy" these values.
PORTAL_STATE_RELIABLE = False
PORTAL_SLACK = 2
PORTAL_REUSE_STALE_FRAMES = True

# Seeding the "hold" action (the follower's own pose, used until the operator's
# first action arrives) reads real hardware, so a single glitched read must not
# decide the session: try a few times, briefly apart, across both sources.
SEED_ATTEMPTS = 3
SEED_RETRY_DELAY_S = 0.2
# How often the Robot side refreshes /portal/status's operator list from the
# live Portal session. Pushed from the control loop (the only thread that holds
# the session), so it is throttled to once a second — the FFI reads are cheap
# but the loop runs at fps.
PARTICIPANT_REFRESH_S = 1.0


def remote_leader_record(robot_name: str) -> dict | None:
    """The robot record named ``robot_name`` when its leader is REMOTE, else None.

    ``None`` is the "behave exactly as before" answer: an unknown/blank name, a
    missing record, or a record whose ``leader_source`` is ``"local"`` (the
    default, including every legacy record) all return None, so every existing
    single-machine flow keeps taking its original code path.
    """
    if not is_valid_robot_name(robot_name):
        return None
    record = get_robot_record(robot_name)
    if record is None or not leader_is_remote(record):
        return None
    return record


class RemoteLeaderSession:
    """Credentials + record for one server-side Portal session.

    Built by :meth:`resolve` on the *request* thread so that a Portal host that
    is missing, disabled, or unable to mint a token fails the start response
    with a legible message, instead of blowing up inside a worker thread where
    the only trace would be a log line.
    """

    def __init__(self, record: dict, session: str, url: str, token: str, fps: int) -> None:
        self.record = record
        self.session = session
        self.url = url
        self.token = token
        self.fps = fps

    @classmethod
    def resolve(cls, record: dict, fps: int = DEFAULT_PORTAL_FPS) -> "RemoteLeaderSession":
        """Mint this machine's Portal token for ``record``'s room.

        The LiveKit API secret never leaves the host: token minting lives in
        ``makerlab/portal_host.py`` (the server-mode feature module), which is
        imported lazily here so a client-only machine — or one without the
        Portal packages at all — never pays for it.
        """
        robot_name = (record or {}).get("name", "")

        try:
            from .portal_host import handle_portal_token, room_name
        except ImportError as e:  # pragma: no cover - depends on optional install
            raise RuntimeError(
                f"Robot '{robot_name}' is configured with a remote leader, but this machine's "
                "Portal host is unavailable (makerlab/portal_host.py could not be imported). "
                "Install the Portal packages on this machine, or set the robot's leader back "
                "to local."
            ) from e

        # The room name comes from portal_host, the same helper the client's
        # token request is answered with — never re-derived here.
        session = room_name(robot_name)
        result = handle_portal_token(ROBOT_IDENTITY, session) or {}
        url = result.get("url") or ""
        token = result.get("token") or ""
        if result.get("success") is False or not url or not token:
            detail = (result.get("message") or "no token was returned").rstrip(".")
            raise RuntimeError(
                f"Could not mint a Portal token for room '{session}': {detail}. "
                "Check that the LiveKit server is running on this machine (/portal/info)."
            )
        return cls(record=record, session=session, url=url, token=token, fps=fps)

    def build_teleoperator(self, robot: Any):
        """The remote leader as a lerobot Teleoperator wrapping ``robot`` (not connected)."""
        return build_remote_teleoperator(
            self.record,
            robot,
            session=self.session,
            url=self.url,
            token=self.token,
            fps=self.fps,
        )


def build_remote_teleoperator(
    record: dict,
    robot: Any,
    session: str,
    url: str,
    token: str,
    fps: int = DEFAULT_PORTAL_FPS,
):
    """A connect-ready ``LiveKitTeleoperator`` standing in for the remote leader.

    ``robot`` is the LOCAL follower (``SO101Follower`` / ``BiSOFollower``): the
    plugin introspects its ``action_features`` / ``observation_features`` to
    derive the wire schema, so the motor names and camera track names on the
    wire are exactly this robot's. No ordering is imposed here — the plugin
    sorts the keys itself (and the client's bridge sorts the same way), so the
    resolved order is read back off the constructed object for the log line
    rather than assumed. Returns the teleoperator unconnected — the caller owns
    ``connect()`` so it can order it against the hardware setup it already does
    (and so a connect failure lands in the caller's existing error path).
    """
    # Lazy import (SPEC hard rule #1): the Portal packages are an optional
    # install, and a local-leader machine must never need them.
    try:
        from lerobot_teleoperator_livekit import (
            LiveKitTeleoperator,
            LiveKitTeleoperatorConfig,
        )
    except ImportError as e:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "This robot's leader is remote, which needs the LiveKit Portal teleoperator "
            "plugin on this machine. Install 'lerobot-teleoperator-livekit' (and "
            "'livekit-portal'), or set the robot's leader back to local."
        ) from e

    class _RemoteLeader(LiveKitTeleoperator):
        """LiveKitTeleoperator that never hands back an EMPTY action.

        Portal's ``get_action()`` is latest-wins: it returns ``None`` only
        until the operator's first action arrives, and the plugin maps that to
        ``{}``. An empty action is poison for both consumers here — lerobot's
        ``build_dataset_frame`` raises ``KeyError`` on a missing motor key
        (recording would die on its first frame), and ``send_action({})``
        writes an empty goal packet to the bus. So the gap is filled: with the
        last action actually received, or — before any has arrived — with the
        follower's own joint positions, i.e. "stay where you are". Both are
        truthful: an operator holding the leader still produces the same
        stream. This is what lets the hot loops stay unchanged.

        The gap-filler is only as good as the seed, so ``connect()`` REFUSES
        the session when it cannot read the follower's pose (retried, and from
        the buses directly if ``get_observation()`` — which also reads the
        cameras — fails). Failing the start is the safe end of that trade:
        recording's first frame would otherwise ``KeyError`` deep inside
        lerobot's ``record_loop``, mid-session, with the arm already energized.
        """

        def __init__(self, config, robot=None):
            super().__init__(config, robot=robot)
            self._local_robot = robot
            self._hold_action: dict[str, Any] = {}
            self._next_participant_report = 0.0

        def connect(self, calibrate: bool = True) -> None:
            super().connect(calibrate)
            try:
                self._seed_hold_action()
            except Exception:
                # Connected but unusable: an unseeded hold action is exactly
                # what lets {} reach send_action()/build_dataset_frame. Close
                # the Portal session here so the caller's setup-failure cleanup
                # doesn't have to know this half-state exists, then fail the
                # start.
                with contextlib.suppress(Exception):
                    super().disconnect()
                raise
            _set_active_room(session)
            self._report_participants()

        def disconnect(self) -> None:
            try:
                super().disconnect()
            finally:
                _set_active_room(None)
                _publish_participants(None)

        def _seed_hold_action(self) -> None:
            """Seed the hold action from the follower's current pose, or raise.

            Retried: the read touches real hardware (and, through
            ``get_observation()``, the cameras), and a single ``read_latest()``
            timeout is an ordinary event — it must not be what decides that the
            session runs without a hold action.
            """
            robot_obj = self._local_robot
            if robot_obj is None:  # pragma: no cover - both call sites pass the follower
                logger.warning(
                    "The remote leader was built without a local follower, so its hold action "
                    "cannot be seeded; the first frames wait for the operator instead."
                )
                return
            for attempt in range(1, SEED_ATTEMPTS + 1):
                seeded = _follower_hold_action(robot_obj, self.action_features)
                if seeded:
                    self._hold_action = seeded
                    logger.info(
                        "Remote leader hold action seeded from the follower's current pose "
                        "(%d motors) — the arm holds this pose until the operator's first action.",
                        len(seeded),
                    )
                    return
                if attempt < SEED_ATTEMPTS:
                    time.sleep(SEED_RETRY_DELAY_S)
            raise RuntimeError(
                "Could not read the follower's current pose after "
                f"{SEED_ATTEMPTS} attempts, so the remote leader has nothing to hold until the "
                "operator's first action arrives. Refusing to start rather than send an empty "
                "action to the servo bus — check the follower arm and its cameras, then retry."
            )

        def get_action(self) -> dict[str, Any]:
            self._maybe_report_participants()
            action = super().get_action()
            if action:
                self._hold_action = action
                return action
            return dict(self._hold_action)

        def _maybe_report_participants(self) -> None:
            """Refresh /portal/status's operator list, at most once a second."""
            now = time.monotonic()
            if now < self._next_participant_report:
                return
            self._next_participant_report = now + PARTICIPANT_REFRESH_S
            self._report_participants()

        def _report_participants(self) -> None:
            # The plugin's PortalRobot instance is the only object that knows
            # who is in the room; read it defensively (same reason as
            # _log_resolved_schema) so a plugin refactor degrades to "no
            # operators reported" instead of breaking teleoperation.
            _publish_participants(getattr(self, "_portal", None))

    config = LiveKitTeleoperatorConfig(
        url=url,
        token=token,
        session=session,
        fps=fps,
        # Tuning that must match the operator side, byte for byte.
        state_reliable=PORTAL_STATE_RELIABLE,
        slack=PORTAL_SLACK,
        reuse_stale_frames=PORTAL_REUSE_STALE_FRAMES,
    )
    teleoperator = _RemoteLeader(config, robot=robot)
    _log_resolved_schema(record, teleoperator, session, fps)
    return teleoperator


def _follower_bus_positions(robot: Any) -> dict[str, float]:
    """``{"<prefix><motor>.pos": value}`` read straight off the follower bus(es).

    Camera-free by construction — that is the whole point: the fallback exists
    for the case where ``get_observation()`` failed because a CAMERA read timed
    out, which says nothing about the motors. Key shape and the bimanual
    ``left_``/``right_`` prefixes follow lerobot's own observation layout (and
    the same bus order teleoperate.py pairs prefixes with).
    """
    # Lazy: teleoperate.py imports this module at import time, so the edge can
    # only be walked in this direction from inside a function.
    from .teleoperate import _device_buses

    buses = _device_buses(robot)
    prefixes = ["left_", "right_"] if len(buses) > 1 else [""]
    positions: dict[str, float] = {}
    for bus, prefix in zip(buses, prefixes, strict=False):
        for motor, value in (bus.sync_read("Present_Position") or {}).items():
            positions[f"{prefix}{motor}.pos"] = float(value)
    return positions


def _follower_hold_action(robot: Any, action_features: dict) -> dict[str, float]:
    """The follower's current pose keyed like ``action_features``; ``{}`` if unreadable.

    Two sources, in order: the robot's own ``get_observation()`` (exactly what a
    local leader's first frame looks like), then a direct bus read. Partial
    results are rejected — a hold action missing one motor is still a
    ``KeyError`` inside ``build_dataset_frame``.
    """
    wanted = list(action_features or {})
    if not wanted:
        return {}
    sources = (
        (lambda: robot.get_observation(), "the follower's observation"),
        (lambda: _follower_bus_positions(robot), "the follower's motor bus(es)"),
    )
    for read, label in sources:
        try:
            pose = read() or {}
        except Exception as e:
            logger.warning("Could not read %s to seed the remote leader's hold action: %s", label, e)
            continue
        seeded = {key: float(pose[key]) for key in wanted if key in pose}
        if len(seeded) == len(wanted):
            return seeded
        logger.warning(
            "Reading %s produced %d of the %d motor positions the wire schema needs.",
            label,
            len(seeded),
            len(wanted),
        )
    return {}


def _metrics_summary(portal: Any) -> dict | None:
    """A small JSON-safe slice of Portal's metrics for /portal/status, or None."""
    try:
        metrics = portal.metrics()
    except Exception as e:  # pragma: no cover - diagnostics only, never fatal
        logger.debug("Reading Portal metrics failed: %s", e)
        return None
    rtt_us = getattr(getattr(metrics, "rtt", None), "rtt_us_last", None)
    transport = getattr(metrics, "transport", None)
    sync = getattr(metrics, "sync", None)
    return {
        "rtt_ms": round(rtt_us / 1e3, 1) if rtt_us else None,
        "actions_received": getattr(transport, "actions_received", None),
        "states_sent": getattr(transport, "states_sent", None),
        "states_dropped": getattr(sync, "states_dropped", None),
    }


def _publish_participants(portal: Any | None) -> None:
    """Push the live operator list to portal_host for /portal/status.

    Best effort in the strongest sense: this runs from the control loop (the
    only thread holding the Portal session), so it must never raise — a status
    endpoint cannot be allowed to stop the arm. ``None`` clears the list, which
    is what a finished session should advertise.
    """
    try:
        from .portal_host import report_participants

        if portal is None:
            report_participants([], None, None)
            return
        report_participants(
            operators=portal.operators(),
            active_operator=portal.active_operator(),
            metrics=_metrics_summary(portal),
        )
    except Exception as e:  # pragma: no cover - purely informational bookkeeping
        logger.debug("Could not report the Portal operator list: %s", e)


def _set_active_room(room: str | None) -> None:
    """Tell portal_host which room this side is serving (best effort, never raises).

    ``/portal/info`` and ``/portal/status`` report it, and on a station with
    more than one robot record it is the only way they can know which room is
    live. Cleared on disconnect so a finished session stops advertising a room.
    """
    try:
        from .portal_host import set_active_room

        set_active_room(room)
    except Exception as e:  # pragma: no cover - purely informational bookkeeping
        logger.debug("Could not record the active Portal room (%s): %s", room, e)


def _log_resolved_schema(record: dict, teleoperator: Any, session: str, fps: int) -> None:
    """Log the wire schema this side RESOLVED — the actual names, in their actual order.

    Portal fingerprints the action/state schema and silently drops packets that
    don't match, warning only. The symptom is "the arm doesn't move", with
    nothing in the UI to explain it. Printing both sides' resolved schema is the
    only cheap way to diagnose that, so this runs on every session start.

    The order is whatever the plugin chose (it sorts the feature keys); it is
    read back off the constructed teleoperator, never assumed here.
    """
    # The plugin's resolved wire names (motor names WITHOUT the ".pos" suffix,
    # which is what Portal actually fingerprints). Read defensively — fall back
    # to the public feature dicts if the plugin's internals ever move.
    action_names = list(getattr(teleoperator, "_action_motors", None) or [])
    state_names = list(getattr(teleoperator, "_state_motors", None) or [])
    camera_names = list(getattr(teleoperator, "_camera_names", None) or [])
    if not action_names:
        action_names = list(getattr(teleoperator, "action_features", {}) or {})
    if not state_names:
        state_names = [
            key
            for key, value in (getattr(teleoperator, "feedback_features", {}) or {}).items()
            if key not in camera_names
        ]

    logger.info(
        "Portal remote leader for robot %r (mode=%s): session=%r fps=%d "
        "state_reliable=%s slack=%s reuse_stale_frames=%s",
        (record or {}).get("name", ""),
        (record or {}).get("mode", "single"),
        session,
        fps,
        PORTAL_STATE_RELIABLE,
        PORTAL_SLACK,
        PORTAL_REUSE_STALE_FRAMES,
    )
    logger.info(
        "Portal schema resolved on THIS side — action (%d): %s | state (%d): %s | video: %s. "
        "The operator MUST resolve the same names in the same order; Portal drops "
        "schema-mismatched packets with only a WARN (symptom: the arm doesn't move).",
        len(action_names),
        ", ".join(action_names) or "<none>",
        len(state_names),
        ", ".join(state_names) or "<none>",
        ", ".join(camera_names) or "<none>",
    )


class PortalTeeRobot:
    """The local follower, plus a Portal publish on every ``get_observation()``.

    Every attribute except ``get_observation`` is delegated to the wrapped
    ``SO101Follower`` / ``BiSOFollower``, so callers (lerobot's ``record_loop``,
    the teleoperation worker, the identity guard, the torque-release cleanup)
    see the real robot: same ``name``, same ``action_features`` /
    ``observation_features``, same buses, same ``send_action``. That is
    deliberate — it is what keeps the dataset path untouched (frames still come
    from ``get_observation()`` straight to disk, uncompressed) and what makes
    modifying lerobot's ``record_loop`` unnecessary.

    Publishing is best-effort and can NEVER raise into the control loop: a
    dropped Portal link must not stop the arm mid-motion. The first failure is
    logged, the rest are counted and summarized once the link recovers.
    """

    def __init__(self, robot: Any, teleoperator: Any) -> None:
        self._robot = robot
        self._teleoperator = teleoperator
        self._publish_failures = 0

    @property
    def wrapped(self) -> Any:
        """The real local follower underneath the tee."""
        return self._robot

    def get_observation(self) -> dict[str, Any]:
        observation = self._robot.get_observation()
        self._publish(observation)
        return observation

    def _publish(self, observation: dict[str, Any]) -> None:
        try:
            self._teleoperator.send_feedback(observation)
        except Exception as e:
            self._publish_failures += 1
            if self._publish_failures == 1:
                logger.warning(
                    "Publishing an observation to Portal failed (%s). The arm keeps being "
                    "driven; the remote operator may lose video/state until the link recovers.",
                    e,
                )
            return
        if self._publish_failures:
            logger.info("Portal publishing recovered after %d failed observation(s).", self._publish_failures)
            self._publish_failures = 0

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes this wrapper does not define itself.
        # object.__getattribute__ avoids recursing through __getattr__ if
        # _robot is somehow missing (e.g. during interpreter teardown).
        try:
            robot = object.__getattribute__(self, "_robot")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(robot, name)

    def __repr__(self) -> str:
        return f"PortalTeeRobot({self._robot!r})"
