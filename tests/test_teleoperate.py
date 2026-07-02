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
"""Tests for lelab.teleoperate — request schema and status handlers."""

from __future__ import annotations

import pytest


def test_teleoperate_request_rejects_missing_fields() -> None:
    from pydantic import ValidationError

    from lelab.teleoperate import TeleoperateRequest

    with pytest.raises(ValidationError):
        TeleoperateRequest()


def test_teleoperate_request_defaults_to_single_arm() -> None:
    """A single-arm request omits the bimanual fields; they default safely."""
    from lelab.teleoperate import TeleoperateRequest

    req = TeleoperateRequest(
        leader_port="/dev/l", follower_port="/dev/f",
        leader_config="L", follower_config="F",
    )
    assert req.mode == "single"
    assert req.right_leader_port == ""
    assert req.right_follower_config == ""


def test_handle_teleoperation_status_returns_dict() -> None:
    from lelab.teleoperate import handle_teleoperation_status

    result = handle_teleoperation_status()
    assert isinstance(result, dict)


def test_handle_get_joint_positions_returns_dict_when_idle() -> None:
    from lelab.teleoperate import handle_get_joint_positions

    result = handle_get_joint_positions()
    assert isinstance(result, dict)


def test_get_joint_positions_from_robot_uses_provided_object() -> None:
    from lelab.teleoperate import get_joint_positions_from_robot
    from tests.mocks import FakeRobot

    robot = FakeRobot()
    robot.connect()
    positions = get_joint_positions_from_robot(robot)
    assert isinstance(positions, dict)


def test_start_teleoperation_reports_connection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A device that fails to connect must make the start handler return
    success=False (so the UI surfaces the error and doesn't navigate to an
    empty teleop screen) and reset state so a retry isn't blocked. Previously
    the connect ran in a worker thread and the handler always claimed success.
    """
    import lelab.teleoperate as teleop

    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(teleop, "setup_calibration_files", lambda leader, follower: ("leader", "follower"))

    class _Bus:
        def connect(self) -> None:
            raise RuntimeError("serial port unavailable")

    class _Device:
        def __init__(self, config) -> None:
            self.bus = _Bus()
            self.cameras: dict = {}
            self.disconnected = False

        def disconnect(self) -> None:
            self.disconnected = True

    monkeypatch.setattr(teleop, "SO101Follower", _Device)
    monkeypatch.setattr(teleop, "SO101Leader", _Device)

    request = teleop.TeleoperateRequest(
        leader_port="COM_LEADER",
        follower_port="COM_FOLLOWER",
        leader_config="leader",
        follower_config="follower",
    )
    result = teleop.handle_start_teleoperation(request)

    assert result["success"] is False
    # The message must name the arm that failed (the follower connects first).
    assert "follower" in result["message"].lower()
    assert "COM_FOLLOWER" in result["message"]
    # State must be reset so the next attempt isn't blocked by the mutex.
    assert teleop.teleoperation_active is False


def test_start_teleoperation_disconnects_follower_when_leader_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The partial-connect path: if the follower connects but the leader then
    fails, the follower must be disconnected so its serial port is released.
    """
    import lelab.teleoperate as teleop

    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(teleop, "setup_calibration_files", lambda leader, follower: ("leader", "follower"))

    class _OkBus:
        def connect(self) -> None:
            pass

    class _FailingBus:
        def connect(self) -> None:
            raise RuntimeError("leader offline")

    class _Follower:
        def __init__(self, config) -> None:
            self.bus = _OkBus()
            self.cameras: dict = {}
            self.disconnected = False

        def disconnect(self) -> None:
            self.disconnected = True

    class _Leader:
        def __init__(self, config) -> None:
            self.bus = _FailingBus()
            self.disconnected = False

        def disconnect(self) -> None:
            self.disconnected = True

    created: dict = {}
    monkeypatch.setattr(
        teleop, "SO101Follower", lambda config: created.setdefault("follower", _Follower(config))
    )
    monkeypatch.setattr(teleop, "SO101Leader", lambda config: created.setdefault("leader", _Leader(config)))

    request = teleop.TeleoperateRequest(
        leader_port="COM_LEADER",
        follower_port="COM_FOLLOWER",
        leader_config="leader",
        follower_config="follower",
    )
    result = teleop.handle_start_teleoperation(request)

    assert result["success"] is False
    assert "leader" in result["message"].lower()
    # The already-connected follower must have been cleaned up.
    assert created["follower"].disconnected is True
    assert teleop.teleoperation_active is False


class _FakeBus:
    """Motor bus double for the explicit torque-disable cleanup step."""

    def __init__(self, port: str = "COM_FAKE", failing: tuple[str, ...] = ()) -> None:
        self.port = port
        self.motors = {"shoulder_pan": 1, "elbow_flex": 3, "gripper": 6}
        self.failing = set(failing)
        self.disabled: list[tuple[str, int]] = []

    def disable_torque(self, motor: str, num_retry: int = 0) -> None:
        if motor in self.failing:
            raise ConnectionError(f"no response from {motor}")
        self.disabled.append((motor, num_retry))


class _FakeArm:
    def __init__(self, bus: _FakeBus) -> None:
        self.bus = bus


def test_force_disable_torque_disables_every_motor() -> None:
    from lelab.teleoperate import force_disable_torque

    bus = _FakeBus()
    problems = force_disable_torque(_FakeArm(bus), "follower arm")

    assert problems == []
    # Every motor is disabled individually, with retries.
    assert [motor for motor, _ in bus.disabled] == list(bus.motors)
    assert all(num_retry == 5 for _, num_retry in bus.disabled)


def test_force_disable_torque_reports_failed_motor_and_port() -> None:
    """One bad motor must not stop the others from being released, and the
    problem message must be unmistakable: it names the port and warns that
    torque may still be enabled (the arm stays rigid until power is pulled).
    """
    from lelab.teleoperate import force_disable_torque

    bus = _FakeBus(port="COM_FOLLOWER", failing=("elbow_flex",))
    problems = force_disable_torque(_FakeArm(bus), "follower arm")

    assert len(problems) == 1
    assert "TORQUE MAY STILL BE ENABLED" in problems[0]
    assert "COM_FOLLOWER" in problems[0]
    assert "elbow_flex" in problems[0]
    # The remaining motors were still disabled despite the failure.
    assert [motor for motor, _ in bus.disabled] == ["shoulder_pan", "gripper"]


def test_force_disable_torque_handles_bimanual_and_none() -> None:
    from lelab.teleoperate import force_disable_torque

    class _BiDevice:
        def __init__(self) -> None:
            self.left_arm = _FakeArm(_FakeBus(port="COM_LEFT"))
            self.right_arm = _FakeArm(_FakeBus(port="COM_RIGHT", failing=("gripper",)))

    device = _BiDevice()
    problems = force_disable_torque(device, "follower arms")

    # Both sub-arm buses are handled; only the right one reports a problem.
    assert len(device.left_arm.bus.disabled) == 3
    assert len(problems) == 1
    assert "COM_RIGHT" in problems[0]

    assert force_disable_torque(None, "nothing") == []


def test_stop_teleoperation_surfaces_cleanup_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the worker's cleanup could not release an arm, the stop response
    must carry a warning instead of claiming a clean disconnect.
    """
    import lelab.teleoperate as teleop

    monkeypatch.setattr(teleop, "teleoperation_active", True)
    monkeypatch.setattr(teleop, "teleoperation_thread", None)
    monkeypatch.setattr(
        teleop, "last_cleanup_error", "TORQUE MAY STILL BE ENABLED on COM_FOLLOWER (follower arm)."
    )

    result = teleop.handle_stop_teleoperation()

    assert result["success"] is True
    assert "TORQUE MAY STILL BE ENABLED" in result["warning"]
    assert teleop.teleoperation_active is False


class _FakeWorker:
    """Thread double: reports alive until joined."""

    def __init__(self, alive: bool = True) -> None:
        self._alive = alive
        self.joined = False

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:
        self.joined = True
        self._alive = False


def test_hold_torque_release_grace_cut_short_by_release_request() -> None:
    """A set release event must end the hold immediately (no 5s sleep)."""
    import threading
    import time

    from lelab.teleoperate import hold_torque_release_grace

    release_now = threading.Event()
    release_now.set()
    start = time.monotonic()
    assert hold_torque_release_grace(release_now, grace_s=30.0) is True
    assert time.monotonic() - start < 1.0


def test_hold_torque_release_grace_elapses_without_release_request() -> None:
    import threading

    from lelab.teleoperate import hold_torque_release_grace

    assert hold_torque_release_grace(threading.Event(), grace_s=0.01) is False


def test_stop_teleoperation_enters_release_return(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first stop of a live session must return immediately (not block
    through the rest-pose return), report `releasing`, and tell the user the
    arm goes back to its starting position and that a second Stop releases it
    now. There is no timed hold anymore — same behavior as the auto-cal stop.
    """
    import lelab.teleoperate as teleop

    worker = _FakeWorker()
    monkeypatch.setattr(teleop, "teleoperation_active", True)
    monkeypatch.setattr(teleop, "teleoperation_thread", worker)
    monkeypatch.setattr(teleop, "last_cleanup_error", None)

    result = teleop.handle_stop_teleoperation()

    assert result["success"] is True
    assert result["releasing"] is True
    assert "returns to its starting position" in result["message"]
    assert "holds its pose" not in result["message"]  # the hold phase is gone
    assert "Stop again" in result["message"]
    # The response must not join through the return.
    assert worker.joined is False
    assert teleop.teleoperation_active is False


def test_second_stop_during_grace_releases_now(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pressing Stop again during the grace hold is the 'skip the wait'
    gesture: it must set the release event and wait for the worker's cleanup.
    """
    import threading

    import lelab.teleoperate as teleop

    worker = _FakeWorker()
    release_now = threading.Event()
    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(teleop, "teleoperation_thread", worker)
    monkeypatch.setattr(teleop, "_release_now", release_now)
    monkeypatch.setattr(teleop, "last_cleanup_error", None)

    result = teleop.handle_stop_teleoperation()

    assert result["success"] is True
    assert release_now.is_set()
    assert worker.joined is True
    assert teleop.teleoperation_thread is None


def test_second_stop_during_grace_surfaces_cleanup_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    import lelab.teleoperate as teleop

    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(teleop, "teleoperation_thread", _FakeWorker())
    monkeypatch.setattr(teleop, "_release_now", threading.Event())
    monkeypatch.setattr(
        teleop, "last_cleanup_error", "TORQUE MAY STILL BE ENABLED on COM_FOLLOWER (follower arm)."
    )

    result = teleop.handle_stop_teleoperation()

    assert result["success"] is True
    assert "TORQUE MAY STILL BE ENABLED" in result["warning"]


def test_finish_pending_release_cuts_grace_short(monkeypatch: pytest.MonkeyPatch) -> None:
    """A start arriving during the grace hold must release the arms and free
    the ports instead of failing port-busy for the rest of the grace.
    """
    import threading

    import lelab.teleoperate as teleop

    worker = _FakeWorker()
    release_now = threading.Event()
    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(teleop, "teleoperation_thread", worker)
    monkeypatch.setattr(teleop, "_release_now", release_now)

    assert teleop.finish_pending_release() is True
    assert release_now.is_set()
    assert worker.joined is True
    assert teleop.teleoperation_thread is None


def test_finish_pending_release_leaves_live_session_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live session (active flag still set) is not a pending release: the
    caller's mutex check reports it, and torque must stay untouched.
    """
    import threading

    import lelab.teleoperate as teleop

    worker = _FakeWorker()
    release_now = threading.Event()
    monkeypatch.setattr(teleop, "teleoperation_active", True)
    monkeypatch.setattr(teleop, "teleoperation_thread", worker)
    monkeypatch.setattr(teleop, "_release_now", release_now)

    assert teleop.finish_pending_release() is False
    assert not release_now.is_set()
    assert worker.joined is False


def test_finish_pending_release_noop_when_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    import lelab.teleoperate as teleop

    monkeypatch.setattr(teleop, "teleoperation_thread", None)
    assert teleop.finish_pending_release() is True


def test_teleoperation_status_reports_releasing(monkeypatch: pytest.MonkeyPatch) -> None:
    """During the post-stop return the status must say the arm is still
    energized and going home (releasing) rather than pretending the session
    is fully over.
    """
    import lelab.teleoperate as teleop

    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(teleop, "releasing", True)

    status = teleop.handle_teleoperation_status()

    assert status["teleoperation_active"] is False
    assert status["releasing"] is True
    assert "returning the arm" in status["message"].lower()


# ---------------------------------------------------------------------------
# Rest-pose return (lelab.rest_pose) and its stop-path integration
# ---------------------------------------------------------------------------


class _RestBus:
    """Bus double for rest-pose capture/return (lelab.rest_pose)."""

    _MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    def __init__(self, positions=None, moving: int = 1, port: str = "COM_FOLLOWER") -> None:
        self.port = port
        self.motors = dict.fromkeys(self._MOTORS)
        self.positions = dict.fromkeys(self._MOTORS, 1000) if positions is None else dict(positions)
        self.moving = moving
        self.fail_reads = False
        self.writes: list[tuple] = []
        self.sync_writes: list[tuple] = []

    def sync_read(self, reg: str, normalize: bool = True) -> dict:
        if self.fail_reads:
            raise ConnectionError("bus gone")
        if reg == "Present_Position":
            return dict(self.positions)
        if reg == "Moving":
            return dict.fromkeys(self.positions, self.moving)
        raise KeyError(reg)

    def write(self, reg: str, motor: str, value: int, normalize: bool = True) -> None:
        self.writes.append((reg, motor, value))

    def sync_write(self, reg: str, values: dict, normalize: bool = True) -> None:
        self.sync_writes.append((reg, dict(values)))


class _RestClock:
    """Simulated time: sleep() advances monotonic()."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def rest_clock(monkeypatch: pytest.MonkeyPatch) -> _RestClock:
    """Drive lelab.rest_pose's time off a simulated clock (no real sleeps)."""
    import lelab.rest_pose as rest_pose

    clock = _RestClock()
    monkeypatch.setattr(rest_pose.time, "sleep", clock.sleep)
    monkeypatch.setattr(rest_pose.time, "monotonic", clock.monotonic)
    return clock


def test_capture_rest_pose_reads_raw_ticks() -> None:
    """Raw Present_Position is captured as-is: teleoperation never rewrites
    Homing_Offset mid-session, so raw ticks are directly replayable later."""
    from lelab.rest_pose import capture_rest_pose

    bus = _RestBus(positions={"shoulder_pan": 123, "gripper": 90})
    assert capture_rest_pose(bus) == {"shoulder_pan": 123, "gripper": 90}

    bus.fail_reads = True
    assert capture_rest_pose(bus) == {}  # never raises — the session must still start


def test_return_to_rest_pose_arrives_and_writes_gentle_goals(rest_clock: _RestClock) -> None:
    """The return writes a gentle profile speed then the captured goals, and
    reports 'returned' once every motor is within tolerance."""
    import lelab.rest_pose as rest_pose

    targets = {"shoulder_pan": 1000, "shoulder_lift": 1005}
    bus = _RestBus(positions={"shoulder_pan": 1000, "shoulder_lift": 1000})

    arrived, reason = rest_pose.return_to_rest_pose(bus, targets, label="follower arm")

    assert arrived is True
    # The completion report carries per-motor |final - target| deltas so a
    # subtly-off landing is diagnosable from the log.
    assert reason.startswith("returned: max delta 5 ticks")
    assert "shoulder_lift=5" in reason
    # Goals land via one sync_write; the finally-restore then zeroes the gentle
    # speed cap (Goal_Velocity is RAM-persistent — a leftover 400 would
    # throttle the next session's follower until a power cycle).
    assert bus.sync_writes == [
        ("Goal_Position", targets),
        ("Goal_Velocity", dict.fromkeys(targets, 0)),
    ]
    speed_writes = [w for w in bus.writes if w[0] == "Goal_Velocity"]
    assert {w[2] for w in speed_writes} == {rest_pose.RETURN_POS_SPEED}
    assert {w[1] for w in speed_writes} == set(targets)


def test_return_to_rest_pose_stalls_without_progress(rest_clock: _RestClock) -> None:
    """Positions that never move toward the target must end in a stall (and
    fall through to the release) instead of looping to the ceiling."""
    from lelab.rest_pose import return_to_rest_pose

    bus = _RestBus(positions={"shoulder_pan": 1000})
    arrived, reason = return_to_rest_pose(bus, {"shoulder_pan": 2000})

    assert arrived is False
    assert reason.startswith("stalled")
    assert "shoulder_pan=1000" in reason  # the culprit and its distance are named
    import lelab.rest_pose as rest_pose

    assert rest_clock.now < rest_pose.RETURN_CEILING_S  # stall beat the ceiling


def test_return_to_rest_pose_reports_settled_short_motor(rest_clock: _RestClock) -> None:
    """A motor that stops moving (Moving == 0) while still far from target is
    NOT a successful return — bench symptom: 'the starting position was not
    right'. It must be reported as its own 'settled' outcome with the deltas."""
    from lelab.rest_pose import return_to_rest_pose

    bus = _RestBus(positions={"shoulder_pan": 1000}, moving=0)
    arrived, reason = return_to_rest_pose(bus, {"shoulder_pan": 2000})

    assert arrived is False
    assert reason.startswith("settled short of target")
    assert "shoulder_pan=1000" in reason


def test_return_to_rest_pose_cut_short_by_abort(rest_clock: _RestClock) -> None:
    """A set abort event (second stop, or a new session start) must end the
    return immediately so the release can run right away."""
    import threading

    from lelab.rest_pose import return_to_rest_pose

    abort = threading.Event()
    abort.set()
    bus = _RestBus(positions={"shoulder_pan": 1000})

    arrived, reason = return_to_rest_pose(bus, {"shoulder_pan": 2000}, abort_event=abort)

    assert (arrived, reason) == (False, "cut-short")


def test_return_to_rest_pose_without_pose_is_a_noop() -> None:
    from lelab.rest_pose import return_to_rest_pose

    bus = _RestBus()
    assert return_to_rest_pose(bus, {}) == (False, "no-pose")
    assert bus.sync_writes == []  # nothing written — straight to the release


def _assert_speed_cap_restored(bus: _RestBus, targets: dict[str, int]) -> None:
    """The last sync_write must zero the gentle Goal_Velocity cap on exactly
    the motors the return drove (RAM-persistent: a leftover cap would throttle
    the next session's follower until a power cycle)."""
    assert bus.sync_writes[-1] == ("Goal_Velocity", dict.fromkeys(targets, 0))


def test_return_restores_speed_cap_on_stall(rest_clock: _RestClock) -> None:
    from lelab.rest_pose import return_to_rest_pose

    bus = _RestBus(positions={"shoulder_pan": 1000})
    arrived, reason = return_to_rest_pose(bus, {"shoulder_pan": 2000})

    assert (arrived, reason[:7]) == (False, "stalled")
    _assert_speed_cap_restored(bus, {"shoulder_pan": 2000})


def test_return_restores_speed_cap_on_settled(rest_clock: _RestClock) -> None:
    from lelab.rest_pose import return_to_rest_pose

    bus = _RestBus(positions={"shoulder_pan": 1000}, moving=0)
    arrived, reason = return_to_rest_pose(bus, {"shoulder_pan": 2000})

    assert arrived is False
    assert reason.startswith("settled")
    _assert_speed_cap_restored(bus, {"shoulder_pan": 2000})


def test_return_restores_speed_cap_on_cut_short(rest_clock: _RestClock) -> None:
    import threading

    from lelab.rest_pose import return_to_rest_pose

    abort = threading.Event()
    abort.set()
    bus = _RestBus(positions={"shoulder_pan": 1000})

    assert return_to_rest_pose(bus, {"shoulder_pan": 2000}, abort_event=abort) == (
        False,
        "cut-short",
    )
    _assert_speed_cap_restored(bus, {"shoulder_pan": 2000})


def test_return_restores_speed_cap_on_ceiling(
    rest_clock: _RestClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force the pathological ceiling exit (positions creep just enough to
    never stall) and check the cap is still zeroed on the way out."""
    import lelab.rest_pose as rest_pose

    bus = _RestBus(positions={"shoulder_pan": 1000})
    original_sync_read = bus.sync_read

    def _creeping_read(reg: str, normalize: bool = True) -> dict:
        # Enough progress per poll to keep resetting the stall window.
        bus.positions["shoulder_pan"] += rest_pose.RETURN_STALL_MIN_PROGRESS + 1
        return original_sync_read(reg, normalize)

    monkeypatch.setattr(bus, "sync_read", _creeping_read)
    arrived, reason = rest_pose.return_to_rest_pose(bus, {"shoulder_pan": 10**6})

    assert arrived is False
    assert reason.startswith("ceiling")
    _assert_speed_cap_restored(bus, {"shoulder_pan": 10**6})


def test_return_restores_speed_cap_on_failed_start(rest_clock: _RestClock) -> None:
    """A comm-error while writing the goals may have already stamped the
    gentle cap on some motors — the best-effort zeroing must still run."""
    from lelab.rest_pose import return_to_rest_pose

    bus = _RestBus(positions={"shoulder_pan": 1000})

    def _failing_write(reg: str, motor: str, value: int, normalize: bool = True) -> None:
        raise ConnectionError("bus gone")

    bus.write = _failing_write
    arrived, reason = return_to_rest_pose(bus, {"shoulder_pan": 2000})

    assert arrived is False
    assert reason.startswith("comm-error")
    _assert_speed_cap_restored(bus, {"shoulder_pan": 2000})


def test_return_speed_cap_restore_failure_never_raises(rest_clock: _RestClock) -> None:
    """The zeroing is best-effort: a dead bus at restore time must not raise —
    the caller's torque release has to run no matter what."""
    from lelab.rest_pose import return_to_rest_pose

    targets = {"shoulder_pan": 1000}
    bus = _RestBus(positions={"shoulder_pan": 1000})
    original_sync_write = bus.sync_write

    def _failing_sync_write(reg: str, values: dict, normalize: bool = True) -> None:
        if reg == "Goal_Velocity":
            raise ConnectionError("bus gone")
        original_sync_write(reg, values, normalize)

    bus.sync_write = _failing_sync_write
    arrived, reason = return_to_rest_pose(bus, targets)

    assert arrived is True  # the return itself still completed and reported
    assert reason.startswith("returned")


def test_return_followers_to_rest_covers_every_follower_bus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bimanual: both follower buses get their return (the leader is never in
    the list — it is human-held with torque off)."""
    import threading

    import lelab.teleoperate as teleop

    calls: list[tuple] = []

    def _spy(bus, pose, abort_event=None, label=""):
        calls.append((bus, pose, abort_event))
        return True, "returned"

    monkeypatch.setattr(teleop, "return_to_rest_pose", _spy)
    abort = threading.Event()
    teleop._return_followers_to_rest([("busL", {"m": 1}), ("busR", {"m": 2})], abort)

    assert [(c[0], c[1]) for c in calls] == [("busL", {"m": 1}), ("busR", {"m": 2})]
    # The worker's abort event is passed through so a second stop cuts the return.
    assert all(c[2] is abort for c in calls)


def test_start_clears_stale_release_state_from_previous_double_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cross-session leak regression: session 1's double-stop sets
    _release_now; if session 2's start didn't clear it (under the state lock),
    every later grace hold AND rest-pose return would be cut short instantly
    until the server restarts."""
    import threading

    import lelab.teleoperate as teleop

    stale = threading.Event()
    stale.set()
    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(teleop, "teleoperation_thread", None)
    monkeypatch.setattr(teleop, "_release_now", stale)
    monkeypatch.setattr(teleop, "releasing", True)
    monkeypatch.setattr(teleop, "setup_calibration_files", lambda leader, follower: ("leader", "follower"))

    class _Bus:
        def connect(self) -> None:
            raise RuntimeError("port busy")

    class _Device:
        def __init__(self, config) -> None:
            self.bus = _Bus()
            self.cameras: dict = {}

        def disconnect(self) -> None:
            pass

    monkeypatch.setattr(teleop, "SO101Follower", _Device)
    monkeypatch.setattr(teleop, "SO101Leader", _Device)

    result = teleop.handle_start_teleoperation(
        teleop.TeleoperateRequest(
            leader_port="COM_LEADER",
            follower_port="COM_FOLLOWER",
            leader_config="leader",
            follower_config="follower",
        )
    )

    # The connect fails, but the per-session reset already ran under the lock.
    assert result["success"] is False
    assert not stale.is_set()
    assert teleop.releasing is False


# ---------------------------------------------------------------------------
# Follower power telemetry (Present_Current, ~1 Hz)
# ---------------------------------------------------------------------------


class _CurrentBus:
    """Bus double serving Present_Current sync_reads."""

    def __init__(self, readings: list[dict], port: str = "COM_FOLLOWER") -> None:
        self._readings = list(readings)
        self.port = port

    def sync_read(self, reg: str, normalize: bool = True) -> dict:
        assert reg == "Present_Current" and normalize is False
        if not self._readings:
            raise ConnectionError("bus gone")
        return self._readings.pop(0)


def test_power_telemetry_tracks_peak_and_mean() -> None:
    """Peaks/means in mA (6.5 mA per register LSB) give the objective A/B for
    the motor-power cap; the summary names the Torque_Limit that was active."""
    import lelab.teleoperate as teleop

    telemetry = teleop.PowerTelemetry()
    bus = _CurrentBus([{"shoulder_pan": 100, "gripper": 20}, {"shoulder_pan": 40, "gripper": 60}])
    telemetry.sample(bus)
    telemetry.sample(bus)

    assert telemetry.peak_ma["shoulder_pan"] == 100 * 6.5
    assert telemetry.latest_ma["shoulder_pan"] == 40 * 6.5
    summary = telemetry.summary(30)
    assert summary is not None
    assert summary.startswith("power telemetry:")
    assert f"shoulder_pan peak {100 * 6.5:.0f}mA / mean {70 * 6.5:.0f}mA" in summary
    assert "motor power 30%, Torque_Limit 300" in summary


def test_power_telemetry_prefixes_bimanual_and_survives_bus_errors() -> None:
    import lelab.teleoperate as teleop

    telemetry = teleop.PowerTelemetry()
    telemetry.sample(_CurrentBus([{"gripper": 10}]), prefix="left_")
    telemetry.sample(_CurrentBus([]))  # dead bus: sample must not raise

    assert set(telemetry.peak_ma) == {"left_gripper"}


def test_power_telemetry_summary_none_without_samples() -> None:
    import lelab.teleoperate as teleop

    assert teleop.PowerTelemetry().summary(100) is None
