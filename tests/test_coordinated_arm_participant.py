"""Offline 4A tests for the coordinated arm participant lifecycle."""

from types import SimpleNamespace

from rocobrick.adapters.moveit.coordinated_arm import CoordinatedArmParticipant
from rocobrick.coordinated_moveit import CoordinatedRobotTrajectory

ARM = tuple(f"piper_0_joint{index}" for index in range(1, 7))


class Point:
    """Minimal trajectory point record for the fake split trajectory."""

    def __init__(self, time_from_start_s, positions):
        self.time_from_start_s = time_from_start_s
        self.positions = positions


def trajectory(points=((0.0, (0.0,) * 6), (1.0, (0.05,) * 6))):
    """Build one valid split trajectory for the fake participant."""
    return CoordinatedRobotTrajectory(
        "piper_0",
        ARM,
        tuple(Point(t, p) for t, p in points),
    )


class FakeFactories:
    """Minimal controller-goal factory set."""

    def follow_joint_trajectory_goal_factory(self):
        goal = SimpleNamespace(
            trajectory=SimpleNamespace(joint_names=[], points=[]),
            goal_tolerance=[],
            path_tolerance=[],
        )
        return goal

    def joint_trajectory_point_factory(self):
        return SimpleNamespace(positions=[], time_from_start=None)

    def joint_tolerance_factory(self):
        return SimpleNamespace(name="", position=0.0, velocity=0.0)

    def duration_factory(self):
        return SimpleNamespace(sec=0, nanosec=0)

    def goal_status_type(self):
        return SimpleNamespace(STATUS_SUCCEEDED=4, STATUS_CANCELED=5)


class FakeClient:
    """FollowJointTrajectory client with a scripted terminal result."""

    def __init__(self, accepted=True, status=4, runtime=None):
        self.accepted = accepted
        self.status = status
        self.sent = 0
        self.runtime = runtime

    def send_goal_async(self, goal):
        self.sent += 1
        if self.runtime is not None:
            self.runtime._command_count += 1
        handle = SimpleNamespace(
            accepted=self.accepted,
            get_result_async=lambda: _done_future(
                SimpleNamespace(status=self.status, result=SimpleNamespace())
            ),
        )
        return _done_future(handle)


def _done_future(value):
    future = SimpleNamespace(done=lambda: True, result=lambda: value)
    return future


class FakeRuntime:
    """Just enough RosMoveItRuntime surface for the participant."""

    def __init__(self, state=None, client=None, ready=True):
        self.arm_joint_names = ARM
        self._state = state or {f"joint{i}": 0.0 for i in range(1, 7)}
        self._arm_client = client or FakeClient(runtime=self)
        if client is not None:
            client.runtime = self
        self._ready = ready
        self._command_count = 0
        self._isaac_state_count = 5
        self._cancel_calls = 0
        self._now = 0.0

    def latest_isaac_state(self):
        return dict(self._state)

    def _resolve_factories(self):
        return FakeFactories()

    def _action_ready(self, client):
        return self._ready

    def _clock_fn(self):
        self._now += 0.01
        return self._now

    def _wait_future(self, future, deadline):
        return future.done()

    def _duration_message(self, seconds, factories):
        return factories.duration_factory()

    def _spin_fn(self):
        pass

    def _sleep_fn(self, _):
        pass

    def cancel_active(self):
        self._cancel_calls += 1
        return {"attempted": True, "terminal": True, "safe": True}


def test_prepare_admits_without_motion_and_rejects_drift():
    participant = CoordinatedArmParticipant(FakeRuntime())
    evidence = participant.prepare(trajectory(), start_at_sim_time_s=10.0)
    assert evidence["prepared"] is True
    assert evidence["motion_sent"] is False

    drifted = FakeRuntime(state={f"joint{i}": 0.5 for i in range(1, 7)})
    assert CoordinatedArmParticipant(drifted).prepare(
        trajectory(), start_at_sim_time_s=10.0
    )["prepared"] is False

    foreign = CoordinatedRobotTrajectory(
        "piper_0",
        tuple(f"piper_1_joint{index}" for index in range(1, 7)),
        trajectory().points,
    )
    assert CoordinatedArmParticipant(FakeRuntime()).prepare(
        foreign, start_at_sim_time_s=10.0
    )["prepared"] is False


def test_start_proves_arrival_from_measured_final_state():
    runtime = FakeRuntime(state={f"joint{i}": 0.0 for i in range(1, 7)})
    participant = CoordinatedArmParticipant(runtime)
    assert participant.prepare(trajectory(), start_at_sim_time_s=10.0)[
        "prepared"
    ] is True
    runtime._state = {f"joint{i}": 0.05 for i in range(1, 7)}
    evidence = participant.start(start_at_sim_time_s=10.0)
    assert evidence["started"] is True
    assert evidence["terminal"] is True
    assert evidence["safe"] is True
    assert evidence["arrival_confirmed"] is True
    assert evidence["command_messages"] >= 1


def test_start_rejection_is_terminal_without_motion_claim():
    runtime = FakeRuntime(client=FakeClient(accepted=False))
    participant = CoordinatedArmParticipant(runtime)
    participant.prepare(trajectory(), start_at_sim_time_s=10.0)
    evidence = participant.start(start_at_sim_time_s=10.0)
    assert evidence["started"] is False
    assert evidence["terminal"] is True
    assert evidence["safe"] is True


def test_start_without_arrival_is_not_safe():
    runtime = FakeRuntime(state={f"joint{i}": 0.0 for i in range(1, 7)})
    participant = CoordinatedArmParticipant(runtime)
    participant.prepare(trajectory(), start_at_sim_time_s=10.0)
    runtime._state = {f"joint{i}": 0.5 for i in range(1, 7)}
    evidence = participant.start(start_at_sim_time_s=10.0)
    assert evidence["started"] is True
    assert evidence["arrival_confirmed"] is False
    assert evidence["safe"] is False


def test_cancel_settles_through_owned_runtime_lifecycle():
    runtime = FakeRuntime()
    participant = CoordinatedArmParticipant(runtime)
    participant.prepare(trajectory(), start_at_sim_time_s=10.0)
    evidence = participant.cancel()
    assert evidence["attempted"] is True
    assert runtime._cancel_calls == 1
