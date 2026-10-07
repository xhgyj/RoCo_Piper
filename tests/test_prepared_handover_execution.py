"""Offline frozen path, shared barrier and stamped tracking regressions."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
from test_handover_joint_planning import planning_request

from rocobrick.adapters.moveit.prepared_execution import perform_prepared_motion
from rocobrick.adapters.moveit.synchronized_feedback import SynchronizedFeedbackMonitor
from rocobrick.coordinated_execution import CoordinatedExecutionCoordinator
from rocobrick.coordinated_moveit import (
    CoordinatedOperation,
    CoordinatedRobotTrajectory,
    CoordinatedTrajectoryPoint,
)
from rocobrick.execution.prepared_motion import (
    PreparedMotionRequest,
    SynchronizationPolicy,
)


def prepared_request():
    """Build an explicitly complete synthetic scene and two-arm path.

    Returns:
        A typed path bound to synthetic full measured state.
    """
    source = planning_request()
    snapshot = copy.deepcopy(source.scene_snapshot)
    snapshot["tcp_provenance"] = {
        "runtime_model_sha256": "a" * 64,
        "source_sha256": "b" * 64,
    }
    snapshot["environment_colliders"] = {
        "floor": {"minimum_m": [-1, -1, -0.1], "maximum_m": [1, 1, 0]}
    }
    joints = dict(source.start_state)
    for robot in source.session_robots:
        joints[f"{robot}_joint7"] = 0.008
        joints[f"{robot}_joint8"] = 0.008
    operation = CoordinatedOperation(
        "prepared-handover-1",
        snapshot["scene_digest"],
        source.session_robots,
        tuple(
            CoordinatedRobotTrajectory(
                robot,
                tuple(f"{robot}_joint{i}" for i in range(1, 7)),
                (
                    CoordinatedTrajectoryPoint(0, (0.0,) * 6),
                    CoordinatedTrajectoryPoint(1, (0.05,) * 6),
                ),
            )
            for robot in source.session_robots
        ),
        0.0,
        0.02,
    )
    return PreparedMotionRequest(
        operation,
        source.session_robots,
        snapshot,
        1,
        joints,
        SynchronizationPolicy(0.02, 0.002, 0.2),
        10.0,
    )


def live_scene(request):
    """Advance query time/counters without moving physical state.

    Returns:
        A semantically unchanged but newly observed scene.
    """
    scene = copy.deepcopy(request.scene_snapshot)
    scene.update(sequence=20, sim_time_s=2.0, scene_digest="e" * 64)
    return scene


def test_prepared_round_trip_and_advancing_query_clock():
    """Static live geometry may advance its query identity without invalidation."""
    request = prepared_request()
    parsed = PreparedMotionRequest.parse(request.serialize())
    assert parsed.serialize() == request.serialize()
    parsed.validate_live(live_scene(request), 1, dict(request.start_state), 2.01)


@pytest.mark.parametrize(
    "change",
    ["joint", "finger", "pose", "environment", "model", "version", "stale", "coverage"],
)
def test_prepared_live_drift_fails_closed(change):
    """No joint, object, environment, model or missing-state drift is repaired."""
    request = prepared_request()
    scene, joints, version, observed = (
        live_scene(request),
        dict(request.start_state),
        1,
        2.01,
    )
    if change == "joint":
        joints["piper_1_joint1"] += 0.1
    elif change == "finger":
        joints["piper_0_joint7"] += 0.1
    elif change == "pose":
        scene["bricks"]["brick-1"]["pose"]["position_xyz"][0] += 0.1
    elif change == "environment":
        scene["environment_colliders"] = {}
    elif change == "model":
        scene["tcp_provenance"] = {}
    elif change == "version":
        version = 2
    elif change == "stale":
        observed = 3.0
    else:
        joints.pop("piper_0_joint7")
    with pytest.raises(ValueError):
        request.validate_live(scene, version, joints, observed)


@pytest.mark.parametrize(
    "change", ["snapshot", "joint", "wire_hash", "unknown_key", "unknown_policy"]
)
def test_prepared_mutation_and_wire_tampering_rejected(change):
    """A frozen request cannot silently rebind mutable nested inputs."""
    request = prepared_request()
    if change == "snapshot":
        request.scene_snapshot["sim_time_s"] += 1
        with pytest.raises(ValueError, match="mutated"):
            request.serialize()
        return
    if change == "joint":
        request.start_state["piper_1_joint1"] = 0.01
        with pytest.raises(ValueError, match="mutated"):
            request.serialize()
        return
    raw = request.serialize()
    if change == "wire_hash":
        raw["binding_sha256"] = "f" * 64
    elif change == "unknown_key":
        raw["repair_plan"] = True
    else:
        raw["policy"]["relax_collision_check"] = True
    with pytest.raises(ValueError):
        PreparedMotionRequest.parse(raw)


def timestamp(seconds):
    """Construct a ROS timestamp without ROS imports.

    Returns:
        Integer seconds/nanoseconds representing the requested time.
    """
    nanos = round(seconds * 1e9)
    return SimpleNamespace(sec=nanos // 1_000_000_000, nanosec=nanos % 1_000_000_000)


def feedback_sample(path, elapsed, *, start=10.0, error=0.0, count=1):
    """Pair a controller phase with independently stamped physics state.

    Returns:
        A controller message and its fresh physics sample.
    """
    positions = (0.05 * min(elapsed, 1.0),) * 6
    message = SimpleNamespace(
        header=SimpleNamespace(stamp=timestamp(start + elapsed)),
        joint_names=path.joint_names,
        desired=SimpleNamespace(
            time_from_start=timestamp(elapsed), positions=positions
        ),
        actual=SimpleNamespace(positions=positions),
    )
    physical = {
        "sample_count": count,
        "sim_time_s": start + elapsed,
        "age_s": 0.01,
        "joints": {f"joint{i}": positions[i - 1] + error for i in range(1, 7)},
    }
    return message, physical


def test_sync_requires_physics_coverage_not_goal_acceptance():
    """Phase and tracking feedback cover the entire path, including its end."""
    request = prepared_request()
    monitor = SynchronizedFeedbackMonitor(
        request.operation.trajectories[0], request.policy, 10.0
    )
    for count, elapsed in enumerate((0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0), 1):
        monitor.consume(*feedback_sample(monitor.trajectory, elapsed, count=count))
    evidence = monitor.evidence()
    assert evidence["synchronization_feedback_confirmed"] is True
    assert evidence["feedback_samples"] == 7
    assert evidence["controller_start_sim_time_s"] == pytest.approx(10.0)
    monitor.consume(*feedback_sample(monitor.trajectory, 1.1, count=8))
    assert monitor.failed is False  # post-end settling is not another tracking sample
    assert monitor.evidence()["feedback_samples"] == 7


@pytest.mark.parametrize(
    "failure",
    [
        "late_phase",
        "tracking",
        "missing_stamp",
        "gap",
        "wrong_robot",
        "late_first",
        "duplicate",
    ],
)
def test_sync_missing_or_invalid_measured_feedback_fails(failure):
    """Controller metadata alone never passes strict synchronization admission."""
    request = prepared_request()
    monitor = SynchronizedFeedbackMonitor(
        request.operation.trajectories[0], request.policy, 10.0
    )
    message, physical = feedback_sample(monitor.trajectory, 0.1)
    if failure == "late_phase":
        message.header.stamp = timestamp(10.3)
    elif failure == "tracking":
        physical["joints"]["joint1"] += 0.1
    elif failure == "missing_stamp":
        physical.pop("sim_time_s")
    elif failure == "gap":
        monitor.consume(message, physical)
        message, physical = feedback_sample(monitor.trajectory, 0.4, count=2)
    elif failure == "wrong_robot":
        message.joint_names = tuple(f"piper_1_joint{i}" for i in range(1, 7))
    elif failure == "late_first":
        message, physical = feedback_sample(monitor.trajectory, 0.5)
    else:
        for _ in range(4):
            monitor.consume(message, physical)
    monitor.consume(message, physical)
    assert monitor.evidence()["synchronization_feedback_confirmed"] is False


def test_prepared_drift_stops_before_any_controller_access():
    """The execution boundary refuses drift before creating action participants."""
    request = prepared_request()
    scene = live_scene(request)
    scene["bricks"]["brick-1"]["pose"]["position_xyz"][0] += 0.1
    session = SimpleNamespace(
        robot_ids=request.robots,
        observe_all=lambda: scene,
        unified_planning_scene_version=lambda: 1,
        unified_start_state=lambda: request.start_state,
        latest_shared_sim_time_s=lambda: 2.01,
        runtime_for=lambda robot: pytest.fail("must not access controller"),
    )
    result = perform_prepared_motion(session, request)
    assert result.passed is False
    assert result.lifecycle["physical_execution"] is False
    assert result.lifecycle["terminal"] is True


def test_second_live_admission_can_block_all_prepared_goals():
    """Scene drift during prepare prevents both participants from starting."""
    from test_coordinated_execution import Participant, operation

    participants = {robot: Participant() for robot in ("piper_0", "piper_1")}

    def drifted():
        raise ValueError("scene changed during prepare")

    result = CoordinatedExecutionCoordinator(participants).execute(
        operation(), before_start=drifted
    )
    assert result.started is False
    assert all(p.calls == ["prepare", "cancel"] for p in participants.values())


def test_strict_coordinator_rejects_acceptance_only_success():
    """Concurrent action success does not substitute for measured phase proof."""
    from test_coordinated_execution import Participant, operation

    result = CoordinatedExecutionCoordinator(
        {robot: Participant() for robot in ("piper_0", "piper_1")}
    ).execute(
        operation(),
        require_synchronization_feedback=True,
    )
    assert result.started is False
    assert set(result.cancel_evidence) == {"piper_0", "piper_1"}


def strict_fake_runtime(robot, *, skew=0.0):
    """Script controller feedback, actual header and separately stamped joints.

    Returns:
        A runtime test double that never opens ROS or Isaac.
    """
    from test_coordinated_arm_participant import FakeRuntime, _done_future

    runtime = FakeRuntime()
    runtime.arm_joint_names = tuple(f"{robot}_joint{i}" for i in range(1, 7))
    runtime.clock_values = lambda: (2_010_000_000,)
    runtime.stamped = {
        "age_s": 0.01,
        "sim_time_s": 2.01,
        "sample_count": 5,
        "joints": runtime._state,
    }
    runtime.latest_stamped_isaac_state = lambda: runtime.stamped
    runtime.sent_goals = []

    def send(goal, *, feedback_callback):
        runtime.sent_goals.append(goal)
        path = CoordinatedRobotTrajectory(
            robot,
            tuple(goal.trajectory.joint_names),
            (
                CoordinatedTrajectoryPoint(0.0, (0.0,) * 6),
                CoordinatedTrajectoryPoint(1.0, (0.05,) * 6),
            ),
        )
        barrier = (
            goal.trajectory.header.stamp.sec
            + goal.trajectory.header.stamp.nanosec * 1e-9
        )
        for count, elapsed in enumerate((0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0), 6):
            message, physical = feedback_sample(
                path, elapsed, start=barrier, count=count
            )
            message.header.stamp = timestamp(barrier + elapsed + skew)
            runtime.stamped = physical
            runtime._state = dict(physical["joints"])
            runtime._isaac_state_count = count
            feedback_callback(message)
        return _done_future(
            SimpleNamespace(
                accepted=True,
                get_result_async=lambda: _done_future(SimpleNamespace(status=4)),
            )
        )

    runtime._arm_client.send_goal_async = send

    def fresh_physics():
        runtime._isaac_state_count += 1
        runtime.stamped = {
            **runtime.stamped,
            "sample_count": runtime._isaac_state_count,
            "sim_time_s": runtime.stamped["sim_time_s"] + 0.01,
        }

    runtime._spin_fn = fresh_physics
    return runtime


@pytest.mark.parametrize("skew", [0.0, 0.03])
def test_prepared_executes_exact_path_and_measures_shared_barrier(skew):
    """The production admission/lifecycle code needs both real feedback bindings."""
    request = prepared_request()
    runtimes = {
        robot: strict_fake_runtime(robot, skew=skew if robot == "piper_1" else 0.0)
        for robot in request.robots
    }
    scene = live_scene(request)
    session = SimpleNamespace(
        robot_ids=request.robots,
        observe_all=lambda: scene,
        unified_planning_scene_version=lambda: 1,
        unified_start_state=lambda: request.start_state,
        latest_shared_sim_time_s=lambda: 2.01,
        runtime_for=runtimes.__getitem__,
    )
    result = perform_prepared_motion(session, request)
    assert result.passed is (skew == 0.0)
    for robot, runtime in runtimes.items():
        goal = runtime.sent_goals[0]
        assert (
            goal.trajectory.header.stamp.sec,
            goal.trajectory.header.stamp.nanosec,
        ) == (3, 10_000_000)
        assert tuple(tuple(p.positions) for p in goal.trajectory.points) == tuple(
            p.positions
            for p in next(
                t for t in request.operation.trajectories if t.robot_id == robot
            ).points
        )
    assert result.lifecycle["replanned"] is False
    if skew == 0.0:
        assert result.lifecycle["prepared_binding_sha256"] == request.binding_sha256
    else:
        assert all(runtime._cancel_calls >= 1 for runtime in runtimes.values())


def test_prepared_cancel_before_admission_sends_no_goal():
    """Cancellation received before preflight never dispatches either arm."""
    request = prepared_request()
    result = perform_prepared_motion(
        SimpleNamespace(robot_ids=request.robots),
        request,
        cancel_requested=lambda: True,
    )
    assert result.passed is False
    assert result.lifecycle["physical_execution"] is False


def test_live_stamped_reader_does_not_borrow_previous_timestamp():
    """Production joint reader only pairs each physics sample with its own stamp."""
    from test_dual_robot_ros_adapter import FakeJointStateMsg, device_state, make_world

    runtime = make_world().runtime("piper_1")
    stamped = FakeJointStateMsg(device_state("piper_1"), [0.0] * 8)
    stamped.header = SimpleNamespace(stamp=timestamp(3.25))
    runtime._on_isaac_state(stamped)
    sample = runtime.latest_stamped_isaac_state()
    assert sample["sim_time_s"] == 3.25
    assert sample["joints"] == runtime.latest_isaac_state()
    runtime._on_isaac_state(FakeJointStateMsg(device_state("piper_1"), [0.0] * 8))
    assert runtime.latest_stamped_isaac_state() is None
    assert runtime.latest_isaac_state() is not None  # legacy PICK channel unchanged


@pytest.mark.parametrize(
    "stamp",
    [
        SimpleNamespace(sec=True, nanosec=0),
        SimpleNamespace(sec=3, nanosec=1_000_000_000),
    ],
)
def test_live_stamped_reader_rejects_invalid_clock_fields(stamp):
    """Boolean and unnormalized clock fields cannot become tracking evidence."""
    from test_dual_robot_ros_adapter import FakeJointStateMsg, device_state, make_world

    runtime = make_world().runtime("piper_0")
    message = FakeJointStateMsg(device_state("piper_0"), [0.0] * 8)
    message.header = SimpleNamespace(stamp=stamp)
    runtime._on_isaac_state(message)
    assert runtime.latest_stamped_isaac_state() is None


def test_second_arm_dispatch_exception_cancels_waiting_first_immediately():
    """Participant order cannot delay cancellation or certify an uncertain send."""
    from threading import Event

    from test_coordinated_execution import Participant, operation

    entered, released = Event(), Event()

    class WaitingArm(Participant):
        """A first arm which can finish only when cancellation reaches it."""

        def start(self, *, start_at_sim_time_s):
            """Wait for peer-failure cancellation, bounded by the test deadline.

            Returns:
                Actual settled state after the cancel request arrives.
            """
            entered.set()
            assert released.wait(1.0), (
                "coordinator waited on first arm instead of canceling it"
            )
            return {"started": True, "terminal": True, "safe": True}

        def cancel(self):
            """Unblock the waiting action channel.

            Returns:
                Settled cancellation evidence for this owned arm.
            """
            self.calls.append("cancel")
            released.set()
            return {"attempted": True, "terminal": True, "safe": True}

    class UnknownDispatch(Participant):
        """A possibly sent second-arm command with no acceptance handle."""

        def start(self, *, start_at_sim_time_s):
            """Raise after the other robot has entered its action lane."""
            assert entered.wait(1.0)
            raise RuntimeError("dispatch outcome unknown")

        def cancel(self):
            """Preserve dispatch uncertainty instead of certifying an idle cancel.

            Returns:
                Non-terminal cancellation that retains resource ownership.
            """
            return {"attempted": True, "terminal": False, "safe": False}

    first = WaitingArm()
    result = CoordinatedExecutionCoordinator(
        {"piper_0": first, "piper_1": UnknownDispatch()}
    ).execute(operation())
    assert released.is_set()
    assert first.calls.count("cancel") >= 2  # early request, then final late-send proof
    assert result.terminal is False
    assert result.safe_to_release is False
