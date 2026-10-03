"""Short contact paths retain every admitted point and controller evidence."""

from types import SimpleNamespace

import pytest
from test_contact_guard import guard, scene
from test_place_down_handler import MODEL_SHA, SOURCE_SHA
from test_ros_moveit_runtime_contract import (
    BASELINE,
    FAKE_GOAL_STATUS,
    FakeGoalHandle,
    make_world,
)

from rocobrick.moveit_contract import TcpPoseTarget
from rocobrick.ros_moveit_runtime import ARM_JOINT_NAMES


def contact_target(z: float = 0.028) -> TcpPoseTarget:
    """Return one admitted TCP target 5 mm below the measured start pose."""
    return TcpPoseTarget(
        frame_id="piper_0_base_link",
        position=(0.32, 0.32, z),
        orientation_xyzw=(1.0, 0.0, 0.0, 0.0),
        position_tolerance_m=0.0003,
        orientation_tolerance_rad=0.015,
        runtime_model_sha256=MODEL_SHA,
        source_sha256=SOURCE_SHA,
    )


def measured_start_tcp(z: float):
    """Return one measured (position, quaternion) start-pose pair."""
    return ((0.32, 0.32, z), (1.0, 0.0, 0.0, 0.0))


def test_contact_controller_retimes_all_points_and_sets_strict_tolerances():
    """The controller receives the full checked path, not an endpoint shortcut."""
    world = make_world()
    world.runtime.latest_isaac_state = lambda: BASELINE
    world.runtime.compute_tcp_fk = lambda state, timeout_s=0.5: measured_start_tcp(
        0.033
    )
    sequence = [20]

    def advancing_scene(timeout_s):
        world.runtime._spin_fn()
        value = scene()
        sequence[0] += 1
        value["sequence"] = sequence[0]
        value["robots"]["piper_0"]["sequence"] = sequence[0]
        return value

    world.runtime.query_scene_state = advancing_scene
    handle = FakeGoalHandle()
    points = [
        SimpleNamespace(
            positions=[
                BASELINE[name] + (0.0001 if index else 0.0) for name in ARM_JOINT_NAMES
            ],
            velocities=[0.001] * 6,
            accelerations=[0.002] * 6,
            time_from_start=SimpleNamespace(sec=0, nanosec=index * 20_000_000),
        )
        for index in range(3)
    ]
    trajectory = SimpleNamespace(joint_names=list(ARM_JOINT_NAMES), points=points)
    world.events.append(lambda: world.arm.pending.set(handle))
    world.events.append(
        lambda: handle.result_future.set(
            SimpleNamespace(
                status=FAKE_GOAL_STATUS.STATUS_ABORTED,
                result=SimpleNamespace(error_code=-5),
            )
        )
    )
    result = world.runtime._execute_contact_trajectory(
        trajectory,
        contact_target(),
        guard(),
        world.clock.now + 10,
        3,
    )
    actual = world.arm.goals[0]
    assert len(actual.trajectory.points) == 3
    assert actual.trajectory.points[-1].positions == points[-1].positions
    # 5 mm measured Cartesian distance at the 1 mm/s contact cap: the 0.04 s
    # plan is stretched to 5.0 s, never shortened.
    assert result.details["contact_cartesian_distance_m"] == pytest.approx(0.005)
    assert actual.trajectory.points[-1].time_from_start.sec == 5
    assert actual.trajectory.points[-1].time_from_start.nanosec == 0
    assert actual.trajectory.points[-1].velocities == pytest.approx([0.000008] * 6)
    # Contact tolerances are deliberately generous: contact loads and
    # uncompensated droop must not abort the supervised descent.
    assert all(tolerance.position == 0.03 for tolerance in actual.goal_tolerance)
    assert all(tolerance.position == 0.05 for tolerance in actual.path_tolerance)
    assert actual.goal_time_tolerance.sec == 5
    assert result.details["contact_stop_reason"] == "contact_tracking_stopped"
    assert result.terminal is True


def test_contact_controller_requires_a_measured_start_pose():
    """Without live FK the contact trajectory never executes."""
    world = make_world()
    world.runtime.latest_isaac_state = lambda: BASELINE
    world.runtime.compute_tcp_fk = lambda state, timeout_s=0.5: None
    trajectory = SimpleNamespace(
        joint_names=list(ARM_JOINT_NAMES),
        points=[
            SimpleNamespace(
                positions=[BASELINE[name] for name in ARM_JOINT_NAMES],
                velocities=[0.0] * 6,
                accelerations=[0.0] * 6,
                time_from_start=SimpleNamespace(sec=0, nanosec=0),
            ),
            SimpleNamespace(
                positions=[BASELINE[name] + 0.0001 for name in ARM_JOINT_NAMES],
                velocities=[0.001] * 6,
                accelerations=[0.002] * 6,
                time_from_start=SimpleNamespace(sec=0, nanosec=20_000_000),
            ),
        ],
    )
    with pytest.raises(RuntimeError, match="measured TCP start pose"):
        world.runtime._execute_contact_trajectory(
            trajectory,
            contact_target(),
            guard(),
            world.clock.now + 10,
            2,
        )


def test_controller_abort_is_never_endpoint_success():
    """A known aborted tracking step enables recovery but not completion."""
    world = make_world()
    handle = FakeGoalHandle()
    handle.result_future.set(
        SimpleNamespace(
            status=FAKE_GOAL_STATUS.STATUS_ABORTED,
            result=SimpleNamespace(error_code=-5),
        )
    )
    world.runtime.query_scene_state = lambda timeout_s: scene()
    result = world.runtime._await_guarded_action(
        handle.result_future,
        guard(),
        world.clock.now + 2,
        {"execution_transport": "arm_follow"},
        0,
    )
    assert result.details["action_status"] == "ABORTED"
    assert result.details["contact_stop_reason"] == "contact_tracking_stopped"
    assert result.terminal is True and result.safe_to_release is True
