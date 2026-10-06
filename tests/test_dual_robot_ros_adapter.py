"""A5 dual-robot ROS adapter routing tests (offline injected transport).

Pins the per-robot isolation of the ROS/MoveIt adapter on injected fakes only
(no rclpy, no MoveIt, no Isaac, no Docker):

* one :class:`RosMoveItRobotSession` hosts one isolated
  :class:`RosMoveItRuntime` per plan robot on ONE node, one ROS domain and the
  shared ``/clock``, consuming the A3 ``RobotRuntimeProfile`` endpoints
  verbatim (``/<robot>/isaac_joint_states``, ``/<robot>/isaac_joint_command``,
  ``/<robot>/joint_states``, ``/<robot>/move_action``, the robot controller /
  FK / plan services and arm+gripper follow actions) with NO canonical
  piper_0 fallback for piper_1;
* cross-robot data cannot pass: state messages, controller answers, action
  readiness, FK requests and scene observations are all keyed by the robot's
  own device names/endpoints, and a piper_1 runtime rejects piper_0 evidence;
* capability readiness covers joints, controllers, MoveGroup, FK, the gripper
  follow action and the robot-scoped scene query;
* one not-ready robot fails the whole session and dispatches NOTHING;
* multi-robot session requests route each operation to its own robot (and the
  scene query is sliced to that robot's block of the global A4 document).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from rocobrick.isaac_backend import (
    RuntimeOperationResult,
    joint_operation,
    readiness_operation,
    scene_query_operation,
)
from rocobrick.moveit_contract import (
    ARM_JOINT_NAMES,
    BASE_FRAME,
    REQUIRED_CONTROLLERS,
    TCP_LINK,
    TCP_STATUS_CALIBRATED_PROJECT,
    TIP_LINK,
    Link6PoseTarget,
    TcpPoseTarget,
)
from rocobrick.moveit_requests import build_joint_target
from rocobrick.robot_runtime_profile import RobotRuntimeProfile
from rocobrick.ros_moveit_executor import RosMoveItExecutor
from rocobrick.ros_moveit_runtime import (
    CAPABILITY_COMPONENTS,
    SCENE_STATE_SCHEMA,
    SERVICE_SCENE_STATE,
    RosMoveItFactories,
    RosMoveItRobotSession,
    RosMoveItRuntime,
    RuntimeActions,
    RuntimeServices,
    RuntimeTopics,
    robot_scene_view,
    runtime_actions_for_profile,
    runtime_services_for_profile,
    runtime_topics_for_profile,
)
from rocobrick.ros_session_entry import (
    SESSION_REQUEST_SCHEMA,
    SESSION_RESULT_SCHEMA,
    SESSION_SUMMARY_SCHEMA,
    RobotOperationRequest,
    RosSessionRequest,
    RosSessionResult,
    parse_robots,
    perform_session,
)
from rocobrick.tcp_contract import EXPECTED_RUNTIME_MODEL_SHA256

ROBOT_0 = "piper_0"
ROBOT_1 = "piper_1"
BASELINE = {
    "joint1": 0.01,
    "joint2": 1.0,
    "joint3": -0.5,
    "joint4": 0.02,
    "joint5": 0.0,
    "joint6": -0.1,
    "gripper_joint1": 0.0,
    "gripper_joint2": 0.0,
}
FAKE_GOAL_STATUS = SimpleNamespace(
    STATUS_SUCCEEDED=4, STATUS_CANCELED=5, STATUS_ABORTED=6
)


def device_joint_names(robot_id: str) -> tuple[str, ...]:
    return tuple(f"{robot_id}_{name}" for name in BASELINE)


def device_state(robot_id: str, positions: dict | None = None) -> dict[str, float]:
    values = BASELINE if positions is None else positions
    return {
        f"{robot_id}_{name}": float(value) for name, value in values.items()
    }


# --------------------------------------------------------------- fake transport


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, delta: float) -> None:
        self.now += float(delta)


class FakeClockMsg:
    def __init__(self, nanos: int) -> None:
        self.clock = SimpleNamespace(
            sec=int(nanos) // 1_000_000_000, nanosec=int(nanos) % 1_000_000_000
        )


class FakeJointStateMsg:
    def __init__(self, positions: dict, velocity: list | None = None) -> None:
        self.name = list(positions)
        self.position = [float(value) for value in positions.values()]
        self.velocity = list(velocity or [])


class FakeFuture:
    def __init__(self, value: object = None) -> None:
        self._value = value

    def done(self) -> bool:
        return True

    def result(self) -> object:
        return self._value


class FakeGoalHandle:
    def __init__(self, accepted: bool) -> None:
        self.accepted = bool(accepted)


class FakeActionClient:
    def __init__(self, node, action_type, name: str) -> None:
        self.node = node
        self.action_type = action_type
        self.name = name
        self.server_ready = True
        self.accept = False
        self.goals: list = []
        self.destroyed = False
        node.action_clients.append(self)

    def wait_for_server(self, timeout_sec=None) -> bool:
        return bool(self.server_ready)

    def send_goal_async(self, goal):
        self.goals.append(goal)
        return FakeFuture(FakeGoalHandle(self.accept))

    def destroy(self) -> None:
        self.destroyed = True


class FakeServiceClient:
    def __init__(self, srv_type, name: str) -> None:
        self.srv_type = srv_type
        self.name = name
        self.ready = True
        self.response: object = None
        self.requests: list = []

    def service_is_ready(self) -> bool:
        return bool(self.ready)

    def call_async(self, request):
        self.requests.append(request)
        return FakeFuture(self.response)


class FakeSubscription:
    def __init__(self, msg_type, topic, callback, qos) -> None:
        self.msg_type = msg_type
        self.topic = topic
        self.callback = callback
        self.qos = qos


class FakeNode:
    def __init__(self) -> None:
        self.subscriptions: list[FakeSubscription] = []
        self.service_clients: list[FakeServiceClient] = []
        self.action_clients: list[FakeActionClient] = []
        self.destroyed: list[tuple[str, str]] = []

    def create_subscription(self, msg_type, topic, callback, qos):
        subscription = FakeSubscription(msg_type, topic, callback, qos)
        self.subscriptions.append(subscription)
        return subscription

    def create_client(self, srv_type, name):
        client = FakeServiceClient(srv_type, name)
        self.service_clients.append(client)
        return client

    def destroy_subscription(self, subscription) -> None:
        self.destroyed.append(("subscription", subscription.topic))

    def destroy_client(self, client) -> None:
        self.destroyed.append(("client", client.name))

    def publish(self, topic: str, message) -> None:
        for subscription in self.subscriptions:
            if subscription.topic == topic:
                subscription.callback(message)

    def subscription_topics(self) -> list[str]:
        return [item.topic for item in self.subscriptions]

    def service_names(self) -> list[str]:
        return [item.name for item in self.service_clients]

    def action_names(self) -> list[str]:
        return [item.name for item in self.action_clients]

    def clients(self, name: str) -> list[FakeServiceClient]:
        return [item for item in self.service_clients if item.name == name]

    def actions(self, name: str) -> list[FakeActionClient]:
        return [item for item in self.action_clients if item.name == name]


# Fake MoveIt/control message classes mirroring the fields the runtime touches.
class FakeMotionPlanRequest:
    def __init__(self) -> None:
        self.group_name = ""
        self.allowed_planning_time = 0.0
        self.num_planning_attempts = 0
        self.max_velocity_scaling_factor = 0.0
        self.max_acceleration_scaling_factor = 0.0
        self.goal_constraints: list = []


class FakeConstraints:
    def __init__(self) -> None:
        self.joint_constraints: list = []
        self.position_constraints: list = []
        self.orientation_constraints: list = []


class FakeJointConstraint:
    def __init__(self) -> None:
        self.joint_name = ""
        self.position = 0.0
        self.tolerance_above = 0.0
        self.tolerance_below = 0.0
        self.weight = 0.0


class FakePositionConstraint:
    def __init__(self) -> None:
        self.header = SimpleNamespace(frame_id="")
        self.link_name = ""
        self.constraint_region = SimpleNamespace(primitives=[], primitive_poses=[])
        self.weight = 0.0


class FakeOrientationConstraint:
    def __init__(self) -> None:
        self.header = SimpleNamespace(frame_id="")
        self.link_name = ""
        self.orientation = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0)
        self.absolute_x_axis_tolerance = 0.0
        self.absolute_y_axis_tolerance = 0.0
        self.absolute_z_axis_tolerance = 0.0
        self.weight = 0.0


class FakeSolidPrimitive:
    SPHERE = 2
    BOX = 1

    def __init__(self) -> None:
        self.type = 0
        self.dimensions: list = []


class FakePose:
    def __init__(self) -> None:
        self.position = SimpleNamespace(x=0.0, y=0.0, z=0.0)
        self.orientation = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0)


class FakeMoveGroupGoal:
    def __init__(self) -> None:
        self.request = None
        self.planning_options = SimpleNamespace(plan_only=None, replan=None)


class FakeTriggerSrv:
    class Request:
        def __init__(self) -> None:
            self.placeholder = True


class FakeFollowJointTrajectoryGoal:
    def __init__(self) -> None:
        self.trajectory = SimpleNamespace(joint_names=[], points=[])
        self.goal_tolerance: list = []


class FakeJointTrajectoryPoint:
    def __init__(self) -> None:
        self.positions: list = []
        self.time_from_start = None


class FakeJointTolerance:
    def __init__(self) -> None:
        self.name = ""
        self.position = 0.0
        self.velocity = 0.0
        self.acceleration = 0.0


class FakeDuration:
    def __init__(self) -> None:
        self.sec = 0
        self.nanosec = 0


def fake_apply_scene_request() -> SimpleNamespace:
    return SimpleNamespace(
        scene=SimpleNamespace(
            is_diff=False,
            allowed_collision_matrix=None,
            world=SimpleNamespace(collision_objects=[]),
        )
    )


def fake_get_scene_request() -> SimpleNamespace:
    return SimpleNamespace(components=SimpleNamespace(components=0))


def fake_collision_object() -> SimpleNamespace:
    return SimpleNamespace(
        ADD=0,
        header=SimpleNamespace(frame_id=""),
        id="",
        operation=None,
        primitives=[],
        primitive_poses=[],
    )


class FakeFKResponse:
    def __init__(self, code: int = 1, pose=None) -> None:
        self.error_code = SimpleNamespace(val=code)
        self.pose_stamped = [] if pose is None else [SimpleNamespace(pose=pose)]


class FakeFKRequest:
    def __init__(self) -> None:
        self.header = SimpleNamespace(frame_id="")
        self.fk_link_names: list = []
        self.robot_state = SimpleNamespace(
            joint_state=SimpleNamespace(name=[], position=[])
        )

    @staticmethod
    def Request() -> "FakeFKRequest":
        return FakeFKRequest()


def identity_pose() -> SimpleNamespace:
    return SimpleNamespace(
        position=SimpleNamespace(x=0.1, y=0.0, z=0.2),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )


def fake_factories() -> RosMoveItFactories:
    return RosMoveItFactories(
        clock_type=lambda: FakeClockMsg,
        joint_state_type=lambda: FakeJointStateMsg,
        list_controllers_type=lambda: SimpleNamespace,
        list_controllers_request_factory=lambda: SimpleNamespace(),
        get_motion_plan_type=lambda: SimpleNamespace,
        get_position_fk_type=lambda: FakeFKRequest,
        move_group_type=lambda: SimpleNamespace,
        follow_joint_trajectory_type=lambda: SimpleNamespace,
        move_group_goal_factory=FakeMoveGroupGoal,
        motion_plan_request_factory=FakeMotionPlanRequest,
        constraints_factory=FakeConstraints,
        joint_constraint_factory=FakeJointConstraint,
        position_constraint_factory=FakePositionConstraint,
        orientation_constraint_factory=FakeOrientationConstraint,
        solid_primitive_factory=FakeSolidPrimitive,
        pose_factory=FakePose,
        action_client_factory=lambda node, action_type, name: FakeActionClient(
            node, action_type, name
        ),
        goal_status_type=lambda: FAKE_GOAL_STATUS,
        apply_planning_scene_type=lambda: SimpleNamespace,
        apply_planning_scene_request_factory=fake_apply_scene_request,
        collision_object_factory=fake_collision_object,
        allowed_collision_entry_factory=lambda: SimpleNamespace(enabled=[]),
        get_planning_scene_type=lambda: SimpleNamespace,
        get_planning_scene_request_factory=fake_get_scene_request,
        trigger_type=lambda: FakeTriggerSrv,
        trigger_request_factory=lambda: FakeTriggerSrv.Request(),
        follow_joint_trajectory_goal_factory=FakeFollowJointTrajectoryGoal,
        joint_trajectory_point_factory=FakeJointTrajectoryPoint,
        joint_tolerance_factory=FakeJointTolerance,
        duration_factory=FakeDuration,
    )


@dataclass
class World:
    node: FakeNode
    clock: FakeClock
    session: RosMoveItRobotSession
    factories: RosMoveItFactories = field(default_factory=fake_factories)
    clock_nanos: int = 0

    def tick_clock(self) -> None:
        """Publish one strictly increasing shared /clock sample."""
        self.clock_nanos += 1_000_000_000
        self.node.publish("/clock", FakeClockMsg(self.clock_nanos))

    def runtime(self, robot_id: str) -> RosMoveItRuntime:
        return self.session.runtime_for(robot_id)

    def controllers(self, robot_id: str) -> FakeServiceClient:
        name = f"/{robot_id}/controller_manager/list_controllers"
        return self.node.clients(name)[0]

    def fk(self, robot_id: str) -> FakeServiceClient:
        return self.node.clients(f"/{robot_id}/compute_fk")[0]

    def scene(self, robot_id: str) -> FakeServiceClient:
        clients = self.node.clients(SERVICE_SCENE_STATE)
        return clients[self.session.robot_ids.index(robot_id)]

    def arm(self, robot_id: str) -> FakeActionClient:
        name = f"/{robot_id}/arm_controller/follow_joint_trajectory"
        return self.node.actions(name)[0]

    def move_group(self, robot_id: str) -> FakeActionClient:
        return self.node.actions(f"/{robot_id}/move_action")[0]

    def gripper(self, robot_id: str) -> FakeActionClient:
        name = f"/{robot_id}/gripper_controller/follow_joint_trajectory"
        return self.node.actions(name)[0]


def make_world(robot_ids: tuple[str, ...] = (ROBOT_0, ROBOT_1)) -> World:
    node = FakeNode()
    clock = FakeClock()
    session = RosMoveItRobotSession(
        node,
        robot_ids,
        factories=fake_factories(),
        profile_factory=RobotRuntimeProfile.from_robot_id,
        qos_state=SimpleNamespace(label="state"),
        qos_sensor=SimpleNamespace(label="sensor"),
        clock=clock,
        sleep=clock.advance,
    )
    session.register()
    return World(node=node, clock=clock, session=session)


def prime_states(world: World, robot_id: str, positions: dict | None = None) -> None:
    world.tick_clock()
    world.tick_clock()
    state = device_state(robot_id, positions)
    velocity = [0.0] * len(BASELINE)
    world.node.publish(f"/{robot_id}/isaac_joint_states", FakeJointStateMsg(state, velocity))
    world.node.publish(f"/{robot_id}/joint_states", FakeJointStateMsg(state, velocity))


def prime_robot(
    world: World,
    robot_id: str,
    *,
    positions: dict | None = None,
    controllers_active: bool = True,
    arm_ready: bool = True,
    move_group_ready: bool = True,
    gripper_ready: bool = True,
    fk_ready: bool = True,
    scene_ready: bool = True,
    scene_document: dict | None = None,
) -> None:
    prime_states(world, robot_id, positions)
    state = "active" if controllers_active else "inactive"
    world.controllers(robot_id).response = SimpleNamespace(
        controller=[
            SimpleNamespace(name=name, state=state) for name in REQUIRED_CONTROLLERS
        ]
    )
    world.arm(robot_id).server_ready = arm_ready
    world.move_group(robot_id).server_ready = move_group_ready
    world.gripper(robot_id).server_ready = gripper_ready
    world.fk(robot_id).ready = fk_ready
    world.fk(robot_id).response = FakeFKResponse(pose=identity_pose())
    world.scene(robot_id).ready = scene_ready
    world.scene(robot_id).response = SimpleNamespace(
        message=json.dumps(
            dual_scene_document() if scene_document is None else scene_document
        )
    )


def dual_scene_document() -> dict:
    return {
        "schema": SCENE_STATE_SCHEMA,
        "sim_time_s": 4.5,
        "bricks": {"brick-44": {"aabb_min": [0.0, 0.0, 0.0]}},
        "connections": {"brick-44|brick-45": True},
        "robots": {
            ROBOT_0: {
                "robot_id": ROBOT_0,
                "sim_time_s": 4.5,
                "gripper": {"gripper_joint1": 0.031, "gripper_joint2": -0.031},
                "opening_m": 0.062,
            },
            ROBOT_1: {
                "robot_id": ROBOT_1,
                "sim_time_s": 4.5,
                "gripper": {"gripper_joint1": 0.005, "gripper_joint2": -0.005},
                "opening_m": 0.01,
                "opening_error": "contact",
            },
        },
    }


def joint_operation_for(robot_id: str, request_id: str):
    target, decision = build_joint_target(
        {name: value for name, value in BASELINE.items()},
        {"joint1": 0.05},
    )
    assert decision.accepted is True
    assert target is not None
    return RobotOperationRequest(
        robot_id=robot_id,
        operation=joint_operation(target, 5.0, request_id),
    )


def session_request(*items: RobotOperationRequest, timeout_s: float = 2.0, request_id: str = "sess-1"):
    return RosSessionRequest(
        request_id=request_id,
        robots=(ROBOT_0, ROBOT_1),
        timeout_s=timeout_s,
        operations=tuple(items),
    )


# ----------------------------------------------------------- endpoint routing


def test_session_uses_only_profile_endpoints_and_shared_clock():
    world = make_world()
    topics = world.node.subscription_topics()
    assert topics.count("/clock") == 2  # one shared clock subscription per robot
    for robot_id in (ROBOT_0, ROBOT_1):
        assert f"/{robot_id}/isaac_joint_states" in topics
        assert f"/{robot_id}/joint_states" in topics
        assert f"/{robot_id}/isaac_joint_command" in topics
    for canonical in (
        "/isaac_joint_states",
        "/isaac_joint_command",
        "/joint_states",
    ):
        assert canonical not in topics

    action_names = set(world.node.action_names())
    for robot_id in (ROBOT_0, ROBOT_1):
        assert f"/{robot_id}/move_action" in action_names
        assert f"/{robot_id}/arm_controller/follow_joint_trajectory" in action_names
        assert (
            f"/{robot_id}/gripper_controller/follow_joint_trajectory" in action_names
        )
    assert "/move_action" not in action_names

    service_names = set(world.node.service_names())
    for robot_id in (ROBOT_0, ROBOT_1):
        assert f"/{robot_id}/controller_manager/list_controllers" in service_names
        assert f"/{robot_id}/plan_kinematic_path" in service_names
        assert f"/{robot_id}/compute_fk" in service_names
        assert f"/{robot_id}/get_planning_scene" in service_names
    assert "/compute_fk" not in service_names
    assert "/plan_kinematic_path" not in service_names
    assert "/get_planning_scene" not in service_names
    assert SERVICE_SCENE_STATE in service_names


def test_profile_derivation_is_verbatim_and_frames_are_prefixed():
    profile = RobotRuntimeProfile.from_robot_id(ROBOT_1)
    runtime = RosMoveItRuntime(None, profile=profile)
    assert runtime.robot_id == ROBOT_1
    assert runtime.topics == runtime_topics_for_profile(profile)
    assert runtime.services == runtime_services_for_profile(profile)
    assert runtime.actions == runtime_actions_for_profile(profile)
    assert runtime.topics.clock == "/clock"
    assert runtime.services.scene_state == SERVICE_SCENE_STATE
    assert runtime.base_frame == f"{ROBOT_1}_base_link"
    assert runtime.link6_link == f"{ROBOT_1}_link6"
    assert runtime.tcp_link == f"{ROBOT_1}_rocobrick_tcp"
    assert runtime.arm_planning_group == f"{ROBOT_1}_arm"
    assert runtime.tcp_planning_group == f"{ROBOT_1}_arm_tcp"
    endpoints = runtime.endpoint_map()
    for name in (
        "isaac_joint_states",
        "isaac_joint_command",
        "joint_states",
        "move_group",
        "arm_follow",
        "gripper_follow",
        "list_controllers",
        "plan",
        "compute_fk",
        "apply_planning_scene",
    ):
        assert endpoints[name].startswith(f"/{ROBOT_1}/")
    assert endpoints["scene_state"] == SERVICE_SCENE_STATE


def test_profile_and_explicit_endpoints_cannot_be_mixed():
    profile = RobotRuntimeProfile.from_robot_id(ROBOT_1)
    with pytest.raises(ValueError, match="must not be overridden"):
        RosMoveItRuntime(
            None, profile=profile, topics=RuntimeTopics(clock="/custom_clock")
        )
    with pytest.raises(ValueError, match="must not be overridden"):
        RosMoveItRuntime(None, profile=profile, actions=RuntimeActions())
    with pytest.raises(ValueError, match="must not be overridden"):
        RosMoveItRuntime(None, profile=profile, services=RuntimeServices())


def test_canonical_single_robot_runtime_is_unchanged():
    runtime = RosMoveItRuntime(None, factories=fake_factories())
    assert runtime.robot_id == ""
    assert runtime.topics == RuntimeTopics()
    assert runtime.services == RuntimeServices()
    assert runtime.actions == RuntimeActions()
    assert runtime.base_frame == BASE_FRAME
    assert runtime.link6_link == TIP_LINK
    assert runtime.tcp_link == TCP_LINK


# ------------------------------------------------------- cross-robot rejection


def test_cross_robot_state_messages_cannot_pass():
    world = make_world()
    # The host bridge mis-maps piper_0's articulation onto piper_1's topic.
    world.node.publish(
        f"/{ROBOT_1}/isaac_joint_states",
        FakeJointStateMsg(device_state(ROBOT_0), [0.0] * 8),
    )
    world.node.publish(
        f"/{ROBOT_1}/joint_states",
        FakeJointStateMsg(device_state(ROBOT_0), [0.0] * 8),
    )
    assert world.runtime(ROBOT_1).latest_isaac_state() is None
    assert world.runtime(ROBOT_1).latest_joint_state() is None
    assert world.runtime(ROBOT_1).readiness_snapshot().isaac_state_complete is False

    prime_states(world, ROBOT_1)
    piper_1 = world.runtime(ROBOT_1)
    assert piper_1.latest_isaac_state() == pytest.approx(BASELINE)
    assert world.runtime(ROBOT_0).latest_isaac_state() is None
    assert world.runtime(ROBOT_0).readiness_snapshot().isaac_state_complete is False


def test_partial_or_duplicated_device_names_are_rejected():
    world = make_world()
    partial = dict(device_state(ROBOT_1))
    partial.pop("piper_1_joint3")
    world.node.publish(
        f"/{ROBOT_1}/isaac_joint_states", FakeJointStateMsg(partial, [0.0] * 8)
    )
    duplicated = [("piper_1_joint1", 0.0), ("piper_1_joint1", 0.0)]
    message = FakeJointStateMsg(device_state(ROBOT_1), [0.0] * 8)
    message.name = [name for name, _ in duplicated]
    message.position = [value for _, value in duplicated]
    world.node.publish(f"/{ROBOT_1}/isaac_joint_states", message)
    assert world.runtime(ROBOT_1).latest_isaac_state() is None


def test_controller_action_fk_and_scene_readiness_are_per_robot():
    world = make_world()
    prime_robot(world, ROBOT_0)
    prime_robot(world, ROBOT_1)
    world.controllers(ROBOT_1).response = SimpleNamespace(
        controller=[
            SimpleNamespace(name=name, state="inactive")
            for name in REQUIRED_CONTROLLERS
        ]
    )
    reports = world.session.capability_readiness_by_robot()
    assert reports[ROBOT_0].controllers_active is True
    assert reports[ROBOT_1].controllers_active is False
    assert reports[ROBOT_1].passed is False
    assert world.session.not_ready_robots() == (ROBOT_1,)
    assert reports[ROBOT_1].robot_id == ROBOT_1
    assert reports[ROBOT_1].endpoints
    assert dict(reports[ROBOT_1].endpoints)["isaac_joint_states"] == (
        f"/{ROBOT_1}/isaac_joint_states"
    )
    # The piper_0 controller answer can never satisfy piper_1.
    world.controllers(ROBOT_1).response = SimpleNamespace(
        controller=[
            SimpleNamespace(name=name, state="active")
            for name in REQUIRED_CONTROLLERS
        ]
    )
    assert world.session.capability_readiness_by_robot()[ROBOT_1].controllers_active


def test_capability_readiness_covers_gripper_and_scene_query():
    world = make_world()
    prime_robot(world, ROBOT_0)
    report = world.runtime(ROBOT_0).capability_readiness()
    assert report.passed is True
    assert report.serialize()["components"] == dict.fromkeys(
        CAPABILITY_COMPONENTS, True
    )

    world.gripper(ROBOT_0).server_ready = False
    report = world.runtime(ROBOT_0).capability_readiness()
    assert report.gripper_action_ready is False
    assert report.passed is False
    # The legacy motion snapshot deliberately stays narrower.
    assert world.runtime(ROBOT_0).readiness_snapshot().passed is True

    world.gripper(ROBOT_0).server_ready = True
    world.scene(ROBOT_0).ready = False
    report = world.runtime(ROBOT_0).capability_readiness()
    assert report.scene_query_ready is False
    assert report.passed is False


def test_readiness_components_are_not_shared_between_robots():
    world = make_world()
    prime_robot(world, ROBOT_0)
    world.arm(ROBOT_0).server_ready = False
    world.gripper(ROBOT_0).server_ready = False
    world.fk(ROBOT_0).ready = False
    report_0 = world.runtime(ROBOT_0).capability_readiness()
    report_1 = world.runtime(ROBOT_1).capability_readiness()
    assert report_0.arm_action_ready is False
    assert report_0.gripper_action_ready is False
    assert report_0.compute_fk_ready is False
    assert report_1.arm_action_ready is True
    assert report_1.gripper_action_ready is True
    assert report_1.compute_fk_ready is True
    # piper_1 has no state at all, so it is not ready either.
    assert report_1.isaac_state_complete is False
    assert report_1.passed is False


def test_joint_goal_constraints_use_the_robot_device_names():
    world = make_world()
    prime_robot(world, ROBOT_1)
    target, decision = build_joint_target(dict(BASELINE), {"joint1": 0.05})
    assert decision.accepted is True and target is not None
    result = world.runtime(ROBOT_1).send_joint(target, 5.0)
    assert result.status.value == "REJECTED"  # fake server refused the goal
    goals = world.move_group(ROBOT_1).goals
    assert len(goals) == 1
    request = goals[0].request
    assert request.group_name == f"{ROBOT_1}_arm"
    constraints = request.goal_constraints[0].joint_constraints
    assert [item.joint_name for item in constraints] == list(
        device_joint_names(ROBOT_1)[:6]
    )
    assert [item.position for item in constraints] == pytest.approx(
        [target.positions[name] for name in ARM_JOINT_NAMES]
    )
    assert world.move_group(ROBOT_0).goals == []


def test_cross_robot_planned_trajectory_is_rejected():
    world = make_world()
    prime_robot(world, ROBOT_1)
    trajectory = SimpleNamespace(
        planned_trajectory=SimpleNamespace(
            joint_trajectory=SimpleNamespace(
                joint_names=list(device_joint_names(ROBOT_0)[:6]),
                points=[SimpleNamespace(positions=[0.0] * 6) for _ in range(3)],
            )
        )
    )
    waypoints, decision = world.runtime(ROBOT_1)._inspect_trajectory(trajectory)
    assert waypoints == 3
    assert decision.accepted is False
    assert set(decision.details["extra"]) == set(device_joint_names(ROBOT_0)[:6])


def test_pose_goal_uses_the_robot_groups_frames_and_links():
    world = make_world()
    prime_robot(world, ROBOT_1)
    link6 = Link6PoseTarget(
        frame_id=BASE_FRAME,
        position=(0.1, 0.0, 0.2),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        position_tolerance_m=0.005,
        orientation_tolerance_rad=0.015,
    )
    world.runtime(ROBOT_1).send_link6_pose(link6, 5.0)
    request = world.move_group(ROBOT_1).goals[0].request
    assert request.group_name == f"{ROBOT_1}_arm"
    position = request.goal_constraints[0].position_constraints[0]
    orientation = request.goal_constraints[0].orientation_constraints[0]
    assert position.link_name == f"{ROBOT_1}_link6"
    assert position.header.frame_id == f"{ROBOT_1}_base_link"
    assert orientation.link_name == f"{ROBOT_1}_link6"
    assert orientation.header.frame_id == f"{ROBOT_1}_base_link"
    assert world.move_group(ROBOT_0).goals == []


def test_tcp_goal_uses_the_calibrated_robot_group_and_link():
    world = make_world()
    prime_robot(world, ROBOT_1)
    tcp = TcpPoseTarget(
        frame_id=BASE_FRAME,
        position=(0.1, 0.0, 0.2),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        position_tolerance_m=0.005,
        orientation_tolerance_rad=0.015,
        tip_link=TCP_LINK,
        tcp_status=TCP_STATUS_CALIBRATED_PROJECT,
        runtime_model_sha256=EXPECTED_RUNTIME_MODEL_SHA256,
        source_sha256="a" * 64,
    )
    world.runtime(ROBOT_1).send_tcp_pose(tcp, 5.0)
    request = world.move_group(ROBOT_1).goals[0].request
    assert request.group_name == f"{ROBOT_1}_arm_tcp"
    position = request.goal_constraints[0].position_constraints[0]
    assert position.link_name == f"{ROBOT_1}_rocobrick_tcp"
    assert position.header.frame_id == f"{ROBOT_1}_base_link"


def test_fk_requests_name_the_robot_device_joints_and_links():
    world = make_world()
    prime_robot(world, ROBOT_1)
    pose = world.runtime(ROBOT_1).compute_link6_fk(dict(BASELINE), timeout_s=1.0)
    assert pose == ((0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0))
    request = world.fk(ROBOT_1).requests[0]
    assert request.header.frame_id == f"{ROBOT_1}_base_link"
    assert request.fk_link_names == [f"{ROBOT_1}_link6"]
    assert request.robot_state.joint_state.name == list(device_joint_names(ROBOT_1))
    assert request.robot_state.joint_state.position == pytest.approx(
        [BASELINE[name] for name in BASELINE]
    )
    world.runtime(ROBOT_1).compute_tcp_fk(dict(BASELINE), timeout_s=1.0)
    assert world.fk(ROBOT_1).requests[1].fk_link_names == [
        f"{ROBOT_1}_rocobrick_tcp"
    ]


def test_fk_rejects_a_cross_robot_state_mapping():
    world = make_world()
    prime_robot(world, ROBOT_1)
    cross = {f"{ROBOT_0}_{name}": value for name, value in BASELINE.items()}
    assert world.runtime(ROBOT_1).compute_link6_fk(cross, timeout_s=1.0) is None
    assert world.fk(ROBOT_1).requests == []


# ------------------------------------------------------------- scene slicing


def test_robot_scene_view_requires_this_robots_own_block():
    document = dual_scene_document()
    view = robot_scene_view(document, ROBOT_1)
    assert view["robot_id"] == ROBOT_1
    assert view["schema"] == SCENE_STATE_SCHEMA
    assert view["bricks"] == document["bricks"]
    assert view["connections"] == document["connections"]
    assert view["gripper"] == {"gripper_joint1": 0.005, "gripper_joint2": -0.005}
    assert view["opening_m"] == pytest.approx(0.01)
    assert view["opening_error"] == "contact"

    with pytest.raises(RuntimeError, match="no block for robot 'piper_2'"):
        robot_scene_view(document, "piper_2")
    with pytest.raises(RuntimeError, match="no robots mapping"):
        robot_scene_view(
            {
                "schema": SCENE_STATE_SCHEMA,
                "sim_time_s": 1.0,
                "bricks": {},
                "connections": {},
                "gripper": {"gripper_joint1": 0.0, "gripper_joint2": 0.0},
            },
            ROBOT_1,
        )
    mismatched = dual_scene_document()
    mismatched["robots"][ROBOT_1] = dict(
        mismatched["robots"][ROBOT_1], robot_id=ROBOT_0
    )
    with pytest.raises(RuntimeError, match="is owned by"):
        robot_scene_view(mismatched, ROBOT_1)
    broken = dual_scene_document()
    broken["robots"][ROBOT_1] = dict(broken["robots"][ROBOT_1])
    broken["robots"][ROBOT_1]["gripper"] = {"gripper_joint1": 0.0}
    with pytest.raises(RuntimeError, match="gripper reading"):
        robot_scene_view(broken, ROBOT_1)


def test_robot_scene_view_fails_closed_on_a_foreign_binding():
    document = dual_scene_document()
    document["robots"][ROBOT_1] = dict(
        document["robots"][ROBOT_1], joint_names=list(device_joint_names(ROBOT_1))
    )
    view = robot_scene_view(
        document, ROBOT_1, joint_names=device_joint_names(ROBOT_1)
    )
    assert view["joint_names"] == list(device_joint_names(ROBOT_1))
    # A snapshot bound to the other robot's joints is never presented as
    # piper_1's observation.
    foreign = dual_scene_document()
    foreign["robots"][ROBOT_1] = dict(
        foreign["robots"][ROBOT_1], joint_names=list(device_joint_names(ROBOT_0))
    )
    with pytest.raises(RuntimeError, match="bound to a different robot"):
        robot_scene_view(foreign, ROBOT_1, joint_names=device_joint_names(ROBOT_1))
    malformed = dual_scene_document()
    malformed["robots"][ROBOT_1] = dict(
        malformed["robots"][ROBOT_1], joint_names=["piper_1_joint1", 7]
    )
    with pytest.raises(RuntimeError, match="malformed"):
        robot_scene_view(malformed, ROBOT_1)


def test_query_robot_scene_state_fails_closed_on_a_foreign_binding():
    world = make_world()
    prime_robot(world, ROBOT_1)
    document = dual_scene_document()
    document["robots"][ROBOT_1] = dict(
        document["robots"][ROBOT_1], joint_names=list(device_joint_names(ROBOT_0))
    )
    for client in world.node.clients(SERVICE_SCENE_STATE):
        client.response = SimpleNamespace(message=json.dumps(document))
    with pytest.raises(RuntimeError, match="bound to a different robot"):
        world.runtime(ROBOT_1).query_robot_scene_state(timeout_s=1.0)


def test_query_robot_scene_state_slices_the_global_service_answer():
    world = make_world()
    prime_robot(world, ROBOT_1)
    document = dual_scene_document()
    for client in world.node.clients(SERVICE_SCENE_STATE):
        client.response = SimpleNamespace(message=json.dumps(document))
    view = world.runtime(ROBOT_1).query_robot_scene_state(timeout_s=1.0)
    assert view["robot_id"] == ROBOT_1
    assert view["gripper"] == {"gripper_joint1": 0.005, "gripper_joint2": -0.005}
    # piper_0's reading is never part of piper_1's observation.
    assert world.runtime(ROBOT_0).query_robot_scene_state(timeout_s=1.0)["gripper"] == {
        "gripper_joint1": 0.031,
        "gripper_joint2": -0.031,
    }
    assert json.dumps(view) .find("0.031") == -1
    # A canonical single-robot document cannot impersonate piper_1's block.
    for client in world.node.clients(SERVICE_SCENE_STATE):
        client.response = SimpleNamespace(
            message=json.dumps(
                {
                    "schema": SCENE_STATE_SCHEMA,
                    "sim_time_s": 1.0,
                    "bricks": {},
                    "connections": {},
                    "gripper": {"gripper_joint1": 0.0, "gripper_joint2": 0.0},
                    "opening_m": 0.0,
                }
            )
        )
    with pytest.raises(RuntimeError, match="no robots mapping"):
        world.runtime(ROBOT_1).query_robot_scene_state(timeout_s=1.0)


def test_query_robot_scene_state_requires_a_profiled_runtime():
    runtime = RosMoveItRuntime(None, factories=fake_factories())
    with pytest.raises(RuntimeError, match="profiled robot runtime"):
        runtime.query_robot_scene_state(timeout_s=1.0)


# ---------------------------------------------------------------- session gate


def test_one_not_ready_robot_fails_the_session_and_dispatches_nothing():
    world = make_world()
    prime_robot(world, ROBOT_0)
    request = session_request(joint_operation_for(ROBOT_0, "op-1"))
    assert world.session.wait_until_ready(timeout_s=0.0) is False
    result = perform_session(world.session, request)
    assert result.all_ready is False
    assert result.passed is False
    assert set(result.readiness_by_robot) == {ROBOT_0, ROBOT_1}
    assert result.readiness_by_robot[ROBOT_0]["passed"] is True
    assert result.readiness_by_robot[ROBOT_1]["passed"] is False
    assert len(result.operations) == 1
    record = result.operations[0]
    assert record["robot_id"] == ROBOT_0
    assert record["dispatched"] is False
    assert record["motion_sent"] is False
    assert record["result"]["passed"] is False
    assert "readiness failed" in record["result"]["error"]
    # Zero dispatch: not even the READY robot's move_group was touched.
    assert world.move_group(ROBOT_0).goals == []
    assert world.move_group(ROBOT_1).goals == []


def test_session_routes_each_operation_to_its_own_robot():
    world = make_world()
    prime_robot(world, ROBOT_0)
    prime_robot(world, ROBOT_1)
    assert world.session.wait_until_ready(timeout_s=1.0) is True
    request = session_request(
        joint_operation_for(ROBOT_1, "op-1"), timeout_s=2.0
    )
    result = perform_session(world.session, request)
    assert result.all_ready is True
    assert result.passed is False  # the fake move_group refuses the goal
    record = result.operations[0]
    assert record["robot_id"] == ROBOT_1
    assert record["dispatched"] is True
    assert record["kind"] == "JOINT_TARGET"
    assert record["motion_sent"] is False
    assert world.move_group(ROBOT_0).goals == []
    goal = world.move_group(ROBOT_1).goals[0]
    assert goal.request.group_name == f"{ROBOT_1}_arm"
    assert [
        item.joint_name for item in goal.request.goal_constraints[0].joint_constraints
    ] == list(device_joint_names(ROBOT_1)[:6])


def test_session_scene_query_is_sliced_per_robot():
    world = make_world()
    prime_robot(world, ROBOT_0)
    prime_robot(world, ROBOT_1)
    document = dual_scene_document()
    for client in world.node.clients(SERVICE_SCENE_STATE):
        client.response = SimpleNamespace(message=json.dumps(document))
    operation = scene_query_operation(
        ["brick-44"], [], include_gripper=True, timeout_s=2.0, request_id="scene-1"
    )
    request = session_request(
        RobotOperationRequest(robot_id=ROBOT_1, operation=operation)
    )
    result = perform_session(world.session, request)
    record = result.operations[0]
    assert record["robot_id"] == ROBOT_1
    assert record["result"]["passed"] is True
    scene = record["result"]["execution"]["scene_state"]
    assert scene["robot_id"] == ROBOT_1
    assert scene["gripper"] == {"gripper_joint1": 0.005, "gripper_joint2": -0.005}
    assert scene["opening_m"] == pytest.approx(0.01)
    assert scene["bricks"] == {"brick-44": document["bricks"]["brick-44"]}
    assert scene["connections"] == {}


def test_session_readiness_only_request_answers_every_robot():
    world = make_world()
    prime_robot(world, ROBOT_0)
    prime_robot(world, ROBOT_1)
    result = perform_session(world.session, session_request())
    assert result.all_ready is True
    assert result.passed is True
    assert result.operations == ()
    assert set(result.readiness_by_robot) == {ROBOT_0, ROBOT_1}
    assert all(
        report["passed"] is True for report in result.readiness_by_robot.values()
    )
    assert RuntimeOperationResult.parse(
        {
            "schema": "rocobrick-runtime-operation-result/v1",
            "request_id": "x",
            "kind": "READINESS",
            "passed": True,
            "execution": {},
            "evidence": {},
            "error": None,
        }
    ).passed is True


def test_session_result_is_strict_and_round_trips():
    world = make_world()
    prime_robot(world, ROBOT_0)
    prime_robot(world, ROBOT_1)
    result = perform_session(
        world.session, session_request(joint_operation_for(ROBOT_1, "op-rt"))
    )
    document = json.loads(json.dumps(result.serialize()))
    assert document["schema"] == SESSION_RESULT_SCHEMA
    parsed = RosSessionResult.parse(document)
    assert parsed == result
    assert parsed.operations[0]["robot_id"] == ROBOT_1
    with pytest.raises(ValueError):
        RosSessionResult.parse({**document, "extra": True})


def test_session_request_contract_is_strict():
    request = session_request(joint_operation_for(ROBOT_1, "op-1"))
    document = request.serialize()
    assert document["schema"] == SESSION_REQUEST_SCHEMA
    assert RosSessionRequest.parse(document) == request
    with pytest.raises(ValueError):
        RosSessionRequest.parse({**document, "extra": True})
    with pytest.raises(ValueError, match="not in"):
        RosSessionRequest.parse(
            session_request(
                joint_operation_for(ROBOT_1, "op-1")
            ).serialize()
            | {
                "operations": [
                    {
                        "robot_id": "piper_2",
                        "operation": joint_operation_for(
                            ROBOT_1, "op-2"
                        ).operation.serialize(),
                    }
                ]
            }
        )
    duplicated = session_request(
        joint_operation_for(ROBOT_0, "op-1"), joint_operation_for(ROBOT_1, "op-1")
    ).serialize()
    with pytest.raises(ValueError, match="duplicated"):
        RosSessionRequest.parse(duplicated)
    with pytest.raises(ValueError, match="timeout_s"):
        RosSessionRequest.parse({**document, "timeout_s": 0.0})
    with pytest.raises(ValueError, match="must not be empty"):
        RosSessionRequest.parse({**document, "robots": []})
    with pytest.raises(ValueError, match="unique"):
        RosSessionRequest.parse({**document, "robots": [ROBOT_0, ROBOT_0]})


def test_session_rejects_a_mismatched_robot_set():
    world = make_world()
    prime_robot(world, ROBOT_0)
    prime_robot(world, ROBOT_1)
    request = RosSessionRequest(
        request_id="sess-mismatch",
        robots=(ROBOT_1,),
        timeout_s=1.0,
        operations=(),
    )
    result = perform_session(world.session, request)
    assert result.passed is False
    assert "do not match" in result.error


def test_parse_robots_accepts_only_valid_unique_ids():
    assert parse_robots(None) is None
    assert parse_robots("piper_0,piper_1") == (ROBOT_0, ROBOT_1)
    with pytest.raises(ValueError, match="comma-separated"):
        parse_robots("piper_0,,piper_1")
    with pytest.raises(ValueError, match="repeat"):
        parse_robots("piper_0,piper_0")
    with pytest.raises(ValueError, match="ROS-safe"):
        parse_robots("piper_0,/piper_1")


def test_scene_probe_is_fail_closed_when_a_robot_block_is_missing():
    world = make_world()
    prime_robot(world, ROBOT_0)
    prime_robot(world, ROBOT_1)
    only_piper_0 = {
        "schema": SCENE_STATE_SCHEMA,
        "sim_time_s": 4.5,
        "bricks": {"brick-44": {"aabb_min": [0.0, 0.0, 0.0]}},
        "connections": {},
        "robots": {ROBOT_0: dual_scene_document()["robots"][ROBOT_0]},
    }
    world.scene(ROBOT_1).response = SimpleNamespace(
        message=json.dumps(only_piper_0)
    )
    # The pure report only sees the endpoint; the probed verdict fails.
    assert world.runtime(ROBOT_1).capability_readiness().scene_query_ready is True
    probed = world.runtime(ROBOT_1).capability_readiness(probe_scene=True)
    assert probed.scene_query_ready is False
    assert probed.passed is False
    assert world.runtime(ROBOT_0).capability_readiness(probe_scene=True).passed is True
    assert world.session.not_ready_robots(probe_scene=True) == (ROBOT_1,)
    assert world.session.wait_until_ready(timeout_s=0.0) is False
    assert world.session.capability_readiness_by_robot(probe_scene=True)[
        ROBOT_1
    ].scene_query_ready is False
    assert world.session.runtime_for(ROBOT_1).scene_query_probe() is None


def test_session_gate_uses_the_live_per_robot_scene_response():
    world = make_world()
    prime_robot(world, ROBOT_0)
    prime_robot(world, ROBOT_1)
    foreign = dual_scene_document()
    foreign["robots"][ROBOT_1] = dict(
        foreign["robots"][ROBOT_1], joint_names=list(device_joint_names(ROBOT_0))
    )
    world.scene(ROBOT_1).response = SimpleNamespace(message=json.dumps(foreign))
    request = session_request(joint_operation_for(ROBOT_0, "op-gate"))
    result = perform_session(world.session, request)
    assert result.all_ready is False
    assert result.operations[0]["dispatched"] is False
    assert result.operations[0]["motion_sent"] is False
    assert world.move_group(ROBOT_0).goals == []
    assert result.readiness_by_robot[ROBOT_1]["components"]["scene_query_ready"] is (
        False
    )
    assert result.readiness_by_robot[ROBOT_1]["passed"] is False
    assert result.readiness_by_robot[ROBOT_0]["passed"] is True


def test_session_wait_until_ready_is_bounded_and_never_sends_motion():
    world = make_world()
    prime_robot(world, ROBOT_0)
    assert world.session.wait_until_ready(timeout_s=0.5) is False
    # The bound is respected up to one polling interval of overshoot.
    assert 0.5 <= world.clock.now <= 0.7
    assert world.move_group(ROBOT_0).goals == []
    with pytest.raises(ValueError):
        world.session.wait_until_ready(timeout_s=-1.0)
    assert world.session.shutdown_errors == []
    world.session.shutdown()
    assert world.session.runtimes[ROBOT_1]._subscriptions == {}


# ------------------------------------------------------------- entry surface


def test_session_entry_main_writes_one_session_result(tmp_path, capsys, monkeypatch):
    import rocobrick.ros_session_entry as entry

    world = make_world()
    prime_robot(world, ROBOT_0)
    prime_robot(world, ROBOT_1)
    request = session_request(joint_operation_for(ROBOT_1, "op-main"))
    request_path = tmp_path / "session-request.json"
    request_path.write_text(json.dumps(request.serialize()), encoding="utf-8")
    result_path = tmp_path / "session-result.json"

    def fake_spin_up(node_name, robots, fk_timeout_s):
        assert robots == (ROBOT_0, ROBOT_1)
        return world.session, None, None, None

    monkeypatch.setattr(entry, "_spin_up_session", fake_spin_up)
    monkeypatch.setattr(entry, "_spin_down_session", lambda *args: None)
    code = entry.main(
        [
            "--request",
            str(request_path),
            "--result",
            str(result_path),
            "--robots",
            "piper_0,piper_1",
        ]
    )
    assert code == 1  # the fake move_group never accepts the joint goal
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 1
    summary = json.loads(lines[0])
    assert summary["schema"] == SESSION_SUMMARY_SCHEMA
    assert summary["result_schema"] == SESSION_RESULT_SCHEMA
    assert summary["robots"] == [ROBOT_0, ROBOT_1]
    assert summary["all_ready"] is True
    document = json.loads(result_path.read_text(encoding="utf-8"))
    parsed = RosSessionResult.parse(document)
    assert parsed.all_ready is True
    assert parsed.operations[0]["robot_id"] == ROBOT_1


def test_session_entry_main_rejects_a_robot_list_mismatch(tmp_path, capsys, monkeypatch):
    import rocobrick.ros_session_entry as entry

    request = session_request()
    request_path = tmp_path / "session-request.json"
    request_path.write_text(json.dumps(request.serialize()), encoding="utf-8")
    result_path = tmp_path / "session-result.json"

    def explode(*args, **kwargs):
        raise AssertionError("no session may be created for a bad request")

    monkeypatch.setattr(entry, "_spin_up_session", explode)
    code = entry.main(
        [
            "--request",
            str(request_path),
            "--result",
            str(result_path),
            "--robots",
            ROBOT_1,
        ]
    )
    assert code == 2
    summary = json.loads(capsys.readouterr().out.splitlines()[0])
    assert summary["schema"] == SESSION_SUMMARY_SCHEMA
    assert summary["passed"] is False
    document = json.loads(result_path.read_text(encoding="utf-8"))
    assert "do not match" in document["error"]


def test_single_robot_operation_mode_still_uses_the_legacy_entry(tmp_path, monkeypatch):
    import rocobrick.ros_session_entry as entry

    world = make_world((ROBOT_1,))
    prime_robot(world, ROBOT_1)
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            readiness_operation(timeout_s=1.0, request_id="ready-legacy").serialize()
        ),
        encoding="utf-8",
    )
    result_path = tmp_path / "result.json"
    seen: dict = {}

    def fake_spin_up(node_name, fk_timeout_s):
        seen["node_name"] = node_name
        seen["fk_timeout_s"] = fk_timeout_s
        return (
            world.runtime(ROBOT_1),
            world.session.executor_for(ROBOT_1),
            None,
            None,
            None,
        )

    monkeypatch.setattr(entry, "_spin_up", fake_spin_up)
    monkeypatch.setattr(entry, "_spin_down", lambda *args: None)
    code = entry.main(["--request", str(request_path), "--result", str(result_path)])
    assert code == 0
    assert seen["node_name"] == "rocobrick_session_entry"
    document = json.loads(result_path.read_text(encoding="utf-8"))
    assert document["schema"] == "rocobrick-runtime-operation-result/v1"
    assert document["request_id"] == "ready-legacy"
    assert document["passed"] is True
    assert document["evidence"]["snapshot"]["arm_action_ready"] is True


def test_executor_is_robot_keyed_and_refuses_a_mismatch():
    world = make_world()
    executor = world.session.executor_for(ROBOT_1)
    assert executor.robot_id == ROBOT_1
    with pytest.raises(ValueError, match="does not match"):
        RosMoveItExecutor(world.runtime(ROBOT_1), robot_id=ROBOT_0)
    # The canonical executor keeps no robot id and no robot_id detail key.
    canonical = RosMoveItRuntime(None, factories=fake_factories())
    legacy = RosMoveItExecutor(canonical)
    assert legacy.robot_id is None
    invalid = Link6PoseTarget(
        frame_id="world",
        position=(0.1, 0.0, 0.2),
        orientation_xyzw=(2.0, 0.0, 0.0, 2.0),
    )
    assert "robot_id" not in legacy.execute_link6_pose(invalid, timeout_s=1.0).details
