"""R2 ROS MoveIt runtime adapter contract tests (offline fakes, no ROS).

Pins the exact runtime adapter behavior around the accepted R1 library layer:
module import without rclpy/moveit_msgs, lazy ROS imports only inside
registration/send methods (AST), no Isaac/tools imports, exact topic/service/
action defaults, readiness snapshot composition over the R1 pure predicates,
bounded readiness waits that never send motion, MoveGroup goal construction
(group arm, plan_only false, exact joint/pose constraints), bounded deadlines
with explicit cancellation, R1 trajectory inspection and final-feedback gates
that never fake success, and the executor admission/delegation boundary.
Everything runs on injected fake message classes and factories; no live ROS,
no Isaac, no Docker.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from rocobrick.moveit_contract import (
    ACTION_ARM_FOLLOW,
    ACTION_GRIPPER_FOLLOW,
    ARM_JOINT_NAMES,
    GRIPPER_JOINT_NAMES,
    JOINT_NAMES,
    MOVE_GROUP_ACTION_CANDIDATES,
    REQUIRED_CONTROLLERS,
    SERVICE_COMPUTE_FK,
    SERVICE_LIST_CONTROLLERS,
    SERVICE_PLAN,
    TOPIC_CLOCK,
    TOPIC_ISAAC_JOINT_COMMAND,
    TOPIC_ISAAC_JOINT_STATES,
    TOPIC_JOINT_STATES,
    GripperTarget,
    JointTarget,
    Link6PoseTarget,
)
from rocobrick.moveit_execution import (
    ExecutionResult,
    ExecutionStatus,
    MoveItExecutor,
    is_success,
)
from rocobrick.moveit_requests import build_joint_target, build_link6_pose_target
from rocobrick.ros_moveit_executor import RosMoveItExecutor
from rocobrick.ros_moveit_runtime import (
    SCENE_STATE_SCHEMA,
    SERVICE_CALL_TIMEOUT_S,
    SERVICE_SCENE_STATE,
    ReadinessSnapshot,
    RosMoveItFactories,
    RosMoveItRuntime,
    RuntimeActions,
    RuntimeServices,
    RuntimeTopics,
)

ROOT = Path(__file__).parents[1]
RUNTIME_MODULES = ("ros_moveit_runtime", "ros_moveit_executor")

FORBIDDEN_LAZY_ROOTS = {
    "rclpy",
    "moveit_msgs",
    "sensor_msgs",
    "rosgraph_msgs",
    "controller_manager_msgs",
    "control_msgs",
    "action_msgs",
    "shape_msgs",
    "geometry_msgs",
    "builtin_interfaces",
}
FORBIDDEN_ANYWHERE_ROOTS = {"isaacsim", "omni", "isaac", "tools"}

SNAPSHOT_FIELDS = (
    "clock_increasing",
    "isaac_state_complete",
    "joint_state_complete",
    "broadcaster_match",
    "controllers_active",
    "arm_action_ready",
    "move_group_action_ready",
    "compute_fk_ready",
)

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


# ---------------------------------------------------------------- fake transport


class FakeClock:
    """Injected monotonic clock the runtime can advance via the sleep hook."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, delta: float) -> None:
        self.now += float(delta)


class FakeFuture:
    def __init__(self) -> None:
        self._done = False
        self._value = None

    def done(self) -> bool:
        return self._done

    def result(self):
        return self._value

    def set(self, value) -> None:
        self._value = value
        self._done = True


class FakeGoalHandle:
    def __init__(self, accepted: bool = True) -> None:
        self.accepted = accepted
        self.result_future = FakeFuture()
        self.cancel_calls = 0

    def get_result_async(self) -> FakeFuture:
        return self.result_future

    def cancel_goal_async(self) -> FakeFuture:
        self.cancel_calls += 1
        future = FakeFuture()
        future.set(SimpleNamespace(return_code=0))
        return future


class FakeActionClient:
    def __init__(self, node: "FakeNode", action_type, name: str) -> None:
        self.node = node
        self.action_type = action_type
        self.name = name
        self.goals: list = []
        self.server_ready = True
        self.destroyed = False
        self.pending: FakeFuture | None = None
        node.action_clients[name] = self

    def wait_for_server(self, timeout_sec=None) -> bool:
        return self.server_ready

    def send_goal_async(self, goal) -> FakeFuture:
        self.goals.append(goal)
        self.pending = FakeFuture()
        return self.pending

    def destroy(self) -> None:
        self.destroyed = True


class FakeServiceClient:
    def __init__(self, name: str) -> None:
        self.name = name
        self.ready = True
        self.response = None
        self.requests: list = []

    def service_is_ready(self) -> bool:
        return self.ready

    def call_async(self, request) -> FakeFuture:
        self.requests.append(request)
        future = FakeFuture()
        if self.response is not None:
            future.set(self.response)
        return future


class FakeNode:
    def __init__(self) -> None:
        self.subscriptions: list = []
        self.service_clients: dict = {}
        self.action_clients: dict = {}
        self.destroyed_subscriptions: list = []
        self.destroyed_clients: list = []

    def create_subscription(self, msg_type, topic, callback, qos):
        subscription = SimpleNamespace(
            msg_type=msg_type, topic=topic, callback=callback, qos=qos
        )
        self.subscriptions.append(subscription)
        return subscription

    def create_client(self, srv_type, name) -> FakeServiceClient:
        client = FakeServiceClient(name)
        self.service_clients[name] = client
        return client

    def destroy_subscription(self, subscription) -> None:
        self.destroyed_subscriptions.append(subscription.topic)

    def destroy_client(self, client) -> None:
        self.destroyed_clients.append(client.name)


def publish(node: FakeNode, topic: str, message) -> None:
    delivered = 0
    for subscription in node.subscriptions:
        if subscription.topic == topic:
            subscription.callback(message)
            delivered += 1
    assert delivered == 1, f"topic {topic} has {delivered} subscribers"


class FakeClockMsg:
    def __init__(self, nanos: int) -> None:
        self.clock = SimpleNamespace(sec=nanos // 1_000_000_000, nanosec=nanos % 1_000_000_000)


class FakeJointStateMsg:
    def __init__(self, positions, velocity=()) -> None:
        self.name = list(positions)
        self.position = [float(value) for value in positions.values()]
        self.velocity = list(velocity)


# Fake MoveIt message classes mirroring the real fields the runtime touches.
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


FAKE_GOAL_STATUS = SimpleNamespace(
    STATUS_SUCCEEDED=4, STATUS_CANCELED=5, STATUS_ABORTED=6
)


# Fake scene-state Trigger and gripper trajectory message classes.
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


def fake_factories() -> RosMoveItFactories:
    return RosMoveItFactories(
        clock_type=lambda: FakeClockMsg,
        joint_state_type=lambda: FakeJointStateMsg,
        list_controllers_type=lambda: SimpleNamespace,
        list_controllers_request_factory=lambda: SimpleNamespace(),
        get_motion_plan_type=lambda: SimpleNamespace,
        get_position_fk_type=lambda: SimpleNamespace,
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
    events: list = field(default_factory=list)
    runtime: RosMoveItRuntime = None  # type: ignore[assignment]
    controllers: FakeServiceClient = None  # type: ignore[assignment]
    fk: FakeServiceClient = None  # type: ignore[assignment]
    scene_state: FakeServiceClient = None  # type: ignore[assignment]
    arm: FakeActionClient = None  # type: ignore[assignment]
    move_group: FakeActionClient = None  # type: ignore[assignment]
    gripper: FakeActionClient = None  # type: ignore[assignment]


def make_world(*, fk_verifier=None) -> World:
    node = FakeNode()
    clock = FakeClock()
    events: list = []

    def spin() -> None:
        if events:
            events.pop(0)()

    runtime = RosMoveItRuntime(
        node,
        qos_state=SimpleNamespace(label="state"),
        qos_sensor=SimpleNamespace(label="sensor"),
        factories=fake_factories(),
        fk_verifier=fk_verifier,
        clock=clock,
        spin=spin,
        sleep=clock.advance,
    )
    runtime.register_subscriptions()
    runtime.register_clients()
    return World(
        node=node,
        clock=clock,
        events=events,
        runtime=runtime,
        controllers=node.service_clients[SERVICE_LIST_CONTROLLERS],
        fk=node.service_clients[SERVICE_COMPUTE_FK],
        scene_state=node.service_clients[SERVICE_SCENE_STATE],
        arm=node.action_clients[ACTION_ARM_FOLLOW],
        move_group=node.action_clients[MOVE_GROUP_ACTION_CANDIDATES[0]],
        gripper=node.action_clients[ACTION_GRIPPER_FOLLOW],
    )


def prime_ready(
    world: World,
    *,
    baseline=None,
    broadcaster=None,
    controllers_active: bool = True,
    clock_samples=(1_000_000_000, 2_000_000_000),
) -> dict:
    state = dict(baseline if baseline is not None else BASELINE)
    for nanos in clock_samples:
        publish(world.node, TOPIC_CLOCK, FakeClockMsg(nanos))
    publish(
        world.node,
        TOPIC_ISAAC_JOINT_STATES,
        FakeJointStateMsg(state, velocity=[0.0] * len(JOINT_NAMES)),
    )
    publish(
        world.node,
        TOPIC_JOINT_STATES,
        FakeJointStateMsg(broadcaster if broadcaster is not None else state),
    )
    world.controllers.response = SimpleNamespace(
        controller=[
            SimpleNamespace(
                name=name, state="active" if controllers_active else "inactive"
            )
            for name in REQUIRED_CONTROLLERS
        ]
    )
    return state


def arm_values(state) -> list[float]:
    return [float(state[name]) for name in ARM_JOINT_NAMES]


def joint_final_state(target: JointTarget) -> dict:
    final = dict(BASELINE)
    final["joint1"] = target.positions["joint1"] + 0.005
    final["joint4"] = target.positions["joint4"] - 0.004
    return final


def trajectory_action_result(target=None, *, points=None, error_code: int = 1):
    if points is None:
        start = arm_values(BASELINE)
        if target is None:
            # Pose goals still return a joint trajectory; any in-limits shape.
            points = [start, start, start]
        else:
            end = arm_values(target.positions)
            middle = [(a + b) / 2.0 for a, b in zip(start, end)]
            points = [start, middle, end]
    trajectory = SimpleNamespace(
        joint_names=list(ARM_JOINT_NAMES),
        points=[
            SimpleNamespace(positions=[float(value) for value in point])
            for point in points
        ],
    )
    return SimpleNamespace(
        error_code=SimpleNamespace(val=error_code),
        planning_time=0.5,
        planned_trajectory=SimpleNamespace(joint_trajectory=trajectory),
    )


def script_success(
    world: World,
    *,
    action_result,
    final_state,
    post_states: int = 2,
    commands: int = 1,
) -> FakeGoalHandle:
    handle = FakeGoalHandle()
    world.events.append(lambda: world.move_group.pending.set(handle))

    def deliver_evidence() -> None:
        for _ in range(max(post_states - 1, 0)):
            publish(
                world.node,
                TOPIC_ISAAC_JOINT_STATES,
                FakeJointStateMsg(final_state, velocity=[0.0] * len(JOINT_NAMES)),
            )
        for _ in range(commands):
            publish(world.node, TOPIC_ISAAC_JOINT_COMMAND, FakeJointStateMsg(final_state))

    world.events.append(deliver_evidence)
    if post_states >= 2:
        world.events.append(
            lambda: publish(
                world.node,
                TOPIC_ISAAC_JOINT_STATES,
                FakeJointStateMsg(final_state, velocity=[0.0] * len(JOINT_NAMES)),
            )
        )
    world.events.append(
        lambda: handle.result_future.set(
            SimpleNamespace(status=FAKE_GOAL_STATUS.STATUS_SUCCEEDED, result=action_result)
        )
    )
    return handle


def script_terminal(world: World, *, status: int, action_result) -> FakeGoalHandle:
    handle = FakeGoalHandle()
    world.events.append(lambda: world.move_group.pending.set(handle))
    world.events.append(
        lambda: handle.result_future.set(
            SimpleNamespace(status=status, result=action_result)
        )
    )
    return handle


@pytest.fixture
def joint_target() -> JointTarget:
    target, decision = build_joint_target(BASELINE, {"joint1": 0.2, "joint4": 0.15})
    assert decision.accepted is True
    assert target is not None
    return target


@pytest.fixture
def pose_target() -> Link6PoseTarget:
    target, decision = build_link6_pose_target(
        frame_id="base_link",
        position=(0.1, 0.0, 0.2),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        position_tolerance_m=0.005,
        orientation_tolerance_rad=0.015,
    )
    assert decision.accepted is True
    assert target is not None
    return target


# ------------------------------------------------------- import and AST contract


def test_runtime_modules_import_without_ros_installed():
    for name in RUNTIME_MODULES:
        assert importlib.import_module(f"rocobrick.{name}") is not None


def _import_roots(nodes) -> set[str]:
    roots: set[str] = set()
    for node in nodes:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in sub.names)
            elif isinstance(sub, ast.ImportFrom) and sub.module:
                roots.add(sub.module.split(".")[0])
    return roots


@pytest.mark.parametrize("module_name", RUNTIME_MODULES)
def test_no_ros_imports_outside_methods(module_name):
    source = (ROOT / "src" / "rocobrick" / f"{module_name}.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    function_roots = _import_roots(functions)
    top_level_roots = _import_roots([tree]) - function_roots
    banned = top_level_roots & FORBIDDEN_LAZY_ROOTS
    assert not banned, f"{module_name} imports {sorted(banned)} outside methods"


def test_runtime_module_imports_ros_only_lazily():
    source = (ROOT / "src" / "rocobrick" / "ros_moveit_runtime.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    assert _import_roots(functions) & FORBIDDEN_LAZY_ROOTS


@pytest.mark.parametrize("module_name", RUNTIME_MODULES)
def test_runtime_modules_never_import_isaac_or_tools(module_name):
    source = (ROOT / "src" / "rocobrick" / f"{module_name}.py").read_text(encoding="utf-8")
    roots = _import_roots([ast.parse(source)])
    banned = roots & FORBIDDEN_ANYWHERE_ROOTS
    assert not banned, f"{module_name} imports forbidden modules: {sorted(banned)}"


# ------------------------------------------------------------- pinned defaults


def test_runtime_topics_default_to_the_pinned_contract():
    topics = RuntimeTopics()
    assert topics.clock == TOPIC_CLOCK
    assert topics.isaac_joint_states == TOPIC_ISAAC_JOINT_STATES
    assert topics.joint_states == TOPIC_JOINT_STATES
    assert topics.isaac_joint_command == TOPIC_ISAAC_JOINT_COMMAND
    with pytest.raises(dataclasses.FrozenInstanceError):
        topics.clock = "/other"  # type: ignore[misc]


def test_runtime_services_default_to_the_live_evidence_names():
    services = RuntimeServices()
    assert services.list_controllers == SERVICE_LIST_CONTROLLERS
    assert services.plan == SERVICE_PLAN
    assert services.compute_fk == SERVICE_COMPUTE_FK
    with pytest.raises(dataclasses.FrozenInstanceError):
        services.plan = "/other"  # type: ignore[misc]


def test_runtime_actions_default_to_arm_follow_and_first_candidate():
    actions = RuntimeActions()
    assert actions.arm_follow == ACTION_ARM_FOLLOW
    assert actions.move_group == MOVE_GROUP_ACTION_CANDIDATES[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        actions.move_group = "/other"  # type: ignore[misc]


def test_runtime_namespaces_are_overridable_and_frozen():
    topics = RuntimeTopics(clock="/custom_clock")
    services = RuntimeServices(plan="/custom_plan")
    actions = RuntimeActions(arm_follow="/custom_arm")
    assert topics.clock == "/custom_clock"
    assert services.plan == "/custom_plan"
    assert actions.arm_follow == "/custom_arm"


# --------------------------------------------------------------- construction


def test_constructor_injects_the_node_and_creates_nothing():
    node = FakeNode()
    runtime = RosMoveItRuntime(node, factories=fake_factories())
    assert runtime.node is node
    assert node.subscriptions == []
    assert node.service_clients == {}
    assert node.action_clients == {}
    assert runtime.topics == RuntimeTopics()
    assert runtime.services == RuntimeServices()
    assert runtime.actions == RuntimeActions()


def test_registration_wires_the_exact_topic_and_client_names():
    world = make_world()
    topics = world.runtime.topics
    assert [subscription.topic for subscription in world.node.subscriptions] == [
        topics.clock,
        topics.isaac_joint_states,
        topics.joint_states,
        topics.isaac_joint_command,
    ]
    assert world.node.subscriptions[0].msg_type is FakeClockMsg
    assert all(
        subscription.msg_type is FakeJointStateMsg
        for subscription in world.node.subscriptions[1:]
    )
    assert world.node.subscriptions[0].qos.label == "sensor"
    assert world.node.subscriptions[2].qos.label == "state"
    assert set(world.node.service_clients) == {
        SERVICE_LIST_CONTROLLERS,
        SERVICE_PLAN,
        SERVICE_COMPUTE_FK,
        SERVICE_SCENE_STATE,
    }
    assert set(world.node.action_clients) == {
        ACTION_ARM_FOLLOW,
        MOVE_GROUP_ACTION_CANDIDATES[0],
        ACTION_GRIPPER_FOLLOW,
    }
    world.runtime.register_subscriptions()
    world.runtime.register_clients()
    assert len(world.node.subscriptions) == 4
    assert len(world.node.service_clients) == 4
    assert len(world.node.action_clients) == 3


def test_scene_state_and_gripper_clients_are_not_readiness_gates():
    """Both new clients register but stay outside the motion-readiness gates."""
    world = make_world()
    prime_ready(world)
    world.scene_state.ready = False
    world.gripper.server_ready = False
    snapshot = world.runtime.readiness_snapshot()
    assert snapshot.passed is True
    assert snapshot.arm_action_ready is True
    assert snapshot.move_group_action_ready is True


# ----------------------------------------------------------- readiness snapshot


def test_readiness_snapshot_passed_requires_every_field():
    all_true = ReadinessSnapshot(**dict.fromkeys(SNAPSHOT_FIELDS, True))
    assert all_true.passed is True
    for name in SNAPSHOT_FIELDS:
        kwargs = dict.fromkeys(SNAPSHOT_FIELDS, True)
        kwargs[name] = False
        assert ReadinessSnapshot(**kwargs).passed is False
    assert ReadinessSnapshot().passed is False


def test_readiness_snapshot_is_frozen():
    snapshot = ReadinessSnapshot(clock_increasing=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.clock_increasing = False  # type: ignore[misc]


def test_readiness_snapshot_all_components_true():
    world = make_world()
    prime_ready(world)
    snapshot = world.runtime.readiness_snapshot()
    assert snapshot == ReadinessSnapshot(**dict.fromkeys(SNAPSHOT_FIELDS, True))
    assert snapshot.passed is True


@pytest.mark.parametrize(
    ("kwargs", "failing_field"),
    [
        ({"clock_samples": (1_000_000_000,)}, "clock_increasing"),
        ({"clock_samples": (1_000_000_000, 1_000_000_000)}, "clock_increasing"),
        (
            {
                "baseline": {
                    name: value for name, value in BASELINE.items() if name != "joint6"
                }
            },
            "isaac_state_complete",
        ),
        ({"controllers_active": False}, "controllers_active"),
    ],
)
def test_readiness_snapshot_flags_each_failed_component(kwargs, failing_field):
    world = make_world()
    prime_ready(world, **kwargs)
    snapshot = world.runtime.readiness_snapshot()
    assert getattr(snapshot, failing_field) is False
    assert snapshot.passed is False


def test_readiness_snapshot_requires_the_broadcaster_state():
    world = make_world()
    for nanos in (1_000_000_000, 2_000_000_000):
        publish(world.node, TOPIC_CLOCK, FakeClockMsg(nanos))
    publish(
        world.node,
        TOPIC_ISAAC_JOINT_STATES,
        FakeJointStateMsg(BASELINE, velocity=[0.0] * len(JOINT_NAMES)),
    )
    world.controllers.response = SimpleNamespace(
        controller=[
            SimpleNamespace(name=name, state="active") for name in REQUIRED_CONTROLLERS
        ]
    )
    snapshot = world.runtime.readiness_snapshot()
    assert snapshot.joint_state_complete is False
    assert snapshot.broadcaster_match is False
    assert snapshot.isaac_state_complete is True
    assert snapshot.passed is False


def test_readiness_snapshot_demands_the_exact_broadcaster_match():
    world = make_world()
    drifted = dict(BASELINE)
    drifted["joint1"] += 1.0e-3
    prime_ready(world, broadcaster=drifted)
    snapshot = world.runtime.readiness_snapshot()
    assert snapshot.isaac_state_complete is True
    assert snapshot.joint_state_complete is True
    assert snapshot.broadcaster_match is False
    assert snapshot.passed is False


@pytest.mark.parametrize(
    ("attribute", "failing_field"),
    [
        ("arm", "arm_action_ready"),
        ("move_group", "move_group_action_ready"),
    ],
)
def test_readiness_snapshot_requires_the_action_servers(attribute, failing_field):
    world = make_world()
    prime_ready(world)
    getattr(world, attribute).server_ready = False
    snapshot = world.runtime.readiness_snapshot()
    assert getattr(snapshot, failing_field) is False
    assert snapshot.passed is False


def test_readiness_snapshot_requires_the_compute_fk_service():
    world = make_world()
    prime_ready(world)
    world.fk.ready = False
    snapshot = world.runtime.readiness_snapshot()
    assert snapshot.compute_fk_ready is False
    assert snapshot.passed is False


def test_monitoring_getters_expose_the_latest_samples():
    world = make_world()
    assert world.runtime.latest_isaac_state() is None
    assert world.runtime.latest_joint_state() is None
    assert world.runtime.clock_values() == ()
    assert world.runtime.controller_states() == {}
    prime_ready(world)
    assert world.runtime.clock_values() == (1_000_000_000, 2_000_000_000)
    assert world.runtime.latest_isaac_state() == BASELINE
    assert world.runtime.latest_joint_state() == BASELINE
    # The earlier poll never completed; past the service bound it is
    # abandoned and re-issued instead of wedging readiness forever.
    world.clock.advance(SERVICE_CALL_TIMEOUT_S + 0.1)
    assert world.runtime.controller_states() == {
        name: "active" for name in REQUIRED_CONTROLLERS
    }
    assert world.controllers.requests  # the list_controllers service was polled


def test_invalid_state_messages_are_ignored_not_faked():
    world = make_world()
    prime_ready(world)
    publish(world.node, TOPIC_ISAAC_JOINT_STATES, FakeJointStateMsg({"joint1": 9.9}))
    publish(
        world.node,
        TOPIC_ISAAC_JOINT_STATES,
        FakeJointStateMsg({**BASELINE, "joint2": float("nan")}),
    )
    assert world.runtime.latest_isaac_state() == BASELINE
    assert world.runtime.readiness_snapshot().isaac_state_complete is True


def test_wait_until_ready_passes_without_sending_motion():
    world = make_world()
    prime_ready(world)
    assert world.runtime.wait_until_ready(5.0) is True
    assert world.move_group.goals == []
    assert world.arm.goals == []


def test_wait_until_ready_is_bounded_when_never_ready():
    world = make_world()
    assert world.runtime.wait_until_ready(0.5) is False
    assert world.clock.now <= 0.75
    assert world.move_group.goals == []
    assert world.arm.goals == []


def test_wait_until_ready_rejects_negative_timeouts():
    world = make_world()
    with pytest.raises(ValueError):
        world.runtime.wait_until_ready(-1.0)


# ------------------------------------------------------- send: goal construction


def test_send_joint_builds_the_exact_move_group_goal(joint_target):
    world = make_world()
    prime_ready(world)
    script_success(
        world,
        action_result=trajectory_action_result(joint_target),
        final_state=joint_final_state(joint_target),
    )
    assert world.runtime.send_joint(joint_target, 20.0).status is ExecutionStatus.SUCCEEDED
    assert len(world.move_group.goals) == 1
    goal = world.move_group.goals[0]
    request = goal.request
    assert request.group_name == "arm"
    assert goal.planning_options.plan_only is False
    assert goal.planning_options.replan is False
    assert request.allowed_planning_time == 5.0
    assert request.num_planning_attempts == 10
    assert request.max_velocity_scaling_factor == 0.1
    assert request.max_acceleration_scaling_factor == 0.1
    assert len(request.goal_constraints) == 1
    constraints = request.goal_constraints[0]
    assert constraints.position_constraints == []
    assert constraints.orientation_constraints == []
    joint_constraints = constraints.joint_constraints
    assert [item.joint_name for item in joint_constraints] == list(ARM_JOINT_NAMES)
    for constraint in joint_constraints:
        assert constraint.position == pytest.approx(joint_target.positions[constraint.joint_name])
        assert constraint.tolerance_above == joint_target.goal_tolerance
        assert constraint.tolerance_below == joint_target.goal_tolerance
        assert constraint.weight == 1.0


def test_send_joint_reports_the_full_success_evidence(joint_target):
    world = make_world()
    prime_ready(world)
    script_success(
        world,
        action_result=trajectory_action_result(joint_target),
        final_state=joint_final_state(joint_target),
    )
    result = world.runtime.send_joint(joint_target, 20.0)
    assert result.status is ExecutionStatus.SUCCEEDED
    assert result.moveit_error_code == 1
    assert result.planned_waypoints == 3
    assert result.command_messages == 1
    assert is_success(result) is True
    final = joint_final_state(joint_target)
    expected = {
        name: abs(final[name] - joint_target.positions[name]) for name in ARM_JOINT_NAMES
    }
    assert result.final_errors == pytest.approx(expected, abs=1e-12)
    assert result.details["max_joint_error"] == pytest.approx(0.005)
    assert result.details["gripper_drift"] == 0.0
    assert result.details["post_state_messages"] == 2


def test_send_joint_rejects_inadmissible_targets_before_any_send(joint_target):
    world = make_world()
    invalid = JointTarget(positions={**joint_target.positions, "joint1": 9.0})
    result = world.runtime.send_joint(invalid, 5.0)
    assert result.status is ExecutionStatus.REJECTED
    assert result.command_messages == 0
    assert result.details["reason"]
    assert world.move_group.goals == []


def test_send_joint_requires_a_positive_finite_timeout(joint_target):
    world = make_world()
    for timeout in (0.0, -1.0, float("nan")):
        result = world.runtime.send_joint(joint_target, timeout)
        assert result.status is ExecutionStatus.REJECTED
        assert result.command_messages == 0
    assert world.move_group.goals == []


def test_send_joint_fails_without_registered_clients(joint_target):
    node = FakeNode()
    runtime = RosMoveItRuntime(
        node, factories=fake_factories(), clock=FakeClock(), sleep=lambda _dt: None
    )
    result = runtime.send_joint(joint_target, 5.0)
    assert result.status is ExecutionStatus.FAILED
    assert "register_clients" in result.details["error"]


def test_send_joint_refuses_motion_before_readiness(joint_target):
    world = make_world()
    result = world.runtime.send_joint(joint_target, 5.0)
    assert result.status is ExecutionStatus.FAILED
    assert "readiness" in result.details["error"]
    assert world.move_group.goals == []


def test_send_joint_timeout_cancels_and_never_reports_success(joint_target):
    world = make_world()
    prime_ready(world)
    handle = FakeGoalHandle()
    world.events.append(lambda: world.move_group.pending.set(handle))
    result = world.runtime.send_joint(joint_target, 0.5)
    assert result.status is ExecutionStatus.TIMED_OUT
    assert handle.cancel_calls == 1
    assert result.details["cancel"]["attempted"] is True
    assert result.details["cancel"]["return_code"] == 0


def test_send_joint_times_out_when_the_goal_response_never_arrives(joint_target):
    world = make_world()
    prime_ready(world)
    result = world.runtime.send_joint(joint_target, 0.5)
    assert result.status is ExecutionStatus.TIMED_OUT
    assert result.details["phase"] == "goal_acceptance"


def test_send_joint_rejects_goals_refused_by_move_group(joint_target):
    world = make_world()
    prime_ready(world)
    handle = FakeGoalHandle(accepted=False)
    world.events.append(lambda: world.move_group.pending.set(handle))
    result = world.runtime.send_joint(joint_target, 5.0)
    assert result.status is ExecutionStatus.REJECTED
    assert result.details["accepted"] is False


def test_send_joint_fails_on_an_aborted_terminal_status(joint_target):
    world = make_world()
    prime_ready(world)
    script_terminal(
        world,
        status=FAKE_GOAL_STATUS.STATUS_ABORTED,
        action_result=trajectory_action_result(joint_target),
    )
    result = world.runtime.send_joint(joint_target, 20.0)
    assert result.status is ExecutionStatus.FAILED
    assert is_success(result) is False
    assert "terminal status" in result.details["error"]


def test_send_joint_fails_when_waypoints_leave_the_limits(joint_target):
    world = make_world()
    prime_ready(world)
    outside = [3.0] + arm_values(BASELINE)[1:]
    script_success(
        world,
        action_result=trajectory_action_result(joint_target, points=[arm_values(BASELINE), outside]),
        final_state=joint_final_state(joint_target),
    )
    result = world.runtime.send_joint(joint_target, 20.0)
    assert result.status is ExecutionStatus.FAILED
    assert is_success(result) is False
    assert result.details["trajectory_accepted"] is False
    assert result.moveit_error_code == 1


def test_send_joint_fails_on_an_empty_planned_trajectory(joint_target):
    world = make_world()
    prime_ready(world)
    script_success(
        world,
        action_result=trajectory_action_result(joint_target, points=[]),
        final_state=joint_final_state(joint_target),
    )
    result = world.runtime.send_joint(joint_target, 20.0)
    assert result.status is ExecutionStatus.FAILED
    assert result.planned_waypoints == 0
    assert result.details["trajectory_accepted"] is False


def test_send_joint_never_fakes_missing_final_feedback(joint_target):
    world = make_world()
    prime_ready(world)
    script_success(
        world,
        action_result=trajectory_action_result(joint_target),
        final_state=joint_final_state(joint_target),
        post_states=0,
        commands=0,
    )
    result = world.runtime.send_joint(joint_target, 20.0)
    assert result.status is ExecutionStatus.FAILED
    assert is_success(result) is False
    assert result.details["post_state_messages"] == 0
    assert "insufficient post-command evidence" in result.details["error"]


def test_send_joint_fails_when_the_final_state_misses_the_target(joint_target):
    world = make_world()
    prime_ready(world)
    bad_final = {**BASELINE, "joint1": joint_target.positions["joint1"] + 0.2}
    script_success(
        world,
        action_result=trajectory_action_result(joint_target),
        final_state=bad_final,
    )
    result = world.runtime.send_joint(joint_target, 20.0)
    assert result.status is ExecutionStatus.FAILED
    assert is_success(result) is False
    assert result.details["max_joint_error"] == pytest.approx(0.2)
    assert result.final_errors["joint1"] == pytest.approx(0.2)


# --------------------------------------------------------- send: pose execution


def test_send_link6_pose_builds_the_exact_pose_goal(pose_target):
    calls: list = []

    def fk_verifier(state):
        calls.append(dict(state))
        return (tuple(pose_target.position), tuple(pose_target.orientation_xyzw))

    world = make_world(fk_verifier=fk_verifier)
    prime_ready(world)
    script_success(
        world, action_result=trajectory_action_result(), final_state=BASELINE
    )
    result = world.runtime.send_link6_pose(pose_target, 20.0)
    assert result.status is ExecutionStatus.SUCCEEDED
    goal = world.move_group.goals[0]
    request = goal.request
    assert request.group_name == "arm"
    assert goal.planning_options.plan_only is False
    assert goal.planning_options.replan is False
    assert len(request.goal_constraints) == 1
    constraints = request.goal_constraints[0]
    assert constraints.joint_constraints == []
    assert len(constraints.position_constraints) == 1
    position = constraints.position_constraints[0]
    assert position.header.frame_id == pose_target.frame_id
    assert position.link_name == pose_target.tip_link
    sphere = position.constraint_region.primitives[0]
    assert sphere.type == FakeSolidPrimitive.SPHERE
    assert sphere.dimensions == [pose_target.position_tolerance_m]
    center = position.constraint_region.primitive_poses[0]
    assert (
        center.position.x,
        center.position.y,
        center.position.z,
    ) == pytest.approx(tuple(pose_target.position))
    assert center.orientation.w == 1.0
    assert len(constraints.orientation_constraints) == 1
    orientation = constraints.orientation_constraints[0]
    assert orientation.header.frame_id == pose_target.frame_id
    assert orientation.link_name == pose_target.tip_link
    assert (
        orientation.orientation.x,
        orientation.orientation.y,
        orientation.orientation.z,
        orientation.orientation.w,
    ) == pytest.approx(tuple(pose_target.orientation_xyzw))
    assert (
        orientation.absolute_x_axis_tolerance,
        orientation.absolute_y_axis_tolerance,
        orientation.absolute_z_axis_tolerance,
    ) == (
        pose_target.orientation_tolerance_rad,
        pose_target.orientation_tolerance_rad,
        pose_target.orientation_tolerance_rad,
    )


def test_send_link6_pose_verifies_through_the_injected_fk_hook_only(pose_target):
    calls: list = []

    def fk_verifier(state):
        calls.append(dict(state))
        return (tuple(pose_target.position), tuple(pose_target.orientation_xyzw))

    world = make_world(fk_verifier=fk_verifier)
    prime_ready(world)
    script_success(
        world, action_result=trajectory_action_result(), final_state=BASELINE
    )
    result = world.runtime.send_link6_pose(pose_target, 20.0)
    assert result.status is ExecutionStatus.SUCCEEDED
    assert is_success(result) is True
    assert calls == [BASELINE]
    assert result.final_errors["position_error_m"] == pytest.approx(0.0, abs=1e-12)
    assert result.final_errors["orientation_error_rad"] == pytest.approx(0.0, abs=1e-12)
    assert result.command_messages == 1


def test_send_link6_pose_fails_when_fk_disagrees(pose_target):
    def fk_verifier(_state):
        shifted = (pose_target.position[0] + 0.05,) + tuple(pose_target.position[1:])
        return (shifted, tuple(pose_target.orientation_xyzw))

    world = make_world(fk_verifier=fk_verifier)
    prime_ready(world)
    script_success(
        world, action_result=trajectory_action_result(), final_state=BASELINE
    )
    result = world.runtime.send_link6_pose(pose_target, 20.0)
    assert result.status is ExecutionStatus.FAILED
    assert is_success(result) is False
    assert result.final_errors["position_error_m"] == pytest.approx(0.05)
    assert result.final_errors["orientation_error_rad"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("hook", [None, lambda _state: None])
def test_send_link6_pose_never_fakes_unavailable_fk_verification(pose_target, hook):
    world = make_world(fk_verifier=hook)
    prime_ready(world)
    script_success(
        world, action_result=trajectory_action_result(), final_state=BASELINE
    )
    result = world.runtime.send_link6_pose(pose_target, 20.0)
    assert result.status is ExecutionStatus.FAILED
    assert is_success(result) is False
    assert "fk verification" in result.details["error"]


def test_send_link6_pose_fails_when_the_fk_hook_raises(pose_target):
    def fk_verifier(_state):
        raise RuntimeError("fk boom")

    world = make_world(fk_verifier=fk_verifier)
    prime_ready(world)
    script_success(
        world, action_result=trajectory_action_result(), final_state=BASELINE
    )
    result = world.runtime.send_link6_pose(pose_target, 20.0)
    assert result.status is ExecutionStatus.FAILED
    assert "fk boom" in result.details["error"]


def test_send_link6_pose_rejects_inadmissible_targets_before_any_send(pose_target):
    world = make_world()
    invalid = Link6PoseTarget(
        frame_id="world",
        position=(0.1, 0.0, 0.2),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    result = world.runtime.send_link6_pose(invalid, 5.0)
    assert result.status is ExecutionStatus.REJECTED
    assert result.command_messages == 0
    assert world.move_group.goals == []


# ------------------------------------------------- send: gripper execution


def gripper_target(
    *,
    opening_m: float = 0.02,
    duration_s: float = 1.0,
    goal_tolerance: float = 0.005,
    expected_opening_m: tuple[float, float] = (0.0, 0.03),
) -> GripperTarget:
    return GripperTarget(
        opening_m=opening_m,
        duration_s=duration_s,
        goal_tolerance=goal_tolerance,
        expected_opening_m=expected_opening_m,
    )


def script_gripper_terminal(
    world: World, *, status: int, final_state=None
) -> FakeGoalHandle:
    handle = FakeGoalHandle()
    world.events.append(lambda: world.gripper.pending.set(handle))
    world.events.append(
        lambda: handle.result_future.set(
            SimpleNamespace(status=status, result=SimpleNamespace())
        )
    )
    if final_state is not None:
        world.events.append(
            lambda: publish(
                world.node, TOPIC_ISAAC_JOINT_STATES, FakeJointStateMsg(final_state)
            )
        )
    return handle


def test_send_gripper_builds_the_exact_two_point_goal():
    world = make_world()
    prime_ready(world)
    script_gripper_terminal(
        world,
        status=FAKE_GOAL_STATUS.STATUS_SUCCEEDED,
        final_state={**BASELINE, "gripper_joint1": 0.01, "gripper_joint2": -0.01},
    )
    target = gripper_target()
    result = world.runtime.send_gripper(target, 20.0)
    assert result.status is ExecutionStatus.SUCCEEDED
    assert result.goal_accepted is True
    assert result.terminal is True and result.safe_to_release is True
    assert result.action_status == "SUCCEEDED"
    assert result.details["post_result_state_messages"] >= 1
    assert len(world.gripper.goals) == 1
    goal = world.gripper.goals[0]
    assert goal.trajectory.joint_names == list(GRIPPER_JOINT_NAMES)
    points = goal.trajectory.points
    assert len(points) == 2
    # Point 0 is the measured state at t=0, point 1 the logical opening
    # mapped onto the symmetric finger positions after duration_s.
    assert points[0].positions == [
        BASELINE["gripper_joint1"],
        BASELINE["gripper_joint2"],
    ]
    assert points[0].time_from_start.sec == 0
    assert points[0].time_from_start.nanosec == 0
    assert points[1].positions == [0.01, -0.01]
    assert points[1].time_from_start.sec == 1
    assert [(item.name, item.position) for item in goal.goal_tolerance] == [
        (name, target.goal_tolerance) for name in GRIPPER_JOINT_NAMES
    ]


def test_send_gripper_rejects_an_aborted_contact_stall_within_window():
    """An in-window opening cannot override an ABORTED action terminal."""
    world = make_world()
    prime_ready(world)
    script_gripper_terminal(
        world,
        status=FAKE_GOAL_STATUS.STATUS_ABORTED,
        final_state={**BASELINE, "gripper_joint1": 0.01, "gripper_joint2": -0.01},
    )
    result = world.runtime.send_gripper(gripper_target(), 20.0)
    assert result.status is ExecutionStatus.FAILED
    assert is_success(result) is False
    assert result.terminal is True and result.safe_to_release is True
    assert result.action_status == "ABORTED"
    assert result.details["action_status"] == "ABORTED"
    assert result.details["measured_opening_m"] == pytest.approx(0.02)
    assert result.details["commanded_opening_m"] == pytest.approx(0.02)
    assert result.details["expected_opening_m"] == [0.0, 0.03]
    assert result.final_errors["gripper_opening_m"] == pytest.approx(0.0, abs=1e-12)
    assert result.planned_waypoints == 2
    assert result.command_messages == 1


def test_send_gripper_requires_post_action_finger_sample():
    """A pre-command opening in range is not final gripper feedback."""
    world = make_world()
    prime_ready(world)
    script_gripper_terminal(world, status=FAKE_GOAL_STATUS.STATUS_SUCCEEDED)
    result = world.runtime.send_gripper(gripper_target(), 20.0)
    assert result.status is ExecutionStatus.FAILED
    assert result.goal_accepted is True
    assert result.terminal is True and result.safe_to_release is True
    assert result.details["post_result_state_messages"] == 0


def test_send_gripper_fails_when_the_measurement_leaves_the_window():
    world = make_world()
    prime_ready(world)
    script_gripper_terminal(
        world,
        status=FAKE_GOAL_STATUS.STATUS_SUCCEEDED,
        final_state={**BASELINE, "gripper_joint1": 0.045, "gripper_joint2": -0.045},
    )
    result = world.runtime.send_gripper(gripper_target(), 20.0)
    assert result.status is ExecutionStatus.FAILED
    assert is_success(result) is False
    assert result.details["measured_opening_m"] == pytest.approx(0.09)
    assert result.final_errors["gripper_opening_m"] == pytest.approx(0.07)
    assert "outside the expected opening window" in result.details["error"]


def test_send_gripper_fails_on_a_mimic_violation_without_faking():
    world = make_world()
    prime_ready(world)
    script_gripper_terminal(
        world,
        status=FAKE_GOAL_STATUS.STATUS_SUCCEEDED,
        final_state={**BASELINE, "gripper_joint1": 0.01},
    )
    result = world.runtime.send_gripper(gripper_target(), 20.0)
    assert result.status is ExecutionStatus.FAILED
    assert is_success(result) is False
    assert result.details["measured_opening_m"] is None
    assert result.final_errors["gripper_opening_m"] is None
    assert "mimic relation" in result.details["measured_opening_error"]
    assert "refusing to claim success" in result.details["error"]


def test_send_gripper_times_out_cancels_and_never_succeeds():
    world = make_world()
    prime_ready(world)
    handle = FakeGoalHandle()
    world.events.append(lambda: world.gripper.pending.set(handle))
    result = world.runtime.send_gripper(gripper_target(), 0.5)
    assert result.status is ExecutionStatus.TIMED_OUT
    assert result.details["phase"] == "execution"
    assert handle.cancel_calls == 1
    assert result.details["cancel"]["attempted"] is True


def test_send_gripper_rejects_inadmissible_targets_before_any_send():
    world = make_world()
    prime_ready(world)
    invalid = gripper_target(opening_m=0.11)
    result = world.runtime.send_gripper(invalid, 5.0)
    assert result.status is ExecutionStatus.REJECTED
    assert result.command_messages == 0
    assert world.gripper.goals == []


def test_send_gripper_refuses_motion_before_readiness():
    world = make_world()
    result = world.runtime.send_gripper(gripper_target(), 5.0)
    assert result.status is ExecutionStatus.FAILED
    assert "readiness" in result.details["error"]
    assert world.gripper.goals == []


def test_send_gripper_fails_without_a_start_state(monkeypatch):
    world = make_world()
    prime_ready(world)
    # Readiness holds, but no finger state is observable at goal-build time.
    monkeypatch.setattr(
        world.runtime,
        "readiness_snapshot",
        lambda: ReadinessSnapshot(**dict.fromkeys(SNAPSHOT_FIELDS, True)),
    )
    monkeypatch.setattr(world.runtime, "latest_isaac_state", lambda: None)
    script_gripper_terminal(world, status=FAKE_GOAL_STATUS.STATUS_SUCCEEDED)
    result = world.runtime.send_gripper(gripper_target(), 5.0)
    assert result.status is ExecutionStatus.FAILED
    assert "start point cannot be built" in result.details["error"]
    assert world.gripper.goals == []


# ------------------------------------------------------------ scene state


def test_query_scene_state_returns_the_parsed_document():
    world = make_world()
    document = {
        "schema": SCENE_STATE_SCHEMA,
        "sim_time_s": 3.25,
        "bricks": {"brick-44": {"aabb_min": [0.0, 0.0, 0.0]}},
        "connections": {"brick-44|brick-45": True},
    }
    world.scene_state.response = SimpleNamespace(message=json.dumps(document))
    assert world.runtime.query_scene_state(5.0) == document
    assert len(world.scene_state.requests) == 1


def test_query_scene_state_rejects_a_wrong_schema():
    world = make_world()
    world.scene_state.response = SimpleNamespace(
        message=json.dumps({"schema": "other-scene/v9"})
    )
    with pytest.raises(RuntimeError, match="schema"):
        world.runtime.query_scene_state(5.0)


def test_query_scene_state_rejects_a_non_json_message():
    world = make_world()
    world.scene_state.response = SimpleNamespace(message="definitely not json")
    with pytest.raises(RuntimeError, match="JSON"):
        world.runtime.query_scene_state(5.0)


def test_query_scene_state_raises_when_the_client_is_unavailable():
    world = make_world()
    world.scene_state.ready = False
    with pytest.raises(RuntimeError, match="not available"):
        world.runtime.query_scene_state(5.0)


def test_query_scene_state_raises_when_the_service_never_answers():
    world = make_world()
    with pytest.raises(RuntimeError, match="did not answer"):
        world.runtime.query_scene_state(0.5)


# ------------------------------------------------------------------- executor


class FakeRuntime:
    def __init__(self) -> None:
        self.joint_calls: list = []
        self.pose_calls: list = []
        self.joint_result = ExecutionResult(
            status=ExecutionStatus.SUCCEEDED,
            moveit_error_code=1,
            planned_waypoints=4,
            command_messages=2,
        )
        self.pose_result = ExecutionResult(
            status=ExecutionStatus.SUCCEEDED,
            moveit_error_code=1,
            planned_waypoints=5,
            command_messages=3,
        )

    def send_joint(self, target, timeout_s) -> ExecutionResult:
        self.joint_calls.append((target, timeout_s))
        return self.joint_result

    def send_link6_pose(self, target, timeout_s) -> ExecutionResult:
        self.pose_calls.append((target, timeout_s))
        return self.pose_result


def test_executor_rejects_invalid_joint_targets_without_runtime_send(joint_target):
    fake = FakeRuntime()
    invalid = JointTarget(positions={**joint_target.positions, "joint1": 9.0})
    result = RosMoveItExecutor(fake).execute_joint(invalid, timeout_s=5.0)
    assert result.status is ExecutionStatus.REJECTED
    assert result.command_messages == 0
    assert fake.joint_calls == []
    assert result.details["reason"]
    assert result.details["admission"]["excess"]["joint1"] == pytest.approx(
        9.0 - 2.6179938
    )


def test_executor_rejects_invalid_pose_targets_without_runtime_send():
    fake = FakeRuntime()
    invalid = Link6PoseTarget(
        frame_id="world",
        position=(0.1, 0.0, 0.2),
        orientation_xyzw=(2.0, 0.0, 0.0, 2.0),
    )
    result = RosMoveItExecutor(fake).execute_link6_pose(invalid, timeout_s=5.0)
    assert result.status is ExecutionStatus.REJECTED
    assert result.command_messages == 0
    assert fake.pose_calls == []
    assert "frame_id" in result.details["reason"]


def test_executor_delegates_valid_targets_exactly_once(joint_target, pose_target):
    fake = FakeRuntime()
    executor = RosMoveItExecutor(fake)
    assert isinstance(executor, MoveItExecutor)
    assert executor.execute_joint(joint_target, timeout_s=7.5) is fake.joint_result
    assert executor.execute_link6_pose(pose_target, timeout_s=3.0) is fake.pose_result
    assert fake.joint_calls == [(joint_target, 7.5)]
    assert fake.pose_calls == [(pose_target, 3.0)]


# ------------------------------------------------------------------- lifecycle


def test_cancel_active_without_an_active_goal_is_a_noop():
    world = make_world()
    assert world.runtime.cancel_active() == {"attempted": False}


def test_shutdown_destroys_every_registered_handle():
    world = make_world()
    world.runtime.shutdown()
    assert sorted(world.node.destroyed_subscriptions) == sorted(
        [
            TOPIC_CLOCK,
            TOPIC_ISAAC_JOINT_STATES,
            TOPIC_JOINT_STATES,
            TOPIC_ISAAC_JOINT_COMMAND,
        ]
    )
    assert sorted(world.node.destroyed_clients) == sorted(
        [
            SERVICE_LIST_CONTROLLERS,
            SERVICE_PLAN,
            SERVICE_COMPUTE_FK,
            SERVICE_SCENE_STATE,
        ]
    )
    assert world.arm.destroyed is True
    assert world.move_group.destroyed is True
    assert world.gripper.destroyed is True
    assert world.runtime.shutdown_errors == []
    snapshot = world.runtime.readiness_snapshot()
    assert snapshot.passed is False
    assert snapshot.arm_action_ready is False
    assert snapshot.move_group_action_ready is False
