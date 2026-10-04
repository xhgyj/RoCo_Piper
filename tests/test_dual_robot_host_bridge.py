"""Offline, injectable tests for the dual/multi-robot P0.5 host bridge.

These tests never boot Kit, Isaac, rclpy, or a container. They import the
bridge module by file path (the same pattern as tests/test_host_bridge_p05.py)
and exercise the multi-robot paths with stubs only:

* the endpoint contract of every validated robot matches the frozen A3
  ``RobotRuntimeProfile.from_robot_id`` namespacing: one shared ``/clock``,
  ``/<robot_id>/isaac_joint_states``, ``/<robot_id>/isaac_joint_command`` and
  the ``<robot_id>_<joint>`` joint names; nothing is defaulted to piper_0;
* robot ids / prim paths / joint sets come from the validated inputs and every
  missing, duplicated, nested or mismatched robot fails closed;
* command routing is per robot: a command published on one robot's topic can
  never drive another robot's articulation, and clamping is per joint;
* a fake rclpy module proves the wiring creates exactly one node with one
  publisher/subscription pair per robot, no command publisher, and that every
  robot's state is published (observable even when it has no operation);
* the scene-state document carries one self-identifying ``robots[robot_id]``
  observation block per robot next to the shared bricks/connections, with no
  cross-robot fallback, while the single-robot/legacy top-level block stays;
* the single-robot snapshot binding is never guessed in a multi-robot scene.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
BRIDGE_PATH = ROOT / "isaac" / "p05_host_bridge.py"

DOF_NAMES = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "gripper_joint1",
    "gripper_joint2",
)
ROBOT_A = "piper_0"
ROBOT_B = "piper_1"
PRIM_A = "/World/Robot_0"
PRIM_B = "/World/Robot_1"
SERVICE_SCENE_STATE = "/p05_host_bridge/scene_state"
SCHEMA = "rocobrick-scene-state/v1"

BRICK_ID_A = "brick-768de5a17827920c48d4"
BRICK_ID_B = "brick-9f3c2216c25b04ba942a"
PAIR_KEY = f"{BRICK_ID_A}|{BRICK_ID_B}"
AABBS_A = ((0.1, -0.05, 0.0), (0.14, 0.03, 0.0112))
AABBS_B = ((0.1, -0.05, 0.0112), (0.14, 0.03, 0.0224))

FINGERS_A = {"gripper_joint1": 0.01, "gripper_joint2": -0.01}
FINGERS_B = {"gripper_joint1": 0.02, "gripper_joint2": -0.02}


@pytest.fixture(scope="module")
def bridge_module():
    """Import the bridge module without booting isaacsim, omni, or rclpy.

    Returns:
        The imported bridge module.
    """
    spec = importlib.util.spec_from_file_location("p05_host_bridge_dual", BRIDGE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bridge_source() -> str:
    """Return the raw bridge script text.

    Returns:
        The bridge script source.
    """
    return BRIDGE_PATH.read_text(encoding="utf-8")


@pytest.fixture
def fake_action_module(monkeypatch):
    """Install a recording ArticulationAction type for offline command routing.

    ``RobotChannel.on_joint_command`` imports
    ``isaacsim.core.utils.types.ArticulationAction`` at call time; the fake
    module keeps the production import untouched while recording actions.

    Returns:
        The recording ArticulationAction class.
    """

    class ArticulationAction:
        def __init__(self, joint_positions=None, joint_indices=None):
            self.joint_positions = tuple(joint_positions or ())
            self.joint_indices = tuple(joint_indices or ())

    module = types.ModuleType("isaacsim.core.utils.types")
    module.ArticulationAction = ArticulationAction
    monkeypatch.setitem(sys.modules, "isaacsim.core.utils.types", module)
    return ArticulationAction


class FakeArticulation:
    """Minimal articulation stub: positions, velocities and applied actions."""

    def __init__(self, positions, initialized=True):
        self.positions = [float(value) for value in positions]
        self.velocities = [0.0] * len(self.positions)
        self.actions: list = []
        self._initialized = initialized

    def get_joint_positions(self):
        return list(self.positions)

    def get_joint_velocities(self):
        return list(self.velocities)

    def apply_action(self, action):
        self.actions.append(action)

    def is_initialized(self):
        return self._initialized


class FakeHeader:
    def __init__(self):
        self.stamp = types.SimpleNamespace(sec=0, nanosec=0)


class FakeJointState:
    """sensor_msgs/JointState stub used by the fake rclpy module."""

    def __init__(self):
        self.name: list = []
        self.position: list = []
        self.velocity: list = []
        self.header = FakeHeader()


class FakeClock:
    """rosgraph_msgs/Clock stub."""

    def __init__(self):
        self.clock = types.SimpleNamespace(sec=0, nanosec=0)


class FakeTrigger:
    """std_srvs/srv/Trigger stub type (only the type is needed for wiring)."""


class FakePublisher:
    """Publisher stub recording every published message per topic."""

    def __init__(self, topic):
        self.topic = topic
        self.messages: list = []

    def publish(self, message):
        self.messages.append(message)


class FakeNode:
    """rclpy node stub recording publishers, subscriptions and services."""

    def __init__(self, name):
        self.name = name
        self.publishers: list = []
        self.subscriptions: list = []
        self.services: list = []
        self.destroyed = False

    def create_publisher(self, message_type, topic, qos):
        publisher = FakePublisher(topic)
        self.publishers.append((message_type, topic, publisher))
        return publisher

    def create_subscription(self, message_type, topic, callback, qos):
        self.subscriptions.append((message_type, topic, callback))
        return types.SimpleNamespace(topic=topic)

    def create_service(self, service_type, name, callback):
        self.services.append((service_type, name, callback))
        return types.SimpleNamespace(name=name)

    def destroy_node(self):
        self.destroyed = True

    def publisher_for(self, topic):
        for _message_type, wired_topic, publisher in self.publishers:
            if wired_topic == topic:
                return publisher
        raise AssertionError(f"no publisher wired for {topic}")

    def subscriber_for(self, topic):
        for _message_type, wired_topic, callback in self.subscriptions:
            if wired_topic == topic:
                return callback
        raise AssertionError(f"no subscription wired for {topic}")


@pytest.fixture
def fake_rclpy(monkeypatch):
    """Install a fake rclpy/rosgraph_msgs/sensor_msgs/std_srvs import tree.

    Returns:
        The (rclpy module, list of created nodes) pair.
    """
    rclpy = types.ModuleType("rclpy")
    nodes: list[FakeNode] = []

    def create_node(name):
        node = FakeNode(name)
        nodes.append(node)
        return node

    rclpy.init = lambda **kwargs: None
    rclpy_signals = types.ModuleType("rclpy.signals")
    rclpy_signals.SignalHandlerOptions = types.SimpleNamespace(NO="no")
    rclpy.create_node = create_node
    rclpy.spin_once = lambda node, timeout_sec=0.0: None
    rclpy.shutdown = lambda: None
    rclpy.ok = lambda: True

    rosgraph_msgs = types.ModuleType("rosgraph_msgs")
    rosgraph_msgs_msg = types.ModuleType("rosgraph_msgs.msg")
    rosgraph_msgs_msg.Clock = FakeClock
    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msgs_msg.JointState = FakeJointState
    std_srvs = types.ModuleType("std_srvs")
    std_srvs_srv = types.ModuleType("std_srvs.srv")
    std_srvs_srv.Trigger = FakeTrigger

    for name, module in (
        ("rclpy", rclpy),
        ("rclpy.signals", rclpy_signals),
        ("rosgraph_msgs", rosgraph_msgs),
        ("rosgraph_msgs.msg", rosgraph_msgs_msg),
        ("sensor_msgs", sensor_msgs),
        ("sensor_msgs.msg", sensor_msgs_msg),
        ("std_srvs", std_srvs),
        ("std_srvs.srv", std_srvs_srv),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return rclpy, nodes


def _spec(prim_path, joints=DOF_NAMES):
    """Build one validated robot spec stub (only the read fields matter).

    Args:
        prim_path: The absolute robot prim path.
        joints: The configured joint order.

    Returns:
        The spec stub.
    """
    return types.SimpleNamespace(prim_path=prim_path, arm_joint_names=tuple(joints))


def _robots():
    """Return the two-robot validated inputs mapping.

    Returns:
        {piper_0: spec, piper_1: spec}.
    """
    return {ROBOT_A: _spec(PRIM_A), ROBOT_B: _spec(PRIM_B)}


def _channels(bridge_module, positions_a=None, positions_b=None):
    """Build two real RobotChannels bound to fake articulations.

    Args:
        bridge_module: The imported bridge module.
        positions_a: Optional piper_0 positions (8 values).
        positions_b: Optional piper_1 positions (8 values).

    Returns:
        The (channels tuple, {robot_id: FakeArticulation}) pair.
    """
    defaults_a = [0.0] * 6 + [0.01, -0.01]
    defaults_b = [0.0] * 6 + [0.02, -0.02]
    articulations = {
        ROBOT_A: FakeArticulation(
            positions_a if positions_a is not None else defaults_a
        ),
        ROBOT_B: FakeArticulation(
            positions_b if positions_b is not None else defaults_b
        ),
    }
    endpoints = bridge_module.build_robot_endpoints(_robots())
    channels = tuple(
        bridge_module.RobotChannel(
            endpoint, articulations[endpoint.robot_id], DOF_NAMES
        )
        for endpoint in endpoints
    )
    return channels, articulations


def _command_message(names, positions):
    """Build a JointState-like command message.

    Args:
        names: Joint names.
        positions: Target positions.

    Returns:
        The message stub.
    """
    return types.SimpleNamespace(name=list(names), position=list(positions))


# --------------------------------------------------------------- endpoints


def test_module_import_is_offline_safe(bridge_module) -> None:
    """Importing the module must not boot isaacsim, omni, rclpy, or pxr."""
    assert bridge_module.SCHEMA == "rocobrick-host-bridge-smoke/v1"
    for heavy in ("isaacsim", "omni", "rclpy", "pxr"):
        assert heavy not in sys.modules, heavy


def test_endpoints_match_frozen_a3_profile_namespacing(bridge_module) -> None:
    """Every validated robot gets the A3 profile's namespaced endpoints."""
    from rocobrick.robot_runtime_profile import RobotRuntimeProfile

    endpoints = bridge_module.build_robot_endpoints(_robots())
    assert [endpoint.robot_id for endpoint in endpoints] == [ROBOT_A, ROBOT_B]
    for endpoint in endpoints:
        profile = RobotRuntimeProfile.from_robot_id(endpoint.robot_id)
        assert endpoint.prim_path == (
            PRIM_A if endpoint.robot_id == ROBOT_A else PRIM_B
        )
        assert endpoint.joint_states_topic == profile.isaac_joint_states
        assert endpoint.joint_command_topic == profile.isaac_joint_command
        assert endpoint.joint_names == profile.joint_names
        assert bridge_module.TOPIC_CLOCK == profile.clock == "/clock"
    assert endpoints[0].joint_states_topic == "/piper_0/isaac_joint_states"
    assert endpoints[1].joint_command_topic == "/piper_1/isaac_joint_command"


def test_per_robot_endpoints_are_disjoint(bridge_module) -> None:
    """No two robots may share a topic, a prefix or a joint name."""
    endpoints = bridge_module.build_robot_endpoints(_robots())
    topics = [
        topic
        for endpoint in endpoints
        for topic in (
            endpoint.joint_states_topic,
            endpoint.joint_command_topic,
            endpoint.joint_states_diag_topic,
        )
    ]
    assert len(set(topics)) == len(topics)
    joint_names = [name for endpoint in endpoints for name in endpoint.joint_names]
    assert len(set(joint_names)) == len(joint_names)
    assert endpoints[0].joint_names == tuple(f"{ROBOT_A}_{name}" for name in DOF_NAMES)


def test_empty_robot_set_fails(bridge_module) -> None:
    """A scene without any validated robot must fail closed."""
    with pytest.raises(bridge_module.RobotEndpointError):
        bridge_module.build_robot_endpoints({})


def test_duplicate_robot_ids_cannot_hide_in_a_sequence(bridge_module) -> None:
    """Duplicate robot entries are rejected upstream; a raw list never binds.

    ``build_runtime_host_inputs`` rejects a repeated robot Name before the
    bridge sees it; the endpoint contract additionally requires the validated
    robot_id -> spec mapping, so a sequence (the shape where duplicate ids
    could hide) fails closed here too.
    """
    with pytest.raises(bridge_module.RobotEndpointError):
        bridge_module.build_robot_endpoints([_spec(PRIM_A), _spec(PRIM_A)])


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param("duplicate_prim", id="duplicate-prim"),
        pytest.param("nested_prim", id="nested-prim"),
        pytest.param("relative_prim", id="relative-prim"),
        pytest.param("bad_robot_id", id="bad-robot-id"),
        pytest.param("short_joint_set", id="short-joint-set"),
        pytest.param("foreign_joint_set", id="foreign-joint-set"),
        pytest.param("duplicate_joints", id="duplicate-joints"),
    ],
)
def test_invalid_robot_inputs_fail_closed(bridge_module, mutation: str) -> None:
    """Duplicate/missing prims, bad ids and joint mismatches must all fail.

    Args:
        bridge_module: The imported bridge module.
        mutation: Which contract clause to violate.
    """
    robots = _robots()
    if mutation == "duplicate_prim":
        robots[ROBOT_B] = _spec(PRIM_A)
    elif mutation == "nested_prim":
        robots[ROBOT_B] = _spec(PRIM_A + "/root_joint")
    elif mutation == "relative_prim":
        robots[ROBOT_B] = _spec("World/Robot_1")
    elif mutation == "bad_robot_id":
        robots["piper-1"] = robots.pop(ROBOT_B)
    elif mutation == "short_joint_set":
        robots[ROBOT_B] = _spec(PRIM_B, DOF_NAMES[:6])
    elif mutation == "foreign_joint_set":
        robots[ROBOT_B] = _spec(PRIM_B, ("j1",) + DOF_NAMES[1:])
    else:
        robots[ROBOT_B] = _spec(PRIM_B, DOF_NAMES[:7] + (DOF_NAMES[0],))
    with pytest.raises(bridge_module.RobotEndpointError):
        bridge_module.build_robot_endpoints(robots)


def test_bind_requires_exact_loaded_robot_set(bridge_module) -> None:
    """A robot missing from the host or loaded without inputs must fail."""
    endpoints = bridge_module.bind_robot_endpoints(
        _robots(), [ROBOT_A, ROBOT_B]
    )
    assert [endpoint.robot_id for endpoint in endpoints] == [ROBOT_A, ROBOT_B]
    with pytest.raises(bridge_module.RobotEndpointError) as missing:
        bridge_module.bind_robot_endpoints(_robots(), [ROBOT_A])
    assert ROBOT_B in str(missing.value)
    with pytest.raises(bridge_module.RobotEndpointError) as unexpected:
        bridge_module.bind_robot_endpoints(_robots(), [ROBOT_A, ROBOT_B, "piper_2"])
    assert "piper_2" in str(unexpected.value)


def test_articulation_joint_mismatch_fails(bridge_module) -> None:
    """An articulation that does not expose the validated joints fails."""
    endpoint = bridge_module.build_robot_endpoints(_robots())[0]
    with pytest.raises(bridge_module.RobotEndpointError):
        bridge_module.RobotChannel(
            endpoint, FakeArticulation([0.0] * 8), DOF_NAMES[:7] + ("joint7",)
        )


# ------------------------------------------------------ snapshot binding


def test_snapshot_binding_is_explicit_and_recorded(bridge_module) -> None:
    """The single-robot binding follows the designated/validated robot only."""
    endpoints = bridge_module.build_robot_endpoints(_robots())
    single, source = bridge_module.select_snapshot_binding_endpoint(
        (endpoints[0],), None
    )
    assert single.robot_id == ROBOT_A
    assert source == "single_robot"
    designated, source = bridge_module.select_snapshot_binding_endpoint(
        endpoints, ROBOT_B
    )
    assert designated.robot_id == ROBOT_B
    assert source == "designated_robot"
    first, source = bridge_module.select_snapshot_binding_endpoint(endpoints, None)
    assert first.robot_id == ROBOT_A
    assert source == "first_validated_robot"
    with pytest.raises(bridge_module.RobotEndpointError):
        bridge_module.select_snapshot_binding_endpoint(endpoints, "piper_2")


# -------------------------------------------------------- command routing


def test_commands_route_only_to_the_addressed_robot(
    bridge_module, fake_action_module
) -> None:
    """A per-robot command drives exactly its own articulation."""
    channels, articulations = _channels(bridge_module)
    channel_a, channel_b = channels
    channel_a._dof_limits = {f"{ROBOT_A}_joint1": (-1.0, 1.0)}

    channel_a.on_joint_command(
        _command_message([f"{ROBOT_A}_joint1"], [0.25])
    )
    assert channel_a.commands_received == 1
    assert channel_a.commands_applied == 1
    assert len(articulations[ROBOT_A].actions) == 1
    action = articulations[ROBOT_A].actions[0]
    assert action.joint_positions == (0.25,)
    assert action.joint_indices == (0,)
    assert articulations[ROBOT_B].actions == []
    assert channel_b.commands_received == 0

    # The other robot's prefixed name is unknown on this channel: ignored,
    # never re-routed.
    applied_before = len(articulations[ROBOT_A].actions)
    channel_a.on_joint_command(
        _command_message([f"{ROBOT_B}_joint1"], [0.5])
    )
    assert channel_a.commands_applied == 1
    assert channel_a.unknown_joint_names == [f"{ROBOT_B}_joint1"]
    assert len(articulations[ROBOT_A].actions) == applied_before
    assert articulations[ROBOT_B].actions == []


def test_command_clamping_and_invalid_payloads(
    bridge_module, fake_action_module
) -> None:
    """Targets clamp to the recorded limits and malformed payloads do nothing."""
    channels, articulations = _channels(bridge_module)
    channel_a = channels[0]
    channel_a._dof_limits = {
        f"{ROBOT_A}_joint1": (-1.0, 1.0),
        f"{ROBOT_A}_joint2": (0.0, 2.0),
    }
    channel_a.on_joint_command(
        _command_message(
            [f"{ROBOT_A}_joint1", f"{ROBOT_A}_joint2"], [5.0, -5.0]
        )
    )
    action = articulations[ROBOT_A].actions[-1]
    assert action.joint_positions == (1.0, 0.0)
    assert action.joint_indices == (0, 1)
    assert channel_a.commands_clamped == 2

    before = len(articulations[ROBOT_A].actions)
    channel_a.on_joint_command(_command_message([], []))
    channel_a.on_joint_command(_command_message([f"{ROBOT_A}_joint1"], []))
    assert len(articulations[ROBOT_A].actions) == before
    assert channel_a.commands_received == 3


def test_gripper_observations_are_per_robot(bridge_module) -> None:
    """Each channel reads its own prefixed gripper DOFs."""
    positions_a = [0.0] * 6 + [0.01, -0.01]
    positions_b = [1.0] * 6 + [0.02, -0.02]
    channels, _articulations = _channels(
        bridge_module, positions_a=positions_a, positions_b=positions_b
    )
    assert channels[0].gripper_positions() == FINGERS_A
    assert channels[1].gripper_positions() == FINGERS_B


# -------------------------------------------------------------- rclpy wiring


def test_one_node_with_per_robot_endpoints(
    bridge_module, fake_rclpy, fake_action_module
) -> None:
    """One node (one participant) wires every robot's own topics."""
    _rclpy, nodes = fake_rclpy
    channels, articulations = _channels(bridge_module)
    state = types.SimpleNamespace(
        ros_node=None, runtime_host=None, scene_manifest_ids=None, world=None
    )
    bridge = bridge_module.InProcessBridge(state, channels, robot_only=False)

    assert len(nodes) == 1
    node = nodes[0]
    assert node.name == bridge_module.ROS_NODE_NAME
    assert state.ros_node is node
    assert [topic for _type, topic, _publisher in node.publishers] == [
        "/clock",
        f"/{ROBOT_A}{bridge_module.TOPIC_JOINT_STATES}",
        f"/{ROBOT_A}{bridge_module.TOPIC_JOINT_STATES_DIAG}",
        f"/{ROBOT_B}{bridge_module.TOPIC_JOINT_STATES}",
        f"/{ROBOT_B}{bridge_module.TOPIC_JOINT_STATES_DIAG}",
    ]
    assert [topic for _type, topic, _callback in node.subscriptions] == [
        f"/{ROBOT_A}{bridge_module.TOPIC_JOINT_COMMANDS}",
        f"/{ROBOT_B}{bridge_module.TOPIC_JOINT_COMMANDS}",
    ]
    assert [name for _type, name, _callback in node.services] == [
        SERVICE_SCENE_STATE
    ]
    for _type, topic, _publisher in node.publishers:
        assert "command" not in topic, topic

    world = types.SimpleNamespace(current_time=2.5)
    bridge.publish_state(world)
    bridge.publish_state(world)
    clock_publisher = node.publisher_for("/clock")
    assert len(clock_publisher.messages) == 2
    assert clock_publisher.messages[0].clock.sec == 2
    for robot_id, expected in ((ROBOT_A, FINGERS_A), (ROBOT_B, FINGERS_B)):
        publisher = node.publisher_for(f"/{robot_id}{bridge_module.TOPIC_JOINT_STATES}")
        assert len(publisher.messages) == 2
        message = publisher.messages[-1]
        assert message.name == tuple(f"{robot_id}_{name}" for name in DOF_NAMES)
        assert message.position == [
            float(value) for value in articulations[robot_id].positions
        ]
        assert len(message.velocity) == len(DOF_NAMES)
        assert message.header.stamp.sec == 2
        assert message.header.stamp.nanosec == 500_000_000
    assert bridge.clock_and_joint_states_published is True
    assert bridge.robots_ready is True
    bridge.spin_once()


def test_readiness_requires_every_robot(bridge_module, fake_rclpy) -> None:
    """One uninitialized or unpublished robot keeps the bridge unready."""
    _rclpy, nodes = fake_rclpy
    channels, articulations = _channels(bridge_module)
    articulations[ROBOT_B]._initialized = False
    state = types.SimpleNamespace(
        ros_node=None, runtime_host=None, scene_manifest_ids=None, world=None
    )
    bridge = bridge_module.InProcessBridge(state, channels, robot_only=False)
    assert bridge.robots_ready is False
    assert bridge.clock_and_joint_states_published is False
    articulations[ROBOT_B]._initialized = True
    assert bridge.robots_ready is True
    assert bridge.clock_and_joint_states_published is False
    bridge.publish_state(types.SimpleNamespace(current_time=1.0))
    assert bridge.clock_and_joint_states_published is True
    assert len(nodes) == 1


def test_robot_only_mode_keeps_the_pinned_single_robot_topics(
    bridge_module, fake_rclpy
) -> None:
    """The legacy robot-only mode keeps its un-namespaced contract."""
    _rclpy, nodes = fake_rclpy
    endpoint = bridge_module.RobotEndpoint(
        robot_id=None,
        prim_path=bridge_module.ROBOT_PRIM_PATH,
        dof_names=DOF_NAMES,
        joint_states_topic=bridge_module.TOPIC_JOINT_STATES,
        joint_command_topic=bridge_module.TOPIC_JOINT_COMMANDS,
        joint_states_diag_topic=bridge_module.TOPIC_JOINT_STATES_DIAG,
    )
    articulation = FakeArticulation([0.0] * 8)
    channel = bridge_module.RobotChannel(endpoint, articulation, DOF_NAMES)
    assert channel.joint_names == DOF_NAMES
    state = types.SimpleNamespace(
        ros_node=None, runtime_host=None, scene_manifest_ids=None, world=None
    )
    bridge = bridge_module.InProcessBridge(state, (channel,), robot_only=True)
    node = nodes[0]
    assert [topic for _type, topic, _publisher in node.publishers] == [
        "/clock",
        "/isaac_joint_states",
        "/joint_states_diag",
    ]
    assert [topic for _type, topic, _callback in node.subscriptions] == [
        "/isaac_joint_command"
    ]
    bridge.publish_state(types.SimpleNamespace(current_time=1.0))
    message = node.publisher_for("/isaac_joint_states").messages[-1]
    assert message.name == DOF_NAMES

    with pytest.raises(RuntimeError):
        bridge_module.InProcessBridge(state, (channel, channel), robot_only=True)
    with pytest.raises(RuntimeError):
        bridge_module.InProcessBridge(state, (channel,), robot_only=False)


def test_evidence_keys_are_json_safe_and_identified(bridge_module) -> None:
    """Every channel evidence key is a string robot id (legacy gets a label)."""
    channels, _articulations = _channels(bridge_module)
    keys = [bridge_module._channel_evidence_key(channel) for channel in channels]
    assert keys == [ROBOT_A, ROBOT_B]
    legacy = bridge_module.RobotEndpoint(
        robot_id=None,
        prim_path=bridge_module.ROBOT_PRIM_PATH,
        dof_names=DOF_NAMES,
        joint_states_topic=bridge_module.TOPIC_JOINT_STATES,
        joint_command_topic=bridge_module.TOPIC_JOINT_COMMANDS,
        joint_states_diag_topic=bridge_module.TOPIC_JOINT_STATES_DIAG,
    )
    legacy_channel = bridge_module.RobotChannel(
        legacy, FakeArticulation([0.0] * 8), DOF_NAMES
    )
    document = {
        bridge_module._channel_evidence_key(channel): {"robot_id": channel.robot_id}
        for channel in (*channels, legacy_channel)
    }
    assert json.loads(json.dumps(document))[
        bridge_module.ROBOT_ONLY_MODE
    ]["robot_id"] is None


def test_initial_evidence_never_claims_scene_robot_paths(bridge_module) -> None:
    """Static evidence is mode-aware and never pins a scene-mode robot."""
    scene_args = bridge_module.parse_args(
        [
            "--scene-manifest", "/tmp/m.json",
            "--scene-snapshot-out", "/tmp/s.json",
            "--host-instance-id", "repro-001",
            "--plan", "/tmp/plan.json",
            "--config", "/tmp/config.json",
        ]
    )
    scene_evidence = bridge_module.initial_evidence(scene_args)
    assert scene_evidence["robot_mode"] == bridge_module.SCENE_MODE
    assert scene_evidence["robot_prim_path"] is None
    assert scene_evidence["articulation_prim_path"] is None
    assert scene_evidence["topics"]["joint_states"] is None
    assert scene_evidence["per_robot_topics"]["joint_states"] == (
        "/<robot_id>/isaac_joint_states"
    )
    assert scene_evidence["rclpy_node_count"] == 1

    robot_only_args = bridge_module.parse_args(["--evidence-out", "/tmp/e.json"])
    robot_only_evidence = bridge_module.initial_evidence(robot_only_args)
    assert robot_only_evidence["robot_mode"] == bridge_module.ROBOT_ONLY_MODE
    assert robot_only_evidence["robot_prim_path"] == "/World/Robot_0"
    assert robot_only_evidence["topics"]["joint_states"] == "/isaac_joint_states"


def test_bridge_never_publishes_a_command_topic(
    bridge_module, bridge_source: str
) -> None:
    """The bridge only publishes clock/joint states, never motion commands."""
    tree = ast.parse(bridge_source)
    published_topics = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "create_publisher"
            and len(node.args) >= 2
        ):
            published_topics.append(ast.unparse(node.args[1]))
    assert published_topics, "expected at least one publisher"
    for topic in published_topics:
        assert "COMMAND" not in topic.upper(), topic
        assert "command" not in topic, topic
    assert "create_publisher(" in bridge_source
    assert "TOPIC_JOINT_COMMANDS" in bridge_source


# ------------------------------------------------- scene observation doc


def test_dual_scene_document_shape(bridge_module) -> None:
    """The dual response carries shared bricks plus per-robot observations."""
    document = bridge_module.build_scene_state_document(
        12.5,
        {BRICK_ID_A: AABBS_A, BRICK_ID_B: AABBS_B},
        {PAIR_KEY: False},
        None,
        robot_grippers={ROBOT_A: FINGERS_A, ROBOT_B: FINGERS_B},
    )
    assert document == {
        "schema": SCHEMA,
        "sim_time_s": 12.5,
        "bricks": {
            BRICK_ID_A: {
                "aabb_min": [0.1, -0.05, 0.0],
                "aabb_max": [0.14, 0.03, 0.0112],
            },
            BRICK_ID_B: {
                "aabb_min": [0.1, -0.05, 0.0112],
                "aabb_max": [0.14, 0.03, 0.0224],
            },
        },
        "connections": {PAIR_KEY: False},
        "robots": {
            ROBOT_A: {
                "robot_id": ROBOT_A,
                "sim_time_s": 12.5,
                "gripper": {"gripper_joint1": 0.01, "gripper_joint2": -0.01},
                "opening_m": 0.02,
            },
            ROBOT_B: {
                "robot_id": ROBOT_B,
                "sim_time_s": 12.5,
                "gripper": {"gripper_joint1": 0.02, "gripper_joint2": -0.02},
                "opening_m": 0.04,
            },
        },
        "scene_digest": document["scene_digest"],
    }
    assert "gripper" not in document
    assert "opening_m" not in document
    assert json.loads(json.dumps(document, sort_keys=True)) == document


def test_single_robot_scene_keeps_the_top_level_block(bridge_module) -> None:
    """A one-robot scene keeps the legacy block next to its robots block."""
    document = bridge_module.build_scene_state_document(
        3.0,
        {BRICK_ID_A: AABBS_A},
        {},
        FINGERS_A,
        robot_grippers={ROBOT_A: FINGERS_A},
    )
    assert document["gripper"] == FINGERS_A
    assert document["opening_m"] == pytest.approx(0.02)
    assert document["robots"][ROBOT_A] == {
        "robot_id": ROBOT_A,
        "sim_time_s": 3.0,
        "gripper": FINGERS_A,
        "opening_m": 0.02,
    }


def test_scene_document_rejects_incomplete_observations(bridge_module) -> None:
    """An unnamed robot, bad fingers or a missing observation fail closed."""
    with pytest.raises(ValueError):
        bridge_module.build_scene_state_document(
            1.0, {BRICK_ID_A: AABBS_A}, {}, robot_grippers={}
        )
    with pytest.raises(ValueError):
        bridge_module.build_scene_state_document(
            1.0, {BRICK_ID_A: AABBS_A}, {}, robot_grippers={"bad/id": FINGERS_A}
        )
    with pytest.raises(ValueError):
        bridge_module.build_scene_state_document(
            1.0,
            {BRICK_ID_A: AABBS_A},
            {},
            robot_grippers={ROBOT_A: {"gripper_joint1": 0.01}},
        )
    with pytest.raises(ValueError):
        bridge_module.build_scene_state_document(1.0, {BRICK_ID_A: AABBS_A}, {})


def _stub_scene_bridge(bridge_module, robot_grippers, articulations=None):
    """Build an InProcessBridge with scene handles for the service callback.

    Args:
        bridge_module: The imported bridge module.
        robot_grippers: ``robot_id -> finger positions`` mapping.
        articulations: Optional mapping of robot id to FakeArticulation.

    Returns:
        The stubbed bridge whose scene-state callback runs offline.
    """
    bridge = bridge_module.InProcessBridge.__new__(bridge_module.InProcessBridge)
    bridge._state = types.SimpleNamespace(
        runtime_host=types.SimpleNamespace(
            physical_instance_paths={BRICK_ID_A: "/World/brick_a"}
        ),
        scene_manifest_ids=sorted([BRICK_ID_A, BRICK_ID_B]),
        world=types.SimpleNamespace(current_time=4.5),
    )
    default_positions = {
        ROBOT_A: [0.0] * 6 + [0.01, -0.01],
        ROBOT_B: [0.0] * 6 + [0.02, -0.02],
    }
    channels = []
    for robot_id, _fingers in robot_grippers.items():
        positions = (
            articulations[robot_id].positions
            if articulations is not None
            else default_positions[robot_id]
        )
        endpoint = bridge_module.RobotEndpoint(
            robot_id=robot_id,
            prim_path=PRIM_A if robot_id == ROBOT_A else PRIM_B,
            dof_names=DOF_NAMES,
            joint_states_topic=f"/{robot_id}{bridge_module.TOPIC_JOINT_STATES}",
            joint_command_topic=f"/{robot_id}{bridge_module.TOPIC_JOINT_COMMANDS}",
            joint_states_diag_topic=f"/{robot_id}{bridge_module.TOPIC_JOINT_STATES_DIAG}",
        )
        channel = bridge_module.RobotChannel(
            endpoint, FakeArticulation(positions), DOF_NAMES
        )
        channels.append(channel)
    bridge._channels = tuple(channels)
    bridge._brick_aabbs = lambda: {BRICK_ID_A: AABBS_A, BRICK_ID_B: AABBS_B}
    bridge._brick_connections = lambda: {PAIR_KEY: True}
    bridge._scene_sequence = 0
    bridge._state.scene_session_id = "test-session"
    bridge._state.scene_manifest_sha256 = "a" * 64
    bridge._live_rigid_poses = lambda: (
        {instance_id: {
            "position_xyz": [0.0, 0.0, 0.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        } for instance_id in (BRICK_ID_A, BRICK_ID_B)},
        {robot_id: {
            "position_xyz": [0.0, 0.0, 0.1],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        } for robot_id in robot_grippers},
    )
    return bridge


def test_callback_observes_both_robots_without_cross_fallback(
    bridge_module,
) -> None:
    """robots[robot_id] carries that robot's own gripper observation."""
    positions_a = [0.0] * 6 + [0.01, -0.01]
    positions_b = [0.0] * 6 + [0.02, -0.02]
    articulations = {
        ROBOT_A: FakeArticulation(positions_a),
        ROBOT_B: FakeArticulation(positions_b),
    }
    bridge = _stub_scene_bridge(
        bridge_module,
        {ROBOT_A: FINGERS_A, ROBOT_B: FINGERS_B},
        articulations=articulations,
    )
    response = types.SimpleNamespace(success=None, message=None)
    returned = bridge._on_scene_state(object(), response)
    assert returned is response
    assert response.success is True
    document = json.loads(response.message)
    assert document["schema"] == SCHEMA
    assert document["sim_time_s"] == 4.5
    assert document["robots"][ROBOT_A]["robot_id"] == ROBOT_A
    assert document["robots"][ROBOT_B]["robot_id"] == ROBOT_B
    assert document["robots"][ROBOT_A]["gripper"] == FINGERS_A
    assert document["robots"][ROBOT_B]["gripper"] == FINGERS_B
    assert document["robots"][ROBOT_A]["opening_m"] == pytest.approx(0.02)
    assert document["robots"][ROBOT_B]["opening_m"] == pytest.approx(0.04)
    assert "gripper" not in document
    assert document["connections"] == {PAIR_KEY: True}
    assert json.dumps(document, sort_keys=True, separators=(",", ":")) == (
        response.message
    )


def test_callback_single_robot_keeps_legacy_top_level(bridge_module) -> None:
    """A one-robot scene answers with both the legacy and the robots block."""
    bridge = _stub_scene_bridge(bridge_module, {ROBOT_A: FINGERS_A}, articulations={
        ROBOT_A: FakeArticulation([0.0] * 6 + [0.01, -0.01])
    })
    response = types.SimpleNamespace(success=None, message=None)
    bridge._on_scene_state(object(), response)
    document = json.loads(response.message)
    assert document["gripper"] == FINGERS_A
    assert document["opening_m"] == pytest.approx(0.02)
    block = document["robots"][ROBOT_A]
    assert block["robot_id"] == ROBOT_A
    assert block["sim_time_s"] == 4.5
    assert block["gripper"] == FINGERS_A
    assert block["opening_m"] == pytest.approx(0.02)
    assert block["sequence"] == document["sequence"]
    assert block["tcp_pose"]["position_xyz"] == [0.0, 0.0, 0.1]


def test_callback_fails_closed_when_a_robot_loses_its_gripper(bridge_module) -> None:
    """A channel without the gripper DOFs becomes an error document."""
    endpoint = bridge_module.RobotEndpoint(
        robot_id=ROBOT_A,
        prim_path=PRIM_A,
        dof_names=DOF_NAMES,
        joint_states_topic=f"/{ROBOT_A}{bridge_module.TOPIC_JOINT_STATES}",
        joint_command_topic=f"/{ROBOT_A}{bridge_module.TOPIC_JOINT_COMMANDS}",
        joint_states_diag_topic=f"/{ROBOT_A}{bridge_module.TOPIC_JOINT_STATES_DIAG}",
    )
    channel = bridge_module.RobotChannel(
        endpoint, FakeArticulation([0.0] * 8), DOF_NAMES
    )
    channel._dof_index.pop(f"{ROBOT_A}_gripper_joint2")
    bridge = _stub_scene_bridge(bridge_module, {ROBOT_A: FINGERS_A})
    bridge._channels = (channel,)
    response = types.SimpleNamespace(success=None, message=None)
    bridge._on_scene_state(object(), response)
    assert response.success is False
    document = json.loads(response.message)
    assert document["schema"] == SCHEMA
    assert document["error"].startswith("KeyError:")


# --------------------------------------------------- SIGINT early stop

EARLY_STOP_CLEANUP = {
    "rclpy_shutdown": True,
    "world_stopped": True,
    "timeline_stopped": True,
    "app_close_initiated": True,
}
SIGINT_MARKER = "sigint_orderly_stop"
KIT_CANCEL_MARKER = "kit_child_cancelled_orderly_stop"


def _early_stop_evidence(**overrides) -> dict:
    """Build an evidence document satisfying the accepted early-stop contract.

    Args:
        overrides: Fields to override on the accepted baseline.

    Returns:
        The evidence dict.
    """
    evidence = {
        "ready": True,
        "stopped": SIGINT_MARKER,
        "cleanup": dict(EARLY_STOP_CLEANUP),
        "commands": {
            "received": 0,
            "applied": 0,
            "clamped_to_limits": 0,
            "unknown_joint_names_ignored": [],
        },
    }
    evidence.update(overrides)
    return evidence


def test_early_stop_accepted_requires_every_condition(bridge_module) -> None:
    """A ready and orderly SIGINT is accepted with audited commands."""
    assert bridge_module.EARLY_STOP_STOPPED == SIGINT_MARKER
    assert bridge_module.evaluate_early_stop_accepted(_early_stop_evidence()) is True
    closed = _early_stop_evidence()
    closed["cleanup"] = {
        **EARLY_STOP_CLEANUP,
        "app_close_initiated": False,
        "app_closed": True,
    }
    assert bridge_module.evaluate_early_stop_accepted(closed) is True
    # A command received but never applied (unknown joint names) is harmless.
    unapplied = _early_stop_evidence()
    unapplied["commands"] = {
        "received": 2,
        "applied": 0,
        "clamped_to_limits": 0,
        "unknown_joint_names_ignored": ["some_other_robot_joint1"],
    }
    assert bridge_module.evaluate_early_stop_accepted(unapplied) is True
    controller_hold = _early_stop_evidence()
    controller_hold["commands"]["applied"] = 2
    assert bridge_module.evaluate_early_stop_accepted(controller_hold) is True


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda e: e.update({"ready": False}), id="not-ready"),
        pytest.param(lambda e: e.pop("ready"), id="ready-missing"),
        pytest.param(lambda e: e.update({"error": "TimeoutError: boom"}), id="error"),
        pytest.param(
            lambda e: e.update({"stopped": KIT_CANCEL_MARKER}),
            id="kit-child-cancellation",
        ),
        pytest.param(lambda e: e.pop("stopped"), id="no-stop-marker"),
        pytest.param(
            lambda e: e.update(
                {"cleanup": {**e["cleanup"], "rclpy_shutdown": "error: boom"}}
            ),
            id="rclpy-failed",
        ),
        pytest.param(
            lambda e: e.update({"cleanup": {**e["cleanup"], "world_stopped": False}}),
            id="world-not-stopped",
        ),
        pytest.param(
            lambda e: e.update({
                "cleanup": {
                    key: value
                    for key, value in e["cleanup"].items()
                    if key != "timeline_stopped"
                }
            }),
            id="timeline-missing",
        ),
        pytest.param(
            lambda e: e.update({"cleanup": {
                "rclpy_shutdown": True,
                "world_stopped": True,
                "timeline_stopped": True,
            }}),
            id="close-stage-missing",
        ),
        pytest.param(lambda e: e.pop("cleanup"), id="cleanup-missing"),
        pytest.param(lambda e: e.update({"cleanup": None}), id="cleanup-null"),
    ],
)
def test_early_stop_rejected_cases(bridge_module, mutate) -> None:
    """Any missing or failed early-stop condition keeps the verdict false.

    Args:
        bridge_module: The imported bridge module.
        mutate: Mutation applied to the accepted baseline evidence.
    """
    evidence = _early_stop_evidence()
    mutate(evidence)
    assert bridge_module.evaluate_early_stop_accepted(evidence) is False


def test_early_stop_never_marks_the_execution_passed(bridge_module) -> None:
    """hold_completed/passed semantics stay governed by evaluate_passed."""
    evidence = _early_stop_evidence()
    assert bridge_module.evaluate_early_stop_accepted(evidence) is True
    assert bridge_module.evaluate_passed(evidence) is False
    assert "hold_completed" not in evidence
    completed = _early_stop_evidence(hold_completed=True)
    assert bridge_module.evaluate_passed(completed) is True


def test_counter_collection_reads_the_live_channels(
    bridge_module, fake_action_module
) -> None:
    """Interrupt evidence carries the exact live per-robot command counters."""
    channels, _articulations = _channels(bridge_module)
    channels[0].on_joint_command(_command_message([f"{ROBOT_A}_joint1"], [0.1]))
    channels[1].on_joint_command(_command_message([f"{ROBOT_B}_joint2"], [0.2]))
    counters = bridge_module.collect_counter_evidence(channels, 7)
    assert counters["commands"]["received"] == 2
    assert counters["commands"]["applied"] == 2
    assert counters["commands"]["clamped_to_limits"] == 0
    assert counters["commands"]["unknown_joint_names_ignored"] == []
    assert counters["commands"]["by_robot"][ROBOT_A]["applied"] == 1
    assert counters["commands"]["by_robot"][ROBOT_B]["applied"] == 1
    assert counters["publish_totals"]["clock"] == 7
    assert set(counters["publish_totals"]) == {
        "clock",
        "joint_states",
        "joint_states_diag",
        "positions_nan",
        "by_robot",
    }
    evidence = _early_stop_evidence()
    evidence.update(counters)
    assert bridge_module.evaluate_early_stop_accepted(evidence) is True


def test_counter_collection_legacy_channel_has_no_robot_breakdown(
    bridge_module,
) -> None:
    """The legacy robot-only channel keeps the flat counter shape."""
    endpoint = bridge_module.RobotEndpoint(
        robot_id=None,
        prim_path=bridge_module.ROBOT_PRIM_PATH,
        dof_names=DOF_NAMES,
        joint_states_topic=bridge_module.TOPIC_JOINT_STATES,
        joint_command_topic=bridge_module.TOPIC_JOINT_COMMANDS,
        joint_states_diag_topic=bridge_module.TOPIC_JOINT_STATES_DIAG,
    )
    channel = bridge_module.RobotChannel(
        endpoint, FakeArticulation([0.0] * 8), DOF_NAMES
    )
    counters = bridge_module.collect_counter_evidence((channel,), 3)
    assert "by_robot" not in counters["commands"]
    assert "by_robot" not in counters["publish_totals"]
    assert set(counters["publish_totals"]) == {
        "clock",
        "joint_states",
        "joint_states_diag",
        "positions_nan",
    }


def test_initial_evidence_defaults_early_stop_rejected(bridge_module) -> None:
    """Static evidence never claims an accepted early stop."""
    args = bridge_module.parse_args(["--evidence-out", "/tmp/e.json"])
    assert bridge_module.initial_evidence(args)["early_stop_accepted"] is False


def test_cleanup_flush_keeps_the_early_stop_verdict(bridge_module, tmp_path) -> None:
    """Every cleanup flush recomputes early_stop_accepted (Kit may die in close)."""
    state = bridge_module.BridgeState()
    state.evidence = _early_stop_evidence()
    state.evidence_path = tmp_path / "evidence.json"
    state._flush(dict(EARLY_STOP_CLEANUP))
    document = json.loads(state.evidence_path.read_text(encoding="utf-8"))
    assert document["early_stop_accepted"] is True
    assert document["passed"] is False
    state._flush({**EARLY_STOP_CLEANUP, "rclpy_shutdown": "error: boom"})
    document = json.loads(state.evidence_path.read_text(encoding="utf-8"))
    assert document["early_stop_accepted"] is False


def test_child_interrupt_path_records_counters_before_final_write(
    bridge_source: str,
) -> None:
    """The Kit child records live counters, then verdicts, in a fixed order."""
    body = bridge_source[bridge_source.index("    async def _child_bridge"):]
    keyboard = body.index("except KeyboardInterrupt")
    cancelled = body.index("except asyncio.CancelledError")
    generic = body.index("except Exception")
    assert 'evidence["stopped"] = EARLY_STOP_STOPPED' in body[keyboard:cancelled]
    assert KIT_CANCEL_MARKER in body[cancelled:generic]
    assert "raise" in body[cancelled:generic]
    counters = body.index("collect_counter_evidence(")
    first_write = body.index("write_evidence(")
    cleanup = body.index('evidence["cleanup"] = state.shutdown()')
    accepted = body.index(
        'evidence["early_stop_accepted"] = evaluate_early_stop_accepted('
    )
    passed = body.index('evidence["passed"] = evaluate_passed(evidence)')
    assert counters < first_write < cleanup < accepted < passed
    finally_start = body.index("finally:")
    assert "bridge = state.bridge" in body[finally_start:first_write]
    assert 'evidence["hold_completed"]' not in body



def test_sigint_is_deferred_to_coroutine_cleanup(bridge_module, tmp_path) -> None:
    """The signal handler records intent without writing or stopping Kit."""
    state = bridge_module.BridgeState()
    state.evidence = _early_stop_evidence()
    state.evidence.pop("stopped")
    state.evidence_path = tmp_path / "evidence.json"
    state.request_stop(bridge_module.signal.SIGTERM, None)
    state.check_stop()
    state.request_stop(bridge_module.signal.SIGINT, None)
    assert state.stop_signal == bridge_module.signal.SIGINT
    assert not state.evidence_path.exists()
    assert "stopped" not in state.evidence
    with pytest.raises(KeyboardInterrupt):
        state.check_stop()
    state.evidence["stopped"] = bridge_module.EARLY_STOP_STOPPED
    state._flush(dict(EARLY_STOP_CLEANUP))
    final = json.loads(state.evidence_path.read_text())
    assert final["early_stop_accepted"] is True
    assert final["passed"] is False


def test_bridge_loops_check_stop_and_preserve_early_evidence_binding(
    bridge_source: str,
) -> None:
    """Each physics loop checks intent before stepping; state owns live evidence."""
    assert bridge_source.count("        state.check_stop()") == 2
    assert "signal.signal(signal.SIGINT, state.request_stop)" in bridge_source
    main_body = bridge_source[bridge_source.index("def main("):]
    assert main_body.index("state.evidence = evidence") < main_body.index(
        "if not _running_inside_kit():"
    )
