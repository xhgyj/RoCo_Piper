"""Offline tests for the P0.5 host bridge scene-state service.

These tests never boot Kit, rclpy, or Isaac: they import the bridge module by
file path (the same pattern as tests/test_host_bridge_p05.py) and exercise the
pure scene-state document assembly plus the fail-closed service callback with
stub objects. Pinned: the ``/p05_host_bridge/scene_state``
``std_srvs/srv/Trigger`` contract served by the single in-process rclpy node,
canonical JSON payloads of schema ``rocobrick-scene-state/v1``, the gripper
opening derivation through ``rocobrick.gripper.fingers_to_logical_gripper``,
ValueError on every input-contract violation, and error documents truncated
to 300 characters.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
BRIDGE_PATH = ROOT / "isaac" / "p05_host_bridge.py"

EXPECTED_NODE_NAME = "piper_l0_host_bridge"
SCHEMA = "rocobrick-scene-state/v1"
SERVICE_NAME = "/p05_host_bridge/scene_state"
DOF_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "gripper_joint1",
    "gripper_joint2",
]

BRICK_ID_A = "brick-768de5a17827920c48d4"
BRICK_ID_B = "brick-9f3c2216c25b04ba942a"
PAIR_KEY = f"{BRICK_ID_A}|{BRICK_ID_B}"

VALID_AABBS_A = ((0.1, -0.05, 0.0), (0.14, 0.03, 0.0112))
VALID_AABBS_B = ((0.1, -0.05, 0.0112), (0.14, 0.03, 0.0224))


@pytest.fixture(scope="module")
def bridge_module():
    """Import the bridge module without booting isaacsim, omni, or rclpy.

    The module is registered in ``sys.modules`` before execution because the
    bridge's frozen value types are dataclasses and dataclasses resolve their
    postponed string annotations through ``sys.modules``.

    Returns:
        The imported bridge module.
    """
    spec = importlib.util.spec_from_file_location(
        "p05_host_bridge_scene_state", BRIDGE_PATH
    )
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


def _valid_inputs() -> tuple[float, dict, dict, dict]:
    """Return scene-state inputs satisfying the document contract.

    Returns:
        The (sim_time_s, brick_aabbs, connection_results, gripper_positions)
        tuple for one valid two-brick scene.
    """
    return (
        12.5,
        {BRICK_ID_A: VALID_AABBS_A, BRICK_ID_B: VALID_AABBS_B},
        {PAIR_KEY: False},
        {"gripper_joint1": 0.01, "gripper_joint2": -0.01},
    )


def _canonical_json(document: dict) -> str:
    """Serialize a document with the repo's canonical JSON convention.

    Args:
        document: The payload dict.

    Returns:
        The canonical JSON string (sorted keys, no spaces).
    """
    return json.dumps(document, sort_keys=True, separators=(",", ":"))


def _stub_bridge(
    bridge_module,
    *,
    runtime_host=None,
    scene_manifest_ids=None,
    current_time=1.5,
    robot_grippers=None,
):
    """Build an InProcessBridge via ``__new__`` with stub scene-mode state.

    Args:
        bridge_module: The imported bridge module.
        runtime_host: Optional stub runtime host holding the prim mapping.
        scene_manifest_ids: Optional ascending physical instance ids.
        current_time: Sim time reported by the stub world.
        robot_grippers: Optional ``robot_id -> [positions]`` mapping; the
            default is one unidentified channel (legacy robot-only mode).

    Returns:
        The bridge instance whose rclpy-free scene-state paths can run offline.
    """
    bridge = bridge_module.InProcessBridge.__new__(bridge_module.InProcessBridge)
    bridge._state = types.SimpleNamespace(
        runtime_host=runtime_host,
        scene_manifest_ids=scene_manifest_ids,
        world=types.SimpleNamespace(current_time=current_time),
    )
    if robot_grippers is None:
        robot_grippers = {None: [0.0] * 6 + [0.01, -0.01]}
    channels = []
    for robot_id, positions in robot_grippers.items():
        channel = bridge_module.RobotChannel.__new__(bridge_module.RobotChannel)
        channel.robot_id = robot_id
        channel._robot = types.SimpleNamespace(get_joint_positions=lambda: positions)
        prefix = "" if robot_id is None else f"{robot_id}_"
        channel._dof_index = {
            prefix + name: index for index, name in enumerate(DOF_NAMES)
        }
        channels.append(channel)
    bridge._channels = tuple(channels)
    bridge._scene_sequence = 0
    bridge._state.scene_session_id = "test-session"
    bridge._state.scene_manifest_sha256 = "a" * 64
    bridge._live_rigid_poses = lambda: (
        {instance_id: {
            "position_xyz": [0.0, 0.0, 0.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        } for instance_id in runtime_host.physical_instance_paths}
        if runtime_host is not None else {},
        {robot_id: {
            "position_xyz": [0.0, 0.0, 0.1],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        } for robot_id in robot_grippers if robot_id is not None},
    )
    return bridge


def test_module_import_is_offline_safe(bridge_module) -> None:
    """Importing the module must not boot isaacsim, omni, rclpy, or pxr."""
    assert bridge_module.SCHEMA.startswith("rocobrick-host-bridge")
    for heavy in ("isaacsim", "omni", "rclpy", "pxr"):
        assert heavy not in sys.modules, heavy


def test_physx_pose_requires_a_live_rigidbody(bridge_module) -> None:
    """No USD or guessed-pose fallback is allowed on a PhysX miss."""
    bridge = _stub_bridge(bridge_module)
    physics = types.SimpleNamespace(
        get_rigidbody_transformation=lambda path: {"ret_val": False}
    )
    with pytest.raises(RuntimeError, match="no live PhysX rigid transform"):
        bridge._physx_pose("/World/brick_a", physics)
    physics = types.SimpleNamespace(
        get_rigidbody_transformation=lambda path: {
            "ret_val": True,
            "position": (0.1, 0.2, 0.3),
            "rotation": (0.0, 0.0, 0.0, 1.0),
        }
    )
    assert bridge._physx_pose("/World/brick_a", physics)["position_xyz"] == [
        0.1, 0.2, 0.3
    ]


def test_service_contract_constants(bridge_module) -> None:
    """The service name, schema, and node are pinned to the fixed contract."""
    assert bridge_module.SCENE_STATE_SERVICE == SERVICE_NAME
    assert bridge_module.SCENE_STATE_SCHEMA == SCHEMA
    assert bridge_module.ROS_NODE_NAME == EXPECTED_NODE_NAME


def test_service_created_on_the_single_node(bridge_source: str) -> None:
    """The Trigger service is created after the node and recorded honestly."""
    assert "from std_srvs.srv import Trigger" in bridge_source
    node_creation = bridge_source.index("rclpy.create_node(")
    service_creation = bridge_source.index("node.create_service(")
    assert node_creation < service_creation
    assert "Trigger, SCENE_STATE_SERVICE, self._on_scene_state" in bridge_source
    assert bridge_source.index("scene_state_service_created = False") < service_creation
    assert 'evidence["scene_state_service"] = {' in bridge_source
    assert '"created": False' in bridge_source
    assert 'evidence["scene_state_service"]["created"] = bool(' in bridge_source
    assert "state.scene_manifest_ids = sorted(manifest_map)" in bridge_source


def test_service_answered_by_bounded_per_step_spin(bridge_source: str) -> None:
    """The loop services requests at the existing bounded per-step point."""
    assert "timeout_sec=0.0" in bridge_source
    assert bridge_source.count("bridge.spin_once()") >= 2


def test_happy_path_document_structure(bridge_module) -> None:
    """A valid input set yields the exact rocobrick-scene-state/v1 document."""
    document = bridge_module.build_scene_state_document(*_valid_inputs())
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
        "gripper": {"gripper_joint1": 0.01, "gripper_joint2": -0.01},
        "opening_m": 0.02,
        "scene_digest": document["scene_digest"],
    }
    assert "opening_error" not in document


def test_canonical_json_round_trip_is_stable(bridge_module) -> None:
    """The serialized document round-trips and re-serializes byte-stably."""
    document = bridge_module.build_scene_state_document(*_valid_inputs())
    first = _canonical_json(document)
    assert json.loads(first) == document
    assert _canonical_json(json.loads(first)) == first
    assert " " not in first


def test_opening_m_matches_gripper_conversion(bridge_module) -> None:
    """opening_m equals fingers_to_logical_gripper(finger_1, finger_2)."""
    from rocobrick.gripper import fingers_to_logical_gripper

    finger_1, finger_2 = 0.0123, -0.01234
    sim_time_s, brick_aabbs, connection_results, _ = _valid_inputs()
    document = bridge_module.build_scene_state_document(
        sim_time_s,
        brick_aabbs,
        connection_results,
        {"gripper_joint1": finger_1, "gripper_joint2": finger_2},
    )
    assert document["opening_m"] == pytest.approx(
        fingers_to_logical_gripper(finger_1, finger_2)
    )


def test_mimic_violation_yields_null_opening_and_error(bridge_module) -> None:
    """A mimic violation keeps the document but nulls opening_m."""
    sim_time_s, brick_aabbs, connection_results, _ = _valid_inputs()
    document = bridge_module.build_scene_state_document(
        sim_time_s,
        brick_aabbs,
        connection_results,
        {"gripper_joint1": 0.01, "gripper_joint2": 0.0},
    )
    assert document["opening_m"] is None
    assert document["opening_error"] == (
        "finger measurements do not satisfy the official mimic relation"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param("non_finite_aabb", id="non-finite-aabb"),
        pytest.param("min_gt_max", id="min-greater-than-max"),
        pytest.param("missing_gripper_joint", id="missing-gripper-joint"),
        pytest.param("empty_brick_id", id="empty-brick-id"),
        pytest.param("non_bool_connection", id="non-bool-connection"),
        pytest.param("negative_sim_time", id="negative-sim-time"),
        pytest.param("non_finite_sim_time", id="non-finite-sim-time"),
    ],
)
def test_contract_violations_raise_value_error(bridge_module, mutation: str) -> None:
    """Every input-contract violation raises ValueError, never a bad document.

    Args:
        bridge_module: The imported bridge module.
        mutation: Which contract clause to violate.
    """
    sim_time_s, brick_aabbs, connection_results, gripper_positions = _valid_inputs()
    if mutation == "non_finite_aabb":
        brick_aabbs[BRICK_ID_A] = ((math.inf, 0.0, 0.0), (1.0, 1.0, 1.0))
    elif mutation == "min_gt_max":
        brick_aabbs[BRICK_ID_A] = ((0.2, 0.0, 0.0), (0.1, 0.03, 0.0112))
    elif mutation == "missing_gripper_joint":
        gripper_positions.pop("gripper_joint2")
    elif mutation == "empty_brick_id":
        brick_aabbs[""] = brick_aabbs.pop(BRICK_ID_A)
    elif mutation == "non_bool_connection":
        connection_results[PAIR_KEY] = 1
    elif mutation == "negative_sim_time":
        sim_time_s = -1.0
    else:
        sim_time_s = math.nan
    with pytest.raises(ValueError):
        bridge_module.build_scene_state_document(
            sim_time_s, brick_aabbs, connection_results, gripper_positions
        )


def test_error_document_truncates_to_300_chars(bridge_module) -> None:
    """scene_state_error_document truncates the error to 300 characters."""
    long_error = "e" * 400
    assert bridge_module.scene_state_error_document(long_error) == {
        "schema": SCHEMA,
        "error": "e" * 300,
    }
    assert bridge_module.scene_state_error_document("boom") == {
        "schema": SCHEMA,
        "error": "boom",
    }


def test_callback_fails_closed_without_scene(bridge_module) -> None:
    """Without a loaded scene the callback returns the scene_not_loaded doc."""
    bridge = _stub_bridge(bridge_module)
    response = types.SimpleNamespace(success=None, message=None)
    returned = bridge._on_scene_state(object(), response)
    assert returned is response
    assert response.success is False
    assert json.loads(response.message) == {
        "schema": SCHEMA,
        "error": "scene_not_loaded",
    }


def test_callback_assembles_live_scene_state(bridge_module) -> None:
    """The happy-path callback reads gripper DOFs via the joint index map."""
    runtime_host = types.SimpleNamespace(
        physical_instance_paths={
            BRICK_ID_B: "/World/brick_b",
            BRICK_ID_A: "/World/brick_a",
        },
    )
    bridge = _stub_bridge(
        bridge_module,
        runtime_host=runtime_host,
        scene_manifest_ids=[BRICK_ID_A, BRICK_ID_B],
        current_time=3.25,
    )
    bridge._brick_aabbs = lambda: {
        BRICK_ID_A: VALID_AABBS_A,
        BRICK_ID_B: VALID_AABBS_B,
    }
    bridge._brick_connections = lambda: {PAIR_KEY: True}
    response = types.SimpleNamespace(success=None, message=None)
    bridge._on_scene_state(object(), response)
    assert response.success is True
    document = json.loads(response.message)
    assert document["schema"] == SCHEMA
    assert document["sim_time_s"] == 3.25
    assert document["connections"] == {PAIR_KEY: True}
    assert document["gripper"] == {"gripper_joint1": 0.01, "gripper_joint2": -0.01}
    assert document["opening_m"] == pytest.approx(0.02)
    assert _canonical_json(document) == response.message


def test_callback_wraps_assembly_errors(bridge_module) -> None:
    """An assembly exception becomes a truncated error document, not a raise."""
    runtime_host = types.SimpleNamespace(physical_instance_paths={BRICK_ID_A: "/p"})
    bridge = _stub_bridge(
        bridge_module,
        runtime_host=runtime_host,
        scene_manifest_ids=[BRICK_ID_A],
    )

    def boom() -> dict:
        """Simulate a failing AABB collection.

        Raises:
            TypeError: Always, to exercise the error wrapping.
        """
        raise TypeError("stage exploded")

    bridge._brick_aabbs = boom
    response = types.SimpleNamespace(success=None, message=None)
    returned = bridge._on_scene_state(object(), response)
    assert returned is response
    assert response.success is False
    assert json.loads(response.message) == {
        "schema": SCHEMA,
        "error": "TypeError: stage exploded",
    }
