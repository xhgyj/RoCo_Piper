"""Pre-contact target derivation from one immutable shared scene."""

from dataclasses import asdict, replace

import pytest
from test_pick_handler import document, operation

from rocobrick.adapters.moveit.transit import TransitContext
from rocobrick.robot_operation_plan import OperationTarget
from rocobrick.skills.pre_phase_targets import (
    derive_pre_phase_targets,
    derive_wire_targets,
)
from rocobrick.skills.targets import RobotBaseFrame, SkillParams, TcpProvenance


def test_pick_target_is_pregrasp_and_other_arm_keeps_measured_tcp():
    """PICK pre-phase must not use generic transport height."""
    shared = document()
    shared["robots"]["piper_1"] = {**shared["robots"]["piper_0"], "robot_id": "piper_1"}
    context = TransitContext(
        RobotBaseFrame((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        TcpProvenance("a" * 64, "b" * 64),
        SkillParams(),
    )
    item = operation()
    result = derive_pre_phase_targets(
        item, shared, {robot: context for robot in ("piper_0", "piper_1")}
    )
    assert result["piper_0"]["tcp_pose"]["position"][2] == pytest.approx(
        0.01 + context.params.pregrasp_standoff_m
    )
    assert result["piper_1"]["tcp_pose"]["position"] == (0.0, 0.0, 0.04)
    assert item.parameters == {}
    transit = replace(
        item,
        skill="TRANSIT",
        target=OperationTarget(
            {
                "from": {
                    "kind": "robot_current",
                    "physical_instance_id": None,
                    "connection_id": None,
                    "robot_id": None,
                },
                "to": {
                    "kind": "part",
                    "physical_instance_id": "brick-a",
                    "connection_id": None,
                    "robot_id": None,
                },
                "carrying_physical_instance_id": None,
            }
        ),
    )
    transit_targets = derive_pre_phase_targets(
        transit, shared, {robot: context for robot in ("piper_0", "piper_1")}
    )
    assert transit_targets["piper_0"] == result["piper_0"]
    support = replace(
        item,
        skill="SUPPORT_TOP",
        target=OperationTarget({"upper_physical_instance_id": "brick-a"}),
    )
    targets = derive_pre_phase_targets(
        support, shared, {robot: context for robot in ("piper_0", "piper_1")}
    )
    assert (
        targets["piper_0"]["tcp_pose"]["position"][2]
        > shared["bricks"]["brick-a"]["aabb_max"][2]
    )
    assert targets["piper_0"]["moveit_pose_goal"] is True
    assert targets["piper_0"]["roll_flip_equivalent"] is False


def test_support_top_carrier_target_defers_without_roll_flip():
    """The carried preplace is a single-branch MoveIt pose goal."""
    shared = document()
    shared["robots"]["piper_1"] = {
        **shared["robots"]["piper_0"],
        "robot_id": "piper_1",
        "tcp_pose": {
            "position_xyz": [0.0, 0.05, 0.028],
            "orientation_xyzw": [1.0, 0.0, 0.0, 0.0],
        },
    }
    shared["bricks"]["brick-b"] = {
        "aabb_min": [-0.01, 0.04, 0.0],
        "aabb_max": [0.01, 0.06, 0.02],
        "pose": {"position_xyz": [0.0, 0.05, 0.01],
                 "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
    }
    shared["connections"] = {"brick-a|brick-b": {"connected": False}}
    shared["placement_interfaces"] = {
        "pairs": {
            "brick-a|brick-b": {
                "lower_to_upper": {
                    "position_xyz": [0.0, 0.032, 0.0192],
                    "orientation_xyzw": [0.0, 0.0, 0.7071, 0.7071],
                },
                "allowed_upper_yaw_quarter_turns": [0, 2],
            }
        }
    }
    context = TransitContext(
        RobotBaseFrame((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        TcpProvenance("a" * 64, "b" * 64),
        SkillParams(),
    )
    support = replace(
        operation(),
        skill="SUPPORT_TOP",
        target=OperationTarget(
            {
                "upper_physical_instance_id": "brick-a",
                "lower_physical_instance_id": "brick-b",
                "placement_robot_id": "piper_1",
            }
        ),
    )
    targets = derive_pre_phase_targets(
        support, shared, {robot: context for robot in ("piper_0", "piper_1")}
    )
    assert targets["piper_0"]["moveit_pose_goal"] is True
    assert targets["piper_0"]["roll_flip_equivalent"] is False
    assert targets["piper_1"]["moveit_pose_goal"] is True
    assert targets["piper_1"]["roll_flip_equivalent"] is False
    assert "tcp_pose" in targets["piper_1"]


def test_wire_target_is_unwrapped_from_asdict_shape():
    """The asdict wire target must bind its values, not the wrapper mapping."""
    shared = document()
    shared["robots"]["piper_1"] = {**shared["robots"]["piper_0"], "robot_id": "piper_1"}
    context = TransitContext(
        RobotBaseFrame((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        TcpProvenance("a" * 64, "b" * 64),
        SkillParams(),
    )
    contexts = {robot: asdict(context) for robot in ("piper_0", "piper_1")}
    wire = asdict(
        replace(
            operation(),
            skill="SUPPORT_TOP",
            target=OperationTarget({"upper_physical_instance_id": "brick-a"}),
        )
    )
    assert wire["target"] == {"values": {"upper_physical_instance_id": "brick-a"}}
    targets = derive_wire_targets(wire, shared, contexts)
    assert (
        targets["piper_0"]["tcp_pose"]["position"][2]
        > shared["bricks"]["brick-a"]["aabb_max"][2]
    )
    unwrapped = {**wire, "target": {"upper_physical_instance_id": "brick-a"}}
    with pytest.raises(ValueError, match="wire operation target"):
        derive_wire_targets(unwrapped, shared, contexts)
    extra_key = {**wire, "target": {"values": {"a": 1}, "meta": True}}
    with pytest.raises(ValueError, match="wire operation target"):
        derive_wire_targets(extra_key, shared, contexts)
