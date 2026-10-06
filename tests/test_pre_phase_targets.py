"""Pre-contact target derivation from one immutable shared scene."""

from dataclasses import replace

import pytest
from test_pick_handler import document, operation

from rocobrick.adapters.moveit.transit import TransitContext
from rocobrick.robot_operation_plan import OperationTarget
from rocobrick.skills.pre_phase_targets import derive_pre_phase_targets
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
