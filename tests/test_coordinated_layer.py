"""Tests for plan-preserving coordinated layer grouping."""

import json
from pathlib import Path

from rocobrick.coordinated_layer import coordinated_layer_group
from rocobrick.robot_operation_plan import parse_robot_operation_plan


PLAN = Path(__file__).parents[1] / "artifacts/scene/type1-c-robot-operation-plan.json"


def test_task_c_transport_layer_pairs_discrete_operations() -> None:
    """TRANSIT and SUPPORT_TOP remain separate ids but share transport grouping."""
    plan = parse_robot_operation_plan(json.loads(PLAN.read_text(encoding="utf-8")))
    operations = tuple(plan.operations[item] for item in plan.execution_layers[2])
    group = coordinated_layer_group(operations)
    assert group is not None
    assert group.operation_ids == (
        "type1-c-piper0-transit-to-place",
        "type1-c-piper1-support-top",
    )
    assert group.robot_ids == ("piper_1", "piper_0")
    assert group.connection_id == "conn-type1-c-lower-upper"


def test_nonpaired_layer_is_not_silently_coordinated() -> None:
    """A layer with no exact connection/role pair is left ungrouped."""
    plan = parse_robot_operation_plan(json.loads(PLAN.read_text(encoding="utf-8")))
    operations = tuple(plan.operations[item] for item in plan.execution_layers[0])
    assert coordinated_layer_group(operations) is None
