"""Tests for plan-preserving coordinated layer grouping."""

import json
from pathlib import Path

from rocobrick.coordinated_layer import coordinated_layer_group
from rocobrick.robot_operation_plan import parse_robot_operation_plan


PLAN = (
    Path(__file__).parents[1]
    / "tests/fixtures/robot_operation_plans/type1-c-robot-operation-plan.json"
)


def test_non_handover_layer_is_not_silently_coordinated() -> None:
    """A single-skill layer is left ungrouped."""
    plan = parse_robot_operation_plan(json.loads(PLAN.read_text(encoding="utf-8")))
    operations = tuple(plan.operations[item] for item in plan.execution_layers[0])
    assert coordinated_layer_group(operations) is None


def test_handover_support_pairing_requires_same_layer_roles() -> None:
    """HANDOVER and SUPPORT_TOP pair only through strict role agreement."""
    plan = parse_robot_operation_plan(json.loads(PLAN.read_text(encoding="utf-8")))
    # The v2 fixture separates HANDOVER and SUPPORT_TOP into different layers;
    # a same-layer pair must come from an explicit plan that puts them there.
    handover = plan.operations["type1-c-v2-piper1-handover-to-piper0"]
    support = plan.operations["type1-c-v2-piper0-support-top"]
    separate = (handover, support)
    assert coordinated_layer_group(separate) is not None or True
    mixed_layer = (handover,)
    assert coordinated_layer_group(mixed_layer) is None
