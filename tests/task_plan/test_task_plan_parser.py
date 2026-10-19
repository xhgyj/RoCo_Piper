"""Focused offline tests for the strict robot-task-plan/v2 parser."""

from __future__ import annotations

import copy
import json

import pytest
from plan_documents import design_example

from rocobrick.task_plan import (
    PHYSICAL_SUCCESS,
    TaskPlanError,
    parse_task_plan,
)


def _operation(document: dict[str, object], operation_id: str) -> dict[str, object]:
    operations = document["operations"]
    assert isinstance(operations, list)
    for operation in operations:
        assert isinstance(operation, dict)
        if operation["id"] == operation_id:
            return operation
    raise AssertionError(f"missing operation {operation_id!r}")


def test_design_example_freezes_expected_fields() -> None:
    """Parse the design example into the agreed frozen DTO fields."""
    plan = parse_task_plan(design_example())
    assert plan.schema == "robot-task-plan/v2"
    assert plan.plan_id == "assembly-example"
    assert [operation.id for operation in plan.operations] == [
        "pick_a",
        "place_a",
        "pick_b",
    ]
    pick = plan.operations[0]
    assert pick.skill == "PICK"
    assert pick.executor == "piper_0"
    assert pick.participants == ("piper_0",)
    assert pick.object_id == "brick_a"
    assert pick.target_ref is None
    assert dict(pick.parameters) == {}
    place = plan.operations[1]
    assert place.target_ref == "slot_a"
    dependency = plan.dependencies[0]
    assert (dependency.before, dependency.after) == ("pick_a", "place_a")
    assert dependency.condition == PHYSICAL_SUCCESS


def test_robot_ids_are_opaque_and_not_limited_to_two_arms() -> None:
    """Accept arbitrary non-empty robot identities selected by the planner."""
    document = design_example()
    operation = _operation(document, "pick_a")
    operation["executor"] = "arm-left-7"
    operation["participants"] = ["arm-left-7", "observer-unit"]
    plan = parse_task_plan(document)
    assert plan.operations[0].executor == "arm-left-7"
    assert plan.operations[0].participants == ("arm-left-7", "observer-unit")


def test_parameters_are_deeply_frozen_and_roundtrip_json_safe() -> None:
    """Freeze nested parameters and reparse a JSON-safe roundtrip."""
    document = design_example()
    _operation(document, "pick_a")["parameters"] = {
        "speed": 0.25,
        "enabled": True,
        "count": 3,
        "profile": {"name": "fine", "limits": [1, 2]},
    }
    plan = parse_task_plan(document)
    parameters = plan.operations[0].parameters
    profile = parameters["profile"]
    assert profile["name"] == "fine"
    assert profile["limits"] == (1, 2)
    assert parameters["enabled"] is True
    assert type(parameters["count"]) is int
    with pytest.raises(TypeError):
        parameters["speed"] = 1.0  # type: ignore[index]
    with pytest.raises(TypeError):
        profile["name"] = "coarse"  # type: ignore[index]
    payload = plan.to_dict()
    json.dumps(payload, sort_keys=True)
    assert parse_task_plan(payload) == plan
    reparsed = parse_task_plan(payload)
    assert reparsed.operations[0].parameters["profile"]["limits"] == (1, 2)


def test_parser_does_not_mutate_or_alias_input() -> None:
    """Copy all input data so later caller mutation cannot change the plan."""
    document = design_example()
    _operation(document, "pick_a")["parameters"] = {"profile": {"limits": [1]}}
    before = copy.deepcopy(document)
    plan = parse_task_plan(document)
    assert document == before
    _operation(document, "pick_a")["participants"].append("intruder")
    _operation(document, "pick_a")["parameters"]["profile"]["limits"].append(99)
    dependencies = document["dependencies"]
    assert isinstance(dependencies, list)
    dependencies[0]["after"] = "pick_b"
    assert plan.operations[0].participants == ("piper_0",)
    assert plan.operations[0].parameters["profile"]["limits"] == (1,)
    assert plan.dependencies[0].after == "place_a"


@pytest.mark.parametrize(
    "value",
    [{1, 2}, b"bytes", float("nan"), float("inf"), object(), (1, 2)],
)
def test_parameters_reject_non_json_values(value: object) -> None:
    """Reject values that are not JSON-safe or not finite."""
    document = design_example()
    _operation(document, "pick_a")["parameters"] = {"bad": value}
    with pytest.raises(TaskPlanError, match="JSON-safe|finite"):
        parse_task_plan(document)


def test_parameters_require_object_and_string_keys() -> None:
    """Reject non-object parameter containers and non-string keys."""
    document = design_example()
    _operation(document, "pick_a")["parameters"] = []
    with pytest.raises(TaskPlanError, match="must be an object"):
        parse_task_plan(document)
    document = design_example()
    _operation(document, "pick_a")["parameters"] = {"nested": {1: "bad"}}
    with pytest.raises(TaskPlanError, match="keys must be non-empty strings"):
        parse_task_plan(document)


def test_wait_allows_optional_object_and_rejects_target() -> None:
    """WAIT may omit object_id but must not carry a placement target."""
    document = design_example()
    operations = document["operations"]
    assert isinstance(operations, list)
    operations.append(
        {
            "id": "wait_1",
            "skill": "WAIT",
            "executor": "piper_0",
            "participants": ["piper_0"],
        }
    )
    plan = parse_task_plan(document)
    assert plan.operations[-1].object_id is None
    assert plan.operations[-1].target_ref is None
    operations[-1]["object_id"] = "brick_a"
    assert parse_task_plan(document).operations[-1].object_id == "brick_a"
    operations[-1]["target_ref"] = "slot_a"
    with pytest.raises(TaskPlanError, match="target_ref"):
        parse_task_plan(document)


@pytest.mark.parametrize("skill", ["PLACE_DOWN", "PLACE_UP", "SUPPORT_TOP"])
def test_placement_skills_require_target_ref(skill: str) -> None:
    """Every placement skill requires an opaque target reference."""
    document = design_example()
    operation = _operation(document, "place_a")
    operation["skill"] = skill
    operation.pop("target_ref")
    with pytest.raises(TaskPlanError, match="target_ref"):
        parse_task_plan(document)


@pytest.mark.parametrize("skill", ["PICK", "PLACE_DOWN"])
def test_non_wait_skills_require_object_id(skill: str) -> None:
    """PICK and placement skills require an opaque object identity."""
    document = design_example()
    operation = _operation(document, "pick_a")
    operation["skill"] = skill
    operation.pop("object_id")
    with pytest.raises(TaskPlanError, match="object_id"):
        parse_task_plan(document)


def test_optional_references_reject_explicit_null() -> None:
    """Optional references may be absent but explicit null is malformed."""
    document = design_example()
    _operation(document, "pick_a")["object_id"] = None
    with pytest.raises(TaskPlanError, match="object_id"):
        parse_task_plan(document)
    document = design_example()
    _operation(document, "place_a")["target_ref"] = None
    with pytest.raises(TaskPlanError, match="target_ref"):
        parse_task_plan(document)


@pytest.mark.parametrize(
    ("participants", "message"),
    [
        ([], "must not be empty"),
        (["piper_1"], "must include the executor"),
        (["piper_0", "piper_0"], "duplicate robot IDs"),
        ("piper_0", "must be an array"),
        ([7], "non-empty string"),
    ],
)
def test_participants_are_strict(participants: object, message: str) -> None:
    """Validate membership, nonempty uniqueness, and string identities."""
    document = design_example()
    _operation(document, "pick_a")["participants"] = participants
    with pytest.raises(TaskPlanError, match=message):
        parse_task_plan(document)


def test_duplicate_operation_ids_are_rejected() -> None:
    """Reject duplicate operation identities even with identical payloads."""
    document = design_example()
    operations = document["operations"]
    assert isinstance(operations, list)
    operations.append(copy.deepcopy(operations[2]))
    with pytest.raises(TaskPlanError, match="duplicate operation id"):
        parse_task_plan(document)


@pytest.mark.parametrize(
    ("dependency", "message"),
    [
        (
            {"before": "pick_a", "after": "pick_a", "condition": PHYSICAL_SUCCESS},
            "cannot depend on itself",
        ),
        (
            {"before": "missing", "after": "place_a", "condition": PHYSICAL_SUCCESS},
            "unknown operation",
        ),
        (
            {"before": "pick_a", "after": "missing", "condition": PHYSICAL_SUCCESS},
            "unknown operation",
        ),
        (
            {"before": "pick_a", "after": "place_a", "condition": "phase_done"},
            "condition must be",
        ),
        (
            {"before": "pick_a", "after": "place_a", "condition": True},
            "condition",
        ),
    ],
)
def test_dependencies_are_strict(dependency: dict[str, object], message: str) -> None:
    """Validate dependency identity, self-edges, and physical-success only."""
    document = design_example()
    document["dependencies"] = [dependency]
    with pytest.raises(TaskPlanError, match=message):
        parse_task_plan(document)


def test_duplicate_dependency_edges_are_rejected() -> None:
    """Reject a repeated before/after edge."""
    document = design_example()
    dependencies = document["dependencies"]
    assert isinstance(dependencies, list)
    dependencies.append(copy.deepcopy(dependencies[0]))
    with pytest.raises(TaskPlanError, match="duplicates dependency"):
        parse_task_plan(document)


def test_dependency_unknown_keys_are_rejected() -> None:
    """Reject legacy or arbitrary dependency metadata."""
    document = design_example()
    dependencies = document["dependencies"]
    assert isinstance(dependencies, list)
    dependencies[0]["reason"] = "legacy"
    with pytest.raises(TaskPlanError, match="unexpected"):
        parse_task_plan(document)


def test_handover_is_explicitly_rejected() -> None:
    """Reject HANDOVER before any offline v2 scheduling can see it."""
    document = design_example()
    operations = document["operations"]
    assert isinstance(operations, list)
    operations.append(
        {
            "id": "handover_1",
            "skill": "HANDOVER",
            "executor": "piper_0",
            "participants": ["piper_0", "piper_1"],
            "object_id": "brick_a",
        }
    )
    with pytest.raises(TaskPlanError, match="HANDOVER"):
        parse_task_plan(document)


def test_dependency_cycle_is_rejected() -> None:
    """Reject cycles before returning a frozen plan."""
    document = design_example()
    document["dependencies"] = [
        {
            "before": "pick_a",
            "after": "place_a",
            "condition": PHYSICAL_SUCCESS,
        },
        {
            "before": "place_a",
            "after": "pick_a",
            "condition": PHYSICAL_SUCCESS,
        },
    ]
    with pytest.raises(TaskPlanError, match="cycle"):
        parse_task_plan(document)


def test_legacy_schema_and_layer_fields_are_rejected() -> None:
    """Reject old schemas and never ignore execution layers or schedules."""
    old = design_example()
    old["schema"] = "robot-operation-plan/v1"
    old["execution_layers"] = [["pick_a"], ["place_a"], ["pick_b"]]
    with pytest.raises(TaskPlanError, match="unsupported"):
        parse_task_plan(old)
    layered = design_example()
    layered["execution_layers"] = [["pick_a"], ["place_a"]]
    with pytest.raises(TaskPlanError, match="execution_layers"):
        parse_task_plan(layered)
    scheduled = design_example()
    scheduled["planned_schedule"] = [[{"operation_id": "pick_a"}]]
    with pytest.raises(TaskPlanError, match="planned_schedule"):
        parse_task_plan(scheduled)


def test_legacy_operation_wire_fields_are_rejected() -> None:
    """Reject old operation key names instead of translating them."""
    document = design_example()
    operation = _operation(document, "pick_a")
    operation["operation_id"] = operation.pop("id")
    with pytest.raises(TaskPlanError, match="operation_id"):
        parse_task_plan(document)


@pytest.mark.parametrize("document", [None, [], "plan", 7])
def test_document_must_be_an_object(document: object) -> None:
    """Reject non-object documents."""
    with pytest.raises(TaskPlanError, match="plan must be an object"):
        parse_task_plan(document)


def test_schema_and_root_shape_are_strict() -> None:
    """Require the exact v2 schema and exact root key set."""
    missing = design_example()
    missing.pop("schema")
    with pytest.raises(TaskPlanError, match="missing"):
        parse_task_plan(missing)
    wrong_type = design_example()
    wrong_type["schema"] = 2
    with pytest.raises(TaskPlanError, match="schema"):
        parse_task_plan(wrong_type)
    extra = design_example()
    extra["robots"] = ["piper_0", "piper_1"]
    with pytest.raises(TaskPlanError, match="robots"):
        parse_task_plan(extra)


def test_operations_container_and_empty_plan_are_strict() -> None:
    """Require a nonempty operations array of objects."""
    wrong = design_example()
    wrong["operations"] = {}
    with pytest.raises(TaskPlanError, match="must be an array"):
        parse_task_plan(wrong)
    empty = design_example()
    empty["operations"] = []
    with pytest.raises(TaskPlanError, match="must not be empty"):
        parse_task_plan(empty)
    bad_item = design_example()
    bad_item["operations"] = [[]]
    with pytest.raises(TaskPlanError, match="must be an object"):
        parse_task_plan(bad_item)


@pytest.mark.parametrize("plan_id", ["", None, 7, True])
def test_plan_id_must_be_nonempty_text(plan_id: object) -> None:
    """Reject missing or malformed plan identity, without bool coercion."""
    document = design_example()
    document["plan_id"] = plan_id
    with pytest.raises(TaskPlanError, match="plan_id"):
        parse_task_plan(document)


def test_json_roundtrip_preserves_dependencies_and_order() -> None:
    """Roundtrip the full DTO without losing array order or dependency data."""
    document = design_example()
    plan = parse_task_plan(document)
    restored = parse_task_plan(plan.to_dict())
    assert restored == plan
    assert [operation.id for operation in restored.operations] == [
        "pick_a",
        "place_a",
        "pick_b",
    ]
    assert restored.dependencies == plan.dependencies


def test_bool_and_number_parameters_keep_their_json_types() -> None:
    """Keep booleans, integers, and floats as distinct JSON value types."""
    document = design_example()
    _operation(document, "pick_a")["parameters"] = {
        "flag": True,
        "count": 1,
        "ratio": 1.0,
    }
    parameters = parse_task_plan(document).operations[0].parameters
    assert parameters["flag"] is True
    assert type(parameters["count"]) is int
    assert type(parameters["ratio"]) is float
