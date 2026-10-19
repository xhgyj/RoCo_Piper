"""Focused tests for direct frozen DTO construction and graph defense."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from rocobrick.task_plan import (
    PHYSICAL_SUCCESS,
    DependencyGraph,
    TaskDependency,
    TaskOperation,
    TaskPlan,
    TaskPlanError,
    parse_task_plan,
)


def _make_operation(**overrides: object) -> TaskOperation:
    values: dict[str, object] = {
        "id": "A",
        "skill": "PICK",
        "executor": "robot",
        "participants": ["robot"],
        "object_id": "brick",
        "target_ref": None,
        "parameters": {},
    }
    values.update(overrides)
    return TaskOperation(**values)


def test_direct_operation_freezes_participants_and_parameters() -> None:
    """Direct construction must copy and freeze mutable inputs."""
    source = {"profile": {"limits": [1]}}
    operation = TaskOperation(
        id="A",
        skill="PICK",
        executor="robot",
        participants=["robot"],
        object_id="brick",
        target_ref=None,
        parameters=source,
    )
    assert operation.participants == ("robot",)
    assert isinstance(operation.parameters, MappingProxyType)
    assert operation.parameters["profile"]["limits"] == (1,)
    source["profile"]["limits"].append(2)
    assert operation.parameters["profile"]["limits"] == (1,)
    assert operation.to_dict()["parameters"] == {"profile": {"limits": [1]}}


def test_direct_operation_defaults_optional_fields_for_wait() -> None:
    """A WAIT operation may omit object, target, and parameters."""
    operation = TaskOperation("wait", "WAIT", "robot", ["robot"])
    assert operation.object_id is None
    assert operation.target_ref is None
    assert operation.parameters == {}
    assert operation.to_dict() == {
        "id": "wait",
        "skill": "WAIT",
        "executor": "robot",
        "participants": ["robot"],
    }


def test_direct_operation_can_rebuild_frozen_parser_output() -> None:
    """Re-freezing an already frozen parameter payload stays idempotent."""
    from plan_documents import design_example

    document = design_example()
    operations = document["operations"]
    assert isinstance(operations, list)
    operations[0]["parameters"] = {"profile": {"limits": [1]}}
    parsed = parse_task_plan(document).operations[0]
    rebuilt = TaskOperation(
        id=parsed.id,
        skill=parsed.skill,
        executor=parsed.executor,
        participants=parsed.participants,
        object_id=parsed.object_id,
        target_ref=parsed.target_ref,
        parameters=parsed.parameters,
    )
    assert rebuilt == parsed
    assert rebuilt.to_dict() == parsed.to_dict()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"skill": "HANDOVER"}, "HANDOVER"),
        ({"skill": "SPIN"}, "unsupported"),
        ({"id": "   "}, "non-empty"),
        ({"executor": "other"}, "include the executor"),
        ({"participants": []}, "must not be empty"),
        ({"participants": ["robot", "robot"]}, "duplicate robot IDs"),
        ({"object_id": None}, "object_id"),
        ({"skill": "PLACE_DOWN", "target_ref": None}, "target_ref"),
        ({"skill": "WAIT", "target_ref": "slot"}, "not allowed for WAIT"),
        ({"parameters": {"bad": {1, 2}}}, "JSON-safe"),
        ({"parameters": {"bad": float("nan")}}, "finite"),
    ],
)
def test_direct_operation_validation(
    overrides: dict[str, object], message: str
) -> None:
    """Invalid operation fields fail without going through the parser."""
    with pytest.raises(TaskPlanError, match=message):
        _make_operation(**overrides)


def test_direct_dependency_validation() -> None:
    """Dependencies enforce text identity, self-edges, and condition."""
    with pytest.raises(TaskPlanError, match="condition"):
        TaskDependency("A", "B", "phase_done")
    with pytest.raises(TaskPlanError, match="depends on itself"):
        TaskDependency("A", "A", PHYSICAL_SUCCESS)
    with pytest.raises(TaskPlanError, match="non-empty"):
        TaskDependency("", "B", PHYSICAL_SUCCESS)


def test_direct_plan_freezes_sequences_and_rejects_bad_roots() -> None:
    """TaskPlan normalizes lists but rejects old schemas and empty plans."""
    first = _make_operation(id="A")
    second = _make_operation(id="B")
    dependency = TaskDependency("A", "B", PHYSICAL_SUCCESS)
    plan = TaskPlan(
        schema="robot-task-plan/v2",
        plan_id="direct",
        operations=[first, second],
        dependencies=[dependency],
    )
    assert isinstance(plan.operations, tuple)
    assert isinstance(plan.dependencies, tuple)
    assert parse_task_plan(plan.to_dict()).to_dict() == plan.to_dict()
    with pytest.raises(TaskPlanError, match="unsupported task plan schema"):
        TaskPlan("robot-operation-plan/v1", "old", [first], [])
    with pytest.raises(TaskPlanError, match="must not be empty"):
        TaskPlan("robot-task-plan/v2", "empty", [], [])
    with pytest.raises(TaskPlanError, match="must be TaskOperation"):
        TaskPlan("robot-task-plan/v2", "bad", [{"id": "A"}], [])


def test_graph_rejects_bypassed_condition_even_for_frozen_plan() -> None:
    """Graph validates condition defensively when a constructor is bypassed."""
    first = _make_operation(id="A")
    second = _make_operation(id="B")
    dependency = TaskDependency("A", "B", PHYSICAL_SUCCESS)
    object.__setattr__(dependency, "condition", "phase_done")
    plan = TaskPlan(
        schema="robot-task-plan/v2",
        plan_id="bypassed",
        operations=(first, second),
        dependencies=(dependency,),
    )
    with pytest.raises(TaskPlanError, match="unsupported condition"):
        DependencyGraph(plan)


def test_parser_rejects_whitespace_id_and_raw_tuple_parameters() -> None:
    """Wire parsing rejects whitespace-only IDs and non-JSON tuples."""
    from plan_documents import design_example

    whitespace = design_example()
    operations = whitespace["operations"]
    assert isinstance(operations, list)
    operations[0]["id"] = "   "
    with pytest.raises(TaskPlanError, match="non-empty"):
        parse_task_plan(whitespace)
    tupled = design_example()
    tupled_operations = tupled["operations"]
    assert isinstance(tupled_operations, list)
    tupled_operations[0]["parameters"] = {"profile": (1, 2)}
    with pytest.raises(TaskPlanError, match="got tuple"):
        parse_task_plan(tupled)
