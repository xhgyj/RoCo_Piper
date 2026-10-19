"""Strict parser for the external ``robot-task-plan/v2`` document.

The parser accepts the design-example shape only: root keys ``schema``,
``plan_id``, ``operations``, and ``dependencies``. Operation array order never
expresses dependency. Legacy layer/schedule fields, old plan schemas,
``HANDOVER``, duplicate identities, malformed participants, and malformed
dependencies fail closed. Nothing is repaired, defaulted, reordered, or bound
to a structure model. Optional skill parameters remain opaque offline JSON
data and grant no motion authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .dependency_graph import DependencyGraph
from .schema import (
    PHYSICAL_SUCCESS,
    PLACEMENT_SKILLS,
    REJECTED_SKILLS,
    SUPPORTED_SKILLS,
    TASK_PLAN_SCHEMA,
    TaskDependency,
    TaskOperation,
    TaskPlan,
    TaskPlanError,
    freeze_parameters,
)

_TOP_KEYS = frozenset({"schema", "plan_id", "operations", "dependencies"})
_OPERATION_REQUIRED = frozenset({"id", "skill", "executor", "participants"})
_OPERATION_OPTIONAL = frozenset({"object_id", "target_ref", "parameters"})
_DEPENDENCY_KEYS = frozenset({"before", "after", "condition"})


def _object(value: object, path: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise TaskPlanError(f"{path} must be an object")
    return value


def _array(value: object, path: str) -> list[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TaskPlanError(f"{path} must be an array")
    return list(value)


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaskPlanError(f"{path} must be a non-empty string")
    return value


def _keys(
    value: Mapping[object, object],
    required: frozenset[str],
    optional: frozenset[str],
    path: str,
) -> None:
    if any(not isinstance(key, str) for key in value):
        raise TaskPlanError(f"{path} keys must be strings")
    actual = frozenset(key for key in value if isinstance(key, str))
    missing = sorted(required - actual)
    unknown = sorted(actual - required - optional)
    if missing or unknown:
        raise TaskPlanError(
            f"{path} keys mismatch: missing {missing}, unexpected {unknown}"
        )


def _participants(value: object, path: str) -> tuple[str, ...]:
    items = tuple(_text(item, f"{path}[]") for item in _array(value, path))
    if not items:
        raise TaskPlanError(f"{path} must not be empty")
    if len(set(items)) != len(items):
        raise TaskPlanError(f"{path} contains duplicate robot IDs")
    return items


def _operation(value: object, index: int) -> TaskOperation:
    path = f"operations[{index}]"
    item = _object(value, path)
    _keys(item, _OPERATION_REQUIRED, _OPERATION_OPTIONAL, path)
    operation_id = _text(item["id"], f"{path}.id")
    skill = _text(item["skill"], f"{path}.skill")
    if skill in REJECTED_SKILLS:
        raise TaskPlanError(
            f"{path}.skill {skill!r} is explicitly rejected by "
            "robot-task-plan/v2 offline mode"
        )
    if skill not in SUPPORTED_SKILLS:
        raise TaskPlanError(f"{path}.skill {skill!r} is unsupported")
    executor = _text(item["executor"], f"{path}.executor")
    participants = _participants(item["participants"], f"{path}.participants")
    if executor not in participants:
        raise TaskPlanError(
            f"{path}.participants must include the executor {executor!r}"
        )
    object_id = (
        _text(item["object_id"], f"{path}.object_id") if "object_id" in item else None
    )
    if skill != "WAIT" and object_id is None:
        raise TaskPlanError(f"{path}.object_id is required for {skill}")
    target_ref = (
        _text(item["target_ref"], f"{path}.target_ref")
        if "target_ref" in item
        else None
    )
    if skill in PLACEMENT_SKILLS and target_ref is None:
        raise TaskPlanError(
            f"{path}.target_ref is required for placement skill {skill}"
        )
    if skill == "WAIT" and target_ref is not None:
        raise TaskPlanError(f"{path}.target_ref is not valid for WAIT")
    parameters = freeze_parameters(
        item.get("parameters", {}),
        f"{path}.parameters",
        allow_tuple=False,
    )
    return TaskOperation(
        id=operation_id,
        skill=skill,
        executor=executor,
        participants=participants,
        object_id=object_id,
        target_ref=target_ref,
        parameters=parameters,
    )


def _dependencies(
    value: object, operations: Mapping[str, TaskOperation]
) -> tuple[TaskDependency, ...]:
    dependencies: list[TaskDependency] = []
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(_array(value, "dependencies")):
        path = f"dependencies[{index}]"
        item = _object(raw, path)
        _keys(item, _DEPENDENCY_KEYS, frozenset(), path)
        before = _text(item["before"], f"{path}.before")
        after = _text(item["after"], f"{path}.after")
        condition = _text(item["condition"], f"{path}.condition")
        if condition != PHYSICAL_SUCCESS:
            raise TaskPlanError(f"{path}.condition must be {PHYSICAL_SUCCESS!r}")
        if before not in operations or after not in operations:
            raise TaskPlanError(
                f"{path} dependency {before!r} -> {after!r} references an "
                "unknown operation"
            )
        if before == after:
            raise TaskPlanError(f"{path} operation {before!r} cannot depend on itself")
        edge = (before, after)
        if edge in seen:
            raise TaskPlanError(f"{path} duplicates dependency {before!r} -> {after!r}")
        seen.add(edge)
        dependencies.append(TaskDependency(before, after, condition))
    return tuple(dependencies)


def parse_task_plan(document: object) -> TaskPlan:
    """Validate and freeze one ``robot-task-plan/v2`` document.

    Args:
        document: Decoded plan document; it is copied and never mutated.

    Returns:
        The immutable task plan, preserving operation array order as data only.

    Raises:
        TaskPlanError: If the schema, shape, identities, participants,
            placement targets, dependencies, or dependency graph are invalid.
    """
    root = _object(document, "plan")
    if "schema" not in root:
        _keys(root, _TOP_KEYS, frozenset(), "plan")
    schema = _text(root["schema"], "plan.schema")
    if schema != TASK_PLAN_SCHEMA:
        raise TaskPlanError(
            f"plan.schema {schema!r} is unsupported; expected {TASK_PLAN_SCHEMA!r}"
        )
    _keys(root, _TOP_KEYS, frozenset(), "plan")
    plan_id = _text(root["plan_id"], "plan.plan_id")
    operations: list[TaskOperation] = []
    seen_ids: set[str] = set()
    for index, value in enumerate(_array(root["operations"], "plan.operations")):
        operation = _operation(value, index)
        if operation.id in seen_ids:
            raise TaskPlanError(f"duplicate operation id {operation.id!r}")
        seen_ids.add(operation.id)
        operations.append(operation)
    if not operations:
        raise TaskPlanError("plan.operations must not be empty")
    by_id = {operation.id: operation for operation in operations}
    dependencies = _dependencies(root["dependencies"], by_id)
    plan = TaskPlan(
        schema=TASK_PLAN_SCHEMA,
        plan_id=plan_id,
        operations=tuple(operations),
        dependencies=dependencies,
    )
    DependencyGraph(plan)
    return plan
