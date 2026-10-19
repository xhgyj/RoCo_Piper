"""Frozen data transfer objects for the ``robot-task-plan/v2`` contract.

The v2 document contains only task identity, executor assignment, opaque
physical references, optional JSON-safe skill parameters, and hard
``physical_success`` dependencies. It carries no execution layers, schedule,
geometry, robot allocation policy, or execution-framework imports. Optional
skill parameters are opaque offline JSON data; this contract does not
interpret them as motion authority or validate them as executable skill
configuration. This is an offline ingress contract only; it is not a
production dispatch integration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

TASK_PLAN_SCHEMA = "robot-task-plan/v2"
PHYSICAL_SUCCESS = "physical_success"
SUPPORTED_SKILLS = frozenset({"PICK", "PLACE_DOWN", "PLACE_UP", "SUPPORT_TOP", "WAIT"})
PLACEMENT_SKILLS = frozenset({"PLACE_DOWN", "PLACE_UP", "SUPPORT_TOP"})
REJECTED_SKILLS = frozenset({"HANDOVER"})


class TaskPlanError(ValueError):
    """Raised when a robot-task-plan/v2 document violates the frozen shape."""


def _freeze_json(value: object, path: str, *, allow_tuple: bool) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TaskPlanError(f"{path} must be a finite JSON number")
        return value
    if isinstance(value, Mapping):
        return freeze_parameters(value, path, allow_tuple=allow_tuple)
    if isinstance(value, list) or (allow_tuple and isinstance(value, tuple)):
        return tuple(
            _freeze_json(item, f"{path}[]", allow_tuple=allow_tuple) for item in value
        )
    raise TaskPlanError(f"{path} must be a JSON-safe value, got {type(value).__name__}")


def freeze_parameters(
    value: object, path: str, *, allow_tuple: bool = True
) -> Mapping[str, object]:
    """Validate and deeply freeze one JSON-safe parameter mapping.

    Args:
        value: Candidate parameter object; it is copied, never aliased.
        path: Diagnostic path used in validation errors.
        allow_tuple: Accept tuples as already-frozen JSON arrays. The
            default keeps direct dataclass construction idempotent; the
            wire parser passes ``False`` to reject non-JSON input tuples.

    Returns:
        A read-only mapping whose nested mappings and arrays are frozen too.

    Raises:
        TaskPlanError: If a key is not a non-empty string, a value is not
            JSON-safe, or a number is not finite.
    """
    if not isinstance(value, Mapping):
        raise TaskPlanError(f"{path} must be an object")
    frozen: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise TaskPlanError(f"{path} keys must be non-empty strings")
        frozen[key] = _freeze_json(item, f"{path}.{key}", allow_tuple=allow_tuple)
    return MappingProxyType(frozen)


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _nonempty_text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaskPlanError(f"{path} must be a non-empty string")
    return value


def _optional_text(value: object, path: str) -> str | None:
    if value is None:
        return None
    return _nonempty_text(value, path)


def _frozen_identities(value: object, path: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TaskPlanError(f"{path} must be an array of robot IDs")
    try:
        items = tuple(value)
    except TypeError as error:
        raise TaskPlanError(f"{path} must be an array of robot IDs") from error
    if not items:
        raise TaskPlanError(f"{path} must not be empty")
    frozen: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        identity = _nonempty_text(item, f"{path}[{index}]")
        if identity in seen:
            raise TaskPlanError(f"{path} contains duplicate robot IDs")
        seen.add(identity)
        frozen.append(identity)
    return tuple(frozen)


def _frozen_sequence(value: object, item_type: type, path: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TaskPlanError(f"{path} must be an array")
    try:
        items = tuple(value)
    except TypeError as error:
        raise TaskPlanError(f"{path} must be an array") from error
    for index, item in enumerate(items):
        if not isinstance(item, item_type):
            raise TaskPlanError(f"{path}[{index}] must be {item_type.__name__}")
    return items


@dataclass(frozen=True)
class TaskOperation:
    """One immutable operation in a ``robot-task-plan/v2`` document.

    Attributes:
        id: Unique operation identity from the wire ``id`` field.
        skill: One of the supported offline v2 skills.
        executor: Opaque robot identity selected by the external planner.
        participants: Opaque robot identities participating in the operation.
        object_id: Opaque physical object identity, optional only for WAIT.
        target_ref: Opaque placement reference, required for placement skills.
        parameters: Opaque deeply frozen JSON-safe skill payload. It is
            copied for immutability but never interpreted as motion
            authority or as validated executable skill configuration.
    """

    id: str
    skill: str
    executor: str
    participants: tuple[str, ...]
    object_id: str | None = None
    target_ref: str | None = None
    parameters: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Return a fresh JSON-safe mapping in the frozen wire shape.

        Returns:
            A plain dictionary suitable for :func:`json.dumps` and reparsing.
        """
        payload: dict[str, object] = {
            "id": self.id,
            "skill": self.skill,
            "executor": self.executor,
            "participants": list(self.participants),
        }
        if self.object_id is not None:
            payload["object_id"] = self.object_id
        if self.target_ref is not None:
            payload["target_ref"] = self.target_ref
        if self.parameters:
            payload["parameters"] = _thaw_json(self.parameters)
        return payload

    def __post_init__(self) -> None:
        """Validate fields and freeze participants and parameters."""
        object.__setattr__(self, "id", _nonempty_text(self.id, "operation.id"))
        skill = _nonempty_text(self.skill, "operation.skill")
        if skill in REJECTED_SKILLS:
            raise TaskPlanError(
                f"skill {skill!r} is not supported by robot-task-plan/v2 offline mode"
            )
        if skill not in SUPPORTED_SKILLS:
            raise TaskPlanError(
                f"unsupported skill {skill!r}; expected one of "
                f"{sorted(SUPPORTED_SKILLS)}"
            )
        executor = _nonempty_text(self.executor, "operation.executor")
        participants = _frozen_identities(self.participants, "operation.participants")
        if executor not in participants:
            raise TaskPlanError("operation.participants must include the executor")
        object_id = _optional_text(self.object_id, "operation.object_id")
        target_ref = _optional_text(self.target_ref, "operation.target_ref")
        if skill == "WAIT":
            if target_ref is not None:
                raise TaskPlanError("operation.target_ref is not allowed for WAIT")
        else:
            if object_id is None:
                raise TaskPlanError(
                    "operation.object_id is required for non-WAIT skills"
                )
            if skill in PLACEMENT_SKILLS and target_ref is None:
                raise TaskPlanError(
                    "operation.target_ref is required for placement skills"
                )
        object.__setattr__(self, "skill", skill)
        object.__setattr__(self, "executor", executor)
        object.__setattr__(self, "participants", participants)
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "target_ref", target_ref)
        object.__setattr__(
            self,
            "parameters",
            freeze_parameters(self.parameters, "operation.parameters"),
        )


@dataclass(frozen=True)
class TaskDependency:
    """One hard external dependency between two operations.

    The only accepted condition is ``physical_success``: the successor may not
    advance merely because an internal phase of its predecessor finished.
    """

    before: str
    after: str
    condition: str

    def __post_init__(self) -> None:
        """Validate identity and the physical-success condition."""
        before = _nonempty_text(self.before, "dependency.before")
        after = _nonempty_text(self.after, "dependency.after")
        condition = _nonempty_text(self.condition, "dependency.condition")
        if condition != PHYSICAL_SUCCESS:
            raise TaskPlanError("dependency.condition must be 'physical_success'")
        if before == after:
            raise TaskPlanError(f"operation {before!r} depends on itself")
        object.__setattr__(self, "before", before)
        object.__setattr__(self, "after", after)
        object.__setattr__(self, "condition", condition)

    def to_dict(self) -> dict[str, object]:
        """Return a fresh JSON-safe dependency mapping.

        Returns:
            A plain dictionary in the frozen wire shape.
        """
        return {
            "before": self.before,
            "after": self.after,
            "condition": self.condition,
        }


@dataclass(frozen=True)
class TaskPlan:
    """Immutable ``robot-task-plan/v2`` document with no scheduling semantics.

    Attributes:
        schema: Exact frozen schema identifier.
        plan_id: External plan identity.
        operations: Operations in document array order, never dependency order.
        dependencies: Explicit hard dependencies; array order is not implicit
            dependency information.
    """

    schema: str
    plan_id: str
    operations: tuple[TaskOperation, ...]
    dependencies: tuple[TaskDependency, ...]

    def __post_init__(self) -> None:
        """Validate roots and freeze operation/dependency sequences."""
        schema = _nonempty_text(self.schema, "plan.schema")
        if schema != TASK_PLAN_SCHEMA:
            raise TaskPlanError(
                f"unsupported task plan schema {schema!r}; "
                f"expected {TASK_PLAN_SCHEMA!r}"
            )
        plan_id = _nonempty_text(self.plan_id, "plan.plan_id")
        operations = _frozen_sequence(self.operations, TaskOperation, "plan.operations")
        if not operations:
            raise TaskPlanError("plan.operations must not be empty")
        dependencies = _frozen_sequence(
            self.dependencies, TaskDependency, "plan.dependencies"
        )
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "plan_id", plan_id)
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "dependencies", dependencies)

    def to_dict(self) -> dict[str, object]:
        """Return a fresh JSON-safe plan mapping in the frozen wire shape.

        Returns:
            A plain dictionary suitable for :func:`json.dumps` and reparsing.
        """
        return {
            "schema": self.schema,
            "plan_id": self.plan_id,
            "operations": [operation.to_dict() for operation in self.operations],
            "dependencies": [dependency.to_dict() for dependency in self.dependencies],
        }
