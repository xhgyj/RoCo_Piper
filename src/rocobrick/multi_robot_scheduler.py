"""Frozen execution-result contracts and the multi-robot operation scheduler.

This module owns the A2 scheduling seam for the zero-motion phase:

* :class:`OperationExecution` and :class:`CancelEvidence` are the only formal
  results accepted from a :class:`RobotOperationExecutor`. They are frozen,
  validate real booleans and JSON-safe evidence at construction time, and are
  never replaced by truthiness, a sent ``asyncio.Task.cancel()`` call, or a
  finished sleep.
* :class:`MultiRobotScheduler` dispatches one layer concurrently, barriers on
  the whole layer, leases the resources declared by each operation before
  dispatch, releases them only after a confirmed terminal result or an
  acknowledged terminal *and* safe cancel, blocks later layers after a failure,
  and keeps every run's state isolated from previous runs.
* ``planned_schedule`` controls evidence display order only; it never controls
  dispatch. Dispatch order is ``execution_layers``.
* Legacy executors returning ``bool``/``Mapping``/objects with a boolean
  ``passed`` attribute are accepted through an explicitly marked compatibility
  shim (``allow_legacy_results``) so pre-contract in-memory stubs keep working;
  their results are recorded with ``legacy=True`` and the
  ``legacy_unvalidated`` evidence marker. Strict mode rejects them.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from rocobrick.compat import StrEnum
from rocobrick.coordinated_layer import coordinated_layer_group
from rocobrick.robot_operation_plan import RobotOperation, RobotOperationPlan
from rocobrick.unified_dual_arm_runtime import (
    SharedSceneSnapshot,
    SuccessorPrePhaseEvidence,
    SuccessorPrePhaseRequest,
    TerminalPrePhasePolicy,
)

JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]


class ExecutionContractError(ValueError):
    """Raised when a formal execution/cancel result violates the contract."""


class OperationStatus(StrEnum):
    """Lifecycle/result status of one scheduled robot operation."""

    READY = "READY"
    RUNNING = "RUNNING"
    CONFIRMED = "CONFIRMED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    BLOCKED = "BLOCKED"


def is_json_safe(value: object) -> bool:
    """Return whether ``value`` can be serialized as strict JSON.

    Args:
        value: Any value. ``Mapping`` keys must be strings; tuples are treated
            as arrays; non-finite floats (NaN/inf) are rejected.

    Returns:
        True when :func:`to_json_value` can copy the value and
        ``json.dumps(..., allow_nan=False)`` can serialize the copy.
    """
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str) and is_json_safe(item) for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return all(is_json_safe(item) for item in value)
    return False


def to_json_value(value: object) -> JSONValue:
    """Copy a JSON-safe value into plain JSON structures.

    Args:
        value: A value accepted by :func:`is_json_safe`.

    Returns:
        A ``dict``/``list``/scalar copy that ``json.dumps`` accepts directly.

    Raises:
        ValueError: If ``value`` is not JSON-safe.
    """
    if not is_json_safe(value):
        raise ValueError("value is not JSON-safe")
    return _copy_json(value)


def _copy_json(value: object) -> JSONValue:
    if isinstance(value, Mapping):
        return {str(key): _copy_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_copy_json(item) for item in value]
    return value  # type: ignore[return-value]


def _legacy_summary(value: object) -> JSONValue:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, Mapping):
        return {str(key): _legacy_summary(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_legacy_summary(item) for item in value]
    return repr(value)


@dataclass(frozen=True)
class OperationExecution:
    """Frozen formal result of one dispatched :class:`RobotOperation`.

    Attributes:
        operation_id: Operation this result belongs to; the scheduler rejects
            a result whose id does not match the dispatched operation.
        robot_id: Robot that produced the result; must match the operation's
            ``executor_robot_id``.
        status: Final :class:`OperationStatus`. A ``CONFIRMED`` result must
            declare ``passed is True`` and ``terminal is True``.
        passed: Explicit boolean success. Only an actual ``True`` is accepted;
            the scheduler never derives it from truthiness.
        terminal: True only when the executor reached a final state. The
            scheduler never equates ``asyncio.Task.cancel()`` with a terminal
            state.
        motion_sent: Explicit boolean claim about sent motion commands.
        evidence: JSON-safe evidence mapping. Every label required by the
            operation and the plan must be present with an explicit value
            before the result can be confirmed.
        reason: Optional human-readable detail; must be ``str`` or ``None``.
    """

    operation_id: str
    robot_id: str
    status: OperationStatus
    passed: bool
    terminal: bool
    motion_sent: bool
    evidence: Mapping[str, JSONValue]
    reason: str | None = None

    def __post_init__(self) -> None:
        """Validate the frozen formal-result fields.

        Raises:
            ExecutionContractError: If any field violates the contract.
        """
        if not isinstance(self.operation_id, str) or not self.operation_id:
            raise ExecutionContractError("operation_id must be a non-empty str")
        if not isinstance(self.robot_id, str) or not self.robot_id:
            raise ExecutionContractError("robot_id must be a non-empty str")
        if not isinstance(self.status, OperationStatus):
            raise ExecutionContractError("status must be an OperationStatus")
        for name in ("passed", "terminal", "motion_sent"):
            if not isinstance(getattr(self, name), bool):
                raise ExecutionContractError(f"{name} must be a real bool")
        if not isinstance(self.evidence, Mapping):
            raise ExecutionContractError("evidence must be a mapping")
        if not is_json_safe(self.evidence):
            raise ExecutionContractError("evidence must be JSON-safe")
        if self.reason is not None and not isinstance(self.reason, str):
            raise ExecutionContractError("reason must be str or None")


@dataclass(frozen=True)
class CancelEvidence:
    """Frozen acknowledgement of one ``cancel(operation_id)`` request.

    Attributes:
        operation_id: Operation the acknowledgement refers to; the scheduler
            rejects mismatched acknowledgements.
        acknowledged: True only when the underlying executor explicitly
            acknowledged the cancel request.
        terminal: True only when the underlying runtime reached a final state.
        safe: True only when the underlying executor explicitly reports the
            cancelled resource as safe to release.
        evidence: JSON-safe evidence mapping produced by the executor.
    """

    operation_id: str
    acknowledged: bool
    terminal: bool
    safe: bool
    evidence: Mapping[str, JSONValue]

    def __post_init__(self) -> None:
        """Validate the frozen cancel-evidence fields.

        Raises:
            ExecutionContractError: If any field violates the contract.
        """
        if not isinstance(self.operation_id, str) or not self.operation_id:
            raise ExecutionContractError("operation_id must be a non-empty str")
        for name in ("acknowledged", "terminal", "safe"):
            if not isinstance(getattr(self, name), bool):
                raise ExecutionContractError(f"{name} must be a real bool")
        if not isinstance(self.evidence, Mapping):
            raise ExecutionContractError("evidence must be a mapping")
        if not is_json_safe(self.evidence):
            raise ExecutionContractError("evidence must be JSON-safe")


class RobotOperationExecutor(Protocol):
    """Formal executor contract consumed by the scheduler and runner."""

    async def execute(self, operation: RobotOperation) -> OperationExecution:
        """Execute one operation and return a frozen formal result.

        Args:
            operation: The validated operation to execute.

        Returns:
            The terminal :class:`OperationExecution` for ``operation``.
        """

    async def execute_layer(
        self, operations: Sequence[RobotOperation]
    ) -> Mapping[str, object]:
        """Optionally execute one same-layer coordinated transport group.

        Returns:
            A mapping keyed by operation id with formal operation results.
        """

    async def cancel(self, operation_id: str) -> CancelEvidence:
        """Request cancellation and return the terminal/safe acknowledgement.

        Args:
            operation_id: Identifier of the operation to cancel.

        Returns:
            The :class:`CancelEvidence` produced by the underlying runtime.
        """


class OperationExecutor(Protocol):
    """Legacy executor seam kept for pre-contract in-memory stubs.

    Results are coerced by the compatibility shim and recorded with
    ``legacy=True``; strict mode (``allow_legacy_results=False``) rejects them.
    """

    async def execute(self, operation: RobotOperation) -> object:
        """Execute one operation and return a legacy result.

        Args:
            operation: The validated operation to execute.

        Returns:
            A legacy ``bool``/``Mapping``/object result.
        """


class OperationCanceler(Protocol):
    """Legacy cancel seam kept for pre-contract in-memory stubs."""

    async def cancel(self, operation_id: str) -> object:
        """Request cancellation and return a legacy acknowledgement.

        Args:
            operation_id: Identifier of the operation to cancel.

        Returns:
            A legacy acknowledgement value.
        """


@dataclass(frozen=True)
class OperationResult:
    """Per-operation scheduler outcome used as run evidence.

    Attributes:
        operation_id: Scheduled operation id.
        robot_id: Robot the operation was dispatched to.
        status: Scheduler status for the operation.
        passed: True only for confirmed terminal success.
        detail: JSON-safe audit detail.
        terminal: True only when a final state was explicitly confirmed.
        motion_sent: Explicit boolean claim, or ``None`` when unknown (for
            example after the scheduler cancelled an in-flight operation).
        evidence: JSON-safe evidence collected for the operation.
        reason: Human-readable outcome detail.
        legacy: True when the result came from the compatibility shim.
        resources: Resources declared by the operation.
        resources_released: True only when all declared resources were
            confirmed released.
        cancel: Cancel acknowledgement when a cancel was requested.
        deadline_exceeded: True when the operation deadline expired before a
            terminal state was observed; the operation and run cannot pass.
        late_terminal_confirmation: True when a validated terminal result
            arrived after the deadline; it may release resources but cannot
            restore deadline success.
    """

    operation_id: str
    robot_id: str
    status: OperationStatus
    passed: bool
    detail: JSONValue = None
    terminal: bool = False
    motion_sent: bool | None = None
    evidence: Mapping[str, JSONValue] = field(default_factory=dict)
    reason: str | None = None
    legacy: bool = False
    resources: tuple[str, ...] = ()
    resources_released: bool = False
    cancel: CancelEvidence | None = None
    deadline_exceeded: bool = False
    late_terminal_confirmation: bool = False


@dataclass(frozen=True)
class ResourceEvidence:
    """Lease record for one resource declared by one operation."""

    resource: str
    operation_id: str
    robot_id: str
    acquired: bool
    released: bool
    reason: str | None = None


@dataclass(frozen=True)
class LayerResult:
    """Outcome of one ``execution_layers`` barrier."""

    layer_index: int
    status: OperationStatus
    operation_ids: tuple[str, ...]
    reason: str | None = None


@dataclass(frozen=True)
class SchedulerResult:
    """Frozen result of one scheduler run.

    Attributes:
        operations: Per-operation outcomes keyed by operation id.
        layers: Per-layer outcomes in execution order.
        passed: True only when every operation is confirmed terminal, every
            required evidence label was present, no unsafe/unknown terminal
            was recorded, and every acquired resource was released.
        evidence: JSON-safe log entries; operation entries follow
            ``planned_schedule`` display order while layers follow execution
            order.
        resources: Lease records for every declared resource.
        cancellations: Cancel acknowledgements collected in this run.
        unsafe_terminals: Operation ids whose terminal state or cancel safety
            was not confirmed; their resources stay held and the run fails.
    """

    operations: Mapping[str, OperationResult]
    layers: tuple[LayerResult, ...]
    passed: bool
    evidence: tuple[Mapping[str, JSONValue], ...] = field(default_factory=tuple)
    resources: tuple[ResourceEvidence, ...] = field(default_factory=tuple)
    cancellations: tuple[CancelEvidence, ...] = field(default_factory=tuple)
    unsafe_terminals: tuple[str, ...] = ()


def _required_evidence(
    plan: RobotOperationPlan, operation: RobotOperation
) -> tuple[str, ...]:
    labels: list[str] = []
    for label in operation.evidence_requirements:
        if label not in labels:
            labels.append(label)
    for operation_id, label in plan.evidence_requirements:
        if operation_id == operation.operation_id and label not in labels:
            labels.append(label)
    return tuple(labels)


def _evidence_present(evidence: Mapping[str, JSONValue], label: str) -> bool:
    if label not in evidence:
        return False
    value = evidence[label]
    return value is not None and value is not False


def _is_confirmation(result: OperationResult) -> bool:
    return (
        result.status is OperationStatus.CONFIRMED and result.passed and result.terminal
    )


def _cancel_releases(cancel: CancelEvidence | None) -> bool:
    return (
        cancel is not None and cancel.acknowledged and cancel.terminal and cancel.safe
    )


def _requires_attention(result: OperationResult) -> bool:
    if _is_confirmation(result):
        return False
    if result.late_terminal_confirmation and result.terminal:
        return False
    if result.cancel is not None:
        return result.motion_sent is True or not _cancel_releases(result.cancel)
    return not result.terminal or result.motion_sent is True


def _predecessors(plan: RobotOperationPlan) -> dict[str, tuple[str, ...]]:
    predecessors: dict[str, list[str]] = {}
    for dependency in plan.dependencies:
        predecessors.setdefault(dependency.successor_operation_id, []).append(
            dependency.predecessor_operation_id
        )
    return {oid: tuple(preds) for oid, preds in predecessors.items()}


def _schedule_order(plan: RobotOperationPlan) -> dict[str, tuple[int, int]]:
    order: dict[str, tuple[int, int]] = {}
    for slot, entries in enumerate(plan.planned_schedule):
        for index, entry in enumerate(entries):
            order[entry.operation_id] = (slot, index)
    return order


def _cancel_document(cancel: CancelEvidence) -> dict[str, JSONValue]:
    return {
        "operation_id": cancel.operation_id,
        "acknowledged": cancel.acknowledged,
        "terminal": cancel.terminal,
        "safe": cancel.safe,
        "evidence": to_json_value(dict(cancel.evidence)),
    }


def _failure_result(
    operation: RobotOperation, reason: str, *, terminal: bool = False
) -> OperationResult:
    return OperationResult(
        operation_id=operation.operation_id,
        robot_id=operation.executor_robot_id,
        status=OperationStatus.FAILED,
        passed=False,
        detail=reason,
        terminal=terminal,
        motion_sent=None,
        evidence={},
        reason=reason,
        resources=operation.resources,
    )


def _blocked_result(operation: RobotOperation, reason: str) -> OperationResult:
    return OperationResult(
        operation_id=operation.operation_id,
        robot_id=operation.executor_robot_id,
        status=OperationStatus.BLOCKED,
        passed=False,
        detail=reason,
        terminal=False,
        motion_sent=None,
        evidence={},
        reason=reason,
        resources=operation.resources,
    )


def _cancelled_result(
    operation: RobotOperation, cancel: CancelEvidence, timed_out: bool
) -> OperationResult:
    if timed_out:
        reason = "operation deadline exceeded; cancel requested"
    else:
        reason = "cancelled after a same-layer peer failed"
    if _cancel_releases(cancel):
        reason += "; cancel acknowledged terminal and safe"
    elif cancel.acknowledged:
        reason += "; cancel acknowledged but terminal/safe not confirmed"
    else:
        reason += "; cancel not acknowledged"
    return OperationResult(
        operation_id=operation.operation_id,
        robot_id=operation.executor_robot_id,
        status=OperationStatus.CANCELLED,
        passed=False,
        detail=reason,
        terminal=cancel.terminal,
        motion_sent=None,
        evidence={"cancel": _cancel_document(cancel)},
        reason=reason,
        resources=operation.resources,
        cancel=cancel,
        deadline_exceeded=timed_out,
    )


def _deadline_settlement(
    operation: RobotOperation, settled: OperationResult
) -> OperationResult:
    evidence = dict(settled.evidence)
    evidence["deadline_exceeded"] = True
    if _is_confirmation(settled):
        evidence["late_terminal_confirmation"] = True
        reason = (
            "operation deadline exceeded; late terminal confirmation cannot "
            "restore deadline success"
        )
        return OperationResult(
            operation_id=operation.operation_id,
            robot_id=operation.executor_robot_id,
            status=OperationStatus.CANCELLED,
            passed=False,
            detail=reason,
            terminal=True,
            motion_sent=settled.motion_sent,
            evidence=evidence,
            reason=reason,
            legacy=settled.legacy,
            resources=operation.resources,
            deadline_exceeded=True,
            late_terminal_confirmation=True,
        )
    reason = f"operation deadline exceeded; {settled.reason or settled.status.value}"
    return dataclasses.replace(
        settled,
        detail=reason,
        reason=reason,
        evidence=evidence,
        deadline_exceeded=True,
    )


def _legacy_cancel(operation_id: str, value: object) -> CancelEvidence:
    acknowledged = False
    if isinstance(value, bool):
        acknowledged = value
    elif isinstance(value, Mapping) and isinstance(value.get("acknowledged"), bool):
        acknowledged = bool(value["acknowledged"])
    elif isinstance(value, Mapping) and isinstance(value.get("passed"), bool):
        acknowledged = bool(value["passed"])
    elif isinstance(getattr(value, "acknowledged", None), bool):
        acknowledged = bool(getattr(value, "acknowledged"))
    return CancelEvidence(
        operation_id=operation_id,
        acknowledged=acknowledged,
        terminal=False,
        safe=False,
        evidence={"legacy_cancel": True, "returned": _legacy_summary(value)},
    )


def _resolve_cancel(
    canceler: object | None, executor: object
) -> Callable[[str], object] | None:
    if canceler is not None:
        candidate = getattr(canceler, "cancel", None)
        if callable(candidate):
            return candidate
    candidate = getattr(executor, "cancel", None)
    return candidate if callable(candidate) else None


@dataclass
class _RunState:
    results: dict[str, OperationResult] = field(default_factory=dict)
    layers: list[LayerResult] = field(default_factory=list)
    held: dict[str, str] = field(default_factory=dict)
    resource_records: dict[tuple[str, str], ResourceEvidence] = field(
        default_factory=dict
    )
    cancellations: list[CancelEvidence] = field(default_factory=list)
    unsafe: list[str] = field(default_factory=list)
    deadline_events: list[Mapping[str, JSONValue]] = field(default_factory=list)
    failed: bool = False
    failure_reasons: list[str] = field(default_factory=list)
    pre_phases: dict[str, Mapping[str, JSONValue]] = field(default_factory=dict)
    transitioning: set[str] = field(default_factory=set)

    def reserve(
        self, operation: RobotOperation, predecessor_ids: tuple[str, ...]
    ) -> str | None:
        """Reserve new resources without changing any predecessor owner.

        Returns:
            A conflict reason, or None when the reservation is complete.
        """
        conflicts = [
            resource
            for resource in operation.resources
            if resource in self.held
            and self.held[resource] not in (*predecessor_ids, operation.operation_id)
        ]
        if conflicts:
            return f"held by other operations: {sorted(conflicts)}"
        for resource in operation.resources:
            if resource not in self.held:
                self.held[resource] = operation.operation_id
                self.resource_records[(operation.operation_id, resource)] = (
                    ResourceEvidence(
                        resource,
                        operation.operation_id,
                        operation.executor_robot_id,
                        True,
                        False,
                        "reserved before successor pre-phase",
                    )
                )
        return None

    def adopt(
        self, request: SuccessorPrePhaseRequest, successor: RobotOperation | None
    ) -> None:
        """Atomically transfer proven physical leases without an unowned interval."""
        if any(
            self.held.get(resource) != owner
            for resource, owner in request.lease_owners.items()
        ):
            raise ExecutionContractError("predecessor lease changed during pre-phase")
        for resource, owner in request.lease_owners.items():
            record = self.resource_records[(owner, resource)]
            self.resource_records[(owner, resource)] = dataclasses.replace(
                record,
                released=True,
                reason=f"adopted by {request.transition_id}",
            )
            if successor is None:
                del self.held[resource]
            else:
                self.held[resource] = successor.operation_id
                self.resource_records[(successor.operation_id, resource)] = (
                    ResourceEvidence(
                        resource,
                        successor.operation_id,
                        successor.executor_robot_id,
                        True,
                        False,
                        f"atomic adoption from {owner}",
                    )
                )
        for owner in request.predecessor_operation_ids:
            result = self.results[owner]
            evidence = dict(result.evidence)
            evidence["physical_departure_confirmed"] = request.transition_id
            self.results[owner] = dataclasses.replace(
                result,
                resources_released=True,
                evidence=evidence,
            )

    def acquire(self, operation: RobotOperation) -> str | None:
        conflicts = sorted(
            {
                resource
                for resource in operation.resources
                if resource in self.held
                and self.held[resource] != operation.operation_id
            }
        )
        if conflicts:
            reason = f"held by other operations: {conflicts}"
            for resource in operation.resources:
                self.resource_records[(operation.operation_id, resource)] = (
                    ResourceEvidence(
                        resource=resource,
                        operation_id=operation.operation_id,
                        robot_id=operation.executor_robot_id,
                        acquired=False,
                        released=False,
                        reason=reason,
                    )
                )
            return reason
        for resource in operation.resources:
            self.held[resource] = operation.operation_id
            self.resource_records.setdefault(
                (operation.operation_id, resource),
                ResourceEvidence(
                    resource=resource,
                    operation_id=operation.operation_id,
                    robot_id=operation.executor_robot_id,
                    acquired=True,
                    released=False,
                    reason="acquired before dispatch",
                ),
            )
        return None

    def release(self, operation: RobotOperation, reason: str) -> None:
        resources = tuple(
            resource
            for resource, owner in self.held.items()
            if owner == operation.operation_id
        )
        for resource in resources:
            del self.held[resource]
            key = (operation.operation_id, resource)
            record = self.resource_records.get(key)
            if record is not None and record.acquired and not record.released:
                self.resource_records[key] = dataclasses.replace(
                    record, released=True, reason=reason
                )

    def mark_failure(self, reason: str) -> None:
        self.failed = True
        self.failure_reasons.append(reason)

    def note_attention(self, operation_id: str) -> None:
        if operation_id not in self.unsafe:
            self.unsafe.append(operation_id)


def _build_evidence(
    plan: RobotOperationPlan, state: _RunState
) -> list[Mapping[str, JSONValue]]:
    entries: list[Mapping[str, JSONValue]] = []
    for layer in state.layers:
        entries.append(
            {
                "kind": "layer",
                "layer_index": layer.layer_index,
                "operation_ids": list(layer.operation_ids),
                "status": layer.status.value,
                "passed": layer.status is OperationStatus.CONFIRMED,
                "reason": layer.reason,
            }
        )
    order = _schedule_order(plan)
    fallback = (len(plan.planned_schedule), 0)
    for operation_id in sorted(
        state.results, key=lambda oid: (order.get(oid, fallback), oid)
    ):
        item = state.results[operation_id]
        entries.append(
            {
                "kind": "operation",
                "operation_id": item.operation_id,
                "robot_id": item.robot_id,
                "status": item.status.value,
                "passed": item.passed,
                "terminal": item.terminal,
                "motion_sent": item.motion_sent,
                "legacy": item.legacy,
                "evidence": to_json_value(dict(item.evidence)),
                "reason": item.reason,
                "resources": list(item.resources),
                "resources_released": item.resources_released,
                "deadline_exceeded": item.deadline_exceeded,
                "late_terminal_confirmation": item.late_terminal_confirmation,
                "cancel": (
                    _cancel_document(item.cancel) if item.cancel is not None else None
                ),
            }
        )
    entries.extend(state.deadline_events)
    return entries


class MultiRobotScheduler:
    """Scheduler for validated ``robot-operation-plan/v1`` documents.

    Every :meth:`schedule` call builds its own state, so one scheduler instance
    can run several plans without leaking cancelled ids, held resources, or
    evidence between runs.

    Args:
        operation_timeout_s: Per-layer (hence per-operation) deadline in
            seconds, measured from dispatch. ``None`` disables the bound.
        cancel_timeout_s: Bound for one cancel acknowledgement round-trip.
        allow_legacy_results: When True, non-contract executor results are
            accepted through the compatibility shim and marked ``legacy``.
        layer_timeout_s_by_index: Optional finite skill-layer deadline overrides;
            other layers retain operation_timeout_s. Coordinated execute_layer
            groups own their lifecycle deadlines and cannot use this override.
    """

    def __init__(
        self,
        *,
        operation_timeout_s: float | None = 30.0,
        cancel_timeout_s: float = 5.0,
        allow_legacy_results: bool = True,
        layer_timeout_s_by_index: Mapping[int, float] | None = None,
    ) -> None:
        """Validate bounds and store the scheduler configuration.

        Raises:
            ValueError: If a timeout bound is not a positive number.
        """
        if operation_timeout_s is not None and (
            not isinstance(operation_timeout_s, (int, float))
            or isinstance(operation_timeout_s, bool)
            or operation_timeout_s <= 0
        ):
            raise ValueError("operation_timeout_s must be a positive number or None")
        if (
            not isinstance(cancel_timeout_s, (int, float))
            or isinstance(cancel_timeout_s, bool)
            or cancel_timeout_s <= 0
        ):
            raise ValueError("cancel_timeout_s must be a positive number")
        self._operation_timeout_s = operation_timeout_s
        self._cancel_timeout_s = float(cancel_timeout_s)
        self._allow_legacy_results = allow_legacy_results
        overrides = {} if layer_timeout_s_by_index is None else layer_timeout_s_by_index
        if not isinstance(overrides, Mapping) or any(
            type(index) is not int
            or index < 0
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 < value <= 3600
            for index, value in overrides.items()
        ):
            raise ValueError(
                "layer timeout overrides require indices and finite bounds"
            )
        self._layer_timeout_s_by_index = dict(overrides)

    async def schedule(
        self,
        plan: RobotOperationPlan,
        executor: RobotOperationExecutor | OperationExecutor,
        canceler: OperationCanceler | RobotOperationExecutor | None = None,
    ) -> SchedulerResult:
        """Run one plan with fresh per-run state.

        Args:
            plan: Parsed ``robot-operation-plan/v1``; ``execution_layers``
                controls dispatch and barriers, ``planned_schedule`` only
                controls evidence display order.
            executor: Formal executor, or a legacy stub when
                ``allow_legacy_results`` is enabled.
            canceler: Optional legacy cancel seam. When omitted, the
                executor's own ``cancel`` method is used if present.

        Returns:
            The frozen :class:`SchedulerResult` for this run.

        Raises:
            Exception: Only exceptions raised outside executor/cancel calls
                propagate; executor and cancel failures are recorded and fail
                the run closed.
        """
        for index in self._layer_timeout_s_by_index:
            if index >= len(plan.execution_layers):
                raise ValueError("layer timeout references an unavailable layer")
            if (
                coordinated_layer_group(
                    tuple(plan.operations[oid] for oid in plan.execution_layers[index])
                )
                is not None
            ):
                raise ValueError("coordinated layer owns its lifecycle deadline")
        state = _RunState()
        predecessors = _predecessors(plan)
        confirmed: set[str] = set()
        for layer_index, operation_ids in enumerate(plan.execution_layers):
            if state.failed:
                reason = "predecessor layer failed"
                for operation_id in operation_ids:
                    state.results[operation_id] = _blocked_result(
                        plan.operations[operation_id], reason
                    )
                state.layers.append(
                    LayerResult(
                        layer_index,
                        OperationStatus.BLOCKED,
                        tuple(operation_ids),
                        reason,
                    )
                )
                continue
            missing_predecessors = {
                operation_id: tuple(
                    predecessor
                    for predecessor in predecessors.get(operation_id, ())
                    if predecessor not in confirmed
                )
                for operation_id in operation_ids
            }
            if any(missing_predecessors.values()):
                blocked_reason = "predecessors not confirmed"
                for operation_id in operation_ids:
                    missing = missing_predecessors[operation_id]
                    reason = (
                        f"predecessors not confirmed: {list(missing)}"
                        if missing
                        else "peer blocked by an unconfirmed predecessor"
                    )
                    state.results[operation_id] = _blocked_result(
                        plan.operations[operation_id], reason
                    )
                state.mark_failure(blocked_reason)
                state.layers.append(
                    LayerResult(
                        layer_index,
                        OperationStatus.BLOCKED,
                        tuple(operation_ids),
                        blocked_reason,
                    )
                )
                continue
            layer_status, layer_results = await self._run_layer(
                plan, layer_index, operation_ids, executor, canceler, state
            )
            state.results.update(layer_results)
            if layer_status is OperationStatus.CONFIRMED:
                confirmed.update(
                    operation_id
                    for operation_id in operation_ids
                    if _is_confirmation(state.results[operation_id])
                )
            else:
                state.mark_failure(
                    f"layer {layer_index} did not confirm every operation"
                )
            state.layers.append(
                LayerResult(
                    layer_index,
                    layer_status,
                    tuple(operation_ids),
                    None
                    if layer_status is OperationStatus.CONFIRMED
                    else "layer failure",
                )
            )
        if not state.failed:
            await self._terminal_departures(plan, executor, state)
        operations = dict(state.results)
        acquired = [
            record for record in state.resource_records.values() if record.acquired
        ]
        passed = (
            not state.failed
            and set(operations) == set(plan.operations)
            and all(_is_confirmation(item) for item in operations.values())
            and not state.unsafe
            and all(record.released for record in acquired)
        )
        return SchedulerResult(
            operations=operations,
            layers=tuple(state.layers),
            passed=passed,
            evidence=tuple(_build_evidence(plan, state)),
            resources=tuple(state.resource_records.values()),
            cancellations=tuple(state.cancellations),
            unsafe_terminals=tuple(state.unsafe),
        )

    async def _run_layer(
        self,
        plan: RobotOperationPlan,
        layer_index: int,
        operation_ids: tuple[str, ...],
        executor: RobotOperationExecutor | OperationExecutor,
        canceler: OperationCanceler | RobotOperationExecutor | None,
        state: _RunState,
    ) -> tuple[OperationStatus, dict[str, OperationResult]]:
        coordinated = coordinated_layer_group(
            tuple(plan.operations[operation_id] for operation_id in operation_ids)
        )
        if coordinated is not None:
            return await self._run_coordinated_layer(
                plan,
                layer_index,
                operation_ids,
                executor,
                canceler,
                state,
            )
        results: dict[str, OperationResult] = {}
        running: dict[str, asyncio.Task[object]] = {}
        order = {
            operation_id: index for index, operation_id in enumerate(operation_ids)
        }
        for operation_id in operation_ids:
            operation = plan.operations[operation_id]
            predecessors = self._physical_predecessors(operation, plan, state)
            conflict = None if predecessors else state.acquire(operation)
            if conflict is not None:
                reason = f"resource unavailable before dispatch: {conflict}"
                results[operation_id] = _failure_result(operation, reason)
                state.mark_failure(reason)
                continue
            try:
                running[operation_id] = asyncio.create_task(
                    self._execute_successor(operation, plan, executor, state)
                )
            except Exception as exc:
                reason = f"dispatch failed for {operation_id!r}: {exc!r}"
                results[operation_id] = _failure_result(operation, reason)
                state.mark_failure(reason)
                state.note_attention(operation_id)
        if not running:
            return OperationStatus.FAILED, results
        owners = {task: oid for oid, task in running.items()}
        pending: set[asyncio.Task[object]] = set(running.values())
        layer_timeout = self._layer_timeout_s_by_index.get(
            layer_index, self._operation_timeout_s
        )
        deadline = (
            time.monotonic() + layer_timeout if layer_timeout is not None else None
        )
        failed = False
        timed_out = False
        try:
            while pending:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    timed_out = True
                    break
                done, pending = await asyncio.wait(
                    pending,
                    timeout=remaining,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    timed_out = True
                    break
                for task in sorted(done, key=lambda item: order[owners[item]]):
                    operation = plan.operations[owners[task]]
                    result = self._evaluate_task(task, operation, plan)
                    results[operation.operation_id] = result
                    if not _is_confirmation(result):
                        failed = True
                    if _requires_attention(result):
                        state.note_attention(operation.operation_id)
                if failed:
                    break
        except BaseException:
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            raise
        pending_ids = [
            operation_id
            for operation_id in operation_ids
            if operation_id in running and running[operation_id] in pending
        ]
        if timed_out:
            state.deadline_events.append(
                {
                    "kind": "deadline_exceeded",
                    "layer_index": layer_index,
                    "operation_ids": list(pending_ids),
                }
            )
        if pending_ids:
            await self._settle_pending(
                plan,
                pending_ids,
                running,
                executor,
                canceler,
                state,
                results,
                timed_out=timed_out,
            )
        for operation_id in operation_ids:
            operation = plan.operations[operation_id]
            result = results.get(operation_id)
            if result is None:
                continue
            released = False
            if result.evidence.get("handoff_required") is True:
                continue
            if _is_confirmation(result):
                state.release(operation, "terminal confirmation")
                released = True
            elif _cancel_releases(result.cancel) and result.motion_sent is not True:
                state.release(operation, "cancel acknowledged terminal and safe")
                released = True
            elif result.late_terminal_confirmation and result.terminal:
                # Terminal proof allows a safe release, but the deadline still
                # failed the operation and the run.
                state.release(operation, "late terminal confirmation after deadline")
                released = True
            if released and not result.resources_released:
                results[operation_id] = dataclasses.replace(
                    result, resources_released=True
                )
        confirmed_layer = bool(results) and all(
            _is_confirmation(item) for item in results.values()
        )
        return (
            OperationStatus.CONFIRMED if confirmed_layer else OperationStatus.FAILED,
            results,
        )

    async def _run_coordinated_layer(
        self,
        plan: RobotOperationPlan,
        layer_index: int,
        operation_ids: tuple[str, ...],
        executor: RobotOperationExecutor | OperationExecutor,
        canceler: OperationCanceler | RobotOperationExecutor | None,
        state: _RunState,
    ) -> tuple[OperationStatus, dict[str, OperationResult]]:
        """Dispatch one eligible layer through a single shared transport.

        Returns:
            Layer status and one result for every original operation id.
        """
        operations = tuple(
            plan.operations[operation_id] for operation_id in operation_ids
        )
        results: dict[str, OperationResult] = {}
        if any(
            self._physical_predecessors(operation, plan, state)
            for operation in operations
        ):
            reason = (
                "coordinated layer requires successor-owned unified pre-phase transport"
            )
            state.mark_failure(reason)
            return OperationStatus.FAILED, {
                operation.operation_id: _failure_result(operation, reason)
                for operation in operations
            }
        for operation in operations:
            conflict = state.acquire(operation)
            if conflict is not None:
                result = _failure_result(
                    operation,
                    f"resource unavailable before coordinated dispatch: {conflict}",
                )
                results[operation.operation_id] = result
                state.mark_failure(result.reason or "coordinated resource conflict")
        if results:
            return OperationStatus.FAILED, results
        dispatch = getattr(executor, "execute_layer", None)
        if not callable(dispatch):
            reason = (
                "coordinated layer requires executor.execute_layer; "
                "independent operation dispatch is forbidden"
            )
            for operation in operations:
                results[operation.operation_id] = _failure_result(operation, reason)
                state.note_attention(operation.operation_id)
            state.mark_failure(reason)
            return OperationStatus.FAILED, results
        try:
            raw = dispatch(operations)
            if inspect.isawaitable(raw):
                raw = await raw
            if not isinstance(raw, Mapping):
                raise TypeError("coordinated execute_layer must return a mapping")
            for operation in operations:
                value = raw.get(operation.operation_id)
                results[operation.operation_id] = self._evaluate_value(
                    value, operation, plan
                )
        except Exception as exc:  # noqa: BLE001 - preserve layer failure evidence
            reason = f"coordinated layer dispatch failed: {exc!r}"
            for operation in operations:
                results[operation.operation_id] = _failure_result(operation, reason)
                state.note_attention(operation.operation_id)
            state.mark_failure(reason)
        for operation in operations:
            result = results[operation.operation_id]
            if _is_confirmation(result):
                if result.evidence.get("handoff_required") is True:
                    continue
                state.release(operation, "coordinated terminal confirmation")
                results[operation.operation_id] = dataclasses.replace(
                    result, resources_released=True
                )
            elif _requires_attention(result):
                state.note_attention(operation.operation_id)
        confirmed = bool(results) and all(
            _is_confirmation(item) for item in results.values()
        )
        return (
            OperationStatus.CONFIRMED if confirmed else OperationStatus.FAILED,
            results,
        )

    def _physical_predecessors(
        self, operation: RobotOperation, plan: RobotOperationPlan, state: _RunState
    ) -> tuple[str, ...]:
        """Select retained endpoints for this successor only.

        Returns:
            Direct physical dependencies or the next shared-robot endpoint owners.
        """
        if operation.skill == "WAIT":
            return ()
        direct = set(_predecessors(plan).get(operation.operation_id, ()))
        selected = []
        for owner in dict.fromkeys(state.held.values()):
            result = state.results.get(owner)
            if result is None or result.evidence.get("handoff_required") is not True:
                continue
            predecessor = plan.operations[owner]
            if owner in direct or set(predecessor.participants) & set(
                operation.participants
            ):
                selected.append(owner)
        return tuple(selected)

    async def _execute_successor(
        self,
        operation: RobotOperation,
        plan: RobotOperationPlan,
        executor: object,
        state: _RunState,
    ) -> object:
        """Run a retained-state transition inside the successor dispatch.

        Returns:
            The contact execution result, or a structured pre-phase failure.
        """

        def failure(reason: str) -> OperationExecution:
            return OperationExecution(
                operation.operation_id,
                operation.executor_robot_id,
                OperationStatus.FAILED,
                False,
                False,
                False,
                {},
                reason,
            )

        predecessors = self._physical_predecessors(operation, plan, state)
        if predecessors:
            if set(predecessors) & state.transitioning:
                return failure("predecessor lease is already transitioning")
            state.transitioning.update(predecessors)
            try:
                conflict = state.reserve(operation, predecessors)
                if conflict:
                    return failure(conflict)
                if not await self._transition(
                    plan, predecessors, operation, executor, state
                ):
                    return failure(
                        "successor pre-phase failed; predecessor leases retained"
                    )
            finally:
                state.transitioning.difference_update(predecessors)
        conflict = state.acquire(operation)
        if conflict:
            return failure(f"resource unavailable before dispatch: {conflict}")
        execute = getattr(executor, "execute")
        raw = await execute(operation)
        if (
            isinstance(raw, OperationExecution)
            and operation.operation_id in state.pre_phases
        ):
            raw = dataclasses.replace(
                raw,
                evidence={
                    **raw.evidence,
                    "pre_phase": state.pre_phases[operation.operation_id],
                },
            )
        return raw

    async def _transition(
        self,
        plan: RobotOperationPlan,
        predecessor_ids: tuple[str, ...],
        successor: RobotOperation | None,
        executor: object,
        state: _RunState,
        policy: TerminalPrePhasePolicy | None = None,
    ) -> bool:
        """Execute a bound transition and atomically adopt its physical leases.

        Returns:
            True only after scene validation and physical arrival proof.
        """
        transition_id = (
            f"pre-phase:{successor.operation_id}"
            if successor
            else f"terminal:{policy.policy_id}"
        )
        execution_attempted = False
        try:
            observe = getattr(executor, "observe_pre_phase", None)
            execute = getattr(executor, "execute_pre_phase", None)
            if not callable(observe) or not callable(execute):
                raise ExecutionContractError(
                    "shared snapshot/pre-phase runtime unavailable"
                )
            operations = tuple(plan.operations[owner] for owner in predecessor_ids)
            participants = tuple(
                sorted(
                    {
                        robot
                        for operation in operations
                        for robot in operation.participants
                    }
                    | (
                        set(successor.participants)
                        if successor
                        else set(policy.participants)
                    )
                )
            )
            snapshot = await asyncio.wait_for(
                observe(participants), self._operation_timeout_s
            )
            if not isinstance(snapshot, SharedSceneSnapshot):
                raise ExecutionContractError(
                    "pre-phase requires one admitted shared snapshot"
                )
            owners = {
                resource: owner
                for resource, owner in state.held.items()
                if owner in predecessor_ids
            }
            request = SuccessorPrePhaseRequest(
                transition_id=transition_id,
                predecessor_operations=operations,
                predecessor_states={
                    owner: dict(state.results[owner].evidence)
                    for owner in predecessor_ids
                },
                next_operation=successor,
                participants=participants,
                resources=tuple(sorted(owners)),
                lease_owners=owners,
                scene_snapshot=snapshot,
                terminal_policy=policy,
            )
            execution_attempted = True
            task = asyncio.create_task(execute(request))
            raw = await asyncio.wait_for(task, self._operation_timeout_s)
            if (
                not isinstance(raw, SuccessorPrePhaseEvidence)
                or not raw.matches(request)
                or not raw.passed
            ):
                raise ExecutionContractError(
                    "pre-phase binding or physical arrival evidence failed"
                )
            robots = snapshot.document["robots"]
            release_support = set(policy.release_supporting) if policy else set()
            for robot_id in participants:
                facts = robots[robot_id]["facts"]
                if set(raw.final_facts[robot_id]) - set(facts):
                    raise ExecutionContractError(
                        "pre-phase cannot create physical facts"
                    )
                if any(
                    not isinstance(proof, Mapping) or not proof
                    for proof in raw.final_facts[robot_id].values()
                ):
                    raise ExecutionContractError(
                        "physical facts require explicit evidence"
                    )
                for key in facts:
                    if key.startswith("holding:") or (
                        key.startswith("supporting:")
                        and robot_id not in release_support
                    ):
                        if not raw.final_facts[robot_id].get(key):
                            raise ExecutionContractError(
                                f"pre-phase lost physical fact {robot_id}/{key}"
                            )
            validate = getattr(executor, "validate_pre_phase", None)
            if (
                not callable(validate)
                or await asyncio.wait_for(
                    validate(request, raw),
                    self._operation_timeout_s,
                )
                is not True
            ):
                raise ExecutionContractError(
                    "scene/version or ledger reconciliation failed"
                )
            state.adopt(request, successor)
            proof = to_json_value(dataclasses.asdict(raw))
            state.pre_phases[successor.operation_id if successor else transition_id] = (
                proof
            )
            state.deadline_events.append(
                {
                    "kind": "pre_phase_adoption",
                    "transition_id": transition_id,
                    "resource_adoption_confirmed": True,
                    "evidence": proof,
                }
            )
            return True
        except asyncio.CancelledError:
            if execution_attempted:
                await self._cancel_transition(executor, transition_id, state)
            state.mark_failure(f"{transition_id}: cancelled; leases retained")
            for owner in predecessor_ids:
                state.note_attention(owner)
            raise
        except Exception as error:
            if execution_attempted:
                await self._cancel_transition(executor, transition_id, state)
            reason = f"{transition_id}: {type(error).__name__}: {error}"
            state.mark_failure(reason)
            for owner in predecessor_ids:
                state.note_attention(owner)
                result = state.results[owner]
                state.results[owner] = dataclasses.replace(
                    result,
                    evidence={**result.evidence, "pre_phase_failure": reason},
                )
            state.deadline_events.append(
                {
                    "kind": "pre_phase_failure",
                    "transition_id": transition_id,
                    "reason": reason,
                }
            )
            return False

    async def _cancel_transition(
        self, executor: object, transition_id: str, state: _RunState
    ) -> None:
        """Record cancellation without releasing physical predecessor leases."""
        cancel = getattr(executor, "cancel_pre_phase", None)
        try:
            if not callable(cancel):
                raise ExecutionContractError("pre-phase cancel unavailable")
            raw = await asyncio.wait_for(cancel(transition_id), self._cancel_timeout_s)
            if not isinstance(raw, Mapping) or not is_json_safe(raw):
                raise ExecutionContractError("invalid pre-phase cancellation evidence")
            evidence = to_json_value(raw)
        except Exception as error:
            evidence = {"terminal": False, "safe": False, "error": repr(error)}
        state.deadline_events.append(
            {
                "kind": "pre_phase_cancel",
                "transition_id": transition_id,
                "evidence": evidence,
            }
        )

    async def _terminal_departures(
        self, plan: RobotOperationPlan, executor: object, state: _RunState
    ) -> None:
        """Require explicit policy for every remaining physical endpoint."""
        owners = tuple(dict.fromkeys(state.held.values()))
        if not owners:
            return
        policy_fn = getattr(executor, "terminal_pre_phase_policy", None)
        try:
            policy = (
                policy_fn(tuple(plan.operations[owner] for owner in owners))
                if callable(policy_fn)
                else None
            )
            if not isinstance(policy, TerminalPrePhasePolicy):
                raise ExecutionContractError(
                    "final endpoint has no explicit terminal pre-phase policy"
                )
        except Exception as error:
            state.mark_failure(str(error))
            for owner in owners:
                state.note_attention(owner)
                result = state.results[owner]
                state.results[owner] = dataclasses.replace(
                    result,
                    evidence={
                        **result.evidence,
                        "terminal_policy_failure": str(error),
                    },
                )
            return
        await self._transition(plan, owners, None, executor, state, policy)

    def _evaluate_task(
        self,
        task: asyncio.Task[object],
        operation: RobotOperation,
        plan: RobotOperationPlan,
    ) -> OperationResult:
        if task.cancelled():
            cancel = CancelEvidence(
                operation_id=operation.operation_id,
                acknowledged=False,
                terminal=False,
                safe=False,
                evidence={
                    "cancel_unavailable": "task cancelled without terminal evidence"
                },
            )
            return _cancelled_result(operation, cancel, False)
        exception = task.exception()
        if exception is not None:
            return _failure_result(operation, f"executor raised {exception!r}")
        return self._evaluate_value(task.result(), operation, plan)

    def _evaluate_value(
        self,
        value: object,
        operation: RobotOperation,
        plan: RobotOperationPlan,
    ) -> OperationResult:
        if isinstance(value, OperationExecution):
            return self._evaluate_execution(value, operation, plan)
        if not self._allow_legacy_results:
            return _failure_result(
                operation,
                f"non-contract executor result rejected: {type(value).__name__!r}",
            )
        return self._evaluate_legacy(value, operation)

    def _evaluate_execution(
        self,
        execution: OperationExecution,
        operation: RobotOperation,
        plan: RobotOperationPlan,
    ) -> OperationResult:
        if execution.operation_id != operation.operation_id:
            return _failure_result(
                operation,
                "result operation_id "
                f"{execution.operation_id!r} does not match dispatched "
                f"{operation.operation_id!r}",
            )
        if execution.robot_id != operation.executor_robot_id:
            return _failure_result(
                operation,
                "result robot_id "
                f"{execution.robot_id!r} does not match executor "
                f"{operation.executor_robot_id!r}",
            )
        if execution.status is OperationStatus.CONFIRMED:
            if not execution.passed:
                return _failure_result(
                    operation, "CONFIRMED result must declare passed=True"
                )
            if not execution.terminal:
                return _failure_result(
                    operation, "CONFIRMED result must declare terminal=True"
                )
            missing = [
                label
                for label in _required_evidence(plan, operation)
                if not _evidence_present(execution.evidence, label)
                and not (
                    execution.evidence.get("handoff_required") is True
                    and label == "retreat_confirmed"
                )
            ]
            if missing:
                return _failure_result(
                    operation,
                    f"missing required evidence labels: {missing}",
                )
            return OperationResult(
                operation_id=operation.operation_id,
                robot_id=operation.executor_robot_id,
                status=OperationStatus.CONFIRMED,
                passed=True,
                detail=execution.reason,
                terminal=True,
                motion_sent=execution.motion_sent,
                evidence=dict(execution.evidence),
                reason=execution.reason,
                resources=operation.resources,
                resources_released=execution.evidence.get("handoff_required")
                is not True,
            )
        if execution.status in (
            OperationStatus.FAILED,
            OperationStatus.CANCELLED,
        ):
            return OperationResult(
                operation_id=operation.operation_id,
                robot_id=operation.executor_robot_id,
                status=execution.status,
                passed=False,
                detail=execution.reason,
                terminal=execution.terminal,
                motion_sent=execution.motion_sent,
                evidence=dict(execution.evidence),
                reason=execution.reason
                or f"executor reported {execution.status.value}",
                resources=operation.resources,
            )
        return _failure_result(
            operation,
            "non-terminal executor status "
            f"{execution.status.value!r} is not a final result",
        )

    def _evaluate_legacy(
        self, value: object, operation: RobotOperation
    ) -> OperationResult:
        mapping = value if isinstance(value, Mapping) else None
        passed: bool | None = None
        if isinstance(value, bool):
            passed = value
        elif mapping is not None and isinstance(mapping.get("passed"), bool):
            passed = bool(mapping["passed"])
        elif isinstance(getattr(value, "passed", None), bool):
            passed = bool(getattr(value, "passed"))
        if passed is None:
            return _failure_result(
                operation,
                "legacy executor result has no explicit boolean 'passed'",
            )
        evidence: dict[str, JSONValue] = {"legacy_unvalidated": True}
        raw_evidence = (
            mapping.get("evidence")
            if mapping is not None
            else getattr(value, "evidence", None)
        )
        if raw_evidence is not None:
            evidence["legacy_evidence"] = _legacy_summary(raw_evidence)
        raw_motion = (
            mapping.get("motion_sent")
            if mapping is not None
            else getattr(value, "motion_sent", None)
        )
        motion_sent = raw_motion if isinstance(raw_motion, bool) else None
        return OperationResult(
            operation_id=operation.operation_id,
            robot_id=operation.executor_robot_id,
            status=(OperationStatus.CONFIRMED if passed else OperationStatus.FAILED),
            passed=passed,
            detail=_legacy_summary(value),
            terminal=True,
            motion_sent=motion_sent,
            evidence=evidence,
            reason=(
                "legacy executor result accepted without strict evidence validation"
            ),
            legacy=True,
            resources=operation.resources,
            resources_released=passed,
        )

    async def _settle_pending(
        self,
        plan: RobotOperationPlan,
        pending_ids: list[str],
        running: dict[str, asyncio.Task[object]],
        executor: RobotOperationExecutor | OperationExecutor,
        canceler: OperationCanceler | RobotOperationExecutor | None,
        state: _RunState,
        results: dict[str, OperationResult],
        *,
        timed_out: bool,
    ) -> None:
        cancel_fn = _resolve_cancel(canceler, executor)
        if cancel_fn is None:
            cancel_evidence = {
                operation_id: CancelEvidence(
                    operation_id=operation_id,
                    acknowledged=False,
                    terminal=False,
                    safe=False,
                    evidence={"cancel_unavailable": "no cancel seam was provided"},
                )
                for operation_id in pending_ids
            }
        else:
            gathered = await asyncio.gather(
                *(
                    self._request_cancel(cancel_fn, operation_id)
                    for operation_id in pending_ids
                ),
                return_exceptions=True,
            )
            cancel_evidence = {}
            for operation_id, item in zip(pending_ids, gathered):
                if isinstance(item, CancelEvidence):
                    cancel_evidence[operation_id] = item
                else:
                    cancel_evidence[operation_id] = CancelEvidence(
                        operation_id=operation_id,
                        acknowledged=False,
                        terminal=False,
                        safe=False,
                        evidence={"cancel_error": repr(item)},
                    )
        state.cancellations.extend(
            cancel_evidence[operation_id] for operation_id in pending_ids
        )
        for operation_id in pending_ids:
            task = running[operation_id]
            if not task.done():
                # asyncio cancellation only releases the coroutine; it is not
                # evidence that the underlying runtime stopped.
                task.cancel()
        await asyncio.gather(
            *(running[operation_id] for operation_id in pending_ids),
            return_exceptions=True,
        )
        for operation_id in pending_ids:
            operation = plan.operations[operation_id]
            task = running[operation_id]
            if task.done() and not task.cancelled():
                exception = task.exception()
                if exception is not None:
                    settled = _failure_result(
                        operation,
                        f"executor raised after cancel request: {exception!r}",
                    )
                else:
                    settled = self._evaluate_task(task, operation, plan)
                if timed_out:
                    settled = _deadline_settlement(operation, settled)
                if not _is_confirmation(settled):
                    settled = dataclasses.replace(
                        settled, cancel=cancel_evidence[operation_id]
                    )
                results[operation_id] = settled
            else:
                results[operation_id] = _cancelled_result(
                    operation, cancel_evidence[operation_id], timed_out
                )
            if _requires_attention(results[operation_id]):
                state.note_attention(operation_id)
            if not _is_confirmation(results[operation_id]):
                state.failed = True

    async def _request_cancel(
        self, cancel_fn: Callable[[str], object], operation_id: str
    ) -> CancelEvidence:
        try:
            value = cancel_fn(operation_id)
            if inspect.isawaitable(value):
                value = await asyncio.wait_for(value, timeout=self._cancel_timeout_s)
        except TimeoutError:
            return CancelEvidence(
                operation_id=operation_id,
                acknowledged=False,
                terminal=False,
                safe=False,
                evidence={"cancel_error": "cancel request timed out"},
            )
        except Exception as exc:
            return CancelEvidence(
                operation_id=operation_id,
                acknowledged=False,
                terminal=False,
                safe=False,
                evidence={"cancel_error": repr(exc)},
            )
        if isinstance(value, CancelEvidence):
            if value.operation_id != operation_id:
                return CancelEvidence(
                    operation_id=operation_id,
                    acknowledged=False,
                    terminal=False,
                    safe=False,
                    evidence={
                        "cancel_error": (
                            "cancel evidence operation_id "
                            f"{value.operation_id!r} does not match"
                        )
                    },
                )
            return value
        if not self._allow_legacy_results:
            return CancelEvidence(
                operation_id=operation_id,
                acknowledged=False,
                terminal=False,
                safe=False,
                evidence={
                    "cancel_error": (
                        f"non-contract cancel result {type(value).__name__!r} rejected"
                    )
                },
            )
        return _legacy_cancel(operation_id, value)


async def schedule(
    plan: RobotOperationPlan,
    executor: RobotOperationExecutor | OperationExecutor,
    canceler: OperationCanceler | RobotOperationExecutor | None = None,
    *,
    operation_timeout_s: float | None = 30.0,
    cancel_timeout_s: float = 5.0,
    allow_legacy_results: bool = True,
) -> SchedulerResult:
    """Run one plan with a fresh scheduler.

    Args:
        plan: Parsed ``robot-operation-plan/v1`` document.
        executor: Formal or legacy executor.
        canceler: Optional legacy cancel seam.
        operation_timeout_s: Per-layer deadline in seconds.
        cancel_timeout_s: Cancel acknowledgement bound in seconds.
        allow_legacy_results: Enable the documented compatibility shim.

    Returns:
        The frozen :class:`SchedulerResult` for this run.
    """
    scheduler = MultiRobotScheduler(
        operation_timeout_s=operation_timeout_s,
        cancel_timeout_s=cancel_timeout_s,
        allow_legacy_results=allow_legacy_results,
    )
    return await scheduler.schedule(plan, executor, canceler)


__all__ = [
    "CancelEvidence",
    "ExecutionContractError",
    "JSONValue",
    "LayerResult",
    "MultiRobotScheduler",
    "OperationCanceler",
    "OperationExecution",
    "OperationExecutor",
    "OperationResult",
    "OperationStatus",
    "ResourceEvidence",
    "RobotOperationExecutor",
    "SchedulerResult",
    "is_json_safe",
    "schedule",
    "to_json_value",
]
