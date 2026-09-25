"""Runner seam that executes one parsed operation plan and serializes it.

The runner keeps the frozen ``rocobrick-operation-plan-run/v1`` document shape
produced for the external operation-plan entry: top-level ``schema``,
``passed``, ``operations``, ``layers`` and ``evidence`` keys stay as they were,
while per-operation terminal/motion/cancel/resource evidence is added
additively and readiness/cleanup evidence is attached only when the caller
supplies it as optional data. The schema identifier never changes here.

Scheduler failures are serialized, not raised: the document records per-robot
isolated operations, layer barriers, cancel acknowledgements, resource leases
and the operation ids whose terminal safety was not established.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

from rocobrick.multi_robot_scheduler import (
    CancelEvidence,
    JSONValue,
    LayerResult,
    MultiRobotScheduler,
    OperationCanceler,
    OperationExecutor,
    OperationResult,
    ResourceEvidence,
    RobotOperationExecutor,
    SchedulerResult,
    is_json_safe,
    to_json_value,
)
from rocobrick.robot_operation_plan import RobotOperationPlan

RUN_SCHEMA = "rocobrick-operation-plan-run/v1"


@dataclass(frozen=True)
class OperationPlanRunner:
    """Execute one parsed plan through the scheduler and serialize its result.

    Attributes:
        plan: Parsed ``robot-operation-plan/v1`` document.
        executor: Formal :class:`RobotOperationExecutor`, or a legacy stub.
        canceler: Optional legacy cancel seam; the executor's ``cancel`` is
            used when this is omitted.
        operation_timeout_s: Override for the scheduler's per-layer deadline.
            ``None`` keeps the scheduler default.
        cancel_timeout_s: Override for the cancel acknowledgement bound.
            ``None`` keeps the scheduler default.
        allow_legacy_results: Enable the documented legacy result shim.
        readiness: Optional JSON-safe readiness evidence (for example
            ``readiness_by_robot``) attached to the serialized document.
        cleanup: Optional JSON-safe cleanup evidence attached to the
            serialized document.
    """

    plan: RobotOperationPlan
    executor: RobotOperationExecutor | OperationExecutor
    canceler: OperationCanceler | RobotOperationExecutor | None = None
    operation_timeout_s: float | None = None
    cancel_timeout_s: float | None = None
    allow_legacy_results: bool = True
    readiness: Mapping[str, JSONValue] | None = None
    cleanup: Mapping[str, JSONValue] | None = None

    async def run(self) -> SchedulerResult:
        """Run the plan with fresh scheduler state.

        Returns:
            The frozen :class:`SchedulerResult` for this run.
        """
        scheduler = self._build_scheduler()
        return await scheduler.schedule(self.plan, self.executor, self.canceler)

    def serialize(self, result: SchedulerResult) -> dict[str, object]:
        """Serialize a run result including the configured optional data.

        Args:
            result: Result returned by :meth:`run`.

        Returns:
            A JSON-safe ``rocobrick-operation-plan-run/v1`` document.

        Raises:
            ValueError: If readiness/cleanup evidence is not JSON-safe.
        """
        return serialize_run_result(
            result, readiness=self.readiness, cleanup=self.cleanup
        )

    def _build_scheduler(self) -> MultiRobotScheduler:
        if self.operation_timeout_s is None and self.cancel_timeout_s is None:
            return MultiRobotScheduler(
                allow_legacy_results=self.allow_legacy_results
            )
        if self.cancel_timeout_s is None:
            return MultiRobotScheduler(
                operation_timeout_s=self.operation_timeout_s,
                allow_legacy_results=self.allow_legacy_results,
            )
        if self.operation_timeout_s is None:
            return MultiRobotScheduler(
                cancel_timeout_s=self.cancel_timeout_s,
                allow_legacy_results=self.allow_legacy_results,
            )
        return MultiRobotScheduler(
            operation_timeout_s=self.operation_timeout_s,
            cancel_timeout_s=self.cancel_timeout_s,
            allow_legacy_results=self.allow_legacy_results,
        )


def _audit_value(value: object) -> JSONValue:
    if is_json_safe(value):
        return to_json_value(value)
    return repr(value)


def _cancel_document(cancel: CancelEvidence) -> dict[str, object]:
    return {
        "operation_id": cancel.operation_id,
        "acknowledged": cancel.acknowledged,
        "terminal": cancel.terminal,
        "safe": cancel.safe,
        "evidence": to_json_value(dict(cancel.evidence)),
    }


def _resource_document(item: ResourceEvidence) -> dict[str, object]:
    return {
        "resource": item.resource,
        "operation_id": item.operation_id,
        "robot_id": item.robot_id,
        "acquired": item.acquired,
        "released": item.released,
        "reason": item.reason,
    }


def _operation_document(item: OperationResult) -> dict[str, object]:
    return {
        "operation_id": item.operation_id,
        "robot_id": item.robot_id,
        "status": item.status.value,
        "passed": item.passed,
        "detail": _audit_value(item.detail),
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


def _layer_document(layer: LayerResult) -> dict[str, object]:
    return {
        "layer_index": layer.layer_index,
        "status": layer.status.value,
        "operation_ids": list(layer.operation_ids),
        "reason": layer.reason,
    }


def _extension_value(value: Mapping[str, JSONValue], name: str) -> JSONValue:
    if not is_json_safe(value):
        raise ValueError(f"{name} evidence must be JSON-safe")
    return to_json_value(value)


def serialize_run_result(
    result: SchedulerResult,
    *,
    readiness: Mapping[str, JSONValue] | None = None,
    cleanup: Mapping[str, JSONValue] | None = None,
) -> dict[str, object]:
    """Serialize a scheduler result as a JSON-safe v1 run document.

    Args:
        result: Result of one scheduler run.
        readiness: Optional readiness evidence, attached as the ``readiness``
            key only when provided.
        cleanup: Optional cleanup evidence, attached as the ``cleanup`` key
            only when provided.

    Returns:
        The ``rocobrick-operation-plan-run/v1`` document ready for
        ``json.dumps``.

    Raises:
        ValueError: If optional readiness/cleanup evidence is not JSON-safe.
    """
    document: dict[str, object] = {
        "schema": RUN_SCHEMA,
        "passed": bool(result.passed),
        "operations": {
            operation_id: _operation_document(item)
            for operation_id, item in result.operations.items()
        },
        "layers": [_layer_document(layer) for layer in result.layers],
        "evidence": [to_json_value(item) for item in result.evidence],
        "resources": [
            _resource_document(item) for item in result.resources
        ],
        "cancellations": [
            _cancel_document(item) for item in result.cancellations
        ],
        "unsafe_terminals": list(result.unsafe_terminals),
    }
    if readiness is not None:
        document["readiness"] = _extension_value(readiness, "readiness")
    if cleanup is not None:
        document["cleanup"] = _extension_value(cleanup, "cleanup")
    json.dumps(document, allow_nan=False)
    return document


__all__ = [
    "OperationPlanRunner",
    "RUN_SCHEMA",
    "serialize_run_result",
]
