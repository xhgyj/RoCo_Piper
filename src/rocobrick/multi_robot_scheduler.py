from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from rocobrick.compat import StrEnum
from typing import Protocol

from rocobrick.robot_operation_plan import RobotOperation, RobotOperationPlan


class OperationStatus(StrEnum):
    READY = "READY"
    RUNNING = "RUNNING"
    CONFIRMED = "CONFIRMED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    BLOCKED = "BLOCKED"


class OperationExecutor(Protocol):
    async def execute(self, operation: RobotOperation) -> object: ...


class OperationCanceler(Protocol):
    async def cancel(self, operation_id: str) -> object: ...


@dataclass(frozen=True)
class OperationResult:
    operation_id: str
    robot_id: str
    status: OperationStatus
    passed: bool
    detail: object = None


@dataclass(frozen=True)
class LayerResult:
    layer_index: int
    status: OperationStatus
    operation_ids: tuple[str, ...]


@dataclass(frozen=True)
class SchedulerResult:
    operations: Mapping[str, OperationResult]
    layers: tuple[LayerResult, ...]
    passed: bool
    evidence: tuple[Mapping[str, object], ...] = field(default_factory=tuple)


def _passed(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, Mapping) and "passed" in value:
        return bool(value["passed"])
    return bool(getattr(value, "passed", value))


class MultiRobotScheduler:
    def __init__(self) -> None:
        self._running: dict[str, asyncio.Task[object]] = {}
        self._cancelled: set[str] = set()

    async def schedule(
        self,
        plan: RobotOperationPlan,
        executor: OperationExecutor,
        canceler: OperationCanceler | None = None,
    ) -> SchedulerResult:
        results: dict[str, OperationResult] = {}
        layers: list[LayerResult] = []
        evidence: list[Mapping[str, object]] = []
        dependencies = {
            operation.successor_operation_id: operation.predecessor_operation_id
            for operation in plan.dependencies
        }
        completed: set[str] = set()
        failed = False
        for layer_index, operation_ids in enumerate(plan.execution_layers):
            if failed:
                for operation_id in operation_ids:
                    operation = plan.operations[operation_id]
                    results[operation_id] = OperationResult(
                        operation_id, operation.executor_robot_id,
                        OperationStatus.BLOCKED, False, "predecessor layer failed",
                    )
                layers.append(LayerResult(layer_index, OperationStatus.BLOCKED, tuple(operation_ids)))
                continue
            blocked = [
                operation_id for operation_id in operation_ids
                if dependencies.get(operation_id) not in completed
                and operation_id in dependencies
            ]
            if blocked:
                for operation_id in operation_ids:
                    operation = plan.operations[operation_id]
                    status = OperationStatus.BLOCKED
                    results[operation_id] = OperationResult(
                        operation_id, operation.executor_robot_id, status, False,
                        "dependency not confirmed",
                    )
                layers.append(LayerResult(layer_index, OperationStatus.BLOCKED, tuple(operation_ids)))
                failed = True
                continue
            layer_results = await self._run_layer(
                plan, operation_ids, executor, canceler
            )
            results.update(layer_results)
            if any(not item.passed for item in layer_results.values()):
                failed = True
                status = OperationStatus.FAILED
            else:
                completed.update(operation_ids)
                status = OperationStatus.CONFIRMED
            layers.append(LayerResult(layer_index, status, tuple(operation_ids)))
            evidence.append({
                "layer_index": layer_index,
                "operation_ids": list(operation_ids),
                "status": status.value,
                "passed": status is OperationStatus.CONFIRMED,
            })
        return SchedulerResult(results, tuple(layers), not failed and all(item.passed for item in results.values()), tuple(evidence))

    async def _run_layer(
        self,
        plan: RobotOperationPlan,
        operation_ids: tuple[str, ...],
        executor: OperationExecutor,
        canceler: OperationCanceler | None,
    ) -> dict[str, OperationResult]:
        running: dict[str, asyncio.Task[object]] = {}
        for operation_id in operation_ids:
            operation = plan.operations[operation_id]
            running[operation_id] = asyncio.create_task(executor.execute(operation))
            self._running[operation_id] = running[operation_id]
        pending: set[asyncio.Task[object]] = set(running.values())
        failed_id: str | None = None
        while pending and failed_id is None:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                operation_id = next(
                    (key for key, value in running.items() if value is task),
                    None,
                )
                if task.cancelled():
                    failed_id = operation_id
                    break
                try:
                    if not _passed(task.result()):
                        failed_id = operation_id
                        break
                except BaseException:
                    failed_id = operation_id
                    break
        if failed_id is not None and pending:
            for operation_id, task in running.items():
                if task in pending:
                    if canceler is not None:
                        try:
                            await canceler.cancel(operation_id)
                        except Exception:
                            pass
                    self._cancelled.add(operation_id)
                    task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        gathered = await asyncio.gather(*running.values(), return_exceptions=True)
        result: dict[str, OperationResult] = {}
        for operation_id, value in zip(operation_ids, gathered):
            operation = plan.operations[operation_id]
            if operation_id in self._cancelled:
                result[operation_id] = OperationResult(
                    operation_id, operation.executor_robot_id,
                    OperationStatus.CANCELLED, False, "cancelled after layer failure",
                )
            elif isinstance(value, BaseException):
                result[operation_id] = OperationResult(
                    operation_id, operation.executor_robot_id,
                    OperationStatus.FAILED, False, repr(value),
                )
            elif _passed(value):
                result[operation_id] = OperationResult(
                    operation_id, operation.executor_robot_id,
                    OperationStatus.CONFIRMED, True, value,
                )
            else:
                result[operation_id] = OperationResult(
                    operation_id, operation.executor_robot_id,
                    OperationStatus.FAILED, False, value,
                )
            self._running.pop(operation_id, None)
        return result


async def schedule(
    plan: RobotOperationPlan,
    executor: OperationExecutor,
    canceler: OperationCanceler | None = None,
) -> SchedulerResult:
    return await MultiRobotScheduler().schedule(plan, executor, canceler)
