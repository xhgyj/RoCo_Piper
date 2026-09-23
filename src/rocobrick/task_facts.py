"""Monotonic task fact projection driven only by structured events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from rocobrick.compat import StrEnum
from rocobrick.models import AssemblyPlan, FailureReason, TaskStatus


class EventType(StrEnum):
    TASK_ASSEMBLED = "TASK_ASSEMBLED"
    TASK_FAILED = "TASK_FAILED"
    RETREAT_FAILED = "RETREAT_FAILED"
    STRUCTURE_INVALIDATED = "STRUCTURE_INVALIDATED"


@dataclass(frozen=True)
class ExecutionEvent:
    event_id: str
    run_id: str
    task_id: str
    intent_id: str
    attempt_id: str
    sequence: int
    event_type: EventType
    reason: FailureReason | None = None


@dataclass(frozen=True)
class TaskFact:
    task_id: str
    status: TaskStatus
    failure_reason: FailureReason | None = None


class TaskFactStore:
    """Owns task facts but no robot, motion, resource, or waiting state."""

    def __init__(self, plan: AssemblyPlan, run_id: str) -> None:
        self._plan = plan
        self._run_id = run_id
        self._facts = {task_id: TaskFact(task_id, TaskStatus.PENDING) for task_id in plan.tasks}
        self._event_ids: set[str] = set()
        self._last_sequence_by_attempt: dict[str, int] = {}
        self._structure_invalidated: set[str] = set()

    @property
    def facts(self) -> Mapping[str, TaskFact]:
        return dict(self._facts)

    @property
    def structure_invalidated_tasks(self) -> frozenset[str]:
        return frozenset(self._structure_invalidated)

    def apply(self, event: ExecutionEvent) -> bool:
        if event.run_id != self._run_id:
            raise ValueError("event run_id does not match this store")
        if event.task_id not in self._facts:
            raise ValueError(f"event references unknown task {event.task_id}")
        if event.event_id in self._event_ids:
            return False
        last = self._last_sequence_by_attempt.get(event.attempt_id, -1)
        if event.sequence <= last:
            raise ValueError(f"out-of-order event for attempt {event.attempt_id}")
        self._event_ids.add(event.event_id)
        self._last_sequence_by_attempt[event.attempt_id] = event.sequence
        current = self._facts[event.task_id]
        if event.event_type == EventType.TASK_ASSEMBLED:
            if current.status == TaskStatus.PENDING:
                self._facts[event.task_id] = TaskFact(event.task_id, TaskStatus.ASSEMBLED)
        elif event.event_type == EventType.TASK_FAILED:
            if event.reason is None:
                raise ValueError("TASK_FAILED requires a reason")
            if current.status == TaskStatus.PENDING:
                self._facts[event.task_id] = TaskFact(event.task_id, TaskStatus.FAILED, event.reason)
                self._propagate_failed_predecessors()
        elif event.event_type == EventType.STRUCTURE_INVALIDATED:
            self._structure_invalidated.add(event.task_id)
        # Retreat failure is an execution/run result and never rewrites assembly facts.
        return True

    def _propagate_failed_predecessors(self) -> None:
        changed = True
        while changed:
            changed = False
            failed_or_blocked = {
                task_id for task_id, fact in self._facts.items()
                if fact.status in (TaskStatus.FAILED, TaskStatus.BLOCKED_BY_FAILED_PREDECESSOR)
            }
            for task_id, definition in self._plan.tasks.items():
                if self._facts[task_id].status != TaskStatus.PENDING:
                    continue
                if definition.predecessors & failed_or_blocked:
                    self._facts[task_id] = TaskFact(task_id, TaskStatus.BLOCKED_BY_FAILED_PREDECESSOR)
                    changed = True
