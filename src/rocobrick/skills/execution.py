"""Skill phase orchestration over an observable physical backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol, runtime_checkable
from uuid import uuid4

from rocobrick.models import FailureReason, SkillType, TaskDefinition
from rocobrick.skills.programs import PhaseResult, PhaseStatus, SkillPhase, program_for
from rocobrick.task_facts import EventType, ExecutionEvent


@runtime_checkable
class TaskRef(Protocol):
    """Minimal task identity: execute_skill needs only these two attributes."""

    task_id: str
    skill: SkillType


class PhysicalPhaseBackend(Protocol):
    """Execution-owned physical operations; implementations must not change phase type."""

    async def execute_phase(
        self,
        task: TaskDefinition,
        phase: SkillPhase,
        intent_id: str,
        attempt_id: str,
    ) -> PhaseResult: ...


@dataclass(frozen=True)
class SkillExecutionResult:
    task_id: str
    attempt_id: str
    assembled: bool
    retreat_confirmed: bool
    events: tuple[ExecutionEvent, ...]
    phases: tuple[PhaseResult, ...]


EventSink = Callable[[ExecutionEvent], None]


async def execute_skill(
    task: TaskRef,
    run_id: str,
    backend: PhysicalPhaseBackend,
    publish: EventSink,
    *,
    intent_id: str | None = None,
    attempt_id: str | None = None,
) -> SkillExecutionResult:
    """Execute one immutable Skill program with first-failure terminal semantics."""
    intent = intent_id or str(uuid4())
    attempt = attempt_id or str(uuid4())
    results: list[PhaseResult] = []
    events: list[ExecutionEvent] = []
    connection_confirmed = False
    release_confirmed = False
    assembled = False
    retreat_confirmed = False

    def emit(event_type: EventType, sequence: int, reason: FailureReason | None = None) -> None:
        event = ExecutionEvent(
            event_id=str(uuid4()),
            run_id=run_id,
            task_id=task.task_id,
            intent_id=intent,
            attempt_id=attempt,
            sequence=sequence,
            event_type=event_type,
            reason=reason,
        )
        publish(event)
        events.append(event)

    for sequence, phase in enumerate(program_for(task.skill).phases):
        if phase.phase_id == "publish_assembled":
            if not connection_confirmed or not release_confirmed:
                failure = PhaseResult(
                    phase,
                    PhaseStatus.FAILED,
                    FailureReason.CONNECTION_FAILED
                    if not connection_confirmed
                    else FailureReason.RELEASE_FAILED,
                )
                results.append(failure)
                emit(EventType.TASK_FAILED, sequence, failure.reason)
                break
            confirmed = PhaseResult(phase, PhaseStatus.CONFIRMED)
            results.append(confirmed)
            emit(EventType.TASK_ASSEMBLED, sequence)
            assembled = True
            continue

        result = await backend.execute_phase(task, phase, intent, attempt)
        if result.phase != phase:
            raise RuntimeError("physical backend changed the requested Skill phase")
        if result.status not in (PhaseStatus.CONFIRMED, PhaseStatus.FAILED):
            raise RuntimeError("physical backend returned a non-terminal phase result")
        results.append(result)
        if result.status == PhaseStatus.FAILED:
            if assembled and phase.phase_id == "retreat":
                emit(EventType.RETREAT_FAILED, sequence, result.reason)
            else:
                emit(EventType.TASK_FAILED, sequence, result.reason)
            break
        if phase.phase_id == "confirm_connection":
            connection_confirmed = True
        elif phase.phase_id == "confirm_release":
            release_confirmed = True
        elif phase.phase_id == "retreat":
            retreat_confirmed = True

    return SkillExecutionResult(
        task.task_id,
        attempt,
        assembled,
        retreat_confirmed,
        tuple(events),
        tuple(results),
    )
