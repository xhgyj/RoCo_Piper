from __future__ import annotations

from rocobrick.models import FailureReason, TaskStatus
from rocobrick.task_facts import EventType, ExecutionEvent, TaskFactStore


def event(task_id, event_type, sequence=0, reason=None, event_id=None):
    return ExecutionEvent(event_id or f"e-{task_id}-{sequence}", "run-1", task_id, "intent-1", f"attempt-{task_id}", sequence, event_type, reason)


def test_single_failure_only_blocks_structural_successors(parsed):
    plan, _ = parsed
    store = TaskFactStore(plan, "run-1")
    store.apply(event("foundation:left", EventType.TASK_FAILED, reason=FailureReason.CONNECTION_FAILED))
    assert store.facts["foundation:left"].status == TaskStatus.FAILED
    assert store.facts["joined/top"].status == TaskStatus.BLOCKED_BY_FAILED_PREDECESSOR
    assert store.facts["cap@right"].status == TaskStatus.PENDING


def test_retreat_failure_never_revokes_assembly(parsed):
    plan, _ = parsed
    store = TaskFactStore(plan, "run-1")
    store.apply(event("foundation:left", EventType.TASK_ASSEMBLED))
    store.apply(event("foundation:left", EventType.RETREAT_FAILED, sequence=1, reason=FailureReason.RETREAT_FAILED, event_id="retreat"))
    assert store.facts["foundation:left"].status == TaskStatus.ASSEMBLED


def test_structure_invalidation_is_separate_from_historical_completion(parsed):
    plan, _ = parsed
    store = TaskFactStore(plan, "run-1")
    store.apply(event("foundation:left", EventType.TASK_ASSEMBLED))
    store.apply(event("foundation:left", EventType.STRUCTURE_INVALIDATED, sequence=1, event_id="slip"))
    assert store.facts["foundation:left"].status == TaskStatus.ASSEMBLED
    assert store.structure_invalidated_tasks == frozenset({"foundation:left"})


def test_duplicate_event_is_idempotent(parsed):
    plan, _ = parsed
    store = TaskFactStore(plan, "run-1")
    assembled = event("cap@right", EventType.TASK_ASSEMBLED)
    assert store.apply(assembled)
    assert not store.apply(assembled)
