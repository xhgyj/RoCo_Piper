"""Deterministic, side-effect-free dispatch selection."""

from __future__ import annotations

from typing import Mapping

from rocobrick.models import (
    ActionType,
    AssemblyPlan,
    Candidate,
    DispatchDecision,
    ExecutionSnapshot,
    FailureReason,
    TaskStatus,
)
from rocobrick.task_facts import TaskFact


def _priority(candidate: Candidate) -> int:
    if candidate.action == ActionType.EMERGENCY_STOP:
        return 0
    if candidate.action == ActionType.CONTROLLED_STOP:
        return 0
    if candidate.resolves_blockage:
        return 1
    if candidate.holding_continuation:
        return 2
    if not candidate.is_prefetch:
        return 3
    return 4


def decide(
    plan: AssemblyPlan,
    facts: Mapping[str, TaskFact],
    execution: ExecutionSnapshot,
    candidates: tuple[Candidate, ...],
) -> DispatchDecision:
    """Select a deterministic maximal compatible set without external access."""
    waiting: dict[str, FailureReason] = {}
    if execution.emergency_stop_requested:
        emergency = tuple(sorted((item for item in candidates if item.action == ActionType.EMERGENCY_STOP), key=lambda item: item.candidate_id))
        return DispatchDecision(emergency[:1], waiting, ("emergency stop has exclusive priority",))
    if execution.safety_stop_requested:
        ordered = sorted(
            (item for item in candidates if item.action == ActionType.CONTROLLED_STOP),
            key=lambda item: (item.robot_id, item.candidate_id),
        )
        by_robot: dict[str, Candidate] = {}
        for item in ordered:
            by_robot.setdefault(item.robot_id, item)
        return DispatchDecision(
            tuple(by_robot.values()),
            waiting,
            ("controlled stop suppresses ordinary actions",),
        )

    eligible: list[Candidate] = []
    for candidate in candidates:
        definition = plan.tasks.get(candidate.task_id)
        if definition is None or candidate.robot_id != definition.primary_robot:
            continue
        if facts[candidate.task_id].status != TaskStatus.PENDING:
            continue
        if candidate.revisions != execution.revisions:
            waiting[candidate.candidate_id] = FailureReason.STALE_WORLD
            continue
        if not candidate.feasible:
            waiting[candidate.candidate_id] = FailureReason.COLLISION_BLOCKED
            continue
        if candidate.is_prefetch:
            waiting[candidate.candidate_id] = FailureReason.RESOURCE_WAIT
            continue
        if any(facts[pred].status != TaskStatus.ASSEMBLED for pred in definition.predecessors) and not candidate.holding_continuation:
            waiting[candidate.candidate_id] = FailureReason.DEPENDENCY_UNMET
            continue
        eligible.append(candidate)

    eligible.sort(key=lambda item: (
        _priority(item),
        plan.tasks[item.task_id].level,
        plan.tasks[item.task_id].schedule_slot,
        item.task_id,
        item.action.value,
        item.candidate_id,
    ))
    selected: list[Candidate] = []
    used_robots: set[str] = set()
    used_resources = set(execution.occupied_resources)
    for candidate in eligible:
        if candidate.robot_id in used_robots or candidate.required_resources & used_resources:
            waiting[candidate.candidate_id] = FailureReason.RESOURCE_WAIT
            continue
        selected.append(candidate)
        used_robots.add(candidate.robot_id)
        used_resources.update(candidate.required_resources)
    rationale = tuple(f"selected {item.candidate_id} at priority {_priority(item)}" for item in selected)
    return DispatchDecision(tuple(selected), dict(sorted(waiting.items())), rationale)
