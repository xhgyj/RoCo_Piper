"""Offline tests for the generic skill programs and the TaskRef seam."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rocobrick.models import ActionType, SkillType
from rocobrick.skill_execution import TaskRef, execute_skill
from rocobrick.skills import (
    PICK_PROGRAM,
    TRANSIT_PROGRAM,
    WAIT_PROGRAM,
    PhaseResult,
    PhaseStatus,
    SkillPhase,
    SkillProgram,
    program_for,
)


class ConfirmBackend:
    """Backend confirming every requested phase."""

    def __init__(self) -> None:
        self.requested: list[str] = []

    async def execute_phase(self, task, phase, intent_id, attempt_id):
        self.requested.append(phase.phase_id)
        return PhaseResult(phase, PhaseStatus.CONFIRMED)


# ---------------------------------------------------------------------------
# Program pinning
# ---------------------------------------------------------------------------


def test_transit_program_is_pinned() -> None:
    assert TRANSIT_PROGRAM == SkillProgram(
        SkillType.TRANSIT,
        (SkillPhase("reach_transit", ActionType.TRANSPORT, "measured_pose"),),
    )


def test_wait_program_is_pinned() -> None:
    assert WAIT_PROGRAM == SkillProgram(
        SkillType.WAIT,
        (SkillPhase("hold", ActionType.YIELD, "clock_elapsed"),),
    )


def test_program_registry_covers_the_executable_skills() -> None:
    from rocobrick.skills import PROGRAMS

    assert set(PROGRAMS) == {
        SkillType.PICK,
        SkillType.TRANSIT,
        SkillType.WAIT,
        SkillType.PLACE_DOWN,
    }
    assert program_for(SkillType.TRANSIT) is TRANSIT_PROGRAM
    assert program_for(SkillType.WAIT) is WAIT_PROGRAM
    assert program_for(SkillType.PICK) is PICK_PROGRAM
    assert [phase.phase_id for phase in TRANSIT_PROGRAM.phases] == [
        "reach_transit"
    ]
    assert [phase.phase_id for phase in WAIT_PROGRAM.phases] == ["hold"]


def test_program_for_unknown_skill_raises_capability_error() -> None:
    with pytest.raises(RuntimeError, match="CAPABILITY_UNAVAILABLE"):
        program_for(SkillType.HANDOVER)


# ---------------------------------------------------------------------------
# TaskRef protocol seam
# ---------------------------------------------------------------------------


def test_task_ref_protocol_accepts_minimal_task_identity() -> None:
    candidate = SimpleNamespace(task_id="skill-0-TRANSIT", skill=SkillType.TRANSIT)
    assert isinstance(candidate, TaskRef)
    assert not isinstance(object(), TaskRef)


def test_execute_skill_runs_the_transit_program_for_a_bare_task_ref() -> None:
    import asyncio

    backend = ConfirmBackend()
    task = SimpleNamespace(task_id="skill-0-TRANSIT", skill=SkillType.TRANSIT)
    outcome = asyncio.run(
        execute_skill(task, "run-1", backend, lambda event: None)
    )
    assert outcome.task_id == "skill-0-TRANSIT"
    assert [result.phase.phase_id for result in outcome.phases] == [
        "reach_transit"
    ]
    assert all(
        result.status is PhaseStatus.CONFIRMED for result in outcome.phases
    )
    assert backend.requested == ["reach_transit"]


def test_execute_skill_runs_the_wait_program_for_a_bare_task_ref() -> None:
    import asyncio

    backend = ConfirmBackend()
    task = SimpleNamespace(task_id="skill-1-WAIT", skill=SkillType.WAIT)
    outcome = asyncio.run(
        execute_skill(task, "run-1", backend, lambda event: None)
    )
    assert [result.phase.phase_id for result in outcome.phases] == ["hold"]
    assert backend.requested == ["hold"]
