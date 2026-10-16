import pytest

from rocobrick.models import FailureReason
from rocobrick.skills.execution import execute_skill
from rocobrick.skills.programs import PhaseResult, PhaseStatus
from rocobrick.task_facts import EventType, TaskFactStore


class Backend:
    def __init__(self, failure=None):
        self.failure = failure
        self.requested = []

    async def execute_phase(self, task, phase, intent_id, attempt_id):
        self.requested.append(phase.phase_id)
        if phase.phase_id == self.failure:
            reason = (
                FailureReason.RETREAT_FAILED
                if phase.phase_id == "retreat"
                else FailureReason.RELEASE_FAILED
            )
            return PhaseResult(phase, PhaseStatus.FAILED, reason)
        return PhaseResult(phase, PhaseStatus.CONFIRMED)


@pytest.mark.asyncio
async def test_place_down_publishes_assembled_only_after_connection_and_release(parsed):
    plan, _ = parsed
    store = TaskFactStore(plan, "run-1")
    backend = Backend()
    result = await execute_skill(plan.tasks["foundation:left"], "run-1", backend, store.apply)
    assert result.assembled
    assert not result.retreat_confirmed
    assert [event.event_type for event in result.events] == [EventType.TASK_ASSEMBLED]
    assert backend.requested.index("confirm_connection") < backend.requested.index("confirm_release")


@pytest.mark.asyncio
async def test_retreat_failure_is_reported_after_assembly_without_revoking_it(parsed):
    plan, _ = parsed
    store = TaskFactStore(plan, "run-1")
    result = await execute_skill(
        plan.tasks["foundation:left"], "run-1", Backend("retreat"), store.apply
    )
    assert result.assembled
    assert not result.retreat_confirmed
    assert [event.event_type for event in result.events] == [EventType.TASK_ASSEMBLED]


@pytest.mark.asyncio
async def test_first_preassembly_failure_stops_remaining_phases(parsed):
    plan, _ = parsed
    store = TaskFactStore(plan, "run-1")
    backend = Backend("confirm_release")
    result = await execute_skill(plan.tasks["foundation:left"], "run-1", backend, store.apply)
    assert not result.assembled
    assert backend.requested[-1] == "confirm_release"
    assert result.events[0].event_type == EventType.TASK_FAILED
