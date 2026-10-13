"""Regression tests for successor-owned physical lease transitions."""

import asyncio
from dataclasses import replace

import pytest
from pre_phase_helpers import snapshot, success
from test_multi_robot_scheduler_contract import (
    StrictExecutor,
    _execution,
    _raw_operation,
    _raw_plan,
)

from rocobrick.multi_robot_scheduler import OperationStatus, schedule
from rocobrick.unified_dual_arm_runtime import TerminalPrePhasePolicy


def endpoint(operation):
    """Return contact completion while retaining physical ownership."""
    return _execution(
        operation, evidence={"handoff_required": True, "terminal_safe": True}
    )


class TransitionExecutor(StrictExecutor):
    """Instrument transitions, shared state and terminal policy admission."""

    def __init__(self, *, mode="success", facts=None, policy=None):
        """Configure scripted physical facts and transition behavior."""
        super().__init__(outcomes={"a": endpoint, "support": endpoint})
        self.mode = mode
        self.facts = facts or {}
        self.policy = policy
        self.requests = []
        self.transition_cancels = []

    async def observe_pre_phase(self, participants):
        """Return a shared scripted snapshot."""
        self.events.append(("snapshot", ",".join(participants)))
        return snapshot(participants, self.facts)

    async def execute_pre_phase(self, request):
        """Return the configured physical transition result."""
        self.requests.append(request)
        self.events.append(("pre_phase", request.transition_id))
        if self.mode == "exception":
            raise RuntimeError("transport error")
        if self.mode == "timeout":
            await asyncio.sleep(10)
        result = success(
            request,
            {robot: self.facts.get(robot, {}) for robot in request.participants},
        )
        if self.mode == "lost_fact":
            return replace(
                result, final_facts={robot: {} for robot in request.participants}
            )
        if self.mode == "wrong_digest":
            return replace(result, scene_digest="b" * 64)
        if self.mode == "wrong_successor":
            return replace(result, successor_operation_ids=("foreign",))
        if self.mode == "wrong_version":
            return replace(result, scene_version=2)
        if self.mode == "wrong_participant":
            return replace(
                result, participants=("piper_1",), final_facts={"piper_1": {}}
            )
        if self.mode == "invented_fact":
            return replace(
                result, final_facts={"piper_0": {"holding:new": {"confirmed": True}}}
            )
        if self.mode == "missing_arrival":
            return replace(result, arrival_confirmed=False)
        return result

    async def validate_pre_phase(self, request, evidence):
        """Return the scripted scene-version readback verdict."""
        return self.mode != "scene_changed"

    async def cancel_pre_phase(self, transition_id):
        """Record cancellation without claiming terminal safety.

        Returns:
            Unsafe cancellation evidence for the scripted participant group.
        """
        self.transition_cancels.append(transition_id)
        return {"transition_id": transition_id, "terminal": False, "safe": False}

    def terminal_pre_phase_policy(self, operations):
        """Return an explicitly configured terminal policy or None."""
        return self.policy


def chain(*, extra=False):
    """Create contact endpoint followed by a motion successor.

    Returns:
        An offline plan with retained physical ownership.
    """
    a = _raw_operation("a", "piper_0", ("robot:piper_0", "part:brick"))
    b = replace(_raw_operation("b", "piper_0", ("robot:piper_0",)), skill="TRANSIT")
    if extra:
        other = _raw_operation("unrelated", "piper_1", ("robot:piper_1",))
        return _raw_plan([a, other, b], (("a",), ("unrelated",), ("b",)))
    return _raw_plan([a, b], (("a",), ("b",)))


def test_late_contact_confirmation_never_releases_physical_endpoint():
    """Late terminal evidence cannot stand in for physical departure."""

    class LateEndpoint(TransitionExecutor):
        async def execute(self, operation):
            if operation.operation_id == "a":
                try:
                    await asyncio.sleep(10)
                except asyncio.CancelledError:
                    return endpoint(operation)
            return await super().execute(operation)

    executor = LateEndpoint()
    result = asyncio.run(
        schedule(chain(), executor, operation_timeout_s=0.01, cancel_timeout_s=0.03)
    )
    assert not result.passed
    assert result.operations["a"].late_terminal_confirmation
    assert not result.operations["a"].resources_released
    assert all(
        not record.released for record in result.resources if record.operation_id == "a"
    )


def test_adopted_resources_remain_accounted_for_across_multiple_successors():
    """Inherited resources remain accounted for on subsequent transfers."""
    plan = chain()
    c = replace(_raw_operation("c", "piper_0", ("robot:piper_0",)), skill="TRANSIT")
    plan = _raw_plan([*plan.operations.values(), c], (("a",), ("b",), ("c",)))
    executor = TransitionExecutor()
    executor.outcomes["b"] = endpoint
    result = asyncio.run(schedule(plan, executor))
    assert result.passed
    assert executor.requests[1].lease_owners["part:brick"] == "b"
    assert all(record.released for record in result.resources if record.acquired)


def test_actual_next_robot_operation_not_unrelated_layer():
    """A lease remains held across unrelated layers and adopts only its successor."""
    executor = TransitionExecutor()
    plan = chain(extra=True)
    result = asyncio.run(schedule(plan, executor, allow_legacy_results=False))
    assert result.passed
    assert len(executor.requests) == 1
    request = executor.requests[0]
    assert request.successor_operation_ids == ("b",)
    assert request.predecessor_operation_ids == ("a",)
    assert request.next_operation is plan.operations["b"]
    assert executor.events.index(("end", "unrelated")) < executor.events.index(
        ("pre_phase", "pre-phase:b")
    )
    assert executor.events.index(("pre_phase", "pre-phase:b")) < executor.events.index(
        ("start", "b")
    )
    adopted = next(
        record
        for record in result.resources
        if record.operation_id == "b" and record.resource == "part:brick"
    )
    assert adopted.acquired and adopted.released
    assert (
        "atomic adoption" in adopted.reason or adopted.reason == "terminal confirmation"
    )


@pytest.mark.parametrize(
    "mode",
    [
        "wrong_digest",
        "wrong_successor",
        "wrong_version",
        "wrong_participant",
        "missing_arrival",
        "exception",
        "scene_changed",
        "invented_fact",
    ],
)
def test_failed_transition_keeps_predecessor_owner_and_blocks_contact(mode):
    """Wrong identity, changed scene or transport failure cannot release ownership."""
    executor = TransitionExecutor(mode=mode)
    result = asyncio.run(schedule(chain(), executor, allow_legacy_results=False))
    assert not result.passed
    assert not result.operations["a"].resources_released
    assert "b" not in executor.dispatched
    assert all(
        not record.released for record in result.resources if record.operation_id == "a"
    )
    assert result.operations["b"].status is OperationStatus.FAILED
    assert "pre_phase_failure" in result.operations["a"].evidence


def test_timeout_cancels_transition_without_releasing_previous_lease():
    """Async cancellation is not physical departure proof."""
    executor = TransitionExecutor(mode="timeout")
    result = asyncio.run(
        schedule(chain(), executor, operation_timeout_s=0.03, cancel_timeout_s=0.03)
    )
    assert not result.passed
    assert executor.transition_cancels == ["pre-phase:b"]
    assert not result.operations["a"].resources_released
    assert "b" not in executor.dispatched


def test_wait_observes_without_triggering_departure():
    """WAIT does not cause an implicit physical transition."""
    plan = chain()
    waiting = _raw_operation("wait", "piper_1", ("robot:piper_1",))
    plan = _raw_plan(
        [plan.operations["a"], waiting, plan.operations["b"]],
        (("a",), ("wait",), ("b",)),
    )
    executor = TransitionExecutor()
    result = asyncio.run(schedule(plan, executor))
    assert result.passed
    assert [request.successor_operation_ids for request in executor.requests] == [
        ("b",)
    ]


def test_final_endpoint_without_policy_is_retained_and_auditable():
    """No successor does not authorize an inferred upward retreat."""
    a = _raw_operation("a", "piper_0", ("robot:piper_0",))
    executor = TransitionExecutor()
    result = asyncio.run(schedule(_raw_plan([a], (("a",),)), executor))
    assert not result.passed
    assert not result.operations["a"].resources_released
    assert not executor.requests
    assert "terminal_policy_failure" in result.operations["a"].evidence


def test_explicit_terminal_policy_requires_real_arrival():
    """An authorized terminal target releases only after physical evidence."""
    policy = TerminalPrePhasePolicy(
        "cleanup", "c" * 64, ("piper_0",), {"piper_0": {"joint_target": "explicit"}}
    )
    a = _raw_operation("a", "piper_0", ("robot:piper_0",))
    executor = TransitionExecutor(policy=policy)
    result = asyncio.run(schedule(_raw_plan([a], (("a",),)), executor))
    assert result.passed
    assert executor.requests[0].terminal_policy is policy
    assert result.operations["a"].resources_released


@pytest.mark.parametrize("fact", ["holding:brick", "supporting:upper"])
def test_physical_facts_must_remain_continuous(fact):
    """Free-space transition cannot drop holding or ongoing support."""
    facts = {"piper_0": {fact: {"confirmed": True}}}
    executor = TransitionExecutor(facts=facts)
    assert asyncio.run(schedule(chain(), executor)).passed
    executor = TransitionExecutor(mode="lost_fact", facts=facts)
    result = asyncio.run(schedule(chain(), executor))
    assert not result.passed
    assert not result.operations["a"].resources_released


def test_multiple_predecessors_are_adopted_by_one_successor():
    """Both controller endpoints participate in one bound transition."""
    a = _raw_operation("a", "piper_0", ("robot:piper_0", "part:brick"))
    support = _raw_operation("support", "piper_1", ("robot:piper_1", "part:upper"))
    b = replace(
        _raw_operation("b", "piper_0", ("robot:piper_0", "robot:piper_1")),
        skill="PLACE_UP",
        participants=("piper_0", "piper_1"),
    )
    executor = TransitionExecutor(
        facts={"piper_1": {"supporting:upper": {"confirmed": True}}}
    )
    result = asyncio.run(
        schedule(_raw_plan([a, support, b], (("a", "support"), ("b",))), executor)
    )
    assert result.passed
    assert executor.requests[0].predecessor_operation_ids == ("a", "support")
    assert executor.requests[0].participants == ("piper_0", "piper_1")
    assert result.operations["support"].resources_released


def test_same_layer_successors_share_one_pre_phase_and_keep_support_leases():
    """PICK and SUPPORT_TOP successors use one combined transition."""
    a = _raw_operation("a", "piper_0", ("robot:piper_0", "part:brick"))
    support = _raw_operation(
        "support", "piper_1", ("robot:piper_1", "part:upper")
    )
    pick = replace(
        _raw_operation("pick", "piper_0", ("robot:piper_0", "part:brick")),
        skill="PICK",
        participants=("piper_0", "piper_1"),
    )
    support_top = replace(
        _raw_operation(
            "support-top", "piper_1", ("robot:piper_1", "part:upper")
        ),
        skill="SUPPORT_TOP",
        participants=("piper_0", "piper_1"),
    )
    executor = TransitionExecutor()
    executor.outcomes.update(
        {
            "pick": lambda operation: _execution(
                operation, evidence={"terminal_safe": True}
            ),
            "support-top": lambda operation: _execution(
                operation, evidence={"terminal_safe": True}
            ),
        }
    )
    result = asyncio.run(
        schedule(
            _raw_plan(
                [a, support, pick, support_top],
                (("a", "support"), ("pick", "support-top")),
            ),
            executor,
            allow_legacy_results=False,
        )
    )
    assert result.passed
    assert len(executor.requests) == 1
    request = executor.requests[0]
    assert request.successor_operation_ids == ("pick", "support-top")
    assert request.participants == ("piper_0", "piper_1")
    assert request.lease_targets["part:brick"] == "pick"
    assert request.lease_targets["part:upper"] == "support-top"
