"""Offline event interleavings; no controller or simulator is constructed."""

# ruff: noqa: D101,D102,D103,D107
from dataclasses import replace

import pytest

from rocobrick.contracts.phases import ResourceKind, ResourceRequirement
from rocobrick.scheduling.resource_manager import ResourceManager
from rocobrick.scheduling.scheduler import (
    DependencyScheduler,
    EventKind,
    OperationEvent,
    OperationState,
)
from rocobrick.task_plan.parser import parse_task_plan


class Verifier:
    """Explicit test double, not a production physical verifier."""

    def __init__(self):
        self.physical = True
        self.completion = True
        self.release = True

    def ready(self, operation, attempt):
        return self.physical

    def completed(self, operation, event):
        return self.completion and event.evidence_ids == ("physical+terminal",)

    def releasable(self, operation, attempt, evidence_ids):
        return self.release and evidence_ids == ("terminal+no-obligations",)


def kernel(*, dependencies=None, same_robot=False):
    plan = parse_task_plan(
        {
            "schema": "robot-task-plan/v2",
            "plan_id": "plan",
            "operations": [
                {
                    "id": "c",
                    "skill": "PLACE_DOWN",
                    "executor": "r0",
                    "participants": ["r0"],
                    "object_id": "brick_a",
                    "target_ref": "slot",
                },
                {
                    "id": "b",
                    "skill": "PICK",
                    "executor": "r0" if same_robot else "r1",
                    "participants": ["r0" if same_robot else "r1"],
                    "object_id": "brick_b",
                },
                {
                    "id": "a",
                    "skill": "PICK",
                    "executor": "r0",
                    "participants": ["r0"],
                    "object_id": "brick_a",
                },
            ],
            "dependencies": dependencies
            if dependencies is not None
            else [{"before": "a", "after": "c", "condition": "physical_success"}],
        }
    )
    verifier = Verifier()
    return DependencyScheduler(plan, "session", verifier), verifier


def event(operation="a", *, kind=EventKind.PHYSICAL_SUCCESS, event_id="e1"):
    return OperationEvent(
        event_id, "session", "plan", operation, 1, kind, ("physical+terminal",)
    )


def test_successor_can_request_while_independent_pick_is_still_admitted():
    scheduler, _ = kernel()
    assert scheduler.eligible() == ("a", "b")
    assert scheduler.admit("a") and scheduler.admit("b")
    assert scheduler.apply(event())
    # Success does not implicitly release carrying/contact resources.
    assert scheduler.eligible() == ()
    assert "resource:robot:r0" in scheduler.waiting_reasons("c")
    # This test's verifier explicitly confirms no continuing obligation. Real
    # carrying transitions will require lease transfer, not this release.
    assert scheduler.release("a", ("terminal+no-obligations",))
    assert scheduler.eligible() == ("c",)
    assert scheduler.admit("c")
    assert scheduler.state("b") is OperationState.ADMITTED


def test_all_explicit_predecessors_must_physically_finish():
    scheduler, _ = kernel(
        dependencies=[
            {"before": "a", "after": "c", "condition": "physical_success"},
            {"before": "b", "after": "c", "condition": "physical_success"},
        ]
    )
    scheduler.admit("a")
    scheduler.admit("b")
    scheduler.apply(event())
    scheduler.release("a", ("terminal+no-obligations",))
    assert scheduler.eligible() == ()
    assert "predecessor:b" in scheduler.waiting_reasons("c")
    scheduler.apply(event("b", event_id="b-done"))
    assert scheduler.eligible() == ("c",)


def test_readiness_is_rechecked_and_truthy_results_fail_closed():
    scheduler, verifier = kernel()
    assert scheduler.eligible() == ("a", "b")
    verifier.physical = {"ready": True}
    assert not scheduler.admit("a")
    assert scheduler.held_resources("a") == ()


def test_same_robot_and_object_are_not_double_owned():
    scheduler, _ = kernel(same_robot=True)
    assert scheduler.admit("b")
    assert not scheduler.admit("a")
    assert scheduler.held_resources("a") == ()
    assert scheduler.held_resources("b")


def test_planner_or_controller_success_is_not_physical_success():
    scheduler, verifier = kernel()
    scheduler.admit("a")
    assert not scheduler.apply(replace(event(), evidence_ids=("planner-success",)))
    verifier.completion = {"ok": True}
    # Explicit port violation (truthy object) must be rejected.
    verifier.completed = lambda operation, evidence: {"ok": True}
    assert not scheduler.apply(event(event_id="e2"))
    assert scheduler.state("a") is OperationState.ADMITTED
    assert scheduler.held_resources("a")


@pytest.mark.parametrize(
    "change",
    [
        {"session_id": "old"},
        {"plan_id": "old"},
        {"attempt": 2},
        {"operation_id": "unknown"},
    ],
)
def test_foreign_and_late_events_cannot_complete_current_operation(change):
    scheduler, _ = kernel()
    scheduler.admit("a")
    assert not scheduler.apply(replace(event(), **change))
    assert scheduler.state("a") is OperationState.ADMITTED


def test_duplicate_event_is_idempotent_but_conflicting_id_is_error():
    scheduler, _ = kernel()
    scheduler.admit("a")
    assert scheduler.apply(event())
    assert not scheduler.apply(event())
    with pytest.raises(ValueError, match="reused"):
        scheduler.apply(replace(event(), kind=EventKind.FAILURE))


def test_cancellation_never_proves_stop_or_releases_leases():
    scheduler, _ = kernel()
    scheduler.admit("a")
    before = scheduler.held_resources("a")
    scheduler.apply(event(kind=EventKind.CANCEL_REQUESTED))
    assert scheduler.state("a") is OperationState.UNCERTAIN
    assert scheduler.state("c") is OperationState.BLOCKED
    assert scheduler.held_resources("a") == before
    assert not scheduler.release("a", ("terminal+no-obligations",))
    assert scheduler.eligible() == ()


def test_failure_quarantines_until_safe_release_then_only_independent_work_runs():
    scheduler, verifier = kernel()
    scheduler.admit("a")
    scheduler.apply(event(kind=EventKind.FAILURE))
    assert scheduler.eligible() == ()
    verifier.release = False
    assert not scheduler.release("a", ("terminal+no-obligations",))
    assert scheduler.held_resources("a")
    verifier.release = True
    assert scheduler.release("a", ("terminal+no-obligations",))
    assert scheduler.eligible() == ("b",)
    assert scheduler.state("c") is OperationState.BLOCKED


def test_revocation_invalidates_running_descendant_and_keeps_its_leases():
    scheduler, _ = kernel()
    scheduler.admit("a")
    scheduler.apply(event())
    scheduler.release("a", ("terminal+no-obligations",))
    scheduler.admit("c")
    scheduler.apply(event(kind=EventKind.FACT_REVOKED, event_id="revoke"))
    assert scheduler.state("c") is OperationState.UNCERTAIN
    assert scheduler.held_resources("c")
    assert scheduler.eligible() == ()
    assert not scheduler.apply(event("c", event_id="late-success"))


def test_continuing_support_obligation_prevents_release_even_after_success():
    scheduler, verifier = kernel()
    scheduler.admit("a")
    scheduler.apply(event())
    verifier.release = False
    assert not scheduler.release("a", ("terminal+no-obligations",))
    assert not scheduler.admit("c")
    assert scheduler.held_resources("a")


def test_resource_claim_is_atomic_and_namespaces_do_not_alias():
    manager = ResourceManager()
    robot = ResourceRequirement(ResourceKind.ROBOT, "robot:same")
    obj = ResourceRequirement(ResourceKind.PART, "part:same")
    assert manager.claim("a", (robot,))
    assert not manager.claim("b", (obj, robot))
    assert manager.held("b") == ()
    assert manager.claim("b", (obj,))


@pytest.mark.parametrize(
    "change",
    [
        {"attempt": True},
        {"attempt": 0},
        {"kind": "physical_success"},
        {"evidence_ids": []},
        {"evidence_ids": ()},
        {"event_id": ""},
    ],
)
def test_events_are_strict_immutable_contracts(change):
    with pytest.raises(ValueError):
        replace(event(), **change)
