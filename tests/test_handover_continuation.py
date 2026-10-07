"""Stationary PICK/lift lease continuation with fresh synthetic physics only."""

from dataclasses import replace

import pytest
from test_coordinated_handover_lifecycle import operation, program
from test_handover_joint_planning import snapshot
from test_robot_operation_plan_and_scheduler import operation as raw_operation
from test_robot_operation_plan_and_scheduler import plan

from rocobrick.execution.operation_state_ledger import OperationStateLedger
from rocobrick.robot_operation_plan import parse_robot_operation_plan
from rocobrick.skills.handover_continuation import StationaryHandoverContinuation
from rocobrick.unified_dual_arm_runtime import (
    SharedSceneSnapshot,
    SuccessorPrePhaseRequest,
)
from rocobrick.unified_runtime_facade import UnifiedDualArmRuntimeFacade


class Policy:
    """Use only the fixture's explicit giver calibration."""

    def __init__(self, prepared):
        """Initialize the explicitly bound fixture state."""
        self.prepared = prepared

    def profile_for(self, op):
        """Return the explicitly captured policy.

        Returns:
            The bound result of this read-only test boundary.
        """
        return self.prepared.request.profile

    def donor_window_for(self, op):
        """Return the giver's own calibrated opening window.

        Returns:
            The bound result of this read-only test boundary.
        """
        return self.prepared.donor_holding_window


class Session:
    """Read-only current scene fixture with no reachable motion methods."""

    def __init__(self, prepared, defect=None):
        """Initialize the explicitly bound fixture state."""
        self.prepared, self.defect = prepared, defect
        self.sequence = 1

    def observe_shared_scene(self):
        """Read current shared physics, full joints and scene version.

        Returns:
            The bound result of this read-only test boundary.
        """
        self.sequence += 1
        raw = snapshot(
            self.sequence, displacement=0.002 if self.defect == "moved" else 0
        )
        if self.defect == "stale":
            raw = snapshot(1)
        if self.defect == "no_contacts":
            raw["pad_object_contacts"]["contacts"] = []
        joints = dict(self.prepared.request.start_state)
        if self.defect == "joint_drift":
            joints["piper_1_joint1"] = self.sequence * 0.01
        if self.defect == "partial_joints":
            joints.pop("piper_0_joint1")
        return {
            "scene_snapshot": raw,
            "start_state": joints,
            "planning_scene_version": 2 if self.defect == "version" else 1,
            "observed_sim_time_s": raw["sim_time_s"]
            + (1 if self.defect == "old_clock" else 0.01),
        }


def fixture(defect=None):
    """Retain the exact preceding PICK's measured holding/lift proof.

    Returns:
        The identified, bound result; physical proof remains separately checked.
    """
    p = program()
    proof = {
        "lift_confirmed": True,
        "gripper_feedback_confirmed": True,
        "holding_confirmed": {
            "robot_id": "piper_1",
            "physical_instance_id": "brick-1",
            "session_id": "session-1",
            "manifest_sha256": "c" * 64,
            "last_sim_time_s": 1.1,
            "lift_m": 0.05,
            "samples": 4,
        },
    }
    ledger = OperationStateLedger(p.request.session_robots)
    ledger.confirm_holding("piper_1", "brick-1", proof)
    raw = raw_operation("pick-original", "piper_1", "PICK")
    pick = parse_robot_operation_plan(
        plan(
            [raw],
            [["pick-original"]],
            [[{"operation_id": "pick-original", "robot_id": "piper_1"}]],
        )
    ).operations["pick-original"]
    document = snapshot()
    document["scene_version"] = 1
    for robot, block in document["robots"].items():
        block.update(
            {
                "authorized": True,
                "terminal_safe": True,
                "holding_known": True,
                "supporting_known": True,
                "facts": {"holding:brick-1": proof} if robot == "piper_1" else {},
            }
        )
    request = SuccessorPrePhaseRequest(
        "pick-to-handover",
        (pick,),
        {"pick-original": {"terminal_safe": True}},
        operation(p),
        p.participants,
        ("robot:piper_1", "part:brick-1"),
        {"robot:piper_1": "pick-original", "part:brick-1": "pick-original"},
        SharedSceneSnapshot(document, 1.11, 0.5),
    )
    session = Session(p, defect)
    continuation = StationaryHandoverContinuation(session, ledger, Policy(p))
    return request, continuation, ledger, session


@pytest.mark.asyncio
async def test_stationary_continuation_keeps_original_lift_and_never_claims_transit():
    """Actual retained grasp refreshes custody without an invented arrival."""
    """Verify test pick lift proof cannot be replaced by planned effect."""
    request, continuation, ledger, _ = fixture()
    evidence = await continuation.admit_stationary(request)
    assert evidence.passed, evidence.reason
    assert evidence.matches(request)
    assert not any(
        (
            evidence.planned,
            evidence.prepared,
            evidence.started,
            evidence.arrival_confirmed,
        )
    )
    assert evidence.transition_mode == "stationary_handover"
    assert await continuation.validate_stationary(request, evidence)
    ledger.reconcile_pre_phase(request, evidence)
    fact = ledger.observe("piper_1", "holding:brick-1")
    assert fact.evidence["lift_confirmed"] is True
    assert fact.evidence["holding_confirmed"]["lift_m"] == 0.05
    assert fact.evidence["stationary_retention"]["samples"] == 4
    assert ledger.owner_of_object("brick-1") == "piper_1"
    assert ledger.observe("piper_0", "holding:brick-1") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "defect",
    [
        "moved",
        "stale",
        "no_contacts",
        "joint_drift",
        "partial_joints",
        "version",
        "old_clock",
    ],
)
async def test_stationary_continuation_rejects_unknown_or_changed_physical_state(
    defect,
):
    """Verify test stationary continuation rejects unknown or changed physical state."""
    request, continuation, ledger, _ = fixture(defect)
    evidence = await continuation.admit_stationary(request)
    assert not evidence.passed
    assert ledger.owner_of_object("brick-1") == "piper_1"
    assert ledger.observe("piper_0", "holding:brick-1") is None


@pytest.mark.asyncio
async def test_issued_evidence_cannot_be_replayed_or_mutated():
    """Verify test issued evidence cannot be replayed or mutated."""
    request, continuation, _, _ = fixture()
    evidence = await continuation.admit_stationary(request)
    forged = replace(evidence, evidence={**evidence.evidence, "extra": True})
    assert not await continuation.validate_stationary(request, forged)
    assert not await continuation.validate_stationary(request, evidence)


@pytest.mark.asyncio
async def test_final_validation_rechecks_physics_not_only_certificate_hash():
    """Verify test final validation rechecks physics not only certificate hash."""
    request, continuation, _, session = fixture()
    evidence = await continuation.admit_stationary(request)
    session.defect = "no_contacts"
    assert not await continuation.validate_stationary(request, evidence)


@pytest.mark.asyncio
async def test_facade_uses_stationary_admission_and_live_revalidation():
    """The facade uses stationary custody and live revalidation, not TRANSIT."""
    request, continuation, ledger, session = fixture()
    facade = UnifiedDualArmRuntimeFacade(session, state_ledger=ledger)
    facade.bind_handover_continuation(continuation)
    evidence = await facade.execute_pre_phase(request)
    assert evidence.passed, evidence.reason
    assert await facade.validate_pre_phase(request, evidence)


@pytest.mark.asyncio
async def test_pick_lift_proof_cannot_be_replaced_by_planned_effect():
    """Planned effects cannot replace the original measured lift proof."""
    request, continuation, ledger, _ = fixture()
    ledger._states["piper_1"]["holding:brick-1"] = replace(
        ledger.observe("piper_1", "holding:brick-1"),
        evidence={"lift_confirmed": True, "planned_effect": True},
    )
    evidence = await continuation.admit_stationary(request)
    assert not evidence.passed
