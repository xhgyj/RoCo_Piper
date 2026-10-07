"""Full transfer lifecycle tests with explicitly synthetic physics/actions."""

import asyncio
from dataclasses import replace

import pytest
from test_handover_joint_planning import planning_request, snapshot
from test_robot_operation_plan_and_scheduler import handover_operation, plan

from rocobrick.adapters.moveit.handover_scene_planner import HandoverPhasePaths
from rocobrick.coordinated_moveit import (
    CoordinatedOperation,
    CoordinatedRobotTrajectory,
    CoordinatedTrajectoryPoint,
)
from rocobrick.execution.operation_state_ledger import OperationStateLedger
from rocobrick.robot_operation_plan import parse_robot_operation_plan
from rocobrick.skills.coordinated_handover import (
    PHASES,
    CoordinatedHandoverBackend,
    CoordinatedHandoverProgram,
    HandoverPhaseReceipt,
)
from rocobrick.skills.handover_planning import generate_candidates
from rocobrick.skills.handover_profile import EXECUTION_CAPABILITIES
from rocobrick.skills.observation import Pose


def program():
    """Use a qualified SYNTHETIC policy, never alter the installed profile.

    Returns:
        The identified, bound result; physical proof remains separately checked.
    """
    request = planning_request()
    request = replace(
        request, profile=replace(request.profile, execution_qualified=True)
    )
    candidate = generate_candidates(request)[0]
    state = dict(request.start_state)
    paths = HandoverPhasePaths(
        candidate.candidate_id,
        (state, state),
        (state, state),
        (state, state),
        (state, state),
        (state, state),
        (state, state),
        Pose((0, 0, 0), (0, 0, 0, 1)),
    )
    trajectories = tuple(
        CoordinatedRobotTrajectory(
            r,
            tuple(f"{r}_joint{i}" for i in range(1, 7)),
            (
                CoordinatedTrajectoryPoint(0, (0.0,) * 6),
                CoordinatedTrajectoryPoint(1, (0.0,) * 6),
            ),
        )
        for r in (request.donor.robot_id, request.receiver.robot_id)
    )
    motion = CoordinatedOperation(
        candidate.candidate_id,
        request.scene_snapshot["scene_digest"],
        (request.donor.robot_id, request.receiver.robot_id),
        trajectories,
        0,
    )
    return CoordinatedHandoverProgram(
        request,
        candidate,
        motion,
        paths,
        EXECUTION_CAPABILITIES,
        (0.015, 0.017),
        (0.015, 0.017),
        (0.075, 0.085),
        2.0,
    )


def operation(p):
    """Parse an unmodified formal HANDOVER, with fixture identities.

    Returns:
        The identified, bound result; physical proof remains separately checked.
    """
    raw = handover_operation(p.identity.operation_id, *p.participants)
    raw["parameters"]["grasp_profile_id"] = p.request.profile.profile_id
    parsed = parse_robot_operation_plan(
        plan(
            [raw],
            [[raw["operation_id"]]],
            [
                [
                    {
                        "operation_id": raw["operation_id"],
                        "robot_id": raw["executor_robot_id"],
                    }
                ]
            ],
        )
    )
    return parsed.operations[raw["operation_id"]]


class Factory:
    """Offline prepared-program provider, with no reachable runtime."""

    def __init__(self, prepared):
        """Initialize the explicitly bound fixture state."""
        self.prepared = prepared

    async def prepare(self, op):
        """Return the already admitted synthetic program.

        Returns:
            The bound result of this read-only test boundary.
        """
        return self.prepared


class Transport:
    """Synthetic current-tick physics and independent owned action completion."""

    def __init__(self, ledger, *, fail=None, defect=None):
        """Initialize the explicitly bound fixture state."""
        self.ledger, self.fail, self.defect = ledger, fail, defect
        self.calls = []
        self.reads = 1
        self.phase = None
        self.released = False
        self.displacement = 0.0
        self.cancelled = False
        self.release_states = []
        self.following = asyncio.Event()

    async def observe(self, p):
        """Read the next identified same-tick physics sample.

        Returns:
            The bound result of this read-only test boundary.
        """
        await asyncio.sleep(0)
        self.reads += 1
        if self.phase == "receiver_follow" and self.defect != "stationary_follow":
            self.displacement = min(0.016, self.displacement + 0.004)
        raw = snapshot(self.reads, self.displacement, released=self.released)
        if self.defect == "no_giver_contacts":
            raw["pad_object_contacts"]["contacts"] = [
                c
                for c in raw["pad_object_contacts"]["contacts"]
                if c["robot_id"] != "piper_1"
            ]
        if self.defect == "stale" and self.phase == "approach":
            raw = snapshot(2)
        if self.defect == "receiver_slip" and self.released:
            raw["robots"]["piper_0"]["tcp_pose"]["position_xyz"][0] += 0.01
        if self.defect == "giver_contact" and self.released:
            raw["pad_object_contacts"]["contacts"] += snapshot(self.reads)[
                "pad_object_contacts"
            ]["contacts"][2:]
        if self.defect == "no_receiver_contacts" and self.phase == "receiver_close":
            raw["pad_object_contacts"]["contacts"] = [
                c
                for c in raw["pad_object_contacts"]["contacts"]
                if c["robot_id"] != "piper_0"
            ]
        return raw

    async def execute_phase(self, p, phase, attachment):
        """Execute the exact bound phase and await physical completion.

        Returns:
            The bound result of this read-only test boundary.
        """
        self.phase = phase
        self.calls.append(phase)
        start = self.reads
        if phase == "donor_open":
            self.release_states.append(
                (
                    self.ledger.owner_of_object("brick-1"),
                    self.ledger.observe("piper_1", "holding:brick-1"),
                    self.ledger.observe("piper_0", "holding:brick-1"),
                )
            )
            self.released = True
        if phase == "receiver_follow":
            self.following.set()
        while self.reads < start + 4 and not self.cancelled:
            await asyncio.sleep(0)
        if self.fail == phase:
            raise RuntimeError("injected physical phase failure")
        proof = {
            "prepared_path_executed": True,
            "arrival_confirmed": True,
            "tracking_certified": True,
            "replanned": False,
            "gripper_feedback_confirmed": True,
            "sweep_certified": True,
            "transfer_status": "COMMITTED",
            "readback_confirmed": True,
        }
        if self.defect == "replanned" and phase == "approach":
            proof["replanned"] = True
        if self.defect == "no_readback" and phase == "scene_transfer":
            proof["readback_confirmed"] = False
        return HandoverPhaseReceipt(
            p.identity.operation_id,
            p.binding_sha256,
            phase,
            not self.cancelled,
            True,
            True,
            proof,
        )

    async def cancel(self, p):
        """Cancel both owned physical action lanes.

        Returns:
            The bound result of this read-only test boundary.
        """
        self.cancelled = True
        return {
            "operation_id": p.identity.operation_id,
            "participants": {
                r: {
                    "terminal": True,
                    "safe_to_release": self.defect != "unsafe_stop",
                }
                for r in p.participants
            },
        }


def fixture(**kwargs):
    """Admit the initial giver and return an entirely offline transfer.

    Returns:
        The identified, bound result; physical proof remains separately checked.
    """
    p = program()
    ledger = OperationStateLedger(p.request.session_robots)
    ledger.confirm("piper_1", "holding:brick-1", {"actual_holding": True})
    transport = Transport(ledger, **kwargs)
    backend = CoordinatedHandoverBackend(Factory(p), transport, ledger)
    return p, ledger, transport, backend


@pytest.mark.asyncio
async def test_full_handover_revokes_giver_before_release_and_commits_after_follow():
    """All eight physical phases precede one and only one final owner adoption."""
    p, ledger, transport, backend = fixture()
    result = await backend.execute(operation(p))
    assert result.passed, result.reason
    assert transport.calls == list(PHASES)
    assert transport.release_states == [("piper_1", None, None)]
    assert ledger.owner_of_object("brick-1") == "piper_0"
    assert ledger.observe("piper_1", "holding:brick-1") is None
    assert ledger.observe("piper_0", "holding:brick-1") is not None
    assert ledger.provisional_handover("brick-1") is None
    assert not ledger.uncertain_keys("piper_0")
    assert result.evidence["receiver_holding_follow"] is True
    assert result.evidence["receiver_follow_proof"]["object_distance_m"] >= 0.012


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", PHASES)
async def test_every_phase_failure_stops_both_and_retains_uncertain_custody(phase):
    """A terminal controller stop cannot become successful custody or lease release."""
    p, ledger, transport, backend = fixture(fail=phase)
    result = await backend.execute(operation(p))
    assert result.passed is False and result.motion_sent is True
    assert result.evidence["safe_to_release"] is False
    assert transport.cancelled is True
    assert ledger.owner_of_object("brick-1") == "piper_1"
    assert ledger.observe("piper_0", "holding:brick-1") is None
    assert ledger.observe("piper_1", "holding:brick-1") is None
    assert ledger.uncertain_handover("brick-1") is not None
    assert "owner_transfer_committed" not in result.evidence


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "defect",
    [
        "stale",
        "receiver_slip",
        "giver_contact",
        "no_receiver_contacts",
        "stationary_follow",
        "replanned",
        "no_readback",
    ],
)
async def test_missing_physical_proof_never_adopts_receiver(defect):
    """Nominal paths/openings and stale/static samples cannot substitute for proof."""
    p, ledger, transport, backend = fixture(defect=defect)
    result = await backend.execute(operation(p))
    assert result.passed is False
    assert result.evidence["safe_to_release"] is False
    assert ledger.owner_of_object("brick-1") == "piper_1"
    if defect == "no_receiver_contacts":
        assert "donor_open" not in transport.calls


@pytest.mark.asyncio
async def test_failure_before_dispatch_reports_no_motion_and_preserves_giver():
    """A role mismatch is rejected before receiver opening or controller creation."""
    p, ledger, transport, backend = fixture()
    wrong = replace(operation(p), executor_robot_id="piper_1")
    result = await backend.execute(wrong)
    assert result.passed is False and result.motion_sent is False
    assert result.terminal is True and result.evidence["safe_to_release"] is True
    assert not transport.calls
    assert ledger.observe("piper_1", "holding:brick-1") is not None


@pytest.mark.asyncio
async def test_cancellation_cannot_claim_safe_object_lease_from_safe_action_stop():
    """Physical cancellation preserves protection of unresolved custody."""
    p, ledger, transport, backend = fixture()
    task = asyncio.create_task(backend.execute(operation(p)))
    await transport.following.wait()
    proof = await backend.cancel(p.identity.operation_id)
    result = await task
    assert proof["safe_to_release"] is False
    assert result.passed is False
    assert transport.cancelled
    assert ledger.uncertain_handover("brick-1") is not None


def test_installed_draft_profile_is_not_upgraded_by_full_handler():
    """Offline lifecycle coverage is not a physical qualification claim."""
    p = program()
    with pytest.raises(ValueError, match="NOT_QUALIFIED"):
        replace(p, request=planning_request())


def test_bound_program_detects_nested_planning_input_mutation():
    """Frozen dataclasses do not hide mutable nested source dictionaries."""
    p = program()
    p.request.start_state["piper_1_joint1"] = 0.3
    assert p.current_binding() != p.binding_sha256


@pytest.mark.asyncio
async def test_missing_initial_giver_contact_revokes_custody_without_motion():
    """Failed retention invalidates custody even before the first motor command."""
    p, ledger, transport, backend = fixture(defect="no_giver_contacts")
    result = await backend.execute(operation(p))
    assert not result.passed and not result.motion_sent
    assert not transport.calls
    assert result.evidence["safe_to_release"] is False
    assert ledger.observe("piper_1", "holding:brick-1") is None
    assert ledger.owner_of_object("brick-1") == "piper_1"
    assert ledger.uncertain_handover("brick-1") is not None
