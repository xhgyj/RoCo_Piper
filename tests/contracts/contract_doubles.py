"""Builders for valid contract values used by the contract tests."""

from __future__ import annotations

from rocobrick.contracts.evidence import (
    EvidencePolarity,
    EvidenceProvenance,
    ObservationChannel,
    PhysicalEvidence,
    PhysicalFact,
    PhysicalFactKind,
    ValidationCheck,
    ValidationProof,
)
from rocobrick.contracts.identity import PhaseIdentity
from rocobrick.contracts.motion import JointSample, JointTrajectory, MotionCommit
from rocobrick.contracts.phases import (
    ContactDeclaration,
    ContactMode,
    GoalDeclaration,
    GoalKind,
    MotionPermission,
    PermissionKind,
    PhaseIntent,
    PhysicalRequirement,
    ResourceKind,
    ResourceRequirement,
)


def identity(phase_id: str = "approach", attempt: int = 1) -> PhaseIdentity:
    """Return one valid phase identity."""
    return PhaseIdentity(
        plan_id="plan-a",
        operation_id="pick-a",
        phase_id=phase_id,
        attempt=attempt,
    )


def validation(
    *, passed: bool = True, check_id: str = "distance_within_tolerance"
) -> ValidationProof:
    """Return one valid validation proof with a single named check."""
    return ValidationProof(
        validator_id="pick_post_grasp",
        validator_version=1,
        checks=(ValidationCheck(check_id=check_id, passed=passed),),
    )


def evidence(
    *,
    evidence_id: str = "ev-1",
    phase_identity: PhaseIdentity | None = None,
    fact: PhysicalFact | None = None,
    entity_ids: tuple[str, ...] = ("piper_0", "brick_a"),
    session_id: str = "session-1",
    observed_at_s: float = 12.5,
    world_revision: int = 7,
    channel: ObservationChannel = ObservationChannel.SIMULATED_SCENE,
    validation_proof: ValidationProof | None = None,
) -> PhysicalEvidence:
    """Return one valid physical evidence record."""
    return PhysicalEvidence(
        evidence_id=evidence_id,
        phase_identity=phase_identity or identity(),
        fact=fact
        or PhysicalFact(
            kind=PhysicalFactKind.HOLDING,
            polarity=EvidencePolarity.CONFIRMED,
        ),
        entity_ids=entity_ids,
        session_id=session_id,
        observed_at_s=observed_at_s,
        world_revision=world_revision,
        provenance=EvidenceProvenance(
            channel=channel, source_id="host_bridge.scene_state"
        ),
        validation=validation_proof or validation(),
    )


def requirement(
    *,
    fact_kind: PhysicalFactKind = PhysicalFactKind.HOLDING,
    polarity: EvidencePolarity = EvidencePolarity.CONFIRMED,
    entity_ids: tuple[str, ...] = ("piper_0", "brick_a"),
    max_age_s: float = 0.25,
    minimum_world_revision: int = 7,
) -> PhysicalRequirement:
    """Return one valid physical requirement."""
    return PhysicalRequirement(
        fact_kind=fact_kind,
        polarity=polarity,
        entity_ids=entity_ids,
        max_age_s=max_age_s,
        minimum_world_revision=minimum_world_revision,
    )


def intent(
    *,
    phase_identity: PhaseIdentity | None = None,
    predecessor_phase_ids: tuple[str, ...] = (),
    participants: tuple[str, ...] = ("piper_0",),
    physical_requirements: tuple[PhysicalRequirement, ...] = (),
    resources: tuple[ResourceRequirement, ...] = (),
    permissions: tuple[MotionPermission, ...] | None = None,
    contact: ContactDeclaration | None = None,
) -> PhaseIntent:
    """Return one valid phase intent."""
    bound = participants
    return PhaseIntent(
        identity=phase_identity or identity(),
        predecessor_phase_ids=predecessor_phase_ids,
        participants=participants,
        physical_requirements=physical_requirements,
        resources=resources
        or (ResourceRequirement(kind=ResourceKind.ROBOT, resource_id="robot:piper_0"),),
        goal=GoalDeclaration(
            kind=GoalKind.RELATIVE_RELATION,
            target_ref="grasp-region:pregrasp",
            candidate_set_refs=("candidate-set:pregrasp",),
            constraint_refs=("constraint:carry-envelope",),
        ),
        permissions=permissions
        or tuple(
            MotionPermission(
                robot_id=robot_id,
                kind=(
                    PermissionKind.CONTACT_PRESERVING
                    if contact is not None
                    else PermissionKind.CARRY_CONSTRAINED
                ),
                writable_joint_names=(f"{robot_id}_joint_1",),
                obligations=("preserve grasp",),
            )
            for robot_id in bound
        ),
        contact=contact,
    )


def trajectory(
    *,
    robot_id: str = "piper_0",
    joint_names: tuple[str, ...] | None = None,
    times: tuple[float, ...] = (0.0, 0.5, 1.0),
) -> JointTrajectory:
    """Return one valid single-robot joint trajectory."""
    names = joint_names or (f"{robot_id}_joint_1", f"{robot_id}_joint_2")
    return JointTrajectory(
        robot_id=robot_id,
        joint_names=names,
        samples=tuple(
            JointSample(
                time_from_start_s=time_from_start_s,
                positions=tuple(0.1 * index for _ in names),
                velocities=tuple(0.0 for _ in names),
            )
            for index, time_from_start_s in enumerate(times)
        ),
    )


def commit(
    *,
    participants: tuple[str, ...] = ("piper_0", "piper_1"),
    trajectories: tuple[JointTrajectory, ...] | None = None,
    start_at_s: float = 100.0,
    session_id: str = "session-1",
) -> MotionCommit:
    """Return one valid two-participant motion commit."""
    return MotionCommit(
        commit_id="commit-1",
        identity=identity(),
        session_id=session_id,
        participants=participants,
        epoch=3,
        world_revision=7,
        trajectory_revision=2,
        start_at_s=start_at_s,
        world_evidence_ids=("ev-world-1",),
        lease_ids=("robot:piper_0", "robot:piper_1"),
        reservation_ids=("reservation-1",),
        trajectories=trajectories
        or tuple(trajectory(robot_id=robot_id) for robot_id in participants),
    )


def contact() -> ContactDeclaration:
    """Return one valid contact declaration."""
    return ContactDeclaration(
        mode=ContactMode.MAINTAIN,
        robot_link_ids=("piper_0_gripper_link1",),
        entity_ids=("brick_a",),
        entry_condition_ref="contact:pregrasp-closed",
        exit_condition_ref=None,
    )
