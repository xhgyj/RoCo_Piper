"""Phase intent declares dependencies, resources and per-arm permissions."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest
from contract_doubles import contact, identity, intent, requirement

from rocobrick.contracts import (
    ContactMode,
    ContractViolationError,
    EvidencePolarity,
    GoalDeclaration,
    GoalKind,
    MotionPermission,
    PermissionKind,
    PhaseIntent,
    PhysicalFactKind,
    ResourceKind,
    ResourceRequirement,
)


def test_roundtrip_preserves_declarations() -> None:
    """A full intent survives a JSON roundtrip unchanged."""
    original = intent(
        phase_identity=identity(phase_id="grasp"),
        predecessor_phase_ids=("approach",),
        participants=("piper_0", "piper_1"),
        physical_requirements=(
            requirement(entity_ids=("piper_0", "brick_a")),
            requirement(
                fact_kind=PhysicalFactKind.SUPPORTING,
                polarity=EvidencePolarity.ABSENT,
                entity_ids=("brick_a",),
            ),
        ),
        contact=contact(),
    )
    payload = json.loads(json.dumps(original.to_dict()))
    parsed = PhaseIntent.from_dict(payload)
    assert parsed == original
    assert parsed.contact is not None
    assert parsed.contact.mode is ContactMode.MAINTAIN


def test_intent_is_immutable_including_nested_values() -> None:
    """Nested declarations are frozen dataclasses and tuples."""
    value = intent()
    with pytest.raises(FrozenInstanceError):
        value.goal.kind = GoalKind.POSE  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        value.permissions[0].kind = PermissionKind.IN_FLIGHT  # type: ignore[misc]
    assert isinstance(value.permissions, tuple)
    assert isinstance(value.permissions[0].writable_joint_names, tuple)


def test_permissions_must_bind_every_participant_once() -> None:
    """Every participating arm needs exactly one permission declaration."""
    with pytest.raises(ContractViolationError, match=r"missing \['piper_1'\]"):
        intent(
            participants=("piper_0", "piper_1"),
            permissions=(
                MotionPermission(
                    robot_id="piper_0",
                    kind=PermissionKind.FREE_TO_COORDINATE,
                    writable_joint_names=("piper_0_joint_1",),
                ),
            ),
        )
    with pytest.raises(ContractViolationError, match="unknown"):
        intent(
            participants=("piper_0",),
            permissions=(
                MotionPermission(
                    robot_id="piper_0",
                    kind=PermissionKind.FREE_TO_COORDINATE,
                    writable_joint_names=("piper_0_joint_1",),
                ),
                MotionPermission(
                    robot_id="piper_9",
                    kind=PermissionKind.FREE_TO_COORDINATE,
                    writable_joint_names=("piper_9_joint_1",),
                ),
            ),
        )


def test_duplicate_permissions_and_resources_are_rejected() -> None:
    """Duplicated ownership declarations are ambiguous and fail closed."""
    permission = MotionPermission(
        robot_id="piper_0",
        kind=PermissionKind.FREE_TO_COORDINATE,
        writable_joint_names=("piper_0_joint_1",),
    )
    with pytest.raises(ContractViolationError, match="duplicate motion permission"):
        intent(permissions=(permission, permission))
    with pytest.raises(ContractViolationError, match="duplicate resource"):
        intent(
            resources=(
                ResourceRequirement(
                    kind=ResourceKind.ROBOT, resource_id="robot:piper_0"
                ),
                ResourceRequirement(
                    kind=ResourceKind.ROBOT, resource_id="robot:piper_0"
                ),
            )
        )


def test_phase_cannot_depend_on_itself() -> None:
    """A phase is never its own predecessor or duplicate requirement."""
    with pytest.raises(ContractViolationError, match="phase itself"):
        intent(predecessor_phase_ids=("approach",))
    duplicate = requirement()
    with pytest.raises(ContractViolationError, match="duplicate physical"):
        intent(physical_requirements=(duplicate, duplicate))


def test_resource_vocabulary_is_closed() -> None:
    """Resources use the shared robot/part/connection vocabulary."""
    assert {kind.value for kind in ResourceKind} == {
        "robot",
        "part",
        "connection",
    }
    with pytest.raises(ContractViolationError, match="resource_id"):
        ResourceRequirement(kind=ResourceKind.PART, resource_id="brick_a")


def test_goal_requires_a_target_reference() -> None:
    """A goal without a target reference is not a goal."""
    with pytest.raises(ContractViolationError, match="target_ref"):
        GoalDeclaration(kind=GoalKind.POSE, target_ref="")


def test_strict_parse_rejects_payload_drift() -> None:
    """Unexpected keys and unknown enum wire values are rejected."""
    payload = intent().to_dict()
    with pytest.raises(ContractViolationError, match="keys mismatch"):
        PhaseIntent.from_dict({**payload, "extra": True})
    with pytest.raises(ContractViolationError, match="goal"):
        PhaseIntent.from_dict(
            {
                **payload,
                "goal": {
                    "kind": "MAGIC",
                    "target_ref": "pose:1",
                    "candidate_set_refs": [],
                    "constraint_refs": [],
                },
            }
        )


def test_contradictory_physical_requirements_are_rejected() -> None:
    """The same fact cannot be required both confirmed and absent."""
    with pytest.raises(ContractViolationError, match="contradictory"):
        intent(
            physical_requirements=(
                requirement(polarity=EvidencePolarity.CONFIRMED),
                requirement(polarity=EvidencePolarity.ABSENT),
            )
        )


def test_same_fact_with_different_entities_is_not_contradictory() -> None:
    """Contradiction is scoped to the fact and its exact entity set."""
    declared = intent(
        physical_requirements=(
            requirement(polarity=EvidencePolarity.CONFIRMED),
            requirement(
                polarity=EvidencePolarity.ABSENT,
                entity_ids=("piper_0", "brick_b"),
            ),
        )
    )
    assert len(declared.physical_requirements) == 2


def test_permissions_must_not_weaken_a_contact_obligation() -> None:
    """A MAINTAIN contact cannot be combined with a weaker permission kind."""
    with pytest.raises(ContractViolationError, match="contact obligation"):
        intent(
            contact=contact(),
            permissions=(
                MotionPermission(
                    robot_id="piper_0",
                    kind=PermissionKind.FREE_TO_COORDINATE,
                    writable_joint_names=("piper_0_joint_1",),
                ),
            ),
        )


def test_contact_preserving_permission_satisfies_contact() -> None:
    """An explicitly contact-preserving permission keeps the obligation."""
    declared = intent(
        contact=contact(),
        permissions=(
            MotionPermission(
                robot_id="piper_0",
                kind=PermissionKind.CONTACT_PRESERVING,
                writable_joint_names=("piper_0_joint_1",),
                obligations=("preserve grasp",),
            ),
        ),
    )
    assert declared.permissions[0].kind is PermissionKind.CONTACT_PRESERVING
