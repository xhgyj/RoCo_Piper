"""Physical evidence is validated observation, never planner or controller success."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest
from contract_doubles import evidence, validation

from rocobrick.contracts import (
    ContractViolationError,
    EvidencePolarity,
    ObservationChannel,
    PhysicalEvidence,
    PhysicalFact,
    PhysicalFactKind,
    ValidationProof,
)


def test_roundtrip_preserves_observation() -> None:
    """Evidence survives a JSON roundtrip with every field preserved."""
    original = evidence(observed_at_s=3.25, world_revision=9)
    payload = json.loads(json.dumps(original.to_dict()))
    parsed = PhysicalEvidence.from_dict(payload)
    assert parsed == original
    assert parsed.observed_at_s == 3.25
    assert parsed.world_revision == 9


def test_evidence_is_immutable_including_nested_values() -> None:
    """Evidence and its nested contracts cannot be mutated in place."""
    value = evidence()
    with pytest.raises(FrozenInstanceError):
        value.evidence_id = "other"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        value.fact.polarity = EvidencePolarity.ABSENT  # type: ignore[misc]
    assert isinstance(value.validation.checks, tuple)
    assert isinstance(value.entity_ids, tuple)


def test_observation_time_must_be_finite_and_nonnegative() -> None:
    """NaN, infinity and negative observation times are rejected."""
    for bad in (float("nan"), float("inf"), -0.1):
        with pytest.raises(ContractViolationError, match="observed_at_s"):
            evidence(observed_at_s=bad)


def test_world_revision_rejects_bool_and_negative() -> None:
    """Revisions are real integers; booleans are not numbers here."""
    with pytest.raises(ContractViolationError, match="world_revision"):
        evidence(world_revision=True)  # type: ignore[arg-type]
    with pytest.raises(ContractViolationError, match="world_revision"):
        evidence(world_revision=-1)


def test_channel_vocabulary_excludes_planner_and_controller() -> None:
    """No channel can attest observation from planning or command success."""
    channels = {channel.name for channel in ObservationChannel}
    assert channels == {
        "SIMULATED_SCENE",
        "SENSOR",
        "GRIPPER_FEEDBACK",
        "OPERATOR_ATTESTATION",
    }
    payload = evidence().to_dict()
    provenance = dict(payload["provenance"])  # type: ignore[arg-type]
    with pytest.raises(ContractViolationError, match="channel"):
        PhysicalEvidence.from_dict(
            {
                **payload,
                "provenance": {**provenance, "channel": "PLANNER_MODEL"},
            }
        )


def test_failed_validation_proof_is_not_evidence() -> None:
    """Evidence requires passing checks; a bare claim is insufficient."""
    with pytest.raises(ContractViolationError, match="not physical evidence"):
        evidence(validation_proof=validation(passed=False))
    with pytest.raises(ContractViolationError, match="checks"):
        ValidationProof(validator_id="v", validator_version=1, checks=())


def test_duplicate_checks_and_entities_are_rejected() -> None:
    """Repeated check ids and repeated entity ids are ambiguous."""
    check = validation().checks[0]
    with pytest.raises(ContractViolationError, match="duplicate validation check"):
        ValidationProof(
            validator_id="v",
            validator_version=1,
            checks=(check, check),
        )
    with pytest.raises(ContractViolationError, match="duplicates"):
        evidence(entity_ids=("brick_a", "brick_a"))


def test_uncertain_and_absent_facts_are_distinct_from_confirmed() -> None:
    """Polarity stays explicit so absence and doubt are never success."""
    fact = PhysicalFact(
        kind=PhysicalFactKind.HOLDING, polarity=EvidencePolarity.UNCERTAIN
    )
    uncertain = evidence(fact=fact)
    assert uncertain.fact.polarity is EvidencePolarity.UNCERTAIN
    assert uncertain.fact.polarity is not EvidencePolarity.CONFIRMED
