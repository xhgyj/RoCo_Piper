"""Phase identity is strict, immutable and duplicate-free across retries."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest
from contract_doubles import identity

from rocobrick.contracts import (
    ContractViolationError,
    PhaseIdentity,
    unique_identities,
)


def test_roundtrip_preserves_identity_and_key() -> None:
    """A parsed identity equals the serialized one and keeps its key."""
    original = identity(phase_id="grasp", attempt=2)
    payload = json.loads(json.dumps(original.to_dict()))
    parsed = PhaseIdentity.from_dict(payload)
    assert parsed == original
    assert parsed.key == ("plan-a", "pick-a", "grasp", 2)


def test_identity_is_immutable() -> None:
    """Identity fields and hashable comparison keys cannot be mutated."""
    value = identity()
    with pytest.raises(FrozenInstanceError):
        value.attempt = 3  # type: ignore[misc]
    assert {value} == {identity()}


def test_attempt_must_be_positive_integer() -> None:
    """Attempt numbering starts at one and rejects bools and floats."""
    with pytest.raises(ContractViolationError, match="attempt must be >= 1"):
        identity(attempt=0)
    with pytest.raises(ContractViolationError, match="attempt must be an integer"):
        identity(attempt=True)  # type: ignore[arg-type]
    with pytest.raises(ContractViolationError, match="attempt must be an integer"):
        identity(attempt=1.0)  # type: ignore[arg-type]


def test_blank_identity_component_is_rejected() -> None:
    """Whitespace-only identifiers are not identities."""
    with pytest.raises(ContractViolationError, match="plan_id"):
        PhaseIdentity(plan_id=" ", operation_id="op", phase_id="p")


def test_duplicate_identities_are_rejected() -> None:
    """Duplicate identities fail before they reach shared state."""
    with pytest.raises(ContractViolationError, match="duplicate phase identity"):
        unique_identities([identity(), identity()])
    ordered = unique_identities([identity(), identity(attempt=2)])
    assert [item.attempt for item in ordered] == [1, 2]


def test_strict_parse_rejects_shape_drift() -> None:
    """Missing, unexpected and wrongly versioned payloads fail closed."""
    payload = identity().to_dict()
    with pytest.raises(ContractViolationError, match="keys mismatch"):
        PhaseIdentity.from_dict({**payload, "extra": 1})
    incomplete = dict(payload)
    del incomplete["attempt"]
    with pytest.raises(ContractViolationError, match="keys mismatch"):
        PhaseIdentity.from_dict(incomplete)
    with pytest.raises(ContractViolationError, match="schema"):
        PhaseIdentity.from_dict({**payload, "schema": "phase-identity/v0"})
