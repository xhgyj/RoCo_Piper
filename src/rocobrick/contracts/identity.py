"""Phase identity shared by every cross-module contract.

A phase is addressed by ``(plan_id, operation_id, phase_id, attempt)``. The
attempt is part of the identity, so a late result from a superseded attempt can
never be mistaken for the current one.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from rocobrick.contracts._validation import (
    ContractViolationError,
    require_int,
    require_keys,
    require_object,
    require_schema,
    require_text,
)

PHASE_IDENTITY_SCHEMA = "phase-identity/v1"

_IDENTITY_KEYS = frozenset({"schema", "plan_id", "operation_id", "phase_id", "attempt"})


@dataclass(frozen=True, slots=True)
class PhaseIdentity:
    """One attempt at one phase of one operation in one plan.

    Attributes:
        plan_id: Frozen upper-layer plan identity.
        operation_id: Operation identity inside that plan.
        phase_id: Phase identity inside that operation.
        attempt: 1-based attempt number; a retry is a new identity.
    """

    plan_id: str
    operation_id: str
    phase_id: str
    attempt: int = 1

    def __post_init__(self) -> None:
        """Validate identity components strictly."""
        require_text(self.plan_id, "identity.plan_id")
        require_text(self.operation_id, "identity.operation_id")
        require_text(self.phase_id, "identity.phase_id")
        require_int(self.attempt, "identity.attempt", minimum=1)

    @property
    def key(self) -> tuple[str, str, str, int]:
        """Return the hashable, comparable identity key.

        Returns:
            The ``(plan_id, operation_id, phase_id, attempt)`` tuple.
        """
        return (self.plan_id, self.operation_id, self.phase_id, self.attempt)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dictionary.

        Returns:
            A dictionary containing only strings and the attempt integer.
        """
        return {
            "schema": PHASE_IDENTITY_SCHEMA,
            "plan_id": self.plan_id,
            "operation_id": self.operation_id,
            "phase_id": self.phase_id,
            "attempt": self.attempt,
        }

    @classmethod
    def from_dict(cls, payload: object) -> PhaseIdentity:
        """Parse one strict JSON phase identity.

        Args:
            payload: Decoded JSON value.

        Returns:
            The parsed identity.
        """
        item = require_object(payload, "identity")
        require_keys(item, _IDENTITY_KEYS, "identity")
        require_schema(item["schema"], PHASE_IDENTITY_SCHEMA, "identity")
        return cls(
            plan_id=require_text(item["plan_id"], "identity.plan_id"),
            operation_id=require_text(item["operation_id"], "identity.operation_id"),
            phase_id=require_text(item["phase_id"], "identity.phase_id"),
            attempt=require_int(item["attempt"], "identity.attempt", minimum=1),
        )


def unique_identities(
    identities: Iterable[PhaseIdentity],
) -> tuple[PhaseIdentity, ...]:
    """Reject duplicate phase identities before they reach shared state.

    Args:
        identities: Identities to check, in their original order.

    Returns:
        The same identities as a tuple.
    """
    seen: set[tuple[str, str, str, int]] = set()
    ordered: list[PhaseIdentity] = []
    for identity in identities:
        if not isinstance(identity, PhaseIdentity):
            raise ContractViolationError("identities must be PhaseIdentity values")
        if identity.key in seen:
            raise ContractViolationError(f"duplicate phase identity {identity.key!r}")
        seen.add(identity.key)
        ordered.append(identity)
    return tuple(ordered)
