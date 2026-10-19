"""Physical evidence: observed physical facts, never planner or command success.

Evidence is the only currency in which a physical fact may be asserted. Its
provenance channel excludes planning, controller and dispatch results outright,
and it must carry an explicit, passing validation proof. A planner success, an
accepted action goal or a returned command is not evidence.

Three outcomes stay distinguishable: ``CONFIRMED`` (observed true), ``ABSENT``
(observed false) and ``UNCERTAIN`` (observation exists but does not settle the
fact). "Never observed" is represented by the absence of any evidence object,
so it can never be read as a confirmed fact.

The proof is structural, not epistemic. A passing check list means the record is
well formed and names checks that claim to have passed; it does not make those
claims true. Trust in a validator identity/version (a trusted validator registry
or allow-list is out of scope here), freshness against a requirement's age and
revision bounds, and session/model/scene provenance checks remain the consumer's
obligation, and these contracts supply no production verifier. A channel is
likewise not a certificate: gripper feedback, an opening or closing command or a
returned status cannot by itself certify ``HOLDING``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from rocobrick.contracts._validation import (
    ContractViolationError,
    parse_enum,
    require_array,
    require_bool,
    require_enum,
    require_float,
    require_int,
    require_keys,
    require_object,
    require_schema,
    require_text,
    require_texts,
)
from rocobrick.contracts.identity import PhaseIdentity

PHYSICAL_EVIDENCE_SCHEMA = "physical-evidence/v1"

_EVIDENCE_KEYS = frozenset(
    {
        "schema",
        "evidence_id",
        "phase_identity",
        "fact",
        "entity_ids",
        "session_id",
        "observed_at_s",
        "world_revision",
        "provenance",
        "validation",
    }
)
_FACT_KEYS = frozenset({"kind", "polarity"})
_PROVENANCE_KEYS = frozenset({"channel", "source_id"})
_VALIDATION_KEYS = frozenset({"validator_id", "validator_version", "checks"})
_CHECK_KEYS = frozenset({"check_id", "passed"})


class ObservationChannel(StrEnum):
    """Channels that may attest a physical fact.

    The vocabulary deliberately has no planner, controller or command member:
    those are solver and transport outcomes, not physical observations.
    """

    SIMULATED_SCENE = "SIMULATED_SCENE"
    SENSOR = "SENSOR"
    GRIPPER_FEEDBACK = "GRIPPER_FEEDBACK"
    OPERATOR_ATTESTATION = "OPERATOR_ATTESTATION"


class PhysicalFactKind(StrEnum):
    """Physical facts a phase may require or verify.

    These are fact kinds only; deciding which observation can establish one is a
    policy question left to the verifier (e.g. gripper command feedback alone
    cannot establish ``HOLDING``).
    """

    HOLDING = "HOLDING"
    SUPPORTING = "SUPPORTING"
    CONNECTED = "CONNECTED"
    RELEASED = "RELEASED"
    RETREATED = "RETREATED"
    AT_POSE = "AT_POSE"
    OBJECT_STABLE = "OBJECT_STABLE"


class EvidencePolarity(StrEnum):
    """Whether an observation confirms, refutes or leaves a fact unsettled."""

    CONFIRMED = "CONFIRMED"
    ABSENT = "ABSENT"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True, slots=True)
class PhysicalFact:
    """One asserted physical fact kind with its observation polarity."""

    kind: PhysicalFactKind
    polarity: EvidencePolarity

    def __post_init__(self) -> None:
        """Validate the fact kind and polarity."""
        require_enum(self.kind, "fact.kind", PhysicalFactKind)
        require_enum(self.polarity, "fact.polarity", EvidencePolarity)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dictionary.

        Returns:
            A dictionary with the wire kind and polarity strings.
        """
        return {"kind": self.kind.value, "polarity": self.polarity.value}

    @classmethod
    def from_dict(cls, payload: object) -> PhysicalFact:
        """Parse one strict JSON physical fact.

        Args:
            payload: Decoded JSON value.

        Returns:
            The parsed fact.
        """
        item = require_object(payload, "fact")
        require_keys(item, _FACT_KEYS, "fact")
        return cls(
            kind=parse_enum(item["kind"], "fact.kind", PhysicalFactKind),
            polarity=parse_enum(item["polarity"], "fact.polarity", EvidencePolarity),
        )


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    """One named predicate that the validating consumer actually evaluated."""

    check_id: str
    passed: bool

    def __post_init__(self) -> None:
        """Validate the check identity and strict boolean outcome."""
        require_text(self.check_id, "validation.check_id")
        require_bool(self.passed, f"validation.{self.check_id}.passed")

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dictionary.

        Returns:
            A dictionary with the check identity and boolean outcome.
        """
        return {"check_id": self.check_id, "passed": self.passed}

    @classmethod
    def from_dict(cls, payload: object) -> ValidationCheck:
        """Parse one strict JSON validation check.

        Args:
            payload: Decoded JSON value.

        Returns:
            The parsed check.
        """
        item = require_object(payload, "validation.check")
        require_keys(item, _CHECK_KEYS, "validation.check")
        return cls(
            check_id=require_text(item["check_id"], "validation.check.check_id"),
            passed=require_bool(item["passed"], "validation.check.passed"),
        )


@dataclass(frozen=True, slots=True)
class ValidationProof:
    """Explicit proof that a validator evaluated and passed named checks.

    An empty check list is rejected, so evidence can never claim a physical
    fact on the strength of a validator name alone. The proof is structural:
    trusting the validator identity and version is the consumer's decision.
    """

    validator_id: str
    validator_version: int
    checks: tuple[ValidationCheck, ...]

    def __post_init__(self) -> None:
        """Validate the validator identity and its non-empty check list."""
        require_text(self.validator_id, "validation.validator_id")
        require_int(self.validator_version, "validation.validator_version", minimum=1)
        if not isinstance(self.checks, tuple):
            raise ContractViolationError("validation.checks must be a tuple")
        if not self.checks:
            raise ContractViolationError("validation.checks must not be empty")
        seen: set[str] = set()
        for check in self.checks:
            if not isinstance(check, ValidationCheck):
                raise ContractViolationError(
                    "validation.checks must contain ValidationCheck values"
                )
            if check.check_id in seen:
                raise ContractViolationError(
                    f"duplicate validation check {check.check_id!r}"
                )
            seen.add(check.check_id)

    @property
    def all_checks_passed(self) -> bool:
        """Return whether every named check passed.

        Returns:
            True when at least one check exists and all of them passed.
        """
        return bool(self.checks) and all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dictionary.

        Returns:
            A dictionary with the validator identity and check list.
        """
        return {
            "validator_id": self.validator_id,
            "validator_version": self.validator_version,
            "checks": [check.to_dict() for check in self.checks],
        }

    @classmethod
    def from_dict(cls, payload: object) -> ValidationProof:
        """Parse one strict JSON validation proof.

        Args:
            payload: Decoded JSON value.

        Returns:
            The parsed proof.
        """
        item = require_object(payload, "validation")
        require_keys(item, _VALIDATION_KEYS, "validation")
        return cls(
            validator_id=require_text(item["validator_id"], "validation.validator_id"),
            validator_version=require_int(
                item["validator_version"],
                "validation.validator_version",
                minimum=1,
            ),
            checks=tuple(
                ValidationCheck.from_dict(entry)
                for entry in require_array(item["checks"], "validation.checks")
            ),
        )


@dataclass(frozen=True, slots=True)
class EvidenceProvenance:
    """Auditable origin of one physical observation."""

    channel: ObservationChannel
    source_id: str

    def __post_init__(self) -> None:
        """Validate the observation channel and source identity."""
        require_enum(self.channel, "provenance.channel", ObservationChannel)
        require_text(self.source_id, "provenance.source_id")

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dictionary.

        Returns:
            A dictionary with the wire channel and source identity.
        """
        return {"channel": self.channel.value, "source_id": self.source_id}

    @classmethod
    def from_dict(cls, payload: object) -> EvidenceProvenance:
        """Parse one strict JSON provenance record.

        Args:
            payload: Decoded JSON value.

        Returns:
            The parsed provenance.
        """
        item = require_object(payload, "provenance")
        require_keys(item, _PROVENANCE_KEYS, "provenance")
        return cls(
            channel=parse_enum(
                item["channel"], "provenance.channel", ObservationChannel
            ),
            source_id=require_text(item["source_id"], "provenance.source_id"),
        )


@dataclass(frozen=True, slots=True)
class PhysicalEvidence:
    """One validated physical observation about specific entities.

    A constructed record is a claim plus its validation provenance, not a
    certificate: consumers must still check validator trust, freshness,
    session/model/scene provenance and polarity before treating it as proof.

    Attributes:
        evidence_id: Unique evidence identity.
        phase_identity: Phase this observation was taken for.
        fact: Observed fact kind and polarity.
        entity_ids: Exact entities the observation is about, in role order.
        session_id: Runtime session that produced the observation.
        observed_at_s: Session-domain observation time in seconds.
        world_revision: World revision the observation belongs to.
        provenance: Observation channel and source identity.
        validation: Explicit validator identity and passed checks.
    """

    evidence_id: str
    phase_identity: PhaseIdentity
    fact: PhysicalFact
    entity_ids: tuple[str, ...]
    session_id: str
    observed_at_s: float
    world_revision: int
    provenance: EvidenceProvenance
    validation: ValidationProof

    def __post_init__(self) -> None:
        """Validate identity, scope, time, revision and validation proof."""
        require_text(self.evidence_id, "evidence.evidence_id")
        if not isinstance(self.phase_identity, PhaseIdentity):
            raise ContractViolationError(
                "evidence.phase_identity must be a PhaseIdentity"
            )
        if not isinstance(self.fact, PhysicalFact):
            raise ContractViolationError("evidence.fact must be a PhysicalFact")
        if not isinstance(self.entity_ids, tuple):
            raise ContractViolationError("evidence.entity_ids must be a tuple")
        require_texts(self.entity_ids, "evidence.entity_ids")
        require_text(self.session_id, "evidence.session_id")
        require_float(self.observed_at_s, "evidence.observed_at_s", minimum=0.0)
        require_int(self.world_revision, "evidence.world_revision")
        if not isinstance(self.provenance, EvidenceProvenance):
            raise ContractViolationError(
                "evidence.provenance must be an EvidenceProvenance"
            )
        if not isinstance(self.validation, ValidationProof):
            raise ContractViolationError(
                "evidence.validation must be a ValidationProof"
            )
        if not self.validation.all_checks_passed:
            raise ContractViolationError(
                "evidence.validation must contain passing checks; "
                "an unvalidated observation is not physical evidence"
            )

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dictionary.

        Returns:
            A dictionary containing only JSON primitives, lists and objects.
        """
        return {
            "schema": PHYSICAL_EVIDENCE_SCHEMA,
            "evidence_id": self.evidence_id,
            "phase_identity": self.phase_identity.to_dict(),
            "fact": self.fact.to_dict(),
            "entity_ids": list(self.entity_ids),
            "session_id": self.session_id,
            "observed_at_s": self.observed_at_s,
            "world_revision": self.world_revision,
            "provenance": self.provenance.to_dict(),
            "validation": self.validation.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: object) -> PhysicalEvidence:
        """Parse one strict JSON physical evidence record.

        Args:
            payload: Decoded JSON value.

        Returns:
            The parsed physical evidence.
        """
        item = require_object(payload, "evidence")
        require_keys(item, _EVIDENCE_KEYS, "evidence")
        require_schema(item["schema"], PHYSICAL_EVIDENCE_SCHEMA, "evidence")
        return cls(
            evidence_id=require_text(item["evidence_id"], "evidence.evidence_id"),
            phase_identity=PhaseIdentity.from_dict(item["phase_identity"]),
            fact=PhysicalFact.from_dict(item["fact"]),
            entity_ids=require_texts(item["entity_ids"], "evidence.entity_ids"),
            session_id=require_text(item["session_id"], "evidence.session_id"),
            observed_at_s=require_float(
                item["observed_at_s"],
                "evidence.observed_at_s",
                minimum=0.0,
            ),
            world_revision=require_int(
                item["world_revision"], "evidence.world_revision"
            ),
            provenance=EvidenceProvenance.from_dict(item["provenance"]),
            validation=ValidationProof.from_dict(item["validation"]),
        )
