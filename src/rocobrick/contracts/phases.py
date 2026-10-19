"""Phase intent: what a phase wants, requires, may move and must preserve.

A ``PhaseIntent`` is descriptive. It declares dependencies, physical
preconditions, resources, goal/candidate/constraint references, contacts and
per-arm motion permissions. It contains no trajectories, no dispatch decision
and no physical threshold: geometry, feasibility and acceptance limits stay
with the owning skill, planner or calibrated configuration.

Declaring an intent is not authorization: a candidate set is not a motion
commit, and ``max_age_s``/``minimum_world_revision`` are bounds the verifier must
enforce, not checks performed here. Opaque ``*_ref`` strings are identities
resolved by the consuming module; they are deliberately not free-form
dictionaries.
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
    require_optional_text,
    require_schema,
    require_text,
    require_texts,
)
from rocobrick.contracts.evidence import EvidencePolarity, PhysicalFactKind
from rocobrick.contracts.identity import PhaseIdentity

PHASE_INTENT_SCHEMA = "phase-intent/v1"

_INTENT_KEYS = frozenset(
    {
        "schema",
        "identity",
        "predecessor_phase_ids",
        "participants",
        "physical_requirements",
        "resources",
        "goal",
        "permissions",
        "contact",
    }
)
_RESOURCE_KEYS = frozenset({"kind", "resource_id", "exclusive"})
_REQUIREMENT_KEYS = frozenset(
    {
        "fact_kind",
        "polarity",
        "entity_ids",
        "max_age_s",
        "minimum_world_revision",
    }
)
_GOAL_KEYS = frozenset({"kind", "target_ref", "candidate_set_refs", "constraint_refs"})
_PERMISSION_KEYS = frozenset(
    {"robot_id", "kind", "writable_joint_names", "obligations"}
)
_CONTACT_KEYS = frozenset(
    {
        "mode",
        "robot_link_ids",
        "entity_ids",
        "entry_condition_ref",
        "exit_condition_ref",
    }
)


class GoalKind(StrEnum):
    """How a phase goal is described to the planner."""

    POSE = "POSE"
    JOINT_REGION = "JOINT_REGION"
    RELATIVE_RELATION = "RELATIVE_RELATION"
    NAMED_TARGET = "NAMED_TARGET"


class ContactMode(StrEnum):
    """Contact obligation declared for a phase."""

    APPROACH = "APPROACH"
    MAKE = "MAKE"
    MAINTAIN = "MAINTAIN"
    BREAK = "BREAK"


class PermissionKind(StrEnum):
    """Motion permission class for one participating arm."""

    FREE_TO_COORDINATE = "FREE_TO_COORDINATE"
    CARRY_CONSTRAINED = "CARRY_CONSTRAINED"
    CONTACT_PRESERVING = "CONTACT_PRESERVING"
    EXPLICIT_HOLD = "EXPLICIT_HOLD"
    IN_FLIGHT = "IN_FLIGHT"


class ResourceKind(StrEnum):
    """Resource vocabulary shared with the operation-plan resources."""

    ROBOT = "robot"
    PART = "part"
    CONNECTION = "connection"


@dataclass(frozen=True, slots=True)
class ResourceRequirement:
    """One resource lease a phase needs before it may proceed."""

    kind: ResourceKind
    resource_id: str
    exclusive: bool = True

    def __post_init__(self) -> None:
        """Validate the resource kind, vocabulary prefix and exclusivity."""
        require_enum(self.kind, "resource.kind", ResourceKind)
        require_text(self.resource_id, "resource.resource_id")
        prefix = f"{self.kind.value}:"
        if not self.resource_id.startswith(prefix):
            raise ContractViolationError(
                f"resource.resource_id must start with {prefix!r}"
            )
        if not self.resource_id[len(prefix) :].strip():
            raise ContractViolationError(
                "resource.resource_id must name a resource after its prefix"
            )
        require_bool(self.exclusive, "resource.exclusive")

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dictionary.

        Returns:
            A dictionary with the wire resource kind and identity.
        """
        return {
            "kind": self.kind.value,
            "resource_id": self.resource_id,
            "exclusive": self.exclusive,
        }

    @classmethod
    def from_dict(cls, payload: object) -> ResourceRequirement:
        """Parse one strict JSON resource requirement.

        Args:
            payload: Decoded JSON value.

        Returns:
            The parsed resource requirement.
        """
        item = require_object(payload, "resource")
        require_keys(item, _RESOURCE_KEYS, "resource")
        return cls(
            kind=parse_enum(item["kind"], "resource.kind", ResourceKind),
            resource_id=require_text(item["resource_id"], "resource.resource_id"),
            exclusive=require_bool(item["exclusive"], "resource.exclusive"),
        )


@dataclass(frozen=True, slots=True)
class PhysicalRequirement:
    """One physical fact a phase requires, with freshness and revision bounds.

    ``max_age_s`` and ``minimum_world_revision`` are supplied by the caller;
    this contract invents no safety threshold.
    """

    fact_kind: PhysicalFactKind
    polarity: EvidencePolarity
    entity_ids: tuple[str, ...]
    max_age_s: float
    minimum_world_revision: int

    def __post_init__(self) -> None:
        """Validate the required fact, its entities, age bound and revision."""
        require_enum(self.fact_kind, "requirement.fact_kind", PhysicalFactKind)
        require_enum(self.polarity, "requirement.polarity", EvidencePolarity)
        if not isinstance(self.entity_ids, tuple):
            raise ContractViolationError("requirement.entity_ids must be a tuple")
        require_texts(self.entity_ids, "requirement.entity_ids")
        require_float(self.max_age_s, "requirement.max_age_s", minimum=0.0)
        require_int(
            self.minimum_world_revision,
            "requirement.minimum_world_revision",
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dictionary.

        Returns:
            A dictionary with the wire fact, entities, age and revision.
        """
        return {
            "fact_kind": self.fact_kind.value,
            "polarity": self.polarity.value,
            "entity_ids": list(self.entity_ids),
            "max_age_s": self.max_age_s,
            "minimum_world_revision": self.minimum_world_revision,
        }

    @classmethod
    def from_dict(cls, payload: object) -> PhysicalRequirement:
        """Parse one strict JSON physical requirement.

        Args:
            payload: Decoded JSON value.

        Returns:
            The parsed physical requirement.
        """
        item = require_object(payload, "requirement")
        require_keys(item, _REQUIREMENT_KEYS, "requirement")
        return cls(
            fact_kind=parse_enum(
                item["fact_kind"], "requirement.fact_kind", PhysicalFactKind
            ),
            polarity=parse_enum(
                item["polarity"], "requirement.polarity", EvidencePolarity
            ),
            entity_ids=require_texts(item["entity_ids"], "requirement.entity_ids"),
            max_age_s=require_float(
                item["max_age_s"], "requirement.max_age_s", minimum=0.0
            ),
            minimum_world_revision=require_int(
                item["minimum_world_revision"],
                "requirement.minimum_world_revision",
            ),
        )


@dataclass(frozen=True, slots=True)
class GoalDeclaration:
    """Reference-based goal declaration resolved by the owning consumer."""

    kind: GoalKind
    target_ref: str
    candidate_set_refs: tuple[str, ...] = ()
    constraint_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the goal kind, target reference and reference lists."""
        require_enum(self.kind, "goal.kind", GoalKind)
        require_text(self.target_ref, "goal.target_ref")
        if not isinstance(self.candidate_set_refs, tuple):
            raise ContractViolationError("goal.candidate_set_refs must be a tuple")
        require_texts(
            self.candidate_set_refs,
            "goal.candidate_set_refs",
            allow_empty=True,
        )
        if not isinstance(self.constraint_refs, tuple):
            raise ContractViolationError("goal.constraint_refs must be a tuple")
        require_texts(self.constraint_refs, "goal.constraint_refs", allow_empty=True)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dictionary.

        Returns:
            A dictionary with the wire goal kind and reference lists.
        """
        return {
            "kind": self.kind.value,
            "target_ref": self.target_ref,
            "candidate_set_refs": list(self.candidate_set_refs),
            "constraint_refs": list(self.constraint_refs),
        }

    @classmethod
    def from_dict(cls, payload: object) -> GoalDeclaration:
        """Parse one strict JSON goal declaration.

        Args:
            payload: Decoded JSON value.

        Returns:
            The parsed goal declaration.
        """
        item = require_object(payload, "goal")
        require_keys(item, _GOAL_KEYS, "goal")
        return cls(
            kind=parse_enum(item["kind"], "goal.kind", GoalKind),
            target_ref=require_text(item["target_ref"], "goal.target_ref"),
            candidate_set_refs=require_texts(
                item["candidate_set_refs"],
                "goal.candidate_set_refs",
                allow_empty=True,
            ),
            constraint_refs=require_texts(
                item["constraint_refs"],
                "goal.constraint_refs",
                allow_empty=True,
            ),
        )


@dataclass(frozen=True, slots=True)
class ContactDeclaration:
    """Contact obligation between named robot links and named entities."""

    mode: ContactMode
    robot_link_ids: tuple[str, ...]
    entity_ids: tuple[str, ...]
    entry_condition_ref: str | None = None
    exit_condition_ref: str | None = None

    def __post_init__(self) -> None:
        """Validate the contact mode, participants and condition references."""
        require_enum(self.mode, "contact.mode", ContactMode)
        if not isinstance(self.robot_link_ids, tuple):
            raise ContractViolationError("contact.robot_link_ids must be a tuple")
        require_texts(self.robot_link_ids, "contact.robot_link_ids")
        if not isinstance(self.entity_ids, tuple):
            raise ContractViolationError("contact.entity_ids must be a tuple")
        require_texts(self.entity_ids, "contact.entity_ids")
        require_optional_text(self.entry_condition_ref, "contact.entry_condition_ref")
        require_optional_text(self.exit_condition_ref, "contact.exit_condition_ref")

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dictionary.

        Returns:
            A dictionary with the wire contact mode and participants.
        """
        return {
            "mode": self.mode.value,
            "robot_link_ids": list(self.robot_link_ids),
            "entity_ids": list(self.entity_ids),
            "entry_condition_ref": self.entry_condition_ref,
            "exit_condition_ref": self.exit_condition_ref,
        }

    @classmethod
    def from_dict(cls, payload: object) -> ContactDeclaration:
        """Parse one strict JSON contact declaration.

        Args:
            payload: Decoded JSON value.

        Returns:
            The parsed contact declaration.
        """
        item = require_object(payload, "contact")
        require_keys(item, _CONTACT_KEYS, "contact")
        return cls(
            mode=parse_enum(item["mode"], "contact.mode", ContactMode),
            robot_link_ids=require_texts(
                item["robot_link_ids"], "contact.robot_link_ids"
            ),
            entity_ids=require_texts(item["entity_ids"], "contact.entity_ids"),
            entry_condition_ref=require_optional_text(
                item["entry_condition_ref"], "contact.entry_condition_ref"
            ),
            exit_condition_ref=require_optional_text(
                item["exit_condition_ref"], "contact.exit_condition_ref"
            ),
        )


@dataclass(frozen=True, slots=True)
class MotionPermission:
    """What one participating arm may change, and what it must preserve."""

    robot_id: str
    kind: PermissionKind
    writable_joint_names: tuple[str, ...]
    obligations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the arm identity, permission class and joint scope."""
        require_text(self.robot_id, "permission.robot_id")
        require_enum(self.kind, "permission.kind", PermissionKind)
        if not isinstance(self.writable_joint_names, tuple):
            raise ContractViolationError(
                "permission.writable_joint_names must be a tuple"
            )
        require_texts(
            self.writable_joint_names,
            "permission.writable_joint_names",
            allow_empty=True,
        )
        if not isinstance(self.obligations, tuple):
            raise ContractViolationError("permission.obligations must be a tuple")
        require_texts(self.obligations, "permission.obligations", allow_empty=True)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dictionary.

        Returns:
            A dictionary with the wire permission class and joint scope.
        """
        return {
            "robot_id": self.robot_id,
            "kind": self.kind.value,
            "writable_joint_names": list(self.writable_joint_names),
            "obligations": list(self.obligations),
        }

    @classmethod
    def from_dict(cls, payload: object) -> MotionPermission:
        """Parse one strict JSON motion permission.

        Args:
            payload: Decoded JSON value.

        Returns:
            The parsed motion permission.
        """
        item = require_object(payload, "permission")
        require_keys(item, _PERMISSION_KEYS, "permission")
        return cls(
            robot_id=require_text(item["robot_id"], "permission.robot_id"),
            kind=parse_enum(item["kind"], "permission.kind", PermissionKind),
            writable_joint_names=require_texts(
                item["writable_joint_names"],
                "permission.writable_joint_names",
                allow_empty=True,
            ),
            obligations=require_texts(
                item["obligations"], "permission.obligations", allow_empty=True
            ),
        )


@dataclass(frozen=True, slots=True)
class PhaseIntent:
    """One phase's declarative intent, before any coordination or planning.

    Attributes:
        identity: Plan/operation/phase/attempt identity.
        predecessor_phase_ids: Intra-operation predecessors that must finish.
        participants: Exact robot ids that must take part.
        physical_requirements: Physical facts required before the phase runs.
        resources: Task resource leases the phase needs.
        goal: Reference-based goal declaration.
        permissions: Exactly one motion permission per participant.
        contact: Optional declared contact obligation. A ``MAKE`` or
            ``MAINTAIN`` contact requires every participant's permission kind
            to be ``CONTACT_PRESERVING`` or ``EXPLICIT_HOLD``; weaker kinds are
            rejected rather than silently dropping the obligation. Physical
            requirements may not require the same fact both confirmed and
            absent.
    """

    identity: PhaseIdentity
    predecessor_phase_ids: tuple[str, ...]
    participants: tuple[str, ...]
    physical_requirements: tuple[PhysicalRequirement, ...]
    resources: tuple[ResourceRequirement, ...]
    goal: GoalDeclaration
    permissions: tuple[MotionPermission, ...]
    contact: ContactDeclaration | None = None

    def __post_init__(self) -> None:
        """Validate identity, dependencies, participants and completeness."""
        if not isinstance(self.identity, PhaseIdentity):
            raise ContractViolationError("phase.identity must be a PhaseIdentity")
        if not isinstance(self.predecessor_phase_ids, tuple):
            raise ContractViolationError("phase.predecessor_phase_ids must be a tuple")
        require_texts(
            self.predecessor_phase_ids,
            "phase.predecessor_phase_ids",
            allow_empty=True,
        )
        if self.identity.phase_id in self.predecessor_phase_ids:
            raise ContractViolationError(
                "phase.predecessor_phase_ids must not contain the phase itself"
            )
        if not isinstance(self.participants, tuple):
            raise ContractViolationError("phase.participants must be a tuple")
        require_texts(self.participants, "phase.participants")
        if not isinstance(self.physical_requirements, tuple):
            raise ContractViolationError("phase.physical_requirements must be a tuple")
        seen_requirements: set[
            tuple[PhysicalFactKind, EvidencePolarity, tuple[str, ...]]
        ] = set()
        polarities_by_scope: dict[
            tuple[PhysicalFactKind, tuple[str, ...]], set[EvidencePolarity]
        ] = {}
        for requirement in self.physical_requirements:
            if not isinstance(requirement, PhysicalRequirement):
                raise ContractViolationError(
                    "phase.physical_requirements must contain "
                    "PhysicalRequirement values"
                )
            key = (
                requirement.fact_kind,
                requirement.polarity,
                requirement.entity_ids,
            )
            if key in seen_requirements:
                raise ContractViolationError(f"duplicate physical requirement {key!r}")
            seen_requirements.add(key)
            scope = (requirement.fact_kind, requirement.entity_ids)
            polarities = polarities_by_scope.setdefault(scope, set())
            polarities.add(requirement.polarity)
            if {EvidencePolarity.CONFIRMED, EvidencePolarity.ABSENT} <= polarities:
                raise ContractViolationError(
                    f"contradictory physical requirements for {scope!r}: "
                    f"a fact cannot be required both CONFIRMED and ABSENT"
                )
        if not isinstance(self.resources, tuple):
            raise ContractViolationError("phase.resources must be a tuple")
        resource_ids: set[str] = set()
        for resource in self.resources:
            if not isinstance(resource, ResourceRequirement):
                raise ContractViolationError(
                    "phase.resources must contain ResourceRequirement values"
                )
            if resource.resource_id in resource_ids:
                raise ContractViolationError(
                    f"duplicate resource {resource.resource_id!r}"
                )
            resource_ids.add(resource.resource_id)
        if not isinstance(self.goal, GoalDeclaration):
            raise ContractViolationError("phase.goal must be a GoalDeclaration")
        if not isinstance(self.permissions, tuple):
            raise ContractViolationError("phase.permissions must be a tuple")
        permission_robots: set[str] = set()
        for permission in self.permissions:
            if not isinstance(permission, MotionPermission):
                raise ContractViolationError(
                    "phase.permissions must contain MotionPermission values"
                )
            if permission.robot_id in permission_robots:
                raise ContractViolationError(
                    f"duplicate motion permission for {permission.robot_id!r}"
                )
            permission_robots.add(permission.robot_id)
        if permission_robots != set(self.participants):
            missing = sorted(set(self.participants) - permission_robots)
            unknown = sorted(permission_robots - set(self.participants))
            raise ContractViolationError(
                f"phase.permissions must bind every participant exactly once: "
                f"missing {missing}, unknown {unknown}"
            )
        if self.contact is not None and not isinstance(
            self.contact, ContactDeclaration
        ):
            raise ContractViolationError(
                "phase.contact must be a ContactDeclaration or None"
            )
        if self.contact is not None and self.contact.mode in (
            ContactMode.MAKE,
            ContactMode.MAINTAIN,
        ):
            weakening = sorted(
                permission.robot_id
                for permission in self.permissions
                if permission.kind
                not in (
                    PermissionKind.CONTACT_PRESERVING,
                    PermissionKind.EXPLICIT_HOLD,
                )
            )
            if weakening:
                raise ContractViolationError(
                    f"phase.permissions must not weaken the "
                    f"{self.contact.mode.value} contact obligation; "
                    f"use CONTACT_PRESERVING or EXPLICIT_HOLD for {weakening}"
                )

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dictionary.

        Returns:
            A dictionary containing only JSON primitives, lists and objects.
        """
        return {
            "schema": PHASE_INTENT_SCHEMA,
            "identity": self.identity.to_dict(),
            "predecessor_phase_ids": list(self.predecessor_phase_ids),
            "participants": list(self.participants),
            "physical_requirements": [
                requirement.to_dict() for requirement in self.physical_requirements
            ],
            "resources": [resource.to_dict() for resource in self.resources],
            "goal": self.goal.to_dict(),
            "permissions": [permission.to_dict() for permission in self.permissions],
            "contact": None if self.contact is None else self.contact.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: object) -> PhaseIntent:
        """Parse one strict JSON phase intent.

        Args:
            payload: Decoded JSON value.

        Returns:
            The parsed phase intent.
        """
        item = require_object(payload, "phase")
        require_keys(item, _INTENT_KEYS, "phase")
        require_schema(item["schema"], PHASE_INTENT_SCHEMA, "phase")
        contact = item["contact"]
        return cls(
            identity=PhaseIdentity.from_dict(item["identity"]),
            predecessor_phase_ids=require_texts(
                item["predecessor_phase_ids"],
                "phase.predecessor_phase_ids",
                allow_empty=True,
            ),
            participants=require_texts(item["participants"], "phase.participants"),
            physical_requirements=tuple(
                PhysicalRequirement.from_dict(entry)
                for entry in require_array(
                    item["physical_requirements"],
                    "phase.physical_requirements",
                )
            ),
            resources=tuple(
                ResourceRequirement.from_dict(entry)
                for entry in require_array(item["resources"], "phase.resources")
            ),
            goal=GoalDeclaration.from_dict(item["goal"]),
            permissions=tuple(
                MotionPermission.from_dict(entry)
                for entry in require_array(item["permissions"], "phase.permissions")
            ),
            contact=(
                None if contact is None else ContactDeclaration.from_dict(contact)
            ),
        )
