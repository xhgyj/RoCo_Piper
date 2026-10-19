"""Structured failure reporting with an explicit, closed failure taxonomy.

Categories separate goal infeasibility, goal collision, path-search exhaustion,
post-validation rejection, execution failure and physical verification failure,
plus ``UNKNOWN`` for causes that are not established. A report never asserts
success: there is no success category and no physical result field. Planning or
search exhaustion is reported as exhaustion, not as proven infeasibility, and no
category can stand in for physical evidence.

Controller terminal evidence (``terminal_evidence_ids``) is recorded separately
from physical evidence: reaching a terminal controller state says nothing about
the physical world. Conversely, a failure report never releases resources --
retained leases and reservations stay retained, and no helper here releases,
reallocates or re-uses them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from rocobrick.contracts._validation import (
    ContractViolationError,
    parse_enum,
    require_bool,
    require_keys,
    require_object,
    require_schema,
    require_text,
    require_texts,
)
from rocobrick.contracts.identity import PhaseIdentity

FAILURE_REPORT_SCHEMA = "failure-report/v1"

_REPORT_KEYS = frozenset(
    {
        "schema",
        "identity",
        "category",
        "reason",
        "sent_motion",
        "stopped",
        "search_exhausted",
        "diagnostic_refs",
        "physical_evidence_ids",
        "terminal_evidence_ids",
        "retained_lease_ids",
        "allowed_recovery_actions",
    }
)

_PRE_SEND_CATEGORIES = frozenset(
    {
        "GOAL_IK_OR_CONSTRAINT",
        "GOAL_COLLISION",
        "PATH_SEARCH_EXHAUSTED",
        "POST_VALIDATION",
    }
)


class FailureCategory(StrEnum):
    """Closed failure taxonomy; there is deliberately no success member."""

    GOAL_IK_OR_CONSTRAINT = "GOAL_IK_OR_CONSTRAINT"
    GOAL_COLLISION = "GOAL_COLLISION"
    PATH_SEARCH_EXHAUSTED = "PATH_SEARCH_EXHAUSTED"
    POST_VALIDATION = "POST_VALIDATION"
    EXECUTION = "EXECUTION"
    PHYSICAL_VERIFICATION = "PHYSICAL_VERIFICATION"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class FailureReport:
    """Auditable description of why a phase did not complete.

    Attributes:
        identity: Phase identity the failure belongs to.
        category: Closed failure category; ``UNKNOWN`` when not established.
        reason: Human-readable, non-empty explanation.
        sent_motion: Whether any motion was sent for this phase.
        stopped: Whether a stop is confirmed after sending motion.
        search_exhausted: Whether a budget, not infeasibility, ended the search.
        diagnostic_refs: Reference identities for candidate/constraint data.
        physical_evidence_ids: Physical evidence collected before failing.
        terminal_evidence_ids: Controller terminal-state evidence identities.
        retained_lease_ids: Leases still held because release is unproven.
        allowed_recovery_actions: Recovery actions the upper layer may take.
    """

    identity: PhaseIdentity
    category: FailureCategory
    reason: str
    sent_motion: bool = False
    stopped: bool = False
    search_exhausted: bool = False
    diagnostic_refs: tuple[str, ...] = ()
    physical_evidence_ids: tuple[str, ...] = ()
    terminal_evidence_ids: tuple[str, ...] = ()
    retained_lease_ids: tuple[str, ...] = ()
    allowed_recovery_actions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate category, flags and their mutual consistency."""
        if not isinstance(self.identity, PhaseIdentity):
            raise ContractViolationError("failure.identity must be a PhaseIdentity")
        if not isinstance(self.category, FailureCategory):
            raise ContractViolationError("failure.category must be a FailureCategory")
        require_text(self.reason, "failure.reason")
        require_bool(self.sent_motion, "failure.sent_motion")
        require_bool(self.stopped, "failure.stopped")
        require_bool(self.search_exhausted, "failure.search_exhausted")
        if self.search_exhausted and (
            self.category is not FailureCategory.PATH_SEARCH_EXHAUSTED
        ):
            raise ContractViolationError(
                "failure.search_exhausted is only valid for PATH_SEARCH_EXHAUSTED"
            )
        if self.category.value in _PRE_SEND_CATEGORIES and self.sent_motion:
            raise ContractViolationError(
                f"failure.category {self.category.value} is decided before "
                f"sending motion"
            )
        if self.stopped and not self.sent_motion:
            raise ContractViolationError("failure.stopped requires failure.sent_motion")
        if self.terminal_evidence_ids and not self.sent_motion:
            raise ContractViolationError(
                "failure.terminal_evidence_ids requires failure.sent_motion"
            )
        if (
            self.category is FailureCategory.PATH_SEARCH_EXHAUSTED
            and not self.search_exhausted
        ):
            raise ContractViolationError(
                "failure.category PATH_SEARCH_EXHAUSTED requires search_exhausted=True"
            )
        for field_name in (
            "diagnostic_refs",
            "physical_evidence_ids",
            "terminal_evidence_ids",
            "retained_lease_ids",
            "allowed_recovery_actions",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, tuple):
                raise ContractViolationError(f"failure.{field_name} must be a tuple")
            require_texts(value, f"failure.{field_name}", allow_empty=True)

    @property
    def reported_no_unproven_motion(self) -> bool:
        """Return whether the report *claims* no unproven motion remains.

        This is a reporting summary only. It is not release authority: leases,
        reservations and ownership epochs stay retained, and nothing may be
        released, reallocated or re-used because this is true. It is also not a
        safety or physical-success statement -- holding/supporting facts and
        in-flight peers still require independent physical evidence.

        Returns:
            True when no motion was sent, or a stop is reported.
        """
        return not self.sent_motion or self.stopped

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dictionary.

        Returns:
            A dictionary containing only JSON primitives, lists and objects.
        """
        return {
            "schema": FAILURE_REPORT_SCHEMA,
            "identity": self.identity.to_dict(),
            "category": self.category.value,
            "reason": self.reason,
            "sent_motion": self.sent_motion,
            "stopped": self.stopped,
            "search_exhausted": self.search_exhausted,
            "diagnostic_refs": list(self.diagnostic_refs),
            "physical_evidence_ids": list(self.physical_evidence_ids),
            "terminal_evidence_ids": list(self.terminal_evidence_ids),
            "retained_lease_ids": list(self.retained_lease_ids),
            "allowed_recovery_actions": list(self.allowed_recovery_actions),
        }

    @classmethod
    def from_dict(cls, payload: object) -> FailureReport:
        """Parse one strict JSON failure report.

        Args:
            payload: Decoded JSON value.

        Returns:
            The parsed failure report.
        """
        item = require_object(payload, "failure")
        require_keys(item, _REPORT_KEYS, "failure")
        require_schema(item["schema"], FAILURE_REPORT_SCHEMA, "failure")
        return cls(
            identity=PhaseIdentity.from_dict(item["identity"]),
            category=parse_enum(item["category"], "failure.category", FailureCategory),
            reason=require_text(item["reason"], "failure.reason"),
            sent_motion=require_bool(item["sent_motion"], "failure.sent_motion"),
            stopped=require_bool(item["stopped"], "failure.stopped"),
            search_exhausted=require_bool(
                item["search_exhausted"], "failure.search_exhausted"
            ),
            diagnostic_refs=require_texts(
                item["diagnostic_refs"],
                "failure.diagnostic_refs",
                allow_empty=True,
            ),
            physical_evidence_ids=require_texts(
                item["physical_evidence_ids"],
                "failure.physical_evidence_ids",
                allow_empty=True,
            ),
            terminal_evidence_ids=require_texts(
                item["terminal_evidence_ids"],
                "failure.terminal_evidence_ids",
                allow_empty=True,
            ),
            retained_lease_ids=require_texts(
                item["retained_lease_ids"],
                "failure.retained_lease_ids",
                allow_empty=True,
            ),
            allowed_recovery_actions=require_texts(
                item["allowed_recovery_actions"],
                "failure.allowed_recovery_actions",
                allow_empty=True,
            ),
        )
