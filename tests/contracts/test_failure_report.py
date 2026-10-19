"""Failures use a closed taxonomy and never assert physical success."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest
from contract_doubles import identity

from rocobrick.contracts import (
    ContractViolationError,
    FailureCategory,
    FailureReport,
)


def _report(**overrides: object) -> FailureReport:
    """Build a valid failure report with optional field overrides.

    Args:
        **overrides: Field values replacing the valid defaults.

    Returns:
        The validated failure report.
    """
    values: dict[str, object] = {
        "identity": identity(),
        "category": FailureCategory.GOAL_IK_OR_CONSTRAINT,
        "reason": "goal unreachable under declared constraints",
    }
    values.update(overrides)
    return FailureReport(**values)  # type: ignore[arg-type]


def test_taxonomy_is_closed_and_has_no_success_member() -> None:
    """Six proven categories plus UNKNOWN exist; none of them means success."""
    assert {category.value for category in FailureCategory} == {
        "GOAL_IK_OR_CONSTRAINT",
        "GOAL_COLLISION",
        "PATH_SEARCH_EXHAUSTED",
        "POST_VALIDATION",
        "EXECUTION",
        "PHYSICAL_VERIFICATION",
        "UNKNOWN",
    }
    assert "SUCCESS" not in FailureCategory.__members__
    assert not hasattr(FailureCategory, "SUCCESS")


def test_roundtrip_and_immutability() -> None:
    """Reports serialize losslessly and cannot be edited afterwards."""
    original = _report(
        category=FailureCategory.EXECUTION,
        sent_motion=True,
        stopped=False,
        diagnostic_refs=("diag-1",),
        physical_evidence_ids=("ev-1",),
        terminal_evidence_ids=("terminal-1",),
        retained_lease_ids=("robot:piper_0",),
        allowed_recovery_actions=("stop_and_verify",),
    )
    parsed = FailureReport.from_dict(json.loads(json.dumps(original.to_dict())))
    assert parsed == original
    assert parsed.terminal_evidence_ids == ("terminal-1",)
    assert parsed.reported_no_unproven_motion is False
    with pytest.raises(FrozenInstanceError):
        parsed.stopped = True  # type: ignore[misc]


def test_search_exhaustion_is_not_infeasibility() -> None:
    """Exhaustion is only reportable as path-search exhaustion."""
    with pytest.raises(ContractViolationError, match="search_exhausted"):
        _report(search_exhausted=True)
    exhausted = _report(
        category=FailureCategory.PATH_SEARCH_EXHAUSTED,
        search_exhausted=True,
        reason="sampling budget exhausted; infeasibility not proven",
    )
    assert exhausted.search_exhausted is True
    assert exhausted.reported_no_unproven_motion is True


def test_path_search_exhausted_requires_the_flag() -> None:
    """PATH_SEARCH_EXHAUSTED and its exhaustion flag cannot disagree."""
    with pytest.raises(ContractViolationError, match="requires search_exhausted=True"):
        _report(category=FailureCategory.PATH_SEARCH_EXHAUSTED)


def test_unknown_category_preserves_an_unestablished_cause() -> None:
    """UNKNOWN records an unproven cause instead of forcing a wrong one."""
    report = _report(
        category=FailureCategory.UNKNOWN,
        reason="planner returned without a diagnosable cause",
        sent_motion=True,
        stopped=True,
    )
    assert report.category is FailureCategory.UNKNOWN
    with pytest.raises(ContractViolationError, match="search_exhausted"):
        _report(
            category=FailureCategory.UNKNOWN,
            reason="cause unknown",
            search_exhausted=True,
        )


def test_pre_send_categories_cannot_claim_sent_motion() -> None:
    """Planning and validation failures are decided before sending."""
    for category in (
        FailureCategory.GOAL_IK_OR_CONSTRAINT,
        FailureCategory.GOAL_COLLISION,
        FailureCategory.PATH_SEARCH_EXHAUSTED,
        FailureCategory.POST_VALIDATION,
    ):
        with pytest.raises(ContractViolationError, match="before sending"):
            _report(category=category, sent_motion=True)


def test_stop_requires_sent_motion() -> None:
    """A stop can only confirm termination of motion that was sent."""
    with pytest.raises(ContractViolationError, match="requires"):
        _report(stopped=True)


def test_terminal_evidence_requires_sent_motion() -> None:
    """Controller terminal evidence presupposes motion that was sent."""
    with pytest.raises(ContractViolationError, match="terminal_evidence_ids"):
        _report(category=FailureCategory.EXECUTION, terminal_evidence_ids=("t-1",))
    report = _report(
        category=FailureCategory.EXECUTION,
        sent_motion=True,
        terminal_evidence_ids=("t-1",),
    )
    assert report.terminal_evidence_ids == ("t-1",)
    assert report.reported_no_unproven_motion is False


def test_physical_verification_failure_is_not_success() -> None:
    """A physical-verification failure carries no success claim at all."""
    report = _report(
        category=FailureCategory.PHYSICAL_VERIFICATION,
        reason="holding fact not observed",
        physical_evidence_ids=(),
    )
    payload = report.to_dict()
    assert "physical_success" not in payload
    assert "success" not in payload
    assert report.reported_no_unproven_motion is True
    assert report.retained_lease_ids == ()


def test_strict_parse_rejects_payload_drift() -> None:
    """Unknown categories and unexpected keys fail closed."""
    payload = _report().to_dict()
    with pytest.raises(ContractViolationError, match="keys mismatch"):
        FailureReport.from_dict({**payload, "extra": 1})
    with pytest.raises(ContractViolationError, match="category"):
        FailureReport.from_dict({**payload, "category": "MAGIC"})
