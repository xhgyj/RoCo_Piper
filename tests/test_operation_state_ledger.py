from __future__ import annotations

import pytest

from rocobrick.execution.operation_state_ledger import OperationStateLedger


def test_ledger_is_robot_isolated_and_evidence_driven() -> None:
    ledger = OperationStateLedger(("piper_0", "piper_1"))
    state = ledger.confirm("piper_0", "holding:brick-1", {"holding_confirmed": True})
    assert state.robot_id == "piper_0"
    assert ledger.observe("piper_1", "holding:brick-1") is None
    assert ledger.observe("piper_0", "holding:brick-1") == state


def test_ledger_rejects_non_json_evidence_and_unknown_robot() -> None:
    ledger = OperationStateLedger(("piper_0",))
    with pytest.raises(ValueError):
        ledger.confirm("piper_0", "robot_at_pick_region", {"value": object()})
    with pytest.raises(ValueError):
        ledger.observe("piper_1", "robot_at_pick_region")


def test_ledger_revoke_removes_only_confirmed_fact() -> None:
    ledger = OperationStateLedger(("piper_0",))
    ledger.confirm("piper_0", "assembled:brick-1", {"connection_confirmed": True})
    assert ledger.revoke("piper_0", "assembled:brick-1", "scene changed") is True
    assert ledger.observe("piper_0", "assembled:brick-1") is None
    assert ledger.revoke("piper_0", "assembled:brick-1", "already absent") is False


def test_holding_owner_is_unique_across_robots() -> None:
    ledger = OperationStateLedger(("piper_0", "piper_1"))
    original = ledger.confirm_holding("piper_0", "brick-1", {"holding_confirmed": True})
    assert ledger.owner_of_object("brick-1") == "piper_0"
    with pytest.raises(ValueError, match="already owned"):
        ledger.confirm("piper_1", "holding:brick-1", {"holding_confirmed": True})
    assert ledger.observe("piper_0", "holding:brick-1") == original
    assert ledger.observe("piper_1", "holding:brick-1") is None
    renewed = ledger.confirm_holding(
        "piper_0", "brick-1", {"holding_confirmed": True, "sample": 2}
    )
    assert ledger.observe("piper_0", "holding:brick-1") == renewed


def test_stale_holding_keeps_ownership_until_explicit_revoke() -> None:
    ledger = OperationStateLedger(("piper_0", "piper_1"))
    ledger.confirm_holding("piper_0", "brick-1", {"holding_confirmed": True})
    assert ledger.invalidate_if_stale("piper_0", "brick-1", "scene stale")
    assert ledger.observe("piper_0", "holding:brick-1") is None
    assert "holding:brick-1" not in ledger.snapshot("piper_0")
    assert ledger.owner_of_object("brick-1") == "piper_0"
    with pytest.raises(ValueError, match="already owned"):
        ledger.confirm_holding("piper_1", "brick-1", {})
    with pytest.raises(ValueError, match="uncertain"):
        ledger.revoke("piper_0", "holding:brick-1", "release confirmed")
    assert ledger.owner_of_object("brick-1") == "piper_0"
    with pytest.raises(ValueError, match="uncertain"):
        ledger.confirm_holding("piper_0", "brick-1", {"holding_confirmed": True})


def test_holding_rejects_empty_instance_id() -> None:
    ledger = OperationStateLedger(("piper_0",))
    with pytest.raises(ValueError):
        ledger.confirm("piper_0", "holding:", {})
    with pytest.raises(ValueError):
        ledger.owner_of_object("")


def test_pre_phase_reconciliation_preserves_holding_and_support():
    """Arrival refreshes physical evidence without freeing the object owner."""
    from pre_phase_helpers import request, success

    ledger = OperationStateLedger(("piper_0",))
    facts = {"holding:brick": {"sample": 2}, "supporting:upper": {"sample": 2}}
    for key in facts:
        ledger.confirm("piper_0", key, {"sample": 1})
    item = request()
    ledger.reconcile_pre_phase(item, success(item, {"piper_0": facts}))
    assert ledger.owner_of_object("brick") == "piper_0"
    assert ledger.observe("piper_0", "supporting:upper").evidence == {"sample": 2}


def test_pre_phase_reconciliation_is_atomic_on_lost_physical_fact():
    """Invalid final facts leave all previous facts and ownership intact."""
    from pre_phase_helpers import request, success

    ledger = OperationStateLedger(("piper_0",))
    ledger.confirm("piper_0", "holding:brick", {"sample": 1})
    ledger.confirm("piper_0", "supporting:upper", {"sample": 1})
    previous = ledger.snapshot("piper_0")
    item = request()
    with pytest.raises(ValueError, match="continuity"):
        ledger.reconcile_pre_phase(
            item, success(item, {"piper_0": {"holding:brick": {"sample": 2}}})
        )
    assert ledger.snapshot("piper_0") == previous
    assert ledger.owner_of_object("brick") == "piper_0"


def test_terminal_policy_can_end_support_but_cannot_create_holding():
    """Only an explicitly authorized support departure may remove its fact."""
    from pre_phase_helpers import request, success

    from rocobrick.unified_dual_arm_runtime import TerminalPrePhasePolicy

    policy = TerminalPrePhasePolicy(
        "cleanup",
        "c" * 64,
        ("piper_0",),
        {"piper_0": {"target": "explicit"}},
        ("piper_0",),
    )
    item = request(policy=policy)
    ledger = OperationStateLedger(("piper_0",))
    ledger.confirm("piper_0", "supporting:upper", {"confirmed": True})
    with pytest.raises(ValueError, match="new physical"):
        ledger.reconcile_pre_phase(
            item, success(item, {"piper_0": {"holding:brick": {"confirmed": True}}})
        )
    assert ledger.observe("piper_0", "supporting:upper") is not None
    ledger.reconcile_pre_phase(item, success(item))
    assert ledger.observe("piper_0", "supporting:upper") is None


def test_unknown_support_cannot_be_recovered_by_free_space_motion():
    """An invalidated support fact remains protected until dedicated recovery."""
    from pre_phase_helpers import request, success

    ledger = OperationStateLedger(("piper_0",))
    ledger.confirm("piper_0", "supporting:upper", {"confirmed": True})
    ledger.invalidate_support("piper_0", "upper", "cancel unsafe")
    item = request()
    with pytest.raises(ValueError, match="uncertain"):
        ledger.reconcile_pre_phase(item, success(item))
