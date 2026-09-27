from __future__ import annotations

import pytest

from rocobrick.operation_state_ledger import OperationStateLedger


def test_ledger_is_robot_isolated_and_evidence_driven() -> None:
    ledger = OperationStateLedger(("piper_0", "piper_1"))
    state = ledger.confirm(
        "piper_0", "holding:brick-1", {"holding_confirmed": True}
    )
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
    original = ledger.confirm_holding(
        "piper_0", "brick-1", {"holding_confirmed": True}
    )
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
    assert ledger.revoke("piper_0", "holding:brick-1", "release confirmed") is False
    assert ledger.owner_of_object("brick-1") is None
    ledger.confirm_holding("piper_1", "brick-1", {"holding_confirmed": True})
    assert ledger.owner_of_object("brick-1") == "piper_1"


def test_holding_rejects_empty_instance_id() -> None:
    ledger = OperationStateLedger(("piper_0",))
    with pytest.raises(ValueError):
        ledger.confirm("piper_0", "holding:", {})
    with pytest.raises(ValueError):
        ledger.owner_of_object("")
