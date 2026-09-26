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
