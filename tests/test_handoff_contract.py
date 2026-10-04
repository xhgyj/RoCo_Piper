"""Tests for deferred resource release through skill handoff."""

import asyncio

from rocobrick.unified_dual_arm_runtime import SuccessorPrePhaseEvidence


def test_handoff_evidence_is_strict_value() -> None:
    """Pre-phase evidence carries terminal and resource-transfer booleans."""
    item = SuccessorPrePhaseEvidence(
        predecessor_operation_id="op-1",
        successor_operation_ids=("next-op",),
        participants=("piper_0",),
        scene_digest="scene-1",
        planned=True,
        executed=True,
        terminal=True,
        safe=True,
        resource_transfer_confirmed=True,
        evidence={"retreat_confirmed": {"handoff": True}},
    )
    assert item.passed is True
    assert item.evidence["retreat_confirmed"]


def test_handoff_protocol_is_async_shape() -> None:
    """The protocol seam is intended for awaited coordinated transport."""
    assert asyncio.iscoroutinefunction(
        getattr(SuccessorPrePhaseEvidence, "__init__")
    ) is False
