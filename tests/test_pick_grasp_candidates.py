"""Legacy geometry and plan-only facade contracts; no physical execution."""

import asyncio
from types import SimpleNamespace

import pytest
from test_pick_handler import document
from test_place_down_handler import context

from rocobrick.coordinated_session import (
    CoordinatedSessionResult,
    FreeSpaceMotionRequest,
)
from rocobrick.execution.operation_state_ledger import OperationStateLedger
from rocobrick.skills.observation import parse_pick_observation
from rocobrick.skills.placement_geometry import place_up_pick_candidates
from rocobrick.unified_runtime_facade import UnifiedDualArmRuntimeFacade


def candidates():
    """Use identified measured geometry, not Task C brick IDs or locations.

    Returns:
        The finite side/oblique family under an identity robot base.
    """
    shared = document()
    sample = parse_pick_observation(
        shared,
        robot_id="piper_0",
        physical_instance_id="brick-a",
        session_id="session-1",
        manifest_sha256="a" * 64,
    )
    ctx = context()
    return place_up_pick_candidates(sample, ctx.base, ctx.provenance, ctx.params)


def test_facade_planning_uses_only_plan_only_lane_and_does_not_confirm_custody():
    """One measured world is shared with execution, but no ledger effect occurs."""
    calls = []

    def plan(request):
        calls.append(request)
        return CoordinatedSessionResult(
            request.coordination_id,
            request.robots,
            {},
            {"points": [1]},
            {
                "plan_only": True,
                "physical_execution": False,
                "prepared": False,
                "started": False,
                "terminal": True,
            },
            True,
            participants=request.participants,
        )

    def execute(request):
        pytest.fail("plan_free_space dispatched physical movement")

    shared = document()
    shared["scene_digest"] = "c" * 64
    ledger = OperationStateLedger(("piper_0", "piper_1"))
    facade = UnifiedDualArmRuntimeFacade(
        SimpleNamespace(
            execute_successor_pre_phase=plan, execute_coordinated_pre_phase=execute
        ),
        state_ledger=ledger,
    )
    request = FreeSpaceMotionRequest(
        "candidate",
        ("piper_0", "piper_1"),
        ("piper_0",),
        shared,
        1,
        {"piper_0": {"tcp_pose": {}}},
        {"piper_0_joint1": 0.0},
        15.0,
    )
    result = asyncio.run(facade.plan_free_space(request))
    assert result.passed and calls[0].plan_only is True
    assert calls[0].physical_facts == {"piper_0": {}}
    assert ledger.owner_of_object("brick-a") is None
