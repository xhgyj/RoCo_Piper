"""Tests for the unified successor pre-phase facade."""

import asyncio
from types import SimpleNamespace

from rocobrick.unified_dual_arm_runtime import SuccessorPrePhaseRequest
from rocobrick.unified_runtime_facade import UnifiedDualArmRuntimeFacade


def request(*, terminal_cleanup=False, parameters=None):
    operation = SimpleNamespace(
        operation_id="next-op",
        executor_robot_id="piper_1",
        participants=("piper_1", "piper_0"),
        parameters=parameters or {},
    )
    return SuccessorPrePhaseRequest(
        predecessor_operation_id="previous-op",
        successor_operation_ids=() if terminal_cleanup else ("next-op",),
        predecessor_state={"endpoint": "release"},
        successor_operations=() if terminal_cleanup else (operation,),
        participants=("piper_1", "piper_0"),
        scene_digest="scene-1",
        resources=("robot:piper_1", "robot:piper_0"),
        terminal_cleanup=terminal_cleanup,
    )


def test_facade_rejects_missing_geometry_targets():
    result = asyncio.run(
        UnifiedDualArmRuntimeFacade(object()).execute_pre_phase(request())
    )
    assert result.passed is False
    assert result.executed is False
    assert result.safe is False
    assert "pre_phase_targets" in (result.reason or "")


def test_facade_does_not_promote_plan_only_to_execution():
    target = {
        "piper_1": {
            "tcp_pose": {
                "position": (0.1, 0.2, 0.3),
                "orientation_xyzw": (0.0, 0.0, 0.0, 1.0),
                "tip_link": "piper_1_rocobrick_tcp",
                "frame_id": "piper_1_base_link",
            }
        },
        "piper_0": {
            "tcp_pose": {
                "position": (0.1, 0.2, 0.3),
                "orientation_xyzw": (0.0, 0.0, 0.0, 1.0),
                "tip_link": "piper_0_rocobrick_tcp",
                "frame_id": "piper_0_base_link",
            }
        },
    }
    result = asyncio.run(
        UnifiedDualArmRuntimeFacade(object()).execute_pre_phase(
            request(parameters={"pre_phase_targets": target})
        )
    )
    assert result.passed is False
    assert result.executed is False
    assert result.terminal is False


def test_terminal_cleanup_is_fail_closed_until_runtime_exists():
    result = asyncio.run(
        UnifiedDualArmRuntimeFacade(object()).execute_pre_phase(
            request(terminal_cleanup=True)
        )
    )
    assert result.passed is False
    assert "cleanup" in (result.reason or "")
