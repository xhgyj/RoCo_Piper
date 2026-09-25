"""Multi-robot runtime entry point (A6).

This module is the import seam the plan entry point (``piper_run``) and the
skill/observation seam (``robot_operation_executor``) use to reach the ONE
real multi-robot live session: :class:`MultiRobotIsaacMoveItSession` in
:mod:`rocobrick.runtime_session` (ONE Isaac host bridge plus ONE ROS/MoveIt
container running the A3 dual launch for every plan robot, with the A5
multi-robot session envelope for readiness and robot-scoped read-only
observations). They are re-exported here so no caller needs to know where the
session lives.

:class:`DryRunMultiRobotSession` is HISTORICAL: it is a no-motion in-memory
helper kept only for its own legacy unit test. No production module may
import it (``tests/test_multi_robot_live_session.py`` pins that) and it must
never be used as the real session or as acceptance evidence.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from rocobrick.robot_operation_plan import RobotOperation
from rocobrick.runtime_session import (
    MultiRobotIsaacMoveItSession,
    MultiRobotObservationEvidence,
    MultiRobotReadinessEvidence,
    MultiRobotSessionConfig,
    MultiRobotSessionError,
    MultiRobotSessionEvidence,
    ProcessHandle,
    ProcessRunner,
    RunOutcome,
    RuntimeSessionError,
    RuntimeSessionStatus,
    SubprocessRunner,
    build_global_scene_document,
)

__all__ = [
    "DryRunMultiRobotSession",
    "DryRunReadinessEvidence",
    "MultiRobotIsaacMoveItSession",
    "MultiRobotObservationEvidence",
    "MultiRobotReadinessEvidence",
    "MultiRobotSessionConfig",
    "MultiRobotSessionError",
    "MultiRobotSessionEvidence",
    "ProcessHandle",
    "ProcessRunner",
    "RunOutcome",
    "RuntimeSessionError",
    "RuntimeSessionStatus",
    "SubprocessRunner",
    "build_global_scene_document",
]


@dataclass(frozen=True)
class DryRunReadinessEvidence:
    """HISTORICAL (test-only) readiness evidence of the dry-run helper."""

    robot_ids: tuple[str, ...]
    passed: bool
    mode: str = "dry_run"

    def serialize(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "robots": list(self.robot_ids),
            "passed": self.passed,
        }


class DryRunMultiRobotSession:
    """HISTORICAL no-motion helper used only to exercise the scheduler seam.

    It is not the real session, it starts nothing, observes nothing and is not
    allowed anywhere in the production path (see the module docstring); the
    real session is :class:`MultiRobotIsaacMoveItSession`.
    """

    def __init__(self, robot_ids: tuple[str, ...]) -> None:
        if not robot_ids or len(robot_ids) != len(set(robot_ids)):
            raise ValueError("robot_ids must be non-empty and unique")
        self.robot_ids = robot_ids
        self.events: list[dict[str, object]] = []
        self.started = False
        self.stopped = False

    def start(self) -> DryRunReadinessEvidence:
        if self.started:
            raise RuntimeError("dry-run session already started")
        self.started = True
        evidence = DryRunReadinessEvidence(self.robot_ids, True)
        self.events.append({"event": "readiness", **evidence.serialize()})
        return evidence

    async def execute(self, operation: RobotOperation) -> dict[str, object]:
        if not self.started or self.stopped:
            raise RuntimeError("dry-run session is not active")
        self.events.append({
            "event": "operation_start",
            "operation_id": operation.operation_id,
            "robot_id": operation.executor_robot_id,
            "skill": operation.skill,
        })
        await asyncio.sleep(0)
        result = {
            "passed": True,
            "mode": "dry_run",
            "operation_id": operation.operation_id,
            "robot_id": operation.executor_robot_id,
            "skill": operation.skill,
            "motion_sent": False,
            "evidence": ["scheduler_dispatch_confirmed"],
        }
        self.events.append({"event": "operation_confirmed", **result})
        return result

    async def cancel(self, operation_id: str) -> dict[str, object]:
        self.events.append({
            "event": "operation_cancel",
            "operation_id": operation_id,
            "mode": "dry_run",
            "motion_sent": False,
        })
        await asyncio.sleep(0)
        return {"passed": True, "operation_id": operation_id, "mode": "dry_run"}

    def stop(self) -> dict[str, object]:
        self.stopped = True
        evidence = {
            "mode": "dry_run",
            "robots": list(self.robot_ids),
            "motion_sent": False,
            "cleanup_passed": True,
        }
        self.events.append({"event": "cleanup", **evidence})
        return evidence
