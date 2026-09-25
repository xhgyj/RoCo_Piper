from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from rocobrick.robot_operation_plan import RobotOperation


@dataclass(frozen=True)
class DryRunReadinessEvidence:
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
    """No-motion session used to exercise the production scheduler seam."""

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
