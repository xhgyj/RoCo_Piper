from __future__ import annotations

import asyncio

from rocobrick.multi_robot_runtime import DryRunMultiRobotSession
from rocobrick.robot_operation_plan import parse_robot_operation_plan
from rocobrick.robot_operation_runner import OperationPlanRunner


def _plan():
    def op(operation_id: str, robot_id: str):
        return {
            "operation_id": operation_id,
            "source_task_id": "task-1",
            "skill": "WAIT",
            "executor_robot_id": robot_id,
            "participants": [robot_id],
            "role_bindings": {"executor": robot_id},
            "target": None,
            "parameters": {},
            "preconditions": [],
            "effects": [],
            "resources": [f"robot:{robot_id}"],
            "evidence_requirements": ["confirmed"],
        }

    return parse_robot_operation_plan({
        "schema": "robot-operation-plan/v1",
        "plan_id": "dry-run",
        "source_plan_id": "assembly",
        "robots": ["piper_0", "piper_1"],
        "operations": [op("a", "piper_0"), op("b", "piper_1")],
        "dependencies": [],
        "execution_layers": [["a", "b"]],
        "planned_schedule": [[
            {"operation_id": "a", "robot_id": "piper_0"},
            {"operation_id": "b", "robot_id": "piper_1"},
        ]],
        "evidence_requirements": [
            {"operation_id": "a", "evidence": "confirmed"},
            {"operation_id": "b", "evidence": "confirmed"},
        ],
    })


def test_dry_run_session_exercises_two_robot_executor() -> None:
    session = DryRunMultiRobotSession(("piper_0", "piper_1"))
    readiness = session.start()
    result = asyncio.run(OperationPlanRunner(_plan(), session, session).run())
    cleanup = session.stop()

    assert readiness.passed
    assert result.passed
    assert {item["robot_id"] for item in session.events if item["event"] == "operation_confirmed"} == {
        "piper_0", "piper_1"
    }
    assert cleanup["motion_sent"] is False
