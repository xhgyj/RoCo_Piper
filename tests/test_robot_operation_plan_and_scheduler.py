from __future__ import annotations

import asyncio

import pytest

from rocobrick.multi_robot_scheduler import OperationStatus, schedule
from rocobrick.robot_operation_plan import (
    RobotOperationPlanError,
    parse_robot_operation_plan,
)


def operation(operation_id: str, robot: str, skill: str = "WAIT", resource: str | None = None) -> dict:
    target = None
    resources = [f"robot:{robot}"]
    if skill == "PICK":
        target = {"physical_instance_id": "brick-1"}
        resources.append("part:brick-1")
    elif skill == "PLACE_DOWN":
        target = {
            "connection_id": "conn-1",
            "upper_physical_instance_id": "brick-1",
            "lower_physical_instance_id": "brick-0",
        }
        resources += ["connection:conn-1", "part:brick-1", "part:brick-0"]
    elif skill == "TRANSIT":
        target = {
            "from": {"kind": "robot_current", "physical_instance_id": None, "connection_id": None, "robot_id": None},
            "to": {"kind": "part", "physical_instance_id": "brick-1", "connection_id": None, "robot_id": None},
            "carrying_physical_instance_id": None,
        }
    if resource:
        resources.append(resource)
    return {
        "operation_id": operation_id,
        "source_task_id": "task-" + operation_id,
        "skill": skill,
        "executor_robot_id": robot,
        "participants": [robot],
        "role_bindings": {"executor": robot},
        "target": target,
        "parameters": {},
        "preconditions": [],
        "effects": [],
        "resources": resources,
        "evidence_requirements": [],
    }


def plan(ops: list[dict], layers: list[list[str]], schedule_slots: list[list[dict]], deps: list[dict] | None = None) -> dict:
    return {
        "schema": "robot-operation-plan/v1",
        "plan_id": "run-1",
        "source_plan_id": "assembly-1",
        "robots": ["piper_0", "piper_1"],
        "operations": ops,
        "dependencies": deps or [],
        "execution_layers": layers,
        "planned_schedule": schedule_slots,
        "evidence_requirements": [],
    }


def test_parses_all_four_skills() -> None:
    ops = [
        operation("pick", "piper_0", "PICK"),
        operation("transit", "piper_1", "TRANSIT"),
        operation("wait", "piper_0", "WAIT"),
        operation("place", "piper_1", "PLACE_DOWN"),
    ]
    parsed = parse_robot_operation_plan(
        plan(ops, [["pick", "transit"], ["wait", "place"]], [
            [{"operation_id": "pick", "robot_id": "piper_0"}, {"operation_id": "transit", "robot_id": "piper_1"}],
            [{"operation_id": "wait", "robot_id": "piper_0"}, {"operation_id": "place", "robot_id": "piper_1"}],
        ])
    )
    assert set(parsed.operations) == {"pick", "transit", "wait", "place"}


def test_rejects_unknown_operation_key() -> None:
    raw = plan([operation("wait", "piper_0")], [["wait"]], [[{"operation_id": "wait", "robot_id": "piper_0"}]])
    raw["operations"][0]["unexpected"] = True
    with pytest.raises(RobotOperationPlanError):
        parse_robot_operation_plan(raw)


def test_rejects_same_layer_resource_conflict() -> None:
    raw = plan(
        [operation("a", "piper_0", "WAIT", "part:shared"), operation("b", "piper_1", "WAIT", "part:shared")],
        [["a", "b"]],
        [[{"operation_id": "a", "robot_id": "piper_0"}, {"operation_id": "b", "robot_id": "piper_1"}]],
    )
    with pytest.raises(RobotOperationPlanError):
        parse_robot_operation_plan(raw)


def test_rejects_dependency_not_in_later_layer() -> None:
    raw = plan(
        [operation("a", "piper_0"), operation("b", "piper_1")],
        [["a", "b"]],
        [[{"operation_id": "a", "robot_id": "piper_0"}, {"operation_id": "b", "robot_id": "piper_1"}]],
        [{"predecessor_operation_id": "a", "successor_operation_id": "b", "reason": "required"}],
    )
    with pytest.raises(RobotOperationPlanError):
        parse_robot_operation_plan(raw)


class Executor:
    def __init__(self, fail: str | None = None):
        self.events: list[str] = []
        self.fail = fail

    async def execute(self, op):
        self.events.append("start:" + op.operation_id)
        await asyncio.sleep(0.02)
        self.events.append("end:" + op.operation_id)
        return op.operation_id != self.fail


class Canceler:
    def __init__(self):
        self.ids: list[str] = []

    async def cancel(self, operation_id: str):
        self.ids.append(operation_id)


def test_scheduler_runs_same_layer_concurrently_and_barriers() -> None:
    raw = plan(
        [operation("a", "piper_0"), operation("b", "piper_1"), operation("c", "piper_0")],
        [["a", "b"], ["c"]],
        [
            [{"operation_id": "a", "robot_id": "piper_0"}, {"operation_id": "b", "robot_id": "piper_1"}],
            [{"operation_id": "c", "robot_id": "piper_0"}],
        ],
    )
    parsed = parse_robot_operation_plan(raw)
    executor = Executor()
    result = asyncio.run(schedule(parsed, executor))
    assert result.passed
    assert executor.events.index("end:a") < executor.events.index("start:c")
    assert executor.events.index("end:b") < executor.events.index("start:c")


def test_scheduler_cancels_pending_and_blocks_later_layers() -> None:
    raw = plan(
        [operation("bad", "piper_0"), operation("other", "piper_1"), operation("later", "piper_0")],
        [["bad", "other"], ["later"]],
        [
            [{"operation_id": "bad", "robot_id": "piper_0"}, {"operation_id": "other", "robot_id": "piper_1"}],
            [{"operation_id": "later", "robot_id": "piper_0"}],
        ],
    )
    parsed = parse_robot_operation_plan(raw)
    canceler = Canceler()
    result = asyncio.run(schedule(parsed, Executor("bad"), canceler))
    assert not result.passed
    assert result.operations["later"].status is OperationStatus.BLOCKED
