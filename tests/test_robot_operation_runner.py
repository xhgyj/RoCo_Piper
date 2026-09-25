import asyncio

from rocobrick.robot_operation_plan import parse_robot_operation_plan
from rocobrick.robot_operation_runner import OperationPlanRunner, serialize_run_result


def plan():
    operation = {
        "operation_id": "wait-0",
        "source_task_id": "task-0",
        "skill": "WAIT",
        "executor_robot_id": "piper_0",
        "participants": ["piper_0"],
        "role_bindings": {"executor": "piper_0"},
        "target": None,
        "parameters": {},
        "preconditions": [],
        "effects": [],
        "resources": ["robot:piper_0"],
        "evidence_requirements": [],
    }
    return parse_robot_operation_plan({
        "schema": "robot-operation-plan/v1",
        "plan_id": "run",
        "source_plan_id": "assembly",
        "robots": ["piper_0"],
        "operations": [operation],
        "dependencies": [],
        "execution_layers": [["wait-0"]],
        "planned_schedule": [[{"operation_id": "wait-0", "robot_id": "piper_0"}]],
        "evidence_requirements": [],
    })


class Executor:
    async def execute(self, operation):
        return {"passed": True, "robot_id": operation.executor_robot_id}


def test_runner_serializes_scheduler_evidence():
    result = asyncio.run(OperationPlanRunner(plan(), Executor()).run())
    document = serialize_run_result(result)
    assert document["schema"] == "rocobrick-operation-plan-run/v1"
    assert document["passed"] is True
    assert document["operations"]["wait-0"]["passed"] is True
