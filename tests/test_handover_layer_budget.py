"""Finite trial-layer budgets must not extend the original PICK-layer deadline."""

import asyncio

import pytest

from rocobrick.multi_robot_scheduler import MultiRobotScheduler
from rocobrick.robot_operation_plan import parse_robot_operation_plan
from rocobrick.robot_operation_runner import OperationPlanRunner


def two_layers():
    """Build ordinary WAIT layers with no simulator or physical motion.

    Returns:
        A valid external-plan contract for testing scheduler budgets only.
    """
    operations = [
        {
            "operation_id": f"wait-{i}",
            "source_task_id": f"task-{i}",
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
        for i in range(2)
    ]
    return parse_robot_operation_plan(
        {
            "schema": "robot-operation-plan/v1",
            "plan_id": "budget-test",
            "source_plan_id": "assembly",
            "robots": ["piper_0"],
            "operations": operations,
            "dependencies": [],
            "execution_layers": [[op["operation_id"]] for op in operations],
            "planned_schedule": [
                [{"operation_id": op["operation_id"], "robot_id": "piper_0"}]
                for op in operations
            ],
            "evidence_requirements": [],
        }
    )


class TimedExecutor:
    """A legacy in-memory timing stub, never a physical executor."""

    def __init__(self, slow_first=False):
        """Choose which synthetic operation takes longer than the default."""
        self.slow_first = slow_first

    async def execute(self, operation):
        """Return a synthetic terminal result after a bounded asynchronous wait.

        Returns:
            A legacy result for scheduler timing tests only.
        """
        slow = (operation.operation_id == "wait-0") == self.slow_first
        await asyncio.sleep(0.05 if slow else 0.001)
        return {"passed": True, "robot_id": operation.executor_robot_id}


def test_only_selected_layer_gets_extended_budget():
    """A longer second layer must not inherit the short default timeout."""
    result = asyncio.run(
        OperationPlanRunner(
            two_layers(),
            TimedExecutor(),
            operation_timeout_s=0.02,
            layer_timeout_s_by_index={1: 0.3},
        ).run()
    )
    assert result.passed


def test_original_layer_deadline_is_preserved():
    """Overriding layer one cannot protect a timed-out layer zero."""
    result = asyncio.run(
        OperationPlanRunner(
            two_layers(),
            TimedExecutor(slow_first=True),
            operation_timeout_s=0.02,
            layer_timeout_s_by_index={1: 0.3},
        ).run()
    )
    assert not result.passed
    assert result.operations["wait-1"].status.value == "BLOCKED"


@pytest.mark.parametrize(
    "overrides",
    [
        {True: 1},
        {-1: 1},
        {0: True},
        {0: 0},
        {0: -1},
        {0: float("inf")},
        {0: float("nan")},
        {0: 3601},
    ],
)
def test_layer_budget_requires_finite_indices_and_bounds(overrides):
    """Malformed trial budgets reject before an operation is dispatched."""
    with pytest.raises(ValueError, match="layer timeout"):
        MultiRobotScheduler(layer_timeout_s_by_index=overrides)


def test_layer_override_cannot_reference_missing_layer():
    """A stale layer index fails before resources or execution are touched."""
    runner = OperationPlanRunner(
        two_layers(),
        TimedExecutor(),
        layer_timeout_s_by_index={2: 1},
    )
    with pytest.raises(ValueError, match="unavailable"):
        asyncio.run(runner.run())
