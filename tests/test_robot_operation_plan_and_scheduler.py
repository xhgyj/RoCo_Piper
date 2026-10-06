from __future__ import annotations

import asyncio

import pytest

from rocobrick.multi_robot_scheduler import OperationStatus, schedule
from rocobrick.robot_operation_plan import (
    RobotOperationPlanError,
    parse_robot_operation_plan,
)


def operation(
    operation_id: str, robot: str, skill: str = "WAIT", resource: str | None = None
) -> dict:
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
            "from": {
                "kind": "robot_current",
                "physical_instance_id": None,
                "connection_id": None,
                "robot_id": None,
            },
            "to": {
                "kind": "part",
                "physical_instance_id": "brick-1",
                "connection_id": None,
                "robot_id": None,
            },
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


def plan(
    ops: list[dict],
    layers: list[list[str]],
    schedule_slots: list[list[dict]],
    deps: list[dict] | None = None,
) -> dict:
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


def handover_operation(operation_id: str, donor: str, receiver: str) -> dict:
    """Build one generic HANDOVER operation document.

    Returns:
        The raw ``robot-operation-plan/v1`` operation document.
    """
    return {
        "operation_id": operation_id,
        "source_task_id": "task-" + operation_id,
        "skill": "HANDOVER",
        "executor_robot_id": receiver,
        "participants": [receiver, donor],
        "role_bindings": {
            "executor": receiver,
            "donor": donor,
            "receiver": receiver,
        },
        "target": {
            "physical_instance_id": "brick-1",
            "donor_robot_id": donor,
            "receiver_robot_id": receiver,
        },
        "parameters": {
            "grasp_profile_id": "brick-side-grasp-v1",
            "co_hold_timeout_s": 10.0,
            "follow_samples": 4,
        },
        "preconditions": [],
        "effects": [],
        "resources": [
            f"robot:{receiver}",
            f"robot:{donor}",
            "part:brick-1",
        ],
        "evidence_requirements": [],
    }


def test_parses_handover_and_core_skills() -> None:
    """HANDOVER parses alongside the core skills and binds donor/receiver."""
    ops = [
        operation("pick", "piper_0", "PICK"),
        handover_operation("handover", "piper_0", "piper_1"),
        operation("wait", "piper_0", "WAIT"),
        operation("place", "piper_1", "PLACE_DOWN"),
    ]
    parsed = parse_robot_operation_plan(
        plan(
            ops,
            [["pick"], ["handover"], ["wait", "place"]],
            [
                [{"operation_id": "pick", "robot_id": "piper_0"}],
                [{"operation_id": "handover", "robot_id": "piper_1"}],
                [
                    {"operation_id": "wait", "robot_id": "piper_0"},
                    {"operation_id": "place", "robot_id": "piper_1"},
                ],
            ],
        )
    )
    assert set(parsed.operations) == {"pick", "handover", "wait", "place"}
    handover = parsed.operations["handover"]
    assert handover.target.values["donor_robot_id"] == "piper_0"
    assert handover.target.values["receiver_robot_id"] == "piper_1"
    assert handover.executor_robot_id == "piper_1"
    assert "part:brick-1" in handover.resources


def test_transit_skill_is_rejected_by_the_frozen_plan() -> None:
    """The plan vocabulary no longer accepts standalone TRANSIT operations."""
    with pytest.raises(RobotOperationPlanError, match="unsupported"):
        parse_robot_operation_plan(
            plan(
                [operation("transit", "piper_1", "TRANSIT")],
                [["transit"]],
                [[{"operation_id": "transit", "robot_id": "piper_1"}]],
            )
        )


def upward_operation(operation_id: str, robot: str, skill: str, partner: str) -> dict:
    """Build one upward-collaboration operation with its partner binding.

    Returns:
        The raw ``robot-operation-plan/v1`` operation document.
    """
    partner_key = "support_robot_id" if skill == "PLACE_UP" else "placement_robot_id"
    target = {
        "connection_id": "conn-up",
        "upper_physical_instance_id": "brick-upper",
        "lower_physical_instance_id": "brick-lower",
        partner_key: partner,
    }
    participants = [robot, partner] if skill == "PLACE_UP" else [robot]
    bindings = {"executor": robot}
    if skill == "PLACE_UP":
        bindings["support"] = partner
    return {
        "operation_id": operation_id,
        "source_task_id": "task-" + operation_id,
        "skill": skill,
        "executor_robot_id": robot,
        "participants": participants,
        "role_bindings": bindings,
        "target": target,
        "parameters": {},
        "preconditions": [],
        "effects": [],
        "resources": [
            f"robot:{robot}",
            *([f"robot:{partner}"] if skill == "PLACE_UP" else []),
            "connection:conn-up",
            "part:brick-upper",
            *(["part:brick-lower"] if skill == "PLACE_UP" else []),
        ],
        "evidence_requirements": [],
    }


def test_parses_upward_collaboration_skills() -> None:
    ops = [
        upward_operation("support", "piper_1", "SUPPORT_TOP", "piper_0"),
        upward_operation("place-up", "piper_0", "PLACE_UP", "piper_1"),
    ]
    parsed = parse_robot_operation_plan(
        plan(
            ops,
            [["support"], ["place-up"]],
            [
                [{"operation_id": "support", "robot_id": "piper_1"}],
                [{"operation_id": "place-up", "robot_id": "piper_0"}],
            ],
            [
                {
                    "predecessor_operation_id": "support",
                    "successor_operation_id": "place-up",
                    "reason": "support first",
                }
            ],
        )
    )
    assert set(parsed.operations) == {"support", "place-up"}
    assert parsed.operations["place-up"].target.values["support_robot_id"] == "piper_1"
    assert parsed.operations["support"].target.values["placement_robot_id"] == "piper_0"


def test_upward_operation_requires_partner_as_participant() -> None:
    """PLACE_UP must include the support partner in its participant set."""
    raw = upward_operation("place-up", "piper_0", "PLACE_UP", "piper_1")
    raw["participants"] = ["piper_0"]
    raw["role_bindings"] = {"executor": "piper_0"}
    raw["resources"] = [
        "robot:piper_0",
        "connection:conn-up",
        "part:brick-upper",
        "part:brick-lower",
    ]
    document = plan(
        [raw],
        [["support"]],
        [
            [{"operation_id": "support", "robot_id": "piper_1"}],
        ],
    )
    with pytest.raises(RobotOperationPlanError, match="exactly the executor"):
        parse_robot_operation_plan(document)


def test_upward_operation_rejects_partner_equal_to_executor() -> None:
    raw = upward_operation("place-up", "piper_0", "PLACE_UP", "piper_0")
    raw["participants"] = ["piper_0", "piper_1"]
    raw["role_bindings"] = {"executor": "piper_0", "supporter": "piper_1"}
    document = plan(
        [raw],
        [["place-up"]],
        [
            [{"operation_id": "place-up", "robot_id": "piper_0"}],
        ],
    )
    with pytest.raises(RobotOperationPlanError, match="collaborating robot"):
        parse_robot_operation_plan(document)


def test_upward_connection_pair_must_be_labelled_consistently() -> None:
    """The same connection id cannot name two different brick pairs."""
    support = upward_operation("support", "piper_1", "SUPPORT_TOP", "piper_0")
    place_up = upward_operation("place-up", "piper_0", "PLACE_UP", "piper_1")
    place_up["target"]["lower_physical_instance_id"] = "brick-other"
    place_up["resources"] = [
        "robot:piper_0",
        "robot:piper_1",
        "connection:conn-up",
        "part:brick-upper",
        "part:brick-other",
    ]
    document = plan(
        [support, place_up],
        [["support"], ["place-up"]],
        [
            [{"operation_id": "support", "robot_id": "piper_1"}],
            [{"operation_id": "place-up", "robot_id": "piper_0"}],
        ],
        [
            {
                "predecessor_operation_id": "support",
                "successor_operation_id": "place-up",
                "reason": "support first",
            }
        ],
    )
    with pytest.raises(RobotOperationPlanError, match="bound to both"):
        parse_robot_operation_plan(document)


def test_rejects_unknown_operation_key() -> None:
    raw = plan(
        [operation("wait", "piper_0")],
        [["wait"]],
        [[{"operation_id": "wait", "robot_id": "piper_0"}]],
    )
    raw["operations"][0]["unexpected"] = True
    with pytest.raises(RobotOperationPlanError):
        parse_robot_operation_plan(raw)


def test_rejects_same_layer_resource_conflict() -> None:
    raw = plan(
        [
            operation("a", "piper_0", "WAIT", "part:shared"),
            operation("b", "piper_1", "WAIT", "part:shared"),
        ],
        [["a", "b"]],
        [
            [
                {"operation_id": "a", "robot_id": "piper_0"},
                {"operation_id": "b", "robot_id": "piper_1"},
            ]
        ],
    )
    with pytest.raises(RobotOperationPlanError):
        parse_robot_operation_plan(raw)


def test_rejects_dependency_not_in_later_layer() -> None:
    raw = plan(
        [operation("a", "piper_0"), operation("b", "piper_1")],
        [["a", "b"]],
        [
            [
                {"operation_id": "a", "robot_id": "piper_0"},
                {"operation_id": "b", "robot_id": "piper_1"},
            ]
        ],
        [
            {
                "predecessor_operation_id": "a",
                "successor_operation_id": "b",
                "reason": "required",
            }
        ],
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


def test_scheduler_accepts_skill_internal_free_space_plan_without_transit() -> None:
    """PICK/SUPPORT_TOP/PLACE_UP carry their own free-space phases."""
    pick = {
        "operation_id": "pick-lower",
        "source_task_id": "task-place-up",
        "skill": "PICK",
        "executor_robot_id": "piper_1",
        "participants": ["piper_1"],
        "role_bindings": {"executor": "piper_1"},
        "target": {"physical_instance_id": "brick-lower"},
        "parameters": {},
        "preconditions": [],
        "effects": ["holding:brick-lower"],
        "resources": ["robot:piper_1", "part:brick-lower"],
        "evidence_requirements": [],
    }
    support = {
        "operation_id": "support-upper",
        "source_task_id": "task-place-up",
        "skill": "SUPPORT_TOP",
        "executor_robot_id": "piper_0",
        "participants": ["piper_0"],
        "role_bindings": {"executor": "piper_0"},
        "target": {
            "connection_id": "conn-lower-upper",
            "upper_physical_instance_id": "brick-upper",
            "lower_physical_instance_id": "brick-lower",
            "placement_robot_id": "piper_1",
        },
        "parameters": {},
        "preconditions": [],
        "effects": ["supporting:brick-upper"],
        "resources": [
            "robot:piper_0",
            "connection:conn-lower-upper",
            "part:brick-upper",
        ],
        "evidence_requirements": [],
    }
    place_up = {
        "operation_id": "place-up-lower",
        "source_task_id": "task-place-up",
        "skill": "PLACE_UP",
        "executor_robot_id": "piper_1",
        "participants": ["piper_1", "piper_0"],
        "role_bindings": {"executor": "piper_1", "support": "piper_0"},
        "target": {
            "connection_id": "conn-lower-upper",
            "upper_physical_instance_id": "brick-upper",
            "lower_physical_instance_id": "brick-lower",
            "support_robot_id": "piper_0",
        },
        "parameters": {},
        "preconditions": ["holding:brick-lower", "supporting:brick-upper"],
        "effects": ["assembled:brick-lower"],
        "resources": [
            "robot:piper_1",
            "robot:piper_0",
            "connection:conn-lower-upper",
            "part:brick-upper",
            "part:brick-lower",
        ],
        "evidence_requirements": [],
    }
    raw = plan(
        [pick, support, place_up],
        [["pick-lower"], ["support-upper"], ["place-up-lower"]],
        [
            [{"operation_id": "pick-lower", "robot_id": "piper_1"}],
            [{"operation_id": "support-upper", "robot_id": "piper_0"}],
            [{"operation_id": "place-up-lower", "robot_id": "piper_1"}],
        ],
        deps=[
            {
                "predecessor_operation_id": "pick-lower",
                "successor_operation_id": "support-upper",
                "reason": "support follows grasp",
            },
            {
                "predecessor_operation_id": "pick-lower",
                "successor_operation_id": "place-up-lower",
                "reason": "placement follows grasp",
            },
            {
                "predecessor_operation_id": "support-upper",
                "successor_operation_id": "place-up-lower",
                "reason": "placement follows support",
            },
        ],
    )
    parsed = parse_robot_operation_plan(raw)
    assert {item.skill for item in parsed.operations.values()} == {
        "PICK",
        "SUPPORT_TOP",
        "PLACE_UP",
    }

    class RecordingExecutor(Executor):
        def __init__(self):
            super().__init__()
            self.skills: list[str] = []

        async def execute(self, op):
            self.skills.append(op.skill)
            return await super().execute(op)

    executor = RecordingExecutor()
    result = asyncio.run(schedule(parsed, executor))
    assert result.passed
    assert executor.skills == ["PICK", "SUPPORT_TOP", "PLACE_UP"]
    assert all("transit" not in operation_id for operation_id in result.operations)

    raw = plan(
        [
            operation("a", "piper_0"),
            operation("b", "piper_1"),
            operation("c", "piper_0"),
        ],
        [["a", "b"], ["c"]],
        [
            [
                {"operation_id": "a", "robot_id": "piper_0"},
                {"operation_id": "b", "robot_id": "piper_1"},
            ],
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
        [
            operation("bad", "piper_0"),
            operation("other", "piper_1"),
            operation("later", "piper_0"),
        ],
        [["bad", "other"], ["later"]],
        [
            [
                {"operation_id": "bad", "robot_id": "piper_0"},
                {"operation_id": "other", "robot_id": "piper_1"},
            ],
            [{"operation_id": "later", "robot_id": "piper_0"}],
        ],
    )
    parsed = parse_robot_operation_plan(raw)
    canceler = Canceler()
    result = asyncio.run(schedule(parsed, Executor("bad"), canceler))
    assert not result.passed
    assert result.operations["later"].status is OperationStatus.BLOCKED
