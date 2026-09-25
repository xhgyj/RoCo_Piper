"""Binding tests for ``robot-operation-plan/v1`` against an authoritative STAP.

These tests pin the frozen wire shape (keys, four skills, target shapes, and
``robot:``/``part:``/``connection:`` resources) and the fail-closed binding to
``assembly-plan/v2``: plan id and robot set equality, source-task/executor
allocation, grounded part/connection references, layer/schedule coverage,
dependency order, and same-layer resource exclusivity. Offline only; no Isaac,
ROS, Docker, or motion.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from rocobrick.models import SkillType
from rocobrick.robot_operation_plan import (
    RobotOperationPlanError,
    ScheduledOperation,
    parse_robot_operation_plan,
)
from rocobrick.stap_contract import (
    StapPlan,
    StructuralPart,
    StructuralTask,
    parse_stap_plan,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
# Committed assembly-plan/v2 fixture whose canonical plan_id is the exact
# ``source_plan_id`` declared by the committed example1 operation-plan
# artifact. All tests stay on tracked files: no run artifact is required.
EXAMPLE1_STAP_PATH = (
    REPO_ROOT / "runs" / "scene-smoke-20260914T093942Z" / "input" / "plan.json"
)
EXAMPLE1_PLAN_PATH = (
    REPO_ROOT / "artifacts" / "scene" / "example1-robot-operation-plan.json"
)


def _load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Synthetic authoritative STAP: base -> mid -> top stack handled by two robots.
# ---------------------------------------------------------------------------


def _part(instance_id, key, start_key, x, y, z, color="Red"):
    return StructuralPart(
        physical_instance_id=instance_id,
        source_goal_key=key,
        source_start_key=start_key,
        brick_id=2,
        x=x,
        y=y,
        z=z,
        orientation_quarter_turns=0,
        color=color,
    )


def _task(task_id, part_id, primary, *, supporting=None, predecessors=(), supports=()):
    return StructuralTask(
        task_id=task_id,
        physical_instance_id=part_id,
        skill=SkillType.PLACE_DOWN,
        primary_robot=primary,
        supporting_robot=supporting,
        predecessors=frozenset(predecessors),
        initial_supports=frozenset(supports),
        level=0,
        schedule_slot=0,
    )


def _synthetic_stap() -> StapPlan:
    parts = (
        _part("brick-base", "0", "s0", 0, 0, 0, "Red"),
        _part("brick-mid", "1", None, 0, 0, 1, "Blue"),
        _part("brick-top", "2", None, 0, 0, 2, "Green"),
        _part("brick-side", "3", None, 1, 0, 1, "Yellow"),
        _part("brick-far", "4", "s4", 5, 0, 0, "White"),
    )
    tasks = (
        _task("task-mid", "brick-mid", "piper_a", supports=("brick-base",)),
        _task("task-top", "brick-top", "piper_b", predecessors=("task-mid",)),
    )
    return StapPlan(
        plan_id="assembly-plan-synthetic",
        robots=("piper_a", "piper_b"),
        source_goal_sha256="a" * 64,
        source_start_sha256="b" * 64,
        goal_parts={part.physical_instance_id: part for part in parts},
        initially_assembled=frozenset({"brick-base", "brick-far"}),
        tasks={task.task_id: task for task in tasks},
        hltl_hierarchy=({"p0": "(<> p1)"},),
        hltl_root_formula="(<> p1)",
        raw={},
    )


# ---------------------------------------------------------------------------
# Plan builders.
# ---------------------------------------------------------------------------


def _ref(kind, *, part=None, connection=None, robot=None):
    return {
        "kind": kind,
        "physical_instance_id": part,
        "connection_id": connection,
        "robot_id": robot,
    }


def _transit(source, destination, carrying=None):
    return {
        "from": source,
        "to": destination,
        "carrying_physical_instance_id": carrying,
    }


def _place(connection, upper, lower):
    return {
        "connection_id": connection,
        "upper_physical_instance_id": upper,
        "lower_physical_instance_id": lower,
    }


def _operation(
    operation_id,
    source_task_id,
    skill,
    executor,
    *,
    participants=None,
    target=None,
    resources=None,
    evidence=(),
    parameters=None,
    role_bindings=None,
):
    participants = [executor] if participants is None else list(participants)
    if role_bindings is None:
        role_bindings = (
            {"executor": executor}
            if len(participants) == 1
            else {"executor": executor, "supporting": participants[1]}
        )
    return {
        "operation_id": operation_id,
        "source_task_id": source_task_id,
        "skill": skill,
        "executor_robot_id": executor,
        "participants": participants,
        "role_bindings": role_bindings,
        "target": target,
        "parameters": {} if parameters is None else parameters,
        "preconditions": [],
        "effects": [],
        "resources": [f"robot:{executor}"] if resources is None else list(resources),
        "evidence_requirements": list(evidence),
    }


def _schedule(*slots):
    return [[dict(entry) for entry in slot] for slot in slots]


def _plan(
    operations,
    layers,
    schedule,
    *,
    dependencies=(),
    evidence=(),
    robots=("piper_a", "piper_b"),
    plan_id="robot-op-synthetic",
    source_plan_id="assembly-plan-synthetic",
):
    return {
        "schema": "robot-operation-plan/v1",
        "plan_id": plan_id,
        "source_plan_id": source_plan_id,
        "robots": list(robots),
        "operations": list(operations),
        "dependencies": [dict(item) for item in dependencies],
        "execution_layers": [list(layer) for layer in layers],
        "planned_schedule": [list(slot) for slot in schedule],
        "evidence_requirements": [dict(item) for item in evidence],
    }


def _default_operations():
    return [
        _operation(
            "a-transit",
            "task-mid",
            "TRANSIT",
            "piper_a",
            target=_transit(
                _ref("robot_current"),
                _ref("part", part="brick-mid"),
            ),
            resources=["robot:piper_a"],
        ),
        _operation(
            "a-pick",
            "task-mid",
            "PICK",
            "piper_a",
            target={"physical_instance_id": "brick-mid"},
            resources=["robot:piper_a", "part:brick-mid"],
            evidence=["holding_confirmed"],
        ),
        _operation(
            "a-place",
            "task-mid",
            "PLACE_DOWN",
            "piper_a",
            target=_place("conn-mid-base", "brick-mid", "brick-base"),
            resources=[
                "robot:piper_a",
                "connection:conn-mid-base",
                "part:brick-mid",
                "part:brick-base",
            ],
            evidence=["connection_confirmed"],
        ),
        _operation(
            "b-transit",
            "task-top",
            "TRANSIT",
            "piper_b",
            target=_transit(
                _ref("part", part="brick-mid"),
                _ref("connection", connection="conn-top-mid"),
                carrying="brick-top",
            ),
            resources=["robot:piper_b", "part:brick-top", "part:brick-mid"],
            evidence=["carrying_state_confirmed"],
        ),
        _operation(
            "b-place",
            "task-top",
            "PLACE_DOWN",
            "piper_b",
            target=_place("conn-top-mid", "brick-top", "brick-mid"),
            resources=[
                "robot:piper_b",
                "connection:conn-top-mid",
                "part:brick-top",
                "part:brick-mid",
            ],
            evidence=["connection_confirmed"],
        ),
    ]


_DEFAULT_LAYERS = (
    ("a-transit",),
    ("a-pick",),
    ("a-place",),
    ("b-transit",),
    ("b-place",),
)
_DEFAULT_SCHEDULE = _schedule(
    ({"operation_id": "a-transit", "robot_id": "piper_a"},),
    ({"operation_id": "a-pick", "robot_id": "piper_a"},),
    ({"operation_id": "a-place", "robot_id": "piper_a"},),
    ({"operation_id": "b-transit", "robot_id": "piper_b"},),
    ({"operation_id": "b-place", "robot_id": "piper_b"},),
)
def _dependency(before, after, reason):
    return {
        "predecessor_operation_id": before,
        "successor_operation_id": after,
        "reason": reason,
    }


_DEFAULT_DEPENDENCIES = (
    _dependency("a-transit", "a-pick", "pick requires transit"),
    _dependency("a-pick", "a-place", "place requires holding"),
    _dependency("a-place", "b-transit", "carry onto placed support"),
    _dependency("b-transit", "b-place", "place requires carry"),
)


def _default_plan(**overrides):
    parameters = {
        "operations": _default_operations(),
        "layers": _DEFAULT_LAYERS,
        "schedule": _DEFAULT_SCHEDULE,
        "dependencies": _DEFAULT_DEPENDENCIES,
    }
    parameters.update(overrides)
    return _plan(
        parameters["operations"],
        parameters["layers"],
        parameters["schedule"],
        dependencies=parameters["dependencies"],
        evidence=parameters.get("evidence", ()),
        robots=parameters.get("robots", ("piper_a", "piper_b")),
        source_plan_id=parameters.get("source_plan_id", "assembly-plan-synthetic"),
    )


def _one(operation, *, layers=None, schedule=None, **overrides):
    layers = [list(layer) for layer in (layers or ([operation["operation_id"]],))]
    entry = {
        "operation_id": operation["operation_id"],
        "robot_id": operation["executor_robot_id"],
    }
    schedule = schedule or _schedule((entry,))
    return _plan([operation], layers, schedule, **overrides)


# ---------------------------------------------------------------------------
# example1 artifact binding.
# ---------------------------------------------------------------------------


def test_example1_stap_fixture_matches_the_artifact() -> None:
    stap = parse_stap_plan(_load(EXAMPLE1_STAP_PATH))
    artifact = _load(EXAMPLE1_PLAN_PATH)
    assert stap.plan_id == "assembly-plan-25a5d5cc586a1e5c65f2141d"
    assert artifact["source_plan_id"] == stap.plan_id
    assert stap.robots == ("piper_0", "piper_1")
    assert set(stap.goal_parts) == {
        "brick-4bbba49bec24496bb21e",
        "brick-768de5a17827920c48d4",
    }
    task = stap.tasks["assemble-768de5a17827920c48d4"]
    assert task.physical_instance_id == "brick-768de5a17827920c48d4"
    assert task.primary_robot == "piper_0"
    assert task.supporting_robot is None
    assert task.initial_supports == frozenset({"brick-4bbba49bec24496bb21e"})


def test_example1_artifact_parses_with_validated_stap() -> None:
    stap = parse_stap_plan(_load(EXAMPLE1_STAP_PATH))
    parsed = parse_robot_operation_plan(_load(EXAMPLE1_PLAN_PATH), source_plan=stap)
    assert parsed.plan_id == "robot-operation-plan-example1-v1"
    assert parsed.source_plan_id == stap.plan_id
    assert parsed.robots == ("piper_0", "piper_1")
    assert [operation.skill for operation in parsed.operations.values()] == [
        "TRANSIT",
        "WAIT",
        "PICK",
        "TRANSIT",
        "PLACE_DOWN",
    ]
    assert parsed.operations["example1-piper0-place-down"].target.values[
        "lower_physical_instance_id"
    ] == "brick-4bbba49bec24496bb21e"


def test_example1_artifact_parses_with_raw_stap_payload() -> None:
    parsed = parse_robot_operation_plan(
        _load(EXAMPLE1_PLAN_PATH), source_plan=_load(EXAMPLE1_STAP_PATH)
    )
    assert set(parsed.operations) == {
        "example1-piper0-transit-to-pick",
        "example1-piper1-wait-stable",
        "example1-piper0-pick",
        "example1-piper0-transit-to-place",
        "example1-piper0-place-down",
    }


def test_example1_artifact_input_order_is_preserved_and_never_mutated() -> None:
    raw = _load(EXAMPLE1_PLAN_PATH)
    before = copy.deepcopy(raw)
    parsed = parse_robot_operation_plan(
        raw, source_plan=parse_stap_plan(_load(EXAMPLE1_STAP_PATH))
    )
    assert raw == before
    assert list(parsed.operations) == [
        operation["operation_id"] for operation in before["operations"]
    ]
    assert parsed.execution_layers == tuple(
        tuple(layer) for layer in before["execution_layers"]
    )
    assert [
        [(entry.slot, entry.operation_id, entry.robot_id) for entry in slot]
        for slot in parsed.planned_schedule
    ] == [
        [(slot, entry["operation_id"], entry["robot_id"]) for entry in entries]
        for slot, entries in enumerate(before["planned_schedule"])
    ]


# ---------------------------------------------------------------------------
# Synthetic happy paths.
# ---------------------------------------------------------------------------


def test_synthetic_plan_binds_to_its_stap() -> None:
    parsed = parse_robot_operation_plan(_default_plan(), source_plan=_synthetic_stap())
    assert list(parsed.operations) == [
        "a-transit",
        "a-pick",
        "a-place",
        "b-transit",
        "b-place",
    ]
    assert parsed.dependencies[1].predecessor_operation_id == "a-pick"
    assert isinstance(parsed.planned_schedule[0][0], ScheduledOperation)
    assert parsed.planned_schedule[0][0].slot == 0


def test_supporting_robot_allocation_is_accepted_when_fully_bound() -> None:
    stap = _synthetic_stap()
    supporting = replace(
        stap,
        tasks={
            **stap.tasks,
            "task-mid": replace(stap.tasks["task-mid"], supporting_robot="piper_b"),
        },
    )
    operation = _operation(
        "a-pick",
        "task-mid",
        "PICK",
        "piper_a",
        participants=["piper_a", "piper_b"],
        role_bindings={"executor": "piper_a", "supporting": "piper_b"},
        target={"physical_instance_id": "brick-mid"},
        resources=["robot:piper_a", "robot:piper_b", "part:brick-mid"],
    )
    parsed = parse_robot_operation_plan(_one(operation), source_plan=supporting)
    assert parsed.operations["a-pick"].role_bindings["supporting"] == "piper_b"


# ---------------------------------------------------------------------------
# STAP cross-checks: identity and allocation.
# ---------------------------------------------------------------------------


def test_rejects_source_plan_id_mismatch() -> None:
    raw = _default_plan(source_plan_id="assembly-plan-other")
    with pytest.raises(RobotOperationPlanError, match="source_plan_id"):
        parse_robot_operation_plan(raw, source_plan=_synthetic_stap())


def test_rejects_robot_set_mismatch() -> None:
    raw = _default_plan(robots=("piper_a",))
    with pytest.raises(RobotOperationPlanError, match="robots"):
        parse_robot_operation_plan(raw, source_plan=_synthetic_stap())


def test_rejects_extra_robot_beyond_the_stap() -> None:
    raw = _default_plan(robots=("piper_a", "piper_b", "piper_c"))
    with pytest.raises(RobotOperationPlanError, match="robots"):
        parse_robot_operation_plan(raw, source_plan=_synthetic_stap())


def test_rejects_empty_robots_even_without_a_stap() -> None:
    raw = _default_plan(robots=())
    with pytest.raises(RobotOperationPlanError, match="robots"):
        parse_robot_operation_plan(raw)


def test_rejects_duplicate_robots() -> None:
    raw = _default_plan(robots=("piper_a", "piper_a"))
    with pytest.raises(RobotOperationPlanError, match="duplicates"):
        parse_robot_operation_plan(raw)


def test_rejects_unknown_source_task() -> None:
    raw = _default_plan()
    raw["operations"][0]["source_task_id"] = "task-absent"
    with pytest.raises(RobotOperationPlanError, match="source_task_id"):
        parse_robot_operation_plan(raw, source_plan=_synthetic_stap())


def test_rejects_motion_executor_outside_allocation() -> None:
    raw = _default_plan()
    operation = raw["operations"][3]
    operation["executor_robot_id"] = "piper_a"
    operation["participants"] = ["piper_a"]
    operation["role_bindings"] = {"executor": "piper_a"}
    operation["resources"] = ["robot:piper_a", "part:brick-top", "part:brick-mid"]
    raw["planned_schedule"][3] = [{"operation_id": "b-transit", "robot_id": "piper_a"}]
    with pytest.raises(RobotOperationPlanError, match="allocation"):
        parse_robot_operation_plan(raw, source_plan=_synthetic_stap())


def test_rejects_motion_participants_beyond_allocation() -> None:
    stap = _synthetic_stap()
    operation = _operation(
        "a-pick",
        "task-mid",
        "PICK",
        "piper_a",
        participants=["piper_a", "piper_b"],
        role_bindings={"executor": "piper_a", "supporting": "piper_b"},
        target={"physical_instance_id": "brick-mid"},
        resources=["robot:piper_a", "robot:piper_b", "part:brick-mid"],
    )
    with pytest.raises(RobotOperationPlanError, match="participants"):
        parse_robot_operation_plan(_one(operation), source_plan=stap)


def test_rejects_motion_skill_on_robot_without_any_allocation() -> None:
    stap = _synthetic_stap()
    operation = _operation(
        "b-pick",
        "task-top",
        "PICK",
        "piper_a",
        target={"physical_instance_id": "brick-top"},
        resources=["robot:piper_a", "part:brick-top"],
    )
    with pytest.raises(RobotOperationPlanError, match="allocation"):
        parse_robot_operation_plan(_one(operation), source_plan=stap)


# ---------------------------------------------------------------------------
# STAP cross-checks: grounded targets.
# ---------------------------------------------------------------------------


def test_rejects_pick_of_a_foreign_part() -> None:
    stap = _synthetic_stap()
    operation = _operation(
        "a-pick",
        "task-mid",
        "PICK",
        "piper_a",
        target={"physical_instance_id": "brick-base"},
        resources=["robot:piper_a", "part:brick-base"],
    )
    with pytest.raises(RobotOperationPlanError, match="PICK"):
        parse_robot_operation_plan(_one(operation), source_plan=stap)


def test_rejects_transit_part_outside_task_and_supports() -> None:
    stap = _synthetic_stap()
    operation = _operation(
        "a-transit",
        "task-mid",
        "TRANSIT",
        "piper_a",
        target=_transit(_ref("robot_current"), _ref("part", part="brick-side")),
        resources=["robot:piper_a"],
    )
    with pytest.raises(RobotOperationPlanError, match="TRANSIT"):
        parse_robot_operation_plan(_one(operation), source_plan=stap)


def test_rejects_transit_carrying_a_foreign_part() -> None:
    stap = _synthetic_stap()
    operation = _operation(
        "b-transit",
        "task-top",
        "TRANSIT",
        "piper_b",
        target=_transit(
            _ref("robot_current"),
            _ref("part", part="brick-mid"),
            carrying="brick-base",
        ),
        resources=["robot:piper_b", "part:brick-base"],
    )
    with pytest.raises(RobotOperationPlanError, match="carries"):
        parse_robot_operation_plan(_one(operation), source_plan=stap)


def test_rejects_transit_connection_not_placed_by_its_source_task() -> None:
    raw = _default_plan()
    raw["operations"][3]["target"]["to"] = _ref(
        "connection", connection="conn-mid-base"
    )
    with pytest.raises(RobotOperationPlanError, match="not placed"):
        parse_robot_operation_plan(raw, source_plan=_synthetic_stap())


def test_rejects_place_down_upper_not_the_task_part() -> None:
    stap = _synthetic_stap()
    operation = _operation(
        "a-place",
        "task-mid",
        "PLACE_DOWN",
        "piper_a",
        target=_place("conn-x", "brick-top", "brick-mid"),
        resources=[
            "robot:piper_a",
            "connection:conn-x",
            "part:brick-top",
            "part:brick-mid",
        ],
    )
    with pytest.raises(RobotOperationPlanError, match="upper"):
        parse_robot_operation_plan(_one(operation), source_plan=stap)


def test_rejects_place_down_lower_not_a_support() -> None:
    stap = _synthetic_stap()
    operation = _operation(
        "b-place",
        "task-top",
        "PLACE_DOWN",
        "piper_b",
        target=_place("conn-x", "brick-top", "brick-far"),
        resources=[
            "robot:piper_b",
            "connection:conn-x",
            "part:brick-top",
            "part:brick-far",
        ],
    )
    with pytest.raises(RobotOperationPlanError, match="support"):
        parse_robot_operation_plan(_one(operation), source_plan=stap)


def test_rejects_place_down_lower_not_directly_beneath() -> None:
    stap = _synthetic_stap()
    off_stack = replace(
        stap,
        tasks={
            **stap.tasks,
            "task-top": replace(
                stap.tasks["task-top"],
                predecessors=frozenset(),
                initial_supports=frozenset({"brick-side"}),
            ),
        },
    )
    operation = _operation(
        "b-place",
        "task-top",
        "PLACE_DOWN",
        "piper_b",
        target=_place("conn-x", "brick-top", "brick-side"),
        resources=[
            "robot:piper_b",
            "connection:conn-x",
            "part:brick-top",
            "part:brick-side",
        ],
    )
    with pytest.raises(RobotOperationPlanError, match="beneath"):
        parse_robot_operation_plan(_one(operation), source_plan=off_stack)


def test_rejects_unknown_part_reference() -> None:
    stap = _synthetic_stap()
    operation = _operation(
        "a-pick",
        "task-mid",
        "PICK",
        "piper_a",
        target={"physical_instance_id": "brick-absent"},
        resources=["robot:piper_a", "part:brick-absent"],
    )
    with pytest.raises(RobotOperationPlanError, match="PICK"):
        parse_robot_operation_plan(_one(operation), source_plan=stap)


# ---------------------------------------------------------------------------
# Connection grounding (opaque ids, never invented).
# ---------------------------------------------------------------------------


def test_rejects_connection_bound_to_two_pairs() -> None:
    first = _operation(
        "p1",
        "task-mid",
        "PLACE_DOWN",
        "piper_a",
        target=_place("conn-shared", "brick-mid", "brick-base"),
        resources=[
            "robot:piper_a",
            "connection:conn-shared",
            "part:brick-mid",
            "part:brick-base",
        ],
    )
    second = _operation(
        "p2",
        "task-top",
        "PLACE_DOWN",
        "piper_b",
        target=_place("conn-shared", "brick-top", "brick-mid"),
        resources=[
            "robot:piper_b",
            "connection:conn-shared",
            "part:brick-top",
            "part:brick-mid",
        ],
    )
    raw = _plan(
        [first, second],
        [["p1"], ["p2"]],
        _schedule(
            ({"operation_id": "p1", "robot_id": "piper_a"},),
            ({"operation_id": "p2", "robot_id": "piper_b"},),
        ),
    )
    with pytest.raises(RobotOperationPlanError, match="bound to both"):
        parse_robot_operation_plan(raw)


def test_rejects_pair_labelled_by_two_connection_ids() -> None:
    first = _operation(
        "p1",
        "task-mid",
        "PLACE_DOWN",
        "piper_a",
        target=_place("conn-one", "brick-mid", "brick-base"),
        resources=[
            "robot:piper_a",
            "connection:conn-one",
            "part:brick-mid",
            "part:brick-base",
        ],
    )
    second = _operation(
        "p2",
        "task-mid",
        "PLACE_DOWN",
        "piper_a",
        target=_place("conn-two", "brick-mid", "brick-base"),
        resources=[
            "robot:piper_a",
            "connection:conn-two",
            "part:brick-mid",
            "part:brick-base",
        ],
    )
    raw = _plan(
        [first, second],
        [["p1"], ["p2"]],
        _schedule(
            ({"operation_id": "p1", "robot_id": "piper_a"},),
            ({"operation_id": "p2", "robot_id": "piper_a"},),
        ),
    )
    with pytest.raises(RobotOperationPlanError, match="labelled"):
        parse_robot_operation_plan(raw)


def test_rejects_ungrounded_connection_without_a_source_plan() -> None:
    operation = _operation(
        "t1",
        "task-mid",
        "TRANSIT",
        "piper_a",
        target=_transit(
            _ref("robot_current"),
            _ref("connection", connection="conn-nowhere"),
        ),
        resources=["robot:piper_a"],
    )
    with pytest.raises(RobotOperationPlanError, match="grounds"):
        parse_robot_operation_plan(_one(operation))


def test_rejects_a_second_connection_for_the_same_source_task() -> None:
    raw = _default_plan()
    raw["operations"].append(
        _operation(
            "a-place-again",
            "task-mid",
            "PLACE_DOWN",
            "piper_a",
            target=_place("conn-mid-base-2", "brick-mid", "brick-far"),
            resources=[
                "robot:piper_a",
                "connection:conn-mid-base-2",
                "part:brick-mid",
                "part:brick-far",
            ],
        )
    )
    raw["execution_layers"].append(["a-place-again"])
    raw["planned_schedule"].append(
        [{"operation_id": "a-place-again", "robot_id": "piper_a"}]
    )
    with pytest.raises(RobotOperationPlanError):
        parse_robot_operation_plan(raw, source_plan=_synthetic_stap())


# ---------------------------------------------------------------------------
# WAIT is a passive single-robot observation.
# ---------------------------------------------------------------------------


def test_wait_may_run_on_a_robot_without_motion_allocation() -> None:
    operation = _operation(
        "w1", "task-mid", "WAIT", "piper_b", target=None, resources=["robot:piper_b"]
    )
    parsed = parse_robot_operation_plan(_one(operation), source_plan=_synthetic_stap())
    assert parsed.operations["w1"].executor_robot_id == "piper_b"
    assert parsed.operations["w1"].target is None


def test_wait_still_requires_an_existing_source_task() -> None:
    operation = _operation(
        "w1", "task-absent", "WAIT", "piper_b", target=None, resources=["robot:piper_b"]
    )
    with pytest.raises(RobotOperationPlanError, match="source_task_id"):
        parse_robot_operation_plan(_one(operation), source_plan=_synthetic_stap())


def test_rejects_wait_with_extra_participant() -> None:
    operation = _operation(
        "w1",
        "task-mid",
        "WAIT",
        "piper_b",
        participants=["piper_b", "piper_a"],
        role_bindings={"executor": "piper_b", "observer": "piper_a"},
        target=None,
        resources=["robot:piper_b", "robot:piper_a"],
    )
    with pytest.raises(RobotOperationPlanError, match="WAIT participants"):
        parse_robot_operation_plan(_one(operation), source_plan=_synthetic_stap())


def test_rejects_wait_with_a_target() -> None:
    stap = _synthetic_stap()
    operation = _operation(
        "w1",
        "task-mid",
        "WAIT",
        "piper_b",
        target={"physical_instance_id": "brick-mid"},
    )
    with pytest.raises(RobotOperationPlanError, match="null"):
        parse_robot_operation_plan(_one(operation), source_plan=stap)


# ---------------------------------------------------------------------------
# Frozen shapes, resources, coverage, ordering.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda op: op.update({"skill": "TELEPORT"}),
            id="unknown-skill",
        ),
        pytest.param(
            lambda op: op["target"].update({"unexpected": 1}),
            id="transit-extra-target-key",
        ),
        pytest.param(
            lambda op: op["target"].update(
                {"to": _ref("part", part=None)}
            ),
            id="part-ref-without-id",
        ),
        pytest.param(
            lambda op: op["target"].update({"from": _ref("unknown_kind")}),
            id="unsupported-ref-kind",
        ),
        pytest.param(
            lambda op: op["target"].update(
                {"from": _ref("robot_current", part="brick-mid")}
            ),
            id="robot-current-with-part",
        ),
        pytest.param(
            lambda op: op["target"].update({"to": _ref("robot_home")}),
            id="robot-home-without-robot",
        ),
        pytest.param(
            lambda op: op["target"].update(
                {"to": _ref("robot_home", robot="piper_b")}
            ),
            id="robot-home-of-another-robot",
        ),
        pytest.param(
            lambda op: op.update({"parameters": []}),
            id="parameters-not-object",
        ),
        pytest.param(
            lambda op: op.update({"resources": ["robot:piper_a", "tool:hammer"]}),
            id="unknown-resource-kind",
        ),
        pytest.param(
            lambda op: op.update({"resources": ["robot:piper_b"]}),
            id="robot-resource-outside-participants",
        ),
        pytest.param(
            lambda op: op.update({"resources": ["robot:piper_a", "part:brick-top"]}),
            id="part-resource-not-in-target",
        ),
    ],
)
def test_single_operation_shape_and_resource_violations_are_rejected(mutate) -> None:
    operation = _operation(
        "a-transit",
        "task-mid",
        "TRANSIT",
        "piper_a",
        target=_transit(_ref("robot_current"), _ref("part", part="brick-mid")),
        resources=["robot:piper_a"],
    )
    mutate(operation)
    with pytest.raises(RobotOperationPlanError):
        parse_robot_operation_plan(_one(operation))


def test_rejects_connection_resource_not_in_target() -> None:
    operation = _operation(
        "a-pick",
        "task-mid",
        "PICK",
        "piper_a",
        target={"physical_instance_id": "brick-mid"},
        resources=["robot:piper_a", "part:brick-mid", "connection:conn-x"],
    )
    with pytest.raises(RobotOperationPlanError, match="connection"):
        parse_robot_operation_plan(_one(operation))


def test_rejects_participants_outside_the_plan() -> None:
    operation = _operation(
        "w1",
        "task-mid",
        "WAIT",
        "piper_a",
        participants=["piper_a", "piper_z"],
        role_bindings={"executor": "piper_a", "observer": "piper_z"},
    )
    with pytest.raises(RobotOperationPlanError, match="participants"):
        parse_robot_operation_plan(_one(operation))


def test_rejects_incomplete_role_bindings() -> None:
    operation = _operation(
        "w1",
        "task-mid",
        "WAIT",
        "piper_a",
        participants=["piper_a", "piper_b"],
        role_bindings={"executor": "piper_a"},
        resources=["robot:piper_a", "robot:piper_b"],
    )
    with pytest.raises(RobotOperationPlanError, match="role_bindings"):
        parse_robot_operation_plan(_one(operation))


def test_rejects_same_layer_resource_conflict() -> None:
    first = _operation(
        "a-pick",
        "task-mid",
        "PICK",
        "piper_a",
        target={"physical_instance_id": "brick-mid"},
        resources=["robot:piper_a", "part:brick-mid"],
    )
    second = _operation(
        "b-transit",
        "task-top",
        "TRANSIT",
        "piper_b",
        target=_transit(
            _ref("part", part="brick-mid"),
            _ref("connection", connection="conn-top-mid"),
            carrying="brick-top",
        ),
        resources=["robot:piper_b", "part:brick-top", "part:brick-mid"],
    )
    third = _operation(
        "b-place",
        "task-top",
        "PLACE_DOWN",
        "piper_b",
        target=_place("conn-top-mid", "brick-top", "brick-mid"),
        resources=[
            "robot:piper_b",
            "connection:conn-top-mid",
            "part:brick-top",
            "part:brick-mid",
        ],
    )
    raw = _plan(
        [first, second, third],
        [["a-pick", "b-transit"], ["b-place"]],
        _schedule(
            (
                {"operation_id": "a-pick", "robot_id": "piper_a"},
                {"operation_id": "b-transit", "robot_id": "piper_b"},
            ),
            ({"operation_id": "b-place", "robot_id": "piper_b"},),
        ),
    )
    with pytest.raises(RobotOperationPlanError, match="resource conflict"):
        parse_robot_operation_plan(raw, source_plan=_synthetic_stap())


def test_rejects_layer_coverage_gap() -> None:
    raw = _default_plan(
        layers=(("a-transit",), ("a-pick",), ("a-place",), ("b-transit",))
    )
    with pytest.raises(RobotOperationPlanError, match="execution_layers"):
        parse_robot_operation_plan(raw, source_plan=_synthetic_stap())


def test_rejects_operation_listed_twice_in_layers() -> None:
    raw = _default_plan(
        layers=(
            ("a-transit",),
            ("a-transit", "a-pick", "a-place", "b-transit", "b-place"),
        )
    )
    with pytest.raises(RobotOperationPlanError, match="duplicated"):
        parse_robot_operation_plan(raw, source_plan=_synthetic_stap())


def test_rejects_schedule_coverage_gap() -> None:
    raw = _default_plan(schedule=_DEFAULT_SCHEDULE[:4])
    with pytest.raises(RobotOperationPlanError, match="planned_schedule"):
        parse_robot_operation_plan(raw, source_plan=_synthetic_stap())


def test_rejects_schedule_robot_mismatch() -> None:
    raw = _default_plan()
    raw["planned_schedule"][0] = [{"operation_id": "a-transit", "robot_id": "piper_b"}]
    with pytest.raises(RobotOperationPlanError, match="executor"):
        parse_robot_operation_plan(raw, source_plan=_synthetic_stap())


def test_rejects_unknown_dependency_operation() -> None:
    raw = _default_plan()
    raw["dependencies"][0]["successor_operation_id"] = "absent"
    with pytest.raises(RobotOperationPlanError, match="dependency"):
        parse_robot_operation_plan(raw, source_plan=_synthetic_stap())


def test_rejects_dependency_without_layer_order() -> None:
    raw = _default_plan(
        layers=(("a-transit", "a-pick"), ("a-place",), ("b-transit",), ("b-place",))
    )
    with pytest.raises(RobotOperationPlanError, match="order"):
        parse_robot_operation_plan(raw, source_plan=_synthetic_stap())


def test_rejects_dependency_without_schedule_slot_order() -> None:
    raw = _default_plan(
        schedule=_schedule(
            ({"operation_id": "a-pick", "robot_id": "piper_a"},),
            ({"operation_id": "a-transit", "robot_id": "piper_a"},),
            ({"operation_id": "a-place", "robot_id": "piper_a"},),
            ({"operation_id": "b-transit", "robot_id": "piper_b"},),
            ({"operation_id": "b-place", "robot_id": "piper_b"},),
        )
    )
    with pytest.raises(RobotOperationPlanError, match="order"):
        parse_robot_operation_plan(raw, source_plan=_synthetic_stap())


def test_rejects_evidence_label_not_declared_by_operation() -> None:
    raw = _default_plan(
        evidence=(),
    )
    raw["evidence_requirements"] = [
        {"operation_id": "a-pick", "evidence": "holding_confirmed"},
    ]
    parsed = parse_robot_operation_plan(raw, source_plan=_synthetic_stap())
    assert parsed.evidence_requirements == (("a-pick", "holding_confirmed"),)
    raw["evidence_requirements"] = [
        {"operation_id": "a-pick", "evidence": "never_declared"},
    ]
    with pytest.raises(RobotOperationPlanError, match="declared"):
        parse_robot_operation_plan(raw, source_plan=_synthetic_stap())


def test_rejects_evidence_for_unknown_operation() -> None:
    raw = _default_plan()
    raw["evidence_requirements"] = [{"operation_id": "absent", "evidence": "x"}]
    with pytest.raises(RobotOperationPlanError, match="unknown"):
        parse_robot_operation_plan(raw)


# ---------------------------------------------------------------------------
# source_plan typing and immutability of failing input.
# ---------------------------------------------------------------------------


def test_rejects_a_source_plan_of_the_wrong_type() -> None:
    with pytest.raises(RobotOperationPlanError, match="source_plan"):
        parse_robot_operation_plan(
            _default_plan(), source_plan="assembly-plan-synthetic"
        )


def test_rejects_a_malformed_raw_source_plan() -> None:
    with pytest.raises(RobotOperationPlanError, match="assembly-plan/v2"):
        parse_robot_operation_plan(
            _default_plan(), source_plan={"schema": "assembly-plan/v2"}
        )


def test_rejected_input_is_never_mutated_or_reordered() -> None:
    raw = _default_plan()
    raw["operations"][2]["target"]["lower_physical_instance_id"] = "brick-top"
    raw["operations"][2]["resources"] = [
        "robot:piper_a",
        "connection:conn-mid-base",
        "part:brick-mid",
        "part:brick-top",
    ]
    before = copy.deepcopy(raw)
    with pytest.raises(RobotOperationPlanError):
        parse_robot_operation_plan(raw, source_plan=_synthetic_stap())
    assert raw == before
