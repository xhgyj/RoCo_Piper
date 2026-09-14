"""Strict ingress contract for structural plans emitted by H-LTL STAP.

This wire contract deliberately carries grid assembly facts, not simulator or
motion geometry.  Conversion to calibrated metric poses belongs to the scene
adapter and is recorded as run evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any

from rocobrick.models import SkillType


STAP_PLAN_SCHEMA = "assembly-plan/v2"


class StapContractError(ValueError):
    pass


@dataclass(frozen=True)
class StructuralPart:
    physical_instance_id: str
    source_goal_key: str
    source_start_key: str | None
    brick_id: int
    x: int
    y: int
    z: int
    orientation_quarter_turns: int
    color: str


@dataclass(frozen=True)
class StructuralTask:
    task_id: str
    physical_instance_id: str
    skill: SkillType
    primary_robot: str
    supporting_robot: str | None
    predecessors: frozenset[str]
    initial_supports: frozenset[str]
    level: int
    schedule_slot: int


@dataclass(frozen=True)
class StapPlan:
    plan_id: str
    robots: tuple[str, ...]
    source_goal_sha256: str
    source_start_sha256: str | None
    goal_parts: Mapping[str, StructuralPart]
    initially_assembled: frozenset[str]
    tasks: Mapping[str, StructuralTask]
    hltl_hierarchy: tuple[Mapping[str, str], ...]
    hltl_root_formula: str
    raw: Mapping[str, object]


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise StapContractError(f"{path} must be an object")
    return value


def _array(value: object, path: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise StapContractError(f"{path} must be an array")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise StapContractError(f"{path} must be a non-empty string")
    return value


def _sha(value: object, path: str) -> str:
    result = _string(value, path)
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise StapContractError(f"{path} must be a lowercase SHA-256 digest")
    return result


def _part(value: object, path: str) -> StructuralPart:
    item = _mapping(value, path)
    values = [item.get(name) for name in ("brick_id", "x", "y", "z", "ori")]
    if not all(type(value) is int for value in values):
        raise StapContractError(f"{path} brick_id/x/y/z/ori must be integers")
    brick_id, x, y, z, orientation = values
    if brick_id < 0 or z < 0 or orientation not in (0, 1):
        raise StapContractError(f"{path} has invalid brick_id, z, or ori")
    start_key = item.get("source_start_key")
    if start_key is not None:
        start_key = _string(start_key, f"{path}.source_start_key")
    return StructuralPart(
        physical_instance_id=_string(item.get("physical_instance_id"), f"{path}.physical_instance_id"),
        source_goal_key=_string(item.get("source_goal_key"), f"{path}.source_goal_key"),
        source_start_key=start_key,
        brick_id=brick_id,
        x=x,
        y=y,
        z=z,
        orientation_quarter_turns=orientation,
        color=_string(item.get("color"), f"{path}.color"),
    )


def _assert_exact(items: list[str], expected: set[str], path: str) -> None:
    if len(items) != len(expected) or set(items) != expected:
        raise StapContractError(f"{path} must contain every task exactly once")


def parse_stap_plan(payload: object) -> StapPlan:
    root = _mapping(payload, "plan")
    if root.get("schema") != STAP_PLAN_SCHEMA:
        raise StapContractError(
            f"schema {root.get('schema')!r} is unsupported; expected {STAP_PLAN_SCHEMA}"
        )
    plan_material = {
        "source": root.get("source"),
        "robots": root.get("robots"),
        "parts": root.get("parts"),
        "dependencies": root.get("dependencies"),
        "initial_dependencies": root.get("initial_dependencies"),
        "allocation": root.get("allocation"),
        "execution_layers": root.get("execution_layers"),
        "planned_schedule": root.get("planned_schedule"),
        "hierarchy": _mapping(root.get("specification"), "plan.specification").get("hierarchy"),
    }
    expected_plan_id = "assembly-plan-" + sha256(
        json.dumps(
            plan_material, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()[:24]
    if root.get("plan_id") != expected_plan_id:
        raise StapContractError("plan_id does not match the canonical plan contents")
    semantics = _mapping(root.get("structural_semantics"), "plan.structural_semantics")
    orientation = _mapping(semantics.get("orientation"), "plan.structural_semantics.orientation")
    if (
        semantics.get("coordinate_system") != "grid"
        or semantics.get("motion_geometry") != "omitted"
        or orientation.get("encoding") != "quarter_turn"
        or list(_array(orientation.get("values"), "orientation.values")) != [0, 1]
    ):
        raise StapContractError("unsupported structural semantics")

    source = _mapping(root.get("source"), "plan.source")
    robots = tuple(
        _string(_mapping(item, "plan.robots[]").get("robot_id"), "plan.robots[].robot_id")
        for item in _array(root.get("robots"), "plan.robots")
    )
    if not robots or len(robots) != len(set(robots)):
        raise StapContractError("plan.robots must contain unique robot ids")

    goal = _mapping(root.get("goal_structure"), "plan.goal_structure")
    goal_parts: dict[str, StructuralPart] = {}
    goal_keys: set[str] = set()
    for index, value in enumerate(_array(goal.get("instances"), "plan.goal_structure.instances")):
        part = _part(value, f"plan.goal_structure.instances[{index}]")
        if part.physical_instance_id in goal_parts or part.source_goal_key in goal_keys:
            raise StapContractError("goal structure has duplicate physical id or source key")
        goal_parts[part.physical_instance_id] = part
        goal_keys.add(part.source_goal_key)
    if not goal_parts:
        raise StapContractError("goal structure must not be empty")

    initial = _mapping(root.get("initial_structure"), "plan.initial_structure")
    initially_assembled: set[str] = set()
    for index, value in enumerate(_array(initial.get("facts"), "plan.initial_structure.facts")):
        fact = _mapping(value, f"plan.initial_structure.facts[{index}]")
        instance = _string(fact.get("physical_instance_id"), f"initial fact {index}.physical_instance_id")
        if fact.get("predicate") != "assembled" or instance not in goal_parts or instance in initially_assembled:
            raise StapContractError(f"invalid initial assembled fact for {instance}")
        if goal_parts[instance].source_start_key is None:
            raise StapContractError(f"initial part {instance} lacks source_start_key")
        if fact.get("source_start_key") != goal_parts[instance].source_start_key:
            raise StapContractError(f"initial source key mismatch for {instance}")
        if fact.get("fact_id") != f"assembled:{instance}":
            raise StapContractError(f"invalid initial fact id for {instance}")
        initially_assembled.add(instance)

    task_rows = list(_array(root.get("parts"), "plan.parts"))
    task_data: dict[str, tuple[Mapping[str, object], StructuralPart]] = {}
    moving_instances: set[str] = set()
    for index, value in enumerate(task_rows):
        row = _mapping(value, f"plan.parts[{index}]")
        task_id = _string(row.get("task_id"), f"plan.parts[{index}].task_id")
        part = _part(row, f"plan.parts[{index}]")
        canonical = goal_parts.get(part.physical_instance_id)
        if canonical is None or canonical != part:
            raise StapContractError(f"task {task_id} does not exactly match its goal instance")
        if task_id in task_data or part.physical_instance_id in moving_instances:
            raise StapContractError("tasks have duplicate ids or physical instances")
        if part.physical_instance_id in initially_assembled:
            raise StapContractError(f"initially assembled part cannot be task {task_id}")
        task_data[task_id] = (row, part)
        moving_instances.add(part.physical_instance_id)
    if moving_instances | initially_assembled != set(goal_parts):
        raise StapContractError("initial facts and tasks must partition the goal structure")

    dependency_rows = _array(root.get("dependencies"), "plan.dependencies")
    predecessors = {task_id: set() for task_id in task_data}
    edges: list[tuple[str, str]] = []
    for index, value in enumerate(dependency_rows):
        pair = list(_array(value, f"plan.dependencies[{index}]"))
        if len(pair) != 2 or not all(isinstance(item, str) for item in pair):
            raise StapContractError(f"plan.dependencies[{index}] must be a task pair")
        before, after = pair
        if before not in task_data or after not in task_data or before == after:
            raise StapContractError(f"invalid task dependency {before!r} -> {after!r}")
        predecessors[after].add(before)
        edges.append((before, after))

    initial_supports = {task_id: set() for task_id in task_data}
    for index, value in enumerate(_array(root.get("initial_dependencies", []), "plan.initial_dependencies")):
        pair = list(_array(value, f"plan.initial_dependencies[{index}]"))
        if len(pair) != 2 or pair[0] not in initially_assembled or pair[1] not in task_data:
            raise StapContractError(f"invalid initial dependency at index {index}")
        initial_supports[pair[1]].add(pair[0])

    allocation = _mapping(root.get("allocation"), "plan.allocation")
    arm_by_task: dict[str, str] = {}
    assigned: list[str] = []
    if set(allocation) != set(robots):
        raise StapContractError("allocation keys must exactly equal plan.robots")
    for robot in robots:
        for task_id in _array(allocation[robot], f"plan.allocation.{robot}"):
            task_id = _string(task_id, f"plan.allocation.{robot}[]")
            assigned.append(task_id)
            arm_by_task[task_id] = robot
    _assert_exact(assigned, set(task_data), "plan.allocation")

    level_of: dict[str, int] = {}
    layered: list[str] = []
    for level, value in enumerate(_array(root.get("execution_layers"), "plan.execution_layers")):
        for task_id in _array(value, f"plan.execution_layers[{level}]"):
            task_id = _string(task_id, f"plan.execution_layers[{level}][]")
            level_of[task_id] = level
            layered.append(task_id)
    _assert_exact(layered, set(task_data), "plan.execution_layers")

    slot_of: dict[str, int] = {}
    scheduled: list[str] = []
    for slot, value in enumerate(_array(root.get("planned_schedule"), "plan.planned_schedule")):
        used: set[str] = set()
        for raw_job in _array(value, f"plan.planned_schedule[{slot}]"):
            job = _mapping(raw_job, f"plan.planned_schedule[{slot}][]")
            task_id = _string(job.get("task"), "schedule.task")
            robot = _string(job.get("robot"), "schedule.robot")
            if robot in used or arm_by_task.get(task_id) != robot:
                raise StapContractError(f"invalid schedule assignment for {task_id}")
            used.add(robot)
            slot_of[task_id] = slot
            scheduled.append(task_id)
    _assert_exact(scheduled, set(task_data), "plan.planned_schedule")

    for before, after in edges:
        if level_of[before] >= level_of[after] or slot_of[before] >= slot_of[after]:
            raise StapContractError(f"ordering violates dependency {before} -> {after}")

    specification = _mapping(root.get("specification"), "plan.specification")
    hierarchy: list[Mapping[str, str]] = []
    for level, value in enumerate(_array(specification.get("hierarchy"), "plan.specification.hierarchy")):
        raw_level = _mapping(value, f"plan.specification.hierarchy[{level}]")
        hierarchy.append({
            _string(key, f"hierarchy[{level}] key"): _string(formula, f"hierarchy[{level}].{key}")
            for key, formula in raw_level.items()
        })
    if specification.get("type") != "hierarchical-ltl" or not hierarchy:
        raise StapContractError("specification must contain a non-empty hierarchical-ltl hierarchy")
    root_formula = _string(specification.get("root_formula"), "plan.specification.root_formula")
    if len(hierarchy[0]) != 1 or root_formula != next(iter(hierarchy[0].values())):
        raise StapContractError("root_formula must exactly preserve the hierarchy root")

    tasks: dict[str, StructuralTask] = {}
    for task_id, (row, _) in task_data.items():
        try:
            skill = SkillType(_string(row.get("skill_type"), f"task {task_id}.skill_type"))
        except ValueError as exc:
            raise StapContractError(f"task {task_id} has unknown skill") from exc
        primary = _string(row.get("primary_robot"), f"task {task_id}.primary_robot")
        supporting = row.get("supporting_robot")
        if supporting is not None:
            supporting = _string(supporting, f"task {task_id}.supporting_robot")
        if skill is not SkillType.PLACE_DOWN or primary != arm_by_task[task_id]:
            raise StapContractError(f"task {task_id} capability or allocation mismatch")
        tasks[task_id] = StructuralTask(
            task_id, row["physical_instance_id"], skill, primary, supporting,
            frozenset(predecessors[task_id]), frozenset(initial_supports[task_id]),
            level_of[task_id], slot_of[task_id],
        )

    return StapPlan(
        plan_id=_string(root.get("plan_id"), "plan.plan_id"),
        robots=robots,
        source_goal_sha256=_sha(source.get("goal_sha256"), "plan.source.goal_sha256"),
        source_start_sha256=(
            None if source.get("start_sha256") is None
            else _sha(source.get("start_sha256"), "plan.source.start_sha256")
        ),
        goal_parts=goal_parts,
        initially_assembled=frozenset(initially_assembled),
        tasks=tasks,
        hltl_hierarchy=tuple(hierarchy),
        hltl_root_formula=root_formula,
        raw=dict(root),
    )
