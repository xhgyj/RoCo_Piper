"""Strict versioned input validation without plan repair or inference."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from rocobrick.capabilities import CAPABILITIES
from rocobrick.models import AssemblyPlan, PartInstance, Pose, SceneManifest, SkillType, TaskDefinition

PLAN_SCHEMA = "assembly-plan/v2"
SCENE_SCHEMA = "assembly-scene/v1"


class ContractError(ValueError):
    """An input is rejected before any physical motion occurs."""


def _object(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{path} must be an object")
    return value


def _array(value: object, path: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ContractError(f"{path} must be an array")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{path} must be a non-empty string")
    return value


def _pose(value: object, path: str, expected_frame: str) -> Pose:
    raw = _object(value, path)
    frame = _string(raw.get("frame_id"), f"{path}.frame_id")
    if frame != expected_frame:
        raise ContractError(f"{path}.frame_id must equal {expected_frame!r}")
    position = tuple(_array(raw.get("position_m"), f"{path}.position_m"))
    orientation = tuple(_array(raw.get("orientation_xyzw"), f"{path}.orientation_xyzw"))
    if len(position) != 3 or not all(type(item) in (int, float) and math.isfinite(item) for item in position):
        raise ContractError(f"{path}.position_m must contain three finite numbers")
    if len(orientation) != 4 or not all(type(item) in (int, float) and math.isfinite(item) for item in orientation):
        raise ContractError(f"{path}.orientation_xyzw must contain four finite numbers")
    norm = math.sqrt(sum(float(item) ** 2 for item in orientation))
    if not math.isclose(norm, 1.0, abs_tol=1e-4):
        raise ContractError(f"{path}.orientation_xyzw must be normalized")
    return Pose(frame, tuple(map(float, position)), tuple(map(float, orientation)))


def _check_dag(tasks: Mapping[str, TaskDefinition]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visiting:
            raise ContractError(f"dependencies contain a cycle at {task_id}")
        if task_id in visited:
            return
        visiting.add(task_id)
        for predecessor in tasks[task_id].predecessors:
            if predecessor not in tasks:
                raise ContractError(f"task {task_id} references unknown predecessor {predecessor}")
            visit(predecessor)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in tasks:
        visit(task_id)


def parse_plan(payload: object) -> AssemblyPlan:
    raw = _object(payload, "plan")
    schema = raw.get("schema")
    if schema != PLAN_SCHEMA:
        raise ContractError(f"schema {schema!r} is unsupported; expected {PLAN_SCHEMA}; automatic conversion is disabled")
    units = _object(raw.get("units"), "plan.units")
    if units.get("length") != "m" or units.get("angle") != "rad":
        raise ContractError("plan.units must be {'length': 'm', 'angle': 'rad'}")
    frame = _string(raw.get("frame_id"), "plan.frame_id")
    robots = {_string(item, "plan.robots[]") for item in _array(raw.get("robots"), "plan.robots")}
    if not robots:
        raise ContractError("plan.robots must not be empty")
    tasks: dict[str, TaskDefinition] = {}
    instances: set[str] = set()
    for index, value in enumerate(_array(raw.get("tasks"), "plan.tasks")):
        item = _object(value, f"plan.tasks[{index}]")
        task_id = _string(item.get("task_id"), f"plan.tasks[{index}].task_id")
        instance = _string(item.get("part_instance_id"), f"plan.tasks[{index}].part_instance_id")
        if task_id in tasks:
            raise ContractError(f"duplicate task_id {task_id}")
        if instance in instances:
            raise ContractError(f"duplicate part_instance_id {instance}")
        try:
            skill = SkillType(_string(item.get("skill_type"), f"plan.tasks[{index}].skill_type"))
        except ValueError as exc:
            raise ContractError(f"plan.tasks[{index}].skill_type is unknown") from exc
        if not CAPABILITIES[skill].available:
            raise ContractError(f"CAPABILITY_UNAVAILABLE: {skill}")
        primary = _string(item.get("primary_robot"), f"plan.tasks[{index}].primary_robot")
        supporting = item.get("supporting_robot")
        if primary not in robots or (supporting is not None and supporting not in robots):
            raise ContractError(f"task {task_id} references an unknown robot")
        if supporting == primary:
            raise ContractError(f"task {task_id} cannot use the same primary and supporting robot")
        brick_id = item.get("brick_id")
        level, slot = item.get("level"), item.get("schedule_slot")
        if type(brick_id) is not int or brick_id < 0:
            raise ContractError(f"task {task_id}.brick_id must be a non-negative integer")
        if type(level) is not int or level < 0 or type(slot) is not int or slot < 0:
            raise ContractError(f"task {task_id} level and schedule_slot must be non-negative integers")
        predecessors = frozenset(_string(p, f"task {task_id}.predecessors[]") for p in _array(item.get("predecessors", []), f"task {task_id}.predecessors"))
        task = TaskDefinition(
            task_id=task_id,
            part_instance_id=instance,
            brick_id=brick_id,
            target=_pose(item.get("target"), f"task {task_id}.target", frame),
            target_interface=_string(item.get("target_interface"), f"task {task_id}.target_interface"),
            skill=skill,
            primary_robot=primary,
            supporting_robot=supporting,
            predecessors=predecessors,
            level=level,
            schedule_slot=slot,
        )
        tasks[task_id] = task
        instances.add(instance)
    if not tasks:
        raise ContractError("plan.tasks must not be empty")
    _check_dag(tasks)
    for task in tasks.values():
        for predecessor in task.predecessors:
            before = tasks[predecessor]
            if before.level >= task.level:
                raise ContractError(f"level violates dependency {predecessor} -> {task.task_id}")
            if before.schedule_slot >= task.schedule_slot:
                raise ContractError(f"schedule_slot violates dependency {predecessor} -> {task.task_id}")
    return AssemblyPlan(
        plan_id=_string(raw.get("plan_id"), "plan.plan_id"),
        frame_id=frame,
        brick_library_version=_string(raw.get("brick_library_version"), "plan.brick_library_version"),
        robot_asset_version=_string(raw.get("robot_asset_version"), "plan.robot_asset_version"),
        tasks=tasks,
    )


def parse_scene(payload: object) -> SceneManifest:
    raw = _object(payload, "scene")
    if raw.get("schema") != SCENE_SCHEMA:
        raise ContractError(f"scene.schema must be {SCENE_SCHEMA}")
    frame = _string(raw.get("frame_id"), "scene.frame_id")
    parts: dict[str, PartInstance] = {}
    for index, value in enumerate(_array(raw.get("parts"), "scene.parts")):
        item = _object(value, f"scene.parts[{index}]")
        instance = _string(item.get("part_instance_id"), f"scene.parts[{index}].part_instance_id")
        brick_id = item.get("brick_id")
        if instance in parts:
            raise ContractError(f"duplicate scene part_instance_id {instance}")
        if type(brick_id) is not int or brick_id < 0:
            raise ContractError(f"scene part {instance}.brick_id must be a non-negative integer")
        parts[instance] = PartInstance(instance, brick_id, _pose(item.get("pose"), f"scene part {instance}.pose", frame))
    return SceneManifest(_string(raw.get("scene_id"), "scene.scene_id"), frame, parts)


def validate_binding(plan: AssemblyPlan, scene: SceneManifest) -> None:
    if plan.frame_id != scene.frame_id:
        raise ContractError("plan and scene frame_id differ")
    for task in plan.tasks.values():
        part = scene.parts.get(task.part_instance_id)
        if part is None:
            raise ContractError(f"scene is missing part_instance_id {task.part_instance_id}")
        if part.brick_id != task.brick_id:
            raise ContractError(f"brick_id mismatch for {task.part_instance_id}")


def load_and_validate_inputs(plan_path: Path, scene_path: Path) -> tuple[AssemblyPlan, SceneManifest]:
    with plan_path.open(encoding="utf-8") as stream:
        plan = parse_plan(json.load(stream))
    with scene_path.open(encoding="utf-8") as stream:
        scene = parse_scene(json.load(stream))
    validate_binding(plan, scene)
    return plan, scene
