"""Compile the structural task schedule without invoking runtime components."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rocobrick.stap_contract import StapPlan, StructuralTask


class ScheduleContractError(ValueError):
    """Raised when a STAP schedule does not satisfy its contract."""


@dataclass(frozen=True)
class ScheduledTask:
    """One task at its validated layer and schedule positions."""

    layer_index: int
    schedule_index: int
    task_id: str
    robot_id: str


def _mapping(value: object, path: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise ScheduleContractError(f"{path} must be a mapping")
    return value


def _sequence(value: object, path: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ScheduleContractError(f"{path} must be an array")
    return value


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ScheduleContractError(f"{path} must be a non-empty string")
    return value


def _index(value: object, path: str) -> int:
    if type(value) is not int or value < 0:
        raise ScheduleContractError(f"{path} must be a non-negative integer")
    return value


def _required(raw: Mapping[object, object], name: str) -> object:
    if name not in raw:
        raise ScheduleContractError(f"plan.raw.{name} is required")
    return raw[name]


def _validate_tasks(plan: StapPlan) -> dict[str, StructuralTask]:
    raw_tasks = _mapping(plan.tasks, "plan.tasks")
    tasks: dict[str, StructuralTask] = {}
    for key, value in raw_tasks.items():
        task_id = _text(key, "plan.tasks key")
        if task_id in tasks:
            raise ScheduleContractError(f"duplicate task key {task_id!r}")
        if not isinstance(value, StructuralTask):
            raise ScheduleContractError(f"plan.tasks[{task_id!r}] must be a StructuralTask")
        declared_id = _text(value.task_id, f"task {task_id}.task_id")
        if declared_id != task_id:
            raise ScheduleContractError(f"task key {task_id!r} does not match task_id")
        _text(value.primary_robot, f"task {task_id}.primary_robot")
        _index(value.level, f"task {task_id}.level")
        _index(value.schedule_slot, f"task {task_id}.schedule_slot")
        tasks[task_id] = value
    return tasks


def _validate_allocation(
    value: object,
    tasks: Mapping[str, StructuralTask],
) -> None:
    allocation = _mapping(value, "plan.raw.allocation")
    task_ids = set(tasks)
    assigned: set[str] = set()
    for raw_robot_id, raw_task_ids in allocation.items():
        robot_id = _text(raw_robot_id, "plan.raw.allocation key")
        task_list = _sequence(raw_task_ids, f"plan.raw.allocation[{robot_id!r}]")
        for position, raw_task_id in enumerate(task_list):
            task_id = _text(
                raw_task_id,
                f"plan.raw.allocation[{robot_id!r}][{position}]",
            )
            if task_id not in tasks:
                raise ScheduleContractError(f"allocation contains unknown task {task_id!r}")
            if task_id in assigned:
                raise ScheduleContractError(f"allocation contains duplicate task {task_id!r}")
            if tasks[task_id].primary_robot != robot_id:
                raise ScheduleContractError(
                    f"allocation robot for {task_id!r} does not match primary_robot"
                )
            assigned.add(task_id)
    if assigned != task_ids:
        raise ScheduleContractError("allocation must contain every task exactly once")


def _validate_layers(
    value: object,
    tasks: Mapping[str, StructuralTask],
) -> tuple[tuple[str, ...], ...]:
    raw_layers = _sequence(value, "plan.raw.execution_layers")
    layers: list[tuple[str, ...]] = []
    seen: set[str] = set()
    for layer_index, raw_layer in enumerate(raw_layers):
        layer = _sequence(raw_layer, f"plan.raw.execution_layers[{layer_index}]")
        layer_tasks: list[str] = []
        for position, raw_task_id in enumerate(layer):
            task_id = _text(
                raw_task_id,
                f"plan.raw.execution_layers[{layer_index}][{position}]",
            )
            if task_id not in tasks:
                raise ScheduleContractError(
                    f"execution_layers contains unknown task {task_id!r}"
                )
            if task_id in seen:
                raise ScheduleContractError(
                    f"execution_layers contains duplicate task {task_id!r}"
                )
            if tasks[task_id].level != layer_index:
                raise ScheduleContractError(
                    f"task {task_id!r} has a level different from its layer"
                )
            seen.add(task_id)
            layer_tasks.append(task_id)
        layers.append(tuple(layer_tasks))
    if seen != set(tasks):
        raise ScheduleContractError(
            "execution_layers must contain every task exactly once"
        )
    return tuple(layers)


def _validate_schedule(
    value: object,
    tasks: Mapping[str, StructuralTask],
) -> tuple[tuple[tuple[str, str], ...], ...]:
    raw_schedule = _sequence(value, "plan.raw.planned_schedule")
    schedule: list[tuple[tuple[str, str], ...]] = []
    seen: set[str] = set()
    expected_keys = {"robot", "task"}
    for schedule_index, raw_slot in enumerate(raw_schedule):
        slot = _sequence(raw_slot, f"plan.raw.planned_schedule[{schedule_index}]")
        seen_robots: set[str] = set()
        slot_tasks: list[tuple[str, str]] = []
        for position, raw_entry in enumerate(slot):
            entry = _mapping(
                raw_entry,
                f"plan.raw.planned_schedule[{schedule_index}][{position}]",
            )
            if set(entry) != expected_keys:
                raise ScheduleContractError(
                    f"planned_schedule entry at slot {schedule_index} has invalid keys"
                )
            task_id = _text(
                entry["task"],
                f"plan.raw.planned_schedule[{schedule_index}][{position}].task",
            )
            robot_id = _text(
                entry["robot"],
                f"plan.raw.planned_schedule[{schedule_index}][{position}].robot",
            )
            if task_id not in tasks:
                raise ScheduleContractError(
                    f"planned_schedule contains unknown task {task_id!r}"
                )
            if task_id in seen:
                raise ScheduleContractError(
                    f"planned_schedule contains duplicate task {task_id!r}"
                )
            if tasks[task_id].primary_robot != robot_id:
                raise ScheduleContractError(
                    f"schedule robot for {task_id!r} does not match primary_robot"
                )
            if tasks[task_id].schedule_slot != schedule_index:
                raise ScheduleContractError(
                    f"task {task_id!r} has a schedule slot different from its slot"
                )
            if robot_id in seen_robots:
                raise ScheduleContractError(
                    f"schedule slot {schedule_index} uses robot {robot_id!r} more than once"
                )
            seen_robots.add(robot_id)
            seen.add(task_id)
            slot_tasks.append((task_id, robot_id))
        schedule.append(tuple(slot_tasks))
    if seen != set(tasks):
        raise ScheduleContractError(
            "planned_schedule must contain every task exactly once"
        )
    return tuple(schedule)


def _compile(plan: StapPlan) -> tuple[ScheduledTask, ...]:
    tasks = _validate_tasks(plan)
    raw = _mapping(plan.raw, "plan.raw")
    _validate_allocation(_required(raw, "allocation"), tasks)
    layers = _validate_layers(_required(raw, "execution_layers"), tasks)
    schedule = _validate_schedule(_required(raw, "planned_schedule"), tasks)

    compiled: list[ScheduledTask] = []
    for layer_index, layer in enumerate(layers):
        layer_tasks = set(layer)
        for schedule_index, slot in enumerate(schedule):
            for task_id, robot_id in slot:
                if task_id in layer_tasks:
                    compiled.append(
                        ScheduledTask(
                            layer_index=layer_index,
                            schedule_index=schedule_index,
                            task_id=task_id,
                            robot_id=robot_id,
                        )
                    )
    return tuple(compiled)


def compile_stap_schedule(plan: StapPlan) -> tuple[ScheduledTask, ...]:
    """Validate and compile a STAP plan's allocation and schedule data."""

    try:
        if not isinstance(plan, StapPlan):
            raise ScheduleContractError("plan must be a StapPlan")
        return _compile(plan)
    except ScheduleContractError:
        raise
    except Exception as exc:
        raise ScheduleContractError("invalid STAP schedule contract") from exc
