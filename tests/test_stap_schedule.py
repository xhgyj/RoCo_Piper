from __future__ import annotations

import ast
import copy
from dataclasses import FrozenInstanceError, replace
import hashlib
import inspect
import json

import pytest

from rocobrick.models import SkillType
from rocobrick.stap_contract import StapPlan, StructuralTask
import rocobrick.stap_schedule as stap_schedule
from rocobrick.stap_schedule import (
    ScheduleContractError,
    ScheduledTask,
    compile_stap_schedule,
)


def _task(task_id: str, robot_id: str, level: int, schedule_slot: int) -> StructuralTask:
    return StructuralTask(
        task_id=task_id,
        physical_instance_id=f"part-{task_id}",
        skill=SkillType.PLACE_DOWN,
        primary_robot=robot_id,
        supporting_robot=None,
        predecessors=frozenset(),
        initial_supports=frozenset(),
        level=level,
        schedule_slot=schedule_slot,
    )


def _plan(
    task_specs: tuple[tuple[str, str, int, int], ...] = (("task", "robot", 0, 0),),
    *,
    allocation: object | None = None,
    execution_layers: object | None = None,
    planned_schedule: object | None = None,
) -> StapPlan:
    tasks = {
        task_id: _task(task_id, robot_id, level, schedule_slot)
        for task_id, robot_id, level, schedule_slot in task_specs
    }
    if allocation is None:
        allocation_map: dict[str, list[str]] = {}
        for task_id, robot_id, _, _ in task_specs:
            allocation_map.setdefault(robot_id, []).append(task_id)
        allocation = allocation_map
    if execution_layers is None:
        layers: dict[int, list[str]] = {}
        for task_id, _, level, _ in task_specs:
            layers.setdefault(level, []).append(task_id)
        execution_layers = [layers[index] for index in range(max(layers, default=-1) + 1)]
    if planned_schedule is None:
        slots: dict[int, list[dict[str, str]]] = {}
        for task_id, robot_id, _, schedule_slot in task_specs:
            slots.setdefault(schedule_slot, []).append({"task": task_id, "robot": robot_id})
        planned_schedule = [slots[index] for index in range(max(slots, default=-1) + 1)]
    return StapPlan(
        plan_id="test-plan",
        robots=tuple(dict.fromkeys(robot_id for _, robot_id, _, _ in task_specs)),
        source_goal_sha256="a" * 64,
        source_start_sha256=None,
        goal_parts={},
        initially_assembled=frozenset(),
        tasks=tasks,
        hltl_hierarchy=(),
        hltl_root_formula="",
        raw={
            "allocation": allocation,
            "execution_layers": execution_layers,
            "planned_schedule": planned_schedule,
        },
    )


def _with_raw(plan: StapPlan, **changes: object) -> StapPlan:
    raw = copy.deepcopy(plan.raw)
    raw.update(changes)
    return replace(plan, raw=raw)


def _without_raw(plan: StapPlan, name: str) -> StapPlan:
    raw = copy.deepcopy(plan.raw)
    raw.pop(name)
    return replace(plan, raw=raw)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def test_compiles_one_task():
    plan = _plan()

    assert compile_stap_schedule(plan) == (
        ScheduledTask(layer_index=0, schedule_index=0, task_id="task", robot_id="robot"),
    )


def test_compiles_multiple_layers():
    plan = _plan(
        (("base", "robot-a", 0, 0), ("top", "robot-b", 1, 1)),
        allocation={"robot-a": ["base"], "robot-b": ["top"]},
        execution_layers=[["base"], ["top"]],
        planned_schedule=[
            [{"task": "base", "robot": "robot-a"}],
            [{"task": "top", "robot": "robot-b"}],
        ],
    )

    assert compile_stap_schedule(plan) == (
        ScheduledTask(0, 0, "base", "robot-a"),
        ScheduledTask(1, 1, "top", "robot-b"),
    )


def test_same_layer_same_slot_keeps_slot_entry_order():
    plan = _plan(
        (("first", "robot-a", 0, 0), ("second", "robot-b", 0, 0)),
        allocation={"robot-a": ["first"], "robot-b": ["second"]},
        execution_layers=[["second", "first"]],
        planned_schedule=[
            [
                {"task": "first", "robot": "robot-a"},
                {"task": "second", "robot": "robot-b"},
            ]
        ],
    )

    assert [item.task_id for item in compile_stap_schedule(plan)] == ["first", "second"]


def test_same_layer_across_slots_keeps_slot_order():
    plan = _plan(
        (("early", "robot-a", 0, 0), ("late", "robot-b", 0, 2)),
        allocation={"robot-a": ["early"], "robot-b": ["late"]},
        execution_layers=[["late", "early"]],
        planned_schedule=[
            [{"task": "early", "robot": "robot-a"}],
            [],
            [{"task": "late", "robot": "robot-b"}],
        ],
    )

    assert compile_stap_schedule(plan) == (
        ScheduledTask(0, 0, "early", "robot-a"),
        ScheduledTask(0, 2, "late", "robot-b"),
    )


def test_layers_are_outer_order_when_layers_and_slots_interleave():
    plan = _plan(
        (("layer-zero-late", "robot-a", 0, 2), ("layer-zero-early", "robot-b", 0, 0), ("layer-one", "robot-c", 1, 1)),
        allocation={
            "robot-a": ["layer-zero-late"],
            "robot-b": ["layer-zero-early"],
            "robot-c": ["layer-one"],
        },
        execution_layers=[["layer-zero-late", "layer-zero-early"], ["layer-one"]],
        planned_schedule=[
            [{"task": "layer-zero-early", "robot": "robot-b"}],
            [{"task": "layer-one", "robot": "robot-c"}],
            [{"task": "layer-zero-late", "robot": "robot-a"}],
        ],
    )

    assert compile_stap_schedule(plan) == (
        ScheduledTask(0, 0, "layer-zero-early", "robot-b"),
        ScheduledTask(0, 2, "layer-zero-late", "robot-a"),
        ScheduledTask(1, 1, "layer-one", "robot-c"),
    )


def test_empty_layers_and_slots_do_not_compress_indices():
    plan = _plan(
        (("task", "robot", 2, 3),),
        allocation={"robot": ["task"]},
        execution_layers=[[], [], ["task"]],
        planned_schedule=[[], [], [], [{"task": "task", "robot": "robot"}]],
    )

    assert compile_stap_schedule(plan) == (ScheduledTask(2, 3, "task", "robot"),)


def test_result_is_tuple_and_scheduled_task_is_frozen():
    item = compile_stap_schedule(_plan())[0]

    assert type(compile_stap_schedule(_plan())) is tuple
    assert ScheduledTask.__dataclass_params__.frozen is True
    with pytest.raises(FrozenInstanceError):
        item.task_id = "changed"


def test_does_not_modify_plan_tasks_or_raw():
    plan = _plan(
        (("a", "robot-a", 0, 0), ("b", "robot-b", 1, 2)),
        allocation={"robot-a": ["a"], "robot-b": ["b"]},
        execution_layers=[["a"], ["b"]],
        planned_schedule=[[{"task": "a", "robot": "robot-a"}], [], [{"task": "b", "robot": "robot-b"}]],
    )
    tasks_before = copy.deepcopy(plan.tasks)
    raw_before = copy.deepcopy(plan.raw)
    raw_before_bytes = _canonical_json_bytes(raw_before)
    raw_before_digest = hashlib.sha256(raw_before_bytes).hexdigest()

    compile_stap_schedule(plan)

    raw_after_bytes = _canonical_json_bytes(plan.raw)
    raw_after_digest = hashlib.sha256(raw_after_bytes).hexdigest()

    assert plan.tasks == tasks_before
    assert plan.raw == raw_before
    assert raw_after_bytes == raw_before_bytes
    assert raw_after_digest == raw_before_digest
    assert hashlib.sha256(raw_before_bytes).hexdigest() == raw_before_digest
    assert hashlib.sha256(raw_after_bytes).hexdigest() == raw_after_digest
    assert hashlib.sha256(b"").hexdigest() == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
    assert len(raw_before_digest) == 64
    assert len(raw_after_digest) == 64


def test_rejects_two_tasks_using_one_robot_in_one_schedule_slot():
    plan = _plan(
        (("first", "robot", 0, 0), ("second", "robot", 0, 0)),
        allocation={"robot": ["first", "second"]},
        execution_layers=[["first", "second"]],
        planned_schedule=[
            [
                {"task": "first", "robot": "robot"},
                {"task": "second", "robot": "robot"},
            ]
        ],
    )

    with pytest.raises(ScheduleContractError, match=r"schedule.*robot"):
        compile_stap_schedule(plan)


@pytest.mark.parametrize(
    ("name", "make_broken"),
    [
        ("wrong plan type", lambda: None),
        ("tasks wrong type", lambda: replace(_plan(), tasks=[])),
        ("task key wrong type", lambda: replace(_plan(), tasks={1: _task("task", "robot", 0, 0)})),
        ("task value wrong type", lambda: replace(_plan(), tasks={"task": object()})),
        ("task id mismatch", lambda: replace(_plan(), tasks={"task": replace(_plan().tasks["task"], task_id="other")})),
        ("primary robot wrong type", lambda: replace(_plan(), tasks={"task": replace(_plan().tasks["task"], primary_robot=1)})),
        ("level wrong type", lambda: replace(_plan(), tasks={"task": replace(_plan().tasks["task"], level="zero")})),
        ("level negative", lambda: replace(_plan(), tasks={"task": replace(_plan().tasks["task"], level=-1)})),
        ("schedule slot wrong type", lambda: replace(_plan(), tasks={"task": replace(_plan().tasks["task"], schedule_slot="zero")})),
        ("schedule slot negative", lambda: replace(_plan(), tasks={"task": replace(_plan().tasks["task"], schedule_slot=-1)})),
        ("raw wrong type", lambda: replace(_plan(), raw=[])),
        ("allocation missing", lambda: _without_raw(_plan(), "allocation")),
        ("allocation wrong type", lambda: _with_raw(_plan(), allocation=[])),
        ("allocation key wrong type", lambda: _with_raw(_plan(), allocation={1: ["task"]})),
        ("allocation task list wrong type", lambda: _with_raw(_plan(), allocation={"robot": "task"})),
        ("allocation unknown task", lambda: _with_raw(_plan(), allocation={"robot": ["unknown"]})),
        ("allocation duplicate task", lambda: _with_raw(_plan(), allocation={"robot": ["task", "task"]})),
        ("allocation missing task", lambda: _with_raw(_plan(), allocation={"robot": []})),
        ("allocation robot mismatch", lambda: _with_raw(_plan(), allocation={"other": ["task"]})),
        ("layers missing", lambda: _without_raw(_plan(), "execution_layers")),
        ("layers wrong type", lambda: _with_raw(_plan(), execution_layers="layers")),
        ("layer wrong type", lambda: _with_raw(_plan(), execution_layers=[None])),
        ("layer unknown task", lambda: _with_raw(_plan(), execution_layers=[["unknown"]])),
        ("layer duplicate task", lambda: _with_raw(_plan(), execution_layers=[["task", "task"]])),
        ("layer missing task", lambda: _with_raw(_plan(), execution_layers=[[]])),
        ("layer level mismatch", lambda: _with_raw(_plan(), execution_layers=[[], ["task"]])),
        ("schedule missing", lambda: _without_raw(_plan(), "planned_schedule")),
        ("schedule wrong type", lambda: _with_raw(_plan(), planned_schedule="schedule")),
        ("slot wrong type", lambda: _with_raw(_plan(), planned_schedule=[None])),
        ("entry wrong type", lambda: _with_raw(_plan(), planned_schedule=[[None]])),
        ("entry missing key", lambda: _with_raw(_plan(), planned_schedule=[[{"task": "task"}] ])),
        ("entry extra key", lambda: _with_raw(_plan(), planned_schedule=[[{"task": "task", "robot": "robot", "extra": True}]])),
        ("entry task wrong type", lambda: _with_raw(_plan(), planned_schedule=[[{"task": 1, "robot": "robot"}]])),
        ("entry robot wrong type", lambda: _with_raw(_plan(), planned_schedule=[[{"task": "task", "robot": 1}]])),
        ("schedule unknown task", lambda: _with_raw(_plan(), planned_schedule=[[{"task": "unknown", "robot": "robot"}]])),
        ("schedule duplicate task", lambda: _with_raw(_plan(), planned_schedule=[[{"task": "task", "robot": "robot"}, {"task": "task", "robot": "robot"}]])),
        ("schedule missing task", lambda: _with_raw(_plan(), planned_schedule=[[]])),
        ("schedule robot mismatch", lambda: _with_raw(_plan(), planned_schedule=[[{"task": "task", "robot": "other"}]])),
        ("schedule slot mismatch", lambda: _with_raw(_plan(), planned_schedule=[[], [{"task": "task", "robot": "robot"}]])),
    ],
)
def test_rejects_contract_errors(name: str, make_broken):
    del name
    broken = make_broken()
    with pytest.raises(ScheduleContractError):
        compile_stap_schedule(broken)


def test_import_hygiene_and_no_builtin_ordering_call():
    source = inspect.getsource(stap_schedule)
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    forbidden = {
        "rocobrick." + "scheduler",
        "rocobrick." + "execution",
        "rocobrick." + "structural_binding",
        "rocobrick." + "piper_run",
    }
    assert not any(
        module in forbidden
        or module.startswith("ros")
        or module.startswith("isaac")
        or module.startswith("tools")
        for module in imported_modules
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "".join(("sor", "ted"))
        for node in ast.walk(tree)
    )
