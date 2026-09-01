"""Pure tests for the planner-driven episode pipeline."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from rocobrick.execution.episode import ExecutionPlanner
from rocobrick.execution.staging import initial_slot_by_target
from rocobrick.planning.task_planner import PlanningProblem, TopologyTaskPlanner
from rocobrick.task_config.episode import EpisodeConfig

ROOT = Path(__file__).resolve().parents[1]


def _task_d_plan():
    topology = json.loads(
        (ROOT / "tasks/type1/D/execution_plan.json").read_text(encoding="utf-8")
    )
    problem = PlanningProblem(
        "task_d", topology, (0,), ("piper_0", "piper_1")
    )
    return TopologyTaskPlanner().plan(problem)


def test_task_d_episode_retains_current_dual_arm_layout() -> None:
    """The generic config preserves the calibrated current robot poses."""
    episode = EpisodeConfig.load(
        ROOT / "config/episodes/task_d/episode.json"
    )
    poses = {item.robot_id: item.position for item in episode.robot_overrides}
    assert episode.available_arm_ids == ("piper_0", "piper_1")
    assert poses == {
        "piper_0": (0.22, -0.40, 0.0),
        "piper_1": (-0.22, -0.40, 0.0),
    }
    assert episode.staging.arm("piper_0").parking_slots[0] == (0.30, -0.17)
    assert episode.staging.arm("piper_1").parking_slots[0] == (-0.30, -0.17)


def test_reference_planner_reproduces_task_d_assignment() -> None:
    """Topology branches yield the currently validated two-arm ownership."""
    plan = _task_d_plan()
    assignments = {
        task.target_part_id: task.assigned_arm for task in plan.tasks
    }
    assert tuple(task.target_part_id for task in plan.tasks) == (
        2,
        1,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        11,
        12,
        10,
    )
    assert {part for part, arm in assignments.items() if arm == "piper_0"} == {
        1,
        2,
        4,
        6,
        8,
        11,
    }
    assert {part for part, arm in assignments.items() if arm == "piper_1"} == {
        3,
        5,
        7,
        9,
        10,
        12,
    }
    bridge = plan.tasks[-1]
    assert bridge.connection_ids == (5, 11)
    assert set(bridge.depends_on) == {"assemble_11", "assemble_12"}


def test_task_d_assigns_initial_slots_without_a_pickup_restage() -> None:
    """Each target keeps the first slot selected for its planner-owned arm."""
    episode = EpisodeConfig.load(
        ROOT / "config/episodes/task_d/episode.json"
    )
    slots = initial_slot_by_target(_task_d_plan(), episode.staging)
    assert slots[2] == (0.30, -0.17)
    assert slots[3] == (-0.30, -0.17)
    assert slots[10] == (-0.20, -0.17)


def test_task_d_disables_prefetch_across_global_task_boundaries() -> None:
    """Task D finishes one placement before dispatching the next pickup."""
    episode = EpisodeConfig.load(
        ROOT / "config/episodes/task_d/episode.json"
    )
    execution = ExecutionPlanner().compile(_task_d_plan(), episode)
    tasks = {item.task.task_id: item for item in execution.tasks}
    assert not episode.allow_prefetch
    assert tasks["assemble_3"].prepare_after == (
        "assemble_1",
        "assemble_2",
    )
    assert tasks["assemble_3"].place_after == (
        "assemble_1",
        "assemble_2",
    )
    assert tasks["assemble_4"].prepare_after == (
        "assemble_1",
        "assemble_3",
    )
    assert tasks["assemble_4"].place_after == (
        "assemble_1",
        "assemble_3",
    )
    assert tasks["assemble_10"].prepare_after == (
        "assemble_11",
        "assemble_12",
    )
    assert tasks["assemble_11"].return_home
    assert tasks["assemble_10"].return_home
    assert not tasks["assemble_4"].return_home


def test_execution_plan_can_enable_one_step_prefetch() -> None:
    """Other episodes may explicitly opt into bounded preparation overlap."""
    episode = EpisodeConfig.load(
        ROOT / "config/episodes/task_d/episode.json"
    )
    execution = ExecutionPlanner().compile(
        _task_d_plan(), replace(episode, allow_prefetch=True)
    )
    tasks = {item.task.task_id: item for item in execution.tasks}
    assert tasks["assemble_3"].prepare_after == ("assemble_2",)
    assert tasks["assemble_4"].prepare_after == ("assemble_1",)


def test_planner_rejects_a_part_without_support_connection() -> None:
    """An ungrounded target fails before simulator execution."""
    topology = {
        "parts": [{"id": 0}, {"id": 1}],
        "connections": [],
    }
    problem = PlanningProblem("broken", topology, (0,), ("piper_0",))
    try:
        TopologyTaskPlanner().plan(problem)
    except ValueError as error:
        assert "no incoming connection" in str(error)
    else:
        raise AssertionError("planner accepted an unsupported target")
