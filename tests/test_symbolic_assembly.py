"""Tests for deterministic symbolic assembly task generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rocobrick.env.loose_parts import aabb_clearance, footprint_aabb
from rocobrick.task_config.symbolic_assembly import (
    GenerationConfig,
    build_single_step_plan,
    generate_demo_task,
    generate_symbolic_tasks,
    oriented_dimensions,
    validate_generated_task,
    validate_goal_structure,
)


def test_basic_task_has_one_preplaced_reference_and_one_target() -> None:
    """The target is absent from start and introduced by the only step."""
    task = generate_demo_task("basic", seed=7)
    assert len(task.start) == 1
    assert len(task.goal) == 2
    assert set(task.goal) - set(task.start) == {str(task.step.target_part_id)}
    assert task.step.additional_connections == ()
    assert validate_generated_task(task).valid


def test_bridge_groups_two_connections_into_one_target_step() -> None:
    """A cross-brick target is represented as one action with two supports."""
    task = generate_demo_task("bridge", seed=3)
    assert len(task.start) == 2
    assert len(task.goal) == 3
    assert len(task.step.additional_connections) == 1
    references = {
        task.step.primary_connection.reference_part_id,
        task.step.additional_connections[0].reference_part_id,
    }
    assert references == {1, 2}
    assert validate_generated_task(task).valid


def test_multilevel_task_keeps_scaffold_in_start() -> None:
    """Every lower scaffold brick is preplaced before the target is spawned."""
    task = generate_demo_task("multilevel", seed=11)
    target = task.goal[str(task.step.target_part_id)]
    assert 1 <= target["z"] <= 3
    assert all(brick["z"] < target["z"] for brick in task.start.values())
    assert task.step.reference_level == target["z"]


def test_plan_is_derived_from_start_goal_difference() -> None:
    """Rebuilding a plan produces the exact generated semantic record."""
    task = generate_demo_task("adjacent", seed=19)
    assert build_single_step_plan(task.start, task.goal, task.family) == task.step


def test_generated_structures_use_a_canonical_local_origin() -> None:
    """Global grid translation is removed while relative geometry is retained."""
    for family in ("basic", "adjacent", "multilevel", "dense", "bridge"):
        task = generate_demo_task(family, seed=29)
        assert min(brick["x"] for brick in task.goal.values()) == 0
        assert min(brick["y"] for brick in task.goal.values()) == 0
        assert min(brick["z"] for brick in task.goal.values()) == 0
        assert all(task.start[key] == task.goal[key] for key in task.start)
        assert validate_generated_task(task).valid


def test_generation_is_reproducible(tmp_path: Path) -> None:
    """Equal configurations produce byte-identical Task-1 files."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_symbolic_tasks(
        GenerationConfig(output_dir=first, seed=23, count_per_family=3)
    )
    generate_symbolic_tasks(
        GenerationConfig(output_dir=second, seed=23, count_per_family=3)
    )
    first_files = {
        path.relative_to(first): path.read_bytes()
        for path in first.rglob("*.json")
    }
    second_files = {
        path.relative_to(second): path.read_bytes()
        for path in second.rglob("*.json")
    }
    assert first_files == second_files
    expected_families = {"basic", "adjacent", "multilevel", "dense", "bridge"}
    assert {path.name for path in first.iterdir()} == expected_families
    for family in expected_families:
        assert {path.name for path in (first / family).iterdir()} == {"1", "2", "3"}
        for sequence in range(1, 4):
            task_dir = first / family / str(sequence)
            assert {path.name for path in task_dir.iterdir()} == {
                "structure_start.json",
                "structure_goal.json",
            }


def test_loose_target_aabb_is_separate_from_plate() -> None:
    """Footprint geometry reports the configured outside-board clearance."""
    plate_transform = np.eye(4)
    plate_transform[1, 3] = -0.2
    target_transform = np.eye(4)
    target_transform[1, 3] = 0.08
    plate = footprint_aabb(plate_transform, 32, 32)
    target = footprint_aabb(target_transform, 4, 2)
    assert aabb_clearance(plate, target) > 0.02


def test_goal_only_validation_accepts_generated_bridge() -> None:
    """Goal validation accepts multi-support topology without a start file."""
    task = generate_demo_task("bridge", seed=7)
    report = validate_goal_structure(task.goal)
    assert report.valid
    assert report.brick_count == 3
    assert report.connection_count >= 4


def test_dense_structure_keeps_parallel_gripper_corridor() -> None:
    """Same-layer context cannot block both sides of the grasp axis."""
    task = generate_demo_task("dense", seed=17)
    target_key = str(task.step.target_part_id)
    target = task.goal[target_key]
    target_length, target_width = oriented_dimensions(
        target["brick_id"], target["ori"]
    )
    same_layer = [
        brick
        for key, brick in task.start.items()
        if key != target_key and brick["z"] == target["z"]
    ]
    target_ranges = (
        (target["x"], target["x"] + target_length),
        (target["y"], target["y"] + target_width),
    )

    def axis_clearance(axis: int) -> float:
        perpendicular = 1 - axis
        nearest = np.inf
        for brick in same_layer:
            length, width = oriented_dimensions(brick["brick_id"], brick["ori"])
            ranges = (
                (brick["x"], brick["x"] + length),
                (brick["y"], brick["y"] + width),
            )
            if not (
                ranges[perpendicular][0] < target_ranges[perpendicular][1]
                and ranges[perpendicular][1] > target_ranges[perpendicular][0]
            ):
                continue
            if ranges[axis][1] <= target_ranges[axis][0]:
                gap = target_ranges[axis][0] - ranges[axis][1]
            elif ranges[axis][0] >= target_ranges[axis][1]:
                gap = ranges[axis][0] - target_ranges[axis][1]
            else:
                gap = 0
            nearest = min(nearest, gap)
        return float(nearest)

    assert max(axis_clearance(0), axis_clearance(1)) >= 1


def test_goal_only_validation_rejects_overlap_and_floating_brick() -> None:
    """Goal validation reports geometry and support failures together."""
    goal = {
        "1": {"x": 0, "y": 0, "z": 0, "ori": 0, "brick_id": 12, "color": "Red"},
        "2": {"x": 1, "y": 0, "z": 0, "ori": 0, "brick_id": 12, "color": "Blue"},
        "3": {"x": 8, "y": 8, "z": 2, "ori": 0, "brick_id": 12, "color": "Green"},
    }
    report = validate_goal_structure(goal)
    assert not report.valid
    assert any("overlap" in error for error in report.errors)
    assert any("unsupported" in error for error in report.errors)
