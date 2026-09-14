from __future__ import annotations

import pytest

from rocobrick.contracts import parse_plan, parse_scene, validate_binding


@pytest.fixture
def valid_payloads():
    plan = {
        "schema": "assembly-plan/v2",
        "plan_id": "arbitrary-plan",
        "frame_id": "assembly_world",
        "units": {"length": "m", "angle": "rad"},
        "brick_library_version": "brick-library/test",
        "robot_asset_version": "agx-piper-l/test",
        "robots": ["piper_0", "piper_1"],
        "tasks": [
            {
                "task_id": "foundation:left",
                "part_instance_id": "physical-brick-44",
                "brick_id": 2,
                "target": {"frame_id": "assembly_world", "position_m": [0.0, 0.0, 0.01], "orientation_xyzw": [0, 0, 0, 1]},
                "target_interface": "baseplate/studs/0-3",
                "skill_type": "PLACE_DOWN",
                "primary_robot": "piper_0",
                "supporting_robot": None,
                "predecessors": [],
                "level": 0,
                "schedule_slot": 0,
            },
            {
                "task_id": "cap@right",
                "part_instance_id": "physical-brick-97",
                "brick_id": 3,
                "target": {"frame_id": "assembly_world", "position_m": [0.2, 0.0, 0.01], "orientation_xyzw": [0, 0, 0, 1]},
                "target_interface": "baseplate/studs/20-23",
                "skill_type": "PLACE_DOWN",
                "primary_robot": "piper_1",
                "supporting_robot": None,
                "predecessors": [],
                "level": 0,
                "schedule_slot": 0,
            },
            {
                "task_id": "joined/top",
                "part_instance_id": "physical-brick-123",
                "brick_id": 9,
                "target": {"frame_id": "assembly_world", "position_m": [0.1, 0.0, 0.02], "orientation_xyzw": [0, 0, 0, 1]},
                "target_interface": "physical-brick-44/top",
                "skill_type": "PLACE_DOWN",
                "primary_robot": "piper_0",
                "supporting_robot": None,
                "predecessors": ["foundation:left"],
                "level": 1,
                "schedule_slot": 1,
            },
        ],
    }
    scene = {
        "schema": "assembly-scene/v1",
        "scene_id": "scene-test",
        "frame_id": "assembly_world",
        "parts": [
            {"part_instance_id": task["part_instance_id"], "brick_id": task["brick_id"], "pose": {"frame_id": "assembly_world", "position_m": [0.0, -0.2, 0.05], "orientation_xyzw": [0, 0, 0, 1]}}
            for task in plan["tasks"]
        ],
    }
    return plan, scene


@pytest.fixture
def parsed(valid_payloads):
    plan_payload, scene_payload = valid_payloads
    plan, scene = parse_plan(plan_payload), parse_scene(scene_payload)
    validate_binding(plan, scene)
    return plan, scene
