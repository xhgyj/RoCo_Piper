#!/usr/bin/env python3
"""Load and verify one assembly-plan/v2 scene without commanding motion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rocobrick.bricksim_topology import BaseplateSpec, build_bricksim_topologies
from rocobrick.isaac_scene import IsaacSceneConfig, load_verified_scene
from rocobrick.models import Pose
from rocobrick.scene_bootstrap import RobotSpawnSpec
from rocobrick.stap_contract import parse_stap_plan


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


async def main():
    args = _arguments()
    plan = parse_stap_plan(json.loads(args.plan.read_text(encoding="utf-8")))
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    task_config = raw["Task_Config"]
    plate = task_config["Base_Plate"]
    topology = build_bricksim_topologies(
        plan, BaseplateSpec(tuple(plate["Dimension"]), plate["Color"])
    )
    robot_specs = {}
    for robot in raw["Robot_Config"]["Robots"]:
        robot_id = robot["Name"]
        if robot_id not in plan.robots:
            continue
        base = robot["Robot_Base_Frame"]
        wxyz = base["Orientation"]
        robot_specs[robot_id] = RobotSpawnSpec(
            robot_id=robot_id,
            prim_path=robot["Robot_Prim_Path"],
            usd_path=str((args.config.parent / robot["Robot_USD_Path"]).resolve()),
            base_pose=Pose(
                "assembly_world", tuple(base["Position"]),
                (wxyz[1], wxyz[2], wxyz[3], wxyz[0]),
            ),
            arm_joint_names=tuple(robot["Joint_Order"][:6]),
            sensor_prim_paths=tuple(
                camera["Prim_Path"] for camera in robot.get("Camera_Config", {}).values()
            ),
        )
    loaded = await load_verified_scene(
        plan, topology, robot_specs,
        IsaacSceneConfig(
            tuple(plate["Position"]), tuple(plate["Orientation"]),
            "/World/LegoWorkspace",
            tuple(raw["Env_Config"]["Storage_Config"]["Size"]),
            tuple(raw["Env_Config"]["Storage_Config"]["Position"]),
            tuple(raw["Env_Config"]["Storage_Config"]["Orientation"]),
            60,
        ),
    )
    result = {
        "valid": True,
        "plan_id": plan.plan_id,
        "robots": {
            key: {"joints": list(value.joint_positions), "joint_names": list(value.joint_names)}
            for key, value in loaded.measured_robots.items()
        },
        "parts": {
            key: {"prim_path": value.prim_path, "position_m": list(value.measured_pose.position)}
            for key, value in loaded.parts.items()
        },
        "initial_connection_count": len(topology.expected_initial_connections),
        "goal_connection_count": len(topology.expected_goal_connections),
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    await loaded.world.pause_async()
    print("ROCOBRICK_SCENE_SMOKE_OK " + json.dumps(result, sort_keys=True), flush=True)
