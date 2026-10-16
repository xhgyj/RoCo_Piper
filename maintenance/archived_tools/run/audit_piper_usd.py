#!/usr/bin/env python3
"""Audit composed Piper USD assets without commanding either articulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ARM_JOINTS = tuple(f"joint{index}" for index in range(1, 7))
GRIPPER_JOINTS = ("gripper_joint1", "gripper_joint2")


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def _applied(prim, schema: str) -> bool:
    return schema in set(prim.GetAppliedSchemas())


def _audit_robot(stage, articulation, config: dict) -> dict:
    import numpy as np
    from isaacsim.core.prims import SingleXFormPrim
    from pxr import Usd, UsdGeom, UsdPhysics

    root_path = config["Robot_Prim_Path"]
    root = stage.GetPrimAtPath(root_path)
    descendants = [
        prim for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
        if prim != root
    ]
    paths = {str(prim.GetPath()): prim for prim in descendants}
    dof_names = tuple(articulation.dof_names or ())

    joints = {}
    for name in ARM_JOINTS + GRIPPER_JOINTS:
        matches = [
            str(path) for path, prim in paths.items()
            if prim.GetName() == name and prim.IsA(UsdPhysics.Joint)
        ]
        joints[name] = {
            "dof_present": name in dof_names,
            "joint_prim_paths": sorted(matches),
            "valid": name in dof_names and len(matches) == 1,
        }

    articulation_root_paths = []
    collision_paths = []
    disabled_collision_paths = []
    rigid_body_paths = []
    unresolved_assets = []
    for path, prim in paths.items():
        schemas = set(prim.GetAppliedSchemas())
        if "PhysicsArticulationRootAPI" in schemas:
            articulation_root_paths.append(path)
        collision_api = UsdPhysics.CollisionAPI(prim)
        collision_attribute = prim.GetAttribute("physics:collisionEnabled")
        if "PhysicsCollisionAPI" in schemas or collision_api or collision_attribute.IsValid():
            enabled = collision_attribute.Get() if collision_attribute.IsValid() else None
            (collision_paths if enabled is not False else disabled_collision_paths).append(path)
        if _applied(prim, "PhysicsRigidBodyAPI"):
            rigid_body_paths.append(path)
        for attribute in prim.GetAttributes():
            if str(attribute.GetTypeName()) != "asset":
                continue
            value = attribute.Get()
            if value and getattr(value, "resolvedPath", "") == "":
                unresolved_assets.append({
                    "prim_path": path,
                    "attribute": attribute.GetName(),
                    "asset_path": getattr(value, "path", str(value)),
                })

    tcp_path = f"{root_path}/link6"
    tcp_link_position, tcp_link_wxyz = SingleXFormPrim(tcp_path).get_world_pose()
    w, x, y, z = (float(value) for value in tcp_link_wxyz)
    local_z = np.asarray((
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y),
    ))
    tcp_position = np.asarray(tcp_link_position, dtype=float) + 0.1358 * local_z
    camera_paths = [
        camera["Prim_Path"]
        for camera in config.get("Camera_Config", {}).values()
    ]
    camera_checks = {
        path: stage.GetPrimAtPath(path).IsValid()
        and stage.GetPrimAtPath(path).IsA(UsdGeom.Camera)
        for path in camera_paths
    }
    arm_ok = all(joints[name]["valid"] for name in ARM_JOINTS)
    gripper_ok = all(joints[name]["valid"] for name in GRIPPER_JOINTS)
    result = {
        "root_path": root_path,
        "articulation_root_paths": sorted(articulation_root_paths),
        "articulation_root": bool(articulation_root_paths),
        "dof_names": list(dof_names),
        "joints": joints,
        "arm_joints_complete": arm_ok,
        "gripper_complete": gripper_ok,
        "tcp": {
            "path": tcp_path,
            "valid": stage.GetPrimAtPath(tcp_path).IsValid(),
            "position_m": [float(value) for value in tcp_position],
            "orientation_xyzw": [x, y, z, w],
        },
        "cameras": camera_checks,
        "collision_prim_paths": sorted(collision_paths),
        "disabled_collision_prim_paths": sorted(disabled_collision_paths),
        "rigid_body_prim_paths": sorted(rigid_body_paths),
        "unresolved_assets": unresolved_assets,
    }
    result["physical_model_complete"] = all((
        result["articulation_root"], arm_ok, gripper_ok,
        result["tcp"]["valid"], all(camera_checks.values()),
        bool(collision_paths), bool(rigid_body_paths),
    ))
    return result


async def main():
    import numpy as np
    import omni.kit.app
    from isaacsim.core.api.world import World
    from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage, open_stage_async
    from bricksim.assets import DEFAULT_STAGE_PATH

    args = _arguments()
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    app = omni.kit.app.get_app()
    if World._world_initialized:
        World.clear_instance()
    await open_stage_async(str(DEFAULT_STAGE_PATH))
    world = World(backend="numpy", device="cpu", physics_prim_path="/physicsScene")
    await world.initialize_simulation_context_async()
    stage = get_current_stage()
    articulations = {}
    for robot in raw["Robot_Config"]["Robots"]:
        usd_path = (args.config.parent / robot["Robot_USD_Path"]).resolve()
        root_path = robot["Robot_Prim_Path"]
        add_reference_to_stage(str(usd_path), root_path)
        base = robot["Robot_Base_Frame"]
        orientation = base["Orientation"]
        SingleXFormPrim(root_path, name=f"{robot['Name']}_audit_root").set_world_pose(
            position=np.asarray(base["Position"], dtype=float),
            orientation=np.asarray(orientation, dtype=float),
        )
        articulation = SingleArticulation(root_path, name=f"{robot['Name']}_audit")
        world.scene.add(articulation)
        articulations[robot["Name"]] = articulation
    await world.reset_async()
    await world.play_async()
    for _ in range(12):
        await app.next_update_async()

    robots = {
        robot["Name"]: _audit_robot(stage, articulations[robot["Name"]], robot)
        for robot in raw["Robot_Config"]["Robots"]
    }
    result = {
        "valid": all(item["physical_model_complete"] for item in robots.values()),
        "motion_commanded": False,
        "robots": robots,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    await world.pause_async()
    print("ROCOBRICK_PIPER_USD_AUDIT " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
