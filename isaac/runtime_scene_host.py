from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType


SCHEMA = "rocobrick-runtime-scene-host/v1"
BRIDGE_SCHEMA = "rocobrick-host-bridge-smoke/v1"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--snapshot-out", type=Path, required=True)
    parser.add_argument("--host-instance-id", required=True)
    parser.add_argument("--robot-id", required=True)
    parser.add_argument("--host-profile", type=Path, required=True)
    parser.add_argument("--runtime-usd", type=Path, required=True)
    parser.add_argument("--evidence-out", type=Path, required=True)
    parser.add_argument("--ros-domain-id", type=int, default=None)
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def main(argv=None) -> int:
    args = parse_args(argv)
    evidence = {
        "schema": SCHEMA,
        "passed": False,
        "plan_path": str(args.plan),
        "config_path": str(args.config),
        "host_instance_id": args.host_instance_id,
        "robot_id": args.robot_id,
    }
    args.evidence_out.parent.mkdir(parents=True, exist_ok=True)
    host = None
    loaded = None
    try:
        from rocobrick.bricksim_topology import BaseplateSpec, build_bricksim_topologies
        from rocobrick.isaac_scene import IsaacSceneConfig, load_verified_scene
        from rocobrick.isaac_snapshot_host_adapter import (
            build_host_binding,
            collect_stage_snapshot,
            load_scene_snapshot_manifest,
            write_snapshot_artifact_atomic,
        )
        from rocobrick.models import Pose
        from rocobrick.runtime_host import LoadedRuntimeHost
        from rocobrick.scene_bootstrap import RobotSpawnSpec
        from rocobrick.stap_contract import parse_stap_plan
        from omni.usd import get_context

        plan_bytes = args.plan.read_bytes()
        evidence["plan_sha256"] = hashlib.sha256(plan_bytes).hexdigest()
        plan = parse_stap_plan(json.loads(plan_bytes))
        config = json.loads(args.config.read_text(encoding="utf-8"))
        task = config["Task_Config"]
        plate = task["Base_Plate"]
        topology = build_bricksim_topologies(
            plan,
            BaseplateSpec(tuple(plate["Dimension"]), plate["Color"]),
        )
        robots = {}
        for raw in config["Robot_Config"]["Robots"]:
            robot_id = raw["Name"]
            if robot_id not in plan.robots:
                continue
            base = raw["Robot_Base_Frame"]
            orientation = tuple(base["Orientation"])
            robots[robot_id] = RobotSpawnSpec(
                robot_id=robot_id,
                prim_path=raw["Robot_Prim_Path"],
                usd_path=str((args.config.parent / raw["Robot_USD_Path"]).resolve()),
                base_pose=Pose(
                    "assembly_world",
                    tuple(base["Position"]),
                    (orientation[1], orientation[2], orientation[3], orientation[0]),
                ),
                arm_joint_names=tuple(raw["Joint_Order"]),
                sensor_prim_paths=tuple(
                    camera["Prim_Path"]
                    for camera in raw.get("Camera_Config", {}).values()
                ),
            )
        storage = config["Env_Config"]["Storage_Config"]
        loaded = await load_verified_scene(
            plan,
            topology,
            robots,
            IsaacSceneConfig(
                tuple(plate["Position"]),
                tuple(plate["Orientation"]),
                storage.get("Prim_Path", "/World/LegoWorkspace"),
                tuple(storage["Size"]),
                tuple(storage["Position"]),
                tuple(storage["Orientation"]),
                60,
            ),
        )
        paths = {key: value.prim_path for key, value in loaded.parts.items()}
        bricks = {key: value.brick_id for key, value in loaded.parts.items()}
        host = LoadedRuntimeHost(
            loaded,
            MappingProxyType(paths),
            MappingProxyType(bricks),
        )
        manifest = load_scene_snapshot_manifest(args.manifest)
        manifest_paths = {
            item.physical_instance_id: item.prim_path for item in manifest.objects
        }
        if manifest_paths != paths:
            raise RuntimeError("manifest does not match loaded BrickSim mapping")
        for item in manifest.objects:
            if bricks[item.physical_instance_id] != item.brick_id:
                raise RuntimeError(f"manifest brick mismatch: {item.physical_instance_id}")
        articulation = loaded.robots[args.robot_id]
        actual_dofs = tuple(str(name) for name in articulation.dof_names)
        binding = build_host_binding(
            host_instance_id=args.host_instance_id,
            ros_domain_id=(
                args.ros_domain_id
                if args.ros_domain_id is not None
                else int(os.environ.get("ROS_DOMAIN_ID", "0"))
            ),
            host_profile_sha256=sha256(args.host_profile),
            host_bridge_schema=BRIDGE_SCHEMA,
            runtime_usd_path=str(args.runtime_usd.resolve()),
            runtime_usd_sha256=sha256(args.runtime_usd),
            robot_id=args.robot_id,
            robot_prim_path=robots[args.robot_id].prim_path,
            articulation_prim_path=robots[args.robot_id].prim_path + "/root_joint",
            joint_dofs=actual_dofs,
            participant_count=1,
        )
        snapshot = collect_stage_snapshot(get_context().get_stage(), manifest, binding)
        write_snapshot_artifact_atomic(snapshot, args.snapshot_out)
        evidence.update({
            "passed": True,
            "snapshot_sha256": snapshot.snapshot_sha256,
            "part_mapping": paths,
            "joint_dofs": list(actual_dofs),
        })
        await loaded.world.pause_async()
        return 0
    except asyncio.CancelledError:
        evidence["error"] = "CancelledError"
        raise
    except Exception as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        return 1
    finally:
        if host is not None:
            host.close()
        args.evidence_out.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
