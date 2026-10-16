#!/usr/bin/env python3
"""Plan and physically execute one pregrasp from assembly-plan/v2."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import time

from google.protobuf.json_format import MessageToDict

from rocobrick.bricksim_backend import BrickSimTrajectoryPort, RobotBinding
from rocobrick.bricksim_topology import BaseplateSpec, build_bricksim_topologies
from rocobrick.isaac_scene import IsaacSceneConfig, load_verified_scene
from rocobrick.models import ActionType, Pose, Revisions, RobotSnapshot
from rocobrick.planner_client import GrpcPlannerClient, _encode_request
from rocobrick.planner_contract import (
    CollisionObject, CollisionShape, ContactAllowance, PlanningRequest, PlanningStatus,
)
from rocobrick.scene_bootstrap import RobotSpawnSpec
from rocobrick.stap_contract import parse_stap_plan
from rocobrick.trajectory_execution import PlaybackStatus, play_trajectory


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--planner", default="127.0.0.1:50052")
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def event(path: Path, name: str, **fields) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": name, "wall_time_s": time.time(), **fields}, sort_keys=True) + "\n")


def fnv1a(payload: bytes) -> str:
    value = 1469598103934665603
    for byte in payload:
        value = ((value ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return format(value, "x")


def aabb(stage, prim_path: str):
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bounds = cache.ComputeWorldBound(stage.GetPrimAtPath(prim_path)).ComputeAlignedRange()
    minimum = tuple(float(value) for value in bounds.GetMin())
    maximum = tuple(float(value) for value in bounds.GetMax())
    if not all(math.isfinite(value) for value in minimum + maximum):
        raise RuntimeError(f"non-finite physical bounds for {prim_path}")
    return minimum, maximum


def box(object_id: str, minimum, maximum) -> CollisionObject:
    center = tuple((low + high) / 2.0 for low, high in zip(minimum, maximum))
    dimensions = tuple(high - low for low, high in zip(minimum, maximum))
    if min(dimensions) <= 0.0:
        raise RuntimeError(f"empty collision bounds for {object_id}")
    material = json.dumps([object_id, minimum, maximum], separators=(",", ":"))
    return CollisionObject(
        object_id, Pose("world", center, (0.0, 0.0, 0.0, 1.0)), CollisionShape.BOX,
        dimensions, sha256(material.encode()).hexdigest(),
    )


def collision_aabbs(stage, root_path: str) -> dict[str, object]:
    from pxr import Usd, UsdPhysics

    root = stage.GetPrimAtPath(root_path)
    result = {}
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        attribute = prim.GetAttribute("physics:collisionEnabled")
        if "PhysicsCollisionAPI" not in set(prim.GetAppliedSchemas()) and not attribute.IsValid():
            continue
        if attribute.IsValid() and attribute.Get() is False:
            continue
        result[str(prim.GetPath())] = aabb(stage, str(prim.GetPath()))
    return result


def snapshots(loaded, specs) -> tuple[RobotSnapshot, ...]:
    result = []
    for robot_id in sorted(specs):
        articulation = loaded.robots[robot_id]
        names = tuple(articulation.dof_names)
        raw = articulation.get_joint_positions()
        joints = tuple(float(raw[names.index(name)]) for name in specs[robot_id].arm_joint_names)
        result.append(RobotSnapshot(robot_id, joints, time.monotonic()))
    return tuple(result)


def response_json(response) -> dict:
    return {
        "request_id": response.request_id, "status": response.status.value,
        "revisions": asdict(response.revisions),
        "required_resources": sorted(response.required_resources),
        "start_digest": response.start_digest, "goal_digest": response.goal_digest,
        "scene_digest": response.scene_digest, "collision_report": list(response.collision_report),
        "trajectory": None if response.trajectory is None else asdict(response.trajectory),
    }


async def main():
    from isaacsim.core.prims import SingleXFormPrim
    from isaacsim.core.utils.stage import get_current_stage

    args = arguments()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    events = args.run_dir / "events.jsonl"
    result_path = args.run_dir / "result.json"
    result = {"success": False, "stage": "STARTUP", "reason": "unexpected_exit"}
    (args.run_dir / "input-plan.json").write_bytes(args.plan.read_bytes())
    (args.run_dir / "config.json").write_bytes(args.config.read_bytes())
    try:
        plan = parse_stap_plan(json.loads(args.plan.read_text(encoding="utf-8")))
        if len(plan.tasks) != 1:
            raise RuntimeError("pregrasp stage requires exactly one pending plan task")
        task = next(iter(plan.tasks.values()))
        raw = json.loads(args.config.read_text(encoding="utf-8"))
        calibration = raw["Manipulation_Config"]
        plate = raw["Task_Config"]["Base_Plate"]
        topology = build_bricksim_topologies(plan, BaseplateSpec(tuple(plate["Dimension"]), plate["Color"]))
        specs = {}
        for robot in raw["Robot_Config"]["Robots"]:
            if robot["Name"] not in plan.robots:
                continue
            base, wxyz = robot["Robot_Base_Frame"], robot["Robot_Base_Frame"]["Orientation"]
            specs[robot["Name"]] = RobotSpawnSpec(
                robot["Name"], robot["Robot_Prim_Path"],
                str((args.config.parent / robot["Robot_USD_Path"]).resolve()),
                Pose("assembly_world", tuple(base["Position"]), (wxyz[1], wxyz[2], wxyz[3], wxyz[0])),
                tuple(robot["Joint_Order"][:6]),
                tuple(camera["Prim_Path"] for camera in robot.get("Camera_Config", {}).values()),
            )
        loaded = await load_verified_scene(
            plan, topology, specs,
            IsaacSceneConfig(
                tuple(plate["Position"]), tuple(plate["Orientation"]), "/World/LegoWorkspace",
                tuple(raw["Env_Config"]["Storage_Config"]["Size"]),
                tuple(raw["Env_Config"]["Storage_Config"]["Position"]),
                tuple(raw["Env_Config"]["Storage_Config"]["Orientation"]), 60,
            ),
        )
        stage, revisions = get_current_stage(), Revisions(1, 1, 0)
        part_path = loaded.parts[task.physical_instance_id].prim_path
        event(events, "SCENE_VERIFIED", revisions=asdict(revisions), task_id=task.task_id,
              robot_id=task.primary_robot, part_instance_id=task.physical_instance_id,
              bounds={key: aabb(stage, part.prim_path) for key, part in loaded.parts.items()})

        port = BrickSimTrajectoryPort(
            loaded.app, loaded.world,
            {key: RobotBinding(loaded.robots[key], specs[key].arm_joint_names) for key in specs},
            lambda _robot_id: None, lambda _robot_id: True,
        )
        for robot_id in sorted(specs):
            port.controlled_stop(robot_id)
        for _ in range(30):
            await port.step()
        event(events, "MEASURED_STATE_HOLDING", robots={
            item.robot_id: item.joints for item in snapshots(loaded, specs)
        })

        client = GrpcPlannerClient(args.planner)
        health = client.health(5.0)
        (args.run_dir / "planner-health.json").write_text(json.dumps(asdict(health), indent=2) + "\n")
        if not health.ready or health.trajectory_execution_enabled:
            raise RuntimeError(f"unsafe planner health: {health}")

        response = request = None
        for attempt in range(1, 4):
            robots = snapshots(loaded, specs)
            start = next(item for item in robots if item.robot_id == task.primary_robot)
            objects = [
                box("part:" + key, *aabb(stage, part.prim_path))
                for key, part in sorted(loaded.parts.items())
            ]
            top, thickness = float(calibration["Ground_Top_Z"]), float(calibration["Ground_Thickness"])
            size = tuple(float(value) for value in calibration["Ground_Size"])
            objects.append(box("ground", (-size[0] / 2, -size[1] / 2, top - thickness),
                               (size[0] / 2, size[1] / 2, top)))
            minimum, maximum = aabb(stage, part_path)
            target = Pose(
                "world",
                ((minimum[0] + maximum[0]) / 2, (minimum[1] + maximum[1]) / 2,
                 maximum[2] + float(calibration["Pregrasp_Clearance"])),
                (1.0, 0.0, 0.0, 0.0),
            )
            request = PlanningRequest(
                f"{plan.plan_id}:pregrasp:{attempt}", ActionType.PICK, task.primary_robot,
                revisions, start, target, robots, tuple(objects), None, (),
            )
            encoded = _encode_request(request)
            request_payload = MessageToDict(encoded, preserving_proto_field_name=True)
            (args.run_dir / f"planning-request-{attempt}.json").write_text(
                json.dumps(request_payload, indent=2) + "\n"
            )
            before_joints = start.joints
            before_part = SingleXFormPrim(part_path).get_world_pose()[0]
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = client.plan_async(request, executor, 35.0)
                while not future.done():
                    await loaded.app.next_update_async()
                response = future.result()
            latest = snapshots(loaded, specs)
            latest_start = next(item for item in latest if item.robot_id == task.primary_robot)
            latest_part = SingleXFormPrim(part_path).get_world_pose()[0]
            robot_delta = max(abs(a - b) for a, b in zip(before_joints, latest_start.joints))
            world_delta = max(abs(float(a) - float(b)) for a, b in zip(before_part, latest_part))
            robot_changed = robot_delta > float(calibration["State_Change_Tolerance_Rad"])
            world_changed = world_delta > float(calibration["World_Change_Tolerance_M"])
            (args.run_dir / f"planning-response-{attempt}.json").write_text(
                json.dumps(response_json(response), indent=2) + "\n"
            )
            if world_changed or robot_changed:
                revisions = Revisions(revisions.world + int(world_changed),
                                      revisions.robot_state + int(robot_changed), revisions.motion)
                event(events, "PLANNING_RESULT_REJECTED_STALE", attempt=attempt,
                      world_changed=world_changed, robot_changed=robot_changed,
                      world_delta_m=world_delta, robot_delta_rad=robot_delta,
                      revisions=asdict(revisions))
                continue
            if response.revisions != request.revisions or response.start_digest != fnv1a(encoded.start.SerializeToString()):
                raise RuntimeError("planner response revision or start digest mismatch")
            if response.status is not PlanningStatus.SUCCESS or response.trajectory is None:
                raise RuntimeError(f"pregrasp planning failed: {response.status}: {response.collision_report}")
            error = max(abs(a - b) for a, b in zip(request.start.joints, latest_start.joints))
            if error > float(calibration["Start_Tolerance_Rad"]):
                revisions = Revisions(revisions.world, revisions.robot_state + 1, revisions.motion)
                event(events, "PLANNING_RESULT_REJECTED_START", attempt=attempt, error_rad=error)
                continue
            break
        else:
            raise RuntimeError("pregrasp could not obtain a fresh admitted trajectory")

        assert request is not None and response is not None and response.trajectory is not None
        event(events, "PREGRASP_ADMITTED", request_id=request.request_id,
              revisions=asdict(request.revisions), start_digest=response.start_digest,
              scene_digest=response.scene_digest, goal_digest=response.goal_digest)
        playback = await play_trajectory(port, task.primary_robot, response.trajectory,
                                         expected_holding_part=None)
        event(events, "PREGRASP_PLAYBACK_FINISHED", status=playback.status.value,
              physics_steps=playback.physics_steps,
              maximum_tracking_error_rad=playback.maximum_tracking_error_rad,
              final_joints=playback.final_state.joints)
        if playback.status is not PlaybackStatus.COMPLETED:
            raise RuntimeError(f"pregrasp playback failed: {playback.status.value}")

        minimum, maximum = aabb(stage, part_path)
        brick_width = min(maximum[0] - minimum[0], maximum[1] - minimum[1])
        open_width = min(0.1, brick_width + float(calibration["Gripper_Open_Clearance"]))
        for _ in range(180):
            port.command_gripper_opening(task.primary_robot, open_width)
            await port.step()
        measured_opening = port.measured_gripper_opening(task.primary_robot)
        if abs(measured_opening - open_width) > float(calibration["Gripper_Position_Tolerance"]):
            raise RuntimeError(
                f"gripper failed to open: target={open_width}, measured={measured_opening}"
            )
        event(events, "GRIPPER_OPEN_CONFIRMED", target_m=open_width, measured_m=measured_opening)

        revisions = Revisions(revisions.world, revisions.robot_state + 1, revisions.motion + 2)
        robots = snapshots(loaded, specs)
        start = next(item for item in robots if item.robot_id == task.primary_robot)
        objects = [
            box("part:" + key, *aabb(stage, part.prim_path))
            for key, part in sorted(loaded.parts.items())
        ]
        top, thickness = float(calibration["Ground_Top_Z"]), float(calibration["Ground_Thickness"])
        size = tuple(float(value) for value in calibration["Ground_Size"])
        objects.append(box("ground", (-size[0] / 2, -size[1] / 2, top - thickness),
                           (size[0] / 2, size[1] / 2, top)))
        minimum, maximum = aabb(stage, part_path)
        approach_target = Pose(
            "world",
            ((minimum[0] + maximum[0]) / 2, (minimum[1] + maximum[1]) / 2,
             maximum[2] + float(calibration["Grasp_TCP_Top_Offset"])),
            (1.0, 0.0, 0.0, 0.0),
        )
        part_object_id = "part:" + task.physical_instance_id
        allowances = tuple(
            ContactAllowance(part_object_id, f"{task.primary_robot}_{link}", 20.0, 0.002, 5.0)
            for link in ("gripper_link1", "gripper_link2")
        )
        approach_request = PlanningRequest(
            f"{plan.plan_id}:approach:1", ActionType.PICK, task.primary_robot, revisions,
            start, approach_target, robots, tuple(objects), None, allowances,
        )
        encoded = _encode_request(approach_request)
        (args.run_dir / "approach-planning-request.json").write_text(json.dumps(
            MessageToDict(encoded, preserving_proto_field_name=True), indent=2
        ) + "\n")
        before_part = SingleXFormPrim(part_path).get_world_pose()[0]
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = client.plan_async(approach_request, executor, 35.0)
            while not future.done():
                await loaded.app.next_update_async()
            approach_response = future.result()
        (args.run_dir / "approach-planning-response.json").write_text(
            json.dumps(response_json(approach_response), indent=2) + "\n"
        )
        latest_start = next(
            item for item in snapshots(loaded, specs) if item.robot_id == task.primary_robot
        )
        latest_part = SingleXFormPrim(part_path).get_world_pose()[0]
        robot_delta = max(abs(a - b) for a, b in zip(start.joints, latest_start.joints))
        world_delta = max(abs(float(a) - float(b)) for a, b in zip(before_part, latest_part))
        if robot_delta > float(calibration["State_Change_Tolerance_Rad"]) or world_delta > float(
            calibration["World_Change_Tolerance_M"]
        ):
            raise RuntimeError(
                f"approach result stale: robot_delta={robot_delta}, world_delta={world_delta}"
            )
        if (
            approach_response.revisions != approach_request.revisions
            or approach_response.start_digest != fnv1a(encoded.start.SerializeToString())
        ):
            raise RuntimeError("approach response revision or start digest mismatch")
        if approach_response.status is not PlanningStatus.SUCCESS or approach_response.trajectory is None:
            raise RuntimeError(
                f"approach planning failed: {approach_response.status}: {approach_response.collision_report}"
            )
        event(events, "APPROACH_ADMITTED", request_id=approach_request.request_id,
              revisions=asdict(revisions), start_digest=approach_response.start_digest,
              scene_digest=approach_response.scene_digest, goal_digest=approach_response.goal_digest)
        approach_playback = await play_trajectory(
            port, task.primary_robot, approach_response.trajectory, expected_holding_part=None
        )
        event(events, "APPROACH_PLAYBACK_FINISHED", status=approach_playback.status.value,
              physics_steps=approach_playback.physics_steps,
              maximum_tracking_error_rad=approach_playback.maximum_tracking_error_rad,
              final_joints=approach_playback.final_state.joints)
        if approach_playback.status is not PlaybackStatus.COMPLETED:
            raise RuntimeError(f"approach playback failed: {approach_playback.status.value}")

        diagnostic_paths = {
            "part": part_path,
            "gripper_link1": specs[task.primary_robot].prim_path + "/gripper_link1",
            "gripper_link2": specs[task.primary_robot].prim_path + "/gripper_link2",
        }
        event(events, "GRASP_GEOMETRY_MEASURED", bounds={
            key: aabb(stage, path) for key, path in diagnostic_paths.items()
        }, collision_bounds={
            key: collision_aabbs(stage, path) for key, path in diagnostic_paths.items()
        })

        stable_steps = 0
        previous_opening = port.measured_gripper_opening(task.primary_robot)
        for _ in range(300):
            port.command_gripper_opening(task.primary_robot, 0.0)
            await port.step()
            measured_opening = port.measured_gripper_opening(task.primary_robot)
            if abs(measured_opening - previous_opening) <= float(calibration["Gripper_Stable_Delta"]):
                stable_steps += 1
            else:
                stable_steps = 0
            previous_opening = measured_opening
            if stable_steps >= int(calibration["Gripper_Stable_Steps"]):
                break
        if stable_steps < int(calibration["Gripper_Stable_Steps"]):
            event(events, "GRIPPER_CLOSE_UNSTABLE", measured_opening_m=measured_opening,
                  measured_fingers_m=port.measured_gripper_fingers(task.primary_robot),
                  stable_steps=stable_steps)
            raise RuntimeError("gripper did not reach a stable physical state while closing")
        event(events, "GRIPPER_CLOSED_GEOMETRY_MEASURED", measured_opening_m=measured_opening,
              measured_fingers_m=port.measured_gripper_fingers(task.primary_robot),
              bounds={key: aabb(stage, path) for key, path in diagnostic_paths.items()},
              collision_bounds={
                  key: collision_aabbs(stage, path) for key, path in diagnostic_paths.items()
              })
        if measured_opening < float(calibration["Minimum_Confirmed_Grasp_Opening"]):
            raise RuntimeError(f"gripper closed without capturing the part: opening={measured_opening}")
        event(events, "GRASP_CLOSURE_CONFIRMED", measured_opening_m=measured_opening,
              measured_fingers_m=port.measured_gripper_fingers(task.primary_robot),
              stable_steps=stable_steps)

        arm_stable_steps = 0
        previous_arm = port.measured_state(task.primary_robot).joints
        port.controlled_stop(task.primary_robot)
        for _ in range(300):
            await port.step()
            current_arm = port.measured_state(task.primary_robot).joints
            arm_delta = max(abs(a - b) for a, b in zip(previous_arm, current_arm))
            arm_stable_steps = arm_stable_steps + 1 if arm_delta <= float(
                calibration["State_Change_Tolerance_Rad"]
            ) else 0
            previous_arm = current_arm
            if arm_stable_steps >= int(calibration["Gripper_Stable_Steps"]):
                break
        if arm_stable_steps < int(calibration["Gripper_Stable_Steps"]):
            raise RuntimeError("arm did not settle after physical grasp confirmation")
        event(events, "POST_GRASP_ARM_STABLE", joints=previous_arm,
              stable_steps=arm_stable_steps, maximum_delta_rad=arm_delta)

        def tcp_position():
            import numpy as np

            position, wxyz = SingleXFormPrim(
                specs[task.primary_robot].prim_path + "/link6"
            ).get_world_pose()
            w, x, y, z = (float(value) for value in wxyz)
            local_z = np.asarray((2 * (x * z + w * y), 2 * (y * z - w * x),
                                  1 - 2 * (x * x + y * y)))
            return tuple(float(value) for value in np.asarray(position) + 0.1358 * local_z)

        captured_part_position = tuple(
            float(value) for value in SingleXFormPrim(part_path).get_world_pose()[0]
        )
        captured_tcp_position = tcp_position()
        captured_relative = tuple(
            part - tcp for part, tcp in zip(captured_part_position, captured_tcp_position)
        )

        def holding_observer(robot_id: str):
            if robot_id != task.primary_robot:
                return None
            part_position = SingleXFormPrim(part_path).get_world_pose()[0]
            relative = tuple(float(part) - tcp for part, tcp in zip(part_position, tcp_position()))
            error = math.sqrt(sum((value - expected) ** 2 for value, expected in zip(relative, captured_relative)))
            return task.physical_instance_id if error <= float(calibration["Holding_Relative_Tolerance"]) else None

        revisions = Revisions(revisions.world + 1, revisions.robot_state + 1, revisions.motion + 2)
        lift_target = Pose(
            "world", (approach_target.position[0], approach_target.position[1],
                      approach_target.position[2] + float(calibration["Lift_Height"])),
            approach_target.orientation_xyzw,
        )
        lift_response = None
        maximum_stale_replans = int(calibration["Maximum_Stale_Replans"])
        for lift_attempt in range(1, maximum_stale_replans + 1):
            robots = tuple(
                RobotSnapshot(item.robot_id, item.joints, item.feedback_monotonic_s,
                              task.physical_instance_id if item.robot_id == task.primary_robot else None)
                for item in snapshots(loaded, specs)
            )
            start = next(item for item in robots if item.robot_id == task.primary_robot)
            objects = []
            for key, part in sorted(loaded.parts.items()):
                item = box("part:" + key, *aabb(stage, part.prim_path))
                if key == task.physical_instance_id:
                    item = CollisionObject(item.object_id, item.pose, item.shape, item.dimensions_m,
                                           item.geometry_digest, task.primary_robot + "_tcp_link")
                objects.append(item)
            objects.append(box("ground", (-size[0] / 2, -size[1] / 2, top - thickness),
                               (size[0] / 2, size[1] / 2, top)))
            lift_request = PlanningRequest(
                f"{plan.plan_id}:lift:{lift_attempt}", ActionType.PICK, task.primary_robot,
                revisions, start, lift_target, robots, tuple(objects), None, allowances,
            )
            encoded = _encode_request(lift_request)
            (args.run_dir / f"lift-planning-request-{lift_attempt}.json").write_text(json.dumps(
                MessageToDict(encoded, preserving_proto_field_name=True), indent=2
            ) + "\n")
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = client.plan_async(lift_request, executor, 35.0)
                while not future.done():
                    await loaded.app.next_update_async()
                lift_response = future.result()
            (args.run_dir / f"lift-planning-response-{lift_attempt}.json").write_text(
                json.dumps(response_json(lift_response), indent=2) + "\n"
            )
            latest_start = next(
                item for item in snapshots(loaded, specs) if item.robot_id == task.primary_robot
            )
            lift_start_delta = max(abs(a - b) for a, b in zip(start.joints, latest_start.joints))
            if lift_start_delta > float(calibration["State_Change_Tolerance_Rad"]):
                event(events, "LIFT_REJECTED_START_MISMATCH", attempt=lift_attempt,
                      request_start=start.joints, measured_start=latest_start.joints,
                      maximum_delta_rad=lift_start_delta,
                      tolerance_rad=float(calibration["State_Change_Tolerance_Rad"]))
                revisions = Revisions(
                    revisions.world + 1, revisions.robot_state + 1, revisions.motion + 1
                )
                continue
            if holding_observer(task.primary_robot) is None:
                raise RuntimeError("holding relation changed while lift planning was pending")
            if (
                lift_response.revisions != lift_request.revisions
                or lift_response.start_digest != fnv1a(encoded.start.SerializeToString())
            ):
                raise RuntimeError("lift response revision or start digest mismatch")
            if lift_response.status is not PlanningStatus.SUCCESS or lift_response.trajectory is None:
                event(events, "LIFT_CANDIDATE_REJECTED", attempt=lift_attempt,
                      status=lift_response.status.value,
                      collision_report=lift_response.collision_report)
                revisions = Revisions(
                    revisions.world + 1, revisions.robot_state + 1, revisions.motion + 1
                )
                continue
            break
        else:
            raise RuntimeError(
                f"lift start remained unstable after {maximum_stale_replans} fresh planning attempts"
            )
        event(events, "LIFT_ADMITTED", request_id=lift_request.request_id,
              revisions=asdict(revisions), start_digest=lift_response.start_digest,
              scene_digest=lift_response.scene_digest, goal_digest=lift_response.goal_digest)
        holding_port = BrickSimTrajectoryPort(
            loaded.app, loaded.world,
            {key: RobotBinding(loaded.robots[key], specs[key].arm_joint_names) for key in specs},
            holding_observer, lambda _robot_id: True,
        )
        lift_playback = await play_trajectory(
            holding_port, task.primary_robot, lift_response.trajectory,
            expected_holding_part=task.physical_instance_id,
        )
        final_part_position = tuple(
            float(value) for value in SingleXFormPrim(part_path).get_world_pose()[0]
        )
        lift_distance = final_part_position[2] - captured_part_position[2]
        event(events, "LIFT_PLAYBACK_FINISHED", status=lift_playback.status.value,
              physics_steps=lift_playback.physics_steps,
              maximum_tracking_error_rad=lift_playback.maximum_tracking_error_rad,
              lift_distance_m=lift_distance, holding=holding_observer(task.primary_robot))
        if lift_playback.status is not PlaybackStatus.COMPLETED:
            raise RuntimeError(f"lift playback failed: {lift_playback.status.value}")
        if lift_distance < float(calibration["Lift_Height"]) - 0.02:
            raise RuntimeError(f"part did not physically lift far enough: {lift_distance}")
        if holding_observer(task.primary_robot) != task.physical_instance_id:
            raise RuntimeError("part holding relation failed after lift")
        event(events, "HOLDING_CONFIRMED", part_instance_id=task.physical_instance_id,
              relative_position_m=captured_relative, lift_distance_m=lift_distance)
        result = {
            "success": True, "stage": "PICK", "plan_id": plan.plan_id,
            "task_id": task.task_id, "robot_id": task.primary_robot,
            "part_instance_id": task.physical_instance_id, "target": asdict(request.target),
            "revisions": asdict(request.revisions), "start_digest": response.start_digest,
            "scene_digest": response.scene_digest, "goal_digest": response.goal_digest,
            "pregrasp_playback": asdict(playback),
            "approach_target": asdict(approach_target),
            "approach_playback": asdict(approach_playback),
            "gripper_opening_m": measured_opening,
            "lift_target": asdict(lift_target),
            "lift_playback": asdict(lift_playback),
            "lift_distance_m": lift_distance,
        }
        client.close()
        await loaded.world.pause_async()
    except Exception as error:
        result = {"success": False, "stage": result["stage"],
                  "reason": type(error).__name__, "message": str(error)}
        event(events, "RUN_FAILED", **result)
        raise
    finally:
        result_path.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
