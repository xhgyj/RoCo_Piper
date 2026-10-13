from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from math import pi
from types import MappingProxyType
from pathlib import Path
from typing import Any

from rocobrick.bricksim_topology import BaseplateSpec, BrickSimTopologyBundle, build_bricksim_topologies
from rocobrick.isaac_scene import IsaacSceneConfig
from rocobrick.initial_part_allocator import allocate_initial_slots
from rocobrick.models import Pose
from rocobrick.scene_bootstrap import RobotSpawnSpec
from rocobrick.stap_contract import StapPlan


class RuntimeHostInputError(ValueError):
    """The immutable plan/config pair cannot describe an Isaac scene."""


@dataclass(frozen=True)
class RuntimeHostInputs:
    topology: BrickSimTopologyBundle
    robots: Mapping[str, RobotSpawnSpec]
    scene_config: IsaacSceneConfig


def _seq(value: object, name: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise RuntimeHostInputError(f"{name} must contain {length} values")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise RuntimeHostInputError(f"{name} must contain numeric values") from exc
    if not all(isfinite(item) for item in result):
        raise RuntimeHostInputError(f"{name} must contain finite numeric values")
    return result


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeHostInputError(f"{name} must be an object")
    return value


def build_runtime_host_inputs(
    plan: StapPlan,
    config: Mapping[str, object],
    *,
    config_dir: Path,
    collision_filter_robot_id: str | None = None,
    collision_filter_instance_ids: tuple[str, ...] = (),
) -> RuntimeHostInputs:
    """Build BrickSim topology and Isaac spawn inputs from validated facts.

    The caller owns STAP parsing. This function only consumes the resulting
    plan and the explicit user config; it does not reorder or repair either.
    """
    task_config = _mapping(config.get("Task_Config"), "Task_Config")
    plate = _mapping(task_config.get("Base_Plate"), "Task_Config.Base_Plate")
    dimensions = plate.get("Dimension")
    if not isinstance(dimensions, (list, tuple)) or len(dimensions) != 2:
        raise RuntimeHostInputError("Task_Config.Base_Plate.Dimension must contain two values")
    try:
        size = (int(dimensions[0]), int(dimensions[1]))
    except (TypeError, ValueError) as exc:
        raise RuntimeHostInputError("baseplate dimensions must be integers") from exc
    if tuple(float(value) for value in dimensions) != tuple(size):
        raise RuntimeHostInputError("baseplate dimensions must be integral")
    color = plate.get("Color")
    if type(color) is not str or not color.strip():
        raise RuntimeHostInputError("baseplate color must be a non-empty string")
    baseplate_position = _seq(plate.get("Position"), "baseplate position", 3)
    baseplate_orientation = _seq(
        plate.get("Orientation"), "baseplate orientation", 4
    )
    topology = build_bricksim_topologies(plan, BaseplateSpec(size, color))

    robot_config = _mapping(config.get("Robot_Config"), "Robot_Config")
    raw_robots = robot_config.get("Robots")
    if not isinstance(raw_robots, list):
        raise RuntimeHostInputError("Robot_Config.Robots must be a list")
    robots: dict[str, RobotSpawnSpec] = {}
    for raw in raw_robots:
        item = _mapping(raw, "Robot_Config.Robots[]")
        robot_id = item.get("Name")
        if type(robot_id) is not str or not robot_id:
            raise RuntimeHostInputError("robot Name must be a non-empty string")
        if robot_id not in plan.robots:
            raise RuntimeHostInputError(f"robot {robot_id} is not present in plan")
        if robot_id in robots:
            raise RuntimeHostInputError(f"duplicate robot: {robot_id}")
        base = _mapping(item.get("Robot_Base_Frame"), f"robot {robot_id} base frame")
        orientation = _seq(base.get("Orientation"), f"robot {robot_id} orientation", 4)
        position = _seq(base.get("Position"), f"robot {robot_id} position", 3)
        prim_path = item.get("Robot_Prim_Path")
        usd_rel = item.get("Robot_USD_Path")
        joints = item.get("Joint_Order")
        if type(prim_path) is not str or not prim_path.startswith("/"):
            raise RuntimeHostInputError(f"robot {robot_id} prim path must be absolute")
        if type(usd_rel) is not str or not usd_rel:
            raise RuntimeHostInputError(f"robot {robot_id} USD path is required")
        if not isinstance(joints, list) or not joints or not all(type(j) is str and j for j in joints):
            raise RuntimeHostInputError(f"robot {robot_id} Joint_Order is invalid")
        cameras = _mapping(item.get("Camera_Config", {}), f"robot {robot_id} Camera_Config")
        sensor_paths = []
        for camera in cameras.values():
            camera_item = _mapping(camera, f"robot {robot_id} camera")
            sensor = camera_item.get("Prim_Path")
            if type(sensor) is not str or not sensor.startswith(prim_path + "/"):
                raise RuntimeHostInputError(f"robot {robot_id} camera prim is outside robot")
            sensor_paths.append(sensor)
        robots[robot_id] = RobotSpawnSpec(
            robot_id=robot_id,
            prim_path=prim_path,
            usd_path=str((config_dir / usd_rel).resolve()),
            base_pose=Pose("assembly_world", position, (orientation[1], orientation[2], orientation[3], orientation[0])),
            arm_joint_names=tuple(joints),
            sensor_prim_paths=tuple(sensor_paths),
        )
    if set(robots) != set(plan.robots):
        raise RuntimeHostInputError(f"robot config does not exactly match plan robots: {sorted(robots)}")
    if collision_filter_robot_id is not None and collision_filter_robot_id not in robots:
        raise RuntimeHostInputError(
            f"collision filter robot is not configured: {collision_filter_robot_id}"
        )
    if len(set(collision_filter_instance_ids)) != len(collision_filter_instance_ids):
        raise RuntimeHostInputError("collision filter instance ids must be unique")
    if any(type(item) is not str or not item for item in collision_filter_instance_ids):
        raise RuntimeHostInputError("collision filter instance ids must be non-empty strings")
    known_instance_ids = set(plan.goal_parts)
    unknown_filter_ids = sorted(set(collision_filter_instance_ids) - known_instance_ids)
    if unknown_filter_ids:
        raise RuntimeHostInputError(
            f"collision filter instances are absent from plan goal parts: {unknown_filter_ids}"
        )
    if collision_filter_instance_ids and collision_filter_robot_id is None:
        raise RuntimeHostInputError("collision filter instances require a robot id")

    # The topology stores numeric BrickSim ids; convert them to physical ids
    # without relying on task-specific brick names.
    loose_document = getattr(topology, "loose", {})
    topology_instances = getattr(topology, "instance_by_topology_id", {})
    loose_parts = (
        loose_document.get("parts", [])
        if isinstance(loose_document, Mapping)
        else []
    )
    loose_instance_ids = tuple(
        topology_instances[int(part["id"])]
        for part in loose_parts
        if isinstance(part, Mapping)
        and int(part["id"]) in topology_instances
    )
    robot_base_xy = {
        robot_id: tuple(spec.base_pose.position[:2])
        for robot_id, spec in robots.items()
    }
    preferred_robots: dict[str, str] = {}
    for instance_id in loose_instance_ids:
        task = next(
            (
                item
                for item in plan.tasks.values()
                if item.physical_instance_id == instance_id
            ),
            None,
        )
        if task is not None and task.primary_robot in robot_base_xy:
            preferred_robots[instance_id] = task.primary_robot
        else:
            preferred_robots[instance_id] = min(
                robot_base_xy,
                key=lambda robot: (
                    (robot_base_xy[robot][0] - baseplate_position[0]) ** 2
                    + (robot_base_xy[robot][1] - baseplate_position[1]) ** 2,
                    robot,
                ),
            )
    allocations = allocate_initial_slots(
        loose_instance_ids,
        preferred_robots=preferred_robots,
        robot_base_xy=robot_base_xy,
        board_center_xy=(baseplate_position[0], baseplate_position[1]),
        board_half_extent_xy=(
            max(0.08, size[0] * 0.008 / 2),
            max(0.08, size[1] * 0.008 / 2),
        ),
    )

    env = _mapping(config.get("Env_Config"), "Env_Config")
    storage = _mapping(env.get("Storage_Config"), "Env_Config.Storage_Config")
    workspace_path = storage.get("Prim_Path", "/World/LegoWorkspace")
    if type(workspace_path) is not str or not workspace_path.startswith("/"):
        raise RuntimeHostInputError("workspace path must be absolute")
    scene_config = IsaacSceneConfig(
        baseplate_position,
        baseplate_orientation,
        workspace_path,
        _seq(storage.get("Size"), "workspace size", 3),
        _seq(storage.get("Position"), "workspace position", 3),
        _seq(storage.get("Orientation"), "workspace orientation", 4),
        60,
        collision_filter_robot_id,
        tuple(collision_filter_instance_ids),
        {item: allocation.position_xy_m for item, allocation in allocations.items()},
        {
            item: float(plan.goal_parts[item].orientation_quarter_turns) * pi / 2.0
            for item in allocations
        },
        {item: dict(allocation.reason) for item, allocation in allocations.items()},
    )
    return RuntimeHostInputs(topology, MappingProxyType(dict(robots)), scene_config)
