"""Concrete BrickSim scene loader with measured post-load verification."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from rocobrick.bricksim_topology import BrickSimTopologyBundle
from rocobrick.models import Pose
from rocobrick.scene_bootstrap import LoadedPart, LoadedRobot, RobotSpawnSpec, SceneBootstrapError
from rocobrick.stap_contract import StapPlan


@dataclass(frozen=True)
class IsaacSceneConfig:
    baseplate_position_m: tuple[float, float, float]
    baseplate_orientation_wxyz: tuple[float, float, float, float]
    workspace_path: str
    workspace_scale: tuple[float, float, float]
    workspace_position_m: tuple[float, float, float]
    workspace_orientation_wxyz: tuple[float, float, float, float]
    physics_hz: int
    collision_filter_robot_id: str | None = None
    collision_filter_instance_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class IsaacLoadedScene:
    app: object
    world: object
    robots: Mapping[str, object]
    measured_robots: Mapping[str, LoadedRobot]
    parts: Mapping[str, LoadedPart]
    topology_paths: Mapping[int, str]
    collision_filter_pairs: tuple[tuple[str, str], ...] = ()


def _apply_collision_filters(
    stage: object,
    *,
    robot: RobotSpawnSpec,
    part_paths: Mapping[str, str],
    instance_ids: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    """Filter only authorized target-brick/gripper collisions."""
    if not instance_ids:
        return ()
    from pxr import Sdf, UsdPhysics

    contact_links = ("gripper_base", "gripper_link1", "gripper_link2")
    pairs: list[tuple[str, str]] = []
    for instance_id in instance_ids:
        part_path = part_paths.get(instance_id)
        if not isinstance(part_path, str) or not part_path:
            raise SceneBootstrapError(
                f"collision filter target is not a loaded instance: {instance_id}"
            )
        part = stage.GetPrimAtPath(part_path)
        if not part.IsValid():
            raise SceneBootstrapError(
                f"collision filter target prim is invalid: {part_path}"
            )
        for link in contact_links:
            link_path = f"{robot.prim_path}/{link}"
            gripper = stage.GetPrimAtPath(link_path)
            if not gripper.IsValid():
                raise SceneBootstrapError(
                    f"collision filter gripper link prim is invalid: {link_path}"
                )
            UsdPhysics.FilteredPairsAPI.Apply(gripper).CreateFilteredPairsRel().AddTarget(
                Sdf.Path(part_path)
            )
            UsdPhysics.FilteredPairsAPI.Apply(part).CreateFilteredPairsRel().AddTarget(
                Sdf.Path(link_path)
            )
            pairs.append((link_path, part_path))
    return tuple(pairs)


def _xyzw(wxyz: object) -> tuple[float, float, float, float]:
    values = tuple(float(value) for value in wxyz)
    return values[1], values[2], values[3], values[0]


def _pose(prim_path: str, frame_id: str) -> Pose:
    from isaacsim.core.prims import SingleXFormPrim

    position, orientation_wxyz = SingleXFormPrim(prim_path=prim_path).get_world_pose()
    return Pose(frame_id, tuple(float(value) for value in position), _xyzw(orientation_wxyz))


async def load_verified_scene(
    plan: StapPlan,
    topology: BrickSimTopologyBundle,
    robots: Mapping[str, RobotSpawnSpec],
    config: IsaacSceneConfig,
) -> IsaacLoadedScene:
    """Load physical assets, then verify bindings from measured simulator state.

    This function contains no articulation action calls.  It is therefore
    suitable for the pre-motion gate of a run.
    """
    import numpy as np
    import omni.kit.app
    from bricksim.assets import DEFAULT_STAGE_PATH
    from bricksim.core import are_parts_connected, arrange_parts_in_workspace, import_lego
    from isaacsim.core.api.world import World
    from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage, open_stage_async
    from pxr import Gf

    if set(robots) != set(plan.robots):
        raise SceneBootstrapError("robot specs must exactly match STAP plan robots")
    app = omni.kit.app.get_app()
    if World._world_initialized:
        World.clear_instance()
    await open_stage_async(str(DEFAULT_STAGE_PATH))
    world = World(backend="numpy", device="cpu", physics_prim_path="/physicsScene")
    await world.initialize_simulation_context_async()
    world.get_physics_context().set_physics_dt(1.0 / config.physics_hz)
    stage = get_current_stage()
    workspace = stage.GetPrimAtPath(config.workspace_path)
    if not workspace.IsValid():
        raise SceneBootstrapError(f"workspace prim does not exist: {config.workspace_path}")
    workspace.GetAttribute("xformOp:scale").Set(Gf.Vec3d(*config.workspace_scale))
    workspace.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*config.workspace_position_m))
    workspace.GetAttribute("xformOp:orient").Set(Gf.Quatd(*config.workspace_orientation_wxyz))

    initial_paths, initial_connections = import_lego(
        json=dict(topology.initial), env_id=-1,
        ref_pos=config.baseplate_position_m,
        ref_rot=config.baseplate_orientation_wxyz,
    )
    expected_initial_ids = {part["id"] for part in topology.initial["parts"]}
    if set(initial_paths) != expected_initial_ids:
        raise SceneBootstrapError("BrickSim did not import every initial topology part")
    expected_connection_ids = {connection["id"] for connection in topology.initial["connections"]}
    if set(initial_connections) != expected_connection_ids:
        raise SceneBootstrapError("BrickSim did not import every initial connection")

    loose_paths, loose_connections = import_lego(json=dict(topology.loose), env_id=-1)
    if loose_connections:
        raise SceneBootstrapError("loose-part import unexpectedly created a connection")
    expected_loose_ids = {part["id"] for part in topology.loose["parts"]}
    if set(loose_paths) != expected_loose_ids:
        raise SceneBootstrapError("BrickSim did not import every loose topology part")
    arranged, unarranged = arrange_parts_in_workspace(
        workspace_path=config.workspace_path,
        parts_to_arrange=[loose_paths[index] for index in sorted(loose_paths)],
        structure_ids=[index for index in sorted(loose_paths)],
    )
    if unarranged or set(arranged) != set(loose_paths.values()):
        raise SceneBootstrapError(f"failed to arrange loose parts: {unarranged}")

    articulations: dict[str, object] = {}
    for robot_id in sorted(robots):
        spec = robots[robot_id]
        usd_path = Path(spec.usd_path).resolve()
        if not usd_path.is_file():
            raise SceneBootstrapError(f"robot USD does not exist: {usd_path}")
        add_reference_to_stage(str(usd_path), spec.prim_path)
        xf = SingleXFormPrim(prim_path=spec.prim_path, name=f"{robot_id}_root")
        xyzw = spec.base_pose.orientation_xyzw
        xf.set_world_pose(
            position=np.asarray(spec.base_pose.position),
            orientation=np.asarray((xyzw[3], xyzw[0], xyzw[1], xyzw[2])),
        )
        articulation = SingleArticulation(prim_path=spec.prim_path, name=robot_id)
        world.scene.add(articulation)
        articulations[robot_id] = articulation
        for sensor_path in spec.sensor_prim_paths:
            if not sensor_path.startswith(spec.prim_path + "/"):
                raise SceneBootstrapError(f"sensor for {robot_id} is outside its robot prim")
            if not stage.GetPrimAtPath(sensor_path).IsValid():
                raise SceneBootstrapError(f"configured sensor missing for {robot_id}: {sensor_path}")

    collision_filter_pairs: tuple[tuple[str, str], ...] = ()
    if config.collision_filter_instance_ids:
        robot_id = config.collision_filter_robot_id
        if not isinstance(robot_id, str) or robot_id not in robots:
            raise SceneBootstrapError(
                "collision filter requires a configured executor robot"
            )
        pre_filter_paths = {**initial_paths, **loose_paths}
        instance_paths = {
            instance_id: pre_filter_paths[topology_id]
            for topology_id, instance_id in topology.instance_by_topology_id.items()
            if topology_id in pre_filter_paths
        }
        collision_filter_pairs = _apply_collision_filters(
            stage,
            robot=robots[robot_id],
            part_paths={
                instance_id: instance_paths[instance_id]
                for instance_id in config.collision_filter_instance_ids
                if instance_id in instance_paths
            },
            instance_ids=config.collision_filter_instance_ids,
        )

    await world.reset_async()
    await world.play_async()
    for _ in range(max(2, config.physics_hz // 10)):
        await app.next_update_async()

    measured_robots: dict[str, LoadedRobot] = {}
    for robot_id, articulation in articulations.items():
        actual_names = tuple(articulation.dof_names or ())
        configured = robots[robot_id].arm_joint_names
        missing = [name for name in configured if name not in actual_names]
        if missing:
            raise SceneBootstrapError(f"{robot_id} missing joints after initialization: {missing}")
        raw = articulation.get_joint_positions()
        values = tuple(float(raw[actual_names.index(name)]) for name in configured)
        if not all(np.isfinite(values)):
            raise SceneBootstrapError(f"{robot_id} returned non-finite measured joints")
        measured_robots[robot_id] = LoadedRobot(robot_id, configured, values)

    topology_paths = {**initial_paths, **loose_paths}
    parts: dict[str, LoadedPart] = {}
    for topology_id, instance_id in topology.instance_by_topology_id.items():
        path = topology_paths.get(topology_id)
        if path is None or not stage.GetPrimAtPath(path).IsValid():
            raise SceneBootstrapError(f"missing physical prim for {instance_id}")
        source = plan.goal_parts[instance_id]
        parts[instance_id] = LoadedPart(
            instance_id, path, source.brick_id, _pose(path, "assembly_world")
        )

    for connection in topology.expected_initial_connections:
        stud_path = topology_paths[connection["stud_id"]]
        hole_path = topology_paths[connection["hole_id"]]
        if not are_parts_connected(stud_path, hole_path):
            raise SceneBootstrapError(
                f"initial connection missing: {connection['stud_id']} -> {connection['hole_id']}"
            )
    return IsaacLoadedScene(
        app, world, articulations, measured_robots, parts, topology_paths,
        collision_filter_pairs,
    )
