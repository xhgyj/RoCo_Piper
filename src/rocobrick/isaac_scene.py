"""Concrete BrickSim scene loader with measured post-load verification."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import atan2, cos, hypot, inf, isfinite, sin
from pathlib import Path

from rocobrick.bricksim_topology import BrickSimTopologyBundle
from rocobrick.feeding_platforms import FeedingPlatform
from rocobrick.models import Pose
from rocobrick.scene_bootstrap import (
    LoadedPart,
    LoadedRobot,
    RobotSpawnSpec,
    SceneBootstrapError,
)
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
    # Allocated XY positions keep the imported BrickSim shape and stud orientation.
    initial_part_positions_xy: Mapping[str, tuple[float, float]] = field(
        default_factory=dict
    )
    initial_part_yaws_rad: Mapping[str, float] = field(default_factory=dict)
    initial_part_allocation: Mapping[str, Mapping[str, object]] = field(
        default_factory=dict
    )
    feeding_platforms: tuple[FeedingPlatform, ...] = ()
    initial_part_support_top_z_m: Mapping[str, float] = field(default_factory=dict)


def _create_feeding_platforms(
    stage: object, platforms: tuple[FeedingPlatform, ...]
) -> None:
    """Author whole static support boxes that the host measures as colliders."""
    from pxr import Gf, UsdGeom, UsdPhysics

    for platform in platforms:
        if stage.GetPrimAtPath(platform.prim_path).IsValid():
            raise SceneBootstrapError(
                f"feeding platform prim already exists: {platform.prim_path}"
            )
        cube = UsdGeom.Cube.Define(stage, platform.prim_path)
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(
            Gf.Vec3d(*platform.center_xy_m, platform.top_z_m - platform.thickness_m / 2)
        )
        cube.AddScaleOp().Set(Gf.Vec3d(*platform.size_xy_m, platform.thickness_m))
        cube.CreateDisplayColorAttr([Gf.Vec3f(0.35, 0.45, 0.38)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)


@dataclass(frozen=True)
class IsaacLoadedScene:
    app: object
    world: object
    robots: Mapping[str, object]
    measured_robots: Mapping[str, LoadedRobot]
    parts: Mapping[str, LoadedPart]
    topology_paths: Mapping[int, str]
    collision_filter_pairs: tuple[tuple[str, str], ...] = ()
    disabled_camera_collisions: tuple[str, ...] = ()
    hidden_camera_visuals: tuple[str, ...] = ()


# The vendor wrist-camera assembly (the mid stand and the d435 body) is a
# safety-margin hazard, not a task sensor: no skill, observation or gate in
# this repository consumes wrist-camera images, and in the simulator the stand
# is collided as the convex hull of a 97 x 48 x 74 mm bracket that fills the
# narrow space over a placement.  Measured on task-c-top-offset12 (2026-10-14):
# the support pre-phase staged the two wrists with the stand interpenetrating
# the peer's finger by 1.87 mm, piper_0's arm controller then aborted on goal
# tolerance, and the collision-free-vs-simulator disagreement came entirely
# from hulling that bracket; with these colliders disabled the same station
# clears by more than 25 mm.  The vendor USD stays byte-identical (its SHA-256
# is bound by the TCP authorization artifact), so the collision is disabled
# here, in our own scene bootstrap, and recorded in the session evidence.
WRIST_CAMERA_COLLISION_LINKS = ("camera_stand_link", "camera_link")


def _disable_camera_collisions(
    stage: object, *, robot: RobotSpawnSpec
) -> tuple[str, ...]:
    """Disable the wrist-camera colliders of one loaded robot.

    Every collision prim under the configured camera links is switched off
    through ``UsdPhysics.CollisionAPI.collisionEnabled``.  The visual geometry
    is hidden as well by :func:`_hide_camera_visuals`, so the headed viewport
    shows the machine the physics models instead of a camera that can no
    longer collide.

    The vendor asset instances each link's ``collisions`` subtree, and instance
    proxies neither report their applied schemas nor accept authoring -- a plain
    walk finds nothing and the colliders keep colliding.  The subtree is
    therefore materialized (``SetInstanceable(False)``) first, exactly like
    ``materialize_robot_visuals`` does for the visuals, and every edit lands in
    the session layer so the vendor USD stays byte-identical.

    A link that carries no collision prim is a contract inconsistency and fails
    the bootstrap closed.

    Returns:
        The disabled collision prim paths, in evidence order.
    """
    from pxr import Usd, UsdPhysics

    candidates: list[tuple[str, object]] = []
    collider_paths: list[str] = []
    for link in WRIST_CAMERA_COLLISION_LINKS:
        link_prim = stage.GetPrimAtPath(f"{robot.prim_path}/{link}")
        if not link_prim.IsValid():
            # A vendor variant without the wrist camera has nothing to disable.
            continue
        link_colliders = [
            str(prim.GetPath())
            for prim in Usd.PrimRange(link_prim, Usd.TraverseInstanceProxies())
            if prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        if not link_colliders:
            raise SceneBootstrapError(
                f"wrist camera link {link_prim.GetPath()} carries no collider "
                "to disable"
            )
        collider_paths.extend(link_colliders)
        collisions = link_prim.GetChild("collisions")
        if collisions.IsValid() and collisions.IsInstance():
            candidates.append((str(link_prim.GetPath()), collisions))
    if not candidates and not collider_paths:
        return ()
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for link_path, collisions in candidates:
            collisions.SetInstanceable(False)
            print(
                f"[rocobrick] materialized collision instance for {link_path}",
                flush=True,
            )
        for path in collider_paths:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid() or not prim.HasAPI(UsdPhysics.CollisionAPI):
                raise SceneBootstrapError(
                    f"wrist camera collider {path} vanished after materializing "
                    "its collision subtree"
                )
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
    return tuple(collider_paths)


def _hide_camera_visuals(stage: object, *, robot: RobotSpawnSpec) -> tuple[str, ...]:
    """Hide the wrist-camera visual geometry of one loaded robot.

    The camera colliders are switched off (see :func:`_disable_camera_collisions`),
    so the camera must not stay drawn either: the headed viewport is the review
    surface for every run, and a camera that is still rendered reads as a part
    that still collides.  Visibility is authored on the link prim and inherits
    down to its visual subtree; the edit lands in the session layer, so the
    vendor USD stays byte-identical and one visibility reset restores the view.

    Returns:
        The hidden link prim paths, in evidence order.
    """
    from pxr import Usd, UsdGeom

    targets: list[object] = []
    for link in WRIST_CAMERA_COLLISION_LINKS:
        prim = stage.GetPrimAtPath(f"{robot.prim_path}/{link}")
        if prim.IsValid():
            targets.append(prim)
    if not targets:
        return ()
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for prim in targets:
            UsdGeom.Imageable(prim).MakeInvisible()
    return tuple(str(prim.GetPath()) for prim in targets)


def _apply_collision_filters(
    stage: object,
    *,
    robot: RobotSpawnSpec,
    part_paths: Mapping[str, str],
    instance_ids: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    """Authorize and record the pick-target/gripper contact pairs.

    No ``UsdPhysics.FilteredPairsAPI`` is applied: a PhysX filter between
    the gripper and the pick target DISABLES their contact, which makes
    every physical grasp of the target impossible (live-evidenced: the pads
    closed to the commanded opening with the brick untouched and the
    follow gate failed with "object did not lift").  Isaac must keep real
    collisions everywhere; unwanted contacts stop the motion and fail the
    operation closed.  This only validates that every authorized pick
    target (and the gripper links that may touch it) is actually loaded,
    and returns the authorized pairs for the session evidence.
    """
    if not instance_ids:
        return ()

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
            pairs.append((link_path, part_path))
    return tuple(pairs)


# Observer framing constants. The camera is not a calibrated robot sensor, so
# the numbers only decide how much of the loaded scene stays in frame.
_OBSERVER_FOCAL_LENGTH_MM = 18.0
_OBSERVER_HORIZONTAL_APERTURE_MM = 20.955
_OBSERVER_ASPECT = 720.0 / 1280.0
_OBSERVER_DIRECTION = (0.0, 0.83, 0.55)
_OBSERVER_FRAME_HEIGHT_M = 0.55
_OBSERVER_FIT_MARGIN = 1.12
# Zero disables camera-side depth of field, independently of render settings.
_OBSERVER_FSTOP = 0.0
# Fallback pose for a stage whose focus prims cannot be measured.
_OBSERVER_FALLBACK = ((0.0, 0.60, 0.42), (0.0, -0.10, 0.16))
# Vendor assets embed a large reference plane: the piper USD's ``agx_arm``
# root measures 50 x 50 m, and framing it would push the observer tens of
# metres away from the workspace. Bounds wider than this are scene artefacts,
# so the observer uses the layout's own points for those participants.
_OBSERVER_MAX_FOCUS_EXTENT_M = 2.0


def _observer_focus_bounds(
    stage: object,
    paths: Sequence[str],
    points: Sequence[tuple[float, float, float]] = (),
) -> (
    tuple[tuple[float, float, float], tuple[float, float, float], tuple[str, ...]]
    | None
):
    """Measure the world-space bounds of what the observer must frame.

    Args:
        stage: Open USD stage.
        paths: Prim paths whose world bounds describe the workspace.
        points: Explicit world points (robot base poses) that join the box
            even when no prim of that participant can be measured.

    Returns:
        Lower corner, upper corner and the skipped artefact prim paths, or
        None when nothing is measurable.
    """
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    lower = [inf, inf, inf]
    upper = [-inf, -inf, -inf]
    skipped: list[str] = []
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if aligned.IsEmpty():
            continue
        minimum = aligned.GetMin()
        maximum = aligned.GetMax()
        extent = max(float(maximum[axis]) - float(minimum[axis]) for axis in range(3))
        if extent > _OBSERVER_MAX_FOCUS_EXTENT_M:
            skipped.append(path)
            continue
        for axis in range(3):
            lower[axis] = min(lower[axis], float(minimum[axis]))
            upper[axis] = max(upper[axis], float(maximum[axis]))
    for point in points:
        for axis in range(3):
            lower[axis] = min(lower[axis], float(point[axis]))
            upper[axis] = max(upper[axis], float(point[axis]))
    if not all(isfinite(value) for value in (*lower, *upper)):
        return None
    return (
        (lower[0], lower[1], lower[2]),
        (upper[0], upper[1], upper[2]),
        tuple(skipped),
    )


def _observer_pose(
    lower: tuple[float, float, float],
    upper: tuple[float, float, float],
    *,
    full_height: bool = False,
) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    """Place the observer so the measured bounds fit the viewport.

    The frame is fitted on the scene footprint plus a bounded height band:
    a robot arm held high may leave the top of the frame, the assembly,
    feeder and storage area may not.

    Returns:
        Camera position, camera target and their distance in metres.
    """
    width = upper[0] - lower[0]
    depth = upper[1] - lower[1]
    height = upper[2] - lower[2]
    if not full_height:
        height = min(height, _OBSERVER_FRAME_HEIGHT_M)
    target = (
        (lower[0] + upper[0]) / 2.0,
        (lower[1] + upper[1]) / 2.0,
        lower[2] + height / 2.0,
    )
    tan_horizontal = 0.5 * _OBSERVER_HORIZONTAL_APERTURE_MM / _OBSERVER_FOCAL_LENGTH_MM
    tan_vertical = tan_horizontal * _OBSERVER_ASPECT
    elevation = atan2(_OBSERVER_DIRECTION[2], _OBSERVER_DIRECTION[1])
    horizontal_half = 0.5 * hypot(width, depth)
    vertical_half = 0.5 * height * cos(elevation) + 0.5 * depth * sin(elevation)
    distance = _OBSERVER_FIT_MARGIN * max(
        horizontal_half / tan_horizontal, vertical_half / tan_vertical
    )
    norm = hypot(*_OBSERVER_DIRECTION)
    position = tuple(
        target[axis] + distance * _OBSERVER_DIRECTION[axis] / norm for axis in range(3)
    )
    return position, target, distance


def _set_initial_viewport_camera(
    stage: object,
    focus_paths: Sequence[str] = (),
    focus_points: Sequence[tuple[float, float, float]] = (),
) -> None:
    """Frame the pick and handover workspace in the headed observer viewport.

    The observer pose is fitted to the measured world bounds of the feeder
    platforms and loaded parts, plus the robot base points, so a scene layout
    change moves the observer with it. The previously fixed pose left the
    scaled2 observer looking at its own foreground geometry (live evidence,
    task-c-top-offset13s-headed 2026-10-14).
    """
    import os

    if os.environ.get("ROCOBRICK_HEADED") != "1":
        return
    from omni.kit.viewport.utility import get_active_viewport
    from omni.kit.viewport.utility.camera_state import ViewportCameraState
    from pxr import Gf, Sdf, UsdGeom

    # The observer records the workspace with a fixed camera on a simulator
    # that advances faster than real time, so the cinematic post effects only
    # smear the recording. Suppress them here as well as on the Kit command
    # line: the renderer writes its own defaults while it initializes, which
    # overwrote the launch-time values (live evidence,
    # task-c-top-offset13v-headed 2026-10-14).
    from carb.settings import get_settings

    settings = get_settings()
    settings.set_bool("/rtx/post/dof/enabled", False)
    settings.set_bool("/rtx/post/motionblur/enabled", False)

    camera_path = "/World/RoCoBrickInitialCamera"
    camera = UsdGeom.Camera.Define(stage, camera_path)
    bounds = _observer_focus_bounds(stage, focus_paths, focus_points)
    if bounds is None:
        position, target = _OBSERVER_FALLBACK
        focus_distance_m = float(
            hypot(
                position[0] - target[0],
                position[1] - target[1],
                position[2] - target[2],
            )
        )
        skipped: tuple[str, ...] = ()
    else:
        lower, upper, skipped = bounds
        overview = os.environ.get("ROCOBRICK_OBSERVER_OVERVIEW") == "1"
        if overview:
            # Include a conservative motion envelope about each robot base;
            # vendor root bounds contain an unrelated 50 m reference plane.
            for point in focus_points:
                lower = tuple(
                    min(lower[i], point[i] - (0.65 if i < 2 else 0.0)) for i in range(3)
                )
                upper = tuple(
                    max(upper[i], point[i] + (0.65 if i < 2 else 0.85))
                    for i in range(3)
                )
        position, target, focus_distance_m = _observer_pose(
            lower, upper, full_height=overview
        )
    # Pull back modestly, without changing the viewing direction or target.
    # The larger overview mode already carries a conservative motion envelope.
    if os.environ.get("ROCOBRICK_OBSERVER_PULLBACK") == "1":
        position = tuple(target[i] + 1.25 * (position[i] - target[i]) for i in range(3))
        focus_distance_m *= 1.25
    focal_length_mm = _OBSERVER_FOCAL_LENGTH_MM
    camera.CreateFocalLengthAttr(focal_length_mm)
    camera.CreateFocusDistanceAttr(focus_distance_m)
    camera.CreateFStopAttr(_OBSERVER_FSTOP)
    # USD's default near plane is too distant for a metre-scale workspace.
    # Keep nearby platforms/tools from being sliced by the view frustum.
    camera.CreateClippingRangeAttr((0.01, 100.0))
    # A newly defined USD camera has no Kit center-of-interest value.  The
    # Isaac viewport camera-state helper assumes this attribute exists when it
    # rotates/moves the camera; seed it before using that helper.
    center_of_interest = camera.GetPrim().CreateAttribute(
        "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d
    )
    center_of_interest.Set(Gf.Vec3d(0.0, 0.0, -1.0))
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("headed Isaac run has no active viewport")
    viewport.camera_path = camera_path
    viewport.resolution = (1920, 1080)
    camera_state = ViewportCameraState(camera_path=camera_path, viewport=viewport)
    camera_state.set_position_world(Gf.Vec3d(*position), True)
    camera_state.set_target_world(Gf.Vec3d(*target), True)
    print(
        "[rocobrick] initial viewport camera configured "
        f"path={camera_path} position={position} target={target} "
        f"focal_length_mm={focal_length_mm} focus_distance_m={focus_distance_m} "
        f"skipped_focus_prims={list(skipped)}",
        flush=True,
    )


def _xyzw(wxyz: object) -> tuple[float, float, float, float]:
    values = tuple(float(value) for value in wxyz)
    return values[1], values[2], values[3], values[0]


def _pose(prim_path: str, frame_id: str) -> Pose:
    from isaacsim.core.prims import SingleXFormPrim

    position, orientation_wxyz = SingleXFormPrim(prim_path=prim_path).get_world_pose()
    return Pose(
        frame_id, tuple(float(value) for value in position), _xyzw(orientation_wxyz)
    )


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
    from bricksim.core import (
        are_parts_connected,
        arrange_parts_in_workspace,
        import_lego,
    )
    from isaacsim.core.api.world import World
    from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
    from isaacsim.core.utils.stage import (
        add_reference_to_stage,
        get_current_stage,
        open_stage_async,
    )
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
        raise SceneBootstrapError(
            f"workspace prim does not exist: {config.workspace_path}"
        )
    workspace.GetAttribute("xformOp:scale").Set(Gf.Vec3d(*config.workspace_scale))
    workspace.GetAttribute("xformOp:translate").Set(
        Gf.Vec3d(*config.workspace_position_m)
    )
    workspace.GetAttribute("xformOp:orient").Set(
        Gf.Quatd(*config.workspace_orientation_wxyz)
    )

    initial_paths, initial_connections = import_lego(
        json=dict(topology.initial),
        env_id=-1,
        ref_pos=config.baseplate_position_m,
        ref_rot=config.baseplate_orientation_wxyz,
    )
    expected_initial_ids = {part["id"] for part in topology.initial["parts"]}
    if set(initial_paths) != expected_initial_ids:
        raise SceneBootstrapError("BrickSim did not import every initial topology part")
    expected_connection_ids = {
        connection["id"] for connection in topology.initial["connections"]
    }
    if set(initial_connections) != expected_connection_ids:
        raise SceneBootstrapError("BrickSim did not import every initial connection")

    _create_feeding_platforms(stage, config.feeding_platforms)
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
    if config.initial_part_positions_xy:
        for instance_id, position_xy in config.initial_part_positions_xy.items():
            topology_id = topology.topology_id_by_instance.get(instance_id)
            if topology_id is None or topology_id not in loose_paths:
                raise SceneBootstrapError(
                    f"allocator target is not a loose loaded part: {instance_id}"
                )
            if not isinstance(position_xy, (tuple, list)) or len(position_xy) != 2:
                raise SceneBootstrapError(
                    f"allocator XY position is invalid for {instance_id}"
                )
            prim = SingleXFormPrim(prim_path=loose_paths[topology_id])
            part_payload = next(
                (
                    item.get("payload", {})
                    for item in topology.goal.get("parts", [])
                    if isinstance(item, Mapping) and item.get("id") == topology_id
                ),
                {},
            )
            height_units = part_payload.get("H", 3)
            if isinstance(height_units, bool) or not isinstance(
                height_units, (int, float)
            ):
                height_units = 3
            # BrickSim H counts 3.2 mm layers; the stud adds 1.7 mm.
            height_m = float(height_units) * 0.0032 + 0.0017
            support_z = config.initial_part_support_top_z_m.get(
                instance_id, float(config.baseplate_position_m[2])
            )
            yaw = float(config.initial_part_yaws_rad.get(instance_id, 0.0))
            prim.set_world_pose(
                position=np.asarray(
                    (
                        float(position_xy[0]),
                        float(position_xy[1]),
                        support_z + height_m / 2.0,
                    )
                ),
                orientation=np.asarray((cos(yaw / 2.0), 0.0, 0.0, sin(yaw / 2.0))),
            )

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
        if os.environ.get("ROCOBRICK_HEADED") == "1":
            from rocobrick.adapters.isaac.visual_instances import (
                materialize_robot_visuals,
            )

            repaired = materialize_robot_visuals(stage, spec.prim_path)
            print(
                f"[rocobrick] materialized robot visual instances: {repaired}",
                flush=True,
            )
        articulation = SingleArticulation(prim_path=spec.prim_path, name=robot_id)
        world.scene.add(articulation)
        articulations[robot_id] = articulation
        for sensor_path in spec.sensor_prim_paths:
            if not sensor_path.startswith(spec.prim_path + "/"):
                raise SceneBootstrapError(
                    f"sensor for {robot_id} is outside its robot prim"
                )
            if not stage.GetPrimAtPath(sensor_path).IsValid():
                raise SceneBootstrapError(
                    f"configured sensor missing for {robot_id}: {sensor_path}"
                )

    disabled_camera_collisions: list[str] = []
    hidden_camera_visuals: list[str] = []
    for robot_id, spec in robots.items():
        disabled_camera_collisions.extend(_disable_camera_collisions(stage, robot=spec))
        hidden_camera_visuals.extend(_hide_camera_visuals(stage, robot=spec))

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
    _set_initial_viewport_camera(
        stage,
        (
            *(platform.prim_path for platform in config.feeding_platforms),
            *initial_paths.values(),
            *loose_paths.values(),
        ),
        tuple(
            tuple(float(value) for value in spec.base_pose.position)
            for spec in robots.values()
        ),
    )
    for _ in range(max(2, config.physics_hz // 10)):
        await app.next_update_async()

    measured_robots: dict[str, LoadedRobot] = {}
    for robot_id, articulation in articulations.items():
        actual_names = tuple(articulation.dof_names or ())
        configured = robots[robot_id].arm_joint_names
        missing = [name for name in configured if name not in actual_names]
        if missing:
            raise SceneBootstrapError(
                f"{robot_id} missing joints after initialization: {missing}"
            )
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
        app,
        world,
        articulations,
        measured_robots,
        parts,
        topology_paths,
        collision_filter_pairs,
        tuple(disabled_camera_collisions),
        tuple(hidden_camera_visuals),
    )
