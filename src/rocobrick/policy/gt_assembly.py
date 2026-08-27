"""Privileged pick preparation and local expert for one-step assembly."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from rocobrick.backends.bricksim import (
    BrickSimConnectionGoal,
    BrickSimRobotBackend,
    BrickSimSuccessCheck,
    BrickSimWorldModel,
)
from rocobrick.controllers.motion import MotionConfig
from rocobrick.execution.types import ExecutionError, HeldObject
from rocobrick.policy.assembly_control import (
    AssemblyExpertConfig,
)
from rocobrick.policy.assembly_runtime import GTAssemblyRuntime
from rocobrick.safety.checks import ForceGuard
from rocobrick.skills.assemble import AssembleRequest, AssembleSkill
from rocobrick.skills.pick import GraspCandidate, PickRequest, PickSkill

SAFE_HEIGHT = 0.06
PICK_APPROACH_HEIGHT = 0.06
PICK_TCP_HEIGHT = 0.002
TRANSIT_HEIGHT = 0.10
POST_ASSEMBLY_RETREAT = 0.06
GOAL_POSITION_CONSISTENCY = 5e-4
GOAL_ROTATION_CONSISTENCY = np.deg2rad(0.5)
SAFE_IK_POSITION_TOLERANCE = 0.004
SAFE_IK_ROTATION_TOLERANCE = np.deg2rad(3.0)
FREE_MOTION_POSITION_TOLERANCE = 0.005
FREE_MOTION_ROTATION_TOLERANCE = np.deg2rad(5.0)
FREE_MOTION_JOINT_TOLERANCE = 0.04
FREE_MOTION_MAX_ARM_STEP = 0.02
FREE_MOTION_MAX_GRIPPER_STEP = 0.001
FREE_MOTION_TIMEOUT = 600
FREE_MOTION_SETTLE_STEPS = 3
GRIPPER_MAX_STEPS = 90
GRIPPER_MIN_STEPS = 6
GRIPPER_STABLE_STEPS = 4
GRIPPER_STALL_DELTA = 0.00015
GRIPPER_OPEN_TOLERANCE = 0.0015
GRIPPER_CONTACT_WIDTH_TOLERANCE = 0.004
GRIPPER_OPEN_MARGIN_PER_FINGER = 0.006
GRIPPER_MAX_JOINT_OPENING = 0.045
MAX_GRASP_FINGER_AXIS_DRIFT = 0.004
MAX_GRASP_VERTICAL_DRIFT = 0.004
MAX_INITIAL_GRASP_VERTICAL_SETTLING = 0.008
MAX_GRASP_ROTATION_DRIFT = np.deg2rad(5.0)
GRASP_DRIFT_COMPARISON_TOLERANCE = 0.0002
FREE_MOTION_TRANSLATION_STEP = 0.003
FREE_MOTION_COMMAND_LEAD = 0.008
HELD_TRANSLATION_STEP = 0.002
HELD_COMMAND_LEAD = 0.004
COMMAND_LEAD_RAMP_STEPS = 15
BRICK_UNIT_LENGTH = 0.008
SAME_LEVEL_Z_TOLERANCE = 0.0048
MIN_GRIPPER_SIDE_CLEARANCE = 0.004
GRIPPER_CLEARANCE_SCORE_CAP = 0.012
LONG_BRICK_ASPECT_RATIO = 3.0
ALIGNMENT_IK_PLANNING_STEP = np.deg2rad(2.0)


@dataclass(frozen=True)
class GTConnection:
    """One simulator-ground-truth target-to-reference connection."""

    reference_id: int
    target_id: int
    reference_path: str
    target_path: str
    stud_iface: int
    hole_iface: int
    offset: tuple[int, int]
    yaw: int
    overlap_studs: int


@dataclass(frozen=True)
class GTAssemblyTask:
    """A single target brick and every connection it must establish."""

    target_id: int
    target_path: str
    dimensions: dict[str, int]
    primary: GTConnection
    additional: tuple[GTConnection, ...]

    @property
    def connections(self) -> tuple[GTConnection, ...]:
        """Return primary and additional connections in verification order."""
        return (self.primary, *self.additional)


@dataclass(frozen=True)
class SafeStart:
    """Verified physical grasp at the start of the assembly trajectory."""

    task: GTAssemblyTask
    arm_index: int
    world_t_goal_brick: np.ndarray
    world_t_goal_tcp: np.ndarray
    world_t_safe_tcp: np.ndarray
    brick_t_tcp: np.ndarray
    grasp_axis: int
    grasp_width: float


@dataclass(frozen=True)
class PickPlan:
    """Unrecorded free-space waypoints that establish a safe start."""

    arm_index: int
    grasp_axis: int
    grasp_width: float
    world_t_pick_tcp: np.ndarray
    world_t_pregrasp_tcp: np.ndarray
    q_pregrasp: np.ndarray
    q_grasp: np.ndarray


class _RuntimeWrenchSource:
    """Expose calibrated legacy feedback through the primitive protocol."""

    def __init__(self, runtime: GTAssemblyRuntime):
        """Bind the calibrated BrickSim assembly runtime."""
        self._runtime = runtime

    def read_wrench_world(self) -> np.ndarray:
        """Read the current estimated world-frame TCP wrench.

        Returns:
            Force followed by torque as a six-vector.
        """
        return self._runtime.read_feedback().wrench_world


@dataclass(frozen=True)
class ExpertResult:
    """Terminal result from one local assembly attempt."""

    success: bool
    steps: int
    failure_reason: str | None


@dataclass(frozen=True)
class AlignmentIKBranch:
    """Verified joint seeds along a high-clearance Cartesian yaw path."""

    rotation_errors: np.ndarray
    configurations: tuple[np.ndarray, ...]


class SafeStartError(RuntimeError):
    """Raised when the privileged safe-start state cannot be established."""


def load_expert_config(path: str | Path) -> AssemblyExpertConfig:
    """Load the deterministic GT expert runtime configuration.

    Returns:
        Cartesian limits, feedback filters, and safety thresholds.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data["max_rotation_step"] = np.deg2rad(data.pop("max_rotation_step_deg"))
    data["max_alignment_rotation_step"] = np.deg2rad(
        data.pop("max_alignment_rotation_step_deg")
    )
    data["max_alignment_rotation_lead"] = np.deg2rad(
        data.pop("max_alignment_rotation_lead_deg")
    )
    return AssemblyExpertConfig(**data)


def resolve_single_step_task(env) -> GTAssemblyTask:
    """Resolve exactly one unplaced target from simulator GT topology.

    Returns:
        Target-centric task with a deterministic primary reference.
    """
    targets = sorted(int(part_id) for part_id in env.to_place_placed)
    if len(targets) != 1:
        raise ValueError(f"expected exactly one unplaced target, found {targets}")
    target_id = targets[0]
    parts = {int(part["id"]): part for part in env.topology["parts"]}
    candidates = [
        connection
        for connection in env.topology["connections"]
        if int(connection["hole_id"]) == target_id
        and int(connection["stud_id"]) in env.pre_placed_parts
        and int(connection["stud_id"]) != 0
    ]
    if not candidates:
        raise ValueError(f"target {target_id} has no preplaced brick reference")

    def converted(connection) -> GTConnection:
        reference_id = int(connection["stud_id"])
        return GTConnection(
            reference_id=reference_id,
            target_id=target_id,
            reference_path=env.pre_placed_parts[reference_id],
            target_path=env.to_place_placed[target_id],
            stud_iface=int(connection["stud_iface"]),
            hole_iface=int(connection["hole_iface"]),
            offset=(int(connection["offset"][0]), int(connection["offset"][1])),
            yaw=int(connection["yaw"]),
            overlap_studs=_connection_overlap(connection, parts),
        )

    connections = sorted(
        (converted(connection) for connection in candidates),
        key=lambda connection: (-connection.overlap_studs, connection.reference_id),
    )
    dimensions = {
        key: int(value)
        for key, value in parts[target_id]["payload"].items()
        if key in {"L", "W", "H"}
    }
    return GTAssemblyTask(
        target_id=target_id,
        target_path=env.to_place_placed[target_id],
        dimensions=dimensions,
        primary=connections[0],
        additional=tuple(connections[1:]),
    )


def compute_goal_brick_pose(env, task: GTAssemblyTask) -> np.ndarray:
    """Compute and cross-check the target brick world pose from GT references.

    Returns:
        Homogeneous target brick pose derived from the primary connection.
    """
    candidates = [_goal_from_connection(env, item) for item in task.connections]
    primary = candidates[0]
    for connection, candidate in zip(task.connections[1:], candidates[1:]):
        position_error = float(np.linalg.norm(candidate[:3, 3] - primary[:3, 3]))
        rotation_error = float(
            Rotation.from_matrix(primary[:3, :3].T @ candidate[:3, :3]).magnitude()
        )
        if (
            position_error > GOAL_POSITION_CONSISTENCY
            or rotation_error > GOAL_ROTATION_CONSISTENCY
        ):
            raise ValueError(
                "inconsistent multi-reference target pose from reference "
                f"{connection.reference_id}: position={position_error:.6f} m, "
                f"rotation={np.rad2deg(rotation_error):.3f} deg"
            )
    return primary


async def prepare_safe_start(env, safe_height: float = SAFE_HEIGHT) -> SafeStart:
    """Physically pick and transport the loose target to the safe start.

    This free-space preparation is deliberately separate from the recorded
    local assembly trajectory.  It may use simulator GT, but it never moves or
    teleports the loose target directly.

    Returns:
        Verified task, active arm, and goal/safe transforms.
    """
    if safe_height <= 0.0:
        raise ValueError("safe_height must be positive")
    task = resolve_single_step_task(env)
    world_t_goal_brick = compute_goal_brick_pose(env, task)
    initial_brick = env.get_prim_world_T(task.target_path)
    print(
        "[preparation] loose target GT: "
        f"xyz={initial_brick[:3, 3].tolist()}",
        flush=True,
    )
    plan, grasp_clearance = _select_pick_plan(
        env, task, initial_brick, world_t_goal_brick, safe_height
    )
    arm_index = plan.arm_index
    print(
        "[preparation] execute PickSkill "
        f"(arm={arm_index}, grasp_axis={'xy'[plan.grasp_axis]})",
        flush=True,
    )
    pick_skill = PickSkill.create(
        BrickSimRobotBackend(env, arm_index), BrickSimWorldModel(env)
    )
    try:
        pick_result = await pick_skill.execute(
            PickRequest(
                object_id=task.target_path,
                candidates=(
                    GraspCandidate(
                        world_t_pregrasp_tcp=plan.world_t_pregrasp_tcp,
                        world_t_grasp_tcp=plan.world_t_pick_tcp,
                        grasp_axis=plan.grasp_axis,
                        grasp_width=plan.grasp_width,
                    ),
                ),
                lift_distance=PICK_APPROACH_HEIGHT,
            )
        )
    except ExecutionError as exc:
        raise SafeStartError(str(exc)) from exc
    brick_t_tcp = pick_result.held.object_t_tcp
    lifted_brick = pick_result.lifted_object_pose
    print(
        "[preparation] PickSkill complete: "
        f"steps={pick_result.steps}, "
        f"brick_xyz={lifted_brick[:3, 3].tolist()}, "
        f"brick_to_tcp={brick_t_tcp[:3, 3].tolist()}",
        flush=True,
    )

    # Establish one physical-grasp baseline after lift.  Every remaining
    # transport segment is checked against this same transform so cumulative
    # slip cannot be hidden by resetting the reference between waypoints.
    lift_q = _arm_configuration(env, arm_index)
    lift_tcp = _tcp_world(env, arm_index, lift_q)
    lifted_brick = env.get_prim_world_T(task.target_path)
    brick_t_tcp = np.linalg.inv(lifted_brick) @ lift_tcp
    world_t_safe_brick = _safe_start_brick_pose(
        world_t_goal_brick,
        lifted_brick,
        safe_height,
    )
    world_t_transit_brick = world_t_safe_brick.copy()
    world_t_transit_brick[:3, 3] += (
        world_t_goal_brick[:3, 2] * (TRANSIT_HEIGHT - safe_height)
    )
    world_t_carry_brick = lifted_brick.copy()
    carry_height = float(
        np.dot(
            world_t_transit_brick[:3, 3] - lifted_brick[:3, 3],
            world_t_goal_brick[:3, 2],
        )
    )
    if carry_height > 0.0:
        world_t_carry_brick[:3, 3] += (
            world_t_goal_brick[:3, 2] * carry_height
        )
    world_t_carry_tcp = world_t_carry_brick @ brick_t_tcp
    carry_q = _solve_verified_ik(
        env, arm_index, world_t_carry_tcp, lift_q, "carry_height"
    )
    print(
        "[preparation] lift -> carry height -> transport -> safe start",
        flush=True,
    )
    await _execute_cartesian_waypoint(
        env,
        arm_index,
        world_t_carry_tcp,
        "carry_height",
        plan.grasp_width,
        close=True,
        grasp_reference=(task.target_path, brick_t_tcp, plan.grasp_axis),
    )

    carry_q = _arm_configuration(env, arm_index)
    world_t_transit_tcp = world_t_transit_brick @ brick_t_tcp
    _solve_verified_ik(
        env, arm_index, world_t_transit_tcp, carry_q, "transport"
    )
    await _execute_cartesian_waypoint(
        env,
        arm_index,
        world_t_transit_tcp,
        "transport",
        plan.grasp_width,
        close=True,
        grasp_reference=(task.target_path, brick_t_tcp, plan.grasp_axis),
    )

    transit_q = _arm_configuration(env, arm_index)
    world_t_safe_tcp = world_t_safe_brick @ brick_t_tcp
    _solve_verified_ik(
        env, arm_index, world_t_safe_tcp, transit_q, "safe_start"
    )
    await _execute_cartesian_waypoint(
        env,
        arm_index,
        world_t_safe_tcp,
        "safe_start",
        plan.grasp_width,
        close=True,
        grasp_reference=(task.target_path, brick_t_tcp, plan.grasp_axis),
    )

    actual_q = _arm_configuration(env, arm_index)
    actual_tcp = _tcp_world(env, arm_index, actual_q)
    actual_brick = env.get_prim_world_T(task.target_path)
    actual_brick_t_tcp = np.linalg.inv(actual_brick) @ actual_tcp
    safe_position_error, safe_rotation_error = _pose_error(
        actual_brick, world_t_safe_brick
    )
    if safe_position_error > 0.005 or safe_rotation_error > np.deg2rad(5.0):
        raise SafeStartError(
            "picked target did not reach the safe assembly pose: "
            f"position={safe_position_error:.4f} m, "
            f"rotation={np.rad2deg(safe_rotation_error):.2f} deg"
        )
    world_t_goal_tcp = world_t_goal_brick @ actual_brick_t_tcp
    goal_position_error, goal_rotation_error = _pose_error(
        actual_brick, world_t_goal_brick
    )
    clearance_text = "open" if np.isinf(grasp_clearance) else (
        f"{grasp_clearance * 1000:.1f} mm"
    )
    print(
        "[preparation] complete; assembly recording may start: "
        f"arm={arm_index}, height={safe_height * 1000:.1f} mm, "
        f"grasp_axis={'xy'[plan.grasp_axis]}, "
        f"side_clearance={clearance_text}, "
        f"region_error={safe_position_error * 1000:.2f} mm/"
        f"{np.rad2deg(safe_rotation_error):.2f} deg, "
        f"goal_error={goal_position_error * 1000:.1f} mm/"
        f"{np.rad2deg(goal_rotation_error):.1f} deg",
        flush=True,
    )
    return SafeStart(
        task=task,
        arm_index=arm_index,
        world_t_goal_brick=world_t_goal_brick,
        world_t_goal_tcp=world_t_goal_tcp,
        world_t_safe_tcp=actual_tcp,
        brick_t_tcp=actual_brick_t_tcp,
        grasp_axis=plan.grasp_axis,
        grasp_width=plan.grasp_width,
    )


async def run_gt_assembly_expert(
    env,
    prepared: SafeStart,
    runtime_config,
) -> ExpertResult:
    """Run deterministic geometric assembly from safe pose to connection.

    Returns:
        Success, executed control steps, and an optional failure reason.
    """
    runtime = GTAssemblyRuntime(env, prepared.arm_index, runtime_config)
    runtime.reset()
    await runtime.calibrate_wrench()
    initial_grasp_error = _grasp_stability_error(env, prepared)
    if initial_grasp_error is not None:
        return ExpertResult(
            False,
            0,
            f"target slipped during pre-trajectory calibration: "
            f"{initial_grasp_error}",
        )
    alignment_reachable, alignment_rotation_hint, _ = _alignment_rotation_hint(
        env,
        prepared,
        runtime_config.max_alignment_rotation_step * 0.5,
    )
    if not alignment_reachable:
        return ExpertResult(
            False,
            0,
            "safe-start pose has no verified high-clearance alignment IK",
        )
    robot = BrickSimRobotBackend(env, prepared.arm_index)
    world = BrickSimWorldModel(env)
    calibrated_preassembly_tcp = robot.read_state().tcp_world
    success_check = _bricksim_success_check(prepared.task)
    held = HeldObject(
        prepared.task.target_path,
        robot.robot_id,
        prepared.brick_t_tcp,
        prepared.grasp_axis,
        prepared.grasp_width,
    )
    direction = -prepared.world_t_goal_brick[:3, 2]
    motion_config = MotionConfig(
        position_tolerance=0.002,
        rotation_tolerance=np.deg2rad(1.0),
        timeout_steps=runtime_config.max_episode_steps,
        translation_step=runtime_config.max_translation_step,
        translation_step_held=runtime_config.max_translation_step,
    )
    print("[expert] execute AssembleSkill", flush=True)
    try:
        result = await AssembleSkill.create(
            robot, world, motion_config
        ).execute(
            AssembleRequest(
                held=held,
                world_t_preassembly_tcp=calibrated_preassembly_tcp,
                world_t_goal_tcp=prepared.world_t_goal_tcp,
                insertion_direction_world=direction,
                success_check=success_check,
                alignment_rotation_hint_goal=alignment_rotation_hint,
                approach_clearance=0.005,
                insertion_distance=0.012,
                insertion_step=0.0001,
                retreat_distance=POST_ASSEMBLY_RETREAT,
                max_insert_steps=runtime_config.max_episode_steps,
                wrench_source=_RuntimeWrenchSource(runtime),
                force_guard=ForceGuard(
                    runtime_config.max_force, runtime_config.max_force
                ),
            )
        )
    except (ExecutionError, ValueError) as exc:
        return ExpertResult(False, 0, str(exc))
    print(
        f"[expert] AssembleSkill complete: steps={result.steps}", flush=True
    )
    return ExpertResult(True, result.steps, None)


async def release_and_return_home(env, prepared: SafeStart) -> None:
    """Return home after AssembleSkill has released and retreated."""
    if not verify_connections(prepared.task):
        raise SafeStartError("cannot return home before every connection is verified")
    arm_index = prepared.arm_index
    home_q = env.robot_pins[arm_index].home_q.copy()
    print("[cleanup] retreat -> home", flush=True)
    await _execute_joint_waypoint(
        env, arm_index, home_q, "home", world_t_tcp=None
    )
    if not verify_connections(prepared.task):
        raise SafeStartError("requested connection was lost while returning home")
    print("[cleanup] complete", flush=True)


def verify_connections(task: GTAssemblyTask) -> bool:
    """Return whether every requested BrickSim connection is active."""
    from bricksim.core import lookup_physics_connection

    for connection in task.connections:
        info = lookup_physics_connection(
            stud_path=connection.reference_path,
            stud_if=connection.stud_iface,
            hole_path=connection.target_path,
            hole_if=connection.hole_iface,
        )
        if (
            info is None
            or tuple(info.offset) != connection.offset
            or int(info.yaw) != connection.yaw
        ):
            return False
    return True


def _bricksim_success_check(task: GTAssemblyTask) -> BrickSimSuccessCheck:
    """Build exact connection verification behind the backend boundary.

    Returns:
        Live semantic success checker for every task connection.
    """
    return BrickSimSuccessCheck(
        tuple(
            BrickSimConnectionGoal(
                connection.reference_path,
                connection.stud_iface,
                connection.target_path,
                connection.hole_iface,
                connection.offset,
                connection.yaw,
            )
            for connection in task.connections
        )
    )


def _connection_conflict(task: GTAssemblyTask) -> str | None:
    """Return an error when a requested interface snapped to the wrong grid pose."""
    from bricksim.core import lookup_physics_connection

    for connection in task.connections:
        info = lookup_physics_connection(
            stud_path=connection.reference_path,
            stud_if=connection.stud_iface,
            hole_path=connection.target_path,
            hole_if=connection.hole_iface,
        )
        if info is None:
            continue
        actual_offset = tuple(info.offset)
        actual_yaw = int(info.yaw)
        if actual_offset != connection.offset or actual_yaw != connection.yaw:
            return (
                "wrong BrickSim connection: "
                f"reference={connection.reference_id} "
                f"actual={actual_offset}/{actual_yaw} "
                f"expected={connection.offset}/{connection.yaw}"
            )
    return None


def _assembly_debug_summary(task: GTAssemblyTask) -> str:
    """Format BrickSim's native detector diagnostics for this target.

    Returns:
        Compact diagnostics for the latest matching detector candidate.
    """
    from bricksim.core import get_assembly_debug_infos, lookup_physics_connection

    active = []
    for connection in task.connections:
        info = lookup_physics_connection(
            stud_path=connection.reference_path,
            stud_if=connection.stud_iface,
            hole_path=connection.target_path,
            hole_if=connection.hole_iface,
        )
        if info is not None:
            active.append(
                f"ref={connection.reference_id} actual={tuple(info.offset)}/"
                f"{int(info.yaw)} expected={connection.offset}/{connection.yaw}"
            )

    endpoints = {
        (
            connection.reference_path,
            connection.stud_iface,
            connection.target_path,
            connection.hole_iface,
        )
        for connection in task.connections
    }
    matching = [
        info
        for info in get_assembly_debug_infos()
        if (
            info.stud_path,
            info.stud_interface,
            info.hole_path,
            info.hole_interface,
        )
        in endpoints
    ]
    if not matching:
        if active:
            return "active " + "; ".join(active)
        return "no detector candidate"
    info = matching[-1]
    return (
        f"accepted={info.accepted} distance={info.relative_distance:.5f} "
        f"tilt={np.rad2deg(info.tilt):.2f}deg "
        f"force={info.projected_force:.3f} "
        f"yaw_error={np.rad2deg(info.yaw_error):.2f}deg "
        f"position_error={info.position_error:.5f} "
        f"grid={info.grid_pos}->{info.grid_pos_snapped}"
    )


def _goal_from_connection(env, connection: GTConnection) -> np.ndarray:
    from bricksim.core import compute_connection_transform

    quaternion, position = compute_connection_transform(
        stud_path=connection.reference_path,
        stud_if=connection.stud_iface,
        hole_path=connection.target_path,
        hole_if=connection.hole_iface,
        offset=connection.offset,
        yaw=connection.yaw,
    )
    reference_t_target = np.eye(4)
    reference_t_target[:3, :3] = Rotation.from_quat(
        [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
    ).as_matrix()
    reference_t_target[:3, 3] = np.asarray(position, dtype=np.float64)
    return env.get_prim_world_T(connection.reference_path) @ reference_t_target


def _select_pick_plan(
    env, task, world_t_pick_brick, world_t_goal_brick, safe_height
) -> tuple[PickPlan, float]:
    """Choose a grasp that is reachable at pickup and clear at assembly.

    Returns:
        Selected free-space pick plan and its goal-side clearance.
    """
    candidates = []
    preferred_axis = _preferred_grasp_axis(task.dimensions)
    pick_tcp_height = PICK_TCP_HEIGHT
    obstacle_bounds = _goal_level_obstacle_bounds(env, task, world_t_goal_brick)
    clearances = tuple(
        _grasp_axis_clearance(task.dimensions, obstacle_bounds, axis)
        for axis in (0, 1)
    )
    viable_axes = [
        axis
        for axis, clearance in enumerate(clearances)
        if clearance >= MIN_GRIPPER_SIDE_CLEARANCE
    ]
    if not viable_axes:
        raise SafeStartError(
            "no collision-free x/y grasp axis at the assembly goal: "
            f"clearance_x={clearances[0]:.4f} m, "
            f"clearance_y={clearances[1]:.4f} m"
        )
    for arm_index, robot_pin in enumerate(env.robot_pins):
        for grasp_axis in viable_axes:
            for sign in (1.0, -1.0):
                pick_tcp = _grasp_tcp_for_brick(
                    world_t_pick_brick,
                    grasp_axis,
                    sign,
                    pick_tcp_height,
                )
                pregrasp_tcp = pick_tcp.copy()
                pregrasp_tcp[:3, 3] += (
                    world_t_pick_brick[:3, 2] * PICK_APPROACH_HEIGHT
                )
                pregrasp = _try_verified_ik(
                    robot_pin, pregrasp_tcp, robot_pin.home_q
                )
                if pregrasp is None:
                    continue
                grasp = _try_verified_ik(robot_pin, pick_tcp, pregrasp)
                if grasp is None:
                    continue

                # Before touching the object, prove that the same local grasp
                # orientation is also reachable above the final structure.
                brick_t_tcp = np.linalg.inv(world_t_pick_brick) @ pick_tcp
                safe_brick = _safe_start_brick_pose(
                    world_t_goal_brick,
                    world_t_pick_brick,
                    safe_height,
                )
                safe_tcp = safe_brick @ brick_t_tcp
                safe = _try_verified_ik(robot_pin, safe_tcp, grasp)
                if safe is None:
                    continue

                # Planning may use GT to reject a dead-end safe start even
                # though preparation must not execute this alignment.  The
                # selected arm/grasp has to retain an IK solution for the
                # final-yaw TCP at the same 60 mm clearance height.
                aligned_safe_brick = _offset_along_local_z(
                    world_t_goal_brick, safe_height
                )
                aligned_safe_tcp = aligned_safe_brick @ brick_t_tcp
                aligned_safe = _try_verified_ik(
                    robot_pin, aligned_safe_tcp, safe
                )
                if aligned_safe is None:
                    continue
                # Endpoint reachability alone is insufficient: a wrist-limit
                # discontinuity can lie between two individually valid poses.
                # Reject that grasp branch before touching the loose target.
                if _build_alignment_ik_branch(
                    robot_pin,
                    safe_tcp,
                    aligned_safe_tcp,
                    safe,
                    ALIGNMENT_IK_PLANNING_STEP,
                    verbose=False,
                ) is None:
                    continue
                ik_cost = float(
                    np.linalg.norm(pregrasp[:6] - robot_pin.home_q[:6])
                    + np.linalg.norm(grasp[:6] - pregrasp[:6])
                    + np.linalg.norm(safe[:6] - grasp[:6])
                    + np.linalg.norm(aligned_safe[:6] - safe[:6])
                )
                pickup_distance = float(
                    np.linalg.norm(
                        world_t_pick_brick[:3, 3] - robot_pin.BASE_T[:3, 3]
                    )
                )
                clearance = clearances[grasp_axis]
                clearance_cost = -min(clearance, GRIPPER_CLEARANCE_SCORE_CAP)
                axis_preference = int(grasp_axis != preferred_axis)
                grasp_width = (
                    task.dimensions["L"]
                    if grasp_axis == 0
                    else task.dimensions["W"]
                ) * BRICK_UNIT_LENGTH
                plan = PickPlan(
                    arm_index=arm_index,
                    grasp_axis=grasp_axis,
                    grasp_width=grasp_width,
                    world_t_pick_tcp=pick_tcp,
                    world_t_pregrasp_tcp=pregrasp_tcp,
                    q_pregrasp=pregrasp,
                    q_grasp=grasp,
                )
                candidates.append(
                    (
                        clearance_cost,
                        axis_preference,
                        pickup_distance,
                        ik_cost,
                        plan,
                        clearance,
                    )
                )
    if not candidates:
        raise SafeStartError(
            "no arm has a valid IK chain for pickup and safe assembly pose"
        )
    selected = min(candidates, key=lambda item: item[:4])
    return selected[4], selected[5]


def _grasp_tcp_for_brick(
    world_t_brick: np.ndarray,
    grasp_axis: int,
    sign: float,
    tcp_height: float,
) -> np.ndarray:
    """Return a top-down TCP pose expressed from one brick-local side axis."""
    if grasp_axis not in (0, 1):
        raise ValueError("grasp_axis must be 0 or 1")
    # Piper's prismatic finger joints move along the tool's local +/-y axis.
    # ``grasp_axis`` therefore denotes the brick-local axis across which the
    # jaws close, not the finger-length/tool-x direction.
    y_axis = sign * world_t_brick[:3, grasp_axis].copy()
    z_axis = -world_t_brick[:3, 2].copy()
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis /= np.linalg.norm(z_axis)
    result = np.eye(4)
    result[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    result[:3, 3] = world_t_brick[:3, 3] + (
        world_t_brick[:3, 2] * tcp_height
    )
    return result


def _preferred_grasp_axis(dimensions: dict[str, int]) -> int:
    """Choose a stable jaw axis from the target footprint.

    Very elongated bricks are gripped across their long dimension to increase
    yaw leverage.  Compact bricks use the short dimension to minimize travel.

    Returns:
        Preferred brick-local jaw-motion axis, either 0 (x) or 1 (y).
    """
    length = int(dimensions["L"])
    width = int(dimensions["W"])
    short_axis = 0 if length <= width else 1
    long_axis = 0 if length >= width else 1
    aspect_ratio = max(length, width) / min(length, width)
    return long_axis if aspect_ratio >= LONG_BRICK_ASPECT_RATIO else short_axis


def _is_short_target(dimensions: dict[str, int]) -> bool:
    """Return whether the footprint is exactly one by two studs."""
    return sorted((int(dimensions["L"]), int(dimensions["W"]))) == [1, 2]


def _try_verified_ik(robot_pin, world_t_tcp, seed):
    """Return a strictly verified IK solution, or None when unreachable."""
    arm_t_tcp = np.linalg.inv(robot_pin.BASE_T) @ world_t_tcp
    q, _ = robot_pin.IK(
        {robot_pin.ee_frames[0]: arm_t_tcp},
        seed,
        robot_pin.controllable_joints,
        ROT_WEIGHT=0.05,
    )
    solved = robot_pin.FK(q, [robot_pin.ee_frames[0]])[
        robot_pin.ee_frames[0]
    ]
    position_error, rotation_error = _pose_error(solved, arm_t_tcp)
    if (
        position_error > SAFE_IK_POSITION_TOLERANCE
        or rotation_error > SAFE_IK_ROTATION_TOLERANCE
    ):
        return None
    return q


def _solve_verified_ik(env, arm_index, world_t_tcp, seed, stage):
    """Solve one free-space waypoint or raise a stage-specific error.

    Returns:
        Verified full robot configuration.
    """
    solution = _try_verified_ik(env.robot_pins[arm_index], world_t_tcp, seed)
    if solution is None:
        raise SafeStartError(f"{stage}: no verified IK solution")
    return solution


def _offset_along_local_z(transform: np.ndarray, distance: float) -> np.ndarray:
    """Translate a copied pose along its positive local z axis.

    Returns:
        Copied and translated homogeneous transform.
    """
    result = np.asarray(transform, dtype=np.float64).copy()
    result[:3, 3] += result[:3, 2] * distance
    return result


def _safe_start_brick_pose(
    world_t_goal_brick: np.ndarray,
    world_t_pickup_brick: np.ndarray,
    safe_height: float,
) -> np.ndarray:
    """Place the brick center directly above the goal without changing yaw.

    Returns:
        Pickup-oriented brick pose at the configured height over goal XY.
    """
    result = np.asarray(world_t_pickup_brick, dtype=np.float64).copy()
    result[:3, 3] = world_t_goal_brick[:3, 3].copy()
    result[:3, 3] += world_t_goal_brick[:3, 2] * safe_height
    return result


def _pose_error(actual: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Return world-frame position and geodesic rotation errors."""
    position = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
    rotation = float(
        Rotation.from_matrix(actual[:3, :3].T @ target[:3, :3]).magnitude()
    )
    return position, rotation


def _goal_level_obstacle_bounds(env, task, world_t_goal_brick):
    """Return preplaced same-level footprints in target-local coordinates."""
    parts = {int(part["id"]): part for part in env.topology["parts"]}
    goal_t_world = np.linalg.inv(world_t_goal_brick)
    bounds = []
    for part_id, path in env.pre_placed_parts.items():
        if int(part_id) == 0:
            continue
        goal_t_part = goal_t_world @ env.get_prim_world_T(path)
        if abs(float(goal_t_part[2, 3])) > SAME_LEVEL_Z_TOLERANCE:
            continue
        payload = parts[int(part_id)]["payload"]
        half_length = int(payload["L"]) * BRICK_UNIT_LENGTH * 0.5
        half_width = int(payload["W"]) * BRICK_UNIT_LENGTH * 0.5
        corners = np.array(
            [
                [-half_length, -half_width, 0.0, 1.0],
                [-half_length, half_width, 0.0, 1.0],
                [half_length, -half_width, 0.0, 1.0],
                [half_length, half_width, 0.0, 1.0],
            ]
        )
        local = (goal_t_part @ corners.T).T
        bounds.append(
            (
                float(local[:, 0].min()),
                float(local[:, 0].max()),
                float(local[:, 1].min()),
                float(local[:, 1].max()),
            )
        )
    return tuple(bounds)


def _grasp_axis_clearance(dimensions, obstacle_bounds, grasp_axis):
    """Return nearest side-obstacle clearance for one target-local axis."""
    half_extents = np.array(
        [dimensions["L"], dimensions["W"]], dtype=np.float64
    ) * (BRICK_UNIT_LENGTH * 0.5)
    axis = int(grasp_axis)
    if axis not in (0, 1):
        raise ValueError("grasp_axis must be 0 or 1")
    perpendicular = 1 - axis
    nearest = np.inf
    for bounds in obstacle_bounds:
        minimum = np.array([bounds[0], bounds[2]], dtype=np.float64)
        maximum = np.array([bounds[1], bounds[3]], dtype=np.float64)
        overlaps_grasp_band = (
            minimum[perpendicular] < half_extents[perpendicular]
            and maximum[perpendicular] > -half_extents[perpendicular]
        )
        if not overlaps_grasp_band:
            continue
        if maximum[axis] <= -half_extents[axis]:
            gap = -half_extents[axis] - maximum[axis]
        elif minimum[axis] >= half_extents[axis]:
            gap = minimum[axis] - half_extents[axis]
        else:
            gap = 0.0
        nearest = min(nearest, float(gap))
    return float(nearest)


def _connection_overlap(connection, parts) -> int:
    stud = parts[int(connection["stud_id"])]["payload"]
    hole = parts[int(connection["hole_id"])]["payload"]
    offset_x, offset_y = (int(value) for value in connection["offset"])
    yaw = int(connection["yaw"]) % 4

    def rotate(x, y):
        if yaw == 0:
            return x, y
        if yaw == 1:
            return -y, x
        if yaw == 2:
            return -x, -y
        return y, -x

    overlap = 0
    for x in range(int(hole["L"])):
        for y in range(int(hole["W"])):
            rotated_x, rotated_y = rotate(x, y)
            stud_x = offset_x + rotated_x
            stud_y = offset_y + rotated_y
            if 0 <= stud_x < int(stud["L"]) and 0 <= stud_y < int(stud["W"]):
                overlap += 1
    return overlap


def _with_gripper(env, arm_index, q, grasp_width, close):
    result = np.asarray(q, dtype=np.float64).copy()
    open_position = _open_gripper_joint_position(grasp_width)
    robot_pin = env.robot_pins[arm_index]
    for name in env.robot_configs[arm_index]["Joint_Order"]:
        if "gripper" not in name:
            continue
        joint = robot_pin.pin_model.joints[robot_pin.pin_model.getJointId(name)]
        if name == "gripper_joint1":
            result[joint.idx_q] = 0.0 if close else open_position
        elif name == "gripper_joint2":
            result[joint.idx_q] = 0.0 if close else -open_position
    return result


def _open_gripper_joint_position(grasp_width: float) -> float:
    """Return each finger's open position for a full object width.

    Returns:
        Positive magnitude applied symmetrically to the two finger joints.
    """
    if grasp_width <= 0.0:
        raise ValueError("grasp_width must be positive")
    return min(
        grasp_width * 0.5 + GRIPPER_OPEN_MARGIN_PER_FINGER,
        GRIPPER_MAX_JOINT_OPENING,
    )


def _transport_jaw_drift_limit(grasp_width: float) -> float:
    """Return a width-aware seating limit along the jaw-motion axis.

    Returns:
        Allowed relative translation before transport declares a dropped brick.
    """
    if grasp_width <= 0.0:
        raise ValueError("grasp_width must be positive")
    return min(0.008, max(0.004, grasp_width * 0.5))


def _fast_alignment_allowed(dimensions: dict[str, int]) -> bool:
    """Return whether target geometry tolerates rotation lead."""
    return not _is_short_target(dimensions)


def _gripper_positions(env, arm_index, q) -> list[float]:
    """Return named gripper joint positions for diagnostics.

    Returns:
        Gripper values in configured joint order.
    """
    values = []
    robot_pin = env.robot_pins[arm_index]
    for name in env.robot_configs[arm_index]["Joint_Order"]:
        if "gripper" not in name:
            continue
        joint = robot_pin.pin_model.joints[robot_pin.pin_model.getJointId(name)]
        values.append(float(q[joint.idx_q]))
    return values


async def _execute_joint_waypoint(
    env,
    arm_index: int,
    goal_q: np.ndarray,
    stage: str,
    world_t_tcp: np.ndarray | None,
) -> None:
    """Track one unrecorded free-space waypoint with feedback checks."""
    settled = 0
    for step in range(1, FREE_MOTION_TIMEOUT + 1):
        actual_q = _arm_configuration(env, arm_index)
        delta = np.asarray(goal_q, dtype=np.float64) - actual_q
        limits = np.full(delta.shape, FREE_MOTION_MAX_ARM_STEP)
        for joint_name in env.robot_configs[arm_index]["Joint_Order"]:
            if "gripper" not in joint_name:
                continue
            robot_pin = env.robot_pins[arm_index]
            joint = robot_pin.pin_model.joints[
                robot_pin.pin_model.getJointId(joint_name)
            ]
            limits[joint.idx_q] = FREE_MOTION_MAX_GRIPPER_STEP
        command_q = actual_q + np.clip(delta, -limits, limits)
        _set_arm_command(env, arm_index, command_q)
        await env.step()
        await env.step()

        actual_q = _arm_configuration(env, arm_index)
        if world_t_tcp is None:
            arm_errors = []
            robot_pin = env.robot_pins[arm_index]
            for joint_name in env.robot_configs[arm_index]["Joint_Order"]:
                if "gripper" in joint_name:
                    continue
                joint = robot_pin.pin_model.joints[
                    robot_pin.pin_model.getJointId(joint_name)
                ]
                arm_errors.append(abs(actual_q[joint.idx_q] - goal_q[joint.idx_q]))
            reached = bool(
                arm_errors and max(arm_errors) <= FREE_MOTION_JOINT_TOLERANCE
            )
            position_error = 0.0
            rotation_error = 0.0
        else:
            actual_tcp = _tcp_world(env, arm_index, actual_q)
            position_error, rotation_error = _pose_error(actual_tcp, world_t_tcp)
            reached = (
                position_error <= FREE_MOTION_POSITION_TOLERANCE
                and rotation_error <= FREE_MOTION_ROTATION_TOLERANCE
            )
        settled = settled + 1 if reached else 0
        if settled >= FREE_MOTION_SETTLE_STEPS:
            print(
                f"[free-motion] {stage}: {step} steps, "
                f"position={position_error:.4f} m, "
                f"rotation={np.rad2deg(rotation_error):.2f} deg",
                flush=True,
            )
            return
    if world_t_tcp is not None:
        actual_tcp = _tcp_world(
            env, arm_index, _arm_configuration(env, arm_index)
        )
        position_error, rotation_error = _pose_error(actual_tcp, world_t_tcp)
        detail = (
            f"position={position_error:.4f} m, "
            f"rotation={np.rad2deg(rotation_error):.2f} deg"
        )
    else:
        detail = "joint target did not settle"
    raise SafeStartError(
        f"{stage}: state timeout after {FREE_MOTION_TIMEOUT} steps ({detail})"
    )


async def _execute_cartesian_waypoint(
    env,
    arm_index: int,
    world_t_tcp: np.ndarray,
    stage: str,
    grasp_width: float,
    close: bool,
    grasp_reference: tuple[str, np.ndarray, int] | None = None,
    allow_vertical_settling: bool = False,
) -> None:
    """Track a straight Cartesian segment with feedback and incremental IK."""
    settled = 0
    commanded_tcp: np.ndarray | None = None
    for step in range(1, FREE_MOTION_TIMEOUT + 1):
        actual_q = _arm_configuration(env, arm_index)
        actual_tcp = _tcp_world(env, arm_index, actual_q)
        if grasp_reference is not None:
            target_path, reference_brick_t_tcp, grasp_axis = grasp_reference
            brick = env.get_prim_world_T(target_path)
            actual_brick_t_tcp = np.linalg.inv(brick) @ actual_tcp
            _, drift_rotation = _pose_error(
                actual_brick_t_tcp, reference_brick_t_tcp
            )
            translation_drift = (
                actual_brick_t_tcp[:3, 3] - reference_brick_t_tcp[:3, 3]
            )
            jaw_drift = abs(float(translation_drift[grasp_axis]))
            finger_drift = abs(float(translation_drift[1 - grasp_axis]))
            vertical_drift = abs(float(translation_drift[2]))
            vertical_limit = (
                MAX_INITIAL_GRASP_VERTICAL_SETTLING
                if allow_vertical_settling
                else MAX_GRASP_VERTICAL_DRIFT
            )
            jaw_limit = _transport_jaw_drift_limit(grasp_width)
            if (
                jaw_drift
                > jaw_limit + GRASP_DRIFT_COMPARISON_TOLERANCE
                or finger_drift
                > MAX_GRASP_FINGER_AXIS_DRIFT
                + GRASP_DRIFT_COMPARISON_TOLERANCE
                or vertical_drift
                > vertical_limit + GRASP_DRIFT_COMPARISON_TOLERANCE
                or drift_rotation > MAX_GRASP_ROTATION_DRIFT
            ):
                raise SafeStartError(
                    f"{stage}: target slipped in gripper "
                    f"(jaw={jaw_drift:.6f}/{jaw_limit:.6f} m, "
                    f"finger={finger_drift:.6f}/"
                    f"{MAX_GRASP_FINGER_AXIS_DRIFT:.6f} m, "
                    f"vertical={vertical_drift:.6f}/{vertical_limit:.6f} m, "
                    f"rotation={np.rad2deg(drift_rotation):.2f} deg)"
                )
        if commanded_tcp is None:
            commanded_tcp = actual_tcp.copy()
        position_error, rotation_error = _pose_error(actual_tcp, world_t_tcp)
        reached = (
            position_error <= FREE_MOTION_POSITION_TOLERANCE
            and rotation_error <= FREE_MOTION_ROTATION_TOLERANCE
        )
        settled = settled + 1 if reached else 0
        if settled >= FREE_MOTION_SETTLE_STEPS:
            print(
                f"[free-motion] {stage}: {step} steps, "
                f"position={position_error:.4f} m, "
                f"rotation={np.rad2deg(rotation_error):.2f} deg",
                flush=True,
            )
            return

        incremental = commanded_tcp.copy()
        translation = world_t_tcp[:3, 3] - commanded_tcp[:3, 3]
        translation_norm = float(np.linalg.norm(translation))
        translation_step = (
            HELD_TRANSLATION_STEP if close else FREE_MOTION_TRANSLATION_STEP
        )
        if translation_norm > translation_step:
            translation *= translation_step / translation_norm
        incremental[:3, 3] += translation
        command_lead = incremental[:3, 3] - actual_tcp[:3, 3]
        command_lead_norm = float(np.linalg.norm(command_lead))
        base_lead_limit = (
            HELD_COMMAND_LEAD if close else FREE_MOTION_COMMAND_LEAD
        )
        ramp = min(1.0, step / COMMAND_LEAD_RAMP_STEPS)
        command_lead_limit = base_lead_limit * ramp
        if command_lead_norm > command_lead_limit:
            incremental[:3, 3] = actual_tcp[:3, 3] + (
                command_lead * (command_lead_limit / command_lead_norm)
            )
        local_rotation = Rotation.from_matrix(
            actual_tcp[:3, :3].T @ world_t_tcp[:3, :3]
        ).as_rotvec()
        rotation_norm = float(np.linalg.norm(local_rotation))
        rotation_limit = np.deg2rad(3.0 if close else 5.0)
        if rotation_norm > rotation_limit:
            local_rotation *= rotation_limit / rotation_norm
        incremental[:3, :3] = actual_tcp[:3, :3] @ Rotation.from_rotvec(
            local_rotation
        ).as_matrix()
        commanded_tcp = incremental.copy()
        q_target = _try_verified_ik(
            env.robot_pins[arm_index], incremental, actual_q
        )
        if q_target is None:
            raise SafeStartError(f"{stage}: incremental Cartesian IK failed")
        q_target = _with_gripper(
            env, arm_index, q_target, grasp_width, close
        )
        delta_q = q_target - actual_q
        for joint_name in env.robot_configs[arm_index]["Joint_Order"]:
            robot_pin = env.robot_pins[arm_index]
            joint = robot_pin.pin_model.joints[
                robot_pin.pin_model.getJointId(joint_name)
            ]
            limit = (
                FREE_MOTION_MAX_GRIPPER_STEP
                if "gripper" in joint_name
                else FREE_MOTION_MAX_ARM_STEP
            )
            delta_q[joint.idx_q] = np.clip(delta_q[joint.idx_q], -limit, limit)
        _set_arm_command(env, arm_index, actual_q + delta_q)
        await env.step()
        await env.step()
    actual_q = _arm_configuration(env, arm_index)
    actual_tcp = _tcp_world(env, arm_index, actual_q)
    position_error, rotation_error = _pose_error(actual_tcp, world_t_tcp)
    raise SafeStartError(
        f"{stage}: Cartesian timeout after {FREE_MOTION_TIMEOUT} steps "
        f"(position={position_error:.4f} m, "
        f"rotation={np.rad2deg(rotation_error):.2f} deg)"
    )


async def _actuate_gripper(
    env,
    arm_index: int,
    arm_q: np.ndarray,
    grasp_width: float,
    close: bool,
    connected_release: bool = False,
) -> None:
    """Actuate and verify the gripper while holding the arm fixed.

    ``connected_release`` is reserved for a target whose requested BrickSim
    connection has already been verified.  A finger obstructed by the final
    structure may stop before the planned clearance; cleanup then continues
    with a vertical retreat and verifies that the connection was retained.
    Normal pre-grasp opening and every closing check retain their original
    criteria.
    """
    if close and connected_release:
        raise ValueError("connected_release is valid only while opening")
    target = _with_gripper(env, arm_index, arm_q, grasp_width, close)
    target_values = _gripper_positions(env, arm_index, target)
    stable_steps = 0
    actual_values: list[float] = []
    for step in range(1, GRIPPER_MAX_STEPS + 1):
        before = _arm_configuration(env, arm_index)
        command = target.copy()
        for joint_name in env.robot_configs[arm_index]["Joint_Order"]:
            if "gripper" not in joint_name:
                continue
            robot_pin = env.robot_pins[arm_index]
            joint = robot_pin.pin_model.joints[
                robot_pin.pin_model.getJointId(joint_name)
            ]
            delta = target[joint.idx_q] - before[joint.idx_q]
            command[joint.idx_q] = before[joint.idx_q] + np.clip(
                delta,
                -FREE_MOTION_MAX_GRIPPER_STEP,
                FREE_MOTION_MAX_GRIPPER_STEP,
            )
        _set_arm_command(env, arm_index, command)
        await env.step()
        await env.step()
        after = _arm_configuration(env, arm_index)
        before_values = np.asarray(_gripper_positions(env, arm_index, before))
        actual_values = _gripper_positions(env, arm_index, after)
        actual_array = np.asarray(actual_values)
        if close:
            gap = float(np.sum(np.abs(actual_array)))
            width_reached = (
                abs(gap - grasp_width) <= GRIPPER_CONTACT_WIDTH_TOLERANCE
            )
            stalled = bool(
                np.max(np.abs(actual_array - before_values))
                <= GRIPPER_STALL_DELTA
            )
            reached = width_reached and stalled
        else:
            error = np.max(np.abs(actual_array - np.asarray(target_values)))
            gap = float(np.sum(np.abs(actual_array)))
            stalled = bool(
                np.max(np.abs(actual_array - before_values))
                <= GRIPPER_STALL_DELTA
            )
            constrained_release = connected_release and stalled
            reached = bool(error <= GRIPPER_OPEN_TOLERANCE) or constrained_release
        stable_steps = stable_steps + 1 if reached else 0
        if (
            step >= GRIPPER_MIN_STEPS
            and stable_steps >= GRIPPER_STABLE_STEPS
        ):
            if close:
                state = "closed on target"
            elif error <= GRIPPER_OPEN_TOLERANCE:
                state = "at planned opening"
            else:
                state = "constrained; vertical release required"
            print(
                f"[free-motion] gripper {state}: "
                f"steps={step}, joints={actual_values}",
                flush=True,
            )
            return
    gap = float(np.sum(np.abs(np.asarray(actual_values))))
    expected = grasp_width if close else 2.0 * target_values[0]
    raise SafeStartError(
        f"gripper {'close' if close else 'open'} verification failed: "
        f"gap={gap:.4f} m, expected={expected:.4f} m, "
        f"joints={actual_values}"
    )


def _set_arm_command(env, arm_index, arm_q) -> None:
    command = np.asarray(env.get_observations()["joint_positions"]).copy()
    robot_pin = env.robot_pins[arm_index]
    robot_config = env.robot_configs[arm_index]
    name = robot_config.get("Name", f"robot_{arm_index}")
    start, _ = env.arm_joint_slices[name]
    for offset, joint_name in enumerate(robot_config["Joint_Order"]):
        joint = robot_pin.pin_model.joints[robot_pin.pin_model.getJointId(joint_name)]
        command[start + offset] = arm_q[joint.idx_q]
    env.robot_apply_action(command)


def _arm_configuration(env, arm_index) -> np.ndarray:
    observation = np.asarray(env.get_observations()["joint_positions"])
    robot_pin = env.robot_pins[arm_index]
    robot_config = env.robot_configs[arm_index]
    name = robot_config.get("Name", f"robot_{arm_index}")
    start, _ = env.arm_joint_slices[name]
    q = robot_pin.home_q.copy()
    for offset, joint_name in enumerate(robot_config["Joint_Order"]):
        joint = robot_pin.pin_model.joints[robot_pin.pin_model.getJointId(joint_name)]
        q[joint.idx_q] = observation[start + offset]
    return q


def _tcp_world(env, arm_index, q) -> np.ndarray:
    robot_pin = env.robot_pins[arm_index]
    ee = robot_pin.ee_frames[0]
    return robot_pin.BASE_T @ robot_pin.FK(q, [ee])[ee]


def _rotate_wrench(world_t_frame, wrench_world):
    rotation = np.asarray(world_t_frame)[:3, :3].T
    result = np.asarray(wrench_world, dtype=np.float64).copy()
    result[:3] = rotation @ result[:3]
    result[3:] = rotation @ result[3:]
    return result


def _alignment_rotation_hint(
    env,
    prepared: SafeStart,
    rotation_step: float,
) -> tuple[bool, np.ndarray | None, AlignmentIKBranch | None]:
    """Return the Cartesian rotation direction of a reachable IK branch.

    Returns:
        Path reachability, an optional unit rotation vector in the goal skill
        frame, and verified joint seeds along the Cartesian rotation path.  The
        hint is needed only for a +/-pi ambiguity.
    """
    arm_index = prepared.arm_index
    current_q = _arm_configuration(env, arm_index)
    current_tcp = _tcp_world(env, arm_index, current_q)
    current_brick = env.get_prim_world_T(prepared.task.target_path)
    brick_t_tcp = np.linalg.inv(current_brick) @ current_tcp
    aligned_brick = prepared.world_t_goal_brick.copy()
    aligned_brick[:3, 3] = current_brick[:3, 3]
    aligned_tcp = aligned_brick @ brick_t_tcp
    aligned_q = _try_verified_ik(
        env.robot_pins[arm_index], aligned_tcp, current_q
    )
    if aligned_q is None:
        return False, None, None

    branch = _build_alignment_ik_branch(
        env.robot_pins[arm_index],
        current_tcp,
        aligned_tcp,
        current_q,
        rotation_step,
    )
    if branch is None:
        return False, None, None

    probe_q = current_q + 0.02 * (aligned_q - current_q)
    probe_tcp = _tcp_world(env, arm_index, probe_q)
    world_t_goal_tcp = prepared.world_t_goal_brick @ brick_t_tcp
    skill_t_current = np.linalg.inv(world_t_goal_tcp) @ current_tcp
    current_rotation_error = Rotation.from_matrix(
        skill_t_current[:3, :3]
    ).magnitude()
    if current_rotation_error < np.deg2rad(170.0):
        # The endpoint IK was verified above.  A direction hint is only
        # meaningful for the +/-pi ambiguity.
        print(
            "[expert] high-clearance alignment endpoint verified; "
            "no pi rotation hint required",
            flush=True,
        )
        return True, None, branch
    skill_t_probe = np.linalg.inv(world_t_goal_tcp) @ probe_tcp
    hint = Rotation.from_matrix(
        skill_t_probe[:3, :3] @ skill_t_current[:3, :3].T
    ).as_rotvec()
    hint_norm = float(np.linalg.norm(hint))
    if hint_norm < 1e-9:
        return False, None, None
    result = hint / hint_norm
    print(
        f"[expert] high-clearance alignment rotation hint={result.tolist()}",
        flush=True,
    )
    return True, result, branch


def _build_alignment_ik_branch(
    robot_pin,
    current_tcp: np.ndarray,
    aligned_tcp: np.ndarray,
    current_q: np.ndarray,
    rotation_step: float,
    *,
    verbose: bool = True,
) -> AlignmentIKBranch | None:
    """Continue IK from the physical safe start over Cartesian yaw.

    Returns:
        Verified seeds indexed by remaining rotation, or ``None`` if any
        intermediate high-clearance pose is not reachable.
    """
    if rotation_step <= 0.0:
        raise ValueError("rotation_step must be positive")
    aligned_rotation = Rotation.from_matrix(aligned_tcp[:3, :3])
    delta = Rotation.from_matrix(
        current_tcp[:3, :3] @ aligned_tcp[:3, :3].T
    ).as_rotvec()
    total_error = float(np.linalg.norm(delta))
    errors = [total_error]
    configurations = [np.asarray(current_q, dtype=np.float64).copy()]
    if total_error <= 1e-9:
        return AlignmentIKBranch(np.asarray(errors), tuple(configurations))
    error = max(0.0, total_error - rotation_step)
    seed = configurations[0]
    while True:
        pose = np.asarray(aligned_tcp, dtype=np.float64).copy()
        pose[:3, :3] = (
            Rotation.from_rotvec(delta * (error / total_error)).as_matrix()
            @ aligned_rotation.as_matrix()
        )
        solved = _try_verified_ik(robot_pin, pose, seed)
        if solved is None:
            if verbose:
                print(
                    "[expert] high-clearance IK continuation failed at "
                    f"remaining_rotation={np.rad2deg(error):.2f} deg",
                    flush=True,
                )
            return None
        errors.append(error)
        configurations.append(solved)
        seed = solved
        if error <= 1e-9:
            break
        error = max(0.0, error - rotation_step)
    order = np.argsort(errors)
    return AlignmentIKBranch(
        np.asarray(errors)[order],
        tuple(configurations[index] for index in order),
    )


def _alignment_branch_seed(
    branch: AlignmentIKBranch | None,
    target_rotation_error: float,
) -> np.ndarray | None:
    """Return the verified branch seed nearest a target rotation error.

    Returns:
        A deterministic nearby joint seed, or ``None`` without an endpoint.
    """
    if branch is None:
        return None
    index = int(
        np.argmin(
            np.abs(branch.rotation_errors - float(target_rotation_error))
        )
    )
    return branch.configurations[index].copy()


def _grasp_stability_error(env, prepared: SafeStart) -> str | None:
    """Return directional physical-grasp drift, or None while stable.

    Returns:
        Diagnostic string when the grasp exceeds the runtime limits.
    """
    q = _arm_configuration(env, prepared.arm_index)
    tcp = _tcp_world(env, prepared.arm_index, q)
    brick = env.get_prim_world_T(prepared.task.target_path)
    actual = np.linalg.inv(brick) @ tcp
    translation = actual[:3, 3] - prepared.brick_t_tcp[:3, 3]
    jaw_drift = abs(float(translation[prepared.grasp_axis]))
    finger_drift = abs(float(translation[1 - prepared.grasp_axis]))
    vertical_drift = abs(float(translation[2]))
    rotation_error = Rotation.from_matrix(
        actual[:3, :3].T @ prepared.brick_t_tcp[:3, :3]
    ).magnitude()
    if (
        jaw_drift < 0.025
        and finger_drift < 0.025
        and vertical_drift < 0.025
        and rotation_error < np.deg2rad(20.0)
    ):
        return None
    _, brick_goal_rotation = _pose_error(
        brick, prepared.world_t_goal_brick
    )
    _, tcp_goal_rotation = _pose_error(tcp, prepared.world_t_goal_tcp)
    return (
        f"jaw={jaw_drift:.4f} m, finger={finger_drift:.4f} m, "
        f"vertical={vertical_drift:.4f} m, "
        f"rotation={np.rad2deg(rotation_error):.2f} deg, "
        f"brick_goal_rotation={np.rad2deg(brick_goal_rotation):.2f} deg, "
        f"tcp_goal_rotation={np.rad2deg(tcp_goal_rotation):.2f} deg"
    )
