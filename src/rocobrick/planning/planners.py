"""Robot-constrained grasp, transport, and assembly planners."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from rocobrick.backends.base import RobotBackend
from rocobrick.execution.types import ExecutionError, FailureCode, HeldObjectState
from rocobrick.planning.geometry import (
    gripper_boxes,
    interpolate_pose,
    interpolate_poses,
    obb_intersects,
    path_clearance,
)
from rocobrick.planning.models import (
    AssemblyGoal,
    AssemblyPlan,
    CartesianPath,
    FloatArray,
    GraspPlan,
    GraspRegion,
    HeldMotionPlan,
    ObjectGeometry,
    SceneGeometry,
)


@dataclass(frozen=True)
class PlannerConfig:
    """Geometry-derived planning limits with no structure-specific cases."""

    pick_clearance: float = 0.06
    assembly_clearance: float = 0.06
    retreat_distance: float = 0.06
    insertion_margin: float = 0.004
    minimum_side_clearance: float = 0.002
    sample_spacing: float = 0.012
    precision_ik_step: float = 0.002
    held_ik_step: float = 0.004
    maximum_ik_joint_step: float = 0.10
    maximum_ik_subdivision_depth: int = 10


def _ik_path(
    robot: RobotBackend,
    poses: tuple[FloatArray, ...],
    seed: FloatArray,
    stage: str,
) -> CartesianPath:
    configurations: list[FloatArray] = []
    current = np.asarray(seed, dtype=np.float64).copy()
    for pose in poses:
        solution = robot.solve_ik(pose, current)
        if solution is None:
            raise ExecutionError(
                FailureCode.UNREACHABLE, stage, "continuous IK path is unreachable"
            )
        if not robot.configuration_is_safe(solution):
            raise ExecutionError(
                FailureCode.COLLISION, stage, "IK path contains an unsafe configuration"
            )
        current = np.asarray(solution, dtype=np.float64).copy()
        configurations.append(current)
    return CartesianPath(poses, tuple(configurations))


def _adaptive_ik_path(
    robot: RobotBackend,
    poses: tuple[FloatArray, ...],
    seed: FloatArray,
    stage: str,
    maximum_joint_step: float,
    maximum_depth: int,
    goal_configuration: FloatArray | None = None,
) -> CartesianPath:
    """Solve a path and subdivide segments with excessive joint motion.

    Returns:
        Continuous IK branch with inserted poses where required.
    """
    if not poses:
        raise ValueError("adaptive IK path cannot be empty")
    solved_poses: list[FloatArray] = [poses[0]]
    first = np.asarray(seed, dtype=np.float64).copy()
    if not robot.configuration_is_safe(first):
        raise ExecutionError(
            FailureCode.COLLISION,
            stage,
            "adaptive IK path starts from an unsafe configuration",
        )
    solved_configurations: list[FloatArray] = [first]
    arm_indices = list(robot.arm_configuration_indices)

    def solve_segment(
        start_pose: FloatArray,
        goal_pose: FloatArray,
        start_q: FloatArray,
        depth: int,
        fixed_goal_q: FloatArray | None = None,
    ) -> list[tuple[FloatArray, FloatArray]]:
        if fixed_goal_q is None:
            goal_q = _ik_path(robot, (goal_pose,), start_q, stage).configurations[0]
        else:
            goal_q = np.asarray(fixed_goal_q, dtype=np.float64).copy()
            if not robot.configuration_is_safe(goal_q):
                raise ExecutionError(
                    FailureCode.COLLISION,
                    stage,
                    "cached IK endpoint is an unsafe configuration",
                )
        joint_step = float(np.max(np.abs(goal_q[arm_indices] - start_q[arm_indices])))
        if joint_step <= maximum_joint_step:
            return [(goal_pose, goal_q)]
        if depth >= maximum_depth:
            largest_index = arm_indices[
                int(np.argmax(np.abs(goal_q[arm_indices] - start_q[arm_indices])))
            ]
            raise ExecutionError(
                FailureCode.UNREACHABLE,
                stage,
                "IK branch requires excessive joint motion after subdivision "
                f"(q[{largest_index}] delta={joint_step:.3f} rad)",
            )
        midpoint = interpolate_pose(start_pose, goal_pose, 0.5)
        first_half = solve_segment(start_pose, midpoint, start_q, depth + 1)
        second_half = solve_segment(
            midpoint,
            goal_pose,
            first_half[-1][1],
            depth + 1,
            fixed_goal_q,
        )
        return [*first_half, *second_half]

    for index, goal_pose in enumerate(poses[1:], start=1):
        fixed_goal_q = goal_configuration if index == len(poses) - 1 else None
        segment = solve_segment(
            solved_poses[-1],
            goal_pose,
            solved_configurations[-1],
            0,
            fixed_goal_q,
        )
        solved_poses.extend(item[0] for item in segment)
        solved_configurations.extend(item[1] for item in segment)
    return CartesianPath(tuple(solved_poses), tuple(solved_configurations))


def _concatenate_paths(paths: tuple[CartesianPath, ...]) -> CartesianPath:
    poses: list[FloatArray] = []
    configurations: list[FloatArray] = []
    for path in paths:
        start = 1 if poses else 0
        poses.extend(path.poses[start:])
        configurations.extend(path.configurations[start:])
    return CartesianPath(tuple(poses), tuple(configurations))


def _joint_space_path(
    robot: RobotBackend,
    start: FloatArray,
    goal: FloatArray,
    maximum_joint_step: float,
    stage: str,
) -> CartesianPath:
    """Interpolate a free-space joint move and recover its actual TCP path.

    Returns:
        Safe configurations paired with their forward-kinematic TCP poses.
    """
    arm_indices = list(robot.arm_configuration_indices)
    distance = float(np.max(np.abs(goal[arm_indices] - start[arm_indices])))
    count = max(1, int(np.ceil(distance / maximum_joint_step)))
    configurations: list[FloatArray] = []
    poses: list[FloatArray] = []
    for index in range(count + 1):
        fraction = index / count
        configuration = start * (1.0 - fraction) + goal * fraction
        if not robot.configuration_is_safe(configuration):
            raise ExecutionError(
                FailureCode.COLLISION,
                stage,
                "joint-space path contains an unsafe configuration",
            )
        configurations.append(configuration)
        poses.append(robot.forward_kinematics(configuration))
    return CartesianPath(tuple(poses), tuple(configurations))


@dataclass(frozen=True)
class _RankedGrasp:
    """Geometry-ranked candidate awaiting full continuous IK verification."""

    score: tuple[float, ...]
    index: int
    grasp_axis: int
    width: float
    minimum_clearance: float
    stability: float
    pregrasp_tcp: FloatArray
    grasp_tcp: FloatArray
    lift_tcp: FloatArray


class GraspPlanner:
    """Generate and rank side grasps for exactly one assigned robot."""

    def __init__(self, config: PlannerConfig | None = None):
        """Store generic sampling and clearance limits."""
        self._config = config or PlannerConfig()

    def plan(
        self,
        robot: RobotBackend,
        scene: SceneGeometry,
        assembly_goal: AssemblyGoal | None = None,
    ) -> GraspPlan:
        """Return the best fully verified grasp without changing robot assignment."""
        target = scene.target
        geometry = scene.gripper
        candidates: list[_RankedGrasp] = []
        rejections: Counter[str] = Counter()
        current = robot.read_state()
        candidate_index = 0
        region_axes = (
            (region, axis)
            for region in target.grasp_regions
            for axis in region.allowed_jaw_axes
        )
        for region, grasp_axis in region_axes:
            width = float(region.half_extents[grasp_axis] * 2.0)
            if not geometry.minimum_opening <= width <= geometry.maximum_opening:
                rejections["opening"] += 1
                continue
            sample_axis = 1 - grasp_axis
            available = max(
                0.0,
                float(region.half_extents[sample_axis]) - geometry.finger_thickness,
            )
            sample_count = max(
                1, int(np.floor(available * 2 / self._config.sample_spacing)) + 1
            )
            if sample_count > 1 and sample_count % 2 == 0:
                sample_count += 1
            offsets = (
                np.array([0.0])
                if sample_count == 1
                else np.linspace(-available, available, sample_count)
            )
            for offset in offsets:
                for sign in (1.0, -1.0):
                    index = candidate_index
                    candidate_index += 1
                    grasp_tcp = self._grasp_pose(
                        target,
                        region,
                        grasp_axis,
                        sign,
                        float(offset),
                    )
                    pregrasp_tcp = grasp_tcp.copy()
                    pregrasp_tcp[:3, 3] += (
                        target.world_t_object[:3, 2] * self._config.pick_clearance
                    )
                    lift_tcp = grasp_tcp.copy()
                    lift_tcp[:3, 3] += (
                        target.world_t_object[:3, 2] * self._config.pick_clearance
                    )
                    collision_poses = (
                        interpolate_poses(pregrasp_tcp, grasp_tcp),
                        interpolate_poses(grasp_tcp, lift_tcp),
                    )
                    clearance_values = tuple(
                        path_clearance(path, width, geometry, scene.obstacles)
                        for path in collision_poses
                    )
                    minimum_clearance = (
                        min(value for value in clearance_values if value is not None)
                        if all(value is not None for value in clearance_values)
                        else -np.inf
                    )
                    if minimum_clearance < self._config.minimum_side_clearance:
                        rejections["pick_collision"] += 1
                        continue
                    centering = 1.0 - (0.5 * abs(float(offset)) / max(available, 1e-9))
                    stability = (
                        width / float(max(region.half_extents[:2]) * 2.0) * centering
                    )
                    path_length = float(
                        np.linalg.norm(pregrasp_tcp[:3, 3] - current.tcp_world[:3, 3])
                        + np.linalg.norm(grasp_tcp[:3, 3] - pregrasp_tcp[:3, 3])
                        + np.linalg.norm(lift_tcp[:3, 3] - grasp_tcp[:3, 3])
                    )
                    score = (
                        -minimum_clearance,
                        -stability,
                        path_length,
                        float(index),
                    )
                    candidates.append(
                        _RankedGrasp(
                            score,
                            index,
                            grasp_axis,
                            width,
                            minimum_clearance,
                            stability,
                            pregrasp_tcp,
                            grasp_tcp,
                            lift_tcp,
                        )
                    )
        for candidate in sorted(candidates, key=lambda item: item.score):
            try:
                q_pregrasp = _ik_path(
                    robot,
                    (candidate.pregrasp_tcp,),
                    current.q,
                    "pick_pregrasp_probe",
                ).configurations[-1]
                q_grasp = _ik_path(
                    robot,
                    (candidate.grasp_tcp,),
                    q_pregrasp,
                    "pick_grasp_probe",
                ).configurations[-1]
                q_lift = _ik_path(
                    robot,
                    (candidate.lift_tcp,),
                    q_grasp,
                    "pick_lift_probe",
                ).configurations[-1]
                if assembly_goal is not None:
                    high_tcp = self._downstream_tcp(
                        target, candidate.grasp_tcp, assembly_goal
                    )
                    _ik_path(
                        robot,
                        (high_tcp,),
                        q_lift,
                        "downstream_assembly_probe",
                    )
                pregrasp = _joint_space_path(
                    robot,
                    current.q,
                    q_pregrasp,
                    self._config.maximum_ik_joint_step,
                    "pick_pregrasp",
                )
                free_clearance = path_clearance(
                    pregrasp.poses,
                    candidate.width,
                    geometry,
                    scene.obstacles,
                )
                if (
                    free_clearance is None
                    or free_clearance < self._config.minimum_side_clearance
                ):
                    raise ExecutionError(
                        FailureCode.COLLISION,
                        "pick_pregrasp",
                        "joint-space gripper path intersects scene geometry",
                    )
                approach = _adaptive_ik_path(
                    robot,
                    interpolate_poses(
                        candidate.pregrasp_tcp,
                        candidate.grasp_tcp,
                        translation_step=self._config.precision_ik_step,
                    ),
                    pregrasp.configurations[-1],
                    "pick_approach",
                    self._config.maximum_ik_joint_step,
                    self._config.maximum_ik_subdivision_depth,
                    q_grasp,
                )
                lift = _adaptive_ik_path(
                    robot,
                    interpolate_poses(
                        candidate.grasp_tcp,
                        candidate.lift_tcp,
                        translation_step=self._config.precision_ik_step,
                    ),
                    approach.configurations[-1],
                    "pick_lift",
                    self._config.maximum_ik_joint_step,
                    self._config.maximum_ik_subdivision_depth,
                    q_lift,
                )
                if assembly_goal is not None:
                    self._require_downstream_transport(
                        robot,
                        scene,
                        target,
                        candidate.grasp_tcp,
                        candidate.lift_tcp,
                        candidate.width,
                        assembly_goal,
                        lift.configurations[-1],
                    )
            except ExecutionError as error:
                rejections[f"{error.stage}:{error.code.value}:{error.detail}"] += 1
                continue
            return GraspPlan(
                robot.robot_id,
                target.object_id,
                np.linalg.inv(target.world_t_object) @ candidate.grasp_tcp,
                candidate.grasp_axis,
                candidate.width,
                pregrasp,
                approach,
                lift,
                min(candidate.minimum_clearance, free_clearance),
                candidate.stability,
                candidate.index,
            )
        rejection_detail = ", ".join(
            f"{reason}={count}" for reason, count in sorted(rejections.items())
        )
        suffix = f"; rejections: {rejection_detail}" if rejection_detail else ""
        raise ExecutionError(
            FailureCode.UNREACHABLE,
            "grasp_planning",
            "assigned robot has no collision-free grasp with continuous IK" + suffix,
        )

    @staticmethod
    def _grasp_pose(
        target: ObjectGeometry,
        region: GraspRegion,
        grasp_axis: int,
        sign: float,
        sample_offset: float,
    ) -> FloatArray:
        world_t_region = target.world_t_object @ region.object_t_region
        y_axis = sign * world_t_region[:3, grasp_axis]
        z_axis = -world_t_region[:3, 2]
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        pose = np.eye(4)
        pose[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
        pose[:3, 3] = (
            world_t_region[:3, 3] + world_t_region[:3, 1 - grasp_axis] * sample_offset
        )
        return pose

    def _require_downstream_transport(
        self,
        robot: RobotBackend,
        scene: SceneGeometry,
        target: ObjectGeometry,
        grasp_tcp: FloatArray,
        lift_tcp: FloatArray,
        grasp_width: float,
        goal: AssemblyGoal,
        seed: FloatArray,
    ) -> None:
        object_t_tcp = np.linalg.inv(target.world_t_object) @ grasp_tcp
        tcp_t_object = np.linalg.inv(object_t_tcp)
        lifted_object = lift_tcp @ tcp_t_object
        direction = goal.insertion_direction_world / np.linalg.norm(
            goal.insertion_direction_world
        )
        high_current_object = lifted_object.copy()
        high_current_object[:3, 3] -= direction * self._config.assembly_clearance
        high_goal_object = goal.world_t_goal_object.copy()
        high_goal_object[:3, 3] -= direction * self._config.assembly_clearance
        preassembly_object = goal.world_t_goal_object.copy()
        preassembly_object[:3, 3] -= direction * self._config.insertion_margin
        waypoints = (
            lift_tcp,
            high_current_object @ object_t_tcp,
            high_goal_object @ object_t_tcp,
            preassembly_object @ object_t_tcp,
        )
        current_seed = seed
        for segment_index, (start, end) in enumerate(zip(waypoints, waypoints[1:])):
            collision_poses = interpolate_poses(start, end)
            if (
                path_clearance(
                    collision_poses,
                    grasp_width,
                    scene.gripper,
                    scene.obstacles,
                )
                is None
            ):
                for pose_index, tcp_pose in enumerate(collision_poses):
                    for gripper_box in gripper_boxes(
                        tcp_pose, grasp_width, scene.gripper
                    ):
                        for obstacle in scene.obstacles:
                            if obb_intersects(gripper_box, obstacle):
                                fraction = pose_index / max(1, len(collision_poses) - 1)
                                raise ExecutionError(
                                    FailureCode.COLLISION,
                                    "downstream_assembly",
                                    "gripper collision on segment "
                                    f"{segment_index} at {fraction:.3f}: "
                                    f"{gripper_box.object_id} intersects "
                                    f"{obstacle.object_id}",
                                )
                raise AssertionError("collision must identify intersecting boxes")
            for tcp_pose in collision_poses:
                held_box = target.collision_box(tcp_pose @ tcp_t_object)
                for obstacle in scene.obstacles:
                    if obb_intersects(held_box, obstacle):
                        raise ExecutionError(
                            FailureCode.COLLISION,
                            "downstream_assembly",
                            f"held-object swept volume intersects {obstacle.object_id}",
                        )
            path = _adaptive_ik_path(
                robot,
                interpolate_poses(
                    start,
                    end,
                    translation_step=self._config.held_ik_step,
                ),
                current_seed,
                "downstream_assembly",
                self._config.maximum_ik_joint_step,
                self._config.maximum_ik_subdivision_depth,
            )
            current_seed = path.configurations[-1]

    def _downstream_tcp(
        self,
        target: ObjectGeometry,
        grasp_tcp: FloatArray,
        goal: AssemblyGoal,
    ) -> FloatArray:
        """Return the high-clearance downstream TCP endpoint.

        Returns:
            Goal-aligned TCP pose above the mating target.
        """
        object_t_tcp = np.linalg.inv(target.world_t_object) @ grasp_tcp
        high_object = goal.world_t_goal_object.copy()
        direction = goal.insertion_direction_world / np.linalg.norm(
            goal.insertion_direction_world
        )
        high_object[:3, 3] -= direction * self._config.assembly_clearance
        return high_object @ object_t_tcp


class AssemblyPlanner:
    """Plan held-object transport and one downward mating operation."""

    def __init__(self, config: PlannerConfig | None = None):
        """Store generic geometry and motion limits."""
        self._config = config or PlannerConfig()

    def plan(
        self,
        robot: RobotBackend,
        scene: SceneGeometry,
        held_state: HeldObjectState,
        goal: AssemblyGoal,
    ) -> AssemblyPlan:
        """Plan a continuous IK branch from current grasp to preassembly.

        Returns:
            Geometry-derived transport and insertion parameters.
        """
        if held_state.held.robot_id != robot.robot_id:
            raise ValueError("held object is assigned to a different robot")
        direction = np.asarray(goal.insertion_direction_world, dtype=np.float64)
        direction /= np.linalg.norm(direction)
        object_t_tcp = held_state.acquisition_object_t_tcp
        current_tcp = robot.read_state().tcp_world
        current_object = current_tcp @ np.linalg.inv(object_t_tcp)
        goal_tcp = goal.world_t_goal_object @ object_t_tcp
        clearance = max(
            self._config.assembly_clearance,
            float(
                scene.target.collision_half_extents[2] * 2.0
                + self._config.insertion_margin
            ),
        )
        high_current_object = current_object.copy()
        high_current_object[:3, 3] -= direction * clearance
        high_goal_object = goal.world_t_goal_object.copy()
        high_goal_object[:3, 3] -= direction * clearance
        preassembly_object = goal.world_t_goal_object.copy()
        preassembly_object[:3, 3] -= direction * self._config.insertion_margin
        tcp_waypoints = (
            current_tcp,
            high_current_object @ object_t_tcp,
            high_goal_object @ object_t_tcp,
            preassembly_object @ object_t_tcp,
        )
        paths: list[CartesianPath] = []
        seed = robot.read_state().q
        minimum_clearance = np.inf
        for start, end in zip(tcp_waypoints, tcp_waypoints[1:]):
            collision_poses = interpolate_poses(start, end)
            self._require_held_clearance(
                collision_poses,
                object_t_tcp,
                scene,
                held_state.held.grasp_width,
            )
            clearance_value = path_clearance(
                collision_poses,
                held_state.held.grasp_width,
                scene.gripper,
                scene.obstacles,
            )
            if clearance_value is None:
                raise ExecutionError(
                    FailureCode.COLLISION,
                    "assembly_transport",
                    "gripper swept volume intersects scene geometry",
                )
            minimum_clearance = min(minimum_clearance, clearance_value)
            poses = interpolate_poses(
                start,
                end,
                translation_step=self._config.held_ik_step,
            )
            path = _adaptive_ik_path(
                robot,
                poses,
                seed,
                "assembly_transport",
                self._config.maximum_ik_joint_step,
                self._config.maximum_ik_subdivision_depth,
            )
            paths.append(path)
            seed = path.configurations[-1]
        stability = float(
            np.clip(
                held_state.held.grasp_width
                / max(
                    float(scene.target.collision_half_extents[0] * 2),
                    float(scene.target.collision_half_extents[1] * 2),
                ),
                0.0,
                1.0,
            )
        )
        rotation_step = np.deg2rad(1.0 + 3.0 * stability)
        insertion_distance = (
            float(scene.target.collision_half_extents[2] * 2.0)
            + self._config.insertion_margin
        )
        return AssemblyPlan(
            robot.robot_id,
            goal,
            HeldMotionPlan(
                robot.robot_id, _concatenate_paths(tuple(paths)), minimum_clearance
            ),
            preassembly_object @ object_t_tcp,
            goal_tcp,
            insertion_distance,
            min(0.0005, insertion_distance / 20.0),
            self._config.retreat_distance,
            rotation_step,
        )

    @staticmethod
    def _require_held_clearance(
        poses: tuple[FloatArray, ...],
        object_t_tcp: FloatArray,
        scene: SceneGeometry,
        grasp_width: float,
    ) -> None:
        del grasp_width
        tcp_t_object = np.linalg.inv(object_t_tcp)
        for tcp_pose in poses[:-1]:
            held_box = scene.target.collision_box(tcp_pose @ tcp_t_object)
            for obstacle in scene.obstacles:
                if obb_intersects(held_box, obstacle):
                    raise ExecutionError(
                        FailureCode.COLLISION,
                        "assembly_transport",
                        f"held object swept volume intersects {obstacle.object_id}",
                    )
