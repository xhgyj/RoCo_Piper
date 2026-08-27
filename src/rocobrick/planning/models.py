"""Immutable contracts between grounding, planning, and execution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rocobrick.safety.checks import SuccessCheck

FloatArray = NDArray[np.float64]


def _validate_transform(value: FloatArray, name: str) -> None:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{name} must be a finite 4x4 transform")


@dataclass(frozen=True)
class OrientedBox:
    """An object-space box and its world pose."""

    object_id: str
    world_t_box: FloatArray
    half_extents: FloatArray

    def __post_init__(self) -> None:
        """Validate pose and strictly positive box extents."""
        _validate_transform(self.world_t_box, "world_t_box")
        extents = np.asarray(self.half_extents, dtype=np.float64)
        if extents.shape != (3,) or not np.isfinite(extents).all():
            raise ValueError("half_extents must be a finite 3-vector")
        if np.any(extents <= 0.0):
            raise ValueError("half_extents must be positive")


@dataclass(frozen=True)
class GripperGeometry:
    """Conservative parallel-jaw geometry used during planning."""

    minimum_opening: float = 0.004
    maximum_opening: float = 0.072
    finger_length: float = 0.035
    finger_thickness: float = 0.006
    finger_height: float = 0.012
    palm_depth: float = 0.018
    palm_width: float = 0.085
    tcp_to_fingertip: float = 0.002


@dataclass(frozen=True)
class GraspRegion:
    """One object-local region in which parallel-jaw grasps may be sampled."""

    object_t_region: FloatArray
    half_extents: FloatArray
    allowed_jaw_axes: tuple[int, ...] = (0, 1)

    def __post_init__(self) -> None:
        """Validate the region frame, volume, and jaw axes."""
        _validate_transform(self.object_t_region, "object_t_region")
        extents = np.asarray(self.half_extents, dtype=np.float64)
        if extents.shape != (3,) or np.any(extents <= 0.0):
            raise ValueError("grasp region half_extents must be positive 3-vector")
        if not self.allowed_jaw_axes or any(
            axis not in (0, 1) for axis in self.allowed_jaw_axes
        ):
            raise ValueError("allowed_jaw_axes must contain only 0 and/or 1")


@dataclass(frozen=True)
class ObjectGeometry:
    """Separate object, collision, and grasp frames for one target."""

    object_id: str
    world_t_object: FloatArray
    object_t_collision: FloatArray
    collision_half_extents: FloatArray
    grasp_regions: tuple[GraspRegion, ...]

    def __post_init__(self) -> None:
        """Validate target geometry independently of simulator frame choices."""
        if not self.object_id:
            raise ValueError("object_id cannot be empty")
        _validate_transform(self.world_t_object, "world_t_object")
        _validate_transform(self.object_t_collision, "object_t_collision")
        extents = np.asarray(self.collision_half_extents, dtype=np.float64)
        if extents.shape != (3,) or np.any(extents <= 0.0):
            raise ValueError("collision_half_extents must be positive 3-vector")
        if not self.grasp_regions:
            raise ValueError("object geometry requires at least one grasp region")

    def collision_box(self, world_t_object: FloatArray | None = None) -> OrientedBox:
        """Build the collision OBB at a supplied or current object pose.

        Returns:
            Target collision box in the world frame.
        """
        object_pose = self.world_t_object if world_t_object is None else world_t_object
        return OrientedBox(
            self.object_id,
            object_pose @ self.object_t_collision,
            self.collision_half_extents,
        )


@dataclass(frozen=True)
class SceneGeometry:
    """Grounded geometry for the target, obstacles, and assigned gripper."""

    target: ObjectGeometry
    obstacles: tuple[OrientedBox, ...]
    gripper: GripperGeometry = GripperGeometry()


@dataclass(frozen=True)
class ConnectionSpec:
    """One exact semantic mating condition."""

    reference_id: str
    interface_id: str
    offset: tuple[int, int]
    yaw_quarter_turns: int


@dataclass(frozen=True)
class AssemblyGoal:
    """Robot-independent object goal for a downward assembly."""

    goal_id: str
    world_t_goal_object: FloatArray
    insertion_direction_world: FloatArray
    connections: tuple[ConnectionSpec, ...]
    allowed_contact_ids: tuple[str, ...]
    success_check: SuccessCheck

    def __post_init__(self) -> None:
        """Validate the goal frame and insertion direction."""
        _validate_transform(self.world_t_goal_object, "world_t_goal_object")
        direction = np.asarray(self.insertion_direction_world, dtype=np.float64)
        if direction.shape != (3,) or not np.isfinite(direction).all():
            raise ValueError("insertion_direction_world must be finite 3-vector")
        if np.linalg.norm(direction) < 1e-9:
            raise ValueError("insertion_direction_world must be nonzero")
        if not self.connections:
            raise ValueError("assembly goal requires at least one connection")


@dataclass(frozen=True)
class CartesianPath:
    """A continuous Cartesian path with a verified IK configuration per sample."""

    poses: tuple[FloatArray, ...]
    configurations: tuple[FloatArray, ...]

    def __post_init__(self) -> None:
        """Require matching nonempty pose and IK sequences."""
        if not self.poses or len(self.poses) != len(self.configurations):
            raise ValueError("path poses and configurations must match")


@dataclass(frozen=True)
class GraspPlan:
    """Fully verified grasp plan for one already-assigned robot."""

    robot_id: str
    object_id: str
    object_t_tcp: FloatArray
    grasp_axis: int
    grasp_width: float
    pregrasp: CartesianPath
    approach: CartesianPath
    lift: CartesianPath
    minimum_clearance: float
    stability_score: float
    candidate_index: int


@dataclass(frozen=True)
class HeldMotionPlan:
    """Verified free-space transport of a held object."""

    robot_id: str
    path: CartesianPath
    minimum_clearance: float


@dataclass(frozen=True)
class AssemblyPlan:
    """Grounded and verified downward assembly plan."""

    robot_id: str
    goal: AssemblyGoal
    transport: HeldMotionPlan
    world_t_preassembly_tcp: FloatArray
    world_t_goal_tcp: FloatArray
    insertion_distance: float
    insertion_step: float
    retreat_distance: float
    rotation_step: float


@dataclass(frozen=True)
class GroundedAction:
    """Geometry supplied by a domain adapter for one symbolic action."""

    scene: SceneGeometry
    assembly_goal: AssemblyGoal | None = None
