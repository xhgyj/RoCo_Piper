"""Insertion-axis-preserving Align primitive."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from rocobrick.execution.types import (
    ExecutionError,
    FailureCode,
    HeldObject,
    OperationResult,
)
from rocobrick.primitives.base import PrimitiveContext
from rocobrick.safety.checks import GraspTracking, SuccessCheck

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class AlignRequest:
    """Goal pose and axis that must remain fixed during alignment."""

    world_t_goal_tcp: FloatArray
    insertion_direction_world: FloatArray
    held: HeldObject
    stage: str = "align"
    lateral_tolerance: float = 0.0005
    rotation_tolerance: float = np.deg2rad(1.0)
    max_steps: int = 600
    settle_steps: int = 3
    success_check: SuccessCheck | None = None
    allow_vertical_settling: bool = True
    max_rotation_step: float = np.deg2rad(1.0)
    max_vertical_drift: float = 0.010
    max_grasp_rotation_drift: float = np.deg2rad(6.0)
    rotation_hint_goal: FloatArray | None = None

    def __post_init__(self) -> None:
        """Validate alignment tolerances and iteration bounds."""
        if (
            self.lateral_tolerance <= 0.0
            or self.rotation_tolerance <= 0.0
            or self.max_rotation_step <= 0.0
            or self.max_vertical_drift <= 0.0
            or self.max_grasp_rotation_drift <= 0.0
        ):
            raise ValueError("alignment tolerances must be positive")
        if self.max_steps <= 0 or self.settle_steps <= 0:
            raise ValueError("alignment step bounds must be positive")
        if self.rotation_hint_goal is not None:
            _unit_vector(self.rotation_hint_goal, "rotation_hint_goal")


class Align:
    """Align transverse position and orientation without descending."""

    def __init__(self, context: PrimitiveContext):
        """Bind shared primitive dependencies."""
        self._context = context

    async def execute(self, request: AlignRequest) -> OperationResult:
        """Servo to the goal manifold while preserving axial displacement.

        Returns:
            Controller completion metrics.
        """
        direction = _unit_vector(
            request.insertion_direction_world, "insertion_direction_world"
        )
        goal_tcp = np.asarray(request.world_t_goal_tcp, dtype=np.float64)
        goal_object = goal_tcp @ np.linalg.inv(request.held.object_t_tcp)
        rotation_hint_world = None
        if request.rotation_hint_goal is not None:
            rotation_hint_world = goal_tcp[:3, :3] @ _unit_vector(
                request.rotation_hint_goal, "rotation_hint_goal"
            )
        tracking = GraspTracking(
            request.held.object_id,
            request.held.object_t_tcp,
            request.held.grasp_axis,
            request.held.grasp_width,
        )
        settled = 0
        lateral_error = 0.0
        rotation_error = 0.0
        for step in range(1, request.max_steps + 1):
            if (
                request.success_check is not None
                and request.success_check.is_satisfied()
            ):
                return OperationResult(step - 1, lateral_error, rotation_error)
            state = self._context.robot.read_state()
            current_object = self._context.world.object_pose(
                request.held.object_id
            )
            displacement = current_object[:3, 3] - goal_object[:3, 3]
            axial_displacement = float(np.dot(displacement, direction))
            lateral = displacement - direction * axial_displacement
            lateral_error = float(np.linalg.norm(lateral))
            rotation_error = float(
                Rotation.from_matrix(
                    current_object[:3, :3].T @ goal_object[:3, :3]
                ).magnitude()
            )
            reached = (
                lateral_error <= request.lateral_tolerance
                and rotation_error <= request.rotation_tolerance
            )
            settled = settled + 1 if reached else 0
            if settled >= request.settle_steps:
                print(
                    f"[motion] {request.stage}: {step} steps, "
                    f"lateral={lateral_error:.4f} m, "
                    f"rotation={np.rad2deg(rotation_error):.2f} deg",
                    flush=True,
                )
                return OperationResult(step, lateral_error, rotation_error)
            target_object = goal_object.copy()
            target_object[:3, 3] += direction * axial_displacement
            current_object_t_tcp = np.linalg.inv(current_object) @ state.tcp_world
            target_tcp = target_object @ current_object_t_tcp
            await self._context.cartesian.servo_step(
                target_tcp,
                request.stage,
                request.held.grasp_width,
                closed=True,
                tracking=tracking,
                allow_vertical_settling=request.allow_vertical_settling,
                rotation_step_limit=request.max_rotation_step,
                max_vertical_drift=request.max_vertical_drift,
                rotation_direction_hint_world=rotation_hint_world,
                max_rotation_drift=request.max_grasp_rotation_drift,
            )
        raise ExecutionError(
            FailureCode.TIMEOUT,
            request.stage,
            f"alignment did not settle after {request.max_steps} steps "
            f"(lateral={lateral_error:.4f} m, "
            f"rotation={np.rad2deg(rotation_error):.2f} deg)",
        )


def _unit_vector(value: FloatArray, name: str) -> FloatArray:
    """Validate and normalize one Cartesian direction.

    Returns:
        Normalized three-vector.
    """
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        raise ValueError(f"{name} must be nonzero")
    return vector / norm
