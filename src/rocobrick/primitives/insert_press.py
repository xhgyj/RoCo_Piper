"""Direction-parameterized InsertPress primitive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from rocobrick.execution.types import (
    ExecutionError,
    FailureCode,
    HeldObject,
    OperationResult,
)
from rocobrick.primitives.align import _unit_vector
from rocobrick.primitives.base import PrimitiveContext
from rocobrick.safety.checks import ForceGuard, GraspTracking, SuccessCheck

FloatArray = NDArray[np.float64]


class WrenchSource(Protocol):
    """Source of a world-frame TCP wrench for guarded contact motion."""

    def read_wrench_world(self) -> FloatArray:
        """Return force followed by torque as a finite six-vector."""
        ...


@dataclass(frozen=True)
class InsertPressRequest:
    """Bounded insertion motion and its semantic completion condition."""

    insertion_direction_world: FloatArray
    held: HeldObject
    success_check: SuccessCheck
    max_distance: float = 0.012
    step_distance: float = 0.0001
    max_steps: int = 180
    stage: str = "insert_press"
    wrench_source: WrenchSource | None = None
    force_guard: ForceGuard | None = None

    def __post_init__(self) -> None:
        """Validate motion bounds and force-feedback pairing."""
        _unit_vector(self.insertion_direction_world, "insertion_direction_world")
        if self.max_distance <= 0.0 or self.step_distance <= 0.0:
            raise ValueError("insertion distances must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.force_guard is not None and self.wrench_source is None:
            raise ValueError("ForceGuard requires a wrench source")


class InsertPress:
    """Advance along one axis until verified assembly or a safety limit."""

    def __init__(self, context: PrimitiveContext):
        """Bind shared primitive dependencies."""
        self._context = context

    async def execute(self, request: InsertPressRequest) -> OperationResult:
        """Perform bounded contact motion and require semantic success.

        Returns:
            Number of servo commands and final pose error.
        """
        direction = _unit_vector(
            request.insertion_direction_world, "insertion_direction_world"
        )
        tracking = GraspTracking(
            request.held.object_id,
            request.held.object_t_tcp,
            request.held.grasp_axis,
            request.held.grasp_width,
        )
        commanded = self._context.robot.read_state().tcp_world.copy()
        travelled = 0.0
        last_result = OperationResult(0)
        for step in range(1, request.max_steps + 1):
            conflict = request.success_check.conflict()
            if conflict is not None:
                raise ExecutionError(
                    FailureCode.VERIFICATION_FAILED,
                    request.stage,
                    conflict,
                )
            if request.success_check.is_satisfied():
                return OperationResult(
                    step - 1,
                    last_result.position_error,
                    last_result.rotation_error,
                )
            if request.force_guard is not None:
                assert request.wrench_source is not None
                request.force_guard.require_safe(
                    request.wrench_source.read_wrench_world(),
                    direction,
                    request.stage,
                )
            remaining = request.max_distance - travelled
            if remaining <= 1e-12:
                break
            increment = min(request.step_distance, remaining)
            commanded[:3, 3] += direction * increment
            last_result = await self._context.cartesian.servo_step(
                commanded,
                request.stage,
                request.held.grasp_width,
                closed=True,
                tracking=tracking,
            )
            travelled += increment
        conflict = request.success_check.conflict()
        if conflict is not None:
            raise ExecutionError(
                FailureCode.VERIFICATION_FAILED, request.stage, conflict
            )
        raise ExecutionError(
            FailureCode.CONTACT_FAILED,
            request.stage,
            f"no verified insertion after {travelled:.4f} m",
        )
