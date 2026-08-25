"""Grasp primitive with gripper-feedback verification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rocobrick.execution.types import ExecutionError, FailureCode, OperationResult
from rocobrick.primitives.base import PrimitiveContext


@dataclass(frozen=True)
class GripperConfig:
    """Gripper motion limits retained from the original expert."""

    max_steps: int = 90
    min_steps: int = 6
    stable_steps: int = 4
    stall_delta: float = 0.00015
    open_tolerance: float = 0.0015
    contact_width_tolerance: float = 0.004
    max_joint_step: float = 0.001


class Grasp:
    """Prepare and close a parallel gripper around an object."""

    def __init__(
        self,
        context: PrimitiveContext,
        config: GripperConfig | None = None,
    ):
        """Bind primitive dependencies and gripper thresholds."""
        self._context = context
        self._config = config or GripperConfig()

    async def prepare(
        self, object_width: float, allow_constrained: bool = False
    ) -> OperationResult:
        """Open and verify clearance for the requested object width.

        Returns:
            Gripper completion metrics.
        """
        return await self._actuate(
            object_width, closed=False, allow_constrained=allow_constrained
        )

    async def execute(self, object_width: float) -> OperationResult:
        """Close and verify stable contact around the requested width.

        Returns:
            Gripper completion metrics.
        """
        return await self._actuate(object_width, closed=True)

    async def _actuate(
        self,
        object_width: float,
        closed: bool,
        allow_constrained: bool = False,
    ) -> OperationResult:
        target = self._context.robot.with_gripper(
            self._context.robot.read_state().q,
            object_width,
            closed=closed,
        )
        gripper_indices = self._context.robot.gripper_configuration_indices
        target_values = target[list(gripper_indices)]
        stable = 0
        actual_values = self._context.robot.read_state().gripper_positions
        for step in range(1, self._config.max_steps + 1):
            before = self._context.robot.read_state()
            # Hold the arm at the pose captured when gripper actuation began.
            # Copying the measured arm state here would accept contact-induced
            # drift and can leave one finger stalled against the brick edge.
            command = target.copy()
            for index in gripper_indices:
                delta = target[index] - before.q[index]
                command[index] = before.q[index] + np.clip(
                    delta,
                    -self._config.max_joint_step,
                    self._config.max_joint_step,
                )
            self._context.collision.require_safe(command, "grasp")
            self._context.robot.command_configuration(command)
            await self._context.world.advance(2)
            after = self._context.robot.read_state()
            actual_values = after.gripper_positions
            movement = float(
                np.max(np.abs(actual_values - before.gripper_positions))
            )
            stalled = movement <= self._config.stall_delta
            if closed:
                gap = after.gripper_width
                reached = (
                    abs(gap - object_width)
                    <= self._config.contact_width_tolerance
                    and stalled
                )
            else:
                error = float(np.max(np.abs(actual_values - target_values)))
                reached = error <= self._config.open_tolerance or (
                    allow_constrained and stalled
                )
            stable = stable + 1 if reached else 0
            if step >= self._config.min_steps and stable >= self._config.stable_steps:
                if closed:
                    state = "closed on target"
                elif error <= self._config.open_tolerance:
                    state = "at planned opening"
                else:
                    state = "constrained; retreat required"
                print(
                    f"[motion] gripper {state}: steps={step}, "
                    f"joints={actual_values.tolist()}",
                    flush=True,
                )
                return OperationResult(step)
        gap = float(np.sum(np.abs(actual_values)))
        expected = object_width if closed else float(np.sum(np.abs(target_values)))
        raise ExecutionError(
            FailureCode.GRIPPER_FAILED,
            "grasp" if closed else "pregrasp_open",
            f"gripper verification failed: gap={gap:.4f} m, "
            f"expected={expected:.4f} m",
        )
