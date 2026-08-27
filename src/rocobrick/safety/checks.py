"""Safety checks shared by manipulation primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from rocobrick.backends.base import RobotBackend, WorldModel
from rocobrick.execution.types import ExecutionError, FailureCode

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class GraspTracking:
    """Reference transform and geometry for held-object monitoring."""

    object_id: str
    object_t_tcp: FloatArray
    grasp_axis: int
    grasp_width: float


@dataclass(frozen=True)
class GraspStabilityConfig:
    """Allowed held-object drift during transport."""

    max_finger_axis_drift: float = 0.004
    max_vertical_drift: float = 0.004
    max_initial_vertical_settling: float = 0.008
    max_rotation_drift: float = np.deg2rad(5.0)
    comparison_tolerance: float = 0.0002


class CollisionCheck:
    """Single checking point for controller configuration commands."""

    def __init__(self, robot: RobotBackend):
        """Bind the robot-specific collision and limit implementation."""
        self._robot = robot

    def require_safe(self, q: FloatArray, stage: str) -> None:
        """Reject a configuration that fails backend checks."""
        if not self._robot.configuration_is_safe(q):
            raise ExecutionError(
                FailureCode.COLLISION,
                stage,
                "commanded configuration failed safety checks",
            )


class SuccessCheck(Protocol):
    """Semantic postcondition injected into contact-phase primitives."""

    def is_satisfied(self) -> bool:
        """Return whether every requested task condition is active."""
        ...

    def conflict(self) -> str | None:
        """Return a conflicting terminal condition, if one exists."""
        ...

    def require_satisfied(self, stage: str) -> None:
        """Raise when the requested task condition is not active."""
        ...


@dataclass(frozen=True)
class ForceGuard:
    """Reject excessive total or insertion-axis force."""

    max_force: float
    max_axial_force: float

    def __post_init__(self) -> None:
        """Require positive force limits."""
        if self.max_force <= 0.0 or self.max_axial_force <= 0.0:
            raise ValueError("force limits must be positive")

    def require_safe(
        self, wrench_world: FloatArray, direction_world: FloatArray, stage: str
    ) -> None:
        """Raise when a wrench exceeds configured force limits."""
        wrench = np.asarray(wrench_world, dtype=np.float64)
        direction = np.asarray(direction_world, dtype=np.float64)
        if wrench.shape != (6,) or not np.isfinite(wrench).all():
            raise ValueError("wrench_world must be a finite 6-vector")
        if direction.shape != (3,) or not np.isfinite(direction).all():
            raise ValueError("direction_world must be a finite 3-vector")
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            raise ValueError("direction_world must be nonzero")
        force = wrench[:3]
        total = float(np.linalg.norm(force))
        axial = abs(float(np.dot(force, direction / norm)))
        if total > self.max_force or axial > self.max_axial_force:
            raise ExecutionError(
                FailureCode.EXCESS_FORCE,
                stage,
                f"force limit exceeded: total={total:.3f}/"
                f"{self.max_force:.3f} N, axial={axial:.3f}/"
                f"{self.max_axial_force:.3f} N",
            )


@dataclass(frozen=True)
class ContactCheck:
    """Detect directional contact from force and commanded travel."""

    min_axial_force: float = 0.5
    min_travel: float = 0.0

    def __post_init__(self) -> None:
        """Validate non-negative contact thresholds."""
        if self.min_axial_force < 0.0 or self.min_travel < 0.0:
            raise ValueError("contact thresholds must be non-negative")

    def detected(
        self,
        wrench_world: FloatArray,
        direction_world: FloatArray,
        travel: float,
    ) -> bool:
        """Return whether force after minimum travel indicates contact."""
        wrench = np.asarray(wrench_world, dtype=np.float64)
        direction = np.asarray(direction_world, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if wrench.shape != (6,) or direction.shape != (3,) or norm < 1e-9:
            raise ValueError("contact check requires a wrench and direction")
        axial = abs(float(np.dot(wrench[:3], direction / norm)))
        return travel >= self.min_travel and axial >= self.min_axial_force


class GraspStabilityCheck:
    """Detect object motion relative to a supposedly rigid grasp."""

    def __init__(
        self,
        robot: RobotBackend,
        world: WorldModel,
        config: GraspStabilityConfig | None = None,
    ):
        """Bind robot/world feedback and drift thresholds."""
        self._robot = robot
        self._world = world
        self._config = config or GraspStabilityConfig()

    def require_stable(
        self,
        tracking: GraspTracking,
        stage: str,
        allow_vertical_settling: bool = False,
        max_vertical_drift: float | None = None,
        max_rotation_drift: float | None = None,
    ) -> None:
        """Reject excessive translation or rotation inside the gripper."""
        tcp = self._robot.read_state().tcp_world
        object_pose = self._world.object_pose(tracking.object_id)
        actual_object_t_tcp = np.linalg.inv(object_pose) @ tcp
        translation = actual_object_t_tcp[:3, 3] - tracking.object_t_tcp[:3, 3]
        rotation = Rotation.from_matrix(
            actual_object_t_tcp[:3, :3].T @ tracking.object_t_tcp[:3, :3]
        ).magnitude()
        jaw_drift = abs(float(translation[tracking.grasp_axis]))
        finger_drift = abs(float(translation[1 - tracking.grasp_axis]))
        vertical_drift = abs(float(translation[2]))
        jaw_limit = min(0.008, max(0.004, tracking.grasp_width * 0.5))
        if max_vertical_drift is None:
            vertical_limit = (
                self._config.max_initial_vertical_settling
                if allow_vertical_settling
                else self._config.max_vertical_drift
            )
        else:
            if max_vertical_drift <= 0.0:
                raise ValueError("max_vertical_drift must be positive")
            vertical_limit = max_vertical_drift
        rotation_limit = (
            self._config.max_rotation_drift
            if max_rotation_drift is None
            else max_rotation_drift
        )
        if rotation_limit <= 0.0:
            raise ValueError("max_rotation_drift must be positive")
        tolerance = self._config.comparison_tolerance
        if (
            jaw_drift > jaw_limit + tolerance
            or finger_drift > self._config.max_finger_axis_drift + tolerance
            or vertical_drift > vertical_limit + tolerance
            or rotation > rotation_limit
        ):
            raise ExecutionError(
                FailureCode.SLIPPED,
                stage,
                "target slipped in gripper "
                f"(jaw={jaw_drift:.6f}/{jaw_limit:.6f} m, "
                f"finger={finger_drift:.6f}/"
                f"{self._config.max_finger_axis_drift:.6f} m, "
                f"vertical={vertical_drift:.6f}/{vertical_limit:.6f} m, "
                f"rotation={np.rad2deg(rotation):.2f}/"
                f"{np.rad2deg(rotation_limit):.2f} deg)",
            )
