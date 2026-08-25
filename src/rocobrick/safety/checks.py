"""Safety checks shared by manipulation primitives."""

from __future__ import annotations

from dataclasses import dataclass

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
        vertical_limit = (
            self._config.max_initial_vertical_settling
            if allow_vertical_settling
            else self._config.max_vertical_drift
        )
        tolerance = self._config.comparison_tolerance
        if (
            jaw_drift > jaw_limit + tolerance
            or finger_drift > self._config.max_finger_axis_drift + tolerance
            or vertical_drift > vertical_limit + tolerance
            or rotation > self._config.max_rotation_drift
        ):
            raise ExecutionError(
                FailureCode.SLIPPED,
                stage,
                "target slipped in gripper "
                f"(jaw={jaw_drift:.6f}/{jaw_limit:.6f} m, "
                f"finger={finger_drift:.6f}/"
                f"{self._config.max_finger_axis_drift:.6f} m, "
                f"vertical={vertical_drift:.6f}/{vertical_limit:.6f} m, "
                f"rotation={np.rad2deg(rotation):.2f} deg)",
            )
