"""Shared results and state passed between manipulation layers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class FailureCode(Enum):
    """Backend-independent manipulation failure categories."""

    UNREACHABLE = "unreachable"
    TIMEOUT = "timeout"
    COLLISION = "collision"
    GRIPPER_FAILED = "gripper_failed"
    SLIPPED = "slipped"
    EXCESS_FORCE = "excess_force"
    CONTACT_FAILED = "contact_failed"
    VERIFICATION_FAILED = "verification_failed"
    INVALID_REQUEST = "invalid_request"


class ActionStatus(Enum):
    """Terminal state of one public manipulation action."""

    SUCCESS = "success"
    INVALID_ACTION = "invalid_action"
    PLANNING_FAILED = "planning_failed"
    EXECUTION_FAILED = "execution_failed"


class ExecutionError(RuntimeError):
    """Raise a structured failure from a controller or primitive."""

    def __init__(self, code: FailureCode, stage: str, detail: str):
        """Initialize the failure code, operation stage, and detail."""
        super().__init__(f"{stage}: {detail}")
        self.code = code
        self.stage = stage
        self.detail = detail


@dataclass(frozen=True)
class OperationResult:
    """Successful completion metrics for one primitive."""

    steps: int
    position_error: float = 0.0
    rotation_error: float = 0.0


@dataclass(frozen=True)
class HeldObject:
    """Verified grasp state handed from Pick to later skills."""

    object_id: str
    robot_id: str
    object_t_tcp: FloatArray
    grasp_axis: int
    grasp_width: float

    def __post_init__(self) -> None:
        """Validate the immutable grasp contract."""
        transform = np.asarray(self.object_t_tcp, dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("object_t_tcp must be a finite 4x4 transform")
        if self.grasp_axis not in (0, 1):
            raise ValueError("grasp_axis must be 0 or 1")
        if self.grasp_width <= 0.0:
            raise ValueError("grasp_width must be positive")


@dataclass(frozen=True)
class ActionFailure:
    """Stable failure information returned to the upper planner."""

    code: FailureCode
    stage: str
    detail: str


@dataclass(frozen=True)
class ExecutionMetrics:
    """Unambiguous execution work reported across all skills."""

    waypoints: int = 0
    control_iterations: int = 0
    simulation_steps: int = 0

    def __post_init__(self) -> None:
        """Reject negative counters."""
        if min(self.waypoints, self.control_iterations, self.simulation_steps) < 0:
            raise ValueError("execution counters must be non-negative")

    def __add__(self, other: ExecutionMetrics) -> ExecutionMetrics:
        """Combine metrics from sequential execution phases.

        Returns:
            Component-wise sum of both metrics.
        """
        return ExecutionMetrics(
            self.waypoints + other.waypoints,
            self.control_iterations + other.control_iterations,
            self.simulation_steps + other.simulation_steps,
        )


@dataclass(frozen=True)
class ManipulationResult:
    """Uniform result returned by every manipulation action."""

    action_id: str
    status: ActionStatus
    robot_ids: tuple[str, ...]
    object_id: str
    held_by: str | None
    metrics: ExecutionMetrics
    failure: ActionFailure | None = None

    @property
    def steps(self) -> int:
        """Return physical simulation steps for concise reporting."""
        return self.metrics.simulation_steps


@dataclass
class HeldObjectState:
    """Persistent grasp state with an immutable acquisition reference."""

    held: HeldObject
    acquisition_object_t_tcp: FloatArray
    settled_object_t_tcp: FloatArray
    current_object_t_tcp: FloatArray
    cumulative_position_drift: float = 0.0
    cumulative_rotation_drift: float = 0.0

    @classmethod
    def from_pick(
        cls,
        held: HeldObject,
        acquisition_object_t_tcp: FloatArray | None = None,
    ) -> HeldObjectState:
        """Create state after Pick has completed its one allowed settling phase.

        Returns:
            Persistent state retaining both acquisition and settled transforms.
        """
        settled = np.asarray(held.object_t_tcp, dtype=np.float64).copy()
        acquisition = (
            settled.copy()
            if acquisition_object_t_tcp is None
            else np.asarray(acquisition_object_t_tcp, dtype=np.float64).copy()
        )
        return cls(held, acquisition, settled.copy(), settled.copy())

    def observe(self, object_t_tcp: FloatArray) -> None:
        """Update current and cumulative drift without rebasing the reference."""
        from scipy.spatial.transform import Rotation

        observed = np.asarray(object_t_tcp, dtype=np.float64)
        if observed.shape != (4, 4) or not np.isfinite(observed).all():
            raise ValueError("object_t_tcp must be a finite 4x4 transform")
        delta = np.linalg.inv(self.acquisition_object_t_tcp) @ observed
        self.cumulative_position_drift = float(np.linalg.norm(delta[:3, 3]))
        self.cumulative_rotation_drift = float(
            Rotation.from_matrix(delta[:3, :3]).magnitude()
        )
        self.current_object_t_tcp = observed.copy()
