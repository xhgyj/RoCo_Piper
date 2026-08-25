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
    INVALID_REQUEST = "invalid_request"


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
