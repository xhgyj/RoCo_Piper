"""Measured-state trajectory playback owned by the physical simulator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
from typing import Protocol

from rocobrick.execution import Trajectory
from rocobrick.models import FailureReason


class PlaybackStatus(StrEnum):
    COMPLETED = "COMPLETED"
    INVALID_TRAJECTORY = "INVALID_TRAJECTORY"
    TRACKING_FAILED = "TRACKING_FAILED"
    FEEDBACK_EXPIRED = "FEEDBACK_EXPIRED"
    HOLDING_FAILED = "HOLDING_FAILED"
    SAFETY_STOPPED = "SAFETY_STOPPED"
    NO_PROGRESS = "NO_PROGRESS"


@dataclass(frozen=True)
class MeasuredState:
    simulation_time_s: float
    joints: tuple[float, ...]
    feedback_age_s: float
    holding_part: str | None
    safety_valid: bool


class PhysicalTrajectoryPort(Protocol):
    """Minimal simulator boundary; MoveIt never implements this protocol."""

    def measured_state(self, robot_id: str) -> MeasuredState: ...
    def command_joint_positions(
        self, robot_id: str, joint_names: tuple[str, ...], positions: tuple[float, ...]
    ) -> None: ...
    async def step(self) -> None: ...
    def controlled_stop(self, robot_id: str) -> None: ...


@dataclass(frozen=True)
class PlaybackResult:
    status: PlaybackStatus
    reason: FailureReason | None
    physics_steps: int
    maximum_tracking_error_rad: float
    final_state: MeasuredState


def validate_trajectory(trajectory: Trajectory) -> None:
    width = len(trajectory.joint_names)
    if width == 0 or len(set(trajectory.joint_names)) != width:
        raise ValueError("trajectory joint names must be non-empty and unique")
    if not trajectory.points or len(trajectory.points) != len(trajectory.times_s):
        raise ValueError("trajectory points and timestamps must have equal non-zero length")
    previous = -math.inf
    for point, timestamp in zip(trajectory.points, trajectory.times_s):
        if len(point) != width or not all(math.isfinite(value) for value in point):
            raise ValueError("trajectory point has invalid width or non-finite position")
        if not math.isfinite(timestamp) or timestamp < 0.0 or timestamp <= previous:
            raise ValueError("trajectory timestamps must be finite and strictly increasing")
        previous = timestamp


def interpolate(trajectory: Trajectory, elapsed_s: float) -> tuple[float, ...]:
    if elapsed_s <= trajectory.times_s[0]:
        return trajectory.points[0]
    if elapsed_s >= trajectory.times_s[-1]:
        return trajectory.points[-1]
    upper = next(index for index, value in enumerate(trajectory.times_s) if value >= elapsed_s)
    lower = upper - 1
    t0, t1 = trajectory.times_s[lower], trajectory.times_s[upper]
    alpha = (elapsed_s - t0) / (t1 - t0)
    return tuple(
        start + alpha * (end - start)
        for start, end in zip(trajectory.points[lower], trajectory.points[upper])
    )


async def play_trajectory(
    port: PhysicalTrajectoryPort,
    robot_id: str,
    trajectory: Trajectory,
    *,
    expected_holding_part: str | None,
    tracking_tolerance_rad: float = 0.08,
    final_tolerance_rad: float = 0.02,
    feedback_timeout_s: float = 0.25,
    no_progress_timeout_s: float = 30.0,
    progress_epsilon_rad: float = 1e-4,
) -> PlaybackResult:
    """Play an admitted trajectory using simulation time and measured feedback."""
    try:
        validate_trajectory(trajectory)
    except ValueError:
        state = port.measured_state(robot_id)
        return PlaybackResult(PlaybackStatus.INVALID_TRAJECTORY, FailureReason.PLANNING_FAILED, 0, 0.0, state)

    initial = port.measured_state(robot_id)
    started_at = initial.simulation_time_s
    last_progress_at = started_at
    last_joints = initial.joints
    maximum_error = 0.0
    steps = 0

    def failed(status: PlaybackStatus, reason: FailureReason, state: MeasuredState) -> PlaybackResult:
        port.controlled_stop(robot_id)
        return PlaybackResult(status, reason, steps, maximum_error, state)

    while True:
        state = port.measured_state(robot_id)
        if not state.safety_valid:
            return failed(PlaybackStatus.SAFETY_STOPPED, FailureReason.SAFETY_STOPPED, state)
        if state.feedback_age_s > feedback_timeout_s:
            return failed(PlaybackStatus.FEEDBACK_EXPIRED, FailureReason.TRACKING_FAILED, state)
        if expected_holding_part is not None and state.holding_part != expected_holding_part:
            return failed(PlaybackStatus.HOLDING_FAILED, FailureReason.HOLDING_FAILED, state)
        elapsed = max(0.0, state.simulation_time_s - started_at)
        command = interpolate(trajectory, elapsed)
        if len(state.joints) != len(command):
            return failed(PlaybackStatus.TRACKING_FAILED, FailureReason.TRACKING_FAILED, state)
        error = max(abs(actual - desired) for actual, desired in zip(state.joints, command))
        maximum_error = max(maximum_error, error)
        if error > tracking_tolerance_rad:
            return failed(PlaybackStatus.TRACKING_FAILED, FailureReason.TRACKING_FAILED, state)
        movement = max(abs(actual - before) for actual, before in zip(state.joints, last_joints))
        if movement >= progress_epsilon_rad:
            last_progress_at = state.simulation_time_s
            last_joints = state.joints
        if elapsed >= trajectory.times_s[-1]:
            final_error = max(
                abs(actual - desired)
                for actual, desired in zip(state.joints, trajectory.points[-1])
            )
            if final_error <= final_tolerance_rad:
                return PlaybackResult(PlaybackStatus.COMPLETED, None, steps, maximum_error, state)
        if state.simulation_time_s - last_progress_at >= no_progress_timeout_s:
            return failed(PlaybackStatus.NO_PROGRESS, FailureReason.TRACKING_FAILED, state)
        port.command_joint_positions(robot_id, trajectory.joint_names, command)
        await port.step()
        steps += 1
