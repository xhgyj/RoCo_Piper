"""Execution-owned state, candidate admission, and watchdog semantics."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Mapping, Protocol

from rocobrick.compat import StrEnum
from rocobrick.models import (
    ActionType,
    Candidate,
    FailureReason,
    Revisions,
    RobotSnapshot,
)


class AdmissionStatus(StrEnum):
    ACCEPTED = "ACCEPTED"
    STALE_WORLD = "STALE_WORLD"
    START_STATE_MISMATCH = "START_STATE_MISMATCH"
    HOLDING_MISMATCH = "HOLDING_MISMATCH"
    RESOURCE_CONFLICT = "RESOURCE_CONFLICT"


@dataclass(frozen=True)
class Trajectory:
    joint_names: tuple[str, ...]
    points: tuple[tuple[float, ...], ...]
    times_s: tuple[float, ...]


@dataclass(frozen=True)
class PlannedMotion:
    candidate: Candidate
    start_joints: tuple[float, ...]
    expected_holding_part: str | None
    trajectory: Trajectory
    start_digest: str
    goal_digest: str
    scene_digest: str


@dataclass(frozen=True)
class AdmissionResult:
    status: AdmissionStatus
    reason: FailureReason | None = None


class CommandSink(Protocol):
    def send_trajectory(self, robot_id: str, trajectory: Trajectory) -> None: ...
    def controlled_stop(self, robot_id: str) -> None: ...
    def emergency_stop(self) -> None: ...


class ExecutionKernel:
    """The only mutable owner of measured robot and resource state."""

    def __init__(self, robots: Mapping[str, RobotSnapshot], command_sink: CommandSink, start_tolerance_rad: float = 0.02) -> None:
        self._robots = dict(robots)
        self._revisions = Revisions(0, 0, 0)
        self._resources: dict[str, str] = {}
        self._sink = command_sink
        self._start_tolerance = start_tolerance_rad
        self._last_progress_s = time.monotonic()

    @property
    def revisions(self) -> Revisions:
        return self._revisions

    def update_measurement(self, snapshot: RobotSnapshot, progressed: bool = False) -> None:
        self._robots[snapshot.robot_id] = snapshot
        self._revisions = replace(self._revisions, robot_state=self._revisions.robot_state + 1)
        if progressed:
            self._last_progress_s = time.monotonic()

    def world_changed(self) -> None:
        self._revisions = replace(self._revisions, world=self._revisions.world + 1)
        self._last_progress_s = time.monotonic()

    def holding_changed(self, robot_id: str, part_instance_id: str | None) -> None:
        self._robots[robot_id] = replace(self._robots[robot_id], holding_part=part_instance_id)
        self._revisions = Revisions(self._revisions.world + 1, self._revisions.robot_state + 1, self._revisions.motion + 1)
        self._last_progress_s = time.monotonic()

    def admit(self, motion: PlannedMotion) -> AdmissionResult:
        candidate = motion.candidate
        if candidate.revisions != self._revisions:
            return AdmissionResult(AdmissionStatus.STALE_WORLD, FailureReason.STALE_WORLD)
        robot = self._robots[candidate.robot_id]
        if robot.holding_part != motion.expected_holding_part:
            return AdmissionResult(AdmissionStatus.HOLDING_MISMATCH, FailureReason.HOLDING_FAILED)
        if len(robot.joints) != len(motion.start_joints) or any(
            not math.isclose(actual, planned, abs_tol=self._start_tolerance)
            for actual, planned in zip(robot.joints, motion.start_joints)
        ):
            return AdmissionResult(AdmissionStatus.START_STATE_MISMATCH, FailureReason.START_STATE_MISMATCH)
        if any(resource in self._resources for resource in candidate.required_resources):
            return AdmissionResult(AdmissionStatus.RESOURCE_CONFLICT, FailureReason.RESOURCE_WAIT)
        for resource in candidate.required_resources:
            self._resources[resource] = candidate.candidate_id
        self._robots[candidate.robot_id] = replace(robot, active_action_id=candidate.candidate_id)
        self._revisions = replace(self._revisions, motion=self._revisions.motion + 1)
        self._sink.send_trajectory(candidate.robot_id, motion.trajectory)
        return AdmissionResult(AdmissionStatus.ACCEPTED)

    def finish(self, candidate: Candidate, confirmed_progress: bool = True) -> None:
        for resource in tuple(self._resources):
            if self._resources[resource] == candidate.candidate_id:
                del self._resources[resource]
        robot = self._robots[candidate.robot_id]
        self._robots[candidate.robot_id] = replace(robot, active_action_id=None)
        self._revisions = replace(self._revisions, motion=self._revisions.motion + 1)
        if confirmed_progress:
            self._last_progress_s = time.monotonic()

    def watchdog_expired(self, now_s: float | None = None, timeout_s: float = 30.0) -> bool:
        now = time.monotonic() if now_s is None else now_s
        return now - self._last_progress_s >= timeout_s

    def stop_robot(self, robot_id: str) -> None:
        self._sink.controlled_stop(robot_id)

    def emergency_stop(self) -> None:
        self._sink.emergency_stop()
