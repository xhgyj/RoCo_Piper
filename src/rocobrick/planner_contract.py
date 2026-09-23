"""Planner request/response types independent of ROS transport details."""

from __future__ import annotations

from dataclasses import dataclass

from rocobrick.compat import StrEnum
from rocobrick.execution import Trajectory
from rocobrick.models import ActionType, Pose, Revisions, RobotSnapshot


class PlanningStatus(StrEnum):
    SUCCESS = "SUCCESS"
    COLLISION_BLOCKED = "COLLISION_BLOCKED"
    TEMPORARILY_UNREACHABLE = "TEMPORARILY_UNREACHABLE"
    PLANNING_FAILED = "PLANNING_FAILED"
    INVALID_REQUEST = "INVALID_REQUEST"


class CollisionShape(StrEnum):
    BOX = "BOX"
    SPHERE = "SPHERE"
    CYLINDER = "CYLINDER"


@dataclass(frozen=True)
class CollisionObject:
    object_id: str
    pose: Pose
    shape: CollisionShape
    dimensions_m: tuple[float, ...]
    geometry_digest: str
    attached_to: str | None = None


@dataclass(frozen=True)
class ContactAllowance:
    first: str
    second: str
    max_force_n: float
    max_penetration_m: float
    max_duration_s: float


@dataclass(frozen=True)
class PlanningRequest:
    request_id: str
    action: ActionType
    robot_id: str
    revisions: Revisions
    start: RobotSnapshot
    target: Pose
    robots: tuple[RobotSnapshot, ...]
    objects: tuple[CollisionObject, ...]
    other_robot_remaining_motion: Trajectory | None
    allowed_contacts: tuple[ContactAllowance, ...]


@dataclass(frozen=True)
class PlanningResponse:
    request_id: str
    status: PlanningStatus
    revisions: Revisions
    trajectory: Trajectory | None
    required_resources: frozenset[str]
    start_digest: str
    goal_digest: str
    scene_digest: str
    collision_report: tuple[str, ...]
