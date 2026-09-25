"""Immutable public domain types shared by the platform core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from rocobrick.compat import StrEnum


class SkillType(StrEnum):
    PICK = "PICK"
    TRANSIT = "TRANSIT"
    WAIT = "WAIT"
    HANDOVER = "HANDOVER"
    PLACE_DOWN = "PLACE_DOWN"
    SUPPORT_BOTTOM = "SUPPORT_BOTTOM"
    PLACE_UP = "PLACE_UP"
    SUPPORT_TOP = "SUPPORT_TOP"


class TaskStatus(StrEnum):
    PENDING = "PENDING"
    ASSEMBLED = "ASSEMBLED"
    FAILED = "FAILED"
    BLOCKED_BY_FAILED_PREDECESSOR = "BLOCKED_BY_FAILED_PREDECESSOR"


class ActionType(StrEnum):
    PICK = "PICK"
    TRANSPORT = "TRANSPORT"
    ALIGN = "ALIGN"
    INSERT = "INSERT"
    VERIFY_CONNECTION = "VERIFY_CONNECTION"
    RELEASE = "RELEASE"
    RETREAT = "RETREAT"
    YIELD = "YIELD"
    CONTROLLED_STOP = "CONTROLLED_STOP"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class FailureReason(StrEnum):
    DEPENDENCY_UNMET = "DEPENDENCY_UNMET"
    RESOURCE_WAIT = "RESOURCE_WAIT"
    COLLISION_BLOCKED = "COLLISION_BLOCKED"
    TEMPORARILY_UNREACHABLE = "TEMPORARILY_UNREACHABLE"
    PLANNING_FAILED = "PLANNING_FAILED"
    START_STATE_MISMATCH = "START_STATE_MISMATCH"
    STALE_WORLD = "STALE_WORLD"
    TRACKING_FAILED = "TRACKING_FAILED"
    HOLDING_FAILED = "HOLDING_FAILED"
    CONNECTION_FAILED = "CONNECTION_FAILED"
    RELEASE_FAILED = "RELEASE_FAILED"
    RETREAT_FAILED = "RETREAT_FAILED"
    SAFETY_STOPPED = "SAFETY_STOPPED"
    CAPABILITY_UNAVAILABLE = "CAPABILITY_UNAVAILABLE"


@dataclass(frozen=True)
class Pose:
    frame_id: str
    position: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    part_instance_id: str
    brick_id: int
    target: Pose
    target_interface: str
    skill: SkillType
    primary_robot: str
    supporting_robot: str | None
    predecessors: frozenset[str]
    level: int
    schedule_slot: int


@dataclass(frozen=True)
class AssemblyPlan:
    plan_id: str
    frame_id: str
    brick_library_version: str
    robot_asset_version: str
    tasks: Mapping[str, TaskDefinition]


@dataclass(frozen=True)
class PartInstance:
    part_instance_id: str
    brick_id: int
    pose: Pose


@dataclass(frozen=True)
class SceneManifest:
    scene_id: str
    frame_id: str
    parts: Mapping[str, PartInstance]


@dataclass(frozen=True)
class Revisions:
    world: int
    robot_state: int
    motion: int


@dataclass(frozen=True)
class RobotSnapshot:
    robot_id: str
    joints: tuple[float, ...]
    feedback_monotonic_s: float
    holding_part: str | None = None
    active_action_id: str | None = None


@dataclass(frozen=True)
class ExecutionSnapshot:
    revisions: Revisions
    robots: Mapping[str, RobotSnapshot]
    occupied_resources: Mapping[str, str] = field(default_factory=dict)
    safety_stop_requested: bool = False
    emergency_stop_requested: bool = False


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    task_id: str
    action: ActionType
    robot_id: str
    revisions: Revisions
    required_resources: frozenset[str]
    feasible: bool
    holding_continuation: bool = False
    resolves_blockage: bool = False
    is_prefetch: bool = False


@dataclass(frozen=True)
class DispatchDecision:
    selected: tuple[Candidate, ...]
    waiting: Mapping[str, FailureReason]
    rationale: tuple[str, ...]
