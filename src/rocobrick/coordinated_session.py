"""Strict coordinated MoveIt session envelope contracts.

A coordinated request is separate from the existing single-robot session
request. It carries one combined MoveIt operation identity and exact
participants; it must never be serialized as two unrelated robot operations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from rocobrick.coordinated_moveit import CoordinatedMoveItError

COORDINATED_REQUEST_SCHEMA = "rocobrick-coordinated-moveit-request/v1"
COORDINATED_RESULT_SCHEMA = "rocobrick-coordinated-moveit-result/v1"


def _identities(values: Sequence[str], label: str) -> tuple[str, ...]:
    """Validate one ordered, unique robot identity sequence."""
    result = tuple(values)
    if (
        not result
        or any(not isinstance(item, str) or not item for item in result)
        or len(result) != len(set(result))
    ):
        raise CoordinatedMoveItError(f"{label} must be unique and non-empty")
    return result


@dataclass(frozen=True)
class FreeSpaceMotionRequest:
    """Skill-neutral free-space motion request.

    ``session_robots`` identifies the complete persistent ROS session. Only
    ``participants`` receive controller goals; the remaining session robots
    still contribute their measured state to the combined planning request.
    """

    motion_id: str
    session_robots: tuple[str, ...]
    participants: tuple[str, ...]
    scene_snapshot: Mapping[str, object]
    scene_version: int
    targets: Mapping[str, Mapping[str, object]]
    start_state: Mapping[str, float]
    timeout_s: float
    physical_facts: Mapping[str, Mapping[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject incomplete identity, participant coverage or evidence."""
        if not isinstance(self.motion_id, str) or not self.motion_id:
            raise CoordinatedMoveItError("free-space motion_id is required")
        session = _identities(self.session_robots, "session_robots")
        participants = _identities(self.participants, "participants")
        if not set(participants) <= set(session):
            raise CoordinatedMoveItError(
                "free-space participants must belong to session_robots"
            )
        if not isinstance(self.scene_snapshot, Mapping):
            raise CoordinatedMoveItError("free-space scene_snapshot must be a mapping")
        if not isinstance(self.targets, Mapping) or set(self.targets) != set(
            participants
        ):
            raise CoordinatedMoveItError(
                "free-space targets must cover participants exactly"
            )
        if any(not isinstance(value, Mapping) for value in self.targets.values()):
            raise CoordinatedMoveItError("free-space targets must be mappings")
        if not isinstance(self.start_state, Mapping):
            raise CoordinatedMoveItError("free-space start_state must be a mapping")
        if not isinstance(self.physical_facts, Mapping) or not all(
            isinstance(value, Mapping) for value in self.physical_facts.values()
        ):
            raise CoordinatedMoveItError("free-space physical_facts must be mappings")
        if not set(self.physical_facts) <= set(participants):
            raise CoordinatedMoveItError(
                "free-space physical_facts must cover participants only"
            )
        if type(self.scene_version) is not int or self.scene_version < 0:
            raise CoordinatedMoveItError("free-space scene_version is invalid")
        if (
            isinstance(self.timeout_s, bool)
            or not isinstance(self.timeout_s, (int, float))
            or not math.isfinite(float(self.timeout_s))
            or float(self.timeout_s) <= 0
        ):
            raise CoordinatedMoveItError("free-space timeout_s is invalid")
        from rocobrick.unified_dual_arm_runtime import _json_safe

        if not all(
            _json_safe(value)
            for value in (
                self.scene_snapshot,
                self.targets,
                self.start_state,
                self.physical_facts,
            )
        ):
            raise CoordinatedMoveItError("free-space request evidence is not JSON-safe")
        object.__setattr__(self, "session_robots", session)
        object.__setattr__(self, "participants", participants)

    def to_coordinated_request(self, *, plan_only: bool) -> "CoordinatedSessionRequest":
        """Convert the skill-neutral request to the ROS planning envelope."""
        scene_digest = self.scene_snapshot.get("scene_digest")
        if not isinstance(scene_digest, str) or not scene_digest:
            raise CoordinatedMoveItError("free-space scene snapshot lacks scene_digest")
        return CoordinatedSessionRequest(
            coordination_id=self.motion_id,
            robots=self.session_robots,
            participants=self.participants,
            scene_digest=scene_digest,
            planning_group="dual_arm",
            plan_only=plan_only,
            timeout_s=float(self.timeout_s),
            targets=self.targets,
            scene_snapshot=self.scene_snapshot,
            scene_version=self.scene_version,
            start_state=self.start_state,
            physical_facts=dict(self.physical_facts),
        )


@dataclass(frozen=True)
class FreeSpaceMotionEvidence:
    """Skill-neutral lifecycle and measured-arrival evidence."""

    motion_id: str
    participants: tuple[str, ...]
    planned: bool
    prepared: bool
    started: bool
    arrival_confirmed: bool
    terminal: bool
    safe_to_release: bool
    participant_evidence: Mapping[str, Mapping[str, object]]
    evidence: Mapping[str, object]
    reason: str | None = None

    def __post_init__(self) -> None:
        """Require strict lifecycle booleans and participant coverage."""
        if not isinstance(self.motion_id, str) or not self.motion_id:
            raise CoordinatedMoveItError("free-space evidence motion_id is required")
        participants = _identities(self.participants, "participants")
        if set(self.participant_evidence) != set(participants):
            raise CoordinatedMoveItError(
                "free-space participant evidence must cover participants exactly"
            )
        for name in (
            "planned",
            "prepared",
            "started",
            "arrival_confirmed",
            "terminal",
            "safe_to_release",
        ):
            if type(getattr(self, name)) is not bool:
                raise CoordinatedMoveItError(f"free-space {name} must be boolean")
        from rocobrick.unified_dual_arm_runtime import _json_safe

        if not _json_safe(self.participant_evidence) or not _json_safe(self.evidence):
            raise CoordinatedMoveItError("free-space evidence must be JSON-safe")
        if self.reason is not None and not isinstance(self.reason, str):
            raise CoordinatedMoveItError("free-space reason must be a string or null")
        object.__setattr__(self, "participants", participants)

    @property
    def passed(self) -> bool:
        """Return true only for planned, terminal-safe measured arrival."""
        return all(
            (
                self.planned,
                self.prepared,
                self.started,
                self.arrival_confirmed,
                self.terminal,
                self.safe_to_release,
            )
        )


@dataclass(frozen=True)
class CoordinatedSessionRequest:
    """One planning/execution request for the combined MoveIt group."""

    coordination_id: str
    robots: tuple[str, ...]
    scene_digest: str
    planning_group: str
    plan_only: bool
    timeout_s: float
    targets: Mapping[str, Mapping[str, object]]
    scene_snapshot: Mapping[str, object] = field(default_factory=dict)
    scene_version: int = 0
    physical_facts: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    operation: Mapping[str, object] = field(default_factory=dict)
    target_contexts: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    start_state: Mapping[str, float] = field(default_factory=dict)
    participants: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate coordinated request identity and participant coverage."""
        if not self.coordination_id or not self.scene_digest or not self.planning_group:
            raise CoordinatedMoveItError("coordinated request identity is incomplete")
        if not self.robots or len(self.robots) != len(set(self.robots)):
            raise CoordinatedMoveItError("coordinated robots must be unique")
        participants = self.participants or self.robots
        if (
            not participants
            or len(participants) != len(set(participants))
            or any(not isinstance(robot, str) or not robot for robot in participants)
            or not set(participants) <= set(self.robots)
        ):
            raise CoordinatedMoveItError(
                "coordinated participants must be unique and belong to robots"
            )
        object.__setattr__(self, "participants", tuple(participants))
        if not self.operation and set(self.targets) != set(self.participants):
            raise CoordinatedMoveItError(
                "coordinated targets must cover every participant"
            )
        if (
            not isinstance(self.plan_only, bool)
            or not isinstance(self.timeout_s, (int, float))
            or self.timeout_s <= 0
        ):
            raise CoordinatedMoveItError("coordinated request mode/timeout is invalid")
        if isinstance(self.timeout_s, bool) or not math.isfinite(self.timeout_s):
            raise CoordinatedMoveItError("coordinated timeout must be finite")
        for value in (
            self.scene_snapshot,
            self.targets,
            self.physical_facts,
            self.operation,
            self.target_contexts,
            self.start_state,
        ):
            if not isinstance(value, Mapping):
                raise CoordinatedMoveItError("combined evidence must be mappings")
        if self.physical_facts and set(self.physical_facts) != set(self.participants):
            raise CoordinatedMoveItError(
                "physical facts must cover every participant"
            )
        from rocobrick.unified_dual_arm_runtime import _json_safe

        if not all(
            _json_safe(value)
            for value in (
                self.scene_snapshot,
                self.targets,
                self.physical_facts,
                self.operation,
                self.target_contexts,
            )
        ):
            raise CoordinatedMoveItError("combined request evidence must be JSON-safe")
        if type(self.scene_version) is not int or self.scene_version < 0:
            raise CoordinatedMoveItError("scene version must be a non-negative integer")

    def serialize(self) -> dict[str, object]:
        """Return the JSON-safe wire document."""
        return {
            "schema": COORDINATED_REQUEST_SCHEMA,
            "coordination_id": self.coordination_id,
            "robots": list(self.robots),
            "participants": list(self.participants),
            "scene_digest": self.scene_digest,
            "planning_group": self.planning_group,
            "plan_only": self.plan_only,
            "timeout_s": float(self.timeout_s),
            "targets": {robot: dict(value) for robot, value in self.targets.items()},
            "scene_snapshot": dict(self.scene_snapshot),
            "scene_version": self.scene_version,
            "operation": dict(self.operation),
            "target_contexts": {
                robot: dict(context) for robot, context in self.target_contexts.items()
            },
            "start_state": dict(self.start_state),
            "physical_facts": {
                robot: dict(facts) for robot, facts in self.physical_facts.items()
            },
        }


@dataclass(frozen=True)
class CoordinatedSessionResult:
    """Strict result envelope for one combined MoveIt session request."""

    coordination_id: str
    robots: tuple[str, ...]
    readiness: Mapping[str, object]
    trajectory: Mapping[str, object]
    lifecycle: Mapping[str, object]
    passed: bool
    error: str | None = None
    participants: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject non-boolean status and non-JSON result evidence."""
        from rocobrick.unified_dual_arm_runtime import _json_safe

        if (
            type(self.passed) is not bool
            or not isinstance(self.coordination_id, str)
            or not self.coordination_id
        ):
            raise CoordinatedMoveItError("combined result status/identity is invalid")
        if (
            not self.robots
            or any(not isinstance(robot, str) or not robot for robot in self.robots)
            or len(self.robots) != len(set(self.robots))
        ):
            raise CoordinatedMoveItError("combined result session robots are invalid")
        participants = self.participants or self.robots
        if (
            not participants
            or any(
                not isinstance(robot, str) or not robot for robot in participants
            )
            or len(participants) != len(set(participants))
            or not set(participants) <= set(self.robots)
        ):
            raise CoordinatedMoveItError("combined result participants are invalid")
        object.__setattr__(self, "participants", tuple(participants))
        if not all(
            isinstance(value, Mapping) and _json_safe(value)
            for value in (self.readiness, self.trajectory, self.lifecycle)
        ):
            raise CoordinatedMoveItError(
                "combined result evidence must be JSON-safe mappings"
            )
        participants = self.participants or self.robots
        if not set(participants) <= set(self.robots):
            raise CoordinatedMoveItError(
                "combined result participants must belong to session robots"
            )
        object.__setattr__(self, "participants", tuple(participants))

    def serialize(self) -> dict[str, object]:
        """Return the JSON-safe result document."""
        return {
            "schema": COORDINATED_RESULT_SCHEMA,
            "coordination_id": self.coordination_id,
            "robots": list(self.robots),
            "participants": list(self.participants),
            "readiness": dict(self.readiness),
            "trajectory": dict(self.trajectory),
            "lifecycle": dict(self.lifecycle),
            "passed": bool(self.passed),
            "error": self.error,
        }

    @classmethod
    def parse(cls, document: object) -> "CoordinatedSessionResult":
        """Parse one exact result envelope.

        Returns:
            The immutable coordinated result.
        """
        expected = frozenset(
            {
                "schema",
                "coordination_id",
                "robots",
                "participants",
                "readiness",
                "trajectory",
                "lifecycle",
                "passed",
                "error",
            }
        )
        if not isinstance(document, Mapping) or frozenset(document) != expected:
            raise CoordinatedMoveItError("coordinated result keys are invalid")
        if document["schema"] != COORDINATED_RESULT_SCHEMA:
            raise CoordinatedMoveItError("coordinated result schema is invalid")
        robots = document["robots"]
        if not isinstance(robots, Sequence) or isinstance(robots, (str, bytes)):
            raise CoordinatedMoveItError("coordinated result robots must be an array")
        if not robots or len(set(robots)) != len(robots):
            raise CoordinatedMoveItError("coordinated result robots must be unique")
        mappings = ("readiness", "trajectory", "lifecycle")
        if any(not isinstance(document[name], Mapping) for name in mappings):
            raise CoordinatedMoveItError("coordinated result evidence must be objects")
        if not isinstance(document["passed"], bool):
            raise CoordinatedMoveItError("coordinated result passed must be boolean")
        error = document["error"]
        if error is not None and not isinstance(error, str):
            raise CoordinatedMoveItError(
                "coordinated result error must be string or null"
            )
        return cls(
            coordination_id=str(document["coordination_id"]),
            robots=tuple(str(robot) for robot in robots),
            participants=tuple(
                str(robot) for robot in document["participants"]
            ),
            readiness=dict(document["readiness"]),
            trajectory=dict(document["trajectory"]),
            lifecycle=dict(document["lifecycle"]),
            passed=bool(document["passed"]),
            error=error,
        )


def parse_coordinated_request(document: object) -> CoordinatedSessionRequest:
    """Parse one exact coordinated request document.

    Returns:
        The immutable coordinated request.
    """
    if not isinstance(document, Mapping) or set(document) != {
        "schema",
        "coordination_id",
        "robots",
        "participants",
        "scene_digest",
        "planning_group",
        "plan_only",
        "timeout_s",
        "targets",
        "scene_snapshot",
        "scene_version",
        "physical_facts",
        "operation",
        "target_contexts",
        "start_state",
    }:
        raise CoordinatedMoveItError("coordinated request keys are invalid")
    if document["schema"] != COORDINATED_REQUEST_SCHEMA:
        raise CoordinatedMoveItError("coordinated request schema is invalid")
    robots = document["robots"]
    targets = document["targets"]
    if not isinstance(robots, Sequence) or isinstance(robots, (str, bytes)):
        raise CoordinatedMoveItError("coordinated robots must be an array")
    participants = document["participants"]
    if not isinstance(participants, Sequence) or isinstance(
        participants, (str, bytes)
    ):
        raise CoordinatedMoveItError("coordinated participants must be an array")
    if not isinstance(targets, Mapping) or not all(
        isinstance(value, Mapping) for value in targets.values()
    ):
        raise CoordinatedMoveItError("coordinated targets must be an object of objects")
    return CoordinatedSessionRequest(
        coordination_id=str(document["coordination_id"]),
        robots=tuple(str(robot) for robot in robots),
        participants=tuple(str(robot) for robot in participants),
        scene_digest=str(document["scene_digest"]),
        planning_group=str(document["planning_group"]),
        plan_only=document["plan_only"],
        timeout_s=document["timeout_s"],
        targets={str(robot): dict(value) for robot, value in targets.items()},
        scene_snapshot=document["scene_snapshot"],
        scene_version=document["scene_version"],
        physical_facts=document["physical_facts"],
        operation=document["operation"],
        target_contexts=document["target_contexts"],
        start_state=document["start_state"],
    )
