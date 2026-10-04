"""Strict coordinated MoveIt session envelope contracts.

A coordinated request is separate from the existing single-robot session
request. It carries one combined MoveIt operation identity and exact
participants; it must never be serialized as two unrelated robot operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from rocobrick.coordinated_moveit import CoordinatedMoveItError

COORDINATED_REQUEST_SCHEMA = "rocobrick-coordinated-moveit-request/v1"
COORDINATED_RESULT_SCHEMA = "rocobrick-coordinated-moveit-result/v1"


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

    def __post_init__(self) -> None:
        """Validate coordinated request identity and participant coverage."""
        if not self.coordination_id or not self.scene_digest or not self.planning_group:
            raise CoordinatedMoveItError("coordinated request identity is incomplete")
        if not self.robots or len(self.robots) != len(set(self.robots)):
            raise CoordinatedMoveItError("coordinated robots must be unique")
        if set(self.targets) != set(self.robots):
            raise CoordinatedMoveItError("coordinated targets must cover every robot")
        if (
            not isinstance(self.plan_only, bool)
            or not isinstance(self.timeout_s, (int, float))
            or self.timeout_s <= 0
        ):
            raise CoordinatedMoveItError("coordinated request mode/timeout is invalid")
        if isinstance(self.timeout_s, bool):
            raise CoordinatedMoveItError("coordinated timeout must be numeric")

    def serialize(self) -> dict[str, object]:
        """Return the JSON-safe wire document."""
        return {
            "schema": COORDINATED_REQUEST_SCHEMA,
            "coordination_id": self.coordination_id,
            "robots": list(self.robots),
            "scene_digest": self.scene_digest,
            "planning_group": self.planning_group,
            "plan_only": self.plan_only,
            "timeout_s": float(self.timeout_s),
            "targets": {robot: dict(value) for robot, value in self.targets.items()},
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

    def serialize(self) -> dict[str, object]:
        """Return the JSON-safe result document."""
        return {
            "schema": COORDINATED_RESULT_SCHEMA,
            "coordination_id": self.coordination_id,
            "robots": list(self.robots),
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
        "schema", "coordination_id", "robots", "scene_digest", "planning_group",
        "plan_only", "timeout_s", "targets",
    }:
        raise CoordinatedMoveItError("coordinated request keys are invalid")
    if document["schema"] != COORDINATED_REQUEST_SCHEMA:
        raise CoordinatedMoveItError("coordinated request schema is invalid")
    robots = document["robots"]
    targets = document["targets"]
    if not isinstance(robots, Sequence) or isinstance(robots, (str, bytes)):
        raise CoordinatedMoveItError("coordinated robots must be an array")
    if (
        not isinstance(targets, Mapping)
        or not all(isinstance(value, Mapping) for value in targets.values())
    ):
        raise CoordinatedMoveItError("coordinated targets must be an object of objects")
    return CoordinatedSessionRequest(
        coordination_id=str(document["coordination_id"]),
        robots=tuple(str(robot) for robot in robots),
        scene_digest=str(document["scene_digest"]),
        planning_group=str(document["planning_group"]),
        plan_only=document["plan_only"],
        timeout_s=document["timeout_s"],
        targets={str(robot): dict(value) for robot, value in targets.items()},
    )
