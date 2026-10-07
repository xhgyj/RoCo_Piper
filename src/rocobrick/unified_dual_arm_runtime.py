"""Bound successor pre-phase contracts for planning and physical lease adoption."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from rocobrick.robot_operation_plan import RobotOperation

JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]


def _identities(values: tuple[str, ...], label: str, *, empty: bool = False) -> None:
    if (
        (not values and not empty)
        or any(not isinstance(value, str) or not value for value in values)
        or len(values) != len(set(values))
    ):
        raise ValueError(f"{label} must contain unique non-empty identities")


def _json_safe(value: object) -> bool:
    if value is None or isinstance(value, (bool, int, str)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str) and _json_safe(item) for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return all(_json_safe(item) for item in value)
    return False


def _digest(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a SHA-256 digest")


@dataclass(frozen=True)
class SharedSceneSnapshot:
    """One immutable-by-copy observation with provenance and a freshness bound."""

    document: Mapping[str, JSONValue]
    observed_sim_time_s: float
    max_age_s: float

    def __post_init__(self) -> None:
        """Reject incomplete provenance, stale snapshots and unknown robot states."""
        if not isinstance(self.document, Mapping) or not _json_safe(self.document):
            raise ValueError("scene snapshot must be a JSON-safe mapping")
        document = json.loads(json.dumps(self.document, allow_nan=False))
        if (
            not isinstance(document.get("session_id"), str)
            or not document["session_id"]
        ):
            raise ValueError("scene snapshot requires session identity")
        _digest(document.get("manifest_sha256"), "manifest_sha256")
        version = document.get("scene_version")
        if type(version) is not int or version < 0:
            raise ValueError("scene_version must be a non-negative integer")
        sim_time = document.get("sim_time_s")
        for value in (sim_time, self.observed_sim_time_s, self.max_age_s):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("scene snapshot times must be finite numbers")
        if (
            sim_time < 0
            or self.max_age_s <= 0
            or not (0 <= self.observed_sim_time_s - sim_time <= self.max_age_s)
        ):
            raise ValueError("scene snapshot is stale or from the future")
        robots = document.get("robots")
        if not isinstance(robots, dict) or not robots:
            raise ValueError("scene snapshot requires shared robot states")
        for robot_id, state in robots.items():
            if (
                not robot_id
                or not isinstance(state, dict)
                or not all(
                    state.get(flag) is True
                    for flag in (
                        "authorized",
                        "terminal_safe",
                        "holding_known",
                        "supporting_known",
                    )
                )
            ):
                raise ValueError(
                    "pre-phase robot state is unknown, unsafe or unauthorized"
                )
            facts = state.get("facts")
            if not isinstance(facts, dict) or any(
                not key or not isinstance(proof, dict) or not proof
                for key, proof in facts.items()
            ):
                raise ValueError("physical facts require explicit evidence")
        object.__setattr__(self, "document", document)

    @property
    def scene_digest(self) -> str:
        """Return the digest of the complete bound shared observation.

        Returns:
            The identified, bound result; physical proof remains separately checked.
        """
        canonical = json.dumps(self.document, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def scene_version(self) -> int:
        """Return the admitted planning-scene version.

        Returns:
            The identified, bound result; physical proof remains separately checked.
        """
        return self.document["scene_version"]  # type: ignore[return-value]


@dataclass(frozen=True)
class TerminalPrePhasePolicy:
    """Explicit authorized terminal departure targets, never an inferred retreat."""

    policy_id: str
    source_sha256: str
    participants: tuple[str, ...]
    targets: Mapping[str, Mapping[str, JSONValue]]
    release_supporting: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate policy provenance and exact participant target coverage."""
        if not isinstance(self.policy_id, str) or not self.policy_id:
            raise ValueError("terminal policy requires identity")
        _digest(self.source_sha256, "terminal policy source_sha256")
        _identities(self.participants, "terminal policy participants")
        if (
            set(self.targets) != set(self.participants)
            or not _json_safe(self.targets)
            or any(
                not isinstance(target, Mapping) or not target
                for target in self.targets.values()
            )
        ):
            raise ValueError("terminal policy requires explicit participant targets")
        _identities(self.release_supporting, "release_supporting", empty=True)
        if not set(self.release_supporting) <= set(self.participants):
            raise ValueError("terminal support release is outside participants")


@dataclass(frozen=True)
class SuccessorPrePhaseRequest:
    """One successor-owned transition adopting one or more predecessor leases."""

    transition_id: str
    predecessor_operations: tuple[RobotOperation, ...]
    predecessor_states: Mapping[str, Mapping[str, JSONValue]]
    next_operation: RobotOperation | None
    participants: tuple[str, ...]
    resources: tuple[str, ...]
    lease_owners: Mapping[str, str]
    scene_snapshot: SharedSceneSnapshot
    terminal_policy: TerminalPrePhasePolicy | None = None

    def __post_init__(self) -> None:
        """Require exact owners, endpoints, participants and snapshot provenance."""
        if not isinstance(self.transition_id, str) or not self.transition_id:
            raise ValueError("transition_id must be non-empty")
        ids = self.predecessor_operation_ids
        _identities(ids, "predecessor operations")
        if set(self.predecessor_states) != set(ids) or not _json_safe(
            self.predecessor_states
        ):
            raise ValueError("predecessor endpoint evidence does not match identities")
        if any(
            state.get("terminal_safe") is not True
            for state in self.predecessor_states.values()
        ):
            raise ValueError("predecessor endpoint lacks terminal-safe evidence")
        _identities(self.participants, "participants")
        _identities(self.resources, "resources")
        if set(self.lease_owners) != set(self.resources) or not set(
            self.lease_owners.values()
        ) <= set(ids):
            raise ValueError("resource lease ownership does not match predecessors")
        expected_resources = {
            resource
            for operation in self.predecessor_operations
            for resource in operation.resources
        }
        if not expected_resources <= set(self.resources):
            raise ValueError("pre-phase must account for every predecessor resource")
        expected_participants = {
            robot
            for operation in self.predecessor_operations
            for robot in operation.participants
        }
        if self.next_operation is not None:
            if self.terminal_policy is not None or self.next_operation.skill == "WAIT":
                raise ValueError("WAIT or terminal policy cannot be a motion successor")
            expected_participants.update(self.next_operation.participants)
        elif self.terminal_policy is None:
            raise ValueError("terminal departure requires an explicit policy")
        else:
            expected_participants.update(self.terminal_policy.participants)
        if set(self.participants) != expected_participants:
            raise ValueError("pre-phase participant set does not match operations")
        robots = self.scene_snapshot.document["robots"]
        if not isinstance(robots, Mapping) or not set(self.participants) <= set(robots):
            raise ValueError("shared snapshot does not cover participants")

    @property
    def predecessor_operation_ids(self) -> tuple[str, ...]:
        """Return every physical predecessor identity.

        Returns:
            The identified, bound result; physical proof remains separately checked.
        """
        return tuple(
            operation.operation_id for operation in self.predecessor_operations
        )

    @property
    def successor_operation_ids(self) -> tuple[str, ...]:
        """Return the actual successor, or none for authorized terminal cleanup.

        Returns:
            The identified, bound result; physical proof remains separately checked.
        """
        return (
            () if self.next_operation is None else (self.next_operation.operation_id,)
        )

    @property
    def successor_operations(self) -> tuple[RobotOperation, ...]:
        """Return the immutable external successor operation.

        Returns:
            The identified, bound result; physical proof remains separately checked.
        """
        return () if self.next_operation is None else (self.next_operation,)

    @property
    def scene_digest(self) -> str:
        """Return the shared snapshot digest.

        Returns:
            The identified, bound result; physical proof remains separately checked.
        """
        return self.scene_snapshot.scene_digest

    @property
    def scene_version(self) -> int:
        """Return the shared snapshot version.

        Returns:
            The identified, bound result; physical proof remains separately checked.
        """
        return self.scene_snapshot.scene_version

    @property
    def terminal_cleanup(self) -> bool:
        """Return whether this is an explicitly authorized terminal departure.

        Returns:
            The identified, bound result; physical proof remains separately checked.
        """
        return self.terminal_policy is not None


@dataclass(frozen=True)
class SuccessorPrePhaseEvidence:
    """Physical arrival proof; scheduler ownership is committed separately."""

    transition_id: str
    predecessor_operation_ids: tuple[str, ...]
    successor_operation_ids: tuple[str, ...]
    participants: tuple[str, ...]
    resources: tuple[str, ...]
    lease_owners: Mapping[str, str]
    scene_digest: str
    scene_version: int
    planned: bool
    prepared: bool
    started: bool
    arrival_confirmed: bool
    terminal: bool
    safe: bool
    adoption_ready: bool
    final_facts: Mapping[str, Mapping[str, JSONValue]]
    evidence: Mapping[str, JSONValue]
    terminal_cleanup: bool = False
    reason: str | None = None
    transition_mode: str = "motion"

    def __post_init__(self) -> None:
        """Validate lifecycle booleans, identity coverage and JSON evidence."""
        if not isinstance(self.transition_id, str) or not self.transition_id:
            raise ValueError("transition_id must be non-empty")
        _identities(self.predecessor_operation_ids, "predecessors")
        _identities(
            self.successor_operation_ids, "successors", empty=self.terminal_cleanup
        )
        _identities(self.participants, "participants")
        _identities(self.resources, "resources")
        _digest(self.scene_digest, "scene_digest")
        if type(self.scene_version) is not int or self.scene_version < 0:
            raise ValueError("scene_version must be a non-negative integer")
        for name in (
            "planned",
            "prepared",
            "started",
            "arrival_confirmed",
            "terminal",
            "safe",
            "adoption_ready",
            "terminal_cleanup",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"pre-phase {name} must be a real bool")
        if set(self.lease_owners) != set(self.resources) or not set(
            self.lease_owners.values()
        ) <= set(self.predecessor_operation_ids):
            raise ValueError("pre-phase owner evidence is inconsistent")
        if set(self.final_facts) != set(self.participants) or not _json_safe(
            self.final_facts
        ):
            raise ValueError("pre-phase final facts must cover participants")
        if not isinstance(self.evidence, Mapping) or not _json_safe(self.evidence):
            raise ValueError("pre-phase evidence must be JSON-safe")
        if self.reason is not None and not isinstance(self.reason, str):
            raise ValueError("reason must be str or None")
        if self.transition_mode not in {"motion", "stationary_handover"}:
            raise ValueError("unknown physical lease transition mode")
        if self.transition_mode == "stationary_handover" and any(
            (
                self.planned,
                self.prepared,
                self.started,
                self.arrival_confirmed,
                self.terminal_cleanup,
            )
        ):
            raise ValueError(
                "stationary HANDOVER adoption cannot claim motion or arrival"
            )

    @property
    def passed(self) -> bool:
        """Return true only after arrival and terminal-safe adoption readiness.

        Returns:
            The identified, bound result; physical proof remains separately checked.
        """
        if self.transition_mode == "stationary_handover":
            return (
                self.terminal
                and self.safe
                and self.adoption_ready
                and (
                    self.evidence.get("stationary_custody_confirmed") is True
                    and self.evidence.get("motion_sent") is False
                )
            )
        return all(
            (
                self.planned,
                self.prepared,
                self.started,
                self.arrival_confirmed,
                self.terminal,
                self.safe,
                self.adoption_ready,
            )
        )

    def matches(self, request: SuccessorPrePhaseRequest) -> bool:
        """Compare every binding before physical leases can change owners.

        Returns:
            Whether this evidence answers the exact admitted request.
        """
        if self.transition_mode == "stationary_handover":
            operation = request.next_operation
            target = operation.target.values if operation and operation.target else {}
            donor, receiver = (
                target.get("donor_robot_id"),
                target.get("receiver_robot_id"),
            )
            physical = target.get("physical_instance_id")
            if (
                operation is None
                or operation.skill != "HANDOVER"
                or request.terminal_cleanup
                or (
                    donor == receiver
                    or {donor, receiver} != set(request.participants)
                    or self.evidence.get("physical_instance_id") != physical
                    or self.evidence.get("donor_robot_id") != donor
                    or self.evidence.get("receiver_robot_id") != receiver
                )
            ):
                return False
        return all(
            (
                self.transition_id == request.transition_id,
                self.predecessor_operation_ids == request.predecessor_operation_ids,
                self.successor_operation_ids == request.successor_operation_ids,
                self.participants == request.participants,
                self.resources == request.resources,
                dict(self.lease_owners) == dict(request.lease_owners),
                self.scene_digest == request.scene_digest,
                self.scene_version == request.scene_version,
                self.terminal_cleanup == request.terminal_cleanup,
            )
        )


def failed_pre_phase(
    request: SuccessorPrePhaseRequest, reason: str
) -> SuccessorPrePhaseEvidence:
    """Create bound failure evidence without inventing physical arrival.

    Returns:
        Evidence that cannot authorize physical lease adoption.
    """
    return SuccessorPrePhaseEvidence(
        transition_id=request.transition_id,
        predecessor_operation_ids=request.predecessor_operation_ids,
        successor_operation_ids=request.successor_operation_ids,
        participants=request.participants,
        resources=request.resources,
        lease_owners=request.lease_owners,
        scene_digest=request.scene_digest,
        scene_version=request.scene_version,
        planned=False,
        prepared=False,
        started=False,
        arrival_confirmed=False,
        terminal=False,
        safe=False,
        adoption_ready=False,
        final_facts={robot: {} for robot in request.participants},
        evidence={},
        terminal_cleanup=request.terminal_cleanup,
        reason=reason,
    )


class UnifiedDualArmRuntime(Protocol):
    """Single combined planner and controller execution boundary."""

    async def execute_pre_phase(
        self, request: SuccessorPrePhaseRequest
    ) -> SuccessorPrePhaseEvidence:
        """Execute an admitted successor-owned physical transition."""
        ...

    async def cancel_pre_phase(self, transition_id: str) -> Mapping[str, object]:
        """Cancel every participant and await physical terminal evidence."""
        ...


__all__ = [
    "SharedSceneSnapshot",
    "SuccessorPrePhaseEvidence",
    "SuccessorPrePhaseRequest",
    "TerminalPrePhasePolicy",
    "UnifiedDualArmRuntime",
    "failed_pre_phase",
]
