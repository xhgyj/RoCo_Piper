"""Confirmed per-robot operation state ledger.

The scheduler owns dependencies and resource leases. This module owns only
state proven by executor evidence, so a planned effect can never masquerade as
a physical fact.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rocobrick.multi_robot_scheduler import JSONValue, is_json_safe, to_json_value
from rocobrick.skills.contracts import HandoverEvidence, HandoverRequest
from rocobrick.unified_dual_arm_runtime import (
    SuccessorPrePhaseEvidence,
    SuccessorPrePhaseRequest,
)


@dataclass(frozen=True)
class ConfirmedState:
    """One state fact confirmed by runtime evidence."""

    robot_id: str
    key: str
    evidence: Mapping[str, JSONValue]


@dataclass(frozen=True)
class UncertainHandover:
    """One physical transfer whose post-release owner is unknown."""

    operation_id: str
    physical_instance_id: str
    donor_robot_id: str
    receiver_robot_id: str
    reason: str
    evidence: Mapping[str, JSONValue]


class OperationStateLedger:
    """Session-scoped, robot-isolated ledger of confirmed physical state."""

    def __init__(self, robot_ids: Sequence[str]) -> None:
        """Create isolated confirmed and uncertain physical-state stores."""
        if (
            isinstance(robot_ids, (str, bytes, bytearray))
            or not robot_ids
            or any(
                not isinstance(robot_id, str) or not robot_id for robot_id in robot_ids
            )
            or len(set(robot_ids)) != len(robot_ids)
        ):
            raise ValueError("robot_ids must be a non-empty sequence of unique strings")
        self._robot_ids = tuple(robot_ids)
        self._states: dict[str, dict[str, ConfirmedState]] = {
            robot_id: {} for robot_id in self._robot_ids
        }
        self._object_owners: dict[str, str] = {}
        self._uncertain: dict[tuple[str, str], str] = {}
        self._uncertain_handovers: dict[str, UncertainHandover] = {}

    @property
    def robot_ids(self) -> tuple[str, ...]:
        """Validated robot ids owned by this ledger."""
        return self._robot_ids

    def observe(self, robot_id: str, key: str) -> ConfirmedState | None:
        """Return a confirmed fact, or None when it has not been proven."""
        self._validate_robot(robot_id)
        self._validate_key(key)
        return self._states[robot_id].get(key)

    def confirm(
        self, robot_id: str, key: str, evidence: Mapping[str, JSONValue]
    ) -> ConfirmedState:
        """Record one fact only when its evidence is a strict JSON mapping.

        Returns:
            The newly confirmed robot-scoped fact.
        """
        self._validate_robot(robot_id)
        self._validate_key(key)
        if not isinstance(evidence, Mapping) or not is_json_safe(evidence):
            raise ValueError("evidence must be a JSON-safe mapping")
        instance_id = self._holding_instance_id(key)
        if instance_id is not None:
            owner = self._object_owners.get(instance_id)
            if owner is not None and owner != robot_id:
                raise ValueError(
                    f"object {instance_id!r} is already owned by {owner!r}"
                )
        if (robot_id, key) in self._uncertain:
            raise ValueError(
                "uncertain physical state requires explicit reconciliation"
            )
        confirmed = ConfirmedState(
            robot_id=robot_id, key=key, evidence=to_json_value(evidence)
        )
        self._states[robot_id][key] = confirmed
        if instance_id is not None:
            self._object_owners[instance_id] = robot_id
        return confirmed

    def confirm_holding(
        self,
        robot_id: str,
        physical_instance_id: str,
        evidence: Mapping[str, JSONValue],
    ) -> ConfirmedState:
        """Confirm exclusive holding for one physical instance.

        Returns:
            The robot-scoped confirmed holding fact.
        """
        self._validate_instance_id(physical_instance_id)
        return self.confirm(robot_id, f"holding:{physical_instance_id}", evidence)

    def owner_of_object(self, physical_instance_id: str) -> str | None:
        """Return the protective owner, including an uncertain holding state."""
        self._validate_instance_id(physical_instance_id)
        return self._object_owners.get(physical_instance_id)

    def invalidate_if_stale(
        self, robot_id: str, physical_instance_id: str, reason: str
    ) -> bool:
        """Remove stale holding confirmation without freeing its object.

        Returns:
            Whether a confirmed holding fact was removed.
        """
        self._validate_robot(robot_id)
        self._validate_instance_id(physical_instance_id)
        self._validate_reason(reason)
        if self._object_owners.get(physical_instance_id) != robot_id:
            return False
        key = f"holding:{physical_instance_id}"
        existed = self._states[robot_id].pop(key, None) is not None
        self._uncertain[(robot_id, key)] = reason
        return existed

    def revoke(self, robot_id: str, key: str, reason: str) -> bool:
        """Remove a known fact.

        Returns:
            Whether a confirmed fact existed.
        """
        self._validate_robot(robot_id)
        self._validate_key(key)
        self._validate_reason(reason)
        if (robot_id, key) in self._uncertain:
            raise ValueError(
                "cannot revoke uncertain physical state without terminal evidence"
            )
        instance_id = self._holding_instance_id(key)
        if instance_id is not None and self._object_owners.get(instance_id) == robot_id:
            del self._object_owners[instance_id]
        return self._states[robot_id].pop(key, None) is not None

    def invalidate_support(
        self, robot_id: str, physical_instance_id: str, reason: str
    ) -> None:
        """Invalidate support without assuming physical departure."""
        self._validate_robot(robot_id)
        self._validate_instance_id(physical_instance_id)
        self._validate_reason(reason)
        key = f"supporting:{physical_instance_id}"
        self._states[robot_id].pop(key, None)
        self._uncertain[(robot_id, key)] = reason

    def mark_handover_uncertain(
        self,
        request: HandoverRequest,
        reason: str,
        evidence: Mapping[str, JSONValue],
    ) -> UncertainHandover:
        """Record an unknown post-release transfer without changing owner."""
        self._validate_robot(request.donor_robot_id)
        self._validate_robot(request.receiver_robot_id)
        self._validate_instance_id(request.physical_instance_id)
        self._validate_reason(reason)
        if not isinstance(evidence, Mapping) or not is_json_safe(evidence):
            raise ValueError("uncertain handover evidence must be JSON-safe")
        key = f"holding:{request.physical_instance_id}"
        self._states[request.donor_robot_id].pop(key, None)
        self._uncertain[(request.donor_robot_id, key)] = reason
        self._uncertain[(request.receiver_robot_id, key)] = reason
        self._object_owners.setdefault(
            request.physical_instance_id, request.donor_robot_id
        )
        record = UncertainHandover(
            request.operation_id,
            request.physical_instance_id,
            request.donor_robot_id,
            request.receiver_robot_id,
            reason,
            to_json_value(evidence),
        )
        self._uncertain_handovers[request.physical_instance_id] = record
        return record

    def uncertain_handover(
        self, physical_instance_id: str
    ) -> UncertainHandover | None:
        """Return unresolved transfer state for one physical instance."""
        self._validate_instance_id(physical_instance_id)
        return self._uncertain_handovers.get(physical_instance_id)

    def reconcile_handover(
        self,
        physical_instance_id: str,
        *,
        owner_robot_id: str | None,
        evidence: Mapping[str, JSONValue],
    ) -> None:
        """Resolve UNKNOWN_HANDOVER only from explicit measured evidence."""
        self._validate_instance_id(physical_instance_id)
        if owner_robot_id is not None:
            self._validate_robot(owner_robot_id)
        if not isinstance(evidence, Mapping) or not is_json_safe(evidence):
            raise ValueError("handover reconciliation evidence must be JSON-safe")
        record = self._uncertain_handovers.get(physical_instance_id)
        if record is None:
            raise ValueError("physical instance has no uncertain handover")
        key = f"holding:{physical_instance_id}"
        for robot_id in (record.donor_robot_id, record.receiver_robot_id):
            self._states[robot_id].pop(key, None)
            self._uncertain.pop((robot_id, key), None)
        if owner_robot_id is None:
            self._object_owners.pop(physical_instance_id, None)
        else:
            self._object_owners.pop(physical_instance_id, None)
            self.confirm_holding(owner_robot_id, physical_instance_id, evidence)
        del self._uncertain_handovers[physical_instance_id]

    def transfer_holding(
        self,
        request: HandoverRequest,
        evidence: HandoverEvidence,
    ) -> ConfirmedState:
        """Atomically transfer one confirmed holding fact to a receiver.

        The donor fact and object owner remain unchanged unless every physical
        HANDOVER gate passes.  A partial transfer therefore fails closed.

        Returns:
            The receiver's newly confirmed holding state.
        """
        if not evidence.matches(request):
            raise ValueError("handover evidence does not match request")
        if not evidence.ready_to_commit:
            raise ValueError("handover requires complete terminal-safe evidence")
        self._validate_robot(request.donor_robot_id)
        self._validate_robot(request.receiver_robot_id)
        self._validate_instance_id(request.physical_instance_id)
        key = f"holding:{request.physical_instance_id}"
        donor_state = self._states[request.donor_robot_id].get(key)
        if (
            donor_state is None
            or self._object_owners.get(request.physical_instance_id)
            != request.donor_robot_id
        ):
            raise ValueError("handover donor holding is not confirmed")
        if self._states[request.receiver_robot_id].get(key) is not None:
            raise ValueError("handover receiver already has holding evidence")
        if any(
            owner == request.donor_robot_id and state_key == key
            for owner, state_key in self._uncertain
        ) or any(
            owner == request.receiver_robot_id and state_key == key
            for owner, state_key in self._uncertain
        ):
            raise ValueError("handover cannot adopt uncertain holding state")
        receiver_evidence = dict(evidence.evidence)
        receiver_evidence.update(
            {
                "holding_confirmed": True,
                "handover_operation_id": request.operation_id,
                "donor_robot_id": request.donor_robot_id,
                "receiver_robot_id": request.receiver_robot_id,
                "physical_instance_id": request.physical_instance_id,
            }
        )
        if not is_json_safe(receiver_evidence):
            raise ValueError("handover adoption evidence is not JSON-safe")
        confirmed = ConfirmedState(
            request.receiver_robot_id, key, to_json_value(receiver_evidence)
        )
        self._states[request.receiver_robot_id][key] = confirmed
        del self._states[request.donor_robot_id][key]
        self._object_owners[request.physical_instance_id] = request.receiver_robot_id
        return confirmed

    def reconcile_pre_phase(
        self, request: SuccessorPrePhaseRequest, evidence: SuccessorPrePhaseEvidence
    ) -> None:
        """Atomically refresh physical facts after terminal-safe arrival.

        Raises:
            ValueError: If any physical fact, participant or evidence is inconsistent.
        """
        if not evidence.matches(request) or not evidence.passed:
            raise ValueError(
                "ledger reconciliation requires bound terminal-safe pre-phase"
            )
        releases = (
            set(request.terminal_policy.release_supporting)
            if request.terminal_policy
            else set()
        )
        replacements: dict[str, dict[str, ConfirmedState]] = {}
        for robot_id in request.participants:
            self._validate_robot(robot_id)
            if any(robot == robot_id for robot, _ in self._uncertain):
                raise ValueError(
                    "uncertain holding/supporting state cannot enter pre-phase"
                )
            previous = self._states[robot_id]
            physical = {
                key for key in previous if key.startswith(("holding:", "supporting:"))
            }
            final = evidence.final_facts[robot_id]
            required = {
                key
                for key in physical
                if not (key.startswith("supporting:") and robot_id in releases)
            }
            if not required <= set(final) or any(
                not isinstance(final[key], Mapping) or not final[key]
                for key in required
            ):
                raise ValueError(
                    "pre-phase lost confirmed holding/supporting continuity"
                )
            if set(final) - physical:
                raise ValueError(
                    "free-space pre-phase cannot establish new physical facts"
                )
            updated = dict(previous)
            for key in physical:
                if key in final:
                    updated[key] = ConfirmedState(
                        robot_id, key, to_json_value(final[key])
                    )
                elif key.startswith("supporting:") and robot_id in releases:
                    del updated[key]
            replacements[robot_id] = updated
        self._states.update(replacements)

    def snapshot(self, robot_id: str) -> Mapping[str, ConfirmedState]:
        """Return an immutable-by-copy view of one robot's confirmed facts."""
        self._validate_robot(robot_id)
        return dict(self._states[robot_id])

    def uncertain_keys(self, robot_id: str) -> tuple[str, ...]:
        """Return this robot's facts awaiting explicit reconciliation.

        A non-empty answer means the robot's holding/supporting state is
        unknown, so pre-phase admission must refuse the robot outright
        instead of guessing a home/empty state.
        """
        self._validate_robot(robot_id)
        return tuple(
            key for (owner, key) in self._uncertain if owner == robot_id
        )

    def _validate_robot(self, robot_id: str) -> None:
        if robot_id not in self._states:
            raise ValueError(f"unknown robot_id: {robot_id!r}")

    @staticmethod
    def _validate_key(key: str) -> None:
        if not isinstance(key, str) or not key:
            raise ValueError("state key must be a non-empty string")
        if key == "holding:":
            raise ValueError("holding state requires a physical instance id")

    @staticmethod
    def _validate_instance_id(physical_instance_id: str) -> None:
        if not isinstance(physical_instance_id, str) or not physical_instance_id:
            raise ValueError("physical_instance_id must be a non-empty string")

    @staticmethod
    def _validate_reason(reason: str) -> None:
        if not isinstance(reason, str) or not reason:
            raise ValueError("reason must be a non-empty string")

    @staticmethod
    def _holding_instance_id(key: str) -> str | None:
        return key.removeprefix("holding:") if key.startswith("holding:") else None


__all__ = ["ConfirmedState", "OperationStateLedger", "UncertainHandover"]
