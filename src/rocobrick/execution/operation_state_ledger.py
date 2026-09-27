"""Confirmed per-robot operation state ledger.

The scheduler owns dependencies and resource leases. This module owns only
state proven by executor evidence, so a planned effect can never masquerade as
a physical fact.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rocobrick.multi_robot_scheduler import JSONValue, is_json_safe, to_json_value


@dataclass(frozen=True)
class ConfirmedState:
    """One state fact confirmed by runtime evidence."""

    robot_id: str
    key: str
    evidence: Mapping[str, JSONValue]


class OperationStateLedger:
    """Session-scoped, robot-isolated ledger of confirmed physical state."""

    def __init__(self, robot_ids: Sequence[str]) -> None:
        if (
            isinstance(robot_ids, (str, bytes, bytearray))
            or not robot_ids
            or any(not isinstance(robot_id, str) or not robot_id for robot_id in robot_ids)
            or len(set(robot_ids)) != len(robot_ids)
        ):
            raise ValueError("robot_ids must be a non-empty sequence of unique strings")
        self._robot_ids = tuple(robot_ids)
        self._states: dict[str, dict[str, ConfirmedState]] = {
            robot_id: {} for robot_id in self._robot_ids
        }
        self._object_owners: dict[str, str] = {}

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
        """Record one fact only when its evidence is a strict JSON mapping."""
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
        return self._states[robot_id].pop(
            f"holding:{physical_instance_id}", None
        ) is not None

    def revoke(self, robot_id: str, key: str, reason: str) -> bool:
        """Remove a fact and return whether a confirmed fact existed."""
        self._validate_robot(robot_id)
        self._validate_key(key)
        self._validate_reason(reason)
        instance_id = self._holding_instance_id(key)
        if instance_id is not None and self._object_owners.get(instance_id) == robot_id:
            del self._object_owners[instance_id]
        return self._states[robot_id].pop(key, None) is not None

    def snapshot(self, robot_id: str) -> Mapping[str, ConfirmedState]:
        """Return an immutable-by-copy view of one robot's confirmed facts."""
        self._validate_robot(robot_id)
        return dict(self._states[robot_id])

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


__all__ = ["ConfirmedState", "OperationStateLedger"]
