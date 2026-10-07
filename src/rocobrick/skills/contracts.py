"""Strict physical HANDOVER contracts.

These contracts describe a real change of gripper ownership.  They do not
create a plan operation or infer a transfer from owner fields alone.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from rocobrick.multi_robot_scheduler import JSONValue, is_json_safe, to_json_value


def _identities(
    values: Sequence[str], label: str, *, empty: bool = False
) -> tuple[str, ...]:
    result = tuple(values)
    if (
        (not result and not empty)
        or any(not isinstance(value, str) or not value for value in result)
        or len(result) != len(set(result))
    ):
        raise ValueError(f"{label} must contain unique non-empty identities")
    return result


def _positive_finite(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


@dataclass(frozen=True)
class HandoverTransferIdentity:
    """Minimal physical identity shared by coordinated custody and the ledger."""

    operation_id: str
    donor_robot_id: str
    receiver_robot_id: str
    physical_instance_id: str

    def __post_init__(self) -> None:
        """Reject missing or ambiguous participants before any custody action."""
        for value in (
            self.operation_id,
            self.donor_robot_id,
            self.receiver_robot_id,
            self.physical_instance_id,
        ):
            if not isinstance(value, str) or not value:
                raise ValueError("HANDOVER physical identity is incomplete")
        if self.donor_robot_id == self.receiver_robot_id:
            raise ValueError("HANDOVER requires different physical holders")


@dataclass(frozen=True)
class HandoverRequest:
    """One admitted donor-to-receiver physical transfer."""

    operation_id: str
    donor_robot_id: str
    receiver_robot_id: str
    physical_instance_id: str
    session_robots: tuple[str, ...]
    scene_snapshot: Mapping[str, JSONValue]
    scene_version: int
    start_state: Mapping[str, JSONValue]
    receiver_target: Mapping[str, JSONValue]
    receiver_gripper: Mapping[str, JSONValue]
    donor_release_gripper: Mapping[str, JSONValue]
    receiver_attach: Mapping[str, JSONValue]
    donor_detach: Mapping[str, JSONValue]
    timeout_s: float
    follow_samples: int = 3

    def __post_init__(self) -> None:
        """Require complete identity, scene provenance and receiver geometry."""
        for label, value in (
            ("operation_id", self.operation_id),
            ("donor_robot_id", self.donor_robot_id),
            ("receiver_robot_id", self.receiver_robot_id),
            ("physical_instance_id", self.physical_instance_id),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a non-empty string")
        if self.donor_robot_id == self.receiver_robot_id:
            raise ValueError("handover donor and receiver must be different robots")
        session = _identities(self.session_robots, "session_robots")
        if not {self.donor_robot_id, self.receiver_robot_id} <= set(session):
            raise ValueError("handover session_robots must cover donor and receiver")
        if type(self.scene_version) is not int or self.scene_version < 0:
            raise ValueError("handover scene_version must be a non-negative integer")
        if not isinstance(self.scene_snapshot, Mapping) or not is_json_safe(
            self.scene_snapshot
        ):
            raise ValueError("handover scene_snapshot must be JSON-safe")
        if not isinstance(self.scene_snapshot.get("scene_digest"), str):
            raise ValueError("handover scene_snapshot requires scene_digest")
        if not isinstance(self.start_state, Mapping) or not is_json_safe(
            self.start_state
        ):
            raise ValueError("handover start_state must be JSON-safe")
        for label, payload in (
            ("receiver_gripper", self.receiver_gripper),
            ("donor_release_gripper", self.donor_release_gripper),
            ("receiver_attach", self.receiver_attach),
            ("donor_detach", self.donor_detach),
        ):
            if not isinstance(payload, Mapping) or not payload:
                raise ValueError(f"handover {label} must be a non-empty mapping")
            if not is_json_safe(payload):
                raise ValueError(f"handover {label} must be JSON-safe")
        target = self.receiver_target
        if not isinstance(target, Mapping) or set(target) != {"tcp_pose"}:
            raise ValueError("handover receiver_target must contain one tcp_pose")
        tcp = target["tcp_pose"]
        required_tcp = {
            "position",
            "orientation_xyzw",
            "frame_id",
            "tip_link",
            "runtime_model_sha256",
            "source_sha256",
            "position_tolerance_m",
            "orientation_tolerance_rad",
        }
        if not isinstance(tcp, Mapping) or set(tcp) != required_tcp:
            raise ValueError("handover receiver tcp target provenance is incomplete")
        if (
            tcp["frame_id"] != f"{self.receiver_robot_id}_base_link"
            or tcp["tip_link"] != f"{self.receiver_robot_id}_rocobrick_tcp"
        ):
            raise ValueError("handover receiver target frame/tip must match receiver")
        for digest_name in ("runtime_model_sha256", "source_sha256"):
            digest = tcp[digest_name]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                raise ValueError(f"handover receiver {digest_name} is invalid")
        timeout = _positive_finite(self.timeout_s, "handover timeout_s")
        if type(self.follow_samples) is not int or self.follow_samples < 3:
            raise ValueError("handover follow_samples must be an integer >= 3")
        object.__setattr__(self, "session_robots", session)
        object.__setattr__(self, "scene_snapshot", to_json_value(self.scene_snapshot))
        object.__setattr__(self, "start_state", to_json_value(self.start_state))
        object.__setattr__(self, "receiver_target", to_json_value(self.receiver_target))
        object.__setattr__(
            self, "receiver_gripper", to_json_value(self.receiver_gripper)
        )
        object.__setattr__(
            self, "donor_release_gripper", to_json_value(self.donor_release_gripper)
        )
        object.__setattr__(self, "receiver_attach", to_json_value(self.receiver_attach))
        object.__setattr__(self, "donor_detach", to_json_value(self.donor_detach))
        object.__setattr__(self, "timeout_s", timeout)

    @property
    def participants(self) -> tuple[str, str]:
        """Return the donor and receiver that participate in the transfer.

        Returns:
            The identified, bound result; physical proof remains separately checked.
        """
        return self.donor_robot_id, self.receiver_robot_id


@dataclass(frozen=True)
class HandoverEvidence:
    """Measured lifecycle evidence for one exact HANDOVER request."""

    operation_id: str
    donor_robot_id: str
    receiver_robot_id: str
    physical_instance_id: str
    receiver_free_space: bool
    receiver_grasp_feedback: bool
    co_hold_confirmed: bool
    scene_transfer_committed: bool
    donor_release_feedback: bool
    receiver_holding_follow: bool
    terminal: bool
    safe_to_release: bool
    owner_transfer_committed: bool
    evidence: Mapping[str, JSONValue]
    reason: str | None = None

    def __post_init__(self) -> None:
        """Reject truthy non-booleans and unbound or unsafe evidence."""
        for label, value in (
            ("operation_id", self.operation_id),
            ("donor_robot_id", self.donor_robot_id),
            ("receiver_robot_id", self.receiver_robot_id),
            ("physical_instance_id", self.physical_instance_id),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"handover {label} must be a non-empty string")
        if self.donor_robot_id == self.receiver_robot_id:
            raise ValueError("handover donor and receiver must be different robots")
        for name in (
            "receiver_free_space",
            "receiver_grasp_feedback",
            "co_hold_confirmed",
            "scene_transfer_committed",
            "donor_release_feedback",
            "receiver_holding_follow",
            "terminal",
            "safe_to_release",
            "owner_transfer_committed",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"handover {name} must be a real bool")
        if not isinstance(self.evidence, Mapping) or not is_json_safe(self.evidence):
            raise ValueError("handover evidence must be a JSON-safe mapping")
        if self.reason is not None and not isinstance(self.reason, str):
            raise ValueError("handover reason must be a string or null")
        object.__setattr__(self, "evidence", to_json_value(self.evidence))

    @property
    def ready_to_commit(self) -> bool:
        """Return whether physical evidence is sufficient for ledger adoption.

        Returns:
            The identified, bound result; physical proof remains separately checked.
        """
        return (
            all(
                (
                    self.receiver_free_space,
                    self.receiver_grasp_feedback,
                    self.co_hold_confirmed,
                    self.scene_transfer_committed,
                    self.donor_release_feedback,
                    self.receiver_holding_follow,
                    self.terminal,
                    self.safe_to_release,
                )
            )
            and not self.owner_transfer_committed
        )

    @property
    def passed(self) -> bool:
        """Return true only after physical evidence and atomic adoption.

        Returns:
            The identified, bound result; physical proof remains separately checked.
        """
        return (
            self.ready_to_commit is False
            and self.owner_transfer_committed
            and all(
                (
                    self.receiver_free_space,
                    self.receiver_grasp_feedback,
                    self.co_hold_confirmed,
                    self.scene_transfer_committed,
                    self.donor_release_feedback,
                    self.receiver_holding_follow,
                    self.terminal,
                    self.safe_to_release,
                )
            )
        )

    def matches(self, request: HandoverRequest | HandoverTransferIdentity) -> bool:
        """Return whether evidence answers the exact admitted transfer.

        Returns:
            The identified, bound result; physical proof remains separately checked.
        """
        return (
            self.operation_id == request.operation_id
            and self.donor_robot_id == request.donor_robot_id
            and self.receiver_robot_id == request.receiver_robot_id
            and self.physical_instance_id == request.physical_instance_id
        )

    def committed(self) -> "HandoverEvidence":
        """Mark owner adoption after the ledger has atomically transferred it.

        Returns:
            Evidence marked as committed after physical gates passed.
        """
        if not self.ready_to_commit:
            raise ValueError("handover physical evidence is not ready for adoption")
        return replace(self, owner_transfer_committed=True)


__all__ = ["HandoverEvidence", "HandoverRequest", "HandoverTransferIdentity"]
