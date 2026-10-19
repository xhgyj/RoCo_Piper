"""Motion commit: the validated, lease-bound right to move named arms.

A ``MotionCommit`` is produced after planning, post-validation, lease and
revision checks. It binds one trajectory per participant, the session clock and
ownership-epoch namespace, the world/version evidence it was validated against,
and the reservation and lease identities it holds.

The commit is data only. Constructing or parsing one is neither authorization
nor execution: an executor must still recheck session, epoch, revisions, leases
and reservations before use. Terminal motion outcome is deliberately out of
scope -- controller terminal evidence (that the controller reached a terminal
state and confirmed a stop) is a separate contract from ``PhysicalEvidence``
(that the world is in the required physical state), and neither implies the
other.

Offline draft: stop handling, switch-point timing, acceptance summaries and
withdrawal markers described by the architecture are not modelled yet.
"""

from __future__ import annotations

from dataclasses import dataclass

from rocobrick.contracts._validation import (
    ContractViolationError,
    require_array,
    require_float,
    require_int,
    require_keys,
    require_object,
    require_schema,
    require_text,
    require_texts,
)
from rocobrick.contracts.identity import PhaseIdentity

MOTION_COMMIT_SCHEMA = "motion-commit/v1"

_COMMIT_KEYS = frozenset(
    {
        "schema",
        "commit_id",
        "identity",
        "session_id",
        "participants",
        "epoch",
        "world_revision",
        "trajectory_revision",
        "start_at_s",
        "world_evidence_ids",
        "lease_ids",
        "reservation_ids",
        "trajectories",
    }
)
_SAMPLE_KEYS = frozenset({"time_from_start_s", "positions", "velocities"})
_TRAJECTORY_KEYS = frozenset({"robot_id", "joint_names", "samples"})


@dataclass(frozen=True, slots=True)
class JointSample:
    """One timed joint state, relative to the commit start time."""

    time_from_start_s: float
    positions: tuple[float, ...]
    velocities: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        """Validate the finite time and the finite joint values."""
        require_float(self.time_from_start_s, "sample.time_from_start_s", minimum=0.0)
        if not isinstance(self.positions, tuple):
            raise ContractViolationError("sample.positions must be a tuple")
        for index, position in enumerate(self.positions):
            require_float(position, f"sample.positions[{index}]")
        if self.velocities is not None:
            if not isinstance(self.velocities, tuple):
                raise ContractViolationError("sample.velocities must be a tuple")
            for index, velocity in enumerate(self.velocities):
                require_float(velocity, f"sample.velocities[{index}]")

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dictionary.

        Returns:
            A dictionary with the relative time, positions and velocities.
        """
        return {
            "time_from_start_s": self.time_from_start_s,
            "positions": list(self.positions),
            "velocities": (None if self.velocities is None else list(self.velocities)),
        }

    @classmethod
    def from_dict(cls, payload: object) -> JointSample:
        """Parse one strict JSON joint sample.

        Args:
            payload: Decoded JSON value.

        Returns:
            The parsed joint sample.
        """
        item = require_object(payload, "sample")
        require_keys(item, _SAMPLE_KEYS, "sample")
        positions = require_array(item["positions"], "sample.positions")
        velocities = item["velocities"]
        return cls(
            time_from_start_s=require_float(
                item["time_from_start_s"],
                "sample.time_from_start_s",
                minimum=0.0,
            ),
            positions=tuple(
                require_float(value, f"sample.positions[{index}]")
                for index, value in enumerate(positions)
            ),
            velocities=(
                None
                if velocities is None
                else tuple(
                    require_float(value, f"sample.velocities[{index}]")
                    for index, value in enumerate(
                        require_array(velocities, "sample.velocities")
                    )
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class JointTrajectory:
    """One robot's joint trajectory with strictly increasing sampled times.

    Times are offsets from the commit start time; the first sample must be
    ``0.0`` so the start time is unambiguous.

    Joint names are model-qualified within the owning robot: ownership is keyed
    by ``(robot_id, joint_name)``, so the same local name on another robot is a
    different joint, never a conflict.
    """

    robot_id: str
    joint_names: tuple[str, ...]
    samples: tuple[JointSample, ...]

    def __post_init__(self) -> None:
        """Validate joint ownership, sample shape and time monotonicity."""
        require_text(self.robot_id, "trajectory.robot_id")
        if not isinstance(self.joint_names, tuple):
            raise ContractViolationError("trajectory.joint_names must be a tuple")
        joint_names = require_texts(self.joint_names, "trajectory.joint_names")
        if not isinstance(self.samples, tuple):
            raise ContractViolationError("trajectory.samples must be a tuple")
        if not self.samples:
            raise ContractViolationError("trajectory.samples must not be empty")
        previous = -1.0
        for index, sample in enumerate(self.samples):
            if not isinstance(sample, JointSample):
                raise ContractViolationError(
                    "trajectory.samples must contain JointSample values"
                )
            if len(sample.positions) != len(joint_names):
                raise ContractViolationError(
                    f"trajectory.samples[{index}].positions must match "
                    f"{len(joint_names)} joint names"
                )
            if sample.velocities is not None and len(sample.velocities) != len(
                joint_names
            ):
                raise ContractViolationError(
                    f"trajectory.samples[{index}].velocities must match "
                    f"{len(joint_names)} joint names"
                )
            if index == 0 and sample.time_from_start_s != 0.0:
                raise ContractViolationError(
                    "trajectory first sample must start at 0.0"
                )
            if index > 0 and sample.time_from_start_s <= previous:
                raise ContractViolationError(
                    "trajectory sample times must strictly increase"
                )
            previous = sample.time_from_start_s

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dictionary.

        Returns:
            A dictionary with the owning robot, joint names and samples.
        """
        return {
            "robot_id": self.robot_id,
            "joint_names": list(self.joint_names),
            "samples": [sample.to_dict() for sample in self.samples],
        }

    @classmethod
    def from_dict(cls, payload: object) -> JointTrajectory:
        """Parse one strict JSON joint trajectory.

        Args:
            payload: Decoded JSON value.

        Returns:
            The parsed joint trajectory.
        """
        item = require_object(payload, "trajectory")
        require_keys(item, _TRAJECTORY_KEYS, "trajectory")
        return cls(
            robot_id=require_text(item["robot_id"], "trajectory.robot_id"),
            joint_names=require_texts(item["joint_names"], "trajectory.joint_names"),
            samples=tuple(
                JointSample.from_dict(entry)
                for entry in require_array(item["samples"], "trajectory.samples")
            ),
        )


@dataclass(frozen=True, slots=True)
class MotionCommit:
    """Lease-bound, version-pinned right for named arms to execute one plan.

    Construction and validation are neither execution nor authorization: the
    executor must still recheck session, epoch, revisions, leases and
    reservations before use.

    Joint ownership is keyed by ``(robot_id, joint_name)``; joint names must be
    qualified within the owning robot's model.

    Attributes:
        commit_id: Unique commit identity; redelivery must not re-move.
        identity: Phase identity the commit was planned for.
        session_id: Runtime session defining the clock domain and the ownership
            epoch namespace; equal epoch numbers in different sessions are
            different epochs and must not be compared.
        participants: Exact robot ids the commit binds.
        epoch: Execution ownership epoch within ``session_id``.
        world_revision: World revision the trajectories were validated on.
        trajectory_revision: Trajectory revision within the epoch.
        start_at_s: Session-domain start time in seconds.
        world_evidence_ids: Evidence records backing the version pins.
        lease_ids: Task resource leases held for this motion.
        reservation_ids: Time-space reservations held for this motion.
        trajectories: Exactly one trajectory per participant.
    """

    commit_id: str
    identity: PhaseIdentity
    session_id: str
    participants: tuple[str, ...]
    epoch: int
    world_revision: int
    trajectory_revision: int
    start_at_s: float
    world_evidence_ids: tuple[str, ...]
    lease_ids: tuple[str, ...]
    reservation_ids: tuple[str, ...]
    trajectories: tuple[JointTrajectory, ...]

    def __post_init__(self) -> None:
        """Validate identity, epoch, evidence pins and joint ownership."""
        require_text(self.commit_id, "commit.commit_id")
        if not isinstance(self.identity, PhaseIdentity):
            raise ContractViolationError("commit.identity must be a PhaseIdentity")
        require_text(self.session_id, "commit.session_id")
        if not isinstance(self.participants, tuple):
            raise ContractViolationError("commit.participants must be a tuple")
        participants = require_texts(self.participants, "commit.participants")
        require_int(self.epoch, "commit.epoch")
        require_int(self.world_revision, "commit.world_revision")
        require_int(self.trajectory_revision, "commit.trajectory_revision")
        require_float(self.start_at_s, "commit.start_at_s", minimum=0.0)
        if not isinstance(self.world_evidence_ids, tuple):
            raise ContractViolationError("commit.world_evidence_ids must be a tuple")
        require_texts(self.world_evidence_ids, "commit.world_evidence_ids")
        if not isinstance(self.lease_ids, tuple):
            raise ContractViolationError("commit.lease_ids must be a tuple")
        require_texts(self.lease_ids, "commit.lease_ids")
        if not isinstance(self.reservation_ids, tuple):
            raise ContractViolationError("commit.reservation_ids must be a tuple")
        require_texts(self.reservation_ids, "commit.reservation_ids")
        if not isinstance(self.trajectories, tuple):
            raise ContractViolationError("commit.trajectories must be a tuple")
        owned_joints: set[tuple[str, str]] = set()
        bound_robots: set[str] = set()
        for trajectory in self.trajectories:
            if not isinstance(trajectory, JointTrajectory):
                raise ContractViolationError(
                    "commit.trajectories must contain JointTrajectory values"
                )
            if trajectory.robot_id in bound_robots:
                raise ContractViolationError(
                    f"duplicate trajectory for {trajectory.robot_id!r}"
                )
            bound_robots.add(trajectory.robot_id)
            for joint_name in trajectory.joint_names:
                ownership = (trajectory.robot_id, joint_name)
                if ownership in owned_joints:
                    raise ContractViolationError(
                        f"joint {joint_name!r} of {trajectory.robot_id!r} is "
                        f"owned by more than one trajectory"
                    )
                owned_joints.add(ownership)
        if bound_robots != set(participants):
            missing = sorted(set(participants) - bound_robots)
            unknown = sorted(bound_robots - set(participants))
            raise ContractViolationError(
                f"commit.trajectories must bind every participant exactly "
                f"once: missing {missing}, unknown {unknown}"
            )

    def trajectory_for(self, robot_id: str) -> JointTrajectory:
        """Return the one trajectory bound to a participant.

        Args:
            robot_id: Participant robot identity.

        Returns:
            The trajectory owned by that robot.
        """
        for trajectory in self.trajectories:
            if trajectory.robot_id == robot_id:
                return trajectory
        raise ContractViolationError(f"{robot_id!r} is not a commit participant")

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dictionary.

        Returns:
            A dictionary containing only JSON primitives, lists and objects.
        """
        return {
            "schema": MOTION_COMMIT_SCHEMA,
            "commit_id": self.commit_id,
            "identity": self.identity.to_dict(),
            "session_id": self.session_id,
            "participants": list(self.participants),
            "epoch": self.epoch,
            "world_revision": self.world_revision,
            "trajectory_revision": self.trajectory_revision,
            "start_at_s": self.start_at_s,
            "world_evidence_ids": list(self.world_evidence_ids),
            "lease_ids": list(self.lease_ids),
            "reservation_ids": list(self.reservation_ids),
            "trajectories": [trajectory.to_dict() for trajectory in self.trajectories],
        }

    @classmethod
    def from_dict(cls, payload: object) -> MotionCommit:
        """Parse one strict JSON motion commit.

        Args:
            payload: Decoded JSON value.

        Returns:
            The parsed motion commit.
        """
        item = require_object(payload, "commit")
        require_keys(item, _COMMIT_KEYS, "commit")
        require_schema(item["schema"], MOTION_COMMIT_SCHEMA, "commit")
        return cls(
            commit_id=require_text(item["commit_id"], "commit.commit_id"),
            identity=PhaseIdentity.from_dict(item["identity"]),
            session_id=require_text(item["session_id"], "commit.session_id"),
            participants=require_texts(item["participants"], "commit.participants"),
            epoch=require_int(item["epoch"], "commit.epoch"),
            world_revision=require_int(item["world_revision"], "commit.world_revision"),
            trajectory_revision=require_int(
                item["trajectory_revision"], "commit.trajectory_revision"
            ),
            start_at_s=require_float(
                item["start_at_s"], "commit.start_at_s", minimum=0.0
            ),
            world_evidence_ids=require_texts(
                item["world_evidence_ids"], "commit.world_evidence_ids"
            ),
            lease_ids=require_texts(item["lease_ids"], "commit.lease_ids"),
            reservation_ids=require_texts(
                item["reservation_ids"], "commit.reservation_ids"
            ),
            trajectories=tuple(
                JointTrajectory.from_dict(entry)
                for entry in require_array(item["trajectories"], "commit.trajectories")
            ),
        )
