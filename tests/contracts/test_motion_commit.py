"""Motion commits bind participants, joints, versions and monotonic times."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest
from contract_doubles import commit, trajectory

from rocobrick.contracts import (
    ContractViolationError,
    JointSample,
    JointTrajectory,
    MotionCommit,
)


def test_roundtrip_preserves_binding_and_timing() -> None:
    """A commit survives JSON roundtrip with participants and times intact."""
    original = commit(start_at_s=250.5)
    payload = json.loads(json.dumps(original.to_dict()))
    parsed = MotionCommit.from_dict(payload)
    assert parsed == original
    assert parsed.start_at_s == 250.5
    assert parsed.session_id == "session-1"
    assert parsed.trajectory_for("piper_1").robot_id == "piper_1"


def test_commit_is_immutable_including_trajectories() -> None:
    """Commit, trajectory and sample values cannot be mutated in place."""
    value = commit()
    with pytest.raises(FrozenInstanceError):
        value.epoch = 4  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        value.trajectories[0].joint_names = ("x",)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        value.trajectories[0].samples[0].positions = (9.0,)  # type: ignore[misc]


def test_every_participant_is_bound_exactly_once() -> None:
    """Missing and unknown trajectories both fail closed."""
    with pytest.raises(ContractViolationError, match=r"missing \['piper_1'\]"):
        commit(trajectories=(trajectory(robot_id="piper_0"),))
    with pytest.raises(ContractViolationError, match="unknown"):
        commit(
            participants=("piper_0",),
            trajectories=(
                trajectory(robot_id="piper_0"),
                trajectory(robot_id="piper_9"),
            ),
        )
    with pytest.raises(ContractViolationError, match="not a commit participant"):
        commit().trajectory_for("piper_9")


def test_session_domain_disambiguates_equal_epochs() -> None:
    """Epochs are scoped to the runtime session that minted them.

    A restart mints a new session, so an equal epoch number in another session
    names a different ownership epoch instead of being ambiguous.
    """
    first = commit(session_id="session-1")
    restarted = commit(session_id="session-2")
    assert first.epoch == restarted.epoch
    assert first != restarted
    assert first.to_dict()["session_id"] == "session-1"


def test_session_id_is_required() -> None:
    """A commit without a session cannot place its epoch in time."""
    with pytest.raises(ContractViolationError, match="commit.session_id"):
        _single_participant_commit(session_id="")
    with pytest.raises(ContractViolationError, match="commit.session_id"):
        _single_participant_commit(session_id=True)


def test_joint_ownership_is_keyed_by_robot_and_joint() -> None:
    """The same local joint name on two robots is two distinct joints."""
    shared = commit(
        trajectories=(
            trajectory(robot_id="piper_0", joint_names=("joint_1",)),
            trajectory(robot_id="piper_1", joint_names=("joint_1",)),
        )
    )
    assert shared.trajectory_for("piper_0").joint_names == ("joint_1",)
    assert shared.trajectory_for("piper_1").joint_names == ("joint_1",)


def test_duplicate_joint_names_within_one_robot_are_rejected() -> None:
    """One robot may not declare the same joint twice in one trajectory."""
    with pytest.raises(ContractViolationError, match="duplicates"):
        trajectory(robot_id="piper_0", joint_names=("joint_1", "joint_1"))


def _single_participant_commit(**overrides: object) -> MotionCommit:
    """Build a one-participant commit payload with optional overrides.

    Args:
        **overrides: Field values replacing the valid defaults.

    Returns:
        The validated commit.
    """
    values: dict[str, object] = {
        "commit_id": "c",
        "identity": commit().identity,
        "session_id": "session-1",
        "participants": ("piper_0",),
        "epoch": 1,
        "world_revision": 7,
        "trajectory_revision": 1,
        "start_at_s": 0.0,
        "world_evidence_ids": ("ev-1",),
        "lease_ids": ("robot:piper_0",),
        "reservation_ids": ("res-1",),
        "trajectories": (trajectory(robot_id="piper_0"),),
    }
    values.update(overrides)
    return MotionCommit(**values)  # type: ignore[arg-type]


def test_versions_and_start_time_reject_bools_and_nonfinite() -> None:
    """Epochs and revisions are integers; times are finite non-negatives."""
    with pytest.raises(ContractViolationError, match="commit.epoch"):
        _single_participant_commit(epoch=True)
    for bad in (float("nan"), float("inf"), -1.0):
        with pytest.raises(ContractViolationError, match="start_at_s"):
            commit(start_at_s=bad)
    with pytest.raises(ContractViolationError, match="commit.world_revision"):
        _single_participant_commit(world_revision=True)


def test_sample_times_must_start_at_zero_and_strictly_increase() -> None:
    """Trajectory timing is unambiguous and monotonic."""
    with pytest.raises(ContractViolationError, match="first sample must start at 0.0"):
        trajectory(times=(0.5, 1.0))
    with pytest.raises(ContractViolationError, match="must strictly increase"):
        trajectory(times=(0.0, 0.5, 0.5))
    with pytest.raises(ContractViolationError, match="must strictly increase"):
        trajectory(times=(0.0, 0.0, 1.0))


def test_sample_shape_must_match_joint_names() -> None:
    """Position and velocity widths match the declared joint names."""
    with pytest.raises(ContractViolationError, match="positions must match"):
        JointTrajectory(
            robot_id="piper_0",
            joint_names=("a", "b"),
            samples=(JointSample(time_from_start_s=0.0, positions=(0.0,)),),
        )
    with pytest.raises(ContractViolationError, match="velocities must match"):
        JointTrajectory(
            robot_id="piper_0",
            joint_names=("a", "b"),
            samples=(
                JointSample(
                    time_from_start_s=0.0,
                    positions=(0.0, 0.0),
                    velocities=(0.0,),
                ),
            ),
        )


def test_leases_reservations_and_evidence_are_required() -> None:
    """A commit without lease, reservation or evidence pins is not valid."""
    for field_name in (
        "world_evidence_ids",
        "lease_ids",
        "reservation_ids",
    ):
        with pytest.raises(ContractViolationError, match=field_name):
            _single_participant_commit(**{field_name: ()})
