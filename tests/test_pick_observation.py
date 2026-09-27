"""Offline checks for PICK identity and relative-pose evidence."""

from __future__ import annotations

from copy import deepcopy

import pytest

from rocobrick.skills.observation import (
    parse_pick_observation,
    verify_pick_following,
)


def scene(z: float, sequence: int) -> dict[str, object]:
    """One robot/object scene reading at a simulated step.

    Returns:
        A scene document with one mapped object and robot.
    """
    return {
        "schema": "rocobrick-scene-state/v1",
        "session_id": "session-1",
        "manifest_sha256": "a" * 64,
        "sequence": sequence,
        "sim_time_s": float(sequence),
        "bricks": {"brick-a": {
            "aabb_min": [0.0, 0.0, z],
            "aabb_max": [0.02, 0.02, z + 0.01],
            "pose": {"position_xyz": [0.0, 0.0, z],
                     "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        }},
        "robots": {"piper_0": {
            "robot_id": "piper_0", "sim_time_s": float(sequence),
            "sequence": sequence, "opening_m": 0.019,
            "tcp_pose": {"position_xyz": [0.0, 0.0, z + 0.02],
                         "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        }},
    }


def parse(document: dict[str, object]):
    """Bind a scene to its expected robot, instance and provenance.

    Returns:
        The parsed physical observation.
    """
    return parse_pick_observation(
        document, robot_id="piper_0", physical_instance_id="brick-a",
        session_id="session-1", manifest_sha256="a" * 64,
    )


def verify(documents):
    """Apply the PICK-specific following envelope.

    Returns:
        The validated relative-pose evidence summary.
    """
    return verify_pick_following(
        [parse(document) for document in documents],
        position_tolerance_m=0.005,
        orientation_tolerance_rad=0.03,
        minimum_lift_m=0.05,
    )


def test_following_requires_synchronized_lift_and_relative_pose() -> None:
    """Height and a stable object/TCP transform are separate evidence."""
    result = verify([scene(0.0, 1), scene(0.05, 2), scene(0.1, 3)])
    assert result["samples"] == 3
    assert result["lift_m"] == pytest.approx(0.1)
    assert result["max_relative_position_error_m"] == pytest.approx(0.0)
    assert "holding_confirmed" not in result


def test_height_change_without_tcp_following_fails() -> None:
    """An independently raised brick cannot establish holding."""
    documents = [scene(0.0, 1), scene(0.05, 2), scene(0.1, 3)]
    documents[-1]["robots"]["piper_0"]["tcp_pose"]["position_xyz"][2] = 0.02
    with pytest.raises(ValueError, match="slipped"):
        verify(documents)


def test_relative_orientation_slip_fails() -> None:
    """A lifted object rotating against TCP is not a confirmed hold."""
    documents = [scene(0.0, 1), scene(0.05, 2), scene(0.1, 3)]
    documents[-1]["bricks"]["brick-a"]["pose"]["orientation_xyzw"] = [
        0.0, 0.0, 0.1, 0.99498743710662
    ]
    with pytest.raises(ValueError, match="slipped"):
        verify(documents)


def test_stale_and_foreign_observations_fail() -> None:
    """Sequence, robot identity and manifest provenance are mandatory."""
    documents = [scene(0.0, 1), scene(0.05, 2), scene(0.1, 3)]
    stale = deepcopy(documents)
    stale[-1]["sequence"] = 2
    stale[-1]["robots"]["piper_0"]["sequence"] = 2
    with pytest.raises(ValueError, match="stale"):
        verify(stale)
    foreign = deepcopy(documents[0])
    foreign["robots"]["piper_0"]["robot_id"] = "piper_1"
    with pytest.raises(ValueError, match="foreign"):
        parse(foreign)
    foreign = deepcopy(documents[0])
    foreign["manifest_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="provenance"):
        parse(foreign)


def test_missing_pose_or_opening_cannot_fallback_to_aabb() -> None:
    """AABB and measured opening alone cannot replace rigid-body poses."""
    document = scene(0.0, 1)
    del document["bricks"]["brick-a"]["pose"]
    with pytest.raises(ValueError, match="rigid pose"):
        parse(document)
    document = scene(0.0, 1)
    document["robots"]["piper_0"]["opening_m"] = None
    with pytest.raises(ValueError, match="opening"):
        parse(document)
