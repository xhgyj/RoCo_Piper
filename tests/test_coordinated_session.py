"""Tests for the dedicated coordinated MoveIt session envelope."""

import pytest

from rocobrick.coordinated_moveit import CoordinatedMoveItError
from rocobrick.coordinated_session import (
    COORDINATED_REQUEST_SCHEMA,
    CoordinatedSessionRequest,
    parse_coordinated_request,
)


def request() -> CoordinatedSessionRequest:
    """Return one valid planning-only request."""
    return CoordinatedSessionRequest(
        coordination_id="coord-1",
        robots=("piper_0", "piper_1"),
        scene_digest="scene-1",
        planning_group="dual_arm",
        plan_only=True,
        timeout_s=30.0,
        targets={"piper_0": {"kind": "pre_support"}, "piper_1": {"kind": "pre_place"}},
    )


def test_coordinated_request_round_trips_without_single_robot_operations() -> None:
    """The wire shape carries one group identity, not two fake operations."""
    document = request().serialize()
    assert document["schema"] == COORDINATED_REQUEST_SCHEMA
    assert "operations" not in document
    parsed = parse_coordinated_request(document)
    assert parsed.planning_group == "dual_arm"
    assert parsed.plan_only is True


def test_coordinated_request_rejects_missing_participant_target() -> None:
    """Every participant needs an explicit pre-contact target."""
    document = request().serialize()
    document["targets"].pop("piper_1")
    with pytest.raises(CoordinatedMoveItError, match="cover"):
        parse_coordinated_request(document)


def test_coordinated_request_rejects_execution_without_positive_timeout() -> None:
    """Planning/execution requests cannot use an unbounded timeout."""
    document = request().serialize()
    document["timeout_s"] = 0
    with pytest.raises(CoordinatedMoveItError, match="timeout"):
        parse_coordinated_request(document)


def test_coordinated_result_round_trips_strictly() -> None:
    """The combined result preserves readiness, trajectory and lifecycle evidence."""
    from rocobrick.coordinated_session import CoordinatedSessionResult

    result = CoordinatedSessionResult(
        coordination_id="coord-1",
        robots=("piper_0", "piper_1"),
        readiness={"scene": True},
        trajectory={"joint_names": ["piper_0_joint1"]},
        lifecycle={"plan_only": True},
        passed=True,
    )
    assert CoordinatedSessionResult.parse(result.serialize()) == result

