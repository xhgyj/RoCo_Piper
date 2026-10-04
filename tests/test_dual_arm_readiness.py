"""Offline tests for combined dual-arm readiness evidence."""

import pytest

from rocobrick.coordinated_moveit import CoordinatedMoveItError
from rocobrick.dual_arm_readiness import require_group_readiness


def readiness() -> dict:
    """Return one passing combined MoveIt readiness document."""
    return {
        "coordination_id": "coord-1",
        "scene_digest": "scene-1",
        "scene_sequence": 42,
        "scene_sim_time_s": 12.5,
        "participants": ["piper_0", "piper_1"],
        "move_group_action": "/dual_arm/move_action",
        "planning_group": "dual_arm",
        "components": {"scene": True, "move_group": True, "joint_state": True},
    }


def test_group_readiness_requires_exact_identity_and_all_components() -> None:
    """A passing document is frozen with the requested coordination identity."""
    result = require_group_readiness(
        readiness(),
        coordination_id="coord-1",
        participants=("piper_0", "piper_1"),
        scene_digest="scene-1",
    )
    assert result.planning_group == "dual_arm"
    assert result.scene_sequence == 42


@pytest.mark.parametrize(
    "field,value,pattern",
    [
        ("coordination_id", "other", "coordination id"),
        ("scene_digest", "other", "scene digest"),
        ("participants", ["piper_1", "piper_0"], "participants"),
    ],
)
def test_group_readiness_rejects_identity_mismatch(field, value, pattern) -> None:
    """A coordinated plan cannot consume another coordination's readiness."""
    document = readiness()
    document[field] = value
    with pytest.raises(CoordinatedMoveItError, match=pattern):
        require_group_readiness(
            document,
            coordination_id="coord-1",
            participants=("piper_0", "piper_1"),
            scene_digest="scene-1",
        )


def test_group_readiness_rejects_any_failed_component() -> None:
    """Every component must pass before a coordinated plan can start."""
    document = readiness()
    document["components"]["scene"] = False
    with pytest.raises(CoordinatedMoveItError, match="components"):
        require_group_readiness(
            document,
            coordination_id="coord-1",
            participants=("piper_0", "piper_1"),
            scene_digest="scene-1",
        )
