"""Strict MoveIt detach/readback regressions; no simulator or ROS launch."""

from types import SimpleNamespace

import pytest
from test_r5_tcp_moveit import make_world

from rocobrick.isaac_backend import (
    OperationContractError,
    detached_planning_scene_operation,
)


def payload():
    """Return one robot-scoped detach payload.

    Returns:
        The strict collision-object removal claim.
    """
    return {
        "scene_id": "detach",
        "object_id": "brick",
        "link_name": "piper_0_rocobrick_tcp",
        "operation": "detach",
    }


@pytest.mark.parametrize("remaining", [False, True])
def test_detach_requires_absence_in_attached_object_readback(remaining):
    """Service success cannot substitute for detached-object absence."""
    world = make_world()
    apply = world.runtime._ensure_planning_scene_client()
    apply.response = SimpleNamespace(success=True)
    get = world.runtime._ensure_get_planning_scene_client()
    get.response = SimpleNamespace(
        scene=SimpleNamespace(
            robot_state=SimpleNamespace(
                attached_collision_objects=[
                    SimpleNamespace(object=SimpleNamespace(id="brick"))
                ]
                if remaining
                else []
            )
        )
    )
    result = world.runtime.detach_attached_planning_scene(payload(), 1)
    assert result["passed"] is (not remaining)
    assert result["readback_confirmed"] is (not remaining)
    request = apply.requests[0]
    assert request.scene.robot_state.is_diff is True
    assert request.scene.robot_state.attached_collision_objects[0].object.operation == 1


def test_detach_payload_rejects_legacy_or_extra_keys():
    """Removal is exact-schema validated before any service call."""
    op = detached_planning_scene_operation(payload(), 1, "request")
    assert op.payload["operation"] == "detach"
    bad = {**payload(), "frame_id": "world"}
    with pytest.raises(OperationContractError):
        detached_planning_scene_operation(bad, 1, "request")
