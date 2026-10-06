"""Strict session-level planning-scene transfer contract tests."""

from dataclasses import replace

import pytest

from rocobrick.isaac_backend import (
    OperationContractError,
    OperationKind,
    transfer_attached_planning_scene_operation,
)
from rocobrick.planning_scene import box_geometry_digest


def _payload() -> dict[str, object]:
    minimum = (-0.01, -0.02, -0.03)
    maximum = (0.01, 0.02, 0.03)
    return {
        "transfer_id": "handover-1-transfer",
        "object_id": "brick-1",
        "donor_robot_id": "piper_0",
        "donor_link_name": "piper_0_rocobrick_tcp",
        "receiver_robot_id": "piper_1",
        "receiver_link_name": "piper_1_rocobrick_tcp",
        "frame_id": "dual_world",
        "minimum_m": list(minimum),
        "maximum_m": list(maximum),
        "geometry_digest": box_geometry_digest(
            object_id="brick-1",
            frame_id="dual_world",
            minimum_m=minimum,
            maximum_m=maximum,
        ),
        "touch_links": ["piper_1_gripper_link1", "piper_1_gripper_link2"],
        "expected_before": {
            "donor_attached": True,
            "receiver_attached": False,
        },
    }


def test_transfer_operation_has_one_explicit_identity() -> None:
    """A valid transfer is one runtime operation, not two independent calls."""
    operation = transfer_attached_planning_scene_operation(
        _payload(), 30.0, "handover-1-transfer-request"
    )

    assert operation.kind is OperationKind.TRANSFER_ATTACHED_PLANNING_SCENE
    assert operation.payload["transfer_id"] == "handover-1-transfer"
    assert operation.serialize()["kind"] == "TRANSFER_ATTACHED_PLANNING_SCENE"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("geometry_digest", "0" * 64, "geometry_digest"),
        ("donor_robot_id", "piper_1", "differ"),
        (
            "expected_before",
            {"donor_attached": False, "receiver_attached": False},
            "expected_before",
        ),
        ("touch_links", ["piper_1_gripper_link1", "piper_1_gripper_link1"], "unique"),
    ],
)
def test_transfer_operation_rejects_tampered_identity(
    field: str, value: object, message: str
) -> None:
    """A transfer cannot hide geometry, role or before-state mismatches."""
    payload = _payload()
    payload[field] = value
    with pytest.raises(OperationContractError, match=message):
        transfer_attached_planning_scene_operation(payload, 30.0, "request")


def test_transfer_operation_rejects_unknown_payload_keys() -> None:
    """The transfer wire shape cannot grow implicit side channels."""
    payload = _payload()
    payload["rollback"] = True
    with pytest.raises(OperationContractError, match="exactly"):
        transfer_attached_planning_scene_operation(payload, 30.0, "request")


def test_transfer_operation_result_binding_is_independent_of_payload_copy() -> None:
    """The request id remains distinct from the physical transfer id."""
    operation = transfer_attached_planning_scene_operation(
        _payload(), 30.0, "request-a"
    )
    copied = replace(operation, request_id="request-b")
    assert copied.request_id == "request-b"
    assert copied.payload["transfer_id"] == "handover-1-transfer"
