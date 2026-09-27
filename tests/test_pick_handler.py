"""Offline admission tests for the independent PICK handler."""

from __future__ import annotations

import asyncio

import pytest

from rocobrick.execution.operation_state_ledger import OperationStateLedger
from rocobrick.multi_robot_scheduler import OperationStatus
from rocobrick.robot_operation_plan import OperationTarget, RobotOperation
from rocobrick.skills.pick import PickHandler


class Observer:
    """Robot-scoped scene transport with a single measured document."""

    def __init__(self, document: dict) -> None:
        """Store the frozen test observation."""
        self.document = document
        self.calls: list[str] = []

    async def observe(self, robot_id: str) -> dict:
        """Return the scene while recording its requested robot.

        Returns:
            The supplied scene document.
        """
        self.calls.append(robot_id)
        return self.document


def operation() -> RobotOperation:
    """Build one formal PICK operation from its public contract.

    Returns:
        The operation targeting brick-a with piper_0.
    """
    return RobotOperation(
        operation_id="pick-1", source_task_id="task-1", skill="PICK",
        executor_robot_id="piper_0", participants=("piper_0",),
        role_bindings={"executor": "piper_0"},
        target=OperationTarget({"physical_instance_id": "brick-a"}),
        parameters={}, preconditions=("robot_at_pick_region",),
        effects=("holding:brick-a",),
        resources=("robot:piper_0", "part:brick-a"),
        evidence_requirements=("holding_confirmed",),
    )


def document() -> dict:
    """Return a co-sampled identified scene without holding evidence.

    Returns:
        The valid pose/target scene document.
    """
    return {
        "schema": "rocobrick-scene-state/v1",
        "session_id": "session-1", "manifest_sha256": "a" * 64,
        "sequence": 3, "sim_time_s": 3.0,
        "bricks": {"brick-a": {
            "aabb_min": [0.0, 0.0, 0.0],
            "aabb_max": [0.02, 0.02, 0.02],
            "pose": {"position_xyz": [0.0, 0.0, 0.01],
                     "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        }},
        "robots": {"piper_0": {
            "robot_id": "piper_0", "sequence": 3, "sim_time_s": 3.0,
            "opening_m": 0.02,
            "tcp_pose": {"position_xyz": [0.0, 0.0, 0.04],
                         "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        }},
    }


def test_pick_handler_checks_identity_then_fails_before_motion() -> None:
    """A valid arrival does not imply planning-scene capability or holding."""
    observer = Observer(document())
    ledger = OperationStateLedger(("piper_0", "piper_1"))
    ledger.confirm("piper_0", "robot_at_pick_region", {
        "arrival": True, "physical_instance_id": "brick-a"
    })
    handler = PickHandler(
        observer, ledger, session_id="session-1", manifest_sha256="a" * 64
    )
    result = asyncio.run(handler.execute(operation()))
    assert result.status is OperationStatus.FAILED
    assert result.terminal is True and result.motion_sent is False
    assert result.reason == "PICK_PLANNING_SCENE_READBACK_UNAVAILABLE"
    assert result.evidence["target_identity_checked"]["sequence"] == 3
    assert "holding_confirmed" not in result.evidence
    assert ledger.owner_of_object("brick-a") is None
    assert observer.calls == ["piper_0"]


@pytest.mark.parametrize("defect", ["foreign", "no_arrival", "owned", "wrong_target"])
def test_pick_handler_rejects_preflight_without_motion(defect: str) -> None:
    """Identity, arrival and exclusive object ownership are motion gates."""
    scene = document()
    observer = Observer(scene)
    ledger = OperationStateLedger(("piper_0", "piper_1"))
    if defect != "no_arrival":
        ledger.confirm("piper_0", "robot_at_pick_region", {
        "arrival": True, "physical_instance_id": "brick-a"
    })
    if defect == "foreign":
        scene["robots"]["piper_0"]["robot_id"] = "piper_1"
    if defect == "wrong_target":
        ledger.confirm("piper_0", "robot_at_pick_region", {
            "physical_instance_id": "brick-b"
        })
    if defect == "owned":
        ledger.confirm_holding("piper_1", "brick-a", {"holding_confirmed": True})
    handler = PickHandler(
        observer, ledger, session_id="session-1", manifest_sha256="a" * 64
    )
    result = asyncio.run(handler.execute(operation()))
    assert result.status is OperationStatus.FAILED
    assert result.motion_sent is False
    assert result.evidence == {}
    assert result.reason.startswith("PICK_PREFLIGHT_REJECTED")
    assert ledger.observe("piper_0", "holding:brick-a") is None
