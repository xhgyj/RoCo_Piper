"""Offline admission tests for the independent PICK handler."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from rocobrick.skills.target_context import GroundSpec

from rocobrick.execution.operation_state_ledger import OperationStateLedger
from rocobrick.isaac_backend import OperationKind, RuntimeOperation
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



class Motion:
    """Minimal runtime transport for phase normalization tests."""

    def run_motion_operation(self, robot_id: str, operation: RuntimeOperation) -> dict:
        return {
            "motion_sent": False,
            "result": {
                "passed": True,
                "execution": {
                    "readback_confirmed": True,
                },
                "kind": operation.kind.value,
            },
        }

    def cancel_motion_operation(self, operation_id: str) -> dict:
        return {"attempted": False, "terminal": False, "safe": False}


def test_planning_scene_phase_does_not_require_motion_safe_to_release() -> None:
    handler = PickHandler(
        Observer(document()),
        OperationStateLedger(("piper_0",)),
        session_id="session-1",
        manifest_sha256="a" * 64,
        motion_transport=Motion(),
    )
    runtime_operation = RuntimeOperation(
        kind=OperationKind.APPLY_PLANNING_SCENE,
        payload={},
        timeout_s=1.0,
        request_id="pick-1:plan_grasp",
    )
    result = asyncio.run(handler._dispatch(operation(), "plan_grasp", runtime_operation))
    assert result.passed is True
    assert result.terminal is True and result.safe is True
    assert result.motion_sent is False


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


def test_pick_planning_scene_includes_protected_ground_box() -> None:
    """The floor stays protected for every robot link, with no ACM waiver.

    The calibrated distal-pad TCP keeps the tool clear of the floor at the
    approach target, so the ground box is a hard collision constraint: an
    unreachable target must fail planning instead of being authorized to
    graze the floor and then stall against it in Isaac.
    """
    ledger = OperationStateLedger(("piper_0",))
    ledger.confirm(
        "piper_0", "robot_at_pick_region",
        {"arrival": True, "physical_instance_id": "brick-a"},
    )
    handler = PickHandler(
        Observer(document()), ledger, session_id="session-1",
        manifest_sha256="a" * 64,
    )
    admission = asyncio.run(handler.preflight(operation()))
    context = SimpleNamespace(ground=GroundSpec(
        center_xy_m=(0.0, -0.2), size_xy_m=(2.0, 2.0),
        top_z_m=-0.001, thickness_m=0.02,
    ))
    payload = handler._planning_payload(admission, context)
    boxes = {box["object_id"]: box for box in payload["boxes"]}
    assert set(boxes) == {"brick-a", "piper_0_ground"}
    ground = boxes["piper_0_ground"]
    assert ground["frame_id"] == "dual_world"
    assert ground["minimum_m"] == (-1.0, -1.2, -0.021)
    assert ground["maximum_m"] == (1.0, 0.8, -0.001)
    assert payload["allowed_collision_pairs"] == ()
    assert set(map(tuple, payload["forbidden_collision_pairs"])) == {
        ("brick-a", "piper_0_gripper_link1"),
        ("brick-a", "piper_0_gripper_link2"),
    }
    contact = handler._planning_payload(admission, context, allow_contact=True)
    assert contact["forbidden_collision_pairs"] == ()
    assert set(map(tuple, contact["allowed_collision_pairs"])) == set(
        map(tuple, payload["forbidden_collision_pairs"])
    )


def test_dispatch_preserves_nested_motion_evidence() -> None:
    """Outer false dispatch metadata must not hide actual command evidence."""
    class MotionWithNestedEvidence(Motion):
        def run_motion_operation(self, robot_id, runtime_operation):
            return {
                "motion_sent": False,
                "result": {
                    "passed": True,
                    "execution": {
                        "terminal": True, "safe_to_release": True,
                        "command_messages": 12,
                    },
                    "evidence": {"motion_sent": True},
                },
            }

    handler = PickHandler(
        Observer(document()), OperationStateLedger(("piper_0",)),
        session_id="session-1", manifest_sha256="a" * 64,
        motion_transport=MotionWithNestedEvidence(),
    )
    request = RuntimeOperation(
        kind=OperationKind.TCP_POSE, payload={}, timeout_s=1.0,
        request_id="pick-1:approach",
    )
    result = asyncio.run(handler._dispatch(operation(), "approach", request))
    assert result.passed is True
    assert result.motion_sent is True


@pytest.mark.parametrize("span", [0.02, 0.079])
def test_pick_wider_opening_keeps_one_unified_approach(span) -> None:
    """Approach opens 10 mm wider, within travel, with no side waypoint."""
    from rocobrick.skills.targets import RobotBaseFrame, SkillParams, TcpProvenance
    from rocobrick.tcp_contract import EXPECTED_RUNTIME_MODEL_SHA256

    scene = document()
    scene["bricks"]["brick-a"]["aabb_max"] = [span, span, 0.02]
    ledger = OperationStateLedger(("piper_0",))
    ledger.confirm("piper_0", "robot_at_pick_region", {
        "physical_instance_id": "brick-a",
    })

    class FreeSpace:
        robot_ids = ("piper_0",)

        async def free_space_context(self, participants):
            return {}, 1, {}

    handler = PickHandler(
        Observer(scene), ledger, session_id="session-1",
        manifest_sha256="a" * 64, motion_transport=Motion(),
        free_space_transport=FreeSpace(),
        context=SimpleNamespace(
            base=RobotBaseFrame((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            provenance=TcpProvenance(EXPECTED_RUNTIME_MODEL_SHA256, "b" * 64),
            params=SkillParams(), ground=None,
        ),
    )
    sent = []
    phases = []

    async def dispatch(operation, phase, request):
        sent.append((phase, request))
        return SimpleNamespace(
            passed=True, terminal=True, safe=True, motion_sent=False,
            evidence={}, reason=None,
        )

    async def approach(operation, **kwargs):
        phases.append(kwargs["phase_id"])
        raise RuntimeError("stop before executing approach")

    handler._dispatch = dispatch
    handler.execute_free_space_phase = approach
    result = asyncio.run(handler.execute(operation()))
    assert result.passed is False
    assert [phase for phase, _ in sent] == ["plan_grasp", "open_gripper"], result.reason
    opening = sent[-1][1].payload
    expected = min(0.1, span + 0.03)
    assert opening["opening_m"] == pytest.approx(expected)
    assert opening["expected_opening_m"] == pytest.approx((
        expected - 0.01, min(0.1, expected + 0.01),
    ))
    assert phases == ["precontact_boundary"]
    assert "approach_corridor" not in result.evidence


def test_pick_enables_contact_only_after_unified_arrival():
    from rocobrick.skills.targets import RobotBaseFrame, SkillParams, TcpProvenance
    from rocobrick.tcp_contract import EXPECTED_RUNTIME_MODEL_SHA256

    ledger = OperationStateLedger(("piper_0",))
    ledger.confirm("piper_0", "robot_at_pick_region", {"physical_instance_id": "brick-a"})

    class FreeSpace:
        robot_ids = ("piper_0",)

        async def free_space_context(self, participants):
            return {}, 1, {}

    handler = PickHandler(
        Observer(document()), ledger, session_id="session-1", manifest_sha256="a" * 64,
        motion_transport=Motion(), free_space_transport=FreeSpace(),
        context=SimpleNamespace(
            base=RobotBaseFrame((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            provenance=TcpProvenance(EXPECTED_RUNTIME_MODEL_SHA256, "b" * 64),
            params=SkillParams(), ground=None,
        ),
    )
    phases = []
    scenes = {}

    async def dispatch(operation, phase, request):
        phases.append(phase)
        if phase in ("plan_grasp", "allow_grasp_contact"):
            scenes[phase] = request.payload
        return SimpleNamespace(
            passed=phase != "close_gripper", terminal=True, safe=True,
            motion_sent=False, evidence={}, reason="stop at closure",
        )

    async def approach(operation, **kwargs):
        phases.append(kwargs["phase_id"])
        return SimpleNamespace(
            planned=True, prepared=True, started=True, arrival_confirmed=True,
            terminal=True, safe_to_release=True, evidence={},
            motion_id="pick-1:precontact_boundary", participants=("piper_0",),
            reason=None,
        )

    handler._dispatch = dispatch
    handler.execute_free_space_phase = approach
    result = asyncio.run(handler.execute(operation()))
    assert phases == ["plan_grasp", "open_gripper", "precontact_boundary",
                      "allow_grasp_contact", "close_gripper"]
    assert scenes["plan_grasp"]["allowed_collision_pairs"] == ()
    assert len(scenes["plan_grasp"]["forbidden_collision_pairs"]) == 2
    assert scenes["allow_grasp_contact"]["forbidden_collision_pairs"] == ()
    assert len(scenes["allow_grasp_contact"]["allowed_collision_pairs"]) == 2
    assert "preclose_observation" in result.evidence
