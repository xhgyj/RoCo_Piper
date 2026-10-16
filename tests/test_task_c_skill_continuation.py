"""Task C thick-brick execution contracts, without ROS or physical motion."""

import asyncio
import copy
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
from pre_phase_helpers import request, success
from test_pick_handler import document as pick_document
from test_place_down_handler import context, document, operation

from rocobrick.coordinated_session import (
    CoordinatedSessionResult,
    FreeSpaceMotionEvidence,
)
from rocobrick.execution.operation_state_ledger import OperationStateLedger
from rocobrick.robot_operation_plan import OperationTarget
from rocobrick.skills.contact import allows_recovery, inspect_contact
from rocobrick.skills.observation import parse_pick_observation
from rocobrick.skills.place_up import PlaceUpHandler
from rocobrick.skills.placement_geometry import (
    place_up_pick_targets,
    support_top_target,
)
from rocobrick.skills.pre_phase_targets import derive_pre_phase_targets
from rocobrick.skills.support_evidence import verify_support_contact
from rocobrick.skills.support_top import SupportTopHandler
from rocobrick.skills.targets import (
    RobotBaseFrame,
    SkillParams,
    _quat_rotate_vector,
    pick_targets,
)
from rocobrick.unified_runtime_facade import UnifiedDualArmRuntimeFacade


def upward_operation():
    """Return the carrier/support roles in the existing external plan."""
    base = operation()
    return replace(
        base,
        skill="PLACE_UP",
        participants=("piper_0", "piper_1"),
        target=OperationTarget({**base.target.values, "support_robot_id": "piper_1"}),
        preconditions=(),
    )


def support_operation():
    """Return the supporting role without claiming the carrier's lower part."""
    return replace(
        upward_operation(),
        operation_id="support-1",
        skill="SUPPORT_TOP",
        executor_robot_id="piper_1",
        participants=("piper_1",),
        target=OperationTarget(
            {
                key: value
                for key, value in {
                    **upward_operation().target.values,
                    "placement_robot_id": "piper_0",
                }.items()
                if key != "support_robot_id"
            }
        ),
    )


class Scene:
    """Advance one shared physics tick and retain explicit contact data."""

    def __init__(self):
        """Create measured geometry and two independently identified robots."""
        self.data = document(sequence=4, upper_z=0.10, tcp_z=0.0368, connection=False)
        self.data["scene_digest"] = "c" * 64
        self.data["ground"] = {"aabb_min": [-1, -1, -0.02], "aabb_max": [1, 1, 0]}
        self.data["robots"]["piper_1"] = {
            **copy.deepcopy(self.data["robots"]["piper_0"]),
            "robot_id": "piper_1",
        }
        self.contacts = True

    async def observe(self, robot):
        """Return an advancing, co-sampled document."""
        self.data["sequence"] += 1
        self.data["sim_time_s"] += 1
        for block in self.data["robots"].values():
            block.update(
                sequence=self.data["sequence"], sim_time_s=self.data["sim_time_s"]
            )
        self.data["pad_object_contacts"] = {
            "schema": "rocobrick-pad-object-contacts/v1",
            "source": "physx_contact_report",
            **{
                key: self.data[key]
                for key in ("session_id", "manifest_sha256", "sequence", "sim_time_s")
            },
            "contacts": [
                {
                    "robot_id": "piper_1",
                    "pad_link": "piper_1_gripper_link1",
                    "physical_instance_id": "brick-upper",
                    "position_world_m": [0.32, 0.32, 0.1112],
                    "normal_world": [0, 0, 1],
                    "normal_force_n": 0.5,
                }
            ]
            if self.contacts
            else [],
        }
        return copy.deepcopy(self.data)


class FreeSpace:
    """Require one combined request, holding the support participant fixed."""

    robot_ids = ("piper_0", "piper_1")

    def __init__(self, scene):
        """Bind the fake physical scene, not a ROS node."""
        self.scene = scene
        self.calls = []

    async def free_space_context(self, participants):
        """Read an advancing scene and both start-state identities.

        Returns:
            The synthetic shared scene, version and measured joints.
        """
        return (
            await self.scene.observe("piper_0"),
            1,
            {"piper_0_joint1": 0.0, "piper_1_joint1": 0.0},
        )

    async def execute_free_space(self, item):
        """Apply only moving participants' targets in the synthetic scene.

        Returns:
            Scripted lifecycle proof, not actual robotics validation.
        """
        self.calls.append(item)
        for robot, target in item.targets.items():
            if target.get("maintain_pose"):
                continue
            self.scene.data["robots"][robot]["tcp_pose"] = {
                "position_xyz": list(target["tcp_pose"]["position"]),
                "orientation_xyzw": list(target["tcp_pose"]["orientation_xyzw"]),
            }
        return FreeSpaceMotionEvidence(
            item.motion_id,
            item.participants,
            True,
            True,
            True,
            True,
            True,
            True,
            {robot: {"arrival_confirmed": True} for robot in item.participants},
            {"physical_execution": True},
        )


class Motion:
    """Model exact connection cancellation, real opening, and planning detach."""

    def __init__(self, scene):
        """Bind phase evidence to the same synthetic physical scene."""
        self.scene = scene
        self.calls = []

    def run_motion_operation(self, robot, item):
        """Model observed connection, opening and safe action termination.

        Returns:
            Synthetic action evidence for the phase under test.
        """
        self.calls.append(item)
        stop = None
        if item.request_id.endswith(":insert_up"):
            guard = item.payload["guard"]
            self.scene.data["bricks"]["brick-lower"]["pose"] = copy.deepcopy(
                guard["desired_lower"]
            )
            for key, value in self.scene.data["connections"].items():
                self.scene.data["connections"][key] = True
            pair = next(iter(self.scene.data["placement_interfaces"]["pairs"].values()))
            pair["actual_connection"] = {
                "stud_interface": pair["stud_interface"],
                "hole_interface": pair["hole_interface"],
                "lower_to_upper": pair["lower_to_upper"],
            }
            tcp = self.scene.data["robots"][robot]["tcp_pose"]
            tcp["position_xyz"][2] = (
                guard["desired_lower"]["position_xyz"][2]
                + guard["held_transform"]["position_xyz"][2]
            )
            stop = "connection_confirmed"
        if item.request_id.endswith(":release"):
            self.scene.data["robots"][robot]["opening_m"] = item.payload["opening_m"]
        return {
            "motion_sent": item.request_id.endswith(
                (":insert_up", ":release", ":regrip")
            ),
            "result": {
                "passed": True,
                "execution": {
                    "terminal": True,
                    "safe_to_release": True,
                    "details": {"contact_stop_reason": stop},
                },
            },
        }


def test_place_up_pick_keeps_studs_open_without_changing_ordinary_pick():
    """A successor intent selects a horizontal tool and a horizontal closing axis."""
    shared = pick_document()
    sample = parse_pick_observation(
        shared,
        robot_id="piper_0",
        physical_instance_id="brick-a",
        session_id="session-1",
        manifest_sha256="a" * 64,
    )
    ctx = context()
    side = place_up_pick_targets(sample, ctx.base, ctx.provenance, ctx.params)
    ordinary = pick_targets(
        sample.aabb_min, sample.aabb_max, ctx.base, ctx.provenance, ctx.params
    )
    assert (
        abs(
            _quat_rotate_vector(side.precontact_boundary.orientation_xyzw, (0, 0, 1))[2]
        )
        < 1e-10
    )
    assert (
        abs(
            _quat_rotate_vector(side.precontact_boundary.orientation_xyzw, (0, 1, 0))[2]
        )
        < 1e-10
    )
    assert (
        _quat_rotate_vector(ordinary.precontact_boundary.orientation_xyzw, (0, 0, 1))[2]
        == -1
    )
    assert side.close_opening_m == ordinary.close_opening_m


def test_support_targets_are_downward_and_separate_approach_from_contact():
    """A brick quaternion is not itself a downward tool quaternion."""
    scene = Scene()
    shared = asyncio.run(scene.observe("piper_1"))
    sample = parse_pick_observation(
        shared,
        robot_id="piper_1",
        physical_instance_id="brick-upper",
        session_id="session-1",
        manifest_sha256="a" * 64,
    )
    ctx = context()
    approach = support_top_target(
        sample, ctx.base, ctx.provenance, ctx.params, approach=True
    )
    contact = support_top_target(
        sample, ctx.base, ctx.provenance, ctx.params, approach=False
    )
    assert _quat_rotate_vector(contact.orientation_xyzw, (0, 0, 1))[2] == -1
    assert contact.position[2] == pytest.approx(
        sample.aabb_max[2] + 0.005211 - ctx.params.support_press_m
    )
    assert approach.position[2] > contact.position[2]
    targets = derive_pre_phase_targets(
        support_operation(), shared, {robot: ctx for robot in ("piper_0", "piper_1")}
    )
    assert targets["piper_0"].get("maintain_pose") is not True
    assert targets["piper_1"]["tcp_pose"]["position"] == approach.position


@pytest.mark.parametrize("base_yaw", [0.0, math.pi / 2])
def test_support_lean_preserves_distal_surface_point_and_normal_gap(base_yaw):
    """A base rotation cannot change the selected world support geometry."""
    scene = Scene()
    shared = asyncio.run(scene.observe("piper_1"))
    sample = parse_pick_observation(
        shared,
        robot_id="piper_1",
        physical_instance_id="brick-upper",
        session_id="session-1",
        manifest_sha256="a" * 64,
    )
    ctx = context()
    base = RobotBaseFrame(
        (-0.35, -0.3, 0.0),
        (0.0, 0.0, math.sin(base_yaw / 2), math.cos(base_yaw / 2)),
    )
    params = replace(ctx.params, support_radial_tilt_deg=5.0)
    targets = [
        support_top_target(sample, base, ctx.provenance, params, approach=approach)
        for approach in (True, False)
    ]
    assert targets[0].orientation_xyzw == targets[1].orientation_xyzw
    for target, gap in zip(
        targets, (params.support_approach_gap_m, -params.support_press_m)
    ):
        tool_axis_base = _quat_rotate_vector(target.orientation_xyzw, (0, 0, 1))
        distal_base = tuple(
            target.position[i] + tool_axis_base[i] * 0.005211 for i in range(3)
        )
        distal_rotated = _quat_rotate_vector(base.orientation_xyzw, distal_base)
        distal_world = tuple(base.position_xyz[i] + distal_rotated[i] for i in range(3))
        assert distal_world == pytest.approx(
            (
                (sample.aabb_min[0] + sample.aabb_max[0]) / 2,
                (sample.aabb_min[1] + sample.aabb_max[1]) / 2,
                sample.aabb_max[2] + gap,
            )
        )
        tool_axis_world = _quat_rotate_vector(base.orientation_xyzw, tool_axis_base)
        assert tool_axis_world[2] == pytest.approx(-math.cos(math.radians(5)))
        radial = tuple(distal_world[i] - base.position_xyz[i] for i in range(2))
        assert sum(tool_axis_world[i] * radial[i] for i in range(2)) > 0


def test_support_bias_stays_inside_the_long_top_face():
    """The support reference reserves pad separation and an eight-mm edge margin."""
    scene = Scene()
    shared = asyncio.run(scene.observe("piper_1"))
    sample = parse_pick_observation(
        shared,
        robot_id="piper_1",
        physical_instance_id="brick-upper",
        session_id="session-1",
        manifest_sha256="a" * 64,
    )
    ctx = context()
    base = replace(ctx.base, position_xyz=(-0.35, -0.3, 0.0))
    params = replace(ctx.params, support_surface_bias_m=0.02)
    target = support_top_target(sample, base, ctx.provenance, params, approach=True)
    rotated = _quat_rotate_vector(base.orientation_xyzw, target.position)
    world = tuple(base.position_xyz[i] + rotated[i] for i in range(3))
    assert sample.aabb_min[0] + 0.008 <= world[0] <= sample.aabb_max[0] - 0.008
    assert sample.aabb_min[1] <= world[1] <= sample.aabb_max[1]


@pytest.mark.parametrize("angle", [-1.0, 10.01, float("nan"), True])
def test_support_lean_rejects_unbounded_or_non_numeric_angles(angle):
    """The support tilt is bounded rather than a free orientation tolerance."""
    with pytest.raises(ValueError):
        SkillParams(support_radial_tilt_deg=angle)


@pytest.mark.parametrize("defect", ["no_contact", "stale", "side_contact", "foreign"])
def test_stability_does_not_substitute_real_top_support(defect):
    """Unchanged upper geometry alone never establishes support."""
    scene = Scene()
    shared = asyncio.run(scene.observe("piper_1"))
    report = shared["pad_object_contacts"]
    if defect == "no_contact":
        report["contacts"] = []
    elif defect == "stale":
        report["sequence"] -= 1
    elif defect == "side_contact":
        report["contacts"][0]["normal_world"] = [1, 0, 0]
    else:
        report["contacts"][0]["robot_id"] = "piper_0"
    with pytest.raises(ValueError):
        verify_support_contact(
            shared,
            robot_id="piper_1",
            upper_id="brick-upper",
            session_id="session-1",
            manifest_sha256="a" * 64,
        )


def test_support_consumes_successor_receipt_and_only_executes_contact():
    """Public-space motion is not dispatched twice."""
    scene = Scene()
    ledger = OperationStateLedger(("piper_0", "piper_1"))
    free = FreeSpace(scene)
    handler = SupportTopHandler(
        scene,
        ledger,
        session_id="session-1",
        manifest_sha256="a" * 64,
        motion_transport=Motion(scene),
        free_space_transport=free,
        context=context(),
        poll_interval_s=0,
    )
    item = support_operation()
    receipt = replace(success(request()), successor_operation_ids=(item.operation_id,))
    handler.bind_pre_phase(item.operation_id, receipt)
    result = asyncio.run(handler.execute(item))
    assert result.passed, result.reason
    assert free.calls == []
    assert result.evidence["support_pose"]["request_id"].endswith(":support_contact")
    assert "support_pose_confirmed" in result.evidence
    assert ledger.owner_of_object("brick-upper") is None


def test_place_up_requires_real_release_and_retreat_before_confirmation():
    """The two missing transport methods are replaced by the existing facade."""
    scene = Scene()
    ledger = OperationStateLedger(("piper_0", "piper_1"))
    ledger.confirm_holding("piper_0", "brick-lower", {"holding_confirmed": True})
    ledger.confirm("piper_1", "supporting:brick-upper", {"confirmed": True})
    free, motion = FreeSpace(scene), Motion(scene)
    handler = PlaceUpHandler(
        scene,
        ledger,
        session_id="session-1",
        manifest_sha256="a" * 64,
        motion_transport=motion,
        free_space_transport=free,
        context=context(),
        poll_interval_s=0,
    )
    result = asyncio.run(handler.execute(upward_operation()))
    assert result.passed, result.reason
    assert [call.motion_id.rsplit(":", 1)[1] for call in free.calls] == [
        "pre_place_up",
        "retreat_up",
    ]
    assert free.calls[-1].targets["piper_1"]["maintain_pose"] is True
    assert (
        next(
            call for call in motion.calls if call.request_id.endswith(":insert_up")
        ).timeout_s
        == 120
    )
    assert set(
        ("connection_confirmed", "release_confirmed", "retreat_confirmed")
    ) <= set(result.evidence)
    assert "retreat_deferred" not in result.evidence
    assert ledger.owner_of_object("brick-lower") is None
    assert ledger.observe("piper_1", "supporting:brick-upper") is None
    assert ledger.observe("piper_0", "placed:brick-lower") is not None


def test_insert_guard_reads_the_shared_scene_not_the_executor_slice():
    """The 13j regression: the peer's support proof needs a dual-robot read.

    Live 2026-10-14 (``runs/task-c-top-offset13j-headless-*``): the support was
    pressing at 5.21 N on piper_1's own slice, yet PLACE_UP stopped before any
    insertion motion with ``support_contact_lost``.  The up-mode contact guard
    proves the SUPPORT robot's pad reaction, but it was evaluated against the
    executor's robot-scoped observation, which by design carries only the
    executor's own contact records.  The guard now reads the shared multi-robot
    scene; this test drives the whole pre-insert path with a sliced observation
    transport and asserts the insertion actually runs.
    """
    scene = Scene()
    ledger = OperationStateLedger(("piper_0", "piper_1"))
    ledger.confirm_holding("piper_0", "brick-lower", {"holding_confirmed": True})
    ledger.confirm("piper_1", "supporting:brick-upper", {"confirmed": True})

    class SlicedObservations:
        """One robot-scoped observation transport (this robot's own contacts)."""

        async def observe(self, robot_id):
            """Return one robot-scoped answer, as the A5 envelope slices it."""
            document = await scene.observe(robot_id)
            document["pad_object_contacts"]["contacts"] = [
                entry
                for entry in document["pad_object_contacts"]["contacts"]
                if entry["robot_id"] == robot_id
            ]
            return document

    free, motion = FreeSpace(scene), Motion(scene)
    handler = PlaceUpHandler(
        SlicedObservations(),
        ledger,
        session_id="session-1",
        manifest_sha256="a" * 64,
        motion_transport=motion,
        free_space_transport=free,
        context=context(),
        poll_interval_s=0,
    )
    result = asyncio.run(handler.execute(upward_operation()))
    assert result.passed, result.reason
    assert "insertion_up" in result.evidence
    assert any(call.request_id.endswith(":insert_up") for call in motion.calls)

    # Control: the same proof fails on the executor's own slice, which is why
    # the guard must not be evaluated against it.
    sliced = asyncio.run(SlicedObservations().observe("piper_0"))
    assert sliced["pad_object_contacts"]["contacts"] == []
    with pytest.raises(ValueError, match="no measured top-surface contact"):
        verify_support_contact(
            sliced,
            robot_id="piper_1",
            upper_id="brick-upper",
            session_id="session-1",
            manifest_sha256="a" * 64,
        )


def test_place_up_regrips_the_held_brick_before_the_insertion():
    """The 13l/13p regression: the insertion's grip is stiffened first.

    Live 2026-10-14 (``runs/task-c-top-offset13l/13p-headless-*``): the upward
    insertion carries its whole load through pad friction, and the support's
    press transfers to the held brick at the stud engagement, so the brick slid
    3.0-3.2 mm in the jaws and the guard stopped the insertion.  The skill now
    presses the already-gripping pads harder before the insertion; the one-shot
    close is untouched, so the brick cannot be ejected during the close.
    """
    scene = Scene()
    ledger = OperationStateLedger(("piper_0", "piper_1"))
    ledger.confirm_holding("piper_0", "brick-lower", {"holding_confirmed": True})
    ledger.confirm("piper_1", "supporting:brick-upper", {"confirmed": True})
    free, motion = FreeSpace(scene), Motion(scene)
    ctx = context()
    # The regrip target is derived from the opening the gripper holds the brick
    # with, captured before the flow (the release later changes the reading).
    held_opening = scene.data["robots"]["piper_0"]["opening_m"]
    tightened = replace(
        ctx, params=replace(ctx.params, place_up_regrip_closing_m=0.002)
    )
    handler = PlaceUpHandler(
        scene,
        ledger,
        session_id="session-1",
        manifest_sha256="a" * 64,
        motion_transport=motion,
        free_space_transport=free,
        context=tightened,
        poll_interval_s=0,
    )
    result = asyncio.run(handler.execute(upward_operation()))
    assert result.passed, result.reason
    assert "regrip" in result.evidence
    ids = [call.request_id.rsplit(":", 1)[1] for call in motion.calls]
    assert ids.index("regrip") < ids.index("insert_up")
    measured = held_opening
    regrip = next(call for call in motion.calls if call.request_id.endswith(":regrip"))
    assert regrip.payload["opening_m"] == pytest.approx(measured - 0.002)
    assert regrip.payload["expected_opening_m"][0] < measured
    assert regrip.payload["expected_opening_m"][1] > measured

    # The default stays a one-shot close: no regrip without the parameter.
    default_scene = Scene()
    default_ledger = OperationStateLedger(("piper_0", "piper_1"))
    default_ledger.confirm_holding(
        "piper_0", "brick-lower", {"holding_confirmed": True}
    )
    default_ledger.confirm("piper_1", "supporting:brick-upper", {"confirmed": True})
    free_default, motion_default = FreeSpace(default_scene), Motion(default_scene)
    default_handler = PlaceUpHandler(
        default_scene,
        default_ledger,
        session_id="session-1",
        manifest_sha256="a" * 64,
        motion_transport=motion_default,
        free_space_transport=free_default,
        context=context(),
        poll_interval_s=0,
    )
    assert asyncio.run(default_handler.execute(upward_operation())).passed
    assert not any(call.request_id.endswith(":regrip") for call in motion_default.calls)


def test_continuous_insertion_retains_the_feedback_trial_slip_bounds():
    """Removing per-step pauses does not relax custody or force protection."""
    scene = Scene()
    ledger = OperationStateLedger(("piper_0", "piper_1"))
    ledger.confirm_holding("piper_0", "brick-lower", {"holding_confirmed": True})
    ledger.confirm("piper_1", "supporting:brick-upper", {"confirmed": True})
    free, motion = FreeSpace(scene), Motion(scene)
    handler = PlaceUpHandler(
        scene,
        ledger,
        session_id="session-1",
        manifest_sha256="a" * 64,
        motion_transport=motion,
        free_space_transport=free,
        context=context(),
        poll_interval_s=0,
    )
    result = asyncio.run(handler.execute(upward_operation()))
    assert result.passed, result.reason
    guard = result.evidence["contact_guard"]
    assert "insertion_feedback" not in result.evidence
    assert guard["grasp_position_m"] == pytest.approx(0.003)
    assert guard["grasp_angle_rad"] == pytest.approx(math.radians(3))
    assert guard["force_limit_n"] == pytest.approx(8.0)
    assert guard["supported_position_m"] == pytest.approx(0.002)
    assert guard["supported_angle_rad"] == pytest.approx(math.radians(3))


def test_pre_phase_cancel_uses_real_session_boundary():
    """A physical facade can no longer manufacture no-motion safe cancellation."""
    calls = []

    def cancel(identity):
        calls.append(identity)
        return {"attempted": True, "terminal": True, "safe": True}

    result = asyncio.run(
        UnifiedDualArmRuntimeFacade(
            SimpleNamespace(cancel_free_space=cancel)
        ).cancel_pre_phase("transition")
    )
    assert calls == ["transition"] and result["safe"] is True


@pytest.mark.parametrize("failure", ["release", "retreat", "support"])
def test_incomplete_physical_proof_never_publishes_assembled(failure):
    """Failed release/retreat keeps support; lost contact forbids insertion."""
    scene = Scene()
    ledger = OperationStateLedger(("piper_0", "piper_1"))
    ledger.confirm_holding("piper_0", "brick-lower", {"holding_confirmed": True})
    ledger.confirm("piper_1", "supporting:brick-upper", {"confirmed": True})
    free, motion = FreeSpace(scene), Motion(scene)
    original_motion = motion.run_motion_operation
    original_free = free.execute_free_space

    def run(robot, item):
        if failure == "release" and item.request_id.endswith(":release"):
            motion.calls.append(item)
            return {
                "motion_sent": True,
                "result": {
                    "passed": False,
                    "execution": {"terminal": True, "safe_to_release": True},
                },
            }
        return original_motion(robot, item)

    async def execute(item):
        result = await original_free(item)
        if failure == "retreat" and item.motion_id.endswith(":retreat_up"):
            return replace(
                result, arrival_confirmed=False, reason="measured retreat failed"
            )
        return result

    motion.run_motion_operation = run
    free.execute_free_space = execute
    if failure == "support":
        scene.contacts = False
    handler = PlaceUpHandler(
        scene,
        ledger,
        session_id="session-1",
        manifest_sha256="a" * 64,
        motion_transport=motion,
        free_space_transport=free,
        context=context(),
        poll_interval_s=0,
    )
    result = asyncio.run(handler.execute(upward_operation()))
    assert not result.passed
    assert ledger.observe("piper_0", "placed:brick-lower") is None
    assert ledger.observe("piper_1", "supporting:brick-upper") is not None
    if failure == "release":
        assert ledger.owner_of_object("brick-lower") == "piper_0"
        assert ledger.uncertain_keys("piper_0") == ("holding:brick-lower",)
        assert not any(item.motion_id.endswith(":retreat_up") for item in free.calls)
    if failure == "support":
        assert not any(item.request_id.endswith(":insert_up") for item in motion.calls)


def test_contact_scene_update_preserves_all_other_structure_obstacles():
    """Contact permission cannot erase the baseplate or an unrelated brick."""
    scene = Scene()
    scene.data["bricks"]["other-brick"] = copy.deepcopy(
        scene.data["bricks"]["brick-upper"]
    )
    scene.data["environment_colliders"] = {
        "baseplate": {"aabb_min": [-1, -1, -0.02], "aabb_max": [1, 1, 0]}
    }
    ledger = OperationStateLedger(("piper_0", "piper_1"))
    ledger.confirm_holding("piper_0", "brick-lower", {"holding_confirmed": True})
    ledger.confirm("piper_1", "supporting:brick-upper", {"confirmed": True})
    handler = PlaceUpHandler(
        scene,
        ledger,
        session_id="session-1",
        manifest_sha256="a" * 64,
        context=context(),
    )
    admission = asyncio.run(handler.preflight(upward_operation()))
    payload = handler._world_payload(admission, admission.sample, "contact", scene.data)
    assert {box["object_id"] for box in payload["boxes"]} == {
        "brick-upper",
        "other-brick",
        "environment:baseplate",
    }
    assert payload["allowed_collision_pairs"] == (("brick-lower", "brick-upper"),)


def test_single_support_successor_does_not_steal_carrier_resources():
    """Support establishes its own role while the PICK endpoint retains custody."""
    from pre_phase_helpers import operation as endpoint
    from pre_phase_helpers import snapshot

    previous = replace(
        endpoint("pick", "piper_0"), resources=("robot:piper_0", "part:lower")
    )
    following = replace(endpoint("support", "piper_1"), skill="SUPPORT_TOP")
    item = replace(
        request(),
        predecessor_operations=(previous,),
        predecessor_states={"pick": {"terminal_safe": True}},
        next_operation=following,
        participants=("piper_0", "piper_1"),
        resources=previous.resources,
        lease_owners={resource: "pick" for resource in previous.resources},
        scene_snapshot=snapshot(("piper_0", "piper_1")),
    )
    assert "robot:piper_0" not in item.lease_targets
    assert "part:lower" not in item.lease_targets
    assert item.lease_targets["robot:piper_1"] == "support"


@pytest.mark.parametrize("mode", ["empty", "contact", "force", "deflected", "stale"])
def test_support_guard_presses_through_contact_and_stops_on_safety_limits(mode):
    """A support descent presses through contact; only its limits stop it.

    Touching the top surface is the expected condition of the controlled press
    (the contact target sits ``support_press_m`` below the surface), so a
    measured contact keeps the descent running and is recorded as evidence.
    Stopping there left the pads hovering above the surface with the press
    unexecuted (live evidence, task-c-top-offset13f 2026-10-14).
    """
    shared = asyncio.run(Scene().observe("piper_1"))
    guard = {
        "robot_id": "piper_1",
        "session_id": "session-1",
        "manifest_sha256": "a" * 64,
        "upper_id": "brick-upper",
        "supported_pose": copy.deepcopy(shared["bricks"]["brick-upper"]["pose"]),
        "supported_position_m": 0.002,
        "supported_angle_rad": 0.05236,
        "force_limit_n": 8.0,
        "mode": "support_top",
    }
    expected = {
        "empty": None,
        "contact": None,
        "force": "support_force_exceeded",
        "deflected": "supported_upper_disturbed",
    }
    if mode == "empty":
        shared["pad_object_contacts"]["contacts"] = []
    elif mode == "force":
        shared["pad_object_contacts"]["contacts"][0]["normal_force_n"] = 9.0
    elif mode == "deflected":
        shared["bricks"]["brick-upper"]["pose"]["position_xyz"][2] += 0.003
    elif mode == "stale":
        shared["pad_object_contacts"]["sequence"] -= 1
        with pytest.raises(ValueError, match="stale"):
            inspect_contact(shared, guard)
        return
    stop, measured = inspect_contact(shared, guard)
    assert stop == expected[mode]
    if mode == "contact":
        # The measured contact is recorded even though the press continues.
        assert measured["support_force_n"] > 0
        assert measured["support_contact_established"] is True
        assert measured["support_contact"]["confirmed"] is True
    assert allows_recovery("support_contact_confirmed")


def test_support_transport_error_never_claims_terminal_safety():
    """Missing action feedback retains unknown physical state, not safe idle."""
    scene = Scene()
    ledger = OperationStateLedger(("piper_0", "piper_1"))
    free, motion = FreeSpace(scene), Motion(scene)
    original = motion.run_motion_operation

    def run(robot, item):
        if item.request_id.endswith(":support_contact"):
            raise RuntimeError("worker feedback lost")
        return original(robot, item)

    motion.run_motion_operation = run
    handler = SupportTopHandler(
        scene,
        ledger,
        session_id="session-1",
        manifest_sha256="a" * 64,
        motion_transport=motion,
        free_space_transport=free,
        context=context(),
        poll_interval_s=0,
    )
    result = asyncio.run(handler.execute(support_operation()))
    assert not result.passed and not result.terminal
    assert result.motion_sent
    assert ledger.observe("piper_1", "supporting:brick-upper") is None


def test_pre_phase_requires_exact_arrived_participants():
    """An unrelated robot's arrival cannot stand in for the requested arm."""
    from test_unified_runtime_facade import FakeSession

    session = FakeSession()
    session.execution_result = CoordinatedSessionResult(
        "transition-1",
        ("piper_0", "piper_1"),
        {},
        {"points": [1]},
        {
            "prepared": True,
            "started": True,
            "terminal": True,
            "safe_to_release": True,
            "participants": {"piper_1": {"arrival_confirmed": True}},
        },
        True,
        participants=("piper_0",),
    )
    facade = UnifiedDualArmRuntimeFacade(
        session,
        contexts={robot: context() for robot in ("piper_0", "piper_1")},
        state_ledger=OperationStateLedger(("piper_0", "piper_1")),
    )
    result = asyncio.run(facade.execute_pre_phase(request()))
    assert not result.passed and not result.arrival_confirmed


def test_pre_phase_revalidation_allows_only_its_own_planned_scene_version():
    """Applying a measured attachment can bump version; unrelated drift cannot."""
    from test_unified_runtime_facade import FakeSession

    session = FakeSession(version=2)
    facade = UnifiedDualArmRuntimeFacade(
        session, state_ledger=OperationStateLedger(("piper_0", "piper_1"))
    )
    item = request()
    proof = replace(
        success(item), evidence={"trajectory": {"planning_scene": {"scene_version": 2}}}
    )
    assert asyncio.run(facade.validate_pre_phase(item, proof))
    session._shared["planning_scene_version"] = 3
    assert not asyncio.run(facade.validate_pre_phase(item, proof))
