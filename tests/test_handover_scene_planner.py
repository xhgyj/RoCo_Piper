"""Native planning-provider tests without ROS/Isaac/controller startup."""

from dataclasses import replace

import pytest
from test_handover_geometry import native_scene

from rocobrick.adapters.moveit.handover_scene_planner import (
    HandoverGripperConfigurations,
    NativeHandoverScenePlanner,
    ReadOnlyCombinedHandoverPlanner,
    ReadOnlyPlannerInternalError,
    ReadOnlyPlanRejectedError,
)
from rocobrick.skills.handover_joint_backend import HandoverJointPlanningBackend
from rocobrick.skills.handover_planning import calibrated_target, generate_candidates
from rocobrick.skills.observation import Pose
from rocobrick.skills.placement_geometry import compose


class RecordingPlanRuntime:
    """Explicit fake ONLY for ROS transport; geometry/IK remain native."""

    def __init__(self):
        """Capture plan-only envelopes; provide no execution/scene-write method."""
        self.requests = []
        self.colliding_midpoint = False
        self.geometry = None
        self.captured = None
        self.pose_shift_m = 0.0
        self.pose_shifts_by_robot = {}

    def plan(self, request, *, timeout_s):
        """Return one transport fixture on the shared time grid."""
        self.requests.append(request)
        names = tuple(
            f"{r}_joint{i}" for r in request.participants for i in range(1, 7)
        )
        initial = tuple(request.start_state[n] for n in names)
        if all("tcp_region" in t for t in request.targets.values()):
            from rocobrick.skills.handover_region import preparation_region

            region = preparation_region(self.captured)
            observed = self.captured.observation(self.captured.donor.robot_id)
            local = Pose(
                tuple(
                    region["center"][i]
                    - (
                        region["precontact_distance_m"] + region["contact_clearance_m"]
                        if i == 2
                        else 0
                    )
                    for i in range(3)
                ),
                (0, 0, 0, 1),
            )
            goal = {}
            for robot, target in (
                (self.captured.donor.robot_id, observed.tcp_pose),
                (self.captured.receiver.robot_id, compose(observed.object_pose, local)),
            ):
                solution = self.geometry.solve_tcp_ik(
                    robot,
                    target,
                    request.start_state,
                    deadline_s=1e9,
                    clock=lambda: 0.0,
                )
                if solution is None:
                    raise ReadOnlyPlanRejectedError(
                        "synthetic region fixture has no endpoint"
                    )
                goal.update(solution)
        elif all("tcp_pose" in t for t in request.targets.values()):
            # A deliberately synthetic MoveGroup stand-in. This does NOT test
            # ROS/KDL pose sampling; it lets native post-plan proofs run on CPU.
            goal = {}
            contexts = (self.captured.donor, self.captured.receiver)
            for ctx in contexts:
                raw = request.targets[ctx.robot_id]["tcp_pose"]
                world = compose(
                    Pose(ctx.base.position_xyz, ctx.base.orientation_xyzw),
                    Pose(tuple(raw["position"]), tuple(raw["orientation_xyzw"])),
                )
                solution = self.geometry.solve_tcp_ik(
                    ctx.robot_id,
                    world,
                    request.start_state,
                    deadline_s=1e9,
                    clock=lambda: 0.0,
                )
                if solution is None:
                    raise ReadOnlyPlanRejectedError(
                        "synthetic transport has no fixture goal"
                    )
                goal.update(solution)
                goal[f"{ctx.robot_id}_joint1"] += self.pose_shifts_by_robot.get(
                    ctx.robot_id, self.pose_shift_m
                )
        else:
            goal = {
                n: v
                for t in request.targets.values()
                for n, v in t["joint_positions"].items()
            }
        points = [{"time_from_start_s": 0.0, "positions": initial}]
        if self.colliding_midpoint:
            midpoint = dict(request.start_state)
            midpoint["piper_0_joint3"] = -0.33
            points.append(
                {
                    "time_from_start_s": 1.0,
                    "positions": tuple(midpoint[n] for n in names),
                }
            )
        points.append(
            {"time_from_start_s": 2.0, "positions": tuple(goal[n] for n in names)}
        )
        return {"status": "SUCCEEDED", "joint_names": names, "points": points}


def native_provider(tmp_path):
    """Construct actual native geometry and a deliberately synthetic ROS transport.

    Returns:
        Provider, captured request/candidate and recording transport.
    """
    scene, request = native_scene(tmp_path)
    # Primitive Cartesian chains translate their toy link geometry through
    # a different volume than Piper. Give this fixture a low floor, plus an
    # obstacle hit only by the deliberately bad transport midpoint.
    request.scene_snapshot["environment_colliders"] = {
        "floor": {"aabb_min": [-2, -2, -1.05], "aabb_max": [2, 2, -1.03]},
        "bad_midpoint": {
            "aabb_min": [0.348, 0.018, 0.015],
            "aabb_max": [0.352, 0.022, 0.025],
        },
    }
    # Keep nominal object/giver pose and an upward, zero-yaw receiver for an
    # analytically reachable primitive fixture; never a physical Piper claim.
    request = replace(
        request,
        profile=replace(
            request.profile,
            object_yaws_rad=(0.0,),
            object_tilts_rad=(0.0,),
            receiver_yaws_rad=(0.0,),
            receiver_tilts_rad=(0.0,),
        ),
    )
    scene = type(scene)(scene.files, request)
    candidate = next(
        c
        for c in generate_candidates(request)
        if c.object_goal.position_xyz
        == request.observation("piper_1").object_pose.position_xyz
    )
    grippers = HandoverGripperConfigurations(
        candidate.candidate_id,
        "f" * 64,
        {"piper_0_gripper_joint1": 0.02, "piper_0_gripper_joint2": -0.02},
        {"piper_0_gripper_joint1": 0.006, "piper_0_gripper_joint2": -0.006},
        {"piper_1_gripper_joint1": 0.02, "piper_1_gripper_joint2": -0.02},
    )
    runtime = RecordingPlanRuntime()
    runtime.geometry, runtime.captured = scene, request
    provider = NativeHandoverScenePlanner(
        scene,
        ReadOnlyCombinedHandoverPlanner(runtime, clock=lambda: 0.0),
        grippers=lambda c: replace(grippers, candidate_id=c.candidate_id),
        read_binding=lambda: request.binding_sha256,
        clock=lambda: 0.0,
    )
    return provider, request, candidate, runtime


def test_pose_first_native_provider_keeps_actual_endpoint_for_phase_chain(
    tmp_path, monkeypatch
):
    """Do not round tolerated MoveGroup endpoints back to nominal native IK."""
    provider, request, candidate, runtime = native_provider(tmp_path)
    runtime.pose_shift_m = 0.0002

    def forbidden(*args, **kwargs):
        pytest.fail("production called native rendezvous IK or joint planning")

    monkeypatch.setattr(provider, "solve_ik", forbidden)
    monkeypatch.setattr(provider, "plan_combined", forbidden)
    result = HandoverJointPlanningBackend(
        provider, clock=lambda: 0.0, use_moveit_pose_goals=True
    ).evaluate_candidate(request, candidate, deadline_s=20.0)
    assert result.passed, (result.reason, provider.last_path_failure)
    assert len(runtime.requests) == 1
    assert runtime.requests[0].timeout_s == 10.0
    assert all(set(t) == {"tcp_pose"} for t in runtime.requests[0].targets.values())
    actual = provider.geometry.tcp_pose(
        request.receiver.robot_id,
        {**request.start_state, **provider._pose_goals[candidate.candidate_id]},
    )
    assert actual.position_xyz[1] - candidate.receiver_precontact.position_xyz[
        1
    ] == pytest.approx(0.0002, abs=0.0001)
    paths = provider.phase_paths(candidate)
    for t in result.joint_plan.trajectories:
        assert t.points[0].positions == tuple(
            request.start_state[n] for n in t.joint_names
        )
        assert t.points[-1].positions == tuple(
            paths.approach[0][n] for n in t.joint_names
        )


@pytest.mark.parametrize("shift, midpoint", [(0.01, False), (0.0, True)])
def test_pose_first_still_rejects_bad_fk_or_interior_collision(
    tmp_path, shift, midpoint
):
    """MoveGroup success does not bypass native calibrated FK or swept geometry."""
    provider, request, candidate, runtime = native_provider(tmp_path)
    runtime.pose_shift_m, runtime.colliding_midpoint = shift, midpoint
    result = HandoverJointPlanningBackend(
        provider, clock=lambda: 0.0, use_moveit_pose_goals=True
    ).evaluate_candidate(request, candidate, deadline_s=20.0)
    assert not result.passed
    assert len(runtime.requests) == 1
    if shift:
        assert provider.last_path_failure["phase"] == "moveit_endpoint"
    else:
        assert result.reason == "PATH_OR_SCENE_VALIDATION_FAILED"


def test_pose_first_cohold_collision_is_rejected_after_moveit_before_approach(
    tmp_path, monkeypatch
):
    """Only rendezvous order changes; colliding simultaneous grasp stays blocked."""
    from rocobrick.adapters.moveit.handover_geometry import CollisionVerdict

    provider, request, candidate, runtime = native_provider(tmp_path)
    original = provider.geometry.check

    def collision(joints, **kwargs):
        if kwargs.get("receiver_touch"):
            return CollisionVerdict(False, 0.0, (), "FULL_GEOMETRY_COLLISION")
        return original(joints, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("approach started before cohold endpoint admission")

    monkeypatch.setattr(provider.geometry, "check", collision)
    monkeypatch.setattr(provider, "_cartesian", forbidden)
    result = HandoverJointPlanningBackend(
        provider, clock=lambda: 0.0, use_moveit_pose_goals=True
    ).evaluate_candidate(request, candidate, deadline_s=20.0)
    assert not result.passed and len(runtime.requests) == 1
    assert provider.last_path_failure["phase"] == "cohold_endpoint"


def test_pose_query_rejection_is_recorded_without_native_fallback(tmp_path):
    """Ordinary unsuccessful read-only query rejects one candidate, not search."""
    provider, request, candidate, runtime = native_provider(tmp_path)

    def rejected(query, *, timeout_s):
        runtime.requests.append(query)
        raise ReadOnlyPlanRejectedError("NO_IK_SOLUTION")

    runtime.plan = rejected
    result = HandoverJointPlanningBackend(
        provider, clock=lambda: 0.0, use_moveit_pose_goals=True
    ).evaluate_candidate(request, candidate, deadline_s=20.0)
    assert result.reason == "MOVEIT_POSE_PLAN_FAILED"
    assert "NO_IK_SOLUTION" in provider.planner.last_planning_failure
    assert len(runtime.requests) == 1


def test_pose_query_does_not_retry_moveit_internal_failure_as_unreachable_target(
    tmp_path,
):
    """A broken sampler must not burn hundreds of candidate requests."""
    provider, request, candidate, runtime = native_provider(tmp_path)

    def fail(query, *, timeout_s):
        runtime.requests.append(query)
        raise ReadOnlyPlannerInternalError("MOVEIT_INTERNAL_FAILURE")

    runtime.plan = fail
    with pytest.raises(ReadOnlyPlannerInternalError):
        HandoverJointPlanningBackend(
            provider, clock=lambda: 0.0, use_moveit_pose_goals=True
        ).evaluate_candidate(request, candidate, deadline_s=20.0)
    assert len(runtime.requests) == 1


def test_deadline_before_query_does_not_erase_previous_moveit_failure(tmp_path):
    """Budget exhaustion must retain the preceding recorded failure reason."""
    provider, request, candidate, runtime = native_provider(tmp_path)
    provider.planner.last_planning_failure = "NO_IK_SOLUTION"
    assert (
        provider.planner.plan_poses(
            request, candidate, request.start_state, deadline_s=0.0
        )
        is None
    )
    assert provider.planner.last_planning_failure == "NO_IK_SOLUTION"
    assert runtime.requests == []


@pytest.mark.parametrize("expired", [True, False])
def test_pose_first_checks_deadline_and_static_binding_before_query(tmp_path, expired):
    """No query can run after deadline or against a changed native scene."""
    provider, request, candidate, runtime = native_provider(tmp_path)
    if not expired:
        provider.read_binding = lambda: "a" * 64
    result = HandoverJointPlanningBackend(
        provider, clock=lambda: 0.0, use_moveit_pose_goals=True
    ).evaluate_candidate(request, candidate, deadline_s=0.0 if expired else 20.0)
    assert not result.passed and runtime.requests == []


def test_pose_first_requires_object_relative_contact_not_just_two_tcp_goals(tmp_path):
    """Individually tolerated endpoints may still disagree across the object."""
    provider, request, candidate, runtime = native_provider(tmp_path)
    runtime.pose_shifts_by_robot = {"piper_0": 0.0009, "piper_1": -0.0009}
    result = HandoverJointPlanningBackend(
        provider, clock=lambda: 0.0, use_moveit_pose_goals=True
    ).evaluate_candidate(request, candidate, deadline_s=20.0)
    assert not result.passed and len(runtime.requests) == 1
    assert provider.last_path_failure["phase"] == "moveit_contact_binding"


def test_native_provider_certifies_real_geometry_and_sends_one_combined_plan(tmp_path):
    """The actual gates operate on native geometry; only ROS transport is faked."""
    provider, request, candidate, runtime = native_provider(tmp_path)
    result = HandoverJointPlanningBackend(
        provider, clock=lambda: 0.0
    ).evaluate_candidate(
        request,
        candidate,
        deadline_s=10.0,
    )
    assert result.passed, result.reason
    assert len(runtime.requests) == 1
    envelope = runtime.requests[0]
    assert envelope.plan_only is True and envelope.planning_group == "dual_arm"
    assert envelope.participants == ("piper_1", "piper_0")
    paths = provider.phase_paths(candidate)
    for a, b in zip(paths.approach, paths.approach[1:]):
        assert all(
            a[f"piper_1_joint{i}"] == b[f"piper_1_joint{i}"] for i in range(1, 7)
        )
    with pytest.raises(TypeError):
        paths.approach[0]["piper_1_joint1"] = 1.0
    assert (
        paths.receiver_follow[-1]["piper_0_joint3"]
        > paths.receiver_follow[0]["piper_0_joint3"]
    )


def test_combined_query_uses_guarded_fresh_snapshot_without_rebinding_geometry(
    tmp_path,
):
    """An old search snapshot is not reused as current-clock admission."""
    _, request, candidate, runtime = native_provider(tmp_path)
    old_digest = request.scene_snapshot["scene_digest"]
    old_binding = request.binding_sha256
    fresh = {
        **request.scene_snapshot,
        "scene_digest": "e" * 64,
        "sequence": request.scene_snapshot["sequence"] + 1,
        "sim_time_s": request.scene_snapshot["sim_time_s"] + 200,
    }
    reads = []

    def read_static(captured):
        reads.append(captured)
        return {
            "scene_snapshot": fresh,
            "planning_scene_version": request.scene_version,
        }

    planner = ReadOnlyCombinedHandoverPlanner(
        runtime, clock=lambda: 0.0, read_static_scene=read_static
    )
    arm_goals = {n: v for n, v in request.start_state.items() if "gripper" not in n}
    operation = planner.plan(
        request, candidate, arm_goals, request.start_state, deadline_s=10.0
    )
    assert reads == [request]
    assert runtime.requests[0].scene_snapshot == fresh
    assert runtime.requests[0].scene_digest == fresh["scene_digest"]
    assert operation.scene_digest == old_digest
    assert request.binding_sha256 == old_binding


def test_failed_static_guard_prevents_combined_query(tmp_path):
    """Movement detected during search cannot be laundered by a new timestamp."""
    _, request, candidate, runtime = native_provider(tmp_path)

    def read_static(captured):
        raise ValueError("arms/fingers moved during HANDOVER planning")

    planner = ReadOnlyCombinedHandoverPlanner(
        runtime, clock=lambda: 0.0, read_static_scene=read_static
    )
    with pytest.raises(ValueError, match="arms/fingers moved"):
        planner.plan(request, candidate, {}, {}, deadline_s=10.0)
    assert runtime.requests == []


def test_fresh_read_consumes_original_planning_budget(tmp_path):
    """No query is sent if the bounded fresh observation uses the last time."""
    _, request, candidate, runtime = native_provider(tmp_path)
    wall = [0.0]

    def read_static(captured):
        wall[0] = 11.0
        return {
            "scene_snapshot": request.scene_snapshot,
            "planning_scene_version": request.scene_version,
        }

    planner = ReadOnlyCombinedHandoverPlanner(
        runtime, clock=lambda: wall[0], read_static_scene=read_static
    )
    assert planner.plan(request, candidate, {}, {}, deadline_s=10.0) is None
    assert runtime.requests == []


def test_native_provider_refuses_actual_returned_trajectory_interior_collision(
    tmp_path,
):
    """Valid paired goals do not make a colliding ROS path safe."""
    provider, request, candidate, runtime = native_provider(tmp_path)
    runtime.colliding_midpoint = True
    result = HandoverJointPlanningBackend(
        provider, clock=lambda: 0.0
    ).evaluate_candidate(
        request,
        candidate,
        deadline_s=10.0,
    )
    assert not result.passed and not result.joint_path_valid
    assert len(runtime.requests) == 1


def test_cohold_collision_is_rejected_before_approach_or_combined_plan(
    tmp_path, monkeypatch
):
    """A colliding closed-jaw goal must not consume path-planning work."""
    from rocobrick.adapters.moveit.handover_geometry import CollisionVerdict

    provider, request, candidate, runtime = native_provider(tmp_path)
    original_check = provider.geometry.check
    pair = ("piper_0_gripper_link1_0", "piper_1_gripper_link2_0")

    def colliding_cohold(joints, **kwargs):
        if kwargs.get("receiver_touch"):
            return CollisionVerdict(False, 0.0, (pair,), "FULL_GEOMETRY_COLLISION")
        return original_check(joints, **kwargs)

    def forbidden_approach(*args, **kwargs):
        pytest.fail("approach was attempted before simultaneous-grasp admission")

    monkeypatch.setattr(provider.geometry, "check", colliding_cohold)
    monkeypatch.setattr(provider, "_cartesian", forbidden_approach)
    result = HandoverJointPlanningBackend(
        provider, clock=lambda: 0.0
    ).evaluate_candidate(request, candidate, deadline_s=10.0)
    assert not result.passed
    assert runtime.requests == []
    assert provider.last_path_failure["phase"] == "cohold_endpoint"
    assert provider.last_path_failure["collision_pairs"] == (pair,)


def test_successful_candidate_does_not_retain_previous_path_failure(tmp_path):
    """A later feasible candidate must not report a rejected predecessor."""
    provider, request, candidate, _ = native_provider(tmp_path)
    provider.last_path_failure = {"phase": "receiver_close", "failure": "old"}
    result = HandoverJointPlanningBackend(
        provider, clock=lambda: 0.0
    ).evaluate_candidate(request, candidate, deadline_s=10.0)
    assert result.passed
    assert provider.last_path_failure is None


def test_readback_mismatch_prevents_movegroup_dispatch(tmp_path):
    """Changed scene identity is rejected without mutating or replanning it."""
    provider, request, candidate, runtime = native_provider(tmp_path)
    provider.read_binding = lambda: "0" * 64
    result = HandoverJointPlanningBackend(
        provider, clock=lambda: 0.0
    ).evaluate_candidate(
        request,
        candidate,
        deadline_s=10.0,
    )
    assert not result.passed and runtime.requests == []


def test_deadline_prevents_even_plan_only_goal_dispatch(tmp_path):
    """Time exhaustion cannot be promoted to synthetic success."""
    provider, request, candidate, runtime = native_provider(tmp_path)
    result = HandoverJointPlanningBackend(
        provider, clock=lambda: 0.0
    ).evaluate_candidate(
        request,
        candidate,
        deadline_s=0.0,
    )
    assert result.reason == "HANDOVER_PLANNING_BUDGET_EXHAUSTED"
    assert runtime.requests == []


def test_exact_gripper_joint_coverage_is_required(tmp_path):
    """Missing finger states cannot use a zero/neutral fallback."""
    provider, request, candidate, _ = native_provider(tmp_path)
    original = provider.grippers(candidate)
    provider.grippers = lambda _: replace(
        original, donor_open={"piper_1_gripper_joint1": 0.02}
    )
    with pytest.raises(ValueError, match="exact exported"):
        provider.validate_pair(request, candidate, {}, deadline_s=10.0)


def test_native_cartesian_certificate_refuses_endpoint_orientation_excursion(tmp_path):
    """Joint endpoint validity cannot substitute for Cartesian-tube proof."""
    provider, request, _, _ = native_provider(tmp_path)
    first = provider.geometry.tcp_pose("piper_0", request.start_state)
    last = {**request.start_state, "piper_0_joint4": 0.05}
    assert not provider.geometry.certify_cartesian_segment(
        request.start_state,
        last,
        robot="piper_0",
        first_pose=first,
        last_pose=first,
        deadline_s=10.0,
        clock=lambda: 0.0,
    )


def test_native_ik_rejects_wrong_calibration_identity(tmp_path):
    """A numerically reachable target must retain its measured model binding."""
    provider, request, _, _ = native_provider(tmp_path)
    target = dict(
        calibrated_target(
            Pose((0.35, 0, 0.35), (0, 0, 0, 1)), request.receiver, request.profile
        )
    )
    target["tcp_pose"] = {**target["tcp_pose"], "source_sha256": "0" * 64}
    with pytest.raises(ValueError, match="calibration/frame"):
        provider.solve_ik(request, request.receiver, target, deadline_s=10.0)


def test_certified_phase_cache_cannot_reuse_changed_gripper_configuration(tmp_path):
    """A cached certificate binds its calibrated finger sweep, not just pose."""
    provider, request, candidate, _ = native_provider(tmp_path)
    result = HandoverJointPlanningBackend(
        provider, clock=lambda: 0.0
    ).evaluate_candidate(
        request,
        candidate,
        deadline_s=10.0,
    )
    assert result.passed
    previous = provider.grippers(candidate)
    provider.grippers = lambda _: replace(previous, calibration_sha256="a" * 64)
    with pytest.raises(ValueError, match="changed during planning"):
        provider.validate_approach(request, candidate, {}, deadline_s=10.0)
