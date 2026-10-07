"""Offline profile, paired geometry and physical proof regressions."""

import copy
import math
from dataclasses import asdict, replace

import pytest

from rocobrick.coordinated_moveit import (
    CoordinatedOperation,
    CoordinatedRobotTrajectory,
    CoordinatedTrajectoryPoint,
)
from rocobrick.skills.handover_observation import (
    CONTACT_SCHEMA,
    parse_handover_observation,
    verify_co_hold,
    verify_receiver_follow,
)
from rocobrick.skills.handover_planning import (
    CandidateEvaluation,
    HandoverPlanningRequest,
    HandoverRobotContext,
    ObjectGraspGeometry,
    calibrated_target,
    generate_candidates,
    plan_handover,
)
from rocobrick.skills.handover_profile import (
    BOTTOM_PROFILE_ID,
    EXECUTION_CAPABILITIES,
    PLANNING_CAPABILITIES,
    HandoverProfile,
    require_handover_capabilities,
    resolve_handover_profile,
)
from rocobrick.skills.placement_geometry import compose, inverse, pose_error
from rocobrick.skills.targets import RobotBaseFrame, TcpProvenance


def profile():
    """Return the packaged, explicitly planning-only policy."""
    return resolve_handover_profile(BOTTOM_PROFILE_ID)


def snapshot(sequence=1, displacement=0.0, *, released=False):
    """Build identified synthetic observations, never run a simulator.

    Returns:
        A same-tick scene and contact report for contract tests.
    """
    obj = {
        "position_xyz": [displacement, 0.0, 0.2],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    time_s = 1.0 + sequence * 0.1
    robots = {}
    contacts = []
    for robot in ("piper_0", "piper_1"):
        donor = robot == "piper_1"
        tcp = {
            "position_xyz": [0.01, 0.0, 0.21] if donor else [displacement, 0, 0.2],
            "orientation_xyzw": [1.0, 0, 0, 0] if donor else [0, 0, 0, 1.0],
        }
        robots[robot] = {
            "robot_id": robot,
            "sequence": sequence,
            "sim_time_s": time_s,
            "tcp_pose": tcp,
            "opening_m": 0.08 if donor and released else 0.016,
        }
        if not donor or not released:
            for pad, sign in ((1, -1), (2, 1)):
                contacts.append(
                    {
                        "robot_id": robot,
                        "pad_link": f"{robot}_gripper_link{pad}",
                        "physical_instance_id": "brick-1",
                        "position_world_m": [displacement, sign * 0.006, 0.2],
                        "normal_world": [0.0, float(sign), 0.0],
                        "normal_force_n": 1.0,
                    }
                )
    return {
        "schema": "rocobrick-scene-state/v1",
        "session_id": "session-1",
        "manifest_sha256": "c" * 64,
        "sequence": sequence,
        "sim_time_s": time_s,
        "scene_digest": "d" * 64,
        "robots": robots,
        "bricks": {
            "brick-1": {
                "pose": obj,
                "aabb_min": [displacement - 0.008, -0.008, 0.1945],
                "aabb_max": [displacement + 0.008, 0.008, 0.2055],
            }
        },
        "pad_object_contacts": {
            "schema": CONTACT_SCHEMA,
            "source": "physx_contact_report",
            "session_id": "session-1",
            "manifest_sha256": "c" * 64,
            "sequence": sequence,
            "sim_time_s": time_s,
            "contacts": contacts,
        },
    }


def planning_request():
    """Use explicit local geometry and both measured arm states.

    Returns:
        A complete synthetic planning request.
    """
    policy = profile()
    provenance = TcpProvenance(policy.runtime_model_sha256, "b" * 64)
    bases = {
        robot: RobotBaseFrame((x, -0.25, 0), (0, 0, 2**-0.5, 2**-0.5))
        for robot, x in (("piper_1", -0.25), ("piper_0", 0.25))
    }
    return HandoverPlanningRequest(
        "handover-1",
        "brick-1",
        ("piper_0", "piper_1"),
        HandoverRobotContext("piper_1", bases["piper_1"], provenance),
        HandoverRobotContext("piper_0", bases["piper_0"], provenance),
        snapshot(),
        1,
        {f"{robot}_joint{i}": 0.0 for robot in bases for i in range(1, 7)},
        {
            "holding_confirmed": True,
            "physical_instance_id": "brick-1",
            "robot_id": "piper_1",
        },
        ObjectGraspGeometry(
            (-0.008, -0.008, -0.0055), (0.008, 0.008, 0.0055), "e" * 64
        ),
        policy,
        4,
        10.0,
        1.15,
    )


@pytest.mark.parametrize("value", ["brick-side-grasp-v1", "../other", "", None])
def test_unknown_profiles_never_silently_become_bottom_handover(value):
    """Legacy and path-like IDs cannot select the new physical strategy."""
    with pytest.raises(ValueError, match="UNSUPPORTED_HANDOVER_PROFILE"):
        resolve_handover_profile(value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("execution_qualified", 1),
        ("max_candidates", True),
        ("planning_timeout_s", float("nan")),
        ("planning_timeout_s", float("inf")),
        ("receiver_tilts_rad", [math.pi / 2]),
        ("receiver_yaws_rad", [0.1]),
        ("runtime_model_sha256", "wrong"),
        ("verification_distance_m", 0.0001),
    ],
)
def test_profile_constraints_are_strict(field, value):
    """Malformed numeric policy cannot expand the safety domain."""
    with pytest.raises(ValueError):
        replace(profile(), **{field: value})


def test_profile_parse_rejects_unknown_keys_and_has_reproducible_digest():
    """A frozen complete profile can be audited without ignored parameters."""
    raw = {"schema": "rocobrick-handover-profile/v1", **asdict(profile())}
    assert HandoverProfile.parse(raw).sha256 == profile().sha256
    with pytest.raises(ValueError, match="keys"):
        HandoverProfile.parse({**raw, "ignore_collisions": True})


def test_candidate_grid_cannot_allocate_unbounded_cartesian_product():
    """A candidate count limit must also bound generation work and memory."""
    with pytest.raises(ValueError, match="generation budget"):
        replace(profile(), object_yaws_rad=tuple(i * 0.001 for i in range(1000)))


@pytest.mark.parametrize("time", [0.5, 3.0, float("nan")])
def test_shared_scene_requires_explicit_fresh_clock(time):
    """Old/future snapshots cannot be admitted by matching their own timestamp."""
    with pytest.raises(ValueError):
        replace(planning_request(), observed_sim_time_s=time)


def test_giver_holding_evidence_cannot_be_borrowed_from_receiver():
    """Physical object identity alone is not enough to bind the current holder."""
    request = planning_request()
    with pytest.raises(ValueError, match="giver holding"):
        replace(
            request,
            giver_holding_evidence={
                **request.giver_holding_evidence,
                "robot_id": request.receiver.robot_id,
            },
        )


def test_candidate_cannot_reuse_identity_after_changing_geometry():
    """All paired targets, not just object center, belong to the candidate SHA."""
    candidate = generate_candidates(planning_request())[0]
    with pytest.raises(ValueError, match="identity"):
        replace(candidate, receiver_span_m=0.02)


def test_planning_capability_is_not_physical_qualification():
    """Even a fully capable transport cannot upgrade a draft profile."""
    require_handover_capabilities(profile(), PLANNING_CAPABILITIES, physical=False)
    with pytest.raises(ValueError, match="NOT_QUALIFIED"):
        require_handover_capabilities(profile(), EXECUTION_CAPABILITIES, physical=True)
    with pytest.raises(ValueError, match="CAPABILITY_UNAVAILABLE"):
        require_handover_capabilities(profile(), (), physical=False)


def test_candidates_preserve_giver_grasp_and_use_bottom_approach():
    """Both robots can relocate without centering the giver TCP on the brick."""
    req = planning_request()
    before = copy.deepcopy(req.scene_snapshot)
    candidates = generate_candidates(req)
    assert candidates == generate_candidates(req)
    assert len(candidates) == req.profile.max_candidates
    giver = req.observation(req.donor.robot_id)
    original_grasp = compose(inverse(giver.object_pose), giver.tcp_pose)
    assert len({c.object_goal.position_xyz for c in candidates}) > 1
    for c in candidates:
        assert pose_error(
            compose(inverse(c.object_goal), c.donor_goal), original_grasp
        ) == pytest.approx((0, 0), abs=1e-7)
        assert (
            c.receiver_precontact.position_xyz[2] < c.receiver_contact.position_xyz[2]
        )
        assert c.approach_direction_world[2] > 0
        assert c.receiver_span_m == pytest.approx(0.016)
        for ctx, pose in (
            (req.donor, c.donor_goal),
            (req.receiver, c.receiver_precontact),
        ):
            target = calibrated_target(pose, ctx, req.profile)["tcp_pose"]
            assert target["frame_id"] == f"{ctx.robot_id}_base_link"
            assert target["tip_link"] == f"{ctx.robot_id}_rocobrick_tcp"
    assert req.scene_snapshot == before


def evaluation(candidate, **overrides):
    """Create explicit no-motion gates for a fake readonly planning backend.

    Returns:
        Synthetic bound gates and a combined trajectory.
    """
    values = {
        name: True
        for name in (
            "giver_ik",
            "receiver_ik",
            "combined_state_valid",
            "gripper_sweeps_valid",
            "approach_valid",
            "separation_valid",
            "scene_unchanged",
            "joint_path_valid",
        )
    }
    values["joint_plan"] = joint_plan(planning_request(), candidate)
    values.update(overrides)
    return CandidateEvaluation(
        candidate.candidate_id, candidate.binding_sha256, **values
    )


def joint_plan(request, candidate):
    """Build one synthetic full pair trajectory, never controller evidence.

    Returns:
        One coordinated pair plan with a shared time grid.
    """
    participants = (request.donor.robot_id, request.receiver.robot_id)
    trajectories = tuple(
        CoordinatedRobotTrajectory(
            robot,
            tuple(f"{robot}_joint{i}" for i in range(1, 7)),
            (
                CoordinatedTrajectoryPoint(0.0, (0.0,) * 6),
                CoordinatedTrajectoryPoint(1.0, (0.1,) * 6),
            ),
        )
        for robot in participants
    )
    return CoordinatedOperation(
        candidate.candidate_id,
        request.scene_snapshot["scene_digest"],
        participants,
        trajectories,
        0.0,
        request.profile.max_sync_skew_s,
    )


def test_ik_success_alone_cannot_admit_a_colliding_or_unplanned_pair():
    """Endpoint IK never substitutes for sweep, approach and separation gates."""
    req = replace(planning_request(), profile=replace(profile(), max_evaluations=3))

    class Backend:
        capabilities = PLANNING_CAPABILITIES

        def evaluate_candidate(self, request, candidate, *, deadline_s):
            """Reject an IK-reachable pair with a gripper-body collision.

            Returns:
                Failed geometry gates with synthetic successful IK.
            """
            return evaluation(
                candidate, combined_state_valid=False, gripper_sweeps_valid=False
            )

    result = plan_handover(req, Backend())
    assert not result.passed and result.evaluated == 3
    assert result.rejected == {"combined_state_valid": 3, "gripper_sweeps_valid": 3}
    assert result.rejection_reasons == {"MISSING_REQUIRED_PLANNING_PROOF": 3}
    assert result.reason == "NO_FEASIBLE_HANDOVER_IN_PROFILE_BUDGET"


def test_planner_accepts_only_a_full_bound_readonly_evaluation():
    """A selected goal is feasible planning evidence, never physical adoption."""

    class Backend:
        capabilities = PLANNING_CAPABILITIES

        def evaluate_candidate(self, request, candidate, *, deadline_s):
            """Return all required endpoint and path gates."""
            return evaluation(candidate)

    result = plan_handover(planning_request(), Backend())
    assert result.passed and result.evaluated == 1
    assert not hasattr(result, "owner_transfer_committed")
    with pytest.raises(ValueError, match="real bool"):
        evaluation(result.candidate, separation_valid=1)


def test_planner_rejects_foreign_evidence_and_late_success():
    """Changing bindings or returning after the budget cannot admit a plan."""

    class Foreign:
        capabilities = PLANNING_CAPABILITIES

        def evaluate_candidate(self, request, candidate, *, deadline_s):
            """Return success for another frozen request."""
            return replace(evaluation(candidate), binding_sha256="f" * 64)

    with pytest.raises(ValueError, match="binding"):
        plan_handover(planning_request(), Foreign())
    ticks = iter((0.0, 0.1, 31.0))

    class Late(Foreign):
        def evaluate_candidate(self, request, candidate, *, deadline_s):
            """Return a valid response after its overall deadline."""
            return evaluation(candidate)

    result = plan_handover(planning_request(), Late(), clock=lambda: next(ticks))
    assert not result.passed and result.reason == "HANDOVER_PLANNING_BUDGET_EXHAUSTED"


def paired(raw):
    """Parse this test's exact shared identities.

    Returns:
        A strict paired observation with actual-contact-shaped records.
    """
    return parse_handover_observation(
        raw,
        donor_id="piper_1",
        receiver_id="piper_0",
        physical_instance_id="brick-1",
        session_id="session-1",
        manifest_sha256="c" * 64,
    )


def cohold(samples):
    """Apply separate giver and receiver windows and the requested N frames.

    Returns:
        The co-hold proof derived from supplied synthetic observations.
    """
    donor = planning_request().observation("piper_1")
    return verify_co_hold(
        samples,
        profile=profile(),
        follow_samples=4,
        donor_window=(0.014, 0.018),
        receiver_window=(0.015, 0.017),
        expected_object_to_donor_tcp=compose(
            inverse(donor.object_pose), donor.tcp_pose
        ),
    )


def test_static_cohold_cannot_hide_giver_grasp_slip_during_relocation():
    """Rigid samples after a prior slip do not preserve the lift-time grasp."""
    frames = []
    for i in range(1, 5):
        raw = snapshot(i)
        raw["robots"]["piper_1"]["tcp_pose"]["position_xyz"][0] += 0.003
        frames.append(paired(raw))
    with pytest.raises(ValueError, match="since admitted lift"):
        cohold(frames)


def test_cohold_requires_actual_opposed_contacts_and_full_sample_count():
    """Static opening and pose alone cannot make either gripper a holder."""
    samples = [paired(snapshot(i)) for i in range(1, 5)]
    assert cohold(samples)["confirmed"] is True
    with pytest.raises(ValueError, match="follow_samples"):
        cohold(samples[:3])
    without_contacts = []
    for i in range(1, 5):
        raw = snapshot(i)
        raw["pad_object_contacts"]["contacts"] = []
        without_contacts.append(paired(raw))
    with pytest.raises(ValueError, match="pads"):
        cohold(without_contacts)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source", "moveit_collision"),
        ("source", "assembly_contact_force"),
        ("sequence", 999),
        ("session_id", "foreign"),
    ],
)
def test_mixed_or_nonphysics_contacts_are_rejected(field, value):
    """Geometry contact and stud-hole debug force do not prove pad contact."""
    raw = snapshot()
    raw["pad_object_contacts"][field] = value
    with pytest.raises(ValueError):
        paired(raw)


def test_missing_contact_signal_fails_closed():
    """Existing scene snapshots must not silently invent pad contacts."""
    raw = snapshot()
    del raw["pad_object_contacts"]
    with pytest.raises(ValueError, match="unavailable"):
        paired(raw)


def test_receiver_follow_requires_actual_motion_and_no_giver_contacts():
    """A moving receiver and object form proof; stationary samples do not."""
    samples = [paired(snapshot(i, (i - 1) * 0.005, released=True)) for i in range(1, 5)]
    args = dict(
        profile=profile(),
        follow_samples=4,
        donor_release_window=(0.07, 0.09),
        receiver_window=(0.015, 0.017),
    )
    assert verify_receiver_follow(samples, **args)["confirmed"] is True
    stationary = [paired(snapshot(i, released=True)) for i in range(1, 5)]
    with pytest.raises(ValueError, match="stationary"):
        verify_receiver_follow(stationary, **args)
    raw = snapshot(4, 0.015, released=True)
    raw["pad_object_contacts"]["contacts"].extend(
        snapshot(4, 0.015)["pad_object_contacts"]["contacts"][-2:]
    )
    with pytest.raises(ValueError, match="giver still"):
        verify_receiver_follow(samples[:3] + [paired(raw)], **args)


def test_follow_checks_rotation_and_does_not_accept_stale_samples():
    """Relative SE(3) and sequence order are independent evidence gates."""
    samples = [paired(snapshot(i, (i - 1) * 0.005, released=True)) for i in range(1, 5)]
    raw = snapshot(4, 0.015, released=True)
    raw["bricks"]["brick-1"]["pose"]["orientation_xyzw"] = [
        0.0,
        0.0,
        math.sin(0.1),
        math.cos(0.1),
    ]
    args = dict(
        profile=profile(),
        follow_samples=4,
        donor_release_window=(0.07, 0.09),
        receiver_window=(0.015, 0.017),
    )
    with pytest.raises(ValueError, match=r"SE\(3\)"):
        verify_receiver_follow(samples[:3] + [paired(raw)], **args)
    with pytest.raises(ValueError, match="stale"):
        verify_receiver_follow([samples[0], samples[0], *samples[2:]], **args)
