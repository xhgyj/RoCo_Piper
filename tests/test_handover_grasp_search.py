"""Bounded complementary-grasp geometry, without simulator or robot motion."""

import math
from dataclasses import asdict, replace

import pytest
from test_handover_joint_planning import joint_plan, planning_request

from rocobrick.skills.handover_planning import (
    CandidateEvaluation,
    generate_candidates,
    grasp_variants,
    plan_handover,
)
from rocobrick.skills.handover_profile import PLANNING_CAPABILITIES, HandoverProfile
from rocobrick.skills.handover_simulation import simulation_profile
from rocobrick.skills.observation import Pose
from rocobrick.skills.placement_geometry import compose, inverse, pose_error
from rocobrick.skills.targets import _quat_rotate_vector


def offset_request():
    """Keep an analytically simple tool frame with several bounded grasps.

    Returns:
        A synthetic no-motion request with lateral and depth search enabled.
    """
    request = planning_request()
    return replace(
        request,
        profile=replace(
            request.profile,
            object_yaws_rad=(0.0,),
            object_tilts_rad=(0.0,),
            receiver_yaws_rad=(0.0,),
            receiver_tilts_rad=(0.0,),
            receiver_lateral_offsets_m=(0.0, -0.004, 0.004, 0.008),
            receiver_depth_offsets_m=(0.0, 0.002),
            max_candidates=128,
            max_evaluations=128,
        ),
    )


def test_offsets_change_relative_grasp_without_shifting_the_closing_axis():
    """Receiver jaws stay centered across opposing faces; giver grasp is fixed."""
    request = offset_request()
    candidates = generate_candidates(request)
    assert candidates == generate_candidates(request)
    assert len({c.candidate_id for c in candidates}) == len(candidates)
    giver = request.observation(request.donor.robot_id)
    held = compose(inverse(giver.object_pose), giver.tcp_pose)
    local_targets = set()
    for candidate in candidates:
        assert pose_error(candidate.object_to_donor_tcp, held) == pytest.approx(
            (0.0, 0.0), abs=1e-7
        )
        local = compose(inverse(candidate.object_goal), candidate.receiver_contact)
        local_targets.add(tuple(round(v, 6) for v in local.position_xyz))
        assert local.position_xyz[1] == pytest.approx(0.0, abs=1e-7)
        assert abs(local.position_xyz[0]) <= 0.004 + 1e-7
        assert candidate.receiver_span_m == pytest.approx(0.016)
        assert math.dist(
            candidate.receiver_precontact.position_xyz,
            candidate.receiver_contact.position_xyz,
        ) == pytest.approx(request.profile.precontact_distance_m)
        assert candidate.approach_direction_world[2] > 0
    assert len(local_targets) == 6


def test_grasp_offsets_and_depth_are_bound_by_profile_and_candidate_identity():
    """Changing the search domain changes provenance, not safety thresholds."""
    request = offset_request()
    central = replace(
        request,
        profile=replace(
            request.profile,
            receiver_lateral_offsets_m=(0.0,),
            receiver_depth_offsets_m=(0.0,),
        ),
    )
    assert request.binding_sha256 != central.binding_sha256
    candidate = generate_candidates(request)[0]
    moved = replace(
        candidate.receiver_contact,
        position_xyz=tuple(v + 0.001 for v in candidate.receiver_contact.position_xyz),
    )
    with pytest.raises(ValueError):
        replace(candidate, receiver_contact=moved)


@pytest.mark.parametrize(
    "name,values",
    [
        ("receiver_lateral_offsets_m", (True,)),
        ("receiver_lateral_offsets_m", (0.0, 0.0)),
        ("receiver_lateral_offsets_m", (0.010001,)),
        ("receiver_depth_offsets_m", (-0.001,)),
        ("receiver_depth_offsets_m", (0.008001,)),
        ("receiver_depth_offsets_m", (float("nan"),)),
        ("receiver_depth_offsets_m", ()),
    ],
)
def test_grasp_search_domain_cannot_expand_silently(name, values):
    """Offsets are finite, unique and bounded; no unchecked target injection."""
    with pytest.raises(ValueError):
        replace(planning_request().profile, **{name: values})


def test_legacy_profile_defaults_and_simulation_only_grasp_search():
    """Existing profiles parse unchanged; search does not qualify hardware."""
    packaged = planning_request().profile
    raw = {"schema": "rocobrick-handover-profile/v1", **asdict(packaged)}
    del raw["receiver_lateral_offsets_m"]
    del raw["receiver_depth_offsets_m"]
    assert HandoverProfile.parse(raw) == packaged
    trial = simulation_profile(packaged.profile_id)
    assert trial.execution_qualified is False
    assert trial.receiver_lateral_offsets_m == (0.0, 0.004, -0.004, 0.008, -0.008)
    assert trial.receiver_depth_offsets_m == (0.0, 0.002, 0.004)
    assert trial.max_candidates == trial.max_evaluations == 1024
    assert trial.joint_hold_tolerance_rad == packaged.joint_hold_tolerance_rad


def test_tilted_offset_is_perpendicular_to_receiver_closing_axis():
    """A tilted tool must use its own tangent, not a hardcoded world axis."""
    request = offset_request()
    request = replace(
        request,
        profile=replace(request.profile, receiver_tilts_rad=(math.pi / 6,)),
    )
    for candidate in generate_candidates(request):
        axis = _quat_rotate_vector(
            candidate.receiver_contact.orientation_xyzw, (0, 1, 0)
        )
        center = compose(candidate.object_goal, Pose((0, 0, 0), (0, 0, 0, 1)))
        delta = tuple(
            a - b
            for a, b in zip(
                candidate.receiver_contact.position_xyz, center.position_xyz
            )
        )
        assert sum(a * b for a, b in zip(delta, axis)) == pytest.approx(0, abs=1e-7)


class RecordingGraspTransport:
    """Synthetic search-feedback fixture, never physical planning evidence."""

    capabilities = PLANNING_CAPABILITIES

    def __init__(self, *, accept_second=False, reachable=True):
        """Select a deterministic response and record bounded evaluations."""
        self.accept_second = accept_second
        self.reachable = reachable
        self.calls = []

    def evaluate_candidate(self, request, candidate, *, deadline_s):
        """Return only synthetic search feedback for budget/order regressions.

        Returns:
            A bound synthetic success or rejection; no actions are dispatched.
        """
        self.calls.append(candidate)
        passed = self.accept_second and len(self.calls) == 2
        return CandidateEvaluation(
            candidate.candidate_id,
            candidate.binding_sha256,
            giver_ik=self.reachable,
            receiver_ik=self.reachable,
            combined_state_valid=self.reachable,
            gripper_sweeps_valid=passed,
            approach_valid=passed,
            separation_valid=passed,
            scene_unchanged=True,
            reason=None if passed else "SYNTHETIC_GRASP_REJECTED",
            joint_path_valid=passed,
            joint_plan=joint_plan(request, candidate) if passed else None,
        )


def test_reachable_pair_is_refined_before_next_world_pose():
    """Search neighboring grasps immediately, without increasing the budget."""
    request = offset_request()
    transport = RecordingGraspTransport(accept_second=True)
    result = plan_handover(request, transport, clock=lambda: 0.0)
    assert result.passed and result.evaluated == 2
    anchor, refined = transport.calls
    assert refined.object_goal == anchor.object_goal
    assert refined.donor_goal == anchor.donor_goal
    assert refined.receiver_contact != anchor.receiver_contact
    assert refined in grasp_variants(request, anchor)


def test_refinement_is_deduplicated_and_uses_the_existing_evaluation_budget():
    """Repeated failures cannot recursively re-evaluate the same grasp."""
    request = offset_request()
    transport = RecordingGraspTransport()
    result = plan_handover(request, transport, clock=lambda: 0.0)
    ids = [c.candidate_id for c in transport.calls]
    assert not result.passed
    assert result.evaluated == len(ids) == len(set(ids))
    assert result.evaluated <= request.profile.max_evaluations
    assert len(ids) == len(generate_candidates(request))


def test_unreachable_anchor_does_not_prioritize_grasp_neighbors():
    """Changing receiver grasp cannot repair an unreachable giver target."""
    request = offset_request()
    transport = RecordingGraspTransport(reachable=False)
    plan_handover(request, transport, clock=lambda: 0.0)
    assert transport.calls[:2] == list(generate_candidates(request)[:2])


def test_refinement_rejects_a_foreign_request_binding():
    """Neighbor generation cannot silently rebind another operation's grasp."""
    request = offset_request()
    candidate = generate_candidates(request)[0]
    foreign = replace(request, operation_id="other-operation")
    with pytest.raises(ValueError, match="foreign request"):
        grasp_variants(foreign, candidate)


def test_invalid_first_lateral_offset_does_not_hide_valid_remaining_grasps():
    """The tangent-width guard skips offsets, not the entire reachable pose."""
    request = offset_request()
    request = replace(
        request,
        profile=replace(
            request.profile,
            receiver_lateral_offsets_m=(0.008, 0.0, 0.004),
        ),
    )
    candidates = generate_candidates(request)
    assert candidates
    x_values = {
        round(compose(inverse(c.object_goal), c.receiver_contact).position_xyz[0], 6)
        for c in candidates
    }
    assert x_values == {0.0, 0.004}
