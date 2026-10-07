"""Simulation trial admission does not promote hardware qualification."""

from dataclasses import replace

import pytest
from test_handover_joint_planning import paired, planning_request, snapshot

from rocobrick.adapters.moveit.handover_live import _sha, contact_geometry
from rocobrick.skills.handover_observation import require_giver_retention_sample
from rocobrick.skills.handover_profile import (
    EXECUTION_CAPABILITIES,
    resolve_handover_profile,
)
from rocobrick.skills.handover_simulation import (
    SimulationHandoverAuthorization,
    simulation_profile,
)
from rocobrick.skills.placement_geometry import compose, inverse


def authorization(request):
    """Bind a synthetic trial to its exact immutable input identities.

    Returns:
        A synthetic session-scoped authorization, not hardware qualification.
    """
    return SimulationHandoverAuthorization(
        request.scene_snapshot["session_id"],
        request.scene_snapshot["manifest_sha256"],
        request.profile.sha256,
        request.profile.runtime_model_sha256,
    )


def test_trial_keeps_installed_profile_unqualified():
    """The trial changes contact allowance but never the qualification flag."""
    profile = resolve_handover_profile("handover-bottom-coordinated-v1")
    trial = simulation_profile(profile.profile_id)
    assert profile.contact_position_tolerance_m == 0.0001
    assert trial.contact_position_tolerance_m == 0.0005
    assert profile.execution_qualified is trial.execution_qualified is False
    assert profile.sha256 != trial.sha256
    assert trial.max_candidates == trial.max_evaluations == 1024
    assert trial.planning_timeout_s == 120


@pytest.mark.parametrize(
    "field", ["session_id", "manifest_sha256", "profile", "model", "capability"]
)
def test_trial_authorization_cannot_be_borrowed(field):
    """Changed identities and missing capabilities reject before dispatch."""
    request = planning_request()
    token = authorization(request)
    profile, scene, caps = (
        request.profile,
        dict(request.scene_snapshot),
        EXECUTION_CAPABILITIES,
    )
    token.validate(profile, scene, caps)
    if field == "session_id":
        scene[field] = "foreign"
    elif field == "manifest_sha256":
        scene[field] = "a" * 64
    elif field == "profile":
        profile = replace(profile, contact_position_tolerance_m=0.0005)
    elif field == "model":
        profile = replace(profile, runtime_model_sha256="b" * 64)
    else:
        caps = caps - {"named_pad_contacts"}
    with pytest.raises(ValueError):
        token.validate(profile, scene, caps)


@pytest.mark.parametrize("excess", [0.000217, 0.0007])
def test_bounded_contact_allowance_still_rejects_bad_positions(excess):
    """Admit a bounded offset while rejecting points outside the trial allowance."""
    raw = snapshot()
    original = paired(raw)
    contact = next(
        c
        for c in raw["pad_object_contacts"]["contacts"]
        if c["robot_id"] == original.donor.robot_id
    )
    contact["position_world_m"][0] = original.donor.aabb_min[0] - excess
    measured = paired(raw)
    kwargs = dict(
        donor_window=(0.015, 0.017),
        expected_object_to_donor_tcp=compose(
            inverse(measured.donor.object_pose), measured.donor.tcp_pose
        ),
    )
    with pytest.raises(ValueError, match="outside"):
        require_giver_retention_sample(
            measured, profile=planning_request().profile, **kwargs
        )
    if excess < 0.0005:
        require_giver_retention_sample(
            measured,
            profile=simulation_profile("handover-bottom-coordinated-v1"),
            **kwargs,
        )
    else:
        with pytest.raises(ValueError, match="outside"):
            require_giver_retention_sample(
                measured,
                profile=simulation_profile("handover-bottom-coordinated-v1"),
                **kwargs,
            )


def test_allowance_cannot_become_an_unbounded_contact_bypass():
    """Keep the contact allowance explicitly bounded to half a millimetre."""
    with pytest.raises(ValueError):
        replace(planning_request().profile, contact_position_tolerance_m=0.001)


def test_local_contact_box_requires_exact_cube_source_binding():
    """Reject changed authored dimensions when old provenance is retained."""
    raw = dict(
        physical_instance_id="brick",
        prim_path="/World/brick/BodyCollider",
        source="usd_object_local_body_collider",
        manifest_sha256="a" * 64,
        minimum_m=[-0.016, -0.008, 0],
        maximum_m=[0.016, 0.008, 0.0096],
        collider_type="Cube",
    )
    raw["source_sha256"] = _sha(raw)
    scene = {"manifest_sha256": "a" * 64, "handover_contact_geometry": {"brick": raw}}
    assert contact_geometry(scene, "brick").maximum_m[2] == 0.0096
    raw["maximum_m"][2] = 0.1
    with pytest.raises(ValueError, match="authored cube"):
        contact_geometry(scene, "brick")
