"""Continuous preparation region, calibrated transport and actual FK binding."""

import copy
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_coordinated_ros_runtime import Factory, runtime
from test_handover_joint_planning import planning_request
from test_handover_scene_planner import native_provider
from test_readonly_handover_transport import planning_session, readonly_request

from rocobrick.adapters.moveit.readonly_combined_plan import (
    perform_readonly_combined_plan,
)
from rocobrick.skills.handover_region import (
    candidate_from_endpoint,
    preparation_region,
    validate_region,
)
from rocobrick.skills.observation import Pose
from rocobrick.skills.placement_geometry import compose
from rocobrick.skills.targets import _quat_rotate_vector


class FloatOnlyMessage(SimpleNamespace):
    """Enforce generated geometry_msgs scalar setters in the CPU regression."""

    def __setattr__(self, name, value):
        """Reject numeric values that generated ROS double setters reject."""
        if name in {"x", "y", "z", "w", "weight"} or name.endswith("tolerance"):
            assert isinstance(value, float), (
                f"The '{name}' field must be of type 'float'"
            )
        if name == "dimensions":
            assert all(isinstance(item, float) for item in value)
        super().__setattr__(name, value)


class StrictRegionFactory(Factory):
    """Keep the existing transport fake but make ROS numerical setters strict."""

    def pose_factory(self):
        """Build float-only position and quaternion fields.

        Returns:
            A pose double with the generated message's scalar type boundary.
        """
        return SimpleNamespace(
            position=FloatOnlyMessage(), orientation=FloatOnlyMessage()
        )

    def orientation_constraint_factory(self):
        """Build float-only orientation and tolerance fields.

        Returns:
            A constraint double with a strict quaternion setter.
        """
        return FloatOnlyMessage(
            header=SimpleNamespace(frame_id=""), orientation=FloatOnlyMessage()
        )

    def solid_primitive_factory(self):
        """Build float-only primitive dimensions.

        Returns:
            A primitive double enforcing its numeric sequence type boundary.
        """
        return FloatOnlyMessage(SPHERE=2, type=0)


@pytest.mark.parametrize("integer_inputs", [False, True])
def test_region_converts_every_ros_double_without_mutating_input(integer_inputs):
    """Integer identity quaternions and valid integer data serialize as doubles."""
    region = preparation_region(planning_request())
    if integer_inputs:
        region["giver_pose"] = {
            "position_xyz": [0, 0, 0],
            "orientation_xyzw": [0, 0, 0, 1],
        }
        region["object_in_giver"] = {
            "position_xyz": [0, 0, 0],
            "orientation_xyzw": [0, 0, 0, 1],
        }
        region["rotation_tolerance_rad"] = 1
    original = copy.deepcopy(region)
    base = readonly_request()
    query = replace(
        base,
        targets={r: {"tcp_region": region} for r in base.participants},
        scene_snapshot={
            **base.scene_snapshot,
            "tcp_provenance": {
                key: region[key] for key in ("runtime_model_sha256", "source_sha256")
            },
        },
    )
    goal = runtime()._build_goal(query, StrictRegionFactory())
    assert goal.request.pipeline_id == "rocobrick_region"
    constraints = goal.request.goal_constraints[0]
    for pc, oc in zip(
        constraints.position_constraints, constraints.orientation_constraints
    ):
        assert all(
            isinstance(getattr(oc.orientation, key), float)
            for key in ("x", "y", "z", "w")
        )
        assert all(
            isinstance(
                getattr(pc.constraint_region.primitive_poses[0].position, key), float
            )
            for key in ("x", "y", "z")
        )
    assert region == original
    assert goal.planning_options.plan_only is True


def test_strict_message_double_reproduces_original_integer_assignment_error():
    """The regression double rejects the precise scalar type from the real run."""
    message = FloatOnlyMessage()
    with pytest.raises(AssertionError, match="'x'.*float"):
        message.x = 0
    message.x = 0.0


def paired_endpoint(request, pitch=0.0):
    """Return analytic object-relative side/oblique preparation geometry.

    Returns:
        Giver and receiver poses, without any planner or simulator.
    """
    region = preparation_region(request)
    observed = request.observation(request.donor.robot_id)
    orientation = (0, math.sin(pitch / 2), 0, math.cos(pitch / 2))
    distance = region["precontact_distance_m"] + region["contact_clearance_m"]
    local = Pose(
        tuple(
            region["center"][i] - distance * d
            for i, d in enumerate((math.sin(pitch), 0, math.cos(pitch)))
        ),
        orientation,
    )
    return observed.tcp_pose, compose(observed.object_pose, local)


@pytest.mark.parametrize("pitch", [-math.pi / 2, -0.6, 0, 0.8, math.pi / 2])
def test_region_accepts_continuous_oblique_and_sideways_approaches(pitch):
    """Sideways is valid even when its world-up component is zero."""
    request = planning_request()
    candidate = candidate_from_endpoint(request, *paired_endpoint(request, pitch))
    assert candidate.binding_sha256 == request.binding_sha256
    assert math.dist(
        candidate.receiver_contact.position_xyz,
        candidate.receiver_precontact.position_xyz,
    ) == pytest.approx(0.04)


def test_region_rejects_approach_covering_stud_face_and_uncoupled_receiver():
    """Broad region constraints do not authorize arbitrary closing geometry."""
    request = planning_request()
    with pytest.raises(ValueError, match="STUD_FACE"):
        candidate_from_endpoint(request, *paired_endpoint(request, math.pi))
    giver, receiver = paired_endpoint(request)
    with pytest.raises(ValueError, match="COUPLED"):
        candidate_from_endpoint(
            request, giver, replace(receiver, position_xyz=(0.5, 0.5, 0.5))
        )


def test_region_accepts_tool_axis_pointing_down_in_world_if_stud_frame_allows():
    """World Z is not a handover predicate: the brick may be rotated."""
    request = planning_request()
    snapshot = copy.deepcopy(request.scene_snapshot)
    snapshot["bricks"]["brick-1"]["pose"]["orientation_xyzw"] = [1, 0, 0, 0]
    request = replace(request, scene_snapshot=snapshot)
    candidate = candidate_from_endpoint(request, *paired_endpoint(request))
    assert candidate.approach_direction_world[2] == pytest.approx(-1)


def test_repeated_region_queries_have_unique_ids_without_changing_region(tmp_path):
    """The persistent read-only mailbox consumes each query identity once."""
    provider, captured, seed, recorded = native_provider(tmp_path)
    region = preparation_region(captured)
    binding = captured.binding_sha256
    for _ in range(2):
        assert (
            provider.planner.plan_poses(
                captured, seed, captured.start_state, deadline_s=10, region=region
            )
            is not None
        )
    first, second = recorded.requests
    assert first.coordination_id != second.coordination_id
    assert first.targets == second.targets
    assert captured.binding_sha256 == binding


def test_region_transport_is_one_coupled_goal_and_requires_native_admission():
    """One plan-only coupled goal retains post-plan FK admission."""
    captured = planning_request()
    region = preparation_region(captured)
    provenance = {key: region[key] for key in ("runtime_model_sha256", "source_sha256")}
    base = readonly_request()
    query = replace(
        base,
        targets={r: {"tcp_region": region} for r in base.participants},
        scene_snapshot={**base.scene_snapshot, "tcp_provenance": provenance},
    )
    goal = runtime()._build_goal(query, Factory())
    constraints = goal.request.goal_constraints[0]
    assert constraints.name.startswith("rocobrick-handover-region/v1;")
    assert len(constraints.position_constraints) == 2
    assert (
        constraints.position_constraints[1].header.frame_id
        == captured.donor.robot_id + "_rocobrick_tcp"
    )
    assert goal.planning_options.plan_only is True
    result = perform_readonly_combined_plan(planning_session(), query)
    assert result.passed, result.error
    assert result.lifecycle["goal_mode"] == "tcp_region"
    assert result.lifecycle["native_fk_validation_required"] is True


@pytest.mark.parametrize(
    "change", ["radius", "calibration", "schema", "partial", "quaternion"]
)
def test_region_policy_cannot_weaken_exact_pose_transport(change):
    """Malformed and expanded regions fail before the MoveGroup query."""
    region = preparation_region(planning_request())
    snapshot = {
        "tcp_provenance": {
            key: region[key] for key in ("runtime_model_sha256", "source_sha256")
        }
    }
    altered = copy.deepcopy(region)
    if change == "radius":
        altered["translation_radius_m"] = 2.0
    elif change == "calibration":
        altered["source_sha256"] = "0" * 64
    elif change == "schema":
        altered["schema"] = "other"
    elif change == "partial":
        altered.pop("object_in_giver")
    else:
        altered["object_in_giver"]["orientation_xyzw"] = [0, 0, 0, 2]
    with pytest.raises(ValueError):
        validate_region(altered, snapshot)


@pytest.mark.parametrize(
    "quaternion",
    [
        (
            -0.011421002447605133,
            0.001834752270951867,
            0.004789809696376324,
            0.9999216198921204,
        ),
        (
            -0.011421273462474346,
            0.0018344814889132977,
            0.004789730999618769,
            0.9999216198921204,
        ),
        (
            -0.011421500705182552,
            0.0018346573924645782,
            0.0047898320481181145,
            0.9999216198921204,
        ),
    ],
)
def test_region_seed_accepts_recorded_lift_rotation_without_weakening_axis_gate(
    quaternion,
):
    """Use the production seed formula with recorded raw rotation, no planner."""
    request = planning_request()
    snapshot = copy.deepcopy(request.scene_snapshot)
    snapshot["bricks"][request.physical_instance_id]["pose"]["orientation_xyzw"] = list(
        quaternion
    )
    request = replace(request, scene_snapshot=snapshot)
    original = copy.deepcopy(snapshot)
    binding = request.binding_sha256
    observed = request.observation(request.donor.robot_id)
    region = preparation_region(request)
    direction = _quat_rotate_vector(observed.object_pose.orientation_xyzw, (0, 0, 1))
    center = compose(observed.object_pose, Pose(tuple(region["center"]), (0, 0, 0, 1)))
    receiver = Pose(
        tuple(
            c - d * (region["precontact_distance_m"] + region["contact_clearance_m"])
            for c, d in zip(center.position_xyz, direction)
        ),
        observed.object_pose.orientation_xyzw,
    )
    candidate = candidate_from_endpoint(request, observed.tcp_pose, receiver)
    assert candidate.receiver_precontact.orientation_xyzw == quaternion
    assert candidate.receiver_contact.orientation_xyzw == quaternion
    assert request.scene_snapshot == original
    assert request.binding_sha256 == binding
    tilted = compose(
        receiver,
        Pose(
            (0, 0, 0),
            (
                0,
                0,
                math.sin(1e-5 / 2),
                math.cos(1e-5 / 2),
            ),
        ),
    )
    with pytest.raises(ValueError, match="precontact does not follow"):
        replace(candidate, receiver_precontact=tilted)
