"""Bound, coupled preparation region; MoveIt owns its sampling and IK.

Brick-local +Z is the stud face. Closing axes are side-face normals; the
receiver's palm stays in the opposite half-space, independently of world Z.
These are preparation constraints, not contact or custody authorization.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from rocobrick.skills.handover_planning import (
    HandoverCandidate,
    HandoverPlanningRequest,
    _hash,
)
from rocobrick.skills.handover_profile import digest, finite
from rocobrick.skills.observation import Pose, _pose
from rocobrick.skills.placement_geometry import compose, inverse, pose_error
from rocobrick.skills.targets import _quat_rotate_vector

REGION_SCHEMA = "rocobrick-handover-region/v1"
# Preparation admits the receiver in a 0.15 m neighborhood around the carried
# object. The post-arrival contact correction must cover that same admitted
# neighborhood; Cartesian clearance and IK still certify every intermediate
# waypoint before any contact command is sent.
LOCAL_ALIGNMENT_DISTANCE_M = 0.15
LOCAL_ALIGNMENT_ANGLE_RAD = math.pi / 2
PREPARATION_MIDPOINT_FORWARD_M = 0.08
PREPARATION_HEIGHT_M = 0.34
PREPARATION_PITCH_RAD = 0.35


def _unit(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    """Normalize one nondegenerate workspace direction.

    Returns:
        A unit-length direction vector.
    """
    norm = math.sqrt(sum(value * value for value in vector))
    if norm < 1e-8:
        raise ValueError("handover bases do not define a horizontal direction")
    return tuple(value / norm for value in vector)


@dataclass(frozen=True)
class HandoverPreparationEndpoint:
    """Actual paired arrival, not a contact or custody certificate."""

    binding_sha256: str
    giver: Pose
    receiver: Pose
    object_pose: Pose

    def __post_init__(self) -> None:
        """Keep preparation identity and complete rigid poses strict."""
        digest(self.binding_sha256, "preparation binding")
        for name in ("giver", "receiver", "object_pose"):
            _pose(asdict(getattr(self, name)), name)


def preparation_from_endpoint(
    request: HandoverPlanningRequest, giver: Pose, receiver: Pose
) -> HandoverPreparationEndpoint:
    """Admit the broad paired region without applying final grasp predicates.

    Returns:
        A bound endpoint that still requires full-path and local-handler proof.
    """
    region = preparation_region(request)
    _pose(asdict(giver), "preparation giver")
    _pose(asdict(receiver), "preparation receiver")
    observed = request.observation(request.donor.robot_id)
    displacement, angle = pose_error(observed.tcp_pose, giver)
    if displacement > region["translation_radius_m"] + 0.001:
        raise ValueError("PREPARATION_TRANSLATION_REGION_ESCAPE")
    if angle > math.sqrt(3) * region["rotation_tolerance_rad"] + 0.015:
        raise ValueError("PREPARATION_ROTATION_REGION_ESCAPE")
    obj = compose(giver, compose(inverse(observed.tcp_pose), observed.object_pose))
    if math.dist(receiver.position_xyz, obj.position_xyz) > 0.15 + 0.001:
        raise ValueError("PREPARATION_RECEIVER_REGION_ESCAPE")
    return HandoverPreparationEndpoint(request.binding_sha256, giver, receiver, obj)


def contact_from_preparation(
    request: HandoverPlanningRequest, preparation: HandoverPreparationEndpoint
) -> HandoverCandidate:
    """Choose a nearby continuous side pinch for the local receiver handler.

    This constructs geometry only, not native global IK or a trajectory.
    Local motion, both pads, collisions and physical custody must be certified.

    Returns:
        A bound final contact candidate within the local correction budget.
    """
    if preparation.binding_sha256 != request.binding_sha256:
        raise ValueError("preparation belongs to a foreign request")
    region = preparation_region(request)
    local = compose(inverse(preparation.object_pose), preparation.receiver)
    closing = _quat_rotate_vector(local.orientation_xyzw, (0, 1, 0))
    planar = math.hypot(closing[0], closing[1])
    if planar < 1e-6:
        raise ValueError("LOCAL_ALIGNMENT_CLOSING_AXIS_DEGENERATE")
    # Continuous side-plane closing direction, not four X/Y-only orientations.
    yaw = math.atan2(-closing[0], closing[1])
    tangent = (math.cos(yaw), math.sin(yaw), 0.0)
    approach = _quat_rotate_vector(local.orientation_xyzw, (0, 0, 1))
    horizontal = sum(a * b for a, b in zip(approach, tangent))
    vertical = max(0.0, approach[2])
    if math.hypot(horizontal, vertical) < 1e-6:
        horizontal = 1.0
    pitch = math.atan2(horizontal, vertical)
    orientation = compose(
        Pose((0, 0, 0), (0, 0, math.sin(yaw / 2), math.cos(yaw / 2))),
        Pose((0, 0, 0), (0, math.sin(pitch / 2), 0, math.cos(pitch / 2))),
    ).orientation_xyzw
    direction = _quat_rotate_vector(orientation, (0, 0, 1))
    # The pitch changes the receiver's local X axis. Use the final orientation
    # for the lateral placement too, otherwise candidate_from_endpoint sees a
    # different local frame and rejects its own constructed target.
    tangent = _quat_rotate_vector(orientation, (1, 0, 0))
    delta = tuple(p - c for p, c in zip(local.position_xyz, region["center"]))
    minimum = region["precontact_distance_m"] + region["contact_clearance_m"]
    depth = min(
        max(-sum(a * b for a, b in zip(delta, direction)), minimum),
        minimum + region["depth_bound_m"],
    )
    lateral = min(
        max(sum(a * b for a, b in zip(delta, tangent)), -region["lateral_bound_m"]),
        region["lateral_bound_m"],
    )
    target = compose(
        preparation.object_pose,
        Pose(
            tuple(
                c + lateral * t - depth * d
                for c, t, d in zip(region["center"], tangent, direction)
            ),
            orientation,
        ),
    )
    displacement, angle = pose_error(preparation.receiver, target)
    if displacement > LOCAL_ALIGNMENT_DISTANCE_M or angle > LOCAL_ALIGNMENT_ANGLE_RAD:
        raise ValueError("LOCAL_ALIGNMENT_BUDGET_EXCEEDED")
    return candidate_from_endpoint(request, preparation.giver, target)


def validate_region(
    region: Mapping[str, object], snapshot: Mapping[str, object]
) -> None:
    """Separate bounded region policy from the unchanged exact-pose contract."""
    required = {
        "schema",
        "binding_sha256",
        "donor",
        "receiver",
        "giver_pose",
        "object_in_giver",
        "center",
        "extents",
        "translation_radius_m",
        "rotation_tolerance_rad",
        "precontact_distance_m",
        "contact_clearance_m",
        "lateral_bound_m",
        "depth_bound_m",
        "runtime_model_sha256",
        "source_sha256",
    }
    if (
        set(region) != required
        or region["schema"] != REGION_SCHEMA
        or ({region["donor"], region["receiver"]} != {"piper_0", "piper_1"})
    ):
        raise ValueError("invalid coupled HANDOVER preparation region")
    source = snapshot.get("tcp_provenance", {})
    for name in ("binding_sha256", "runtime_model_sha256", "source_sha256"):
        digest(region[name], name)
        if name != "binding_sha256" and region[name] != source.get(name):
            raise ValueError("region calibration differs from captured snapshot")
    for name in ("giver_pose", "object_in_giver"):
        raw = region[name]
        if not isinstance(raw, Mapping) or set(raw) != {
            "position_xyz",
            "orientation_xyzw",
        }:
            raise ValueError("region requires complete rigid poses")
        for key, count in (("position_xyz", 3), ("orientation_xyzw", 4)):
            if not isinstance(raw[key], (list, tuple)) or len(raw[key]) != count:
                raise ValueError("region pose vector is incomplete")
            for value in raw[key]:
                finite(value, name)
        if abs(sum(v * v for v in raw["orientation_xyzw"]) - 1) > 1e-6:
            raise ValueError("region quaternion is not normalized")
    for name in ("center", "extents"):
        if not isinstance(region[name], (list, tuple)) or len(region[name]) != 3:
            raise ValueError("region object box is incomplete")
        for value in region[name]:
            finite(value, name, positive=name == "extents")
    for name, maximum in (
        ("translation_radius_m", 0.24),
        ("rotation_tolerance_rad", 1.0),
        ("precontact_distance_m", 0.08),
        ("contact_clearance_m", 0.01),
        ("lateral_bound_m", 0.004),
        ("depth_bound_m", 0.004),
    ):
        if finite(region[name], name, positive=True) > maximum:
            raise ValueError("region exceeds preparation policy")


def preparation_region(request: HandoverPlanningRequest) -> dict[str, object]:
    """Anchor a continuous neighborhood to the measured lift, not a height grid.

    Returns:
        Immutable-input-derived region metadata for the combined MoveIt sampler.
    """
    observation = request.observation(request.donor.robot_id)
    geometry = request.object_geometry
    object_in_giver = compose(inverse(observation.tcp_pose), observation.object_pose)
    # Put the preparation endpoint near the horizontal midpoint of the two
    # bases, with a small forward offset into the shared workspace.  The height
    # is deliberately above the post-lift height rather than anchored to the
    # giver's current TCP position.  The object-relative
    # transform remains exact, so the brick follows the giver without a
    # release.  The bounded region around this pose lets MoveIt choose a
    # collision-free joint configuration and a receiver approach orientation.
    donor_base = request.donor.base.position_xyz
    receiver_base = request.receiver.base.position_xyz
    along = _unit(
        (
            receiver_base[0] - donor_base[0],
            receiver_base[1] - donor_base[1],
            0.0,
        )
    )
    forward = (-along[1], along[0], 0.0)
    facing = tuple(
        sum(
            _quat_rotate_vector(ctx.base.orientation_xyzw, (1.0, 0.0, 0.0))[i]
            for ctx in (request.donor, request.receiver)
        )
        for i in range(3)
    )
    if sum(a * b for a, b in zip(forward, facing)) < 0.0:
        forward = tuple(-value for value in forward)
    midpoint = tuple(
        (donor_base[i] + receiver_base[i]) / 2.0 for i in range(3)
    )
    target_position = tuple(
        midpoint[i] + PREPARATION_MIDPOINT_FORWARD_M * forward[i]
        if i < 2
        else PREPARATION_HEIGHT_M
        for i in range(3)
    )
    half_pitch = PREPARATION_PITCH_RAD / 2.0
    target_orientation = compose(
        observation.tcp_pose,
        Pose((0.0, 0.0, 0.0), (0.0, math.sin(half_pitch), 0.0, math.cos(half_pitch))),
    )
    giver_target = Pose(target_position, target_orientation.orientation_xyzw)
    return {
        "schema": REGION_SCHEMA,
        "binding_sha256": request.binding_sha256,
        "donor": request.donor.robot_id,
        "receiver": request.receiver.robot_id,
        "giver_pose": asdict(giver_target),
        "object_in_giver": asdict(object_in_giver),
        "center": [(a + b) / 2 for a, b in zip(geometry.minimum_m, geometry.maximum_m)],
        "extents": [b - a for a, b in zip(geometry.minimum_m, geometry.maximum_m)],
        # Keep MoveIt sampling around the midpoint endpoint;
        # the endpoint itself is not giver-local, while preparation can
        # sample small collision-free variations.
        "translation_radius_m": 0.14,
        "rotation_tolerance_rad": 1.0,
        "precontact_distance_m": request.profile.precontact_distance_m,
        "contact_clearance_m": request.profile.contact_clearance_m,
        "lateral_bound_m": min(
            0.004,
            min(b - a for a, b in zip(geometry.minimum_m[:2], geometry.maximum_m[:2]))
            / 4,
        ),
        "depth_bound_m": 0.004,
        "runtime_model_sha256": request.donor.provenance.runtime_model_sha256,
        "source_sha256": request.donor.provenance.source_sha256,
    }


def candidate_from_endpoint(
    request: HandoverPlanningRequest, giver: Pose, receiver: Pose
) -> HandoverCandidate:
    """Reconstruct contact from actual returned FK, and reject region escape.

    Returns:
        A contact candidate bound to the original input and actual endpoint.
    """
    region = preparation_region(request)
    observed = request.observation(request.donor.robot_id)
    displacement, rotation = pose_error(observed.tcp_pose, giver)
    if displacement > region["translation_radius_m"] + 0.001:
        raise ValueError("PREPARATION_TRANSLATION_REGION_ESCAPE")
    # MoveIt uses axis-wise rotational bounds. This conservative aggregate
    # check is additional to its complete endpoint constraint validation.
    if rotation > math.sqrt(3) * region["rotation_tolerance_rad"] + 0.015:
        raise ValueError("PREPARATION_ROTATION_REGION_ESCAPE")
    held = compose(inverse(observed.object_pose), observed.tcp_pose)
    obj = compose(giver, inverse(held))
    local = compose(inverse(obj), receiver)
    direction_local = _quat_rotate_vector(local.orientation_xyzw, (0, 0, 1))
    closing_local = _quat_rotate_vector(local.orientation_xyzw, (0, 1, 0))
    # No world-up gate: tool +Z may point in any world direction. Require
    # a side-face pinch and a receiver palm outside the exposed stud half-space.
    if direction_local[2] < -0.015 or abs(closing_local[2]) > 0.015:
        raise ValueError("STUD_FACE_OR_SIDE_PINCH_CONSTRAINT")
    direction = _quat_rotate_vector(receiver.orientation_xyzw, (0, 0, 1))
    tangent_local = _quat_rotate_vector(local.orientation_xyzw, (1, 0, 0))
    delta = tuple(p - c for p, c in zip(local.position_xyz, region["center"]))
    depth = -sum(d * v for d, v in zip(direction_local, delta))
    lateral = sum(d * v for d, v in zip(tangent_local, delta))
    closing_offset = abs(sum(d * v for d, v in zip(closing_local, delta)))
    minimum = region["precontact_distance_m"] + region["contact_clearance_m"]
    tolerance = request.profile.relative_position_tolerance_m
    if not minimum - tolerance <= depth <= minimum + region[
        "depth_bound_m"
    ] + tolerance or (
        abs(lateral) > region["lateral_bound_m"] + tolerance
        or closing_offset > tolerance
    ):
        raise ValueError("COUPLED_PRECONTACT_REGION_ESCAPE")
    contact = Pose(
        tuple(
            p + request.profile.precontact_distance_m * d
            for p, d in zip(receiver.position_xyz, direction)
        ),
        receiver.orientation_xyzw,
    )
    span = sum(abs(v) * size for v, size in zip(closing_local, region["extents"]))
    fields = {
        "binding_sha256": request.binding_sha256,
        "object_goal": asdict(obj),
        "object_to_donor_tcp": asdict(held),
        "donor_goal": asdict(giver),
        "receiver_contact": asdict(contact),
        "receiver_precontact": asdict(receiver),
        "approach_direction_world": direction,
        "receiver_span_m": span,
    }
    return HandoverCandidate(
        _hash(fields),
        request.binding_sha256,
        obj,
        held,
        giver,
        contact,
        receiver,
        direction,
        span,
    )


__all__ = [
    "REGION_SCHEMA",
    "HandoverPreparationEndpoint",
    "preparation_region",
    "preparation_from_endpoint",
    "contact_from_preparation",
    "candidate_from_endpoint",
]
