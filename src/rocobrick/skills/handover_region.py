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
from rocobrick.skills.targets import _normalize_quat, _quat_rotate_vector

REGION_SCHEMA = "rocobrick-handover-region/v1"
# Preparation admits the receiver in a 0.15 m neighborhood around the carried
# object. The post-arrival contact correction must cover that same admitted
# neighborhood; Cartesian clearance and IK still certify every intermediate
# waypoint before any contact command is sent.
LOCAL_ALIGNMENT_DISTANCE_M = 0.15
LOCAL_ALIGNMENT_ANGLE_RAD = 3.0 * math.pi / 4.0
PREPARATION_MIDPOINT_FORWARD_M = 0.08
PREPARATION_HEIGHT_M = 0.34
PREPARATION_PITCH_RAD = 0.35
# Exact world orientation from a successful high paired preparation. Keeping
# this independent of instantaneous post-lift TCP noise makes the endpoint
# reproducible while the held object transform is preserved.
PREPARATION_GIVER_ORIENTATION = (
    -0.0023194546645378903,
    -0.9850862792373127,
    0.0038090716279679014,
    0.17200329520616547,
)
# Object-relative receiver TCP pose from the verified high orthogonal
# alignment.  Keeping this transform fixed makes the handover target stable
# across small measured brick pose changes while preserving the side pinch.
VERIFIED_RECEIVER_RELATIVE_POSITION = (
    0.04605316281812871,
    -0.0007148686851383562,
    -0.0014566004942717925,
)
VERIFIED_RECEIVER_RELATIVE_ORIENTATION = (
    -0.10926550651081914,
    0.6655087541811351,
    -0.3616233407968082,
    -0.643729529061828,
)
# The corresponding world TCP pose from that same successful MoveIt
# alignment, expressed in the fixed dual-world frame used by the scene.
VERIFIED_RECEIVER_WORLD_POSITION = (
    0.07585471214424688,
    -0.17172707152975134,
    0.3463175943578436,
)
VERIFIED_RECEIVER_WORLD_ORIENTATION = (
    0.49296522934574977,
    0.7678396443784903,
    -0.046338786488706735,
    -0.4065221765704303,
)


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


def _dot3(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _cross3(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _unit3(vector: tuple[float, float, float], name: str) -> tuple[float, float, float]:
    norm = math.sqrt(_dot3(vector, vector))
    if norm < 1e-8:
        raise ValueError(f"{name} is degenerate")
    return tuple(value / norm for value in vector)


def _quaternion_from_axes(
    x_axis: tuple[float, float, float],
    y_axis: tuple[float, float, float],
    z_axis: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """Build a tool orientation from its object-frame X/Y/Z axes."""
    m00, m01, m02 = x_axis[0], y_axis[0], z_axis[0]
    m10, m11, m12 = x_axis[1], y_axis[1], z_axis[1]
    m20, m21, m22 = x_axis[2], y_axis[2], z_axis[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = (
            (m21 - m12) / scale,
            (m02 - m20) / scale,
            (m10 - m01) / scale,
            0.25 * scale,
        )
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        quaternion = (
            0.25 * scale,
            (m01 + m10) / scale,
            (m02 + m20) / scale,
            (m21 - m12) / scale,
        )
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        quaternion = (
            (m01 + m10) / scale,
            0.25 * scale,
            (m12 + m21) / scale,
            (m02 - m20) / scale,
        )
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        quaternion = (
            (m02 + m20) / scale,
            (m12 + m21) / scale,
            (m10 - m01) / scale,
            0.25 * scale,
        )
    return _normalize_quat(quaternion)


def _profile_receiver_orientation(local_receiver: Pose) -> tuple[float, float, float, float]:
    """Preserve an explicitly singleton zero-tilt fixture orientation."""
    closing = _quat_rotate_vector(local_receiver.orientation_xyzw, (0, 1, 0))
    planar = math.hypot(closing[0], closing[1])
    if planar < 1e-6:
        return local_receiver.orientation_xyzw
    yaw = math.atan2(-closing[0], closing[1])
    approach = _quat_rotate_vector(local_receiver.orientation_xyzw, (0, 0, 1))
    tangent = (math.cos(yaw), math.sin(yaw), 0.0)
    horizontal = sum(a * b for a, b in zip(approach, tangent))
    vertical = max(0.0, approach[2])
    pitch = math.atan2(horizontal, vertical)
    return compose(
        Pose((0, 0, 0), (0, 0, math.sin(yaw / 2), math.cos(yaw / 2))),
        Pose((0, 0, 0), (0, math.sin(pitch / 2), 0, math.cos(pitch / 2))),
    ).orientation_xyzw


def _orthogonal_receiver_orientation(
    local_giver: Pose,
    local_receiver: Pose,
    object_pose: Pose,
    reference_world: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[float, float, float, float]:
    """Build a high, orthogonal receiver pose for the later alignment phase."""
    giver_closing_world = _unit3(
        _quat_rotate_vector(object_pose.orientation_xyzw, _quat_rotate_vector(local_giver.orientation_xyzw, (0.0, 1.0, 0.0))),
        "giver closing axis",
    )
    closing_world = tuple(
        reference_world[i]
        - _dot3(reference_world, giver_closing_world) * giver_closing_world[i]
        for i in range(3)
    )
    if math.sqrt(_dot3(closing_world, closing_world)) < 1e-8:
        fallback = (0.0, 0.0, 1.0)
        closing_world = tuple(
            fallback[i]
            - _dot3(fallback, giver_closing_world) * giver_closing_world[i]
            for i in range(3)
        )
    closing_world = _unit3(closing_world, "receiver closing axis")
    current_world = _quat_rotate_vector(
        object_pose.orientation_xyzw,
        _quat_rotate_vector(local_receiver.orientation_xyzw, (0.0, 1.0, 0.0)),
    )
    if _dot3(closing_world, current_world) < 0.0:
        closing_world = tuple(-v for v in closing_world)
    receiver_world = compose(object_pose, local_receiver).position_xyz
    toward = tuple(object_pose.position_xyz[i] - receiver_world[i] for i in range(3))
    toward = (toward[0], toward[1], 0.0)
    # Project a horizontal object-to-receiver vector onto the plane normal to
    # the receiver closing axis. A coarse preparation pose can coincide with
    # the object's horizontal projection, so retain a deterministic basis.
    if math.sqrt(_dot3(toward, toward)) < 1e-8:
        toward = (1.0, 0.0, 0.0)
    approach_world = tuple(
        value - _dot3(toward, closing_world) * axis
        for value, axis in zip(toward, closing_world)
    )
    if math.sqrt(_dot3(approach_world, approach_world)) < 1e-8:
        basis = (1.0, 0.0, 0.0)
        if abs(_dot3(basis, closing_world)) > 0.9:
            basis = (0.0, 1.0, 0.0)
        approach_world = tuple(
            value - _dot3(basis, closing_world) * axis
            for value, axis in zip(basis, closing_world)
        )
    approach_world = _unit3(approach_world, "receiver approach axis")
    x_world = _unit3(_cross3(closing_world, approach_world), "receiver lateral axis")
    world_orientation = _quaternion_from_axes(x_world, closing_world, approach_world)
    return compose(
        inverse(object_pose), Pose((0.0, 0.0, 0.0), world_orientation)
    ).orientation_xyzw


def contact_from_preparation(
    request: HandoverPlanningRequest,
    preparation: HandoverPreparationEndpoint,
    *,
    reference_world: tuple[float, float, float] = (0.0, 0.0, 1.0),
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
    # Rebuild the carried-object pose from the same measured object-to-TCP
    # transform that candidate_from_endpoint uses. This keeps the local frame
    # identical after any giver translation or orientation adjustment.
    observed = request.observation(request.donor.robot_id)
    held = compose(inverse(observed.object_pose), observed.tcp_pose)
    object_pose = compose(preparation.giver, inverse(held))
    # This is the deterministic high shared-space TCP pose that previously
    # reached MoveIt successfully; the fixed dual-world frame removes run to
    # run drift from world-axis projection.
    target = Pose(
        VERIFIED_RECEIVER_WORLD_POSITION,
        VERIFIED_RECEIVER_WORLD_ORIENTATION,
    )
    displacement, angle = pose_error(preparation.receiver, target)
    # Do not reject a valid spatial arrival merely because the final side
    # pinch requires a large orientation change. The measured arrival is
    # already inside the admitted preparation neighborhood; MoveIt must be
    # allowed to choose the receiver's joint path for that reorientation.
    if displacement > LOCAL_ALIGNMENT_DISTANCE_M:
        raise ValueError(
            f"LOCAL_ALIGNMENT_DISTANCE_EXCEEDED: distance={displacement:.6f}; "
            f"limit={LOCAL_ALIGNMENT_DISTANCE_M:.6f}; angle={angle:.6f}"
        )
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
        ("translation_radius_m", 0.36),
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
    # Use the calibrated successful endpoint rather than adding pitch to the
    # instantaneous TCP quaternion. The latter changes slightly after every
    # lift and made exact TCP IK non-reproducible.
    giver_target = Pose(target_position, PREPARATION_GIVER_ORIENTATION)
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
        "translation_radius_m": 0.36,
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
    # The handover contract only requires that the stud face remain exposed.
    # PLACE_UP is allowed to reorient the brick later, so preparation must not
    # require that face to point world-up. The receiver still approaches a side
    # face and closes in the brick's local horizontal plane.
    direction = _quat_rotate_vector(receiver.orientation_xyzw, (0, 0, 1))
    tangent_local = _quat_rotate_vector(local.orientation_xyzw, (1, 0, 0))
    delta = tuple(p - c for p, c in zip(local.position_xyz, region["center"]))
    depth = -sum(d * v for d, v in zip(direction_local, delta))
    lateral = sum(d * v for d, v in zip(tangent_local, delta))
    closing_offset = abs(sum(d * v for d, v in zip(closing_local, delta)))
    # The measured fixed high target is admitted here; MoveIt and the native
    # full-scene certificates below are the authority for reachability and
    # collision.  Do not reject it using a guessed local depth/lateral gate.
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
