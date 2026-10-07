"""Bound, coupled preparation region; MoveIt owns its sampling and IK.

Brick-local +Z is the stud face. Closing axes are side-face normals; the
receiver's palm stays in the opposite half-space, independently of world Z.
These are preparation constraints, not contact or custody authorization.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict

from rocobrick.skills.handover_planning import (
    HandoverCandidate,
    HandoverPlanningRequest,
    _hash,
)
from rocobrick.skills.handover_profile import digest, finite
from rocobrick.skills.observation import Pose
from rocobrick.skills.placement_geometry import compose, inverse, pose_error
from rocobrick.skills.targets import _quat_rotate_vector

REGION_SCHEMA = "rocobrick-handover-region/v1"


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
        ("translation_radius_m", 0.12),
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
    return {
        "schema": REGION_SCHEMA,
        "binding_sha256": request.binding_sha256,
        "donor": request.donor.robot_id,
        "receiver": request.receiver.robot_id,
        "giver_pose": asdict(observation.tcp_pose),
        "object_in_giver": asdict(object_in_giver),
        "center": [(a + b) / 2 for a, b in zip(geometry.minimum_m, geometry.maximum_m)],
        "extents": [b - a for a, b in zip(geometry.minimum_m, geometry.maximum_m)],
        "translation_radius_m": 0.12,
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
    if (
        direction_local[2] < -0.015
        or abs(closing_local[2]) > 0.015
        or (max(abs(closing_local[0]), abs(closing_local[1])) < 0.999)
    ):
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


__all__ = ["REGION_SCHEMA", "preparation_region", "candidate_from_endpoint"]
