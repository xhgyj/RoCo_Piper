"""Deterministic paired TCP goals in a shared, table-relative handover zone.

Geometry is an inexpensive proposal, not a reachability certificate. The
existing combined MoveIt planner validates each proposal before any dispatch.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from rocobrick.skills.handover_planning import (
    HandoverPlanningRequest,
    _rotation,
    calibrated_target,
)
from rocobrick.skills.observation import Pose
from rocobrick.skills.placement_geometry import compose, inverse, pose_error
from rocobrick.skills.targets import _normalize_quat, _quat_rotate_vector


@dataclass(frozen=True)
class HandoverZone:
    """Horizontal box around the base midpoint, with live tabletop height."""

    midpoint: tuple[float, float, float]
    along: tuple[float, float, float]
    forward: tuple[float, float, float]
    tabletop_z_m: float
    horizontal_half_width_m: float = 0.10
    height_range_m: tuple[float, float] = (0.15, 0.25)

    def evidence(self) -> dict[str, object]:
        """Describe world-frame geometry without claiming IK feasibility."""
        return {"frame_id": "dual_world", **asdict(self)}


@dataclass(frozen=True)
class ZoneTarget:
    """A rigid giver grasp and a separate receiver precontact pose."""

    center: tuple[float, float, float]
    object_goal: Pose
    giver_goal: Pose
    receiver_precontact: Pose
    receiver_contact: Pose
    giver_approach: tuple[float, float, float]
    receiver_approach: tuple[float, float, float]
    score: float

    def targets(self, request: HandoverPlanningRequest) -> dict[str, dict[str, object]]:
        """Convert each world goal with its own base and TCP provenance."""
        return {
            request.donor.robot_id: dict(
                calibrated_target(self.giver_goal, request.donor, request.profile)
            ),
            request.receiver.robot_id: dict(
                calibrated_target(
                    self.receiver_precontact, request.receiver, request.profile
                )
            ),
        }


def _unit(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm < 1e-8:
        raise ValueError("handover geometry contains a degenerate direction")
    return tuple(value / norm for value in vector)


def _dot(left, right) -> float:
    return sum(a * b for a, b in zip(left, right))


def _turn(first, last) -> Pose:
    """Shortest rotation taking the measured tool axis to the proposed axis."""
    cosine = _dot(first, last)
    cross = (
        first[1] * last[2] - first[2] * last[1],
        first[2] * last[0] - first[0] * last[2],
        first[0] * last[1] - first[1] * last[0],
    )
    if cosine < -1 + 1e-8:
        axis = _unit((first[1], -first[0], 0.0))
        return Pose((0.0, 0.0, 0.0), (*axis, 0.0))
    return Pose((0.0, 0.0, 0.0), _normalize_quat((*cross, 1.0 + cosine)))


def zone_from_request(request: HandoverPlanningRequest) -> HandoverZone:
    """Read the tabletop collider; never confuse base height with table height."""
    bases = (request.donor.base.position_xyz, request.receiver.base.position_xyz)
    along = _unit((bases[1][0] - bases[0][0], bases[1][1] - bases[0][1], 0.0))
    forward = (-along[1], along[0], 0.0)
    facing = tuple(
        sum(_quat_rotate_vector(ctx.base.orientation_xyzw, (1.0, 0.0, 0.0))[i]
            for ctx in (request.donor, request.receiver))
        for i in range(3)
    )
    if _dot(forward, facing) < 0:
        forward = tuple(-v for v in forward)
    colliders = request.scene_snapshot.get("environment_colliders", {})
    table_heights = [
        float(row["aabb_max"][2])
        for path, row in colliders.items()
        if path.rsplit("/", 1)[-1] == "tableTopActor"
    ]
    if len(table_heights) != 1 or not math.isfinite(table_heights[0]):
        raise ValueError("handover zone needs the measured tabletop collider")
    return HandoverZone(
        tuple((a + b) / 2 for a, b in zip(*bases)),
        along, forward, table_heights[0],
    )


def generate_zone_targets(
    request: HandoverPlanningRequest,
) -> tuple[HandoverZone, tuple[ZoneTarget, ...]]:
    """Generate a small reproducible family anchored to this measured lift.

    Both tool axes point into the object from their respective base sides.
    Giver wrist rotation preserves the measured object-to-TCP transform. The
    receiver closing axis remains in the object's local side-face plane.
    """
    zone = zone_from_request(request)
    observed = request.observation(request.donor.robot_id)
    receiver = request.observation(request.receiver.robot_id)
    geometry = request.object_geometry
    local_center = tuple((a + b) / 2 for a, b in zip(geometry.minimum_m, geometry.maximum_m))
    measured_center = compose(observed.object_pose, Pose(local_center, (0, 0, 0, 1))).position_xyz
    held = compose(inverse(observed.object_pose), observed.tcp_pose)
    measured_axis = _quat_rotate_vector(observed.tcp_pose.orientation_xyzw, (0, 0, 1))
    preferred_height = min(0.23, max(0.17, measured_center[2] - zone.tabletop_z_m + 0.06))
    preferred_along = min(0.05, max(-0.05, _dot(
        tuple(a - b for a, b in zip(measured_center, zone.midpoint)), zone.along
    )))
    centers = []
    for along in (preferred_along, 0.0, -preferred_along):
        for height in (preferred_height, 0.20, 0.23):
            center = tuple(
                zone.midpoint[i] + along * zone.along[i] + 0.08 * zone.forward[i]
                if i < 2 else zone.tabletop_z_m + height
                for i in range(3)
            )
            if center not in centers:
                centers.append(center)
    proposals = []
    for center in centers:
        donor_inward = _unit((center[0] - request.donor.base.position_xyz[0],
                              center[1] - request.donor.base.position_xyz[1], 0.0))
        receiver_inward = _unit((center[0] - request.receiver.base.position_xyz[0],
                                 center[1] - request.receiver.base.position_xyz[1], 0.0))
        # Preserve the measured object/wrist vertical convention. Only yaw the
        # receiver around the object normal; horizontal tool axes caused the
        # previous receiver IK family to fail for every zone height.
        object_orientation = observed.object_pose.orientation_xyzw
        object_goal = Pose(
            tuple(center[i] - _quat_rotate_vector(object_orientation, local_center)[i]
                  for i in range(3)), object_orientation
        )
        giver_goal = compose(object_goal, held)
        orientation_candidates = [
            ("receiver-measured", receiver.tcp_pose.orientation_xyzw, 0.0),
        ]
        for yaw in (math.radians(90), math.radians(-90), 0.0):
            local_tool = _rotation("z", yaw)
            tool = compose(object_goal, local_tool)
            orientation_candidates.append(("object-derived", tool.orientation_xyzw, 0.04))
        for _, tool_orientation, orientation_penalty in orientation_candidates:
            contact = Pose(tuple(p - request.profile.contact_clearance_m * d
                                 for p, d in zip(center, receiver_inward)), tool_orientation)
            precontact = Pose(tuple(p - request.profile.precontact_distance_m * d
                                    for p, d in zip(contact.position_xyz, receiver_inward)), tool_orientation)
            giver_distance, giver_rotation = pose_error(observed.tcp_pose, giver_goal)
            receiver_distance, receiver_rotation = pose_error(receiver.tcp_pose, precontact)
            midpoint_distance = math.hypot(center[0] - zone.midpoint[0], center[1] - zone.midpoint[1])
            score = (giver_distance + 0.35 * receiver_distance + 0.03 * giver_rotation
                     + 0.01 * receiver_rotation + 0.5 * midpoint_distance
                     + orientation_penalty)
            proposals.append(ZoneTarget(center, object_goal, giver_goal, precontact,
                                        contact, donor_inward, receiver_inward, score))
    return zone, tuple(sorted(proposals, key=lambda value: value.score))


def generate_preparation_targets(
    request: HandoverPlanningRequest,
) -> tuple[HandoverZone, tuple[ZoneTarget, ...]]:
    """Generate paired post-lift preparation targets.

    The giver carries the measured grasp to a larger central move and changes
    the object's wrist orientation. The receiver approaches using its measured
    TCP orientation, which gives MoveIt a reachable preparation problem before
    any contact-specific orientation is requested.
    """
    zone = zone_from_request(request)
    observed = request.observation(request.donor.robot_id)
    receiver_observed = request.observation(request.receiver.robot_id)
    geometry = request.object_geometry
    local_center = tuple(
        (a + b) / 2 for a, b in zip(geometry.minimum_m, geometry.maximum_m)
    )
    measured_center = compose(
        observed.object_pose, Pose(local_center, (0.0, 0.0, 0.0, 1.0))
    ).position_xyz
    held = compose(inverse(observed.object_pose), observed.tcp_pose)
    receiver_current = receiver_observed.tcp_pose
    proposals: list[ZoneTarget] = []
    # Keep the object inside the table, but move it considerably farther than
    # the old 0.35/0.55/0.75 interpolation. The receiver enters from its side.
    for along in (-0.04, 0.0, 0.04):
        for forward in (0.06, 0.10):
            for height in (0.17, 0.21):
                center = tuple(
                    zone.midpoint[i] + along * zone.along[i]
                    + forward * zone.forward[i]
                    if i < 2 else zone.tabletop_z_m + height
                    for i in range(3)
                )
                donor_inward = _unit((
                    center[0] - request.donor.base.position_xyz[0],
                    center[1] - request.donor.base.position_xyz[1],
                    0.0,
                ))
                receiver_inward = _unit((
                    center[0] - request.receiver.base.position_xyz[0],
                    center[1] - request.receiver.base.position_xyz[1],
                    0.0,
                ))
                approach = tuple(
                    center[i] - 0.11 * receiver_inward[i] for i in range(3)
                )
                for pitch in (-0.35, 0.0, 0.35):
                    for roll in (0.0,):
                        local_rotation = compose(
                            _rotation("y", pitch), _rotation("x", roll)
                        )
                        object_pose = compose(
                            observed.object_pose,
                            Pose((0.0, 0.0, 0.0), local_rotation.orientation_xyzw),
                        )
                        object_goal = Pose(
                            tuple(
                                center[i]
                                - _quat_rotate_vector(
                                    object_pose.orientation_xyzw, local_center
                                )[i]
                                for i in range(3)
                            ),
                            object_pose.orientation_xyzw,
                        )
                        giver_goal = compose(object_goal, held)
                        giver_distance, giver_rotation = pose_error(
                            observed.tcp_pose, giver_goal
                        )
                        for receiver_blend in (0.60, 0.80):
                            receiver_position = tuple(
                                receiver_current.position_xyz[i]
                                * (1.0 - receiver_blend)
                                + approach[i] * receiver_blend
                                for i in range(3)
                            )
                            receiver_precontact = Pose(
                                receiver_position,
                                receiver_current.orientation_xyzw,
                            )
                            receiver_distance, receiver_rotation = pose_error(
                                receiver_current, receiver_precontact
                            )
                            midpoint_distance = math.hypot(
                                center[0] - zone.midpoint[0],
                                center[1] - zone.midpoint[1],
                            )
                            score = (
                                giver_distance
                                + 0.25 * receiver_distance
                                + 0.03 * giver_rotation
                                + 0.01 * receiver_rotation
                                + 0.5 * midpoint_distance
                                + 0.02 * abs(pitch)
                                + 0.01 * abs(roll)
                            )
                            proposals.append(
                                ZoneTarget(
                                    center,
                                    object_goal,
                                    giver_goal,
                                    receiver_precontact,
                                    receiver_precontact,
                                    donor_inward,
                                    receiver_inward,
                                    score,
                                )
                            )
    return zone, tuple(sorted(proposals, key=lambda value: value.score))
