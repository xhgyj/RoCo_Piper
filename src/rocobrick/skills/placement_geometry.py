"""Rigid placement geometry from a stud-hole interface and measured grasp."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from rocobrick.skills.observation import Pose, _pose
from rocobrick.skills.targets import (
    PickTargets,
    PlaceTargets,
    RobotBaseFrame,
    SkillParams,
    TcpProvenance,
    _normalize_quat,
    _quat_conjugate,
    _quat_mul,
    _quat_rotate_vector,
    _tcp_pose_target,
    pad_separation_to_logical_opening,
    pick_targets,
    world_orientation_to_base,
    world_position_to_base,
)


def compose(left: Pose, right: Pose) -> Pose:
    """Compose rigid poses expressed in XYZW order.

    Returns:
        ``T_left * T_right``.
    """
    offset = _quat_rotate_vector(left.orientation_xyzw, right.position_xyz)
    return Pose(
        tuple(left.position_xyz[i] + offset[i] for i in range(3)),
        _normalize_quat(_quat_mul(left.orientation_xyzw, right.orientation_xyzw)),
    )


def inverse(pose: Pose) -> Pose:
    """Invert a rigid pose.

    Returns:
        The inverse transform in XYZW order.
    """
    orientation = _quat_conjugate(pose.orientation_xyzw)
    return Pose(
        _quat_rotate_vector(orientation, tuple(-x for x in pose.position_xyz)),
        orientation,
    )


def pose_error(measured: Pose, expected: Pose) -> tuple[float, float]:
    """Compare full rigid poses without assuming brick rotational symmetry.

    Validate poses before normalizing sensor roundoff. Quaternion sign is
    irrelevant, and the chord-based angle stays accurate near zero rotation.

    Returns:
        Position distance in metres and quaternion angle in radians.
    """
    left, right = (
        _normalize_quat(
            _pose(
                {
                    "position_xyz": pose.position_xyz,
                    "orientation_xyzw": pose.orientation_xyzw,
                },
                name,
            ).orientation_xyzw
        )
        for name, pose in (("measured pose", measured), ("expected pose", expected))
    )
    if sum(a * b for a, b in zip(left, right)) < 0.0:
        right = tuple(-value for value in right)
    difference = math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))
    total = math.sqrt(sum((a + b) ** 2 for a, b in zip(left, right)))
    return math.dist(measured.position_xyz, expected.position_xyz), (
        4.0 * math.atan2(difference, total)
    )


def interpolate_pose(first: Pose, last: Pose, fraction: float) -> Pose:
    """Interpolate position and the shortest quaternion arc without mutation.

    Returns:
        A validated rigid pose; midpoint interpolation also bisects rotation.
    """
    if (
        isinstance(fraction, bool)
        or not math.isfinite(fraction)
        or not 0 <= fraction <= 1
    ):
        raise ValueError("pose interpolation fraction must be in [0, 1]")
    pose_error(first, last)  # Preserve the strict input pose contract.
    left, right = (_normalize_quat(pose.orientation_xyzw) for pose in (first, last))
    if sum(a * b for a, b in zip(left, right)) < 0:
        right = tuple(-value for value in right)
    return Pose(
        tuple(
            a + fraction * (b - a)
            for a, b in zip(first.position_xyz, last.position_xyz)
        ),
        _normalize_quat(
            tuple((1 - fraction) * a + fraction * b for a, b in zip(left, right))
        ),
    )


def interface_place_targets(
    lower_world: Pose,
    lower_to_upper: Pose,
    measured_upper: Pose,
    measured_tcp: Pose,
    measured_opening_m: float,
    base: RobotBaseFrame,
    provenance: TcpProvenance,
    params: SkillParams,
    yaw_symmetries: tuple[int, ...] = (0,),
) -> tuple[PlaceTargets, Pose, Pose]:
    """Use exact interface geometry and the current held-object transform.

    No brick dimensions, centre, layer height or rotational symmetry are
    assumed. The desired upper pose is ``T_world_lower * T_lower_upper``;
    the tool pose is that result times the measured ``T_upper_tcp``.
    Offsets follow the lower brick's stud axis, not the robot base Z axis.

    Returns:
        Tool targets, desired upper pose and measured upper-to-TCP transform.
    """
    held = compose(inverse(measured_upper), measured_tcp)
    if not yaw_symmetries or any(
        type(k) is not int or k not in range(4) for k in yaw_symmetries
    ):
        raise ValueError("invalid verified shape symmetry group")
    nominal_upper = compose(lower_world, lower_to_upper)
    candidates = [
        compose(
            nominal_upper,
            Pose(
                (0.0, 0.0, 0.0),
                (0.0, 0.0, math.sin(k * math.pi / 4), math.cos(k * math.pi / 4)),
            ),
        )
        for k in yaw_symmetries
    ]
    desired_upper = min(
        candidates,
        key=lambda candidate: pose_error(compose(candidate, held), measured_tcp)[1],
    )
    desired_tcp = compose(desired_upper, held)
    normal = _quat_rotate_vector(lower_world.orientation_xyzw, (0.0, 0.0, 1.0))
    if normal[2] <= 0.0:
        raise ValueError("placement stud axis is not upward; unsupported insertion")
    orientation = world_orientation_to_base(desired_tcp.orientation_xyzw, base)

    def target(offset: float):
        world = tuple(
            desired_tcp.position_xyz[i] + normal[i] * offset for i in range(3)
        )
        return _tcp_pose_target(
            world_position_to_base(world, base), orientation, provenance, params
        )

    release = pad_separation_to_logical_opening(
        measured_opening_m + params.open_extra_m
    )
    low = max(0.0, release - min(params.free_window_slop_m, params.open_extra_m * 0.25))
    high = min(0.1, release + params.free_window_slop_m)
    targets = PlaceTargets(
        preplace=target(params.place_standoff_m),
        contact=target(-params.place_press_m),
        retreat=target(params.place_retreat_height_m),
        release_opening_m=release,
        release_expected_window_m=(low, high),
        upper_span_m=measured_opening_m,
        closing_yaw_deg=0.0,
        contact_world_z_m=desired_tcp.position_xyz[2],
    )
    return targets, desired_upper, held


def interface_place_up_targets(
    upper_world: Pose,
    lower_to_upper: Pose,
    measured_lower: Pose,
    measured_tcp: Pose,
    measured_opening_m: float,
    base: RobotBaseFrame,
    provenance: TcpProvenance,
    params: SkillParams,
    yaw_symmetries: tuple[int, ...] = (0,),
) -> tuple[PlaceTargets, Pose, Pose]:
    """Solve the upward insertion targets from the FIXED upper interface.

    The mirror of :func:`interface_place_targets`: the executor grips the
    LOWER part and inserts it upward into the UPPER part stabilized by the
    support robot. The desired lower pose is
    ``T_world_upper * T_upper_lower`` with ``T_upper_lower =
    inverse(T_lower_upper)``; the tool pose composes that with the measured
    ``T_lower_tcp``. Offsets follow the upper brick's stud axis and the
    ascent bound sits ABOVE the seated depth by ``place_press_m`` (a bound
    the live monitor cancels, not a press to hold).

    A yaw-symmetric brick seats under several rotations, but the measured
    grasp offset from the brick centre must stay on the side AWAY from the
    mated upper part: only such candidates are eligible, and the
    minimum-tool-rotation ordering picks within that set. A fully vertical
    overlap has no preferred side, so every candidate stays eligible.

    Returns:
        Tool targets, desired lower pose and measured lower-to-TCP transform.
    """
    held = compose(inverse(measured_lower), measured_tcp)
    if not yaw_symmetries or any(
        type(k) is not int or k not in range(4) for k in yaw_symmetries
    ):
        raise ValueError("invalid verified shape symmetry group")
    nominal_lower = compose(upper_world, inverse(lower_to_upper))
    candidates = [
        compose(
            nominal_lower,
            Pose(
                (0.0, 0.0, 0.0),
                (0.0, 0.0, math.sin(k * math.pi / 4), math.cos(k * math.pi / 4)),
            ),
        )
        for k in yaw_symmetries
    ]
    toward_upper = (
        upper_world.position_xyz[0] - nominal_lower.position_xyz[0],
        upper_world.position_xyz[1] - nominal_lower.position_xyz[1],
    )

    def keeps_tool_clear(candidate: Pose) -> bool:
        tcp = compose(candidate, held)
        offset = (
            tcp.position_xyz[0] - candidate.position_xyz[0],
            tcp.position_xyz[1] - candidate.position_xyz[1],
        )
        return offset[0] * toward_upper[0] + offset[1] * toward_upper[1] <= 1e-4

    clearing = [candidate for candidate in candidates if keeps_tool_clear(candidate)]
    if not clearing:
        raise ValueError(
            "no verified yaw symmetry keeps the tool clear of the upper part"
        )
    desired_lower = min(
        clearing,
        key=lambda candidate: pose_error(compose(candidate, held), measured_tcp)[1],
    )
    desired_tcp = compose(desired_lower, held)
    normal = _quat_rotate_vector(upper_world.orientation_xyzw, (0.0, 0.0, 1.0))
    if normal[2] <= 0.0:
        raise ValueError("upward insertion axis is not upward; unsupported")
    orientation = world_orientation_to_base(desired_tcp.orientation_xyzw, base)

    def target(offset: float):
        world = tuple(
            desired_tcp.position_xyz[i] + normal[i] * offset for i in range(3)
        )
        return _tcp_pose_target(
            world_position_to_base(world, base), orientation, provenance, params
        )

    release = pad_separation_to_logical_opening(
        measured_opening_m + params.open_extra_m
    )
    low = max(0.0, release - min(params.free_window_slop_m, params.open_extra_m * 0.25))
    high = min(0.1, release + params.free_window_slop_m)
    targets = PlaceTargets(
        preplace=target(-params.place_standoff_m),
        contact=target(params.place_press_m),
        retreat=target(-params.place_retreat_height_m),
        release_opening_m=release,
        release_expected_window_m=(low, high),
        upper_span_m=measured_opening_m,
        closing_yaw_deg=0.0,
        contact_world_z_m=desired_tcp.position_xyz[2],
    )
    return targets, desired_lower, held


def offset_pose_target(target, normal_world, distance: float, base: RobotBaseFrame):
    """Offset a tool target along a measured stud axis.

    Returns:
        The target displaced in the base frame by the requested distance.
    """
    rotated = _quat_rotate_vector(_quat_conjugate(base.orientation_xyzw), normal_world)
    return replace(
        target,
        position=tuple(target.position[i] + rotated[i] * distance for i in range(3)),
    )


def support_top_target(sample, base, provenance, params, *, approach: bool):
    """Keep the calibrated tool pointing into the measured upper surface.

    The distal tips extend 5.211 mm below the calibrated TCP. Public-space
    approach leaves a 10 mm gap; only the contact phase consumes that gap.

    Returns:
        The calibrated approach or contact target in this robot's base.
    """
    normal = _quat_rotate_vector(sample.object_pose.orientation_xyzw, (0.0, 0.0, 1.0))
    if normal[2] <= 0.0:
        raise ValueError("SUPPORT_TOP requires an upward upper surface")
    surface = (
        (sample.aabb_min[0] + sample.aabb_max[0]) / 2,
        (sample.aabb_min[1] + sample.aabb_max[1]) / 2,
        sample.aabb_max[2],
    )
    offset = 0.005211 + (0.01 if approach else -params.support_press_m)
    world = tuple(surface[i] + normal[i] * offset for i in range(3))
    # Support is not a grasp: leave the opening untouched, but align the
    # separated fingertips with the long top dimension so neither tip is
    # deliberately parked outside a narrow face (Task C's 64 x 32 mm upper).
    axes = [
        _quat_rotate_vector(sample.object_pose.orientation_xyzw, axis)
        for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    ]
    spans = [
        sum(abs(axis[i]) * (sample.aabb_max[i] - sample.aabb_min[i]) for i in range(3))
        for axis in axes
    ]
    yaw = math.pi / 2 if spans[0] >= spans[1] else 0.0
    orientations = []
    for angle in (yaw, yaw + math.pi):
        around = (0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2))
        down = _quat_mul(around, (1.0, 0.0, 0.0, 0.0))
        orientations.append(_quat_mul(sample.object_pose.orientation_xyzw, down))
    orientation = min(
        orientations,
        key=lambda value: pose_error(
            Pose((0.0, 0.0, 0.0), value),
            Pose((0.0, 0.0, 0.0), sample.tcp_pose.orientation_xyzw),
        )[1],
    )
    return _tcp_pose_target(
        world_position_to_base(world, base),
        world_orientation_to_base(orientation, base),
        provenance,
        params,
    )


@dataclass(frozen=True)
class PlaceUpPickCandidate:
    """One geometry-only grasp; neither IK nor physical custody is implied."""

    candidate_id: str
    face_quarter_turn: int
    downward_tilt_deg: float
    targets: PickTargets


def place_up_pick_candidates(sample, base, provenance, params):
    """Enumerate side/oblique tools without changing the measured brick slot.

    Closing stays in the brick's horizontal plane. Tilt is downward from
    horizontal, keeping the palm to the side rather than using a top grasp.
    MoveIt, not this priority order, decides reachability and collisions.

    Returns:
        Sixteen identified geometry candidates in deterministic priority order.
    """
    original = pick_targets(sample.aabb_min, sample.aabb_max, base, provenance, params)
    toward = tuple(
        sample.object_pose.position_xyz[i] - base.position_xyz[i] for i in range(3)
    )
    faces = sorted(
        range(4),
        key=lambda quarter: (
            -sum(
                _quat_rotate_vector(
                    sample.object_pose.orientation_xyzw,
                    (
                        math.cos(quarter * math.pi / 2),
                        math.sin(quarter * math.pi / 2),
                        0.0,
                    ),
                )[i]
                * toward[i]
                for i in range(3)
            )
        ),
    )
    result = []
    for tilt in (20.0, 35.0, 50.0, 0.0):
        for quarter in faces:
            yaw = (
                0.0,
                0.0,
                math.sin(quarter * math.pi / 4),
                math.cos(quarter * math.pi / 4),
            )
            pitch = math.pi / 2 + math.radians(tilt)
            sideways = _quat_mul(
                yaw, (0.0, math.sin(pitch / 2), 0.0, math.cos(pitch / 2))
            )
            world = _quat_mul(sample.object_pose.orientation_xyzw, sideways)
            closing = _quat_rotate_vector(world, (0.0, 1.0, 0.0))
            span = sum(
                abs(closing[i]) * (sample.aabb_max[i] - sample.aabb_min[i])
                for i in range(3)
            )
            reach = pad_separation_to_logical_opening(span + params.open_extra_m)
            close = pad_separation_to_logical_opening(span - params.squeeze_m)
            contact = pad_separation_to_logical_opening(span)
            orientation = world_orientation_to_base(world, base)
            targets = replace(
                original,
                pregrasp=replace(original.pregrasp, orientation_xyzw=orientation),
                precontact_boundary=replace(
                    original.precontact_boundary, orientation_xyzw=orientation
                ),
                lift=replace(original.lift, orientation_xyzw=orientation),
                brick_span_m=span,
                reach_opening_m=reach,
                close_opening_m=close,
                reach_expected_window_m=(
                    reach - params.free_window_slop_m,
                    reach + params.free_window_slop_m,
                ),
                close_expected_window_m=(
                    contact - params.contact_window_slop_m,
                    contact + params.contact_window_slop_m,
                ),
            )
            result.append(
                PlaceUpPickCandidate(
                    f"side-{quarter}-tilt-{int(tilt)}", quarter, tilt, targets
                )
            )
    return tuple(result)


def place_up_pick_targets(sample, base, provenance, params):
    """Reuse PICK phases with a side-entry tool, leaving the stud face open.

    The same rigid measured lower-to-TCP transform is subsequently consumed
    by interface_place_up_targets. There is no extra clearance waypoint,
    regrasp, reversed finger model or bottom grab through the ground.

    Returns:
        Existing PICK phase targets with a successor-compatible tool pose.
    """
    # Compatibility geometry view only. Production PICK now submits an
    # object-relative grasp_region to MoveIt, not this finite candidate family.
    return next(
        candidate.targets
        for candidate in place_up_pick_candidates(sample, base, provenance, params)
        if candidate.downward_tilt_deg == 0.0
    )
