"""Rigid placement geometry from a stud-hole interface and measured grasp."""

from __future__ import annotations

import math
from dataclasses import replace

from rocobrick.skills.observation import Pose
from rocobrick.skills.targets import (
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

    Returns:
        Position distance in metres and quaternion angle in radians.
    """
    dot = abs(
        sum(a * b for a, b in zip(measured.orientation_xyzw, expected.orientation_xyzw))
    )
    return math.dist(measured.position_xyz, expected.position_xyz), (
        2.0 * math.acos(min(1.0, dot))
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
