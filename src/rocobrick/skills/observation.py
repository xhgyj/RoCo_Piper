"""Strict, transport-free evidence checks for one observed PICK."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Pose:
    """A measured world-frame rigid pose in xyzw quaternion order."""

    position_xyz: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class PickObservation:
    """A same-step, robot-bound TCP and physical-instance sample."""

    robot_id: str
    physical_instance_id: str
    session_id: str
    manifest_sha256: str
    sequence: int
    sim_time_s: float
    object_pose: Pose
    tcp_pose: Pose
    opening_m: float
    aabb_min: tuple[float, float, float]
    aabb_max: tuple[float, float, float]


def _vector(raw: object, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(raw, (list, tuple)) or len(raw) != length:
        raise ValueError(f"{name} must have {length} components")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in raw
    ):
        raise ValueError(f"{name} must contain finite numbers")
    return tuple(float(value) for value in raw)


def _pose(raw: object, name: str) -> Pose:
    if not isinstance(raw, Mapping) or set(raw) != {
        "position_xyz", "orientation_xyzw"
    }:
        raise ValueError(f"{name} must be a complete rigid pose")
    position = _vector(raw["position_xyz"], 3, f"{name}.position_xyz")
    quaternion = _vector(raw["orientation_xyzw"], 4, f"{name}.orientation_xyzw")
    norm = math.sqrt(sum(value * value for value in quaternion))
    if abs(norm - 1.0) > 1e-3:
        raise ValueError(f"{name}.orientation_xyzw must be normalized")
    return Pose(position, quaternion)


def parse_pick_observation(
    document: Mapping[str, object],
    *,
    robot_id: str,
    physical_instance_id: str,
    session_id: str,
    manifest_sha256: str,
) -> PickObservation:
    """Accept only a co-sampled, identified object and robot TCP observation.

    Returns:
        An immutable sample with no inferred position or holding state.
    """
    if not all(
        isinstance(value, str) and value
        for value in (robot_id, physical_instance_id, session_id, manifest_sha256)
    ):
        raise ValueError("expected observation identities must be nonempty")
    if not isinstance(document, Mapping) or document.get("schema") != (
        "rocobrick-scene-state/v1"
    ):
        raise ValueError("scene-state schema is missing or wrong")
    if document.get("session_id") != session_id or document.get(
        "manifest_sha256"
    ) != manifest_sha256:
        raise ValueError("scene observation provenance does not match")
    sequence = document.get("sequence")
    if type(sequence) is not int or sequence < 0:
        raise ValueError("scene sequence must be a nonnegative integer")
    sim_time = document.get("sim_time_s")
    if (
        isinstance(sim_time, bool)
        or not isinstance(sim_time, (int, float))
        or not math.isfinite(float(sim_time))
        or sim_time < 0
    ):
        raise ValueError("scene simulation time is invalid")
    robots = document.get("robots")
    bricks = document.get("bricks")
    if not isinstance(robots, Mapping) or not isinstance(bricks, Mapping):
        raise ValueError("scene robot or brick mapping is missing")
    robot = robots.get(robot_id)
    brick = bricks.get(physical_instance_id)
    if not isinstance(robot, Mapping) or robot.get("robot_id") != robot_id:
        raise ValueError("scene robot identity is missing or foreign")
    if not isinstance(brick, Mapping):
        raise ValueError("scene physical instance is missing")
    if type(robot.get("sequence")) is not int or robot["sequence"] != sequence:
        raise ValueError("robot and scene sequence do not match")
    robot_time = robot.get("sim_time_s")
    if (
        isinstance(robot_time, bool)
        or not isinstance(robot_time, (int, float))
        or float(robot_time) != float(sim_time)
    ):
        raise ValueError("robot and scene simulation time do not match")
    minimum = _vector(brick.get("aabb_min"), 3, "aabb_min")
    maximum = _vector(brick.get("aabb_max"), 3, "aabb_max")
    if any(low > high for low, high in zip(minimum, maximum)):
        raise ValueError("brick AABB is inverted")
    opening = robot.get("opening_m")
    if (
        isinstance(opening, bool)
        or not isinstance(opening, (int, float))
        or not math.isfinite(float(opening))
    ):
        raise ValueError("measured gripper opening is unavailable")
    return PickObservation(
        robot_id,
        physical_instance_id,
        session_id,
        manifest_sha256,
        sequence,
        float(sim_time),
        _pose(brick.get("pose"), "brick pose"),
        _pose(robot.get("tcp_pose"), "TCP pose"),
        float(opening),
        minimum,
        maximum,
    )


def parse_place_observation(
    document: Mapping[str, object],
    *,
    robot_id: str,
    upper_physical_instance_id: str,
    lower_physical_instance_id: str,
    session_id: str,
    manifest_sha256: str,
) -> "PlaceObservation":
    """Accept one co-sampled placement observation with the live connection.

    Both bricks and this robot's TCP/opening come from the SAME scene
    document (same sequence and simulation time); the connection value is
    the live bricksim ``are_parts_connected`` verdict for exactly the
    requested pair, never an inferred stacking geometry.

    Returns:
        An immutable sample with no inferred placement state.

    Raises:
        ValueError: When any identity, freshness or shape guarantee fails.
    """
    upper = parse_pick_observation(
        document,
        robot_id=robot_id,
        physical_instance_id=upper_physical_instance_id,
        session_id=session_id,
        manifest_sha256=manifest_sha256,
    )
    lower = parse_pick_observation(
        document,
        robot_id=robot_id,
        physical_instance_id=lower_physical_instance_id,
        session_id=session_id,
        manifest_sha256=manifest_sha256,
    )
    connections = document.get("connections")
    if not isinstance(connections, Mapping):
        raise ValueError("scene connections mapping is missing")
    pair = "|".join(sorted((upper_physical_instance_id, lower_physical_instance_id)))
    value = connections.get(pair)
    if isinstance(value, bool):
        connected = value
    elif isinstance(value, Mapping) and isinstance(value.get("connected"), bool):
        connected = value["connected"]
    else:
        raise ValueError(f"scene connection {pair!r} is missing or not boolean")
    return PlaceObservation(upper=upper, lower=lower, connection=connected,
                            interfaces=document.get("placement_interfaces"))


@dataclass(frozen=True)
class PlaceObservation:
    """One same-step placement sample: both bricks, the connection verdict."""

    upper: PickObservation
    lower: PickObservation
    connection: bool
    interfaces: object = None


def verify_place_bricks(
    samples: Sequence[PlaceObservation],
    *,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
    require_connected: bool = True,
    minimum_tcp_rise_m: float = 0.0,
) -> Mapping[str, object]:
    """Require advancing, identified samples with stable bricks and TCP rise.

    Every sample must belong to the same robot/session/manifest and carry a
    strictly advancing scene sequence and simulation time.  Both bricks must
    stay within ``position_tolerance_m``/``orientation_tolerance_rad`` of
    their first-sample poses (a disturbed stack never passes); when
    ``require_connected`` every sample's live connection verdict must be
    true; when ``minimum_tcp_rise_m`` is positive the TCP must have risen at
    least that far from the first to the last sample (retreat proof).

    Returns:
        JSON-safe measurement summary; no placement confirmation label.

    Raises:
        ValueError: When any identity, advancement, stability, connection
            or rise requirement fails.
    """
    if len(samples) < 2:
        raise ValueError("placement verification requires at least two observations")
    for name, value in (
        ("position_tolerance_m", position_tolerance_m),
        ("orientation_tolerance_rad", orientation_tolerance_rad),
        ("minimum_tcp_rise_m", minimum_tcp_rise_m),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
        ):
            raise ValueError(f"{name} must be finite and non-negative")
    first = samples[0]
    if first.upper.physical_instance_id == first.lower.physical_instance_id:
        raise ValueError("placement upper and lower parts are identical")
    max_upper_position = 0.0
    max_upper_orientation = 0.0
    max_lower_position = 0.0
    max_lower_orientation = 0.0
    previous = first
    for sample in samples:
        upper = sample.upper
        lower = sample.lower
        if (
            upper.robot_id != first.upper.robot_id
            or upper.physical_instance_id != first.upper.physical_instance_id
            or lower.physical_instance_id != first.lower.physical_instance_id
            or upper.session_id != first.upper.session_id
            or upper.manifest_sha256 != first.upper.manifest_sha256
        ):
            raise ValueError("placement observation identity changed")
        if (
            sample is not first
            and (
                upper.sequence <= previous.upper.sequence
                or upper.sim_time_s <= previous.upper.sim_time_s
                or lower.sequence != upper.sequence
                or lower.sim_time_s != upper.sim_time_s
            )
        ):
            raise ValueError("placement observations are stale or reordered")
        if require_connected and sample.connection is not True:
            raise ValueError("the live connection verdict is not connected")
        for measured, reference, label in (
            (upper.object_pose, first.upper.object_pose, "upper"),
            (lower.object_pose, first.lower.object_pose, "lower"),
        ):
            difference = math.dist(
                measured.position_xyz, reference.position_xyz
            )
            dot = abs(sum(a * b for a, b in zip(
                measured.orientation_xyzw, reference.orientation_xyzw
            )))
            angle = 2.0 * math.acos(min(1.0, dot))
            if label == "upper":
                max_upper_position = max(max_upper_position, difference)
                max_upper_orientation = max(max_upper_orientation, angle)
            else:
                max_lower_position = max(max_lower_position, difference)
                max_lower_orientation = max(max_lower_orientation, angle)
        previous = sample
    tcp_rise = (
        samples[-1].upper.tcp_pose.position_xyz[2]
        - first.upper.tcp_pose.position_xyz[2]
    )
    if minimum_tcp_rise_m > 0.0 and tcp_rise < minimum_tcp_rise_m:
        raise ValueError("the TCP did not rise the required retreat distance")
    if (
        max_upper_position > position_tolerance_m
        or max_upper_orientation > orientation_tolerance_rad
        or max_lower_position > position_tolerance_m
        or max_lower_orientation > orientation_tolerance_rad
    ):
        raise ValueError("a placed brick moved beyond the placement tolerance")
    return {
        "robot_id": first.upper.robot_id,
        "upper_physical_instance_id": first.upper.physical_instance_id,
        "lower_physical_instance_id": first.lower.physical_instance_id,
        "session_id": first.upper.session_id,
        "manifest_sha256": first.upper.manifest_sha256,
        "first_sequence": first.upper.sequence,
        "last_sequence": samples[-1].upper.sequence,
        "first_sim_time_s": first.upper.sim_time_s,
        "last_sim_time_s": samples[-1].upper.sim_time_s,
        "samples": len(samples),
        "connected": True,
        "upper_max_position_change_m": max_upper_position,
        "upper_max_orientation_change_rad": max_upper_orientation,
        "lower_max_position_change_m": max_lower_position,
        "lower_max_orientation_change_rad": max_lower_orientation,
        "tcp_rise_m": tcp_rise,
        "position_tolerance_m": float(position_tolerance_m),
        "orientation_tolerance_rad": float(orientation_tolerance_rad),
        "minimum_tcp_rise_m": float(minimum_tcp_rise_m),
    }


def _conjugate(quat: tuple[float, float, float, float]) -> tuple[float, ...]:
    return (-quat[0], -quat[1], -quat[2], quat[3])


def _multiply(left: Sequence[float], right: Sequence[float]) -> tuple[float, ...]:
    x, y, z, w = left
    a, b, c, d = right
    return (w * a + x * d + y * c - z * b,
            w * b - x * c + y * d + z * a,
            w * c + x * b - y * a + z * d,
            w * d - x * a - y * b - z * c)


def _relative(tcp: Pose, obj: Pose) -> Pose:
    inverse = _conjugate(tcp.orientation_xyzw)
    offset = tuple(a - b for a, b in zip(obj.position_xyz, tcp.position_xyz))
    rotated = _multiply(_multiply(inverse, (*offset, 0.0)), tcp.orientation_xyzw)
    orientation = _multiply(inverse, obj.orientation_xyzw)
    return Pose(rotated[:3], orientation)


def verify_pick_following(
    samples: Sequence[PickObservation],
    *,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
    minimum_lift_m: float,
) -> Mapping[str, object]:
    """Require advancing, stable object-to-TCP transforms and a real lift.

    Returns:
        JSON-safe measurement summary; no holding confirmation label.
    """
    if len(samples) < 3:
        raise ValueError("holding requires at least three observations")
    for name, value in (
        ("position_tolerance_m", position_tolerance_m),
        ("orientation_tolerance_rad", orientation_tolerance_rad),
        ("minimum_lift_m", minimum_lift_m),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError(f"{name} must be finite and positive")
    first = samples[0]
    reference = _relative(first.tcp_pose, first.object_pose)
    max_position = 0.0
    max_orientation = 0.0
    previous = first
    for sample in samples:
        if (
            sample.robot_id != first.robot_id
            or sample.physical_instance_id != first.physical_instance_id
            or sample.session_id != first.session_id
            or sample.manifest_sha256 != first.manifest_sha256
        ):
            raise ValueError("following observation identity changed")
        if sample is not first and (
            sample.sequence <= previous.sequence
            or sample.sim_time_s <= previous.sim_time_s
        ):
            raise ValueError("following observations are stale or reordered")
        current = _relative(sample.tcp_pose, sample.object_pose)
        difference = math.dist(current.position_xyz, reference.position_xyz)
        dot = abs(sum(a * b for a, b in zip(
            current.orientation_xyzw, reference.orientation_xyzw
        )))
        angle = 2 * math.acos(min(1.0, dot))
        max_position = max(max_position, difference)
        max_orientation = max(max_orientation, angle)
        previous = sample
    lift = (
        samples[-1].object_pose.position_xyz[2]
        - first.object_pose.position_xyz[2]
    )
    if lift < minimum_lift_m:
        raise ValueError("object did not lift from its initial height")
    if (
        max_position > position_tolerance_m
        or max_orientation > orientation_tolerance_rad
    ):
        raise ValueError("object slipped relative to observed TCP")
    return {
        "robot_id": first.robot_id,
        "physical_instance_id": first.physical_instance_id,
        "session_id": first.session_id,
        "manifest_sha256": first.manifest_sha256,
        "first_sequence": first.sequence,
        "last_sequence": samples[-1].sequence,
        "first_sim_time_s": first.sim_time_s,
        "last_sim_time_s": samples[-1].sim_time_s,
        "samples": len(samples),
        "lift_m": lift,
        "max_relative_position_error_m": max_position,
        "max_relative_orientation_error_rad": max_orientation,
        "position_tolerance_m": float(position_tolerance_m),
        "orientation_tolerance_rad": float(orientation_tolerance_rad),
    }
