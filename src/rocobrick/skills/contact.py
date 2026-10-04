"""Transport-free contact guards for measured insertion and unloading."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict

from rocobrick.skills.observation import _pose, parse_place_observation
from rocobrick.skills.placement_geometry import compose, inverse, pose_error

GUARD_KEYS = frozenset(
    {
        "robot_id",
        "session_id",
        "manifest_sha256",
        "upper_id",
        "lower_id",
        "support_pose",
        "held_transform",
        "desired_upper",
        "support_position_m",
        "support_angle_rad",
        "grasp_position_m",
        "grasp_angle_rad",
        "force_limit_n",
        "mode",
    }
)

# Upward guards swap the roles: the UPPER part is the stationary partner
# stabilized by the support robot and the LOWER part is the executor's held,
# moving object. Key names state their upward meaning explicitly.
UP_GUARD_KEYS = frozenset(
    {
        "robot_id",
        "session_id",
        "manifest_sha256",
        "upper_id",
        "lower_id",
        "supported_pose",
        "held_transform",
        "desired_lower",
        "supported_position_m",
        "supported_angle_rad",
        "grasp_position_m",
        "grasp_angle_rad",
        "force_limit_n",
        "mode",
    }
)
UP_MODES = ("insert_up", "unload_up")

# Probe mode stops at the FIRST measured interface contact: the held and
# support AABBs overlap laterally and their world-Z separation has closed to
# this bound. The stud axis is the (already validated) near-vertical support
# +Z, so world-Z separation is the axial interface gap for any brick shape.
PROBE_CONTACT_GAP_M = 0.0005


def allows_recovery(reason: object) -> bool:
    """Distinguish a safely stopped contact step from transport failure.

    Returns:
        Whether a terminal stopped step may enter the recovery controller.
    """
    return reason in {
        "connection_confirmed",
        "support_deflected",
        "supported_upper_disturbed",
        "grasp_slipped",
        "interface_force_exceeded",
        "wrong_interface_connection",
        "connection_changed_during_unload",
        "unload_object_uncontrolled",
        "unload_force_exceeded",
        "contact_tracking_stopped",
        "probe_contact_detected",
    }


def validate_guard(document: object) -> Mapping[str, object]:
    """Validate an exact robot-scoped guard before a command can be sent.

    Returns:
        The validated guard document.
    """
    if not isinstance(document, Mapping):
        raise ValueError("contact guard must be a mapping")
    mode = document.get("mode")
    if mode in UP_MODES:
        expected: frozenset[str] = UP_GUARD_KEYS
        poses = ("supported_pose", "held_transform", "desired_lower")
        numbers = (
            "supported_position_m",
            "supported_angle_rad",
            "grasp_position_m",
            "grasp_angle_rad",
            "force_limit_n",
        )
    else:
        expected = GUARD_KEYS
        poses = ("support_pose", "held_transform", "desired_upper")
        numbers = (
            "support_position_m",
            "support_angle_rad",
            "grasp_position_m",
            "grasp_angle_rad",
            "force_limit_n",
        )
    if set(document) != expected:
        raise ValueError("contact guard keys do not match the strict schema")
    for key in ("robot_id", "session_id", "manifest_sha256", "upper_id", "lower_id"):
        if not isinstance(document[key], str) or not document[key]:
            raise ValueError(f"contact guard {key} must be nonempty")
    if document["upper_id"] == document["lower_id"]:
        raise ValueError("contact guard pair must contain distinct objects")
    if mode not in ("insert", "unload", "probe", *UP_MODES):
        raise ValueError("contact guard mode is unsupported")
    for key in poses:
        _pose(document[key], key)
    for key in numbers:
        value = document[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or (not math.isfinite(value) or value <= 0)
        ):
            raise ValueError(f"contact guard {key} must be finite and positive")
    return document


def _interface_gap_m(sample) -> float | None:
    """Measure the world-Z interface gap between laterally overlapping AABBs.

    Returns:
        The upper-AABB-bottom to lower-AABB-top separation in metres, or
        ``None`` when the two boxes do not overlap laterally (no contact is
        geometrically possible on this descent).
    """
    for axis in (0, 1):
        upper_min = float(sample.upper.aabb_min[axis])
        upper_max = float(sample.upper.aabb_max[axis])
        lower_min = float(sample.lower.aabb_min[axis])
        lower_max = float(sample.lower.aabb_max[axis])
        if upper_min > lower_max or lower_min > upper_max:
            return None
    return float(sample.upper.aabb_min[2]) - float(sample.lower.aabb_max[2])


def _live_pair(sample, guard):
    """Return the identified live interface pair of the guard's brick pair.

    Returns:
        The pair mapping of ``placement_interfaces.pairs`` for this exact
        upper/lower pair.

    Raises:
        ValueError: When the pair has no identified live interface.
    """
    info = sample.interfaces
    pairs = info.get("pairs") if isinstance(info, Mapping) else None
    key = "|".join(sorted((guard["upper_id"], guard["lower_id"])))
    pair = pairs.get(key) if isinstance(pairs, Mapping) else None
    if not isinstance(pair, Mapping):
        raise ValueError("contact guard has no identified live interface")
    return pair


def _pair_force(pair: Mapping[str, object]) -> float | None:
    """Validate and return the pair's native interface force diagnostic.

    Returns:
        The finite diagnostic force in newtons, or ``None`` when absent.
    """
    force = pair.get("contact_force_n")
    if force is not None and (
        isinstance(force, bool)
        or not isinstance(force, (int, float))
        or not math.isfinite(force)
    ):
        raise ValueError("interface contact force is nonfinite")
    return force


def _inspect_upward(sample, scene, guard):
    """Guard the upward insertion: fixed upper, executor-held moving lower.

    The support robot's stabilization is proven physically by the UPPER
    brick holding still against its admitted pose; the executor's grip is
    proven by the LOWER brick following its measured TCP transform. The
    connection identity check mirrors the downward guard in the admitted
    upper frame.

    Returns:
        Stop reason (or None) and JSON-safe measurements.
    """
    supported_pose = _pose(guard["supported_pose"], "supported")
    supported_p, supported_a = pose_error(sample.upper.object_pose, supported_pose)
    predicted_lower = compose(
        sample.lower.tcp_pose, inverse(_pose(guard["held_transform"], "held"))
    )
    grasp_p, grasp_a = pose_error(sample.lower.object_pose, predicted_lower)
    pair = _live_pair(sample, guard)
    actual = pair.get("actual_connection")
    exact = False
    conflict = False
    if sample.connection:
        if not isinstance(actual, Mapping):
            raise ValueError("connected pair has no exact interface evidence")
        if actual.get("stud_interface") != pair.get("stud_interface") or (
            actual.get("hole_interface") != pair.get("hole_interface")
        ):
            conflict = True
        else:
            actual_fixed_lower = compose(
                supported_pose,
                inverse(_pose(actual.get("lower_to_upper"), "actual interface")),
            )
            distance, angle = pose_error(
                actual_fixed_lower, _pose(guard["desired_lower"], "desired lower")
            )
            exact = distance <= 1e-5 and angle <= 1e-4
            conflict = not exact
    force = _pair_force(pair)
    dynamics = scene.get("robots", {}).get(guard["robot_id"], {}).get("dynamics")
    measurements = {
        "dynamics": dynamics,
        "sequence": sample.upper.sequence,
        "sim_time_s": sample.upper.sim_time_s,
        "supported_position_m": supported_p,
        "supported_angle_rad": supported_a,
        "grasp_position_m": grasp_p,
        "grasp_angle_rad": grasp_a,
        "connected": sample.connection,
        "exact_connection": exact,
        "contact_force_n": force,
        "contact_gap_m": _interface_gap_m(sample),
        "upper_pose": asdict(sample.upper.object_pose),
        "lower_pose": asdict(sample.lower.object_pose),
        "tcp_pose": asdict(sample.upper.tcp_pose),
    }
    if guard["mode"] == "unload_up":
        if sample.connection:
            return "connection_changed_during_unload", measurements
        if grasp_p > guard["grasp_position_m"] or grasp_a > guard["grasp_angle_rad"]:
            return "unload_object_uncontrolled", measurements
        if force is not None and abs(force) > guard["force_limit_n"]:
            return "unload_force_exceeded", measurements
        return None, measurements
    if conflict:
        return "wrong_interface_connection", measurements
    if (
        supported_p > guard["supported_position_m"]
        or supported_a > guard["supported_angle_rad"]
    ):
        return "supported_upper_disturbed", measurements
    if exact:
        return "connection_confirmed", measurements
    if grasp_p > guard["grasp_position_m"] or grasp_a > guard["grasp_angle_rad"]:
        return "grasp_slipped", measurements
    if force is not None and abs(force) > guard["force_limit_n"]:
        return "interface_force_exceeded", measurements
    return None, measurements


def inspect_contact(scene: Mapping[str, object], guard: Mapping[str, object]):
    """Inspect live support/grasp safety and the exact interface connection.

    Native interface force is diagnostic contact force, not a calibrated TCP
    wrench. Missing force is recorded explicitly; geometric guards remain
    active even before the native contact detector produces a diagnostic.

    Returns:
        Stop reason (or None) and JSON-safe measurements of this observation.
    """
    validate_guard(guard)
    sample = parse_place_observation(
        scene,
        robot_id=guard["robot_id"],
        upper_physical_instance_id=guard["upper_id"],
        lower_physical_instance_id=guard["lower_id"],
        session_id=guard["session_id"],
        manifest_sha256=guard["manifest_sha256"],
    )
    if guard["mode"] in UP_MODES:
        return _inspect_upward(sample, scene, guard)
    support_p, support_a = pose_error(
        sample.lower.object_pose, _pose(guard["support_pose"], "support")
    )
    predicted = compose(
        sample.upper.tcp_pose, inverse(_pose(guard["held_transform"], "held"))
    )
    grasp_p, grasp_a = pose_error(sample.upper.object_pose, predicted)
    info = sample.interfaces
    pairs = info.get("pairs") if isinstance(info, Mapping) else None
    key = "|".join(sorted((guard["upper_id"], guard["lower_id"])))
    pair = pairs.get(key) if isinstance(pairs, Mapping) else None
    if not isinstance(pair, Mapping):
        raise ValueError("contact guard has no identified live interface")
    actual = pair.get("actual_connection")
    exact = False
    conflict = False
    if sample.connection:
        if not isinstance(actual, Mapping):
            raise ValueError("connected pair has no exact interface evidence")
        if actual.get("stud_interface") != pair.get("stud_interface") or (
            actual.get("hole_interface") != pair.get("hole_interface")
        ):
            conflict = True
        else:
            # Support deflection must not turn a correct interface into a
            # wrong label; compare in the fixed admitted support frame.
            actual_fixed = compose(
                _pose(guard["support_pose"], "support"),
                _pose(actual.get("lower_to_upper"), "actual interface"),
            )
            distance, angle = pose_error(
                actual_fixed, _pose(guard["desired_upper"], "desired upper")
            )
            exact = distance <= 1e-5 and angle <= 1e-4
            conflict = not exact
    force = pair.get("contact_force_n")
    if force is not None and (
        isinstance(force, bool)
        or not isinstance(force, (int, float))
        or not math.isfinite(force)
    ):
        raise ValueError("interface contact force is nonfinite")
    dynamics = scene.get("robots", {}).get(guard["robot_id"], {}).get("dynamics")
    measurements = {
        "dynamics": dynamics,
        "sequence": sample.upper.sequence,
        "sim_time_s": sample.upper.sim_time_s,
        "support_position_m": support_p,
        "support_angle_rad": support_a,
        "grasp_position_m": grasp_p,
        "grasp_angle_rad": grasp_a,
        "connected": sample.connection,
        "exact_connection": exact,
        "contact_force_n": force,
        "contact_gap_m": _interface_gap_m(sample),
        "upper_pose": asdict(sample.upper.object_pose),
        "tcp_pose": asdict(sample.upper.tcp_pose),
    }
    if guard["mode"] == "unload":
        # Only reduction of an existing contact is authorized during unload.
        if sample.connection:
            return "connection_changed_during_unload", measurements
        if grasp_p > guard["grasp_position_m"] or grasp_a > guard["grasp_angle_rad"]:
            return "unload_object_uncontrolled", measurements
        if force is not None and abs(force) > guard["force_limit_n"]:
            return "unload_force_exceeded", measurements
        return None, measurements
    if conflict:
        return "wrong_interface_connection", measurements
    if (
        support_p > guard["support_position_m"]
        or support_a > guard["support_angle_rad"]
    ):
        return "support_deflected", measurements
    if exact:
        return "connection_confirmed", measurements
    if guard["mode"] == "probe":
        gap = measurements["contact_gap_m"]
        if gap is not None and gap <= PROBE_CONTACT_GAP_M:
            return "probe_contact_detected", measurements
        return None, measurements
    if grasp_p > guard["grasp_position_m"] or grasp_a > guard["grasp_angle_rad"]:
        return "grasp_slipped", measurements
    if force is not None and abs(force) > guard["force_limit_n"]:
        return "interface_force_exceeded", measurements
    return None, measurements
