"""Transport-free contact guards for measured insertion and unloading."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict

from rocobrick.skills.observation import (
    _pose,
    parse_pick_observation,
    parse_place_observation,
)
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
SUPPORT_GUARD_KEYS = frozenset(
    {
        "robot_id",
        "session_id",
        "manifest_sha256",
        "upper_id",
        "supported_pose",
        "supported_position_m",
        "supported_angle_rad",
        "force_limit_n",
        "mode",
    }
)

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
        "support_contact_confirmed",
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
    if document.get("mode") in ("pick_descent", "pick_entry"):
        from rocobrick.skills.pick_contact import validate_pick_guard

        return validate_pick_guard(document)
    mode = document.get("mode")
    if mode == "support_top":
        expected = SUPPORT_GUARD_KEYS
        if "target_force_n" in document:
            expected = expected | {"target_force_n"}
            target_force = document["target_force_n"]
            if (
                type(target_force) not in (int, float)
                or not math.isfinite(target_force)
                or not 0 < target_force <= 3.0
            ):
                raise ValueError("support target force must be in (0, 3] N")
        poses = ("supported_pose",)
        numbers = ("supported_position_m", "supported_angle_rad", "force_limit_n")
    elif mode in UP_MODES:
        expected = UP_GUARD_KEYS
        if "support_robot_id" in document:
            expected = expected | {"support_robot_id"}
            if (
                not isinstance(document["support_robot_id"], str)
                or not document["support_robot_id"]
                or document["support_robot_id"] == document.get("robot_id")
            ):
                raise ValueError("contact guard support robot identity is invalid")
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
    identity_keys = ("robot_id", "session_id", "manifest_sha256", "upper_id")
    for key in identity_keys + (() if mode == "support_top" else ("lower_id",)):
        if not isinstance(document[key], str) or not document[key]:
            raise ValueError(f"contact guard {key} must be nonempty")
    if mode != "support_top" and document["upper_id"] == document["lower_id"]:
        raise ValueError("contact guard pair must contain distinct objects")
    if mode not in ("insert", "unload", "probe", "support_top", *UP_MODES):
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
    # Compare in the LIVE supported interface frame, not world-Z AABB extrema.
    live_seat = compose(
        compose(sample.upper.object_pose, inverse(supported_pose)),
        _pose(guard["desired_lower"], "desired lower"),
    )
    seat_delta = compose(inverse(live_seat), sample.lower.object_pose)
    _, seat_angle = pose_error(sample.lower.object_pose, live_seat)
    native = sample.interfaces.get("diagnostics", [])
    if not isinstance(native, list):
        raise ValueError("native assembly diagnostics must be a list")
    native_pair = [
        dict(item)
        for item in native
        if isinstance(item, Mapping)
        and item.get("stud_path") == pair.get("stud_path")
        and item.get("hole_path") == pair.get("hole_path")
        and item.get("stud_interface") == pair.get("stud_interface")
        and item.get("hole_interface") == pair.get("hole_interface")
    ]
    report = scene.get("pad_object_contacts", {})
    carrier_contacts = (
        [
            dict(item)
            for item in report.get("contacts", [])
            if isinstance(item, Mapping)
            and item.get("robot_id") == guard["robot_id"]
            and item.get("physical_instance_id")
            in (guard["lower_id"], guard["upper_id"])
        ]
        if isinstance(report, Mapping)
        else []
    )
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
        "contact_gap_source": "world_z_aabb",
        "seat_error_xyz_m": list(seat_delta.position_xyz),
        "seat_axial_error_m": seat_delta.position_xyz[2],
        "seat_lateral_error_m": math.hypot(*seat_delta.position_xyz[:2]),
        "seat_angle_rad": seat_angle,
        "native_assembly_diagnostics": native_pair,
        "native_assembly_diagnostics_available": bool(native_pair),
        "native_assembly_thresholds": dict(sample.interfaces.get("thresholds", {})),
        "interface_force_available": force is not None,
        "carrier_pad_contacts": carrier_contacts,
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
    if "support_robot_id" in guard:
        from rocobrick.skills.support_evidence import verify_support_contact

        try:
            _, support = verify_support_contact(
                scene,
                robot_id=guard["support_robot_id"],
                upper_id=guard["upper_id"],
                session_id=guard["session_id"],
                manifest_sha256=guard["manifest_sha256"],
            )
            measurements["support_contact"] = support
        except ValueError as error:
            measurements["support_contact_error"] = str(error)
            return "support_contact_lost", measurements
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
    if guard["mode"] in ("pick_descent", "pick_entry"):
        from rocobrick.skills.pick_contact import inspect_pick_contact

        return inspect_pick_contact(scene, guard)
    if guard["mode"] == "support_top":
        from rocobrick.skills.support_evidence import verify_support_contact

        sample = parse_pick_observation(
            scene,
            robot_id=guard["robot_id"],
            physical_instance_id=guard["upper_id"],
            session_id=guard["session_id"],
            manifest_sha256=guard["manifest_sha256"],
        )
        position, angle = pose_error(
            sample.object_pose, _pose(guard["supported_pose"], "supported upper")
        )
        measured = {
            "sequence": sample.sequence,
            "sim_time_s": sample.sim_time_s,
            "supported_position_m": position,
            "supported_angle_rad": angle,
        }
        if (
            position > guard["supported_position_m"]
            or angle > guard["supported_angle_rad"]
        ):
            return "supported_upper_disturbed", measured
        report = scene.get("pad_object_contacts")
        if not isinstance(report, Mapping) or not isinstance(
            report.get("contacts"), list
        ):
            raise ValueError("support approach requires actual contact sensing")
        if (
            report.get("schema") != "rocobrick-pad-object-contacts/v1"
            or report.get("source") != "physx_contact_report"
            or any(
                report.get(key) != scene.get(key)
                for key in ("session_id", "manifest_sha256", "sequence", "sim_time_s")
            )
        ):
            raise ValueError("support approach contact report is stale or foreign")
        force = sum(
            float(item["normal_force_n"])
            for item in report["contacts"]
            if item.get("robot_id") == guard["robot_id"]
            and item.get("physical_instance_id") == guard["upper_id"]
        )
        if not math.isfinite(force) or force < 0:
            raise ValueError("support reaction is not finite")
        measured["support_force_n"] = force
        if force > guard["force_limit_n"]:
            return "support_force_exceeded", measured
        if force > 0:
            _, contact = verify_support_contact(
                scene,
                robot_id=guard["robot_id"],
                upper_id=guard["upper_id"],
                session_id=guard["session_id"],
                manifest_sha256=guard["manifest_sha256"],
            )
            measured["support_contact"] = contact
            measured["support_contact_established"] = True
            if "target_force_n" in guard and force >= guard["target_force_n"]:
                return "support_contact_confirmed", measured
        # A support descent is a controlled PRESS: touching the top surface is
        # the expected condition, not a terminal one. Stopping here left the
        # pads hovering above the surface with the designed press (the contact
        # target sits support_press_m below it) never executed, so no
        # maintained contact existed to prove and SUPPORT_TOP failed (live
        # evidence, task-c-top-offset13f 2026-10-14: one 0.12 N graze at
        # sim 90.22 s, pads settled 0.9 mm ABOVE the surface, then "no measured
        # top-surface contact"). The descent runs to its endpoint; the force
        # limit above and the upper-brick tolerance before it stay the stops.
        return None, measured
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
