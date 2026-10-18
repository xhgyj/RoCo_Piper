"""PICK-owned fixed-orientation entry contracts; no free-space planning."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict

from rocobrick.skills.grasp_region import validate_grasp_region
from rocobrick.skills.observation import _pose, parse_pick_observation
from rocobrick.skills.placement_geometry import compose, pose_error

MODES = ("pick_descent", "pick_entry")
KEYS = {"mode", "robot_id", "session_id", "manifest_sha256", "region", "start", "end"}
# Live servo residual admitted around the commanded vertical line.  The arm is
# position-controlled through spring drives that settle several millimetres and
# tens of milliradians short of the commanded spline (2026-10-18: 2.76 mm
# lateral / 10.3-15.8 mrad of the requested hover), so a 2 mm lateral demand
# rejected the measured start of every descent.  These bounds admit that
# residual and stay inside the open-finger corridor: the approach opening is
# the brick span plus 30 mm, i.e. ~15 mm of pad clearance per side.
PICK_LATERAL_BOUND_M = 0.010
PICK_ANGLE_BOUND_RAD = 0.03


def validate_pick_guard(guard):
    """Reject missing identity, tilted paths and excessive open-loop travel.

    Returns:
        The validated guard, without widening any contact admission bound.
    """
    if not isinstance(guard, Mapping) or set(guard) != KEYS:
        raise ValueError("PICK contact guard keys mismatch")
    if guard["mode"] not in MODES:
        raise ValueError("PICK contact mode unsupported")
    validate_grasp_region(guard["region"])
    if guard["robot_id"] != guard["region"]["robot_id"]:
        raise ValueError("PICK contact robot mismatch")
    for key in ("session_id", "manifest_sha256"):
        if not isinstance(guard[key], str) or not guard[key]:
            raise ValueError("PICK contact identity missing")
    start, end = (_pose(guard[k], k) for k in ("start", "end"))
    _, angle = pose_error(start, end)
    distance = start.position_xyz[2] - end.position_xyz[2]
    if (
        math.dist(start.position_xyz[:2], end.position_xyz[:2]) > PICK_LATERAL_BOUND_M
        or angle > PICK_ANGLE_BOUND_RAD
        or not 0 < distance <= 0.065
        or guard["mode"] == "pick_entry"
        and distance > 0.005
    ):
        raise ValueError(
            "PICK contact requires bounded vertical fixed-orientation entry"
        )
    return guard


def inspect_pick_contact(scene, guard):
    """Stop on scene drift, off-axis tracking or premature finger contact.

    Returns:
        Stop reason and same-step physical evidence, never holding proof.
    """
    validate_pick_guard(guard)
    region = guard["region"]
    sample = parse_pick_observation(
        scene,
        robot_id=guard["robot_id"],
        physical_instance_id=region["physical_instance_id"],
        session_id=guard["session_id"],
        manifest_sha256=guard["manifest_sha256"],
    )
    expected = compose(
        _pose(region["base_in_world"], "base"),
        _pose(region["object_in_base"], "object"),
    )
    displacement, angle = pose_error(sample.object_pose, expected)
    start, end = (_pose(guard[k], k) for k in ("start", "end"))
    _, tcp_angle = pose_error(sample.tcp_pose, end)
    lateral = math.dist(sample.tcp_pose.position_xyz[:2], end.position_xyz[:2])
    measured = {
        "sequence": sample.sequence,
        "sim_time_s": sample.sim_time_s,
        "object_displacement_m": displacement,
        "object_angle_rad": angle,
        "tcp_lateral_error_m": lateral,
        "tcp_angle_rad": tcp_angle,
        "tcp_pose": asdict(sample.tcp_pose),
    }
    measured["opening_m"] = sample.opening_m
    if abs(sample.opening_m - region["approach_opening_m"]) > 0.001:
        return "pick_opening_drift", measured
    if displacement > 0.001 or angle > 0.015:
        return "pick_object_disturbed", measured
    if lateral > PICK_LATERAL_BOUND_M or tcp_angle > PICK_ANGLE_BOUND_RAD:
        return "pick_entry_tracking_error", measured
    if (
        not end.position_xyz[2] - 0.001
        <= sample.tcp_pose.position_xyz[2]
        <= start.position_xyz[2] + 0.002
    ):
        return "pick_entry_axial_bound", measured
    report = scene.get("pad_object_contacts")
    if (
        not isinstance(report, Mapping)
        or report.get("schema") != "rocobrick-pad-object-contacts/v1"
        or report.get("source") != "physx_contact_report"
        or not isinstance(report.get("contacts"), list)
        or any(
            report.get(k) != scene.get(k)
            for k in ("session_id", "manifest_sha256", "sequence", "sim_time_s")
        )
    ):
        raise ValueError("PICK entry contact sensing missing/stale")
    contacts = [c for c in report["contacts"] if c.get("robot_id") == guard["robot_id"]]
    force = sum(float(c["normal_force_n"]) for c in contacts)
    if not math.isfinite(force) or force < 0:
        raise ValueError("PICK entry force invalid")
    measured["finger_force_n"] = force
    # This stage positions OPEN fingers. Contact is allowed only by the later
    # handler-owned close phase, never by silently accepting a premature hit.
    if force > 0:
        return "pick_premature_contact", measured
    return None, measured
