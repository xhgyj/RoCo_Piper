"""Object-relative PICK regions sampled inside MoveIt, not by the host."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path

from rocobrick.gripper import pad_separation_to_logical_opening
from rocobrick.moveit_contract import JOINT_NAMES
from rocobrick.skills.observation import (
    PickObservation,
    Pose,
    _pose,
    _vector,
    parse_pick_observation,
)
from rocobrick.skills.placement_geometry import compose, inverse, pose_error
from rocobrick.skills.targets import (
    PickTargets,
    _quat_rotate_vector,
    world_orientation_to_base,
    world_position_to_base,
)

GRASP_REGION_SCHEMA = "rocobrick-grasp-region/v1"
# Measured-arrival admission bound.  The position-controlled arm settles
# 10-16 mrad off the orientation it was commanded (2026-10-18 evidence), so the
# closing-axis alignment check has to admit that residual instead of 15 mrad.
ARRIVAL_ANGLE_TOLERANCE_RAD = 0.05
# Experimental PICK-only platform allowance: retain 4 mm positive clearance.
# This is not a calibrated physical safety guarantee or a global scene padding.
PICK_PLATFORM_MARGIN_M = 0.004
CALIBRATION_PATH = (
    Path(__file__).resolve().parents[3]
    / "artifacts/model-audit/piper_l_tcp_calibration.json"
)


def calibrated_pad_geometry(path: Path, expected_sha256: str) -> dict[str, object]:
    """Read the existing pinned contact patches in the calibrated TCP frame.

    Returns:
        Contact-patch vertices at zero opening and the measured flange offset.
    """
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("grasp contact geometry calibration digest mismatch")
    data = json.loads(raw)
    if data.get("passed") is not True or data.get("method") != (
        "symmetric-opposed-contact-patch-centroids"
    ):
        raise ValueError("grasp contact geometry requires passed TCP calibration")
    translation = _vector(data["aggregate"]["translation_median_m"], 3, "TCP offset")
    sample = next(s for s in data["samples"] if s["opening_m"] == 0.0)
    matrix = _vector(sample["link6_world_matrix"], 16, "link6 matrix")
    vertices = []
    for patch in sample["patches"].values():
        for triangle in patch["triangles"]:
            for raw_vertex in triangle["world_vertices_m"]:
                point = _vector(raw_vertex, 3, "patch vertex")
                vertices.append(
                    [
                        sum(
                            matrix[4 * j + i] * (point[j] - matrix[4 * j + 3])
                            for j in range(3)
                        )
                        - translation[i]
                        for i in range(3)
                    ]
                )
    if not vertices:
        raise ValueError("calibration has no contact-patch vertices")
    return {"pad_vertices_tcp": vertices, "flange_to_tcp_m": translation[2]}


def supporting_surface_top(sample: PickObservation, context) -> float:
    """Resolve the measured brick's support, never a lower floor beneath it.

    Configured platforms use the same boxes as simulator initialization. A
    1 mm observation tolerance permits settling noise, not a different level.
    Without platform configuration the existing ground-only contract remains.

    Returns:
        The world Z of the unique measured supporting surface.
    """
    platforms = context.feeding_platforms
    if not platforms:
        return context.ground.top_z_m if context.ground else sample.aabb_min[2]
    candidates = []
    for platform in platforms:
        if (
            all(
                sample.aabb_min[i] >= platform.minimum_m[i] - 0.001
                and sample.aabb_max[i] <= platform.maximum_m[i] + 0.001
                for i in (0, 1)
            )
            and abs(sample.aabb_min[2] - platform.top_z_m) <= 0.001
        ):
            candidates.append(platform.top_z_m)
    ground = context.ground
    if (
        ground is not None
        and all(
            abs((sample.aabb_min[i] + sample.aabb_max[i]) / 2 - ground.center_xy_m[i])
            + (sample.aabb_max[i] - sample.aabb_min[i]) / 2
            <= ground.size_xy_m[i] / 2 + 0.001
            for i in (0, 1)
        )
        and abs(sample.aabb_min[2] - ground.top_z_m) <= 0.001
    ):
        candidates.append(ground.top_z_m)
    if len(candidates) != 1:
        raise ValueError("PICK_SUPPORT_UNRESOLVED_OR_AMBIGUOUS")
    return candidates[0]


def place_up_grasp_region(sample: PickObservation, context, *, geometry=None):
    """Build one continuous side-pinch region without relocating the brick.

    Returns:
        A JSON-safe region bound to the live object and immutable calibration.
    """
    if geometry is None:
        geometry = calibrated_pad_geometry(
            CALIBRATION_PATH, context.provenance.source_sha256
        )
    local = inverse(sample.object_pose)
    corners = [
        compose(local, Pose((x, y, z), (0.0, 0.0, 0.0, 1.0))).position_xyz
        for x in (sample.aabb_min[0], sample.aabb_max[0])
        for y in (sample.aabb_min[1], sample.aabb_max[1])
        for z in (sample.aabb_min[2], sample.aabb_max[2])
    ]
    minimum = [min(p[i] for p in corners) for i in range(3)]
    maximum = [max(p[i] for p in corners) for i in range(3)]
    base = Pose(context.base.position_xyz, context.base.orientation_xyzw)
    support_top = supporting_surface_top(sample, context)
    on_platform = any(
        abs(platform.top_z_m - support_top) <= 1e-9
        for platform in context.feeding_platforms
    )
    region = {
        "schema": GRASP_REGION_SCHEMA,
        "robot_id": sample.robot_id,
        "physical_instance_id": sample.physical_instance_id,
        "frame_id": f"{sample.robot_id}_base_link",
        "tip_link": f"{sample.robot_id}_rocobrick_tcp",
        "runtime_model_sha256": context.provenance.runtime_model_sha256,
        "source_sha256": context.provenance.source_sha256,
        "object_in_base": asdict(compose(inverse(base), sample.object_pose)),
        "center": [(a + b) / 2.0 for a, b in zip(minimum, maximum, strict=True)],
        "half_extents": [(b - a) / 2.0 for a, b in zip(minimum, maximum, strict=True)],
        "ground_top_world_m": support_top,
        "base_in_world": asdict(base),
        "floor_margin_m": PICK_PLATFORM_MARGIN_M if on_platform else 0.005,
        "contact_margin_m": min(0.001, (maximum[2] - minimum[2]) * 0.1),
        "pad_vertices_tcp": geometry["pad_vertices_tcp"],
        "flange_to_tcp_m": geometry["flange_to_tcp_m"],
        "approach_opening_m": min(
            0.1,
            max(maximum[i] - minimum[i] for i in (0, 1))
            + context.params.open_extra_m
            + 0.01,
        ),
    }
    validate_grasp_region(region)
    return region


def validate_grasp_region(region: Mapping[str, object]) -> None:
    """Validate the wire contract before creating MoveIt constraints."""
    required = {
        "schema",
        "robot_id",
        "physical_instance_id",
        "frame_id",
        "tip_link",
        "runtime_model_sha256",
        "source_sha256",
        "object_in_base",
        "center",
        "half_extents",
        "ground_top_world_m",
        "base_in_world",
        "floor_margin_m",
        "contact_margin_m",
        "pad_vertices_tcp",
        "flange_to_tcp_m",
        "approach_opening_m",
    }
    if not isinstance(region, Mapping) or set(region) != required:
        raise ValueError("grasp_region keys are invalid")
    robot = region["robot_id"]
    if region["schema"] != GRASP_REGION_SCHEMA or robot not in ("piper_0", "piper_1"):
        raise ValueError("grasp_region schema/robot is invalid")
    if (
        region["frame_id"] != f"{robot}_base_link"
        or region["tip_link"] != f"{robot}_rocobrick_tcp"
    ):
        raise ValueError("grasp_region frame/tip is not robot-scoped")
    if (
        not isinstance(region["physical_instance_id"], str)
        or not region["physical_instance_id"]
    ):
        raise ValueError("grasp_region object identity is required")
    for key in ("runtime_model_sha256", "source_sha256"):
        digest = region[key]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("grasp_region calibration digest is invalid")
    for key in ("object_in_base", "base_in_world"):
        _pose(region[key], key)
    _vector(region["center"], 3, "center")
    extents = _vector(region["half_extents"], 3, "half extents")
    if min(extents) <= 0:
        raise ValueError("grasp_region extents must be positive")
    for key in (
        "ground_top_world_m",
        "floor_margin_m",
        "contact_margin_m",
        "flange_to_tcp_m",
        "approach_opening_m",
    ):
        _vector([region[key]], 1, key)
    if (
        not 0 < region["approach_opening_m"] <= 0.1
        or region["floor_margin_m"] < 0
        or not 0 < region["contact_margin_m"] < extents[2]
    ):
        raise ValueError("grasp_region opening/margins are invalid")
    if region["flange_to_tcp_m"] <= 0:
        raise ValueError("grasp_region flange offset is invalid")
    vertices = region["pad_vertices_tcp"]
    if not isinstance(vertices, (tuple, list)) or not vertices or len(vertices) > 128:
        raise ValueError("grasp_region pad vertices are invalid")
    for point in vertices:
        _vector(point, 3, "pad vertex")


def validate_grasp_scene(region, scene_snapshot) -> None:
    """Require the region's object pose to come from the bound shared scene."""
    validate_grasp_region(region)
    bricks = scene_snapshot.get("bricks")
    brick = (
        bricks.get(region["physical_instance_id"])
        if isinstance(bricks, Mapping)
        else None
    )
    if not isinstance(brick, Mapping):
        raise ValueError("grasp region object is missing from shared snapshot")
    measured = _pose(brick.get("pose"), "scene object")
    bound = compose(
        _pose(region["base_in_world"], "base"),
        _pose(region["object_in_base"], "object"),
    )
    distance, angle = pose_error(measured, bound)
    if distance > 1e-6 or angle > 1e-5:
        raise ValueError("grasp region object pose does not match shared snapshot")


def admit_grasp_endpoint(
    region,
    sample: PickObservation,
    *,
    tolerance_m=0.001,
    angle_tolerance_rad=0.015,
    center_tolerance_m=None,
):
    """Check actual pad centroids, floor envelope and exposed stud corridor.

    ``tolerance_m`` stays the object-drift and floor-envelope bound; the
    centring checks use ``center_tolerance_m`` (defaulting to it) because the
    measured arrival cannot be tighter than the servo residual.

    Returns:
        The actual brick span on the measured closing axis, not holding proof.
    """
    validate_grasp_region(region)
    if center_tolerance_m is None:
        center_tolerance_m = tolerance_m
    if (
        sample.robot_id != region["robot_id"]
        or sample.physical_instance_id != region["physical_instance_id"]
    ):
        raise ValueError("grasp endpoint identity mismatch")
    base = _pose(region["base_in_world"], "base")
    expected_object = compose(base, _pose(region["object_in_base"], "object"))
    displacement, angle = pose_error(expected_object, sample.object_pose)
    if displacement > tolerance_m or angle > angle_tolerance_rad:
        raise ValueError("brick moved during grasp approach")
    relative = compose(inverse(sample.object_pose), sample.tcp_pose)
    closing = _quat_rotate_vector(relative.orientation_xyzw, (0.0, 1.0, 0.0))
    axis = max((0, 1), key=lambda i: abs(closing[i]))
    if abs(closing[axis]) < math.cos(angle_tolerance_rad) or abs(closing[2]) > math.sin(
        angle_tolerance_rad
    ):
        raise ValueError("grasp closing axis misses opposed brick side faces")
    center = region["center"]
    extents = region["half_extents"]
    delta = [relative.position_xyz[i] - center[i] for i in range(3)]
    tangent = 1 - axis
    margin = region["contact_margin_m"]
    if (
        abs(delta[axis]) > center_tolerance_m
        or abs(delta[tangent]) > extents[tangent] * 0.5 + center_tolerance_m
        or abs(delta[2]) > extents[2] - margin + center_tolerance_m
    ):
        raise ValueError("TCP pad centroids miss the graspable side-wall region")
    approach = _quat_rotate_vector(relative.orientation_xyzw, (0.0, 0.0, 1.0))
    flange = [delta[i] - region["flange_to_tcp_m"] * approach[i] for i in range(3)]
    if (
        abs(flange[0]) < extents[0]
        and abs(flange[1]) < extents[1]
        and flange[2] > extents[2]
    ):
        raise ValueError("gripper palm occupies the exposed stud corridor")
    if abs(approach[2]) >= math.cos(angle_tolerance_rad):
        raise ValueError("PLACE_UP requires side-entry, not top/bottom grasp")
    for vertex in region["pad_vertices_tcp"]:
        world = compose(sample.tcp_pose, Pose(tuple(vertex), (0.0, 0.0, 0.0, 1.0)))
        if (
            world.position_xyz[2]
            < region["ground_top_world_m"] + region["floor_margin_m"] - tolerance_m
        ):
            raise ValueError("grasp contact-patch envelope crosses the floor")
    span = 2.0 * sum(abs(closing[i]) * extents[i] for i in range(3))
    if sample.opening_m < span + margin:
        raise ValueError("gripper is not open beyond the brick span")
    return span


def targets_from_grasp_endpoint(region, sample, context, original: PickTargets):
    """Use measured selected orientation for close/lift and later attachment.

    Returns:
        Existing PICK targets updated from the actual arrived TCP.
    """
    # The pads still close onto the wall while the finger centre sits offset by
    # less than the closing squeeze, so that is the physical centring bound for
    # a measured arrival (the servo holds it several millimetres off target).
    span = admit_grasp_endpoint(
        region,
        sample,
        center_tolerance_m=context.params.squeeze_m,
        angle_tolerance_rad=ARRIVAL_ANGLE_TOLERANCE_RAD,
    )
    position = world_position_to_base(sample.tcp_pose.position_xyz, context.base)
    orientation = world_orientation_to_base(
        sample.tcp_pose.orientation_xyzw, context.base
    )
    lift_world = (
        *sample.tcp_pose.position_xyz[:2],
        sample.tcp_pose.position_xyz[2] + context.params.lift_height_m,
    )
    contact = pad_separation_to_logical_opening(span)
    return replace(
        original,
        precontact_boundary=replace(
            original.precontact_boundary,
            position=position,
            orientation_xyzw=orientation,
        ),
        lift=replace(
            original.lift,
            position=world_position_to_base(lift_world, context.base),
            orientation_xyzw=orientation,
        ),
        brick_span_m=span,
        close_opening_m=pad_separation_to_logical_opening(
            span - context.params.squeeze_m
        ),
        close_expected_window_m=(
            contact - context.params.contact_window_slop_m,
            contact + context.params.contact_window_slop_m,
        ),
    )


def planned_grasp_endpoints(session, request, joint_names, points):
    """Verify planned region endpoints by model FK before any controller prepare.

    Returns:
        Selected TCP poses/spans, still explicitly no physical holding proof.
    """
    endpoints = {}
    final = dict(zip(joint_names, points[-1]["positions"], strict=True))
    combined = {**request.start_state, **final}
    for robot, target in request.targets.items():
        if "grasp_region" not in target:
            continue
        region = target["grasp_region"]
        compute = getattr(session.runtime_for(robot), "compute_tcp_fk", None)
        if not callable(compute):
            raise ValueError("grasp region requires endpoint FK verification")
        # Combined MoveIt state includes the mimic parent ``gripper``. Keep
        # it in planning; the existing FK adapter requires exactly 8 joints.
        missing = [name for name in JOINT_NAMES if f"{robot}_{name}" not in combined]
        if missing:
            raise ValueError(f"grasp endpoint FK missing joints for {robot}: {missing}")
        canonical = {name: combined[f"{robot}_{name}"] for name in JOINT_NAMES}
        fk = compute(canonical, timeout_s=min(2.0, request.timeout_s))
        if fk is None:
            raise ValueError("grasp region endpoint FK is unavailable")
        base = _pose(region["base_in_world"], "base")
        selected = compose(base, Pose(tuple(fk[0]), tuple(fk[1])))
        snapshot = request.scene_snapshot
        measured = parse_pick_observation(
            snapshot,
            robot_id=robot,
            physical_instance_id=region["physical_instance_id"],
            session_id=snapshot["session_id"],
            manifest_sha256=snapshot["manifest_sha256"],
        )
        predicted = replace(measured, tcp_pose=selected)
        span = admit_grasp_endpoint(region, predicted)
        endpoints[robot] = {
            "tcp_pose": asdict(selected),
            "span_m": span,
            "physical_holding_confirmed": False,
        }
    return endpoints
