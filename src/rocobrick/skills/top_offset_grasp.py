"""Vertical offset PICK geometry derived from immutable structural footprints."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace

from rocobrick.skills.observation import PickObservation, Pose
from rocobrick.skills.placement_geometry import compose, inverse, pose_error
from rocobrick.skills.targets import (
    PickTargets,
    _quat_mul,
    _quat_rotate_vector,
    world_orientation_to_base,
    world_position_to_base,
)
from rocobrick.stap_contract import StapPlan


@dataclass(frozen=True)
class TopOffsetBinding:
    """Structural free strip expressed in the lower brick's canonical frame."""

    lower_id: str
    upper_id: str
    free_axis: int
    free_sign: int
    free_length_m: float
    span_m: float
    footprint_m: tuple[float, float]
    overlap_fraction: float


def top_offset_bindings(
    structure: StapPlan, operations, library: Mapping[str, object]
) -> Mapping[str, TopOffsetBinding]:
    """Bind PICK to its explicit PLACE_UP successor, without changing operations.

    Returns:
        Operation-scoped rectangular free strips from the original grid records.
    """
    bindings = {}
    for pick in operations.operations.values():
        if pick.skill != "PICK" or pick.parameters.get("successor_skill") != "PLACE_UP":
            continue
        lower_id = pick.target.values["physical_instance_id"]
        successors = [
            item
            for item in operations.operations.values()
            if item.skill == "PLACE_UP"
            and item.source_task_id == pick.source_task_id
            and item.target.values["lower_physical_instance_id"] == lower_id
        ]
        if len(successors) != 1:
            raise ValueError(
                "vertical offset PICK requires one bound PLACE_UP successor"
            )
        upper_id = successors[0].target.values["upper_physical_instance_id"]
        lower, upper = (structure.goal_parts[key] for key in (lower_id, upper_id))

        def dimensions(part):
            row = library[str(part.brick_id)]
            size = (int(row["height"]), int(row["width"]))
            return size[::-1] if part.orientation_quarter_turns % 2 else size

        ld, ud = dimensions(lower), dimensions(upper)
        lo, up = (lower.x, lower.y), (upper.x, upper.y)
        overlap = tuple(
            (max(lo[i], up[i]), min(lo[i] + ld[i], up[i] + ud[i])) for i in (0, 1)
        )
        if upper.z != lower.z + 1 or any(b <= a for a, b in overlap):
            raise ValueError(
                "vertical offset PICK requires adjacent overlapping bricks"
            )
        strips = []
        for axis in (0, 1):
            if overlap[1 - axis] != (lo[1 - axis], lo[1 - axis] + ld[1 - axis]):
                continue
            strips.extend(
                [
                    (overlap[axis][0] - lo[axis], axis, -1),
                    (lo[axis] + ld[axis] - overlap[axis][1], axis, 1),
                ]
            )
        length, axis, sign = max(strips, default=(0, 0, 1))
        if length <= 0:
            raise ValueError("vertical offset PICK has no exposed rectangular strip")
        # BrickSim canonicalizes the longer dimension onto local X.
        canonical_yaw = 0 if ld[0] >= ld[1] else 1
        vector = [0, 0]
        vector[axis] = sign
        local = (vector[1], -vector[0]) if canonical_yaw else tuple(vector)
        local_axis = 0 if local[0] else 1
        footprint = (max(ld) * 0.008, min(ld) * 0.008)
        bindings[pick.operation_id] = TopOffsetBinding(
            lower_id,
            upper_id,
            local_axis,
            local[local_axis],
            length * 0.008,
            footprint[1 - local_axis],
            footprint,
            math.prod(b - a for a, b in overlap) / math.prod(ld),
        )
    return bindings


def top_offset_targets(
    binding: TopOffsetBinding, sample: PickObservation, context, original: PickTargets
) -> tuple[PickTargets, Pose, dict[str, object]]:
    """Place the pad centre in the free strip with a downward tool and horizontal close.

    Returns:
        Exact targets, contact pose and auditable structural geometry.
    """
    if sample.physical_instance_id != binding.lower_id:
        raise ValueError("vertical offset PICK object identity mismatch")
    normal = _quat_rotate_vector(sample.object_pose.orientation_xyzw, (0, 0, 1))
    if normal[2] < math.cos(0.03):
        raise ValueError("vertical offset PICK requires upright studs")
    local = [0.0, 0.0, 0.0]
    # The free strip is the only thing keeping the jaws out of the mated upper
    # brick, and the strip's *centre* left the jaws' upper part over the
    # covered half: at the place-up seat the jaw faces grazed the upper brick
    # and MoveIt rejected the whole insertion plan as a collision (live
    # evidence, task-c-top-offset13k-headless 2026-10-14: invalid LIN states
    # 6..11 of 12, contacts between brick-794d and piper_0_gripper_link1/2).
    # Deeper is not better either: 6 mm of shift bought 3.3 mm of seat
    # clearance but the longer moment arm let the brick slip in the jaws at
    # the stud engagement (task-c-top-offset13l-headless: grasp_slipped after
    # 3.2 mm of slide, ~2 mm below the seat).  The pad centre therefore moves
    # by min(4.5 mm, 30% of the strip): 3 mm still left the plan's last states
    # in contact (13o: invalid 10..13 of 14), 6 mm cleared it but let the brick
    # slip at the engagement (13l).  The jaws are far wider than the strip, so
    # every strip point stays covered by the pads.
    deeper = min(0.0045, binding.free_length_m * 0.3)
    local[binding.free_axis] = binding.free_sign * (
        binding.footprint_m[binding.free_axis] / 2
        - binding.free_length_m / 2
        + deeper
    )
    # The live AABB anchors contact height; no platform height is hardcoded.
    local[2] = (sample.aabb_max[2] - sample.aabb_min[2]) / 2 + 0.0035
    yaw = math.pi / 2 if binding.free_axis == 1 else 0.0
    orientation = _quat_mul(
        (0, 0, math.sin(yaw / 2), math.cos(yaw / 2)),
        (1, 0, 0, 0),
    )
    # Rotate about the horizontal closing axis, leaning the wrist toward the
    # free strip while keeping both pad normals aligned with the brick sides.
    free = [0.0, 0.0, 0.0]
    free[binding.free_axis] = binding.free_sign
    axis = (-free[1], free[0], 0.0)
    half_angle = math.radians(context.params.top_offset_pick_tilt_deg) / 2
    lean = (
        *tuple(value * math.sin(half_angle) for value in axis),
        math.cos(half_angle),
    )
    orientation = _quat_mul(lean, orientation)
    contact = compose(sample.object_pose, Pose(tuple(local), orientation))
    base_orientation = world_orientation_to_base(contact.orientation_xyzw, context.base)

    def target(template, height):
        world = (*contact.position_xyz[:2], contact.position_xyz[2] + height)
        return replace(
            template,
            position=world_position_to_base(world, context.base),
            orientation_xyzw=base_orientation,
            position_tolerance_m=0.002,
        )

    closing = _quat_rotate_vector(contact.orientation_xyzw, (0, 1, 0))
    span = sum(
        abs(closing[i]) * (sample.aabb_max[i] - sample.aabb_min[i]) for i in range(3)
    )
    # Wider approach opening: tracking error on a tilted free-roll grasp left
    # only ~7 mm of finger clearance and pushed the brick (live evidence,
    # task-c-top-offset3). 80 mm keeps ~24 mm per side for the 32 mm brick.
    opening = min(0.1, max(span + context.params.open_extra_m + 0.01, 0.08))
    targets = replace(
        original,
        precontact_boundary=target(original.precontact_boundary, 0),
        lift=target(original.lift, context.params.lift_height_m),
        brick_span_m=span,
        reach_opening_m=opening,
        reach_expected_window_m=(opening - 0.01, min(0.1, opening + 0.01)),
        close_opening_m=span - context.params.squeeze_m,
        close_expected_window_m=(
            span - context.params.contact_window_slop_m,
            span + context.params.contact_window_slop_m,
        ),
    )
    return (
        targets,
        contact,
        {
            **asdict(binding),
            "mode": "vertical_offset",
            "downward_tilt_deg": context.params.top_offset_pick_tilt_deg,
            "orientation_branches": 2,
            "approach_opening_m": opening,
            "local_contact_xyz_m": local,
            "planned_contact_pose": asdict(contact),
            "place_up_compatibility_confirmed": False,
        },
    )


def verify_top_offset_arrival(reference, current, contact: Pose) -> None:
    """Require unchanged brick, target position, tool-down and side closing.

    The wrist roll about the tool axis is planner-selected (both jaw
    orientations are physically equivalent), so the measured orientation is
    checked through the grasp geometry instead of a pinned quaternion.
    """
    displacement, angle = pose_error(reference.object_pose, current.object_pose)
    if displacement > 0.001 or angle > 0.015:
        raise ValueError("brick moved during vertical grasp approach")
    if math.dist(current.tcp_pose.position_xyz, contact.position_xyz) > 0.004:
        raise ValueError("vertical offset grasp arrival pose is outside target")
    relative = compose(inverse(current.object_pose), current.tcp_pose)
    rotation = relative.orientation_xyzw
    measured_axis = _quat_rotate_vector(rotation, (0, 0, 1))
    expected_relative = compose(inverse(current.object_pose), contact)
    expected_axis = _quat_rotate_vector(expected_relative.orientation_xyzw, (0, 0, 1))
    if expected_axis[2] > -math.cos(math.radians(35.0) + 0.03) or sum(
        a * b for a, b in zip(measured_axis, expected_axis)
    ) < math.cos(0.03):
        raise ValueError(
            "vertical offset grasp tool is not downward at its target axis"
        )
    closing = _quat_rotate_vector(rotation, (0, 1, 0))
    axis = max((0, 1), key=lambda i: abs(closing[i]))
    if abs(closing[axis]) < math.cos(0.03) or abs(closing[2]) > math.sin(0.03):
        raise ValueError("vertical offset grasp closing axis misses the side faces")
