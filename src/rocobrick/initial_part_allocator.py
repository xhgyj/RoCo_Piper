"""Task-independent initial loose-part slot allocation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import hypot


@dataclass(frozen=True)
class InitialPartAllocation:
    """One deterministic loose-part placement decision."""

    physical_instance_id: str
    position_xy_m: tuple[float, float]
    preferred_robot_id: str
    reason: Mapping[str, object]


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return hypot(left[0] - right[0], left[1] - right[1])


def allocate_initial_slots(
    physical_instance_ids: Sequence[str],
    *,
    preferred_robots: Mapping[str, str],
    robot_base_xy: Mapping[str, tuple[float, float]],
    board_center_xy: tuple[float, float],
    board_half_extent_xy: tuple[float, float],
    minimum_part_clearance_m: float = 0.055,
    minimum_robot_clearance_m: float = 0.09,
    approach_offset_m: float = 0.03,
) -> Mapping[str, InitialPartAllocation]:
    """Allocate loose bricks near their first execution robot.

    The allocator intentionally does not perform IK or reject a slot using a
    static reachability heuristic. It chooses a collision-free ground slot
    outside the baseplate footprint; MoveIt remains responsible for actual
    motion feasibility.
    """
    if not physical_instance_ids:
        return {}
    if (
        minimum_part_clearance_m <= 0
        or minimum_robot_clearance_m <= 0
        or approach_offset_m < 0
    ):
        raise ValueError("allocator clearances must be positive")
    if board_half_extent_xy[0] <= 0 or board_half_extent_xy[1] <= 0:
        raise ValueError("board extents must be positive")
    missing = set(physical_instance_ids) - set(preferred_robots)
    if missing:
        raise ValueError(f"missing preferred robot for loose parts: {sorted(missing)}")
    if set(preferred_robots.values()) - set(robot_base_xy):
        raise ValueError("preferred robot is not present in robot_base_xy")

    cx, cy = board_center_xy
    hx, hy = board_half_extent_xy
    candidates: list[tuple[float, float]] = []
    # Loose parts belong on the ground outside the baseplate. Generate slots
    # just beyond the nearest baseplate edge, with a small lateral fan. This
    # keeps the slot in the robot's working corridor; distance from the base
    # alone is not an IK/reachability criterion.
    for robot_id in sorted(robot_base_xy):
        bx, by = robot_base_xy[robot_id]
        dx, dy = cx - bx, cy - by
        if abs(dx) >= abs(dy):
            edge_x = cx + (hx + 0.06) * (1.0 if bx >= cx else -1.0)
            # Move away from the robot base. For piper_0, whose base is below
            # the board (by < cy), this increases world Y rather than moving
            # the loose part back toward the base.
            forward = 1.0 if by < cy else -1.0
            for lateral in (
                -0.08 + forward * approach_offset_m,
                0.0 + forward * approach_offset_m,
                0.08 + forward * approach_offset_m,
            ):
                candidates.append((edge_x, cy + lateral))
        else:
            edge_y = cy + (hy + 0.06) * (1.0 if by >= cy else -1.0)
            for lateral in (-0.08, 0.0, 0.08):
                candidates.append((cx + lateral, edge_y))

    allocated: dict[str, InitialPartAllocation] = {}
    occupied: list[tuple[float, float]] = []
    for instance_id in sorted(physical_instance_ids):
        robot_id = preferred_robots[instance_id]
        base = robot_base_xy[robot_id]
        ranked = sorted(
            candidates,
            key=lambda point: (
                _distance(point, base),
                _distance(point, (cx, cy)),
                point[0],
                point[1],
            ),
        )
        selected = next(
            (
                point
                for point in ranked
                if _distance(point, base) >= minimum_robot_clearance_m
                and not (
                    abs(point[0] - cx) < hx + minimum_part_clearance_m
                    and abs(point[1] - cy) < hy + minimum_part_clearance_m
                )
                and all(
                    _distance(point, previous) >= minimum_part_clearance_m
                    for previous in occupied
                )
            ),
            None,
        )
        if selected is None:
            raise ValueError(f"no collision-free initial slot for {instance_id}")
        occupied.append(selected)
        allocated[instance_id] = InitialPartAllocation(
            instance_id,
            selected,
            robot_id,
            {
                "preferred_robot_id": robot_id,
                "distance_to_robot_base_m": _distance(selected, base),
                "distance_to_board_center_m": _distance(selected, (cx, cy)),
                "allocator": "initial_part_allocator/v1",
            },
        )
    return allocated


__all__ = ["InitialPartAllocation", "allocate_initial_slots"]
