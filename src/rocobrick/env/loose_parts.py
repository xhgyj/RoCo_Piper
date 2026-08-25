"""Geometry checks for loose bricks spawned outside the base plate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BRICK_UNIT_LENGTH = 0.008


@dataclass(frozen=True)
class RectangleAabb:
    """Axis-aligned XY bounds of an oriented rectangular footprint."""

    minimum: tuple[float, float]
    maximum: tuple[float, float]


def footprint_aabb(
    world_transform: np.ndarray, length_studs: int, width_studs: int
) -> RectangleAabb:
    """Return the world XY AABB of a brick footprint centered at its origin."""
    transform = np.asarray(world_transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("world_transform must be 4x4")
    half_length = length_studs * BRICK_UNIT_LENGTH * 0.5
    half_width = width_studs * BRICK_UNIT_LENGTH * 0.5
    local = np.array(
        [
            [-half_length, -half_width, 0.0, 1.0],
            [-half_length, half_width, 0.0, 1.0],
            [half_length, -half_width, 0.0, 1.0],
            [half_length, half_width, 0.0, 1.0],
        ]
    )
    world = (transform @ local.T).T
    return RectangleAabb(
        minimum=(float(world[:, 0].min()), float(world[:, 1].min())),
        maximum=(float(world[:, 0].max()), float(world[:, 1].max())),
    )


def aabb_clearance(first: RectangleAabb, second: RectangleAabb) -> float:
    """Return XY separation, or zero when the two AABBs overlap."""
    gap_x = max(
        first.minimum[0] - second.maximum[0],
        second.minimum[0] - first.maximum[0],
        0.0,
    )
    gap_y = max(
        first.minimum[1] - second.maximum[1],
        second.minimum[1] - first.maximum[1],
        0.0,
    )
    return float(np.hypot(gap_x, gap_y))


def format_aabb(bounds: RectangleAabb) -> str:
    """Return concise bounds for simulator error messages."""
    return f"min={bounds.minimum}, max={bounds.maximum}"
