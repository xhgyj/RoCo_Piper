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


def planar_yaw_degrees(world_transform: np.ndarray) -> float:
    """Return the world heading of a flat brick's local positive x axis.

    Returns:
        Heading in degrees in ``[-180, 180]``.
    """
    transform = np.asarray(world_transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("world_transform must be 4x4")
    x_axis = transform[:3, 0]
    if np.linalg.norm(x_axis[:2]) < 1e-9:
        raise ValueError("brick local x axis has no planar projection")
    return float(np.rad2deg(np.arctan2(x_axis[1], x_axis[0])))


def with_planar_yaw(
    world_transform: np.ndarray, yaw_degrees: float
) -> np.ndarray:
    """Return a transform with the requested absolute world yaw.

    The rotation is applied about world Z through the existing brick center,
    preserving position and the brick's current tilt.

    Returns:
        Copied homogeneous transform at the requested planar heading.
    """
    transform = np.asarray(world_transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("world_transform must be 4x4")
    if not np.isfinite(transform).all() or not np.isfinite(yaw_degrees):
        raise ValueError("yaw transform inputs must be finite")
    current = np.deg2rad(planar_yaw_degrees(transform))
    desired = np.deg2rad(float(yaw_degrees))
    delta = desired - current
    cosine = np.cos(delta)
    sine = np.sin(delta)
    rotate_z = np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    result = transform.copy()
    result[:3, :3] = rotate_z @ transform[:3, :3]
    return result


def aabb_inside(
    inner: RectangleAabb,
    outer: RectangleAabb,
    tolerance: float = 1e-9,
) -> bool:
    """Return whether one XY AABB is fully contained in another."""
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    return bool(
        inner.minimum[0] >= outer.minimum[0] - tolerance
        and inner.minimum[1] >= outer.minimum[1] - tolerance
        and inner.maximum[0] <= outer.maximum[0] + tolerance
        and inner.maximum[1] <= outer.maximum[1] + tolerance
    )


def centered_aabb(
    center_xy: tuple[float, float], size_xy: tuple[float, float]
) -> RectangleAabb:
    """Return axis-aligned bounds from a center and full XY size."""
    if len(center_xy) != 2 or len(size_xy) != 2:
        raise ValueError("center_xy and size_xy must contain two values")
    if size_xy[0] <= 0.0 or size_xy[1] <= 0.0:
        raise ValueError("AABB size must be positive")
    half_x = float(size_xy[0]) * 0.5
    half_y = float(size_xy[1]) * 0.5
    return RectangleAabb(
        minimum=(float(center_xy[0]) - half_x, float(center_xy[1]) - half_y),
        maximum=(float(center_xy[0]) + half_x, float(center_xy[1]) + half_y),
    )
