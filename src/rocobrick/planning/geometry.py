"""OBB collision and discretized swept-volume utilities."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from rocobrick.planning.models import FloatArray, GripperGeometry, OrientedBox


def interpolate_poses(
    start: FloatArray,
    goal: FloatArray,
    translation_step: float = 0.001,
    rotation_step: float = np.deg2rad(2.0),
) -> tuple[FloatArray, ...]:
    """Sample a rigid motion under translation and rotation step bounds.

    Returns:
        Endpoint-inclusive sequence of homogeneous poses.
    """
    relative = start[:3, :3].T @ goal[:3, :3]
    rotvec = Rotation.from_matrix(relative).as_rotvec()
    count = max(
        1,
        math.ceil(float(np.linalg.norm(goal[:3, 3] - start[:3, 3])) / translation_step),
        math.ceil(float(np.linalg.norm(rotvec)) / rotation_step),
    )
    poses: list[FloatArray] = []
    for index in range(count + 1):
        fraction = index / count
        pose = np.eye(4)
        pose[:3, :3] = (
            start[:3, :3] @ Rotation.from_rotvec(rotvec * fraction).as_matrix()
        )
        pose[:3, 3] = start[:3, 3] * (1.0 - fraction) + goal[:3, 3] * fraction
        poses.append(pose)
    return tuple(poses)


def interpolate_pose(
    start: FloatArray, goal: FloatArray, fraction: float
) -> FloatArray:
    """Interpolate one rigid pose at a normalized path fraction.

    Returns:
        Homogeneous pose between the supplied endpoints.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    relative = start[:3, :3].T @ goal[:3, :3]
    rotvec = Rotation.from_matrix(relative).as_rotvec()
    pose = np.eye(4)
    pose[:3, :3] = start[:3, :3] @ Rotation.from_rotvec(rotvec * fraction).as_matrix()
    pose[:3, 3] = start[:3, 3] * (1.0 - fraction) + goal[:3, 3] * fraction
    return pose


def obb_axis_separations(left: OrientedBox, right: OrientedBox) -> NDArray[np.float64]:
    """Return SAT separation on all nondegenerate candidate axes."""
    axes: list[NDArray[np.float64]] = []
    left_axes = left.world_t_box[:3, :3]
    right_axes = right.world_t_box[:3, :3]
    axes.extend(left_axes[:, index] for index in range(3))
    axes.extend(right_axes[:, index] for index in range(3))
    for left_index in range(3):
        for right_index in range(3):
            axis = np.cross(left_axes[:, left_index], right_axes[:, right_index])
            norm = float(np.linalg.norm(axis))
            if norm > 1e-9:
                axes.append(axis / norm)
    center_delta = right.world_t_box[:3, 3] - left.world_t_box[:3, 3]
    separations = []
    for axis in axes:
        left_radius = float(np.sum(np.abs(left_axes.T @ axis) * left.half_extents))
        right_radius = float(np.sum(np.abs(right_axes.T @ axis) * right.half_extents))
        separations.append(
            abs(float(np.dot(center_delta, axis))) - left_radius - right_radius
        )
    return np.asarray(separations, dtype=np.float64)


def obb_intersects(left: OrientedBox, right: OrientedBox) -> bool:
    """Return whether two oriented boxes overlap under the SAT."""
    _, intersects = obb_clearance_and_intersection(left, right)
    return intersects


def obb_clearance(left: OrientedBox, right: OrientedBox) -> float:
    """Return a conservative signed separation estimate."""
    clearance, _ = obb_clearance_and_intersection(left, right)
    return clearance


def obb_clearance_and_intersection(
    left: OrientedBox, right: OrientedBox
) -> tuple[float, bool]:
    """Compute clearance and intersection from one SAT evaluation.

    Returns:
        Conservative signed clearance and whether the boxes intersect.
    """
    separations = obb_axis_separations(left, right)
    return float(np.max(separations)), bool(np.all(separations <= 1e-9))


def transform_box(box: OrientedBox, world_t_box: FloatArray) -> OrientedBox:
    """Copy a box to a new world pose.

    Returns:
        The same box geometry expressed at the supplied pose.
    """
    return OrientedBox(box.object_id, world_t_box, box.half_extents)


def gripper_boxes(
    world_t_tcp: FloatArray,
    opening: float,
    geometry: GripperGeometry,
) -> tuple[OrientedBox, ...]:
    """Build conservative palm and finger OBBs at one TCP pose.

    Returns:
        Two finger boxes followed by one palm box.
    """
    finger_x = geometry.finger_length * 0.5
    finger_y = geometry.finger_thickness * 0.5
    finger_z = geometry.finger_height * 0.5
    boxes: list[OrientedBox] = []
    for sign in (-1.0, 1.0):
        tcp_t_finger = np.eye(4)
        tcp_t_finger[:3, 3] = [
            finger_x - geometry.tcp_to_fingertip,
            sign * (opening * 0.5 + finger_y),
            0.0,
        ]
        boxes.append(
            OrientedBox(
                f"gripper_finger_{sign:+.0f}",
                world_t_tcp @ tcp_t_finger,
                np.array([finger_x, finger_y, finger_z]),
            )
        )
    tcp_t_palm = np.eye(4)
    tcp_t_palm[:3, 3] = [
        geometry.finger_length + geometry.palm_depth * 0.5,
        0.0,
        0.0,
    ]
    boxes.append(
        OrientedBox(
            "gripper_palm",
            world_t_tcp @ tcp_t_palm,
            np.array(
                [
                    geometry.palm_depth * 0.5,
                    geometry.palm_width * 0.5,
                    geometry.finger_height * 0.75,
                ]
            ),
        )
    )
    return tuple(boxes)


def path_clearance(
    poses: tuple[FloatArray, ...],
    opening: float,
    geometry: GripperGeometry,
    obstacles: tuple[OrientedBox, ...],
) -> float | None:
    """Return minimum gripper clearance, or None if the swept path collides."""
    minimum = np.inf
    for pose in poses:
        for gripper_box in gripper_boxes(pose, opening, geometry):
            for obstacle in obstacles:
                clearance, intersects = obb_clearance_and_intersection(
                    gripper_box, obstacle
                )
                if intersects:
                    return None
                minimum = min(minimum, clearance)
    return float(minimum)
