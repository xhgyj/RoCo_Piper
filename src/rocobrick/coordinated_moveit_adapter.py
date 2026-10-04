"""Transport-free admission for a combined MoveIt joint trajectory."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rocobrick.coordinated_moveit import (
    CoordinatedMoveItError,
    CoordinatedOperation,
    CoordinatedRobotTrajectory,
    CoordinatedTrajectoryPoint,
    validate_joint_binding,
)


def coordinated_operation_from_joint_trajectory(
    *,
    coordination_id: str,
    scene_digest: str,
    participants: Sequence[str],
    joint_names: Sequence[str],
    points: Sequence[Mapping[str, object]],
    joint_binding: Mapping[str, Sequence[str]],
    start_at_sim_time_s: float,
    synchronization_tolerance_s: float = 0.02,
) -> CoordinatedOperation:
    """Split one admitted dual-arm MoveIt trajectory by robot identity.

    Args:
        coordination_id: Exact external coordination identity.
        scene_digest: Planning-scene digest used for the plan request.
        participants: Exact ordered robot participant set.
        joint_names: Joint names returned by the combined MoveIt plan.
        points: Mapping records with ``time_from_start_s`` and ``positions``.
        joint_binding: ``robot_id -> exact expected joint names`` mapping.
        start_at_sim_time_s: Shared simulation-time start barrier.
        synchronization_tolerance_s: Maximum participant start skew.

    Returns:
        Strict per-robot trajectories wrapped in one coordinated operation.

    Raises:
        CoordinatedMoveItError: If the returned trajectory is incomplete,
            foreign, nonmonotonic, or cannot be partitioned exactly.
    """
    names = tuple(joint_names)
    if not names or len(names) != len(set(names)):
        raise CoordinatedMoveItError("combined trajectory joint names must be unique")
    if tuple(participants) != tuple(joint_binding):
        raise CoordinatedMoveItError("joint binding keys must match participants order")
    index_by_name = {name: index for index, name in enumerate(names)}
    trajectories: list[CoordinatedRobotTrajectory] = []
    for robot_id in participants:
        expected = tuple(joint_binding[robot_id])
        validate_names = [name for name in expected if name in index_by_name]
        if tuple(validate_names) != tuple(expected):
            raise CoordinatedMoveItError(
                f"combined trajectory is missing joints for {robot_id}"
            )
        robot_points: list[CoordinatedTrajectoryPoint] = []
        for point_index, raw in enumerate(points):
            if not isinstance(raw, Mapping):
                raise CoordinatedMoveItError(f"trajectory point {point_index} is not an object")
            time_value = raw.get("time_from_start_s")
            positions = raw.get("positions")
            if not isinstance(positions, Sequence) or isinstance(positions, (str, bytes)):
                raise CoordinatedMoveItError(f"trajectory point {point_index}.positions is invalid")
            if len(positions) != len(names):
                raise CoordinatedMoveItError(f"trajectory point {point_index}.positions length mismatch")
            robot_points.append(
                CoordinatedTrajectoryPoint(
                    time_value,
                    tuple(positions[index_by_name[name]] for name in expected),
                )
            )
        trajectories.append(CoordinatedRobotTrajectory(robot_id, expected, tuple(robot_points)))
    return CoordinatedOperation(
        coordination_id=coordination_id,
        scene_digest=scene_digest,
        participants=tuple(participants),
        trajectories=tuple(trajectories),
        start_at_sim_time_s=start_at_sim_time_s,
        synchronization_tolerance_s=synchronization_tolerance_s,
    )
