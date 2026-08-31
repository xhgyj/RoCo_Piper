"""Shared deterministic staging for from-zero BrickSim assembly."""

from __future__ import annotations

PICKUP_XY_BY_ARM = {0: (0.1625, 0.005), 1: (-0.1625, 0.005)}
INITIAL_YAW_DEGREES = 90.0
DUAL_PARKING_SLOTS = {
    0: tuple((x, y) for y in (0.11, 0.20) for x in (0.08, 0.17, 0.26)),
    1: tuple((x, y) for y in (0.11, 0.20) for x in (-0.08, -0.17, -0.26)),
}


def part_order(topology: dict[str, object]) -> tuple[int, ...]:
    """Return a support-complete part order using BrickSim BFS connections.

    Returns:
        Ordered non-base part identifiers.
    """
    from bricksim.topology.ordering import bfs_sort_connections

    ordered = bfs_sort_connections(topology)
    connections = ordered["connections"]
    remaining = {
        int(part["id"]) for part in topology["parts"] if int(part["id"]) != 0
    }
    incoming: dict[int, set[int]] = {part_id: set() for part_id in remaining}
    first_seen: dict[int, int] = {}
    for index, connection in enumerate(connections):
        stud_id = int(connection["stud_id"])
        hole_id = int(connection["hole_id"])
        if hole_id in remaining:
            incoming[hole_id].add(stud_id)
            first_seen.setdefault(hole_id, index)

    assembled = {0}
    result: list[int] = []
    while remaining:
        ready = [
            part_id
            for part_id in remaining
            if incoming[part_id] and incoming[part_id] <= assembled
        ]
        if not ready:
            raise ValueError(
                "topology has no support-complete next part; "
                f"assembled={sorted(assembled)} remaining={sorted(remaining)}"
            )
        target_id = min(ready, key=lambda item: (first_seen[item], item))
        result.append(target_id)
        assembled.add(target_id)
        remaining.remove(target_id)
    return tuple(result)


async def arrange_dual_staging(
    env,
    order: tuple[int, ...],
    assignments: dict[int, int],
) -> None:
    """Park loose parts on their assigned sides at one fixed initial yaw."""
    import numpy as np
    from isaacsim.core.prims import SingleXFormPrim
    from scipy.spatial.transform import Rotation

    from rocobrick.env.loose_parts import with_planar_yaw

    arm_slot_indices = {0: 0, 1: 0}
    for target_id in order:
        arm_index = assignments[target_id]
        slots = DUAL_PARKING_SLOTS[arm_index]
        slot_index = arm_slot_indices[arm_index]
        if slot_index >= len(slots):
            raise ValueError(f"arm {arm_index} parking is full")
        x_position, y_position = slots[slot_index]
        arm_slot_indices[arm_index] += 1
        path = env.to_place_placed[target_id]
        desired = with_planar_yaw(
            env.get_prim_world_T(path), INITIAL_YAW_DEGREES
        )
        desired[0, 3] = x_position
        desired[1, 3] = y_position
        quaternion = Rotation.from_matrix(desired[:3, :3]).as_quat()
        SingleXFormPrim(
            prim_path=path, name=f"from_zero_parking_{target_id}"
        ).set_world_pose(
            position=desired[:3, 3],
            orientation=np.array(
                [quaternion[3], quaternion[0], quaternion[1], quaternion[2]],
                dtype=np.float64,
            ),
        )
    for _ in range(30):
        await env.step()


async def stage_target(env, target_id: int, arm_index: int) -> None:
    """Move one target into its assigned arm's fixed pickup slot."""
    import numpy as np
    from isaacsim.core.prims import SingleXFormPrim
    from scipy.spatial.transform import Rotation

    from rocobrick.env.loose_parts import with_planar_yaw

    try:
        pickup_xy = PICKUP_XY_BY_ARM[arm_index]
    except KeyError as error:
        raise ValueError(f"unsupported staging arm {arm_index}") from error
    path = env.to_place_placed[target_id]
    desired = with_planar_yaw(
        env.get_prim_world_T(path), INITIAL_YAW_DEGREES
    )
    desired[0, 3], desired[1, 3] = pickup_xy
    quaternion = Rotation.from_matrix(desired[:3, :3]).as_quat()
    SingleXFormPrim(
        prim_path=path, name=f"from_zero_pickup_{target_id}"
    ).set_world_pose(
        position=desired[:3, 3],
        orientation=np.array(
            [quaternion[3], quaternion[0], quaternion[1], quaternion[2]],
            dtype=np.float64,
        ),
    )
    for _ in range(30):
        await env.step()
