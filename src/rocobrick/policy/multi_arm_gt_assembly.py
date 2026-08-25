"""Multi-arm adapters around the unchanged single-arm GT assembly expert.

The single-arm expert remains the canonical strategy.  This module contains
only the extra bookkeeping needed when several arms take turns: selecting a
target from an evolving assembled set and restricting pickup planning to the
arm assigned by the sequence policy.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Mapping

from rocobrick.policy import gt_assembly as single_arm


def resolve_single_step_task(
    env,
    target_id: int,
    assembled_parts: Mapping[int, str],
) -> single_arm.GTAssemblyTask:
    """Resolve one target using already assembled parts as references.

    Returns:
        Target-centric task consumed by the unchanged single-arm expert.
    """
    targets = sorted(int(part_id) for part_id in env.to_place_placed)
    if target_id not in targets:
        raise ValueError(
            f"target {target_id} is not an unplaced part; available={targets}"
        )
    parts = {int(part["id"]): part for part in env.topology["parts"]}
    candidates = [
        connection
        for connection in env.topology["connections"]
        if int(connection["hole_id"]) == target_id
        and int(connection["stud_id"]) in assembled_parts
    ]
    if not candidates:
        raise ValueError(f"target {target_id} has no assembled brick reference")

    def converted(connection):
        reference_id = int(connection["stud_id"])
        return single_arm.GTConnection(
            reference_id=reference_id,
            target_id=target_id,
            reference_path=assembled_parts[reference_id],
            target_path=env.to_place_placed[target_id],
            stud_iface=int(connection["stud_iface"]),
            hole_iface=int(connection["hole_iface"]),
            offset=(int(connection["offset"][0]), int(connection["offset"][1])),
            yaw=int(connection["yaw"]),
            overlap_studs=single_arm._connection_overlap(connection, parts),
        )

    connections = sorted(
        (converted(connection) for connection in candidates),
        key=lambda connection: (-connection.overlap_studs, connection.reference_id),
    )
    dimensions = {
        key: int(value)
        for key, value in parts[target_id]["payload"].items()
        if key in {"L", "W", "H"}
    }
    return single_arm.GTAssemblyTask(
        target_id=target_id,
        target_path=env.to_place_placed[target_id],
        dimensions=dimensions,
        primary=connections[0],
        additional=tuple(connections[1:]),
    )


@contextmanager
def _patched_pickup(task, arm_index: int) -> Iterator[None]:
    """Temporarily constrain the unchanged picker to one arm and one task."""
    original_resolver = single_arm.resolve_single_step_task
    original_selector = single_arm._select_pick_plan

    def resolver(_env):
        return task

    def selector(env, selected_task, pick, goal, height):
        return original_selector(
            env, selected_task, pick, goal, height, arm_indices=(arm_index,)
        )

    single_arm.resolve_single_step_task = resolver
    single_arm._select_pick_plan = selector
    try:
        yield
    finally:
        single_arm.resolve_single_step_task = original_resolver
        single_arm._select_pick_plan = original_selector


async def prepare_safe_start(env, safe_height, task, arm_index):
    """Prepare a target with the single-arm picker on the assigned arm.

    Returns:
        Verified safe-start state returned by the single-arm expert.
    """
    with _patched_pickup(task, arm_index):
        return await single_arm.prepare_safe_start(env, safe_height=safe_height)


def pickup_reachability_error(env, target_path, dimensions, arm_index):
    """Return why the assigned arm cannot pregrasp, grasp, and lift a brick."""
    if arm_index < 0 or arm_index >= len(env.robot_pins):
        return f"invalid arm index {arm_index}"
    brick = env.get_prim_world_T(target_path)
    robot_pin = env.robot_pins[arm_index]
    for grasp_axis in (0, 1):
        for sign in (1.0, -1.0):
            pick_tcp = single_arm._grasp_tcp_for_brick(
                brick, grasp_axis, sign, single_arm.PICK_TCP_HEIGHT
            )
            pregrasp_tcp = pick_tcp.copy()
            pregrasp_tcp[:3, 3] += brick[:3, 2] * single_arm.PICK_APPROACH_HEIGHT
            pregrasp = single_arm._try_verified_ik(
                robot_pin, pregrasp_tcp, robot_pin.home_q
            )
            if pregrasp is None:
                continue
            grasp = single_arm._try_verified_ik(robot_pin, pick_tcp, pregrasp)
            if grasp is None:
                continue
            lift_tcp = pick_tcp.copy()
            lift_tcp[:3, 3] += brick[:3, 2] * single_arm.PICK_APPROACH_HEIGHT
            if single_arm._try_verified_ik(robot_pin, lift_tcp, grasp) is not None:
                return None
    size = {key: int(dimensions[key]) for key in ("L", "W", "H")}
    return f"no verified pregrasp/grasp/lift IK for dimensions={size}"


def arm_home_error(env, arm_index: int, tolerance: float = 0.04):
    """Return why an arm is not at its configured home position."""
    if tolerance <= 0.0:
        raise ValueError("home tolerance must be positive")
    actual = single_arm._arm_configuration(env, arm_index)
    home = env.robot_pins[arm_index].home_q
    joint_error = float(max(abs(actual - home)))
    if joint_error > tolerance:
        return f"arm {arm_index} home joint error={joint_error:.4f} rad/m"
    return None
