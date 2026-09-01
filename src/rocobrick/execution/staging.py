"""Derive immutable loose-part pickup slots from an episode plan."""

from __future__ import annotations

from rocobrick.planning.task_planner import TaskPlan
from rocobrick.task_config.episode import StagingConfig


def initial_slot_by_target(
    plan: TaskPlan, staging: StagingConfig
) -> dict[int, tuple[float, float]]:
    """Assign every planned loose part one static initial pickup slot.

    The returned positions are consumed while the simulator is initialized.
    They are not motion commands: a part remains at its slot until the arm
    picks it there.

    Returns:
        Target part IDs mapped to their immutable initial XY positions.
    """
    slot_indices = {arm.robot_id: 0 for arm in staging.arms}
    slots_by_target: dict[int, tuple[float, float]] = {}
    for task in plan.tasks:
        arm = staging.arm(task.assigned_arm)
        slot_index = slot_indices[task.assigned_arm]
        if slot_index >= len(arm.parking_slots):
            raise ValueError(
                f"arm {task.assigned_arm} has {len(arm.parking_slots)} parking "
                f"slots but needs more for target {task.target_part_id}"
            )
        slots_by_target[task.target_part_id] = arm.parking_slots[slot_index]
        slot_indices[task.assigned_arm] += 1
    return slots_by_target
