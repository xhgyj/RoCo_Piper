"""Execution-layer grouping for operations that share one coordinated transport."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from rocobrick.robot_operation_plan import RobotOperation


@dataclass(frozen=True)
class CoordinatedLayerGroup:
    """One plan-preserving group dispatched through a shared transport."""

    operation_ids: tuple[str, ...]
    robot_ids: tuple[str, ...]
    connection_id: str
    upper_physical_instance_id: str
    lower_physical_instance_id: str
    pre_phase_targets: Mapping[str, Mapping[str, object]] = field(default_factory=dict)


def coordinated_layer_group(
    operations: Sequence[RobotOperation],
) -> CoordinatedLayerGroup | None:
    """Recognize a task-C HANDOVER plus SUPPORT_TOP pairing.

    Both operations are in the same scheduler layer, target the same
    connection and physical instances, and involve different executor robots.
    Any ambiguity returns ``None`` so the caller can fail closed.

    Returns:
        The strict paired group, or ``None`` when the layer is not eligible.
    """
    handover = [item for item in operations if item.skill == "HANDOVER"]
    support = [item for item in operations if item.skill == "SUPPORT_TOP"]
    if len(handover) != 1 or len(support) != 1 or len(operations) != 2:
        return None
    handover_item = handover[0]
    support_item = support[0]
    handover_target = handover_item.target
    support_target = support_item.target
    if handover_target is None or support_target is None:
        return None
    handover_values = handover_target.values
    support_values = support_target.values
    physical = handover_values.get("physical_instance_id")
    connection_id = support_values.get("connection_id")
    upper = support_values.get("upper_physical_instance_id")
    lower = support_values.get("lower_physical_instance_id")
    if not all(
        isinstance(value, str) and value
        for value in (physical, connection_id, upper, lower)
    ):
        return None
    if physical != lower:
        return None
    if support_values.get("placement_robot_id") != handover_item.executor_robot_id:
        return None
    if handover_item.executor_robot_id == support_item.executor_robot_id:
        return None
    return CoordinatedLayerGroup(
        operation_ids=(handover_item.operation_id, support_item.operation_id),
        robot_ids=(
            handover_item.executor_robot_id,
            support_item.executor_robot_id,
        ),
        connection_id=connection_id,
        upper_physical_instance_id=upper,
        lower_physical_instance_id=lower,
    )


__all__ = ["CoordinatedLayerGroup", "coordinated_layer_group"]
