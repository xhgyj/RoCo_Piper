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
    """Recognize task-C transport pairing without changing operation semantics.

    The external plan remains a TRANSIT plus SUPPORT_TOP pair. The pair is
    eligible only when both operations are in the same scheduler layer and
    identify the same connection and upper/lower physical instances. Any
    ambiguity returns ``None`` so the caller can fail closed.

    Returns:
        The strict paired group, or ``None`` when the layer is not eligible.
    """
    transit = [item for item in operations if item.skill == "TRANSIT"]
    support = [item for item in operations if item.skill == "SUPPORT_TOP"]
    if len(transit) != 1 or len(support) != 1 or len(operations) != 2:
        return None
    transit_item = transit[0]
    support_item = support[0]
    transit_target = transit_item.target
    support_target = support_item.target
    if transit_target is None or support_target is None:
        return None
    transit_values = transit_target.values
    support_values = support_target.values
    destination = transit_values.get("to")
    carrying = transit_values.get("carrying_physical_instance_id")
    if not isinstance(destination, dict) or destination.get("kind") != "connection":
        return None
    connection_id = destination.get("connection_id")
    upper = support_values.get("upper_physical_instance_id")
    lower = support_values.get("lower_physical_instance_id")
    if not all(
        isinstance(value, str) and value
        for value in (connection_id, upper, lower)
    ):
        return None
    if carrying != lower or support_values.get("connection_id") != connection_id:
        return None
    if support_values.get("placement_robot_id") != transit_item.executor_robot_id:
        return None
    if transit_item.executor_robot_id == support_item.executor_robot_id:
        return None
    return CoordinatedLayerGroup(
        operation_ids=(transit_item.operation_id, support_item.operation_id),
        robot_ids=(transit_item.executor_robot_id, support_item.executor_robot_id),
        connection_id=connection_id,
        upper_physical_instance_id=upper,
        lower_physical_instance_id=lower,
    )


__all__ = ["CoordinatedLayerGroup", "coordinated_layer_group"]
