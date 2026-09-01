"""Resolve planner-selected topology tasks into BrickSim assembly goals."""

from __future__ import annotations

from collections.abc import Mapping

from rocobrick.policy import gt_assembly


def resolve_planned_task(
    env,
    target_id: int,
    assembled_parts: Mapping[int, str],
    connection_ids: tuple[int, ...],
) -> gt_assembly.GTAssemblyTask:
    """Resolve exactly the connections selected by the upper planner.

    Returns:
        Target-centric task consumed by BrickSim grounding.
    """
    if target_id not in env.to_place_placed:
        raise ValueError(f"target {target_id} is not an unplaced part")
    expected = set(connection_ids)
    if not expected:
        raise ValueError(f"target {target_id} has no requested connections")
    parts = {int(part["id"]): part for part in env.topology["parts"]}
    candidates = [
        connection
        for connection in env.topology["connections"]
        if int(connection["hole_id"]) == target_id
        and int(connection["id"]) in expected
        and int(connection["stud_id"]) in assembled_parts
    ]
    actual = {int(connection["id"]) for connection in candidates}
    if actual != expected:
        raise ValueError(
            f"target {target_id} connections are not ready; "
            f"missing={sorted(expected - actual)}"
        )

    def converted(connection):
        reference_id = int(connection["stud_id"])
        return gt_assembly.GTConnection(
            reference_id=reference_id,
            target_id=target_id,
            reference_path=assembled_parts[reference_id],
            target_path=env.to_place_placed[target_id],
            stud_iface=int(connection["stud_iface"]),
            hole_iface=int(connection["hole_iface"]),
            offset=(int(connection["offset"][0]), int(connection["offset"][1])),
            yaw=int(connection["yaw"]),
            overlap_studs=gt_assembly._connection_overlap(connection, parts),
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
    return gt_assembly.GTAssemblyTask(
        target_id=target_id,
        target_path=env.to_place_placed[target_id],
        dimensions=dimensions,
        primary=connections[0],
        additional=tuple(connections[1:]),
    )
