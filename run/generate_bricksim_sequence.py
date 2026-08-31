"""Generate NaivePolicy-style tasks from a BrickSim connection ordering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bricksim.topology.ordering import bfs_sort_connections

from rocobrick.task_config.Task import TaskConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a serialized part-level plan for one Type-1 task."
    )
    parser.add_argument("task_dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def generate_sequence(task_dir: Path) -> dict[str, object]:
    """Return BrickSim's filtered BFS connections as serialized tasks."""
    task_config = TaskConfig(
        {
            "Task_Config": {
                "Task_Path": str(task_dir),
                "Task_Type": "1",
                "Base_Plate": {
                    "Dimension": [32, 32],
                    "Position": [0.0, -0.2, 0.0],
                    "Orientation": [0.7071, 0.0, 0.0, 0.7071],
                    "Color": "Light Gray",
                },
            }
        }
    )
    topology = task_config.topology
    preplaced_topology = task_config.pre_placed_topology
    if topology is None or preplaced_topology is None:
        raise RuntimeError("TaskConfig did not produce task topologies")
    ordered = bfs_sort_connections(topology)
    preplaced = {int(part["id"]) for part in preplaced_topology["parts"]}

    candidate_connections = [
        connection
        for connection in ordered["connections"]
        if not (
            connection["stud_id"] in preplaced
            and connection["hole_id"] in preplaced
        )
    ]

    tasks = [
        {
            "task_index": index,
            "connection_id": connection["id"],
            "stud_part_id": connection["stud_id"],
            "stud_iface": connection["stud_iface"],
            "hole_part_id": connection["hole_id"],
            "hole_iface": connection["hole_iface"],
            "offset": connection["offset"],
            "yaw": connection["yaw"],
        }
        for index, connection in enumerate(candidate_connections, start=1)
    ]
    return {
        "schema": "rocobrick/bricksim_connection_sequence@1",
        "preplaced_part_ids": sorted(preplaced),
        "tasks": tasks,
    }


def main() -> None:
    """Generate and write the requested sequence file."""
    args = _parse_args()
    task_dir = args.task_dir.resolve()
    output = args.output or task_dir / "assembly_sequence.json"
    plan = generate_sequence(task_dir)
    output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
