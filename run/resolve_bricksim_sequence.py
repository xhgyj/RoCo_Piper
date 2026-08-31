#!/usr/bin/env python3
"""Resolve a NaivePolicy-style connection plan to live BrickSim USD paths."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from bricksim.topology.ordering import bfs_sort_connections

from rocobrick.env.Env import Env
from rocobrick.env.lifecycle import close_kit_app

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent


def _parse_args() -> argparse.Namespace:
    """Parse task and output paths.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _runtime_plan(env) -> dict[str, object]:
    """Reproduce the historical NaivePolicy connection-plan conversion.

    Returns:
        Ordered connection tasks with live USD paths and symbolic part IDs.
    """
    sorted_topology = bfs_sort_connections(env.topology)
    preplaced_ids = set(env.pre_placed_parts)
    tasks = []
    for connection in sorted_topology["connections"]:
        stud_id = int(connection["stud_id"])
        hole_id = int(connection["hole_id"])
        if stud_id in preplaced_ids and hole_id in preplaced_ids:
            continue
        stud_path = (
            env.pre_placed_parts[stud_id]
            if stud_id in preplaced_ids
            else env.to_place_placed[stud_id]
        )
        hole_path = (
            env.pre_placed_parts[hole_id]
            if hole_id in preplaced_ids
            else env.to_place_placed[hole_id]
        )
        tasks.append(
            {
                "task_index": len(tasks) + 1,
                "connection_id": int(connection["id"]),
                "stud_part_id": stud_id,
                "stud_path": str(stud_path),
                "stud_iface": int(connection["stud_iface"]),
                "hole_part_id": hole_id,
                "hole_path": str(hole_path),
                "hole_iface": int(connection["hole_iface"]),
                "offset": [int(value) for value in connection["offset"]],
                "yaw": int(connection["yaw"]),
            }
        )
    return {
        "schema": "rocobrick/bricksim_runtime_connection_sequence@1",
        "preplaced_part_ids": sorted(preplaced_ids),
        "to_place_part_ids": sorted(int(part_id) for part_id in env.to_place_placed),
        "tasks": tasks,
    }


async def main() -> None:
    """Initialize BrickSim, resolve live paths, and write the runtime plan."""
    env = None
    return_code = 0
    try:
        args = _parse_args()
        task_dir = args.task_dir.resolve()
        output = (
            args.output.resolve()
            if args.output is not None
            else task_dir / "runtime_assembly_sequence.json"
        )
        with tempfile.TemporaryDirectory(prefix="roco-sequence-") as temporary:
            config = json.loads(
                (REPOSITORY_ROOT / "config/user_config.json").read_text(
                    encoding="utf-8"
                )
            )
            config["Task_Config"]["Task_Path"] = str(task_dir)
            config["Task_Config"]["Task_Type"] = "1"
            config_path = Path(temporary) / "user_config.json"
            config_path.write_text(
                json.dumps(config, indent=2) + "\n", encoding="utf-8"
            )
            env = Env(
                root_dir=str(SCRIPT_DIR),
                user_config_path=str(config_path),
                system_config_path="../config/system_config.json",
            )
            await env.reset()
            plan = _runtime_plan(env)
            output.write_text(
                json.dumps(plan, indent=2) + "\n", encoding="utf-8"
            )
            print(output, flush=True)
            print(json.dumps(plan, indent=2), flush=True)
    except BaseException:
        return_code = 1
        raise
    finally:
        await close_kit_app(env, return_code)


if __name__ == "__main__":
    raise RuntimeError(
        "launch with: uv run bricksim ./run/resolve_bricksim_sequence.py"
    )
