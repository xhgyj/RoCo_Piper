#!/usr/bin/env python3
"""Run physical pickup, the local GT assembly expert, and cleanup."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from rocobrick.env.Env import Env
from rocobrick.env.lifecycle import close_kit_app
from rocobrick.policy.gt_assembly import (
    load_expert_config,
    prepare_safe_start,
    release_and_return_home,
    run_gt_assembly_expert,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    """Parse demo options.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=Path("tasks/type1/basic/1"),
    )
    parser.add_argument("--safe-height-mm", type=float, default=60.0)
    parser.add_argument("--final-hold-seconds", type=float, default=10.0)
    return parser.parse_args()


async def main() -> None:
    """Initialize one safe start and execute only the local assembly expert."""
    env = None
    return_code = 0
    try:
        args = parse_args()
        task_dir = args.task_dir.resolve()
        for name in ("structure_start.json", "structure_goal.json"):
            if not (task_dir / name).is_file():
                raise FileNotFoundError(task_dir / name)
        if args.safe_height_mm <= 0 or args.final_hold_seconds < 0:
            raise ValueError(
                "safe height must be positive and hold time non-negative"
            )
        with tempfile.TemporaryDirectory(
            prefix="roco-gt-assembly-"
        ) as temporary:
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
            runtime_config = load_expert_config(
                REPOSITORY_ROOT / "config/gt_assembly_expert.json"
            )
            env = Env(
                root_dir=str(SCRIPT_DIR),
                user_config_path=str(config_path),
                system_config_path="../config/system_config.json",
            )
            await env.reset()
            await env.play()
            await env.get_robot_ready()
            prepared = await prepare_safe_start(
                env, safe_height=args.safe_height_mm / 1000.0
            )
            print(
                "[assembly] trajectory starts at verified safe pose",
                flush=True,
            )
            result = await run_gt_assembly_expert(env, prepared, runtime_config)
            print(f"[assembly] result={result}", flush=True)
            if not result.success:
                raise RuntimeError(result.failure_reason)
            await release_and_return_home(env, prepared)
            for _ in range(round(args.final_hold_seconds * 60)):
                await env.step()
    except BaseException:
        return_code = 1
        raise
    finally:
        await close_kit_app(env, return_code)


if __name__ == "__main__":
    raise RuntimeError("launch with: uv run bricksim ./run/demo_gt_assembly.py")
