#!/usr/bin/env python3
"""Run unified goal-aware Pick -> PlaceDown tests for Task-1 directories."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path


def _arguments() -> argparse.Namespace:
    """Parse task discovery, arm assignment, and reporting options.

    Returns:
        Command-line arguments for one or more test episodes.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "task_dirs",
        nargs="*",
        type=Path,
        help=(
            "Task-1 directories containing structure_start.json and "
            "structure_goal.json"
        ),
    )
    parser.add_argument(
        "--tasks-root",
        action="append",
        default=[],
        type=Path,
        help="recursively discover Task-1 directories below this root",
    )
    parser.add_argument(
        "--arm-index",
        action="append",
        type=int,
        help="explicit upper-planner arm assignment; repeat to test multiple arms",
    )
    parser.add_argument(
        "--yaw-degrees",
        type=float,
        help="optional absolute yaw applied to the single loose target",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=10,
        help="physics steps after optional target-yaw placement (default: 10)",
    )
    parser.add_argument(
        "--final-hold-seconds",
        type=float,
        default=0.0,
        help="keep the successful final scene visible before shutdown",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON summary path",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="stop the batch after the first failed episode",
    )
    return parser.parse_args()


def _discover_tasks(explicit: list[Path], roots: list[Path]) -> tuple[Path, ...]:
    """Resolve and validate explicit and recursively discovered Task-1 paths.

    Returns:
        Unique absolute task directories in deterministic order.
    """
    candidates = [path.resolve() for path in explicit]
    for root in roots:
        resolved_root = root.resolve()
        candidates.extend(
            goal.parent
            for goal in resolved_root.rglob("structure_goal.json")
            if (goal.parent / "structure_start.json").is_file()
        )
    unique = tuple(sorted(set(candidates), key=str))
    if not unique:
        raise ValueError("provide a Task-1 directory or --tasks-root")
    for task_dir in unique:
        if not task_dir.is_dir():
            raise ValueError(f"task directory does not exist: {task_dir}")
        missing = [
            name
            for name in ("structure_start.json", "structure_goal.json")
            if not (task_dir / name).is_file()
        ]
        if missing:
            raise ValueError(f"task directory {task_dir} is missing {missing}")
    return unique


def _temporary_user_config(
    base_config: dict[str, object], task_dir: Path, output_dir: Path
) -> Path:
    """Write a Task-1 override without changing repository configuration.

    Returns:
        Temporary user-configuration path for ``Env.reset``.
    """
    config = json.loads(json.dumps(base_config))
    task_config = config.get("Task_Config")
    if not isinstance(task_config, dict):
        raise ValueError("base user config has no Task_Config object")
    task_config["Task_Path"] = str(task_dir)
    task_config["Task_Type"] = "1"
    path = output_dir / f"user_{len(tuple(output_dir.iterdir())):04d}.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def _action_payload(result) -> dict[str, object]:
    """Convert a unified manipulation result into JSON-safe diagnostics.

    Returns:
        Status, ownership, metrics, and optional structured failure.
    """
    return {
        "action_id": result.action_id,
        "status": result.status.value,
        "robot_ids": list(result.robot_ids),
        "object_id": result.object_id,
        "held_by": result.held_by,
        "metrics": {
            "waypoints": result.metrics.waypoints,
            "control_iterations": result.metrics.control_iterations,
            "simulation_steps": result.metrics.simulation_steps,
        },
        "failure": (
            None
            if result.failure is None
            else {
                "code": result.failure.code.value,
                "stage": result.failure.stage,
                "detail": result.failure.detail,
            }
        ),
    }


async def _run_episode(
    env,
    task_dir: Path,
    config_path: Path,
    arm_index: int,
    yaw_degrees: float | None,
    settle_steps: int,
    final_hold_seconds: float,
):
    """Run one explicitly assigned Pick -> PlaceDown episode.

    Returns:
        JSON-safe episode result with both public action results.
    """
    from rocobrick.backends.bricksim import BrickSimRobotBackend, BrickSimWorldModel
    from rocobrick.execution import ActionStatus, ManipulationExecutor
    from rocobrick.policy.bricksim_grounder import BrickSimActionGrounder
    from rocobrick.policy.gt_assembly import resolve_single_step_task
    from rocobrick.skills import ManipulationAction, ManipulationSkillType

    started = time.perf_counter()
    env.user_config_path = str(config_path)
    await env.reset()
    await env.play()
    await env.get_robot_ready()
    if arm_index < 0 or arm_index >= len(env.robot_pins):
        raise ValueError(f"arm index {arm_index} is outside [0, {len(env.robot_pins)})")
    if yaw_degrees is not None:
        env.set_loose_target_yaw(yaw_degrees)
        await BrickSimWorldModel(env).advance(settle_steps)

    task = resolve_single_step_task(env)
    robot = BrickSimRobotBackend(env, arm_index)
    executor = ManipulationExecutor(
        robots={robot.robot_id: robot},
        world=BrickSimWorldModel(env),
        grounder=BrickSimActionGrounder(env),
    )
    goal_id = BrickSimActionGrounder.assembly_goal_id(task.target_path)
    prefix = f"{task_dir.name}-{robot.robot_id}"
    pick = await executor.execute(
        ManipulationAction(
            action_id=f"{prefix}-pick",
            robot_ids=(robot.robot_id,),
            skill_type=ManipulationSkillType.PICK,
            object_id=task.target_path,
            goal_id=goal_id,
        )
    )
    pick_payload = _action_payload(pick)
    print(json.dumps({"phase": "pick", **pick_payload}, indent=2), flush=True)
    place_payload = None
    success = False
    if pick.status is ActionStatus.SUCCESS:
        place = await executor.execute(
            ManipulationAction(
                action_id=f"{prefix}-place-down",
                robot_ids=(robot.robot_id,),
                skill_type=ManipulationSkillType.PLACE_DOWN,
                object_id=task.target_path,
                goal_id=goal_id,
            )
        )
        place_payload = _action_payload(place)
        print(
            json.dumps({"phase": "place_down", **place_payload}, indent=2),
            flush=True,
        )
        success = place.status is ActionStatus.SUCCESS
    if success and final_hold_seconds > 0.0:
        print(
            f"[episode] holding final scene for {final_hold_seconds:.1f}s",
            flush=True,
        )
        for _ in range(round(final_hold_seconds * 60)):
            await env.step()
    return {
        "task_dir": str(task_dir),
        "arm_index": arm_index,
        "robot_id": robot.robot_id,
        "target_id": task.target_path,
        "success": success,
        "elapsed_seconds": time.perf_counter() - started,
        "pick": pick_payload,
        "place_down": place_payload,
    }


async def main() -> None:
    """Run every requested task/arm pair and emit a batch summary."""
    args = _arguments()
    task_dirs = _discover_tasks(args.task_dirs, args.tasks_root)
    arm_indices = tuple(args.arm_index or [0])
    if args.settle_steps < 0:
        raise ValueError("settle-steps cannot be negative")
    if args.final_hold_seconds < 0.0:
        raise ValueError("final-hold-seconds cannot be negative")

    script_dir = Path(__file__).resolve().parent
    repository = script_dir.parent
    base_config = json.loads(
        (repository / "config/user_config.json").read_text(encoding="utf-8")
    )
    env = None
    return_code = 0
    records: list[dict[str, object]] = []
    try:
        from rocobrick.env.Env import Env

        with tempfile.TemporaryDirectory(prefix="roco-pick-assemble-") as temporary:
            output_dir = Path(temporary)
            configs = {
                task_dir: _temporary_user_config(base_config, task_dir, output_dir)
                for task_dir in task_dirs
            }
            env = Env(
                root_dir=str(script_dir),
                user_config_path=str(configs[task_dirs[0]]),
                system_config_path="../config/system_config.json",
            )
            stop = False
            for task_dir in task_dirs:
                for arm_index in arm_indices:
                    print(
                        f"[episode] task={task_dir} arm_index={arm_index}",
                        flush=True,
                    )
                    try:
                        record = await _run_episode(
                            env,
                            task_dir,
                            configs[task_dir],
                            arm_index,
                            args.yaw_degrees,
                            args.settle_steps,
                            args.final_hold_seconds,
                        )
                    except Exception as error:  # noqa: BLE001 -- batch diagnostics.
                        record = {
                            "task_dir": str(task_dir),
                            "arm_index": arm_index,
                            "success": False,
                            "setup_failure": (f"{type(error).__name__}: {error}"),
                        }
                    records.append(record)
                    if not bool(record["success"]):
                        return_code = 1
                        if args.stop_on_failure:
                            stop = True
                            break
                if stop:
                    break
    except BaseException:
        return_code = 1
        raise
    finally:
        summary = {
            "success": sum(bool(record["success"]) for record in records),
            "total": len(records),
            "records": records,
        }
        rendered = json.dumps(summary, ensure_ascii=False, indent=2)
        print(rendered, flush=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        from rocobrick.env.lifecycle import close_kit_app

        await close_kit_app(env, return_code)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
