#!/usr/bin/env python3
"""Run one planner-driven BrickSim episode through the unified pipeline."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from rocobrick.env.Env import Env
from rocobrick.env.lifecycle import close_kit_app
from rocobrick.execution.episode import ExecutionPlanner, MultiArmScheduler
from rocobrick.execution.staging import initial_slot_by_target
from rocobrick.planning.task_planner import (
    PlanningProblem,
    TaskPlan,
    TopologyTaskPlanner,
)
from rocobrick.task_config.episode import EpisodeConfig
from rocobrick.task_config.Task import TaskConfig
from rocobrick.utils import deep_merge


def _arguments() -> argparse.Namespace:
    """Parse the single episode entry point.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--final-hold-seconds", type=float, default=0.0)
    return parser.parse_args()


def _runtime_user_config(
    episode: EpisodeConfig, temporary_path: Path
) -> tuple[Path, TaskPlan]:
    """Materialize scene config and its immutable initial loose-part slots.

    Returns:
        Generated user-config path and the upper planner's task plan.
    """
    task_path = temporary_path / "task"
    task_path.mkdir()
    (task_path / "structure_start.json").write_text(
        episode.initial_structure_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (task_path / "structure_goal.json").write_text(
        episode.goal_structure_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    config = json.loads(episode.user_config_path.read_text(encoding="utf-8"))
    config["Task_Config"]["Task_Path"] = str(task_path)
    config["Task_Config"]["Task_Type"] = "1"

    overrides = {item.robot_id: item for item in episode.robot_overrides}
    configured_names = set()
    for index, robot in enumerate(config["Robot_Config"]["Robots"]):
        robot_id = str(robot.get("Name", f"robot_{index}"))
        configured_names.add(robot_id)
        if robot_id not in overrides:
            continue
        override = overrides[robot_id]
        robot["Robot_Base_Frame"]["Position"] = list(override.position)
        robot["Robot_Base_Frame"]["Orientation"] = list(override.orientation)
    missing = set(overrides) - configured_names
    if missing:
        raise ValueError(
            "episode robot overrides are not configured: "
            f"{sorted(missing)}"
        )

    storage = config["Env_Config"]["Storage_Config"]
    storage["Size"] = list(episode.staging.workspace_size)
    storage["Position"] = list(episode.staging.workspace_position)
    task_config = TaskConfig(
        deep_merge(
            config,
            json.loads(episode.system_config_path.read_text(encoding="utf-8")),
        )
    )
    task_plan = TopologyTaskPlanner().plan(
        PlanningProblem(
            episode.episode_id,
            task_config.topology,
            tuple(
                int(part["id"])
                for part in task_config.pre_placed_topology["parts"]
            ),
            episode.available_arm_ids,
        )
    )
    slot_by_target = initial_slot_by_target(task_plan, episode.staging)
    config["Env_Config"]["Initial_Loose_Part_Slots"] = {
        str(target_id): {
            "xy": list(slot_by_target[target_id]),
            "yaw_degrees": episode.staging.initial_yaw_degrees,
        }
        for target_id in sorted(slot_by_target)
    }
    path = temporary_path / "user_config.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path, task_plan


def _result_payload(result) -> dict[str, object]:
    """Convert an episode result into one stable JSON report.

    Returns:
        JSON-serializable episode result.
    """

    def action_payload(action):
        if action is None:
            return None
        return {
            "action_id": action.action_id,
            "status": action.status.value,
            "held_by": action.held_by,
            "failure": (
                None
                if action.failure is None
                else {
                    "code": action.failure.code.value,
                    "stage": action.failure.stage,
                    "detail": action.failure.detail,
                }
            ),
        }

    return {
        "schema": "rocobrick/episode_result@1",
        "episode_id": result.episode_id,
        "success": result.success,
        "completed_task_ids": list(result.completed_task_ids),
        "failure_reason": result.failure_reason,
        "records": [
            {
                "task_id": record.task_id,
                "target_part_id": record.target_part_id,
                "assigned_arm": record.assigned_arm,
                "success": record.success,
                "pick": action_payload(record.pick),
                "place": action_payload(record.place),
                "failure_reason": record.failure_reason,
            }
            for record in result.records
        ],
    }


async def main() -> None:
    """Plan, execute, verify, and close one configured episode."""
    args = _arguments()
    if args.final_hold_seconds < 0.0:
        raise ValueError("final-hold-seconds cannot be negative")
    episode = EpisodeConfig.load(args.episode)
    env = None
    return_code = 1
    try:
        with tempfile.TemporaryDirectory(prefix=f"roco-{episode.episode_id}-") as temp:
            temporary_path = Path(temp)
            user_config, task_plan = _runtime_user_config(episode, temporary_path)
            env = Env(
                root_dir=str(Path(__file__).resolve().parent),
                user_config_path=str(user_config),
                system_config_path=str(episode.system_config_path),
            )
            await env.reset()
            await env.play()
            await env.get_robot_ready()

            execution_plan = ExecutionPlanner().compile(task_plan, episode)
            assignments = {
                task.task_id: task.assigned_arm for task in task_plan.tasks
            }
            print(
                f"[episode] id={episode.episode_id} "
                f"allow_prefetch={episode.allow_prefetch} "
                f"assignments={assignments}",
                flush=True,
            )
            result = await MultiArmScheduler(env, episode).run(execution_plan)
            payload = _result_payload(result)
            print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
            if args.output is not None:
                output = args.output.resolve()
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            if not result.success:
                raise RuntimeError(result.failure_reason or "episode failed")
            for _ in range(round(args.final_hold_seconds * 60)):
                await env.step()
            return_code = 0
    finally:
        await close_kit_app(env, return_code)
