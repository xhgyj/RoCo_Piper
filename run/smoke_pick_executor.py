#!/usr/bin/env python3
"""Smoke-test one goal-aware Pick through ManipulationExecutor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rocobrick.backends.bricksim import BrickSimRobotBackend, BrickSimWorldModel
from rocobrick.env.Env import Env
from rocobrick.env.lifecycle import close_kit_app
from rocobrick.execution import ActionStatus, ManipulationExecutor
from rocobrick.policy.bricksim_grounder import BrickSimActionGrounder
from rocobrick.policy.gt_assembly import resolve_single_step_task
from rocobrick.skills import ManipulationAction, ManipulationSkillType


def _arguments() -> argparse.Namespace:
    """Parse the upper-planner robot assignment.

    Returns:
        Command-line arguments for this smoke run.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm-index",
        type=int,
        default=0,
        help="zero-based robot index selected by the upper planner",
    )
    return parser.parse_args()


async def main() -> None:
    """Initialize BrickSim, execute one new-path Pick, and verify held state."""
    args = _arguments()
    script_dir = Path(__file__).resolve().parent
    env = None
    return_code = 0
    try:
        env = Env(
            root_dir=str(script_dir),
            user_config_path="../config/user_config.json",
            system_config_path="../config/system_config.json",
        )
        await env.reset()
        await env.play()
        await env.get_robot_ready()
        if args.arm_index < 0 or args.arm_index >= len(env.robot_pins):
            raise ValueError(
                f"arm index {args.arm_index} is outside [0, {len(env.robot_pins)})"
            )

        task = resolve_single_step_task(env)
        robot = BrickSimRobotBackend(env, args.arm_index)
        executor = ManipulationExecutor(
            robots={robot.robot_id: robot},
            world=BrickSimWorldModel(env),
            grounder=BrickSimActionGrounder(env),
        )
        action = ManipulationAction(
            action_id="smoke-pick-001",
            robot_ids=(robot.robot_id,),
            skill_type=ManipulationSkillType.PICK,
            object_id=task.target_path,
            goal_id=BrickSimActionGrounder.assembly_goal_id(task.target_path),
        )
        print(
            "[smoke] upper planner assigned "
            f"action={action.action_id} robot={robot.robot_id} "
            f"object={action.object_id}",
            flush=True,
        )
        result = await executor.execute(action)
        payload = {
            "action_id": result.action_id,
            "status": result.status.value,
            "robot_ids": result.robot_ids,
            "object_id": result.object_id,
            "held_by": result.held_by,
            "steps": result.steps,
            "metrics": {
                "waypoints": result.metrics.waypoints,
                "control_iterations": result.metrics.control_iterations,
                "simulation_steps": result.metrics.simulation_steps,
            },
            "failure": None
            if result.failure is None
            else {
                "code": result.failure.code.value,
                "stage": result.failure.stage,
                "detail": result.failure.detail,
            },
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        if result.status is not ActionStatus.SUCCESS:
            stage = result.failure.stage if result.failure else "unknown"
            detail = result.failure.detail if result.failure else "no detail"
            raise RuntimeError(f"new-path Pick failed: {stage}: {detail}")
        held = executor.held_state(robot.robot_id)
        if held is None or held.held.object_id != task.target_path:
            raise RuntimeError("Pick succeeded without persistent held-object state")
        print(
            "[smoke] held state verified: "
            f"grasp_axis={held.held.grasp_axis}, "
            f"grasp_width={held.held.grasp_width:.4f} m, "
            f"drift={held.cumulative_position_drift * 1000.0:.2f} mm/"
            f"{held.cumulative_rotation_drift * 180.0 / 3.141592653589793:.2f} deg",
            flush=True,
        )
    except BaseException:
        return_code = 1
        raise
    finally:
        await close_kit_app(env, return_code)
