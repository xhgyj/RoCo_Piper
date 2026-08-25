#!/usr/bin/env python3
"""Evaluation entry point for the complete GT assembly workflow."""

from pathlib import Path

from rocobrick.env.Env import Env
from rocobrick.env.lifecycle import close_kit_app
from rocobrick.policy.gt_assembly import (
    load_expert_config,
    prepare_safe_start,
    release_and_return_home,
    run_gt_assembly_expert,
)


async def main():
    """Evaluate one configured local assembly episode."""
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
        prepared = await prepare_safe_start(env)
        config = load_expert_config(
            script_dir / "../config/gt_assembly_expert.json"
        )
        result = await run_gt_assembly_expert(env, prepared, config)
        print("Policy execution completed:", result)
        if not result.success:
            raise RuntimeError(result.failure_reason)
        await release_and_return_home(env, prepared)
    except BaseException:
        return_code = 1
        raise
    finally:
        await close_kit_app(env, return_code)
