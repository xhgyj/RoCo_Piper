#!/usr/bin/env python3
"""Visualize the Mate-down expert pipeline without saving a dataset."""

from __future__ import annotations

import os
from pathlib import Path

from rocobrick.env.Env import Env
from rocobrick.policy.mate_down import ScriptedMateDownExpert
from rocobrick.policy.mate_down_runtime import (
    SimulatorMateDownRuntime,
    load_runtime_config,
    prepare_mate_down,
    release_and_retreat,
    rotate_wrench_to_frame,
    transport_grasp_is_valid,
    true_skill_pose,
)


async def main() -> None:
    """Run one visible Pick-to-Assembly episode without recording data."""
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    runtime_config, _, _ = load_runtime_config(
        script_dir / "../config/mate_down_config.json"
    )
    env = Env(
        root_dir=str(script_dir),
        user_config_path="../config/user_config.json",
        system_config_path="../config/system_config.json",
    )
    await env.reset()
    await env.play()
    await env.get_robot_ready()

    prepared = await prepare_mate_down(env)
    runtime = SimulatorMateDownRuntime(
        env, prepared.arm_index, runtime_config
    )
    runtime.reset()
    print("[mate-down-demo] calibrating wrench at safe preplace")
    await runtime.calibrate_wrench()
    print("[mate-down-demo] Assembly policy starts now")
    teacher = ScriptedMateDownExpert(
        stable_force_min=runtime_config.stable_force_min,
        stable_force_max=runtime_config.stable_force_max,
    )

    for _ in range(runtime_config.max_episode_steps):
        feedback, observation, _, safety = runtime.observe(
            prepared.true_world_t_skill
        )
        if safety.done and not safety.success_candidate:
            raise RuntimeError(
                f"supervisor stopped demo: {safety.failure.name}; "
                f"position={observation.tcp_position}, "
                f"wrench={observation.external_wrench}"
            )
        teacher_pose = true_skill_pose(
            prepared.true_world_t_skill, feedback.tcp_world
        )
        teacher_wrench = rotate_wrench_to_frame(
            prepared.true_world_t_skill, feedback.wrench_world
        )
        output = teacher.act(
            teacher_pose,
            teacher_wrench,
            connected=prepared.assembly_expert._verify_connection(),
        )
        runtime.apply_action(
            prepared.true_world_t_skill, feedback, output.action
        )
        await env.step()
        await env.step()
        if not transport_grasp_is_valid(env, prepared.assembly_expert):
            raise RuntimeError("brick slipped during Mate-down demo")
        if output.done:
            await release_and_retreat(env, prepared)
            if not prepared.assembly_expert._verify_connection():
                raise RuntimeError("BrickSim did not verify the connection")
            print("[mate-down-demo] success; no dataset was written")
            return
    raise RuntimeError("Mate-down demo timed out")
