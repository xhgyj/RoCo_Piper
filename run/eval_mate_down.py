#!/usr/bin/env python3
"""Evaluate a proprioceptive Mate-down ACT checkpoint without policy truth."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from rocobrick.env.Env import Env
from rocobrick.policy.mate_down import FailureType
from rocobrick.policy.mate_down_runtime import (
    NoisyTargetProvider,
    SimulatorMateDownRuntime,
    load_runtime_config,
    prepare_mate_down,
    release_and_retreat,
)
from rocobrick.policy.proprio_act import load_policy


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


async def main():
    """Evaluate the learned policy across reset simulator episodes."""
    args = _arguments()
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    runtime_config, noise_config, _ = load_runtime_config(
        script_dir / "../config/mate_down_config.json"
    )
    policy = load_policy(args.checkpoint, args.device)
    env = Env(
        root_dir=str(script_dir),
        user_config_path="../config/user_config.json",
        system_config_path="../config/system_config.json",
    )
    successes = 0
    arm_attempts = {0: 0, 1: 0}
    arm_successes = {0: 0, 1: 0}
    failures = {failure.name: 0 for failure in FailureType}
    for episode in range(args.episodes):
        await env.reset()
        await env.play()
        await env.get_robot_ready()
        seed = args.seed + episode
        prepared = await prepare_mate_down(env)
        runtime = SimulatorMateDownRuntime(
            env, prepared.arm_index, runtime_config
        )
        runtime.reset()
        await runtime.calibrate_wrench()
        provider = NoisyTargetProvider(noise_config, seed=seed)
        provider.reset()
        policy.reset()
        failure = FailureType.TIMEOUT
        for _ in range(runtime_config.max_episode_steps):
            estimated_target = provider.update(prepared.true_world_t_skill)
            try:
                feedback, _, state, supervisor = runtime.observe(estimated_target)
                if supervisor.done:
                    failure = supervisor.failure
                    break
                action = policy.select_action(state)
                runtime.apply_action(estimated_target, feedback, action)
            except ValueError:
                failure = FailureType.INVALID_FEEDBACK
                break
            except RuntimeError:
                failure = FailureType.IK_FAILED
                break
            await env.step()
            await env.step()
        await release_and_retreat(env, prepared)
        connected = prepared.assembly_expert._verify_connection()
        arm_attempts[prepared.arm_index] += 1
        if connected:
            successes += 1
            arm_successes[prepared.arm_index] += 1
            failure = FailureType.NONE
        elif failure == FailureType.NONE:
            failure = FailureType.NOT_CONNECTED
        failures[failure.name] += 1
        print(
            f"episode={episode + 1}/{args.episodes} arm={prepared.arm_index} "
            f"success={connected} failure={failure.name}"
        )
    print(f"overall: {successes}/{args.episodes} = {successes / args.episodes:.1%}")
    for arm in sorted(arm_attempts):
        attempts = arm_attempts[arm]
        rate = arm_successes[arm] / attempts if attempts else 0.0
        print(f"arm {arm}: {arm_successes[arm]}/{attempts} = {rate:.1%}")
    print("failures:", failures)
    if successes / args.episodes < 0.8:
        raise RuntimeError(
            "Mate-down success rate is below the 80% acceptance threshold"
        )
