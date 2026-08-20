#!/usr/bin/env python3
"""Collect proprioceptive Mate-down demonstrations from a privileged teacher."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from rocobrick.env.Env import Env
from rocobrick.policy.mate_down import (
    FRAME_STATE_DIM,
    FailureType,
    ScriptedMateDownExpert,
)
from rocobrick.policy.mate_down_runtime import (
    MateDownInitializationError,
    NoisyTargetProvider,
    SimulatorMateDownRuntime,
    load_runtime_config,
    prepare_mate_down,
    release_and_retreat,
    rotate_wrench_to_frame,
    transport_grasp_is_valid,
    true_skill_pose,
)


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-attempts", type=int, default=600)
    parser.add_argument("--repo-id", default="local/roco-mate-down")
    parser.add_argument("--output", default="../datasets/roco_mate_down")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def _features(state_dim):
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [f"proprio_history_{index}" for index in range(state_dim)],
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["dx", "dy", "dz", "drx", "dry", "drz"],
        },
        "phase": {"dtype": "int64", "shape": (1,), "names": ["phase"]},
        "contact": {"dtype": "int64", "shape": (1,), "names": ["contact"]},
        "success": {"dtype": "int64", "shape": (1,), "names": ["success"]},
        "failure_type": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["failure_type"],
        },
        "active_arm": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["active_arm"],
        },
        "timestamp_ns": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["timestamp_ns"],
        },
    }


def _new_dataset(repo_id, root, fps, state_dim):
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=root,
        robot_type="dual_piper_l_mate_down",
        features=_features(state_dim),
        use_videos=False,
    )


def _commit(dataset, frames, success, failure):
    if not frames:
        return False
    for frame in frames:
        dataset.add_frame(
            {
                **frame,
                "success": np.array([int(success)], dtype=np.int64),
                "failure_type": np.array([int(failure)], dtype=np.int64),
                "task": "mate a grasped brick downward",
            }
        )
    dataset.save_episode()
    return True


async def main():
    """Collect requested successful and failed Mate-down episodes."""
    args = _arguments()
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    output = (script_dir / args.output).resolve()
    success_root = output / "successes"
    failure_root = output / "failures"
    if success_root.exists() or failure_root.exists():
        raise FileExistsError(f"dataset output already exists: {output}")
    runtime_config, noise_config, _ = load_runtime_config(
        script_dir / "../config/mate_down_config.json"
    )
    state_dim = runtime_config.history_steps * FRAME_STATE_DIM
    successes = _new_dataset(
        f"{args.repo_id}-successes",
        success_root,
        runtime_config.control_hz,
        state_dim,
    )
    failures = _new_dataset(
        f"{args.repo_id}-failures",
        failure_root,
        runtime_config.control_hz,
        state_dim,
    )
    env = Env(
        root_dir=str(script_dir),
        user_config_path="../config/user_config.json",
        system_config_path="../config/system_config.json",
    )
    saved_successes = 0
    attempts = 0
    try:
        while saved_successes < args.episodes and attempts < args.max_attempts:
            attempts += 1
            await env.reset()
            await env.play()
            await env.get_robot_ready()
            episode_seed = args.seed + attempts
            try:
                frames, connected, failure = await _collect_attempt(
                    env, runtime_config, noise_config, episode_seed
                )
            except MateDownInitializationError as exc:
                print(
                    f"[mate-down] initializer failed on attempt {attempts}: "
                    f"{exc}; resetting and retrying"
                )
                continue
            if connected:
                _commit(successes, frames, True, FailureType.NONE)
                saved_successes += 1
                print(
                    f"[mate-down] success {saved_successes}/{args.episodes} "
                    f"on attempt {attempts} ({len(frames)} frames)"
                )
            else:
                _commit(failures, frames, False, failure)
                print(
                    f"[mate-down] saved failure {failure.name} on attempt "
                    f"{attempts} ({len(frames)} frames)"
                )
    finally:
        successes.finalize()
        failures.finalize()
    if saved_successes < args.episodes:
        raise RuntimeError(
            f"collected {saved_successes}/{args.episodes} successes in "
            f"{attempts} attempts"
        )


async def _collect_attempt(env, runtime_config, noise_config, seed):
    prepared = await prepare_mate_down(env)
    runtime = SimulatorMateDownRuntime(env, prepared.arm_index, runtime_config)
    runtime.reset()
    await runtime.calibrate_wrench()
    target_provider = NoisyTargetProvider(noise_config, seed=seed)
    target_provider.reset()
    teacher = ScriptedMateDownExpert(
        stable_force_min=runtime_config.stable_force_min,
        stable_force_max=runtime_config.stable_force_max,
    )
    teacher.reset()
    frames = []
    failure = FailureType.TIMEOUT
    completion_candidate = False
    for _ in range(runtime_config.max_episode_steps):
        estimated_target = target_provider.update(prepared.true_world_t_skill)
        try:
            feedback, observation, history, safety = runtime.observe(estimated_target)
        except (RuntimeError, ValueError) as exc:
            print(f"[mate-down] invalid feedback: {exc}")
            failure = FailureType.INVALID_FEEDBACK
            break
        if safety.done and not safety.success_candidate:
            print(
                "[mate-down] supervisor stop "
                f"{safety.failure.name}: position={observation.tcp_position}, "
                f"wrench={observation.external_wrench}"
            )
            failure = safety.failure
            break
        teacher_wrench = rotate_wrench_to_frame(
            prepared.true_world_t_skill, feedback.wrench_world
        )
        teacher_pose = true_skill_pose(
            prepared.true_world_t_skill, feedback.tcp_world
        )
        teacher_output = teacher.act(
            teacher_pose,
            teacher_wrench,
            connected=prepared.assembly_expert._verify_connection(),
        )
        action = _rotate_action(
            prepared.true_world_t_skill, estimated_target, teacher_output.action
        )
        try:
            executed = runtime.apply_action(estimated_target, feedback, action)
        except RuntimeError as exc:
            print(f"[mate-down] action failed: {exc}")
            failure = FailureType.IK_FAILED
            break
        frames.append(
            {
                "observation.state": history.astype(np.float32),
                "action": executed.astype(np.float32),
                "phase": np.array([int(teacher_output.phase)], dtype=np.int64),
                "contact": np.array(
                    [
                        int(
                            abs(teacher_pose[2, 3])
                            <= teacher.contact_activation_distance
                            and abs(teacher_wrench[2])
                            >= runtime_config.contact_force
                        )
                    ],
                    dtype=np.int64,
                ),
                "active_arm": np.array([prepared.arm_index], dtype=np.int64),
                "timestamp_ns": np.array([time.monotonic_ns()], dtype=np.int64),
            }
        )
        await env.step()
        await env.step()
        if not transport_grasp_is_valid(env, prepared.assembly_expert):
            failure = FailureType.DROPPED
            print("[mate-down] supervisor stop DROPPED: grasp transform lost")
            break
        if teacher_output.done:
            completion_candidate = True
            failure = FailureType.NONE
            break
    connected = False
    if completion_candidate:
        await release_and_retreat(env, prepared)
        connected = prepared.assembly_expert._verify_connection()
    if connected:
        failure = FailureType.NONE
    elif completion_candidate:
        failure = FailureType.NOT_CONNECTED
    return frames, connected, failure


def _rotate_action(true_world_t_skill, estimated_world_t_skill, action):
    rotation = (
        np.asarray(estimated_world_t_skill)[:3, :3].T
        @ np.asarray(true_world_t_skill)[:3, :3]
    )
    result = np.asarray(action, dtype=np.float64).copy()
    result[:3] = rotation @ result[:3]
    result[3:] = rotation @ result[3:]
    return result
