#!/usr/bin/env python3
"""Collect successful scripted-expert episodes in LeRobot v3 format."""

import argparse
import os
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from rocobrick.env.Env import Env
from rocobrick.policy.Policy import Policy


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=300)
    parser.add_argument("--repo-id", default="local/roco-piper-act")
    parser.add_argument("--output", default="../datasets/roco_piper_act")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=3600)
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def _rgb_frame(value, expected_size):
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise RuntimeError(f"invalid RGB shape: {image.shape}")
    image = image[:, :, :3]
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and image.max(initial=0) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    expected_width, expected_height = expected_size
    if image.shape != (expected_height, expected_width, 3):
        raise RuntimeError(
            f"RGB shape {image.shape} does not match "
            f"{(expected_height, expected_width, 3)}"
        )
    if int(image.max()) - int(image.min()) < 3:
        raise RuntimeError("camera frame has no useful dynamic range")
    return np.ascontiguousarray(image)


def _features(env, use_videos):
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(env.global_joint_order),),
            "names": env.global_joint_order,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(env.global_joint_order),),
            "names": env.global_joint_order,
        },
    }
    image_dtype = "video" if use_videos else "image"
    for arm_cfg in env.robot_configs:
        arm_name = arm_cfg.get("Name")
        camera_cfg = arm_cfg["Camera_Config"]["Wrist_Camera"]
        width, height = camera_cfg["Resolution"]
        features[f"observation.images.{arm_name}_wrist"] = {
            "dtype": image_dtype,
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def _camera_frames(env, obs):
    frames = {}
    for arm_cfg in env.robot_configs:
        arm_name = arm_cfg.get("Name")
        camera_cfg = arm_cfg["Camera_Config"]["Wrist_Camera"]
        source_key = f"{arm_name}_Wrist_Camera_rgb"
        if source_key not in obs["images"]:
            raise RuntimeError(
                f"missing {source_key}; camera health={env.camera_health()}"
            )
        frames[f"observation.images.{arm_name}_wrist"] = _rgb_frame(
            obs["images"][source_key], camera_cfg["Resolution"]
        )
    return frames


async def main():
    args = _arguments()
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    output = (script_dir / args.output).resolve()
    if output.exists():
        raise FileExistsError(
            f"dataset output already exists; choose a new --output path: {output}"
        )

    env = Env(
        root_dir=str(script_dir),
        user_config_path="../config/user_config.json",
        system_config_path="../config/system_config.json",
    )
    await env.reset()
    await env.play()
    await env.get_robot_ready()
    use_videos = not args.no_video
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=output,
        robot_type="dual_piper_l",
        features=_features(env, use_videos),
        use_videos=use_videos,
    )

    successes = 0
    attempts = 0
    try:
        while successes < args.episodes and attempts < args.max_attempts:
            attempts += 1
            if attempts > 1:
                await env.reset()
                await env.play()
                await env.get_robot_ready()

            health = env.camera_health()
            if len(health) != len(env.robot_configs) or not all(
                item["valid"] for item in health.values()
            ):
                raise RuntimeError(f"wrist cameras are not ready: {health}")

            expert = Policy(env, strict_demo=True)
            frame_count = 0
            episode_error = None
            while not expert.is_done() and frame_count < args.max_frames:
                obs = env.get_observations()
                action = expert.get_action(obs).astype(np.float32)
                try:
                    frame = {
                        "observation.state": np.asarray(
                            obs["joint_positions"], dtype=np.float32
                        ),
                        "action": action,
                        "task": "assemble the configured brick structure",
                        **_camera_frames(env, obs),
                    }
                    dataset.add_frame(frame)
                except Exception as exc:
                    episode_error = str(exc)
                    break
                env.robot_apply_action(action)
                # 30 Hz demonstration/control on top of 60 Hz simulation.
                await env.step()
                await env.step()
                frame_count += 1

            if expert.succeeded() and episode_error is None:
                dataset.save_episode()
                successes += 1
                print(
                    f"[collector] saved success {successes}/{args.episodes} "
                    f"(attempt {attempts}, {frame_count} frames)"
                )
            else:
                dataset.clear_episode_buffer(delete_images=True)
                reason = episode_error or expert.episode_result()
                print(f"[collector] discarded attempt {attempts}: {reason}")
    finally:
        dataset.finalize()

    if successes < args.episodes:
        raise RuntimeError(
            f"collected {successes}/{args.episodes} successes in "
            f"{attempts} attempts"
        )
    print(f"[collector] dataset ready at {output}")
