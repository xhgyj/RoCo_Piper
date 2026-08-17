#!/usr/bin/env python3
"""Collect successful scripted-expert episodes in LeRobot v3 format."""

import argparse
import json
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
    parser.add_argument(
        "--review-dir",
        default=None,
        help="Optional directory for human-review MP4 previews and metadata.",
    )
    parser.add_argument(
        "--review-stride",
        type=int,
        default=4,
        help="Write one review frame every N collected frames.",
    )
    parser.add_argument(
        "--cameras",
        default="Global_Camera",
        help="Comma-separated camera keys to record, or 'all'.",
    )
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


def _selected_camera_keys(env, cameras_arg):
    if cameras_arg == "all":
        return sorted(env.cameras)
    selected = [key.strip() for key in cameras_arg.split(",") if key.strip()]
    missing = [key for key in selected if key not in env.cameras]
    if missing:
        raise RuntimeError(
            f"requested cameras are missing: {missing}; available={sorted(env.cameras)}"
        )
    return selected


def _features(env, use_videos, camera_keys):
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
    for cam_key in camera_keys:
        camera_cfg = env.camera_specs[cam_key]
        width, height = camera_cfg["Resolution"]
        features[f"observation.images.{_camera_feature_name(cam_key)}"] = {
            "dtype": image_dtype,
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def _camera_frames(env, obs, camera_keys):
    frames = {}
    for cam_key in camera_keys:
        camera_cfg = env.camera_specs[cam_key]
        source_key = f"{cam_key}_rgb"
        if source_key not in obs["images"]:
            raise RuntimeError(
                f"missing {source_key}; camera health={env.camera_health()}"
            )
        frames[f"observation.images.{_camera_feature_name(cam_key)}"] = _rgb_frame(
            obs["images"][source_key], camera_cfg["Resolution"]
        )
    return frames


def _camera_feature_name(cam_key):
    name = cam_key
    if name.endswith("_Wrist_Camera"):
        name = f"{name[:-len('_Wrist_Camera')]}_wrist"
    elif name.endswith("_Camera"):
        name = name[:-len("_Camera")]
    return name.lower()


def _matrix_list(value):
    return np.asarray(value, dtype=float).tolist()


def _task_metadata(env, expert, health, camera_keys):
    tasks = []
    for task in expert.plan:
        tasks.append({
            "stud_path": task["stud_path"],
            "hole_path": task["hole_path"],
            "stud_iface": task["stud_iface"],
            "hole_iface": task["hole_iface"],
            "offset": list(task["offset"]),
            "yaw": task["yaw"],
            "dimensions": task["dimensions"],
            "stud_world_T": _matrix_list(env.get_prim_world_T(task["stud_path"])),
            "hole_initial_world_T": _matrix_list(
                env.get_prim_world_T(task["hole_path"])
            ),
        })
    return {
        "camera_health": health,
        "camera_specs": {
            key: env.camera_specs[key]
            for key in camera_keys
        },
        "tasks": tasks,
        "global_joint_order": env.global_joint_order,
    }


class ReviewRecorder:
    def __init__(self, root, fps, stride):
        self.root = None if root is None else Path(root)
        self.fps = fps
        self.stride = max(1, stride)
        self.writer = None
        self.tmp_path = None
        self.frame_count = 0
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)

    def start_attempt(self, attempt):
        if self.root is None:
            return
        import imageio.v2 as imageio

        self.tmp_path = self.root / f".attempt_{attempt:06d}.mp4"
        self.writer = imageio.get_writer(
            self.tmp_path,
            fps=max(1, round(self.fps / self.stride)),
            macro_block_size=1,
        )
        self.frame_count = 0

    def add(self, frames):
        if self.writer is None:
            return
        if self.frame_count % self.stride == 0:
            ordered = [frames[key] for key in sorted(frames)]
            if ordered:
                self.writer.append_data(np.concatenate(ordered, axis=1))
        self.frame_count += 1

    def save_success(self, episode_id, metadata):
        if self.root is None:
            return
        self._close()
        video_path = self.root / f"episode_{episode_id:06d}.mp4"
        meta_path = self.root / f"episode_{episode_id:06d}.json"
        if self.tmp_path is not None and self.tmp_path.exists():
            self.tmp_path.replace(video_path)
        meta_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.tmp_path = None

    def discard(self):
        self._close()
        if self.tmp_path is not None and self.tmp_path.exists():
            self.tmp_path.unlink()
        self.tmp_path = None

    def _close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None


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
    camera_keys = _selected_camera_keys(env, args.cameras)
    use_videos = not args.no_video
    review_root = None
    if args.review_dir:
        review_root = (script_dir / args.review_dir).resolve()
    reviewer = ReviewRecorder(review_root, args.fps, args.review_stride)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=output,
        robot_type="dual_piper_l",
        features=_features(env, use_videos, camera_keys),
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

            full_health = env.camera_health()
            health = {key: full_health.get(key) for key in camera_keys}
            if len(health) != len(camera_keys) or not all(
                item is not None and item["valid"] for item in health.values()
            ):
                raise RuntimeError(
                    f"selected cameras are not ready: {health}; "
                    f"all camera health={full_health}"
                )

            expert = Policy(env, strict_demo=True)
            episode_metadata = _task_metadata(env, expert, health, camera_keys)
            reviewer.start_attempt(attempts)
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
                        **_camera_frames(env, obs, camera_keys),
                    }
                    dataset.add_frame(frame)
                    reviewer.add({
                        key: value
                        for key, value in frame.items()
                        if key.startswith("observation.images.")
                    })
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
                episode_metadata.update({
                    "episode_index": successes,
                    "attempt": attempts,
                    "frames": frame_count,
                    "expert_result": expert.episode_result(),
                })
                reviewer.save_success(successes, episode_metadata)
                print(
                    f"[collector] saved success {successes}/{args.episodes} "
                    f"(attempt {attempts}, {frame_count} frames)"
                )
            else:
                reviewer.discard()
                dataset.clear_episode_buffer(delete_images=True)
                reason = episode_error or expert.episode_result()
                print(f"[collector] discarded attempt {attempts}: {reason}")
    finally:
        reviewer.discard()
        dataset.finalize()

    if successes < args.episodes:
        raise RuntimeError(
            f"collected {successes}/{args.episodes} successes in "
            f"{attempts} attempts"
        )
    print(f"[collector] dataset ready at {output}")
