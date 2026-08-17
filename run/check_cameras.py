#!/usr/bin/env python3
"""Check wrist camera health and save one RGB snapshot per camera."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from rocobrick.env.Env import Env


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="../camera_check")
    parser.add_argument("--warmup-steps", type=int, default=30)
    return parser.parse_args()


def _rgb_image(value):
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise RuntimeError(f"invalid RGB shape: {image.shape}")
    image = image[:, :, :3]
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and image.max(initial=0) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    if int(image.max()) - int(image.min()) < 3:
        raise RuntimeError("camera frame has no useful dynamic range")
    return np.ascontiguousarray(image)


async def main():
    args = _arguments()
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    output = (script_dir / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    env = Env(
        root_dir=str(script_dir),
        user_config_path="../config/user_config.json",
        system_config_path="../config/system_config.json",
    )
    await env.reset()
    await env.play()
    await env.get_robot_ready()
    for _ in range(args.warmup_steps):
        await env.step()

    health = env.camera_health()
    print(json.dumps(health, ensure_ascii=False, indent=2))
    if len(health) != len(env.cameras) or not all(
        item["valid"] for item in health.values()
    ):
        raise RuntimeError(f"cameras are not ready: {health}")

    obs = env.get_observations()
    saved = []
    for cam_key in sorted(env.cameras):
        key = f"{cam_key}_rgb"
        if key not in obs["images"]:
            raise RuntimeError(f"missing {key}; camera health={health}")
        path = output / f"{cam_key.lower()}.png"
        Image.fromarray(_rgb_image(obs["images"][key])).save(path)
        saved.append(str(path))
    print("saved snapshots:")
    for path in saved:
        print(path)
