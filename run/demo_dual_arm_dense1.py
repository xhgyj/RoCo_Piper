#!/usr/bin/env python3
"""Execute dense1 from an empty plate with strict dual-arm alternation."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime
from pathlib import Path

from rocobrick.env.Env import Env
from rocobrick.env.lifecycle import close_kit_app
from rocobrick.env.video import (
    GlobalCameraVideoRecorder,
    configure_global_camera_view,
)
from rocobrick.policy.gt_assembly import load_expert_config
from rocobrick.policy.sequence_assembly import (
    DENSE1_STAGING_SIZE,
    DENSE1_WORKSPACE_CENTER,
    arrange_dense1_workspace,
    run_dense1_sequence,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    """Parse options forwarded by the non-Isaac launcher.

    Returns:
        Parsed worker arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-output", type=Path)
    parser.add_argument(
        "--video-view",
        choices=("assembly-close", "overview"),
        default="assembly-close",
    )
    parser.add_argument("--final-hold-seconds", type=float, default=5.0)
    return parser.parse_args()


def _video_path(requested: Path | None) -> Path:
    if requested is not None:
        return requested.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPOSITORY_ROOT / "validation_reports" / (
        f"dense1_dual_arm_{timestamp}.mp4"
    )


async def main() -> None:
    """Initialize, preflight, execute, record, and close one dense1 run."""
    env = None
    recorder = None
    return_code = 0
    try:
        args = parse_args()
        if args.final_hold_seconds < 0.0:
            raise ValueError("final hold duration cannot be negative")
        task_dir = REPOSITORY_ROOT / "tasks/type2/dense1"
        with tempfile.TemporaryDirectory(prefix="roco-dense1-dual-") as temporary:
            config = json.loads(
                (REPOSITORY_ROOT / "config/user_config.json").read_text(
                    encoding="utf-8"
                )
            )
            config["Task_Config"]["Task_Path"] = str(task_dir)
            config["Task_Config"]["Task_Type"] = "2"
            configure_global_camera_view(config, args.video_view)
            storage = config["Env_Config"]["Storage_Config"]
            storage["Position"][0] = DENSE1_WORKSPACE_CENTER[0]
            storage["Position"][1] = DENSE1_WORKSPACE_CENTER[1]
            storage["Size"][0] = DENSE1_STAGING_SIZE[0]
            storage["Size"][1] = DENSE1_STAGING_SIZE[1]
            config_path = Path(temporary) / "user_config.json"
            config_path.write_text(
                json.dumps(config, indent=2) + "\n", encoding="utf-8"
            )
            env = Env(
                root_dir=str(SCRIPT_DIR),
                user_config_path=str(config_path),
                system_config_path="../config/system_config.json",
            )
            await env.reset()
            await env.play()
            env.hold_idle_arms_home = True
            if args.save_video:
                recorder = GlobalCameraVideoRecorder(
                    _video_path(args.video_output), fps=30
                )
                recorder.attach(env)
                print(f"[video] recording {recorder.output_path}", flush=True)
            await env.get_robot_ready()
            await arrange_dense1_workspace(env)
            runtime_config = load_expert_config(
                REPOSITORY_ROOT / "config/gt_assembly_expert.json"
            )
            result = await run_dense1_sequence(env, runtime_config)
            print(f"[sequence] result={result}", flush=True)
            if not result.success:
                raise RuntimeError(result.failure_reason)
            for _ in range(round(args.final_hold_seconds * 60)):
                await env.step()
    except BaseException:
        return_code = 1
        raise
    finally:
        try:
            if recorder is not None:
                recorder.close()
                print(
                    f"[video] saved {recorder.frame_count} frames to "
                    f"{recorder.output_path}",
                    flush=True,
                )
        finally:
            await close_kit_app(env, return_code)


if __name__ == "__main__":
    raise RuntimeError(
        "launch with: uv run python ./run/launch_dual_arm_dense1.py --show"
    )
