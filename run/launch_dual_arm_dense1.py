#!/usr/bin/env python3
"""Launch the dual-arm dense1 workflow visibly or headlessly."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
WORKER = SCRIPT_DIR / "demo_dual_arm_dense1.py"


def parse_args() -> argparse.Namespace:
    """Parse launcher and recording options.

    Returns:
        Parsed launcher arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--show", action="store_true", help="show the Kit window")
    display.add_argument(
        "--headless", action="store_true", help="disable the Kit window"
    )
    video = parser.add_mutually_exclusive_group()
    video.add_argument("--save-video", action="store_true")
    video.add_argument("--no-save-video", action="store_true")
    parser.add_argument("--video-output", type=Path)
    parser.add_argument(
        "--video-view",
        choices=("assembly-close", "overview"),
        default="assembly-close",
        help="global-camera framing used by the saved video",
    )
    parser.add_argument("--final-hold-seconds", type=float, default=5.0)
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    """Build the BrickSim worker command without starting Isaac Sim.

    Returns:
        BrickSim executable and forwarded worker arguments.
    """
    bricksim = shutil.which("bricksim")
    if bricksim is None:
        raise RuntimeError("bricksim executable is unavailable; run with uv run")
    command = [bricksim]
    if args.headless:
        command.append("--/app/window/enabled=false")
    command.append(str(WORKER))
    if args.save_video:
        command.append("--save-video")
    if args.video_output is not None:
        command.extend(("--video-output", str(args.video_output.resolve())))
    command.extend(("--video-view", args.video_view))
    command.extend(("--final-hold-seconds", str(args.final_hold_seconds)))
    return command


def main() -> int:
    """Launch BrickSim and return its process status.

    Returns:
        BrickSim process exit status.
    """
    args = parse_args()
    if args.final_hold_seconds < 0.0:
        raise ValueError("final hold duration cannot be negative")
    if args.video_output is not None and not args.save_video:
        raise ValueError("--video-output requires --save-video")
    completed = subprocess.run(build_command(args), check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
