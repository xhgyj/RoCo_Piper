#!/usr/bin/env python3
"""Visualize a symbolic task and run the full scripted assembly expert."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, UsdGeom
from scipy.spatial.transform import Rotation

from rocobrick.env.Env import Env
from rocobrick.env.lifecycle import close_kit_app
from rocobrick.env.video import (
    GlobalCameraVideoRecorder,
    configure_global_camera_view,
)
from rocobrick.policy.gt_assembly import (
    compute_goal_brick_pose,
    load_expert_config,
    prepare_safe_start,
    release_and_return_home,
    resolve_single_step_task,
    run_gt_assembly_expert,
)
from rocobrick.task_config.symbolic_assembly import (
    GeneratedTask,
    generate_demo_task,
    write_generated_task,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent
PREVIEW_PATH = "/World/SymbolicGoalPreview"


def parse_args() -> argparse.Namespace:
    """Parse visual-demo options.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--task-dir", type=Path)
    source.add_argument(
        "--family",
        choices=("basic", "adjacent", "multilevel", "dense", "bridge"),
        default="basic",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--inspect-seconds", type=float, default=5.0)
    parser.add_argument("--final-hold-seconds", type=float, default=10.0)
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
    yaw = parser.add_mutually_exclusive_group()
    yaw.add_argument(
        "--initial-yaw-deg",
        type=float,
        help="set the loose target to this absolute world yaw before pickup",
    )
    yaw.add_argument(
        "--random-initial-yaw",
        action="store_true",
        help="sample loose-target world yaw continuously from [-180, 180)",
    )
    parser.add_argument(
        "--yaw-seed",
        type=int,
        default=0,
        help="reproducible seed used by --random-initial-yaw",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="stop after the loose target reaches the safe pose above the goal",
    )
    return parser.parse_args()


async def main() -> None:
    """Show the initial scene, execute the expert, and hold the final state."""
    env = None
    recorder = None
    return_code = 0
    try:
        args = parse_args()
        if args.inspect_seconds < 0 or args.final_hold_seconds < 0:
            raise ValueError("demo hold durations cannot be negative")
        if args.video_output is not None and not args.save_video:
            raise ValueError("--video-output requires --save-video")
        with tempfile.TemporaryDirectory(
            prefix="roco-symbolic-demo-"
        ) as temporary:
            temporary_path = Path(temporary)
            task_dir, generated = _resolve_task(args, temporary_path)
            config_path = _write_demo_config(task_dir, temporary_path)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            configure_global_camera_view(config, args.video_view)
            config_path.write_text(
                json.dumps(config, indent=2) + "\n", encoding="utf-8"
            )
            env = Env(
                root_dir=str(SCRIPT_DIR),
                user_config_path=str(config_path),
                system_config_path="../config/system_config.json",
            )
            await env.reset()
            applied_yaw = _apply_initial_yaw(env, args)
            await env.play()
            await env.get_robot_ready()
            if args.save_video:
                recorder = GlobalCameraVideoRecorder(
                    _video_path(args.video_output), fps=30
                )
                recorder.attach(env)
                print(f"[video] recording {recorder.output_path}", flush=True)

            task = resolve_single_step_task(env)
            goal_brick = compute_goal_brick_pose(env, task)
            target_world = env.get_prim_world_T(task.target_path)
            if applied_yaw is not None:
                actual_yaw = Rotation.from_matrix(
                    target_world[:3, :3]
                ).as_euler("zyx", degrees=True)[0]
                print(
                    "[symbolic-demo] loose target initial yaw: "
                    f"requested={applied_yaw:.3f} deg, "
                    f"actual={actual_yaw:.3f} deg, seed={args.yaw_seed}"
                )
            print(
                "[symbolic-demo] loose target starts outside plate: "
                f"xyz={target_world[:3, 3].tolist()}"
            )
            print(
                "[symbolic-demo] connection: "
                f"offset={task.primary.offset} yaw={task.primary.yaw} "
                f"additional={len(task.additional)}"
            )
            _add_goal_preview(goal_brick, task.dimensions)
            print(
                f"[symbolic-demo] inspect initial structure for "
                f"{args.inspect_seconds:.1f}s"
            )
            await _hold(env, args.inspect_seconds)
            _remove_goal_preview()
            prepared = await prepare_safe_start(env)
            if args.prepare_only:
                print(
                    "[symbolic-demo] preparation-only success; "
                    f"holding safe start for {args.final_hold_seconds:.1f}s"
                )
                await _hold(env, args.final_hold_seconds)
                return
            runtime_config = load_expert_config(
                REPOSITORY_ROOT / "config/gt_assembly_expert.json"
            )
            print("[symbolic-demo] assembly trajectory starts at safe pose")
            result = await run_gt_assembly_expert(env, prepared, runtime_config)
            print(f"[symbolic-demo] result: {result}")
            if not result.success:
                raise RuntimeError(f"scripted expert failed: {result}")
            await release_and_return_home(env, prepared)
            print(
                f"[symbolic-demo] holding successful final structure for "
                f"{args.final_hold_seconds:.1f}s"
            )
            await _hold(env, args.final_hold_seconds)
            if generated is not None:
                print(
                    f"[symbolic-demo] generated sample={generated.sample_id} "
                    f"family={generated.family}"
                )
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


def _video_path(requested: Path | None) -> Path:
    if requested is not None:
        return requested.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPOSITORY_ROOT / "validation_reports" / (
        f"symbolic_assembly_{timestamp}.mp4"
    )


def _resolve_task(
    args: argparse.Namespace, temporary_path: Path
) -> tuple[Path, GeneratedTask | None]:
    if args.task_dir is not None:
        task_dir = args.task_dir.resolve()
        for name in ("structure_start.json", "structure_goal.json"):
            if not (task_dir / name).is_file():
                raise FileNotFoundError(task_dir / name)
        return task_dir, None
    generated = generate_demo_task(args.family, args.seed)
    task_dir = write_generated_task(temporary_path / "tasks", generated)
    return task_dir, generated


def _write_demo_config(task_dir: Path, temporary_path: Path) -> Path:
    config = json.loads(
        (REPOSITORY_ROOT / "config/user_config.json").read_text(encoding="utf-8")
    )
    config["Task_Config"]["Task_Path"] = str(task_dir)
    config["Task_Config"]["Task_Type"] = "1"
    path = temporary_path / "user_config.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def _add_goal_preview(
    desired: np.ndarray, dimensions: dict[str, int]
) -> None:
    """Draw a translucent target brick at the computed GT goal pose."""
    center = desired[:3, 3] + desired[:3, 2] * 0.0048
    rotation = Rotation.from_matrix(desired[:3, :3]).as_quat()
    stage = get_current_stage()
    preview = UsdGeom.Cube.Define(stage, PREVIEW_PATH)
    preview.GetSizeAttr().Set(1.0)
    preview.GetDisplayColorAttr().Set([Gf.Vec3f(0.1, 1.0, 0.1)])
    preview.GetDisplayOpacityAttr().Set([0.35])
    xformable = UsdGeom.Xformable(preview.GetPrim())
    xformable.AddTranslateOp().Set(Gf.Vec3d(*center))
    xformable.AddOrientOp().Set(
        Gf.Quatf(rotation[3], rotation[0], rotation[1], rotation[2])
    )
    xformable.AddScaleOp().Set(
        Gf.Vec3d(
            dimensions["L"] * 0.008,
            dimensions["W"] * 0.008,
            0.0096,
        )
    )


def _remove_goal_preview() -> None:
    get_current_stage().RemovePrim(PREVIEW_PATH)


async def _hold(env: Env, seconds: float) -> None:
    for _ in range(round(seconds * 30)):
        await env.step()


def _apply_initial_yaw(env: Env, args: argparse.Namespace) -> float | None:
    """Apply the requested expert-validation yaw before physics starts.

    Returns:
        Applied yaw, or None when retaining BrickSim arranger behavior.
    """
    if args.initial_yaw_deg is not None:
        return env.set_loose_target_yaw(args.initial_yaw_deg)
    if args.random_initial_yaw:
        return env.randomize_loose_target_yaw(args.yaw_seed)
    return None
