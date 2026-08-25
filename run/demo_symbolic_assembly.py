#!/usr/bin/env python3
"""Visualize a symbolic task and run the full scripted assembly expert."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, UsdGeom
from scipy.spatial.transform import Rotation

from rocobrick.env.Env import Env
from rocobrick.env.lifecycle import close_kit_app
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
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="stop after the loose target reaches the safe pose above the goal",
    )
    return parser.parse_args()


async def main() -> None:
    """Show the initial scene, execute the expert, and hold the final state."""
    env = None
    return_code = 0
    try:
        args = parse_args()
        if args.inspect_seconds < 0 or args.final_hold_seconds < 0:
            raise ValueError("demo hold durations cannot be negative")
        with tempfile.TemporaryDirectory(
            prefix="roco-symbolic-demo-"
        ) as temporary:
            temporary_path = Path(temporary)
            task_dir, generated = _resolve_task(args, temporary_path)
            config_path = _write_demo_config(task_dir, temporary_path)
            env = Env(
                root_dir=str(SCRIPT_DIR),
                user_config_path=str(config_path),
                system_config_path="../config/system_config.json",
            )
            await env.reset()
            await env.play()
            await env.get_robot_ready()

            task = resolve_single_step_task(env)
            goal_brick = compute_goal_brick_pose(env, task)
            target_world = env.get_prim_world_T(task.target_path)
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
        await close_kit_app(env, return_code)


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
