#!/usr/bin/env python3
"""Build Task D with a synchronized dual-Piper Pick/PlaceDown pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from rocobrick.policy.from_zero_staging import (
    INITIAL_YAW_DEGREES,
    arrange_dual_staging,
    part_order,
    stage_target,
)

BASE_TARGET = 2
PIPER_0_CHAIN = (1, 4, 6, 8, 11)
PIPER_1_CHAIN = (3, 5, 7, 9, 12)
BRIDGE_TARGET = 10
ASSIGNMENTS = {
    BASE_TARGET: 0,
    **{target_id: 0 for target_id in PIPER_0_CHAIN},
    **{target_id: 1 for target_id in PIPER_1_CHAIN},
    BRIDGE_TARGET: 1,
}


class _DualCameraVideoRecorder:
    """Record the global overview and close-up cameras from one simulation."""

    def __init__(self, output_dir: Path, fps: int = 30) -> None:
        self.output_dir = output_dir
        self.fps = fps
        self.frame_count = 0
        self._step_count = 0
        self._writers = {}
        self._attached_env = None
        self._original_step = None

    def attach(self, env) -> None:
        """Attach to ``env.step`` and start both MP4 writers."""
        if self._attached_env is not None:
            raise RuntimeError("video recorder is already attached")
        required = ("Global_Camera", "Close_Camera")
        missing = [name for name in required if name not in env.cameras]
        if missing:
            raise RuntimeError(f"video cameras unavailable: {missing}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for name, filename in (
            ("Global_Camera", "taskD_global.mp4"),
            ("Close_Camera", "taskD_close.mp4"),
        ):
            self._writers[name] = imageio.get_writer(
                self.output_dir / filename,
                fps=self.fps,
                codec="libx264",
                macro_block_size=None,
            )
        self._attached_env = env
        self._original_step = env.step

        async def recording_step() -> None:
            await self._original_step()
            self._step_count += 1
            self._append_frames(env)

        env.step = recording_step

    def _append_frames(self, env) -> None:
        """Append one RGB frame from each camera."""
        for name, writer in self._writers.items():
            value = env.cameras[name].get_rgb()
            if value is None:
                continue
            frame = np.asarray(value)
            if frame.ndim != 3 or frame.shape[2] not in (3, 4):
                raise RuntimeError(f"invalid {name} RGB shape: {frame.shape}")
            frame = frame[:, :, :3]
            if frame.dtype != np.uint8:
                if (
                    np.issubdtype(frame.dtype, np.floating)
                    and frame.max(initial=0) <= 1.0
                ):
                    frame = frame * 255.0
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            writer.append_data(np.ascontiguousarray(frame))
        self.frame_count += 1

    def close(self) -> None:
        """Detach and finalize both MP4 files."""
        if self._attached_env is not None:
            self._attached_env.step = self._original_step
            self._attached_env = None
            self._original_step = None
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()


class _SynchronizedWorld:
    """Serialize shared simulation stepping across concurrent arm actions."""

    def __init__(self, world):
        self._world = world
        self._step_lock = asyncio.Lock()

    def object_pose(self, object_id):
        """Forward an object-pose query to the BrickSim world.

        Returns:
            Current world transform for the requested object.
        """
        return self._world.object_pose(object_id)

    async def advance(self, steps: int = 1) -> None:
        """Advance BrickSim without overlapping another arm's step call."""
        async with self._step_lock:
            await self._world.advance(steps)


def _arguments() -> argparse.Namespace:
    """Parse dual-arm task and report options.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_dir", type=Path)
    parser.add_argument("--final-hold-seconds", type=float, default=0.0)
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("validation_reports/taskD_dual_videos"),
        help="directory receiving taskD_global.mp4 and taskD_close.mp4",
    )
    return parser.parse_args()


def _action_payload(result) -> dict[str, object]:
    """Convert one action result into JSON-safe diagnostics.

    Returns:
        Action status and optional structured failure.
    """
    return {
        "action_id": result.action_id,
        "status": result.status.value,
        "object_id": result.object_id,
        "held_by": result.held_by,
        "failure": (
            None
            if result.failure is None
            else {
                "code": result.failure.code.value,
                "stage": result.failure.stage,
                "detail": result.failure.detail,
            }
        ),
    }


async def _return_home(robot, world) -> None:
    """Move one released arm home before the other enters the structure."""
    current = robot.read_state().q
    target = robot.home_configuration
    arm_indices = list(robot.arm_configuration_indices)
    distance = float(np.max(np.abs(target[arm_indices] - current[arm_indices])))
    waypoint_count = max(1, int(np.ceil(distance / 0.04)))
    for index in range(1, waypoint_count + 1):
        fraction = index / waypoint_count
        command = current * (1.0 - fraction) + target * fraction
        if not robot.configuration_is_safe(command):
            raise RuntimeError(f"{robot.robot_id} home path became unsafe")
        robot.command_configuration(command)
        await world.advance(2)
    for _ in range(30):
        robot.command_configuration(target)
        await world.advance(1)
    actual = robot.read_state().q
    error = float(np.max(np.abs(actual[arm_indices] - target[arm_indices])))
    if error > 0.04:
        raise RuntimeError(
            f"{robot.robot_id} failed to return home: joint error={error:.4f}"
        )
    print(f"[dual] {robot.robot_id} returned home", flush=True)


async def main() -> None:
    """Run Task D with at most one concurrent PlaceDown and one Pick."""
    args = _arguments()
    if args.final_hold_seconds < 0.0:
        raise ValueError("final-hold-seconds cannot be negative")
    task_dir = args.task_dir.resolve()
    goal_path = task_dir / "structure_goal.json"
    if not goal_path.is_file():
        raise FileNotFoundError(goal_path)

    script_dir = Path(__file__).resolve().parent
    repository = script_dir.parent
    env = None
    recorder = None
    return_code = 1
    report: dict[str, object] = {}
    try:
        from rocobrick.backends.bricksim import (
            BrickSimRobotBackend,
            BrickSimWorldModel,
        )
        from rocobrick.env.Env import Env
        from rocobrick.execution import ActionStatus, ManipulationExecutor
        from rocobrick.policy.bricksim_grounder import BrickSimActionGrounder
        from rocobrick.skills import ManipulationAction, ManipulationSkillType

        with tempfile.TemporaryDirectory(prefix="roco-dual-zero-") as temporary:
            temporary_path = Path(temporary)
            empty_task = temporary_path / "task"
            empty_task.mkdir()
            (empty_task / "structure_start.json").write_text(
                "{}\n", encoding="utf-8"
            )
            (empty_task / "structure_goal.json").write_text(
                goal_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
            config = json.loads(
                (repository / "config/user_config.json").read_text(encoding="utf-8")
            )
            config["Task_Config"]["Task_Path"] = str(empty_task)
            config["Task_Config"]["Task_Type"] = "1"
            config["Env_Config"]["Storage_Config"].update(
                {"Size": [0.64, 0.34, 0.10], "Position": [0.0, 0.12, 0.05]}
            )
            camera_config = config["Env_Config"]["Camera_Config"]
            camera_config["Global_Camera"].update(
                {
                    "Position": [0.0, 1.05, 0.95],
                    "Target": [0.0, -0.12, 0.24],
                    "Focal_Length": 16.0,
                    "Resolution": [960, 540],
                }
            )
            camera_config["Close_Camera"] = {
                "FPS": 30,
                "Resolution": [1280, 720],
                "Prim_Path": "/World/Close_Camera",
                "Position": [0.0, 0.42, 0.42],
                "Target": [0.0, -0.20, 0.035],
                "Focal_Length": 36.0,
                "Clipping_Range": [0.01, 10.0],
            }
            config_path = temporary_path / "user_config.json"
            config_path.write_text(
                json.dumps(config, indent=2) + "\n", encoding="utf-8"
            )

            env = Env(
                root_dir=str(script_dir),
                user_config_path=str(config_path),
                system_config_path="../config/system_config.json",
            )
            await env.reset()
            await env.play()
            recorder = _DualCameraVideoRecorder(args.video_dir.resolve(), fps=30)
            recorder.attach(env)
            print(
                f"[video] recording global and close-up views to {recorder.output_dir}",
                flush=True,
            )
            await env.get_robot_ready()
            if len(env.robot_pins) < 2:
                raise ValueError("dual pipeline requires piper_0 and piper_1")

            order = part_order(env.topology)
            if set(order) != set(ASSIGNMENTS):
                raise ValueError(
                    "dual Task D assignment does not match topology: "
                    f"order={order}"
                )
            await arrange_dual_staging(env, order, ASSIGNMENTS)
            print(
                f"[dual] parked {len(order)} parts at "
                f"yaw={INITIAL_YAW_DEGREES:.1f} deg",
                flush=True,
            )

            robots = {
                arm_index: BrickSimRobotBackend(env, arm_index)
                for arm_index in (0, 1)
            }
            world = _SynchronizedWorld(BrickSimWorldModel(env))
            assembled = {
                int(part_id): path
                for part_id, path in env.pre_placed_parts.items()
            }
            records: dict[int, dict[str, object]] = {}
            contexts = {}
            started = time.perf_counter()

            def prepare_turn(target_id: int):
                arm_index = ASSIGNMENTS[target_id]
                robot = robots[arm_index]
                target_path = env.to_place_placed[target_id]
                grounder = BrickSimActionGrounder(
                    env, target_id=target_id, assembled_parts=assembled
                )
                executor = ManipulationExecutor(
                    robots={robot.robot_id: robot},
                    world=world,
                    grounder=grounder,
                )
                goal_id = grounder.assembly_goal_id(target_path)
                prefix = f"{task_dir.name}-{robot.robot_id}-{target_id:02d}"
                pick_action = ManipulationAction(
                    action_id=f"{prefix}-pick",
                    robot_ids=(robot.robot_id,),
                    skill_type=ManipulationSkillType.PICK,
                    object_id=target_path,
                    goal_id=goal_id,
                )
                place_action = ManipulationAction(
                    action_id=f"{prefix}-place-down",
                    robot_ids=(robot.robot_id,),
                    skill_type=ManipulationSkillType.PLACE_DOWN,
                    object_id=target_path,
                    goal_id=goal_id,
                )
                contexts[target_id] = (executor, pick_action, place_action)
                records[target_id] = {
                    "target_id": target_id,
                    "arm_index": arm_index,
                    "robot_id": robot.robot_id,
                    "pick": None,
                    "place_down": None,
                }

            async def pick(target_id: int):
                executor, pick_action, _ = contexts[target_id]
                print(
                    f"[dual] {pick_action.robot_ids[0]} pick target={target_id}",
                    flush=True,
                )
                result = await executor.execute(pick_action)
                records[target_id]["pick"] = _action_payload(result)
                return result

            async def place(target_id: int):
                executor, _, place_action = contexts[target_id]
                print(
                    f"[dual] {place_action.robot_ids[0]} place target={target_id}",
                    flush=True,
                )
                result = await executor.execute(place_action)
                records[target_id]["place_down"] = _action_payload(result)
                return result

            def promote(target_id: int) -> None:
                path = env.to_place_placed.pop(target_id)
                env.pre_placed_parts[target_id] = path
                assembled[target_id] = path

            async def stage_and_prepare(target_id: int) -> None:
                await stage_target(env, target_id, ASSIGNMENTS[target_id])
                prepare_turn(target_id)

            async def require_success(result, target_id: int, phase: str) -> bool:
                if result.status is ActionStatus.SUCCESS:
                    return True
                print(
                    f"[dual] stop target={target_id} phase={phase} "
                    f"status={result.status.value}",
                    flush=True,
                )
                return False

            complete = False
            await stage_and_prepare(BASE_TARGET)
            base_pick = await pick(BASE_TARGET)
            if await require_success(base_pick, BASE_TARGET, "pick"):
                base_place = await place(BASE_TARGET)
                if await require_success(base_place, BASE_TARGET, "place_down"):
                    promote(BASE_TARGET)

                    await stage_and_prepare(PIPER_0_CHAIN[0])
                    await stage_and_prepare(PIPER_1_CHAIN[0])
                    initial_pick = await pick(PIPER_0_CHAIN[0])
                    running = await require_success(
                        initial_pick, PIPER_0_CHAIN[0], "pick"
                    )
                    if running:
                        first_place, first_pick = await asyncio.gather(
                            place(PIPER_0_CHAIN[0]), pick(PIPER_1_CHAIN[0])
                        )
                        running = await require_success(
                            first_place, PIPER_0_CHAIN[0], "place_down"
                        ) and await require_success(
                            first_pick, PIPER_1_CHAIN[0], "pick"
                        )
                        if running:
                            promote(PIPER_0_CHAIN[0])
                        for index in range(len(PIPER_0_CHAIN) - 1):
                            if not running:
                                break
                            next_piper_0 = PIPER_0_CHAIN[index + 1]
                            await stage_and_prepare(next_piper_0)
                            left_place, right_pick = await asyncio.gather(
                                place(PIPER_1_CHAIN[index]), pick(next_piper_0)
                            )
                            running = await require_success(
                                left_place,
                                PIPER_1_CHAIN[index],
                                "place_down",
                            ) and await require_success(
                                right_pick, next_piper_0, "pick"
                            )
                            if not running:
                                break
                            promote(PIPER_1_CHAIN[index])

                            next_piper_1 = PIPER_1_CHAIN[index + 1]
                            await stage_and_prepare(next_piper_1)
                            right_place, left_pick = await asyncio.gather(
                                place(next_piper_0), pick(next_piper_1)
                            )
                            running = await require_success(
                                right_place, next_piper_0, "place_down"
                            ) and await require_success(
                                left_pick, next_piper_1, "pick"
                            )
                            if running:
                                promote(next_piper_0)
                        if running:
                            final_pillar = PIPER_1_CHAIN[-1]
                            await _return_home(robots[0], world)
                            pillar_place = await place(final_pillar)
                            running = await require_success(
                                pillar_place, final_pillar, "place_down"
                            )
                            if running:
                                promote(final_pillar)
                        if running:
                            await stage_and_prepare(BRIDGE_TARGET)
                            bridge_pick = await pick(BRIDGE_TARGET)
                            running = await require_success(
                                bridge_pick, BRIDGE_TARGET, "pick"
                            )
                        if running:
                            bridge_place = await place(BRIDGE_TARGET)
                            running = await require_success(
                                bridge_place, BRIDGE_TARGET, "place_down"
                            )
                            if running:
                                promote(BRIDGE_TARGET)
                                complete = True

            ordered_records = [records[item] for item in order if item in records]
            report = {
                "success": complete,
                "task_dir": str(task_dir),
                "mode": "dual_arm_pick_pipeline",
                "yaw_degrees": INITIAL_YAW_DEGREES,
                "assignments": {str(key): value for key, value in ASSIGNMENTS.items()},
                "completed_parts": sum(
                    record["place_down"] is not None
                    and record["place_down"]["status"] == "success"
                    for record in ordered_records
                ),
                "elapsed_seconds": time.perf_counter() - started,
                "records": ordered_records,
            }
            return_code = 0 if complete else 1
            print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
            if complete and args.final_hold_seconds > 0.0:
                for _ in range(round(args.final_hold_seconds * 60)):
                    await env.step()
    except BaseException as error:
        report = {
            "success": False,
            "setup_failure": f"{type(error).__name__}: {error}",
        }
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    finally:
        if recorder is not None:
            recorder.close()
            print(
                f"[video] saved {recorder.frame_count} frames per view to "
                f"{recorder.output_dir}",
                flush=True,
            )
        from rocobrick.env.lifecycle import close_kit_app

        await close_kit_app(env, return_code)


if __name__ == "__main__":
    asyncio.run(main())
