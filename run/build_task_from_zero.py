#!/usr/bin/env python3
"""Build a Type-1 goal from an empty base plate with the unified API."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

PICKUP_XY_BY_ARM = {0: (0.1625, 0.005), 1: (-0.1625, 0.005)}
INITIAL_YAW_DEGREES = 90.0
PARKING_X = (0.02, 0.10, 0.18, 0.26)
PARKING_Y = (0.10, 0.17, 0.24)
DUAL_PARKING_SLOTS = {
    0: tuple((x, y) for y in (0.11, 0.20) for x in (0.08, 0.17, 0.26)),
    1: tuple((x, y) for y in (0.11, 0.20) for x in (-0.08, -0.17, -0.26)),
}


def _arguments() -> argparse.Namespace:
    """Parse task, robot assignment, visualization, and report options.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_dir", type=Path)
    parser.add_argument("--arm-index", type=int, default=0)
    parser.add_argument("--final-hold-seconds", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _part_order(topology: dict[str, object]) -> tuple[int, ...]:
    """Return a support-complete part order using BrickSim's BFS connections."""
    from bricksim.topology.ordering import bfs_sort_connections

    ordered = bfs_sort_connections(topology)
    connections = ordered["connections"]
    remaining = {
        int(part["id"]) for part in topology["parts"] if int(part["id"]) != 0
    }
    incoming: dict[int, set[int]] = {part_id: set() for part_id in remaining}
    first_seen: dict[int, int] = {}
    for index, connection in enumerate(connections):
        stud_id = int(connection["stud_id"])
        hole_id = int(connection["hole_id"])
        if hole_id in remaining:
            incoming[hole_id].add(stud_id)
            first_seen.setdefault(hole_id, index)

    assembled = {0}
    result: list[int] = []
    while remaining:
        ready = [
            part_id
            for part_id in remaining
            if incoming[part_id] and incoming[part_id] <= assembled
        ]
        if not ready:
            raise ValueError(
                "topology has no support-complete next part; "
                f"assembled={sorted(assembled)} remaining={sorted(remaining)}"
            )
        target_id = min(ready, key=lambda item: (first_seen[item], item))
        result.append(target_id)
        assembled.add(target_id)
        remaining.remove(target_id)
    return tuple(result)


def _action_payload(result) -> dict[str, object]:
    """Convert one public action result into JSON-safe diagnostics.

    Returns:
        Action status and structured failure details.
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


async def _arrange_staging(
    env,
    order: tuple[int, ...],
    assignments: dict[int, int] | None = None,
) -> None:
    """Park all loose parts separately with one fixed initial yaw."""
    import numpy as np
    from isaacsim.core.prims import SingleXFormPrim
    from scipy.spatial.transform import Rotation

    from rocobrick.env.loose_parts import with_planar_yaw

    single_slots = tuple((x, y) for y in PARKING_Y for x in PARKING_X)
    arm_slot_indices = {0: 0, 1: 0}
    for sequence_index, target_id in enumerate(order):
        arm_index = 0 if assignments is None else assignments[target_id]
        if assignments is None:
            if sequence_index >= len(single_slots):
                raise ValueError(
                    f"parking has {len(single_slots)} slots for {len(order)} parts"
                )
            x_position, y_position = single_slots[sequence_index]
        else:
            slots = DUAL_PARKING_SLOTS[arm_index]
            slot_index = arm_slot_indices[arm_index]
            if slot_index >= len(slots):
                raise ValueError(f"arm {arm_index} parking is full")
            x_position, y_position = slots[slot_index]
            arm_slot_indices[arm_index] += 1
        path = env.to_place_placed[target_id]
        desired = with_planar_yaw(
            env.get_prim_world_T(path), INITIAL_YAW_DEGREES
        )
        desired[0, 3] = x_position
        desired[1, 3] = y_position
        quaternion = Rotation.from_matrix(desired[:3, :3]).as_quat()
        SingleXFormPrim(
            prim_path=path, name=f"from_zero_parking_{target_id}"
        ).set_world_pose(
            position=desired[:3, 3],
            orientation=np.array(
                [quaternion[3], quaternion[0], quaternion[1], quaternion[2]],
                dtype=np.float64,
            ),
        )
    for _ in range(30):
        await env.step()


async def _stage_target(env, target_id: int, arm_index: int = 0) -> None:
    """Move the current target into its assigned arm's pickup slot."""
    import numpy as np
    from isaacsim.core.prims import SingleXFormPrim
    from scipy.spatial.transform import Rotation

    from rocobrick.env.loose_parts import with_planar_yaw

    path = env.to_place_placed[target_id]
    desired = with_planar_yaw(
        env.get_prim_world_T(path), INITIAL_YAW_DEGREES
    )
    try:
        desired[0, 3], desired[1, 3] = PICKUP_XY_BY_ARM[arm_index]
    except KeyError as error:
        raise ValueError(f"unsupported staging arm {arm_index}") from error
    quaternion = Rotation.from_matrix(desired[:3, :3]).as_quat()
    SingleXFormPrim(
        prim_path=path, name=f"from_zero_pickup_{target_id}"
    ).set_world_pose(
        position=desired[:3, 3],
        orientation=np.array(
            [quaternion[3], quaternion[0], quaternion[1], quaternion[2]],
            dtype=np.float64,
        ),
    )
    for _ in range(30):
        await env.step()


async def main() -> None:
    """Run the first failing turn or the complete from-zero sequence."""
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

        with tempfile.TemporaryDirectory(prefix="roco-from-zero-") as temporary:
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
                {
                    "Size": [0.32, 0.34, 0.10],
                    "Position": [0.15, 0.12, 0.05],
                }
            )
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
            await env.get_robot_ready()
            if args.arm_index < 0 or args.arm_index >= len(env.robot_pins):
                raise ValueError(f"invalid arm index {args.arm_index}")

            order = _part_order(env.topology)
            print(f"[sequence] from-zero order={order}", flush=True)
            await _arrange_staging(env, order)
            print(
                f"[sequence] parked {len(order)} parts at "
                f"yaw={INITIAL_YAW_DEGREES:.1f} deg",
                flush=True,
            )
            robot = BrickSimRobotBackend(env, args.arm_index)
            world = BrickSimWorldModel(env)
            assembled = {
                int(part_id): path
                for part_id, path in env.pre_placed_parts.items()
            }
            records: list[dict[str, object]] = []
            started = time.perf_counter()
            for turn_index, target_id in enumerate(order, start=1):
                await _stage_target(env, target_id)
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
                prefix = f"{task_dir.name}-{robot.robot_id}-{turn_index:02d}"
                print(
                    f"[sequence] turn={turn_index}/{len(order)} "
                    f"target={target_id} pick",
                    flush=True,
                )
                pick = await executor.execute(
                    ManipulationAction(
                        action_id=f"{prefix}-pick",
                        robot_ids=(robot.robot_id,),
                        skill_type=ManipulationSkillType.PICK,
                        object_id=target_path,
                        goal_id=goal_id,
                    )
                )
                record = {
                    "turn": turn_index,
                    "target_id": target_id,
                    "pick": _action_payload(pick),
                    "place_down": None,
                }
                records.append(record)
                if pick.status is not ActionStatus.SUCCESS:
                    break
                print(
                    f"[sequence] turn={turn_index}/{len(order)} "
                    f"target={target_id} place_down",
                    flush=True,
                )
                place = await executor.execute(
                    ManipulationAction(
                        action_id=f"{prefix}-place-down",
                        robot_ids=(robot.robot_id,),
                        skill_type=ManipulationSkillType.PLACE_DOWN,
                        object_id=target_path,
                        goal_id=goal_id,
                    )
                )
                record["place_down"] = _action_payload(place)
                if place.status is not ActionStatus.SUCCESS:
                    break
                path = env.to_place_placed.pop(target_id)
                env.pre_placed_parts[target_id] = path
                assembled[target_id] = path

            complete = len(records) == len(order) and all(
                record["place_down"] is not None
                and record["place_down"]["status"] == "success"
                for record in records
            )
            report = {
                "success": complete,
                "task_dir": str(task_dir),
                "robot_id": robot.robot_id,
                "order": list(order),
                "completed_parts": sum(
                    record["place_down"] is not None
                    and record["place_down"]["status"] == "success"
                    for record in records
                ),
                "elapsed_seconds": time.perf_counter() - started,
                "records": records,
            }
            return_code = 0 if complete else 1
            print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
            if complete and args.final_hold_seconds > 0.0:
                print(
                    f"[sequence] holding final scene for "
                    f"{args.final_hold_seconds:.1f}s",
                    flush=True,
                )
                for _ in range(round(args.final_hold_seconds * 60)):
                    await env.step()
    except BaseException as error:
        report = {
            "success": False,
            "setup_failure": f"{type(error).__name__}: {error}",
        }
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    finally:
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        from rocobrick.env.lifecycle import close_kit_app

        await close_kit_app(env, return_code)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
