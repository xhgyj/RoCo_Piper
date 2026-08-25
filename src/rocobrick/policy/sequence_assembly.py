"""Strictly serialized dual-arm execution for the dense1 structure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from rocobrick.env.loose_parts import (
    aabb_clearance,
    aabb_inside,
    centered_aabb,
    footprint_aabb,
    with_planar_yaw,
)
from rocobrick.policy.assembly_control import AssemblyExpertConfig
from rocobrick.policy.gt_assembly import (
    ExpertResult,
    release_and_return_home,
    run_gt_assembly_expert,
    verify_connections,
)
from rocobrick.policy.multi_arm_gt_assembly import (
    arm_home_error,
    pickup_reachability_error,
    prepare_safe_start,
    resolve_single_step_task,
)

DENSE1_TARGET_ORDER = (2, 3, 1, 6, 4, 5)
DENSE1_WORKSPACE_CENTER = (0.0, 0.025)
DENSE1_WORKSPACE_SIZE = (0.18, 0.22)
DENSE1_STAGING_SIZE = (0.24, 0.22)
MIN_LOOSE_PART_CLEARANCE = 0.025
MIN_PLATE_CLEARANCE = 0.008


@dataclass(frozen=True)
class AssemblyTurn:
    """One target and the only arm authorized to assemble it."""

    target_id: int
    arm_index: int


@dataclass(frozen=True)
class SequenceResult:
    """Terminal result for a complete alternating assembly sequence."""

    success: bool
    completed_targets: tuple[int, ...]
    failed_target: int | None
    failure_reason: str | None


@dataclass(frozen=True)
class WorkspaceSlot:
    """One deterministic loose-part pose relative to the workspace center."""

    offset_xy: tuple[float, float]
    yaw_degrees: float = 0.0


def dense1_turns(starting_arm: int = 0) -> tuple[AssemblyTurn, ...]:
    """Return the fixed support-first, strictly alternating dense1 plan."""
    if starting_arm not in (0, 1):
        raise ValueError("starting_arm must be 0 or 1")
    return tuple(
        AssemblyTurn(target_id, (starting_arm + index) % 2)
        for index, target_id in enumerate(DENSE1_TARGET_ORDER)
    )


def dense1_workspace_slots() -> Mapping[int, WorkspaceSlot]:
    """Return separated pickup poses for each target's assigned arm."""
    return {
        1: WorkspaceSlot((-0.075, 0.0), yaw_degrees=90.0),
        2: WorkspaceSlot((0.012, -0.045), yaw_degrees=90.0),
        3: WorkspaceSlot((0.054, -0.045)),
        4: WorkspaceSlot((-0.025, 0.035)),
        5: WorkspaceSlot((0.025, 0.035)),
        6: WorkspaceSlot((-0.075, 0.080), yaw_degrees=90.0),
    }


async def arrange_dense1_workspace(env, settle_steps: int = 30) -> None:
    """Move staged dense1 bricks into deterministic assigned-arm slots."""
    from isaacsim.core.prims import SingleXFormPrim

    if settle_steps < 0:
        raise ValueError("settle_steps cannot be negative")
    slots = dense1_workspace_slots()
    if set(int(part_id) for part_id in env.to_place_placed) != set(slots):
        raise ValueError(
            "dense1 workspace requires loose parts 1..6; found "
            f"{sorted(env.to_place_placed)}"
        )
    for part_id, slot in slots.items():
        path = env.to_place_placed[part_id]
        desired = with_planar_yaw(
            env.get_prim_world_T(path), slot.yaw_degrees
        )
        desired[0, 3] = DENSE1_WORKSPACE_CENTER[0] + slot.offset_xy[0]
        desired[1, 3] = DENSE1_WORKSPACE_CENTER[1] + slot.offset_xy[1]
        quaternion = Rotation.from_matrix(desired[:3, :3]).as_quat()
        SingleXFormPrim(
            prim_path=path, name=f"dense1_workspace_part_{part_id}"
        ).set_world_pose(
            position=desired[:3, 3],
            orientation=np.array(
                [quaternion[3], quaternion[0], quaternion[1], quaternion[2]],
                dtype=np.float64,
            ),
        )
    for _ in range(settle_steps):
        await env.step()
    validate_dense1_workspace(env)


def validate_dense1_workspace(env) -> None:
    """Validate spacing and pickup IK for every target's assigned arm."""
    parts = {int(part["id"]): part for part in env.topology["parts"]}
    workspace = centered_aabb(DENSE1_WORKSPACE_CENTER, DENSE1_WORKSPACE_SIZE)
    plate_path = env.pre_placed_parts[0]
    plate_payload = parts[0]["payload"]
    plate_bounds = footprint_aabb(
        env.get_prim_world_T(plate_path),
        int(plate_payload["L"]),
        int(plate_payload["W"]),
    )
    bounds = {}
    for part_id, path in env.to_place_placed.items():
        payload = parts[int(part_id)]["payload"]
        part_bounds = footprint_aabb(
            env.get_prim_world_T(path), int(payload["L"]), int(payload["W"])
        )
        if not aabb_inside(part_bounds, workspace, tolerance=0.001):
            raise RuntimeError(f"loose part {part_id} lies outside shared workspace")
        plate_clearance = aabb_clearance(part_bounds, plate_bounds)
        if plate_clearance < MIN_PLATE_CLEARANCE:
            raise RuntimeError(
                f"loose part {part_id} plate clearance "
                f"{plate_clearance:.4f} m is below {MIN_PLATE_CLEARANCE:.4f} m"
            )
        bounds[int(part_id)] = part_bounds
    identifiers = sorted(bounds)
    for index, first_id in enumerate(identifiers):
        for second_id in identifiers[index + 1 :]:
            clearance = aabb_clearance(bounds[first_id], bounds[second_id])
            if clearance < MIN_LOOSE_PART_CLEARANCE:
                raise RuntimeError(
                    f"loose parts {first_id}/{second_id} clearance "
                    f"{clearance:.4f} m is below {MIN_LOOSE_PART_CLEARANCE:.4f} m"
                )
    assigned_arms = {turn.target_id: turn.arm_index for turn in dense1_turns()}
    for part_id, path in sorted(env.to_place_placed.items()):
        dimensions = parts[int(part_id)]["payload"]
        arm_index = assigned_arms[int(part_id)]
        error = pickup_reachability_error(env, path, dimensions, arm_index)
        if error is not None:
            raise RuntimeError(
                f"loose part {part_id} is unreachable by assigned arm "
                f"{arm_index}: {error}"
            )


async def run_dense1_sequence(
    env,
    runtime_config: AssemblyExpertConfig,
    safe_height: float = 0.06,
) -> SequenceResult:
    """Execute dense1 with a home-gated piper_0/piper_1 alternation.

    Returns:
        Completed targets and the first terminal failure, if any.
    """
    if len(env.robot_pins) != 2:
        raise ValueError(
            f"dense1 sequence requires two arms, found {len(env.robot_pins)}"
        )
    assembled = {
        int(part_id): path for part_id, path in env.pre_placed_parts.items()
    }
    completed: list[int] = []
    verified_tasks = []
    for turn in dense1_turns():
        for arm_index in range(2):
            error = arm_home_error(env, arm_index)
            if error is not None:
                return SequenceResult(
                    False, tuple(completed), turn.target_id, error
                )
        print(
            f"[sequence] target={turn.target_id} arm={turn.arm_index} start",
            flush=True,
        )
        try:
            task = resolve_single_step_task(
                env, target_id=turn.target_id, assembled_parts=assembled
            )
            prepared = await prepare_safe_start(
                env,
                safe_height=safe_height,
                task=task,
                arm_index=turn.arm_index,
            )
            result: ExpertResult = await run_gt_assembly_expert(
                env, prepared, runtime_config
            )
            if not result.success:
                return SequenceResult(
                    False,
                    tuple(completed),
                    turn.target_id,
                    result.failure_reason,
                )
            await release_and_return_home(env, prepared)
            home_error = arm_home_error(env, turn.arm_index)
            if home_error is not None:
                return SequenceResult(
                    False, tuple(completed), turn.target_id, home_error
                )
            path = env.to_place_placed.pop(turn.target_id)
            env.pre_placed_parts[turn.target_id] = path
            assembled[turn.target_id] = path
            completed.append(turn.target_id)
            verified_tasks.append(task)
            print(
                f"[sequence] target={turn.target_id} arm={turn.arm_index} "
                "connected and home",
                flush=True,
            )
        except (RuntimeError, ValueError) as exc:
            return SequenceResult(
                False, tuple(completed), turn.target_id, str(exc)
            )
    if not all(verify_connections(task) for task in verified_tasks):
        return SequenceResult(
            False, tuple(completed), None, "a completed connection was lost"
        )
    return SequenceResult(True, tuple(completed), None, None)
