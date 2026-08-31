"""BrickSim adapter for the unified manipulation executor."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from rocobrick.planning.models import (
    AssemblyGoal,
    ConnectionSpec,
    GraspRegion,
    GroundedAction,
    ObjectGeometry,
    OrientedBox,
    SceneGeometry,
)
from rocobrick.policy.gt_assembly import (
    _bricksim_success_check,
    compute_goal_brick_pose,
    resolve_single_step_task,
)
from rocobrick.policy.multi_arm_gt_assembly import (
    resolve_single_step_task as resolve_sequence_task,
)
from rocobrick.skills.base_skill import ManipulationAction

BRICK_STUD_PITCH = 0.008
BRICK_UNIT_HEIGHT = 0.0096


class BrickSimActionGrounder:
    """Ground a one-step downward assembly without selecting an arm."""

    def __init__(
        self,
        env,
        target_id: int | None = None,
        assembled_parts: Mapping[int, str] | None = None,
    ):
        """Bind world geometry and an optional sequence-selected target."""
        self._env = env
        self._target_id = target_id
        self._assembled_parts = (
            None if assembled_parts is None else dict(assembled_parts)
        )

    @staticmethod
    def assembly_goal_id(target_object_id: str) -> str:
        """Return the stable downward-assembly goal ID for one target.

        Returns:
            Domain-qualified goal identifier resolved by this grounder.
        """
        if not target_object_id:
            raise ValueError("target_object_id cannot be empty")
        return f"bricksim:place_down:{target_object_id}"

    def ground(self, action: ManipulationAction) -> GroundedAction:
        """Resolve symbolic IDs into OBBs and exact connection semantics.

        Returns:
            Geometry for Pick and, when requested, the downward assembly goal.
        """
        task = self._resolve_task()
        if action.object_id != task.target_path:
            raise ValueError(
                "BrickSim action object must be the target prim path "
                f"{task.target_path}"
            )
        parts = {int(part["id"]): part for part in self._env.topology["parts"]}
        target = self._target_geometry(
            task.target_path,
            self._env.get_prim_world_T(task.target_path),
            parts[task.target_id]["payload"],
        )
        obstacles = []
        for part_id, path in self._env.pre_placed_parts.items():
            if int(part_id) == 0:
                continue
            obstacles.append(
                self._box(
                    str(path),
                    self._env.get_prim_world_T(path),
                    parts[int(part_id)]["payload"],
                )
            )
        scene = SceneGeometry(target, tuple(obstacles))
        if action.goal_id is None:
            return GroundedAction(scene)
        expected_goal_id = self.assembly_goal_id(task.target_path)
        if action.goal_id != expected_goal_id:
            raise ValueError(
                f"unknown goal_id {action.goal_id}; expected {expected_goal_id}"
            )
        goal_pose = compute_goal_brick_pose(self._env, task)
        direction = -goal_pose[:3, 2]
        connections = tuple(
            ConnectionSpec(
                str(connection.reference_id),
                f"{connection.stud_iface}:{connection.hole_iface}",
                connection.offset,
                connection.yaw,
            )
            for connection in task.connections
        )
        goal = AssemblyGoal(
            expected_goal_id,
            goal_pose,
            direction,
            connections,
            tuple(connection.reference_path for connection in task.connections),
            _bricksim_success_check(task),
        )
        return GroundedAction(scene, goal)

    def _resolve_task(self):
        """Resolve the original task or the sequence-selected turn.

        Returns:
            Target-centric task consumed by BrickSim grounding.
        """
        if self._target_id is None:
            return resolve_single_step_task(self._env)
        if self._assembled_parts is None:
            raise ValueError("sequence grounding requires assembled_parts")
        return resolve_sequence_task(
            self._env,
            target_id=self._target_id,
            assembled_parts=self._assembled_parts,
        )

    def _target_geometry(self, object_id, pose, payload) -> ObjectGeometry:
        """Build separate collision and grasp frames for a BrickSim target.

        Returns:
            Target geometry with an asset-calibrated side-grasp region.
        """
        object_t_collision, collision_half_extents = self._collision_geometry(
            str(object_id), payload
        )
        grasp_half_extents = self._half_extents(payload)
        object_t_grasp = np.eye(4)
        object_t_grasp[2, 3] = 0.002
        return ObjectGeometry(
            str(object_id),
            pose,
            object_t_collision,
            collision_half_extents,
            (GraspRegion(object_t_grasp, grasp_half_extents),),
        )

    def _box(self, object_id, pose, payload) -> OrientedBox:
        """Convert BrickSim stud dimensions into a metric centered OBB.

        Returns:
            Metric object box at its current simulator pose.
        """
        object_t_collision, half_extents = self._collision_geometry(
            str(object_id), payload
        )
        return OrientedBox(str(object_id), pose @ object_t_collision, half_extents)

    @classmethod
    def _collision_geometry(cls, object_id, payload) -> tuple[np.ndarray, np.ndarray]:
        """Read the asset-local bound, falling back to symbolic dimensions.

        Returns:
            Object-to-collision transform and metric OBB half extents.
        """
        try:
            import omni.usd
            from pxr import Usd, UsdGeom

            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(object_id)
            if not prim.IsValid():
                raise ValueError(f"invalid prim path {object_id}")
            cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [
                    UsdGeom.Tokens.default_,
                    UsdGeom.Tokens.render,
                    UsdGeom.Tokens.proxy,
                ],
                useExtentsHint=True,
            )
            bounds = cache.ComputeLocalBound(prim).ComputeAlignedRange()
            minimum = np.asarray(bounds.GetMin(), dtype=np.float64)
            maximum = np.asarray(bounds.GetMax(), dtype=np.float64)
            half_extents = (maximum - minimum) * 0.5
            if not np.isfinite(half_extents).all() or np.any(half_extents <= 0.0):
                raise ValueError(f"invalid local bounds for {object_id}")
            object_t_collision = np.eye(4)
            object_t_collision[:3, 3] = (minimum + maximum) * 0.5
            return object_t_collision, half_extents
        except (ImportError, AttributeError, RuntimeError, ValueError):
            return np.eye(4), cls._half_extents(payload)

    @staticmethod
    def _half_extents(payload) -> np.ndarray:
        """Convert BrickSim dimensions into metric half extents.

        Returns:
            Length, width, and height half extents in meters.
        """
        return np.array(
            [
                float(payload["L"]) * BRICK_STUD_PITCH * 0.5,
                float(payload["W"]) * BRICK_STUD_PITCH * 0.5,
                float(payload["H"]) * BRICK_UNIT_HEIGHT * 0.5,
            ],
            dtype=np.float64,
        )
