"""BrickSim adapters for backend-independent manipulation code."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from rocobrick.backends.base import RobotState
from rocobrick.execution.types import ExecutionError, FailureCode

FloatArray = NDArray[np.float64]

SAFE_IK_POSITION_TOLERANCE = 0.004
SAFE_IK_ROTATION_TOLERANCE = np.deg2rad(3.0)
GRIPPER_OPEN_MARGIN_PER_FINGER = 0.006
GRIPPER_MAX_JOINT_OPENING = 0.045


@dataclass(frozen=True)
class BrickSimConnectionGoal:
    """One requested BrickSim connection without policy-layer dependencies."""

    reference_path: str
    stud_iface: int
    target_path: str
    hole_iface: int
    offset: tuple[int, int]
    yaw: int


class BrickSimSuccessCheck:
    """Verify exact BrickSim offsets and yaw for requested connections."""

    def __init__(self, goals: tuple[BrickSimConnectionGoal, ...]):
        """Store at least one immutable requested connection."""
        if not goals:
            raise ValueError("at least one connection goal is required")
        self._goals = goals

    def is_satisfied(self) -> bool:
        """Return whether every exact requested connection is active.

        Returns:
            True only when all offsets and yaw values match.
        """
        return all(self._status(goal) == "matched" for goal in self._goals)

    def conflict(self) -> str | None:
        """Return the first wrong active connection.

        Returns:
            Conflict detail, or None when connections are absent or correct.
        """
        for goal in self._goals:
            status = self._status(goal)
            if status not in {"absent", "matched"}:
                return status
        return None

    def require_satisfied(self, stage: str) -> None:
        """Raise unless every exact requested connection is active."""
        conflict = self.conflict()
        if conflict is not None:
            raise ExecutionError(FailureCode.VERIFICATION_FAILED, stage, conflict)
        if not self.is_satisfied():
            raise ExecutionError(
                FailureCode.VERIFICATION_FAILED,
                stage,
                "BrickSim did not verify every requested connection",
            )

    def _status(self, goal: BrickSimConnectionGoal) -> str:
        from bricksim.core import lookup_physics_connection

        info = lookup_physics_connection(
            stud_path=goal.reference_path,
            stud_if=goal.stud_iface,
            hole_path=goal.target_path,
            hole_if=goal.hole_iface,
        )
        if info is None:
            return "absent"
        actual_offset = tuple(info.offset)
        actual_yaw = int(info.yaw)
        if actual_offset == goal.offset and actual_yaw == goal.yaw:
            return "matched"
        return (
            "wrong BrickSim connection: "
            f"reference={goal.reference_path} "
            f"actual={actual_offset}/{actual_yaw} "
            f"expected={goal.offset}/{goal.yaw}"
        )


class BrickSimRobotBackend:
    """Adapt one robot in the current BrickSim environment."""

    def __init__(self, env, arm_index: int):
        """Bind one environment arm without exposing it to skill code."""
        if arm_index < 0 or arm_index >= len(env.robot_pins):
            raise ValueError(f"invalid arm index {arm_index}")
        self._env = env
        self._arm_index = arm_index
        self._robot_pin = env.robot_pins[arm_index]
        self._robot_config = env.robot_configs[arm_index]
        self._joint_order = tuple(self._robot_config["Joint_Order"])
        self._arm_indices = self._indices(gripper=False)
        self._gripper_indices = self._indices(gripper=True)

    @property
    def robot_id(self) -> str:
        """Return the configured robot name."""
        return str(self._robot_config.get("Name", f"robot_{self._arm_index}"))

    @property
    def home_configuration(self) -> FloatArray:
        """Return a copy of the Pinocchio home configuration."""
        return np.asarray(self._robot_pin.home_q, dtype=np.float64).copy()

    @property
    def arm_configuration_indices(self) -> tuple[int, ...]:
        """Return Pinocchio configuration indices for arm joints."""
        return self._arm_indices

    @property
    def gripper_configuration_indices(self) -> tuple[int, ...]:
        """Return Pinocchio configuration indices for finger joints."""
        return self._gripper_indices

    def read_state(self) -> RobotState:
        """Map the global BrickSim observation into one robot state.

        Returns:
            Current configuration, TCP pose, and finger positions.
        """
        observation = np.asarray(
            self._env.get_observations()["joint_positions"], dtype=np.float64
        )
        start, _ = self._env.arm_joint_slices[self.robot_id]
        q = self.home_configuration
        for offset, joint_name in enumerate(self._joint_order):
            joint = self._joint(joint_name)
            q[joint.idx_q] = observation[start + offset]
        tcp = self._tcp_world(q)
        gripper = q[list(self._gripper_indices)].copy()
        return RobotState(q, tcp, gripper)

    def solve_ik(self, world_t_tcp: FloatArray, seed: FloatArray) -> FloatArray | None:
        """Solve and strictly verify one world-frame TCP target.

        Returns:
            Verified configuration, or None when unreachable.
        """
        arm_t_tcp = np.linalg.inv(self._robot_pin.BASE_T) @ world_t_tcp
        q, _ = self._robot_pin.IK(
            {self._robot_pin.ee_frames[0]: arm_t_tcp},
            seed,
            self._robot_pin.controllable_joints,
            ROT_WEIGHT=0.05,
        )
        solved = self._robot_pin.FK(q, [self._robot_pin.ee_frames[0]])[
            self._robot_pin.ee_frames[0]
        ]
        position_error = float(np.linalg.norm(solved[:3, 3] - arm_t_tcp[:3, 3]))
        rotation_error = float(
            Rotation.from_matrix(solved[:3, :3].T @ arm_t_tcp[:3, :3]).magnitude()
        )
        if (
            position_error > SAFE_IK_POSITION_TOLERANCE
            or rotation_error > SAFE_IK_ROTATION_TOLERANCE
        ):
            return None
        return np.asarray(q, dtype=np.float64)

    def forward_kinematics(self, q: FloatArray) -> FloatArray:
        """Return the world-frame TCP pose for one configuration.

        Returns:
            TCP transform computed by the configured Pinocchio model.
        """
        return self._tcp_world(np.asarray(q, dtype=np.float64))

    def command_configuration(self, q: FloatArray) -> None:
        """Write one configuration only to this robot's articulation."""
        command = np.empty(len(self._joint_order), dtype=np.float32)
        for offset, joint_name in enumerate(self._joint_order):
            joint = self._joint(joint_name)
            command[offset] = q[joint.idx_q]
        self._env.robot_apply_arm_action(self._arm_index, command)

    def configuration_is_safe(self, q: FloatArray) -> bool:
        """Check finiteness and configured Pinocchio position limits.

        Returns:
            Whether the configuration passes the available checks.
        """
        candidate = np.asarray(q, dtype=np.float64)
        if candidate.shape != self.home_configuration.shape:
            return False
        if not np.isfinite(candidate).all():
            return False
        lower = np.asarray(self._robot_pin.pin_model.lowerPositionLimit)
        upper = np.asarray(self._robot_pin.pin_model.upperPositionLimit)
        tolerance = 1e-6
        return bool(
            np.all(candidate >= lower - tolerance)
            and np.all(candidate <= upper + tolerance)
        )

    def with_gripper(
        self, q: FloatArray, object_width: float, closed: bool
    ) -> FloatArray:
        """Apply the Piper parallel-gripper target to a copied state.

        Returns:
            Copied configuration with updated finger joints.
        """
        if object_width <= 0.0:
            raise ValueError("object_width must be positive")
        result = np.asarray(q, dtype=np.float64).copy()
        opening = min(
            object_width * 0.5 + GRIPPER_OPEN_MARGIN_PER_FINGER,
            GRIPPER_MAX_JOINT_OPENING,
        )
        for joint_name in self._joint_order:
            if "gripper" not in joint_name:
                continue
            joint = self._joint(joint_name)
            if joint_name == "gripper_joint1":
                result[joint.idx_q] = 0.0 if closed else opening
            elif joint_name == "gripper_joint2":
                result[joint.idx_q] = 0.0 if closed else -opening
        return result

    def _indices(self, gripper: bool) -> tuple[int, ...]:
        return tuple(
            self._joint(name).idx_q
            for name in self._joint_order
            if ("gripper" in name) is gripper
        )

    def _joint(self, name: str):
        joint_id = self._robot_pin.pin_model.getJointId(name)
        return self._robot_pin.pin_model.joints[joint_id]

    def _tcp_world(self, q: FloatArray) -> FloatArray:
        ee = self._robot_pin.ee_frames[0]
        return self._robot_pin.BASE_T @ self._robot_pin.FK(q, [ee])[ee]


class BrickSimWorldModel:
    """Expose BrickSim stepping and object poses through the world protocol."""

    def __init__(self, env):
        """Bind the current BrickSim environment."""
        self._env = env

    def object_pose(self, object_id: str) -> FloatArray:
        """Read an opaque object identifier as a BrickSim prim path.

        Returns:
            Copied world transform for the requested object.
        """
        return np.asarray(
            self._env.get_prim_world_T(object_id), dtype=np.float64
        ).copy()

    async def advance(self, steps: int = 1) -> None:
        """Advance the simulator by a positive number of cycles."""
        if steps <= 0:
            raise ValueError("steps must be positive")
        for _ in range(steps):
            await self._env.step()
