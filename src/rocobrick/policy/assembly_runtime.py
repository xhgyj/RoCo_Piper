"""BrickSim runtime for the safe-start GT assembly expert."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from rocobrick.policy.assembly_control import (
    AssemblyExpertConfig,
    AssemblyObservation,
    AssemblyObservationBuilder,
    AssemblySupervisor,
    CartesianActionAdapter,
    ExternalWrenchEstimator,
    SupervisorResult,
)


@dataclass(frozen=True)
class RuntimeFeedback:
    """Robot feedback needed by the local skill runtime."""

    q: np.ndarray
    dq: np.ndarray
    tcp_world: np.ndarray
    gripper_width: float
    wrench_world: np.ndarray


class GTAssemblyRuntime:
    """Read robot feedback and execute safe local Cartesian actions."""

    def __init__(self, env, arm_index: int, config: AssemblyExpertConfig):
        """Bind one active arm and construct its skill adapters."""
        self.env = env
        self.arm_index = arm_index
        self.config = config
        self.robot = env.robots[arm_index]
        self.robot_pin = env.robot_pins[arm_index]
        self.robot_config = env.robot_configs[arm_index]
        self.observation_builder = AssemblyObservationBuilder(config)
        self.wrench_estimator = ExternalWrenchEstimator(
            config.wrench_filter_alpha,
            config.jacobian_rcond,
            config.wrench_damping,
        )
        self.action_adapter = CartesianActionAdapter(config)
        self.supervisor = AssemblySupervisor(config)
        self._previous_dq: np.ndarray | None = None
        self._commanded_tcp_world: np.ndarray | None = None
        self._arm_joint_names = [
            name
            for name in self.robot_config["Joint_Order"]
            if "gripper" not in name
        ]

    def reset(self) -> None:
        """Reset all episode-local filters and counters."""
        self.observation_builder.reset()
        self.wrench_estimator.reset()
        self.supervisor.reset()
        self._previous_dq = None
        self._commanded_tcp_world = None

    def read_feedback(self) -> RuntimeFeedback:
        """Read articulation feedback and estimate external TCP wrench.

        Returns:
            Current kinematic and model-compensated force feedback.
        """
        self.env._ensure_dof_maps()
        dof_map = self.env._arm_dof_maps[self.arm_index]
        raw_q = _required(self.robot.get_joint_positions(), "joint positions")
        raw_dq = _required(self.robot.get_joint_velocities(), "joint velocities")
        raw_tau = _required(
            self.robot.get_measured_joint_efforts(), "measured joint efforts"
        )
        q = self.robot_pin.home_q.copy()
        dq = np.zeros(self.robot_pin.pin_model.nv, dtype=np.float64)
        measured = []
        velocity_indices = []
        for name in self.robot_config["Joint_Order"]:
            if name not in dof_map:
                continue
            joint = self.robot_pin.pin_model.joints[
                self.robot_pin.pin_model.getJointId(name)
            ]
            q[joint.idx_q] = raw_q[dof_map[name]]
            dq[joint.idx_v] = raw_dq[dof_map[name]]
            if name in self._arm_joint_names:
                measured.append(raw_tau[dof_map[name]])
                velocity_indices.append(joint.idx_v)
        if len(measured) != len(self._arm_joint_names):
            raise RuntimeError("arm joint effort feedback is incomplete")
        self._previous_dq = dq.copy()
        # Local mating is deliberately slow.  Isaac's finite-difference joint
        # acceleration is too noisy for inverse-dynamics subtraction and
        # creates false Cartesian force spikes, so use quasi-static gravity
        # compensation here.  A real torque driver can provide a filtered
        # full-dynamics model behind the same estimator interface.
        model_tau = pin.computeGeneralizedGravity(
            self.robot_pin.pin_model, self.robot_pin.pin_data, q
        )
        ee = self.robot_pin.ee_frames[0]
        ee_id = self.robot_pin.pin_model.getFrameId(ee)
        jacobian_full = pin.computeFrameJacobian(
            self.robot_pin.pin_model,
            self.robot_pin.pin_data,
            q,
            ee_id,
            pin.LOCAL_WORLD_ALIGNED,
        )
        jacobian = jacobian_full[:, velocity_indices]
        wrench_arm = self.wrench_estimator.update(
            np.asarray(measured), model_tau[velocity_indices], jacobian
        )
        wrench_world = wrench_arm.copy()
        base_rotation = self.robot_pin.BASE_T[:3, :3]
        wrench_world[:3] = base_rotation @ wrench_arm[:3]
        wrench_world[3:] = base_rotation @ wrench_arm[3:]
        tcp_world = self.robot_pin.BASE_T @ self.robot_pin.FK(q, [ee])[ee]
        gripper_width = self._gripper_width(q)
        return RuntimeFeedback(q, dq, tcp_world, gripper_width, wrench_world)

    async def calibrate_wrench(
        self, sample_count: int = 30, settle_count: int = 30
    ) -> None:
        """Calibrate model residuals while holding a contact-free pre-mate pose."""
        if sample_count <= 0 or settle_count < 0:
            raise ValueError(
                "sample_count must be positive and settle_count non-negative"
            )
        self.wrench_estimator.calibrate([np.zeros(6)])
        for _ in range(settle_count):
            self.read_feedback()
            await self.env.step()
            await self.env.step()
        self.wrench_estimator.reset()
        samples = []
        base_rotation = self.robot_pin.BASE_T[:3, :3]
        for _ in range(sample_count):
            feedback = self.read_feedback()
            wrench_arm = feedback.wrench_world.copy()
            wrench_arm[:3] = base_rotation.T @ wrench_arm[:3]
            wrench_arm[3:] = base_rotation.T @ wrench_arm[3:]
            samples.append(wrench_arm)
            await self.env.step()
            await self.env.step()
        self.wrench_estimator.calibrate(samples)
        self.observation_builder.reset()
        self.supervisor.reset()

    def observe_brick_goal(
        self,
        world_t_goal_brick: np.ndarray,
        world_t_current_brick: np.ndarray,
    ) -> tuple[
        RuntimeFeedback,
        AssemblyObservation,
        np.ndarray,
        SupervisorResult,
        np.ndarray,
    ]:
        """Observe feedback relative to the GT brick goal.

        Returns:
            Feedback, observation, history, safety state, and the TCP target
            recomputed from the current physical grasp transform.
        """
        feedback = self.read_feedback()
        brick_t_tcp = np.linalg.inv(world_t_current_brick) @ feedback.tcp_world
        world_t_goal_tcp = world_t_goal_brick @ brick_t_tcp
        observation, history = self.observation_builder.build(
            world_t_goal_tcp,
            feedback.tcp_world,
            feedback.gripper_width,
            feedback.wrench_world,
        )
        return (
            feedback,
            observation,
            history,
            self.supervisor.update(observation),
            world_t_goal_tcp,
        )

    def apply_action(
        self,
        estimated_world_t_skill: np.ndarray,
        feedback: RuntimeFeedback,
        action: np.ndarray,
        fast_alignment: bool = False,
    ) -> np.ndarray:
        """Apply a bounded Cartesian action.

        Returns:
            Executed bounded action for dataset recording.
        """
        if self._commanded_tcp_world is None:
            self._commanded_tcp_world = feedback.tcp_world.copy()
        bounded, world_t_target = self.action_adapter.target(
            estimated_world_t_skill,
            self._commanded_tcp_world,
            action,
            max_rotation_step=(
                self.config.max_alignment_rotation_step
                if fast_alignment
                else None
            ),
        )
        world_t_target = _limit_pose_tracking_error(
            feedback.tcp_world,
            world_t_target,
            max_translation=0.005,
            max_rotation=(
                self.config.max_alignment_rotation_lead
                if fast_alignment
                else self.config.max_rotation_step
            ),
        )
        arm_t_target = np.linalg.inv(self.robot_pin.BASE_T) @ world_t_target
        q_target, log = self.robot_pin.IK(
            {self.robot_pin.ee_frames[0]: arm_t_target},
            feedback.q,
            self._arm_joint_names,
            ROT_WEIGHT=0.05,
            STEP_SIZE=0.35,
            TOL=2e-3,
            MAX_ITERS=80,
            method="jacobian",
            random_restarts=0,
        )
        if not log["success"]:
            ee = self.robot_pin.ee_frames[0]
            solved = self.robot_pin.FK(q_target, [ee])[ee]
            position_error = np.linalg.norm(
                solved[:3, 3] - arm_t_target[:3, 3]
            )
            rotation_error = Rotation.from_matrix(
                solved[:3, :3].T @ arm_t_target[:3, :3]
            ).magnitude()
            position_limit = 0.004
            rotation_limit = np.deg2rad(2.0)
            if (
                position_error > position_limit
                or rotation_error > rotation_limit
            ):
                raise RuntimeError(
                    "GT assembly IK failed: "
                    f"position={position_error:.4f} m, "
                    f"rotation={np.rad2deg(rotation_error):.2f} deg"
                )
        self._commanded_tcp_world = world_t_target
        command = np.asarray(
            self.env.get_observations()["joint_positions"], dtype=np.float32
        ).copy()
        name = self.robot_config.get("Name", f"robot_{self.arm_index}")
        start, _ = self.env.arm_joint_slices[name]
        for offset, joint_name in enumerate(self.robot_config["Joint_Order"]):
            joint = self.robot_pin.pin_model.joints[
                self.robot_pin.pin_model.getJointId(joint_name)
            ]
            if "gripper" in joint_name:
                command[start + offset] = self.config.gripper_hold_position
            else:
                command[start + offset] = q_target[joint.idx_q]
        self.env.robot_apply_action(command)
        return bounded.astype(np.float32)

    def _gripper_width(self, q: np.ndarray) -> float:
        values = []
        for name in self.robot_config["Joint_Order"]:
            if "gripper" not in name:
                continue
            joint = self.robot_pin.pin_model.joints[
                self.robot_pin.pin_model.getJointId(name)
            ]
            values.append(abs(float(q[joint.idx_q])))
        return float(sum(values))


def _required(value, name: str) -> np.ndarray:
    if value is None:
        raise RuntimeError(f"missing {name}")
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise RuntimeError(f"{name} contains non-finite values")
    return result


def _limit_pose_tracking_error(
    actual: np.ndarray,
    commanded: np.ndarray,
    max_translation: float,
    max_rotation: float,
) -> np.ndarray:
    """Bound Cartesian command lead while retaining accumulated servo motion.

    Returns:
        Pose target constrained to the configured feedback tracking window.
    """
    result = np.asarray(commanded, dtype=np.float64).copy()
    actual_pose = np.asarray(actual, dtype=np.float64)
    translation = result[:3, 3] - actual_pose[:3, 3]
    translation_norm = np.linalg.norm(translation)
    if translation_norm > max_translation:
        result[:3, 3] = (
            actual_pose[:3, 3]
            + translation * (max_translation / translation_norm)
        )
    rotation = Rotation.from_matrix(
        result[:3, :3] @ actual_pose[:3, :3].T
    ).as_rotvec()
    rotation_norm = np.linalg.norm(rotation)
    if rotation_norm > max_rotation:
        rotation *= max_rotation / rotation_norm
        result[:3, :3] = (
            Rotation.from_rotvec(rotation).as_matrix()
            @ actual_pose[:3, :3]
        )
    return result
