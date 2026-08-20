"""BrickSim adapter for the hardware-shaped Mate-down interfaces."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from rocobrick.policy.mate_down import (
    CartesianActionAdapter,
    ExternalWrenchEstimator,
    MateDownConfig,
    MateObservation,
    MateObservationBuilder,
    MateSupervisor,
    SupervisorResult,
)


@dataclass(frozen=True)
class TargetNoiseConfig:
    """Distribution used to emulate an upstream pose estimator."""

    xy_m: float = 0.006
    z_m: float = 0.003
    yaw_deg: float = 8.0
    tilt_deg: float = 3.0
    max_delay_frames: int = 3


@dataclass(frozen=True)
class RuntimeFeedback:
    """Robot feedback needed by the local skill runtime."""

    q: np.ndarray
    dq: np.ndarray
    tcp_world: np.ndarray
    gripper_width: float
    wrench_world: np.ndarray


@dataclass(frozen=True)
class PreparedMateDown:
    """Privileged initialization result kept outside policy observations."""

    assembly_expert: object
    arm_index: int
    true_world_t_skill: np.ndarray


class MateDownInitializationError(RuntimeError):
    """Raised when Pick or safe-pose transport fails before collection."""


SAFE_TRANSPORT_TRANSLATION_STEP = 0.002
SAFE_TRANSPORT_ROTATION_STEP = np.deg2rad(2.0)
SAFE_TRANSPORT_LATERAL_TOLERANCE = 0.003
SAFE_TRANSPORT_ROTATION_TOLERANCE = np.deg2rad(1.0)


class NoisyTargetProvider:
    """Expose delayed noisy target poses instead of simulator truth."""

    def __init__(self, config: TargetNoiseConfig, seed: int | None = None):
        """Initialize a reproducible noisy target provider."""
        self.config = config
        self.rng = np.random.default_rng(seed)
        self._delay = 0
        self._noise = np.eye(4)
        self._poses: deque[np.ndarray] = deque()

    def reset(self) -> None:
        """Sample one estimator bias and delay for an episode."""
        translation = np.array(
            [
                self.rng.uniform(-self.config.xy_m, self.config.xy_m),
                self.rng.uniform(-self.config.xy_m, self.config.xy_m),
                self.rng.uniform(-self.config.z_m, self.config.z_m),
            ]
        )
        angles = np.deg2rad(
            [
                self.rng.uniform(-self.config.tilt_deg, self.config.tilt_deg),
                self.rng.uniform(-self.config.tilt_deg, self.config.tilt_deg),
                self.rng.uniform(-self.config.yaw_deg, self.config.yaw_deg),
            ]
        )
        self._noise = np.eye(4)
        self._noise[:3, :3] = Rotation.from_euler("xyz", angles).as_matrix()
        self._noise[:3, 3] = translation
        self._delay = int(self.rng.integers(0, self.config.max_delay_frames + 1))
        self._poses.clear()

    def update(self, true_world_t_skill: np.ndarray) -> np.ndarray:
        """Return a delayed pose with a fixed per-episode estimator bias."""
        estimated = np.asarray(true_world_t_skill, dtype=np.float64) @ self._noise
        self._poses.append(estimated.copy())
        while len(self._poses) > self._delay + 1:
            self._poses.popleft()
        return self._poses[0].copy()


class SimulatorMateDownRuntime:
    """Read robot feedback and execute safe local Cartesian actions."""

    def __init__(self, env, arm_index: int, config: MateDownConfig):
        """Bind one active arm and construct its skill adapters."""
        self.env = env
        self.arm_index = arm_index
        self.config = config
        self.robot = env.robots[arm_index]
        self.robot_pin = env.robot_pins[arm_index]
        self.robot_config = env.robot_configs[arm_index]
        self.observation_builder = MateObservationBuilder(config)
        self.wrench_estimator = ExternalWrenchEstimator(
            config.wrench_filter_alpha,
            config.jacobian_rcond,
            config.wrench_damping,
        )
        self.action_adapter = CartesianActionAdapter(config)
        self.supervisor = MateSupervisor(config)
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

    def observe(
        self, estimated_world_t_skill: np.ndarray
    ) -> tuple[RuntimeFeedback, MateObservation, np.ndarray, SupervisorResult]:
        """Return raw feedback, structured observation, history, and safety state."""
        feedback = self.read_feedback()
        observation, history = self.observation_builder.build(
            estimated_world_t_skill,
            feedback.tcp_world,
            feedback.gripper_width,
            feedback.wrench_world,
        )
        return feedback, observation, history, self.supervisor.update(observation)

    def apply_action(
        self,
        estimated_world_t_skill: np.ndarray,
        feedback: RuntimeFeedback,
        action: np.ndarray,
    ) -> np.ndarray:
        """Apply a bounded Cartesian action.

        Returns:
            Executed bounded action for dataset recording.
        """
        if self._commanded_tcp_world is None:
            self._commanded_tcp_world = feedback.tcp_world.copy()
        bounded, world_t_target = self.action_adapter.target(
            estimated_world_t_skill, self._commanded_tcp_world, action
        )
        world_t_target = _limit_pose_tracking_error(
            feedback.tcp_world,
            world_t_target,
            max_translation=0.005,
            max_rotation=np.deg2rad(2.0),
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
            free_space = (
                self.config.max_translation_step
                >= SAFE_TRANSPORT_TRANSLATION_STEP
            )
            position_limit = 0.006 if free_space else 0.004
            rotation_limit = np.deg2rad(3.0 if free_space else 2.0)
            if (
                position_error > position_limit
                or rotation_error > rotation_limit
            ):
                raise RuntimeError(
                    "Mate-down IK failed: "
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


def load_runtime_config(
    path: str | Path,
) -> tuple[MateDownConfig, TargetNoiseConfig, dict[str, object]]:
    """Load the checked-in skill configuration.

    Returns:
        Runtime, target-noise, and ACT configurations.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    runtime_values = {
        key: value
        for key, value in data.items()
        if key not in {"target_noise", "act", "max_rotation_step_deg"}
    }
    runtime_values["max_rotation_step"] = np.deg2rad(
        data["max_rotation_step_deg"]
    )
    return (
        MateDownConfig(**runtime_values),
        TargetNoiseConfig(**data["target_noise"]),
        data["act"],
    )


def true_skill_pose(
    true_world_t_skill: np.ndarray, world_t_tcp: np.ndarray
) -> np.ndarray:
    """Compute teacher-only TCP pose in the true skill frame.

    Returns:
        Homogeneous TCP transform in the true frame.
    """
    return np.linalg.inv(true_world_t_skill) @ world_t_tcp


def rotate_wrench_to_frame(
    world_t_frame: np.ndarray, wrench_world: np.ndarray
) -> np.ndarray:
    """Rotate a TCP wrench from world axes into frame axes.

    Returns:
        Wrench expressed in the requested frame axes.
    """
    rotation = np.asarray(world_t_frame)[:3, :3].T
    result = np.asarray(wrench_world, dtype=np.float64).copy()
    result[:3] = rotation @ result[:3]
    result[3:] = rotation @ result[3:]
    return result


async def prepare_mate_down(env) -> PreparedMateDown:
    """Run Pick and conventional transport to the safe preplace pose.

    Returns:
        Privileged initialization result kept outside policy input.
    """
    from rocobrick.policy.Policy import Policy

    expert = Policy(env, strict_demo=True)
    for _ in range(4000):
        if expert.is_done():
            result = expert.episode_result()
            raise MateDownInitializationError(
                f"assembly initializer failed: {result}"
            )
        observation = env.get_observations()
        action = expert.get_action(observation)
        # The lift has settled and _plan_place has computed the collision-free
        # preplace waypoint.  Stop advancing Policy before it enters the
        # mating descent; transport is executed below and is never recorded.
        if expert.state == "preplace":
            break
        env.robot_apply_action(action)
        await env.step()
        await env.step()
    else:
        raise MateDownInitializationError(
            "Pick initializer timed out before stable vertical lift"
        )

    if not transport_grasp_is_valid(env, expert):
        raise MateDownInitializationError(
            "brick slipped during grasp or vertical lift"
        )

    true_world_t_skill = expert.targets["place"].copy()
    await _transport_to_safe_preplace(env, expert, true_world_t_skill)
    actual_q = expert._split_actual(
        env.get_observations()["joint_positions"]
    )[expert.active_arm]
    position_error, rotation_error = expert._pose_errors(
        actual_q, expert.targets["preplace"]
    )
    if position_error > 0.008 or rotation_error > np.deg2rad(3.0):
        raise MateDownInitializationError(
            "safe preplace pose was not reached: "
            f"position={position_error:.4f} m, "
            f"rotation={np.rad2deg(rotation_error):.2f} deg"
        )
    print(
        "[mate-down] Pick complete; Assembly starts at safe preplace: "
        f"position residual={position_error:.4f} m, "
        f"rotation residual={np.rad2deg(rotation_error):.2f} deg"
    )

    return PreparedMateDown(expert, expert.active_arm, true_world_t_skill)


async def _transport_to_safe_preplace(
    env, expert, true_world_t_skill: np.ndarray
) -> None:
    """Move a grasped brick to preplace with a non-learned Cartesian servo."""
    config = MateDownConfig(
        max_translation_step=SAFE_TRANSPORT_TRANSLATION_STEP,
        max_rotation_step=SAFE_TRANSPORT_ROTATION_STEP,
    )
    runtime = SimulatorMateDownRuntime(env, expert.active_arm, config)
    runtime.reset()
    skill_t_safe = (
        np.linalg.inv(true_world_t_skill) @ expert.targets["preplace"]
    )
    for step in range(500):
        feedback = runtime.read_feedback()
        skill_t_tcp = np.linalg.inv(true_world_t_skill) @ feedback.tcp_world
        action = _safe_transport_action(skill_t_tcp, skill_t_safe)
        position_error = np.linalg.norm(
            skill_t_safe[:3, 3] - skill_t_tcp[:3, 3]
        )
        rotation_error = Rotation.from_matrix(
            skill_t_safe[:3, :3] @ skill_t_tcp[:3, :3].T
        ).magnitude()
        if position_error < 0.003 and rotation_error < np.deg2rad(1.0):
            print(
                "[mate-down] conventional safe transport complete in "
                f"{step} control frames"
            )
            return
        try:
            runtime.apply_action(true_world_t_skill, feedback, action)
        except RuntimeError as exc:
            raise MateDownInitializationError(
                f"safe-pose Cartesian transport failed: {exc}"
            ) from exc
        await env.step()
        await env.step()
        if not transport_grasp_is_valid(env, expert):
            raise MateDownInitializationError(
                "brick slipped during transport to the safe assembly pose"
            )
    raise MateDownInitializationError(
        "safe-pose Cartesian transport timed out"
    )


def _safe_transport_action(
    skill_t_tcp: np.ndarray, skill_t_safe: np.ndarray
) -> np.ndarray:
    """Return a staged free-space command that keeps descent collision-free."""
    current = np.asarray(skill_t_tcp, dtype=np.float64)
    target = np.asarray(skill_t_safe, dtype=np.float64)
    translation = target[:3, 3] - current[:3, 3]
    rotation = Rotation.from_matrix(
        target[:3, :3] @ current[:3, :3].T
    ).as_rotvec()
    if (
        np.linalg.norm(translation[:2])
        > SAFE_TRANSPORT_LATERAL_TOLERANCE
        or np.linalg.norm(rotation) > SAFE_TRANSPORT_ROTATION_TOLERANCE
    ):
        translation[2] = 0.0
    return np.concatenate((translation, rotation))


async def release_and_retreat(env, prepared: PreparedMateDown) -> None:
    """Run the conventional release and retreat used by final evaluation."""
    expert = prepared.assembly_expert
    await _drive_arm_joint_goal(
        env, prepared.arm_index, expert.waypoints["open_gripper"], 60
    )
    await _drive_arm_joint_goal(
        env, prepared.arm_index, expert.waypoints["retreat"], 600
    )


async def _drive_arm_joint_goal(
    env,
    arm_index: int,
    goal: np.ndarray,
    steps: int,
    arm_step: float = 0.006,
) -> None:
    robot_pin = env.robot_pins[arm_index]
    robot_config = env.robot_configs[arm_index]
    name = robot_config.get("Name", f"robot_{arm_index}")
    start, _ = env.arm_joint_slices[name]
    for _ in range(steps):
        global_q = np.asarray(
            env.get_observations()["joint_positions"], dtype=np.float32
        ).copy()
        deltas = np.empty(len(robot_config["Joint_Order"]), dtype=np.float64)
        limits = np.empty_like(deltas)
        for offset, joint_name in enumerate(robot_config["Joint_Order"]):
            joint_id = robot_pin.pin_model.getJointId(joint_name)
            joint = robot_pin.pin_model.joints[joint_id]
            current = float(global_q[start + offset])
            deltas[offset] = goal[joint.idx_q] - current
            limits[offset] = 0.001 if "gripper" in joint_name else arm_step
        scale = max(1.0, float(np.max(np.abs(deltas) / limits)))
        global_q[start : start + len(deltas)] += (deltas / scale).astype(
            np.float32
        )
        env.robot_apply_action(global_q)
        await env.step()
        await env.step()
        if np.max(np.abs(deltas)) < 1e-4:
            return


def transport_grasp_is_valid(env, expert) -> bool:
    """Check that the transported brick retained its grasp transform.

    Returns:
        Whether the brick remains close to its post-lift grasp transform.
    """
    observation = env.get_observations()
    actual_q = expert._split_actual(observation["joint_positions"])[
        expert.active_arm
    ]
    if expert.brick_to_tcp is None:
        return False
    task = expert.plan[expert.task_index]
    brick_world = env.get_prim_world_T(task["hole_path"])
    tcp_world = expert._tcp_world(actual_q)
    brick_to_tcp = np.linalg.inv(brick_world) @ tcp_world
    translation_error = np.linalg.norm(
        brick_to_tcp[:3, 3] - expert.brick_to_tcp[:3, 3]
    )
    rotation_error = np.linalg.norm(
        Rotation.from_matrix(
            brick_to_tcp[:3, :3].T @ expert.brick_to_tcp[:3, :3]
        ).as_rotvec()
    )
    lift = brick_world[2, 3] - expert.grasp_start_T[2, 3]
    tcp_distance = np.linalg.norm(brick_world[:3, 3] - tcp_world[:3, 3])
    valid = (
        tcp_distance < 0.16
        and translation_error < 0.025
        and rotation_error < np.deg2rad(20.0)
    )
    if not valid:
        print(
            "[mate-down] grasp check failed: "
            f"lift={lift:.4f} m, tcp_distance={tcp_distance:.4f} m, "
            f"relative_translation={translation_error:.4f} m, "
            f"relative_rotation={np.rad2deg(rotation_error):.2f} deg"
        )
    return valid


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
