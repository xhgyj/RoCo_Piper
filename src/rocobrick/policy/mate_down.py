"""Proprioceptive interfaces for the local downward mating skill.

The classes in this module deliberately do not import Isaac Sim or BrickSim.
They form the hardware-shaped boundary used by both simulation and a future
real-robot adapter: observations contain only robot feedback and actions are
bounded Cartesian increments.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from scipy.spatial.transform import Rotation

ACTION_DIM = 6
FRAME_STATE_DIM = 23


class MatePhase(IntEnum):
    """Teacher-only phase labels stored with demonstrations."""

    ALIGN = 0
    APPROACH = 1
    FIRST_CONTACT = 2
    PRESS = 3
    HOLD = 4
    COMPLETE = 5


class FailureType(IntEnum):
    """Terminal result labels for collection and evaluation."""

    NONE = 0
    TIMEOUT = 1
    EXCESSIVE_FORCE = 2
    IK_FAILED = 3
    DROPPED = 4
    INVALID_FEEDBACK = 5
    NOT_CONNECTED = 6


@dataclass(frozen=True)
class MateDownConfig:
    """Runtime limits shared by expert and learned policies."""

    control_hz: int = 30
    history_steps: int = 10
    max_translation_step: float = 0.0015
    max_rotation_step: float = np.deg2rad(2.0)
    max_force: float = 15.0
    max_torque: float = 1.5
    safety_activation_distance: float = 0.01
    safety_violation_steps: int = 3
    max_episode_steps: int = 150
    contact_force: float = 0.5
    stable_force_min: float = 0.8
    stable_force_max: float = 8.0
    stable_steps: int = 5
    gripper_closed_width: float = 0.01
    gripper_hold_position: float = 0.0
    wrench_filter_alpha: float = 0.05
    jacobian_rcond: float = 1e-5
    wrench_damping: float = 0.0


@dataclass(frozen=True)
class MateObservation:
    """One frame of hardware-available proprioceptive feedback."""

    tcp_position: np.ndarray
    tcp_rotation_6d: np.ndarray
    tcp_twist: np.ndarray
    gripper_width: float
    gripper_closed: bool
    external_wrench: np.ndarray

    def vector(self) -> np.ndarray:
        """Return the canonical 23-dimensional float32 representation."""
        values = np.concatenate(
            (
                _vector(self.tcp_position, 3, "tcp_position"),
                _vector(self.tcp_rotation_6d, 6, "tcp_rotation_6d"),
                _vector(self.tcp_twist, 6, "tcp_twist"),
                np.array(
                    [self.gripper_width, float(self.gripper_closed)],
                    dtype=np.float64,
                ),
                _vector(self.external_wrench, 6, "external_wrench"),
            )
        )
        if not np.isfinite(values).all():
            raise ValueError("mate observation contains non-finite values")
        return values.astype(np.float32)


class ObservationHistory:
    """Fixed-size, oldest-to-newest history with deterministic reset padding."""

    def __init__(self, steps: int, state_dim: int = FRAME_STATE_DIM):
        """Initialize an empty fixed-length history."""
        if steps <= 0:
            raise ValueError("history steps must be positive")
        self.steps = steps
        self.state_dim = state_dim
        self._frames: deque[np.ndarray] = deque(maxlen=steps)

    @property
    def output_dim(self) -> int:
        """Return the flattened history dimension."""
        return self.steps * self.state_dim

    def reset(self) -> None:
        """Remove all frames from the preceding episode."""
        self._frames.clear()

    def append(self, frame: np.ndarray) -> np.ndarray:
        """Append a frame.

        Returns:
            Padded flattened history.
        """
        value = _vector(frame, self.state_dim, "frame").astype(np.float32)
        if not np.isfinite(value).all():
            raise ValueError("history frame contains non-finite values")
        if not self._frames:
            for _ in range(self.steps):
                self._frames.append(value.copy())
        else:
            self._frames.append(value.copy())
        return np.concatenate(tuple(self._frames), dtype=np.float32)


class ExternalWrenchEstimator:
    """Estimate TCP wrench from model-compensated joint torque residuals."""

    def __init__(
        self, alpha: float = 0.2, rcond: float = 1e-5, damping: float = 0.0
    ):
        """Initialize filter strength and pseudoinverse tolerance."""
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.rcond = rcond
        self.damping = damping
        self._bias = np.zeros(6, dtype=np.float64)
        self._filtered: np.ndarray | None = None

    def reset(self) -> None:
        """Reset filter state while retaining the calibrated bias."""
        self._filtered = None

    def calibrate(self, samples: list[np.ndarray]) -> None:
        """Set the wrench zero from stationary, contact-free samples."""
        if not samples:
            raise ValueError("at least one calibration sample is required")
        values = np.stack([_vector(item, 6, "wrench sample") for item in samples])
        if not np.isfinite(values).all():
            raise ValueError("wrench calibration contains non-finite values")
        self._bias = np.median(values, axis=0)
        self.reset()

    def raw_wrench(
        self,
        measured_joint_torque: np.ndarray,
        model_joint_torque: np.ndarray,
        jacobian: np.ndarray,
    ) -> np.ndarray:
        """Compute an unfiltered wrench in the Jacobian reference frame.

        Returns:
            Six-dimensional unfiltered wrench.
        """
        measured = np.asarray(measured_joint_torque, dtype=np.float64)
        model = np.asarray(model_joint_torque, dtype=np.float64)
        jac = np.asarray(jacobian, dtype=np.float64)
        if measured.ndim != 1 or model.shape != measured.shape:
            raise ValueError("measured and model joint torques must be equal vectors")
        if jac.shape != (6, measured.size):
            raise ValueError(
                f"jacobian must have shape {(6, measured.size)}, got {jac.shape}"
            )
        if not all(np.isfinite(item).all() for item in (measured, model, jac)):
            raise ValueError("wrench estimator input contains non-finite values")
        torque_residual = measured - model
        if self.damping > 0.0:
            regularized = jac @ jac.T + self.damping**2 * np.eye(6)
            return np.linalg.solve(regularized, jac @ torque_residual)
        return np.linalg.pinv(jac.T, rcond=self.rcond) @ torque_residual

    def update(
        self,
        measured_joint_torque: np.ndarray,
        model_joint_torque: np.ndarray,
        jacobian: np.ndarray,
    ) -> np.ndarray:
        """Return the bias-corrected, low-pass-filtered wrench."""
        value = self.raw_wrench(measured_joint_torque, model_joint_torque, jacobian)
        value = value - self._bias
        if self._filtered is None:
            self._filtered = value
        else:
            self._filtered = self.alpha * value + (1.0 - self.alpha) * self._filtered
        if not np.isfinite(self._filtered).all():
            raise ValueError("estimated wrench is non-finite")
        return self._filtered.copy()


class MateObservationBuilder:
    """Build target-frame proprioception without exposing target truth."""

    def __init__(self, config: MateDownConfig):
        """Initialize episode-local history and finite differences."""
        self.config = config
        self.history = ObservationHistory(config.history_steps)
        self._previous_position: np.ndarray | None = None
        self._previous_rotation: Rotation | None = None

    def reset(self) -> None:
        """Reset finite differences and episode history."""
        self.history.reset()
        self._previous_position = None
        self._previous_rotation = None

    def build(
        self,
        estimated_world_t_skill: np.ndarray,
        world_t_tcp: np.ndarray,
        gripper_width: float,
        wrench_world: np.ndarray,
    ) -> tuple[MateObservation, np.ndarray]:
        """Return the current structured observation and flattened history."""
        world_t_skill = _transform(estimated_world_t_skill, "estimated_world_t_skill")
        tcp_world = _transform(world_t_tcp, "world_t_tcp")
        skill_t_tcp = np.linalg.inv(world_t_skill) @ tcp_world
        position = skill_t_tcp[:3, 3]
        rotation = Rotation.from_matrix(skill_t_tcp[:3, :3])
        if self._previous_position is None or self._previous_rotation is None:
            twist = np.zeros(6, dtype=np.float64)
        else:
            linear = (position - self._previous_position) * self.config.control_hz
            angular = (
                rotation * self._previous_rotation.inv()
            ).as_rotvec() * self.config.control_hz
            twist = np.concatenate((linear, angular))
        self._previous_position = position.copy()
        self._previous_rotation = rotation
        rotation_6d = skill_t_tcp[:3, :2].T.reshape(6)
        wrench = _vector(wrench_world, 6, "wrench_world").copy()
        wrench[:3] = world_t_skill[:3, :3].T @ wrench[:3]
        wrench[3:] = world_t_skill[:3, :3].T @ wrench[3:]
        observation = MateObservation(
            tcp_position=position,
            tcp_rotation_6d=rotation_6d,
            tcp_twist=twist,
            gripper_width=float(gripper_width),
            gripper_closed=gripper_width <= self.config.gripper_closed_width,
            external_wrench=wrench,
        )
        return observation, self.history.append(observation.vector())


class CartesianActionAdapter:
    """Limit a policy action and turn it into a world-frame TCP target."""

    def __init__(self, config: MateDownConfig):
        """Initialize the configured action limits."""
        self.config = config

    def limit(self, action: np.ndarray) -> np.ndarray:
        """Limit translational and rotational vector norms independently.

        Returns:
            Bounded six-dimensional action.
        """
        value = _vector(action, ACTION_DIM, "action").copy()
        if not np.isfinite(value).all():
            raise ValueError("action contains non-finite values")
        value[:3] = _clip_norm(value[:3], self.config.max_translation_step)
        value[3:] = _clip_norm(value[3:], self.config.max_rotation_step)
        return value

    def target(
        self,
        estimated_world_t_skill: np.ndarray,
        world_t_tcp: np.ndarray,
        action: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the bounded action and resulting world-frame TCP target."""
        world_t_skill = _transform(estimated_world_t_skill, "estimated_world_t_skill")
        tcp_world = _transform(world_t_tcp, "world_t_tcp")
        bounded = self.limit(action)
        skill_t_tcp = np.linalg.inv(world_t_skill) @ tcp_world
        skill_t_target = skill_t_tcp.copy()
        skill_t_target[:3, 3] += bounded[:3]
        skill_t_target[:3, :3] = (
            Rotation.from_rotvec(bounded[3:]).as_matrix() @ skill_t_tcp[:3, :3]
        )
        return bounded, world_t_skill @ skill_t_target


@dataclass(frozen=True)
class SupervisorResult:
    """Runtime-only termination decision derived from robot feedback."""

    done: bool
    success_candidate: bool
    failure: FailureType


class MateSupervisor:
    """Apply time and wrench safety limits without simulator truth."""

    def __init__(self, config: MateDownConfig):
        """Initialize feedback-only termination counters."""
        self.config = config
        self.steps = 0
        self.stable_steps = 0
        self.violation_steps = 0

    def reset(self) -> None:
        """Reset episode counters."""
        self.steps = 0
        self.stable_steps = 0
        self.violation_steps = 0

    def update(self, observation: MateObservation) -> SupervisorResult:
        """Return termination state using only the current observation."""
        self.steps += 1
        wrench = observation.external_wrench
        contact_region = (
            observation.tcp_position[2]
            <= self.config.safety_activation_distance
        )
        excessive_wrench = (
            np.linalg.norm(wrench[:3]) > self.config.max_force
            or np.linalg.norm(wrench[3:]) > self.config.max_torque
        )
        self.violation_steps = (
            self.violation_steps + 1
            if contact_region and excessive_wrench
            else 0
        )
        if self.violation_steps >= self.config.safety_violation_steps:
            return SupervisorResult(True, False, FailureType.EXCESSIVE_FORCE)
        normal_force = abs(float(wrench[2]))
        stable = (
            self.config.stable_force_min
            <= normal_force
            <= self.config.stable_force_max
            and np.linalg.norm(observation.tcp_twist[:3]) < 0.003
        )
        self.stable_steps = self.stable_steps + 1 if stable else 0
        if self.stable_steps >= self.config.stable_steps:
            return SupervisorResult(True, True, FailureType.NONE)
        if self.steps >= self.config.max_episode_steps:
            return SupervisorResult(True, False, FailureType.TIMEOUT)
        return SupervisorResult(False, False, FailureType.NONE)


@dataclass(frozen=True)
class ExpertOutput:
    """One teacher command with labels excluded from policy input."""

    action: np.ndarray
    phase: MatePhase
    done: bool


class ScriptedMateDownExpert:
    """Privileged assembly teacher starting from a safe preplace pose."""

    def __init__(
        self,
        hold_steps: int = 6,
        stable_force_min: float = 0.8,
        stable_force_max: float = 8.0,
        lateral_tolerance: float = 0.0005,
        rotation_tolerance: float = np.deg2rad(1.0),
        contact_activation_distance: float = 0.005,
        blend_activation_distance: float = 0.01,
    ):
        """Initialize press and hold counters."""
        self.hold_steps = hold_steps
        self.stable_force_min = stable_force_min
        self.stable_force_max = stable_force_max
        self.lateral_tolerance = lateral_tolerance
        self.rotation_tolerance = rotation_tolerance
        self.contact_activation_distance = contact_activation_distance
        self.blend_activation_distance = blend_activation_distance
        self._press_steps = 0
        self._hold_count = 0

    def reset(self) -> None:
        """Reset teacher phase counters."""
        self._press_steps = 0
        self._hold_count = 0

    def act(
        self,
        true_skill_t_tcp: np.ndarray,
        wrench_skill: np.ndarray,
        connected: bool = False,
    ) -> ExpertOutput:
        """Generate an unclipped action.

        Returns:
            Teacher action and teacher-only phase labels.
        """
        pose = _transform(true_skill_t_tcp, "true_skill_t_tcp")
        wrench = _vector(wrench_skill, 6, "wrench_skill")
        position_error = -pose[:3, 3]
        rotation_error = Rotation.from_matrix(pose[:3, :3]).inv().as_rotvec()
        lateral_error = np.linalg.norm(position_error[:2])
        rotation_norm = np.linalg.norm(rotation_error)
        near_contact = abs(float(pose[2, 3])) <= self.contact_activation_distance
        in_contact = near_contact and abs(float(wrench[2])) >= 0.5

        if self._hold_count:
            self._hold_count += 1
            done = self._hold_count > self.hold_steps
            phase = MatePhase.COMPLETE if done else MatePhase.HOLD
            return ExpertOutput(np.zeros(6), phase, done)

        if connected:
            self._hold_count = 1
            return ExpertOutput(np.zeros(6), MatePhase.HOLD, False)

        needs_alignment = (
            lateral_error > self.lateral_tolerance
            or rotation_norm > self.rotation_tolerance
        )
        axial_error = float(position_error[2])
        if needs_alignment:
            action = np.concatenate((position_error, rotation_error))
            # Far above contact, blend alignment with descent instead of
            # producing a visible stop between two state-machine phases.
            # Inside the final 10 mm, hold height until alignment is precise.
            action[:2] *= 0.7
            action[3:] *= 0.7
            if abs(axial_error) > self.blend_activation_distance:
                action[2] = np.copysign(0.0015, axial_error)
            else:
                action[2] = 0.0
            return ExpertOutput(action, MatePhase.ALIGN, False)

        if not in_contact and abs(axial_error) > 0.0005:
            if abs(axial_error) > 0.01:
                step_size = 0.0015
            elif abs(axial_error) > 0.003:
                step_size = 0.0005
            else:
                step_size = 0.0001
            normal_step = np.copysign(step_size, axial_error)
            action = np.array([0.0, 0.0, normal_step, 0.0, 0.0, 0.0])
            return ExpertOutput(action, MatePhase.APPROACH, False)

        if not in_contact:
            # The desired mating TCP has its +z axis along the downward tool
            # direction, so continue gently through the nominal pose.
            action = np.array([0.0, 0.0, 0.0001, 0.0, 0.0, 0.0])
            return ExpertOutput(action, MatePhase.FIRST_CONTACT, False)

        normal_force = abs(float(wrench[2]))
        if normal_force > self.stable_force_max:
            action = np.array([0.0, 0.0, -0.0001, 0.0, 0.0, 0.0])
            return ExpertOutput(action, MatePhase.PRESS, False)
        self._press_steps += 1
        action = np.array([0.0, 0.0, 0.00002, 0.0, 0.0, 0.0])
        return ExpertOutput(action, MatePhase.PRESS, False)


def _clip_norm(value: np.ndarray, limit: float) -> np.ndarray:
    norm = np.linalg.norm(value)
    if norm <= limit or norm < 1e-12:
        return value
    return value * (limit / norm)


def _transform(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def _vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape {(size,)}, got {result.shape}")
    return result
