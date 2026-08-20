"""Unit tests for the hardware-shaped Mate-down skill interfaces."""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from rocobrick.policy.mate_down import (
    FRAME_STATE_DIM,
    CartesianActionAdapter,
    ExternalWrenchEstimator,
    MateDownConfig,
    MateObservation,
    MateObservationBuilder,
    MatePhase,
    MateSupervisor,
    ObservationHistory,
    ScriptedMateDownExpert,
)
from rocobrick.policy.mate_down_runtime import (
    NoisyTargetProvider,
    TargetNoiseConfig,
    _limit_pose_tracking_error,
    _safe_transport_action,
    load_runtime_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_observation_vector_and_history_are_robot_only():
    """Observation history has the declared shape and resets between episodes."""
    observation = MateObservation(
        tcp_position=np.array([1.0, 2.0, 3.0]),
        tcp_rotation_6d=np.arange(6),
        tcp_twist=np.arange(6),
        gripper_width=0.005,
        gripper_closed=True,
        external_wrench=np.arange(6),
    )
    vector = observation.vector()
    assert vector.shape == (FRAME_STATE_DIM,)
    history = ObservationHistory(3)
    initial = history.append(vector)
    assert initial.shape == (3 * FRAME_STATE_DIM,)
    np.testing.assert_array_equal(initial[:FRAME_STATE_DIM], vector)
    history.reset()
    changed = vector + 1
    reset = history.append(changed)
    np.testing.assert_array_equal(reset[:FRAME_STATE_DIM], changed)


def test_observation_builder_uses_estimated_skill_frame():
    """Builder expresses pose, twist, and wrench in the supplied target frame."""
    config = MateDownConfig(control_hz=10, history_steps=2)
    builder = MateObservationBuilder(config)
    world_t_skill = np.eye(4)
    world_t_skill[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    world_t_tcp = world_t_skill.copy()
    world_t_tcp[:3, 3] += world_t_skill[:3, 0] * 0.01
    observation, history = builder.build(
        world_t_skill,
        world_t_tcp,
        0.005,
        np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
    )
    np.testing.assert_allclose(observation.tcp_position, [0.01, 0.0, 0.0])
    np.testing.assert_allclose(
        observation.external_wrench[:3], [1.0, 0.0, 0.0], atol=1e-12
    )
    assert observation.gripper_closed
    assert history.shape == (2 * FRAME_STATE_DIM,)


def test_wrench_estimator_uses_torque_residual_and_bias():
    """Known identity Jacobian produces the expected compensated wrench."""
    estimator = ExternalWrenchEstimator(alpha=1.0)
    measured = np.arange(1.0, 7.0)
    model = np.ones(6)
    jacobian = np.eye(6)
    np.testing.assert_allclose(
        estimator.raw_wrench(measured, model, jacobian), measured - model
    )
    estimator.calibrate([np.ones(6)])
    np.testing.assert_allclose(
        estimator.update(measured, model, jacobian), measured - model - 1.0
    )


def test_cartesian_action_is_norm_limited_in_skill_frame():
    """Translation and rotation are clipped independently before composition."""
    config = MateDownConfig(max_translation_step=0.001, max_rotation_step=0.1)
    adapter = CartesianActionAdapter(config)
    action = np.array([0.003, 0.004, 0.0, 0.0, 0.0, 0.2])
    bounded, target = adapter.target(np.eye(4), np.eye(4), action)
    assert np.isclose(np.linalg.norm(bounded[:3]), 0.001)
    assert np.isclose(np.linalg.norm(bounded[3:]), 0.1)
    np.testing.assert_allclose(target[:3, 3], bounded[:3])


def test_teacher_starts_with_assembly_alignment_then_approach():
    """Teacher data begins at preplace and contains no transport phase."""
    teacher = ScriptedMateDownExpert()
    aligning_pose = np.eye(4)
    aligning_pose[:3, 3] = [0.002, 0.0, 0.06]
    aligning_pose[:3, :3] = Rotation.from_euler("z", 5, degrees=True).as_matrix()
    output = teacher.act(aligning_pose, np.zeros(6))
    assert output.phase == MatePhase.ALIGN
    assert np.linalg.norm(output.action[3:]) > 0.0
    assert output.action[2] < 0.0

    aligned_pose = np.eye(4)
    aligned_pose[2, 3] = -0.06
    output = teacher.act(aligned_pose, np.zeros(6))
    assert output.phase == MatePhase.APPROACH
    assert output.action[2] > 0.0
    assert np.isclose(output.action[2], 0.0015)

    output = teacher.act(
        aligned_pose, np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    )
    assert output.phase == MatePhase.APPROACH


def test_teacher_completes_only_after_privileged_connection():
    """Stable force alone cannot label a simulator demonstration successful."""
    teacher = ScriptedMateDownExpert(hold_steps=2)
    pose = np.eye(4)
    wrench = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    assert teacher.act(pose, wrench).phase == MatePhase.PRESS
    assert teacher.act(pose, wrench, connected=True).phase == MatePhase.HOLD
    assert not teacher.act(pose, wrench, connected=True).done
    assert teacher.act(pose, wrench, connected=True).done


def test_supervisor_uses_only_feedback_for_stop_conditions():
    """Stable force completes and excessive force fails without simulator truth."""
    config = MateDownConfig(
        stable_steps=2, max_force=10.0, safety_violation_steps=1
    )
    supervisor = MateSupervisor(config)
    observation = MateObservation(
        np.zeros(3),
        np.zeros(6),
        np.zeros(6),
        0.0,
        True,
        np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    )
    assert not supervisor.update(observation).done
    result = supervisor.update(observation)
    assert result.done and result.success_candidate
    unsafe = MateObservation(
        **{**observation.__dict__, "external_wrench": np.array([11.0, 0, 0, 0, 0, 0])}
    )
    result = MateSupervisor(config).update(unsafe)
    assert result.done and not result.success_candidate


def test_noisy_target_provider_is_reproducible_and_bounded():
    """Target provider returns estimator-like poses with deterministic seeds."""
    config = TargetNoiseConfig(max_delay_frames=0)
    first = NoisyTargetProvider(config, seed=3)
    second = NoisyTargetProvider(config, seed=3)
    first.reset()
    second.reset()
    pose_a = first.update(np.eye(4))
    pose_b = second.update(np.eye(4))
    np.testing.assert_allclose(pose_a, pose_b)
    assert abs(pose_a[0, 3]) <= config.xy_m
    assert abs(pose_a[1, 3]) <= config.xy_m
    assert abs(pose_a[2, 3]) <= config.z_m


def test_cartesian_servo_command_lead_is_bounded():
    """Accumulated targets cannot run arbitrarily far ahead of feedback."""
    commanded = np.eye(4)
    commanded[:3, 3] = [0.03, 0.04, 0.0]
    commanded[:3, :3] = Rotation.from_euler("z", 20, degrees=True).as_matrix()
    limited = _limit_pose_tracking_error(
        np.eye(4), commanded, max_translation=0.01, max_rotation=np.deg2rad(5)
    )
    assert np.isclose(np.linalg.norm(limited[:3, 3]), 0.01)
    angle = Rotation.from_matrix(limited[:3, :3]).magnitude()
    assert np.isclose(angle, np.deg2rad(5))


def test_conventional_transport_aligns_before_descending():
    """Free-space transport keeps height until lateral pose is aligned."""
    current = np.eye(4)
    target = np.eye(4)
    target[:3, 3] = [0.05, 0.0, -0.06]
    target[:3, :3] = Rotation.from_euler("z", 20, degrees=True).as_matrix()
    action = _safe_transport_action(current, target)
    np.testing.assert_allclose(action[:3], [0.05, 0.0, 0.0])
    assert np.isclose(np.linalg.norm(action[3:]), np.deg2rad(20))

    aligned = target.copy()
    aligned[2, 3] = 0.0
    action = _safe_transport_action(aligned, target)
    np.testing.assert_allclose(action[:3], [0.0, 0.0, -0.06])
    np.testing.assert_allclose(action[3:], np.zeros(3), atol=1e-12)


def test_checked_in_runtime_config_matches_state_contract():
    """Checked-in defaults preserve the agreed 10-frame, 230-value state."""
    runtime, _, act = load_runtime_config(ROOT / "config/mate_down_config.json")
    assert runtime.history_steps == 10
    assert runtime.history_steps * FRAME_STATE_DIM == 230
    assert act["chunk_size"] == 10
