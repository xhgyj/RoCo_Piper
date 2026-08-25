"""Unit tests for the safe-start GT assembly control interfaces."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rocobrick.policy.assembly_control import (
    FRAME_STATE_DIM,
    AssemblyExpertConfig,
    AssemblyObservation,
    AssemblyObservationBuilder,
    AssemblyPhase,
    AssemblySupervisor,
    CartesianActionAdapter,
    ExternalWrenchEstimator,
    GTAssemblyExpert,
    ObservationHistory,
)
from rocobrick.policy.assembly_runtime import _limit_pose_tracking_error


def test_observation_vector_and_history_are_robot_only():
    """Observation history has the declared shape and resets between episodes."""
    observation = AssemblyObservation(
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
    config = AssemblyExpertConfig(control_hz=10, history_steps=2)
    builder = AssemblyObservationBuilder(config)
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
    config = AssemblyExpertConfig(max_translation_step=0.001, max_rotation_step=0.1)
    adapter = CartesianActionAdapter(config)
    action = np.array([0.003, 0.004, 0.0, 0.0, 0.0, 0.2])
    bounded, target = adapter.target(np.eye(4), np.eye(4), action)
    assert np.isclose(np.linalg.norm(bounded[:3]), 0.001)
    assert np.isclose(np.linalg.norm(bounded[3:]), 0.1)
    np.testing.assert_allclose(target[:3, 3], bounded[:3])


def test_cartesian_action_can_use_high_clearance_rotation_limit():
    """Only an explicit high-clearance call may exceed the local yaw limit."""
    config = AssemblyExpertConfig(
        max_rotation_step=np.deg2rad(2.0),
        max_alignment_rotation_step=np.deg2rad(4.0),
    )
    adapter = CartesianActionAdapter(config)
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.deg2rad(20.0)])
    regular = adapter.limit(action)
    accelerated = adapter.limit(
        action,
        max_rotation_step=config.max_alignment_rotation_step,
    )
    assert np.linalg.norm(regular[3:]) == pytest.approx(np.deg2rad(2.0))
    assert np.linalg.norm(accelerated[3:]) == pytest.approx(np.deg2rad(4.0))


def test_teacher_starts_with_assembly_alignment_then_approach():
    """Teacher data begins at preplace and contains no transport phase."""
    teacher = GTAssemblyExpert()
    aligning_pose = np.eye(4)
    aligning_pose[:3, 3] = [0.002, 0.0, 0.06]
    aligning_pose[:3, :3] = Rotation.from_euler("z", 5, degrees=True).as_matrix()
    output = teacher.act(aligning_pose, np.zeros(6))
    assert output.phase == AssemblyPhase.ALIGN
    assert np.linalg.norm(output.action[3:]) > 0.0
    assert output.action[2] == 0.0

    aligned_pose = np.eye(4)
    aligned_pose[2, 3] = -0.06
    output = teacher.act(aligned_pose, np.zeros(6))
    assert output.phase == AssemblyPhase.APPROACH
    assert output.action[2] > 0.0
    assert np.isclose(output.action[2], 0.0015)

    output = teacher.act(
        aligned_pose, np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    )
    assert output.phase == AssemblyPhase.APPROACH

    below_nominal = np.eye(4)
    below_nominal[2, 3] = 0.001
    output = teacher.act(below_nominal, np.zeros(6))
    assert output.phase == AssemblyPhase.FIRST_CONTACT
    assert output.action[2] > 0.0


def test_teacher_finishes_large_yaw_alignment_before_descent():
    """A 180-degree safe start rotates at clearance height before approach."""
    teacher = GTAssemblyExpert()
    pose = np.eye(4)
    pose[2, 3] = -0.06
    pose[:3, :3] = Rotation.from_euler("z", 180, degrees=True).as_matrix()
    output = teacher.act(pose, np.zeros(6))
    assert output.phase == AssemblyPhase.ALIGN
    assert output.action[2] == 0.0
    assert np.isclose(np.linalg.norm(output.action[3:]), np.pi * 0.7)


def test_teacher_resolves_pi_rotation_toward_reachable_ik_branch():
    """The endpoint IK hint selects one sign of the ambiguous pi rotation."""
    teacher = GTAssemblyExpert(initial_rotation_hint=np.array([0.0, 0.0, 1.0]))
    pose = np.eye(4)
    pose[2, 3] = -0.06
    pose[:3, :3] = Rotation.from_euler("z", 180, degrees=True).as_matrix()
    output = teacher.act(pose, np.zeros(6))
    assert output.phase == AssemblyPhase.ALIGN
    assert np.dot(output.action[3:], teacher.initial_rotation_hint) > 0.0


def test_teacher_uses_alignment_hysteresis_during_approach():
    """Minor post-alignment drift is corrected without phase chattering."""
    teacher = GTAssemblyExpert()
    aligned = np.eye(4)
    aligned[2, 3] = -0.06
    assert teacher.act(aligned, np.zeros(6)).phase == AssemblyPhase.APPROACH

    minor_drift = np.eye(4)
    minor_drift[:3, 3] = [0.0008, 0.0, -0.04]
    minor_drift[:3, :3] = Rotation.from_euler(
        "z", 1.5, degrees=True
    ).as_matrix()
    output = teacher.act(minor_drift, np.zeros(6))
    assert output.phase == AssemblyPhase.APPROACH
    assert output.action[0] < 0.0
    assert output.action[2] > 0.0
    assert np.linalg.norm(output.action[3:]) > 0.0

    excessive_drift = minor_drift.copy()
    excessive_drift[0, 3] = 0.002
    output = teacher.act(excessive_drift, np.zeros(6))
    assert output.phase == AssemblyPhase.ALIGN
    assert output.action[2] == 0.0


def test_teacher_completes_only_after_privileged_connection():
    """Stable force alone cannot label a simulator demonstration successful."""
    teacher = GTAssemblyExpert(hold_steps=2)
    pose = np.eye(4)
    wrench = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    assert teacher.act(pose, wrench).phase == AssemblyPhase.PRESS
    assert teacher.act(pose, wrench, connected=True).phase == AssemblyPhase.HOLD
    assert not teacher.act(pose, wrench, connected=True).done
    assert teacher.act(pose, wrench, connected=True).done


def test_teacher_latches_press_after_first_contact():
    """Contact deflection cannot send the expert back to alignment."""
    teacher = GTAssemblyExpert()
    pose = np.eye(4)
    contact = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    assert teacher.act(pose, contact).phase == AssemblyPhase.PRESS

    deflected = np.eye(4)
    deflected[:3, 3] = [0.001, 0.0, 0.001]
    output = teacher.act(deflected, np.zeros(6))
    assert output.phase == AssemblyPhase.PRESS
    assert output.action[2] > 0.0


def test_teacher_retracts_when_pressed_brick_loses_alignment():
    """Persistent press misalignment retreats to clearance before realignment."""
    teacher = GTAssemblyExpert()
    pose = np.eye(4)
    contact = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    assert teacher.act(pose, contact).phase == AssemblyPhase.PRESS

    jammed = np.eye(4)
    jammed[:3, :3] = Rotation.from_euler("x", 3, degrees=True).as_matrix()
    assert teacher.act(jammed, contact).phase == AssemblyPhase.PRESS
    assert teacher.act(jammed, contact).phase == AssemblyPhase.PRESS
    output = teacher.act(jammed, contact)
    assert output.phase == AssemblyPhase.ALIGN
    assert output.action[2] < 0.0

    # Even if lateral/yaw error disappears immediately, recovery must unload
    # the contact before the expert can descend again.
    near_contact = np.eye(4)
    near_contact[2, 3] = -0.004
    output = teacher.act(near_contact, np.zeros(6))
    assert output.phase == AssemblyPhase.ALIGN
    assert output.action[2] < 0.0

    clear = np.eye(4)
    clear[2, 3] = -teacher.blend_activation_distance
    output = teacher.act(clear, np.zeros(6))
    assert output.phase == AssemblyPhase.APPROACH
    assert output.action[2] > 0.0


def test_supervisor_uses_only_feedback_for_stop_conditions():
    """Stable force completes and excessive force fails without simulator truth."""
    config = AssemblyExpertConfig(
        stable_steps=2, max_force=10.0, safety_violation_steps=1
    )
    supervisor = AssemblySupervisor(config)
    observation = AssemblyObservation(
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
    unsafe = AssemblyObservation(
        **{**observation.__dict__, "external_wrench": np.array([11.0, 0, 0, 0, 0, 0])}
    )
    result = AssemblySupervisor(config).update(unsafe)
    assert result.done and not result.success_candidate


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
