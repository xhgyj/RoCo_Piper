"""Regression tests for expert IK solution validation."""

import sys
from types import SimpleNamespace
from types import ModuleType

import numpy as np

# Policy's unit tests do not need the native BrickSim/Omniverse runtime.  Stub
# only the imported symbols so this file also runs under ordinary pytest.
core = ModuleType("bricksim.core")
core.compute_connection_transform = lambda **kwargs: None
core.lookup_physics_connection = lambda **kwargs: None
ordering = ModuleType("bricksim.topology.ordering")
ordering.bfs_sort_connections = lambda topology: topology
sys.modules["bricksim.core"] = core
sys.modules["bricksim.topology.ordering"] = ordering

from rocobrick.policy.Policy import EpisodeResult, Policy


class _Robot:
    BASE_T = np.eye(4)
    ee_frames = ["grasp_tcp"]
    controllable_joints = [f"joint{i}" for i in range(1, 7)]

    def IK(self, targets, seed, moving_joints, ROT_WEIGHT):
        # A large log6 translation can coexist with an exact Cartesian
        # position because SE(3) logarithm coordinates couple rotation and
        # translation.
        return seed.copy(), {"final_error": np.array([0.026, 0, 0, 0, 0, 1.0])}

    def FK(self, q, frame_names):
        return {frame_names[0]: np.eye(4)}


def _policy():
    policy = Policy.__new__(Policy)
    policy.env = SimpleNamespace(robot_pins=[_Robot()])
    policy.active_arm = 0
    policy.state = "home"
    policy.settle_count = 0
    policy.state_frame_counts = {}
    policy.result = EpisodeResult()
    return policy


def test_solve_uses_fk_cartesian_distance_not_log6_translation():
    policy = _policy()
    q = policy._solve(np.eye(4), np.zeros(6), "plan_pregrasp")
    np.testing.assert_array_equal(q, np.zeros(6))
    assert not policy.result.done


def test_solve_reports_planning_stage_for_unreachable_target():
    policy = _policy()
    target = np.eye(4)
    target[0, 3] = 0.1
    policy._solve(target, np.zeros(6), "plan_pregrasp")
    assert policy.result.done
    assert policy.result.failure_stage == "plan_pregrasp"


def test_motion_settle_count_is_phase_specific():
    policy = _policy()
    assert not policy._stable(True)
    assert policy._stable(True)  # home requires only two stable frames

    policy.state = "grasp"
    policy.settle_count = 0
    for _ in range(4):
        assert not policy._stable(True)
    assert policy._stable(True)  # contact approach remains more conservative


def test_gripper_contact_and_release_conditions():
    previous = np.zeros(8)
    moving = previous.copy()
    moving[6:] = [0.001, -0.001]
    assert not Policy._gripper_stalled(moving, previous)
    assert Policy._gripper_stalled(moving, moving + 1e-5)

    goal = np.zeros(8)
    goal[6:] = [0.03, -0.03]
    current = np.zeros(8)
    current[6:] = [0.02, -0.02]
    assert not Policy._gripper_closed_enough(current, goal)
    current[6:] = [0.018, -0.018]
    assert Policy._gripper_closed_enough(current, goal)
    assert not Policy._gripper_command_reached(current, np.zeros(8))
    assert Policy._gripper_command_reached(np.zeros(8), np.zeros(8))

    current[6:] = [0.014, -0.014]
    assert not Policy._gripper_open_enough(current, goal)
    current[6:] = [0.015, -0.015]
    assert Policy._gripper_open_enough(current, goal)


def test_interpolation_uses_separate_arm_and_gripper_limits():
    current = np.zeros(8)
    goal = np.ones(8)
    command = Policy._interp(current, goal)
    assert np.max(np.abs(command[:6])) <= 0.012
    assert np.max(np.abs(command[6:])) <= 0.001
