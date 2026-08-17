"""Regression tests for expert IK solution validation."""

from types import SimpleNamespace

import numpy as np

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
