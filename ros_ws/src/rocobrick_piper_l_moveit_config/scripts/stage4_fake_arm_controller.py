#!/usr/bin/env python3
"""Fake dual-arm FollowJointTrajectory servers for stage-4 acceptance tests.

This test-only process exposes the two robot-scoped arm controller action
channels and never publishes Isaac commands or touches a real controller. It
accepts a goal, waits for the requested trajectory duration using ROS time,
and returns SUCCESS; a cancel request returns CANCELED after the goal settles.
"""

from __future__ import annotations

import asyncio
import math
import os
import time

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node


class FakeArmController(Node):
    """Own one fake arm action endpoint with strict trajectory validation."""

    def __init__(self) -> None:
        """Create both namespaced arm action servers."""
        super().__init__("stage4_fake_arm_controller")
        self._servers = []
        self._callback_group = ReentrantCallbackGroup()
        self._cancel_requested: set[object] = set()
        self._delay_s = float(os.environ.get("ROCOBRICK_FAKE_CONTROLLER_DELAY_S", "0"))
        self._reject = os.environ.get("ROCOBRICK_FAKE_CONTROLLER_REJECT") == "1"
        for robot_id in ("piper_0", "piper_1"):
            endpoint = f"/{robot_id}/arm_controller/follow_joint_trajectory"
            self._servers.append(
                ActionServer(
                    self,
                    FollowJointTrajectory,
                    endpoint,
                    execute_callback=self._execute,
                    goal_callback=self._goal,
                    cancel_callback=self._cancel,
                    callback_group=self._callback_group,
                )
            )
            self.get_logger().info(f"fake arm controller ready: {endpoint}")

    def _goal(self, goal_request):
        """Accept only non-empty, finite arm trajectories."""
        if self._reject or not goal_request.trajectory.points:
            return GoalResponse.REJECT
        names = tuple(goal_request.trajectory.joint_names)
        if len(names) != 6 or len(names) != len(set(names)):
            return GoalResponse.REJECT
        if any(
            not math.isfinite(float(position))
            for point in goal_request.trajectory.points
            for position in point.positions
        ):
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel(self, cancel_request):
        """Accept cooperative cancellation."""
        return CancelResponse.ACCEPT

    def _execute(self, goal_handle):
        """Wait the requested duration, then publish a successful result."""
        trajectory = goal_handle.request.trajectory
        point = trajectory.points[-1]
        duration = float(point.time_from_start.sec) + point.time_from_start.nanosec * 1e-9
        duration += self._delay_s
        deadline = time.monotonic() + max(0.0, duration)
        while time.monotonic() < deadline:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return FollowJointTrajectory.Result()
            time.sleep(0.02)
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
        else:
            goal_handle.succeed()
        return FollowJointTrajectory.Result()


def main() -> None:
    """Spin fake action servers until SIGINT."""
    rclpy.init()
    node = FakeArmController()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
