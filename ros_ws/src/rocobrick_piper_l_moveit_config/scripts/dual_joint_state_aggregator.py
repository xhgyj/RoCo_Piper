#!/usr/bin/env python3
"""Aggregate prefixed joint states for the combined dual-arm MoveIt model."""

from __future__ import annotations

from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


@dataclass
class _State:
    """Latest state received from one robot topic."""

    message: JointState


class DualJointStateAggregator(Node):
    """Publish one complete prefixed joint state from two robot topics."""

    def __init__(self) -> None:
        super().__init__("dual_joint_state_aggregator")
        self._states: dict[str, _State] = {}
        robot_ids = self.declare_parameter("robot_ids", ["piper_0", "piper_1"]).value
        if not isinstance(robot_ids, list) or not all(
            isinstance(robot_id, str) and robot_id for robot_id in robot_ids
        ):
            raise ValueError("robot_ids must be a non-empty string array")
        for robot_id in robot_ids:
            self.create_subscription(
                JointState,
                f"/{robot_id}/joint_states",
                lambda message, current=robot_id: self._receive(current, message),
                10,
            )
        self._publisher = self.create_publisher(JointState, "/joint_states", 10)

    def _receive(self, robot_id: str, message: JointState) -> None:
        self._states[robot_id] = _State(message)
        if len(self._states) < 2:
            return
        names: list[str] = []
        positions: list[float] = []
        velocities: list[float] = []
        efforts: list[float] = []
        has_velocity = True
        has_effort = True
        latest = None
        for state in self._states.values():
            latest = state.message.header.stamp
            names.extend(str(name) for name in state.message.name)
            positions.extend(float(value) for value in state.message.position)
            if len(state.message.velocity) == len(state.message.name):
                velocities.extend(float(value) for value in state.message.velocity)
            else:
                has_velocity = False
            if len(state.message.effort) == len(state.message.name):
                efforts.extend(float(value) for value in state.message.effort)
            else:
                has_effort = False
        output = JointState()
        output.header.stamp = latest
        output.name = names
        output.position = positions
        output.velocity = velocities if has_velocity else []
        output.effort = efforts if has_effort else []
        self._publisher.publish(output)


def main() -> None:
    """Run the aggregator until ROS shutdown."""
    rclpy.init()
    node = DualJointStateAggregator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
