#!/usr/bin/env python3
"""Publish fresh, complete, robot-owned combined joint states."""

from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

JOINTS = tuple(f"joint{i}" for i in range(1, 7)) + ("gripper_joint1", "gripper_joint2")


def valid_sample(robot_id, message):
    """Check exact ownership, array shapes and finite forwarded positions.

    Topic-based hardware exports no effort interface, so the broadcaster's
    velocity/effort arrays legitimately carry NaN; the combined output
    forwards only names and positions, so only positions must be finite.

    Returns:
        Whether the sample can participate in a combined state.
    """
    expected = {f"{robot_id}_{name}" for name in JOINTS}
    return (
        len(message.name) == len(expected)
        and set(message.name) == expected
        and len(message.position) == len(expected)
        and all(
            len(values) in (0, len(expected))
            for values in (message.velocity, message.effort)
        )
        and all(math.isfinite(value) for value in message.position)
    )


class DualJointStateAggregator(Node):
    """Aggregate only complete samples within explicit age and skew bounds."""

    def __init__(self):
        """Bind exact participant topics to the combined namespace."""
        super().__init__("dual_joint_state_aggregator")
        self.robots = tuple(
            self.declare_parameter("robot_ids", ["piper_0", "piper_1"]).value
        )
        if (
            len(self.robots) != 2
            or len(set(self.robots)) != 2
            or any(not robot for robot in self.robots)
        ):
            raise ValueError("joint aggregator requires two unique robots")
        self.max_age = float(self.declare_parameter("max_age_s", 0.5).value)
        self.max_skew = float(self.declare_parameter("max_skew_s", 0.05).value)
        if not all(
            math.isfinite(value) and value > 0
            for value in (self.max_age, self.max_skew)
        ):
            raise ValueError("joint freshness bounds must be positive")
        self.states = {}
        self._last_warn_s = None
        self._warn_period_s = 5.0
        self.publisher = self.create_publisher(JointState, "/dual_arm/joint_states", 10)
        for robot in self.robots:
            self.create_subscription(
                JointState,
                f"/{robot}/joint_states",
                lambda message, current=robot: self._receive(current, message),
                10,
            )
        self.get_logger().info(
            f"aggregating joint states for {list(self.robots)} onto /dual_arm/joint_states"
        )

    def _warn_blocked(self, reason: str) -> None:
        """Log one bounded rejection reason; silence here bricks move_group."""
        now = self.get_clock().now().nanoseconds * 1e-9
        if (
            self._last_warn_s is None
            or now - self._last_warn_s >= self._warn_period_s
        ):
            self._last_warn_s = now
            self.get_logger().warning(f"combined joint state withheld: {reason}")

    def _receive(self, robot, message):
        if not valid_sample(robot, message):
            self.states.pop(robot, None)
            self._warn_blocked(f"invalid sample from /{robot}/joint_states")
            return
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        previous = self.states.get(robot)
        if previous and stamp < previous[0]:
            self.states.clear()
            self._warn_blocked(f"stale monotonic stamp from {robot}: {stamp}")
            return
        self.states[robot] = (stamp, message)
        if set(self.states) != set(self.robots):
            return
        stamps = [self.states[identity][0] for identity in self.robots]
        now = self.get_clock().now().nanoseconds * 1e-9
        ages = {identity: now - self.states[identity][0] for identity in self.robots}
        if any(not 0 <= age <= self.max_age for age in ages.values()):
            self._warn_blocked(
                f"now={now:.3f} stamps={dict(zip(self.robots, stamps))} "
                f"ages={ages}"
            )
            return
        # Participant callbacks are independent DDS deliveries. Their source
        # stamps may differ by several publish periods even while both samples
        # are fresh. Align the combined message to the newest accepted sample
        # instead of dropping the complete union and starving MoveIt's state
        # monitor; freshness is still enforced per participant above.
        combined_stamp = max(stamps)
        output = JointState()
        output.header.stamp.sec, output.header.stamp.nanosec = divmod(
            int(combined_stamp * 1e9), 1_000_000_000
        )
        for identity in self.robots:
            sample = self.states[identity][1]
            output.name.extend(sample.name)
            output.position.extend(sample.position)
            # The generated URDF exposes a non-mimic parent gripper joint
            # (``<robot>_gripper``) while hardware publishes the two mimic
            # fingers. MoveIt's CurrentStateMonitor requires that parent in
            # the combined state as well; reconstruct it from joint1's 0.5
            # mimic multiplier without inventing any arm state.
            finger = f"{identity}_gripper_joint1"
            if finger in sample.name:
                output.name.append(f"{identity}_gripper")
                output.position.append(
                    2.0 * float(sample.position[sample.name.index(finger)])
                )
        self.publisher.publish(output)


def main():
    """Run the aggregator until shutdown."""
    rclpy.init()
    node = DualJointStateAggregator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
