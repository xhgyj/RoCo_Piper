"""Launch the stage-4 fake arm FollowJointTrajectory endpoints only."""

from launch import LaunchDescription
from launch_ros.actions import Node


PACKAGE = "rocobrick_piper_l_moveit_config"


def generate_launch_description() -> LaunchDescription:
    """Return the isolated fake-controller launch description."""
    return LaunchDescription(
        [
            Node(
                package=PACKAGE,
                executable="stage4_fake_arm_controller.py",
                output="screen",
                parameters=[{"use_sim_time": True}],
            )
        ]
    )
