"""Launch one unified dual-arm MoveIt scene with execution disabled."""

from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    config = (
        MoveItConfigsBuilder("rocobrick_dual_piper_l", package_name="rocobrick_moveit_config")
        .robot_description(file_path="config/dual_piper.urdf")
        .robot_description_semantic(file_path="config/dual_piper.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    parameters = [
        config.to_dict(),
        {
            "allow_trajectory_execution": False,
            "disable_capabilities": "move_group/MoveGroupExecuteTrajectoryAction",
            "publish_planning_scene": True,
            "publish_geometry_updates": True,
            "publish_state_updates": True,
            "publish_transforms_updates": True,
        },
    ]
    return LaunchDescription([
        Node(package="robot_state_publisher", executable="robot_state_publisher", parameters=[config.robot_description]),
        Node(package="moveit_ros_move_group", executable="move_group", output="screen", parameters=parameters),
        Node(package="rocobrick_planner", executable="planner_server", output="screen", parameters=[config.to_dict(), {"listen_address": "0.0.0.0:50051"}]),
    ])
