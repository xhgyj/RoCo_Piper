"""Minimal move_group launch for the container-side Piper_L execution stack.

Adapted from the pinned vendor launches in
isaac/vendor/agx_arm_sim/Moveit2/piper_l_gripper_moveit_config/launch/
(move_group.launch.py / gazebo_moveit.launch.py), minus Gazebo and RViz:
the robot hardware is the host Isaac bridge, reached through
topic_based_ros2_control/TopicBasedSystem via /isaac_joint_command and
/isaac_joint_states.

use_sim_time defaults to true and expects the host bridge /clock
(isaac/p05_host_bridge.py); pass use_sim_time:=false for offline smokes.
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

PACKAGE_NAME = "rocobrick_piper_l_moveit_config"

# P1.B live evidence: with use_sim_time:=true the host Isaac bridge /clock
# advances sim time ~8x faster than wall time. MoveIt's execution-duration
# monitor computes its bound as scaling * trajectory_duration in SIM seconds
# while the arm_controller result propagation costs a fixed ~0.2 s of WALL
# time, so the default scaling of 1.1 races the JTC success notification and
# aborts with "Controller is taking too long to execute trajectory"
# (TIMED_OUT). Scale the bound so the monitor stays meaningful while
# tolerating the sim-rate compression.
EXECUTION_DURATION_SCALING = 5.0


def generate_launch_description():
    package_share = Path(get_package_share_directory(PACKAGE_NAME))

    use_sim_time = LaunchConfiguration("use_sim_time")

    moveit_config = (
        MoveItConfigsBuilder("piper_l", package_name=PACKAGE_NAME)
        .robot_description(file_path="urdf/piper_l.ros2_control.urdf.xacro")
        .robot_description_semantic(file_path="config/piper_l.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"])
        .to_moveit_configs()
    )

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {
                "use_sim_time": use_sim_time,
                "trajectory_execution": {
                    "allowed_execution_duration_scaling": EXECUTION_DURATION_SCALING,
                },
            },
        ],
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[
            moveit_config.robot_description,
            {"use_sim_time": use_sim_time},
        ],
    )

    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        output="screen",
        parameters=[
            moveit_config.robot_description,
            str(package_share / "config" / "ros2_controllers.yaml"),
            {"use_sim_time": use_sim_time},
        ],
    )

    controller_spawners = [
        Node(
            package="controller_manager",
            executable="spawner",
            output="screen",
            arguments=[
                controller,
                "--controller-manager",
                "/controller_manager",
                "--controller-manager-timeout",
                "60",
            ],
        )
        for controller in (
            "joint_state_broadcaster",
            "arm_controller",
            "gripper_controller",
        )
    ]

    start_controllers_after_control_node = RegisterEventHandler(
        OnProcessStart(
            target_action=ros2_control_node,
            on_start=controller_spawners,
        )
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                description=(
                    "Use the simulation clock from the host Isaac bridge (/clock);"
                    " set false for offline smokes without the bridge."
                ),
            ),
            ros2_control_node,
            start_controllers_after_control_node,
            robot_state_publisher_node,
            move_group_node,
        ]
    )
