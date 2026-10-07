"""Launch one combined dual-arm MoveIt planning instance.

This launch is intentionally separate from ``dual_move_group.launch.py``:
the latter remains the legacy isolated per-robot diagnostic stack, while this
entry owns one combined robot_description/SRDF/planning scene for coordinated
free-space planning. Controller execution is not silently claimed atomic;
the runtime coordinator must validate and dispatch the returned per-robot
segments explicitly.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

PACKAGE_NAME = "rocobrick_piper_l_moveit_config"
HELPER_FILE = "dual_robot_config.py"
COMBINED_HELPER = "dual_arm_config.py"


def _boolean(value: str) -> bool:
    """Parse one launch boolean."""
    return value.strip().lower() in ("1", "true", "yes", "on")


def _load_module(path: Path, name: str):
    """Load one local launch helper without installing it as a package."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load launch helper {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _moveit_config(paths):
    """Build MoveIt parameters from one combined model/config directory."""
    return (
        MoveItConfigsBuilder("rocobrick_dual_arm", package_name=PACKAGE_NAME)
        .robot_description(file_path=str(paths.urdf))
        .robot_description_semantic(file_path=str(paths.srdf))
        .robot_description_kinematics(file_path=str(paths.kinematics))
        .joint_limits(file_path=str(paths.joint_limits))
        .trajectory_execution(file_path=str(paths.moveit_controllers))
        .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"])
        .to_moveit_configs()
    )


def _launch_setup(context):
    """Generate the combined model and launch one MoveIt instance."""
    package_dir = Path(get_package_share_directory(PACKAGE_NAME))
    output = LaunchConfiguration("config_output_dir").perform(context).strip()
    output_dir = (
        Path(output) if output else Path(tempfile.mkdtemp(prefix="rocobrick_dual_arm_"))
    )
    robot_ids = ("piper_0", "piper_1")
    single = _load_module(package_dir / "launch" / HELPER_FILE, "dual_robot_config")
    combined = _load_module(package_dir / "launch" / COMBINED_HELPER, "dual_arm_config")
    configs = single.generate_dual_robot_configs(
        robot_ids=robot_ids,
        output_dir=output_dir / "robots",
        package_dir=package_dir,
    )
    config_path = Path(LaunchConfiguration("runtime_config_path").perform(context))
    config_sha256 = LaunchConfiguration("runtime_config_sha256").perform(context)
    base_poses = combined.load_base_poses(config_path, config_sha256, robot_ids)
    paths = combined.generate_dual_arm_configs(configs, output_dir, base_poses)
    moveit_config = _moveit_config(paths)
    use_sim_time = _boolean(LaunchConfiguration("use_sim_time").perform(context))
    robot_description = moveit_config.robot_description
    state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        namespace="/dual_arm",
        output="screen",
        remappings=[("joint_states", "/dual_arm/joint_states")],
        parameters=[robot_description, {"use_sim_time": use_sim_time}],
    )
    joint_aggregator = Node(
        package=PACKAGE_NAME,
        executable="dual_joint_state_aggregator.py",
        namespace="/dual_arm",
        output="screen",
        parameters=[{"robot_ids": list(robot_ids), "use_sim_time": use_sim_time}],
    )
    node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        namespace="/dual_arm",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {
                "use_sim_time": use_sim_time,
                "constraint_samplers": "rocobrick/CleanReferenceSamplerAllocator",
                "allow_trajectory_execution": _boolean(
                    LaunchConfiguration("allow_trajectory_execution").perform(context)
                ),
                "trajectory_execution": {
                    "allowed_execution_duration_scaling": 1.2,
                },
            },
        ],
    )
    controllers = []
    for index, config in enumerate(configs):
        namespace = config.endpoints.namespace
        control_node = Node(
            package="controller_manager",
            executable="ros2_control_node",
            namespace=namespace,
            output="screen",
            parameters=[
                {"robot_description": config.urdf.read_text(encoding="utf-8")},
                str(config.ros2_controllers),
                {"use_sim_time": use_sim_time},
            ],
        )
        spawners = [
            Node(
                package="controller_manager",
                executable="spawner",
                namespace=namespace,
                output="screen",
                arguments=[
                    controller,
                    "--controller-manager",
                    f"{namespace}/controller_manager",
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
        controllers.extend(
            [
                (control_node if index == 0 else TimerAction(period=3.0, actions=[control_node])),
                RegisterEventHandler(
                    OnProcessStart(target_action=control_node, on_start=spawners)
                ),
            ]
        )
    return [
        LogInfo(msg=f"[dual-arm] generated combined config at {paths.directory}"),
        *controllers,
        state_publisher,
        joint_aggregator,
        node,
    ]


def generate_launch_description() -> LaunchDescription:
    """Declare combined MoveIt launch arguments."""
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("runtime_config_path"),
            DeclareLaunchArgument("runtime_config_sha256"),
            DeclareLaunchArgument("allow_trajectory_execution", default_value="false"),
            DeclareLaunchArgument("config_output_dir", default_value=""),
            OpaqueFunction(function=_launch_setup),
        ]
    )
