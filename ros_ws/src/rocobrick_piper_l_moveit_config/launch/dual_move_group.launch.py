r"""Dual-robot MoveIt execution launch: one container, two isolated stacks.

Starts, for every id in ``robot_ids``, an independent namespaced stack in the
same ROS/DDS session (same container, same ``ROS_DOMAIN_ID``):

* ``/<robot_id>/robot_state_publisher`` publishing the generated
  ``<robot_id>_``-prefixed model on robot-scoped ``/<robot_id>/tf`` and
  ``/<robot_id>/tf_static`` topics;
* ``/<robot_id>/controller_manager`` (``ros2_control_node``) with the
  ``TopicBasedSystem`` hardware on the frozen host-bridge topics
  ``/<robot_id>/isaac_joint_command`` and ``/<robot_id>/isaac_joint_states``,
  plus ``joint_state_broadcaster``/``arm_controller``/``gripper_controller``;
* ``/<robot_id>/move_group`` with the generated SRDF/kinematics/joint-limits
  and controller configuration (planning group ``<robot_id>_arm_tcp``,
  calibrated tip ``<robot_id>_rocobrick_tcp``).

All derived names live under the robot namespace or carry the robot prefix,
so two stacks never share a topic, action, service, joint, frame, or group.
``/clock`` stays global: the single host bridge publishes the one shared sim
clock and the DDS domain/listener topology is unchanged (no bridge start
here, this launch is container-side only).

Each robot stack remaps both TF topics symmetrically for its own
``robot_state_publisher`` and ``move_group``. The frame ids remain prefixed,
and the topic remaps prevent one planning scene monitor from attempting to
transform the other robot's disconnected tree.

CLI::

    ros2 launch rocobrick_piper_l_moveit_config dual_move_group.launch.py \
        robot_ids:=piper_0,piper_1 use_sim_time:=true \
        [config_output_dir:=/path/to/run/dir]

``use_sim_time:=false`` is the offline smoke mode (no host bridge). The
generated per-robot config files are written under
``config_output_dir/<robot_id>/`` (a fresh temp dir by default) and are the
files named in the ``dual-robot`` (A3) readiness evidence.
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
)
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

PACKAGE_NAME = "rocobrick_piper_l_moveit_config"
HELPER_FILE = "dual_robot_config.py"

# Same sim-time rationale as launch/move_group.launch.py (P1.B evidence): with
# use_sim_time:=true the host bridge clock advances faster than wall time while
# the controller result propagates in wall time; keep the monitor meaningful.
EXECUTION_DURATION_SCALING = 5.0


def _load_helper():
    """Import the pure generator module shipped next to this launch file."""
    path = Path(__file__).resolve().parent / HELPER_FILE
    spec = importlib.util.spec_from_file_location("rocobrick_dual_robot_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load the dual-robot config helper {path}")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve annotations through sys.modules during class
    # creation, so register the module before executing it.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _boolean(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def _moveit_config(config):
    """MoveIt parameters for one generated robot model (absolute paths)."""
    return (
        MoveItConfigsBuilder("piper_l", package_name=PACKAGE_NAME)
        .robot_description(file_path=str(config.urdf))
        .robot_description_semantic(file_path=str(config.srdf))
        .robot_description_kinematics(file_path=str(config.kinematics))
        .joint_limits(file_path=str(config.joint_limits))
        .trajectory_execution(file_path=str(config.moveit_controllers))
        .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"])
        .to_moveit_configs()
    )


def _robot_stack(helper, config, use_sim_time: bool):
    """Build the four nodes (plus spawner event) of one robot stack."""
    namespace = config.endpoints.namespace
    moveit_config = _moveit_config(config)
    tf_remappings = [
        ("/tf", f"{namespace}/tf"),
        ("/tf_static", f"{namespace}/tf_static"),
    ]

    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        namespace=namespace,
        output="screen",
        parameters=[
            moveit_config.robot_description,
            str(config.ros2_controllers),
            {"use_sim_time": use_sim_time},
        ],
    )
    controller_spawners = [
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
        for controller in helper.REQUIRED_CONTROLLERS
    ]
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        namespace=namespace,
        remappings=tf_remappings,
        output="screen",
        parameters=[
            moveit_config.robot_description,
            {"use_sim_time": use_sim_time},
        ],
    )
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        namespace=namespace,
        remappings=tf_remappings,
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

    return [
        LogInfo(
            msg=(
                f"[dual-robot] {config.robot_id}: namespace={namespace} "
                f"move_group={config.endpoints.move_group_action} "
                f"arm={config.endpoints.arm_follow} "
                f"controllers={config.endpoints.list_controllers} "
                f"states={config.endpoints.isaac_joint_states} "
                f"commands={config.endpoints.isaac_joint_command} "
                f"config_dir={config.directory}"
            )
        ),
        control_node,
        RegisterEventHandler(
            OnProcessStart(
                target_action=control_node,
                on_start=controller_spawners,
            )
        ),
        robot_state_publisher_node,
        move_group_node,
    ]


def _launch_setup(context):
    helper = _load_helper()
    robot_ids = helper.parse_robot_ids(
        LaunchConfiguration("robot_ids").perform(context)
    )
    use_sim_time = _boolean(LaunchConfiguration("use_sim_time").perform(context))
    output_dir = LaunchConfiguration("config_output_dir").perform(context).strip()
    if not output_dir:
        output_dir = tempfile.mkdtemp(prefix="rocobrick_dual_move_group_")
    package_dir = Path(get_package_share_directory(PACKAGE_NAME))
    configs = helper.generate_dual_robot_configs(
        robot_ids=robot_ids,
        output_dir=Path(output_dir),
        package_dir=package_dir,
    )
    actions = []
    for config in configs:
        actions.extend(_robot_stack(helper, config, use_sim_time))
    return actions


def generate_launch_description():
    """Declare the dual-robot arguments and the per-robot stacks."""
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "robot_ids",
                default_value=",".join(("piper_0", "piper_1")),
                description=(
                    "Comma-separated robot ids; one namespaced "
                    "RobotRuntimeProfile stack is generated per id."
                ),
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                description=(
                    "Use the single shared host bridge clock (/clock); set "
                    "false for offline smokes without the bridge."
                ),
            ),
            DeclareLaunchArgument(
                "config_output_dir",
                default_value="",
                description=(
                    "Directory for the generated per-robot model/config "
                    "files; empty means a fresh temp directory."
                ),
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
