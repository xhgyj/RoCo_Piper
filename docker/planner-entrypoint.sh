#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
source /workspace/planner_ws/install/setup.bash
set -u
exec ros2 launch rocobrick_moveit_config planning_only.launch.py
