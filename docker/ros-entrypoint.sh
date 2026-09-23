#!/usr/bin/env bash
set -eo pipefail

set +u
source /opt/ros/humble/setup.bash

if [[ -f /workspace/ros_ws/install/setup.bash ]]; then
  source /workspace/ros_ws/install/setup.bash
fi
set -u

if [[ -n "${FASTRTPS_DEFAULT_PROFILES_FILE:-}" && ! -r "$FASTRTPS_DEFAULT_PROFILES_FILE" ]]; then
  printf 'Fast DDS profile is not readable: %s\n' "$FASTRTPS_DEFAULT_PROFILES_FILE" >&2
  exit 2
fi

if [[ "${1:-}" == "self-check" ]]; then
  python3 - <<'PY'
import hashlib
import json
import os
import subprocess
from pathlib import Path

packages = [
    "controller_manager",
    "joint_state_broadcaster",
    "joint_trajectory_controller",
    "moveit_ros_move_group",
    "rmw_fastrtps_cpp",
]
available = set(subprocess.check_output(["ros2", "pkg", "list"], text=True).splitlines())
missing = sorted(set(packages) - available)
profile_value = os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE")
profile = Path(profile_value) if profile_value else None
profile_sha256 = None
if profile is not None and profile.is_file():
    profile_sha256 = hashlib.sha256(profile.read_bytes()).hexdigest()
report = {
    "schema": "rocobrick-ros-self-check/v1",
    "ros_distro": os.environ.get("ROS_DISTRO"),
    "rmw_implementation": os.environ.get("RMW_IMPLEMENTATION"),
    "ros_localhost_only": os.environ.get("ROS_LOCALHOST_ONLY"),
    "ros_domain_id": os.environ.get("ROS_DOMAIN_ID"),
    "fastdds_profile": profile_value,
    "fastdds_profile_sha256": profile_sha256,
    "required_packages": packages,
    "missing_packages": missing,
    "ready": not missing,
}
print(json.dumps(report, sort_keys=True))
raise SystemExit(1 if missing else 0)
PY
  exit $?
fi

exec "$@"
