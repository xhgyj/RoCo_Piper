#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAAC_ROOT="${ISAAC_ROOT:-$ROOT/.venv/lib/python3.11/site-packages/isaacsim}"
BRIDGE="$ISAAC_ROOT/exts/isaacsim.ros2.bridge"
PROFILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-$ROOT/.runtime/dds/fastdds-host.xml}"

[[ -r "$PROFILE" ]] || { printf 'DDS profile is not readable: %s\n' "$PROFILE" >&2; exit 2; }
[[ -d "$BRIDGE/humble/lib" ]] || { printf 'Isaac Humble bridge not found: %s\n' "$BRIDGE" >&2; exit 2; }

export ROS_DISTRO=humble
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-73}"
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE="$PROFILE"
export LD_LIBRARY_PATH="$BRIDGE/humble/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$BRIDGE/humble/rclpy${PYTHONPATH:+:$PYTHONPATH}"

sha256sum "$PROFILE" >&2
exec "$@"
