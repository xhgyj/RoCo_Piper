#!/usr/bin/env python3
"""P5 offline TCP-pose planning probe (container move_group, no host Isaac).

Regenerates ``artifacts/phase5/tcp-pose-planning-offline.json``. The probe
runs INSIDE the ROS Humble container against the single-robot ``move_group``
stack; the host Isaac bridge is replaced ONLY as the ``/isaac_joint_states``
source by ``tools/p1_fake_joint_states.py`` (canonical 8-DOF zeros). It
proves the calibrated-TCP planning chain offline:

1. Wait (bounded) for ``/joint_states`` carrying the complete 8-DOF state and
   for the ``/compute_fk`` and ``/plan_kinematic_path`` services.
2. Call ``/compute_fk`` (GetPositionFK) for ``rocobrick_tcp`` in
   ``base_link`` at the current (canonical zero) state. The goal is exactly
   that returned pose — no offset arithmetic anywhere.
3. Plan (plan-only, service ``/plan_kinematic_path`` = GetMotionPlan) one
   goal on the ``arm_tcp`` group: one sphere PositionConstraint plus one
   OrientationConstraint on ``rocobrick_tcp``, both in ``base_link``, with
   the recorded 0.01 m / 0.03 rad tolerances. No action is ever sent.
4. Print exactly one machine-readable JSON line on stdout and exit 0 only
   when every phase passed.

The HOST side of this script (default mode) starts the container, runs the
feeder and the in-container probe, merges the ROS evidence with the formal
``build_tcp_pose_target`` admission metadata and the kinematics.yaml digest,
re-checks every planned waypoint against the narrowed execution limits, and
writes the artifact JSON.

Usage:
    uv run python tools/p5_tcp_pose_offline_probe.py \
        --output artifacts/phase5/tcp-pose-planning-offline.json
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCHEMA = "rocobrick-phase5-tcp-pose-planning-offline/v1"
BRIDGE = "none (fake 8-DOF /joint_states from the probe; no host Isaac)"
SERVICE = "/plan_kinematic_path"
FALLBACK_SERVICE = "/move_group/plan_kinematic_path"
SERVICE_FK = "/compute_fk"
TOPIC_JOINT_STATES = "/joint_states"

BASE_FRAME = "base_link"
TIP_LINK = "rocobrick_tcp"
PLANNING_GROUP = "arm_tcp"
GOAL_POSITION_TOLERANCE_M = 0.01
GOAL_ORIENTATION_TOLERANCE_RAD = 0.03
ALLOWED_PLANNING_TIME = 5.0
NUM_PLANNING_ATTEMPTS = 1
MAX_VELOCITY_SCALING = 1.0
MAX_ACCELERATION_SCALING = 1.0

EXPECTED_DOF_NAMES = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "gripper_joint1",
    "gripper_joint2",
)
ARM_JOINT_NAMES = EXPECTED_DOF_NAMES[:6]

KINEMATICS_YAML = (
    ROOT
    / "ros_ws/src/rocobrick_piper_l_moveit_config/config/kinematics.yaml"
)
KINEMATICS_ARM_TCP_ENTRY = (
    "mirrors arm (KDL, resolution copied; timeout raised to 0.05 s)"
)
RUNTIME_USD = ROOT / "robot_assets/piper_l/USD/piper_l_gripper_d435.usd"
CALIBRATION_ARTIFACT = (
    ROOT / "artifacts/model-audit/piper_l_tcp_calibration.json"
)

READINESS_TIMEOUT_S = 90.0
PROBE_TIMEOUT_S = 150.0
FEEDER_DURATION_S = 150

CONTAINER_IMAGE = "roco-piper-ros:humble"
CONTAINER_LAUNCH = (
    "set -euo pipefail\n"
    "set +u\n"
    "source /opt/ros/humble/setup.bash\n"
    "source /workspace/ros_ws/install/setup.bash\n"
    "set -u\n"
    "exec ros2 launch rocobrick_piper_l_moveit_config "
    "move_group.launch.py use_sim_time:=false\n"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# in-container probe
# --------------------------------------------------------------------------


def run_in_container(output_path: Path) -> int:
    import rclpy
    from moveit_msgs.msg import (
        Constraints,
        MotionPlanRequest,
        OrientationConstraint,
        PositionConstraint,
    )
    from moveit_msgs.srv import GetMotionPlan, GetPositionFK
    from geometry_msgs.msg import Pose
    from shape_msgs.msg import SolidPrimitive
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    rclpy.init()
    node = Node("p5_tcp_pose_offline_probe")
    result: dict = {"schema": SCHEMA, "in_container": True}
    try:
        state_seen: dict = {}
        joint_states: list[JointState] = []

        def on_joint_states(message: JointState) -> None:
            joint_states.append(message)
            state_seen["seen"] = True

        node.create_subscription(JointState, TOPIC_JOINT_STATES, on_joint_states, 10)
        fk_client = node.create_client(GetPositionFK, SERVICE_FK)
        plan_client = node.create_client(GetMotionPlan, SERVICE)
        fallback_client = node.create_client(GetMotionPlan, FALLBACK_SERVICE)
        deadline = time.monotonic() + READINESS_TIMEOUT_S
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
            # The broadcaster publishes the 8 DOFs in hardware-declaration
            # order (joint2, joint3, joint1, ...), so readiness and capture
            # must match by name SET, never by tuple order.
            complete = any(
                set(message.name) == set(EXPECTED_DOF_NAMES)
                and len(message.position) == len(EXPECTED_DOF_NAMES)
                and all(math.isfinite(float(value)) for value in message.position)
                for message in joint_states
            )
            if (
                complete
                and fk_client.service_is_ready()
                and (plan_client.service_is_ready() or fallback_client.service_is_ready())
            ):
                break
        else:
            result["error"] = "readiness phase failed: joint state or services not ready"
            print(json.dumps(result), flush=True)
            return 1

        current = next(
            message
            for message in reversed(joint_states)
            if set(message.name) == set(EXPECTED_DOF_NAMES)
        )
        by_name = dict(zip(current.name, current.position))
        state = {
            "frame": BASE_FRAME,
            "names": list(EXPECTED_DOF_NAMES),
            "positions": [float(by_name[name]) for name in EXPECTED_DOF_NAMES],
        }
        result["state"] = state

        fk_request = GetPositionFK.Request()
        fk_request.header.frame_id = BASE_FRAME
        fk_request.fk_link_names = [TIP_LINK]
        fk_request.header.stamp = current.header.stamp
        fk_future = fk_client.call_async(fk_request)
        rclpy.spin_until_future_complete(node, fk_future, timeout_sec=30.0)
        if fk_future.result() is None or fk_future.result().error_code.val != 1:
            result["error"] = "compute_fk for rocobrick_tcp failed"
            print(json.dumps(result), flush=True)
            return 1
        fk_pose = fk_future.result().pose_stamped[0].pose
        fk_block = {
            "service": SERVICE_FK,
            "link": TIP_LINK,
            "frame": fk_future.result().pose_stamped[0].header.frame_id,
            "position_m": [
                float(fk_pose.position.x),
                float(fk_pose.position.y),
                float(fk_pose.position.z),
            ],
            "orientation_xyzw": [
                float(fk_pose.orientation.x),
                float(fk_pose.orientation.y),
                float(fk_pose.orientation.z),
                float(fk_pose.orientation.w),
            ],
        }
        result["fk"] = fk_block
        if not all(
            math.isfinite(value)
            for value in fk_block["position_m"] + fk_block["orientation_xyzw"]
        ):
            result["error"] = "compute_fk returned non-finite values"
            print(json.dumps(result), flush=True)
            return 1

        request = MotionPlanRequest()
        request.group_name = PLANNING_GROUP
        request.allowed_planning_time = ALLOWED_PLANNING_TIME
        request.num_planning_attempts = NUM_PLANNING_ATTEMPTS
        request.max_velocity_scaling_factor = MAX_VELOCITY_SCALING
        request.max_acceleration_scaling_factor = MAX_ACCELERATION_SCALING
        goal = Constraints()
        position = PositionConstraint()
        position.header.frame_id = BASE_FRAME
        position.link_name = TIP_LINK
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [GOAL_POSITION_TOLERANCE_M]
        position.constraint_region.primitives.append(sphere)
        center = Pose()
        (
            center.position.x,
            center.position.y,
            center.position.z,
        ) = fk_block["position_m"]
        center.orientation.w = 1.0
        position.constraint_region.primitive_poses.append(center)
        position.weight = 1.0
        goal.position_constraints.append(position)
        orientation = OrientationConstraint()
        orientation.header.frame_id = BASE_FRAME
        orientation.link_name = TIP_LINK
        (
            orientation.orientation.x,
            orientation.orientation.y,
            orientation.orientation.z,
            orientation.orientation.w,
        ) = fk_block["orientation_xyzw"]
        orientation.absolute_x_axis_tolerance = GOAL_ORIENTATION_TOLERANCE_RAD
        orientation.absolute_y_axis_tolerance = GOAL_ORIENTATION_TOLERANCE_RAD
        orientation.absolute_z_axis_tolerance = GOAL_ORIENTATION_TOLERANCE_RAD
        orientation.weight = 1.0
        goal.orientation_constraints.append(orientation)
        request.goal_constraints.append(goal)

        client = plan_client if plan_client.service_is_ready() else fallback_client
        service_request = GetMotionPlan.Request()
        service_request.motion_plan_request = request
        plan_future = client.call_async(service_request)
        rclpy.spin_until_future_complete(node, plan_future, timeout_sec=90.0)
        response = plan_future.result()
        if response is None:
            result["error"] = "plan service call did not settle"
            print(json.dumps(result), flush=True)
            return 1
        plan_response = response.motion_plan_response
        trajectory = plan_response.trajectory.joint_trajectory
        result["result"] = {
            "moveit_error_code": int(plan_response.error_code.val),
            "success_code_expected": 1,
            "planning_time_s": float(plan_response.planning_time),
            "joint_names": [str(name) for name in trajectory.joint_names],
            "waypoint_count": len(trajectory.points),
            "waypoints": [
                [float(value) for value in point.positions]
                for point in trajectory.points
            ],
        }
        result["service"] = (
            SERVICE if client is plan_client else FALLBACK_SERVICE
        )
        passed = (
            result["result"]["moveit_error_code"] == 1
            and result["result"]["waypoint_count"] >= 2
            and result["result"]["joint_names"] == list(ARM_JOINT_NAMES)
        )
        result["passed"] = passed
        if output_path != Path("/dev/stdout"):
            output_path.write_text(json.dumps(result) + "\n", encoding="utf-8")
        print(json.dumps(result), flush=True)
        return 0 if passed else 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


# --------------------------------------------------------------------------
# host orchestration
# --------------------------------------------------------------------------


def run_host(output_path: Path) -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from rocobrick.moveit_contract import (
        LIMIT_EPSILON,
        EXECUTION_POSITION_LIMITS,
    )
    from rocobrick.moveit_requests import build_tcp_pose_target
    from rocobrick.tcp_contract import Transform

    if not RUNTIME_USD.exists() or not CALIBRATION_ARTIFACT.exists():
        print("runtime USD or calibration artifact missing", file=sys.stderr)
        return 2
    calibration = json.loads(CALIBRATION_ARTIFACT.read_text(encoding="utf-8"))
    if calibration.get("passed") is not True:
        print("calibration artifact did not pass; refusing to build the target",
              file=sys.stderr)
        return 2
    runtime_model_sha256 = sha256(RUNTIME_USD)
    transform = Transform(
        parent_frame="link6",
        child_frame="rocobrick_tcp",
        translation_m=tuple(calibration["aggregate"]["translation_median_m"]),
        rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
        source=str(CALIBRATION_ARTIFACT.relative_to(ROOT)),
        sha256=sha256(CALIBRATION_ARTIFACT),
    )

    session_dir = Path(tempfile.mkdtemp(prefix="p5-tcp-probe-"))
    container = f"p5-tcp-pose-offline-{uuid.uuid4().hex[:10]}"
    runner = subprocess.run
    docker_run = [
        "docker", "run", "--rm", "-d", "--name", container,
        "-e", "ROS_DOMAIN_ID=0",
        "-e", "ROS_LOCALHOST_ONLY=0",
        "-e", "RMW_IMPLEMENTATION=rmw_fastrtps_cpp",
        "-v", f"{ROOT / 'ros_ws'}:/workspace/ros_ws",
        "-v", f"{ROOT / 'src'}:/workspace/src:ro",
        "-v", f"{ROOT / 'tools'}:/workspace/tools:ro",
        "-v", f"{session_dir}:/session",
        CONTAINER_IMAGE,
        "bash", "-c", CONTAINER_LAUNCH,
    ]
    outcome = runner(docker_run, capture_output=True, text=True, timeout=60)
    if outcome.returncode != 0:
        print(f"docker run failed: {outcome.stderr[-400:]}", file=sys.stderr)
        return 2
    try:
        time.sleep(12.0)
        feeder = runner(
            [
                "docker", "exec", "-d", container, "bash", "-c",
                "source /opt/ros/humble/setup.bash && "
                "source /workspace/ros_ws/install/setup.bash && "
                f"python3 /workspace/tools/p1_fake_joint_states.py "
                f"--duration {FEEDER_DURATION_S}",
            ],
            capture_output=True, text=True, timeout=30,
        )
        if feeder.returncode != 0:
            print(f"feeder exec failed: {feeder.stderr[-400:]}", file=sys.stderr)
            return 2
        probe = runner(
            [
                "docker", "exec", container, "bash", "-c",
                "source /opt/ros/humble/setup.bash && "
                "source /workspace/ros_ws/install/setup.bash && "
                "export PYTHONPATH=/workspace/src:${PYTHONPATH:-} && "
                "timeout 140 python3 "
                "/workspace/tools/p5_tcp_pose_offline_probe.py --in-container",
            ],
            capture_output=True, text=True, timeout=PROBE_TIMEOUT_S,
        )
        raw_line = None
        for line in reversed((probe.stdout or "").splitlines()):
            line = line.strip()
            if line.startswith("{"):
                raw_line = line
                break
        if raw_line is None:
            logs = subprocess.run(
                ["docker", "logs", "--tail", "120", container],
                capture_output=True, text=True, timeout=30,
            )
            print(
                f"probe produced no JSON: {probe.stdout[-400:]} {probe.stderr[-400:]}",
                file=sys.stderr,
            )
            print(f"container logs:\n{logs.stdout[-3000:]}", file=sys.stderr)
            return 2
        raw = json.loads(raw_line)
        if raw.get("passed") is not True:
            logs = subprocess.run(
                ["docker", "logs", "--tail", "120", container],
                capture_output=True, text=True, timeout=30,
            )
            print(f"probe failed: {json.dumps(raw)[:600]}", file=sys.stderr)
            print(f"container logs:\n{logs.stdout[-3000:]}", file=sys.stderr)
            return 1
    finally:
        subprocess.run(["docker", "stop", "-t", "10", container],
                       capture_output=True, timeout=60)
        subprocess.run(["docker", "rm", "-f", container],
                       capture_output=True, timeout=30)

    fk_block = raw["fk"]
    target, admission = build_tcp_pose_target(
        frame_id=BASE_FRAME,
        position=fk_block["position_m"],
        orientation_xyzw=fk_block["orientation_xyzw"],
        runtime_model_sha256=runtime_model_sha256,
        source_sha256=transform.sha256,
        position_tolerance_m=GOAL_POSITION_TOLERANCE_M,
        orientation_tolerance_rad=GOAL_ORIENTATION_TOLERANCE_RAD,
    )
    if target is None or admission is None or not admission.accepted:
        print(f"formal target admission failed: {admission}", file=sys.stderr)
        return 1
    if (
        target.tip_link != TIP_LINK
        or target.frame_id != BASE_FRAME
        or list(target.position) != fk_block["position_m"]
    ):
        print("formal target does not match the live FK goal", file=sys.stderr)
        return 1

    result_block = raw["result"]
    max_excess = {name: 0.0 for name in ARM_JOINT_NAMES}
    for point in result_block["waypoints"]:
        if len(point) != len(ARM_JOINT_NAMES):
            print("planned waypoint arity mismatch", file=sys.stderr)
            return 1
        for name, value in zip(ARM_JOINT_NAMES, point):
            lower, upper = EXECUTION_POSITION_LIMITS[name]
            max_excess[name] = max(
                max_excess[name], max(lower - value, value - upper, 0.0)
            )
    for name in ARM_JOINT_NAMES:
        if max_excess[name] > LIMIT_EPSILON:
            print(f"waypoint exceeds execution limit for {name}", file=sys.stderr)
            return 1

    artifact = {
        "schema": SCHEMA,
        "collected_date_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "offline": True,
        "use_sim_time": False,
        "bridge": BRIDGE,
        "action_sent": False,
        "plan_only": True,
        "service": raw["service"],
        "kinematics_yaml_sha256": sha256(KINEMATICS_YAML),
        "kinematics_arm_tcp_entry": KINEMATICS_ARM_TCP_ENTRY,
        "state": raw["state"],
        "fk": fk_block,
        "formal_target": {
            "builder": "rocobrick.moveit_requests.build_tcp_pose_target",
            "tip_link": target.tip_link,
            "tcp_status": "calibrated_project",
            "runtime_model_sha256": runtime_model_sha256,
            "source_sha256": transform.sha256,
            "position_tolerance_m": GOAL_POSITION_TOLERANCE_M,
            "orientation_tolerance_rad": GOAL_ORIENTATION_TOLERANCE_RAD,
        },
        "goal": {
            "group_name": PLANNING_GROUP,
            "link_name": TIP_LINK,
            "frame_id": BASE_FRAME,
            "position_m": fk_block["position_m"],
            "orientation_xyzw": fk_block["orientation_xyzw"],
            "position_tolerance_m": GOAL_POSITION_TOLERANCE_M,
            "orientation_tolerance_rad": GOAL_ORIENTATION_TOLERANCE_RAD,
            "plan_only": True,
            "goal_source": "live /compute_fk of rocobrick_tcp at the fake state",
        },
        "result": {
            **result_block,
            "max_limit_excess": max_excess,
            "limit_epsilon": LIMIT_EPSILON,
        },
        "passed": True,
        "checks_failed": [],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {output_path}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/phase5/tcp-pose-planning-offline.json",
        help="artifact output path (host mode)",
    )
    parser.add_argument(
        "--in-container",
        action="store_true",
        help="run the ROS probe inside the container and print one JSON line",
    )
    args = parser.parse_args(argv)
    if args.in_container:
        return run_in_container(Path("/dev/stdout"))
    return run_host(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
