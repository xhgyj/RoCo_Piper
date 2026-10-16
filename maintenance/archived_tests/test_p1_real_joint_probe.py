"""Offline structural tests for the P1.B real joint-target execution smoke.

The probe and the move_group stack run inside the ROS container (where rclpy
exists); the offline suite here therefore parses their source and AST only
(never imports them, never touches docker or the GPU). Pinned contract: the
probe waits for the FULL readiness set (clock, /isaac_joint_states,
/joint_states via the broadcaster, three active controllers, the arm
FollowJointTrajectory action, the MoveGroup action, planning service as a
diagnostic), admits the joint target STRICTLY on the caller side (deltas
joint1 +0.20 / joint4 +0.15, margin 0.03, opposite sign fallback) BEFORE
sending, then runs two real plan-and-execute MoveGroup actions (outbound
target, then the exact captured baseline) demanding terminal SUCCEEDED +
MoveIt SUCCESS(1), in-limits planned waypoints, observed /isaac_joint_command
traffic and converged live /isaac_joint_states feedback. The smoke
orchestrates one host bridge plus one rootless container with a client-only
TCP profile (P1.A contract preserved; the additive container mesh profile adds
only a netns-local rendezvous and never a TCP listener, and no port is ever
published).
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
PROBE_PATH = ROOT / "tools" / "p1_real_joint_probe.py"
SMOKE_PATH = ROOT / "tools" / "smoke_p1_real_joint.sh"
GENERATOR_PATH = ROOT / "tools" / "generate_dds_profiles.py"
TEMPLATE_PATH = ROOT / "config" / "dds" / "fastdds-rootless-tcp.xml.in"
MESH_TEMPLATE_PATH = ROOT / "config" / "dds" / "fastdds-container-mesh.xml.in"
ARTIFACT_PATH = ROOT / "artifacts" / "phase1-real" / "joint-target.json"

EXPECTED_DOF_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "gripper_joint1",
    "gripper_joint2",
]
ARM_JOINT_NAMES = {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6"}

# Narrowed execution limits (model contract) the probe must admit against.
NARROW_EXECUTION_LIMITS = {
    "joint1": (-2.6179938, 2.6179938),
    "joint2": (0.0, 3.1415926),
    "joint3": (-2.9670597, 0.0),
    "joint4": (-1.7453292, 1.7453292),
    "joint5": (-1.2217304, 1.2217304),
    "joint6": (-2.0943951, 2.0943951),
}

FORBIDDEN_MODULES = {"socket", "grpc", "zmq", "isaacsim", "omni", "paramiko"}
# The chain is JOINT TARGET ONLY: no pose goal, no example1, no tcp_link.
FORBIDDEN_TOKENS = ("example1", "tcp_link", "PoseGoal", "position_x",
                    "orientation_w", "piper-run")
POSE_TOKEN = re.compile(r"\bpose\b|\bPose\b|\bPOSE\b")


@pytest.fixture(scope="module")
def probe_source() -> str:
    """Return the raw probe source text."""
    return PROBE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def probe_tree(probe_source: str):
    """Return the parsed AST of the probe script."""
    return ast.parse(probe_source)


@pytest.fixture(scope="module")
def smoke_source() -> str:
    """Return the raw smoke script source text."""
    return SMOKE_PATH.read_text(encoding="utf-8")


def module_constants(source: str) -> dict:
    """Evaluate only pure UPPERCASE constant assignments, without imports.

    Args:
        source: The script source text.

    Returns:
        A dict of the module-level constants that evaluate without imports.
    """
    namespace: dict = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and all(
            isinstance(target, ast.Name) and target.id.isupper()
            for target in node.targets
        ):
            try:
                code = compile(
                    ast.Module(body=[node], type_ignores=[]), "<module>", "exec"
                )
                exec(code, {"__builtins__": __builtins__}, namespace)
            except Exception:  # constants referencing imports are skipped
                continue
    return namespace


# ----------------------------------------------------------------- probe


def test_probe_exists_and_needs_only_container_modules(probe_source: str) -> None:
    """The probe must exist and only need container-side ROS modules."""
    assert PROBE_PATH.is_file()
    imported = set()
    for node in ast.walk(ast.parse(probe_source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert {"rclpy", "moveit_msgs", "control_msgs", "controller_manager_msgs",
            "sensor_msgs", "rosgraph_msgs", "action_msgs"} <= imported
    assert imported.isdisjoint(FORBIDDEN_MODULES)


def test_probe_no_pose_example1_or_tcp_tokens(probe_source: str) -> None:
    """JOINT TARGET ONLY: no pose goal, no example1, no tcp_link anywhere."""
    for token in FORBIDDEN_TOKENS:
        assert token not in probe_source, token
    assert not POSE_TOKEN.search(probe_source), "a pose token leaked into the probe"


def test_probe_pinned_schema_topics_and_actions(probe_source: str) -> None:
    """Schema, the three contract topics, and both action names."""
    constants = module_constants(probe_source)
    assert constants["SCHEMA"] == "rocobrick-p1-real-joint-probe/v1"
    assert constants["PHASE"] == "P1.B"
    assert constants["TOPIC_CLOCK"] == "/clock"
    assert constants["TOPIC_JOINT_STATES"] == "/isaac_joint_states"
    assert constants["TOPIC_BROADCAST_STATES"] == "/joint_states"
    assert constants["TOPIC_JOINT_COMMANDS"] == "/isaac_joint_command"
    assert constants["ACTION_ARM_FOLLOW"] == "/arm_controller/follow_joint_trajectory"
    assert list(constants["MOVE_GROUP_ACTION_CANDIDATES"]) == [
        "/move_action",
        "/move_group",
    ]
    assert constants["SERVICE_LIST_CONTROLLERS"] == "/controller_manager/list_controllers"
    assert list(constants["SERVICE_PLAN_CANDIDATES"]) == [
        "/plan_kinematic_path",
        "/move_group/plan_kinematic_path",
    ]


def test_probe_pins_the_eight_dof_names(probe_source: str) -> None:
    """The exact 8-DOF Piper_L names in contract order."""
    constants = module_constants(probe_source)
    assert list(constants["EXPECTED_DOF_NAMES"]) == EXPECTED_DOF_NAMES
    assert set(constants["GRIPPER_JOINT_NAMES"]) == {"gripper_joint1", "gripper_joint2"}


def test_probe_hardcodes_the_narrow_execution_limits(probe_source: str) -> None:
    """The admission table must match the model contract exactly."""
    constants = module_constants(probe_source)
    limits = constants["EXECUTION_POSITION_LIMITS"]
    assert set(limits) == ARM_JOINT_NAMES
    for name, (lower, upper) in NARROW_EXECUTION_LIMITS.items():
        assert limits[name] == (pytest.approx(lower), pytest.approx(upper)), name
    assert "joint_limits.yaml" in probe_source  # cites the planning config
    assert "Assuming maximum bounds" in probe_source  # never relied on


def test_probe_readiness_requires_the_full_set(probe_source: str) -> None:
    """No motion before clock, states, broadcaster, controllers, actions."""
    start = probe_source.index("def run_probe")
    readiness_start = probe_source.index("readiness phase", start)
    readiness_end = probe_source.index("baseline capture", readiness_start)
    readiness = probe_source[readiness_start:readiness_end]
    assert "MIN_BASELINE_STATE_MESSAGES" in readiness
    assert "later > earlier" in readiness or "clock_increasing" in readiness
    assert "REQUIRED_CONTROLLERS" in readiness
    for token in (
        "joint_state_broadcaster",
        "arm_controller",
        "gripper_controller",
    ):
        assert token in probe_source.split("def run_probe")[0], token
    for token in (
        '"active"',
        "arm_follow_action_ready",
        "move_group_action_name",
        "plan_service_name",
        "broadcast_states",
    ):
        assert token in readiness, token


def test_probe_baseline_requires_three_valid_messages(probe_source: str) -> None:
    """Baseline only after >= 3 valid states and a fresh complete snapshot."""
    constants = module_constants(probe_source)
    assert constants["MIN_BASELINE_STATE_MESSAGES"] == 3
    assert constants["STATE_FRESH_SECONDS"] <= 5.0
    assert 'result["baseline_fresh"]' in probe_source
    assert 'result["baseline_matches_broadcast"]' in probe_source
    assert "is_valid_state" in probe_source
    assert 'len(msg.velocity) not in (0, len(EXPECTED_DOF_NAMES))' in probe_source
    assert constants["EXACT_STATE_MATCH_TOLERANCE"] <= 1.0e-3


def test_probe_strict_admission_happens_before_the_goal_send(
    probe_source: str,
) -> None:
    """Admission (deltas, margin, sign fallback) runs BEFORE send_goal_async."""
    constants = module_constants(probe_source)
    assert constants["TARGET_DELTAS"] == {"joint1": pytest.approx(0.20),
                                          "joint4": pytest.approx(0.15)}
    assert constants["ADMISSION_MARGIN"] == pytest.approx(0.03)
    admission_index = probe_source.index("strict goal admission")
    send_call_index = probe_source.index('"outbound",')
    assert admission_index < send_call_index, "goal must be admitted before sending"
    # The actual wire send lives in the shared execution helper.
    assert "send_goal_async(goal)" in probe_source
    # Opposite-sign fallback, then hard failure before the action call.
    assert "for sign in (1.0, -1.0)" in probe_source
    assert "chosen_deltas" in probe_source
    body = probe_source[admission_index:send_call_index]
    assert "ADMISSION_MARGIN < value < upper - ADMISSION_MARGIN" in body
    assert "no sign of delta" in body


def test_probe_movegroup_goal_is_plan_and_execute(probe_source: str) -> None:
    """plan_only False, replan False, empty start state, request contract."""
    constants = module_constants(probe_source)
    assert 'goal.planning_options.plan_only = False' in probe_source
    assert 'goal.planning_options.replan = False' in probe_source
    assert ".start_state" not in probe_source  # start state stays empty
    assert constants["GOAL_TOLERANCE"] == pytest.approx(0.01)
    assert constants["ALLOWED_PLANNING_TIME"] == pytest.approx(5.0)
    assert constants["MAX_VELOCITY_SCALING"] == pytest.approx(0.1)
    assert constants["MAX_ACCELERATION_SCALING"] == pytest.approx(0.1)
    assert "STATUS_SUCCEEDED = GoalStatus.STATUS_SUCCEEDED" in probe_source
    assert "from action_msgs.msg import GoalStatus" in probe_source
    assert constants["SUCCESS_CODE"] == 1
    body = probe_source[probe_source.index("def build_request") :]
    assert 'request.group_name = "arm"' in body
    assert "constraint.tolerance_above = GOAL_TOLERANCE" in body
    assert "constraint.tolerance_below = GOAL_TOLERANCE" in body
    assert "for name in ARM_JOINT_NAMES:" in body  # all six arm joints
    # The request carries exactly the six arm joints (the admission table keys).
    constants = module_constants(probe_source)
    assert set(constants["EXECUTION_POSITION_LIMITS"]) == ARM_JOINT_NAMES


def test_probe_goal_is_bounded_and_times_out_to_cancel(probe_source: str) -> None:
    """Wall deadline; an overrun is cancelled and the cancel code recorded."""
    constants = module_constants(probe_source)
    assert constants["MIN_WAYPOINTS"] >= 2
    options = {
        node.args[0].value
        for node in ast.walk(ast.parse(probe_source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }
    assert {"--timeout", "--goal-timeout", "--settle-timeout"} <= options
    assert "must be positive" in probe_source
    assert "wait_future(result_future, hard_deadline)" in probe_source
    assert 'summary["timed_out"] = True' in probe_source
    assert "cancel_goal_async" in probe_source
    assert 'summary["cancel"]["return_code"]' in probe_source
    assert 'error_code"] == SUCCESS_CODE' in probe_source
    assert 'summary["status"] == STATUS_SUCCEEDED' in probe_source


def test_probe_checks_every_planned_waypoint_against_the_limits(
    probe_source: str,
) -> None:
    """Normalize the waypoint order; every waypoint finite and in-limits."""
    body = probe_source[probe_source.index("def inspect_planned_trajectory") :]
    assert "EXECUTION_POSITION_LIMITS[name]" in body
    assert "LIMIT_EPSILON" in body
    assert "waypoint_names.index(name)" in body  # normalization
    assert "min_waypoints_met" in body
    assert "waypoints_within_limits" in body
    assert "nonfinite" in body
    assert "arm_joints_exact" in body
    assert "planned_trajectory" in probe_source


def test_probe_subscribes_the_live_command_topic(probe_source: str) -> None:
    """/isaac_joint_command is monitored inside the probe."""
    assert "TOPIC_JOINT_COMMANDS, monitor.on_command" in probe_source
    assert "command_count" in probe_source
    assert "command_unknown_names" in probe_source
    assert "command_min" in probe_source and "command_max" in probe_source
    constants = module_constants(probe_source)
    assert constants["MIN_COMMAND_MESSAGES"] == 1


def test_probe_final_live_feedback_tolerances(probe_source: str) -> None:
    """joint1/joint4 0.03, other arm joints 0.04, gripper drift 0.01."""
    constants = module_constants(probe_source)
    assert constants["FINAL_TARGET_ERROR_MAX_MAIN"] == pytest.approx(0.03)
    assert constants["FINAL_TARGET_ERROR_MAX_OTHER"] == pytest.approx(0.04)
    assert constants["GRIPPER_DRIFT_MAX"] == pytest.approx(0.01)
    assert constants["RETURN_ARM_ERROR_MAX"] == pytest.approx(0.03)
    assert 'post_command_state_messages' in probe_source
    assert "MIN_POST_COMMAND_STATE_MESSAGES" in probe_source
    assert "settle_deadline = time.monotonic() + settle_timeout" in probe_source


def test_probe_requires_the_settled_arm_to_be_at_rest(probe_source: str) -> None:
    """The next goal is only sent once the arm is at rest (stable samples).

    MoveIt validates the planned start point against the live state with a
    0.01 tolerance; planning from a still-settling state aborts execution
    (CONTROL_FAILED) even when the position is already inside the final
    tolerance. The settle gate therefore demands consecutive stable samples.
    """
    constants = module_constants(probe_source)
    assert constants["SETTLE_STABLE_SAMPLES"] >= 5
    assert constants["SETTLE_STABLE_DELTA_MAX"] <= 1.0e-3
    body = probe_source[probe_source.index("Bounded convergence wait") :]
    body = body[: body.index("def run_probe")]
    assert "stable_count" in body
    assert "max_step(states)" in body
    assert '"settled_at_rest"' in probe_source
    assert 'summary["feedback_passed"] = bool(' in body
    # The at-rest condition sits inside the SHARED verdict (covers both legs).
    assert probe_source.count('and summary["settled_at_rest"]') == 1


def test_probe_return_to_baseline_is_a_real_second_action(
    probe_source: str,
) -> None:
    """A second plan-and-execute goal targets the exact captured baseline."""
    assert probe_source.count("execute_move_group_goal(") >= 3  # def + 2 calls
    body = probe_source[probe_source.index("return to baseline") :]
    assert "{name: baseline[name] for name in ARM_JOINT_NAMES}" in body
    assert '"return"' in body
    assert "RETURN_ARM_ERROR_MAX" in body
    assert 'result["return"] = return_summary' in body


def test_probe_one_json_line_and_exit_code_contract(probe_source: str) -> None:
    """Exactly one final JSON result line decides the exit code."""
    assert 'SCHEMA = "rocobrick-p1-real-joint-probe/v1"' in probe_source
    assert probe_source.count("print(json.dumps(result, sort_keys=True)") == 1
    assert re.search(r"return 0 if result\[.passed.\] else 1", probe_source)


def test_probe_single_node_and_clean_shutdown(probe_tree) -> None:
    """One rclpy node; destroy/shutdown run from the finally block."""
    assert probe_tree.body.count(
        next(
            node
            for node in probe_tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "run_probe"
        )
    ) >= 1
    source = ast.unparse(probe_tree)
    assert source.count("create_node(") == 1
    finally_nodes = [
        node for node in ast.walk(probe_tree) if isinstance(node, ast.Try) and node.finalbody
    ]
    assert finally_nodes
    attrs = {
        sub.func.attr
        for node in finally_nodes
        for sub in ast.walk(ast.Module(body=node.finalbody, type_ignores=[]))
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
    }
    assert {"destroy_node", "shutdown"} <= attrs


def test_probe_is_fully_wall_clock_bounded(probe_source: str) -> None:
    """Every phase uses time.monotonic() deadlines."""
    assert probe_source.count("time.monotonic() +") >= 4
    assert "while time.monotonic() <" in probe_source


# ----------------------------------------------------- generator (mesh profile)


def _generate(tmp_path: Path, *extra: str) -> tuple[Path, dict]:
    output = tmp_path / "dds"
    result = subprocess.run(
        [sys.executable, str(GENERATOR_PATH), "--output-dir", str(output), *extra],
        capture_output=True,
        text=True,
        check=True,
    )
    return output, json.loads(result.stdout)


def test_mesh_template_exists_and_is_additive(tmp_path: Path) -> None:
    """The mesh template exists; the DEFAULT generator output is unchanged."""
    assert MESH_TEMPLATE_PATH.is_file()
    output, report = _generate(tmp_path / "plain")
    assert "fastdds-container-mesh.xml" not in {p.name for p in output.glob("*.xml")}
    assert "container-mesh" not in report["profiles"]
    assert set(report["roles"]) == {"host-listener", "container-client"}


def test_mesh_profile_is_client_only_with_local_rendezvous(tmp_path: Path) -> None:
    """Mesh profile: no TCP listener, host peer, local SPDP multicast peer."""
    output, report = _generate(tmp_path / "mesh", "--container-mesh",
                               "--ros-domain-id", "73")
    entry = report["profiles"]["container-mesh"]
    assert entry["role"] == "container-client-mesh"
    assert entry["listening_ports"] == []
    assert entry["initial_peers"] == [[report["host_address"], report["host_port"]]]
    assert entry["local_spdp_multicast"] == ["239.255.0.1", 7400 + 250 * 73]
    xml = re.sub(r"<!--.*?-->", "",
                 (output / "fastdds-container-mesh.xml").read_text(encoding="utf-8"),
                 flags=re.DOTALL)
    assert "<listening_ports>" not in xml  # still no TCP listener anywhere
    assert "<initialPeersList>" in xml
    assert f"<address>{report['host_address']}</address>" in xml
    assert f"<physical_port>{report['host_port']}</physical_port>" in xml
    assert "<address>239.255.0.1</address>" in xml
    assert "<port>25650</port>" in xml
    assert "<type>UDPv4</type>" in xml
    # The boundary contract of the P1.A profiles is untouched by the flag.
    plain = tmp_path / "plain-dds"
    subprocess.run(
        [sys.executable, str(GENERATOR_PATH), "--output-dir", str(plain)],
        capture_output=True, text=True, check=True,
    )
    for name in ("host", "container", "container-p1", "container-p4"):
        assert (plain / f"fastdds-{name}.xml").read_bytes() == (
            output / f"fastdds-{name}.xml"
        ).read_bytes(), name


# ------------------------------------------------------------------- smoke


def test_smoke_exists_and_is_bash(smoke_source: str) -> None:
    """The smoke script must exist with a bounded shebang header."""
    assert SMOKE_PATH.is_file()
    assert smoke_source.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in smoke_source


def test_smoke_requires_the_p1a_gate_prerequisite(smoke_source: str) -> None:
    """The P1.A multiprocess gate artifact must exist and have passed all."""
    assert "dds-multiprocess-gate.json" in smoke_source
    assert "rocobrick-p1-dds-multiprocess-gate/v1" in smoke_source
    assert '"p1a_prerequisite"' in smoke_source
    assert "did not pass all checks" in smoke_source


def test_smoke_runs_the_model_audit_and_generates_client_only_profiles(
    smoke_source: str,
) -> None:
    """Offline preflight audit + profiles with the additive mesh flag."""
    assert "piper_l_model_audit.py" in smoke_source
    assert "generate_dds_profiles.py" in smoke_source
    assert "--container-mesh" in smoke_source
    assert "fastdds-container-mesh.xml" in smoke_source
    assert '"host_port"' not in smoke_source  # the host port stays default-driven
    assert "DDS_HOST_PORT" in smoke_source


def test_smoke_publishes_no_port_and_has_no_obsolete_tokens(
    smoke_source: str,
) -> None:
    """Nothing is published across the boundary; no stale container port."""
    assert re.search(r"docker run[^\n]*\s-p\b", smoke_source) is None
    assert "-p $" not in smoke_source and "-p ${" not in smoke_source
    assert "17651" not in smoke_source
    assert "DDS_CONTAINER_PORT" not in smoke_source
    assert "network_mode: host" not in smoke_source


def test_smoke_diagnoses_the_gpu_without_killing_foreign_processes(
    smoke_source: str,
) -> None:
    """A recorded GPU diagnostic; no pkill/killall of anything foreign."""
    assert "nvidia-smi" in smoke_source
    assert "gpu-diagnostic" in smoke_source
    assert "pkill" not in smoke_source
    assert "killall" not in smoke_source


def test_smoke_starts_the_host_bridge_with_long_bounds(smoke_source: str) -> None:
    """Host bridge via isaac-ros-env.sh, hold/time limits, marker wait."""
    assert "tools/isaac-ros-env.sh" in smoke_source
    assert "p05_host_bridge.py" in smoke_source
    assert 'MARKER="p0.5_host_bridge_ready"' in smoke_source
    assert 'HOST_EVIDENCE="$OUT_DIR/host-joint-bridge.json"' in smoke_source
    assert 'HOST_LOG="$OUT_DIR/host-joint-bridge.log"' in smoke_source
    assert '--time-limit "$TIME_LIMIT"' in smoke_source
    assert '--hold-seconds "$HOLD_SECONDS"' in smoke_source
    assert '--hold-steps "$HOLD_STEPS"' in smoke_source
    assert '--max-steps "$MAX_STEPS"' in smoke_source
    # Hold (240s) shorter than the stop grace (300s): Kit intercepts SIGINT
    # and exits 0 without the python cleanup, so the bridge must always
    # finish its hold and self-exit before any escalation could fire.
    assert 'HOLD_SECONDS="${P1_HOLD_SECONDS:-240}"' in smoke_source
    assert 'STOP_WAIT="${P1_STOP_WAIT:-300}"' in smoke_source
    assert 'TIME_LIMIT="${P1_TIME_LIMIT:-600}"' in smoke_source
    assert 'grep -q "\\"marker\\": \\"$MARKER\\""' in smoke_source


def test_smoke_runs_one_container_launching_move_group_on_sim_time(
    smoke_source: str,
) -> None:
    """One rootless container; move_group with use_sim_time:=true; probe."""
    assert "docker run --rm" in smoke_source
    assert 'CONTAINER_NAME="rocobrick-p1-real-joint-$$"' in smoke_source
    assert '-v "$WS_DIR:/workspace/ros_ws"' in smoke_source
    assert '-v "$RUNTIME:/dds:ro"' in smoke_source
    assert '-e FASTRTPS_DEFAULT_PROFILES_FILE=/dds/fastdds-container-mesh.xml' \
        in smoke_source
    assert (
        "ros2 launch rocobrick_piper_l_moveit_config move_group.launch.py"
        in smoke_source
    )
    assert "use_sim_time:=true" in smoke_source
    assert "p1_real_joint_probe.py" in smoke_source
    assert "--goal-timeout" in smoke_source and "--settle-timeout" in smoke_source
    assert "ros2 control list_controllers" in smoke_source


def test_smoke_captures_the_in_namespace_listener_evidence(
    smoke_source: str,
) -> None:
    """Any in-namespace TCP listener must be loopback-bound (no boundary)."""
    assert "net-tcp-listeners.txt" in smoke_source
    assert '/proc/net/tcp' in smoke_source
    assert '"container_profile_client_only_no_listener"' in smoke_source
    assert "LISTEN_ADDRESSES" in smoke_source  # loopback-only allow-list
    assert "not listener_rows" in smoke_source


def test_smoke_stops_the_host_orderly_and_cleans_up(smoke_source: str) -> None:
    """Orderly host stop with escalation; the trap removes the container."""
    assert "trap cleanup EXIT" in smoke_source
    assert "docker rm -f" in smoke_source
    assert "stop_host" in smoke_source
    assert 'kill -INT "$HOST_PID"' in smoke_source
    assert 'kill -KILL "$HOST_PID"' in smoke_source
    assert "STOP_WAIT" in smoke_source


def test_smoke_writes_the_aggregate_artifact_with_all_checks(
    smoke_source: str,
) -> None:
    """Schema, artifact path, embedded evidence, and the full check set."""
    assert 'AGGREGATE="$OUT_DIR/joint-target.json"' in smoke_source
    assert "artifacts/phase1-real" in smoke_source
    assert "rocobrick-p1-real-joint-smoke/v1" in smoke_source
    assert "rocobrick-p1-real-joint-probe/v1" in smoke_source
    for token in (
        '"readiness"',
        '"probe_passed"',
        '"outbound_target_passed"',
        '"return_passed"',
        '"command_observed"',
        '"host_commands_applied"',
        '"host_cleanup_orderly"',
        '"host_marker"',
        '"log_scan_clean"',
        '"bounded_and_no_leftovers"',
        'report["passed"] = all(checks.values())',
        "controllers_listing",
        "host_log_tail",
        "container_log_tail",
        "move_group_log_tail",
        "profiles",
    ):
        assert token in smoke_source, token
    # applied == received, no clamps, no unknown names on the host side.
    assert 'host_commands.get("applied") == host_commands.get("received")' \
        in smoke_source
    assert 'host_commands.get("clamped_to_limits") == 0' in smoke_source
    assert 'host_commands.get("unknown_joint_names_ignored") == []' in smoke_source


def test_smoke_scans_all_logs_for_transport_and_state_failures(
    smoke_source: str,
) -> None:
    """Address-in-use, binding, SHM, UDP/multicast, hardware, stale, NaN."""
    for pattern in (
        '"address already in use"',
        '"error binding"',
        '"binding at port"',
        '"shared memory"',
        'SHM_PATTERN = "SHM"',
        '"udpv4"',
        '"udpv6"',
        '"multicast"',
        '"hardware init"',
        '"failed to activate"',
        '"stale"',
    ):
        assert pattern in smoke_source, pattern
    assert re.search(r"nan\\b.*IGNORECASE|IGNORECASE.*nan\\b", smoke_source)
    # The bare lowercase "shm" substring is deliberately NOT a pattern: Kit
    # CUDA error file paths contain it without any DDS meaning.
    bare_shm = re.search(r'"shm",\n', smoke_source)
    assert bare_shm is None
    assert '"host-joint-bridge.log"' in smoke_source
    assert '"container.log"' in smoke_source
    assert '"probe.log"' in smoke_source
    assert '"move_group.log.tail"' in smoke_source


def test_smoke_is_hard_bounded(smoke_source: str) -> None:
    """Every execution stage is wrapped in `timeout` with escalation."""
    assert smoke_source.count("timeout --signal=INT --kill-after=30") >= 2
    assert "timeout --signal=INT --kill-after=15" in smoke_source
    assert smoke_source.count("timeout --signal=INT --kill-after=10") >= 1
    assert 'TOTAL_TIMEOUT="${P1_TOTAL_TIMEOUT:-780}"' in smoke_source
    assert 'PROBE_TIMEOUT="${P1_PROBE_TIMEOUT:-480}"' in smoke_source
    assert 'MOVE_GROUP_WAIT="${P1_MOVE_GROUP_WAIT:-150}"' in smoke_source


def test_smoke_artifact_when_present_is_truthful() -> None:
    """A committed artifact must report honest pass/fail with all checks."""
    if not ARTIFACT_PATH.is_file():
        pytest.skip("live P1.B artifact not collected yet")
    report = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert report["schema"] == "rocobrick-p1-real-joint-smoke/v1"
    assert set(report["checks"]) == {
        "p1a_prerequisite",
        "host_marker",
        "container_profile_client_only_no_listener",
        "readiness",
        "probe_passed",
        "outbound_target_passed",
        "return_passed",
        "command_observed",
        "host_commands_applied",
        "host_cleanup_orderly",
        "log_scan_clean",
        "bounded_and_no_leftovers",
    }
    assert report["passed"] == all(report["checks"].values())
