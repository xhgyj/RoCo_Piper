"""Offline structural tests for the P1 container MoveIt planning smoke.

The feeder and the probe run inside the ROS container, where rclpy exists;
the offline suite here therefore only parses their source and AST (never
imports them, never touches docker). Pinned contract: the feeder replaces
ONLY the host Isaac bridge by publishing the complete 8-DOF state on the
plural /isaac_joint_states topic; the probe waits for /joint_states plus the
move_group planning service, plans a VALID goal inside the narrow execution
limits, and demands the REJECTION of the wide-only joint4 = 2.0 rad goal
(inside the wide URDF limit, outside the narrowed execution limit); the
smoke orchestrates it all in one container-only docker run.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
FEEDER_PATH = ROOT / "tools" / "p1_fake_joint_states.py"
PROBE_PATH = ROOT / "tools" / "p1_moveit_offline_probe.py"
SMOKE_PATH = ROOT / "tools" / "smoke_p1_moveit_offline.sh"

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

# Narrowed execution limits (model contract): the values the probe must
# enforce on every planned waypoint.
NARROW_EXECUTION_LIMITS = {
    "joint1": (-2.6179938, 2.6179938),
    "joint2": (0.0, 3.1415926),
    "joint3": (-2.9670597, 0.0),
    "joint4": (-1.7453292, 1.7453292),
    "joint5": (-1.2217304, 1.2217304),
    "joint6": (-2.0943951, 2.0943951),
}

FORBIDDEN_MODULES = {"socket", "grpc", "zmq", "isaacsim", "omni", "paramiko"}


@pytest.fixture(scope="module")
def feeder_source() -> str:
    """Return the raw feeder source text.

    Returns:
        The feeder script source.
    """
    return FEEDER_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def probe_source() -> str:
    """Return the raw probe source text.

    Returns:
        The probe script source.
    """
    return PROBE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def probe_tree(probe_source: str):
    """Return the parsed AST of the probe script.

    Args:
        probe_source: The probe script source text.

    Returns:
        The probe script AST.
    """
    return ast.parse(probe_source)


@pytest.fixture(scope="module")
def smoke_source() -> str:
    """Return the raw smoke script source text.

    Returns:
        The smoke script source.
    """
    return SMOKE_PATH.read_text(encoding="utf-8")


def module_constants(source: str) -> dict:
    """Evaluate only UPPERCASE constant assignments, without imports.

    Args:
        source: The script source text.

    Returns:
        A dict of the module-level constant names to their values.
    """
    namespace: dict = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and all(
            isinstance(target, ast.Name) and target.id.isupper()
            for target in node.targets
        ):
            code = compile(ast.Module(body=[node], type_ignores=[]), "<module>", "exec")
            exec(code, {"__builtins__": __builtins__}, namespace)
    return namespace


# ---------------------------------------------------------------- feeder


def test_feeder_exists_and_is_offline_import_free(feeder_source: str) -> None:
    """The feeder must exist and only need container-side ROS modules."""
    assert FEEDER_PATH.is_file()
    imported = set()
    for node in ast.walk(ast.parse(feeder_source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert {"rclpy", "sensor_msgs"} <= imported
    assert imported.isdisjoint(FORBIDDEN_MODULES)


def test_feeder_publishes_the_plural_states_topic(feeder_source: str) -> None:
    """Topic is the plural /isaac_joint_states; never a command publisher."""
    assert 'TOPIC_JOINT_STATES = "/isaac_joint_states"' in feeder_source
    assert "TOPIC_JOINT_COMMANDS" not in feeder_source
    assert "create_publisher(JointState, TOPIC_JOINT_STATES" in feeder_source


def test_feeder_publishes_exactly_the_eight_names(feeder_source: str) -> None:
    """The message carries the exact 8-DOF names, in the contract order."""
    constants = module_constants(feeder_source)
    assert list(constants["EXPECTED_DOF_NAMES"]) == EXPECTED_DOF_NAMES
    assert list(constants["INITIAL_POSITIONS"]) == EXPECTED_DOF_NAMES
    assert "message.name = list(EXPECTED_DOF_NAMES)" in feeder_source
    assert "message.position = [INITIAL_POSITIONS[name]" in feeder_source
    assert 'config/initial_positions.yaml' in feeder_source


def test_feeder_is_bounded_and_shuts_down_cleanly(feeder_source: str) -> None:
    """Lifetime is bounded by --duration; node shutdown runs in a finally."""
    assert "--duration" in feeder_source
    assert "deadline = time.monotonic() + args.duration" in feeder_source
    assert "while time.monotonic() < deadline" in feeder_source
    assert "must be positive" in feeder_source
    finally_nodes = [
        node
        for node in ast.walk(ast.parse(feeder_source))
        if isinstance(node, ast.Try) and node.finalbody
    ]
    assert finally_nodes
    attrs = {
        sub.func.attr
        for node in finally_nodes
        for sub in ast.walk(ast.Module(body=node.finalbody, type_ignores=[]))
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
    }
    assert {"destroy_node", "shutdown"} <= attrs
    assert "replaces ONLY" in feeder_source  # feeder == bridge replacement


# ----------------------------------------------------------------- probe


def test_probe_exists_and_is_offline_import_free(probe_source: str) -> None:
    """The probe must exist and only need container-side ROS modules."""
    assert PROBE_PATH.is_file()
    imported = set()
    for node in ast.walk(ast.parse(probe_source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert {"rclpy", "moveit_msgs", "sensor_msgs"} <= imported
    assert imported.isdisjoint(FORBIDDEN_MODULES)


def test_probe_pinned_schema_and_standard_topic(probe_source: str) -> None:
    """Schema, phase, and the broadcaster-republished /joint_states topic."""
    constants = module_constants(probe_source)
    assert constants["SCHEMA"] == "rocobrick-p1-moveit-offline/v1"
    assert constants["PHASE"] == "P1"
    assert 'TOPIC_JOINT_STATES = "/joint_states"' in probe_source


def test_probe_waits_for_service_candidates(probe_source: str) -> None:
    """Both planning service names are tried; the one used is recorded."""
    constants = module_constants(probe_source)
    assert list(constants["SERVICE_CANDIDATES"]) == [
        "/plan_kinematic_path",
        "/move_group/plan_kinematic_path",
    ]
    assert 'result["service_name_used"]' in probe_source
    assert "create_client" in probe_source


def test_probe_hardcodes_the_narrow_execution_limits(probe_source: str) -> None:
    """The narrow table must match the model contract, joint4/joint5 narrow."""
    constants = module_constants(probe_source)
    limits = constants["EXECUTION_POSITION_LIMITS"]
    assert set(limits) == ARM_JOINT_NAMES
    for name, (lower, upper) in NARROW_EXECUTION_LIMITS.items():
        assert limits[name] == (pytest.approx(lower), pytest.approx(upper)), name
    assert limits["joint4"] == (pytest.approx(-1.7453292), pytest.approx(1.7453292))
    assert limits["joint5"] == (pytest.approx(-1.2217304), pytest.approx(1.2217304))
    assert "joint_limits.yaml" in probe_source  # cites the planning config


def test_probe_goal_targets_are_inside_the_narrow_limits(probe_source: str) -> None:
    """The VALID goal must sit strictly inside the narrow table."""
    constants = module_constants(probe_source)
    valid = constants["VALID_GOAL_POSITIONS"]
    assert set(valid) == ARM_JOINT_NAMES
    for name, target in valid.items():
        lower, upper = NARROW_EXECUTION_LIMITS[name]
        assert lower < target < upper, name
    assert constants["GOAL_TOLERANCE"] == pytest.approx(0.01)
    assert constants["ALLOWED_PLANNING_TIME"] == pytest.approx(5.0)
    assert constants["MIN_WAYPOINTS"] >= 2


def test_probe_wide_only_goal_is_the_narrowing_proof(probe_source: str) -> None:
    """joint4 = 2.0 (wide-URDF-only) may never yield an out-of-limits plan.

    MoveIt clamps out-of-bounds joint goals to the bound instead of rejecting
    ("Assuming maximum bounds instead"), so the acceptance property is: the
    response is either hard-rejected or clamped with every waypoint inside the
    narrow limits — never a trajectory beyond the narrowed execution limits.
    """
    constants = module_constants(probe_source)
    wide_joint = constants["WIDE_JOINT_NAME"]
    wide_position = constants["WIDE_JOINT4_POSITION"]
    assert wide_joint == "joint4"
    assert wide_position == pytest.approx(2.0)
    # 2.0 is inside the WIDE official URDF limit, outside the narrowed limit.
    assert NARROW_EXECUTION_LIMITS[wide_joint][1] < wide_position < 2.2165681
    assert 'wide_summary["error_code"] != SUCCESS_CODE' in probe_source
    assert 'wide_summary["hard_rejected"]' in probe_source
    assert 'wide_summary["clamped_within_limits"]' in probe_source
    assert 'wide_summary["never_exceeds_limits"]' in probe_source
    assert constants["WIDE_EXPECTED_ERROR_CODE"] == -1
    passed_expr = " ".join(
        probe_source.split()
    )
    assert (
        'result["passed"] = bool( result["joint_states_flow"]'
        ' and wide_summary["never_exceeds_limits"] )' in passed_expr
    )


def test_probe_checks_every_waypoint_against_the_narrow_table(
    probe_source: str,
) -> None:
    """Waypoint bounds use EXECUTION_POSITION_LIMITS over all arm joints."""
    body = probe_source[probe_source.index("def call_plan") :]
    assert "EXECUTION_POSITION_LIMITS[name]" in body
    assert "waypoints_within_limits" in probe_source
    assert "LIMIT_EPSILON" in probe_source


def test_probe_run_is_fully_bounded(probe_tree, probe_source: str) -> None:
    """Ready wait and each planning call need wall-clock deadlines."""
    options = {
        node.args[0].value
        for node in ast.walk(probe_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }
    assert {"--timeout", "--plan-timeout"} <= options
    assert probe_source.count("time.monotonic() +") >= 2
    assert probe_source.count("while time.monotonic() <") >= 1
    assert (
        "while not future.done() and time.monotonic() < deadline:" in probe_source
    )  # the bounded wait inside call_plan covers BOTH goals
    assert probe_source.count("call_plan(") >= 3  # definition + two bounded calls
    assert "must be positive" in probe_source


def test_probe_one_json_line_and_exit_code_contract(probe_source: str) -> None:
    """One JSON result line decides the exit code."""
    assert 'SCHEMA = "rocobrick-p1-moveit-offline/v1"' in probe_source
    assert 'print(json.dumps(result, sort_keys=True), flush=True)' in probe_source
    assert re.search(r"return 0 if result\[.passed.\] else 1", probe_source)


def test_probe_clean_shutdown_in_finally(probe_tree) -> None:
    """Node destruction and rclpy shutdown must run from a finally block."""
    finally_nodes = [
        node for node in ast.walk(probe_tree) if isinstance(node, ast.Try) and node.finalbody
    ]
    assert finally_nodes
    for node in finally_nodes:
        attrs = {
            sub.func.attr
            for sub in ast.walk(ast.Module(body=node.finalbody, type_ignores=[]))
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
        }
        assert {"destroy_node", "shutdown"} <= attrs


# ----------------------------------------------------------------- smoke


def test_smoke_exists_and_is_bash() -> None:
    """The smoke script must exist with a bounded shebang header."""
    assert SMOKE_PATH.is_file()
    text = SMOKE_PATH.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in text


def test_smoke_runs_one_container_with_the_mounted_ws_and_image(
    smoke_source: str,
) -> None:
    """docker run --rm, unique name, ./ros_ws mounted, pinned image."""
    assert "docker run --rm" in smoke_source
    assert "rocobrick-p1-moveit-offline-$$" in smoke_source
    assert '-v "$WS_DIR:/workspace/ros_ws"' in smoke_source
    assert 'IMAGE="${ROS_IMAGE:-roco-piper-ros:humble}"' in smoke_source
    assert "bash -s" in smoke_source


def test_smoke_passes_no_fastdds_profile(smoke_source: str) -> None:
    """Container-only default DDS: no Fast DDS profile may be passed."""
    assert "FASTRTPS_DEFAULT_PROFILES_FILE" not in smoke_source
    assert "fastdds" not in smoke_source.lower()
    # No published port either: everything shares the container's own DDS.
    assert "-p $" not in smoke_source
    assert "-p ${" not in smoke_source


def test_smoke_starts_feeder_and_move_group_offline(smoke_source: str) -> None:
    """Feeder in background + launch with use_sim_time:=false."""
    assert "p1_fake_joint_states.py" in smoke_source
    assert "nohup" in smoke_source
    assert "ros2 launch rocobrick_piper_l_moveit_config move_group.launch.py" in smoke_source
    assert "use_sim_time:=false" in smoke_source
    assert ">/tmp/p1-feeder.log 2>&1 &" in smoke_source
    assert ">/tmp/p1-move_group.log 2>&1 &" in smoke_source


def test_smoke_waits_then_probes_then_lists_controllers(smoke_source: str) -> None:
    """Bounded precondition wait, probe run, diagnostic controller listing."""
    assert "plan_kinematic_path" in smoke_source
    assert "p1_moveit_offline_probe.py" in smoke_source
    assert "ros2 control list_controllers" in smoke_source
    assert "DIAGNOSTIC ONLY" in smoke_source
    assert "must NOT fail this smoke" in smoke_source


def test_smoke_teardown_and_cleanup_are_always_run(smoke_source: str) -> None:
    """Background jobs are killed; a trap always removes the container."""
    assert "trap cleanup EXIT" in smoke_source
    assert "docker rm -f" in smoke_source
    assert 'kill -INT "$LAUNCH_PID"' in smoke_source
    assert 'kill -KILL "$FEEDER_PID"' in smoke_source


def test_smoke_writes_the_aggregate_artifact(smoke_source: str) -> None:
    """Aggregate embeds probe JSON + controllers + bounded log tails."""
    assert (
        'AGGREGATE="$OUT_DIR/moveit-offline-planning.json"' in smoke_source
        and "artifacts/phase1" in smoke_source
    )
    assert "controllers_listing" in smoke_source
    assert "feeder_log_tail" in smoke_source
    assert "move_group_log_tail" in smoke_source
    assert "tail -c 20000" in smoke_source


def test_smoke_exit_zero_only_when_probe_passed(smoke_source: str) -> None:
    """Every step is timeout-bounded; aggregate passed gates the exit."""
    assert smoke_source.count("timeout ") >= 2
    assert '"probe_passed"' in smoke_source
    assert '"valid_goal_planned"' in smoke_source
    assert '"wide_goal_never_exceeds_limits"' in smoke_source
    assert 'report["passed"] = all(checks.values())' in smoke_source
    assert "p1 moveit offline smoke FAILED" in smoke_source


def test_no_gpu_or_isaac_tokens_required(smoke_source: str) -> None:
    """The smoke is CPU/container-only: no GPU or Isaac launch tokens."""
    lowered = smoke_source.lower()
    for token in ("--gpus", "nvidia", "isaac-ros-env", "p05_host_bridge", "omni.kit"):
        assert token not in lowered, token
