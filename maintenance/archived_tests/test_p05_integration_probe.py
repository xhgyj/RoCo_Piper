"""Offline structural tests for the P0.5 ROS container integration probe.

The archived probe (maintenance/archived_tools/tools/p05_integration_probe.py) runs inside the rootless ROS
container, where rclpy exists; the offline suite here therefore only parses
its source and AST (never imports it). Pinned contract: subscribe /clock and
/isaac_joint_states, verify the complete 8-DOF names and an increasing clock,
publish a safe named JointState delta on /isaac_joint_command for a subset of the
six arm joints clamped inside the execution limits, and verify commanded
movement plus bounded idling of every uncommanded joint — all bounded and
with a machine-readable result.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
PROBE_PATH = ROOT / "maintenance" / "archived_tools" / "tools" / "p05_integration_probe.py"

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
GRIPPER_JOINT_NAMES = {"gripper_joint1", "gripper_joint2"}
FORBIDDEN_MODULES = {"socket", "grpc", "zmq", "isaacsim", "omni"}


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

    Returns:
        The probe script AST.
    """
    return ast.parse(probe_source)


@pytest.fixture(scope="module")
def probe_module_ns(probe_source: str) -> dict:
    """Evaluate only the probe's constant assignments, without imports.

    Args:
        probe_source: The probe script source text.

    Returns:
        A dict of the module-level constant names to their values.
    """
    namespace: dict = {}
    for node in ast.parse(probe_source).body:
        if isinstance(node, ast.Assign) and all(
            isinstance(target, ast.Name) and target.id.isupper()
            for target in node.targets
        ):
            code = compile(ast.Module(body=[node], type_ignores=[]), "<probe>", "exec")
            exec(code, {"__builtins__": __builtins__}, namespace)
    return namespace


def test_probe_exists_and_is_offline_import_free() -> None:
    """The probe must exist and only need container-side ROS modules."""
    assert PROBE_PATH.is_file()


def test_imports_are_container_ros_only(probe_tree) -> None:
    """Only rclpy, rosgraph_msgs and sensor_msgs may be imported."""
    imported = set()
    for node in ast.walk(probe_tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert {"rclpy", "rosgraph_msgs", "sensor_msgs"} <= imported
    assert imported.isdisjoint(FORBIDDEN_MODULES)


def test_expected_dof_names_are_complete(probe_module_ns) -> None:
    """The probe must demand the complete Piper L 8-DOF state, in order."""
    assert list(probe_module_ns["EXPECTED_DOF_NAMES"]) == EXPECTED_DOF_NAMES


def test_command_is_a_small_safe_arm_subset(probe_module_ns) -> None:
    """Command one or two arm joints with a small, verifiable delta."""
    command_names = list(probe_module_ns["COMMAND_JOINT_NAMES"])
    assert 1 <= len(command_names) <= 2
    assert set(command_names) <= ARM_JOINT_NAMES
    assert set(command_names).isdisjoint(GRIPPER_JOINT_NAMES)
    delta = probe_module_ns["COMMAND_DELTA"]
    move_min = probe_module_ns["COMMAND_MOVE_MIN"]
    idle_max = probe_module_ns["IDLE_DELTA_MAX"]
    assert 0.0 < move_min <= delta <= 0.3
    assert idle_max >= delta  # idling noise tolerance exceeds the command size


def test_execution_arm_limits_are_pinned(probe_module_ns) -> None:
    """Targets must honor the narrow USD limits at joint4 and joint5."""
    limits = probe_module_ns["JOINT_LIMITS"]
    assert set(limits) == ARM_JOINT_NAMES
    for lower, upper in limits.values():
        assert lower < upper
    assert limits["joint1"] == (pytest.approx(-2.6179938), pytest.approx(2.6179938))
    assert limits["joint4"] == (pytest.approx(-1.7453292), pytest.approx(1.7453292))
    assert limits["joint5"] == (pytest.approx(-1.2217304), pytest.approx(1.2217304))


def test_targets_are_clamped_inside_limits(probe_source: str) -> None:
    """safe_targets must clamp the delta away from the joint limits."""
    start = probe_source.index("def safe_targets")
    body = probe_source[start : probe_source.index("def main")]
    assert "lower + COMMAND_LIMIT_MARGIN" in body
    assert "upper - COMMAND_LIMIT_MARGIN" in body
    assert "min(max(" in body


def test_subscribes_clock_and_joint_states(probe_source: str) -> None:
    """The probe subscribes /clock and /isaac_joint_states and checks both."""
    assert 'TOPIC_CLOCK = "/clock"' in probe_source
    assert 'TOPIC_JOINT_STATES = "/isaac_joint_states"' in probe_source
    assert "create_subscription" in probe_source
    assert "rosgraph_msgs" in probe_source
    assert probe_source.count("Clock") >= 2
    assert "clock_increasing" in probe_source
    assert "zip(clock_values, clock_values[1:])" in probe_source


def test_publishes_named_joint_state_command(probe_source: str) -> None:
    """The command is a named JointState on /isaac_joint_command (singular)."""
    assert 'TOPIC_JOINT_COMMANDS = "/isaac_joint_command"' in probe_source
    assert "create_publisher" in probe_source
    assert "command.name = list(COMMAND_JOINT_NAMES)" in probe_source
    assert "command.position" in probe_source
    assert "positions_by_name" in probe_source


def test_verifies_commanded_move_and_idle_bound(probe_source: str) -> None:
    """Result must gate on commanded movement and bounded idling."""
    assert '"commanded_moved"' in probe_source
    assert '"idle_within_limit"' in probe_source
    assert "COMMAND_MOVE_MIN" in probe_source
    assert "IDLE_DELTA_MAX" in probe_source
    assert '"command_published"' in probe_source
    assert (
        'result["passed"] = bool(' in probe_source
        and 'result["command_published"]' in probe_source
    )


def test_run_is_fully_bounded(probe_tree, probe_source: str) -> None:
    """Both phases need wall-clock deadlines; args must be positive-checked."""
    options = {
        node.args[0].value
        for node in ast.walk(probe_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }
    assert {"--timeout", "--drive-timeout"} <= options
    assert probe_source.count("time.monotonic() +") >= 2
    assert probe_source.count("while time.monotonic() <") >= 2
    assert "must be positive" in probe_source


def test_machine_readable_result_and_exit_code(probe_source: str) -> None:
    """One JSON line with the pinned schema decides the exit code."""
    assert 'SCHEMA = "rocobrick-p05-integration-probe/v1"' in probe_source
    assert 'PHASE = "P0.5"' in probe_source
    assert 'print(json.dumps(result, sort_keys=True), flush=True)' in probe_source
    assert re.search(r"return 0 if result\[.passed.\] else 1", probe_source)


def test_clean_shutdown_in_finally(probe_tree) -> None:
    """Node destruction and rclpy shutdown must run from a finally block."""
    finally_nodes = [
        node
        for node in ast.walk(probe_tree)
        if isinstance(node, ast.Try) and node.finalbody
    ]
    assert finally_nodes
    for node in finally_nodes:
        attrs = {
            sub.func.attr
            for sub in ast.walk(ast.Module(body=node.finalbody, type_ignores=[]))
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
        }
        assert {"destroy_node", "shutdown"} <= attrs
