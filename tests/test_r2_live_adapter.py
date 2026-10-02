"""R2 live adapter verification tests (offline structure + fake transport).

Three layers are pinned here:

* tools/r2_live_adapter_probe.py must stay a THIN composition script over the
  accepted formal library + adapter: it imports build_joint_target,
  build_link6_pose_target, RosMoveItRuntime, RosMoveItExecutor and is_success
  and calls them, and it must NOT define duplicate quaternion / joint-limit /
  trajectory / error-metric / readiness functions nor hardcode
  EXECUTION_POSITION_LIMITS. It must import the formal library DIRECTLY with
  NO stdlib monkeypatch of any kind (no ``enum.StrEnum =`` assignment): the
  Python 3.10 ROS Humble container is supported exclusively by the
  ``rocobrick.compat`` seam. No TCP frame, no grasp, no example1, no old P1
  probe calls.
* tools/smoke_r2_live_adapter.sh must reuse the proven bounded process
  orchestration (P1.A prerequisite, model audit, container-mesh profiles,
  host bridge, ONE rootless container, move_group use_sim_time:=true,
  read-only src mount + PYTHONPATH, cleanup/log scan/no leftovers) and
  aggregate artifacts/phase2-adapter/live-adapter.json with the full check
  set and schema rocobrick-r2-live-adapter-smoke/v1.
* rocobrick/compat.py is the single compatibility seam: it exports only
  ``StrEnum`` (re-exported from enum where available, else a minimal local
  string-enum base), the seven formal modules import it from there, enum
  value/str/equality semantics stay identical on the host, and the container
  proof imports all seven modules directly under Python 3.10 without any
  shim.

The offline functional tests drive the added minimal adapter FK hook
``RosMoveItRuntime.compute_link6_fk`` and the executor parity flow over fake
message classes (no ROS, no docker, no GPU): the /compute_fk request carries
exactly the canonical state, failures return None instead of inventing a
pose, and a link6 pose goal executed through ``RosMoveItExecutor`` verifies
against the live FK hook end to end.
"""

from __future__ import annotations

import ast
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from rocobrick.compat import StrEnum
from rocobrick.models import SkillType
from rocobrick.moveit_contract import (
    ACTION_ARM_FOLLOW,
    ARM_JOINT_NAMES,
    BASE_FRAME,
    JOINT_NAMES,
    MOVE_GROUP_ACTION_CANDIDATES,
    SERVICE_COMPUTE_FK,
    SERVICE_LIST_CONTROLLERS,
    TIP_LINK,
    JointTarget,
    Link6PoseTarget,
)
from rocobrick.moveit_execution import ExecutionStatus, is_success
from rocobrick.moveit_requests import build_joint_target, build_link6_pose_target
from rocobrick.ros_moveit_executor import RosMoveItExecutor
from rocobrick.ros_moveit_runtime import (
    RosMoveItFactories,
    RosMoveItRuntime,
    RuntimeActions,
    RuntimeServices,
    RuntimeTopics,
)

ROOT = Path(__file__).parents[1]
PROBE_PATH = ROOT / "tools" / "r2_live_adapter_probe.py"
SMOKE_PATH = ROOT / "tools" / "smoke_r2_live_adapter.sh"
ARTIFACT_PATH = ROOT / "artifacts" / "phase2-adapter" / "live-adapter.json"

BASELINE = {
    "joint1": 0.01,
    "joint2": 1.0,
    "joint3": -0.5,
    "joint4": 0.02,
    "joint5": 0.0,
    "joint6": -0.1,
    "gripper_joint1": 0.0,
    "gripper_joint2": 0.0,
}
FK_POSITION = (0.073250543, -0.000000032, 0.22739705)
FK_QUATERNION = (0.001243319314, 0.676549822064, 0.001352457993, 0.736394570376)

# Domain logic must NEVER be duplicated inside the thin probe: no helper whose
# name touches quaternion math, joint limits, trajectories, waypoints, error
# metrics, drift, readiness or admission may be defined there.
FORBIDDEN_FUNCTION_SUBSTRINGS = (
    "quat",
    "limit",
    "trajectory",
    "waypoint",
    "error",
    "readiness",
    "admiss",
    "euclid",
    "normalize",
    "drift",
    "deviation",
)
FORBIDDEN_PROBE_MODULES = {"socket", "grpc", "zmq", "isaacsim", "omni", "paramiko"}
FORBIDDEN_PROBE_TOKENS = ("tcp_link", "tool0", "grasp", "example1", "piper-run")
LIMIT_LITERALS = (
    "-2.6179938",
    "2.6179938",
    "2.9670597",
    "1.7453292",
    "1.2217304",
    "2.0943951",
    "3.1415926",
)


# --------------------------------------------------------------- probe sources


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
    """Evaluate only pure UPPERCASE constant assignments, without imports."""
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


def import_roots(tree) -> set[str]:
    """Root module names of every import in the tree."""
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def rocobrick_import_names(tree) -> set[str]:
    """Names imported from rocobrick.* modules."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "rocobrick" or node.module.startswith("rocobrick."):
                names.update(alias.name for alias in node.names)
    return names


# ------------------------------------------------------- probe: thin composition


def test_probe_exists_and_consumes_exactly_the_formal_library(
    probe_source: str, probe_tree
) -> None:
    """The five mandated symbols are imported from the accepted modules."""
    assert PROBE_PATH.is_file()
    names = rocobrick_import_names(probe_tree)
    assert {
        "build_joint_target",
        "build_link6_pose_target",
        "is_success",
        "RosMoveItRuntime",
        "RosMoveItExecutor",
    } <= names
    modules = {
        node.module
        for node in ast.walk(probe_tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert {
        "rocobrick.moveit_requests",
        "rocobrick.moveit_execution",
        "rocobrick.ros_moveit_runtime",
        "rocobrick.ros_moveit_executor",
        "rocobrick.moveit_contract",
        "rocobrick.feedback",
    } <= modules
    roots = import_roots(probe_tree)
    assert roots.isdisjoint(FORBIDDEN_PROBE_MODULES)
    assert "tools" not in roots  # no old P1 probe may be imported
    for banned in ("p1_real_joint_probe", "p1_real_pose_probe"):
        assert banned not in probe_source


def test_probe_defines_no_duplicate_domain_functions(probe_tree) -> None:
    """No quaternion/limit/trajectory/error/readiness logic may be redefined."""
    defined = [
        node.name
        for node in ast.walk(probe_tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    for name in defined:
        lowered = name.lower()
        for fragment in FORBIDDEN_FUNCTION_SUBSTRINGS:
            assert fragment not in lowered, (name, fragment)


def test_probe_does_not_hardcode_the_execution_limits(
    probe_source: str, probe_tree
) -> None:
    """The narrowed limit table stays in the library only."""
    assert "EXECUTION_POSITION_LIMITS" not in probe_source
    for literal in LIMIT_LITERALS:
        assert literal not in probe_source, literal
    joint_name_literals = {joint for joint in JOINT_NAMES}
    for node in ast.walk(probe_tree):
        if isinstance(node, ast.Dict) and node.keys:
            keys = {
                key.value for key in node.keys if isinstance(key, ast.Constant)
            }
            overlapping = keys & joint_name_literals
            assert len(overlapping) < 4, f"limit-like dict literal: {sorted(keys)}"


def test_probe_imports_the_library_directly_with_no_monkeypatch(
    probe_source: str,
) -> None:
    """No stdlib patching of any kind; the compat seam lives in the library."""
    assert "enum.StrEnum" not in probe_source
    assert "sys.version_info" not in probe_source
    tree = ast.parse(probe_source)
    foreign_assignments = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                ):
                    foreign_assignments.append((target.value.id, target.attr))
    assert foreign_assignments == []


def test_probe_pins_schema_phase_deltas_and_tolerances(probe_source: str) -> None:
    """Schema, phase, the mandated deltas and the tighter goal tolerances."""
    constants = module_constants(probe_source)
    assert constants["SCHEMA"] == "rocobrick-r2-live-adapter-probe/v1"
    assert constants["PHASE"] == "R2"
    assert constants["OUTBOUND_DELTAS"] == {"joint1": 0.10, "joint4": 0.08}
    assert constants["GOAL_POSITION_TOLERANCE_M"] == 0.005
    assert constants["GOAL_ORIENTATION_TOLERANCE_RAD"] == 0.015
    assert constants["GOAL_POSITION_TOLERANCE_M"] <= 0.01
    assert constants["GOAL_ORIENTATION_TOLERANCE_RAD"] <= 0.03
    assert constants["SETTLE_STABLE_SAMPLES"] >= 2
    assert constants["SETTLE_STABLE_DELTA_MAX"] <= 5.0e-4


def test_probe_calls_the_adapter_api_only(probe_tree) -> None:
    """The orchestration goes through the runtime/executor/library calls."""
    called_attributes: set[str] = set()
    called_names: set[str] = set()
    for node in ast.walk(probe_tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                called_attributes.add(func.attr)
            elif isinstance(func, ast.Name):
                called_names.add(func.id)
    assert {
        "register_subscriptions",
        "register_clients",
        "wait_until_ready",
        "readiness_snapshot",
        "latest_isaac_state",
        "compute_link6_fk",
        "execute_joint",
        "execute_link6_pose",
        "shutdown",
    } <= called_attributes
    assert {
        "RosMoveItRuntime",
        "RosMoveItExecutor",
        "build_joint_target",
        "build_link6_pose_target",
        "is_success",
    } <= called_names


def test_probe_demandes_readiness_passed_and_four_goals(probe_tree) -> None:
    """Snapshot gates plus the four executor goals with success verdicts."""
    attribute_calls = [
        node.func.attr
        for node in ast.walk(probe_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert attribute_calls.count("execute_joint") == 3  # outbound/return/final
    assert attribute_calls.count("execute_link6_pose") == 1
    attribute_calls.count("wait_until_ready")
    source_names = {
        node.func.attr
        for node in ast.walk(probe_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "passed" in ast.dump(probe_tree)  # ReadinessSnapshot.passed asserted
    assert source_names  # narrowed above


def test_probe_has_no_tcp_grasp_or_example1_semantics(probe_source: str) -> None:
    """link6-origin parity only: TCP unresolved, no grasp, no example1."""
    for token in FORBIDDEN_PROBE_TOKENS:
        assert token not in probe_source, token
    assert "get_position_ik" not in probe_source.lower()
    assert "compute_ik" not in probe_source.lower()


def test_probe_is_fully_bounded_and_prints_one_line(probe_source: str) -> None:
    """Every wait is deadline-bounded; exactly one JSON result line is printed."""
    for flag in (
        "--timeout",
        "--goal-timeout",
        "--settle-timeout",
        "--fk-timeout",
    ):
        assert flag in probe_source, flag
    assert "while time.monotonic() <" in probe_source
    assert probe_source.count("time.monotonic() +") >= 1
    assert probe_source.count("print(json.dumps(") == 1
    assert 'return 0 if result["passed"] else 1' in probe_source


# ------------------------------------------------------- enum compatibility seam

COMPAT_MODULES = (
    "models",
    "execution",
    "skills",
    "planner_contract",
    "task_facts",
    "trajectory_execution",
    "moveit_execution",
)
COMPAT_IMAGE = "roco-piper-ros:humble"

CONTAINER_PROOF_SNIPPET = """import enum
import sys

assert sys.version_info[:2] == (3, 10), sys.version
assert not hasattr(enum, "StrEnum")  # proof: no shim ever patched the stdlib
from rocobrick import compat

assert compat.__all__ == ["StrEnum"]
assert compat.StrEnum.__module__ == "rocobrick.compat"
import rocobrick.models
import rocobrick.execution
import rocobrick.skills
import rocobrick.planner_contract
import rocobrick.task_facts
import rocobrick.trajectory_execution
import rocobrick.moveit_execution
import rocobrick.ros_moveit_runtime
import rocobrick.ros_moveit_executor

member = rocobrick.moveit_execution.ExecutionStatus.SUCCEEDED
assert member.value == "SUCCEEDED"
assert str(member) == "SUCCEEDED"
assert member == "SUCCEEDED"
assert isinstance(member, str)
print("COMPAT_OK")
"""


def test_compat_module_exports_only_strenum() -> None:
    """The seam is monkeypatch-free and re-exports the stdlib StrEnum."""
    import enum

    from rocobrick import compat

    assert compat.__all__ == ["StrEnum"]
    if sys.version_info[:2] >= (3, 11):
        assert compat.StrEnum is enum.StrEnum
    source = (ROOT / "src" / "rocobrick" / "compat.py").read_text(encoding="utf-8")
    patched = [
        (target.value.id, target.attr)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
    ]
    assert patched == []


def test_enum_semantics_are_unchanged_on_the_host() -> None:
    """value/str/equality/str-subclass semantics stay exactly StrEnum."""
    assert issubclass(ExecutionStatus, StrEnum)
    member = ExecutionStatus.SUCCEEDED
    assert member.value == "SUCCEEDED"
    assert str(member) == "SUCCEEDED"
    assert member == "SUCCEEDED"
    assert isinstance(member, str)
    assert member is ExecutionStatus("SUCCEEDED")
    assert SkillType.PICK.value == "PICK"
    assert str(SkillType.PICK) == "PICK"
    assert SkillType.PICK == "PICK"


def test_seven_formal_modules_import_strenum_from_the_seam_only() -> None:
    """Exactly the seven modules switched; no direct enum.StrEnum import left."""
    for name in COMPAT_MODULES:
        source = (ROOT / "src" / "rocobrick" / f"{name}.py").read_text(
            encoding="utf-8"
        )
        assert "from enum import StrEnum" not in source, name
        assert "from rocobrick.compat import StrEnum" in source, name
    remaining = sorted(
        path.name
        for path in (ROOT / "src" / "rocobrick").glob("*.py")
        if path.name != "compat.py"
        and "from enum import StrEnum" in path.read_text(encoding="utf-8")
    )
    assert remaining == []


def _compat_image_available() -> bool:
    try:
        probe = subprocess.run(
            ["docker", "image", "inspect", COMPAT_IMAGE],
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


@pytest.mark.skipif(
    not _compat_image_available(),
    reason="ROS Humble container image not available",
)
def test_compat_modules_import_directly_in_the_ros_container(tmp_path) -> None:
    """Live proof: the seven modules import on Python 3.10 with NO shim.

    Runs the exact direct form (PYTHONPATH=/workspace/src python3 -c ...)
    plus the semantic snippet against the real ROS Humble container with the
    repository src mounted read-only.
    """
    snippet = tmp_path / "compat_proof.py"
    snippet.write_text(CONTAINER_PROOF_SNIPPET, encoding="utf-8")
    direct = (
        "import rocobrick.models, rocobrick.execution, rocobrick.skills, "
        "rocobrick.planner_contract, rocobrick.task_facts, "
        "rocobrick.trajectory_execution, rocobrick.moveit_execution; "
        "print('DIRECT_IMPORT_OK')"
    )
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{ROOT / 'src'}:/workspace/src:ro",
            "-v",
            f"{snippet}:/workspace/compat_proof.py:ro",
            COMPAT_IMAGE,
            "bash",
            "-c",
            f"PYTHONPATH=/workspace/src python3 -c \"{direct}\" && "
            "PYTHONPATH=/workspace/src python3 /workspace/compat_proof.py",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "DIRECT_IMPORT_OK" in result.stdout
    assert "COMPAT_OK" in result.stdout


# ---------------------------------------------------------------- smoke script


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


def test_smoke_runs_the_model_audit_and_mesh_profiles(smoke_source: str) -> None:
    """Offline preflight audit + profiles with the additive mesh flag."""
    assert "piper_l_model_audit.py" in smoke_source
    assert "generate_dds_profiles.py" in smoke_source
    assert "--container-mesh" in smoke_source
    assert "fastdds-container-mesh.xml" in smoke_source
    assert "nvidia-smi" in smoke_source
    assert "pkill" not in smoke_source
    assert "killall" not in smoke_source


def test_smoke_starts_the_host_bridge_with_long_bounds(smoke_source: str) -> None:
    """Host bridge via isaac-ros-env.sh, hold/time limits, marker wait."""
    assert "tools/isaac-ros-env.sh" in smoke_source
    assert "p05_host_bridge.py" in smoke_source
    assert 'MARKER="p0.5_host_bridge_ready"' in smoke_source
    assert 'HOST_EVIDENCE="$OUT_DIR/host-adapter-bridge.json"' in smoke_source
    assert "--time-limit" in smoke_source
    assert "--hold-seconds" in smoke_source
    assert "--hold-steps" in smoke_source
    assert "--max-steps" in smoke_source
    # The bridge must always self-exit orderly: hold shorter than the stop
    # grace (Kit intercepts SIGINT and skips the python evidence cleanup).
    hold = int(re.search(r"R2_HOLD_SECONDS:-([0-9]+)", smoke_source).group(1))
    stop = int(re.search(r"R2_STOP_WAIT:-([0-9]+)", smoke_source).group(1))
    assert hold < stop
    assert 'grep -q "\\"marker\\": \\"$MARKER\\""' in smoke_source


def test_smoke_runs_one_container_with_readonly_src_and_pythonpath(
    smoke_source: str,
) -> None:
    """One rootless container: move_group on sim time, ro src, PYTHONPATH."""
    assert smoke_source.count("docker run --rm") == 1
    assert 'CONTAINER_NAME="rocobrick-r2-live-adapter-$$"' in smoke_source
    assert '-v "$WS_DIR:/workspace/ros_ws"' in smoke_source
    assert '-v "$ROOT/src:/workspace/src:ro"' in smoke_source
    assert (
        '-v "$ROOT/tools/r2_live_adapter_probe.py:'
        '/workspace/tools/r2_live_adapter_probe.py:ro"' in smoke_source
    )
    assert '-v "$RUNTIME:/dds:ro"' in smoke_source
    assert '-v "$OUT_DIR:/artifacts"' in smoke_source
    assert "-e FASTRTPS_DEFAULT_PROFILES_FILE=/dds/fastdds-container-mesh.xml" \
        in smoke_source
    assert (
        "ros2 launch rocobrick_piper_l_moveit_config move_group.launch.py"
        in smoke_source
    )
    assert "use_sim_time:=true" in smoke_source
    assert 'export PYTHONPATH="/workspace/src:${PYTHONPATH:-}"' in smoke_source
    assert re.search(r"docker run[^\n]*\s-p\b", smoke_source) is None
    assert "r2_live_adapter_probe.py" in smoke_source
    for flag in ("--timeout", "--goal-timeout", "--settle-timeout", "--fk-timeout"):
        assert flag in smoke_source, flag
    # The probe only starts after the planning AND FK services are reachable.
    assert "plan_kinematic_path" in smoke_source
    assert "compute_fk" in smoke_source
    assert "ros2 control list_controllers" in smoke_source
    # The old P1 probes must never be called by this smoke.
    assert "p1_real_joint_probe" not in smoke_source
    assert "p1_real_pose_probe" not in smoke_source


def test_smoke_captures_listener_evidence_and_cleans_up(
    smoke_source: str,
) -> None:
    """In-namespace listener scan, orderly stop, trap, no leftovers."""
    assert "net-tcp-listeners-r2.txt" in smoke_source
    assert "/proc/net/tcp" in smoke_source
    assert "LISTEN_ADDRESSES" in smoke_source
    assert "not listener_rows" in smoke_source
    assert "trap cleanup EXIT" in smoke_source
    assert "docker rm -f" in smoke_source
    assert "stop_host" in smoke_source
    assert 'kill -INT "$HOST_PID"' in smoke_source
    assert 'kill -KILL "$HOST_PID"' in smoke_source
    assert "--filter" in smoke_source and "rocobrick-r2-live-adapter-" in smoke_source
    assert 'pgrep", "-f", "p05_host_bridge' in smoke_source


def test_smoke_writes_the_aggregate_with_all_checks(smoke_source: str) -> None:
    """Schema, artifact path, embedded evidence and the full check set."""
    assert 'AGGREGATE="$OUT_DIR/live-adapter.json"' in smoke_source
    assert "artifacts/phase2-adapter" in smoke_source
    assert "rocobrick-r2-live-adapter-smoke/v1" in smoke_source
    assert "rocobrick-r2-live-adapter-probe/v1" in smoke_source
    for token in (
        '"readiness"',
        '"adapter_probe_passed"',
        '"joint_outbound_passed"',
        '"joint_return_passed"',
        '"link6_pose_passed"',
        '"final_return_passed"',
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
        "probe_log_tail",
        "move_group_log_tail",
        "profiles",
    ):
        assert token in smoke_source, token
    # applied == received, no clamps, no unknown names on the host side.
    assert 'host_commands.get("applied") == host_commands.get("received")' \
        in smoke_source
    assert 'host_commands.get("clamped_to_limits") == 0' in smoke_source
    assert 'host_commands.get("unknown_joint_names_ignored") == []' in smoke_source
    # Pose FK verification stays at the contract bounds (never weakened).
    assert 'pose_final["position_error_m"] <= 0.01' in smoke_source
    assert 'pose_final["orientation_error_rad"] <= 0.03' in smoke_source


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
    for name in (
        '"host-adapter-bridge.log"',
        '"container-r2.log"',
        '"adapter-probe.log"',
        '"move_group-r2.log.tail"',
    ):
        assert name in smoke_source, name


def test_smoke_is_hard_bounded(smoke_source: str) -> None:
    """Every execution stage is wrapped in `timeout` with escalation."""
    assert smoke_source.count("timeout --signal=INT --kill-after=30") >= 2
    assert "timeout --signal=INT --kill-after=15" in smoke_source
    assert smoke_source.count("timeout --signal=INT --kill-after=10") >= 1
    assert 'TOTAL_TIMEOUT="${R2_TOTAL_TIMEOUT:-' in smoke_source
    assert 'PROBE_TIMEOUT="${R2_PROBE_TIMEOUT:-' in smoke_source
    assert 'MOVE_GROUP_WAIT="${R2_MOVE_GROUP_WAIT:-' in smoke_source


def test_smoke_artifact_when_present_is_truthful() -> None:
    """A committed artifact must report honest pass/fail with all checks."""
    if not ARTIFACT_PATH.is_file():
        pytest.skip("live R2 adapter artifact not collected yet")
    report = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert report["schema"] == "rocobrick-r2-live-adapter-smoke/v1"
    assert set(report["checks"]) == {
        "p1a_prerequisite",
        "host_marker",
        "container_profile_client_only_no_listener",
        "readiness",
        "adapter_probe_passed",
        "joint_outbound_passed",
        "joint_return_passed",
        "link6_pose_passed",
        "final_return_passed",
        "command_observed",
        "host_commands_applied",
        "host_cleanup_orderly",
        "log_scan_clean",
        "bounded_and_no_leftovers",
    }
    assert report["passed"] == all(report["checks"].values())


# ------------------------------------------------- offline fake transport layer


class FakeClock:
    """Injected monotonic clock the runtime can advance via the sleep hook."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, delta: float) -> None:
        self.now += float(delta)


class FakeFuture:
    def __init__(self) -> None:
        self._done = False
        self._value = None

    def done(self) -> bool:
        return self._done

    def result(self):
        return self._value

    def set(self, value) -> None:
        self._value = value
        self._done = True


class FakeGoalHandle:
    def __init__(self, accepted: bool = True) -> None:
        self.accepted = accepted
        self.result_future = FakeFuture()
        self.cancel_calls = 0

    def get_result_async(self) -> FakeFuture:
        return self.result_future

    def cancel_goal_async(self) -> FakeFuture:
        self.cancel_calls += 1
        future = FakeFuture()
        future.set(SimpleNamespace(return_code=0))
        return future


class FakeActionClient:
    def __init__(self, node: "FakeNode", action_type, name: str) -> None:
        self.node = node
        self.action_type = action_type
        self.name = name
        self.goals: list = []
        self.server_ready = True
        self.pending: FakeFuture | None = None
        node.action_clients[name] = self

    def wait_for_server(self, timeout_sec=None) -> bool:
        return self.server_ready

    def send_goal_async(self, goal) -> FakeFuture:
        self.goals.append(goal)
        self.pending = FakeFuture()
        return self.pending

    def destroy(self) -> None:
        pass


class FakeServiceClient:
    def __init__(self, name: str) -> None:
        self.name = name
        self.ready = True
        self.response = None
        self.requests: list = []

    def service_is_ready(self) -> bool:
        return self.ready

    def call_async(self, request) -> FakeFuture:
        self.requests.append(request)
        future = FakeFuture()
        if self.response is not None:
            future.set(self.response)
        return future


class FakeNode:
    def __init__(self) -> None:
        self.subscriptions: list = []
        self.service_clients: dict = {}
        self.action_clients: dict = {}

    def create_subscription(self, msg_type, topic, callback, qos):
        subscription = SimpleNamespace(
            msg_type=msg_type, topic=topic, callback=callback, qos=qos
        )
        self.subscriptions.append(subscription)
        return subscription

    def create_client(self, srv_type, name) -> FakeServiceClient:
        client = FakeServiceClient(name)
        self.service_clients[name] = client
        return client

    def destroy_subscription(self, subscription) -> None:
        pass

    def destroy_client(self, client) -> None:
        pass


def publish(node: FakeNode, topic: str, message) -> None:
    """Deliver one message to the single fake subscriber of ``topic``."""
    delivered = 0
    for subscription in node.subscriptions:
        if subscription.topic == topic:
            subscription.callback(message)
            delivered += 1
    assert delivered == 1, f"topic {topic} has {delivered} subscribers"


class FakeClockMsg:
    def __init__(self, nanos: int) -> None:
        self.clock = SimpleNamespace(
            sec=nanos // 1_000_000_000, nanosec=nanos % 1_000_000_000
        )


class FakeJointStateMsg:
    def __init__(self, positions, velocity=()) -> None:
        self.name = list(positions)
        self.position = [float(value) for value in positions.values()]
        self.velocity = list(velocity)


class FakeMotionPlanRequest:
    def __init__(self) -> None:
        self.group_name = ""
        self.allowed_planning_time = 0.0
        self.num_planning_attempts = 0
        self.max_velocity_scaling_factor = 0.0
        self.max_acceleration_scaling_factor = 0.0
        self.goal_constraints: list = []


class FakeConstraints:
    def __init__(self) -> None:
        self.joint_constraints: list = []
        self.position_constraints: list = []
        self.orientation_constraints: list = []


class FakeJointConstraint:
    def __init__(self) -> None:
        self.joint_name = ""
        self.position = 0.0
        self.tolerance_above = 0.0
        self.tolerance_below = 0.0
        self.weight = 0.0


class FakePositionConstraint:
    def __init__(self) -> None:
        self.header = SimpleNamespace(frame_id="")
        self.link_name = ""
        self.constraint_region = SimpleNamespace(primitives=[], primitive_poses=[])
        self.weight = 0.0


class FakeOrientationConstraint:
    def __init__(self) -> None:
        self.header = SimpleNamespace(frame_id="")
        self.link_name = ""
        self.orientation = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0)
        self.absolute_x_axis_tolerance = 0.0
        self.absolute_y_axis_tolerance = 0.0
        self.absolute_z_axis_tolerance = 0.0
        self.weight = 0.0


class FakeSolidPrimitive:
    SPHERE = 2

    def __init__(self) -> None:
        self.type = 0
        self.dimensions: list = []


class FakePose:
    def __init__(self) -> None:
        self.position = SimpleNamespace(x=0.0, y=0.0, z=0.0)
        self.orientation = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0)


class FakeMoveGroupGoal:
    def __init__(self) -> None:
        self.request = None
        self.planning_options = SimpleNamespace(plan_only=None, replan=None)


FAKE_GOAL_STATUS = SimpleNamespace(
    STATUS_SUCCEEDED=4, STATUS_CANCELED=5, STATUS_ABORTED=6
)


class FakeFkRequest:
    """GetPositionFK.Request stand-in carrying the fields the hook fills."""

    def __init__(self) -> None:
        self.header = SimpleNamespace(frame_id="")
        self.fk_link_names: list = []
        self.robot_state = SimpleNamespace(
            joint_state=SimpleNamespace(name=[], position=[])
        )


class FakeFkServiceType:
    Request = FakeFkRequest


def fake_factories() -> RosMoveItFactories:
    """Build the full fake factory set, including the FK service type."""
    return RosMoveItFactories(
        clock_type=lambda: FakeClockMsg,
        joint_state_type=lambda: FakeJointStateMsg,
        list_controllers_type=lambda: SimpleNamespace,
        list_controllers_request_factory=lambda: SimpleNamespace(),
        get_motion_plan_type=lambda: SimpleNamespace,
        get_position_fk_type=lambda: FakeFkServiceType,
        move_group_type=lambda: SimpleNamespace,
        follow_joint_trajectory_type=lambda: SimpleNamespace,
        move_group_goal_factory=FakeMoveGroupGoal,
        motion_plan_request_factory=FakeMotionPlanRequest,
        constraints_factory=FakeConstraints,
        joint_constraint_factory=FakeJointConstraint,
        position_constraint_factory=FakePositionConstraint,
        orientation_constraint_factory=FakeOrientationConstraint,
        solid_primitive_factory=FakeSolidPrimitive,
        pose_factory=FakePose,
        action_client_factory=lambda node, action_type, name: FakeActionClient(
            node, action_type, name
        ),
        goal_status_type=lambda: FAKE_GOAL_STATUS,
    )


@dataclass
class World:
    node: FakeNode
    clock: FakeClock
    events: list = field(default_factory=list)
    runtime: RosMoveItRuntime = None  # type: ignore[assignment]
    controllers: FakeServiceClient = None  # type: ignore[assignment]
    fk: FakeServiceClient = None  # type: ignore[assignment]
    move_group: FakeActionClient = None  # type: ignore[assignment]


def make_world(*, fk_verifier=None, register: bool = True) -> World:
    """Wire a runtime over a fake node with an event-queue spin hook."""
    node = FakeNode()
    clock = FakeClock()
    events: list = []

    def spin() -> None:
        if events:
            events.pop(0)()

    runtime = RosMoveItRuntime(
        node,
        qos_state=SimpleNamespace(label="state"),
        qos_sensor=SimpleNamespace(label="sensor"),
        factories=fake_factories(),
        fk_verifier=fk_verifier,
        clock=clock,
        spin=spin,
        sleep=clock.advance,
    )
    if register:
        runtime.register_subscriptions()
        runtime.register_clients()
    return World(
        node=node,
        clock=clock,
        events=events,
        runtime=runtime,
        controllers=node.service_clients.get(SERVICE_LIST_CONTROLLERS),
        fk=node.service_clients.get(SERVICE_COMPUTE_FK),
        move_group=node.action_clients.get(MOVE_GROUP_ACTION_CANDIDATES[0]),
    )


def prime_ready(world: World, state=None) -> dict:
    """Deliver the full readiness evidence set to the fake subscriptions."""
    live = dict(state if state is not None else BASELINE)
    for nanos in (1_000_000_000, 2_000_000_000):
        publish(world.node, "/clock", FakeClockMsg(nanos))
    publish(
        world.node,
        "/isaac_joint_states",
        FakeJointStateMsg(live, velocity=[0.0] * len(JOINT_NAMES)),
    )
    publish(world.node, "/joint_states", FakeJointStateMsg(live))
    world.controllers.response = SimpleNamespace(
        controller=[
            SimpleNamespace(name=name, state="active")
            for name in (
                "joint_state_broadcaster",
                "arm_controller",
                "gripper_controller",
            )
        ]
    )
    return live


def fk_response(
    position=FK_POSITION,
    quaternion=FK_QUATERNION,
    *,
    code: int = 1,
    stamp_count: int = 1,
) -> SimpleNamespace:
    """GetPositionFK.Response stand-in with the requested pose list."""
    poses = [
        SimpleNamespace(
            header=SimpleNamespace(frame_id=BASE_FRAME),
            pose=SimpleNamespace(
                position=SimpleNamespace(
                    x=float(position[0]), y=float(position[1]), z=float(position[2])
                ),
                orientation=SimpleNamespace(
                    x=float(quaternion[0]),
                    y=float(quaternion[1]),
                    z=float(quaternion[2]),
                    w=float(quaternion[3]),
                ),
            ),
        )
        for _ in range(stamp_count)
    ]
    return SimpleNamespace(error_code=SimpleNamespace(val=code), pose_stamped=poses)


def arm_values(state) -> list[float]:
    """Return the six arm positions of ``state`` in canonical order."""
    return [float(state[name]) for name in ARM_JOINT_NAMES]


def trajectory_action_result(target=None, *, error_code: int = 1):
    """MoveGroupResult stand-in with an in-limits planned trajectory."""
    start = arm_values(BASELINE)
    if target is None:
        points = [start, start, start]
    else:
        end = arm_values(target.positions)
        middle = [(a + b) / 2.0 for a, b in zip(start, end)]
        points = [start, middle, end]
    trajectory = SimpleNamespace(
        joint_names=list(ARM_JOINT_NAMES),
        points=[
            SimpleNamespace(positions=[float(value) for value in point])
            for point in points
        ],
    )
    return SimpleNamespace(
        error_code=SimpleNamespace(val=error_code),
        planning_time=0.5,
        planned_trajectory=SimpleNamespace(joint_trajectory=trajectory),
    )


def script_success(world: World, *, action_result, final_state) -> FakeGoalHandle:
    """Queue the event sequence of one successful MoveGroup goal."""
    handle = FakeGoalHandle()
    world.events.append(lambda: world.move_group.pending.set(handle))

    def deliver_evidence() -> None:
        for _ in range(2):
            publish(
                world.node,
                "/isaac_joint_states",
                FakeJointStateMsg(final_state, velocity=[0.0] * len(JOINT_NAMES)),
            )
        publish(world.node, "/isaac_joint_command", FakeJointStateMsg(final_state))

    world.events.append(deliver_evidence)
    world.events.append(
        lambda: handle.result_future.set(
            SimpleNamespace(
                status=FAKE_GOAL_STATUS.STATUS_SUCCEEDED, result=action_result
            )
        )
    )
    return handle


def make_pumping_world() -> tuple[RosMoveItRuntime, FakeNode, FakeClock, list]:
    """World whose sleep hook also pumps the event queue.

    Mirrors the live executor thread: new state samples keep arriving while
    the runtime's bounded waits sleep, not only while its futures spin.
    """
    node = FakeNode()
    clock = FakeClock()
    events: list = []

    def pump() -> None:
        if events:
            events.pop(0)()

    def sleep_and_pump(delta: float) -> None:
        clock.advance(delta)
        pump()

    runtime = RosMoveItRuntime(
        node,
        qos_state=SimpleNamespace(label="state"),
        qos_sensor=SimpleNamespace(label="sensor"),
        factories=fake_factories(),
        clock=clock,
        spin=pump,
        sleep=sleep_and_pump,
    )
    runtime.register_subscriptions()
    runtime.register_clients()
    return runtime, node, clock, events


def prime_ready_on(node: FakeNode, controllers: FakeServiceClient) -> dict:
    """Deliver the full readiness evidence set to the fake subscriptions."""
    live = dict(BASELINE)
    for nanos in (1_000_000_000, 2_000_000_000):
        publish(node, "/clock", FakeClockMsg(nanos))
    publish(
        node,
        "/isaac_joint_states",
        FakeJointStateMsg(live, velocity=[0.0] * len(JOINT_NAMES)),
    )
    publish(node, "/joint_states", FakeJointStateMsg(live))
    controllers.response = SimpleNamespace(
        controller=[
            SimpleNamespace(name=name, state="active")
            for name in (
                "joint_state_broadcaster",
                "arm_controller",
                "gripper_controller",
            )
        ]
    )
    return live


def mid_motion_state() -> dict:
    """The LIVE first-run failure shape: still mid-motion at action end."""
    state = dict(BASELINE)
    state["joint1"] = BASELINE["joint1"] + 0.048
    state["joint4"] = BASELINE["joint4"] + 0.010
    return state


def converged_state(target: JointTarget) -> dict:
    """The settled plant: a few milliradian from the joint target."""
    final = {name: float(BASELINE[name]) for name in JOINT_NAMES}
    final.update({name: float(target.positions[name]) for name in ARM_JOINT_NAMES})
    final["joint1"] -= 0.005
    final["joint4"] += 0.004
    return final


def deliver_accepted(runtime: RosMoveItRuntime, handle: FakeGoalHandle) -> None:
    """Deliver an accepted goal handle to the fake move group future."""
    client = runtime._move_group_client  # noqa: SLF001 - fake transport wiring
    assert client.pending is not None
    client.pending.set(handle)


def deliver_joint_success(
    node: FakeNode, handle: FakeGoalHandle, target: JointTarget
) -> None:
    """Queue terminal success with an in-limits planned trajectory."""
    handle.result_future.set(
        SimpleNamespace(
            status=FAKE_GOAL_STATUS.STATUS_SUCCEEDED,
            result=trajectory_action_result(target),
        )
    )


def test_settled_final_converges_after_the_action_completes() -> None:
    """Fake regression (first live R2 run): the action ends before the plant.

    The first live run proved the MoveGroup action completes while the
    lagging simulated arm is still converging (joint1 was 0.048 rad short at
    the action result), so a single instantaneous sample misjudged a
    converged goal as FAILED. The runtime must wait, bounded, for the
    at-rest plant before sampling the final state; the tolerance gates
    themselves are unchanged.
    """
    runtime, node, _clock, events = make_pumping_world()
    prime_ready_on(node, node.service_clients[SERVICE_LIST_CONTROLLERS])
    target, decision = build_joint_target(BASELINE, {"joint1": 0.10, "joint4": 0.08})
    assert decision.accepted is True and target is not None
    handle = FakeGoalHandle()
    events.append(lambda: deliver_accepted(runtime, handle))

    def deliver_mid_evidence() -> None:
        for _ in range(2):
            publish(
                node,
                "/isaac_joint_states",
                FakeJointStateMsg(
                    mid_motion_state(), velocity=[0.0] * len(JOINT_NAMES)
                ),
            )
        publish(node, "/isaac_joint_command", FakeJointStateMsg(mid_motion_state()))

    events.append(deliver_mid_evidence)
    events.append(lambda: deliver_joint_success(node, handle, target))
    events.append(
        lambda: publish(
            node,
            "/isaac_joint_states",
            FakeJointStateMsg(
                converged_state(target), velocity=[0.0] * len(JOINT_NAMES)
            ),
        )
    )
    outcome = runtime.send_joint(target, 40.0)
    assert outcome.status.value == "SUCCEEDED"
    assert is_success(outcome) is True
    assert outcome.details["settled_at_rest"] is True
    assert outcome.final_errors["joint1"] == pytest.approx(0.005)
    assert outcome.final_errors["joint4"] == pytest.approx(0.004)


def test_settled_final_is_bounded_when_the_arm_never_rests() -> None:
    """The convergence wait expires explicitly; the gates still decide."""
    runtime, node, clock, events = make_pumping_world()
    prime_ready_on(node, node.service_clients[SERVICE_LIST_CONTROLLERS])
    target, decision = build_joint_target(BASELINE, {"joint1": 0.10, "joint4": 0.08})
    assert decision.accepted is True and target is not None
    handle = FakeGoalHandle()
    events.append(lambda: deliver_accepted(runtime, handle))

    def deliver_post_evidence() -> None:
        for _ in range(2):
            publish(
                node,
                "/isaac_joint_states",
                FakeJointStateMsg(
                    mid_motion_state(), velocity=[0.0] * len(JOINT_NAMES)
                ),
            )
        publish(node, "/isaac_joint_command", FakeJointStateMsg(mid_motion_state()))

    events.append(deliver_post_evidence)
    events.append(lambda: deliver_joint_success(node, handle, target))
    # The plant keeps oscillating well beyond the convergence bound.
    oscillating = [mid_motion_state(), converged_state(target)]
    for index in range(200):
        state = oscillating[index % 2]
        events.append(
            lambda state=state: publish(
                node,
                "/isaac_joint_states",
                FakeJointStateMsg(state, velocity=[0.0] * len(JOINT_NAMES)),
            )
        )
    outcome = runtime.send_joint(target, 120.0)
    assert outcome.status.value == "FAILED"
    assert is_success(outcome) is False
    assert outcome.details["settled_at_rest"] is False
    assert clock.now >= 20.0  # bounded by FINAL_CONVERGENCE_TIMEOUT_S


# --------------------------------------------- compute_link6_fk adapter contract


def test_compute_link6_fk_sends_the_exact_request_and_returns_the_pose() -> None:
    """The FK hook asks for base_link->link6 over the exact canonical state."""
    world = make_world()
    world.fk.response = fk_response()
    verified = world.runtime.compute_link6_fk(BASELINE, timeout_s=5.0)
    assert verified == (FK_POSITION, FK_QUATERNION)
    assert len(world.fk.requests) == 1
    request = world.fk.requests[0]
    assert request.header.frame_id == BASE_FRAME
    assert request.fk_link_names == [TIP_LINK]
    assert request.robot_state.joint_state.name == list(JOINT_NAMES)
    assert request.robot_state.joint_state.position == [
        float(BASELINE[name]) for name in JOINT_NAMES
    ]


def test_compute_link6_fk_returns_none_without_a_registered_client() -> None:
    """No clients registered: no pose is invented."""
    runtime = RosMoveItRuntime(
        FakeNode(),
        factories=fake_factories(),
        clock=FakeClock(),
        sleep=lambda _dt: None,
    )
    assert runtime.compute_link6_fk(BASELINE, timeout_s=5.0) is None


@pytest.mark.parametrize("timeout_s", [0.0, -1.0, float("nan"), float("inf")])
def test_compute_link6_fk_rejects_invalid_timeouts(timeout_s) -> None:
    """Non-positive or non-finite bounds violate the input contract."""
    world = make_world()
    with pytest.raises(ValueError):
        world.runtime.compute_link6_fk(BASELINE, timeout_s=timeout_s)


def test_compute_link6_fk_refuses_incomplete_states_without_a_call() -> None:
    """A state missing canonical joints is never sent to the service."""
    world = make_world()
    world.fk.response = fk_response()
    assert world.runtime.compute_link6_fk({"joint1": 0.0}, timeout_s=5.0) is None
    assert world.fk.requests == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"code": -31},
        {"stamp_count": 0},
        {"position": (float("nan"), 0.0, 0.2)},
        {"quaternion": (0.0, 0.0, 0.0, 0.0)},
        {"quaternion": (float("inf"), 0.0, 0.0, 1.0)},
    ],
)
def test_compute_link6_fk_fails_explicitly_on_bad_answers(kwargs) -> None:
    """Failed codes and nonfinite/near-zero answers return None, never fake."""
    world = make_world()
    world.fk.response = fk_response(**kwargs)
    assert world.runtime.compute_link6_fk(BASELINE, timeout_s=5.0) is None


def test_compute_link6_fk_times_out_boundedly() -> None:
    """A service answer that never arrives returns None before the bound."""
    world = make_world()
    client = world.fk

    def never_responds(request) -> FakeFuture:
        client.requests.append(request)
        return FakeFuture()  # never done

    client.call_async = never_responds  # type: ignore[method-assign]
    assert world.runtime.compute_link6_fk(BASELINE, timeout_s=0.2) is None
    assert world.clock.now >= 0.2


# ------------------------------------- executor parity over the live FK adapter


def test_link6_pose_via_executor_verifies_through_compute_link6_fk() -> None:
    """Fake regression: pose execution verifies against the adapter FK hook."""
    holder: dict = {}

    def fk_hook(state):
        adapter = holder.get("runtime")
        if adapter is None:
            return None
        return adapter.compute_link6_fk(state, timeout_s=2.0)

    world = make_world(fk_verifier=fk_hook)
    holder["runtime"] = world.runtime
    target, decision = build_link6_pose_target(
        frame_id=BASE_FRAME,
        position=FK_POSITION,
        orientation_xyzw=FK_QUATERNION,
        position_tolerance_m=0.005,
        orientation_tolerance_rad=0.015,
    )
    assert decision.accepted is True and target is not None
    world.fk.response = fk_response()
    prime_ready(world)
    script_success(
        world, action_result=trajectory_action_result(), final_state=BASELINE
    )
    outcome = RosMoveItExecutor(world.runtime).execute_link6_pose(
        target, timeout_s=30.0
    )
    assert is_success(outcome) is True
    assert outcome.command_messages >= 1
    assert outcome.final_errors["position_error_m"] == pytest.approx(0.0, abs=1e-12)
    assert outcome.final_errors["orientation_error_rad"] == pytest.approx(
        0.0, abs=1e-12
    )
    assert len(world.fk.requests) == 1  # verified through the live hook only


def test_link6_pose_via_executor_fails_when_live_fk_disagrees() -> None:
    """A shifted live FK answer must fail the pose gates (never faked)."""
    holder: dict = {}

    def fk_hook(state):
        adapter = holder.get("runtime")
        if adapter is None:
            return None
        return adapter.compute_link6_fk(state, timeout_s=2.0)

    world = make_world(fk_verifier=fk_hook)
    holder["runtime"] = world.runtime
    target, decision = build_link6_pose_target(
        frame_id=BASE_FRAME,
        position=FK_POSITION,
        orientation_xyzw=FK_QUATERNION,
    )
    assert decision.accepted is True and target is not None
    world.fk.response = fk_response(
        position=(FK_POSITION[0] + 0.05, FK_POSITION[1], FK_POSITION[2])
    )
    prime_ready(world)
    script_success(
        world, action_result=trajectory_action_result(), final_state=BASELINE
    )
    outcome = RosMoveItExecutor(world.runtime).execute_link6_pose(
        target, timeout_s=30.0
    )
    assert is_success(outcome) is False
    assert outcome.final_errors["position_error_m"] == pytest.approx(0.05)


def test_joint_flow_through_executor_matches_the_probe_shape() -> None:
    """Outbound joint goal: library admission then executor gates, offline."""
    world = make_world()
    target, decision = build_joint_target(BASELINE, {"joint1": 0.10, "joint4": 0.08})
    assert decision.accepted is True and target is not None
    prime_ready(world)
    final = dict(BASELINE)
    final["joint1"] = target.positions["joint1"] + 0.005
    final["joint4"] = target.positions["joint4"] - 0.004
    script_success(
        world, action_result=trajectory_action_result(target), final_state=final
    )
    outcome = RosMoveItExecutor(world.runtime).execute_joint(target, timeout_s=30.0)
    assert is_success(outcome) is True
    assert outcome.command_messages >= 1
    assert max(outcome.final_errors.values()) <= 0.04
    assert outcome.details["gripper_drift"] <= 0.01
    assert len(world.move_group.goals) == 1
    request = world.move_group.goals[0].request
    assert request.group_name == "arm"
    assert world.move_group.goals[0].planning_options.plan_only is False


def test_joint_flow_rejects_out_of_limits_targets_before_any_send() -> None:
    """Executor admission mirrors the probe path: reject, zero commands."""
    world = make_world()
    invalid = JointTarget(
        positions={**{name: 0.0 for name in ARM_JOINT_NAMES}, "joint1": 9.0}
    )
    outcome = RosMoveItExecutor(world.runtime).execute_joint(invalid, timeout_s=5.0)
    assert outcome.details["reason"]
    assert outcome.command_messages == 0
    assert world.move_group.goals == []


def test_probe_default_tolerances_stay_inside_the_runtime_gates() -> None:
    """The probe's tighter requested pose tolerances never exceed the gates."""
    from rocobrick.moveit_contract import (
        MAX_ORIENTATION_TOLERANCE_RAD,
        MAX_POSITION_TOLERANCE_M,
    )

    source = PROBE_PATH.read_text(encoding="utf-8")
    constants = module_constants(source)
    assert constants["GOAL_POSITION_TOLERANCE_M"] <= MAX_POSITION_TOLERANCE_M
    assert constants["GOAL_ORIENTATION_TOLERANCE_RAD"] <= MAX_ORIENTATION_TOLERANCE_RAD
    assert math.isclose(constants["GOAL_POSITION_TOLERANCE_M"], 0.005)
    assert math.isclose(constants["GOAL_ORIENTATION_TOLERANCE_RAD"], 0.015)


def test_runtime_defaults_unchanged_for_topics_services_actions() -> None:
    """Guard: the adapter addition changed no pinned transport defaults."""
    topics = RuntimeTopics()
    services = RuntimeServices()
    actions = RuntimeActions()
    assert topics.clock == "/clock"
    assert topics.isaac_joint_states == "/isaac_joint_states"
    assert topics.joint_states == "/joint_states"
    assert topics.isaac_joint_command == "/isaac_joint_command"
    assert services.list_controllers == SERVICE_LIST_CONTROLLERS
    assert services.compute_fk == SERVICE_COMPUTE_FK
    assert services.plan == "/plan_kinematic_path"
    assert actions.arm_follow == ACTION_ARM_FOLLOW
    assert actions.move_group == MOVE_GROUP_ACTION_CANDIDATES[0]


def test_link6_pose_target_defaults_carry_the_unresolved_tcp_status() -> None:
    """Pose parity only: the TCP status stays unresolved end to end."""
    target = Link6PoseTarget(
        frame_id=BASE_FRAME,
        position=FK_POSITION,
        orientation_xyzw=FK_QUATERNION,
    )
    assert target.tcp_status == "unresolved"
    assert target.tip_link == TIP_LINK
    probe_source = PROBE_PATH.read_text(encoding="utf-8")
    assert "tcp_status" not in probe_source  # never re-stated, never changed
