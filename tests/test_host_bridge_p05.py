"""Offline structural and policy tests for the P0.5 host Isaac bridge.

These tests never boot Kit, touch the network, or start containers. They pin
the corrected P0.5 architecture: the host process runs exactly ONE DDS
participant — a single in-process rclpy node (``piper_l0_host_bridge``) created
from the Humble rclpy bundled inside the Isaac install and wired by
``tools/isaac-ros-env.sh`` — publishing /clock and /isaac_joint_states
in-process and subscribing /isaac_joint_command (sensor_msgs/JointState), with named joints
mapped to DOF indices and applied as a clamped ArticulationAction. The
isaacsim.ros2.bridge Kit extension is NEVER enabled: enabling it would spawn
additional in-process DDS participants, and the previous per-OmniGraph-node
Fast DDS participants re-bound the host profile TCP 17650 listening port and
aborted with "Address already in use". No OmniGraph is created. Also pinned:
the Piper L USD prim at /World/Robot_0 with articulation root
/World/Robot_0/root_joint, the readiness marker, the bounded
external-validation hold, the orderly cleanup, and the smoke runner's
host-background + container-probe orchestration.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
BRIDGE_PATH = ROOT / "isaac" / "p05_host_bridge.py"
PROBE_PATH = ROOT / "tools" / "p05_integration_probe.py"
RUNNER_PATH = ROOT / "tools" / "smoke_p05_host_bridge.sh"
ENV_SCRIPT_PATH = ROOT / "tools" / "isaac-ros-env.sh"
USD_PATH = ROOT / "robot_assets" / "piper_l" / "USD" / "piper_l_gripper_d435.usd"

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

EXPECTED_ROS_NODE_NAME = "piper_l0_host_bridge"
EXPECTED_BRIDGE_IMPORTS = {"rclpy", "rosgraph_msgs.msg", "sensor_msgs.msg"}

# The OmniGraph approach must be gone: no Controller.edit graph authoring, no
# leftover graph constants, and no usdrt targetPrim plumbing.
FORBIDDEN_SOURCE_TOKENS = (
    "og.Controller.edit",
    "CREATE_NODES",
    "CONNECT",
    "SET_VALUES",
    "GRAPH_PATH",
    "usdrt",
    "OnPlaybackTick",
)

MODE_B_FORBIDDEN_MODULES = {"socket", "grpc", "zmq"}


@pytest.fixture(scope="module")
def bridge_module():
    """Import the bridge module without booting isaacsim, omni, or rclpy.

    The module is registered in ``sys.modules`` before execution because the
    bridge's frozen value types are dataclasses and dataclasses resolve their
    postponed string annotations through ``sys.modules``.

    Returns:
        The imported bridge module.
    """
    spec = importlib.util.spec_from_file_location("p05_host_bridge", BRIDGE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bridge_source() -> str:
    """Return the raw bridge script text.

    Returns:
        The bridge script source.
    """
    return BRIDGE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def bridge_tree(bridge_source: str):
    """Return the parsed AST of the bridge script.

    Args:
        bridge_source: The raw bridge script text.

    Returns:
        The bridge script AST.
    """
    return ast.parse(bridge_source)


@pytest.fixture(scope="module")
def runner_source() -> str:
    """Return the raw smoke runner text.

    Returns:
        The smoke runner source.
    """
    return RUNNER_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def env_script_source() -> str:
    """Return the raw host ROS environment script text.

    Returns:
        The tools/isaac-ros-env.sh source.
    """
    return ENV_SCRIPT_PATH.read_text(encoding="utf-8")


def test_bridge_script_exists() -> None:
    """The bridge script and the target USD asset must both exist."""
    assert BRIDGE_PATH.is_file()
    assert USD_PATH.is_file()


def test_module_import_is_offline_safe(bridge_module) -> None:
    """Importing the module must not boot isaacsim, omni, or rclpy."""
    assert bridge_module.SCHEMA == "rocobrick-host-bridge-smoke/v1"
    for heavy in ("isaacsim", "omni", "rclpy", "pxr"):
        assert heavy not in sys.modules, heavy


def test_single_rclpy_participant_contract(bridge_module) -> None:
    """The host must run exactly one in-process rclpy node (the P0.5 fix).

    The previous OmniGraph ROS 2 nodes each created their own Fast DDS
    participant; with the host TCPv4 profile the first bound TCP 17650 and the
    rest aborted with "Address already in use". The evidence must pin the
    single in-process participant that replaces them.
    """
    assert bridge_module.ROS_NODE_NAME == EXPECTED_ROS_NODE_NAME
    evidence_defaults = bridge_module.initial_evidence(
        bridge_module.parse_args(
            [
                "--max-steps", "10",
                "--time-limit", "30",
                "--ready-timeout", "5",
                "--hold-steps", "5",
                "--hold-seconds", "1",
            ]
        )
    )
    assert evidence_defaults["in_process_rclpy"] is True
    assert evidence_defaults["rclpy_node_count"] == 1
    assert evidence_defaults["rclpy_node_name"] == EXPECTED_ROS_NODE_NAME
    assert evidence_defaults["omnigraph_created"] is False
    assert evidence_defaults["bridge_extension_enabled"] is False
    assert evidence_defaults["headless"] is True


def test_ros_topics_and_message_types_pinned(bridge_module) -> None:
    """Topics must match the official TopicBasedSystem contract exactly.

    The pinned agx_arm_sim piper_l.ros2_control.xacro wires
    topic_based_ros2_control/TopicBasedSystem to ``/isaac_joint_states``
    (plural) and ``/isaac_joint_command`` (singular); matching them here lets
    P1 attach ros2_control/MoveIt with zero bridge-side changes.
    """
    assert bridge_module.TOPIC_CLOCK == "/clock"
    assert bridge_module.TOPIC_JOINT_STATES == "/isaac_joint_states"
    assert bridge_module.TOPIC_JOINT_COMMANDS == "/isaac_joint_command"
    assert bridge_module.JOINT_COMMAND_MESSAGE_TYPE == "sensor_msgs/JointState"


def test_no_omnigraph_anywhere(bridge_source: str, bridge_tree) -> None:
    """No OmniGraph authoring, graph constants, or usdrt plumbing may remain."""
    for token in FORBIDDEN_SOURCE_TOKENS:
        assert token not in bridge_source, token
    calls = {
        node.func.attr
        for node in ast.walk(bridge_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "edit" not in calls, "og.Controller.edit must not be used"


def test_bridge_core_imports_and_entity_creation(bridge_tree) -> None:
    """The rclpy and ROS message modules must be imported and used in-process."""
    imported = set()
    for node in ast.walk(bridge_tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert EXPECTED_BRIDGE_IMPORTS <= imported, imported
    calls = {
        node.func.attr
        for node in ast.walk(bridge_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert {"init", "create_node", "create_publisher", "create_subscription"} <= calls
    assert {"spin_once", "destroy_node", "shutdown"} <= calls


def test_publishers_and_subscription_topics_wired(bridge_tree) -> None:
    """The one node must publish /clock + /isaac_joint_states and subscribe commands."""
    publishers = []
    subscriptions = []
    for node in ast.walk(bridge_tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            args = node.args
            if node.func.attr == "create_publisher" and len(args) >= 2:
                publishers.append((args[0].id, args[1].id))
            if node.func.attr == "create_subscription" and len(args) >= 2:
                subscriptions.append((args[0].id, args[1].id))
    assert ("Clock", "TOPIC_CLOCK") in publishers
    assert ("JointState", "TOPIC_JOINT_STATES") in publishers
    assert ("JointState", "TOPIC_JOINT_COMMANDS") in subscriptions


def test_commands_follow_articulation_action_pattern(bridge_source: str) -> None:
    """Named commands map to DOF indices, clamp to limits, and apply actions."""
    assert "ArticulationAction(" in bridge_source
    assert "joint_positions=tuple(joint_positions)" in bridge_source
    assert "joint_indices=tuple(joint_indices)" in bridge_source
    assert "apply_action(" in bridge_source
    assert "dof_properties" in bridge_source
    assert "min(max(" in bridge_source
    assert "self._dof_index.get(" in bridge_source
    assert "unknown_joint_names" in bridge_source


def test_clock_message_uses_sim_seconds_and_nanoseconds(bridge_source: str) -> None:
    """The /clock payload must come from the simulation time."""
    assert "world.current_time" in bridge_source
    assert "message.clock.sec = seconds" in bridge_source
    assert "message.clock.nanosec = nanoseconds" in bridge_source
    assert "NANOS_PER_SECOND" in bridge_source


def test_bridge_extension_never_enabled(bridge_source: str) -> None:
    """The Kit extension must never be enabled and no startup gate may remain.

    Enabling ``isaacsim.ros2.bridge`` makes the extension spawn its own DDS
    participants in-process, which re-bind the host profile TCP 17650
    listening port. The bundled Humble rclpy is importable through
    ``tools/isaac-ros-env.sh`` alone, so the extension enable path and its
    OmniGraph registry startup gate are gone entirely.
    """
    for token in (
        "set_extension_enabled_immediate",
        "_wait_for_bridge_startup",
        "get_registered_nodes",
        "BRIDGE_OGN_TYPES",
        "registry_confirmed",
    ):
        assert token not in bridge_source, token
    assert "import rclpy" in bridge_source
    assert "rclpy.init(signal_handler_options=SignalHandlerOptions.NO)" in bridge_source


def test_run_bridge_pins_disabled_extension_and_bundled_rclpy(
    bridge_source: str,
) -> None:
    """run_bridge must verify the disabled extension, then import the rclpy.

    The extension-disabled check and the bundled Humble rclpy provenance check
    must both run before the single InProcessBridge participant is created, so
    a wrong environment fails fast before any DDS participant exists.
    """
    assert "is_extension_enabled(BRIDGE_EXTENSION)" in bridge_source
    assert 'evidence["bridge_extension_enabled"] = False' in bridge_source
    assert '"humble/rclpy" in rclpy_path' in bridge_source
    assert "tools/isaac-ros-env.sh" in bridge_source
    start = bridge_source.index("def run_bridge")
    end = bridge_source.index("def write_evidence")
    run_body = bridge_source[start:end]
    extension_check = run_body.index("is_extension_enabled(BRIDGE_EXTENSION)")
    rclpy_import = run_body.index("import rclpy")
    participant = run_body.index("InProcessBridge(")
    assert extension_check < rclpy_import < participant, (
        "extension-disabled check and bundled rclpy import must precede "
        "participant creation"
    )


def test_no_post_participant_environment_rewrites(bridge_source: str) -> None:
    """After the participant is created the DDS environment must stay fixed.

    The retired workaround swapped FASTRTPS_DEFAULT_PROFILES_FILE to a
    secondary host port after creating the node — pointless now that the
    extension (the source of the extra participants) is never enabled, and
    harmful because it rewrote the environment the single participant had
    already consumed.
    """
    for token in ("secondary", "with_name", "os.environ["):
        assert token not in bridge_source, token
    run_body_start = bridge_source.index("InProcessBridge(")
    run_body = bridge_source[run_body_start:]
    assert "os.environ" not in run_body


def test_env_script_wires_bundled_humble_rclpy(env_script_source: str) -> None:
    """The env script must expose the bundled Humble rclpy and its libraries.

    The bridge never enables the Kit extension, so this environment is the
    only path through which rclpy, rosgraph_msgs and sensor_msgs are imported.
    """
    assert 'BRIDGE="$ISAAC_ROOT/exts/isaacsim.ros2.bridge"' in env_script_source
    assert '[[ -d "$BRIDGE/humble/lib" ]]' in env_script_source
    assert 'PYTHONPATH="$BRIDGE/humble/rclpy' in env_script_source
    assert 'LD_LIBRARY_PATH="$BRIDGE/humble/lib' in env_script_source
    assert "ROS_DISTRO=humble" in env_script_source
    assert "RMW_IMPLEMENTATION=rmw_fastrtps_cpp" in env_script_source


def test_joint_state_message_fields(bridge_source: str) -> None:
    """/isaac_joint_states must carry the DOF names, positions, and velocities."""
    assert "message.name = self._dof_names" in bridge_source
    assert "message.position = [float(value) for value in positions]" in bridge_source
    assert "message.velocity = [float(value) for value in velocities]" in bridge_source
    assert "get_joint_positions()" in bridge_source


def test_kit_child_regime_without_simulation_app(
    bridge_module, bridge_tree, bridge_source: str
) -> None:
    """The bridge must run only inside the Kit child regime.

    SimulationApp must never be constructed (its standalone regime segfaults
    inside BrickSim scene loading; the bricksim launcher regime does not).
    run_bridge must fail closed outside the Kit child regime, the Kit launch
    argv must disable the ros2 bridge extension and re-exec this exact file
    via --exec, and no UI may be imported.
    """
    calls = [
        node
        for node in ast.walk(bridge_tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == "SimulationApp"
    ]
    assert calls == []
    assert "omni.ui" not in ast.dump(bridge_tree)
    start = bridge_source.index("def run_bridge")
    end = bridge_source.index("def write_evidence")
    run_body = bridge_source[start:end]
    assert "if not _running_inside_kit():" in run_body
    assert "must run inside the Kit child regime" in run_body
    args = bridge_module.parse_args(["--evidence-out", "/tmp/e.json"])
    kit_argv = bridge_module._kit_launch_argv(args)
    assert kit_argv[0] == "isaacsim.exp.full"
    assert kit_argv[kit_argv.index("--disable") + 1] == "isaacsim.ros2.bridge"
    assert "--/app/content/emptyStageOnStart=false" in kit_argv
    exec_index = kit_argv.index("--exec")
    assert kit_argv[exec_index + 1].split()[0] == str(bridge_module.BRIDGE_FILE)


def test_kit_launch_argv_round_trips_every_argument(bridge_module) -> None:
    """The rebuilt bridge argv must parse back into the identical arguments."""
    base = [
        "--max-steps", "600", "--time-limit", "180", "--ready-timeout", "90",
        "--hold-steps", "60", "--hold-seconds", "2.0",
        "--evidence-out", "/tmp/e.json",
        "--scene-manifest", "/tmp/m.json", "--scene-snapshot-out", "/tmp/s.json",
        "--host-instance-id", "repro-001",
        "--plan", "/tmp/plan.json", "--config", "/tmp/config.json",
    ]
    parsed = bridge_module.parse_args(base)
    rebuilt = bridge_module.parse_args(bridge_module._argv_from_args(parsed))
    for field in (
        "max_steps", "time_limit", "ready_timeout", "hold_steps", "hold_seconds",
        "evidence_out", "scene_manifest", "scene_snapshot_out",
        "host_instance_id", "plan", "config",
    ):
        assert getattr(rebuilt, field) == getattr(parsed, field)
    minimal = bridge_module.parse_args(["--evidence-out", "/tmp/e.json"])
    argv = bridge_module._argv_from_args(minimal)
    assert "--scene-manifest" not in argv
    assert "--plan" not in argv


def test_run_is_bounded(bridge_module, bridge_tree) -> None:
    """Steps and wall-clock time must both bound ready and hold phases."""
    assert bridge_module.DEFAULT_MAX_STEPS >= 100000
    assert bridge_module.DEFAULT_TIME_LIMIT_S > 0
    assert bridge_module.DEFAULT_READY_TIMEOUT_S > 0
    assert bridge_module.DEFAULT_HOLD_STEPS > 0
    assert bridge_module.DEFAULT_HOLD_SECONDS_S > 0
    options = {
        node.args[0].value
        for node in ast.walk(bridge_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }
    bounded_options = {
        "--max-steps",
        "--time-limit",
        "--ready-timeout",
        "--hold-steps",
        "--hold-seconds",
    }
    assert bounded_options <= options


def test_articulation_initialized_before_marker(bridge_source: str) -> None:
    """The articulation must be initialized through a reset before the marker."""
    assert "SingleArticulation(" in bridge_source
    assert "world.reset()" in bridge_source
    added = bridge_source.index("SingleArticulation(")
    initialized = bridge_source.index("world.reset()")
    marker = bridge_source.index('"stage": "ready"')
    assert added < initialized < marker


def test_world_reset_then_play_with_timeline_evidence(bridge_source: str) -> None:
    """After the reset the timeline must be re-played and recorded."""
    assert bridge_source.index("world.reset()") < bridge_source.index("world.play()")
    assert '"timeline_playing"' in bridge_source


def test_marker_only_after_articulation_and_first_publish(bridge_source: str) -> None:
    """The marker must follow the in-process gate and first publishes."""
    gate = bridge_source.index("clock_and_joint_states_published")
    published = bridge_source.index("bridge.publish_state")
    marker = bridge_source.index('"marker": READY_MARKER')
    assert gate < marker
    assert published < marker
    assert '"stage": "ready"' in bridge_source
    assert "flush=True" in bridge_source


def test_step_loop_publish_spin_no_post_init_kit_pump(bridge_source: str) -> None:
    """Steps publish throttled, spin non-blocking, and never pump Kit after init.

    app.update() after rclpy.init() makes the bridge extension spawn its own
    DDS participants, which then fail to bind the profile's TCP port and
    poison endpoint discovery for the larger message types.
    """
    assert "PUBLISH_PERIOD_S" in bridge_source
    stepped = bridge_source.index("world.step(render=False)")
    published = bridge_source.index("bridge.publish_state")
    spun = bridge_source.index("bridge.spin_once()")
    assert stepped < published < spun
    assert "timeout_sec=0.0" in bridge_source
    assert "last_publish >= PUBLISH_PERIOD_S" in bridge_source
    run_body_start = bridge_source.index("loop_started = time.monotonic()")
    assert "app.update" not in bridge_source[run_body_start:]


def test_dof_count_check(bridge_module, bridge_source: str) -> None:
    """The articulation must report the complete 8-DOF Piper L set."""
    assert bridge_module.EXPECTED_DOF_COUNT == 8
    assert bridge_module.EXPECTED_DOF_COUNT == len(EXPECTED_DOF_NAMES)
    assert "expected {EXPECTED_DOF_COUNT} DOFs" in bridge_source


def test_hold_phase_for_external_validation(bridge_source: str) -> None:
    """The host must keep stepping after the marker so the probe can verify."""
    marker = bridge_source.index('"stage": "ready"')
    hold = bridge_source.index('evidence["hold_completed"] = True')
    assert marker < hold
    assert "world.step(render=False)" in bridge_source
    assert '"hold"' in bridge_source
    assert '"hold_samples"' in bridge_source


def test_orderly_cleanup_ros_first(bridge_tree, bridge_source: str) -> None:
    """Cleanup must run rclpy, world, timeline, app in order from finally."""
    calls = {
        node.func.attr
        for node in ast.walk(bridge_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert {"destroy_node", "shutdown", "stop"} <= calls
    has_finally = any(
        node.finalbody for node in ast.walk(bridge_tree) if isinstance(node, ast.Try)
    )
    assert has_finally, "cleanup must run from a finally block"
    start = bridge_source.index("def shutdown")
    end = bridge_source.index("def parse_args")
    shutdown_body = bridge_source[start:end]
    order_ok = (
        shutdown_body.index("node.destroy_node()")
        < shutdown_body.index("try_shutdown")
        < shutdown_body.index("world.stop()")
        < shutdown_body.index("get_timeline_interface().stop()")
        < shutdown_body.index("get_app().shutdown()")
    )
    assert order_ok, "cleanup order must be rclpy, world, timeline, app"
    evidence_first = bridge_source.index("finally:")
    finally_body = bridge_source[evidence_first:]
    assert finally_body.index("write_evidence") < finally_body.index(
        "state.shutdown()"
    ), "evidence must be written before teardown"


def test_passed_protocol_survives_kit_fast_shutdown(
    bridge_source: str, bridge_module
) -> None:
    """passed persists progressively and uses Kit's supported shutdown API."""
    shutdown_start = bridge_source.index("def shutdown")
    shutdown_end = bridge_source.index("def parse_args")
    shutdown_body = bridge_source[shutdown_start:shutdown_end]
    assert shutdown_body.index('status["app_close_initiated"] = True') < (
        shutdown_body.index("                get_app().shutdown()")
    )
    assert (
        'self.evidence["passed"] = evaluate_passed(self.evidence)'
        in bridge_source
    )
    complete = {
        "ready": True,
        "hold_completed": True,
        "error": None,
        "cleanup": {
            "rclpy_shutdown": True,
            "world_stopped": True,
            "timeline_stopped": True,
            "app_close_initiated": True,
        },
    }
    assert bridge_module.evaluate_passed(complete) is True
    without_close = {
        **complete,
        "cleanup": {**complete["cleanup"], "app_close_initiated": False},
    }
    assert bridge_module.evaluate_passed(without_close) is False
    assert bridge_module.evaluate_passed({**complete, "hold_completed": False}) is False


def test_no_mode_b_protocol(bridge_source: str, bridge_tree) -> None:
    """The old Mode B socket protocol must not reappear."""
    imported = set()
    for node in ast.walk(bridge_tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint(MODE_B_FORBIDDEN_MODULES)
    assert "mode_b" not in bridge_source.lower()
    assert "mode-b" not in bridge_source.lower()


def test_runner_profiles_and_host_background(runner_source: str) -> None:
    """The runner must generate DDS profiles then background the host bridge."""
    assert "tools/generate_dds_profiles.py" in runner_source
    assert ".runtime/dds" in runner_source
    assert "ROS_DOMAIN_ID:-73" in runner_source
    assert "tools/isaac-ros-env.sh" in runner_source
    assert "isaac/p05_host_bridge.py" in runner_source
    assert '>"$HOST_LOG" 2>&1 &' in runner_source
    assert "HOST_PID=$!" in runner_source


def test_runner_waits_for_marker_then_runs_container_probe(runner_source: str) -> None:
    """The runner must gate on the marker and validate via the ROS container."""
    assert 'MARKER="p0.5_host_bridge_ready"' in runner_source
    assert "readiness_marker_seen" in runner_source
    assert "docker run --rm --name" in runner_source
    assert '-p "$CONTAINER_PORT:$CONTAINER_PORT/tcp"' in runner_source
    assert "FASTRTPS_DEFAULT_PROFILES_FILE=/dds/fastdds-container.xml" in runner_source
    assert "tools/p05_integration_probe.py" in runner_source
    assert "probe.py" in runner_source


def test_runner_stops_host_orderly_and_writes_aggregate(runner_source: str) -> None:
    """The runner must wait/stop the host and aggregate both evidence sides."""
    assert "stop_host" in runner_source
    assert "kill -INT" in runner_source
    assert "kill -KILL" in runner_source
    assert "wait \"$HOST_PID\"" in runner_source
    assert "artifacts/phase0.5/p05-integration.json" in runner_source
    assert "rocobrick-p05-integration/v1" in runner_source
    assert "rocobrick-p05-integration-probe/v1" in runner_source


def test_runner_is_bounded(runner_source: str) -> None:
    """Every long-running process must be wrapped in timeout."""
    assert "timeout --signal=INT --kill-after=30" in runner_source
    assert "timeout --signal=INT --kill-after=15" in runner_source
    assert "P05_SIM_TIMEOUT:-900" in runner_source
    assert "P05_PROBE_TIMEOUT:-240" in runner_source
    assert "P05_READY_WAIT:-600" in runner_source
    assert "P05_STOP_WAIT:-150" in runner_source


def test_runner_max_steps_pinned(runner_source: str) -> None:
    """The runner must default the simulation bound to 12000 steps."""
    assert "P05_MAX_STEPS:-1000000" in runner_source


def test_runner_preflight_model_audit(runner_source: str) -> None:
    """The runner must gate the GPU smoke on the offline model audit.

    The preflight runs the deterministic tools/piper_l_model_audit.py into the
    smoke output dir, so runtime USD/URDF drift from the pinned agx_arm_sim
    fails before any GPU time is spent.
    """
    assert "tools/piper_l_model_audit.py" in runner_source
    assert "preflight-model-audit.json" in runner_source


def test_runner_cleans_up_processes_and_container(runner_source: str) -> None:
    """The EXIT trap must clean the host process and the container."""
    assert "trap cleanup EXIT" in runner_source
    assert "docker rm -f" in runner_source


def test_probe_file_exists_and_runner_is_executable() -> None:
    """The probe script and the executable runner must be in place."""
    assert PROBE_PATH.is_file()
    assert os.access(RUNNER_PATH, os.X_OK)
