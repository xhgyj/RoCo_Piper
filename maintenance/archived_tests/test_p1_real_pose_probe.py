"""Offline structural tests for the P1.C real pose-target execution smoke.

The probe and the move_group stack run inside the ROS container (where rclpy
exists); the offline suite here therefore parses their source and AST only
(never imports them, never touches docker or the GPU). Pinned contract: the
probe waits for the FULL readiness set (clock, /isaac_joint_states,
/joint_states via the broadcaster, three active controllers, the arm
FollowJointTrajectory action, the MoveGroup action, /compute_fk ready), waits
for the arm to be at rest, derives the link6 target pose from the LIVE
/compute_fk service over the captured baseline (no hardcoded world
coordinates), optionally applies a tiny +0.005 m displacement ONLY when a
collision-aware GetPositionIK preflight proves it reachable, admits the pose
goal STRICTLY on the caller side (finite/normalized quaternion, exact
frame/link, tolerance bounds, NO simultaneous joint target) BEFORE sending,
then runs two real plan-and-execute MoveGroup goals (plan_only false) — the
displaced FK pose, then the exact baseline FK pose — demanding terminal
SUCCEEDED + MoveIt SUCCESS(1), in-limits planned waypoints, observed
/isaac_joint_command traffic and a live /compute_fk proof that the final
base_link -> link6 pose is within 0.01 m / 0.03 rad of the requested pose.
The scope is the link6 ORIGIN only: the model contract records the TCP as
unresolved, so tcp_link frames, grasps, example1 and piper-run are forbidden,
and the probe never synthesizes or mutates plans itself.
"""

from __future__ import annotations

import ast
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
PROBE_PATH = ROOT / "tools" / "p1_real_pose_probe.py"
SMOKE_PATH = ROOT / "tools" / "smoke_p1_real_pose.sh"
GENERATOR_PATH = ROOT / "tools" / "generate_dds_profiles.py"
ARTIFACT_PATH = ROOT / "artifacts" / "phase1-real" / "pose-target.json"

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

# Narrowed execution limits (model contract) the waypoints must be checked
# against.
NARROW_EXECUTION_LIMITS = {
    "joint1": (-2.6179938, 2.6179938),
    "joint2": (0.0, 3.1415926),
    "joint3": (-2.9670597, 0.0),
    "joint4": (-1.7453292, 1.7453292),
    "joint5": (-1.2217304, 1.2217304),
    "joint6": (-2.0943951, 2.0943951),
}

FORBIDDEN_MODULES = {"socket", "grpc", "zmq", "isaacsim", "omni", "paramiko"}
# link6 pose scope ONLY: no TCP frame, no grasp, no example1, no piper-run,
# and no client-side plan synthesis or mutation (the MoveGroup action plans).
FORBIDDEN_TOKENS = (
    "tcp_link",
    "grasp",
    "example1",
    "piper-run",
    "PoseGoal",
    "GetMotionPlan",
    "plan_kinematic_path",
    "RobotTrajectory",
    ".points.append",
)
# No hardcoded world coordinates: no list/tuple literal of three decimal
# numbers anywhere in the probe (the target comes from the live FK service).
HARDCODED_POINT = re.compile(
    r"[\[\(]\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*[\]\)]"
)


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


def extract_functions(source: str, names: set[str]) -> dict:
    """Exec pure module-level functions with only ``math`` available.

    Args:
        source: The probe source text.
        names: The function names to extract (``finite`` is always included).

    Returns:
        A namespace dict with the requested callables.
    """
    namespace: dict = {"math": math, "__builtins__": __builtins__}
    namespace.update(module_constants(source))  # pinned constants are pure
    wanted = set(names) | {"finite"}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            code = compile(
                ast.Module(body=[node], type_ignores=[]), "<probe-extract>", "exec"
            )
            exec(code, namespace)
    assert names <= set(namespace), names - set(namespace)
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
    assert {
        "rclpy",
        "moveit_msgs",
        "geometry_msgs",
        "shape_msgs",
        "builtin_interfaces",
        "control_msgs",
        "controller_manager_msgs",
        "sensor_msgs",
        "rosgraph_msgs",
        "action_msgs",
    } <= imported
    assert imported.isdisjoint(FORBIDDEN_MODULES)


def test_probe_has_no_forbidden_tokens_or_plan_synthesis(probe_source: str) -> None:
    """No TCP frame, no grasp, no example1, no piper-run, no plan mutation."""
    for token in FORBIDDEN_TOKENS:
        assert token not in probe_source, token
    assert "open(" not in probe_source  # no file IO at all
    assert "Path(" not in probe_source
    assert HARDCODED_POINT.search(probe_source) is None, (
        "a hardcoded world coordinate leaked into the probe"
    )


def test_probe_pinned_schema_topics_actions_and_services(probe_source: str) -> None:
    """Schema, the three contract topics, both actions, FK and IK services."""
    constants = module_constants(probe_source)
    assert constants["SCHEMA"] == "rocobrick-p1-real-pose-probe/v1"
    assert constants["PHASE"] == "P1.C"
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
    assert constants["SERVICE_FK"] == "/compute_fk"
    # Live evidence: the pinned Humble move_group exposes "/compute_ik"; the
    # "/get_position_ik" alias is kept for other MoveIt2 releases.
    assert list(constants["SERVICE_IK_CANDIDATES"]) == [
        "/compute_ik",
        "/get_position_ik",
    ]


def test_probe_scope_is_link6_in_base_link_with_unresolved_tcp(
    probe_source: str,
) -> None:
    """tip_link=link6, base frame=base_link, tcp_status=unresolved, recorded."""
    constants = module_constants(probe_source)
    assert constants["TIP_LINK"] == "link6"
    assert constants["BASE_FRAME"] == "base_link"
    assert constants["PLANNING_GROUP"] == "arm"
    assert constants["TCP_STATUS"] == "unresolved"
    assert '"tip_link": TIP_LINK' in probe_source
    assert '"tcp_status": TCP_STATUS' in probe_source
    assert "unresolved" in probe_source  # the contract wording is recorded
    assert '"fk_link_names": [TIP_LINK]' in probe_source


def test_probe_pins_the_eight_dof_names_and_execution_limits(
    probe_source: str,
) -> None:
    """The exact 8-DOF Piper_L names and the narrowed admission table."""
    constants = module_constants(probe_source)
    assert list(constants["EXPECTED_DOF_NAMES"]) == EXPECTED_DOF_NAMES
    assert set(constants["GRIPPER_JOINT_NAMES"]) == {"gripper_joint1", "gripper_joint2"}
    limits = constants["EXECUTION_POSITION_LIMITS"]
    assert set(limits) == ARM_JOINT_NAMES
    for name, (lower, upper) in NARROW_EXECUTION_LIMITS.items():
        assert limits[name] == (pytest.approx(lower), pytest.approx(upper)), name
    assert "joint_limits.yaml" in probe_source  # cites the planning config
    assert "Assuming maximum bounds" in probe_source  # never relied on


def test_probe_readiness_requires_the_full_set_including_fk(probe_source: str) -> None:
    """No motion before clock, states, broadcaster, controllers, actions, FK."""
    start = probe_source.index("def run_probe")
    readiness_start = probe_source.index("readiness phase", start)
    readiness_end = probe_source.index("at-rest baseline capture", readiness_start)
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
        "fk_service_ready",
        "broadcast_states",
    ):
        assert token in readiness, token
    # The FK client is created against the pinned /compute_fk service name;
    # the IK preflight client resolves the ready candidate name.
    assert "node.create_client(GetPositionFK, SERVICE_FK)" in probe_source
    assert "node.create_client(GetPositionIK, name) for name in SERVICE_IK_CANDIDATES" in probe_source
    assert '"ik_service_name": ik_name' in probe_source
    constants = module_constants(probe_source)
    assert constants["MIN_BASELINE_STATE_MESSAGES"] == 3
    assert constants["STATE_FRESH_SECONDS"] <= 5.0
    assert constants["EXACT_STATE_MATCH_TOLERANCE"] <= 1.0e-3
    assert "is_valid_state" in probe_source
    assert 'len(msg.velocity) not in (0, len(EXPECTED_DOF_NAMES))' in probe_source
    assert 'result["baseline_fresh"]' in probe_source
    assert 'result["baseline_matches_broadcast"]' in probe_source


def test_probe_baseline_is_captured_at_rest_before_the_target(
    probe_source: str,
) -> None:
    """At-rest settle gate (P1.B pattern) runs before the FK derivation."""
    constants = module_constants(probe_source)
    assert constants["SETTLE_STABLE_SAMPLES"] >= 5
    assert constants["SETTLE_STABLE_DELTA_MAX"] <= 1.0e-3
    rest_index = probe_source.index("at-rest baseline capture")
    fk_index = probe_source.index("FK target source")
    assert rest_index < fk_index, "the arm must be at rest before deriving a target"
    body = probe_source[rest_index:fk_index]
    assert "stable_count" in body
    assert "max_step(states)" in body
    assert 'result["baseline_at_rest"]' in probe_source
    assert "baseline_at_rest" in probe_source[probe_source.index("def run_probe") :]


def test_probe_fk_request_matches_the_service_contract(probe_source: str) -> None:
    """GetPositionFK with link6 in base_link over the full live joint state."""
    body = probe_source[probe_source.index("def call_fk") :]
    body = body[: body.index("def displaced_pose")]
    assert "request.header.frame_id = BASE_FRAME" in body
    assert "request.fk_link_names = [TIP_LINK]" in body
    assert "request.robot_state.joint_state.name" in body
    assert "request.robot_state.joint_state.position" in body
    assert "call_async(request)" in body
    assert "error_code.val) != SUCCESS_CODE" in body
    # The target is derived from the CAPTURED BASELINE (live state), and the
    # verification recomputes FK from the final live state.
    assert "call_fk(fk_client, baseline)" in probe_source
    assert "call_fk(fk_client, final_state)" in probe_source
    assert 'result["fk_source_state"]' in probe_source
    assert 'result["target_fk"]' in probe_source
    assert 'result["fk_request"]' in probe_source


def test_probe_target_is_fk_derived_not_hardcoded(probe_source: str) -> None:
    """The goal pose flows from the FK response into the request builder."""
    constants = module_constants(probe_source)
    fk_index = probe_source.index("target_fk = call_fk")
    build_index = probe_source.index('request = build_pose_request(target_pose)')
    assert fk_index < build_index, "the target pose must exist before the request"
    builder = probe_source[probe_source.index("def build_pose_request") :]
    builder = builder[: builder.index("def admit_pose_request")]
    assert "pose[" in builder  # built from the passed-in pose, not literals
    assert 'pose["position"]' in builder
    assert 'pose["orientation"]' in builder
    assert constants["POSE_DISPLACEMENT_M"] == pytest.approx(0.005)
    assert list(constants["DISPLACEMENT_AXES"]) == ["x", "z"]
    assert "def displaced_pose" in probe_source
    assert 'target_pose = candidate' in probe_source  # displacement only when proved


def test_probe_displacement_requires_an_ik_reachability_preflight(
    probe_source: str,
) -> None:
    """The +0.005 m shift is applied ONLY when GetPositionIK proves it."""
    body = probe_source[probe_source.index("def ik_preflight") :]
    body = body[: body.index("def build_pose_request")]
    assert "request.ik_request.group_name = PLANNING_GROUP" in body
    assert "request.ik_request.ik_link_name = TIP_LINK" in body
    assert "request.ik_request.avoid_collisions = True" in body
    assert "EXECUTION_POSITION_LIMITS[name]" in body
    assert "ADMISSION_MARGIN" in body
    section = probe_source[probe_source.index("displacement preflight") :]
    section = section[: section.index("strict admission")]
    assert "ik_preflight(ik_client, baseline, candidate)" in section
    assert 'evidence["proved"]' in section
    assert 'displacement["applied"] = True' in section
    assert "exact FK pose used" in section  # the fallback reason is recorded


def test_quaternion_helpers_are_mathematically_correct(probe_source: str) -> None:
    """The extracted helpers reject bad input and measure angles correctly."""
    namespace = extract_functions(
        probe_source,
        {"quat_norm", "quaternion_is_admissible", "quat_normalize", "quat_angle_between"},
    )
    good = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 2.0}
    admissible, reason = namespace["quaternion_is_admissible"](good)
    assert admissible and reason == "ok"
    normalized = namespace["quat_normalize"](good)
    assert namespace["quat_norm"](normalized) == pytest.approx(1.0)
    admissible, _ = namespace["quaternion_is_admissible"](
        {"x": 0.0, "y": 0.0, "z": 0.0, "w": 0.0}
    )
    assert not admissible  # near-zero quaternion rejected
    admissible, _ = namespace["quaternion_is_admissible"](
        {"x": float("nan"), "y": 0.0, "z": 0.0, "w": 1.0}
    )
    assert not admissible  # nonfinite quaternion rejected
    half_sqrt = math.sqrt(0.5)
    identity = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
    quarter_turn = {"x": 0.0, "y": 0.0, "z": half_sqrt, "w": half_sqrt}
    assert namespace["quat_angle_between"](identity, identity) == pytest.approx(0.0)
    assert namespace["quat_angle_between"](identity, quarter_turn) == pytest.approx(
        math.pi / 2
    )


def test_probe_pose_goal_constraints_and_tolerances(probe_source: str) -> None:
    """One sphere position constraint + one orientation constraint, bounded."""
    constants = module_constants(probe_source)
    # Verification thresholds are the mandated bounds and are never weakened.
    assert constants["POSITION_TOLERANCE_M"] == pytest.approx(0.01)
    assert constants["POSITION_TOLERANCE_M"] <= 0.01
    assert constants["ORIENTATION_TOLERANCE_RAD"] == pytest.approx(0.03)
    assert constants["ORIENTATION_TOLERANCE_RAD"] <= 0.03
    # The REQUESTED constraints are tighter than the verification thresholds
    # so the planner cannot legally settle on the tolerance boundary.
    assert constants["GOAL_POSITION_TOLERANCE_M"] == pytest.approx(0.005)
    assert constants["GOAL_POSITION_TOLERANCE_M"] < constants["POSITION_TOLERANCE_M"]
    assert constants["GOAL_ORIENTATION_TOLERANCE_RAD"] == pytest.approx(0.015)
    assert (
        constants["GOAL_ORIENTATION_TOLERANCE_RAD"]
        < constants["ORIENTATION_TOLERANCE_RAD"]
    )
    body = probe_source[probe_source.index("def build_pose_request") :]
    body = body[: body.index("def admit_pose_request")]
    assert 'request.group_name = PLANNING_GROUP' in body
    assert "sphere.type = SolidPrimitive.SPHERE" in body
    assert "sphere.dimensions = [GOAL_POSITION_TOLERANCE_M]" in body
    assert "position.header.frame_id = BASE_FRAME" in body
    assert "position.link_name = TIP_LINK" in body
    assert "goal.position_constraints.append(position)" in body
    assert "orientation.header.frame_id = BASE_FRAME" in body
    assert "orientation.link_name = TIP_LINK" in body
    assert "orientation.absolute_x_axis_tolerance = GOAL_ORIENTATION_TOLERANCE_RAD" in body
    assert "orientation.absolute_y_axis_tolerance = GOAL_ORIENTATION_TOLERANCE_RAD" in body
    assert "orientation.absolute_z_axis_tolerance = GOAL_ORIENTATION_TOLERANCE_RAD" in body
    assert "goal.orientation_constraints.append(orientation)" in body
    assert "from moveit_msgs.msg import" in probe_source
    assert "PositionConstraint" in probe_source and "OrientationConstraint" in probe_source
    assert "Constraints" in probe_source


def test_probe_admission_runs_before_the_goal_send(probe_source: str) -> None:
    """Strict caller-side admission; MoveIt clamping is never relied on."""
    admission_index = probe_source.index("strict admission")
    send_index = probe_source.index('"outbound",')
    assert admission_index < send_index, "the goal must be admitted before sending"
    body = probe_source[admission_index:send_index]
    assert 'admission = admit_pose_request(request, target_pose)' in body
    assert 'result["admission"] = admission' in body
    assert "pose goal admission failed" in probe_source
    checker = probe_source[probe_source.index("def admit_pose_request") :]
    checker = checker[: checker.index("def serialize_pose")]
    assert "no simultaneous joint target" in checker
    assert "no_joint_target" in checker
    assert "header.frame_id != BASE_FRAME" in checker
    assert "link_name != TIP_LINK" in checker
    assert "radius > POSITION_TOLERANCE_M" in checker
    assert "ORIENTATION_TOLERANCE_RAD" in checker
    assert "quaternion_is_admissible" in checker
    assert "not normalized" in checker
    # The actual wire send lives in the shared execution helper.
    assert "send_goal_async(goal)" in probe_source


def test_probe_movegroup_goal_is_plan_and_execute(probe_source: str) -> None:
    """plan_only False, replan False, empty start state, request contract."""
    constants = module_constants(probe_source)
    assert 'goal.planning_options.plan_only = False' in probe_source
    assert 'goal.planning_options.replan = False' in probe_source
    assert ".start_state" not in probe_source  # start state stays empty
    assert constants["ALLOWED_PLANNING_TIME"] == pytest.approx(5.0)
    assert constants["MAX_VELOCITY_SCALING"] == pytest.approx(0.1)
    assert constants["MAX_ACCELERATION_SCALING"] == pytest.approx(0.1)
    assert "STATUS_SUCCEEDED = GoalStatus.STATUS_SUCCEEDED" in probe_source
    assert "from action_msgs.msg import GoalStatus" in probe_source
    assert constants["SUCCESS_CODE"] == 1
    assert '"plan_only": False' in probe_source  # recorded in the request log


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
    assert 'summary["status"] == STATUS_SUCCEEDED' in probe_source
    assert 'summary["error_code"] == SUCCESS_CODE' in probe_source


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


def test_probe_subscribes_command_and_both_state_topics(probe_source: str) -> None:
    """/isaac_joint_command and both state topics are monitored."""
    assert "TOPIC_CLOCK, monitor.on_clock" in probe_source
    assert "TOPIC_JOINT_STATES, monitor.on_isaac_state" in probe_source
    assert "TOPIC_BROADCAST_STATES, monitor.on_broadcast_state" in probe_source
    assert "TOPIC_JOINT_COMMANDS, monitor.on_command" in probe_source
    assert "command_count" in probe_source
    assert "command_unknown_names" in probe_source
    assert "command_min" in probe_source and "command_max" in probe_source
    constants = module_constants(probe_source)
    assert constants["MIN_COMMAND_MESSAGES"] == 1
    assert constants["MIN_POST_COMMAND_STATE_MESSAGES"] == 2
    assert 'result["command_stats"]' in probe_source
    assert "host_correlation" in probe_source


def test_probe_verifies_the_final_pose_with_live_fk(probe_source: str) -> None:
    """Final live FK within the request tolerances: 0.01 m and 0.03 rad."""
    constants = module_constants(probe_source)
    body = probe_source[probe_source.index("Bounded convergence wait") :]
    body = body[: body.index("def command_stats")]
    assert 'fk_result["position_error_m"] <= POSITION_TOLERANCE_M' in body
    assert 'fk_result["orientation_error_rad"] <= ORIENTATION_TOLERANCE_RAD' in body
    assert '"fk_verification"' in probe_source
    assert '"position_error_m"' in probe_source
    assert '"orientation_error_rad"' in probe_source
    assert '"fk_source_state"' in probe_source
    assert '"fk_pose"' in probe_source
    assert "quat_angle_between" in body  # angular error vs the requested pose
    assert "settle_deadline = time.monotonic() + settle_timeout" in body
    assert '"settled_at_rest"' in probe_source
    assert 'summary["feedback_passed"] = bool(' in body


def test_probe_return_to_baseline_is_a_real_fk_pose_action(
    probe_source: str,
) -> None:
    """A second plan-and-execute goal targets the exact baseline FK pose."""
    assert probe_source.count("execute_pose_goal(") >= 3  # def + 2 calls
    body = probe_source[probe_source.index("return to baseline") :]
    assert "return_request = build_pose_request(target_fk)" in body
    assert 'admit_pose_request(return_request, target_fk)' in body
    assert '"return",' in body
    assert 'result["return"] = return_summary' in body
    assert 'result["passed"] = bool(outbound["passed"] and return_summary["passed"])' in body


def test_probe_one_json_line_and_exit_code_contract(probe_source: str) -> None:
    """Exactly one final JSON result line decides the exit code."""
    assert 'SCHEMA = "rocobrick-p1-real-pose-probe/v1"' in probe_source
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


def test_probe_has_no_undefined_names() -> None:
    """No NameError can kill the probe before its first JSON line.

    The offline suite cannot import the probe (no rclpy on the host), so a
    missing module constant would otherwise only surface inside the live
    container run; ruff's F821 catches it here instead.
    """
    ruff = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--select", "F821", str(PROBE_PATH)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if ruff.returncode not in (0, 1):
        pytest.skip(f"ruff not runnable: {ruff.stderr.strip()[:200]}")
    assert ruff.returncode == 0, ruff.stdout


# ------------------------------------------------------------------- smoke


def test_smoke_exists_and_is_bash(smoke_source: str) -> None:
    """The smoke script must exist with a bounded shebang header."""
    assert SMOKE_PATH.is_file()
    assert smoke_source.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in smoke_source


def test_smoke_has_no_forbidden_scope_tokens(smoke_source: str) -> None:
    """No TCP frame, no grasp, no example1, no piper-run in the smoke."""
    for token in ("tcp_link", "grasp", "example1", "piper-run"):
        assert token not in smoke_source, token


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


def test_smoke_publishes_no_port(smoke_source: str) -> None:
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
    assert "gpu-diagnostic-pose" in smoke_source
    assert "pkill" not in smoke_source
    assert "killall" not in smoke_source


def test_smoke_starts_the_host_bridge_with_long_bounds(smoke_source: str) -> None:
    """Host bridge via isaac-ros-env.sh, hold/time limits, marker wait."""
    assert "tools/isaac-ros-env.sh" in smoke_source
    assert "p05_host_bridge.py" in smoke_source
    assert 'MARKER="p0.5_host_bridge_ready"' in smoke_source
    assert 'HOST_EVIDENCE="$OUT_DIR/host-pose-bridge.json"' in smoke_source
    assert 'HOST_LOG="$OUT_DIR/host-pose-bridge.log"' in smoke_source
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
    assert 'CONTAINER_NAME="rocobrick-p1-real-pose-$$"' in smoke_source
    assert '-v "$WS_DIR:/workspace/ros_ws"' in smoke_source
    assert '-v "$RUNTIME:/dds:ro"' in smoke_source
    assert '-e FASTRTPS_DEFAULT_PROFILES_FILE=/dds/fastdds-container-mesh.xml' \
        in smoke_source
    assert (
        "ros2 launch rocobrick_piper_l_moveit_config move_group.launch.py"
        in smoke_source
    )
    assert "use_sim_time:=true" in smoke_source
    assert "p1_real_pose_probe.py" in smoke_source
    assert "--goal-timeout" in smoke_source and "--settle-timeout" in smoke_source
    assert "ros2 control list_controllers" in smoke_source
    # The inner readiness wait covers the planning service AND /compute_fk.
    assert "grep -q compute_fk" in smoke_source
    assert "grep -q plan_kinematic_path" in smoke_source


def test_smoke_captures_the_in_namespace_listener_evidence(
    smoke_source: str,
) -> None:
    """Any in-namespace TCP listener must be loopback-bound (no boundary)."""
    assert "net-tcp-listeners-pose.txt" in smoke_source
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
    assert 'AGGREGATE="$OUT_DIR/pose-target.json"' in smoke_source
    assert "artifacts/phase1-real" in smoke_source
    assert "rocobrick-p1-real-pose-smoke/v1" in smoke_source
    assert "rocobrick-p1-real-pose-probe/v1" in smoke_source
    for token in (
        '"readiness"',
        '"probe_passed"',
        '"pose_outbound_passed"',
        '"pose_fk_verification"',
        '"pose_return_passed"',
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
    # The aggregate FK gate pins the request tolerances verbatim.
    assert 'outbound_fk["position_error_m"] <= 0.01' in smoke_source
    assert 'outbound_fk["orientation_error_rad"] <= 0.03' in smoke_source
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
    assert '"host-pose-bridge.log"' in smoke_source
    assert '"container-pose.log"' in smoke_source
    assert '"pose-probe.log"' in smoke_source
    assert '"move_group-pose.log.tail"' in smoke_source


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
        pytest.skip("live P1.C artifact not collected yet")
    report = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert report["schema"] == "rocobrick-p1-real-pose-smoke/v1"
    assert set(report["checks"]) == {
        "p1a_prerequisite",
        "host_marker",
        "container_profile_client_only_no_listener",
        "readiness",
        "probe_passed",
        "pose_outbound_passed",
        "pose_fk_verification",
        "pose_return_passed",
        "command_observed",
        "host_commands_applied",
        "host_cleanup_orderly",
        "log_scan_clean",
        "bounded_and_no_leftovers",
    }
    assert report["passed"] == all(report["checks"].values())
    probe = report.get("probe") or {}
    assert probe.get("schema") == "rocobrick-p1-real-pose-probe/v1"
    assert probe.get("tip_link") == "link6"
    assert probe.get("tcp_status") == "unresolved"
    assert probe.get("target_fk")  # the target really came from the FK service
    outbound = probe.get("outbound") or {}
    fk = outbound.get("fk_verification") or {}
    assert fk.get("passed") is True
    assert fk.get("position_error_m") <= 0.01
    assert fk.get("orientation_error_rad") <= 0.03
