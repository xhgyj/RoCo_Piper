"""ROS 2 / MoveIt runtime adapter around the pure P1 library layer (R2).

Connects the accepted transport-free R1 modules (moveit_contract,
moveit_requests, admission, readiness, feedback, moveit_execution) to a live
ROS 2 Humble move_group stack. The ROS behavior mirrors the live-verified
P1.B / P1.C probes (tools/p1_real_joint_probe.py, tools/p1_real_pose_probe.py
and artifacts/phase1-real/*.json) without importing them: readiness before
any motion, strict caller-side admission, plan_only=false MoveGroup goals,
bounded deadlines with explicit cancellation, R1 waypoint inspection of the
returned trajectory and R1 final-feedback gates. Results never fake success:
when the exact final state cannot be observed the runtime returns FAILED with
explicit details.

All ROS / MoveIt imports are LAZY (only inside registration and send
methods), so this module imports cleanly on a host without rclpy. Every
transport boundary (message types, action clients, goal construction, the
list_controllers request and the GoalStatus constants) is injectable through
:data:`RosMoveItFactories`, and the clock/spin/sleep hooks are injectable too,
so offline fakes can drive the full flow. No Isaac, no tools imports; no node
is created here - the caller injects an existing node.

Per-robot routing (A5): a runtime constructed with a
:class:`rocobrick.robot_runtime_profile.RobotRuntimeProfile` takes EVERY
endpoint from that profile (robot-scoped state topics, controller/FK/plan
services, arm+gripper actions, MoveGroup action, MoveGroup planning-scene
services) and no canonical default is ever substituted, so a piper_1 runtime
cannot observe, command or report on piper_0. The injected scene-state
Trigger stays the single global host-bridge service of the A4 contract and
is sliced per robot through :func:`robot_scene_view`. State messages are
normalized onto the canonical joint names at ingest and are rejected when
they do not carry exactly this robot's ``<robot_id>_<joint>`` names, so
cross-injected samples can never become readiness or feedback evidence.
:class:`RosMoveItRobotSession` hosts one isolated runtime per plan robot on
ONE node, one ROS domain and one shared /clock; its
:meth:`RosMoveItRobotSession.wait_until_ready` requires every robot's
capability readiness (joints, controllers, MoveGroup, FK, gripper and the
robot-scoped scene query).
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping

from rocobrick.admission import (
    strict_action_admission,
    validate_trajectory_waypoints,
)
from rocobrick.feedback import (
    max_joint_drift,
    max_joint_error,
    orientation_error_rad,
    position_error_m,
    state_is_stable,
)
from rocobrick.gripper import fingers_to_logical_gripper, logical_gripper_to_fingers
from rocobrick.moveit_contract import (
    ACTION_ARM_FOLLOW,
    ACTION_GRIPPER_FOLLOW,
    ARM_JOINT_NAMES,
    BASE_FRAME,
    EXACT_STATE_MATCH_TOLERANCE,
    GRIPPER_JOINT_NAMES,
    JOINT_NAMES,
    MAX_ORIENTATION_TOLERANCE_RAD,
    MAX_POSITION_TOLERANCE_M,
    MOVE_GROUP_ACTION_CANDIDATES,
    QUAT_NORM_MIN,
    SERVICE_COMPUTE_FK,
    SERVICE_LIST_CONTROLLERS,
    SERVICE_PLAN,
    SUCCESS_CODE,
    TCP_LINK,
    TIP_LINK,
    TOPIC_CLOCK,
    TOPIC_ISAAC_JOINT_COMMAND,
    TOPIC_ISAAC_JOINT_STATES,
    TOPIC_JOINT_STATES,
    AdmissionDecision,
    GripperTarget,
    JointTarget,
    Link6PoseTarget,
    TcpPoseTarget,
)
from rocobrick.moveit_execution import ExecutionResult, ExecutionStatus
from rocobrick.planning_scene import (
    AxisAlignedBox,
    PlanningSceneResult,
    PlanningSceneUpdate,
    scene_digest,
)
from rocobrick.readiness import (
    clock_is_increasing,
    controllers_are_active,
    state_is_complete,
    states_match,
)
from rocobrick.robot_runtime_profile import (
    RobotRuntimeProfile,
    assert_profiles_disjoint,
)

PLANNING_GROUP = "arm"
# MoveGroup group for project-calibrated TCP goals (SRDF chain arm_tcp).
TCP_PLANNING_GROUP = "arm_tcp"
ALLOWED_PLANNING_TIME_S = 5.0
NUM_PLANNING_ATTEMPTS = 10
MAX_SCALING_FACTOR = 0.1

# Formal planning-scene service (moveit_msgs/srv/ApplyPlanningScene).
SERVICE_APPLY_PLANNING_SCENE = "/apply_planning_scene"
SERVICE_GET_PLANNING_SCENE = "/get_planning_scene"
PLANNING_SCENE_ALLOWED_COLLISION_MATRIX = 128
PLANNING_SCENE_ROBOT_STATE_ATTACHED_OBJECTS = 4
PLANNING_SCENE_WORLD_OBJECT_NAMES = 8
PLANNING_SCENE_WORLD_OBJECT_GEOMETRY = 16
PLANNING_SCENE_WORLD_OBJECT_READBACK = (
    PLANNING_SCENE_WORLD_OBJECT_NAMES | PLANNING_SCENE_WORLD_OBJECT_GEOMETRY
)
# Live host-bridge scene-state service (std_srvs/srv/Trigger) answering the
# canonical JSON scene-state document of schema rocobrick-scene-state/v1.
SERVICE_SCENE_STATE = "/p05_host_bridge/scene_state"
SCENE_STATE_SCHEMA = "rocobrick-scene-state/v1"

# Bounded-wait pacing and deadlines (mirrors the live P1 probe discipline).
POLL_INTERVAL_S = 0.2
WAIT_STEP_S = 0.05
ACCEPT_TIMEOUT_S = 15.0
CANCEL_TIMEOUT_S = 10.0
SERVICE_CALL_TIMEOUT_S = 5.0
# Bound of the read-only robot-scoped scene probe used by readiness.
SCENE_PROBE_TIMEOUT_S = 2.0

# Feedback evidence gates (mirrors the live P1 probe final-state contract).
STATE_FRESH_SECONDS = 3.0
STATE_HISTORY = 64
MIN_POST_COMMAND_STATES = 2
MIN_COMMAND_MESSAGES = 1
FINAL_ARM_ERROR_MAX = 0.04
GRIPPER_DRIFT_MAX = 0.01

# Bounded post-action convergence wait. LIVE R2 EVIDENCE (the first
# artifacts/phase2-adapter run): the MoveGroup action completes while the
# lagging simulated arm is still converging (its commanded targets stream to
# Isaac at a lower rate than the state feedback), so an instantaneous sample
# taken at the action result misjudged a converged goal as a 0.048 rad joint
# error. The live-verified P1 settle gate is therefore reproduced here: the
# final sample is taken only after FINAL_SETTLE_SAMPLES consecutive live
# samples move less than FINAL_SETTLE_STEP_MAX, never later than the bound.
FINAL_CONVERGENCE_TIMEOUT_S = 20.0
FINAL_CONVERGENCE_POLL_S = 0.2
FINAL_SETTLE_SAMPLES = 8
FINAL_SETTLE_STEP_MAX = 5.0e-4

FkVerification = tuple[tuple[float, float, float], tuple[float, float, float, float]]
FkVerifier = Callable[[Mapping[str, float]], "FkVerification | None"]


@dataclass(frozen=True)
class RuntimeTopics:
    """Pinned Isaac<->MoveIt topic contract (vendor ros2_control xacro)."""

    clock: str = TOPIC_CLOCK
    isaac_joint_states: str = TOPIC_ISAAC_JOINT_STATES
    joint_states: str = TOPIC_JOINT_STATES
    isaac_joint_command: str = TOPIC_ISAAC_JOINT_COMMAND


@dataclass(frozen=True)
class RuntimeServices:
    """Live-resolved service names recorded in the P1 evidence.

    ``apply_planning_scene`` is the formal R5.A planning-scene service and
    ``scene_state`` the host-bridge Trigger service; the P1-evidence names
    above are unchanged.
    """

    list_controllers: str = SERVICE_LIST_CONTROLLERS
    plan: str = SERVICE_PLAN
    compute_fk: str = SERVICE_COMPUTE_FK
    apply_planning_scene: str = SERVICE_APPLY_PLANNING_SCENE
    get_planning_scene: str = SERVICE_GET_PLANNING_SCENE
    scene_state: str = SERVICE_SCENE_STATE


@dataclass(frozen=True)
class RuntimeActions:
    """Arm and gripper follow actions plus the first MoveGroup candidate.

    The gripper follow action is registered for gripper goals only and is
    deliberately NOT part of the motion-readiness gates.
    """

    arm_follow: str = ACTION_ARM_FOLLOW
    move_group: str = MOVE_GROUP_ACTION_CANDIDATES[0]
    gripper_follow: str = ACTION_GRIPPER_FOLLOW


def robot_endpoint_root(profile: RobotRuntimeProfile) -> str:
    """Namespace prefix of a profile's endpoints ("" for the ROS root)."""
    namespace = str(profile.namespace)
    if namespace in ("", "/"):
        return ""
    return namespace


def runtime_topics_for_profile(profile: RobotRuntimeProfile) -> RuntimeTopics:
    """The profile's exact robot-scoped state topics (shared ``/clock``)."""
    return RuntimeTopics(
        clock=profile.clock,
        isaac_joint_states=profile.isaac_joint_states,
        joint_states=profile.joint_states,
        isaac_joint_command=profile.isaac_joint_command,
    )


def runtime_services_for_profile(profile: RobotRuntimeProfile) -> RuntimeServices:
    """The profile's robot-scoped MoveIt services.

    ``list_controllers``/``plan``/``compute_fk`` come from the profile
    verbatim. A namespaced move_group exposes its planning-scene pair below
    the same namespace, so ``apply_planning_scene``/``get_planning_scene``
    follow the profile namespace. The host-bridge scene-state Trigger is a
    SINGLE global service by the A4 contract (robot slicing happens in the
    response), never a namespaced per-robot endpoint, so it is never derived
    from the namespace.
    """
    root = robot_endpoint_root(profile)
    return RuntimeServices(
        list_controllers=profile.list_controllers,
        plan=profile.plan,
        compute_fk=profile.compute_fk,
        apply_planning_scene=root + SERVICE_APPLY_PLANNING_SCENE,
        get_planning_scene=root + SERVICE_GET_PLANNING_SCENE,
        scene_state=SERVICE_SCENE_STATE,
    )


def runtime_actions_for_profile(profile: RobotRuntimeProfile) -> RuntimeActions:
    """The profile's exact arm/gripper/MoveGroup actions."""
    return RuntimeActions(
        arm_follow=profile.arm_follow,
        move_group=profile.move_group_action,
        gripper_follow=profile.gripper_follow,
    )


# Every capability component a robot-scoped operation readiness must satisfy
# (A5): the motion-readiness components of one robot plus its gripper follow
# action and its robot-scoped scene observation.
CAPABILITY_COMPONENTS = (
    "clock_increasing",
    "isaac_state_complete",
    "joint_state_complete",
    "broadcaster_match",
    "controllers_active",
    "arm_action_ready",
    "move_group_action_ready",
    "gripper_action_ready",
    "compute_fk_ready",
    "scene_query_ready",
)


@dataclass(frozen=True)
class RobotCapabilityReadiness:
    """One robot's capability readiness verdict, with its exact endpoints.

    Extends :class:`ReadinessSnapshot` with the gripper follow action and the
    robot-scoped scene-observation service, keeps the robot identity and the
    endpoints the verdict was measured on, and passes only when every
    component holds. Every component is measured on THIS robot's own
    subscriptions and clients, so a verdict can never mix robots.
    """

    robot_id: str = ""
    namespace: str = ""
    clock_increasing: bool = False
    isaac_state_complete: bool = False
    joint_state_complete: bool = False
    broadcaster_match: bool = False
    controllers_active: bool = False
    arm_action_ready: bool = False
    move_group_action_ready: bool = False
    gripper_action_ready: bool = False
    compute_fk_ready: bool = False
    scene_query_ready: bool = False
    endpoints: tuple[tuple[str, str], ...] = ()

    @property
    def passed(self) -> bool:
        """True only when every capability component of this robot holds."""
        return all(getattr(self, name) for name in CAPABILITY_COMPONENTS)

    @classmethod
    def component_names(cls) -> tuple[str, ...]:
        """The exact capability component vocabulary."""
        return CAPABILITY_COMPONENTS

    def serialize(self) -> dict[str, object]:
        """JSON-safe per-robot readiness report."""
        return {
            "robot_id": self.robot_id,
            "namespace": self.namespace,
            "passed": self.passed,
            "components": {
                name: bool(getattr(self, name)) for name in CAPABILITY_COMPONENTS
            },
            "endpoints": {name: value for name, value in self.endpoints},
        }


def robot_scene_view(
    document: Mapping[str, object],
    robot_id: str,
    *,
    joint_names: Sequence[str] | None = None,
) -> dict[str, object]:
    """Robot-scoped slice of the A4 dual scene-state document.

    The global ``/p05_host_bridge/scene_state`` document carries the shared
    ``sim_time_s``/``bricks``/``connections`` plus an additive
    ``robots[robot_id]`` block with that robot's ``robot_id``, ``sim_time_s``,
    ``gripper`` (exactly ``gripper_joint1``/``gripper_joint2``), ``opening_m``
    and the optional ``opening_error``. The returned view carries the shared
    scene plus only this robot's observation, in the shape the scene-query
    filter consumes. There is NO fallback: a missing ``robots`` mapping, a
    missing block for ``robot_id``, a block owned by another robot or a
    malformed gripper block raise ``RuntimeError`` instead of answering with
    another robot's (or the top-level single-robot) observation.

    When the block carries the host binding evidence ``joint_names`` it is
    validated: it must be exactly the device joint names of this robot
    (``joint_names`` when supplied, fail-closed on any mismatch), so a scene
    snapshot bound to a designated robot can never be presented as another
    robot's observation.
    """
    if not isinstance(robot_id, str) or not robot_id or "/" in robot_id:
        raise ValueError("robot_id must be a non-empty ROS-safe id")
    if (
        not isinstance(document, Mapping)
        or document.get("schema") != SCENE_STATE_SCHEMA
    ):
        raise RuntimeError(
            f"scene-state schema must be exactly {SCENE_STATE_SCHEMA!r}"
        )
    robots = document.get("robots")
    if not isinstance(robots, Mapping):
        raise RuntimeError(
            "scene-state document carries no robots mapping; refusing to "
            f"fall back to another robot's observation for {robot_id!r}"
        )
    block = robots.get(robot_id)
    if not isinstance(block, Mapping):
        raise RuntimeError(
            f"scene-state document carries no block for robot {robot_id!r} "
            f"(present: {sorted(str(key) for key in robots)})"
        )
    if str(block.get("robot_id")) != robot_id:
        raise RuntimeError(
            f"scene-state block for robot {robot_id!r} is owned by "
            f"{block.get('robot_id')!r}"
        )
    bound_joint_names = block.get("joint_names")
    if bound_joint_names is not None:
        if (
            not isinstance(bound_joint_names, Sequence)
            or isinstance(bound_joint_names, (str, bytes, bytearray))
            or not bound_joint_names
            or not all(
                isinstance(name, str) and name for name in bound_joint_names
            )
        ):
            raise RuntimeError(
                f"scene-state binding joint_names of robot {robot_id!r} are "
                "malformed"
            )
        expected = (
            tuple(joint_names) if joint_names is not None else None
        )
        if expected is not None and (
            len(bound_joint_names) != len(expected)
            or set(str(name) for name in bound_joint_names) != set(expected)
        ):
            raise RuntimeError(
                f"scene-state block for robot {robot_id!r} is bound to a "
                "different robot's joint names; refusing the observation"
            )
    bricks = document.get("bricks")
    connections = document.get("connections")
    if not isinstance(bricks, Mapping) or not isinstance(connections, Mapping):
        raise RuntimeError(
            "scene-state document carries no shared bricks/connections mapping"
        )
    gripper = block.get("gripper")
    if not isinstance(gripper, Mapping) or set(gripper) != set(GRIPPER_JOINT_NAMES):
        raise RuntimeError(
            f"scene-state block for robot {robot_id!r} carries no complete "
            f"gripper reading {sorted(GRIPPER_JOINT_NAMES)}"
        )
    for name in GRIPPER_JOINT_NAMES:
        value = gripper[name]
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise RuntimeError(
                f"scene-state gripper joint {name!r} of robot {robot_id!r} "
                "is not a finite number"
            )
    sim_time = block.get("sim_time_s", document.get("sim_time_s"))
    if not isinstance(sim_time, (int, float)) or not math.isfinite(float(sim_time)):
        raise RuntimeError(
            f"scene-state block for robot {robot_id!r} carries no finite sim_time_s"
        )
    opening = block.get("opening_m")
    if opening is not None and (
        not isinstance(opening, (int, float)) or not math.isfinite(float(opening))
    ):
        raise RuntimeError(
            f"scene-state opening_m of robot {robot_id!r} is not a finite number"
        )
    view: dict[str, object] = {
        "schema": SCENE_STATE_SCHEMA,
        "robot_id": robot_id,
        "sim_time_s": float(sim_time),
        "bricks": bricks,
        "connections": connections,
        "gripper": {name: float(gripper[name]) for name in GRIPPER_JOINT_NAMES},
        "opening_m": None if opening is None else float(opening),
    }
    if "opening_error" in block:
        view["opening_error"] = block["opening_error"]
    if bound_joint_names is not None:
        view["joint_names"] = [str(name) for name in bound_joint_names]
    if "session_id" in document:
        view["session_id"] = document["session_id"]
    if "manifest_sha256" in document:
        view["manifest_sha256"] = document["manifest_sha256"]
    if "sequence" in document:
        view["sequence"] = document["sequence"]
    if "sequence" in block:
        view["robot_sequence"] = block["sequence"]
    if "tcp_pose" in block:
        view["tcp_pose"] = block["tcp_pose"]
    return view


@dataclass(frozen=True)
class ReadinessSnapshot:
    """One readiness evaluation over the R1 predicates plus transport flags."""

    clock_increasing: bool = False
    isaac_state_complete: bool = False
    joint_state_complete: bool = False
    broadcaster_match: bool = False
    controllers_active: bool = False
    arm_action_ready: bool = False
    move_group_action_ready: bool = False
    compute_fk_ready: bool = False

    @property
    def passed(self) -> bool:
        """True only when every readiness component holds."""
        return all(
            (
                self.clock_increasing,
                self.isaac_state_complete,
                self.joint_state_complete,
                self.broadcaster_match,
                self.controllers_active,
                self.arm_action_ready,
                self.move_group_action_ready,
                self.compute_fk_ready,
            )
        )


@dataclass(frozen=True)
class RosMoveItFactories:
    """Injected construction boundaries for every ROS/MoveIt type.

    Production instances are built lazily by
    :meth:`RosMoveItRuntime._default_factories` (real rclpy message classes).
    Offline fakes supply plain Python classes so the full goal construction,
    readiness and execution flow runs without rclpy installed.
    """

    clock_type: Callable[[], Any]
    joint_state_type: Callable[[], Any]
    list_controllers_type: Callable[[], Any]
    list_controllers_request_factory: Callable[[], Any]
    get_motion_plan_type: Callable[[], Any]
    get_position_fk_type: Callable[[], Any]
    move_group_type: Callable[[], Any]
    follow_joint_trajectory_type: Callable[[], Any]
    move_group_goal_factory: Callable[[], Any]
    motion_plan_request_factory: Callable[[], Any]
    constraints_factory: Callable[[], Any]
    joint_constraint_factory: Callable[[], Any]
    position_constraint_factory: Callable[[], Any]
    orientation_constraint_factory: Callable[[], Any]
    solid_primitive_factory: Callable[[], Any]
    pose_factory: Callable[[], Any]
    action_client_factory: Callable[[Any, Any, str], Any]
    goal_status_type: Callable[[], Any]
    # R5.A planning-scene transport boundaries; production types are resolved
    # lazily in _default_factories, offline fakes may inject plain classes.
    apply_planning_scene_type: Callable[[], Any] | None = None
    apply_planning_scene_request_factory: Callable[[], Any] | None = None
    collision_object_factory: Callable[[], Any] | None = None
    attached_collision_object_factory: Callable[[], Any] | None = None
    allowed_collision_entry_factory: Callable[[], Any] | None = None
    get_planning_scene_type: Callable[[], Any] | None = None
    get_planning_scene_request_factory: Callable[[], Any] | None = None
    # Scene-state Trigger and gripper trajectory transport boundaries; the
    # same lazy-resolved/optional-injection pattern as the planning scene.
    trigger_type: Callable[[], Any] | None = None
    trigger_request_factory: Callable[[], Any] | None = None
    follow_joint_trajectory_goal_factory: Callable[[], Any] | None = None
    joint_trajectory_point_factory: Callable[[], Any] | None = None
    joint_tolerance_factory: Callable[[], Any] | None = None
    duration_factory: Callable[[], Any] | None = None


def _clock_nanos(message: Any) -> int:
    """Clock message as integer nanoseconds."""
    return int(message.clock.sec) * 1_000_000_000 + int(message.clock.nanosec)


def _parse_controller_states(future: Any) -> dict[str, str] | None:
    """Controller name/state mapping from a completed future, else None."""
    try:
        response = future.result()
        return {str(item.name): str(item.state) for item in response.controller}
    except Exception:
        return None


def _map_goal_status(status: Any, goal_status: Any) -> ExecutionStatus:
    """Terminal action status to ExecutionStatus; unknown values never succeed."""
    value = int(status)
    if value == int(goal_status.STATUS_SUCCEEDED):
        return ExecutionStatus.SUCCEEDED
    if value == int(goal_status.STATUS_CANCELED):
        return ExecutionStatus.CANCELED
    return ExecutionStatus.FAILED


def _terminal_status_name(status: Any, goal_status: Any) -> str:
    """Human-readable terminal action status name for the evidence details."""
    value = int(status)
    if value == int(goal_status.STATUS_SUCCEEDED):
        return "SUCCEEDED"
    if value == int(goal_status.STATUS_CANCELED):
        return "CANCELED"
    aborted = getattr(goal_status, "STATUS_ABORTED", None)
    if aborted is not None and value == int(aborted):
        return "ABORTED"
    return f"UNKNOWN({value})"


def _collision_box_matches(item: object, box: AxisAlignedBox) -> bool:
    """Check the read-back MoveIt box against the submitted geometry.

    Returns:
        True only for the same frame, shape, dimensions and center.
    """
    try:
        if item.header.frame_id != box.frame_id:
            return False
        if len(item.primitives) != 1 or len(item.primitive_poses) != 1:
            return False
        primitive = item.primitives[0]
        pose = item.primitive_poses[0]
        if primitive.type != 1 or len(primitive.dimensions) != 3:
            return False
        dimensions = tuple(float(value) for value in primitive.dimensions)
        center = (float(pose.position.x), float(pose.position.y), float(pose.position.z))
        quaternion = (
            float(pose.orientation.x), float(pose.orientation.y),
            float(pose.orientation.z), float(pose.orientation.w),
        )
    except (AttributeError, IndexError, TypeError, ValueError):
        return False
    return (
        all(math.isfinite(value) for value in (*dimensions, *center, *quaternion))
        and all(abs(a - b) <= 1e-5 for a, b in zip(dimensions, box.dimensions_m))
        and all(abs(a - b) <= 1e-5 for a, b in zip(center, box.center_m))
        and all(abs(a - b) <= 1e-5 for a, b in zip(quaternion, (0.0, 0.0, 0.0, 1.0)))
    )


def _attached_object_matches(
    item: object, *, object_id: str, link_name: str
) -> bool:
    """Require one attached object to carry exact object/link identity.

    Returns:
        True only when the attached-object identity and collision geometry are
        present; missing fields never fall back to a positive result.
    """
    try:
        attached = item
        collision = attached.object
        if collision.id != object_id or attached.link_name != link_name:
            return False
        return bool(collision.primitives or collision.meshes or collision.planes)
    except (AttributeError, TypeError, ValueError):
        return False


def _merge_allowed_collision_pairs(
    matrix: Any,
    pairs: tuple[tuple[str, str], ...],
    entry_factory: Callable[[], Any],
) -> Any:
    """Append allowed pairs without discarding MoveIt's existing ACM.

    ``ApplyPlanningScene`` treats a non-empty ACM in a diff as a replacement
    matrix.  The caller therefore reads the current full matrix first, then
    extends its rows and changes only the requested pairs.
    """
    try:
        names = [str(name) for name in matrix.entry_names]
        source_rows = [list(entry.enabled) for entry in matrix.entry_values]
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"current allowed collision matrix is unreadable: {error}") from error
    if not names or len(set(names)) != len(names):
        raise ValueError("current allowed collision matrix has no unique entries")
    if len(source_rows) != len(names) or any(
        len(row) != len(names) for row in source_rows
    ):
        raise ValueError(
            "current allowed collision matrix is not square: "
            f"entries={len(names)} rows={len(source_rows)}"
        )
    rows = [[bool(value) for value in row] for row in source_rows]
    for first, second in pairs:
        for name in (first, second):
            if name in names:
                continue
            names.append(name)
            for row in rows:
                row.append(False)
            rows.append([False] * len(names))
        first_index = names.index(first)
        second_index = names.index(second)
        rows[first_index][second_index] = True
        rows[second_index][first_index] = True
    matrix.entry_names = list(names)
    matrix.entry_values = []
    for row in rows:
        entry = entry_factory()
        entry.enabled = list(row)
        matrix.entry_values.append(entry)
    return matrix


class RosMoveItRuntime:
    """ROS-side MoveIt runtime; every transport boundary is injectable.

    The caller owns the node: nothing is created in ``__init__``. Call
    :meth:`register_subscriptions` and :meth:`register_clients` once, wait for
    :meth:`wait_until_ready` and only then send goals. QoS profiles default to
    the live P1 probe profiles (RELIABLE / BEST_EFFORT, KEEP_LAST depth 50)
    and are resolved lazily, so injected fakes can pass opaque placeholders.

    With a ``profile`` every endpoint is the profile's robot-scoped name and
    states are only accepted when they carry exactly that robot's device
    joint names; without a profile the historical canonical single-robot
    names are used (no profile ever falls back to them per endpoint).
    """

    def __init__(
        self,
        node: Any,
        *,
        profile: RobotRuntimeProfile | None = None,
        topics: RuntimeTopics | None = None,
        services: RuntimeServices | None = None,
        actions: RuntimeActions | None = None,
        base_frame: str | None = None,
        link6_link: str | None = None,
        tcp_link: str | None = None,
        arm_planning_group: str | None = None,
        tcp_planning_group: str | None = None,
        qos_sensor: Any = None,
        qos_state: Any = None,
        factories: RosMoveItFactories | None = None,
        fk_verifier: FkVerifier | None = None,
        tcp_fk_verifier: FkVerifier | None = None,
        planning_scene_applier: (
            Callable[[PlanningSceneUpdate, float], PlanningSceneResult] | None
        ) = None,
        clock: Callable[[], float] | None = None,
        spin: Callable[[], None] | None = None,
        sleep: Callable[[float], None] | None = None,
        poll_interval_s: float = POLL_INTERVAL_S,
        wait_step_s: float = WAIT_STEP_S,
    ) -> None:
        if profile is not None:
            if not isinstance(profile, RobotRuntimeProfile):
                raise ValueError(
                    "profile must be a RobotRuntimeProfile, got "
                    f"{type(profile).__name__}"
                )
            if topics is not None or services is not None or actions is not None:
                raise ValueError(
                    "topics/services/actions must not be overridden when a "
                    "RobotRuntimeProfile is supplied"
                )
            topics = runtime_topics_for_profile(profile)
            services = runtime_services_for_profile(profile)
            actions = runtime_actions_for_profile(profile)
        self.node = node
        self.profile = profile
        self.topics = topics if topics is not None else RuntimeTopics()
        self.services = services if services is not None else RuntimeServices()
        self.actions = actions if actions is not None else RuntimeActions()
        self._robot_id = profile.robot_id if profile is not None else ""
        self._namespace = profile.namespace if profile is not None else ""
        self._joint_names = (
            tuple(profile.joint_names) if profile is not None else JOINT_NAMES
        )
        self._arm_joint_names = (
            tuple(profile.arm_joint_names) if profile is not None else ARM_JOINT_NAMES
        )
        self._gripper_joint_names = (
            tuple(profile.gripper_joint_names)
            if profile is not None
            else GRIPPER_JOINT_NAMES
        )
        if (
            len(self._joint_names) != len(JOINT_NAMES)
            or len(self._arm_joint_names) != len(ARM_JOINT_NAMES)
            or len(self._gripper_joint_names) != len(GRIPPER_JOINT_NAMES)
            or self._joint_names
            != self._arm_joint_names + self._gripper_joint_names
        ):
            raise ValueError(
                "robot joint names must be the 6 arm + 2 gripper joints of "
                "the canonical order"
            )
        self._base_frame = (
            base_frame
            if base_frame is not None
            else (profile.base_frame if profile is not None else BASE_FRAME)
        )
        self._link6_link = (
            link6_link
            if link6_link is not None
            else (profile.link6_link if profile is not None else TIP_LINK)
        )
        self._tcp_link = (
            tcp_link
            if tcp_link is not None
            else (profile.tip_link if profile is not None else TCP_LINK)
        )
        self._arm_planning_group = (
            arm_planning_group
            if arm_planning_group is not None
            else (
                profile.link6_planning_group
                if profile is not None
                else PLANNING_GROUP
            )
        )
        self._tcp_planning_group = (
            tcp_planning_group
            if tcp_planning_group is not None
            else (
                profile.planning_group if profile is not None else TCP_PLANNING_GROUP
            )
        )
        self._qos_sensor = qos_sensor
        self._qos_state = qos_state
        self._factories = factories
        self._fk_verifier = fk_verifier
        self._tcp_fk_verifier = tcp_fk_verifier
        self._planning_scene_applier = planning_scene_applier
        self._clock_fn = clock if clock is not None else time.monotonic
        self._spin_fn = spin if spin is not None else (lambda: None)
        self._sleep_fn = sleep if sleep is not None else time.sleep
        self._poll_interval_s = float(poll_interval_s)
        self._wait_step_s = float(wait_step_s)
        self._lock = threading.Lock()
        self._clock_values: deque[int] = deque(maxlen=STATE_HISTORY)
        self._isaac_states: deque[dict[str, float]] = deque(maxlen=STATE_HISTORY)
        self._isaac_receipts: deque[float] = deque(maxlen=STATE_HISTORY)
        self._joint_states: deque[dict[str, float]] = deque(maxlen=STATE_HISTORY)
        self._isaac_state_count = 0
        self._command_count = 0
        self._controller_states: dict[str, str] = {}
        self._controllers_future: Any = None
        self._controllers_requested_at = 0.0
        self._subscriptions: dict[str, Any] = {}
        self._list_controllers_client: Any = None
        self._plan_client: Any = None
        self._fk_client: Any = None
        self._planning_scene_client: Any = None
        self._get_planning_scene_client: Any = None
        self._scene_state_client: Any = None
        self._arm_client: Any = None
        self._move_group_client: Any = None
        self._gripper_client: Any = None
        self._active_handle: Any = None
        # Keep both futures so a timeout can handle late goal acceptance and
        # wait for the action result after cancel_goal_async().
        self._active_goal_future: Any = None
        self._active_result_future: Any = None
        self._resolved_factories: RosMoveItFactories | None = None
        self.shutdown_errors: list[str] = []

    # --------------------------------------------------------------- identity

    @property
    def robot_id(self) -> str:
        """The profile's robot id, or "" for a canonical single-robot runtime."""
        return self._robot_id

    @property
    def namespace(self) -> str:
        """The profile's ROS namespace, or "" for the canonical runtime."""
        return self._namespace

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Device joint names of this runtime (canonical without a profile)."""
        return self._joint_names

    @property
    def arm_joint_names(self) -> tuple[str, ...]:
        """Device arm joint names of this runtime."""
        return self._arm_joint_names

    @property
    def gripper_joint_names(self) -> tuple[str, ...]:
        """Device gripper joint names of this runtime."""
        return self._gripper_joint_names

    @property
    def base_frame(self) -> str:
        """FK/pose base frame of this runtime."""
        return self._base_frame

    @property
    def link6_link(self) -> str:
        """Raw link6 flange frame of this runtime."""
        return self._link6_link

    @property
    def tip_link(self) -> str:
        """Calibrated skill TCP frame of this runtime."""
        return self._tcp_link

    @property
    def tcp_link(self) -> str:
        """Calibrated TCP frame of this runtime (alias of ``tip_link``)."""
        return self._tcp_link

    @property
    def arm_planning_group(self) -> str:
        """MoveIt group of the raw link6 arm chain of this runtime."""
        return self._arm_planning_group

    @property
    def tcp_planning_group(self) -> str:
        """MoveIt group of the calibrated TCP chain of this runtime."""
        return self._tcp_planning_group

    def endpoint_map(self) -> dict[str, str]:
        """Every transport endpoint this runtime measures, verbatim."""
        return {
            "clock": self.topics.clock,
            "isaac_joint_states": self.topics.isaac_joint_states,
            "joint_states": self.topics.joint_states,
            "isaac_joint_command": self.topics.isaac_joint_command,
            "list_controllers": self.services.list_controllers,
            "plan": self.services.plan,
            "compute_fk": self.services.compute_fk,
            "apply_planning_scene": self.services.apply_planning_scene,
            "get_planning_scene": self.services.get_planning_scene,
            "scene_state": self.services.scene_state,
            "arm_follow": self.actions.arm_follow,
            "move_group": self.actions.move_group,
            "gripper_follow": self.actions.gripper_follow,
            "base_frame": self._base_frame,
            "link6_link": self._link6_link,
            "tcp_link": self._tcp_link,
            "arm_planning_group": self._arm_planning_group,
            "tcp_planning_group": self._tcp_planning_group,
        }

    # ------------------------------------------------------------- transport

    def _default_factories(self) -> RosMoveItFactories:
        """Real ROS/MoveIt factories; the lazy ROS imports live only here."""
        from action_msgs.msg import GoalStatus
        from builtin_interfaces.msg import Duration
        from control_msgs.action import FollowJointTrajectory
        from control_msgs.msg import JointTolerance
        from controller_manager_msgs.srv import ListControllers
        from geometry_msgs.msg import Pose
        from moveit_msgs.action import MoveGroup
        from moveit_msgs.msg import (
            AllowedCollisionEntry,
            CollisionObject,
            Constraints,
            JointConstraint,
            MotionPlanRequest,
            OrientationConstraint,
            PositionConstraint,
        )
        from moveit_msgs.srv import (
            ApplyPlanningScene,
            GetMotionPlan,
            GetPlanningScene,
            GetPositionFK,
        )
        from rclpy.action import ActionClient
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import JointState
        from shape_msgs.msg import SolidPrimitive
        from std_srvs.srv import Trigger
        from trajectory_msgs.msg import JointTrajectoryPoint

        def action_client(node: Any, action_type: Any, name: str) -> Any:
            return ActionClient(node, action_type, name)

        return RosMoveItFactories(
            clock_type=lambda: Clock,
            joint_state_type=lambda: JointState,
            list_controllers_type=lambda: ListControllers,
            list_controllers_request_factory=lambda: ListControllers.Request(),
            get_motion_plan_type=lambda: GetMotionPlan,
            get_position_fk_type=lambda: GetPositionFK,
            move_group_type=lambda: MoveGroup,
            follow_joint_trajectory_type=lambda: FollowJointTrajectory,
            move_group_goal_factory=MoveGroup.Goal,
            motion_plan_request_factory=MotionPlanRequest,
            constraints_factory=Constraints,
            joint_constraint_factory=JointConstraint,
            position_constraint_factory=PositionConstraint,
            orientation_constraint_factory=OrientationConstraint,
            solid_primitive_factory=SolidPrimitive,
            pose_factory=Pose,
            action_client_factory=action_client,
            goal_status_type=lambda: GoalStatus,
            apply_planning_scene_type=lambda: ApplyPlanningScene,
            apply_planning_scene_request_factory=ApplyPlanningScene.Request,
            collision_object_factory=CollisionObject,
            attached_collision_object_factory=lambda: __import__(
                "moveit_msgs.msg", fromlist=["AttachedCollisionObject"]
            ).AttachedCollisionObject,
            allowed_collision_entry_factory=AllowedCollisionEntry,
            get_planning_scene_type=lambda: GetPlanningScene,
            get_planning_scene_request_factory=GetPlanningScene.Request,
            trigger_type=lambda: Trigger,
            trigger_request_factory=Trigger.Request,
            follow_joint_trajectory_goal_factory=FollowJointTrajectory.Goal,
            joint_trajectory_point_factory=JointTrajectoryPoint,
            joint_tolerance_factory=JointTolerance,
            duration_factory=Duration,
        )

    def _resolve_factories(self) -> RosMoveItFactories:
        if self._factories is not None:
            return self._factories
        if self._resolved_factories is None:
            self._resolved_factories = self._default_factories()
        return self._resolved_factories

    def _resolve_qos(self, profile: Any, *, reliable: bool) -> Any:
        """Injected profile wins; otherwise the live P1 probe profile."""
        if profile is not None:
            return profile
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

        return QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
            reliability=(
                ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT
            ),
        )

    def register_subscriptions(self) -> None:
        """Create the four state subscriptions once (lazy ROS imports)."""
        if self._subscriptions:
            return
        factories = self._resolve_factories()
        clock_type = factories.clock_type()
        joint_state_type = factories.joint_state_type()
        qos_state = self._resolve_qos(self._qos_state, reliable=True)
        qos_sensor = self._resolve_qos(self._qos_sensor, reliable=False)
        self._subscriptions = {
            self.topics.clock: self.node.create_subscription(
                clock_type, self.topics.clock, self._on_clock, qos_sensor
            ),
            self.topics.isaac_joint_states: self.node.create_subscription(
                joint_state_type,
                self.topics.isaac_joint_states,
                self._on_isaac_state,
                qos_sensor,
            ),
            self.topics.joint_states: self.node.create_subscription(
                joint_state_type, self.topics.joint_states, self._on_joint_state, qos_state
            ),
            self.topics.isaac_joint_command: self.node.create_subscription(
                joint_state_type,
                self.topics.isaac_joint_command,
                self._on_command,
                qos_state,
            ),
        }

    def register_clients(self) -> None:
        """Create the service and action clients once (lazy ROS imports).

        The scene-state Trigger client and the gripper follow action client
        are registered too. They are not components of the historical motion
        snapshot (:meth:`readiness_snapshot`), but they ARE components of
        :meth:`capability_readiness`, which is the gate a per-robot session
        runtime dispatches on.
        """
        if self._move_group_client is not None:
            return
        factories = self._resolve_factories()
        node = self.node
        self._list_controllers_client = node.create_client(
            factories.list_controllers_type(), self.services.list_controllers
        )
        self._plan_client = node.create_client(
            factories.get_motion_plan_type(), self.services.plan
        )
        self._fk_client = node.create_client(
            factories.get_position_fk_type(), self.services.compute_fk
        )
        if (
            factories.get_planning_scene_type is not None
            and factories.get_planning_scene_request_factory is not None
        ):
            self._get_planning_scene_client = node.create_client(
                factories.get_planning_scene_type(), self.services.get_planning_scene
            )
        if factories.trigger_type is not None:
            self._scene_state_client = node.create_client(
                factories.trigger_type(), self.services.scene_state
            )
        self._arm_client = factories.action_client_factory(
            node, factories.follow_joint_trajectory_type(), self.actions.arm_follow
        )
        self._move_group_client = factories.action_client_factory(
            node, factories.move_group_type(), self.actions.move_group
        )
        self._gripper_client = factories.action_client_factory(
            node,
            factories.follow_joint_trajectory_type(),
            self.actions.gripper_follow,
        )

    # ------------------------------------------------------------- callbacks

    def _on_clock(self, message: Any) -> None:
        with self._lock:
            self._clock_values.append(_clock_nanos(message))

    def _normalized_state(self, message: Any) -> dict[str, float] | None:
        """Normalize a state message onto the canonical joint names, or None.

        The message must carry EXACTLY this runtime's device joint names (the
        profile's ``<robot_id>_<joint>`` names) with finite positions and a
        velocity array of length 0 or 8. Any other name set - in particular a
        state message of a different robot delivered to this subscription -
        is rejected outright, so cross-injected samples never become
        readiness or feedback evidence here. Accepted states are keyed by the
        canonical joint names, which is the API contract of every state
        getter; the wire-level ``<robot_id>_`` prefix never appears in them.
        """
        try:
            raw = {
                str(name): float(value)
                for name, value in zip(message.name, message.position)
            }
            velocity = list(message.velocity)
        except (AttributeError, TypeError, ValueError):
            return None
        if len(raw) != len(self._joint_names) or set(raw) != set(self._joint_names):
            return None
        if len(velocity) not in (0, len(self._joint_names)):
            return None
        state: dict[str, float] = {}
        for canonical, device in zip(JOINT_NAMES, self._joint_names):
            value = raw[device]
            if not math.isfinite(value):
                return None
            state[canonical] = value
        return state

    def _on_isaac_state(self, message: Any) -> None:
        state = self._normalized_state(message)
        if state is None:
            return
        with self._lock:
            self._isaac_states.append(state)
            self._isaac_receipts.append(self._clock_fn())
            self._isaac_state_count += 1

    def _on_joint_state(self, message: Any) -> None:
        state = self._normalized_state(message)
        if state is None:
            return
        with self._lock:
            self._joint_states.append(state)

    def _on_command(self, message: Any) -> None:
        with self._lock:
            self._command_count += 1

    # ------------------------------------------------------------ monitoring

    def latest_isaac_state(self) -> dict[str, float] | None:
        """Latest valid /isaac_joint_states sample or None."""
        with self._lock:
            if not self._isaac_states:
                return None
            return dict(self._isaac_states[-1])

    def latest_joint_state(self) -> dict[str, float] | None:
        """Latest valid /joint_states (broadcaster) sample or None."""
        with self._lock:
            if not self._joint_states:
                return None
            return dict(self._joint_states[-1])

    def clock_values(self) -> tuple[int, ...]:
        """Recent /clock samples as integer nanoseconds."""
        with self._lock:
            return tuple(self._clock_values)

    def controller_states(self) -> dict[str, str]:
        """Cached controller states, refreshed via a bounded service poll.

        An in-flight poll never blocks readiness for longer than
        ``SERVICE_CALL_TIMEOUT_S``; a stale pending poll is abandoned and
        re-issued so a dead service call cannot wedge the readiness loop.
        """
        with self._lock:
            cached = dict(self._controller_states)
        client = self._list_controllers_client
        if client is None:
            return cached
        try:
            if not client.service_is_ready():
                return cached
            future = self._controllers_future
            pending_is_fresh = (
                future is not None
                and not future.done()
                and (self._clock_fn() - self._controllers_requested_at)
                <= SERVICE_CALL_TIMEOUT_S
            )
            if pending_is_fresh:
                return cached
            if future is not None:
                with self._lock:
                    self._controllers_future = None
                states = _parse_controller_states(future) if future.done() else None
                if states is not None:
                    with self._lock:
                        self._controller_states = states
                    return dict(states)
            new_future = client.call_async(
                self._resolve_factories().list_controllers_request_factory()
            )
            with self._lock:
                self._controllers_future = new_future
                self._controllers_requested_at = self._clock_fn()
            states = (
                _parse_controller_states(new_future) if new_future.done() else None
            )
            if states is not None:
                with self._lock:
                    self._controller_states = states
                return dict(states)
        except Exception:
            return cached
        return cached

    # ------------------------------------------------------------- readiness

    def _action_ready(self, client: Any) -> bool:
        if client is None:
            return False
        try:
            return bool(client.wait_for_server(timeout_sec=0.0))
        except Exception:
            return False

    def _service_ready(self, client: Any) -> bool:
        if client is None:
            return False
        try:
            return bool(client.service_is_ready())
        except Exception:
            return False

    def readiness_snapshot(self) -> ReadinessSnapshot:
        """Aggregate the R1 pure predicates plus transport readiness flags."""
        isaac = self.latest_isaac_state()
        joint = self.latest_joint_state()
        return ReadinessSnapshot(
            clock_increasing=clock_is_increasing(self.clock_values()),
            isaac_state_complete=isaac is not None and state_is_complete(isaac),
            joint_state_complete=joint is not None and state_is_complete(joint),
            broadcaster_match=(
                isaac is not None
                and joint is not None
                and states_match(isaac, joint, tolerance=EXACT_STATE_MATCH_TOLERANCE)
            ),
            controllers_active=controllers_are_active(self.controller_states()),
            arm_action_ready=self._action_ready(self._arm_client),
            move_group_action_ready=self._action_ready(self._move_group_client),
            compute_fk_ready=self._service_ready(self._fk_client),
        )

    def wait_until_ready(self, timeout_s: float) -> bool:
        """Bounded readiness wait; never sends motion.

        True only when every readiness component holds before the deadline
        (injected clock). The spin/sleep hooks are the injected event pump.
        """
        if not math.isfinite(float(timeout_s)) or float(timeout_s) < 0.0:
            raise ValueError(
                f"timeout_s must be finite and non-negative, got {timeout_s!r}"
            )
        deadline = self._clock_fn() + float(timeout_s)
        while True:
            self._spin_fn()
            if self.readiness_snapshot().passed:
                return True
            if self._clock_fn() >= deadline:
                return False
            self._sleep_fn(self._poll_interval_s)

    # ---------------------------------------------------- capability readiness

    def scene_query_probe(
        self, timeout_s: float = SCENE_PROBE_TIMEOUT_S
    ) -> dict[str, object] | None:
        """Bounded live robot-scoped scene probe; None when unverifiable.

        Read-only: it performs the same bounded global Trigger call as
        :meth:`query_robot_scene_state` and returns the validated view of
        THIS robot's block, or ``None`` when the client is unavailable, the
        query fails, the schema is wrong, or the block is missing, foreign or
        bound to another robot's joint names. Readiness uses it so a robot
        whose scene response cannot be verified never becomes ready.
        """
        try:
            return self.query_robot_scene_state(timeout_s=timeout_s)
        except (RuntimeError, ValueError):
            return None

    def capability_readiness(
        self, *, probe_scene: bool = False
    ) -> RobotCapabilityReadiness:
        """This robot's full capability readiness, endpoints included.

        Without ``probe_scene`` it is read-only and side-effect free: it
        reads the injected pump's latest state, the cached controller states
        and the action/service availability of THIS runtime's clients (the
        scene component is then the Trigger client availability). No motion
        is ever sent.

        With ``probe_scene`` the scene component is verified against the LIVE
        robot-scoped response: the bounded Trigger answer must carry this
        robot's own block (and, when the block carries the host binding
        evidence, this robot's device joint names), so a snapshot bound to
        another robot can never satisfy it. The verdict carries the profiled
        robot id, its namespace and the exact endpoints so a caller can audit
        that no canonical fallback leaked in.
        """
        snapshot = self.readiness_snapshot()
        scene_query_ready = self._service_ready(self._scene_state_client)
        if probe_scene and scene_query_ready:
            scene_query_ready = self.scene_query_probe() is not None
        return RobotCapabilityReadiness(
            robot_id=self._robot_id,
            namespace=self._namespace,
            clock_increasing=snapshot.clock_increasing,
            isaac_state_complete=snapshot.isaac_state_complete,
            joint_state_complete=snapshot.joint_state_complete,
            broadcaster_match=snapshot.broadcaster_match,
            controllers_active=snapshot.controllers_active,
            arm_action_ready=snapshot.arm_action_ready,
            move_group_action_ready=snapshot.move_group_action_ready,
            gripper_action_ready=self._action_ready(self._gripper_client),
            compute_fk_ready=snapshot.compute_fk_ready,
            scene_query_ready=scene_query_ready,
            endpoints=tuple(sorted(self.endpoint_map().items())),
        )

    def wait_until_capability_ready(
        self, timeout_s: float, *, probe_scene: bool = True
    ) -> bool:
        """Bounded per-robot capability readiness wait; never sends motion.

        The scene component is probed live by default, so a robot whose
        per-robot scene response is missing, foreign or mis-bound never
        reaches readiness.
        """
        if not math.isfinite(float(timeout_s)) or float(timeout_s) < 0.0:
            raise ValueError(
                f"timeout_s must be finite and non-negative, got {timeout_s!r}"
            )
        deadline = self._clock_fn() + float(timeout_s)
        while True:
            self._spin_fn()
            if self.capability_readiness(probe_scene=probe_scene).passed:
                return True
            if self._clock_fn() >= deadline:
                return False
            self._sleep_fn(self._poll_interval_s)

    # ------------------------------------------------------------------ FK

    def compute_link6_fk(
        self, state: Mapping[str, float], timeout_s: float = SERVICE_CALL_TIMEOUT_S
    ) -> FkVerification | None:
        """Live raw-flange (link6) FK of ``state`` via this robot's compute_fk.

        The explicitly minimal public FK adapter hook: the GetPositionFK
        request is built through the injected factories and the registered
        client from the EXACT canonical 8-DOF names of ``state`` (the wire
        names of this runtime's profile are used transparently), waited on
        bounded through the injected clock/spin/sleep hooks, and answered with
        ``((x, y, z), (x, y, z, w))`` or ``None``. No pose is ever hardcoded
        or interpolated: a non-SUCCESS code, an incomplete state, a missing
        answer, a nonfinite or near-zero result all return ``None`` so the
        caller fails explicitly instead of faking a pose.
        """
        return self._compute_fk(state, self._link6_link, timeout_s)

    def compute_tcp_fk(
        self, state: Mapping[str, float], timeout_s: float = SERVICE_CALL_TIMEOUT_S
    ) -> FkVerification | None:
        """Live calibrated-TCP FK of ``state`` via this robot's compute_fk.

        Identical contract to :meth:`compute_link6_fk` but the requested link
        is the project-calibrated TCP frame of that robot's expanded MoveIt
        model, so the answered pose is exactly the link6 FK composed with the
        generated calibration joint. The same failure rules apply: nothing is
        hardcoded, interpolated, or faked.
        """
        return self._compute_fk(state, self._tcp_link, timeout_s)

    def _compute_fk(
        self, state: Mapping[str, float], link: str, timeout_s: float
    ) -> FkVerification | None:
        """Shared bounded /compute_fk call for one exact link name."""
        if (
            not isinstance(timeout_s, (int, float))
            or not math.isfinite(float(timeout_s))
            or float(timeout_s) <= 0.0
        ):
            raise ValueError(
                f"timeout_s must be finite and positive, got {timeout_s!r}"
            )
        client = self._fk_client
        if client is None or not self._service_ready(client):
            return None
        if not state_is_complete(state):
            return None
        request = self._resolve_factories().get_position_fk_type().Request()
        request.header.frame_id = self._base_frame
        request.fk_link_names = [link]
        request.robot_state.joint_state.name = list(self._joint_names)
        request.robot_state.joint_state.position = [
            float(state[name]) for name in JOINT_NAMES
        ]
        future = client.call_async(request)
        if not self._wait_future(future, self._clock_fn() + float(timeout_s)):
            return None
        try:
            response = future.result()
            if response is None or int(response.error_code.val) != SUCCESS_CODE:
                return None
            if len(response.pose_stamped) != 1:
                return None
            pose = response.pose_stamped[0].pose
            position = (
                float(pose.position.x),
                float(pose.position.y),
                float(pose.position.z),
            )
            quaternion = (
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            )
        except (AttributeError, IndexError, TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in position + quaternion):
            return None
        if math.sqrt(sum(value * value for value in quaternion)) < QUAT_NORM_MIN:
            return None
        return position, quaternion

    # -------------------------------------------------------- planning scene

    def apply_planning_scene(
        self, update: PlanningSceneUpdate, timeout_s: float
    ) -> PlanningSceneResult:
        """Apply a formal planning-scene update (R5.A), never faking success.

        Uses the injected ``planning_scene_applier`` when one was supplied and
        otherwise the lazy production ``/apply_planning_scene`` adapter. The
        update claim is never modified: the result reports exactly which
        object ids were applied and the scene digest observed, so the caller
        can verify them against ``update.expected_scene_digest``. Applier
        exceptions are reported as failed results, never raised past the
        transport boundary.
        """
        if not isinstance(update, PlanningSceneUpdate):
            raise ValueError(
                f"update must be a PlanningSceneUpdate, got {type(update).__name__}"
            )
        if (
            not isinstance(timeout_s, (int, float))
            or not math.isfinite(float(timeout_s))
            or float(timeout_s) <= 0.0
        ):
            raise ValueError(
                f"timeout_s must be finite and positive, got {timeout_s!r}"
            )
        applier = (
            self._planning_scene_applier
            if self._planning_scene_applier is not None
            else self._default_planning_scene_applier
        )
        try:
            result = applier(update, float(timeout_s))
        except Exception as exc:
            return PlanningSceneResult(
                passed=False,
                details={
                    "scene_id": update.scene_id,
                    "error": (
                        "planning-scene applier raised: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                },
            )
        if not isinstance(result, PlanningSceneResult):
            return PlanningSceneResult(
                passed=False,
                details={
                    "scene_id": update.scene_id,
                    "error": (
                        "planning-scene applier returned "
                        f"{type(result).__name__}, not PlanningSceneResult"
                    ),
                },
            )
        return result

    def _ensure_planning_scene_client(self) -> Any:
        """Lazily create the /apply_planning_scene client (never at init)."""
        if self._planning_scene_client is not None:
            return self._planning_scene_client
        factories = self._resolve_factories()
        if factories.apply_planning_scene_type is None:
            return None
        self._planning_scene_client = self.node.create_client(
            factories.apply_planning_scene_type(),
            self.services.apply_planning_scene,
        )
        return self._planning_scene_client

    def _ensure_get_planning_scene_client(self) -> Any:
        """Return the lazy GetPlanningScene client, if the transport exists."""
        if self._get_planning_scene_client is not None:
            return self._get_planning_scene_client
        factories = self._resolve_factories()
        if (
            factories.get_planning_scene_type is None
            or factories.get_planning_scene_request_factory is None
        ):
            return None
        self._get_planning_scene_client = self.node.create_client(
            factories.get_planning_scene_type(), self.services.get_planning_scene
        )
        return self._get_planning_scene_client

    def _read_current_allowed_collision_matrix(
        self, timeout_s: float, factories: RosMoveItFactories
    ) -> tuple[Any | None, str | None]:
        """Read the complete MoveIt ACM before constructing a scene diff."""
        client = self._ensure_get_planning_scene_client()
        if client is None or not self._service_ready(client):
            return None, (
                f"{self.services.get_planning_scene} client is not available; "
                "refusing to replace the existing allowed collision matrix"
            )
        request_factory = factories.get_planning_scene_request_factory
        if request_factory is None:
            return None, "no GetPlanningScene request factory is available"
        request = request_factory()
        components = getattr(request, "components", None)
        if components is None:
            return None, "GetPlanningScene request has no components field"
        components.components = PLANNING_SCENE_ALLOWED_COLLISION_MATRIX
        future = client.call_async(request)
        if not self._wait_future(future, self._clock_fn() + float(timeout_s)):
            return None, (
                f"{self.services.get_planning_scene} did not answer within "
                f"{timeout_s}s"
            )
        try:
            response = future.result()
            scene = response.scene
            matrix = scene.allowed_collision_matrix
            names = list(matrix.entry_names)
            rows = list(matrix.entry_values)
        except Exception as error:
            return None, f"GetPlanningScene response is invalid: {error}"
        if not names or len(rows) != len(names) or any(
            len(entry.enabled) != len(names) for entry in rows
        ):
            return None, (
                "GetPlanningScene returned a non-square or empty allowed "
                "collision matrix; refusing fail-open scene application"
            )
        return matrix, None

    def _read_world_collision_objects(
        self, timeout_s: float, factories: RosMoveItFactories
    ) -> tuple[Mapping[str, object] | None, str | None]:
        """Read the MoveIt world geometry after a scene application.

        Returns:
            Object IDs mapped to read-back collision objects, or a failure.
        """
        client = self._ensure_get_planning_scene_client()
        request_factory = factories.get_planning_scene_request_factory
        if client is None or not self._service_ready(client) or request_factory is None:
            return None, "GetPlanningScene world geometry is unavailable"
        request = request_factory()
        if getattr(request, "components", None) is None:
            return None, "GetPlanningScene components are unavailable"
        request.components.components = PLANNING_SCENE_WORLD_OBJECT_READBACK
        future = client.call_async(request)
        if not self._wait_future(future, self._clock_fn() + timeout_s):
            return None, "GetPlanningScene world geometry timed out"
        try:
            objects = future.result().scene.world.collision_objects
            by_id = {item.id: item for item in objects}
            if len(by_id) != len(objects):
                raise ValueError("duplicate collision object ids")
        except (AttributeError, TypeError, ValueError) as error:
            return None, f"GetPlanningScene world geometry is malformed: {error}"
        return by_id, None

    def _read_attached_collision_object(
        self,
        timeout_s: float,
        factories: RosMoveItFactories,
        *,
        object_id: str,
        link_name: str,
    ) -> tuple[bool, str | None]:
        """Read back one exact robot-attached collision object.

        Returns:
            ``(True, None)`` only for exactly one matching attached object.
        """
        client = self._ensure_get_planning_scene_client()
        request_factory = factories.get_planning_scene_request_factory
        if client is None or not self._service_ready(client) or request_factory is None:
            return False, "GetPlanningScene attached objects are unavailable"
        request = request_factory()
        if getattr(request, "components", None) is None:
            return False, "GetPlanningScene components are unavailable"
        request.components.components = PLANNING_SCENE_ROBOT_STATE_ATTACHED_OBJECTS
        future = client.call_async(request)
        if not self._wait_future(future, self._clock_fn() + timeout_s):
            return False, "GetPlanningScene attached objects timed out"
        try:
            attached = future.result().scene.robot_state.attached_collision_objects
            matches = [
                item for item in attached
                if _attached_object_matches(
                    item, object_id=object_id, link_name=link_name
                )
            ]
        except (AttributeError, TypeError, ValueError) as error:
            return False, f"GetPlanningScene attached objects malformed: {error}"
        if len(matches) != 1:
            return False, (
                f"GetPlanningScene returned {len(matches)} matching attached objects "
                f"for {object_id!r}/{link_name!r}"
            )
        return True, None

    def apply_attached_planning_scene(
        self, payload: Mapping[str, object], timeout_s: float
    ) -> Mapping[str, object]:
        """Apply and read back one exact robot-attached collision object."""
        factories = self._resolve_factories()
        required = (
            factories.apply_planning_scene_type,
            factories.apply_planning_scene_request_factory,
            factories.collision_object_factory,
            factories.attached_collision_object_factory,
            factories.solid_primitive_factory,
            factories.pose_factory,
        )
        if any(item is None for item in required):
            return {"passed": False, "readback_confirmed": False,
                    "error": "attached-scene factories are unavailable"}
        client = self._ensure_planning_scene_client()
        if client is None or not self._service_ready(client):
            return {"passed": False, "readback_confirmed": False,
                    "error": "apply planning-scene client is unavailable"}
        try:
            request = factories.apply_planning_scene_request_factory()
            request.scene.is_diff = True
            collision = factories.collision_object_factory()
            collision.id = str(payload["object_id"])
            collision.header.frame_id = str(payload["frame_id"])
            collision.operation = getattr(type(collision), "ADD")
            primitive = factories.solid_primitive_factory()
            primitive.type = getattr(type(primitive), "BOX")
            minimum = tuple(float(value) for value in payload["minimum_m"])
            maximum = tuple(float(value) for value in payload["maximum_m"])
            primitive.dimensions = [high - low for low, high in zip(minimum, maximum)]
            pose = factories.pose_factory()
            center = [(low + high) / 2.0 for low, high in zip(minimum, maximum)]
            pose.position.x, pose.position.y, pose.position.z = center
            pose.orientation.w = 1.0
            collision.primitives.append(primitive)
            collision.primitive_poses.append(pose)
            attached = request.scene.robot_state.attached_collision_objects
            attached_object = factories.attached_collision_object_factory()
            attached_object.link_name = str(payload["link_name"])
            attached_object.object = collision
            attached_object.touch_links = list(payload["touch_links"])
            attached.append(attached_object)
            future = client.call_async(request)
            if not self._wait_future(future, self._clock_fn() + float(timeout_s)):
                return {"passed": False, "readback_confirmed": False,
                        "error": "apply planning-scene timed out"}
            if future.result().success is not True:
                return {"passed": False, "readback_confirmed": False,
                        "error": "apply planning-scene reported failure"}
        except (AttributeError, TypeError, ValueError) as error:
            return {"passed": False, "readback_confirmed": False,
                    "error": f"attached-scene request malformed: {error}"}
        ok, error = self._read_attached_collision_object(
            float(timeout_s), factories,
            object_id=str(payload["object_id"]), link_name=str(payload["link_name"]),
        )
        return {
            "passed": ok,
            "readback_confirmed": ok,
            "attached_object_id": str(payload["object_id"]),
            "attached_link": str(payload["link_name"]),
            "error": error,
        }

    def _default_planning_scene_applier(
        self, update: PlanningSceneUpdate, timeout_s: float
    ) -> PlanningSceneResult:
        """Production adapter over moveit_msgs/srv/ApplyPlanningScene.

        Each box is one ADD CollisionObject with a single BOX SolidPrimitive
        at the AABB center with identity orientation, sized to the AABB
        dimensions, in the box frame; the scene diff flag is set so the
        update is additive. Standalone collision objects only: no attached
        objects and no contact allowances are ever constructed here.
        """
        factories = self._resolve_factories()
        if (
            factories.apply_planning_scene_type is None
            or factories.apply_planning_scene_request_factory is None
            or factories.collision_object_factory is None
        ):
            return PlanningSceneResult(
                passed=False,
                details={
                    "scene_id": update.scene_id,
                    "error": (
                        "no injected planning-scene applier and the factories "
                        "carry no ApplyPlanningScene transport"
                    ),
                },
            )
        client = self._ensure_planning_scene_client()
        if client is None or not self._service_ready(client):
            return PlanningSceneResult(
                passed=False,
                details={
                    "scene_id": update.scene_id,
                    "error": (
                        f"{self.services.apply_planning_scene} client is not "
                        "available"
                    ),
                },
            )
        request = factories.apply_planning_scene_request_factory()
        request.scene.is_diff = True
        if update.allowed_collision_pairs:
            entry_factory = factories.allowed_collision_entry_factory
            if entry_factory is None:
                return PlanningSceneResult(
                    passed=False,
                    details={
                        "scene_id": update.scene_id,
                        "error": (
                            "allowed collision pairs were requested but the "
                            "AllowedCollisionEntry factory is unavailable"
                        ),
                    },
                )
            matrix, matrix_error = self._read_current_allowed_collision_matrix(
                timeout_s, factories
            )
            if matrix is None:
                return PlanningSceneResult(
                    passed=False,
                    details={"scene_id": update.scene_id, "error": matrix_error},
                )
            try:
                request.scene.allowed_collision_matrix = _merge_allowed_collision_pairs(
                    matrix, update.allowed_collision_pairs, entry_factory
                )
            except ValueError as error:
                return PlanningSceneResult(
                    passed=False,
                    details={"scene_id": update.scene_id, "error": str(error)},
                )
        for box in update.boxes:
            # Explicitly remove a previous object with the same id before ADD.
            # Some MoveIt/warehouse combinations update the primitive dimensions
            # but retain the previous primitive pose on an ADD with a reused id.
            # REMOVE is harmless when the object is absent and makes the pose
            # replacement deterministic without weakening readback validation.
            removal = factories.collision_object_factory()
            removal.header.frame_id = box.frame_id
            removal.id = box.object_id
            removal.operation = getattr(type(removal), "REMOVE")
            request.scene.world.collision_objects.append(removal)

            collision_object = factories.collision_object_factory()
            collision_object.header.frame_id = box.frame_id
            collision_object.id = box.object_id
            collision_object.operation = getattr(type(collision_object), "ADD")
            primitive = factories.solid_primitive_factory()
            primitive.type = getattr(type(primitive), "BOX")
            primitive.dimensions = [float(value) for value in box.dimensions_m]
            collision_object.primitives.append(primitive)
            center = box.center_m
            pose = factories.pose_factory()
            pose.position.x = float(center[0])
            pose.position.y = float(center[1])
            pose.position.z = float(center[2])
            pose.orientation.w = 1.0
            collision_object.primitive_poses.append(pose)
            request.scene.world.collision_objects.append(collision_object)
        applied = tuple(box.object_id for box in update.boxes)
        future = client.call_async(request)
        if not self._wait_future(future, self._clock_fn() + float(timeout_s)):
            return PlanningSceneResult(
                passed=False,
                details={
                    "scene_id": update.scene_id,
                    "error": (
                        f"{self.services.apply_planning_scene} did not answer "
                        f"within {timeout_s}s"
                    ),
                },
            )
        try:
            response = future.result()
            service_success = bool(response.success)
        except Exception as exc:
            return PlanningSceneResult(
                passed=False,
                details={
                    "scene_id": update.scene_id,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
        computed = scene_digest(update)
        details: dict[str, object] = {
            "scene_id": update.scene_id,
            "service": self.services.apply_planning_scene,
            "service_success": service_success,
            "expected_scene_digest": update.expected_scene_digest,
            "scene_digest": computed,
        }
        if not service_success:
            details["error"] = (
                f"{self.services.apply_planning_scene} reported failure"
            )
            return PlanningSceneResult(
                passed=False, applied_object_ids=(), details=details
            )
        if computed != update.expected_scene_digest:
            details["error"] = (
                "applied scene digest does not match the expected scene digest"
            )
            return PlanningSceneResult(
                passed=False, applied_object_ids=applied, details=details
            )
        observed: Mapping[str, object] | None = None
        readback_error: str | None = None
        matched = False
        for attempt in range(3):
            observed, readback_error = self._read_world_collision_objects(
                timeout_s, factories
            )
            if observed is None:
                break
            matched = True
            for box in update.boxes:
                item = observed.get(box.object_id)
                if item is None:
                    matched = False
                    details["error"] = (
                        f"GetPlanningScene did not return box {box.object_id!r}; "
                        f"observed_ids={sorted(str(value) for value in observed)}"
                    )
                    break
                if not _collision_box_matches(item, box):
                    matched = False
                    details["error"] = (
                        f"GetPlanningScene returned mismatched box {box.object_id!r}; "
                        f"frame={getattr(getattr(item, 'header', None), 'frame_id', None)!r} "
                        f"primitives={len(getattr(item, 'primitives', ()))!r} "
                        f"dimensions={getattr(getattr(item, 'primitives', [None])[0], 'dimensions', None)!r} "
                        f"center={getattr(getattr(getattr(item, 'primitive_poses', [None])[0], 'position', None), 'x', None)!r},"
                        f"{getattr(getattr(getattr(item, 'primitive_poses', [None])[0], 'position', None), 'y', None)!r},"
                        f"{getattr(getattr(getattr(item, 'primitive_poses', [None])[0], 'position', None), 'z', None)!r}"
                    )
                    break
            if matched:
                break
            if attempt < 2:
                self._sleep_fn(min(self._poll_interval_s, float(timeout_s)))
        if observed is None:
            details["error"] = readback_error
            return PlanningSceneResult(
                passed=False, applied_object_ids=applied, details=details
            )
        if not matched:
            return PlanningSceneResult(
                passed=False, applied_object_ids=applied, details=details
            )
        if update.allowed_collision_pairs:
            matrix, matrix_error = self._read_current_allowed_collision_matrix(
                timeout_s, factories
            )
            if matrix is None:
                details["error"] = matrix_error
                return PlanningSceneResult(
                    passed=False, applied_object_ids=applied, details=details
                )
            names = list(matrix.entry_names)
            rows = list(matrix.entry_values)
            for first, second in update.allowed_collision_pairs:
                if (
                    first not in names
                    or second not in names
                    or rows[names.index(first)].enabled[names.index(second)] is not True
                    or rows[names.index(second)].enabled[names.index(first)] is not True
                ):
                    details["error"] = (
                        f"GetPlanningScene did not confirm contact {first!r}/{second!r}"
                    )
                    return PlanningSceneResult(
                        passed=False, applied_object_ids=applied, details=details
                    )
            details["allowed_collision_pairs_confirmed"] = True
        details["readback_confirmed"] = True
        return PlanningSceneResult(
            passed=True,
            applied_object_ids=applied,
            scene_digest=computed,
            details=details,
        )

    # ------------------------------------------------------------ scene state

    def query_scene_state(self, timeout_s: float) -> dict[str, object]:
        """Fetch the live scene-state document from the host bridge (bounded).

        Calls the ``std_srvs/srv/Trigger`` service on
        ``/p05_host_bridge/scene_state`` and parses ``response.message`` as
        canonical JSON. Only a document whose ``schema`` is exactly
        ``rocobrick-scene-state/v1`` is returned; an unavailable client, a
        timeout, a malformed message or a wrong schema raise ``RuntimeError``
        so the caller reports a failed result instead of guessing.
        """
        if (
            not isinstance(timeout_s, (int, float))
            or not math.isfinite(float(timeout_s))
            or float(timeout_s) <= 0.0
        ):
            raise ValueError(
                f"timeout_s must be finite and positive, got {timeout_s!r}"
            )
        client = self._scene_state_client
        if client is None or not self._service_ready(client):
            raise RuntimeError(
                f"{self.services.scene_state} client is not available"
            )
        factories = self._resolve_factories()
        if factories.trigger_request_factory is None:
            raise RuntimeError(
                "the injected factories carry no Trigger transport for the "
                "scene-state query"
            )
        future = client.call_async(factories.trigger_request_factory())
        if not self._wait_future(future, self._clock_fn() + float(timeout_s)):
            raise RuntimeError(
                f"{self.services.scene_state} did not answer within "
                f"{timeout_s}s"
            )
        try:
            response = future.result()
            message = str(response.message)
        except Exception as exc:
            raise RuntimeError(
                f"{self.services.scene_state} answer is malformed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        try:
            document = json.loads(message)
        except ValueError as exc:
            raise RuntimeError(
                f"{self.services.scene_state} message is not JSON: {exc}"
            ) from exc
        if not isinstance(document, dict) or document.get("schema") != (
            SCENE_STATE_SCHEMA
        ):
            raise RuntimeError(
                f"{self.services.scene_state} message schema is not "
                f"{SCENE_STATE_SCHEMA!r}"
            )
        return document

    def query_robot_scene_state(self, timeout_s: float) -> dict[str, object]:
        """This robot's own slice of the live scene-state document (bounded).

        Requires a profiled robot runtime: the global A4 service is queried
        once (same bounded call as :meth:`query_scene_state`) and the answer
        is sliced by :func:`robot_scene_view` with this robot's device joint
        names, so only this robot's gripper reading can ever be reported, a
        missing/foreign block fails explicitly, and a snapshot bound to
        another robot's joint names is rejected instead of falling back to
        another robot's observation.
        """
        if not self._robot_id:
            raise RuntimeError(
                "robot-scoped scene observation requires a profiled robot "
                "runtime; this runtime has no robot id"
            )
        document = self.query_scene_state(timeout_s)
        return robot_scene_view(
            document, self._robot_id, joint_names=self._joint_names
        )

    # ------------------------------------------------------------------ sends

    def send_joint(self, target: JointTarget, timeout_s: float) -> ExecutionResult:
        """Admit, build and send one arm joint MoveGroup goal (bounded)."""
        return self._send_goal("joint", target, timeout_s)

    def send_link6_pose(self, target: Link6PoseTarget, timeout_s: float) -> ExecutionResult:
        """Admit, build and send one link6 pose MoveGroup goal (bounded)."""
        return self._send_goal("pose", target, timeout_s)

    def send_tcp_pose(self, target: TcpPoseTarget, timeout_s: float) -> ExecutionResult:
        """Admit, build and send one calibrated TCP MoveGroup goal (bounded).

        The goal targets the ``arm_tcp`` group and the ``rocobrick_tcp``
        constraint link in ``base_link`` with ``plan_only=false``. No tool
        offset is applied at request time: the calibrated link6->rocobrick_tcp
        transform lives only in the model, so it can never be double-applied.
        """
        return self._send_goal("tcp_pose", target, timeout_s)

    def send_gripper(self, target: GripperTarget, timeout_s: float) -> ExecutionResult:
        """Admit, build and send one two-point gripper trajectory (bounded).

        Point 0 holds the latest observed finger positions at t=0 so the
        command starts from the measured state; point 1 commands the logical
        opening mapped onto the symmetric finger positions through
        ``rocobrick.gripper.logical_gripper_to_fingers``. The verdict is
        measured, never assumed: fingers stalled on a brick legitimately
        terminate the action ABORTED, so after ANY terminal state the latest
        finger positions are re-sampled and the measured opening decides
        success against ``expected_opening_m`` (inclusive). A missing or
        mimic-violating measurement fails explicitly instead of faking
        success.
        """
        decision = strict_action_admission(target)
        if not decision.accepted:
            return ExecutionResult(
                status=ExecutionStatus.REJECTED,
                details={
                    "kind": "gripper",
                    "reason": decision.reason,
                    "admission": dict(decision.details),
                },
                command_messages=0,
            )
        if (
            not isinstance(timeout_s, (int, float))
            or not math.isfinite(float(timeout_s))
            or float(timeout_s) <= 0.0
        ):
            return ExecutionResult(
                status=ExecutionStatus.REJECTED,
                details={
                    "kind": "gripper",
                    "reason": (
                        f"timeout_s must be finite and positive, got {timeout_s!r}"
                    ),
                },
                command_messages=0,
            )
        if self._gripper_client is None:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                details={
                    "kind": "gripper",
                    "error": (
                        "gripper action client is not registered; call "
                        "register_clients() first"
                    ),
                },
                command_messages=0,
            )
        factories = self._resolve_factories()
        if (
            factories.follow_joint_trajectory_goal_factory is None
            or factories.joint_trajectory_point_factory is None
            or factories.joint_tolerance_factory is None
            or factories.duration_factory is None
        ):
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                details={
                    "kind": "gripper",
                    "error": (
                        "the injected factories carry no FollowJointTrajectory "
                        "goal transport for the gripper"
                    ),
                },
                command_messages=0,
            )
        snapshot = self.readiness_snapshot()
        if not snapshot.passed:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                details={
                    "kind": "gripper",
                    "error": (
                        "runtime readiness failed; no motion is allowed until "
                        "wait_until_ready passes"
                    ),
                    "readiness": asdict(snapshot),
                },
                command_messages=0,
            )
        start_state = self.latest_isaac_state()
        if (
            start_state is None
            or "gripper_joint1" not in start_state
            or "gripper_joint2" not in start_state
        ):
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                details={
                    "kind": "gripper",
                    "error": (
                        "no complete isaac finger state was observed; the "
                        "trajectory start point cannot be built"
                    ),
                },
                command_messages=0,
            )
        goal = self._build_gripper_goal(target, start_state, factories)
        return self._execute_gripper_goal(
            goal,
            target=target,
            timeout_s=float(timeout_s),
            factories=factories,
        )

    def _duration_message(self, seconds: float, factories: RosMoveItFactories) -> Any:
        """builtin_interfaces/Duration message for a non-negative seconds."""
        duration = factories.duration_factory()
        whole = int(seconds)
        duration.sec = whole
        duration.nanosec = int(round((float(seconds) - whole) * 1.0e9))
        return duration

    def _build_gripper_goal(
        self,
        target: GripperTarget,
        start_state: Mapping[str, float],
        factories: RosMoveItFactories,
    ) -> Any:
        """Two-point FollowJointTrajectory goal on the gripper joints."""
        goal = factories.follow_joint_trajectory_goal_factory()
        trajectory = goal.trajectory
        trajectory.joint_names = list(self._gripper_joint_names)
        start = factories.joint_trajectory_point_factory()
        start.positions = [
            float(start_state["gripper_joint1"]),
            float(start_state["gripper_joint2"]),
        ]
        start.time_from_start = self._duration_message(0.0, factories)
        commanded = logical_gripper_to_fingers(float(target.opening_m))
        end = factories.joint_trajectory_point_factory()
        end.positions = [float(commanded[0]), float(commanded[1])]
        end.time_from_start = self._duration_message(
            float(target.duration_s), factories
        )
        trajectory.points = [start, end]
        goal.goal_tolerance = []
        for name in self._gripper_joint_names:
            tolerance = factories.joint_tolerance_factory()
            tolerance.name = name
            tolerance.position = float(target.goal_tolerance)
            goal.goal_tolerance.append(tolerance)
        return goal

    def _execute_gripper_goal(
        self,
        goal: Any,
        *,
        target: GripperTarget,
        timeout_s: float,
        factories: RosMoveItFactories,
    ) -> ExecutionResult:
        """Bounded gripper action with terminal and fresh measured feedback."""
        details: dict[str, object] = {"kind": "gripper"}
        now = self._clock_fn()
        deadline = now + timeout_s
        client = self._gripper_client
        try:
            future = client.send_goal_async(goal)
            self._active_goal_future = future
        except Exception as exc:
            details["error"] = f"send_goal_async failed: {type(exc).__name__}: {exc}"
            return ExecutionResult(
                status=ExecutionStatus.FAILED, details=details, command_messages=0
            )
        if not self._wait_future(future, min(now + ACCEPT_TIMEOUT_S, deadline)):
            details["phase"] = "goal_acceptance"
            details["error"] = (
                "gripper goal response not received before the acceptance deadline"
            )
            details["cancel"] = self.cancel_active()
            lifecycle = details["cancel"]
            self._active_goal_future = None
            return ExecutionResult(
                status=ExecutionStatus.TIMED_OUT, details=details, command_messages=1,
                goal_accepted=lifecycle.get("goal_accepted"),
                terminal=bool(lifecycle.get("terminal", False)),
                safe_to_release=bool(lifecycle.get("safe", False)),
                cancel_evidence=lifecycle,
            )
        handle = future.result()
        if handle is None or not getattr(handle, "accepted", False):
            details["phase"] = "goal_acceptance"
            details["accepted"] = False
            details["error"] = "gripper_controller did not accept the goal"
            return ExecutionResult(
                status=ExecutionStatus.REJECTED, details=details, command_messages=0,
                goal_accepted=False, terminal=True, safe_to_release=True,
                action_status="REJECTED",
            )
        self._active_handle = handle
        details["accepted"] = True
        self._active_result_future = handle.get_result_async()
        result_future = self._active_result_future
        if not self._wait_future(result_future, deadline):
            details["cancel"] = self.cancel_active()
            lifecycle = details["cancel"]
            self._active_handle = None
            self._active_goal_future = None
            self._active_result_future = None
            details["phase"] = "execution"
            details["error"] = (
                f"gripper action exceeded the {timeout_s}s wall deadline "
                "(cancel attempted)"
            )
            return ExecutionResult(
                status=ExecutionStatus.TIMED_OUT, details=details, command_messages=1,
                goal_accepted=True,
                terminal=lifecycle.get("terminal") is True,
                safe_to_release=(
                    lifecycle.get("terminal") is True
                    and lifecycle.get("safe") is True
                ),
                cancel_evidence=lifecycle,
            )
        self._active_handle = None
        self._active_goal_future = None
        self._active_result_future = None
        wrapper = result_future.result()
        details["goal_status"] = int(wrapper.status)
        details["action_status"] = _terminal_status_name(
            wrapper.status, factories.goal_status_type()
        )
        result_states = self._isaac_state_count
        while self._isaac_state_count <= result_states and self._clock_fn() < deadline:
            self._spin_fn()
            if self._isaac_state_count > result_states:
                break
            self._sleep_fn(self._wait_step_s)
        details["post_result_state_messages"] = self._isaac_state_count - result_states
        common = {
            "goal_accepted": True,
            "terminal": True,
            "safe_to_release": True,
            "action_status": details["action_status"],
        }
        commanded = float(target.opening_m)
        final = self.latest_isaac_state()
        measured: float | None = None
        if (
            final is None
            or "gripper_joint1" not in final
            or "gripper_joint2" not in final
        ):
            details["measured_opening_error"] = (
                "no complete isaac finger state was observed after the action"
            )
        else:
            try:
                measured = fingers_to_logical_gripper(
                    float(final["gripper_joint1"]), float(final["gripper_joint2"])
                )
            except ValueError as exc:
                details["measured_opening_error"] = str(exc)
        window_low, window_high = (
            float(target.expected_opening_m[0]),
            float(target.expected_opening_m[1]),
        )
        details["commanded_opening_m"] = commanded
        details["measured_opening_m"] = measured
        details["expected_opening_m"] = [window_low, window_high]
        details["duration_s"] = float(target.duration_s)
        final_errors: dict[str, float | None] = {
            "gripper_opening_m": (
                None if measured is None else abs(measured - commanded)
            )
        }
        if (
            measured is None
            or self._isaac_state_count <= result_states
            or not self._state_fresh()
        ):
            details["error"] = (
                "no fresh post-action gripper opening was measured; refusing "
                "to claim success"
            )
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                planned_waypoints=2,
                command_messages=1,
                final_errors=final_errors,
                details=details,
                **common,
            )
        if not (window_low <= measured <= window_high):
            details["error"] = (
                "measured gripper opening is outside the expected opening window"
            )
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                planned_waypoints=2,
                command_messages=1,
                final_errors=final_errors,
                details=details,
                **common,
            )
        if details["action_status"] != "SUCCEEDED":
            details["error"] = "gripper action did not report SUCCEEDED"
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                planned_waypoints=2,
                command_messages=1,
                final_errors=final_errors,
                details=details,
                **common,
            )
        return ExecutionResult(
            status=ExecutionStatus.SUCCEEDED,
            moveit_error_code=SUCCESS_CODE,
            planned_waypoints=2,
            command_messages=1,
            final_errors=final_errors,
            details=details,
            **common,
        )

    def _send_goal(
        self, kind: str, target: JointTarget | Link6PoseTarget | TcpPoseTarget, timeout_s: float
    ) -> ExecutionResult:
        decision = strict_action_admission(target)
        if not decision.accepted:
            return ExecutionResult(
                status=ExecutionStatus.REJECTED,
                details={
                    "kind": kind,
                    "reason": decision.reason,
                    "admission": dict(decision.details),
                },
                command_messages=0,
            )
        if (
            not isinstance(timeout_s, (int, float))
            or not math.isfinite(float(timeout_s))
            or float(timeout_s) <= 0.0
        ):
            return ExecutionResult(
                status=ExecutionStatus.REJECTED,
                details={
                    "kind": kind,
                    "reason": (
                        f"timeout_s must be finite and positive, got {timeout_s!r}"
                    ),
                },
                command_messages=0,
            )
        if self._move_group_client is None:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                details={
                    "kind": kind,
                    "error": (
                        "move_group action client is not registered; call "
                        "register_clients() first"
                    ),
                },
                command_messages=0,
            )
        snapshot = self.readiness_snapshot()
        if not snapshot.passed:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                details={
                    "kind": kind,
                    "error": (
                        "runtime readiness failed; no motion is allowed until "
                        "wait_until_ready passes"
                    ),
                    "readiness": asdict(snapshot),
                },
                command_messages=0,
            )
        factories = self._resolve_factories()
        if kind == "joint":
            goal = self._build_joint_goal(target, factories)
        else:
            goal = self._build_pose_goal(target, factories)
        baseline = self.latest_isaac_state()
        return self._execute_goal(
            goal,
            kind=kind,
            target=target,
            timeout_s=float(timeout_s),
            baseline=baseline,
            factories=factories,
        )

    def _build_joint_goal(self, target: JointTarget, factories: RosMoveItFactories) -> Any:
        """MoveGroup joint goal: plan_only false, exact arm JointConstraints.

        The admitted target carries the canonical arm joint names; the
        constraints are emitted with THIS robot's device joint names in the
        same canonical order, so a piper_1 goal can never command piper_0's
        joints.
        """
        goal = factories.move_group_goal_factory()
        request = factories.motion_plan_request_factory()
        request.group_name = self._arm_planning_group
        request.allowed_planning_time = ALLOWED_PLANNING_TIME_S
        request.num_planning_attempts = NUM_PLANNING_ATTEMPTS
        request.max_velocity_scaling_factor = MAX_SCALING_FACTOR
        request.max_acceleration_scaling_factor = MAX_SCALING_FACTOR
        goal_constraints = factories.constraints_factory()
        tolerance = float(target.goal_tolerance)
        for index, name in enumerate(ARM_JOINT_NAMES):
            constraint = factories.joint_constraint_factory()
            constraint.joint_name = self._arm_joint_names[index]
            constraint.position = float(target.positions[name])
            constraint.tolerance_above = tolerance
            constraint.tolerance_below = tolerance
            constraint.weight = 1.0
            goal_constraints.joint_constraints.append(constraint)
        request.goal_constraints.append(goal_constraints)
        goal.request = request
        goal.planning_options.plan_only = False
        goal.planning_options.replan = False
        return goal

    def _device_link(self, canonical_link: str) -> str:
        """Map an admitted canonical constraint link onto this robot's link.

        ``rocobrick_tcp``/``link6`` are the only links strict admission lets
        through; with a profile they resolve to that robot's calibrated TCP /
        raw flange names, so a piper_1 goal can never constrain piper_0's
        flange. Without a profile the canonical names are already device
        names and pass through unchanged.
        """
        if canonical_link == TCP_LINK:
            return self._tcp_link
        if canonical_link == TIP_LINK:
            return self._link6_link
        return canonical_link

    def _device_frame(self, canonical_frame: str) -> str:
        """Map an admitted canonical frame onto this robot's base frame."""
        if canonical_frame == BASE_FRAME:
            return self._base_frame
        return canonical_frame

    def _build_pose_goal(
        self, target: Link6PoseTarget | TcpPoseTarget, factories: RosMoveItFactories
    ) -> Any:
        """MoveGroup pose goal: one SPHERE position + one orientation constraint.

        The planning group follows the target: the raw link6 group for link6
        goals and the calibrated TCP group for project-calibrated TCP goals,
        both resolved from this robot's profile. The admitted canonical
        constraint link/frame are mapped onto this robot's device names; the
        pose values themselves are used verbatim (never offset or rewritten).
        """
        group = (
            self._tcp_planning_group
            if isinstance(target, TcpPoseTarget)
            else self._arm_planning_group
        )
        link = self._device_link(target.tip_link)
        frame = self._device_frame(target.frame_id)
        goal = factories.move_group_goal_factory()
        request = factories.motion_plan_request_factory()
        request.group_name = group
        request.allowed_planning_time = ALLOWED_PLANNING_TIME_S
        request.num_planning_attempts = NUM_PLANNING_ATTEMPTS
        request.max_velocity_scaling_factor = MAX_SCALING_FACTOR
        request.max_acceleration_scaling_factor = MAX_SCALING_FACTOR
        goal_constraints = factories.constraints_factory()

        position = factories.position_constraint_factory()
        position.header.frame_id = frame
        position.link_name = link
        sphere = factories.solid_primitive_factory()
        sphere.type = getattr(type(sphere), "SPHERE")
        sphere.dimensions = [float(target.position_tolerance_m)]
        position.constraint_region.primitives.append(sphere)
        center = factories.pose_factory()
        center.position.x = float(target.position[0])
        center.position.y = float(target.position[1])
        center.position.z = float(target.position[2])
        center.orientation.w = 1.0
        position.constraint_region.primitive_poses.append(center)
        position.weight = 1.0
        goal_constraints.position_constraints.append(position)

        orientation = factories.orientation_constraint_factory()
        orientation.header.frame_id = frame
        orientation.link_name = link
        orientation.orientation.x = float(target.orientation_xyzw[0])
        orientation.orientation.y = float(target.orientation_xyzw[1])
        orientation.orientation.z = float(target.orientation_xyzw[2])
        orientation.orientation.w = float(target.orientation_xyzw[3])
        orientation_tolerance = float(target.orientation_tolerance_rad)
        orientation.absolute_x_axis_tolerance = orientation_tolerance
        orientation.absolute_y_axis_tolerance = orientation_tolerance
        orientation.absolute_z_axis_tolerance = orientation_tolerance
        orientation.weight = 1.0
        goal_constraints.orientation_constraints.append(orientation)

        request.goal_constraints.append(goal_constraints)
        goal.request = request
        goal.planning_options.plan_only = False
        goal.planning_options.replan = False
        return goal

    def _wait_future(self, future: Any, deadline: float) -> bool:
        """Bounded wait on a future via the injected clock/spin/sleep hooks."""
        while not future.done():
            self._spin_fn()
            if future.done():
                break
            if self._clock_fn() >= deadline:
                return False
            self._sleep_fn(self._wait_step_s)
        return True

    def _execute_goal(
        self,
        goal: Any,
        *,
        kind: str,
        target: JointTarget | Link6PoseTarget | TcpPoseTarget,
        timeout_s: float,
        baseline: Mapping[str, float] | None,
        factories: RosMoveItFactories,
    ) -> ExecutionResult:
        """Bounded MoveGroup send; timeout cancels and never reports success."""
        details: dict[str, object] = {"kind": kind}
        send_commands = self._command_count
        send_states = self._isaac_state_count
        now = self._clock_fn()
        deadline = now + timeout_s
        client = self._move_group_client
        try:
            future = client.send_goal_async(goal)
            self._active_goal_future = future
        except Exception as exc:
            details["error"] = f"send_goal_async failed: {type(exc).__name__}: {exc}"
            return ExecutionResult(
                status=ExecutionStatus.FAILED, details=details, command_messages=0
            )
        if not self._wait_future(future, min(now + ACCEPT_TIMEOUT_S, deadline)):
            details["phase"] = "goal_acceptance"
            details["error"] = (
                "move_group goal response not received before the acceptance deadline"
            )
            details["cancel"] = self.cancel_active()
            lifecycle = details["cancel"]
            self._active_goal_future = None
            return ExecutionResult(
                status=ExecutionStatus.TIMED_OUT,
                details=details,
                command_messages=self._command_count - send_commands,
                goal_accepted=lifecycle.get("goal_accepted"),
                terminal=bool(lifecycle.get("terminal", False)),
                safe_to_release=bool(lifecycle.get("safe", False)),
                cancel_evidence=lifecycle,
            )
        handle = future.result()
        if handle is None or not getattr(handle, "accepted", False):
            details["phase"] = "goal_acceptance"
            details["accepted"] = False
            details["error"] = "move_group did not accept the goal"
            return ExecutionResult(
                status=ExecutionStatus.REJECTED,
                details=details,
                command_messages=self._command_count - send_commands,
            )
        self._active_handle = handle
        details["accepted"] = True
        self._active_result_future = handle.get_result_async()
        result_future = self._active_result_future
        if not self._wait_future(result_future, deadline):
            details["cancel"] = self.cancel_active()
            lifecycle = details["cancel"]
            self._active_handle = None
            self._active_goal_future = None
            self._active_result_future = None
            details["phase"] = "execution"
            details["error"] = (
                f"move_group action exceeded the {timeout_s}s wall deadline "
                "(cancel attempted)"
            )
            return ExecutionResult(
                status=ExecutionStatus.TIMED_OUT,
                details=details,
                command_messages=self._command_count - send_commands,
                goal_accepted=True,
                terminal=bool(lifecycle.get("terminal", False)),
                safe_to_release=bool(lifecycle.get("safe", False)),
                cancel_evidence=lifecycle,
            )
        self._active_handle = None
        self._active_goal_future = None
        self._active_result_future = None
        wrapper = result_future.result()
        action_result = wrapper.result
        status = _map_goal_status(wrapper.status, factories.goal_status_type())
        error_code = int(getattr(action_result.error_code, "val"))
        planned_waypoints, trajectory_decision = self._inspect_trajectory(action_result)
        details["goal_status"] = int(wrapper.status)
        details["action_status"] = _terminal_status_name(
            wrapper.status, factories.goal_status_type()
        )
        details["trajectory_accepted"] = trajectory_decision.accepted
        if not trajectory_decision.accepted:
            details["trajectory"] = dict(trajectory_decision.details)
        common = dict(
            moveit_error_code=error_code,
            planned_waypoints=planned_waypoints,
            command_messages=self._command_count - send_commands,
            goal_accepted=True,
            terminal=True,
            safe_to_release=True,
            action_status=details["action_status"],
        )
        if status is not ExecutionStatus.SUCCEEDED:
            details["error"] = "move_group terminal status did not report success"
            return ExecutionResult(status=status, details=details, **common)
        if error_code != SUCCESS_CODE:
            details["error"] = (
                f"moveit error code {error_code} is not SUCCESS ({SUCCESS_CODE})"
            )
            return ExecutionResult(status=ExecutionStatus.FAILED, details=details, **common)
        if not trajectory_decision.accepted:
            details["error"] = (
                "planned trajectory failed caller-side waypoint admission"
            )
            return ExecutionResult(status=ExecutionStatus.FAILED, details=details, **common)
        return self._verify_final(
            kind, target, baseline, send_states, send_commands, common, details
        )

    def _inspect_trajectory(self, action_result: Any) -> tuple[int, AdmissionDecision]:
        """R1 waypoint admission over the returned planned trajectory.

        A MoveGroup answer names the ARM device joints of the robot whose
        move_group produced it. Only a trajectory naming exactly this
        runtime's arm joints is projected onto the canonical order and
        admitted; a trajectory naming any other robot's joints keeps its
        names and is rejected by the R1 admission (whose evidence lists the
        foreign names), so cross-robot plan output can never be accepted.
        """
        trajectory = action_result.planned_trajectory.joint_trajectory
        names = [str(name) for name in trajectory.joint_names]
        try:
            points = [
                [float(value) for value in point.positions]
                for point in trajectory.points
            ]
        except (AttributeError, TypeError, ValueError):
            return 0, AdmissionDecision(
                accepted=False,
                reason="planned trajectory waypoints are not numeric",
                details={},
            )
        if len(names) == len(self._arm_joint_names) and set(names) == set(
            self._arm_joint_names
        ):
            by_device = {
                device: canonical
                for canonical, device in zip(ARM_JOINT_NAMES, self._arm_joint_names)
            }
            names = [by_device[name] for name in names]
        return len(points), validate_trajectory_waypoints(names, points)

    def _verify_final(
        self,
        kind: str,
        target: JointTarget | Link6PoseTarget | TcpPoseTarget,
        baseline: Mapping[str, float] | None,
        send_states: int,
        send_commands: int,
        common: dict[str, object],
        details: dict[str, object],
    ) -> ExecutionResult:
        """Final live-feedback gates; failures are explicit, never faked."""
        post_states = self._isaac_state_count - send_states
        post_commands = self._command_count - send_commands
        details["post_state_messages"] = post_states
        if post_states < MIN_POST_COMMAND_STATES or post_commands < MIN_COMMAND_MESSAGES:
            details["error"] = (
                f"insufficient post-command evidence: states={post_states} "
                f"(< {MIN_POST_COMMAND_STATES}) or commands={post_commands} "
                f"(< {MIN_COMMAND_MESSAGES})"
            )
            return ExecutionResult(
                status=ExecutionStatus.FAILED, details=details, **common
            )
        final, settled = self._await_settled_final()
        details["settled_at_rest"] = settled
        if final is None or not self._state_fresh():
            details["error"] = (
                "no fresh complete final state was observed; refusing to fake success"
            )
            return ExecutionResult(
                status=ExecutionStatus.FAILED, details=details, **common
            )
        if kind == "joint":
            return self._verify_joint_final(target, final, baseline, common, details)
        return self._verify_pose_final(target, final, common, details)

    def _await_settled_final(self) -> tuple[dict[str, float] | None, bool]:
        """Bounded at-rest wait over live samples before the final verdict.

        Returns ``(final_state, settled_at_rest)``. The state is sampled only
        once FINAL_SETTLE_SAMPLES consecutive live samples moved less than
        FINAL_SETTLE_STEP_MAX (the R1 ``state_is_stable`` predicate over
        runtime samples); when the bound expires first, the latest sample is
        returned with ``settled=False`` so the tolerance gates still decide
        the outcome explicitly. No motion is ever sent from here.
        """
        previous = self.latest_isaac_state()
        if previous is None:
            return None, False
        stable = 0
        deadline = self._clock_fn() + FINAL_CONVERGENCE_TIMEOUT_S
        while True:
            self._sleep_fn(FINAL_CONVERGENCE_POLL_S)
            if self._clock_fn() >= deadline:
                break
            current = self.latest_isaac_state()
            if current is None:
                stable = 0
                continue
            stable = (
                stable + 1
                if state_is_stable(previous, current, max_step=FINAL_SETTLE_STEP_MAX)
                else 0
            )
            previous = current
            if stable >= FINAL_SETTLE_SAMPLES:
                return self.latest_isaac_state(), True
        return self.latest_isaac_state(), False

    def _verify_joint_final(
        self,
        target: JointTarget,
        final: Mapping[str, float],
        baseline: Mapping[str, float] | None,
        common: dict[str, object],
        details: dict[str, object],
    ) -> ExecutionResult:
        if baseline is None:
            details["error"] = (
                "no pre-command baseline state was observed; gripper drift "
                "cannot be verified"
            )
            return ExecutionResult(
                status=ExecutionStatus.FAILED, details=details, **common
            )
        max_error = max_joint_error(final, target.positions)
        drift = max_joint_drift(final, baseline, names=GRIPPER_JOINT_NAMES)
        final_errors = {
            name: abs(float(final[name]) - float(target.positions[name]))
            for name in ARM_JOINT_NAMES
        }
        details["max_joint_error"] = max_error
        details["gripper_drift"] = drift
        details["final_arm_error_max"] = FINAL_ARM_ERROR_MAX
        details["gripper_drift_max"] = GRIPPER_DRIFT_MAX
        if max_error > FINAL_ARM_ERROR_MAX or drift > GRIPPER_DRIFT_MAX:
            details["error"] = (
                "final live state did not converge to the joint target within tolerance"
            )
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                final_errors=final_errors,
                details=details,
                **common,
            )
        return ExecutionResult(
            status=ExecutionStatus.SUCCEEDED,
            final_errors=final_errors,
            details=details,
            **common,
        )

    def _verify_pose_final(
        self,
        target: Link6PoseTarget | TcpPoseTarget,
        final: Mapping[str, float],
        common: dict[str, object],
        details: dict[str, object],
    ) -> ExecutionResult:
        verifier = self._fk_verifier
        if isinstance(target, TcpPoseTarget) and self._tcp_fk_verifier is not None:
            verifier = self._tcp_fk_verifier
        if verifier is None:
            details["error"] = (
                "no fk verification hook was supplied; the exact final pose "
                "cannot be observed"
            )
            return ExecutionResult(
                status=ExecutionStatus.FAILED, details=details, **common
            )
        try:
            verified = verifier(dict(final))
        except Exception as exc:
            details["error"] = f"fk verification hook raised: {type(exc).__name__}: {exc}"
            return ExecutionResult(
                status=ExecutionStatus.FAILED, details=details, **common
            )
        if verified is None:
            details["error"] = "fk verification failed for the final live state"
            return ExecutionResult(
                status=ExecutionStatus.FAILED, details=details, **common
            )
        position, quaternion = verified
        position_error = position_error_m(position, target.position)
        orientation_error = orientation_error_rad(
            quaternion, target.orientation_xyzw
        )
        final_errors = {
            "position_error_m": position_error,
            "orientation_error_rad": orientation_error,
        }
        details["final_fk_position_m"] = [float(value) for value in position]
        details["final_fk_orientation_xyzw"] = [float(value) for value in quaternion]
        details["target_position_m"] = [float(value) for value in target.position]
        details["target_orientation_xyzw"] = [float(value) for value in target.orientation_xyzw]
        details["position_residual_m"] = [
            float(actual) - float(goal)
            for actual, goal in zip(position, target.position)
        ]
        details["final_arm_joint_positions_rad"] = {
            name: float(final[name]) for name in ARM_JOINT_NAMES if name in final
        }
        details["position_tolerance_m"] = MAX_POSITION_TOLERANCE_M
        details["orientation_tolerance_rad"] = MAX_ORIENTATION_TOLERANCE_RAD
        if (
            position_error > MAX_POSITION_TOLERANCE_M
            or orientation_error > MAX_ORIENTATION_TOLERANCE_RAD
        ):
            details["error"] = (
                "final live fk did not confirm the requested pose within tolerance"
            )
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                final_errors=final_errors,
                details=details,
                **common,
            )
        return ExecutionResult(
            status=ExecutionStatus.SUCCEEDED,
            final_errors=final_errors,
            details=details,
            **common,
        )

    def _state_fresh(self) -> bool:
        with self._lock:
            if not self._isaac_receipts:
                return False
            return (self._clock_fn() - self._isaac_receipts[-1]) <= STATE_FRESH_SECONDS

    # ---------------------------------------------------------------- control

    def cancel_active(self) -> dict[str, object]:
        """Cancel the active goal and wait for its final action result.

        A cancel request acknowledgement alone is not terminal evidence. This
        method also settles a late goal acceptance and then waits for the
        accepted goal's result future.
        """
        info: dict[str, object] = {
            "attempted": True,
            "goal_accepted": None,
            "terminal": False,
            "safe": False,
        }
        deadline = self._clock_fn() + CANCEL_TIMEOUT_S
        try:
            handle = self._active_handle
            goal_future = self._active_goal_future
            if handle is None and goal_future is not None:
                while not goal_future.done():
                    self._spin_fn()
                    if self._clock_fn() >= deadline:
                        info["error"] = "goal acceptance did not settle before cancel deadline"
                        return info
                    self._sleep_fn(self._wait_step_s)
                handle = goal_future.result()
                if handle is None or not getattr(handle, "accepted", False):
                    info["goal_accepted"] = False
                    info["terminal"] = True
                    info["safe"] = True
                    info["final_action_status"] = "REJECTED"
                    return info
                self._active_handle = handle
                self._active_result_future = handle.get_result_async()
            if handle is None:
                return {"attempted": False}
            info["goal_accepted"] = True
            cancel_future = handle.cancel_goal_async()
            while not cancel_future.done():
                self._spin_fn()
                if self._clock_fn() >= deadline:
                    info["error"] = "cancel did not complete before its deadline"
                    return info
                self._sleep_fn(self._wait_step_s)
            cancel_response = cancel_future.result()
            info["return_code"] = int(getattr(cancel_response, "return_code"))
            result_future = self._active_result_future
            if result_future is None:
                info["error"] = "accepted goal has no result future"
                return info
            while not result_future.done():
                self._spin_fn()
                if self._clock_fn() >= deadline:
                    info["error"] = "action result did not settle after cancel"
                    return info
                self._sleep_fn(self._wait_step_s)
            wrapper = result_future.result()
            status = int(getattr(wrapper, "status"))
            info["final_action_status"] = status
            info["terminal"] = True
            info["safe"] = True
            return info
        except Exception as exc:
            info["error"] = f"{type(exc).__name__}: {exc}"
            return info

    def shutdown(self) -> None:
        """Best-effort teardown of every registered handle; errors recorded."""
        destroyer = getattr(self.node, "destroy_subscription", None)
        for topic, subscription in list(self._subscriptions.items()):
            try:
                if destroyer is not None:
                    destroyer(subscription)
            except Exception as exc:
                self.shutdown_errors.append(
                    f"subscription {topic}: {type(exc).__name__}: {exc}"
                )
        self._subscriptions.clear()
        client_destroyer = getattr(self.node, "destroy_client", None)
        for name, client in (
            ("list_controllers", self._list_controllers_client),
            ("plan", self._plan_client),
            ("compute_fk", self._fk_client),
            ("apply_planning_scene", self._planning_scene_client),
            ("get_planning_scene", self._get_planning_scene_client),
            ("scene_state", self._scene_state_client),
        ):
            try:
                if client is not None and client_destroyer is not None:
                    client_destroyer(client)
            except Exception as exc:
                self.shutdown_errors.append(
                    f"client {name}: {type(exc).__name__}: {exc}"
                )
        for name, client in (
            ("arm_follow", self._arm_client),
            ("move_group", self._move_group_client),
            ("gripper_follow", self._gripper_client),
        ):
            try:
                destroy = getattr(client, "destroy", None)
                if client is not None and destroy is not None:
                    destroy()
            except Exception as exc:
                self.shutdown_errors.append(
                    f"action {name}: {type(exc).__name__}: {exc}"
                )
        self._list_controllers_client = None
        self._plan_client = None
        self._fk_client = None
        self._planning_scene_client = None
        self._get_planning_scene_client = None
        self._scene_state_client = None
        self._arm_client = None
        self._move_group_client = None
        self._gripper_client = None
        self._active_handle = None


class RosMoveItRobotSession:
    """One ROS node hosting ONE isolated runtime per plan robot (A5).

    Every robot gets its own :class:`RosMoveItRuntime` built from its
    :class:`RobotRuntimeProfile`: robot-scoped state topics, controller/FK/
    plan clients, MoveGroup action and planning-scene clients, arm and
    gripper follow actions, plus its own state/controller/feedback
    containers. Nothing is shared except the injected node, the ROS domain,
    the single injected event pump and the shared ``/clock`` that every
    profile pins; the profile disjointness is asserted at construction. The
    host-bridge scene-state Trigger stays the one global A4 service, and each
    runtime slices its own robot block out of the answer, so there is no
    canonical fallback of any endpoint and no way to read another robot's
    observation.

    Read-only readiness is evaluated per robot through
    :meth:`capability_readiness_by_robot` (joints, controllers, MoveGroup,
    arm/gripper actions, FK and the robot-scoped scene query) and
    :meth:`wait_until_ready` requires EVERY robot to pass before the
    deadline. Nothing here ever sends motion.
    """

    def __init__(
        self,
        node: Any,
        robot_ids: Sequence[str],
        *,
        factories: RosMoveItFactories | None = None,
        qos_sensor: Any = None,
        qos_state: Any = None,
        profile_factory: Callable[[str], RobotRuntimeProfile] = (
            RobotRuntimeProfile.from_robot_id
        ),
        runtime_factory: (
            Callable[[str, RobotRuntimeProfile], RosMoveItRuntime] | None
        ) = None,
        executor_factory: Callable[[RosMoveItRuntime], Any] | None = None,
        fk_timeout_s: float = SERVICE_CALL_TIMEOUT_S,
        clock: Callable[[], float] | None = None,
        spin: Callable[[], None] | None = None,
        sleep: Callable[[float], None] | None = None,
        poll_interval_s: float = POLL_INTERVAL_S,
        wait_step_s: float = WAIT_STEP_S,
    ) -> None:
        ids = tuple(robot_ids)
        if not ids:
            raise ValueError("a robot session needs at least one robot id")
        if len(set(ids)) != len(ids):
            raise ValueError(f"robot ids must be unique, got {list(ids)}")
        if (
            not isinstance(fk_timeout_s, (int, float))
            or not math.isfinite(float(fk_timeout_s))
            or float(fk_timeout_s) <= 0.0
        ):
            raise ValueError(
                f"fk_timeout_s must be finite and positive, got {fk_timeout_s!r}"
            )
        profiles = {
            robot_id: profile_factory(robot_id) for robot_id in ids
        }
        for robot_id, profile in profiles.items():
            if profile.robot_id != robot_id:
                raise ValueError(
                    f"profile for {robot_id!r} reports robot_id "
                    f"{profile.robot_id!r}"
                )
        assert_profiles_disjoint(tuple(profiles[robot_id] for robot_id in ids))
        self.node = node
        self.robot_ids = ids
        self.profiles: dict[str, RobotRuntimeProfile] = profiles
        self._clock_fn = clock if clock is not None else time.monotonic
        self._spin_fn = spin if spin is not None else (lambda: None)
        self._sleep_fn = sleep if sleep is not None else time.sleep
        self._poll_interval_s = float(poll_interval_s)
        self._wait_step_s = float(wait_step_s)
        self.shutdown_errors: list[str] = []

        def default_runtime(
            robot_id: str, profile: RobotRuntimeProfile
        ) -> RosMoveItRuntime:
            holder: dict[str, Any] = {}

            def fk_hook(state: Mapping[str, float]) -> Any:
                runtime = holder.get("runtime")
                if runtime is None:
                    return None
                return runtime.compute_link6_fk(state, timeout_s=float(fk_timeout_s))

            def tcp_fk_hook(state: Mapping[str, float]) -> Any:
                runtime = holder.get("runtime")
                if runtime is None:
                    return None
                return runtime.compute_tcp_fk(state, timeout_s=float(fk_timeout_s))

            runtime = RosMoveItRuntime(
                node,
                profile=profile,
                qos_sensor=qos_sensor,
                qos_state=qos_state,
                factories=factories,
                fk_verifier=fk_hook,
                tcp_fk_verifier=tcp_fk_hook,
                clock=self._clock_fn,
                spin=self._spin_fn,
                sleep=self._sleep_fn,
                poll_interval_s=self._poll_interval_s,
                wait_step_s=self._wait_step_s,
            )
            holder["runtime"] = runtime
            return runtime

        create_runtime = runtime_factory if runtime_factory is not None else default_runtime
        self.runtimes: dict[str, RosMoveItRuntime] = {
            robot_id: create_runtime(robot_id, profiles[robot_id]) for robot_id in ids
        }
        if executor_factory is None:
            from rocobrick.ros_moveit_executor import RosMoveItExecutor

            def executor_factory(runtime: RosMoveItRuntime) -> Any:  # noqa: E301
                return RosMoveItExecutor(runtime)
        self.executors: dict[str, Any] = {
            robot_id: executor_factory(self.runtimes[robot_id]) for robot_id in ids
        }

    # ------------------------------------------------------------ registration

    def register(self) -> None:
        """Register every robot's subscriptions and clients exactly once."""
        self.register_subscriptions()
        self.register_clients()

    def register_subscriptions(self) -> None:
        for runtime in self.runtimes.values():
            runtime.register_subscriptions()

    def register_clients(self) -> None:
        for runtime in self.runtimes.values():
            runtime.register_clients()

    # -------------------------------------------------------------- accessors

    def runtime_for(self, robot_id: str) -> RosMoveItRuntime:
        """The isolated runtime of ``robot_id``; unknown ids raise KeyError."""
        try:
            return self.runtimes[robot_id]
        except KeyError as error:
            raise KeyError(
                f"unknown robot {robot_id!r}; session robots: "
                f"{list(self.robot_ids)}"
            ) from error

    def executor_for(self, robot_id: str) -> Any:
        """The executor bound to ``robot_id``'s runtime; unknown ids raise."""
        try:
            return self.executors[robot_id]
        except KeyError as error:
            raise KeyError(
                f"unknown robot {robot_id!r}; session robots: "
                f"{list(self.robot_ids)}"
            ) from error

    # -------------------------------------------------------------- readiness

    def capability_readiness_by_robot(
        self, *, probe_scene: bool = False
    ) -> dict[str, RobotCapabilityReadiness]:
        """One read-only capability verdict per robot, keyed by robot id.

        With ``probe_scene`` every robot's scene component is verified
        against the live per-robot scene response (fail-closed), which is what
        the session readiness gate uses.
        """
        return {
            robot_id: self.runtimes[robot_id].capability_readiness(
                probe_scene=probe_scene
            )
            for robot_id in self.robot_ids
        }

    def not_ready_robots(self, *, probe_scene: bool = False) -> tuple[str, ...]:
        """Robot ids whose capability readiness does not pass right now."""
        return tuple(
            robot_id
            for robot_id, report in self.capability_readiness_by_robot(
                probe_scene=probe_scene
            ).items()
            if not report.passed
        )

    def wait_until_ready(self, timeout_s: float) -> bool:
        """Bounded all-robot capability readiness wait; never sends motion.

        True only when EVERY robot's capability readiness holds before the
        deadline (scene response probed live per robot), so one not-ready
        robot fails the whole session. The shared spin/sleep hooks are the
        single injected event pump.
        """
        if not math.isfinite(float(timeout_s)) or float(timeout_s) < 0.0:
            raise ValueError(
                f"timeout_s must be finite and non-negative, got {timeout_s!r}"
            )
        deadline = self._clock_fn() + float(timeout_s)
        while True:
            self._spin_fn()
            if not self.not_ready_robots(probe_scene=True):
                return True
            if self._clock_fn() >= deadline:
                return False
            self._sleep_fn(self._poll_interval_s)

    # --------------------------------------------------------------- teardown

    def shutdown(self) -> None:
        """Best-effort teardown of every robot runtime; errors recorded."""
        for robot_id in self.robot_ids:
            runtime = self.runtimes[robot_id]
            try:
                runtime.shutdown()
            except Exception as exc:
                self.shutdown_errors.append(
                    f"{robot_id}: {type(exc).__name__}: {exc}"
                )
            for message in runtime.shutdown_errors:
                self.shutdown_errors.append(f"{robot_id}: {message}")
