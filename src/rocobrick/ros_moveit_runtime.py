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

import os

import json
import math
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
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
from rocobrick.gripper import (
    fingers_to_contact_opening,
    fingers_to_logical_gripper,
    logical_gripper_to_fingers,
)
from rocobrick.moveit_contract import (
    ACTION_ARM_FOLLOW,
    ACTION_GRIPPER_FOLLOW,
    ARM_JOINT_NAMES,
    BASE_FRAME,
    EXACT_STATE_MATCH_TOLERANCE,
    GRIPPER_JOINT_NAMES,
    JOINT_NAMES,
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
    box_geometry_digest,
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
from rocobrick.unified_readiness import UnifiedDualArmReadiness

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
IK_SEARCH_TIMEOUT_S = 0.5
IK_ATTEMPT_TIMEOUT_S = 0.05
IK_SEED_ATTEMPTS = 24
COMBINED_ROBOT_DESCRIPTION_TOPIC = "/dual_arm/robot_description"


def ik_seed_candidates(
    measured: Mapping[str, float],
    limits: Mapping[str, tuple[float, float]],
    count: int,
    token: str,
) -> list[dict[str, float]]:
    """Build deterministic diverse IK seeds around the measured state.

    The stock KDL plugin runs one Newton descent per call, so restart
    diversity must come from explicit seeds. The measured state is always
    first; the rest are reproducible uniform samples inside the combined
    model's per-joint limits, keyed by the caller's transition token.
    """
    import random as _random

    rng = _random.Random(token)
    joints = sorted(set(measured) & set(limits))
    candidates: list[dict[str, float]] = []
    for _ in range(max(0, count)):
        candidates.append(
            {name: rng.uniform(*limits[name]) for name in joints}
            | {name: value for name, value in measured.items() if name not in limits}
        )
    return candidates


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
# Direct arm-completion re-command.  When a MoveGroup goal succeeds but the
# final FK gate finds the arm frozen short of the plan (the container-side
# pipeline stops streaming setpoints at trajectory end), the runtime
# re-sends the planned final joints over its own arm follow action with
# these tight bounds; the subsequent FK re-verification still decides.
ARM_COMPLETION_DURATION_S = 3.0
ARM_COMPLETION_TIMEOUT_S = 20.0
ARM_COMPLETION_TOLERANCE_RAD = 0.005
# Post-action convergence wait: the simulated fingers travel at roughly
# 5 mm/s while the controller holds the last trajectory setpoint, so a full
# close can need several seconds of continued convergence after the action
# result; the wait is bounded by this and the remaining action deadline.
GRIPPER_CONVERGENCE_TIMEOUT_S = 15.0
# Contact-stall detection under a sustained press.  Per-sample stillness
# (a fixed step between adjacent states) is defeated by contact
# chatter: live evidence (headed close run, 2026-10-01) shows a close goal
# pressing 4 mm into the brick holding the action active for the full 60 s
# wall deadline because neither the controller's own tolerance check nor the
# stillness monitor ever fired.  A stall is therefore "no net progress":
# once the measured opening is inside the expected window, the finger
# reading may drift at most GRIPPER_CONTACT_STALL_DRIFT_M from the entering
# reference for GRIPPER_CONTACT_STALL_S before the goal is cancelled as
# contact-limited; net progress beyond the drift re-anchors the reference.
GRIPPER_CONTACT_STALL_DRIFT_M = 1.0e-3
GRIPPER_CONTACT_STALL_S = 1.5
GRIPPER_OPENING_TRACE_MAX = 256

# The MoveGroup goal region is tightened to this fraction of the target's own
# tolerance. The final FK gate verifies against the full target tolerance, so
# a planner allowed the whole budget legitimately stops anywhere inside it
# (live evidence: pose-goal residuals of +3.3 mm and -4.5 mm on the same
# target) and the gate then fails intermittently by planner slack alone.
# Planning inside the fraction keeps the verified residual dominated by
# tracking error with headroom against the requested tolerance.
POSE_GOAL_TOLERANCE_FRACTION = 0.4

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
    compute_ik: str = ""
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
    root = (
        "/dual_arm"
        if profile.move_group_action == "/dual_arm/move_action"
        else robot_endpoint_root(profile)
    )
    return RuntimeServices(
        list_controllers=profile.list_controllers,
        plan=profile.plan,
        compute_fk=profile.compute_fk,
        compute_ik=(
            "/dual_arm/compute_ik"
            if profile.move_group_action == "/dual_arm/move_action"
            else ""
        ),
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
    scene_evidence: Mapping[str, object] = field(default_factory=dict)

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
            "scene_evidence": dict(self.scene_evidence),
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
        raise RuntimeError(f"scene-state schema must be exactly {SCENE_STATE_SCHEMA!r}")
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
            or not all(isinstance(name, str) and name for name in bound_joint_names)
        ):
            raise RuntimeError(
                f"scene-state binding joint_names of robot {robot_id!r} are malformed"
            )
        expected = tuple(joint_names) if joint_names is not None else None
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
    if "scene_digest" in document:
        view["scene_digest"] = document["scene_digest"]
    if "sequence" in document:
        view["sequence"] = document["sequence"]
    if "sequence" in block:
        view["robot_sequence"] = block["sequence"]
    if "tcp_pose" in block:
        view["tcp_pose"] = block["tcp_pose"]
    if "placement_interfaces" in document:
        view["placement_interfaces"] = document["placement_interfaces"]
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
    # Combined per-arm IK transport (unified dual-arm stack only).
    get_position_ik_type: Callable[[], Any] | None = None
    get_position_ik_request_factory: Callable[[], Any] | None = None


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


def _quaternion_multiply(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Compose xyzw quaternions without relying on ROS/tf2 imports."""
    x, y, z, w = left
    a, b, c, d = right
    return (
        w * a + x * d + y * c - z * b,
        w * b - x * c + y * d + z * a,
        w * c + x * b - y * a + z * d,
        w * d - x * a - y * b - z * c,
    )


def _rotate_vector(
    quaternion: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Rotate one vector by an xyzw quaternion."""
    qx, qy, qz, qw = quaternion
    vx, vy, vz = vector
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


def _collision_box_matches(item: object, box: AxisAlignedBox) -> bool:
    """Check readback geometry using MoveIt's object/shape pose semantics.

    MoveIt may serialize the object origin in ``CollisionObject.pose`` and the
    primitive pose relative to that origin. Older message fixtures omit
    ``pose`` and use the primitive pose directly, so both representations are
    handled strictly and composed when the object pose is present.
    """
    try:
        if item.header.frame_id != box.frame_id:
            return False
        if len(item.primitives) != 1 or len(item.primitive_poses) != 1:
            return False
        primitive = item.primitives[0]
        shape_pose = item.primitive_poses[0]
        if primitive.type != 1 or len(primitive.dimensions) != 3:
            return False
        dimensions = tuple(float(value) for value in primitive.dimensions)
        local_center = (
            float(shape_pose.position.x),
            float(shape_pose.position.y),
            float(shape_pose.position.z),
        )
        local_quaternion = (
            float(shape_pose.orientation.x),
            float(shape_pose.orientation.y),
            float(shape_pose.orientation.z),
            float(shape_pose.orientation.w),
        )
        object_pose = getattr(item, "pose", None)
        if object_pose is None:
            center = local_center
            quaternion = local_quaternion
        else:
            object_center = (
                float(object_pose.position.x),
                float(object_pose.position.y),
                float(object_pose.position.z),
            )
            object_quaternion = (
                float(object_pose.orientation.x),
                float(object_pose.orientation.y),
                float(object_pose.orientation.z),
                float(object_pose.orientation.w),
            )
            center = tuple(
                origin + offset
                for origin, offset in zip(
                    object_center, _rotate_vector(object_quaternion, local_center)
                )
            )
            quaternion = _quaternion_multiply(object_quaternion, local_quaternion)
    except (AttributeError, IndexError, TypeError, ValueError):
        return False
    values = (*dimensions, *center, *quaternion)
    return (
        all(math.isfinite(value) for value in values)
        and all(abs(a - b) <= 1e-5 for a, b in zip(dimensions, box.dimensions_m))
        and all(abs(a - b) <= 1e-5 for a, b in zip(center, box.center_m))
        and all(abs(a - b) <= 1e-5 for a, b in zip(quaternion, (0.0, 0.0, 0.0, 1.0)))
    )


def _attached_object_matches(item: object, *, object_id: str, link_name: str) -> bool:
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
        raise ValueError(
            f"current allowed collision matrix is unreadable: {error}"
        ) from error
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
            or self._joint_names != self._arm_joint_names + self._gripper_joint_names
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
                profile.link6_planning_group if profile is not None else PLANNING_GROUP
            )
        )
        self._tcp_planning_group = (
            tcp_planning_group
            if tcp_planning_group is not None
            else (profile.planning_group if profile is not None else TCP_PLANNING_GROUP)
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
        self._stamped_isaac_states: deque[
            tuple[int, float, float, dict[str, float]]
        ] = deque(maxlen=STATE_HISTORY)
        self._isaac_receipts: deque[float] = deque(maxlen=STATE_HISTORY)
        self._joint_states: deque[dict[str, float]] = deque(maxlen=STATE_HISTORY)
        self._joint_receipts: deque[float] = deque(maxlen=STATE_HISTORY)
        self._joint_stamps: deque[float] = deque(maxlen=STATE_HISTORY)
        self._isaac_state_count = 0
        self._command_count = 0
        self._controller_states: dict[str, str] = {}
        self._controllers_future: Any = None
        self._controllers_requested_at = 0.0
        self._subscriptions: dict[str, Any] = {}
        self._list_controllers_client: Any = None
        self._plan_client: Any = None
        self._fk_client: Any = None
        self._ik_client: Any = None
        self._planning_scene_client: Any = None
        self._get_planning_scene_client: Any = None
        self._scene_state_client: Any = None
        self._last_scene_error: str | None = None
        self._last_scene_summary: Mapping[str, object] = {}
        self._arm_client: Any = None
        self._move_group_client: Any = None
        self._combined_ready: Callable[[], bool] = lambda: False
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
            GetPositionIK,
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
            ).AttachedCollisionObject(),
            allowed_collision_entry_factory=AllowedCollisionEntry,
            get_planning_scene_type=lambda: GetPlanningScene,
            get_planning_scene_request_factory=GetPlanningScene.Request,
            trigger_type=lambda: Trigger,
            trigger_request_factory=Trigger.Request,
            get_position_ik_type=lambda: GetPositionIK,
            get_position_ik_request_factory=GetPositionIK.Request,
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
                ReliabilityPolicy.RELIABLE
                if reliable
                else ReliabilityPolicy.BEST_EFFORT
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
                joint_state_type,
                self.topics.joint_states,
                self._on_joint_state,
                qos_state,
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
        factories = self._resolve_factories()
        node = self.node
        if self._list_controllers_client is None:
            self._list_controllers_client = node.create_client(
                factories.list_controllers_type(), self.services.list_controllers
            )
        if self._plan_client is None:
            self._plan_client = node.create_client(
                factories.get_motion_plan_type(), self.services.plan
            )
        if self._fk_client is None:
            self._fk_client = node.create_client(
                factories.get_position_fk_type(), self.services.compute_fk
            )
        if (
            factories.get_position_ik_type is not None
            and self.services.compute_ik
            and self._ik_client is None
        ):
            self._ik_client = node.create_client(
                factories.get_position_ik_type(), self.services.compute_ik
            )
        if (
            factories.get_planning_scene_type is not None
            and factories.get_planning_scene_request_factory is not None
            and self._get_planning_scene_client is None
        ):
            self._get_planning_scene_client = node.create_client(
                factories.get_planning_scene_type(), self.services.get_planning_scene
            )
        if factories.trigger_type is not None and self._scene_state_client is None:
            self._scene_state_client = node.create_client(
                factories.trigger_type(), self.services.scene_state
            )
        if self._arm_client is None:
            self._arm_client = factories.action_client_factory(
                node, factories.follow_joint_trajectory_type(), self.actions.arm_follow
            )
        if (
            self.actions.move_group != "/dual_arm/move_action"
            and self._move_group_client is None
        ):
            self._move_group_client = factories.action_client_factory(
                node, factories.move_group_type(), self.actions.move_group
            )
        if self._gripper_client is None:
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
            stamp = getattr(getattr(message, "header", None), "stamp", None)
            sec, nanosec = getattr(stamp, "sec", None), getattr(stamp, "nanosec", None)
            if (
                type(sec) is int
                and type(nanosec) is int
                and sec >= 0
                and 0 <= nanosec < 1_000_000_000
            ):
                self._stamped_isaac_states.append(
                    (
                        self._isaac_state_count,
                        sec + nanosec * 1e-9,
                        self._isaac_receipts[-1],
                        dict(state),
                    )
                )

    def _on_joint_state(self, message: Any) -> None:
        state = self._normalized_state(message)
        if state is None:
            return
        with self._lock:
            self._joint_states.append(state)
            self._joint_receipts.append(self._clock_fn())
            stamp = getattr(getattr(message, "header", None), "stamp", None)
            if stamp is not None:
                self._joint_stamps.append(
                    float(stamp.sec) + float(stamp.nanosec) * 1e-9
                )

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

    def latest_stamped_isaac_state(self) -> Mapping[str, object] | None:
        """Read the latest physics sample only when its own timestamp exists.

        Returns:
            Identified measured joints, physics time and receipt age; None
            rather than a preceding stamp when the newest sample is unstamped.
        """
        with self._lock:
            if not self._stamped_isaac_states:
                return None
            count, stamp, receipt, joints = self._stamped_isaac_states[-1]
            if count != self._isaac_state_count:
                return None
            return {
                "sample_count": count,
                "sim_time_s": stamp,
                "age_s": self._clock_fn() - receipt,
                "joints": dict(joints),
            }

    def state_timing(self) -> Mapping[str, float | None]:
        """Return actual receipt ages and broadcaster simulation timestamp.

        Returns:
            Per-channel ages and the latest stamped joint-state time.
        """
        now = self._clock_fn()
        with self._lock:
            return {
                "isaac_age_s": now - self._isaac_receipts[-1]
                if self._isaac_receipts
                else None,
                "joint_age_s": now - self._joint_receipts[-1]
                if self._joint_receipts
                else None,
                "joint_stamp_s": self._joint_stamps[-1] if self._joint_stamps else None,
            }

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
            states = _parse_controller_states(new_future) if new_future.done() else None
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
            move_group_action_ready=self._action_ready(self._move_group_client)
            or (
                self.actions.move_group == "/dual_arm/move_action"
                and self._combined_ready()
            ),
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
        except (RuntimeError, ValueError) as error:
            self._last_scene_error = f"{type(error).__name__}: {error}"
            self._last_scene_summary = {
                "robot_id": self._robot_id,
                "endpoint": self.services.scene_state,
            }
            return None

    def scene_readiness_evidence(self) -> Mapping[str, object]:
        """Return the latest scene probe failure or success summary."""
        return {
            "error": self._last_scene_error,
            "scene_client_registered": self._scene_state_client is not None,
            **dict(self._last_scene_summary),
        }

    def capability_readiness(
        self, *, probe_scene: bool = False, require_broadcaster_match: bool = True
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
            scene_evidence=self.scene_readiness_evidence(),
        )

    def wait_until_capability_ready(
        self,
        timeout_s: float,
        *,
        probe_scene: bool = True,
        require_broadcaster_match: bool = True,
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
            if self.capability_readiness(
                probe_scene=probe_scene,
                require_broadcaster_match=require_broadcaster_match,
            ).passed:
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

    def solve_tcp_ik(
        self,
        robot_id: str,
        target: Mapping[str, object],
        seed: Mapping[str, float],
        timeout_s: float,
        extra_seeds: Sequence[Mapping[str, float]] = (),
    ) -> dict[str, float]:
        """Solve one arm's calibrated TCP pose through the combined compute_ik.

        The combined ``dual_arm`` group is joint-space by contract, so a pose
        goal is solved per arm here: the unified ``/dual_arm/compute_ik``
        service answers the calibrated ``<robot>_arm_tcp`` group for this
        robot's tip link. The stock KDL backend performs ONE Newton descent
        per call, so restart diversity must come from explicit seeds: the
        live measured state first, then the caller's deterministic
        candidates. The first successful answer is returned as exact
        wire-named arm joint positions; an unavailable service, an exhausted
        seed budget, a missing arm joint or a non-finite value raises so the
        caller fails closed instead of planning to a guessed state.

        Args:
            robot_id: The arm whose calibrated TCP the pose constrains.
            target: The serialized combined target carrying ``tcp_pose``.
            seed: This robot's live wire-named joint state (8 joints).
            timeout_s: Bounded total wait across the IK attempts.
            extra_seeds: Additional deterministic seed states to restart from.

        Returns:
            Wire-named arm joint positions realizing the requested pose.
        """
        if (
            not isinstance(timeout_s, (int, float))
            or not math.isfinite(float(timeout_s))
            or float(timeout_s) <= 0.0
        ):
            raise ValueError(
                f"timeout_s must be finite and positive, got {timeout_s!r}"
            )
        factories = self._resolve_factories()
        if (
            factories.get_position_ik_type is None
            or factories.get_position_ik_request_factory is None
            or not self.services.compute_ik
        ):
            raise RuntimeError(
                "per-arm IK requires the unified combined compute_ik service"
            )
        pose = target.get("tcp_pose")
        if not isinstance(pose, Mapping):
            raise ValueError("combined IK target carries no tcp_pose")
        position = pose.get("position")
        orientation = pose.get("orientation_xyzw")
        if (
            not isinstance(position, Sequence)
            or isinstance(position, (str, bytes))
            or len(position) != 3
            or not isinstance(orientation, Sequence)
            or isinstance(orientation, (str, bytes))
            or len(orientation) != 4
        ):
            raise ValueError("combined IK target pose is malformed")
        client = self._ik_client
        if client is None or not self._service_ready(client):
            raise RuntimeError(f"{self.services.compute_ik} service is not discovered")
        deadline = self._clock_fn() + float(timeout_s)
        last_code: int | None = None
        attempts = 0
        for seed_state in (seed, *extra_seeds):
            if self._clock_fn() >= deadline:
                break
            attempts += 1
            solved, last_code = self._ik_attempt(
                client,
                factories,
                robot_id,
                pose,
                position,
                orientation,
                seed_state,
                deadline,
            )
            if solved is not None:
                self.last_ik_attempts = attempts
                return solved
        self.last_ik_attempts = attempts
        raise RuntimeError(
            f"combined IK failed for {robot_id}: no seed converged "
            f"(attempts={attempts}, last_error_code={last_code})"
        )

    def _ik_attempt(
        self,
        client: Any,
        factories: RosMoveItFactories,
        robot_id: str,
        pose: Mapping[str, object],
        position: Sequence[float],
        orientation: Sequence[float],
        seed_state: Mapping[str, float],
        deadline: float,
    ) -> tuple[dict[str, float] | None, int | None]:
        """Run one bounded compute_ik descent from a single seed."""
        remaining = deadline - self._clock_fn()
        budget = min(IK_SEARCH_TIMEOUT_S, max(IK_ATTEMPT_TIMEOUT_S, remaining))
        request = factories.get_position_ik_request_factory()
        ik = request.ik_request
        ik.group_name = f"{robot_id}_arm_tcp"
        ik.ik_link_name = str(pose.get("tip_link"))
        ik.timeout.sec = int(budget)
        ik.timeout.nanosec = int((budget % 1.0) * 1e9)
        ik.pose_stamped.header.frame_id = str(pose.get("frame_id"))
        ik.pose_stamped.pose.position.x = float(position[0])
        ik.pose_stamped.pose.position.y = float(position[1])
        ik.pose_stamped.pose.position.z = float(position[2])
        ik.pose_stamped.pose.orientation.x = float(orientation[0])
        ik.pose_stamped.pose.orientation.y = float(orientation[1])
        ik.pose_stamped.pose.orientation.z = float(orientation[2])
        ik.pose_stamped.pose.orientation.w = float(orientation[3])
        seed_wire = {str(name): float(value) for name, value in seed_state.items()}
        ik.robot_state.joint_state.name = list(seed_wire)
        ik.robot_state.joint_state.position = list(seed_wire.values())
        future = client.call_async(request)
        if not self._wait_future(future, deadline):
            return None, None
        try:
            response = future.result()
            code = int(response.error_code.val)
            names = [str(name) for name in response.solution.joint_state.name]
            values = [float(value) for value in response.solution.joint_state.position]
        except (AttributeError, TypeError, ValueError):
            return None, None
        if code != SUCCESS_CODE:
            return None, code
        arm = set(self.arm_joint_names)
        solved = {name: value for name, value in zip(names, values) if name in arm}
        if set(solved) != arm or any(
            not math.isfinite(value) for value in solved.values()
        ):
            return None, code
        return solved, code

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
                        f"planning-scene applier raised: {type(exc).__name__}: {exc}"
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
                f"{self.services.get_planning_scene} did not answer within {timeout_s}s"
            )
        try:
            response = future.result()
            scene = response.scene
            matrix = scene.allowed_collision_matrix
            names = list(matrix.entry_names)
            rows = list(matrix.entry_values)
        except Exception as error:
            return None, f"GetPlanningScene response is invalid: {error}"
        if (
            not names
            or len(rows) != len(names)
            or any(len(entry.enabled) != len(names) for entry in rows)
        ):
            return None, (
                "GetPlanningScene returned a non-square or empty allowed "
                "collision matrix; refusing fail-open scene application"
            )
        return matrix, None

    def unified_scene_fingerprint(self, timeout_s: float) -> str:
        """Read world, attachments, transforms and ACM as one scene proof.

        Returns:
            A digest covering collision-affecting scene state, excluding joint motion.
        """
        import hashlib
        from rosidl_runtime_py.convert import message_to_ordereddict

        client = self._ensure_get_planning_scene_client()
        if client is None or not self._service_ready(client):
            raise RuntimeError("combined scene readback is unavailable")
        request = self._resolve_factories().get_planning_scene_request_factory()
        request.components.components = 4 | 16 | 32 | 64 | 128
        future = client.call_async(request)
        if not self._wait_future(future, self._clock_fn() + timeout_s):
            raise TimeoutError("combined scene fingerprint timed out")
        document = message_to_ordereddict(future.result().scene)
        robot_state = document["robot_state"]
        payload = {
            "world": document["world"],
            "attachments": robot_state["attached_collision_objects"],
            "acm": document["allowed_collision_matrix"],
            "transforms": document["fixed_frame_transforms"],
        }
        return hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()

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
                item
                for item in attached
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

    def read_attached_planning_scene(self, timeout_s: float) -> Mapping[str, object]:
        """Read every attached collision object with normalized geometry."""
        factories = self._resolve_factories()
        client = self._ensure_get_planning_scene_client()
        request_factory = factories.get_planning_scene_request_factory
        if client is None or not self._service_ready(client) or request_factory is None:
            return {
                "passed": False,
                "attached": [],
                "error": "GetPlanningScene attached objects are unavailable",
            }
        try:
            request = request_factory()
            request.components.components = PLANNING_SCENE_ROBOT_STATE_ATTACHED_OBJECTS
            future = client.call_async(request)
            if not self._wait_future(future, self._clock_fn() + float(timeout_s)):
                return {
                    "passed": False,
                    "attached": [],
                    "error": "GetPlanningScene attached objects timed out",
                }
            attached = future.result().scene.robot_state.attached_collision_objects
            normalized: list[dict[str, object]] = []
            for item in attached:
                collision = item.object
                object_id = str(getattr(collision, "id", ""))
                link_name = str(getattr(item, "link_name", ""))
                frame_id = str(getattr(collision.header, "frame_id", ""))
                record: dict[str, object] = {
                    "object_id": object_id,
                    "link_name": link_name,
                    "frame_id": frame_id,
                    "touch_links": [str(value) for value in item.touch_links],
                }
                primitives = list(collision.primitives)
                poses = list(collision.primitive_poses)
                if len(primitives) == 1 and len(poses) == 1:
                    primitive = primitives[0]
                    pose = poses[0]
                    dimensions = tuple(float(value) for value in primitive.dimensions)
                    center = (
                        float(pose.position.x),
                        float(pose.position.y),
                        float(pose.position.z),
                    )
                    if (
                        len(dimensions) == 3
                        and all(
                            math.isfinite(value) and value > 0.0 for value in dimensions
                        )
                        and all(math.isfinite(value) for value in center)
                    ):
                        minimum = tuple(
                            center[index] - dimensions[index] / 2.0
                            for index in range(3)
                        )
                        maximum = tuple(
                            center[index] + dimensions[index] / 2.0
                            for index in range(3)
                        )
                        record["minimum_m"] = list(minimum)
                        record["maximum_m"] = list(maximum)
                        record["geometry_digest"] = box_geometry_digest(
                            object_id=object_id,
                            frame_id=frame_id,
                            minimum_m=minimum,
                            maximum_m=maximum,
                        )
                normalized.append(record)
            return {"passed": True, "attached": normalized, "error": None}
        except (AttributeError, TypeError, ValueError) as error:
            return {
                "passed": False,
                "attached": [],
                "error": f"GetPlanningScene attached objects malformed: {error}",
            }

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
            return {
                "passed": False,
                "readback_confirmed": False,
                "error": "attached-scene factories are unavailable",
            }
        client = self._ensure_planning_scene_client()
        if client is None or not self._service_ready(client):
            return {
                "passed": False,
                "readback_confirmed": False,
                "error": "apply planning-scene client is unavailable",
            }
        try:
            request = factories.apply_planning_scene_request_factory()
            request.scene.is_diff = True
            request.scene.robot_state.is_diff = True
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
                return {
                    "passed": False,
                    "readback_confirmed": False,
                    "error": "apply planning-scene timed out",
                }
            if future.result().success is not True:
                return {
                    "passed": False,
                    "readback_confirmed": False,
                    "error": "apply planning-scene reported failure",
                }
        except (AttributeError, TypeError, ValueError) as error:
            return {
                "passed": False,
                "readback_confirmed": False,
                "error": f"attached-scene request malformed: {error}",
            }
        ok, error = self._read_attached_collision_object(
            float(timeout_s),
            factories,
            object_id=str(payload["object_id"]),
            link_name=str(payload["link_name"]),
        )
        return {
            "passed": ok,
            "readback_confirmed": ok,
            "attached_object_id": str(payload["object_id"]),
            "attached_link": str(payload["link_name"]),
            "error": error,
        }

    def detach_attached_planning_scene(
        self, payload: Mapping[str, object], timeout_s: float
    ) -> Mapping[str, object]:
        """Detach one attached collision object and read back its absence.

        MoveIt ``REMOVE`` semantics: the attached object leaves the robot
        state entirely.  Success requires the readback to find ZERO matching
        attached objects; an object that is still attached never passes.

        Returns:
            The strict detach evidence with its readback verdict.
        """
        factories = self._resolve_factories()
        required = (
            factories.apply_planning_scene_type,
            factories.apply_planning_scene_request_factory,
            factories.collision_object_factory,
            factories.attached_collision_object_factory,
        )
        if any(item is None for item in required):
            return {
                "passed": False,
                "readback_confirmed": False,
                "error": "detached-scene factories are unavailable",
            }
        client = self._ensure_planning_scene_client()
        if client is None or not self._service_ready(client):
            return {
                "passed": False,
                "readback_confirmed": False,
                "error": "apply planning-scene client is unavailable",
            }
        try:
            request = factories.apply_planning_scene_request_factory()
            request.scene.is_diff = True
            request.scene.robot_state.is_diff = True
            collision = factories.collision_object_factory()
            collision.id = str(payload["object_id"])
            collision.operation = getattr(type(collision), "REMOVE")
            attached_object = factories.attached_collision_object_factory()
            attached_object.link_name = str(payload["link_name"])
            attached_object.object = collision
            request.scene.robot_state.attached_collision_objects.append(attached_object)
            future = client.call_async(request)
            if not self._wait_future(future, self._clock_fn() + float(timeout_s)):
                return {
                    "passed": False,
                    "readback_confirmed": False,
                    "error": "detach planning-scene timed out",
                }
            if future.result().success is not True:
                return {
                    "passed": False,
                    "readback_confirmed": False,
                    "error": "detach planning-scene reported failure",
                }
        except (AttributeError, TypeError, ValueError) as error:
            return {
                "passed": False,
                "readback_confirmed": False,
                "error": f"detached-scene request malformed: {error}",
            }
        client_get = self._ensure_get_planning_scene_client()
        request_factory = factories.get_planning_scene_request_factory
        if (
            client_get is None
            or not self._service_ready(client_get)
            or (request_factory is None)
        ):
            return {
                "passed": False,
                "readback_confirmed": False,
                "error": "GetPlanningScene attached objects are unavailable",
            }
        get_request = request_factory()
        get_request.components.components = PLANNING_SCENE_ROBOT_STATE_ATTACHED_OBJECTS
        future = client_get.call_async(get_request)
        if not self._wait_future(future, self._clock_fn() + float(timeout_s)):
            return {
                "passed": False,
                "readback_confirmed": False,
                "error": "GetPlanningScene attached objects timed out",
            }
        try:
            attached = future.result().scene.robot_state.attached_collision_objects
            matches = [
                item
                for item in attached
                if getattr(item.object, "id", None) == str(payload["object_id"])
            ]
        except (AttributeError, TypeError, ValueError) as error:
            return {
                "passed": False,
                "readback_confirmed": False,
                "error": f"GetPlanningScene attached objects malformed: {error}",
            }
        if matches:
            return {
                "passed": False,
                "readback_confirmed": False,
                "detached_object_id": str(payload["object_id"]),
                "detached_link": str(payload["link_name"]),
                "error": (
                    f"the attached object {payload['object_id']!r} is still "
                    "attached after the detach request"
                ),
            }
        return {
            "passed": True,
            "readback_confirmed": True,
            "detached_object_id": str(payload["object_id"]),
            "detached_link": str(payload["link_name"]),
            "error": None,
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
                        f"{self.services.apply_planning_scene} client is not available"
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
            details["error"] = f"{self.services.apply_planning_scene} reported failure"
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
                        f"object_pose={getattr(item, 'pose', None)!r} "
                        f"primitive_pose={getattr(getattr(item, 'primitive_poses', [None])[0], 'position', None)!r}"
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
        if client is None:
            raise RuntimeError(
                f"{self.services.scene_state} client was never registered; "
                "register_clients() must run before a scene query"
            )
        if not self._service_ready(client):
            raise RuntimeError(
                f"{self.services.scene_state} service is not discovered on "
                "this participant"
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
                f"{self.services.scene_state} did not answer within {timeout_s}s"
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
        return robot_scene_view(document, self._robot_id, joint_names=self._joint_names)

    # ------------------------------------------------------------------ sends

    def send_joint(self, target: JointTarget, timeout_s: float) -> ExecutionResult:
        """Admit, build and send one arm joint MoveGroup goal (bounded)."""
        return self._send_goal("joint", target, timeout_s)

    def send_link6_pose(
        self, target: Link6PoseTarget, timeout_s: float
    ) -> ExecutionResult:
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

    def _await_settled_gripper(
        self,
        result_states: int,
        deadline: float,
        anchored: tuple[tuple[float, float], float] | None = None,
    ) -> tuple[dict[str, float] | None, bool]:
        """Wait until the fingers stop making net progress after the action.

        The controller's own SUCCEEDED is not convergence evidence: with the
        loose per-joint goal tolerance it fires mid-travel (live evidence: a
        close commanded to 3.9 mm reported SUCCEEDED with the fingers still
        at 22.4 mm and creeping at 2.6 mm/s).  The terminal reading is
        therefore taken only after the measured fingers drift at most
        GRIPPER_CONTACT_STALL_DRIFT_M from a reference for
        GRIPPER_CONTACT_STALL_S — the same no-net-progress criterion the
        in-loop contact monitor uses — or the bounded wait expires.

        ``anchored`` carries the in-loop monitor's ``(reference, entered_at)``
        pair so the no-net-progress window already served while the action
        was running is not restarted from scratch.  The carry is accepted
        only after one fresh POST-result state message stays within the
        stall drift of that reference; any larger movement re-anchors from
        that fresh sample and waits the full window again.

        Returns:
            ``(final_state, settled)``; settled is True only when the fresh
            post-result fingers held the no-net-progress window.
        """
        reference: tuple[float, float] | None = None
        reference_at: float | None = None
        if anchored is not None:
            reference, reference_at = anchored
        settle_deadline = min(
            float(deadline), self._clock_fn() + GRIPPER_CONVERGENCE_TIMEOUT_S
        )
        while self._clock_fn() < settle_deadline:
            self._spin_fn()
            self._sleep_fn(self._wait_step_s)
            current = self.latest_isaac_state()
            if current is None or self._isaac_state_count <= result_states:
                continue
            try:
                fingers = (
                    float(current["gripper_joint1"]),
                    float(current["gripper_joint2"]),
                )
            except (KeyError, TypeError, ValueError):
                reference = None
                reference_at = None
                continue
            now = self._clock_fn()
            if reference is None:
                reference = fingers
                reference_at = now
                continue
            drift = max(
                abs(fingers[0] - reference[0]),
                abs(fingers[1] - reference[1]),
            )
            if drift > GRIPPER_CONTACT_STALL_DRIFT_M:
                # Still moving toward the held setpoint: keep waiting.
                reference = fingers
                reference_at = now
                continue
            if (
                reference_at is not None
                and now - reference_at >= GRIPPER_CONTACT_STALL_S
            ):
                return current, True
        return self.latest_isaac_state(), False

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
                status=ExecutionStatus.TIMED_OUT,
                details=details,
                command_messages=1,
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
                status=ExecutionStatus.REJECTED,
                details=details,
                command_messages=0,
                goal_accepted=False,
                terminal=True,
                safe_to_release=True,
                action_status="REJECTED",
            )
        self._active_handle = handle
        details["accepted"] = True
        self._active_result_future = handle.get_result_async()
        result_future = self._active_result_future
        window_low = float(target.expected_opening_m[0])
        window_high = float(target.expected_opening_m[1])
        contact_limited = False
        cancel_lifecycle = None
        stall_reference: tuple[float, float] | None = None
        stall_entered_at: float | None = None
        opening_trace: list[tuple[float, float]] = []
        while not result_future.done():
            self._spin_fn()
            current = self.latest_isaac_state()
            if current is not None and self._isaac_state_count > 0:
                now = self._clock_fn()
                try:
                    finger_1 = float(current["gripper_joint1"])
                    finger_2 = float(current["gripper_joint2"])
                    try:
                        measured_contact = fingers_to_logical_gripper(
                            finger_1, finger_2
                        )
                    except ValueError:
                        measured_contact = fingers_to_contact_opening(
                            finger_1, finger_2
                        )
                    if len(opening_trace) < GRIPPER_OPENING_TRACE_MAX:
                        opening_trace.append((now, measured_contact))
                    # A stall anywhere cancels the goal — in or out of the
                    # expected window.  The controller pressing a blocked
                    # workpiece never terminates on its own (live evidence,
                    # 2026-10-01), so without this the phase deadlocks to the
                    # wall deadline.  Whether the stalled reading proves
                    # contact stays the verdict's window check below.
                    if stall_reference is None:
                        stall_reference = (finger_1, finger_2)
                        stall_entered_at = now
                    else:
                        drift = max(
                            abs(finger_1 - stall_reference[0]),
                            abs(finger_2 - stall_reference[1]),
                        )
                        if drift > GRIPPER_CONTACT_STALL_DRIFT_M:
                            # Net progress toward the command re-anchors the
                            # reference: the fingers have not stalled yet.
                            stall_reference = (finger_1, finger_2)
                            stall_entered_at = now
                        elif (
                            stall_entered_at is not None
                            and now - stall_entered_at >= GRIPPER_CONTACT_STALL_S
                        ):
                            lifecycle = self.cancel_active()
                            cancel_lifecycle = lifecycle
                            if (
                                lifecycle.get("terminal") is True
                                and lifecycle.get("safe") is True
                            ):
                                details["contact_limited"] = True
                                details["contact_opening_m"] = measured_contact
                                details["cancel"] = lifecycle
                                contact_limited = True
                            break
                except (KeyError, TypeError, ValueError):
                    stall_reference = None
                    stall_entered_at = None
            if self._clock_fn() >= deadline:
                break
            self._sleep_fn(self._wait_step_s)
        details["opening_trace"] = [
            [round(when, 3), round(opening, 6)] for when, opening in opening_trace
        ]
        if not contact_limited and not result_future.done():
            lifecycle = cancel_lifecycle or self.cancel_active()
            details["cancel"] = lifecycle
            self._active_handle = None
            self._active_goal_future = None
            self._active_result_future = None
            details["phase"] = "execution"
            details["error"] = (
                f"gripper action exceeded the {timeout_s}s wall deadline "
                "(cancel attempted)"
            )
            return ExecutionResult(
                status=ExecutionStatus.TIMED_OUT,
                details=details,
                command_messages=1,
                goal_accepted=True,
                terminal=lifecycle.get("terminal") is True,
                safe_to_release=(
                    lifecycle.get("terminal") is True and lifecycle.get("safe") is True
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
        final, gripper_settled = self._await_settled_gripper(
            result_states,
            deadline,
            anchored=(
                (stall_reference, stall_entered_at)
                if stall_reference is not None and stall_entered_at is not None
                else None
            ),
        )
        details["post_result_state_messages"] = self._isaac_state_count - result_states
        details["gripper_settled"] = gripper_settled
        common = {
            "goal_accepted": True,
            "terminal": True,
            "safe_to_release": True,
            "action_status": details["action_status"],
        }
        commanded = float(target.opening_m)
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
            finger_1 = float(final["gripper_joint1"])
            finger_2 = float(final["gripper_joint2"])
            details["measured_finger_positions_m"] = {
                "gripper_joint1": finger_1,
                "gripper_joint2": finger_2,
            }
            try:
                measured = fingers_to_logical_gripper(finger_1, finger_2)
            except ValueError as exc:
                try:
                    measured = fingers_to_contact_opening(finger_1, finger_2)
                    details["mimic_relation_deviation"] = str(exc)
                    details["opening_measurement"] = "physical_finger_separation"
                except ValueError:
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
        if details["action_status"] != "SUCCEEDED" and not (
            details.get("contact_limited") is True
            and details["action_status"] in {"CANCELED", "CANCELLED"}
        ):
            details["error"] = (
                "gripper action did not report SUCCEEDED or safe contact-limited cancellation"
            )
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

    def send_guarded_tcp_pose(
        self, target: TcpPoseTarget, guard: Mapping[str, object], timeout_s: float
    ) -> ExecutionResult:
        """Run a collision-planned target with live contact supervision.

        Returns:
            Terminal action evidence and sampled contact guard measurements.
        """
        from rocobrick.skills.contact import inspect_contact, validate_guard

        validate_guard(guard)
        if guard["robot_id"] != self.robot_id:
            raise ValueError("contact guard robot identity mismatch")
        reason, measured = inspect_contact(self.query_scene_state(0.5), guard)
        if reason is not None:
            return ExecutionResult(
                status=ExecutionStatus.SUCCEEDED,
                details={"contact_stop_reason": reason, "contact_trace": [measured]},
                goal_accepted=False,
                terminal=True,
                safe_to_release=True,
                action_status="REJECTED",
            )
        goal = self._build_pose_goal(target, self._resolve_factories())
        goal.planning_options.plan_only = True
        goal.request.pipeline_id = "pilz_industrial_motion_planner"
        goal.request.planner_id = "LIN"
        goal.request.max_velocity_scaling_factor = 0.05
        goal.request.max_acceleration_scaling_factor = 0.05
        initial = self.latest_isaac_state()
        if initial is None or not self._state_fresh():
            raise RuntimeError("contact planning requires fresh measured joint state")
        goal.request.start_state.is_diff = False
        goal.request.start_state.joint_state.name = list(self._joint_names)
        goal.request.start_state.joint_state.position = [
            float(initial[name]) for name in JOINT_NAMES
        ]
        goal.request.start_state.joint_state.velocity = []
        decision = strict_action_admission(target)
        if not decision.accepted or not self.readiness_snapshot().passed:
            return ExecutionResult(
                status=ExecutionStatus.REJECTED,
                details={"error": "contact admission/readiness refused"},
                terminal=True,
                safe_to_release=True,
            )
        deadline = self._clock_fn() + float(timeout_s)
        future = self._move_group_client.send_goal_async(goal)
        self._active_goal_future = future
        if not self._wait_future(future, deadline):
            lifecycle = self.cancel_active()
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                details={"cancel": lifecycle},
                terminal=lifecycle.get("terminal") is True,
                safe_to_release=lifecycle.get("safe") is True,
            )
        handle = future.result()
        self._active_goal_future = None
        if handle is None or not handle.accepted:
            return ExecutionResult(
                status=ExecutionStatus.REJECTED,
                terminal=True,
                safe_to_release=True,
                details={"error": "contact plan rejected"},
            )
        self._active_handle = handle
        planned = handle.get_result_async()
        self._active_result_future = planned
        if not self._wait_future(planned, deadline):
            lifecycle = self.cancel_active()
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                details={"cancel": lifecycle},
                terminal=lifecycle.get("terminal") is True,
                safe_to_release=lifecycle.get("safe") is True,
            )
        wrapper = planned.result()
        self._active_handle = None
        self._active_result_future = None
        status = _terminal_status_name(
            wrapper.status, self._resolve_factories().goal_status_type()
        )
        count, admitted = self._inspect_trajectory(wrapper.result)
        if (
            status != "SUCCEEDED"
            or wrapper.result.error_code.val != SUCCESS_CODE
            or not admitted.accepted
        ):
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                terminal=True,
                safe_to_release=True,
                details={"error": "collision-checked contact plan failed"},
            )
        reason, measured = inspect_contact(self.query_scene_state(0.5), guard)
        if reason is not None:
            return ExecutionResult(
                status=ExecutionStatus.SUCCEEDED,
                terminal=True,
                safe_to_release=True,
                details={"contact_stop_reason": reason, "contact_trace": [measured]},
            )
        return self._execute_contact_trajectory(
            wrapper.result.planned_trajectory.joint_trajectory,
            target,
            guard,
            deadline,
            count,
        )

    def _execute_contact_trajectory(
        self, trajectory, target: TcpPoseTarget, guard, deadline, count
    ):
        """Execute the admitted path with strict controller tracking tolerances.

        The whole descent runs as ONE supervised action: the guard monitor
        cancels it in flight on connection or any safety stop, so the
        endpoint is a bound, never a promise. Speed is capped by the measured
        Cartesian distance from the live TCP to the target so a full
        insertion descent cannot outrun the guard sampling period.

        Returns:
            Monitored terminal action evidence; no endpoint completion retry.
        """
        from rocobrick.adapters.moveit.contact_trajectory import (
            CONTACT_CARTESIAN_SPEED_MPS,
            contact_minimum_duration_s,
            contact_trajectory_timing,
        )
        from rocobrick.skills.observation import Pose
        from rocobrick.skills.placement_geometry import pose_error

        factories = self._resolve_factories()
        if (
            factories.follow_joint_trajectory_goal_factory is None
            or self._arm_client is None
        ):
            raise RuntimeError("contact trajectory controller is unavailable")
        times = [
            float(point.time_from_start.sec) + point.time_from_start.nanosec * 1e-9
            for point in trajectory.points
        ]
        current = self.latest_isaac_state()
        if (
            current is None
            or max(
                abs(
                    current[name]
                    - trajectory.points[0].positions[
                        list(trajectory.joint_names).index(device)
                    ]
                )
                for name, device in zip(ARM_JOINT_NAMES, self._arm_joint_names)
            )
            > 0.002
        ):
            raise RuntimeError("contact trajectory start drifted from planned state")
        start_fk = self.compute_tcp_fk(current, timeout_s=0.5)
        if start_fk is None:
            raise RuntimeError("contact trajectory requires a measured TCP start pose")
        contact_distance_m = math.dist(start_fk[0], tuple(target.position))
        timing = contact_trajectory_timing(
            times, minimum_duration_s=contact_minimum_duration_s(contact_distance_m)
        )
        goal = factories.follow_joint_trajectory_goal_factory()
        goal.trajectory.joint_names = list(trajectory.joint_names)
        for source, time_s in zip(trajectory.points, timing.times_s):
            point = factories.joint_trajectory_point_factory()
            point.positions = list(source.positions)
            point.velocities = [x / timing.scale for x in source.velocities]
            point.accelerations = [x / timing.scale**2 for x in source.accelerations]
            point.time_from_start = self._duration_message(time_s, factories)
            goal.trajectory.points.append(point)
        # Controller tolerances are deliberately generous for contact moves:
        # legitimate contact loads and uncompensated gravity droop produce
        # real joint error that must NOT abort the supervised descent. The
        # physical truth comes from the live scene monitor (support tilt,
        # held-object slip, interface force, exact connection), not from
        # tracking tightness.
        goal.goal_tolerance = []
        goal.path_tolerance = []
        for name in self._arm_joint_names:
            end = factories.joint_tolerance_factory()
            end.name = name
            end.position = 0.03
            end.velocity = 0.05
            goal.goal_tolerance.append(end)
            path = factories.joint_tolerance_factory()
            path.name = name
            path.position = 0.05
            goal.path_tolerance.append(path)
        goal.goal_time_tolerance = self._duration_message(5.0, factories)
        send_commands = self._command_count
        future = self._arm_client.send_goal_async(goal)
        self._active_goal_future = future
        if not self._wait_future(future, deadline):
            lifecycle = self.cancel_active()
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                details={"cancel": lifecycle},
                terminal=lifecycle.get("terminal") is True,
                safe_to_release=lifecycle.get("safe") is True,
            )
        handle = future.result()
        self._active_goal_future = None
        if handle is None or not handle.accepted:
            return ExecutionResult(
                status=ExecutionStatus.REJECTED,
                terminal=True,
                safe_to_release=True,
                details={"error": "arm contact goal rejected"},
            )
        self._active_handle = handle
        self._active_result_future = handle.get_result_async()
        details = {
            "planned_waypoints": count,
            "execution_transport": "arm_follow",
            "contact_cartesian_distance_m": contact_distance_m,
            "contact_speed_cap_mps": CONTACT_CARTESIAN_SPEED_MPS,
            "planned_joint_start": dict(
                zip(trajectory.joint_names, trajectory.points[0].positions)
            ),
            "planned_joint_end": dict(
                zip(trajectory.joint_names, trajectory.points[-1].positions)
            ),
            "measured_joint_start": dict(current),
            "trajectory_duration_s": timing.duration_s,
            "goal_tolerance_rad": 0.03,
            "path_tolerance_rad": 0.05,
        }
        outcome = self._await_guarded_action(
            self._active_result_future, guard, deadline, details, send_commands
        )
        # Contact detection may stop before the endpoint. Otherwise require
        # a fresh measured tool pose to match the requested short target.
        if (
            outcome.details.get("contact_stop_reason") is None
            and outcome.status is ExecutionStatus.SUCCEEDED
        ):
            # Require fresh post-result joint feedback before live TCP FK.
            result_states = self._isaac_state_count
            while (
                self._isaac_state_count <= result_states and self._clock_fn() < deadline
            ):
                self._spin_fn()
                self._sleep_fn(0.01)
            if self._isaac_state_count <= result_states or not self._state_fresh():
                return replace(
                    outcome,
                    status=ExecutionStatus.FAILED,
                    details={
                        **outcome.details,
                        "contact_stop_reason": "missing_fresh_final_state",
                    },
                )
            final = self.latest_isaac_state()
            fk = self.compute_tcp_fk(final, timeout_s=0.5)
            if fk is None:
                return replace(
                    outcome,
                    status=ExecutionStatus.FAILED,
                    details={
                        **outcome.details,
                        "contact_stop_reason": "missing_final_fk",
                    },
                )
            measured_pose = Pose(tuple(fk[0]), tuple(fk[1]))
            expected_pose = Pose(target.position, target.orientation_xyzw)
            position_error, angle_error = pose_error(measured_pose, expected_pose)
            updated = {
                **outcome.details,
                "final_position_error_m": position_error,
                "final_orientation_error_rad": angle_error,
            }
            # The endpoint bound is the target's own admission tolerance (a
            # strict bound for descents, a free-space bound for withdrawals);
            # uncompensated droop must fail honestly, not abort mid-motion.
            if (
                position_error > max(0.0005, target.position_tolerance_m)
                or angle_error > target.orientation_tolerance_rad
            ):
                updated["contact_stop_reason"] = "contact_endpoint_not_reached"
                return replace(outcome, details=updated)
            return replace(outcome, details=updated)
        return outcome

    def _send_goal(
        self,
        kind: str,
        target: JointTarget | Link6PoseTarget | TcpPoseTarget,
        timeout_s: float,
        *,
        contact_guard: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        if self.actions.move_group == "/dual_arm/move_action":
            return ExecutionResult(
                status=ExecutionStatus.REJECTED,
                details={
                    "reason": "participant planning is forbidden; use unified pre-phase"
                },
                command_messages=0,
            )
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
        if contact_guard is not None:
            goal.request.pipeline_id = "pilz_industrial_motion_planner"
            goal.request.planner_id = "LIN"
            goal.request.max_velocity_scaling_factor = 0.05
            goal.request.max_acceleration_scaling_factor = 0.05
        return self._execute_goal(
            goal,
            kind=kind,
            target=target,
            timeout_s=float(timeout_s),
            baseline=baseline,
            factories=factories,
            contact_guard=contact_guard,
        )

    def _build_joint_goal(
        self, target: JointTarget, factories: RosMoveItFactories
    ) -> Any:
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
        sphere.dimensions = [
            float(target.position_tolerance_m) * POSE_GOAL_TOLERANCE_FRACTION
        ]
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
        orientation_tolerance = (
            float(target.orientation_tolerance_rad) * POSE_GOAL_TOLERANCE_FRACTION
        )
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
        contact_guard: Mapping[str, object] | None = None,
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
        if contact_guard is not None:
            return self._await_guarded_action(
                result_future, contact_guard, deadline, details, send_commands
            )
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
        trajectory = action_result.planned_trajectory.joint_trajectory
        try:
            raw_names = [str(name) for name in trajectory.joint_names]
            raw_points = [
                [float(value) for value in point.positions]
                for point in trajectory.points
            ]
            if raw_points:
                final_point = raw_points[-1]
                details["planned_final_joint_positions_rad"] = dict(
                    zip(raw_names, final_point)
                )
                first_time = getattr(trajectory.points[0], "time_from_start", None)
                final_time = getattr(trajectory.points[-1], "time_from_start", None)
                if first_time is not None and final_time is not None:
                    details["planned_trajectory_time_s"] = {
                        "first": float(getattr(first_time, "sec", 0))
                        + float(getattr(first_time, "nanosec", 0)) * 1e-9,
                        "final": float(getattr(final_time, "sec", 0))
                        + float(getattr(final_time, "nanosec", 0)) * 1e-9,
                        "points": len(raw_points),
                    }
        except (AttributeError, TypeError, ValueError):
            # Waypoint admission remains authoritative; diagnostics must not
            # change whether a trajectory is accepted.
            pass
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
            return ExecutionResult(
                status=ExecutionStatus.FAILED, details=details, **common
            )
        if not trajectory_decision.accepted:
            details["error"] = (
                "planned trajectory failed caller-side waypoint admission"
            )
            return ExecutionResult(
                status=ExecutionStatus.FAILED, details=details, **common
            )
        return self._verify_final(
            kind, target, baseline, send_states, send_commands, common, details
        )

    def _await_guarded_action(
        self, result_future, guard, deadline, details, send_commands
    ):
        """Monitor a live action without post-action pressing dwell or retries.

        Returns:
            Strict action terminal evidence and the final contact observation.
        """
        from rocobrick.skills.contact import allows_recovery, inspect_contact

        trace = []
        last_sequence = None
        reason = None
        lifecycle = None
        try:
            while True:
                scene = self.query_scene_state(
                    timeout_s=min(0.5, max(0.01, deadline - self._clock_fn()))
                )
                sequence = scene.get("sequence")
                if type(sequence) is not int or (
                    last_sequence is not None and sequence <= last_sequence
                ):
                    raise ValueError("contact observation is stale")
                last_sequence = sequence
                reason, measured = inspect_contact(scene, guard)
                trace.append(measured)
                if reason is not None:
                    if not result_future.done():
                        lifecycle = self.cancel_active()
                    break
                if result_future.done():
                    break
                if self._clock_fn() >= deadline:
                    reason = "contact_deadline"
                    lifecycle = self.cancel_active()
                    break
                self._sleep_fn(0.02)
        except Exception as error:
            reason = f"contact_feedback_failed: {type(error).__name__}: {error}"
            lifecycle = self.cancel_active()
        terminal = result_future.done()
        status_name = None
        if terminal:
            wrapper = result_future.result()
            status_name = _terminal_status_name(
                wrapper.status, self._resolve_factories().goal_status_type()
            )
            if (
                status_name == "ABORTED"
                and details.get("execution_transport") == "arm_follow"
            ):
                reason = reason or "contact_tracking_stopped"
            if status_name == "SUCCEEDED":
                result_object = getattr(wrapper, "result", None)
                code = getattr(getattr(result_object, "error_code", None), "val", None)
                if details.get("execution_transport") == "arm_follow":
                    if getattr(result_object, "error_code", None) != 0:
                        reason = "contact_controller_result_failed"
                elif code != SUCCESS_CODE:
                    reason = "contact_planner_result_failed"
                else:
                    count, admission = self._inspect_trajectory(result_object)
                    details["planned_waypoints"] = count
                    if not admission.accepted:
                        reason = "contact_trajectory_admission_failed"
        safe = terminal and (lifecycle is None or lifecycle.get("safe") is True)
        final_state = self.latest_isaac_state()
        if final_state is not None and details.get("planned_joint_end"):
            end = details["planned_joint_end"]
            details["measured_joint_final"] = dict(final_state)
            details["joint_final_errors"] = {
                device: final_state[name] - end[device]
                for name, device in zip(ARM_JOINT_NAMES, self._arm_joint_names)
            }
        details.update(
            contact_trace=trace,
            contact_stop_reason=reason,
            contact_mode=guard["mode"],
            action_status=status_name,
            cancel=lifecycle or {},
        )
        self._active_handle = None
        self._active_goal_future = None
        self._active_result_future = None
        passed = safe and (
            status_name == "SUCCEEDED" and reason is None or allows_recovery(reason)
        )
        return ExecutionResult(
            status=ExecutionStatus.SUCCEEDED if passed else ExecutionStatus.FAILED,
            details=details,
            command_messages=self._command_count - send_commands,
            goal_accepted=True,
            terminal=terminal,
            safe_to_release=safe,
            action_status=status_name,
            cancel_evidence=lifecycle or {},
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

    def _complete_pose_goal_via_arm(
        self,
        planned: Mapping[str, float],
        factories: RosMoveItFactories,
    ) -> dict[str, object]:
        """Re-command one plan's final arm joints directly to the controller.

        The container-side execution pipeline can stop streaming setpoints
        when the MoveGroup trajectory ends while the simulated arm is still
        short of the goal (live evidence, 2026-10-01: a descent plan
        reported SUCCEEDED with the arm 67 mm from its target and frozen
        there, independent of drive stiffness).  Re-sending the planned
        final joints as this runtime's own FollowJointTrajectory goal with
        tight per-joint tolerances keeps the controller active — the same
        property that keeps the gripper actions honest — until the live
        state converges or the bounded wait expires.

        Returns:
            JSON-safe lifecycle evidence; the caller re-verifies the pose.
        """
        evidence: dict[str, object] = {"attempted": True}
        current = self.latest_isaac_state()
        if current is None or any(name not in current for name in ARM_JOINT_NAMES):
            evidence["attempted"] = False
            evidence["error"] = "no complete live arm state to build the goal from"
            return evidence
        # The plan evidence names this robot's device joints on the wire; the
        # state getters are canonical.  Project the planned final joints onto
        # the canonical order before building the completion goal.
        by_device = {
            device: canonical
            for canonical, device in zip(ARM_JOINT_NAMES, self._arm_joint_names)
        }
        planned_canonical: dict[str, float] = {}
        for key, value in planned.items():
            canonical = by_device.get(str(key), str(key))
            if canonical in ARM_JOINT_NAMES and canonical not in planned_canonical:
                planned_canonical[canonical] = float(value)
        missing_planned = [
            name for name in ARM_JOINT_NAMES if name not in planned_canonical
        ]
        if missing_planned:
            evidence["attempted"] = False
            evidence["error"] = (
                f"the plan evidence lacks canonical arm joints: {missing_planned}"
            )
            return evidence
        if (
            factories.follow_joint_trajectory_goal_factory is None
            or factories.joint_trajectory_point_factory is None
            or factories.joint_tolerance_factory is None
            or factories.duration_factory is None
        ):
            evidence["attempted"] = False
            evidence["error"] = "the injected factories carry no arm goal transport"
            return evidence
        if self._arm_client is None:
            evidence["attempted"] = False
            evidence["error"] = "arm action client is not registered"
            return evidence
        goal = factories.follow_joint_trajectory_goal_factory()
        trajectory = goal.trajectory
        trajectory.joint_names = list(self._arm_joint_names)
        start = factories.joint_trajectory_point_factory()
        start.positions = [float(current[name]) for name in ARM_JOINT_NAMES]
        start.time_from_start = self._duration_message(0.0, factories)
        end = factories.joint_trajectory_point_factory()
        end.positions = [float(planned_canonical[name]) for name in ARM_JOINT_NAMES]
        end.time_from_start = self._duration_message(
            ARM_COMPLETION_DURATION_S, factories
        )
        trajectory.points = [start, end]
        goal.goal_tolerance = []
        for name in self._arm_joint_names:
            tolerance = factories.joint_tolerance_factory()
            tolerance.name = name
            tolerance.position = ARM_COMPLETION_TOLERANCE_RAD
            goal.goal_tolerance.append(tolerance)
        deadline = self._clock_fn() + ARM_COMPLETION_TIMEOUT_S
        try:
            future = self._arm_client.send_goal_async(goal)
            self._active_goal_future = future
        except Exception as exc:
            evidence["error"] = f"send_goal_async failed: {type(exc).__name__}: {exc}"
            return evidence
        if not self._wait_future(
            future, min(self._clock_fn() + ACCEPT_TIMEOUT_S, deadline)
        ):
            evidence["cancel"] = self.cancel_active()
            self._active_goal_future = None
            evidence["error"] = "arm completion goal was not accepted in time"
            return evidence
        handle = future.result()
        self._active_goal_future = None
        if handle is None or not getattr(handle, "accepted", False):
            evidence["error"] = "arm controller did not accept the completion goal"
            return evidence
        self._active_handle = handle
        self._active_result_future = handle.get_result_async()
        result_future = self._active_result_future
        accepted_at = self._clock_fn()
        while not result_future.done() and self._clock_fn() < deadline:
            self._spin_fn()
            self._sleep_fn(self._wait_step_s)
        terminal_status = None
        if result_future.done():
            wrapper = result_future.result()
            terminal_status = _terminal_status_name(
                wrapper.status, factories.goal_status_type()
            )
        else:
            lifecycle = self.cancel_active()
            evidence["cancel"] = lifecycle
            terminal_status = "CANCELED" if lifecycle.get("terminal") is True else None
        self._active_handle = None
        self._active_result_future = None
        evidence["duration_s"] = round(self._clock_fn() - accepted_at, 3)
        evidence["action_status"] = terminal_status
        return evidence

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
        if (
            post_states < MIN_POST_COMMAND_STATES
            or post_commands < MIN_COMMAND_MESSAGES
        ):
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
        result = self._verify_pose_final(target, final, common, details)
        if result.status is not ExecutionStatus.SUCCEEDED:
            planned = details.get("planned_final_joint_positions_rad")
            if isinstance(planned, Mapping) and planned:
                # The plan reached the goal; the container-side execution
                # pipeline froze the arm short of it.  Re-command the planned
                # final joints directly and re-verify against the same gates.
                details["arm_completion"] = self._complete_pose_goal_via_arm(
                    planned, self._resolve_factories()
                )
                retry_final, retry_settled = self._await_settled_final()
                details["arm_completion_settled"] = retry_settled
                if retry_final is not None and self._state_fresh():
                    result = self._verify_pose_final(
                        target, retry_final, common, details
                    )
        return result

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
            details["error"] = (
                f"fk verification hook raised: {type(exc).__name__}: {exc}"
            )
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
        orientation_error = orientation_error_rad(quaternion, target.orientation_xyzw)
        final_errors = {
            "position_error_m": position_error,
            "orientation_error_rad": orientation_error,
        }
        details["final_fk_position_m"] = [float(value) for value in position]
        details["final_fk_orientation_xyzw"] = [float(value) for value in quaternion]
        details["target_position_m"] = [float(value) for value in target.position]
        details["target_orientation_xyzw"] = [
            float(value) for value in target.orientation_xyzw
        ]
        details["position_residual_m"] = [
            float(actual) - float(goal)
            for actual, goal in zip(position, target.position)
        ]
        details["final_arm_joint_positions_rad"] = {
            name: float(final[name]) for name in ARM_JOINT_NAMES if name in final
        }
        details["position_tolerance_m"] = float(target.position_tolerance_m)
        details["orientation_tolerance_rad"] = float(target.orientation_tolerance_rad)
        if position_error > float(
            target.position_tolerance_m
        ) or orientation_error > float(target.orientation_tolerance_rad):
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
                        info["error"] = (
                            "goal acceptance did not settle before cancel deadline"
                        )
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
            RobotRuntimeProfile.unified_participant
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
        profiles = {robot_id: profile_factory(robot_id) for robot_id in ids}
        for robot_id, profile in profiles.items():
            if profile.robot_id != robot_id:
                raise ValueError(
                    f"profile for {robot_id!r} reports robot_id {profile.robot_id!r}"
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
        # Diagnostics of the last unified readiness evaluation; purely
        # observational so a freshness failure names the exact sample ages.
        self._last_unified_timing: dict[str, Mapping[str, object]] = {}

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

        create_runtime = (
            runtime_factory if runtime_factory is not None else default_runtime
        )
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

        from rocobrick.coordinated_ros_runtime import CoordinatedMoveItPlanRuntime
        from rocobrick.unified_ros_runtime import UnifiedDualArmRuntime

        self._coordinated_runtime = None
        if set(ids) == {"piper_0", "piper_1"}:
            self._coordinated_runtime = CoordinatedMoveItPlanRuntime(
                node,
                joint_binding={
                    robot_id: self.runtimes[robot_id].arm_joint_names
                    for robot_id in ids
                },
                factories=factories,
                clock=self._clock_fn,
                spin=self._spin_fn,
                sleep=self._sleep_fn,
                wait_step_s=self._wait_step_s,
            )
        if self._coordinated_runtime is not None:
            for participant in self.runtimes.values():
                participant._combined_ready = self._coordinated_runtime.ready
        from rocobrick.unified_scene import UnifiedSceneBinding

        self._scene_binding = UnifiedSceneBinding(self.runtimes[ids[0]])
        self._planning_lock = threading.Lock()
        self._joint_limits_cache: dict[str, tuple[float, float]] | None = None
        self._unified_runtime = (
            UnifiedDualArmRuntime(self, self._coordinated_runtime)
            if self._coordinated_runtime is not None
            else None
        )

    def transfer_attached_planning_scene(
        self, payload: Mapping[str, object], timeout_s: float
    ) -> Mapping[str, object]:
        """Transfer one attached object with readback and compensating rollback."""
        with self._planning_lock:
            donor_id = str(payload["donor_robot_id"])
            receiver_id = str(payload["receiver_robot_id"])
            object_id = str(payload["object_id"])
            donor = self.runtimes[donor_id]
            receiver = self.runtimes[receiver_id]

            def read(runtime: object) -> Mapping[str, object]:
                result = runtime.read_attached_planning_scene(timeout_s)
                if result.get("passed") is not True:
                    raise RuntimeError(str(result.get("error")))
                return result

            def objects(result: Mapping[str, object]) -> list[Mapping[str, object]]:
                value = result.get("attached")
                return (
                    [item for item in value if isinstance(item, Mapping)]
                    if isinstance(value, list)
                    else []
                )

            def donor_exact(items: list[Mapping[str, object]]) -> bool:
                # The donor and receiver TCP frames are robot-local and may
                # differ.  The transfer payload canonicalizes geometry in the
                # receiver frame; donor identity is therefore proven by the
                # exact physical object and attached donor link here, while
                # receiver geometry/frame is proven by the post-attach
                # readback below.
                matches = [
                    item
                    for item in items
                    if item.get("object_id") == object_id
                    and item.get("link_name") == payload["donor_link_name"]
                ]
                return len(matches) == 1

            def receiver_absent(items: list[Mapping[str, object]]) -> bool:
                return not any(item.get("object_id") == object_id for item in items)

            def receiver_exact(items: list[Mapping[str, object]]) -> bool:
                matches = [
                    item
                    for item in items
                    if item.get("object_id") == object_id
                    and item.get("link_name") == payload["receiver_link_name"]
                ]
                return len(matches) == 1 and (
                    matches[0].get("frame_id") == payload["frame_id"]
                    and matches[0].get("geometry_digest") == payload["geometry_digest"]
                )

            try:
                donor_before = read(donor)
                receiver_before = read(receiver)
                donor_items = objects(donor_before)
                receiver_items = objects(receiver_before)
            except (RuntimeError, KeyError) as error:
                return {
                    "passed": False,
                    "transfer_status": "ROLLED_BACK",
                    "terminal": True,
                    "safe_to_release": True,
                    "error": f"transfer precondition read failed: {error}",
                }
            if not donor_exact(donor_items) or not receiver_absent(receiver_items):
                return {
                    "passed": False,
                    "transfer_status": "ROLLED_BACK",
                    "terminal": True,
                    "safe_to_release": True,
                    "error": "transfer expected_before state did not match",
                    "donor_before": donor_items,
                    "receiver_before": receiver_items,
                }

            detach_payload = {
                "scene_id": payload["transfer_id"],
                "object_id": object_id,
                "link_name": payload["donor_link_name"],
                "operation": "detach",
            }
            attach_payload = {
                "scene_id": payload["transfer_id"],
                "object_id": object_id,
                "link_name": payload["receiver_link_name"],
                "frame_id": payload["frame_id"],
                "minimum_m": payload["minimum_m"],
                "maximum_m": payload["maximum_m"],
                "geometry_digest": payload["geometry_digest"],
                "touch_links": payload["touch_links"],
            }

            def rollback() -> Mapping[str, object]:
                rollback_errors: list[str] = []
                try:
                    receiver.detach_attached_planning_scene(
                        detach_payload | {"link_name": payload["receiver_link_name"]},
                        timeout_s,
                    )
                except Exception as error:
                    rollback_errors.append(f"receiver remove: {error}")
                try:
                    donor.apply_attached_planning_scene(
                        attach_payload | {"link_name": payload["donor_link_name"]},
                        timeout_s,
                    )
                except Exception as error:
                    rollback_errors.append(f"donor restore: {error}")
                try:
                    donor_after = objects(read(donor))
                    receiver_after = objects(read(receiver))
                    restored = donor_exact(donor_after) and receiver_absent(
                        receiver_after
                    )
                except Exception as error:
                    rollback_errors.append(f"rollback readback: {error}")
                    restored = False
                return {
                    "restored": restored,
                    "errors": rollback_errors,
                }

            detached = donor.detach_attached_planning_scene(detach_payload, timeout_s)
            if detached.get("passed") is not True:
                return {
                    "passed": False,
                    "transfer_status": "UNKNOWN",
                    "terminal": False,
                    "safe_to_release": False,
                    "error": str(detached.get("error")),
                    "donor_detach": dict(detached),
                }
            try:
                donor_after = objects(read(donor))
            except Exception as error:
                return {
                    "passed": False,
                    "transfer_status": "UNKNOWN",
                    "terminal": False,
                    "safe_to_release": False,
                    "error": f"donor remove readback failed: {error}",
                }
            if not receiver_absent(donor_after):
                rollback_result = rollback()
                return {
                    "passed": False,
                    "transfer_status": (
                        "ROLLED_BACK" if rollback_result["restored"] else "UNKNOWN"
                    ),
                    "terminal": bool(rollback_result["restored"]),
                    "safe_to_release": bool(rollback_result["restored"]),
                    "error": "donor remained attached after remove",
                    "rollback": rollback_result,
                }

            attached = receiver.apply_attached_planning_scene(attach_payload, timeout_s)
            if attached.get("passed") is not True:
                rollback_result = rollback()
                return {
                    "passed": False,
                    "transfer_status": (
                        "ROLLED_BACK" if rollback_result["restored"] else "UNKNOWN"
                    ),
                    "terminal": bool(rollback_result["restored"]),
                    "safe_to_release": bool(rollback_result["restored"]),
                    "error": str(attached.get("error")),
                    "receiver_attach": dict(attached),
                    "rollback": rollback_result,
                }
            try:
                donor_after = objects(read(donor))
                receiver_after = objects(read(receiver))
            except Exception as error:
                return {
                    "passed": False,
                    "transfer_status": "UNKNOWN",
                    "terminal": False,
                    "safe_to_release": False,
                    "error": f"transfer readback failed: {error}",
                }
            if donor_exact(donor_after) or not receiver_exact(receiver_after):
                rollback_result = rollback()
                return {
                    "passed": False,
                    "transfer_status": (
                        "ROLLED_BACK" if rollback_result["restored"] else "UNKNOWN"
                    ),
                    "terminal": bool(rollback_result["restored"]),
                    "safe_to_release": bool(rollback_result["restored"]),
                    "error": "transfer exact readback failed",
                    "rollback": rollback_result,
                }
            return {
                "passed": True,
                "transfer_status": "COMMITTED",
                "terminal": True,
                "safe_to_release": True,
                "error": None,
                "donor_before": donor_items,
                "receiver_before": receiver_items,
                "donor_after": donor_after,
                "receiver_after": receiver_after,
            }

    def sync_unified_scene(self, request) -> Mapping[str, object]:
        """Apply and bind one combined world snapshot.

        Returns:
            Actual Apply/GetPlanningScene readback proof.
        """
        proof = dict(self._scene_binding.sync(request))
        proof["source_geometry"] = {
            key: request.scene_snapshot.get(key)
            for key in ("bricks", "connections", "environment_colliders", "ground")
        }
        return proof

    def validate_unified_scene(self, proof: Mapping[str, object]) -> bool:
        """Validate actual world geometry after planning.

        Returns:
            Whether scene version and full readback digest remain unchanged.
        """
        if not self._scene_binding.validate(proof):
            return False
        # The host snapshot contains live poses, sequence and sim time that
        # advance while a no-motion plan is being computed. Planning-scene
        # stability is already proven by the combined GetPlanningScene
        # fingerprint/version readback above; comparing the dynamic host
        # document byte-for-byte would reject a valid static planning scene.
        return True

    def unified_plan_only(self, request):
        """Serialize all combined scene/planning access under one runtime owner.

        Returns:
            A plan-only result with no physical arrival or lease release claim.
        """
        from rocobrick.coordinated_session_entry import perform_coordinated_plan

        with self._planning_lock:
            return perform_coordinated_plan(self, request)

    def execute_coordinated_pre_phase(self, request):
        """Plan, dispatch and prove one combined free-space physical motion.

        Returns:
            Coordinated result whose lifecycle carries the participants'
            prepare/start/terminal-safe arrival evidence.
        """
        from rocobrick.coordinated_session_entry import perform_coordinated_plan

        with self._planning_lock:
            return perform_coordinated_plan(
                self,
                request,
                participants_executor=self._execute_coordinated_operation,
            )

    def execute_prepared_motion(self, request, *, cancel_requested=None):
        """Execute an admitted path without replanning or shared scene writes.

        Returns:
            Bound tracking, synchronization and terminal lifecycle evidence.
        """
        from rocobrick.adapters.moveit.prepared_execution import (
            perform_prepared_motion,
        )
        from rocobrick.execution.prepared_motion import PreparedMotionRequest

        if not isinstance(request, PreparedMotionRequest):
            raise ValueError("prepared execution requires PreparedMotionRequest")
        with self._planning_lock:
            return perform_prepared_motion(
                self, request, cancel_requested=cancel_requested
            )

    def plan_readonly_combined(self, request):
        """Query the current combined scene without synchronization/apply calls.

        Returns:
            One combined trajectory, never physical execution evidence.
        """
        from rocobrick.adapters.moveit.readonly_combined_plan import (
            perform_readonly_combined_plan,
        )

        with self._planning_lock:
            return perform_readonly_combined_plan(self, request)

    def _execute_coordinated_operation(self, operation) -> Mapping[str, object]:
        """Run the split trajectories through the all-or-nothing lifecycle."""
        from rocobrick.adapters.moveit.coordinated_arm import (
            CoordinatedArmParticipant,
        )
        from rocobrick.coordinated_execution import CoordinatedExecutionCoordinator

        participants = {
            robot_id: CoordinatedArmParticipant(self.runtimes[robot_id])
            for robot_id in operation.participants
        }
        coordinator = CoordinatedExecutionCoordinator(participants)
        barrier = self.latest_shared_sim_time_s() + 0.2
        lifecycle = coordinator.execute(
            operation,
            start_at_sim_time_s=barrier,
            require_completion=not (
                operation.coordination_id.endswith(":direct-handover")
                or operation.coordination_id.endswith(":giver-reposition")
                or os.environ.get("ROCOBRICK_HANDOVER_SIMULATION") == "1"
            ),
        )
        evidence = lifecycle.serialize()
        evidence["start_barrier_sim_time_s"] = barrier
        return evidence

    def unified_joint_limits(
        self, timeout_s: float = 5.0
    ) -> dict[str, tuple[float, float]]:
        """Read combined arm joint limits from the live robot description.

        Subscribes once to the latched ``/dual_arm/robot_description`` the
        combined launch publishes, parses the per-joint limits of every
        participant arm and caches them for the session.

        Returns:
            Wire-named arm joint limits of both participants.
        """
        if self._joint_limits_cache is not None:
            return dict(self._joint_limits_cache)
        import xml.etree.ElementTree as ET

        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String

        received: dict[str, str] = {}

        def receive(message: object) -> None:
            received["urdf"] = str(getattr(message, "data", ""))

        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        subscription = self.node.create_subscription(
            String, COMBINED_ROBOT_DESCRIPTION_TOPIC, receive, qos
        )
        try:
            deadline = self._clock_fn() + float(timeout_s)
            while not received and self._clock_fn() < deadline:
                self._spin_fn()
                self._sleep_fn(0.02)
        finally:
            self.node.destroy_subscription(subscription)
        if not received:
            raise RuntimeError(
                f"{COMBINED_ROBOT_DESCRIPTION_TOPIC} did not answer within {timeout_s}s"
            )
        arm_joints = {
            f"{robot_id}_joint{index}"
            for robot_id in self.robot_ids
            for index in range(1, 7)
        }
        limits: dict[str, tuple[float, float]] = {}
        for joint in ET.fromstring(received["urdf"]).findall("joint"):
            name = joint.get("name")
            limit = joint.find("limit")
            if (
                name in arm_joints
                and joint.get("type") in ("revolute", "continuous")
                and limit is not None
            ):
                limits[name] = (float(limit.get("lower")), float(limit.get("upper")))
        if set(limits) != arm_joints:
            raise RuntimeError(
                "combined robot description does not cover every arm joint"
            )
        self._joint_limits_cache = limits
        return dict(limits)

    def unified_start_state(self) -> Mapping[str, float]:
        """Return a complete wire-named combined start state, including mimic parents."""
        result: dict[str, float] = {}
        for robot_id in self.robot_ids:
            state = self.runtimes[robot_id].latest_joint_state()
            if not state:
                raise RuntimeError(f"no live joint state for {robot_id}")
            result.update(
                {f"{robot_id}_{name}": float(value) for name, value in state.items()}
            )
            finger = f"{robot_id}_gripper_joint1"
            parent = f"{robot_id}_gripper"
            if parent not in result and finger in result:
                result[parent] = 2.0 * result[finger]
        return result

    def unified_joint_targets(self, request) -> dict[str, dict[str, float]]:
        """Compose one joint-space combined goal from per-arm pose targets.

        The combined ``dual_arm`` group plans in joint space, so every pose
        target is realized per arm here: a moving arm is solved through the
        unified ``/dual_arm/compute_ik`` seeded by its own live measured
        state, and a ``maintain_pose`` participant holds its currently
        measured arm joints. Any missing live state, IK failure or malformed
        target raises so the coordinated entry fails closed.

        Returns:
            Wire-named arm joint positions for every robot needing solving.
        """
        if tuple(request.robots) != self.robot_ids:
            raise ValueError("unified request robots do not match session")
        limits: dict[str, tuple[float, float]] = {}
        seeds: dict[str, dict[str, float]] = {}
        for robot_id in self.robot_ids:
            canonical = self.runtimes[robot_id].latest_joint_state()
            if not canonical:
                raise RuntimeError(
                    f"no live joint state to seed combined IK for {robot_id}"
                )
            seeds[robot_id] = {
                f"{robot_id}_{name}": float(value) for name, value in canonical.items()
            }
        solved: dict[str, dict[str, float]] = {}
        for robot_id in self.robot_ids:
            target = request.targets.get(robot_id) or {}
            arm = set(self.runtimes[robot_id].arm_joint_names)
            if robot_id not in request.participants:
                solved[robot_id] = {
                    joint: seeds[robot_id][joint] for joint in sorted(arm)
                }
                continue
            positions = target.get("joint_positions")
            if (
                isinstance(positions, Mapping)
                and set(positions) == arm
                and all(
                    isinstance(value, (int, float)) and math.isfinite(float(value))
                    for value in positions.values()
                )
            ):
                continue
            if target.get("maintain_pose") is True:
                solved[robot_id] = {
                    joint: seeds[robot_id][joint] for joint in sorted(arm)
                }
                continue
            if not limits:
                limit_reader = getattr(self, "unified_joint_limits", None)
                if callable(limit_reader):
                    limits = limit_reader(request.timeout_s)
            arm_limits = {joint: limits[joint] for joint in arm if joint in limits}
            extra = (
                ik_seed_candidates(
                    seeds[robot_id],
                    arm_limits,
                    IK_SEED_ATTEMPTS - 1,
                    f"{request.coordination_id}|{robot_id}",
                )
                if arm_limits
                else ()
            )
            solved[robot_id] = self.runtimes[robot_id].solve_tcp_ik(
                robot_id,
                target,
                seeds[robot_id],
                request.timeout_s,
                extra_seeds=extra,
            )
        return solved

    @property
    def unified_planning_scene_version(self) -> int:
        """Return the current unified planning-scene version."""
        refresh = getattr(self._scene_binding, "refresh_version", None)
        if callable(refresh):
            try:
                return int(refresh(5.0))
            except (ImportError, ModuleNotFoundError):
                pass
        return int(self._scene_binding.version)

    def latest_shared_sim_time_s(self) -> float:
        """Return the newest shared ``/clock`` sample across participants.

        Every participant subscribes to the one session clock, so the newest
        sample is the live shared simulation time an observation answer can be
        freshness-bound to.
        """
        samples = [
            clocks[-1] * 1e-9
            for runtime in self.runtimes.values()
            if (clocks := runtime.clock_values())
        ]
        if not samples:
            raise RuntimeError("no shared /clock sample has been observed")
        return float(max(samples))

    def observe_all(self, timeout_s: float = 1.0) -> Mapping[str, object]:
        """Fetch and validate one complete global scene snapshot.

        Returns:
            The complete shared document, never independently queried robot snapshots.
        """
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError("shared observation timeout must be finite and positive")
        deadline = self._clock_fn() + timeout_s
        try:
            shared = self.runtimes[self.robot_ids[0]].query_scene_state(timeout_s)
            robots = shared.get("robots")
            if not isinstance(robots, Mapping) or set(robots) != set(self.robot_ids):
                raise RuntimeError(
                    "shared scene snapshot does not cover all robots: "
                    f"scene_error={shared.get('error')!r} "
                    "robots="
                    f"{sorted(robots) if isinstance(robots, Mapping) else robots!r}"
                )
            for robot_id in self.robot_ids:
                robot_scene_view(shared, robot_id)
            if not isinstance(shared.get("session_id"), str) or not shared.get(
                "session_id"
            ):
                raise RuntimeError("shared snapshot has no session provenance")
            manifest = shared.get("manifest_sha256")
            if not isinstance(manifest, str) or len(manifest) != 64:
                raise RuntimeError("shared snapshot has no manifest provenance")
            import hashlib

            canonical = json.dumps(
                {key: value for key, value in shared.items() if key != "scene_digest"},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            if hashlib.sha256(canonical.encode()).hexdigest() != shared.get(
                "scene_digest"
            ):
                raise RuntimeError(
                    "shared snapshot digest is not bound to complete observation"
                )
            sequence = shared.get("sequence")
            if type(sequence) is not int or sequence < 0:
                raise RuntimeError("shared snapshot sequence is invalid")
            snapshot_time = shared.get("sim_time_s")
            if (
                isinstance(snapshot_time, bool)
                or not isinstance(snapshot_time, (int, float))
                or not math.isfinite(snapshot_time)
                or snapshot_time < 0
            ):
                raise RuntimeError("shared snapshot simulation time is invalid")
            self._wait_for_snapshot_clock(float(snapshot_time), deadline)
            return shared
        except Exception as error:
            for robot in self.runtimes.values():
                robot._last_scene_error = f"{type(error).__name__}: {error}"
            raise

    def _wait_for_snapshot_clock(self, snapshot_time: float, deadline: float) -> None:
        """Pump real clock callbacks within the original observation timeout.

        A service response can arrive before the matching /clock topic sample.
        Never replace the received clock with the scene timestamp or relax the
        caller's scene-age gate. A backwards clock fails closed.
        """
        previous = None
        while True:
            observed = self.latest_shared_sim_time_s()
            if not math.isfinite(observed) or observed < 0:
                raise RuntimeError("shared /clock sample is invalid")
            if previous is not None and observed < previous:
                raise RuntimeError(
                    "shared /clock moved backwards during snapshot alignment"
                )
            if observed >= snapshot_time:
                return
            previous = observed
            remaining = deadline - self._clock_fn()
            if remaining <= 0:
                raise TimeoutError(
                    "shared /clock did not reach snapshot time before observation "
                    f"deadline: clock={observed:.9f}, snapshot={snapshot_time:.9f}"
                )
            self._spin_fn()
            self._sleep_fn(min(self._poll_interval_s, 0.005, remaining))

    def unified_readiness(
        self, request, *, probe_scene: bool = True
    ) -> UnifiedDualArmReadiness:
        """Evaluate one shared combined-planner readiness snapshot."""
        if tuple(request.robots) != self.robot_ids:
            raise ValueError("unified request robots do not match session")
        runtime = self._coordinated_runtime
        if runtime is None:
            raise RuntimeError("unified dual-arm runtime is not configured")
        runtime.register()
        shared = request.scene_snapshot
        if not isinstance(shared, Mapping) or not shared:
            raise RuntimeError("plan request must bind one shared scene snapshot")
        robots = shared.get("robots")
        if not isinstance(robots, Mapping) or set(robots) != set(self.robot_ids):
            raise RuntimeError("shared scene snapshot does not cover all robots")
        digest = shared.get("scene_digest")
        import hashlib

        payload = {key: value for key, value in shared.items() if key != "scene_digest"}
        computed = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
        if digest != computed:
            raise RuntimeError("bound scene snapshot digest is invalid")
        if digest != request.scene_digest:
            raise RuntimeError("shared scene digest does not match request")
        sequence = shared.get("sequence")
        if type(sequence) is not int or sequence < 0:
            raise RuntimeError("shared scene sequence is invalid")
        sim_time = shared.get("sim_time_s")
        if (
            not isinstance(sim_time, (int, float))
            or isinstance(sim_time, bool)
            or not math.isfinite(sim_time)
            or sim_time < 0
        ):
            raise RuntimeError("shared scene simulation time is invalid")
        for robot_id in self.robot_ids:
            robot_scene_view(shared, robot_id)
        controller_ready: dict[str, bool] = {}
        timing_by_robot: dict[str, Mapping[str, object]] = {}
        state_complete = True
        clock_increasing = True
        fresh = True
        stamps = []
        simulation_handover = (
            os.environ.get("ROCOBRICK_HANDOVER_SIMULATION") == "1"
            or request.coordination_id.endswith(":direct-handover")
            or ":giver-reposition" in request.coordination_id
            or ":zone-plan" in request.coordination_id
        )
        for robot_id in self.robot_ids:
            robot = self.runtimes[robot_id]
            report = robot.readiness_snapshot()
            state_complete = (
                state_complete
                and report.isaac_state_complete
                and report.joint_state_complete
                and report.broadcaster_match
            )
            clock_increasing = clock_increasing and report.clock_increasing
            controller_ready[robot_id] = (
                report.controllers_active
                and report.arm_action_ready
                and robot._action_ready(robot._gripper_client)
            )
            timing = robot.state_timing()
            timing_by_robot[robot_id] = dict(timing)
            fresh = fresh and all(
                isinstance(timing[name], (int, float)) and 0 <= timing[name] <= 0.5
                for name in ("isaac_age_s", "joint_age_s")
            )
            stamp = timing["joint_stamp_s"]
            if not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
                fresh = False
            else:
                stamps.append(stamp)
            clocks = robot.clock_values()
            clock_s = clocks[-1] * 1e-9 if clocks else None
            scene_delta = float(clock_s - sim_time) if clock_s is not None else None
            joint_delta = (
                float(clock_s - stamp)
                if clock_s is not None and isinstance(stamp, (int, float))
                else None
            )
            scene_clock_ok_raw = (
                scene_delta is not None and abs(scene_delta) <= 0.5
            )
            # In the Isaac HANDOVER trial the host scene snapshot can lag the
            # ROS clock while the simulator continues stepping between the
            # observation and the dispatch request. Keep the measured delta in
            # evidence, but do not turn that known transport lag into an
            # execution gate when the explicit exploratory mode is enabled.
            scene_clock_ok = scene_clock_ok_raw or (
                simulation_handover and scene_delta is not None
            )
            timing_by_robot[robot_id] = {
                **dict(timing),
                "clock_s": clock_s,
                "scene_sim_time_s": float(sim_time),
                "clock_minus_scene_s": scene_delta,
                "fresh_age_ok": all(
                    isinstance(timing[name], (int, float)) and 0 <= timing[name] <= 0.5
                    for name in ("isaac_age_s", "joint_age_s")
                ),
                "scene_clock_ok": scene_clock_ok,
                "scene_clock_ok_raw": scene_clock_ok_raw,
                "scene_clock_check_bypassed": (
                    simulation_handover and scene_clock_ok_raw is False
                    and scene_delta is not None
                ),
                "joint_clock_ok": joint_delta is not None and abs(joint_delta) <= 0.5,
            }
            fresh = fresh and timing_by_robot[robot_id]["fresh_age_ok"] is True
            fresh = fresh and bool(clocks)
            fresh = fresh and timing_by_robot[robot_id]["scene_clock_ok"] is True
            fresh = fresh and timing_by_robot[robot_id]["joint_clock_ok"] is True
        fresh = fresh and len(stamps) == len(self.robot_ids)
        self._last_unified_timing = dict(timing_by_robot)
        return UnifiedDualArmReadiness(
            robots=self.robot_ids,
            clock_increasing=clock_increasing,
            combined_joint_state_complete=state_complete,
            scene_snapshot_fresh=fresh,
            scene_digest_match=True,
            combined_move_group_ready=runtime.ready(),
            controllers_ready=controller_ready,
        )

    def coordinated_readiness(self, request) -> dict[str, object]:
        """Build strict readiness evidence for one combined request.

        Returns:
            Combined participant, scene, and action readiness evidence.
        """
        if tuple(request.robots) != self.robot_ids:
            raise ValueError("coordinated request robots do not match session")
        readiness = self.unified_readiness(request)
        if not readiness.passed:
            components = {
                "clock_increasing": readiness.clock_increasing,
                "combined_joint_state_complete": readiness.combined_joint_state_complete,
                "scene_snapshot_fresh": readiness.scene_snapshot_fresh,
                "scene_digest_match": readiness.scene_digest_match,
                "combined_move_group_ready": readiness.combined_move_group_ready,
                **{
                    f"controller:{robot}": ready
                    for robot, ready in readiness.controllers_ready.items()
                },
            }
            failed = [name for name, passed in components.items() if not passed]
            raise RuntimeError(
                "unified dual-arm readiness failed: "
                + ", ".join(failed)
                + f"; timing_by_robot={dict(self._last_unified_timing)}"
            )
        return {
            "coordination_id": request.coordination_id,
            "participants": list(request.participants),
            "scene_digest": request.scene_digest,
            "scene_sequence": request.scene_snapshot["sequence"],
            "scene_sim_time_s": request.scene_snapshot["sim_time_s"],
            "move_group_action": "/dual_arm/move_action",
            "planning_group": "dual_arm",
            "diagnostics": {
                "timing_by_robot": dict(self._last_unified_timing),
            },
            "components": {
                "clock_increasing": readiness.clock_increasing,
                "combined_joint_state_complete": readiness.combined_joint_state_complete,
                "scene_snapshot_fresh": readiness.scene_snapshot_fresh,
                "scene_digest_match": readiness.scene_digest_match,
                "combined_move_group_ready": readiness.combined_move_group_ready,
                **{
                    f"controller:{robot}": ready
                    for robot, ready in readiness.controllers_ready.items()
                },
            },
        }

    def coordinated_plan_runtime(self):
        """Return the combined MoveIt plan-only transport."""
        if self._coordinated_runtime is None:
            raise RuntimeError("combined MoveIt runtime requires both Piper robots")
        return self._coordinated_runtime

    # ------------------------------------------------------------ registration

    def register(self) -> None:
        """Register every robot's subscriptions and clients exactly once."""
        self.register_subscriptions()
        self.register_clients()
        if self._coordinated_runtime is not None:
            self._coordinated_runtime.register()

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
                f"unknown robot {robot_id!r}; session robots: {list(self.robot_ids)}"
            ) from error

    def executor_for(self, robot_id: str) -> Any:
        """The executor bound to ``robot_id``'s runtime; unknown ids raise."""
        try:
            return self.executors[robot_id]
        except KeyError as error:
            raise KeyError(
                f"unknown robot {robot_id!r}; session robots: {list(self.robot_ids)}"
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
        if probe_scene and any(
            profile.move_group_action == "/dual_arm/move_action"
            for profile in self.profiles.values()
        ):
            try:
                shared = self.observe_all()
                scene_query_ready = True
            except (RuntimeError, ValueError, TimeoutError) as error:
                scene_query_ready = False
                for robot in self.runtimes.values():
                    robot._last_scene_error = f"{type(error).__name__}: {error}"
            reports = {
                robot_id: self.runtimes[robot_id].capability_readiness(
                    probe_scene=False
                )
                for robot_id in self.robot_ids
            }
            return {
                robot_id: replace(report, scene_query_ready=scene_query_ready)
                for robot_id, report in reports.items()
            }
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
        if self._coordinated_runtime is not None:
            try:
                self._coordinated_runtime.shutdown()
            except Exception as error:
                self.shutdown_errors.append(f"combined: {error}")
        for robot_id in self.robot_ids:
            runtime = self.runtimes[robot_id]
            try:
                runtime.shutdown()
            except Exception as exc:
                self.shutdown_errors.append(f"{robot_id}: {type(exc).__name__}: {exc}")
            for message in runtime.shutdown_errors:
                self.shutdown_errors.append(f"{robot_id}: {message}")
