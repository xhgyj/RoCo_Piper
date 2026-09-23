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
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
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
from rocobrick.moveit_contract import (
    ACTION_ARM_FOLLOW,
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
    JointTarget,
    Link6PoseTarget,
    TcpPoseTarget,
)
from rocobrick.moveit_execution import ExecutionResult, ExecutionStatus
from rocobrick.planning_scene import (
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

PLANNING_GROUP = "arm"
# MoveGroup group for project-calibrated TCP goals (SRDF chain arm_tcp).
TCP_PLANNING_GROUP = "arm_tcp"
ALLOWED_PLANNING_TIME_S = 5.0
NUM_PLANNING_ATTEMPTS = 10
MAX_SCALING_FACTOR = 0.1

# Formal planning-scene service (moveit_msgs/srv/ApplyPlanningScene).
SERVICE_APPLY_PLANNING_SCENE = "/apply_planning_scene"

# Bounded-wait pacing and deadlines (mirrors the live P1 probe discipline).
POLL_INTERVAL_S = 0.2
WAIT_STEP_S = 0.05
ACCEPT_TIMEOUT_S = 15.0
CANCEL_TIMEOUT_S = 10.0
SERVICE_CALL_TIMEOUT_S = 5.0

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

    ``apply_planning_scene`` is the formal R5.A planning-scene service; the
    P1-evidence names above are unchanged.
    """

    list_controllers: str = SERVICE_LIST_CONTROLLERS
    plan: str = SERVICE_PLAN
    compute_fk: str = SERVICE_COMPUTE_FK
    apply_planning_scene: str = SERVICE_APPLY_PLANNING_SCENE


@dataclass(frozen=True)
class RuntimeActions:
    """Arm follow action plus the first live MoveGroup action candidate."""

    arm_follow: str = ACTION_ARM_FOLLOW
    move_group: str = MOVE_GROUP_ACTION_CANDIDATES[0]


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


def _positions_by_name(message: Any) -> dict[str, float]:
    """JointState message positions keyed by joint name."""
    return {
        str(name): float(value)
        for name, value in zip(message.name, message.position)
    }


def _clock_nanos(message: Any) -> int:
    """Clock message as integer nanoseconds."""
    return int(message.clock.sec) * 1_000_000_000 + int(message.clock.nanosec)


def _state_message_is_valid(message: Any) -> bool:
    """Complete 8-DOF finite positions and a velocity array of length 0 or 8."""
    try:
        state = _positions_by_name(message)
        velocity = list(message.velocity)
    except (AttributeError, TypeError, ValueError):
        return False
    if not state_is_complete(state):
        return False
    return len(velocity) in (0, len(JOINT_NAMES))


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


class RosMoveItRuntime:
    """ROS-side MoveIt runtime; every transport boundary is injectable.

    The caller owns the node: nothing is created in ``__init__``. Call
    :meth:`register_subscriptions` and :meth:`register_clients` once, wait for
    :meth:`wait_until_ready` and only then send goals. QoS profiles default to
    the live P1 probe profiles (RELIABLE / BEST_EFFORT, KEEP_LAST depth 50)
    and are resolved lazily, so injected fakes can pass opaque placeholders.
    """

    def __init__(
        self,
        node: Any,
        *,
        topics: RuntimeTopics | None = None,
        services: RuntimeServices | None = None,
        actions: RuntimeActions | None = None,
        qos_sensor: Any = None,
        qos_state: Any = None,
        factories: RosMoveItFactories | None = None,
        fk_verifier: FkVerifier | None = None,
        planning_scene_applier: (
            Callable[[PlanningSceneUpdate, float], PlanningSceneResult] | None
        ) = None,
        clock: Callable[[], float] | None = None,
        spin: Callable[[], None] | None = None,
        sleep: Callable[[float], None] | None = None,
        poll_interval_s: float = POLL_INTERVAL_S,
        wait_step_s: float = WAIT_STEP_S,
    ) -> None:
        self.node = node
        self.topics = topics if topics is not None else RuntimeTopics()
        self.services = services if services is not None else RuntimeServices()
        self.actions = actions if actions is not None else RuntimeActions()
        self._qos_sensor = qos_sensor
        self._qos_state = qos_state
        self._factories = factories
        self._fk_verifier = fk_verifier
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
        self._arm_client: Any = None
        self._move_group_client: Any = None
        self._active_handle: Any = None
        self._resolved_factories: RosMoveItFactories | None = None
        self.shutdown_errors: list[str] = []

    # ------------------------------------------------------------- transport

    def _default_factories(self) -> RosMoveItFactories:
        """Real ROS/MoveIt factories; the lazy ROS imports live only here."""
        from action_msgs.msg import GoalStatus
        from control_msgs.action import FollowJointTrajectory
        from controller_manager_msgs.srv import ListControllers
        from geometry_msgs.msg import Pose
        from moveit_msgs.action import MoveGroup
        from moveit_msgs.msg import (
            CollisionObject,
            Constraints,
            JointConstraint,
            MotionPlanRequest,
            OrientationConstraint,
            PositionConstraint,
        )
        from moveit_msgs.srv import ApplyPlanningScene, GetMotionPlan, GetPositionFK
        from rclpy.action import ActionClient
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import JointState
        from shape_msgs.msg import SolidPrimitive

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
        """Create the service and action clients once (lazy ROS imports)."""
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
        self._arm_client = factories.action_client_factory(
            node, factories.follow_joint_trajectory_type(), self.actions.arm_follow
        )
        self._move_group_client = factories.action_client_factory(
            node, factories.move_group_type(), self.actions.move_group
        )

    # ------------------------------------------------------------- callbacks

    def _on_clock(self, message: Any) -> None:
        with self._lock:
            self._clock_values.append(_clock_nanos(message))

    def _on_isaac_state(self, message: Any) -> None:
        if not _state_message_is_valid(message):
            return
        state = _positions_by_name(message)
        with self._lock:
            self._isaac_states.append(state)
            self._isaac_receipts.append(self._clock_fn())
            self._isaac_state_count += 1

    def _on_joint_state(self, message: Any) -> None:
        if not _state_message_is_valid(message):
            return
        state = _positions_by_name(message)
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

    # ------------------------------------------------------------------ FK

    def compute_link6_fk(
        self, state: Mapping[str, float], timeout_s: float = SERVICE_CALL_TIMEOUT_S
    ) -> FkVerification | None:
        """Live ``base_link -> link6`` pose of ``state`` via /compute_fk.

        The explicitly minimal public FK adapter hook: the GetPositionFK
        request is built through the injected factories and the registered
        client from the EXACT canonical 8-DOF names of ``state``, waited on
        bounded through the injected clock/spin/sleep hooks, and answered with
        ``((x, y, z), (x, y, z, w))`` or ``None``. No pose is ever hardcoded
        or interpolated: a non-SUCCESS code, an incomplete state, a missing
        answer, a nonfinite or near-zero result all return ``None`` so the
        caller fails explicitly instead of faking a pose.
        """
        return self._compute_fk(state, TIP_LINK, timeout_s)

    def compute_tcp_fk(
        self, state: Mapping[str, float], timeout_s: float = SERVICE_CALL_TIMEOUT_S
    ) -> FkVerification | None:
        """Live ``base_link -> rocobrick_tcp`` pose of ``state`` via /compute_fk.

        Identical contract to :meth:`compute_link6_fk` but the requested link
        is the project-calibrated ``rocobrick_tcp`` frame of the expanded
        MoveIt model, so the answered pose is exactly the link6 FK composed
        with the generated calibration joint. The same failure rules apply:
        nothing is hardcoded, interpolated, or faked.
        """
        return self._compute_fk(state, TCP_LINK, timeout_s)

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
        request.header.frame_id = BASE_FRAME
        request.fk_link_names = [link]
        request.robot_state.joint_state.name = list(JOINT_NAMES)
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
        return PlanningSceneResult(
            passed=True,
            applied_object_ids=applied,
            scene_digest=computed,
            details=details,
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
        """MoveGroup joint goal: plan_only false, exact arm JointConstraints."""
        goal = factories.move_group_goal_factory()
        request = factories.motion_plan_request_factory()
        request.group_name = PLANNING_GROUP
        request.allowed_planning_time = ALLOWED_PLANNING_TIME_S
        request.num_planning_attempts = NUM_PLANNING_ATTEMPTS
        request.max_velocity_scaling_factor = MAX_SCALING_FACTOR
        request.max_acceleration_scaling_factor = MAX_SCALING_FACTOR
        goal_constraints = factories.constraints_factory()
        tolerance = float(target.goal_tolerance)
        for name in ARM_JOINT_NAMES:
            constraint = factories.joint_constraint_factory()
            constraint.joint_name = name
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

    def _build_pose_goal(
        self, target: Link6PoseTarget | TcpPoseTarget, factories: RosMoveItFactories
    ) -> Any:
        """MoveGroup pose goal: one SPHERE position + one orientation constraint.

        The planning group follows the target: ``arm`` for link6 goals and
        ``arm_tcp`` for project-calibrated TCP goals; the constraint link and
        frame come from the target verbatim (never offset or rewritten).
        """
        group = TCP_PLANNING_GROUP if isinstance(target, TcpPoseTarget) else PLANNING_GROUP
        goal = factories.move_group_goal_factory()
        request = factories.motion_plan_request_factory()
        request.group_name = group
        request.allowed_planning_time = ALLOWED_PLANNING_TIME_S
        request.num_planning_attempts = NUM_PLANNING_ATTEMPTS
        request.max_velocity_scaling_factor = MAX_SCALING_FACTOR
        request.max_acceleration_scaling_factor = MAX_SCALING_FACTOR
        goal_constraints = factories.constraints_factory()

        position = factories.position_constraint_factory()
        position.header.frame_id = target.frame_id
        position.link_name = target.tip_link
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
        orientation.header.frame_id = target.frame_id
        orientation.link_name = target.tip_link
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
            return ExecutionResult(
                status=ExecutionStatus.TIMED_OUT,
                details=details,
                command_messages=self._command_count - send_commands,
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
        result_future = handle.get_result_async()
        if not self._wait_future(result_future, deadline):
            details["cancel"] = self.cancel_active()
            self._active_handle = None
            details["phase"] = "execution"
            details["error"] = (
                f"move_group action exceeded the {timeout_s}s wall deadline "
                "(cancel attempted)"
            )
            return ExecutionResult(
                status=ExecutionStatus.TIMED_OUT,
                details=details,
                command_messages=self._command_count - send_commands,
            )
        self._active_handle = None
        wrapper = result_future.result()
        action_result = wrapper.result
        status = _map_goal_status(wrapper.status, factories.goal_status_type())
        error_code = int(getattr(action_result.error_code, "val"))
        planned_waypoints, trajectory_decision = self._inspect_trajectory(action_result)
        details["goal_status"] = int(wrapper.status)
        details["trajectory_accepted"] = trajectory_decision.accepted
        if not trajectory_decision.accepted:
            details["trajectory"] = dict(trajectory_decision.details)
        common = dict(
            moveit_error_code=error_code,
            planned_waypoints=planned_waypoints,
            command_messages=self._command_count - send_commands,
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
        """R1 waypoint admission over the returned planned trajectory."""
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
        if self._fk_verifier is None:
            details["error"] = (
                "no fk verification hook was supplied; the exact final pose "
                "cannot be observed"
            )
            return ExecutionResult(
                status=ExecutionStatus.FAILED, details=details, **common
            )
        try:
            verified = self._fk_verifier(dict(final))
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
        """Best-effort bounded cancellation of the active MoveGroup goal."""
        handle = self._active_handle
        if handle is None:
            return {"attempted": False}
        info: dict[str, object] = {"attempted": True}
        try:
            future = handle.cancel_goal_async()
            deadline = self._clock_fn() + CANCEL_TIMEOUT_S
            while not future.done():
                self._spin_fn()
                if future.done():
                    break
                if self._clock_fn() >= deadline:
                    info["error"] = "cancel did not complete before its deadline"
                    return info
                self._sleep_fn(self._wait_step_s)
            info["return_code"] = int(future.result().return_code)
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
        self._arm_client = None
        self._move_group_client = None
        self._active_handle = None
