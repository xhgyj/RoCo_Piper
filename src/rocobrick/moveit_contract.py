"""P1 execution contract for real Piper MoveIt joint and link6 pose goals.

Frozen value types and pinned constants mirroring the live-verified P1.B /
P1.C behavior (tools/p1_real_joint_probe.py, tools/p1_real_pose_probe.py and
artifacts/phase1-real/joint-target.json / pose-target.json). Pure data only:
no ROS, no Isaac, no transport imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

# Same canonical order as the pinned model contract
# (tests/test_piper_l_model_contract.py): six arm joints then two grippers.
JOINT_NAMES: tuple[str, ...] = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "gripper_joint1",
    "gripper_joint2",
)
ARM_JOINT_NAMES: tuple[str, ...] = JOINT_NAMES[:6]
GRIPPER_JOINT_NAMES: tuple[str, ...] = JOINT_NAMES[6:]

# Narrowed EXECUTION limits (rad): intersection of the pinned official sources
# as enforced for planning by the MoveIt joint_limits.yaml. These are pinned
# here for CALLER-SIDE goal admission: MoveIt's documented out-of-bounds
# clamping ("Assuming maximum bounds instead") is never relied on.
EXECUTION_POSITION_LIMITS: Mapping[str, tuple[float, float]] = MappingProxyType(
    {
        "joint1": (-2.6179938, 2.6179938),
        "joint2": (0.0, 3.1415926),
        "joint3": (-2.9670597, 0.0),
        "joint4": (-1.7453292, 1.7453292),
        "joint5": (-1.2217304, 1.2217304),
        "joint6": (-2.0943951, 2.0943951),
    }
)

# ROS topic contract from the pinned vendor ros2_control xacro
# (TopicBasedSystem): the Isaac states topic is plural, the command topic is
# singular, and the broadcaster republishes on the standard /joint_states.
TOPIC_CLOCK: str = "/clock"
TOPIC_ISAAC_JOINT_STATES: str = "/isaac_joint_states"
TOPIC_JOINT_STATES: str = "/joint_states"
TOPIC_ISAAC_JOINT_COMMAND: str = "/isaac_joint_command"

# Actions and services (live-resolved names recorded in the P1 evidence).
MOVE_GROUP_ACTION_CANDIDATES: tuple[str, ...] = ("/move_action", "/move_group")
ACTION_ARM_FOLLOW: str = "/arm_controller/follow_joint_trajectory"
SERVICE_LIST_CONTROLLERS: str = "/controller_manager/list_controllers"
SERVICE_PLAN: str = "/plan_kinematic_path"
SERVICE_COMPUTE_FK: str = "/compute_fk"

# Controllers that must report "active" before any motion.
REQUIRED_CONTROLLERS: tuple[str, ...] = (
    "joint_state_broadcaster",
    "arm_controller",
    "gripper_controller",
)

# moveit_msgs.msg.MoveItErrorCodes.SUCCESS.
SUCCESS_CODE: int = 1
MIN_WAYPOINTS: int = 2
# Numeric slack for floating-point noise only (rad) around the limits.
LIMIT_EPSILON: float = 1.0e-4

# Pose scope: the link6 ORIGIN in base_link. The pinned piper_l model has no
# tool0 or dedicated official TCP frame, so the link6 TCP status stays
# unresolved and no link6 TCP semantics exist beyond that record. The
# PROJECT-CALIBRATED frame rocobrick_tcp (R5.A) is separate: it is generated
# into the model from the accepted binding-contract artifact and carries the
# calibrated_project status below (project/simulation-only/runtime-USD-scoped;
# no hardware claim).
BASE_FRAME: str = "base_link"
TIP_LINK: str = "link6"
TCP_STATUS: str = "unresolved"

# R5.A project-calibrated TCP scope (single artifact authority: the accepted
# artifacts/model-audit/piper_l_tcp_binding_contract.json; the MoveIt model
# overlay is generated from it and tests pin its hash and values).
TCP_LINK: str = "rocobrick_tcp"
TCP_JOINT_NAME: str = "rocobrick_tcp_joint"
TCP_STATUS_CALIBRATED_PROJECT: str = "calibrated_project"
# MoveGroup planning group for the TCP chain (SRDF group arm_tcp).
TCP_PLANNING_GROUP: str = "arm_tcp"

# Contract maxima for requested goal tolerances (live P1.C verification uses
# these same bounds; requested constraints may be tighter but never looser).
MAX_POSITION_TOLERANCE_M: float = 0.01
MAX_ORIENTATION_TOLERANCE_RAD: float = 0.03

# Quaternion gates: reject nonfinite components and near-zero quaternions,
# then normalize and demand a normalized result.
QUAT_NORM_MIN: float = 1.0e-6
QUAT_NORM_EPS: float = 1.0e-6

# Strict caller-side admission margin (rad) inside the execution limits.
ADMISSION_MARGIN: float = 0.03
# Max |/joint_states - /isaac_joint_states| while proving the broadcaster path.
EXACT_STATE_MATCH_TOLERANCE: float = 1.0e-4


@dataclass(frozen=True)
class JointTarget:
    """Arm joint goal keyed by canonical joint name, sent as MoveGroup goal."""

    positions: Mapping[str, float]
    goal_tolerance: float = 0.01


@dataclass(frozen=True)
class Link6PoseTarget:
    """link6 origin pose goal in base_link; the TCP status stays unresolved."""

    frame_id: str
    position: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    position_tolerance_m: float = 0.01
    orientation_tolerance_rad: float = 0.03
    tip_link: str = "link6"
    tcp_status: str = "unresolved"


@dataclass(frozen=True)
class TcpPoseTarget:
    """Project-calibrated TCP pose goal in base_link (R5.A).

    Same pose contract as :class:`Link6PoseTarget` but the goal frame is the
    generated zero-geometry ``rocobrick_tcp`` link of the MoveIt model and the
    provenance metadata is mandatory: ``runtime_model_sha256`` scopes the
    calibrated claim to the exact runtime USD and ``source_sha256`` pins the
    accepted calibration artifact the model overlay was generated from. Both
    must be lowercase SHA-256 digests and are never defaulted.
    """

    frame_id: str
    position: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    position_tolerance_m: float = MAX_POSITION_TOLERANCE_M
    orientation_tolerance_rad: float = MAX_ORIENTATION_TOLERANCE_RAD
    tip_link: str = TCP_LINK
    tcp_status: str = TCP_STATUS_CALIBRATED_PROJECT
    runtime_model_sha256: str = field(kw_only=True)
    source_sha256: str = field(kw_only=True)


@dataclass(frozen=True)
class AdmissionDecision:
    """Outcome of a pure admission/readiness check with evidence details."""

    accepted: bool
    reason: str
    details: Mapping[str, object] = field(default_factory=dict)
