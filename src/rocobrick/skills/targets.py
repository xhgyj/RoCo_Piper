"""Pure skill-target geometry: world-frame brick AABBs to base_link TCP goals.

No ROS, no Isaac, no runtime imports: this module derives the calibrated-TCP
pose targets and gripper openings of the generic skill programs
(:mod:`rocobrick.skills`) from three inputs only — the measured world-frame
axis-aligned bounding box of one physical brick, the robot base pose in the
world frame (read from the user config), and the frozen
:class:`SkillParams` document (``config/skill_params.json``). Every TCP pose
is expressed in ``base_link`` with the project-calibrated TCP contract, so
each derived target passes strict admission unchanged; the provenance
SHA-256 fields are mandatory and never defaulted.

Orientation semantics: the grasp orientation is ``qz(closing_yaw) ⊗ qx(180°)``
(xyzw, normalized) so the tool z axis points DOWN onto the brick and the tool
y axis stays horizontal, rotated by the closing yaw. The closing yaw is 0°
when the brick's y-extent does not exceed its x-extent (fingers close along
world y) and 90° otherwise.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path

from rocobrick.moveit_contract import (
    BASE_FRAME,
    MAX_ORIENTATION_TOLERANCE_RAD,
    MAX_POSITION_TOLERANCE_M,
    TCP_LINK,
    TCP_STATUS_CALIBRATED_PROJECT,
    TcpPoseTarget,
)

Vector3 = tuple[float, float, float]
QuaternionXYZW = tuple[float, float, float, float]


@dataclass(frozen=True)
class RobotBaseFrame:
    """The robot base pose in the world frame (config identity only)."""

    position_xyz: Vector3  # base origin in world
    orientation_xyzw: QuaternionXYZW  # base orientation in world (normalized)


@dataclass(frozen=True)
class TcpProvenance:
    """Mandatory calibrated-TCP provenance carried onto every pose target."""

    runtime_model_sha256: str
    source_sha256: str


@dataclass(frozen=True)
class SkillParams:
    """Every numeric knob of the generic skill programs (all finite floats)."""

    pregrasp_standoff_m: float = 0.08  # > 0
    lift_height_m: float = 0.12  # > 0
    transit_extra_height_m: float = 0.10  # > 0
    open_extra_m: float = 0.02  # >= 0
    squeeze_m: float = 0.004  # >= 0
    contact_window_slop_m: float = 0.006  # > 0
    free_window_slop_m: float = 0.01  # > 0
    gripper_duration_s: float = 2.5  # > 0
    gripper_goal_tolerance: float = 0.01  # > 0
    wait_duration_s: float = 2.0  # > 0
    position_tolerance_m: float = 0.005  # in (0, 0.01]
    orientation_tolerance_rad: float = 0.015  # in (0, 0.03]
    still_tolerance_m: float = 0.01  # > 0
    hold_follow_tolerance_m: float = 0.02  # > 0
    scene_query_poll_s: float = 0.2  # > 0

    def __post_init__(self) -> None:
        bounds = {
            "pregrasp_standoff_m": (0.0, math.inf),
            "lift_height_m": (0.0, math.inf),
            "transit_extra_height_m": (0.0, math.inf),
            "open_extra_m": (0.0, math.inf),
            "squeeze_m": (0.0, math.inf),
            "contact_window_slop_m": (0.0, math.inf),
            "free_window_slop_m": (0.0, math.inf),
            "gripper_duration_s": (0.0, math.inf),
            "gripper_goal_tolerance": (0.0, math.inf),
            "wait_duration_s": (0.0, math.inf),
            "still_tolerance_m": (0.0, math.inf),
            "hold_follow_tolerance_m": (0.0, math.inf),
            "scene_query_poll_s": (0.0, math.inf),
        }
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number, got {value!r}")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"{name} must be finite, got {value!r}")
            object.__setattr__(self, name, number)
        for name, (low, high) in bounds.items():
            value = float(getattr(self, name))
            if not value > low:
                raise ValueError(f"{name} must be > {low}, got {value!r}")
        for name in ("position_tolerance_m", "orientation_tolerance_rad"):
            value = float(getattr(self, name))
            maximum = (
                MAX_POSITION_TOLERANCE_M
                if name == "position_tolerance_m"
                else MAX_ORIENTATION_TOLERANCE_RAD
            )
            if not 0.0 < value <= maximum:
                raise ValueError(
                    f"{name} must be in (0, {maximum}], got {value!r}"
                )


def load_skill_params(path: Path) -> SkillParams:
    """Load a strict exact-key ``skill_params.json`` document.

    The document must carry exactly the :class:`SkillParams` field names and
    nothing else; every value must satisfy the field bounds. Any missing or
    extra key, unreadable file, or non-JSON content raises ``ValueError``.

    Raises:
        ValueError: On any deviation from the exact key set, the JSON
            syntax, or the value bounds.
    """
    try:
        payload = json.loads(Path(path).read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"skill params {path} is unreadable: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("skill params must be a JSON object")
    expected = {field.name for field in fields(SkillParams)}
    actual = {str(key) for key in payload}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            f"skill params must carry exactly {sorted(expected)}; "
            f"missing={missing} extra={extra}"
        )
    try:
        return SkillParams(**{str(key): payload[key] for key in payload})
    except ValueError as error:
        raise ValueError(f"skill params {path} is invalid: {error}") from error


def _normalize_quat(quat: Sequence[float]) -> QuaternionXYZW:
    """Return the xyzw quaternion normalized to unit length."""
    x, y, z, w = (float(value) for value in quat)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("quaternion must be finite and non-zero")
    return (x / norm, y / norm, z / norm, w / norm)


def _quat_conjugate(quat: QuaternionXYZW) -> QuaternionXYZW:
    """Return the conjugate (-x, -y, -z, w) of an xyzw quaternion."""
    x, y, z, w = quat
    return (-x, -y, -z, w)


def _quat_mul(
    left: QuaternionXYZW, right: QuaternionXYZW
) -> QuaternionXYZW:
    """Return the Hamilton product left ⊗ right of two xyzw quaternions."""
    xl, yl, zl, wl = left
    xr, yr, zr, wr = right
    return (
        wl * xr + xl * wr + yl * zr - zl * yr,
        wl * yr - xl * zr + yl * wr + zl * xr,
        wl * zr + xl * yr - yl * xr + zl * wr,
        wl * wr - xl * xr - yl * yr - zl * zr,
    )


def _quat_rotate_vector(
    quat: QuaternionXYZW, vector: Sequence[float]
) -> Vector3:
    """Rotate a 3-vector by an xyzw quaternion (v' = q ⊗ v ⊗ q*)."""
    x, y, z = (float(value) for value in vector)
    qx, qy, qz, qw = quat
    # t = 2 * cross(q.xyz, v); v' = v + w * t + cross(q.xyz, t)
    tx = 2.0 * (qy * z - qz * y)
    ty = 2.0 * (qz * x - qx * z)
    tz = 2.0 * (qx * y - qy * x)
    return (
        x + qw * tx + (qy * tz - qz * ty),
        y + qw * ty + (qz * tx - qx * tz),
        z + qw * tz + (qx * ty - qy * tx),
    )


def grasp_orientation_xyzw(closing_yaw_deg: float) -> QuaternionXYZW:
    """Return the calibrated-TCP grasp orientation for one closing yaw.

    ``qz(closing_yaw) ⊗ qx(180°)`` in xyzw, normalized: the tool z axis
    points down (rotating (0, 0, 1) yields (0, 0, -1)) and the tool y axis
    stays horizontal, rotated by the closing yaw.
    """
    yaw = math.radians(float(closing_yaw_deg))
    half = 0.5 * yaw
    qz: QuaternionXYZW = (0.0, 0.0, math.sin(half), math.cos(half))
    qx_180: QuaternionXYZW = (1.0, 0.0, 0.0, 0.0)
    return _normalize_quat(_quat_mul(qz, qx_180))


def closing_yaw_deg(
    aabb_min: Sequence[float], aabb_max: Sequence[float]
) -> float:
    """Return 0.0 when the fingers close along world y, else 90.0.

    The y-extent vs x-extent comparison decides the closing axis: 0.0 when
    ``dy <= dx`` (close along world y) and 90.0 otherwise (close along x).
    """
    dx = float(aabb_max[0]) - float(aabb_min[0])
    dy = float(aabb_max[1]) - float(aabb_min[1])
    if not (math.isfinite(dx) and math.isfinite(dy)):
        raise ValueError("brick AABB extents must be finite")
    return 0.0 if dy <= dx else 90.0


def brick_span_m(
    aabb_min: Sequence[float], aabb_max: Sequence[float]
) -> float:
    """Return the brick extent along the closing axis (dy for yaw 0)."""
    dx = float(aabb_max[0]) - float(aabb_min[0])
    dy = float(aabb_max[1]) - float(aabb_min[1])
    if not (math.isfinite(dx) and math.isfinite(dy)):
        raise ValueError("brick AABB extents must be finite")
    return dy if closing_yaw_deg(aabb_min, aabb_max) == 0.0 else dx


def world_position_to_base(
    position_xyz: Sequence[float], base: RobotBaseFrame
) -> Vector3:
    """Transform one world-frame point into the robot base frame.

    ``p_base = conj(q_base) ⊗ (p_world − t_base)``.
    """
    px, py, pz = (float(value) for value in position_xyz)
    tx, ty, tz = base.position_xyz
    delta = (px - tx, py - ty, pz - tz)
    return _quat_rotate_vector(_quat_conjugate(base.orientation_xyzw), delta)


def world_orientation_to_base(
    orientation_xyzw: Sequence[float], base: RobotBaseFrame
) -> QuaternionXYZW:
    """Transform a world-frame orientation into the requested base frame."""
    world = _normalize_quat(orientation_xyzw)
    return _normalize_quat(_quat_mul(_quat_conjugate(base.orientation_xyzw), world))


def _brick_center(aabb_min: Sequence[float], aabb_max: Sequence[float]) -> Vector3:
    """Return the world-frame center of the brick AABB."""
    return tuple(  # type: ignore[return-value]
        (float(low) + float(high)) / 2.0
        for low, high in zip(aabb_min, aabb_max)
    )


def _tcp_pose_target(
    position: Vector3,
    orientation_xyzw: QuaternionXYZW,
    provenance: TcpProvenance,
    params: SkillParams,
) -> TcpPoseTarget:
    """Build one calibrated TCP pose target under the frozen contract."""
    return TcpPoseTarget(
        frame_id=BASE_FRAME,
        position=position,
        orientation_xyzw=orientation_xyzw,
        position_tolerance_m=params.position_tolerance_m,
        orientation_tolerance_rad=params.orientation_tolerance_rad,
        tip_link=TCP_LINK,
        tcp_status=TCP_STATUS_CALIBRATED_PROJECT,
        runtime_model_sha256=provenance.runtime_model_sha256,
        source_sha256=provenance.source_sha256,
    )


def transit_target(
    aabb_min: Sequence[float],
    aabb_max: Sequence[float],
    base: RobotBaseFrame,
    provenance: TcpProvenance,
    params: SkillParams,
) -> TcpPoseTarget:
    """Derive the TRANSIT pose above the brick (grasp orientation)."""
    center_x, center_y, center_z = _brick_center(aabb_min, aabb_max)
    orientation = world_orientation_to_base(
        grasp_orientation_xyzw(closing_yaw_deg(aabb_min, aabb_max)), base
    )
    center_in_base = world_position_to_base((center_x, center_y, center_z), base)
    apex_in_base = world_position_to_base(
        (
            center_x,
            center_y,
            center_z + params.pregrasp_standoff_m + params.transit_extra_height_m,
        ),
        base,
    )
    return _tcp_pose_target(
        (center_in_base[0], center_in_base[1], apex_in_base[2]),
        orientation,
        provenance,
        params,
    )


@dataclass(frozen=True)
class PickTargets:
    """Every derived target of one PICK program over one measured brick."""

    pregrasp: TcpPoseTarget  # xy = brick center; z = center + standoff
    approach: TcpPoseTarget  # z = brick center (TCP contact mid-height)
    lift: TcpPoseTarget  # z = center + standoff + lift height
    reach_opening_m: float  # brick_span + open_extra_m
    close_opening_m: float  # brick_span − squeeze_m
    reach_expected_window_m: tuple[float, float]  # reach opening ± free slop
    close_expected_window_m: tuple[float, float]  # brick span ± contact slop
    brick_span_m: float
    closing_yaw_deg: float


def pick_targets(
    aabb_min: Sequence[float],
    aabb_max: Sequence[float],
    base: RobotBaseFrame,
    provenance: TcpProvenance,
    params: SkillParams,
) -> PickTargets:
    """Derive every PICK target from one world-frame brick AABB."""
    center_x, center_y, center_z = _brick_center(aabb_min, aabb_max)
    orientation = world_orientation_to_base(
        grasp_orientation_xyzw(closing_yaw_deg(aabb_min, aabb_max)), base
    )
    center_in_base = world_position_to_base((center_x, center_y, center_z), base)
    pregrasp_in_base = world_position_to_base(
        (center_x, center_y, center_z + params.pregrasp_standoff_m), base
    )
    lift_in_base = world_position_to_base(
        (
            center_x,
            center_y,
            center_z + params.pregrasp_standoff_m + params.lift_height_m,
        ),
        base,
    )
    center_xy = (center_in_base[0], center_in_base[1])
    span = brick_span_m(aabb_min, aabb_max)
    reach_opening = span + params.open_extra_m
    close_opening = span - params.squeeze_m
    return PickTargets(
        pregrasp=_tcp_pose_target(
            (center_xy[0], center_xy[1], pregrasp_in_base[2]),
            orientation,
            provenance,
            params,
        ),
        approach=_tcp_pose_target(
            (center_xy[0], center_xy[1], center_in_base[2]),
            orientation,
            provenance,
            params,
        ),
        lift=_tcp_pose_target(
            (center_xy[0], center_xy[1], lift_in_base[2]),
            orientation,
            provenance,
            params,
        ),
        reach_opening_m=reach_opening,
        close_opening_m=close_opening,
        reach_expected_window_m=(
            reach_opening - params.free_window_slop_m,
            reach_opening + params.free_window_slop_m,
        ),
        close_expected_window_m=(
            span - params.contact_window_slop_m,
            span + params.contact_window_slop_m,
        ),
        brick_span_m=span,
        closing_yaw_deg=closing_yaw_deg(aabb_min, aabb_max),
    )


def load_robot_base_frame(config_path: Path, robot_id: str) -> RobotBaseFrame:
    """Read one robot's base pose from the strict user_config.json layout.

    The structure is ``Robot_Config.Robots[]``: a list of dicts keyed by
    ``"Name"`` whose ``"Robot_Base_Frame"`` carries ``"Position"`` (three
    numbers) and ``"Orientation"`` (four numbers, WXYZ order, converted to
    xyzw exactly like ``rocobrick.runtime_host_inputs``). The quaternion is
    normalized before it is stored.

    Raises:
        ValueError: When the config is unreadable or malformed, the robot
            is missing, any value is non-finite, or the quaternion has zero
            norm.
    """
    try:
        payload = json.loads(Path(config_path).read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(
            f"robot config {config_path} is unreadable: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise ValueError("robot config must be a JSON object")
    robot_config = payload.get("Robot_Config")
    if not isinstance(robot_config, Mapping):
        raise ValueError("robot config must carry a Robot_Config object")
    robots = robot_config.get("Robots")
    if not isinstance(robots, list):
        raise ValueError("Robot_Config.Robots must be a list")
    entry = None
    for raw in robots:
        if isinstance(raw, Mapping) and raw.get("Name") == robot_id:
            entry = raw
            break
    if entry is None:
        raise ValueError(f"robot {robot_id!r} is not present in {config_path}")
    base = entry.get("Robot_Base_Frame")
    if not isinstance(base, Mapping):
        raise ValueError(f"robot {robot_id} base frame must be an object")
    position = _finite_tuple(base.get("Position"), 3, f"robot {robot_id} Position")
    orientation = _finite_tuple(
        base.get("Orientation"), 4, f"robot {robot_id} Orientation"
    )
    try:
        normalized = _normalize_quat(
            (orientation[1], orientation[2], orientation[3], orientation[0])
        )
    except ValueError as error:
        raise ValueError(f"robot {robot_id} orientation is invalid: {error}") from error
    return RobotBaseFrame(position_xyz=position, orientation_xyzw=normalized)


def _finite_tuple(value: object, length: int, label: str) -> tuple[float, ...]:
    """Require ``value`` to be a sequence of exactly ``length`` finite numbers."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must contain exactly {length} numbers")
    if len(value) != length:
        raise ValueError(f"{label} must contain exactly {length} numbers")
    numbers: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{label} must contain only finite numbers")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{label} must contain only finite numbers")
        numbers.append(number)
    return tuple(numbers)


__all__ = [
    "PickTargets",
    "RobotBaseFrame",
    "SkillParams",
    "TcpProvenance",
    "brick_span_m",
    "closing_yaw_deg",
    "grasp_orientation_xyzw",
    "load_robot_base_frame",
    "load_skill_params",
    "pick_targets",
    "transit_target",
    "world_orientation_to_base",
    "world_position_to_base",
]
