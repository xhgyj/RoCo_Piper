"""Pure request-spec builders for P1 joint and link6 pose MoveGroup goals.

No ROS transport or Isaac imports: callers translate admitted targets into
moveit_msgs elsewhere. Behavioral reference: tools/p1_real_joint_probe.py
(strict goal admission with margin, sign-flip fallback, no clamping),
tools/p1_real_pose_probe.py (frame, quaternion and tolerance gates) and
artifacts/phase1-real/joint-target.json / pose-target.json.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from rocobrick.moveit_contract import (
    ADMISSION_MARGIN,
    ARM_JOINT_NAMES,
    BASE_FRAME,
    EXECUTION_POSITION_LIMITS,
    MAX_ORIENTATION_TOLERANCE_RAD,
    MAX_POSITION_TOLERANCE_M,
    QUAT_NORM_MIN,
    TCP_LINK,
    TCP_STATUS,
    TCP_STATUS_CALIBRATED_PROJECT,
    TIP_LINK,
    AdmissionDecision,
    JointTarget,
    Link6PoseTarget,
    TcpPoseTarget,
)
from rocobrick.tcp_contract import (
    EXPECTED_RUNTIME_MODEL_SHA256,
    PROJECT_TCP_CHILD_FRAME,
    PROJECT_TCP_PARENT_FRAME,
    TcpContract,
    TcpStatus,
    Transform,
)


def _is_finite(value: object) -> bool:
    """Return True when the value is a finite int/float."""
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _rejected(reason: str, details: Mapping[str, object]) -> AdmissionDecision:
    """Build a rejected decision with the given evidence."""
    return AdmissionDecision(accepted=False, reason=reason, details=details)


def normalize_quaternion(q: Sequence[float]) -> tuple[float, float, float, float]:
    """Normalize an xyzw quaternion; reject nonfinite and near-zero input.

    Raises:
        ValueError: when the sequence is not four finite components or the
            norm is below the contract near-zero threshold. No clamping is
            applied; the caller must fix the input.
    """
    if len(q) != 4:
        raise ValueError(f"quaternion must have exactly 4 components, got {len(q)}")
    values = [float(component) for component in q]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("quaternion components must all be finite")
    norm = math.sqrt(sum(value * value for value in values))
    if norm < QUAT_NORM_MIN:
        raise ValueError(
            f"near-zero quaternion (norm {norm:.3e} < {QUAT_NORM_MIN:.3e})"
        )
    return (values[0] / norm, values[1] / norm, values[2] / norm, values[3] / norm)


def build_joint_target(
    current: Mapping[str, float],
    deltas: Mapping[str, float],
    *,
    margin: float = ADMISSION_MARGIN,
) -> tuple[JointTarget | None, AdmissionDecision]:
    """Build an arm joint target from the live state and signed deltas.

    Every moved joint tries +delta first and then -delta (opposite sign when
    the baseline sits near a bound); the chosen direction is recorded. A moved
    target must sit strictly inside the narrowed execution limits with the
    margin; unchanged joints must sit inside the raw limits. Nothing is ever
    clamped: an inadmissible target is rejected BEFORE any action send.
    """
    missing = [name for name in ARM_JOINT_NAMES if name not in current]
    if missing:
        return None, _rejected(
            f"current state missing arm joints: {missing}", {"missing": tuple(missing)}
        )
    nonfinite = [name for name in ARM_JOINT_NAMES if not _is_finite(current[name])]
    if nonfinite:
        return None, _rejected(
            f"current state has non-finite positions: {nonfinite}",
            {"nonfinite": tuple(nonfinite)},
        )
    if not _is_finite(margin) or margin < 0.0:
        return None, _rejected(
            f"margin must be finite and non-negative, got {margin}", {"margin": margin}
        )
    unknown = sorted(set(deltas) - set(ARM_JOINT_NAMES))
    if unknown:
        return None, _rejected(
            f"deltas carry unknown joint names: {unknown}", {"unknown": tuple(unknown)}
        )
    nonfinite_deltas = [name for name, delta in deltas.items() if not _is_finite(delta)]
    if nonfinite_deltas:
        return None, _rejected(
            f"deltas have non-finite values: {nonfinite_deltas}",
            {"nonfinite": tuple(nonfinite_deltas)},
        )

    targets: dict[str, float] = {}
    chosen: dict[str, float] = {}
    for name, delta in deltas.items():
        lower, upper = EXECUTION_POSITION_LIMITS[name]
        base = float(current[name])
        candidate = None
        for sign in (1.0, -1.0):
            value = base + sign * float(delta)
            if lower + margin < value < upper - margin:
                candidate = value
                chosen[name] = sign * float(delta)
                break
        if candidate is None:
            return None, _rejected(
                (
                    f"{name}: no sign of delta {float(delta)} keeps the target "
                    f"inside [{lower}, {upper}] with margin {margin}"
                ),
                {
                    "joint": name,
                    "delta": float(delta),
                    "baseline": base,
                    "lower": lower,
                    "upper": upper,
                    "margin": margin,
                },
            )
        targets[name] = candidate
    for name in ARM_JOINT_NAMES:
        if name in targets:
            continue
        lower, upper = EXECUTION_POSITION_LIMITS[name]
        value = float(current[name])
        if not lower <= value <= upper:
            return None, _rejected(
                f"{name}: baseline {value} outside its execution limits "
                f"[{lower}, {upper}]",
                {"joint": name, "baseline": value, "lower": lower, "upper": upper},
            )
        targets[name] = value

    positions = {name: targets[name] for name in ARM_JOINT_NAMES}
    target = JointTarget(positions=positions)
    decision = AdmissionDecision(
        accepted=True,
        reason="ok",
        details={
            "margin": margin,
            "chosen_deltas": dict(chosen),
            "target": dict(positions),
        },
    )
    return target, decision


def build_link6_pose_target(
    *,
    frame_id: str,
    position: Sequence[float],
    orientation_xyzw: Sequence[float],
    position_tolerance_m: float = MAX_POSITION_TOLERANCE_M,
    orientation_tolerance_rad: float = MAX_ORIENTATION_TOLERANCE_RAD,
) -> tuple[Link6PoseTarget | None, AdmissionDecision]:
    """Build a link6 pose goal in base_link; the TCP stays unresolved.

    Gates: exact frame ``base_link``, three finite position values, a finite
    non-near-zero quaternion that is normalized, tip link exactly ``link6``
    and tolerances positive and inside the contract maxima. No TCP semantics
    exist beyond the recorded unresolved status.
    """
    problems: list[str] = []
    if frame_id != BASE_FRAME:
        problems.append(f"frame_id must be exactly {BASE_FRAME!r}, got {frame_id!r}")
    position_values = tuple(position)
    if len(position_values) != 3 or not all(
        _is_finite(value) for value in position_values
    ):
        problems.append("position must be exactly 3 finite values")
    try:
        quaternion = normalize_quaternion(orientation_xyzw)
    except ValueError as exc:
        problems.append(f"orientation rejected: {exc}")
        quaternion = None
    for label, tolerance, maximum in (
        ("position_tolerance_m", position_tolerance_m, MAX_POSITION_TOLERANCE_M),
        (
            "orientation_tolerance_rad",
            orientation_tolerance_rad,
            MAX_ORIENTATION_TOLERANCE_RAD,
        ),
    ):
        if not _is_finite(tolerance) or not (0.0 < float(tolerance) <= maximum):
            problems.append(
                f"{label} must be finite in (0, {maximum}], got {tolerance}"
            )
    if problems:
        return None, _rejected(
            "pose target rejected: " + "; ".join(problems),
            {"problems": tuple(problems)},
        )

    target = Link6PoseTarget(
        frame_id=frame_id,
        position=tuple(float(value) for value in position_values),
        orientation_xyzw=quaternion,
        position_tolerance_m=float(position_tolerance_m),
        orientation_tolerance_rad=float(orientation_tolerance_rad),
        tip_link=TIP_LINK,
        tcp_status=TCP_STATUS,
    )
    decision = AdmissionDecision(
        accepted=True,
        reason="ok",
        details={
            "frame_id": target.frame_id,
            "tip_link": target.tip_link,
            "tcp_status": target.tcp_status,
            "position_tolerance_m": target.position_tolerance_m,
            "orientation_tolerance_rad": target.orientation_tolerance_rad,
        },
    )
    return target, decision


def _sha256_or_none(value: object) -> str | None:
    """Return the value when it is a lowercase SHA-256 digest, else None."""
    if not isinstance(value, str):
        return None
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        return None
    return value


def _resolved_calibration_transform(
    calibration: TcpContract | Transform,
) -> Transform | None:
    """Extract the project-calibrated link6->rocobrick_tcp transform.

    Accepts a ``RESOLVED`` :class:`TcpContract` or an already-extracted
    :class:`Transform`. Any other shape, a non-resolved contract, or a
    transform with frames other than exactly link6 -> rocobrick_tcp is
    rejected (None): the project calibration is scoped to those frames.
    """
    if isinstance(calibration, TcpContract):
        if calibration.status is not TcpStatus.RESOLVED:
            return None
        transform = calibration.transform
    elif isinstance(calibration, Transform):
        transform = calibration
    else:
        return None
    if transform is None:
        return None
    if (
        transform.parent_frame != PROJECT_TCP_PARENT_FRAME
        or transform.child_frame != PROJECT_TCP_CHILD_FRAME
    ):
        return None
    return transform


def build_tcp_pose_target(
    *,
    frame_id: str,
    position: Sequence[float],
    orientation_xyzw: Sequence[float],
    calibration: TcpContract | Transform | None = None,
    runtime_model_sha256: str | None = None,
    source_sha256: str | None = None,
    position_tolerance_m: float = MAX_POSITION_TOLERANCE_M,
    orientation_tolerance_rad: float = MAX_ORIENTATION_TOLERANCE_RAD,
) -> tuple[TcpPoseTarget | None, AdmissionDecision]:
    """Build a project-calibrated TCP pose goal in base_link (R5.A).

    Provenance comes from exactly one of two artifact-derived paths that must
    agree with the frozen scope:

    - ``calibration``: a ``RESOLVED`` :class:`rocobrick.tcp_contract.TcpContract`
      or the extracted :class:`rocobrick.tcp_contract.Transform` (frames
      exactly ``link6 -> rocobrick_tcp``); its ``sha256`` pins the source
      artifact and an explicit ``runtime_model_sha256`` scopes it, or
    - explicit ``runtime_model_sha256`` + ``source_sha256`` metadata read from
      the verified artifact.

    The runtime model hash must equal the frozen expected runtime USD digest
    (the calibration is runtime-USD-scoped, never hardware validated). Pose
    gates mirror :func:`build_link6_pose_target`: exact ``base_link`` frame,
    three finite position values, a normalized quaternion and tolerances in
    ``(0, maxima]``. The position is copied verbatim — no tool offset
    arithmetic is applied here or anywhere downstream: the 63 mm link6 to TCP
    translation lives only in the generated model overlay joint.

    Returns:
        ``(TcpPoseTarget | None, AdmissionDecision)``; a rejected target is
        never constructed.
    """
    problems: list[str] = []
    transform: Transform | None = None
    if calibration is not None:
        transform = _resolved_calibration_transform(calibration)
        if transform is None:
            problems.append(
                "calibration must be a RESOLVED TcpContract or Transform for "
                f"{PROJECT_TCP_PARENT_FRAME} -> {PROJECT_TCP_CHILD_FRAME}"
            )
        else:
            source_sha256 = transform.sha256

    if source_sha256 is None:
        problems.append(
            "source_sha256 is mandatory (the accepted calibration artifact "
            "digest from the verified TcpContract/Transform or explicit "
            "artifact metadata)"
        )
    elif _sha256_or_none(source_sha256) is None:
        problems.append(f"source_sha256 must be a lowercase SHA-256 digest, got {source_sha256!r}")

    if runtime_model_sha256 is None:
        problems.append(
            "runtime_model_sha256 is mandatory (the runtime-USD scope of the "
            "calibrated TCP)"
        )
    elif _sha256_or_none(runtime_model_sha256) is None:
        problems.append(
            f"runtime_model_sha256 must be a lowercase SHA-256 digest, got "
            f"{runtime_model_sha256!r}"
        )
    elif runtime_model_sha256 != EXPECTED_RUNTIME_MODEL_SHA256:
        problems.append(
            "runtime_model_sha256 is not the expected runtime-USD scope of "
            "the project calibration"
        )

    if frame_id != BASE_FRAME:
        problems.append(f"frame_id must be exactly {BASE_FRAME!r}, got {frame_id!r}")
    position_values = tuple(position)
    if len(position_values) != 3 or not all(
        _is_finite(value) for value in position_values
    ):
        problems.append("position must be exactly 3 finite values")
    try:
        quaternion = normalize_quaternion(orientation_xyzw)
    except ValueError as exc:
        problems.append(f"orientation rejected: {exc}")
        quaternion = None
    for label, tolerance, maximum in (
        ("position_tolerance_m", position_tolerance_m, MAX_POSITION_TOLERANCE_M),
        (
            "orientation_tolerance_rad",
            orientation_tolerance_rad,
            MAX_ORIENTATION_TOLERANCE_RAD,
        ),
    ):
        if not _is_finite(tolerance) or not (0.0 < float(tolerance) <= maximum):
            problems.append(
                f"{label} must be finite in (0, {maximum}], got {tolerance}"
            )
    if problems:
        return None, _rejected(
            "tcp pose target rejected: " + "; ".join(problems),
            {"problems": tuple(problems)},
        )

    target = TcpPoseTarget(
        frame_id=frame_id,
        position=tuple(float(value) for value in position_values),
        orientation_xyzw=quaternion,
        position_tolerance_m=float(position_tolerance_m),
        orientation_tolerance_rad=float(orientation_tolerance_rad),
        tip_link=TCP_LINK,
        tcp_status=TCP_STATUS_CALIBRATED_PROJECT,
        runtime_model_sha256=runtime_model_sha256,
        source_sha256=source_sha256,
    )
    decision = AdmissionDecision(
        accepted=True,
        reason="ok",
        details={
            "frame_id": target.frame_id,
            "tip_link": target.tip_link,
            "tcp_status": target.tcp_status,
            "runtime_model_sha256": target.runtime_model_sha256,
            "source_sha256": target.source_sha256,
            "calibration_source": (
                transform.source if transform is not None else "explicit_artifact_metadata"
            ),
        },
    )
    return target, decision


def validate_target_within_limits(
    target: Mapping[str, float], *, margin: float = 0.0
) -> AdmissionDecision:
    """Check a joint mapping against the narrowed execution limits.

    Requires exactly the six canonical arm joint names with finite values
    sitting inside the limits widened by ``margin``. Never clamps.
    """
    if not _is_finite(margin) or margin < 0.0:
        raise ValueError(f"margin must be finite and non-negative, got {margin}")
    keys = set(target)
    if keys != set(ARM_JOINT_NAMES):
        missing = sorted(set(ARM_JOINT_NAMES) - keys)
        extra = sorted(keys - set(ARM_JOINT_NAMES))
        return _rejected(
            "target must carry exactly the six canonical arm joint names",
            {"missing": tuple(missing), "extra": tuple(extra)},
        )
    nonfinite = [name for name in ARM_JOINT_NAMES if not _is_finite(target[name])]
    if nonfinite:
        return _rejected(
            f"target has non-finite positions: {nonfinite}",
            {"nonfinite": tuple(nonfinite)},
        )
    excess: dict[str, float] = {}
    for name in ARM_JOINT_NAMES:
        lower, upper = EXECUTION_POSITION_LIMITS[name]
        value = float(target[name])
        over = max(lower + margin - value, value - (upper - margin), 0.0)
        if over > 0.0:
            excess[name] = over
    if excess:
        return _rejected(
            "target outside the execution limits with margin",
            {"margin": margin, "excess": excess},
        )
    return AdmissionDecision(
        accepted=True,
        reason="ok",
        details={"margin": margin, "joints": tuple(ARM_JOINT_NAMES)},
    )
