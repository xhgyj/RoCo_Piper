"""Strict tool-center-point resolution contract for the Piper L model.

The TCP is a metrological claim: only a transform that an official pinned
URDF/USD source both fixes explicitly and designates as the tool point may
resolve it. Local calibration offsets, historical runtime constructs, and
geometric inference from link geometry are recorded as evidence but can never
resolve the TCP, and two disagreeing official designations are a conflict.

One narrow exception exists: a *project calibration* — a live, hash-pinned,
simulation-only measurement of the symmetric opposed contact-patch centroids
on the current runtime model — may resolve the TCP when every condition of
``_project_calibration_transform`` holds. Its authority is explicitly
``project`` (never official), scoped to the exact runtime model hash, and
never hardware validated; a disagreeing official designation still wins as a
``CONFLICT``. This module is pure: it imports nothing from ROS, Isaac, or
execution code and performs no file or network access.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rocobrick.compat import StrEnum

QUATERNION_NORM_TOLERANCE = 1e-6

KIND_PINNED_EXPLICIT_TRANSFORM = "pinned_explicit_transform"
KIND_PINNED_GEOMETRY = "pinned_geometry"
KIND_LOCAL_CONFIG = "local_config"
KIND_HISTORICAL_RUNTIME = "historical_runtime"
KIND_LEGACY_ASSUMPTION = "legacy_assumption"
KIND_PROJECT_CALIBRATION = "project_calibration"
EVIDENCE_KINDS = (
    KIND_PINNED_EXPLICIT_TRANSFORM,
    KIND_PINNED_GEOMETRY,
    KIND_LOCAL_CONFIG,
    KIND_HISTORICAL_RUNTIME,
    KIND_LEGACY_ASSUMPTION,
    KIND_PROJECT_CALIBRATION,
)

# Fixed vocabulary a passed project calibration must declare.
PROJECT_TCP_METHOD = "symmetric-opposed-contact-patch-centroids"
PROJECT_TCP_SCOPE = "runtime_model_hash"
PROJECT_TCP_PARENT_FRAME = "link6"
PROJECT_TCP_CHILD_FRAME = "rocobrick_tcp"
# The only runtime model the project calibration may ever resolve against:
# the frozen runtime USD bytes pinned in the model contract.
EXPECTED_RUNTIME_MODEL_SHA256 = (
    "a781d52f9d55291fca9c8f2695abf9e311640d996314d272f47639d5323e2375"
)


class TcpContractError(ValueError):
    """Raised when TCP evidence or a contract is structurally invalid."""


class UnresolvedTcpError(RuntimeError):
    """Raised when execution requires a TCP that the contract did not resolve."""


class TcpStatus(StrEnum):
    """Resolution state of the tool-center-point claim."""

    RESOLVED = "RESOLVED"
    UNRESOLVED = "UNRESOLVED"
    CONFLICT = "CONFLICT"


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise TcpContractError(f"{field} must be a non-empty string")
    return value


def _sha(value: object, field: str) -> str:
    result = _string(value, field)
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise TcpContractError(f"{field} must be a lowercase SHA-256 digest")
    return result


def _float_tuple(value: object, field: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TcpContractError(f"{field} must be a sequence of {length} numbers")
    if len(value) != length:
        raise TcpContractError(f"{field} must contain exactly {length} numbers")
    numbers: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TcpContractError(f"{field} entries must be numbers")
        number = float(item)
        if not math.isfinite(number):
            raise TcpContractError(f"{field} entries must be finite")
        numbers.append(number)
    return tuple(numbers)


def _translation(value: object, field: str) -> tuple[float, float, float]:
    result = _float_tuple(value, field, 3)
    return result[0], result[1], result[2]


def _rotation(value: object, field: str) -> tuple[float, float, float, float]:
    result = _float_tuple(value, field, 4)
    rotation = result[0], result[1], result[2], result[3]
    norm = math.sqrt(sum(component * component for component in rotation))
    if not math.isfinite(norm) or abs(norm - 1.0) > QUATERNION_NORM_TOLERANCE:
        raise TcpContractError(f"{field} must be a normalized quaternion")
    return rotation


@dataclass(frozen=True)
class Transform:
    """A rigid transform between two distinct frames with provenance.

    ``source`` names the file or record the transform was read from and
    ``sha256`` is the lowercase digest of exactly that source, so a transform
    can always be traced back to immutable bytes.
    """

    parent_frame: str
    child_frame: str
    translation_m: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]
    source: str
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "parent_frame", _string(self.parent_frame, "parent_frame")
        )
        object.__setattr__(
            self, "child_frame", _string(self.child_frame, "child_frame")
        )
        if self.parent_frame == self.child_frame:
            raise TcpContractError("parent_frame and child_frame must be distinct")
        object.__setattr__(
            self, "translation_m", _translation(self.translation_m, "translation_m")
        )
        object.__setattr__(
            self, "rotation_xyzw", _rotation(self.rotation_xyzw, "rotation_xyzw")
        )
        object.__setattr__(self, "source", _string(self.source, "source"))
        object.__setattr__(self, "sha256", _sha(self.sha256, "sha256"))

    @property
    def identity(self) -> tuple[str, str, tuple[float, ...], tuple[float, ...]]:
        """Value identity used to detect conflicting transform claims."""
        return (
            self.parent_frame,
            self.child_frame,
            self.translation_m,
            self.rotation_xyzw,
        )


@dataclass(frozen=True)
class TcpContract:
    """The classified TCP claim for one robot model.

    ``transform`` is present only when ``status`` is ``RESOLVED``.
    ``candidate_frames`` lists every frame the evidence mentions, ``evidence``
    lists the evidence item identifiers considered, and ``blockers`` records
    every reason the strict resolution rule did or did not pass.
    """

    robot_model: str
    status: TcpStatus
    transform: Transform | None
    candidate_frames: tuple[str, ...]
    evidence: tuple[str, ...]
    blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "robot_model", _string(self.robot_model, "robot_model")
        )
        if not isinstance(self.status, TcpStatus):
            raise TcpContractError("status must be a TcpStatus member")
        if self.transform is not None and not isinstance(self.transform, Transform):
            raise TcpContractError("transform must be a Transform or None")
        if self.status is TcpStatus.RESOLVED and self.transform is None:
            raise TcpContractError("a RESOLVED contract must carry a transform")
        if self.status is not TcpStatus.RESOLVED and self.transform is not None:
            raise TcpContractError("only a RESOLVED contract may carry a transform")
        for field in ("candidate_frames", "evidence", "blockers"):
            value = getattr(self, field)
            if not isinstance(value, tuple) or not all(
                isinstance(item, str) and item for item in value
            ):
                raise TcpContractError(f"{field} must be a tuple of non-empty strings")
        object.__setattr__(
            self, "candidate_frames", tuple(dict.fromkeys(self.candidate_frames))
        )
        object.__setattr__(self, "evidence", tuple(dict.fromkeys(self.evidence)))
        object.__setattr__(self, "blockers", tuple(dict.fromkeys(self.blockers)))
        if self.status is TcpStatus.RESOLVED:
            if self.blockers:
                raise TcpContractError("a RESOLVED contract must not carry blockers")
        elif not self.blockers:
            raise TcpContractError(
                "an UNRESOLVED or CONFLICT contract must record blockers"
            )


def _items(evidence: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    raw = evidence.get("items")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise TcpContractError("evidence.items must be an array")
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise TcpContractError(f"evidence.items[{index}] must be an object")
    return tuple(raw)


def _item_kind(item: Mapping[str, object], index: int) -> str:
    kind = _string(item.get("kind"), f"evidence.items[{index}].kind")
    if kind not in EVIDENCE_KINDS:
        raise TcpContractError(f"evidence.items[{index}].kind {kind!r} is unsupported")
    return kind


def _item_identifier(item: Mapping[str, object], index: int) -> str:
    return _string(item.get("id"), f"evidence.items[{index}].id")


def _item_frames(
    item: Mapping[str, object], index: int
) -> tuple[str, str] | None:
    parent, child = item.get("parent_frame"), item.get("child_frame")
    if parent is None and child is None:
        return None
    return (
        _string(parent, f"evidence.items[{index}].parent_frame"),
        _string(child, f"evidence.items[{index}].child_frame"),
    )


def _pinned_sources(evidence: Mapping[str, object]) -> Mapping[str, str]:
    raw = evidence.get("pinned_sources", {})
    if not isinstance(raw, Mapping):
        raise TcpContractError("evidence.pinned_sources must be an object")
    for path, digest in raw.items():
        _string(path, "evidence.pinned_sources key")
        _sha(digest, f"evidence.pinned_sources[{path!r}]")
    return raw


def _project_calibration_transform(
    item: Mapping[str, object],
    index: int,
    item_id: str,
    frames: tuple[str, str] | None,
    pinned_sources: Mapping[str, str],
    notes: list[str],
) -> Transform | None:
    """Evaluate one declared project calibration item, purely.

    A project calibration may resolve the TCP only when every condition
    holds: the calibration passed; the method is the frozen
    symmetric-opposed-contact-patch-centroids measurement; the scope is the
    runtime model hash with ``hardware_validated`` false; the selector source
    and artifact digests are present and the selector checks passed; the
    frames are exactly ``link6 -> rocobrick_tcp``; the cited calibration
    artifact is a pinned source whose digest matches; and the claimed runtime
    model hash equals the frozen expected runtime USD hash and is itself one
    of the pinned source digests. Any failed condition is recorded as a
    blocker and the item can never resolve or conflict the claim.

    Returns:
        The transform offered by the item, or ``None`` when the item failed
        a resolution condition (in which case a blocker was recorded).
    """
    prefix = f"evidence.items[{index}]"
    passed = item.get("passed")
    if not isinstance(passed, bool):
        raise TcpContractError(f"{prefix}.passed must be a boolean")
    if not passed:
        notes.append(
            f"project calibration {item_id!r} records passed=false; a failed "
            "project calibration can never resolve the TCP"
        )
        return None
    method = _string(item.get("method"), f"{prefix}.method")
    scope = _string(item.get("scope"), f"{prefix}.scope")
    hardware_validated = item.get("hardware_validated")
    if not isinstance(hardware_validated, bool):
        raise TcpContractError(f"{prefix}.hardware_validated must be a boolean")
    selector_source_sha256 = _sha(
        item.get("selector_source_sha256"), f"{prefix}.selector_source_sha256"
    )
    selector_artifact_sha256 = _sha(
        item.get("selector_artifact_sha256"), f"{prefix}.selector_artifact_sha256"
    )
    selector_checks_passed = item.get("selector_checks_passed")
    if not isinstance(selector_checks_passed, bool):
        raise TcpContractError(f"{prefix}.selector_checks_passed must be a boolean")
    runtime_model_sha256 = _sha(
        item.get("runtime_model_sha256"), f"{prefix}.runtime_model_sha256"
    )

    def _blocked(reason: str) -> None:
        notes.append(f"project calibration {item_id!r} cannot resolve: {reason}")

    if method != PROJECT_TCP_METHOD:
        _blocked(f"method {method!r} is not {PROJECT_TCP_METHOD!r}")
        return None
    if scope != PROJECT_TCP_SCOPE:
        _blocked(f"scope {scope!r} is not {PROJECT_TCP_SCOPE!r}")
        return None
    if hardware_validated:
        _blocked("hardware_validated must be false; the calibration is "
                 "simulation-only and must never claim hardware authority")
        return None
    if not selector_checks_passed:
        _blocked("selector_checks_passed is false")
        return None
    if frames is None:
        _blocked("the item carries no parent/child frames")
        return None
    if (
        frames[0] != PROJECT_TCP_PARENT_FRAME
        or frames[1] != PROJECT_TCP_CHILD_FRAME
    ):
        _blocked(
            f"frames {frames[0]} -> {frames[1]} are not "
            f"{PROJECT_TCP_PARENT_FRAME} -> {PROJECT_TCP_CHILD_FRAME}"
        )
        return None
    if runtime_model_sha256 != EXPECTED_RUNTIME_MODEL_SHA256:
        _blocked(
            "runtime_model_sha256 does not equal the expected runtime model "
            "hash"
        )
        return None
    if runtime_model_sha256 not in set(pinned_sources.values()):
        _blocked(
            "runtime_model_sha256 does not match any pinned runtime model "
            "source"
        )
        return None
    source = _string(item.get("source"), f"{prefix}.source")
    digest = _sha(item.get("source_sha256"), f"{prefix}.source_sha256")
    if pinned_sources.get(source) != digest:
        _blocked(
            f"the cited calibration artifact {source} is not pinned with this "
            "exact digest"
        )
        return None
    try:
        return Transform(
            parent_frame=frames[0],
            child_frame=frames[1],
            translation_m=item.get("translation_m", ()),
            rotation_xyzw=item.get("rotation_xyzw", ()),
            source=source,
            sha256=digest,
        )
    except TcpContractError as error:
        _blocked(f"the transform is invalid: {error}")
        return None


def assess_piper_l_tcp(evidence: Mapping[str, object]) -> TcpContract:
    """Classify the Piper L TCP claim from structured evidence, purely.

    The evidence mapping carries ``robot_model``, an optional
    ``pinned_sources`` mapping of pinned official file path to SHA-256, and an
    ``items`` array. Every item must declare one of the five evidence kinds and
    an identifier, and may carry frames, translation, quaternion, the source it
    was read from with that source's digest, and whether its source itself
    designates the child frame as the tool point (``declared_tool_frame``).

    Resolution rule: the contract is ``RESOLVED`` only when exactly one
    distinct transform is offered either by items of kind
    ``pinned_explicit_transform`` whose ``declared_tool_frame`` is true, whose
    source digest matches the pinned sources, and whose frames, translation,
    and quaternion are valid, or by one fully valid passed
    ``project_calibration`` item (see ``_project_calibration_transform``),
    with no other official or passed project designation disagreeing. Two such
    distinct designations yield ``CONFLICT``. Local configuration,
    historical runtime constructs, failed project calibrations, and geometric
    inference are recorded as evidence and, when the claim fails, as blockers,
    but they can never resolve or conflict the claim. A ``RESOLVED`` contract
    carries no blockers; its provenance makes the resolution authority clear
    (the transform's ``source`` names the official pinned file or the project
    calibration artifact).

    Args:
        evidence: The structured evidence mapping described above.

    Returns:
        The deterministic ``TcpContract`` classification for the model.

    Raises:
        TcpContractError: If the evidence mapping or any item is malformed.
    """
    robot_model = _string(evidence.get("robot_model"), "evidence.robot_model")
    pinned_sources = _pinned_sources(evidence)
    candidate_frames: list[str] = []
    evidence_ids: list[str] = []
    notes: list[str] = []
    candidates: list[Transform] = []
    conflicts: list[str] = []
    for index, item in enumerate(_items(evidence)):
        item_id = _item_identifier(item, index)
        kind = _item_kind(item, index)
        evidence_ids.append(item_id)
        frames = _item_frames(item, index)
        if frames is not None:
            candidate_frames.append(frames[1])
        declared = item.get("declared_tool_frame", False)
        if not isinstance(declared, bool):
            raise TcpContractError(
                f"evidence.items[{index}].declared_tool_frame must be a boolean"
            )
        if kind == KIND_PROJECT_CALIBRATION:
            if not declared:
                notes.append(
                    f"evidence {item_id!r} (project_calibration) does not "
                    "declare its child frame as a tool frame; a project "
                    "calibration must declare it explicitly to be evaluated"
                )
                continue
            transform = _project_calibration_transform(
                item, index, item_id, frames, pinned_sources, notes
            )
            if transform is None:
                continue
        elif kind != KIND_PINNED_EXPLICIT_TRANSFORM:
            if declared:
                notes.append(
                    f"evidence {item_id!r} ({kind}) claims a tool frame but a "
                    f"{kind} source can never designate the official TCP"
                )
            continue
        else:
            if frames is None:
                raise TcpContractError(
                    f"evidence.items[{index}] must carry parent and child frames"
                )
            if not declared:
                source = _string(
                    item.get("source"), f"evidence.items[{index}].source"
                )
                notes.append(
                    f"pinned explicit transform {item_id!r} in {source} fixes "
                    f"{frames[0]} -> {frames[1]} but does not designate a tool frame"
                )
                continue
            source = _string(item.get("source"), f"evidence.items[{index}].source")
            digest = _sha(
                item.get("source_sha256"), f"evidence.items[{index}].source_sha256"
            )
            if pinned_sources.get(source) != digest:
                notes.append(
                    f"evidence {item_id!r} cites {source} but its digest does not "
                    "match the pinned official sources"
                )
                continue
            try:
                transform = Transform(
                    parent_frame=frames[0],
                    child_frame=frames[1],
                    translation_m=item.get("translation_m", ()),
                    rotation_xyzw=item.get("rotation_xyzw", ()),
                    source=source,
                    sha256=digest,
                )
            except TcpContractError as error:
                notes.append(
                    f"evidence {item_id!r} has an invalid transform: {error}"
                )
                continue
        for existing in candidates:
            if existing.identity != transform.identity:
                conflicts.append(
                    f"official tool designations disagree: {existing.parent_frame}"
                    f" -> {existing.child_frame} from {existing.source} versus "
                    f"{transform.parent_frame} -> {transform.child_frame} from "
                    f"{transform.source}"
                )
        candidates.append(transform)
    unique: dict[tuple[str, str, tuple[float, ...], tuple[float, ...]], Transform] = {}
    for transform in candidates:
        unique.setdefault(transform.identity, transform)
    blockers: list[str] = []
    if conflicts:
        status = TcpStatus.CONFLICT
        transform = None
        blockers.extend(conflicts)
        blockers.extend(notes)
    elif len(unique) == 1:
        status = TcpStatus.RESOLVED
        transform = next(iter(unique.values()))
    else:
        status = TcpStatus.UNRESOLVED
        transform = None
        blockers.extend(notes)
        blockers.append(
            f"no pinned official source or valid passed project calibration "
            f"designates a TCP transform for {robot_model}"
        )
    return TcpContract(
        robot_model=robot_model,
        status=status,
        transform=transform,
        candidate_frames=tuple(candidate_frames),
        evidence=tuple(evidence_ids),
        blockers=tuple(blockers),
    )


def require_resolved_tcp(contract: TcpContract) -> Transform:
    """Return the resolved TCP transform or raise ``UnresolvedTcpError``.

    Raises:
        UnresolvedTcpError: If the contract is not ``RESOLVED`` or its
            transform no longer validates.
    """
    if not isinstance(contract, TcpContract):
        raise TcpContractError("contract must be a TcpContract")
    if contract.status is not TcpStatus.RESOLVED or contract.transform is None:
        raise UnresolvedTcpError(
            f"TCP for {contract.robot_model} is {contract.status}: "
            + "; ".join(contract.blockers)
        )
    # Re-validate so a resolved contract can never yield a stale transform.
    Transform(
        parent_frame=contract.transform.parent_frame,
        child_frame=contract.transform.child_frame,
        translation_m=contract.transform.translation_m,
        rotation_xyzw=contract.transform.rotation_xyzw,
        source=contract.transform.source,
        sha256=contract.transform.sha256,
    )
    return contract.transform
