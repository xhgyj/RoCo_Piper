"""Pure structural binding between STAP plans and task structure bytes.

A binding is an identity statement, not a pose: it ties exactly one pending
PLACE_DOWN task of one validated ``assembly-plan/v2`` plan to the exact task
structure files whose canonical digests the plan records. The binding carries
grid integers and plan hashes only — no metric pose, frame, or TCP value is
derived, repaired, or inferred here. This module imports nothing from ROS,
Isaac, the metric plan contracts, or execution code.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType

from rocobrick.stap_contract import StapContractError, parse_stap_plan

MOTION_GEOMETRY_OMITTED = "omitted"
_BINDING_SHA_FIELDS = ("plan_sha256", "task_start_sha256", "task_goal_sha256")


def canonical_json_digest(payload: object) -> str:
    """Return the lowercase SHA-256 of canonical JSON bytes for ``payload``.

    The canonical form is the exact recipe the STAP contract already uses for
    plan identifiers: ``json.dumps`` with sorted keys, comma separators, and
    ASCII-only output, encoded as UTF-8. Task structure files are hashed with
    the same recipe so a plan's ``source`` digests are comparable byte-for-byte.

    Args:
        payload: A JSON-compatible object.

    Returns:
        The 64-character lowercase SHA-256 hex digest.
    """
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _sha(value: str, field: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


@dataclass(frozen=True)
class StructuralBinding:
    """Grid-level identity of one pending STAP task against one robot.

    All fields are structural: grid coordinates are integer stud/layer units,
    ``orientation`` is the quarter-turn count, and the three digests pin the
    exact plan and task structure bytes this binding was derived from.
    ``motion_geometry`` is always ``"omitted"``: a structural binding never
    carries metric pose information.
    """

    plan_id: str
    task_id: str
    physical_instance_id: str
    brick_id: int
    robot_id: str
    source_goal_key: str
    grid_xyz: tuple[int, int, int]
    orientation: int
    color: str
    motion_geometry: str
    plan_sha256: str
    task_start_sha256: str
    task_goal_sha256: str

    def __post_init__(self) -> None:
        for field in ("plan_id", "task_id", "physical_instance_id", "robot_id",
                      "source_goal_key", "color"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field} must be a non-empty string")
        object.__setattr__(self, "brick_id", _int(self.brick_id, "brick_id"))
        if self.brick_id < 0:
            raise ValueError("brick_id must not be negative")
        grid = self.grid_xyz
        if (
            not isinstance(grid, Sequence)
            or isinstance(grid, (str, bytes))
            or len(grid) != 3
        ):
            raise ValueError("grid_xyz must contain exactly three integers")
        object.__setattr__(
            self,
            "grid_xyz",
            tuple(_int(value, "grid_xyz") for value in grid),
        )
        if any(value < 0 for value in self.grid_xyz):
            raise ValueError("grid_xyz entries must not be negative")
        object.__setattr__(self, "orientation", _int(self.orientation, "orientation"))
        if self.orientation not in (0, 1):
            raise ValueError("orientation must be a quarter-turn count of 0 or 1")
        if self.motion_geometry != MOTION_GEOMETRY_OMITTED:
            raise ValueError(
                f"motion_geometry must be {MOTION_GEOMETRY_OMITTED!r}; a structural "
                "binding never carries metric motion data"
            )
        for field in _BINDING_SHA_FIELDS:
            object.__setattr__(self, field, _sha(getattr(self, field), field))


@dataclass(frozen=True)
class BindingDecision:
    """The outcome of one binding attempt.

    ``binding`` is present exactly when ``accepted`` is true. ``details``
    carries the audit trail (input digests before and after parsing, the
    allocation that was verified, and the recorded plan semantics) and is
    exposed as an immutable mapping.
    """

    accepted: bool
    reason: str
    binding: StructuralBinding | None
    details: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.accepted, bool):
            raise ValueError("accepted must be a boolean")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be a non-empty string")
        if self.accepted != (self.binding is not None):
            raise ValueError("binding must be present exactly when accepted")
        if not isinstance(self.details, Mapping):
            raise ValueError("details must be a mapping")
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))


def _decode_json(payload: bytes) -> object:
    return json.loads(payload.decode("utf-8"))


def _reject(reason: str, details: Mapping[str, object]) -> BindingDecision:
    return BindingDecision(accepted=False, reason=reason, binding=None, details=details)


def bind_single_stap_task(
    *, plan_bytes: bytes, task_start_bytes: bytes, task_goal_bytes: bytes, robot_id: str
) -> BindingDecision:
    """Bind the single pending task of a STAP plan to exact task bytes.

    The plan bytes are hashed before and after validation, validated
    authoritatively with ``parse_stap_plan`` (the metric plan contract is
    deliberately not invoked), checked against the canonical digests of the
    supplied task structure bytes, and required to allocate exactly one pending
    task to exactly ``robot_id``. Nothing is repaired, reassigned, or reordered:
    any mismatch, extra task, wrong robot, or malformed input is rejected as-is.

    Args:
        plan_bytes: Exact bytes of the ``assembly-plan/v2`` document.
        task_start_bytes: Exact bytes of the task ``structure_start.json``.
        task_goal_bytes: Exact bytes of the task ``structure_goal.json``.
        robot_id: The robot that must be the task's selected executor.

    Returns:
        A ``BindingDecision`` whose ``details`` include the immutability proof
        ``input_sha256_before`` / ``input_sha256_after`` and the explicit
        ``motion_geometry`` record.
    """
    digests_before = (
        sha256(plan_bytes).hexdigest(),
        sha256(task_start_bytes).hexdigest(),
        sha256(task_goal_bytes).hexdigest(),
    )
    details: dict[str, object] = {
        "input_sha256_before": {
            "plan": digests_before[0],
            "task_start": digests_before[1],
            "task_goal": digests_before[2],
        },
        "input_bytes": {
            "plan": len(plan_bytes),
            "task_start": len(task_start_bytes),
            "task_goal": len(task_goal_bytes),
        },
        "motion_geometry": MOTION_GEOMETRY_OMITTED,
    }
    if not isinstance(robot_id, str) or not robot_id:
        return _reject("robot_id must be a non-empty string", details)
    try:
        plan_payload = _decode_json(plan_bytes)
        start_payload = _decode_json(task_start_bytes)
        goal_payload = _decode_json(task_goal_bytes)
    except (UnicodeDecodeError, ValueError) as error:
        return _reject(f"malformed JSON input: {error}", details)
    if not isinstance(plan_payload, Mapping):
        return _reject("plan_bytes must decode to a JSON object", details)

    start_digest = canonical_json_digest(start_payload)
    goal_digest = canonical_json_digest(goal_payload)
    source = plan_payload.get("source")
    recorded_start = source.get("start_sha256") if isinstance(source, Mapping) else None
    recorded_goal = source.get("goal_sha256") if isinstance(source, Mapping) else None
    if recorded_goal != goal_digest or recorded_start != start_digest:
        details["task_sha256"] = {"start": start_digest, "goal": goal_digest}
        return _reject(
            "plan source digests do not equal the canonical digests of the "
            "supplied task structure files",
            details,
        )

    try:
        plan = parse_stap_plan(plan_payload)
    except StapContractError as error:
        return _reject(f"plan failed authoritative STAP validation: {error}", details)
    details["input_sha256_after"] = {
        "plan": sha256(plan_bytes).hexdigest(),
        "task_start": sha256(task_start_bytes).hexdigest(),
        "task_goal": sha256(task_goal_bytes).hexdigest(),
    }
    if details["input_sha256_before"] != details["input_sha256_after"]:
        return _reject("input bytes changed during validation", details)
    details["plan_id"] = plan.plan_id
    details["robots"] = list(plan.robots)

    if len(plan.tasks) != 1:
        return _reject(
            f"binding requires exactly one pending task, plan has {len(plan.tasks)}",
            details,
        )
    task = next(iter(plan.tasks.values()))
    details["pending_task_ids"] = [task.task_id]
    if task.primary_robot != robot_id:
        return _reject(
            f"task {task.task_id} is allocated to {task.primary_robot}, not {robot_id}",
            details,
        )
    if robot_id not in plan.robots:
        return _reject(f"robot {robot_id} is not declared by the plan", details)
    allocation = plan.raw.get("allocation")
    if not isinstance(allocation, Mapping):
        return _reject("plan allocation must be an object", details)
    expected_allocation = {name: [] for name in plan.robots}
    expected_allocation[robot_id] = [task.task_id]
    if allocation != expected_allocation:
        return _reject(
            "plan must allocate exactly the one pending task to the selected "
            f"robot and nothing else: expected {expected_allocation}",
            details,
        )
    details["allocation"] = {
        name: list(value) if isinstance(value, Sequence) and not isinstance(value, str)
        else value
        for name, value in allocation.items()
    }
    part = plan.goal_parts[task.physical_instance_id]
    binding = StructuralBinding(
        plan_id=plan.plan_id,
        task_id=task.task_id,
        physical_instance_id=task.physical_instance_id,
        brick_id=part.brick_id,
        robot_id=robot_id,
        source_goal_key=part.source_goal_key,
        grid_xyz=(part.x, part.y, part.z),
        orientation=part.orientation_quarter_turns,
        color=part.color,
        motion_geometry=MOTION_GEOMETRY_OMITTED,
        plan_sha256=digests_before[0],
        task_start_sha256=digests_before[1],
        task_goal_sha256=digests_before[2],
    )
    details["plan_source_sha256"] = {
        "start": plan.source_start_sha256,
        "goal": plan.source_goal_sha256,
    }
    return BindingDecision(
        accepted=True,
        reason=(
            f"task {task.task_id} is the plan's single pending PLACE_DOWN task, "
            f"allocated exactly to {robot_id}, and bound to the exact supplied "
            "task structure bytes"
        ),
        binding=binding,
        details=details,
    )


def binding_requires_tcp(binding: StructuralBinding) -> bool:
    """Whether executing the bound task requires a resolved TCP.

    Always true under the current execution semantics, for an exact reason:
    ``parse_stap_plan`` admits only ``PLACE_DOWN`` tasks — the STAP grammar's
    single assembly action — so a ``StructuralBinding`` is by construction a
    PLACE_DOWN assembly step and no skill conversion exists to model. Every
    PLACE_DOWN execution path in this repository is PICK-related work
    (pregrasp, approach, grasp, lift, place) whose metric targets are expressed
    relative to a tool point offset from ``link6``: the calibrated offsets in
    ``config/user_config.json`` and the legacy ``0.1358`` m link6 offset in
    ``run/pregrasp_v2.py``. A structural binding deliberately carries no metric
    pose, so converting the bound grid cell into a safe grasp or place
    trajectory requires a resolved tool point; while the TCP is unresolved the
    binding may only drive link6-referenced free-space planning and no grasp
    capability may be claimed.

    Args:
        binding: The structural binding to classify.

    Returns:
        True: the bound PLACE_DOWN/PICK-related execution requires a TCP.

    Raises:
        TypeError: If ``binding`` is not a ``StructuralBinding``.
    """
    if not isinstance(binding, StructuralBinding):
        raise TypeError("binding must be a StructuralBinding")
    return True
