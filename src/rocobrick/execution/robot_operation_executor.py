"""A7 zero-motion skill executor for the formal operation scheduler.

This module maps one validated
:class:`~rocobrick.robot_operation_plan.RobotOperation` onto the formal
:class:`~rocobrick.multi_robot_scheduler.RobotOperationExecutor` seam. It
consumes the frozen A2 result types unchanged and never sends motion in this
phase:

* ``WAIT`` is the only skill that can be confirmed. It polls one injected
  read-only per-robot observation transport (:class:`ObservationTransport`),
  requires the observed ``sim_time_s`` of the operation's own executor robot
  to advance by the requested ``duration_s``, and requires the requested
  ``physical_instance_id`` to exist with a measured world AABB that stays
  within ``aabb_tolerance_m`` of the first measured AABB. Only then are the
  ``real_scene_observation`` and ``wait_duration_confirmed`` labels emitted.
  Sleeping is polling only and is never evidence; a missing, malformed,
  regressed or stagnant observation is rejected against a bounded monotonic
  wall-clock deadline.
* ``PICK``, ``TRANSIT`` and ``PLACE_DOWN`` are first admitted (validated
  executor robot id and target shape) and then fail closed before any motion:
  ``terminal=True``, ``passed=False``, ``motion_sent=False``, ``evidence={}``
  and ``reason == "CAPABILITY_UNAVAILABLE"``. No required evidence label is
  ever synthesized for them.
* ``cancel(operation_id)`` returns a safe terminal
  :class:`~rocobrick.multi_robot_scheduler.CancelEvidence` only for an
  operation whose dispatch record proves that no motion was sent (every
  operation of this phase). An unknown operation id, or a record that might
  have sent motion, fails closed with ``safe=False``.

Boundaries recorded for later phases (deliberately NOT implemented here):

* ``PICK`` will reuse the measured-AABB TCP targets of
  :mod:`rocobrick.skill_targets` and the phase program of
  :mod:`rocobrick.session_phase_backend`; this executor neither imports nor
  drives them yet.
* ``TRANSIT`` can currently at most reach a measured pose above a brick
  (``skill_targets.transit_target``); carrying/connection transport is not
  implemented and must never be reported as ``carrying_state_confirmed``.
* ``PLACE_DOWN`` has only phase definitions
  (:data:`rocobrick.skills.PLACE_DOWN_PROGRAM`); no insert, connection,
  release or retreat implementation exists.
* The legacy ``--skill-request/v1`` path is not an entry point for this
  executor.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from rocobrick.multi_robot_scheduler import (
    CancelEvidence,
    JSONValue,
    OperationExecution,
    OperationStatus,
)
from rocobrick.operation_state_ledger import OperationStateLedger
from rocobrick.skills import PhaseStatus, SkillPhase, program_for
from rocobrick.robot_operation_plan import RobotOperation
from rocobrick.transit_adapter import (
    MotionExecution,
    SessionTcpMotionAdapter,
    TransitContext,
    load_transit_contexts,
    resolve_transit_target,
)

SCENE_STATE_SCHEMA = "rocobrick-scene-state/v1"
GRIPPER_JOINT_NAMES = ("gripper_joint1", "gripper_joint2")

WAIT = "WAIT"
PICK = "PICK"
TRANSIT = "TRANSIT"
PLACE_DOWN = "PLACE_DOWN"
MOTION_SKILLS = frozenset({PICK, TRANSIT, PLACE_DOWN})

LABEL_REAL_SCENE_OBSERVATION = "real_scene_observation"
LABEL_WAIT_DURATION_CONFIRMED = "wait_duration_confirmed"
WAIT_EVIDENCE_LABELS = frozenset({
    LABEL_REAL_SCENE_OBSERVATION,
    LABEL_WAIT_DURATION_CONFIRMED,
})

REASON_CAPABILITY_UNAVAILABLE = "CAPABILITY_UNAVAILABLE"
REASON_ROBOT_NOT_VALIDATED = "ROBOT_NOT_VALIDATED"
REASON_TARGET_INVALID = "TARGET_INVALID"
REASON_WAIT_PARAMETERS_INVALID = "WAIT_PARAMETERS_INVALID"
REASON_WAIT_EVIDENCE_UNSUPPORTED = "WAIT_EVIDENCE_UNSUPPORTED"
REASON_OBSERVATION_UNAVAILABLE = "WAIT_OBSERVATION_UNAVAILABLE"
REASON_OBSERVATION_STAGNANT = "WAIT_OBSERVATION_STAGNANT"
REASON_WAIT_DEADLINE_EXCEEDED = "WAIT_DEADLINE_EXCEEDED"
REASON_STRUCTURAL_FACT_UNSTABLE = "WAIT_STRUCTURAL_FACT_UNSTABLE"
REASON_OPERATION_ALREADY_ACTIVE = "OPERATION_ALREADY_ACTIVE"
REASON_CANCELLED = "CANCELLED"

_REF_KEYS = frozenset({
    "kind",
    "physical_instance_id",
    "connection_id",
    "robot_id",
})
_REF_KINDS = frozenset({"robot_current", "robot_home", "part", "connection"})
_PICK_KEYS = frozenset({"physical_instance_id"})
_TRANSIT_KEYS = frozenset({"from", "to", "carrying_physical_instance_id"})
_PLACE_DOWN_KEYS = frozenset({
    "connection_id",
    "upper_physical_instance_id",
    "lower_physical_instance_id",
})
_WAIT_PARAMETER_KEYS = frozenset({"observation", "duration_s"})
_OBSERVATION_KEYS = frozenset({"kind", "fact", "physical_instance_id"})


class ObservationTransport(Protocol):
    """Injected read-only per-robot scene observation transport (A6 seam).

    Implementations perform a read-only scene query only; they never command
    motion. The returned mapping is the frozen A4/A5 scene-state document.
    """

    async def observe(self, robot_id: str) -> Mapping[str, JSONValue]:
        """Return one A4/A5 scene-state document for ``robot_id``.

        Args:
            robot_id: One of the validated plan robot ids.

        Returns:
            The ``rocobrick-scene-state/v1`` document carrying the shared
            ``sim_time_s``/``bricks``/``connections`` keys and a
            ``robots[robot_id]`` block with that robot's ``robot_id``,
            ``sim_time_s``, ``gripper`` and ``opening_m``. There is no
            fallback to another robot's block.
        """


class _WaitParameterError(ValueError):
    """One WAIT parameter document violates the frozen observation shape."""


class _ObservationError(ValueError):
    """One scene document does not prove the A4/A5 observation contract."""


@dataclass(frozen=True)
class _Aabb:
    """One measured world-frame axis-aligned bounding box.

    Attributes:
        minimum: Measured ``aabb_min`` corner.
        maximum: Measured ``aabb_max`` corner.
    """

    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]


@dataclass(frozen=True)
class _Observation:
    """One validated robot-scoped observation of the waited instance.

    Attributes:
        sim_time_s: The observed simulation time of the executor robot.
        aabb: The measured AABB of the waited physical instance.
    """

    sim_time_s: float
    aabb: _Aabb


@dataclass(frozen=True)
class _WaitParameters:
    """Validated WAIT parameters of one operation.

    Attributes:
        duration_s: Requested observed sim-time advance in seconds (> 0).
        physical_instance_id: Physical instance whose structural fact must
            stay stable for the whole wait.
    """

    duration_s: float
    physical_instance_id: str


@dataclass
class _ActiveOperation:
    """Dispatch record proving whether one operation could have sent motion.

    Attributes:
        operation_id: Scheduler operation id.
        robot_id: Validated executor robot id.
        skill: Operation skill name.
        cancel_event: Cooperative cancel request flag.
        finished: Resolved when ``execute`` returns; ``cancel`` awaits it.
        motion_sent: True only when a motion command was actually sent. This
            module has no motion path, so it stays False; ``cancel`` fails
            closed if it is ever True.
    """

    operation_id: str
    robot_id: str
    skill: str
    cancel_event: asyncio.Event
    finished: asyncio.Future[None]
    motion_sent: bool = False


def _finite_number(value: object) -> float | None:
    """Return ``value`` as a finite float, or None when it is not one.

    Args:
        value: Candidate value; ``bool`` is not accepted as a number.

    Returns:
        The finite float, or None.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _positive_number(value: object, name: str) -> float:
    """Validate one strictly positive finite constructor knob.

    Args:
        value: Candidate value.
        name: Parameter name for the error message.

    Returns:
        The validated float.

    Raises:
        ValueError: If the value is not a positive finite number.
    """
    number = _finite_number(value)
    if number is None or number <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


def _non_negative_number(value: object, name: str) -> float:
    """Validate one finite constructor knob that may be zero.

    Args:
        value: Candidate value.
        name: Parameter name for the error message.

    Returns:
        The validated float.

    Raises:
        ValueError: If the value is not a finite number >= 0.
    """
    number = _finite_number(value)
    if number is None or number < 0:
        raise ValueError(f"{name} must be a finite number >= 0")
    return number


def _text(value: object) -> bool:
    """Return whether ``value`` is a non-empty string.

    Args:
        value: Candidate value.

    Returns:
        True only for a non-empty ``str``.
    """
    return isinstance(value, str) and bool(value)


def _reference_valid(value: object, executor_robot_id: str) -> bool:
    """Return whether one TRANSIT reference matches the frozen shape.

    Args:
        value: Candidate ``from``/``to`` reference.
        executor_robot_id: Operation executor robot id.

    Returns:
        True only for an exactly shaped reference whose ``robot_id`` (when
        present) is the executor robot.
    """
    if not isinstance(value, Mapping) or set(value) != _REF_KEYS:
        return False
    kind = value["kind"]
    if kind not in _REF_KINDS:
        return False
    physical = value["physical_instance_id"]
    connection = value["connection_id"]
    robot = value["robot_id"]
    if kind == "robot_current":
        return connection is None and physical is None and robot is None
    if kind == "robot_home":
        return (
            connection is None and physical is None and robot == executor_robot_id
        )
    if kind == "part":
        return _text(physical) and connection is None and robot is None
    return _text(connection) and physical is None and robot is None


def _target_valid(skill: str, target: object, executor_robot_id: str) -> bool:
    """Return whether ``target`` matches the frozen shape of ``skill``.

    Args:
        skill: Operation skill name.
        target: ``operation.target`` mapping, or None for WAIT.
        executor_robot_id: Operation executor robot id.

    Returns:
        True only for a target exactly shaped for ``skill``; nothing is
        repaired, defaulted or filled in.
    """
    if skill == WAIT:
        return target is None
    if not isinstance(target, Mapping):
        return False
    if skill == PICK:
        return set(target) == _PICK_KEYS and _text(target["physical_instance_id"])
    if skill == TRANSIT:
        if set(target) != _TRANSIT_KEYS:
            return False
        carrying = target["carrying_physical_instance_id"]
        if carrying is not None and not _text(carrying):
            return False
        return all(
            _reference_valid(target[key], executor_robot_id)
            for key in ("from", "to")
        )
    if skill == PLACE_DOWN:
        return set(target) == _PLACE_DOWN_KEYS and all(
            _text(target[key]) for key in _PLACE_DOWN_KEYS
        )
    return False


def _parse_wait_parameters(parameters: object) -> _WaitParameters:
    """Validate the frozen WAIT parameter document of one operation.

    Args:
        parameters: ``operation.parameters`` value.

    Returns:
        The validated duration and waited physical instance.

    Raises:
        _WaitParameterError: If a required key is missing, unknown keys are
            present, or ``duration_s``/``observation`` is unsupported.
    """
    if not isinstance(parameters, Mapping):
        raise _WaitParameterError("parameters must be an object")
    if set(parameters) != _WAIT_PARAMETER_KEYS:
        raise _WaitParameterError(
            f"parameters must carry exactly {sorted(_WAIT_PARAMETER_KEYS)}"
        )
    duration = _finite_number(parameters["duration_s"])
    if duration is None or duration <= 0:
        raise _WaitParameterError("duration_s must be a positive finite number")
    observation = parameters["observation"]
    if not isinstance(observation, Mapping):
        raise _WaitParameterError("observation must be an object")
    if set(observation) != _OBSERVATION_KEYS:
        raise _WaitParameterError(
            f"observation must carry exactly {sorted(_OBSERVATION_KEYS)}"
        )
    if observation["kind"] != "structural_fact":
        raise _WaitParameterError("observation.kind must be 'structural_fact'")
    if observation["fact"] != "stable":
        raise _WaitParameterError("observation.fact must be 'stable'")
    instance = observation["physical_instance_id"]
    if not _text(instance):
        raise _WaitParameterError(
            "observation.physical_instance_id must be a non-empty string"
        )
    return _WaitParameters(duration, instance)


def _vector3(raw: object, path: str) -> tuple[float, float, float]:
    """Validate one measured 3-vector.

    Args:
        raw: Candidate list/tuple of three finite numbers.
        path: Document path for error messages.

    Returns:
        The copied vector.

    Raises:
        _ObservationError: If the value is not a finite 3-vector.
    """
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise _ObservationError(f"{path} must be a 3-vector")
    values: list[float] = []
    for item in raw:
        number = _finite_number(item)
        if number is None:
            raise _ObservationError(f"{path} must hold finite numbers")
        values.append(number)
    return (values[0], values[1], values[2])


def _measured_aabb(brick: Mapping, physical_instance_id: str) -> _Aabb:
    """Validate and copy the measured AABB of one brick block.

    Args:
        brick: ``bricks[physical_instance_id]`` mapping.
        physical_instance_id: Instance id for error messages.

    Returns:
        The measured corners.

    Raises:
        _ObservationError: If the AABB is missing, malformed or inverted.
    """
    path = f"bricks[{physical_instance_id!r}]"
    minimum = _vector3(brick.get("aabb_min"), f"{path}.aabb_min")
    maximum = _vector3(brick.get("aabb_max"), f"{path}.aabb_max")
    if any(low > high for low, high in zip(minimum, maximum)):
        raise _ObservationError(f"{path}.aabb_min exceeds aabb_max")
    return _Aabb(minimum, maximum)


def _observation(
    robot_id: str, physical_instance_id: str, document: object
) -> _Observation:
    """Validate one A4/A5 scene document for one robot and one instance.

    Args:
        robot_id: Validated executor robot id.
        physical_instance_id: Physical instance whose AABB is measured.
        document: Candidate ``observe`` return value.

    Returns:
        The observed robot sim time and the measured instance AABB.

    Raises:
        _ObservationError: If any shared or robot-scoped field is missing,
            malformed, or owned by another robot.
    """
    if not isinstance(document, Mapping):
        raise _ObservationError("observation is not an object")
    if document.get("schema") != SCENE_STATE_SCHEMA:
        raise _ObservationError(
            f"observation schema must be {SCENE_STATE_SCHEMA!r}"
        )
    if _finite_number(document.get("sim_time_s")) is None:
        raise _ObservationError("shared sim_time_s is not a finite number")
    bricks = document.get("bricks")
    if not isinstance(bricks, Mapping):
        raise _ObservationError("shared bricks mapping is missing")
    if not isinstance(document.get("connections"), Mapping):
        raise _ObservationError("shared connections mapping is missing")
    robots = document.get("robots")
    if not isinstance(robots, Mapping):
        raise _ObservationError("robots mapping is missing")
    block = robots.get(robot_id)
    if not isinstance(block, Mapping):
        raise _ObservationError(f"robots[{robot_id!r}] block is missing")
    if block.get("robot_id") != robot_id:
        raise _ObservationError(
            f"robots[{robot_id!r}] block is owned by "
            f"{block.get('robot_id')!r}"
        )
    sim_time = _finite_number(block.get("sim_time_s"))
    if sim_time is None:
        raise _ObservationError(
            f"robots[{robot_id!r}].sim_time_s is not a finite number"
        )
    gripper = block.get("gripper")
    if not isinstance(gripper, Mapping) or set(gripper) != set(GRIPPER_JOINT_NAMES):
        raise _ObservationError(
            f"robots[{robot_id!r}].gripper must carry exactly "
            f"{sorted(GRIPPER_JOINT_NAMES)}"
        )
    for name in GRIPPER_JOINT_NAMES:
        if _finite_number(gripper[name]) is None:
            raise _ObservationError(
                f"robots[{robot_id!r}].gripper[{name!r}] is not a finite number"
            )
    opening = block.get("opening_m")
    if opening is not None and _finite_number(opening) is None:
        raise _ObservationError(
            f"robots[{robot_id!r}].opening_m is not a finite number or null"
        )
    brick = bricks.get(physical_instance_id)
    if not isinstance(brick, Mapping):
        raise _ObservationError(
            f"bricks[{physical_instance_id!r}] is missing from the observation"
        )
    return _Observation(sim_time, _measured_aabb(brick, physical_instance_id))


def _aabb_stable(reference: _Aabb, current: _Aabb, tolerance_m: float) -> bool:
    """Return whether two measured AABBs agree within a tolerance.

    Args:
        reference: First measured AABB of the wait.
        current: Later measured AABB of the same instance.
        tolerance_m: Maximum allowed absolute delta per corner coordinate.

    Returns:
        True only when all six coordinates stay within ``tolerance_m``.
    """
    return all(
        abs(now - then) <= tolerance_m
        for then, now in zip(
            reference.minimum + reference.maximum,
            current.minimum + current.maximum,
        )
    )


class SkillOperationExecutor:
    """Zero-motion :class:`RobotOperationExecutor` implementation (A7).

    The executor is constructed with the validated plan robot ids and one
    injected per-robot read-only observation transport. It sends no motion,
    writes no runtime document and imports no runtime/session/skill module;
    the scheduler owns result validation and serialization.
    """

    def __init__(
        self,
        robot_ids: Sequence[str],
        observation_transport: ObservationTransport,
        *,
        poll_interval_s: float = 0.2,
        wait_timeout_s: float = 30.0,
        cancel_timeout_s: float = 5.0,
        aabb_tolerance_m: float = 0.01,
        transit_adapters: Mapping[str, object] | None = None,
        transit_contexts: Mapping[str, TransitContext] | None = None,
        pick_backends: Mapping[str, object] | None = None,
        state_ledger: OperationStateLedger | None = None,
    ) -> None:
        """Validate the robot set, transport and bounded polling knobs.

        Args:
            robot_ids: Non-empty sequence of unique, non-empty validated plan
                robot ids; no other robot id is ever observed or executed.
            observation_transport: Injected read-only transport exposing
                ``async observe(robot_id)``.
            poll_interval_s: Sleep between two observations; sleeping is
                polling only and never evidence.
            wait_timeout_s: Monotonic wall-clock bound for an unproved WAIT;
                keep it below the scheduler's operation deadline.
            cancel_timeout_s: Bound for one cooperative cancel acknowledgement
                awaited by :meth:`cancel`.
            aabb_tolerance_m: Maximum allowed absolute AABB corner delta for
                the structural fact to count as stable.

        Raises:
            ValueError: If the robot ids are empty, duplicated or malformed,
                the transport exposes no callable ``observe``, or a numeric
                knob is out of range.
        """
        if (
            isinstance(robot_ids, (str, bytes, bytearray))
            or not isinstance(robot_ids, Sequence)
            or not robot_ids
        ):
            raise ValueError("robot_ids must be a non-empty sequence")
        validated = tuple(robot_ids)
        for robot_id in validated:
            if not _text(robot_id):
                raise ValueError("every robot id must be a non-empty string")
        if len(set(validated)) != len(validated):
            raise ValueError("robot_ids must be unique")
        if not callable(getattr(observation_transport, "observe", None)):
            raise ValueError(
                "observation_transport must expose a callable observe(robot_id)"
            )
        self._robot_ids = validated
        self._observation_transport = observation_transport
        self._poll_interval_s = _positive_number(
            poll_interval_s, "poll_interval_s"
        )
        self._wait_timeout_s = _positive_number(wait_timeout_s, "wait_timeout_s")
        self._cancel_timeout_s = _positive_number(
            cancel_timeout_s, "cancel_timeout_s"
        )
        self._aabb_tolerance_m = _non_negative_number(
            aabb_tolerance_m, "aabb_tolerance_m"
        )
        self._transit_adapters = dict(transit_adapters or {})
        self._transit_contexts = dict(transit_contexts or {})
        self._pick_backends = dict(pick_backends or {})
        self._state_ledger = state_ledger
        self._active: dict[str, _ActiveOperation] = {}
        self._finished: dict[str, bool] = {}

    @property
    def robot_ids(self) -> tuple[str, ...]:
        """The validated robot ids this executor accepts.

        Returns:
            The robot ids in constructor order.
        """
        return self._robot_ids

    async def execute(self, operation: RobotOperation) -> OperationExecution:
        """Execute one validated operation without ever sending motion.

        Args:
            operation: Operation dispatched by the formal scheduler.

        Returns:
            A frozen :class:`OperationExecution`: ``CONFIRMED`` only for a
            genuinely observed WAIT, otherwise terminal ``FAILED`` (motion
            skills fail with ``CAPABILITY_UNAVAILABLE`` before any motion) or
            a terminal ``CANCELLED`` after a cooperative cancel request.
        """
        robot_id = operation.executor_robot_id
        if operation.operation_id in self._active:
            return self._failed(
                operation, REASON_OPERATION_ALREADY_ACTIVE
            )
        if robot_id not in self._robot_ids:
            return self._failed(
                operation,
                f"{REASON_ROBOT_NOT_VALIDATED}: {robot_id!r} is not a "
                "validated robot id",
            )
        skill = operation.skill
        target = None if operation.target is None else operation.target.values
        if skill in MOTION_SKILLS:
            if not _target_valid(skill, target, robot_id):
                return self._failed(
                    operation,
                    f"{REASON_TARGET_INVALID}: {skill} target does not match "
                    "the frozen shape",
                )
            if skill == TRANSIT and robot_id in self._transit_adapters:
                return await self._transit(operation, robot_id)
            if skill == PICK and robot_id in self._pick_backends:
                return await self._pick(operation, robot_id)
            return self._failed(operation, REASON_CAPABILITY_UNAVAILABLE)
        if skill != WAIT:
            return self._failed(operation, REASON_CAPABILITY_UNAVAILABLE)
        if not _target_valid(WAIT, target, robot_id):
            return self._failed(
                operation,
                f"{REASON_TARGET_INVALID}: WAIT target must be null",
            )
        unsupported = sorted(
            set(operation.evidence_requirements) - WAIT_EVIDENCE_LABELS
        )
        if unsupported:
            return self._failed(
                operation,
                f"{REASON_WAIT_EVIDENCE_UNSUPPORTED}: {unsupported} cannot be "
                "proved by this executor",
            )
        try:
            parameters = _parse_wait_parameters(operation.parameters)
        except _WaitParameterError as error:
            return self._failed(
                operation,
                f"{REASON_WAIT_PARAMETERS_INVALID}: {error}",
            )
        return await self._wait(operation, robot_id, parameters)

    async def cancel(self, operation_id: str) -> CancelEvidence:
        """Request cancellation and report terminal safety explicitly.

        Args:
            operation_id: Operation to cancel.

        Returns:
            ``acknowledged``/``terminal``/``safe`` all True only for an
            operation whose dispatch record proves that no motion was sent
            and whose polling loop has actually stopped. An unknown id, or a
            record that may have sent motion, returns ``safe=False`` (and
            ``terminal=False`` while it cannot be proven).

        Raises:
            ValueError: If ``operation_id`` is not a non-empty string.
        """
        if not _text(operation_id):
            raise ValueError("operation_id must be a non-empty string")
        record = self._active.get(operation_id)
        if record is None:
            no_motion = self._finished.get(operation_id)
            if no_motion is False:
                return CancelEvidence(
                    operation_id=operation_id,
                    acknowledged=True,
                    terminal=True,
                    safe=True,
                    evidence={
                        "mode": "no_motion",
                        "already_finished": True,
                        "motion_sent": False,
                    },
                )
            return CancelEvidence(
                operation_id=operation_id,
                acknowledged=False,
                terminal=False,
                safe=False,
                evidence={
                    "cancel_error": (
                        "operation is unknown to this executor; no terminal "
                        "or safe state can be claimed"
                    )
                },
            )
        record.cancel_event.set()
        if record.motion_sent:
            return CancelEvidence(
                operation_id=operation_id,
                acknowledged=True,
                terminal=False,
                safe=False,
                evidence={
                    "cancel_error": (
                        "motion may be in flight and no safe stop is "
                        "implemented"
                    ),
                    "motion_sent": True,
                },
            )
        try:
            await asyncio.wait_for(
                asyncio.shield(record.finished), timeout=self._cancel_timeout_s
            )
        except TimeoutError:
            return CancelEvidence(
                operation_id=operation_id,
                acknowledged=True,
                terminal=False,
                safe=False,
                evidence={
                    "cancel_error": (
                        "operation did not acknowledge cancellation within "
                        f"{self._cancel_timeout_s} s"
                    )
                },
            )
        return CancelEvidence(
            operation_id=operation_id,
            acknowledged=True,
            terminal=True,
            safe=True,
            evidence={"mode": "no_motion", "motion_sent": False},
        )

    async def _transit(
        self, operation: RobotOperation, robot_id: str
    ) -> OperationExecution:
        """Execute one TRANSIT from a live scene snapshot through TCP MoveIt."""
        adapter = self._transit_adapters.get(robot_id)
        context = self._transit_contexts.get(robot_id)
        if adapter is None or context is None:
            return self._failed(operation, REASON_CAPABILITY_UNAVAILABLE)
        try:
            scene = await self._observe_or_cancel(robot_id, asyncio.Event())
            if scene is None:
                return self._failed(operation, "STALE_WORLD")
            target = resolve_transit_target(scene, operation, context)
        except ValueError as error:
            return self._failed(operation, str(error))
        execute = getattr(adapter, "execute", None)
        if not callable(execute):
            return self._failed(operation, "TRANSIT_ADAPTER_INVALID")
        try:
            result = await asyncio.to_thread(execute, operation, target)
        except Exception as error:
            return self._failed(operation, f"TRANSIT_ADAPTER_FAILED: {error}")
        if not isinstance(result, MotionExecution):
            return self._failed(operation, "TRANSIT_RESULT_INVALID")
        evidence: dict[str, JSONValue] = dict(result.evidence)
        target_values = operation.target.values if operation.target is not None else {}
        destination = target_values.get("to")
        destination_kind = (
            destination.get("kind")
            if isinstance(destination, Mapping)
            else None
        )
        carrying = target_values.get("carrying_physical_instance_id")
        if result.passed and result.terminal and result.safe_to_release:
            final_scene = await self._observe_or_cancel(robot_id, asyncio.Event())
            if final_scene is None or not isinstance(final_scene.get("bricks"), Mapping):
                evidence["transit_scene_feedback_confirmed"] = False
                return OperationExecution(
                    operation_id=operation.operation_id,
                    robot_id=robot_id,
                    status=OperationStatus.FAILED,
                    passed=False,
                    terminal=result.terminal,
                    motion_sent=result.motion_sent,
                    evidence=evidence,
                    reason="STALE_WORLD_AFTER_TRANSIT",
                )
            evidence["transit_scene_feedback_confirmed"] = True
            evidence["transit_feedback_confirmed"] = True
            if carrying is not None:
                evidence["carrying_state_confirmed"] = False
                return OperationExecution(
                    operation_id=operation.operation_id,
                    robot_id=robot_id,
                    status=OperationStatus.FAILED,
                    passed=False,
                    terminal=result.terminal,
                    motion_sent=result.motion_sent,
                    evidence=evidence,
                    reason="CARRYING_STATE_UNCONFIRMED",
                )
            if self._state_ledger is not None:
                key = (
                    "robot_at_pick_region"
                    if destination_kind == "part"
                    else "robot_at_place_region"
                )
                self._state_ledger.confirm(robot_id, key, evidence)
            return OperationExecution(
                operation_id=operation.operation_id,
                robot_id=robot_id,
                status=OperationStatus.CONFIRMED,
                passed=True,
                terminal=True,
                motion_sent=result.motion_sent,
                evidence=evidence,
                reason="TRANSIT_FEEDBACK_CONFIRMED",
            )
        return OperationExecution(
            operation_id=operation.operation_id,
            robot_id=robot_id,
            status=OperationStatus.FAILED,
            passed=False,
            terminal=result.terminal,
            motion_sent=result.motion_sent,
            evidence=evidence,
            reason=result.reason or "TRANSIT_ACTION_FAILED",
        )

    async def _pick(self, operation: RobotOperation, robot_id: str) -> OperationExecution:
        """Run the complete PICK phase program through the real session backend."""
        backend = self._pick_backends.get(robot_id)
        execute_phase = getattr(backend, "execute_phase", None)
        if not callable(execute_phase):
            return self._failed(operation, "PICK_BACKEND_INVALID")
        phase_evidence: list[dict[str, JSONValue]] = []
        motion_sent = False
        for phase in program_for(SkillType.PICK).phases:
            try:
                result = await execute_phase(
                    operation, phase, operation.operation_id, operation.operation_id
                )
            except Exception as error:
                return OperationExecution(operation.operation_id, robot_id, OperationStatus.FAILED,
                    False, True, motion_sent, {"phases": phase_evidence},
                    f"PICK_BACKEND_FAILED: {error}")
            status = getattr(result, "status", None)
            reason = getattr(result, "reason", None)
            phase_evidence.append({
                "phase": phase.phase_id,
                "status": str(getattr(status, "value", status)),
                "reason": None if reason is None else str(getattr(reason, "value", reason)),
            })
            if phase.phase_id in {"reach_pregrasp", "approach", "close_gripper", "lift"}:
                motion_sent = True
            if status is not PhaseStatus.CONFIRMED:
                return OperationExecution(operation.operation_id, robot_id, OperationStatus.FAILED,
                    False, True, motion_sent, {"phases": phase_evidence},
                    str(getattr(reason, "value", reason) or "PICK_PHASE_FAILED"))
        evidence: dict[str, JSONValue] = {
            "phases": phase_evidence,
            "pregrasp_confirmed": True,
            "approach_confirmed": True,
            "gripper_feedback_confirmed": True,
            "grasp_confirmed": True,
            "lift_confirmed": True,
            "holding_confirmed": True,
        }
        instance_id = operation.target.values.get("physical_instance_id") if operation.target else None
        if self._state_ledger is not None and isinstance(instance_id, str):
            self._state_ledger.confirm(robot_id, f"holding:{instance_id}", evidence)
        return OperationExecution(operation.operation_id, robot_id, OperationStatus.CONFIRMED,
            True, True, motion_sent, evidence, "PICK_CONFIRMED")

    async def _wait(
        self,
        operation: RobotOperation,
        robot_id: str,
        parameters: _WaitParameters,
    ) -> OperationExecution:
        """Run one bounded, observation-proved WAIT.

        Args:
            operation: The dispatched WAIT operation.
            robot_id: Validated executor robot id.
            parameters: Validated WAIT duration and instance.

        Returns:
            The terminal execution result of the wait.
        """
        record = _ActiveOperation(
            operation_id=operation.operation_id,
            robot_id=robot_id,
            skill=operation.skill,
            cancel_event=asyncio.Event(),
            finished=asyncio.get_running_loop().create_future(),
        )
        self._active[operation.operation_id] = record
        try:
            return await self._poll_until_proved(
                operation, robot_id, parameters, record
            )
        finally:
            self._active.pop(operation.operation_id, None)
            self._finished[operation.operation_id] = record.motion_sent
            if not record.finished.done():
                record.finished.set_result(None)

    async def _poll_until_proved(
        self,
        operation: RobotOperation,
        robot_id: str,
        parameters: _WaitParameters,
        record: _ActiveOperation,
    ) -> OperationExecution:
        """Poll the transport until the wait is proved or bounded out.

        Args:
            operation: The dispatched WAIT operation.
            robot_id: Validated executor robot id.
            parameters: Validated WAIT duration and instance.
            record: Active dispatch record of this operation.

        Returns:
            ``CONFIRMED`` with both wait labels only when the observed robot
            sim time advanced by ``duration_s`` while the measured AABB of
            the requested instance stayed stable; otherwise terminal
            ``FAILED`` (or terminal ``CANCELLED``).
        """
        deadline = time.monotonic() + self._wait_timeout_s
        try:
            document = await self._observe_or_cancel(robot_id, record.cancel_event)
        except asyncio.CancelledError:
            raise
        except Exception:
            return self._failed(operation, REASON_OBSERVATION_UNAVAILABLE)
        if document is None:
            return self._cancelled(operation)
        try:
            first = _observation(
                robot_id, parameters.physical_instance_id, document
            )
        except _ObservationError:
            return self._failed(operation, REASON_OBSERVATION_UNAVAILABLE)
        previous = first
        polls = 1
        while previous.sim_time_s - first.sim_time_s < parameters.duration_s:
            if record.cancel_event.is_set():
                return self._cancelled(operation)
            if time.monotonic() >= deadline:
                reason = (
                    REASON_OBSERVATION_STAGNANT
                    if previous.sim_time_s <= first.sim_time_s
                    else REASON_WAIT_DEADLINE_EXCEEDED
                )
                return self._failed(operation, reason)
            # Polling sleep only: it is never itself evidence of the wait.
            await asyncio.sleep(self._poll_interval_s)
            try:
                document = await self._observe_or_cancel(
                    robot_id, record.cancel_event
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                return self._failed(operation, REASON_OBSERVATION_UNAVAILABLE)
            if document is None:
                return self._cancelled(operation)
            try:
                current = _observation(
                    robot_id, parameters.physical_instance_id, document
                )
            except _ObservationError:
                return self._failed(operation, REASON_OBSERVATION_UNAVAILABLE)
            if current.sim_time_s < previous.sim_time_s:
                return self._failed(operation, REASON_OBSERVATION_STAGNANT)
            if not _aabb_stable(
                first.aabb, current.aabb, self._aabb_tolerance_m
            ):
                return self._failed(
                    operation, REASON_STRUCTURAL_FACT_UNSTABLE
                )
            previous = current
            polls += 1
        if record.cancel_event.is_set():
            return self._cancelled(operation)
        evidence: dict[str, JSONValue] = {
            LABEL_REAL_SCENE_OBSERVATION: {
                "robot_id": robot_id,
                "physical_instance_id": parameters.physical_instance_id,
                "polls": polls,
                "first_sim_time_s": first.sim_time_s,
                "last_sim_time_s": previous.sim_time_s,
                "aabb_min": list(first.aabb.minimum),
                "aabb_max": list(first.aabb.maximum),
                "aabb_tolerance_m": self._aabb_tolerance_m,
            },
            LABEL_WAIT_DURATION_CONFIRMED: {
                "requested_duration_s": parameters.duration_s,
                "observed_advance_s": previous.sim_time_s - first.sim_time_s,
                "first_sim_time_s": first.sim_time_s,
                "last_sim_time_s": previous.sim_time_s,
            },
        }
        return OperationExecution(
            operation_id=operation.operation_id,
            robot_id=robot_id,
            status=OperationStatus.CONFIRMED,
            passed=True,
            terminal=True,
            motion_sent=False,
            evidence=evidence,
            reason=(
                "observed sim_time_s advanced by >= duration_s with a stable "
                f"measured AABB of {parameters.physical_instance_id!r}"
            ),
        )

    async def _observe_or_cancel(
        self, robot_id: str, cancel_event: asyncio.Event
    ) -> Mapping[str, JSONValue] | None:
        """Await one read-only observation, or None when cancel wins.

        Args:
            robot_id: Validated executor robot id.
            cancel_event: Cooperative cancel flag of the operation.

        Returns:
            The returned scene document, or None when cancellation was
            requested before the observation completed.

        Raises:
            Exception: Whatever the injected transport raised.
        """
        observe_task = asyncio.ensure_future(
            self._observation_transport.observe(robot_id)
        )
        cancel_task = asyncio.ensure_future(cancel_event.wait())
        try:
            done, _ = await asyncio.wait(
                {observe_task, cancel_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
        except BaseException:
            observe_task.cancel()
            cancel_task.cancel()
            await asyncio.gather(
                observe_task, cancel_task, return_exceptions=True
            )
            raise
        if observe_task in done:
            cancel_task.cancel()
            await asyncio.gather(cancel_task, return_exceptions=True)
            return observe_task.result()
        observe_task.cancel()
        await asyncio.gather(observe_task, return_exceptions=True)
        return None

    @staticmethod
    def _failed(operation: RobotOperation, reason: str) -> OperationExecution:
        """Build one terminal no-motion failure without any evidence label.

        Args:
            operation: The dispatched operation.
            reason: Explicit failure reason.

        Returns:
            A frozen terminal ``FAILED`` execution with empty evidence.
        """
        return OperationExecution(
            operation_id=operation.operation_id,
            robot_id=operation.executor_robot_id,
            status=OperationStatus.FAILED,
            passed=False,
            terminal=True,
            motion_sent=False,
            evidence={},
            reason=reason,
        )

    @staticmethod
    def _cancelled(operation: RobotOperation) -> OperationExecution:
        """Build the cooperative-cancel terminal result.

        Args:
            operation: The dispatched operation.

        Returns:
            A frozen terminal ``CANCELLED`` execution with no motion claim
            beyond the explicit cancellation marker.
        """
        return OperationExecution(
            operation_id=operation.operation_id,
            robot_id=operation.executor_robot_id,
            status=OperationStatus.CANCELLED,
            passed=False,
            terminal=True,
            motion_sent=False,
            evidence={"cancel_requested": True},
            reason=REASON_CANCELLED,
        )


__all__ = [
    "GRIPPER_JOINT_NAMES",
    "LABEL_REAL_SCENE_OBSERVATION",
    "LABEL_WAIT_DURATION_CONFIRMED",
    "MOTION_SKILLS",
    "ObservationTransport",
    "REASON_CANCELLED",
    "REASON_CAPABILITY_UNAVAILABLE",
    "REASON_OBSERVATION_STAGNANT",
    "REASON_OBSERVATION_UNAVAILABLE",
    "REASON_OPERATION_ALREADY_ACTIVE",
    "REASON_ROBOT_NOT_VALIDATED",
    "REASON_STRUCTURAL_FACT_UNSTABLE",
    "REASON_TARGET_INVALID",
    "REASON_WAIT_DEADLINE_EXCEEDED",
    "REASON_WAIT_EVIDENCE_UNSUPPORTED",
    "REASON_WAIT_PARAMETERS_INVALID",
    "SCENE_STATE_SCHEMA",
    "SkillOperationExecutor",
    "WAIT_EVIDENCE_LABELS",
]
