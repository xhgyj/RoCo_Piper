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
* ``PLACE_DOWN`` dispatches to an injected per-robot backend only after the
  ledger confirms the carrying state; the backend owns the separated
  connection, release and retreat evidence.
* The legacy ``--skill-request/v1`` path is not an entry point for this
  executor.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from rocobrick.coordinated_layer import coordinated_layer_group
from rocobrick.multi_robot_scheduler import (
    CancelEvidence,
    JSONValue,
    OperationExecution,
    OperationStatus,
)
from rocobrick.operation_state_ledger import OperationStateLedger
from rocobrick.robot_operation_plan import RobotOperation
from rocobrick.skills.observation import (
    Pose,
    _pose,
    parse_pick_observation,
    parse_place_observation,
)
from rocobrick.skills.placement_geometry import interface_place_up_targets
from rocobrick.skills.targets import (
    _tcp_pose_target,
    world_orientation_to_base,
    world_position_to_base,
)
from rocobrick.transit_adapter import TransitContext
from rocobrick.unified_dual_arm_runtime import (
    SharedSceneSnapshot,
    SuccessorPrePhaseEvidence,
    SuccessorPrePhaseRequest,
    TerminalPrePhasePolicy,
    failed_pre_phase,
)

SCENE_STATE_SCHEMA = "rocobrick-scene-state/v1"
GRIPPER_JOINT_NAMES = ("gripper_joint1", "gripper_joint2")

WAIT = "WAIT"
PICK = "PICK"
PLACE_DOWN = "PLACE_DOWN"
PLACE_UP = "PLACE_UP"
SUPPORT_TOP = "SUPPORT_TOP"
HANDOVER = "HANDOVER"
MOTION_SKILLS = frozenset(
    {PICK, HANDOVER, PLACE_DOWN, PLACE_UP, SUPPORT_TOP}
)

LABEL_REAL_SCENE_OBSERVATION = "real_scene_observation"
LABEL_WAIT_DURATION_CONFIRMED = "wait_duration_confirmed"
WAIT_EVIDENCE_LABELS = frozenset(
    {
        LABEL_REAL_SCENE_OBSERVATION,
        LABEL_WAIT_DURATION_CONFIRMED,
    }
)

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

_REF_KEYS = frozenset(
    {
        "kind",
        "physical_instance_id",
        "connection_id",
        "robot_id",
    }
)
_REF_KINDS = frozenset({"robot_current", "robot_home", "part", "connection"})
_PICK_KEYS = frozenset({"physical_instance_id"})
_HANDOVER_KEYS = frozenset(
    {"physical_instance_id", "donor_robot_id", "receiver_robot_id"}
)
_PLACE_DOWN_KEYS = frozenset(
    {
        "connection_id",
        "upper_physical_instance_id",
        "lower_physical_instance_id",
    }
)
_PLACE_UP_KEYS = frozenset(
    {
        "connection_id",
        "upper_physical_instance_id",
        "lower_physical_instance_id",
        "support_robot_id",
    }
)
_SUPPORT_TOP_KEYS = frozenset(
    {
        "connection_id",
        "upper_physical_instance_id",
        "lower_physical_instance_id",
        "placement_robot_id",
    }
)
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
        return connection is None and physical is None and robot == executor_robot_id
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
    if skill == HANDOVER:
        return (
            set(target) == _HANDOVER_KEYS
            and all(_text(target[key]) for key in _HANDOVER_KEYS)
            and target["donor_robot_id"] != target["receiver_robot_id"]
            and executor_robot_id == target["receiver_robot_id"]
        )
    if skill == PLACE_DOWN:
        return set(target) == _PLACE_DOWN_KEYS and all(
            _text(target[key]) for key in _PLACE_DOWN_KEYS
        )
    if skill == PLACE_UP:
        return set(target) == _PLACE_UP_KEYS and all(
            _text(target[key]) for key in _PLACE_UP_KEYS
        )
    if skill == SUPPORT_TOP:
        return set(target) == _SUPPORT_TOP_KEYS and all(
            _text(target[key]) for key in _SUPPORT_TOP_KEYS
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
        raise _ObservationError(f"observation schema must be {SCENE_STATE_SCHEMA!r}")
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
            f"robots[{robot_id!r}] block is owned by {block.get('robot_id')!r}"
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


def _pose_from_mapping(value: object) -> Pose:
    """Parse one interface rigid pose.

    Returns:
        The validated rigid pose.
    """
    if not isinstance(value, Mapping):
        raise ValueError("interface pose is missing")
    return _pose(value, "interface pose")


def _tcp_target_mapping(target: object, robot_id: str) -> Mapping[str, object]:
    """Serialize one calibrated TCP target.

    Returns:
        A JSON-safe pre-phase target mapping.
    """
    required = (
        "position",
        "orientation_xyzw",
        "runtime_model_sha256",
        "source_sha256",
        "position_tolerance_m",
        "orientation_tolerance_rad",
    )
    if any(not hasattr(target, name) for name in required):
        raise ValueError("pre-phase target is not a calibrated TcpPoseTarget")
    return {
        "kind": "pre_phase",
        "tcp_pose": {
            "position": tuple(target.position),
            "orientation_xyzw": tuple(target.orientation_xyzw),
            "tip_link": f"{robot_id}_rocobrick_tcp",
            "frame_id": f"{robot_id}_base_link",
            "runtime_model_sha256": target.runtime_model_sha256,
            "source_sha256": target.source_sha256,
            "position_tolerance_m": target.position_tolerance_m,
            "orientation_tolerance_rad": target.orientation_tolerance_rad,
        },
    }


def _support_pre_phase_target(
    observation, context: TransitContext
) -> Mapping[str, object]:
    """Derive support TCP pose above a measured upper brick.

    Returns:
        A geometry-derived pre-phase target mapping.
    """
    minimum = observation.aabb_min
    maximum = observation.aabb_max
    world = (
        (minimum[0] + maximum[0]) / 2.0,
        (minimum[1] + maximum[1]) / 2.0,
        maximum[2] - context.params.support_press_m,
    )
    target = _tcp_pose_target(
        world_position_to_base(world, context.base),
        world_orientation_to_base(
            observation.tcp_pose.orientation_xyzw, context.base
        ),
        context.provenance,
        context.params,
    )
    return _tcp_target_mapping(target, observation.robot_id)


class SkillOperationExecutor:
    """Zero-motion :class:`RobotOperationExecutor` implementation (A7)."""

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
        place_backends: Mapping[str, object] | None = None,
        place_up_backends: Mapping[str, object] | None = None,
        support_backends: Mapping[str, object] | None = None,
        handover_backend: object | None = None,
        coordinated_layer_transport: object | None = None,
        pre_phase_runtime: object | None = None,
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

        Returns:
            Constructor state is stored after validating transport boundaries.

        Args:
            coordinated_layer_transport: Optional layer-level transport.
            pre_phase_runtime: Unified successor pre-phase runtime.
            transit_adapters: Per-robot free-space adapters.
            transit_contexts: Per-robot calibrated target contexts.
            pick_backends: Per-robot PICK handlers.
            place_backends: Per-robot PLACE_DOWN handlers.
            place_up_backends: Per-robot PLACE_UP handlers.
            support_backends: Per-robot SUPPORT_TOP handlers.
            state_ledger: Optional confirmed physical-state ledger.
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
        self._poll_interval_s = _positive_number(poll_interval_s, "poll_interval_s")
        self._wait_timeout_s = _positive_number(wait_timeout_s, "wait_timeout_s")
        self._cancel_timeout_s = _positive_number(cancel_timeout_s, "cancel_timeout_s")
        self._aabb_tolerance_m = _non_negative_number(
            aabb_tolerance_m, "aabb_tolerance_m"
        )
        self._transit_adapters = dict(transit_adapters or {})
        self._transit_contexts = dict(transit_contexts or {})
        self._pick_backends = dict(pick_backends or {})
        self._place_backends = dict(place_backends or {})
        self._place_up_backends = dict(place_up_backends or {})
        self._support_backends = dict(support_backends or {})
        self._handover_backend = handover_backend
        self._coordinated_layer_transport = coordinated_layer_transport
        self._pre_phase_runtime = pre_phase_runtime
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

    async def execute_layer(
        self, operations: Sequence[RobotOperation]
    ) -> Mapping[str, object]:
        """Execute an eligible layer through one coordinated transport.

        Returns:
            One strict result per original operation id.
        """
        group = coordinated_layer_group(tuple(operations))
        if group is None:
            raise ValueError("layer is not an eligible coordinated operation group")
        transport = self._coordinated_layer_transport
        execute = getattr(transport, "execute_layer", None)
        if not callable(execute):
            raise RuntimeError("coordinated layer transport is unavailable")
        observations: dict[str, Mapping[str, object]] = {}
        for robot_id in group.robot_ids:
            observations[robot_id] = await self._observe_or_cancel(
                robot_id,
                asyncio.Event(),
            ) or {}
        targets = self._derive_coordinated_targets(group, observations)
        group = replace(group, pre_phase_targets=targets)
        raw = execute(group, tuple(operations))
        if inspect.isawaitable(raw):
            raw = await raw
        if not isinstance(raw, Mapping):
            raise TypeError("coordinated layer transport must return a mapping")
        return raw

    def _derive_coordinated_targets(
        self,
        group: object,
        observations: Mapping[str, Mapping[str, object]],
    ) -> Mapping[str, Mapping[str, object]]:
        """Derive geometry-backed TCP targets for a coordinated pre-phase."""
        if not hasattr(group, "robot_ids"):
            raise ValueError("coordinated group identity is unavailable")
        targets: dict[str, Mapping[str, object]] = {}
        for robot_id in group.robot_ids:
            context = self._transit_contexts.get(robot_id)
            if context is None:
                raise ValueError(f"no calibrated context for {robot_id}")
            document = observations.get(robot_id)
            block = (
                document.get("robots", {}).get(robot_id)
                if isinstance(document, Mapping)
                else None
            )
            tcp = block.get("tcp_pose") if isinstance(block, Mapping) else None
            if not isinstance(tcp, Mapping):
                raise ValueError(f"missing measured TCP pose for {robot_id}")
            pose = parse_pick_observation(
                document,
                robot_id=robot_id,
                physical_instance_id=group.upper_physical_instance_id,
                session_id=str(document.get("session_id", "")),
                manifest_sha256=str(document.get("manifest_sha256", "")),
            )
            if robot_id == group.robot_ids[1]:
                targets[robot_id] = _support_pre_phase_target(pose, context)
                continue
            upper = parse_place_observation(
                document,
                robot_id=robot_id,
                upper_physical_instance_id=group.upper_physical_instance_id,
                lower_physical_instance_id=group.lower_physical_instance_id,
                session_id=pose.session_id,
                manifest_sha256=pose.manifest_sha256,
            )
            interfaces = upper.interfaces
            pair = "|".join(
                sorted(
                    (
                        group.upper_physical_instance_id,
                        group.lower_physical_instance_id,
                    )
                )
            )
            interface = (
                interfaces.get("pairs", {}).get(pair)
                if isinstance(interfaces, Mapping)
                else None
            )
            if not isinstance(interface, Mapping):
                raise ValueError("coordinated pre-phase lacks interface geometry")
            targets_data, _, _ = interface_place_up_targets(
                upper.upper.object_pose,
                _pose_from_mapping(interface.get("lower_to_upper")),
                upper.lower.object_pose,
                upper.lower.tcp_pose,
                upper.lower.opening_m,
                context.base,
                context.provenance,
                context.params,
                yaw_symmetries=tuple(
                    interface.get("allowed_upper_yaw_quarter_turns", (0,))
                ),
            )
            targets[robot_id] = _tcp_target_mapping(targets_data.preplace, robot_id)
        return targets

    async def observe_pre_phase(
        self, participants: tuple[str, ...]
    ) -> SharedSceneSnapshot:
        """Read one authorized shared snapshot, never synthesize it from robot queries.

        Returns:
            An admitted snapshot supplied by the unified transport.
        """
        observe = getattr(self._pre_phase_runtime, "observe_pre_phase", None)
        if not callable(observe):
            raise RuntimeError("unified shared snapshot admission is unavailable")
        snapshot = await observe(participants)
        if not isinstance(snapshot, SharedSceneSnapshot):
            raise RuntimeError("unified transport returned an invalid shared snapshot")
        return snapshot

    async def execute_pre_phase(
        self, request: SuccessorPrePhaseRequest
    ) -> SuccessorPrePhaseEvidence:
        """Dispatch the original operation and shared snapshot without rewriting input.

        Returns:
            Strict bound physical evidence or fail-closed evidence.
        """
        execute = getattr(self._pre_phase_runtime, "execute_pre_phase", None)
        if not callable(execute):
            return failed_pre_phase(request, "unified dual-arm runtime unavailable")
        raw = await execute(request)
        if not isinstance(raw, SuccessorPrePhaseEvidence):
            return failed_pre_phase(
                request, "unified runtime returned invalid evidence"
            )
        return raw

    async def validate_pre_phase(
        self, request: SuccessorPrePhaseRequest, evidence: SuccessorPrePhaseEvidence
    ) -> bool:
        """Validate live scene binding and reconcile physical facts before adoption.

        Returns:
            True only when scene binding and ledger continuity both hold.
        """
        validate = getattr(self._pre_phase_runtime, "validate_pre_phase", None)
        if not callable(validate) or await validate(request, evidence) is not True:
            return False
        if self._state_ledger is None:
            return False
        self._state_ledger.reconcile_pre_phase(request, evidence)
        return True

    async def cancel_pre_phase(self, transition_id: str) -> Mapping[str, object]:
        """Cancel the unified participant group rather than a completed predecessor.

        Returns:
            Explicit participant terminal-safety cancellation evidence.
        """
        cancel = getattr(self._pre_phase_runtime, "cancel_pre_phase", None)
        if not callable(cancel):
            return {"transition_id": transition_id, "terminal": False, "safe": False}
        return await cancel(transition_id)

    def terminal_pre_phase_policy(
        self, operations: Sequence[RobotOperation]
    ) -> TerminalPrePhasePolicy | None:
        """Delegate the explicit terminal departure policy to the unified runtime.

        Returns:
            The runtime's authorized policy, or None when no explicit policy
            source exists so the scheduler fails closed.
        """
        policy = getattr(self._pre_phase_runtime, "terminal_pre_phase_policy", None)
        if not callable(policy):
            return None
        return policy(tuple(operations))

    def _derive_successor_targets(
        self,
        request: SuccessorPrePhaseRequest,
        observations: Mapping[str, Mapping[str, object]],
    ) -> Mapping[str, Mapping[str, object]]:
        """Derive geometry-backed targets for any supported successor skill."""
        targets: dict[str, Mapping[str, object]] = {}
        for operation in request.successor_operations:
            target = operation.target.values if operation.target is not None else {}
            robot_id = operation.executor_robot_id
            context = self._transit_contexts.get(robot_id)
            document = observations.get(robot_id)
            if context is None or not isinstance(document, Mapping):
                raise ValueError(f"successor pre-phase lacks context for {robot_id}")
            session_id = str(document.get("session_id", ""))
            manifest_sha = str(document.get("manifest_sha256", ""))
            if operation.skill == "PICK":
                physical_id = target.get("physical_instance_id")
                if not isinstance(physical_id, str) or not physical_id:
                    raise ValueError(
                        f"successor {operation.operation_id} lacks part target geometry"
                    )
                sample = parse_pick_observation(
                    document,
                    robot_id=robot_id,
                    physical_instance_id=physical_id,
                    session_id=session_id,
                    manifest_sha256=manifest_sha,
                )
                from rocobrick.skills.targets import transit_target

                target_obj = transit_target(
                    sample.aabb_min,
                    sample.aabb_max,
                    context.base,
                    context.provenance,
                    context.params,
                )
                targets[robot_id] = _tcp_target_mapping(target_obj, robot_id)
                continue
            if operation.skill == "HANDOVER":
                raise ValueError(
                    "HANDOVER owns its receiver precontact free-space phase; "
                    "it has no successor pre-phase transport"
                )
            if operation.skill == "SUPPORT_TOP":
                upper_id = target.get("upper_physical_instance_id")
                if not isinstance(upper_id, str) or not upper_id:
                    raise ValueError("SUPPORT_TOP lacks upper geometry identity")
                sample = parse_pick_observation(
                    document,
                    robot_id=robot_id,
                    physical_instance_id=upper_id,
                    session_id=session_id,
                    manifest_sha256=manifest_sha,
                )
                targets[robot_id] = _support_pre_phase_target(sample, context)
                continue
            if operation.skill in {"PLACE_DOWN", "PLACE_UP"}:
                upper_id = target.get("upper_physical_instance_id")
                lower_id = target.get("lower_physical_instance_id")
                if not all(
                    isinstance(item, str) and item
                    for item in (upper_id, lower_id)
                ):
                    raise ValueError(
                        f"{operation.skill} lacks pair geometry identity"
                    )
                sample = parse_place_observation(
                    document,
                    robot_id=robot_id,
                    upper_physical_instance_id=upper_id,
                    lower_physical_instance_id=lower_id,
                    session_id=session_id,
                    manifest_sha256=manifest_sha,
                )
                interfaces = sample.interfaces
                pair = "|".join(sorted((upper_id, lower_id)))
                interface = (
                    interfaces.get("pairs", {}).get(pair)
                    if isinstance(interfaces, Mapping)
                    else None
                )
                if not isinstance(interface, Mapping):
                    raise ValueError(f"{operation.skill} lacks interface geometry")
                connection_pose = _pose_from_mapping(
                    interface.get("lower_to_upper")
                )
                yaw_symmetries = tuple(
                    interface.get("allowed_upper_yaw_quarter_turns", (0,))
                )
                if operation.skill == "PLACE_UP":
                    targets_obj, _, _ = interface_place_up_targets(
                        sample.upper.object_pose,
                        connection_pose,
                        sample.lower.object_pose,
                        sample.lower.tcp_pose,
                        sample.lower.opening_m,
                        context.base,
                        context.provenance,
                        context.params,
                        yaw_symmetries=yaw_symmetries,
                    )
                else:
                    from rocobrick.skills.placement_geometry import (
                        interface_place_targets,
                    )

                    targets_obj, _, _ = interface_place_targets(
                        sample.lower.object_pose,
                        connection_pose,
                        sample.upper.object_pose,
                        sample.upper.tcp_pose,
                        sample.upper.opening_m,
                        context.base,
                        context.provenance,
                        context.params,
                        yaw_symmetries=yaw_symmetries,
                    )
                targets[robot_id] = _tcp_target_mapping(
                    targets_obj.preplace, robot_id
                )
                continue
            raise ValueError(f"unsupported successor pre-phase skill {operation.skill!r}")
        if set(targets) != set(request.participants):
            raise ValueError("successor pre-phase targets do not cover participants")
        return targets

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
            return self._failed(operation, REASON_OPERATION_ALREADY_ACTIVE)
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
            if skill == PICK and robot_id in self._pick_backends:
                return await self._run_motion_operation(
                    operation, lambda: self._pick(operation, robot_id)
                )
            if skill == HANDOVER and self._handover_backend is not None:
                return await self._run_motion_operation(
                    operation, lambda: self._handover(operation, robot_id)
                )
            if skill == PLACE_DOWN and robot_id in self._place_backends:
                return await self._run_motion_operation(
                    operation, lambda: self._place_down(operation, robot_id)
                )
            if skill == PLACE_UP and robot_id in self._place_up_backends:
                return await self._run_motion_operation(
                    operation, lambda: self._place_up(operation, robot_id)
                )
            if skill == SUPPORT_TOP and robot_id in self._support_backends:
                return await self._run_motion_operation(
                    operation, lambda: self._support_top(operation, robot_id)
                )
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

    async def _run_motion_operation(
        self, operation: RobotOperation, callback: object
    ) -> OperationExecution:
        """Track one motion operation independently from other robots."""
        record = _ActiveOperation(
            operation_id=operation.operation_id,
            robot_id=operation.executor_robot_id,
            skill=operation.skill,
            cancel_event=asyncio.Event(),
            finished=asyncio.get_running_loop().create_future(),
        )
        self._active[operation.operation_id] = record
        try:
            result = callback()
            if not inspect.isawaitable(result):
                raise TypeError("motion callback must return an awaitable")
            return await result
        finally:
            self._active.pop(operation.operation_id, None)
            self._finished[operation.operation_id] = record.motion_sent
            if not record.finished.done():
                record.finished.set_result(None)

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
        if record.skill == HANDOVER:
            cancel_handover = getattr(self._handover_backend, "cancel", None)
            if not callable(cancel_handover):
                return CancelEvidence(operation_id, False, False, False, {})
            raw = await cancel_handover(operation_id)
            bound = isinstance(raw, Mapping) and raw.get("operation_id") == operation_id
            return CancelEvidence(
                operation_id=operation_id, acknowledged=bound,
                terminal=bound and raw.get("terminal") is True,
                safe=bound and raw.get("safe_to_release") is True,
                evidence=dict(raw) if isinstance(raw, Mapping) else {},
            )
        if record.skill in {PLACE_DOWN, PLACE_UP, SUPPORT_TOP}:
            backends = {
                PLACE_DOWN: self._place_backends,
                PLACE_UP: self._place_up_backends,
                SUPPORT_TOP: self._support_backends,
            }
            backend = backends[record.skill].get(record.robot_id)
            cancel_place = getattr(backend, "cancel", None)
            if not callable(cancel_place):
                return CancelEvidence(
                    operation_id=operation_id,
                    acknowledged=False,
                    terminal=False,
                    safe=False,
                    evidence={},
                )
            raw = await cancel_place(operation_id)
            return CancelEvidence(
                operation_id=operation_id,
                acknowledged=isinstance(raw, Mapping) and raw.get("attempted") is True,
                terminal=isinstance(raw, Mapping) and raw.get("terminal") is True,
                safe=isinstance(raw, Mapping) and raw.get("safe") is True,
                evidence=dict(raw) if isinstance(raw, Mapping) else {},
            )
        record.cancel_event.set()
        if record.motion_sent:
            # TRANSIT adapters were removed with the TRANSIT skill; an active
            # motion operation that may have sent commands can never prove a
            # safe terminal state through this generic path, so fail closed.
            return CancelEvidence(
                operation_id=operation_id,
                acknowledged=True,
                terminal=False,
                safe=False,
                evidence={
                    "cancel_error": (
                        "skill dispatch has no dedicated safe cancel boundary"
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

    async def _handover(
        self, operation: RobotOperation, robot_id: str
    ) -> OperationExecution:
        """Execute one generic HANDOVER through the shared handover backend.

        The operation is validated against the frozen plan contract before the
        backend runs: the executor must be the receiver, participants must be
        exactly donor/receiver, and the ledger must already prove the donor
        holds the physical instance. The donor's holding fact is never assumed
        from a predecessor skill; PICK is only one possible producer.
        """
        if self._handover_backend is None:
            return self._failed(operation, REASON_CAPABILITY_UNAVAILABLE)
        target = operation.target.values if operation.target is not None else {}
        donor = target.get("donor_robot_id")
        part = target.get("physical_instance_id")
        if self._state_ledger is None:
            return self._failed(operation, "HANDOVER_LEDGER_UNAVAILABLE")
        if (
            not isinstance(donor, str)
            or not isinstance(part, str)
            or self._state_ledger.observe(donor, f"holding:{part}") is None
        ):
            return self._failed(operation, "HANDOVER_DONOR_HOLDING_UNCONFIRMED")
        execute = getattr(self._handover_backend, "execute", None)
        if not callable(execute):
            return self._failed(operation, "HANDOVER_BACKEND_INVALID")
        try:
            result = await execute(operation)
        except Exception as error:  # noqa: BLE001 - preserve failure evidence
            return self._failed(operation, f"HANDOVER_BACKEND_FAILED: {error}")
        if (
            not isinstance(result, OperationExecution)
            or result.operation_id != operation.operation_id
            or result.robot_id != robot_id
        ):
            return self._failed(operation, "HANDOVER_RESULT_INVALID")
        return result

    async def _pick(
        self, operation: RobotOperation, robot_id: str
    ) -> OperationExecution:
        """Execute PICK from the current pose directly to pregrasp.

        A direct PICK plan treats the former empty TRANSIT as part of the
        pregrasp motion.  The handler therefore sends one TCP pose from the
        current robot state to the calibrated pregrasp target instead of
        first sending a separate high transit pose and replanning downward.
        """
        handler = self._pick_backends.get(robot_id)
        execute = getattr(handler, "execute", None)
        if not callable(execute):
            return self._failed(operation, "PICK_HANDLER_INVALID")
        execution_operation = operation
        if self._state_ledger is not None:
            arrived = self._state_ledger.observe(robot_id, "robot_at_pick_region")
            if arrived is None and "robot_at_pick_region" in operation.preconditions:
                execution_operation = replace(
                    operation,
                    preconditions=tuple(
                        item
                        for item in operation.preconditions
                        if item != "robot_at_pick_region"
                    ),
                )
        try:
            result = await execute(execution_operation)
        except Exception as error:
            return self._failed(operation, f"PICK_HANDLER_FAILED: {error}")
        if (
            not isinstance(result, OperationExecution)
            or result.operation_id != operation.operation_id
            or result.robot_id != robot_id
        ):
            return self._failed(operation, "PICK_RESULT_INVALID")
        return result

    async def _place_down(
        self, operation: RobotOperation, robot_id: str
    ) -> OperationExecution:
        """Execute PLACE_DOWN with the carrying gate before any motion.

        The ledger must already confirm this robot holds the upper part; a
        carrying request without that confirmation is refused before the
        handler (and therefore before any motion command) runs.  A direct
        plan without a prior TRANSIT carries its own transport phase, so an
        unconfirmed ``robot_at_place_region`` precondition is removed the
        same way the direct PICK path removes its arrival precondition.

        Returns:
            The backend's operation execution, or a closed failure.
        """
        handler = self._place_backends.get(robot_id)
        execute = getattr(handler, "execute", None)
        if not callable(execute):
            return self._failed(operation, "PLACE_DOWN_HANDLER_INVALID")
        target_values = operation.target.values if operation.target is not None else {}
        upper = target_values.get("upper_physical_instance_id")
        if self._state_ledger is None or not isinstance(upper, str) or not upper:
            return self._failed(operation, "PLACE_DOWN_CARRYING_UNCONFIRMED")
        if self._state_ledger.owner_of_object(upper) != robot_id:
            return self._failed(
                operation,
                "PLACE_DOWN_CARRYING_UNCONFIRMED: the ledger does not confirm "
                f"{robot_id} holding {upper!r}",
            )
        execution_operation = operation
        if self._state_ledger.observe(robot_id, "robot_at_place_region") is None:
            execution_operation = replace(
                operation,
                preconditions=tuple(
                    item
                    for item in operation.preconditions
                    if item != "robot_at_place_region"
                ),
            )
        try:
            result = await execute(execution_operation)
        except Exception as error:
            return self._failed(operation, f"PLACE_DOWN_HANDLER_FAILED: {error}")
        if (
            not isinstance(result, OperationExecution)
            or result.operation_id != operation.operation_id
            or result.robot_id != robot_id
        ):
            return self._failed(operation, "PLACE_DOWN_RESULT_INVALID")
        return result

    async def _place_up(
        self, operation: RobotOperation, robot_id: str
    ) -> OperationExecution:
        """Execute PLACE_UP with carrying and support gates before motion.

        The ledger must confirm this robot holds the moving LOWER part, and
        must confirm the target's support robot actively supports the fixed
        UPPER part; either missing fact refuses the operation before any
        motion command runs.

        Returns:
            The backend's operation execution, or a closed failure.
        """
        handler = self._place_up_backends.get(robot_id)
        execute = getattr(handler, "execute", None)
        if not callable(execute):
            return self._failed(operation, "PLACE_UP_HANDLER_INVALID")
        values = operation.target.values if operation.target is not None else {}
        lower = values.get("lower_physical_instance_id")
        upper = values.get("upper_physical_instance_id")
        support_robot = values.get("support_robot_id")
        if self._state_ledger is None or not all(
            isinstance(item, str) and item for item in (lower, upper, support_robot)
        ):
            return self._failed(operation, "PLACE_UP_CARRYING_UNCONFIRMED")
        if self._state_ledger.owner_of_object(lower) != robot_id:
            return self._failed(
                operation,
                "PLACE_UP_CARRYING_UNCONFIRMED: the ledger does not confirm "
                f"{robot_id} holding {lower!r}",
            )
        if self._state_ledger.observe(support_robot, f"supporting:{upper}") is None:
            return self._failed(
                operation,
                "PLACE_UP_SUPPORT_UNCONFIRMED: the ledger does not confirm "
                f"{support_robot!r} supporting {upper!r}",
            )
        execution_operation = operation
        if self._state_ledger.observe(robot_id, "robot_at_place_region") is None:
            execution_operation = replace(
                operation,
                preconditions=tuple(
                    item
                    for item in operation.preconditions
                    if item != "robot_at_place_region"
                ),
            )
        try:
            result = await execute(execution_operation)
        except Exception as error:
            return self._failed(operation, f"PLACE_UP_HANDLER_FAILED: {error}")
        if (
            not isinstance(result, OperationExecution)
            or result.operation_id != operation.operation_id
            or result.robot_id != robot_id
        ):
            return self._failed(operation, "PLACE_UP_RESULT_INVALID")
        return result

    async def _support_top(
        self, operation: RobotOperation, robot_id: str
    ) -> OperationExecution:
        """Execute SUPPORT_TOP: stabilize the fixed upper part from below/side.

        The backend owns the support grasp, its stability evidence and the
        ``supporting:<upper>`` ledger confirmation; a failed support keeps the
        placement gate closed (PLACE_UP refuses without the support fact).

        Returns:
            The backend's operation execution, or a closed failure.
        """
        handler = self._support_backends.get(robot_id)
        execute = getattr(handler, "execute", None)
        if not callable(execute):
            return self._failed(operation, "SUPPORT_TOP_HANDLER_INVALID")
        try:
            result = await execute(operation)
        except Exception as error:
            return self._failed(operation, f"SUPPORT_TOP_HANDLER_FAILED: {error}")
        if (
            not isinstance(result, OperationExecution)
            or result.operation_id != operation.operation_id
            or result.robot_id != robot_id
        ):
            return self._failed(operation, "SUPPORT_TOP_RESULT_INVALID")
        return result

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
            first = _observation(robot_id, parameters.physical_instance_id, document)
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
                document = await self._observe_or_cancel(robot_id, record.cancel_event)
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
            if not _aabb_stable(first.aabb, current.aabb, self._aabb_tolerance_m):
                return self._failed(operation, REASON_STRUCTURAL_FACT_UNSTABLE)
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
            await asyncio.gather(observe_task, cancel_task, return_exceptions=True)
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
