"""Contract tests for the A7 zero-motion skill executor.

The tests run fully offline: the per-robot observation transport is an
in-memory stub, no ROS/Isaac/runtime module is imported, and the injected
documents follow the frozen A4/A5 ``rocobrick-scene-state/v1`` shape.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import math
import time

import pytest

from rocobrick.multi_robot_scheduler import (
    CancelEvidence,
    MultiRobotScheduler,
    OperationExecution,
    OperationStatus,
    is_json_safe,
)
from rocobrick.robot_operation_executor import (
    LABEL_REAL_SCENE_OBSERVATION,
    LABEL_WAIT_DURATION_CONFIRMED,
    REASON_CAPABILITY_UNAVAILABLE,
    REASON_OBSERVATION_STAGNANT,
    REASON_OBSERVATION_UNAVAILABLE,
    REASON_OPERATION_ALREADY_ACTIVE,
    REASON_ROBOT_NOT_VALIDATED,
    REASON_STRUCTURAL_FACT_UNSTABLE,
    REASON_TARGET_INVALID,
    REASON_WAIT_DEADLINE_EXCEEDED,
    REASON_WAIT_EVIDENCE_UNSUPPORTED,
    REASON_WAIT_PARAMETERS_INVALID,
    SCENE_STATE_SCHEMA,
    SkillOperationExecutor,
)
from rocobrick.robot_operation_plan import (
    OperationTarget,
    RobotOperation,
    parse_robot_operation_plan,
)

ROBOT_0 = "piper_0"
ROBOT_1 = "piper_1"
ROBOTS = (ROBOT_0, ROBOT_1)
BRICK_A = "brick-768de5a17827920c48d4"
BRICK_B = "brick-4bbba49bec24496bb21e"
AABB_A = ((0.1, -0.05, 0.0), (0.14, 0.03, 0.0112))
AABB_DRIFTED = ((0.15, -0.05, 0.0), (0.19, 0.03, 0.0112))


def robot_block(
    robot_id: str, sim_time_s: float, *, opening_m: float | None = 0.02
) -> dict:
    """Build one per-robot observation block of the A4/A5 document.

    Args:
        robot_id: Robot owning the block.
        sim_time_s: Observed simulation time of that robot.
        opening_m: Logical gripper opening, or None for a degraded reading.

    Returns:
        The ``robots[robot_id]`` block.
    """
    return {
        "robot_id": robot_id,
        "sim_time_s": sim_time_s,
        "gripper": {"gripper_joint1": 0.01, "gripper_joint2": -0.01},
        "opening_m": opening_m,
    }


def brick_block(bounds: tuple = AABB_A) -> dict:
    """Build one measured brick AABB block.

    Args:
        bounds: ``(aabb_min, aabb_max)`` corners.

    Returns:
        The ``bricks[physical_instance_id]`` block.
    """
    return {"aabb_min": list(bounds[0]), "aabb_max": list(bounds[1])}


def scene_document(
    sim_time_s: float,
    *,
    brick_id: str = BRICK_A,
    bounds: tuple = AABB_A,
    robot_id: str = ROBOT_1,
    robot_time_s: float | None = None,
    robots: object = None,
    bricks: object = None,
    connections: object = None,
    schema: object = SCENE_STATE_SCHEMA,
) -> dict:
    """Build one A4/A5 scene-state document.

    Args:
        sim_time_s: Shared simulation time.
        brick_id: Physical instance carrying the measured AABB.
        bounds: AABB corners of ``brick_id``.
        robot_id: Robot owning the default observation block.
        robot_time_s: Per-robot time; defaults to ``sim_time_s``.
        robots: Override the whole ``robots`` mapping.
        bricks: Override the whole ``bricks`` mapping.
        connections: Override the whole ``connections`` mapping.
        schema: Document schema value.

    Returns:
        The scene-state document mapping.
    """
    if robots is None:
        robots = {
            robot_id: robot_block(
                robot_id, sim_time_s if robot_time_s is None else robot_time_s
            )
        }
    if bricks is None:
        bricks = {brick_id: brick_block(bounds)}
    return {
        "schema": schema,
        "sim_time_s": sim_time_s,
        "bricks": bricks,
        "connections": {} if connections is None else connections,
        "robots": robots,
    }


def without(document: dict, key: str) -> dict:
    """Return a copy of ``document`` without ``key``.

    Args:
        document: Source mapping.
        key: Key to drop.

    Returns:
        The copied mapping.
    """
    copy = dict(document)
    copy.pop(key, None)
    return copy


class RecordingTransport:
    """Read-only observation stub driven by queued or computed documents.

    Args:
        documents: Per-robot list of documents (the last one repeats), or a
            ``(robot_id, index) -> document`` callable.
    """

    def __init__(self, documents: object) -> None:
        """Store the scripted documents and reset the call log.

        Args:
            documents: Per-robot document list or document callable.
        """
        self._documents = documents
        self.calls: list[str] = []

    async def observe(self, robot_id: str) -> object:
        """Return the next scripted document for ``robot_id``.

        Args:
            robot_id: Requested robot id.

        Returns:
            The scripted scene-state document.
        """
        self.calls.append(robot_id)
        if callable(self._documents):
            index = sum(1 for call in self.calls if call == robot_id) - 1
            return self._documents(robot_id, index)
        queue = self._documents[robot_id]
        index = sum(1 for call in self.calls if call == robot_id) - 1
        return queue[min(index, len(queue) - 1)]


def advancing_transport(
    *,
    start_s: float = 10.0,
    step_s: float = 0.4,
    duration_calls: int = 64,
) -> RecordingTransport:
    """Return a transport whose robot time advances per observation.

    Args:
        start_s: First observed simulation time.
        step_s: Simulation-time advance per observation.
        duration_calls: Number of distinct documents before the last repeats.

    Returns:
        The recording transport.
    """

    def documents(robot_id: str, index: int) -> dict:
        sim_time_s = start_s + step_s * min(index, duration_calls)
        return scene_document(sim_time_s)

    return RecordingTransport(documents)


def wait_operation(
    *,
    operation_id: str = "wait-op",
    robot_id: str = ROBOT_1,
    instance_id: str = BRICK_A,
    duration_s: object = 1.0,
    parameters: object = None,
    target: object = None,
    evidence_requirements: tuple[str, ...] = (
        LABEL_REAL_SCENE_OBSERVATION,
        LABEL_WAIT_DURATION_CONFIRMED,
    ),
) -> RobotOperation:
    """Build one validated-shape WAIT operation.

    Args:
        operation_id: Operation id.
        robot_id: Executor robot id.
        instance_id: Waited physical instance id.
        duration_s: Requested sim-time advance.
        parameters: Override the whole parameter mapping.
        target: Override the target.
        evidence_requirements: Declared evidence labels.

    Returns:
        The WAIT operation.
    """
    if parameters is None:
        parameters = {
            "observation": {
                "kind": "structural_fact",
                "fact": "stable",
                "physical_instance_id": instance_id,
            },
            "duration_s": duration_s,
        }
    return RobotOperation(
        operation_id=operation_id,
        source_task_id="task-wait",
        skill="WAIT",
        executor_robot_id=robot_id,
        participants=(robot_id,),
        role_bindings={"executor": robot_id},
        target=target,
        parameters=parameters,
        preconditions=("scene_ready",),
        effects=("scene_stable_observed",),
        resources=(f"robot:{robot_id}",),
        evidence_requirements=tuple(evidence_requirements),
    )


def reference(
    kind: str,
    *,
    physical_instance_id: str | None = None,
    connection_id: str | None = None,
    robot_id: str | None = None,
) -> dict:
    """Build one frozen four-key target reference.

    Args:
        kind: Reference kind.
        physical_instance_id: Physical instance, when the kind uses it.
        connection_id: Connection id, when the kind uses it.
        robot_id: Robot id, when the kind uses it.

    Returns:
        The reference mapping.
    """
    return {
        "kind": kind,
        "physical_instance_id": physical_instance_id,
        "connection_id": connection_id,
        "robot_id": robot_id,
    }


def default_target(skill: str) -> dict:
    """Return one valid target for a motion skill.

    Args:
        skill: ``PICK``, ``TRANSIT`` or ``PLACE_DOWN``.

    Returns:
        The target mapping.
    """
    if skill == "PICK":
        return {"physical_instance_id": BRICK_A}
    if skill == "TRANSIT":
        return {
            "from": reference("robot_current"),
            "to": reference("part", physical_instance_id=BRICK_A),
            "carrying_physical_instance_id": None,
        }
    return {
        "connection_id": "conn-upper-lower",
        "upper_physical_instance_id": BRICK_A,
        "lower_physical_instance_id": BRICK_B,
    }


_DEFAULT_TARGET = object()


def motion_operation(
    skill: str,
    *,
    operation_id: str = "motion-op",
    robot_id: str = ROBOT_0,
    target: object = _DEFAULT_TARGET,
) -> RobotOperation:
    """Build one motion operation, valid unless ``target`` is overridden.

    Args:
        skill: ``PICK``, ``TRANSIT`` or ``PLACE_DOWN``.
        operation_id: Operation id.
        robot_id: Executor robot id.
        target: Target mapping; the default sentinel selects the valid
            default, while an explicit ``None`` builds a targetless
            operation.

    Returns:
        The motion operation.
    """
    chosen = default_target(skill) if target is _DEFAULT_TARGET else target
    return RobotOperation(
        operation_id=operation_id,
        source_task_id="task-motion",
        skill=skill,
        executor_robot_id=robot_id,
        participants=(robot_id,),
        role_bindings={"executor": robot_id},
        target=OperationTarget(chosen),
        parameters={},
        preconditions=(),
        effects=(),
        resources=(f"robot:{robot_id}",),
        evidence_requirements=("motion_label",),
    )


def make_executor(
    transport: object,
    *,
    poll_interval_s: float = 0.005,
    wait_timeout_s: float = 5.0,
    cancel_timeout_s: float = 1.0,
    aabb_tolerance_m: float = 0.01,
) -> SkillOperationExecutor:
    """Build an executor over the two test robots.

    Args:
        transport: Injected observation transport.
        poll_interval_s: Polling interval.
        wait_timeout_s: Internal monotonic wait deadline.
        cancel_timeout_s: Cooperative cancel acknowledgement bound.
        aabb_tolerance_m: Structural stability tolerance.

    Returns:
        The configured executor.
    """
    return SkillOperationExecutor(
        ROBOTS,
        transport,
        poll_interval_s=poll_interval_s,
        wait_timeout_s=wait_timeout_s,
        cancel_timeout_s=cancel_timeout_s,
        aabb_tolerance_m=aabb_tolerance_m,
    )


# ------------------------------------------------------------- construction


def test_constructor_validates_robot_ids_transport_and_bounds() -> None:
    """Malformed ids, a transport without observe and bad bounds raise."""
    transport = RecordingTransport({})
    for robot_ids in ((), (ROBOT_0, ROBOT_0), (ROBOT_0, ""), (ROBOT_0, 7)):
        with pytest.raises(ValueError):
            SkillOperationExecutor(robot_ids, transport)
    with pytest.raises(ValueError):
        SkillOperationExecutor(ROBOT_0, transport)
    with pytest.raises(ValueError):
        SkillOperationExecutor(ROBOTS, object())
    for name, value in (
        ("poll_interval_s", 0.0),
        ("poll_interval_s", -1.0),
        ("wait_timeout_s", 0.0),
        ("cancel_timeout_s", float("nan")),
        ("aabb_tolerance_m", -0.1),
    ):
        with pytest.raises(ValueError):
            SkillOperationExecutor(ROBOTS, transport, **{name: value})
    executor = make_executor(transport)
    assert executor.robot_ids == ROBOTS


def test_executor_implements_the_scheduler_executor_protocol() -> None:
    """The public methods are coroutines and results are A2 contract types."""
    for name in ("execute", "cancel"):
        assert asyncio.iscoroutinefunction(getattr(SkillOperationExecutor, name))
    result = asyncio.run(make_executor(advancing_transport()).execute(
        wait_operation()
    ))
    assert isinstance(result, OperationExecution)


# ---------------------------------------------------------------- WAIT proof


def test_wait_confirms_only_on_observed_clock_advance_and_stable_aabb() -> None:
    """Both labels are emitted only after a measured advance and stable AABB."""
    transport = advancing_transport(start_s=10.0, step_s=0.4)
    executor = make_executor(transport)
    result = asyncio.run(executor.execute(wait_operation(duration_s=1.0)))
    assert result.status is OperationStatus.CONFIRMED
    assert result.passed is True
    assert result.terminal is True
    assert result.motion_sent is False
    assert isinstance(result.evidence, dict)
    assert is_json_safe(result.evidence)
    json.dumps(result.evidence, allow_nan=False)
    assert result.evidence[LABEL_REAL_SCENE_OBSERVATION]
    assert result.evidence[LABEL_WAIT_DURATION_CONFIRMED]
    confirmed = result.evidence[LABEL_WAIT_DURATION_CONFIRMED]
    assert confirmed["observed_advance_s"] >= 1.0
    assert confirmed["requested_duration_s"] == 1.0
    observed = result.evidence[LABEL_REAL_SCENE_OBSERVATION]
    assert observed["robot_id"] == ROBOT_1
    assert observed["physical_instance_id"] == BRICK_A
    assert set(transport.calls) == {ROBOT_1}
    assert len(transport.calls) > 1


def test_wait_rejects_a_stagnant_clock_within_a_bounded_deadline() -> None:
    """A constant sim clock fails closed at the monotonic deadline."""
    transport = RecordingTransport(
        lambda robot_id, index: scene_document(10.0)
    )
    executor = make_executor(
        transport, poll_interval_s=0.005, wait_timeout_s=0.2
    )
    started = time.monotonic()
    result = asyncio.run(executor.execute(wait_operation(duration_s=1.0)))
    elapsed = time.monotonic() - started
    assert elapsed < 2.0
    assert result.status is OperationStatus.FAILED
    assert result.passed is False
    assert result.terminal is True
    assert result.motion_sent is False
    assert result.evidence == {}
    assert result.reason == REASON_OBSERVATION_STAGNANT
    assert len(transport.calls) > 1


def test_wait_rejects_partial_advance_when_the_deadline_expires() -> None:
    """A clock that advances too slowly fails at the wall-clock deadline."""
    transport = RecordingTransport(
        lambda robot_id, index: scene_document(10.0 + 0.01 * index)
    )
    executor = make_executor(
        transport, poll_interval_s=0.005, wait_timeout_s=0.15
    )
    result = asyncio.run(executor.execute(wait_operation(duration_s=5.0)))
    assert result.status is OperationStatus.FAILED
    assert result.terminal is True
    assert result.motion_sent is False
    assert result.evidence == {}
    assert result.reason == REASON_WAIT_DEADLINE_EXCEEDED


def test_wait_rejects_a_regressed_clock() -> None:
    """An observed sim time that moves backwards is never proof."""
    transport = RecordingTransport(
        lambda robot_id, index: scene_document(10.0 if index == 0 else 9.5)
    )
    executor = make_executor(
        transport, poll_interval_s=0.005, wait_timeout_s=0.2
    )
    result = asyncio.run(executor.execute(wait_operation(duration_s=1.0)))
    assert result.status is OperationStatus.FAILED
    assert result.reason == REASON_OBSERVATION_STAGNANT
    assert result.evidence == {}


def test_wait_rejects_a_drifting_structural_fact() -> None:
    """A measured AABB outside the tolerance is never a stable fact."""
    transport = RecordingTransport(
        lambda robot_id, index: scene_document(
            10.0 + 0.5 * index,
            bounds=AABB_A if index == 0 else AABB_DRIFTED,
        )
    )
    executor = make_executor(
        transport, poll_interval_s=0.005, aabb_tolerance_m=0.01
    )
    result = asyncio.run(executor.execute(wait_operation(duration_s=1.0)))
    assert result.status is OperationStatus.FAILED
    assert result.reason == REASON_STRUCTURAL_FACT_UNSTABLE
    assert result.evidence == {}
    assert result.motion_sent is False


def _invalid_documents() -> list:
    """Return (id, factory) pairs of documents that must be rejected.

    Returns:
        The parametrization cases for the missing/malformed observation test.
    """
    valid = scene_document(10.0)
    foreign = scene_document(10.0)
    foreign["robots"] = {ROBOT_1: robot_block(ROBOT_0, 10.0)}
    incomplete = scene_document(10.0)
    incomplete["robots"][ROBOT_1]["gripper"] = {"gripper_joint1": 0.01}
    bad_opening = scene_document(10.0)
    bad_opening["robots"][ROBOT_1]["opening_m"] = "wide"
    bad_time = scene_document(10.0)
    bad_time["robots"][ROBOT_1]["sim_time_s"] = math.nan
    short_aabb = scene_document(10.0, bricks={BRICK_A: {
        "aabb_min": [0.0, 0.0], "aabb_max": [1.0, 1.0, 1.0]
    }})
    inverted = scene_document(10.0, bricks={BRICK_A: {
        "aabb_min": [1.0, 0.0, 0.0], "aabb_max": [0.0, 1.0, 1.0]
    }})
    nan_corner = scene_document(10.0, bricks={BRICK_A: {
        "aabb_min": [math.nan, 0.0, 0.0], "aabb_max": [1.0, 1.0, 1.0]
    }})
    return [
        pytest.param(lambda: ["not-a-document"], id="not-an-object"),
        pytest.param(lambda: {"schema": SCENE_STATE_SCHEMA}, id="empty-object"),
        pytest.param(
            lambda: {**valid, "schema": "other/v1"}, id="wrong-schema"
        ),
        pytest.param(
            lambda: without(valid, "sim_time_s"), id="missing-shared-time"
        ),
        pytest.param(
            lambda: {**valid, "sim_time_s": "ten"}, id="non-numeric-shared-time"
        ),
        pytest.param(lambda: without(valid, "bricks"), id="missing-bricks"),
        pytest.param(
            lambda: {**valid, "bricks": {"other-brick": brick_block()}},
            id="missing-waited-instance",
        ),
        pytest.param(
            lambda: without(valid, "connections"), id="missing-connections"
        ),
        pytest.param(lambda: without(valid, "robots"), id="missing-robots"),
        pytest.param(
            lambda: {**valid, "robots": {ROBOT_0: robot_block(ROBOT_0, 10.0)}},
            id="missing-own-robot-block",
        ),
        pytest.param(lambda: foreign, id="foreign-robot-block"),
        pytest.param(
            lambda: without(scene_document(10.0, robot_time_s=10.0), "robots")
            | {"robots": {ROBOT_1: {
                "robot_id": ROBOT_1,
                "gripper": {"gripper_joint1": 0.01, "gripper_joint2": -0.01},
                "opening_m": 0.02,
            }}},
            id="missing-robot-time",
        ),
        pytest.param(lambda: incomplete, id="incomplete-gripper"),
        pytest.param(lambda: bad_opening, id="non-numeric-opening"),
        pytest.param(lambda: bad_time, id="nan-robot-time"),
        pytest.param(lambda: short_aabb, id="short-aabb"),
        pytest.param(lambda: inverted, id="inverted-aabb"),
        pytest.param(lambda: nan_corner, id="nan-aabb-corner"),
    ]


@pytest.mark.parametrize("factory", _invalid_documents())
def test_wait_rejects_missing_or_malformed_observations(factory) -> None:
    """Every missing/malformed field fails closed without any label."""
    transport = RecordingTransport(lambda robot_id, index: factory())
    executor = make_executor(transport)
    result = asyncio.run(executor.execute(wait_operation(duration_s=1.0)))
    assert result.status is OperationStatus.FAILED
    assert result.terminal is True
    assert result.passed is False
    assert result.motion_sent is False
    assert result.evidence == {}
    assert result.reason == REASON_OBSERVATION_UNAVAILABLE


def test_wait_rejects_unsupported_parameters_before_observing() -> None:
    """Unsupported observation shapes never reach the transport."""
    observations = [
        {"kind": "pose", "fact": "stable", "physical_instance_id": BRICK_A},
        {"kind": "structural_fact", "fact": "moving",
         "physical_instance_id": BRICK_A},
        {"kind": "structural_fact", "fact": "stable",
         "physical_instance_id": ""},
        {"kind": "structural_fact", "fact": "stable"},
        {"kind": "structural_fact", "fact": "stable",
         "physical_instance_id": BRICK_A, "extra": 1},
    ]
    for duration in (0, -1.0, math.nan, "1", None):
        parameters = {"observation": observations[0], "duration_s": duration}
        transport = RecordingTransport({})
        executor = make_executor(transport)
        result = asyncio.run(executor.execute(wait_operation(parameters=parameters)))
        assert result.status is OperationStatus.FAILED
        assert result.reason is not None
        assert result.reason.startswith(REASON_WAIT_PARAMETERS_INVALID)
        assert result.evidence == {}
        assert result.terminal is True
        assert result.motion_sent is False
        assert transport.calls == []
    for observation in observations:
        parameters = {
            "observation": observation,
            "duration_s": 1.0,
            "extra": True,
        }
        transport = RecordingTransport({})
        executor = make_executor(transport)
        result = asyncio.run(executor.execute(wait_operation(parameters=parameters)))
        assert result.reason is not None
        assert result.reason.startswith(REASON_WAIT_PARAMETERS_INVALID)
        assert transport.calls == []


def test_wait_rejects_labels_it_cannot_prove() -> None:
    """A declared label outside the two wait labels fails closed unobserved."""
    transport = RecordingTransport({})
    executor = make_executor(transport)
    operation = wait_operation(
        evidence_requirements=(
            LABEL_REAL_SCENE_OBSERVATION,
            "carrying_state_confirmed",
        )
    )
    result = asyncio.run(executor.execute(operation))
    assert result.status is OperationStatus.FAILED
    assert result.reason is not None
    assert result.reason.startswith(REASON_WAIT_EVIDENCE_UNSUPPORTED)
    assert result.evidence == {}
    assert transport.calls == []


def test_wait_rejects_a_non_null_target() -> None:
    """The frozen WAIT target is null; anything else fails closed."""
    transport = RecordingTransport({})
    executor = make_executor(transport)
    operation = wait_operation(
        target=OperationTarget({"physical_instance_id": BRICK_A})
    )
    result = asyncio.run(executor.execute(operation))
    assert result.status is OperationStatus.FAILED
    assert result.reason is not None
    assert result.reason.startswith(REASON_TARGET_INVALID)
    assert transport.calls == []


# ------------------------------------------------------------- motion skills


@pytest.mark.parametrize("skill", ["PICK", "TRANSIT", "PLACE_DOWN"])
def test_motion_skills_fail_capability_unavailable_before_any_motion(
    skill,
) -> None:
    """The three motion skills never observe and never claim evidence."""
    transport = RecordingTransport({})
    executor = make_executor(transport)
    result = asyncio.run(executor.execute(motion_operation(skill)))
    assert result.status is OperationStatus.FAILED
    assert result.passed is False
    assert result.terminal is True
    assert result.motion_sent is False
    assert result.evidence == {}
    assert result.reason == REASON_CAPABILITY_UNAVAILABLE
    assert transport.calls == []


@pytest.mark.parametrize(
    ("skill", "target"),
    [
        ("PICK", {"physical_instance_id": BRICK_A, "extra": 1}),
        ("PICK", {"physical_instance_id": ""}),
        ("PICK", None),
        (
            "TRANSIT",
            {
                "from": reference("robot_current"),
                "to": reference("part", physical_instance_id=BRICK_A),
            },
        ),
        (
            "TRANSIT",
            {
                "from": reference("robot_current"),
                "to": reference(
                    "part", physical_instance_id=BRICK_A, robot_id=ROBOT_1
                ),
                "carrying_physical_instance_id": None,
            },
        ),
        (
            "TRANSIT",
            {
                "from": reference("robot_home"),
                "to": reference("connection", connection_id="conn-1"),
                "carrying_physical_instance_id": BRICK_A,
            },
        ),
        ("PLACE_DOWN", {"connection_id": "conn-1"}),
        (
            "PLACE_DOWN",
            {
                "connection_id": "conn-1",
                "upper_physical_instance_id": BRICK_A,
                "lower_physical_instance_id": 7,
            },
        ),
    ],
)
def test_motion_skills_reject_invalid_targets_before_any_motion(
    skill, target
) -> None:
    """A malformed target fails closed as TARGET_INVALID without motion."""
    transport = RecordingTransport({})
    executor = make_executor(transport)
    result = asyncio.run(
        executor.execute(motion_operation(skill, target=target))
    )
    assert result.status is OperationStatus.FAILED
    assert result.terminal is True
    assert result.motion_sent is False
    assert result.evidence == {}
    assert result.reason is not None
    assert result.reason.startswith(REASON_TARGET_INVALID)
    assert transport.calls == []


@pytest.mark.parametrize("skill", ["PICK", "TRANSIT", "PLACE_DOWN", "WAIT"])
def test_unvalidated_executor_robot_fails_before_observing(skill) -> None:
    """An executor robot id outside the validated set is never dispatched."""
    transport = RecordingTransport({})
    executor = make_executor(transport)
    if skill == "WAIT":
        operation = wait_operation(robot_id="piper_9")
    else:
        operation = motion_operation(skill, robot_id="piper_9")
    result = asyncio.run(executor.execute(operation))
    assert result.status is OperationStatus.FAILED
    assert result.terminal is True
    assert result.motion_sent is False
    assert result.evidence == {}
    assert result.reason is not None
    assert result.reason.startswith(REASON_ROBOT_NOT_VALIDATED)
    assert transport.calls == []


# -------------------------------------------------------------------- cancel


def test_cancel_of_an_unknown_operation_fails_closed() -> None:
    """An id this executor never dispatched gets no safety claim."""
    executor = make_executor(RecordingTransport({}))
    evidence = asyncio.run(executor.cancel("never-dispatched"))
    assert isinstance(evidence, CancelEvidence)
    assert evidence.operation_id == "never-dispatched"
    assert evidence.acknowledged is False
    assert evidence.terminal is False
    assert evidence.safe is False
    assert is_json_safe(evidence.evidence)
    assert "cancel_error" in evidence.evidence


def test_cancel_of_an_active_wait_is_terminal_and_safe_without_motion() -> None:
    """Cancelling a read-only wait stops it and proves safety explicitly."""
    transport = RecordingTransport(lambda robot_id, index: scene_document(10.0))
    executor = make_executor(transport, wait_timeout_s=30.0)
    operation = wait_operation(duration_s=1000.0)

    async def scenario() -> tuple[OperationExecution, CancelEvidence]:
        task = asyncio.ensure_future(executor.execute(operation))
        await asyncio.sleep(0.05)
        duplicate = await executor.execute(operation)
        assert duplicate.status is OperationStatus.FAILED
        assert duplicate.reason == REASON_OPERATION_ALREADY_ACTIVE
        evidence = await executor.cancel(operation.operation_id)
        result = await task
        return result, evidence

    result, evidence = asyncio.run(scenario())
    assert isinstance(evidence, CancelEvidence)
    assert evidence.acknowledged is True
    assert evidence.terminal is True
    assert evidence.safe is True
    assert evidence.evidence["motion_sent"] is False
    assert result.status is OperationStatus.CANCELLED
    assert result.terminal is True
    assert result.passed is False
    assert result.motion_sent is False
    assert LABEL_REAL_SCENE_OBSERVATION not in result.evidence
    assert LABEL_WAIT_DURATION_CONFIRMED not in result.evidence
    assert transport.calls


def test_cancel_after_a_confirmed_wait_reports_a_terminal_safe_state() -> None:
    """A finished no-motion operation may still answer a late cancel."""
    executor = make_executor(advancing_transport())
    operation = wait_operation(duration_s=1.0)
    result = asyncio.run(executor.execute(operation))
    assert result.status is OperationStatus.CONFIRMED
    evidence = asyncio.run(executor.cancel(operation.operation_id))
    assert evidence.acknowledged is True
    assert evidence.terminal is True
    assert evidence.safe is True
    assert evidence.evidence["already_finished"] is True
    assert evidence.evidence["motion_sent"] is False


def test_cancel_rejects_a_non_string_operation_id() -> None:
    """A malformed cancel id is a programming error, not a safety claim."""
    executor = make_executor(RecordingTransport({}))
    with pytest.raises(ValueError):
        asyncio.run(executor.cancel(""))


# ------------------------------------------------------ scheduler integration


def _wait_plan(duration_s: float) -> dict:
    """Build one scheduler plan with a WAIT on piper_1.

    Args:
        duration_s: Requested wait duration.

    Returns:
        The ``robot-operation-plan/v1`` payload.
    """
    return {
        "schema": "robot-operation-plan/v1",
        "plan_id": "plan-a7-wait",
        "source_plan_id": "source-a7",
        "robots": list(ROBOTS),
        "operations": [
            {
                "operation_id": "plan-wait",
                "source_task_id": "task-wait",
                "skill": "WAIT",
                "executor_robot_id": ROBOT_1,
                "participants": [ROBOT_1],
                "role_bindings": {"executor": ROBOT_1},
                "target": None,
                "parameters": {
                    "observation": {
                        "kind": "structural_fact",
                        "fact": "stable",
                        "physical_instance_id": BRICK_A,
                    },
                    "duration_s": duration_s,
                },
                "preconditions": [],
                "effects": [],
                "resources": [f"robot:{ROBOT_1}"],
                "evidence_requirements": [
                    LABEL_REAL_SCENE_OBSERVATION,
                    LABEL_WAIT_DURATION_CONFIRMED,
                ],
            }
        ],
        "dependencies": [],
        "execution_layers": [["plan-wait"]],
        "planned_schedule": [
            [{"operation_id": "plan-wait", "robot_id": ROBOT_1}]
        ],
        "evidence_requirements": [
            {
                "operation_id": "plan-wait",
                "evidence": LABEL_WAIT_DURATION_CONFIRMED,
            }
        ],
    }


def test_scheduler_confirms_the_wait_and_fails_the_motion_skill() -> None:
    """The A2 scheduler consumes the A7 executor without legacy shims."""
    payload = _wait_plan(duration_s=1.0)
    payload["operations"].append(
        {
            "operation_id": "plan-transit",
            "source_task_id": "task-transit",
            "skill": "TRANSIT",
            "executor_robot_id": ROBOT_0,
            "participants": [ROBOT_0],
            "role_bindings": {"executor": ROBOT_0},
            "target": {
                "from": reference("robot_current"),
                "to": reference("part", physical_instance_id=BRICK_A),
                "carrying_physical_instance_id": None,
            },
            "parameters": {},
            "preconditions": [],
            "effects": [],
            "resources": [f"robot:{ROBOT_0}"],
            "evidence_requirements": ["transit_feedback_confirmed"],
        }
    )
    payload["execution_layers"] = [["plan-wait"], ["plan-transit"]]
    payload["planned_schedule"] = [
        [{"operation_id": "plan-wait", "robot_id": ROBOT_1}],
        [{"operation_id": "plan-transit", "robot_id": ROBOT_0}],
    ]
    payload["dependencies"] = [
        {
            "predecessor_operation_id": "plan-wait",
            "successor_operation_id": "plan-transit",
            "reason": "test ordering",
        }
    ]
    plan = parse_robot_operation_plan(payload)
    transport = advancing_transport(step_s=0.4)
    executor = make_executor(transport)
    scheduler = MultiRobotScheduler(
        operation_timeout_s=5.0, allow_legacy_results=False
    )
    result = asyncio.run(scheduler.schedule(plan, executor))
    assert result.passed is False
    wait = result.operations["plan-wait"]
    assert wait.status is OperationStatus.CONFIRMED
    assert wait.passed is True
    assert wait.terminal is True
    assert wait.motion_sent is False
    assert wait.evidence[LABEL_REAL_SCENE_OBSERVATION]
    assert wait.evidence[LABEL_WAIT_DURATION_CONFIRMED]
    transit = result.operations["plan-transit"]
    assert transit.status is OperationStatus.FAILED
    assert transit.passed is False
    assert transit.terminal is True
    assert transit.motion_sent is False
    assert transit.evidence == {}
    assert transit.reason == REASON_CAPABILITY_UNAVAILABLE
    assert set(transport.calls) == {ROBOT_1}


def test_scheduler_timeout_cancel_is_safe_without_motion() -> None:
    """A timed-out wait acknowledges a safe terminal cancel; no motion."""
    payload = _wait_plan(duration_s=1000.0)
    plan = parse_robot_operation_plan(payload)
    transport = RecordingTransport(lambda robot_id, index: scene_document(10.0))
    executor = make_executor(
        transport, poll_interval_s=0.005, wait_timeout_s=60.0
    )
    scheduler = MultiRobotScheduler(
        operation_timeout_s=0.25,
        cancel_timeout_s=1.0,
        allow_legacy_results=False,
    )
    result = asyncio.run(scheduler.schedule(plan, executor))
    assert result.passed is False
    assert result.unsafe_terminals == ()
    operation = result.operations["plan-wait"]
    assert operation.status is OperationStatus.CANCELLED
    assert operation.terminal is True
    assert operation.motion_sent is False
    assert operation.deadline_exceeded is True
    assert len(result.cancellations) == 1
    cancel = result.cancellations[0]
    assert cancel.acknowledged is True
    assert cancel.terminal is True
    assert cancel.safe is True
    assert cancel.evidence["motion_sent"] is False


def test_executor_does_not_mutate_the_dispatched_operation() -> None:
    """The executor returns frozen results and keeps the operation intact."""
    operation = wait_operation(duration_s=1.0)
    before = dataclasses.asdict(operation)
    asyncio.run(make_executor(advancing_transport()).execute(operation))
    assert dataclasses.asdict(operation) == before
