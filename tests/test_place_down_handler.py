"""Offline admission and evidence tests for the PLACE_DOWN handler."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from rocobrick.adapters.moveit.transit import GroundSpec, TransitContext
from rocobrick.execution.operation_state_ledger import OperationStateLedger
from rocobrick.execution.robot_operation_executor import SkillOperationExecutor
from rocobrick.isaac_backend import OperationKind, RuntimeOperation
from rocobrick.multi_robot_scheduler import OperationStatus
from rocobrick.robot_operation_plan import OperationTarget, RobotOperation
from rocobrick.skills.observation import (
    parse_place_observation,
    verify_place_bricks,
)
from rocobrick.skills.place_down import PlaceDownHandler
from rocobrick.skills.targets import (
    RobotBaseFrame,
    SkillParams,
    TcpProvenance,
)

ROBOT = "piper_0"
UPPER = "brick-upper"
LOWER = "brick-lower"
SESSION = "session-1"
MANIFEST = "a" * 64
MODEL_SHA = "a781d52f9d55291fca9c8f2695abf9e311640d996314d272f47639d5323e2375"
SOURCE_SHA = "3ba48537ed4f453c7e2507e042056128515e8a064adbab7a4c871b3ed2bc94cb"
GRASP_ORIENTATION = (1.0, 0.0, 0.0, 0.0)  # qz(0) ⊗ qx(180°), yaw 0 layout
BRICK_ORIENTATION = (0.0, 0.0, 0.0, 1.0)  # upright brick in world frame
# Lower brick at rest; upper held/placed with the same footprint.
LOWER_AABB = ((0.30, 0.30, 0.0112), (0.34, 0.34, 0.0224))
UPPER_HEIGHT = 0.0112


def provenance() -> TcpProvenance:
    """Return the calibrated TCP provenance of the test."""
    return TcpProvenance(runtime_model_sha256=MODEL_SHA, source_sha256=SOURCE_SHA)


def context() -> TransitContext:
    """Return one identity-base transit context with the frozen params."""
    return TransitContext(
        base=RobotBaseFrame(
            position_xyz=(0.0, 0.0, 0.0), orientation_xyzw=(0.0, 0.0, 0.0, 1.0)
        ),
        provenance=provenance(),
        params=SkillParams(),
        ground=GroundSpec(
            center_xy_m=(0.0, -0.2),
            size_xy_m=(2.0, 2.0),
            top_z_m=0.0,
            thickness_m=0.02,
        ),
    )


def operation() -> RobotOperation:
    """Build one formal PLACE_DOWN operation from its public contract.

    Returns:
    The operation targeting the upper/lower pair with piper_0.
    """
    return RobotOperation(
        operation_id="place-1",
        source_task_id="task-1",
        skill="PLACE_DOWN",
        executor_robot_id=ROBOT,
        participants=(ROBOT,),
        role_bindings={"executor": ROBOT},
        target=OperationTarget(
            {
                "connection_id": "conn-1",
                "upper_physical_instance_id": UPPER,
                "lower_physical_instance_id": LOWER,
            }
        ),
        parameters={},
        preconditions=("robot_at_place_region",),
        effects=(f"placed:{UPPER}",),
        resources=(f"robot:{ROBOT}", f"part:{UPPER}", f"part:{LOWER}"),
        evidence_requirements=(
            "connection_confirmed",
            "release_confirmed",
            "retreat_confirmed",
        ),
    )


def document(
    *,
    sequence: int,
    upper_z: float,
    upper_xy: tuple[float, float] = (0.30, 0.34),
    tcp_z: float,
    connection: bool,
) -> dict:
    """Return one co-sampled identified scene with both bricks."""
    upper_min = (upper_xy[0], 0.30, upper_z)
    upper_max = (upper_xy[1], 0.34, upper_z + UPPER_HEIGHT)
    return {
        "schema": "rocobrick-scene-state/v1",
        "session_id": SESSION,
        "manifest_sha256": MANIFEST,
        "sequence": sequence,
        "sim_time_s": float(sequence),
        "bricks": {
            LOWER: {
                "aabb_min": list(LOWER_AABB[0]),
                "aabb_max": list(LOWER_AABB[1]),
                "pose": {
                    "position_xyz": [0.32, 0.32, 0.0168],
                    "orientation_xyzw": list(BRICK_ORIENTATION),
                },
            },
            UPPER: {
                "aabb_min": list(upper_min),
                "aabb_max": list(upper_max),
                "pose": {
                    "position_xyz": [
                        (upper_min[0] + upper_max[0]) / 2.0,
                        0.32,
                        upper_z + UPPER_HEIGHT / 2.0,
                    ],
                    "orientation_xyzw": list(BRICK_ORIENTATION),
                },
            },
        },
        "connections": {f"{LOWER}|{UPPER}": connection},
        "placement_interfaces": {
            "pairs": {
                f"{LOWER}|{UPPER}": {
                    "actual_connection": (
                        {
                            "stud_interface": 1,
                            "hole_interface": 0,
                            "lower_to_upper": {
                                "position_xyz": [0.0, 0.0, 0.0112],
                                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                            },
                            "offset": [0, 0],
                            "yaw": 0,
                        }
                        if connection
                        else None
                    ),
                    "stud_interface": 1,
                    "hole_interface": 0,
                    "contact_force_n": 0.0,
                    "lower_physical_instance_id": LOWER,
                    "upper_physical_instance_id": UPPER,
                    "lower_to_upper": {
                        "position_xyz": [0.0, 0.0, 0.0112],
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                }
            }
        },
        "robots": {
            ROBOT: {
                "robot_id": ROBOT,
                "sequence": sequence,
                "sim_time_s": float(sequence),
                "opening_m": 0.012,
                "tcp_pose": {
                    "position_xyz": [0.32, 0.32, tcp_z],
                    "orientation_xyzw": list(GRASP_ORIENTATION),
                },
            }
        },
    }


def script(
    *, placed_tcp_z: float = 0.0224 + 0.0035, connection_break: int | None = None
) -> list[dict]:
    """Script the advancing observations of one successful place.

    Returns:
        The scripted observation documents in consumption order.

    Observation 1 is the preflight (held high, not connected); 2 is the
    post-transport alignment check; 3-5 prove the connection on the placed
    stack; 6-8 prove the release; 9 rebuilds the placed world; 10-12 prove
    the retreat with the TCP risen.
    """
    documents: list[dict] = []
    placed_upper_z = LOWER_AABB[1][2]
    for index in range(16):
        sequence = index + 4
        if index == 0:
            documents.append(
                document(
                    sequence=sequence,
                    upper_z=0.12,
                    tcp_z=0.14,
                    connection=False,
                )
            )
        elif index == 1:
            documents.append(
                document(
                    sequence=sequence,
                    upper_z=0.10,
                    tcp_z=0.12,
                    connection=False,
                )
            )
        else:
            connected = not (connection_break is not None and index >= connection_break)
            risen = index >= 11
            documents.append(
                document(
                    sequence=sequence,
                    upper_z=placed_upper_z,
                    tcp_z=(placed_tcp_z + 0.05) if risen else placed_tcp_z,
                    connection=connected,
                )
            )
    return documents


class Observer:
    """Robot-scoped scene transport replaying one scripted observation list."""

    def __init__(self, documents: list[dict]) -> None:
        """Store the scripted observations."""
        self.documents = list(documents)
        self.calls: list[str] = []

    async def observe(self, robot_id: str) -> dict:
        """Return the next observation while recording the robot."""
        self.calls.append(robot_id)
        if not self.documents:
            raise AssertionError("the scripted observations were exhausted")
        return self.documents.pop(0)


class Motion:
    """Runtime transport that succeeds every phase with strict records."""

    def __init__(self) -> None:
        """Prepare the dispatch log."""
        self.dispatched: list[tuple[str, str]] = []
        self.release_window: tuple[float, float] | None = None

    def run_motion_operation(self, robot_id: str, operation: RuntimeOperation) -> dict:
        """Record the phase and answer a strict success record.

        Returns:
            The strict runtime record of this dispatched phase.
        """
        self.dispatched.append((operation.kind.value, operation.request_id))
        if operation.kind is OperationKind.GRIPPER_TARGET:
            window = self.release_window or (0.0, 0.09)
            return {
                "motion_sent": False,
                "result": {
                    "passed": True,
                    "execution": {
                        "goal_accepted": True,
                        "terminal": True,
                        "safe_to_release": True,
                        "action_status": "SUCCEEDED",
                        "command_messages": 1,
                        "details": {
                            "measured_opening_m": sum(window) / 2.0,
                            "expected_opening_m": list(window),
                            "post_result_state_messages": 3,
                        },
                    },
                    "kind": operation.kind.value,
                },
            }
        if operation.kind in (
            OperationKind.APPLY_PLANNING_SCENE,
            OperationKind.ATTACH_PLANNING_SCENE,
        ):
            return {
                "motion_sent": False,
                "result": {
                    "passed": True,
                    "execution": {"readback_confirmed": True},
                    "kind": operation.kind.value,
                },
            }
        return {
            "motion_sent": False,
            "result": {
                "passed": True,
                "execution": {
                    "terminal": True,
                    "safe_to_release": True,
                    "command_messages": 1,
                },
                "kind": operation.kind.value,
            },
        }

    def cancel_motion_operation(self, operation_id: str) -> dict:
        """Answer one inactive cancel.

        Returns:
            The inert cancel evidence document.
        """
        return {"attempted": False, "terminal": False, "safe": False}


def holding_ledger() -> OperationStateLedger:
    """Return a ledger whose piper_0 confirmedly holds the upper brick."""
    ledger = OperationStateLedger((ROBOT, "piper_1"))
    ledger.confirm_holding(ROBOT, UPPER, {"holding_confirmed": True})
    return ledger


def direct_operation() -> RobotOperation:
    """The direct-dispatch form whose arrival precondition was stripped.

    Returns:
        The same operation without the place-region arrival precondition.
    """
    return replace(operation(), preconditions=())


def handler(
    observer: Observer, ledger: OperationStateLedger, motion: Motion | None = None
) -> PlaceDownHandler:
    """Build the handler under test.

    Returns:
    The PlaceDownHandler wired to the test doubles.
    """
    return PlaceDownHandler(
        observer,
        ledger,
        session_id=SESSION,
        manifest_sha256=MANIFEST,
        motion_transport=motion,
        context=context(),
    )


def test_parse_place_observation_requires_the_live_pair_verdict() -> None:
    """The exact pair connection verdict must be present and boolean."""
    good = document(sequence=5, upper_z=0.10, tcp_z=0.12, connection=True)
    sample = parse_place_observation(
        good,
        robot_id=ROBOT,
        upper_physical_instance_id=UPPER,
        lower_physical_instance_id=LOWER,
        session_id=SESSION,
        manifest_sha256=MANIFEST,
    )
    assert sample.connection is True
    assert sample.upper.physical_instance_id == UPPER
    assert sample.lower.physical_instance_id == LOWER
    bad = document(sequence=6, upper_z=0.10, tcp_z=0.12, connection=True)
    bad["connections"] = {}
    with pytest.raises(ValueError, match="connection"):
        parse_place_observation(
            bad,
            robot_id=ROBOT,
            upper_physical_instance_id=UPPER,
            lower_physical_instance_id=LOWER,
            session_id=SESSION,
            manifest_sha256=MANIFEST,
        )


def test_verify_place_bricks_enforces_connection_stability_and_rise() -> None:
    """Connection, stillness and TCP rise are separate hard requirements."""

    def sample(
        sequence: int, *, connection: bool, rise: float = 0.0, shift: float = 0.0
    ) -> object:
        return parse_place_observation(
            document(
                sequence=sequence,
                upper_z=LOWER_AABB[1][2] + shift,
                tcp_z=0.06 + rise,
                connection=connection,
            ),
            robot_id=ROBOT,
            upper_physical_instance_id=UPPER,
            lower_physical_instance_id=LOWER,
            session_id=SESSION,
            manifest_sha256=MANIFEST,
        )

    okay = verify_place_bricks(
        [sample(3, connection=True), sample(4, connection=True, rise=0.03)],
        position_tolerance_m=0.01,
        orientation_tolerance_rad=0.05,
        minimum_tcp_rise_m=0.02,
    )
    assert okay["connected"] is True
    assert okay["tcp_rise_m"] == pytest.approx(0.03)
    with pytest.raises(ValueError, match="connection verdict"):
        verify_place_bricks(
            [sample(3, connection=True), sample(4, connection=False)],
            position_tolerance_m=0.01,
            orientation_tolerance_rad=0.05,
        )
    with pytest.raises(ValueError, match="moved beyond"):
        verify_place_bricks(
            [sample(3, connection=True), sample(4, connection=True, shift=0.02)],
            position_tolerance_m=0.01,
            orientation_tolerance_rad=0.05,
        )
    with pytest.raises(ValueError, match="did not rise"):
        verify_place_bricks(
            [sample(3, connection=True), sample(4, connection=True)],
            position_tolerance_m=0.01,
            orientation_tolerance_rad=0.05,
            minimum_tcp_rise_m=0.02,
        )
    with pytest.raises(ValueError, match="stale or reordered"):
        verify_place_bricks(
            [sample(5, connection=True), sample(5, connection=True)],
            position_tolerance_m=0.01,
            orientation_tolerance_rad=0.05,
        )


def test_place_preflight_rejects_unproven_carrying_before_motion() -> None:
    """Without a confirmed holding fact no motion may ever be sent."""
    observer = Observer(script())
    ledger = OperationStateLedger((ROBOT,))
    result = asyncio.run(handler(observer, ledger).execute(direct_operation()))
    assert result.status is OperationStatus.FAILED
    assert result.motion_sent is False
    assert result.evidence == {}
    assert result.reason.startswith("PLACE_DOWN_PREFLIGHT_REJECTED")
    assert "carrying" in result.reason


@pytest.mark.parametrize("defect", ["foreign_owner", "same_part", "bad_shape"])
def test_place_preflight_rejects_invalid_targets(defect: str) -> None:
    """Pair identity and shape are motion gates."""
    observer = Observer(script())
    ledger = holding_ledger()
    op = operation()
    if defect == "foreign_owner":
        ledger2 = OperationStateLedger(("piper_1",))
        ledger2.confirm_holding("piper_1", UPPER, {"holding_confirmed": True})
        ledger = ledger2
    elif defect == "same_part":
        op = replace(
            op,
            target=OperationTarget(
                {
                    "connection_id": "conn-1",
                    "upper_physical_instance_id": UPPER,
                    "lower_physical_instance_id": UPPER,
                }
            ),
        )
    else:
        op = replace(
            op,
            target=OperationTarget({"connection_id": "conn-1"}),
        )
    result = asyncio.run(handler(observer, ledger).execute(op))
    assert result.status is OperationStatus.FAILED
    assert result.motion_sent is False
    assert result.reason.startswith("PLACE_DOWN_PREFLIGHT_REJECTED")


def test_place_executes_all_phases_and_commits_placement() -> None:
    """The full program confirms only with all three separated proofs."""
    motion = Motion()
    observer = Observer(script())
    ledger = holding_ledger()
    result = asyncio.run(handler(observer, ledger, motion).execute(direct_operation()))
    assert result.status is OperationStatus.CONFIRMED
    assert result.reason == "PLACE_DOWN_CONFIRMED"
    assert result.motion_sent is True
    kinds = [kind for kind, _ in motion.dispatched]
    assert kinds == [
        "APPLY_PLANNING_SCENE",  # plan_place
        "TCP_POSE",  # transport
        "TCP_POSE",  # descend
        "GRIPPER_TARGET",  # release
        "ATTACH_PLANNING_SCENE",  # detach
        "APPLY_PLANNING_SCENE",  # world_placed
    ]
    assert result.evidence["handoff_required"] is True
    assert "retreat_confirmed" not in result.evidence
    placed = ledger.observe(ROBOT, f"placed:{UPPER}")
    assert placed is not None
    assert placed.evidence["connection_id"] == "conn-1"
    assert placed.evidence["handoff_required"] is True
    assert ledger.owner_of_object(UPPER) is None
    assert ledger.observe(ROBOT, f"holding:{UPPER}") is None


def test_place_fails_closed_when_the_connection_never_confirms() -> None:
    """A dropped connection keeps the part held; nothing is committed."""
    motion = Motion()
    observer = Observer(script(connection_break=4))
    ledger = holding_ledger()
    result = asyncio.run(handler(observer, ledger, motion).execute(direct_operation()))
    assert result.status is OperationStatus.FAILED
    assert result.motion_sent is True
    assert "PLACE_DOWN_FAILED" in result.reason
    assert "connection verdict" in result.reason
    assert ledger.owner_of_object(UPPER) == ROBOT
    assert ledger.observe(ROBOT, f"placed:{UPPER}") is None


def test_probe_contact_stops_at_first_touch_and_fails_after_withdrawal() -> None:
    """Probe mode never places: first contact, one withdrawal, closed failure."""

    class ProbeStopMotion(Motion):
        def run_motion_operation(self, robot_id, operation):
            if operation.kind is OperationKind.GUARDED_TCP_POSE and (
                ":insert_descent" in operation.request_id
            ):
                return {
                    "motion_sent": True,
                    "result": {
                        "passed": True,
                        "execution": {
                            "goal_accepted": True,
                            "terminal": True,
                            "safe_to_release": True,
                            "action_status": "CANCELED",
                            "command_messages": 2,
                            "details": {
                                "contact_stop_reason": "probe_contact_detected",
                                "contact_trace": [
                                    document(
                                        sequence=30,
                                        upper_z=0.0224,
                                        tcp_z=0.030,
                                        connection=False,
                                    )
                                ],
                            },
                        },
                        "kind": operation.kind.value,
                    },
                }
            return super().run_motion_operation(robot_id, operation)

    documents = [
        document(sequence=40, upper_z=0.12, tcp_z=0.14, connection=False),
        document(sequence=41, upper_z=0.10, tcp_z=0.12, connection=False),
        document(sequence=42, upper_z=0.03, tcp_z=0.05, connection=False),
        document(sequence=43, upper_z=0.03, tcp_z=0.05, connection=False),
        document(sequence=44, upper_z=0.03, tcp_z=0.05, connection=False),
        document(sequence=45, upper_z=0.042, tcp_z=0.062, connection=False),
    ]
    motion = ProbeStopMotion()
    observer = Observer(documents)
    ledger = holding_ledger()
    place_handler = PlaceDownHandler(
        observer,
        ledger,
        session_id=SESSION,
        manifest_sha256=MANIFEST,
        motion_transport=motion,
        context=context(),
        probe_contact=True,
    )
    result = asyncio.run(place_handler.execute(direct_operation()))
    assert result.status is OperationStatus.FAILED
    assert result.motion_sent is True
    assert "probe stopped at the first interface contact" in result.reason
    assert result.evidence["contact_guard"]["mode"] == "probe"
    assert result.evidence["contact_probe"]["stop_reason"] == "probe_contact_detected"
    unload = result.evidence["failure_unload"]
    assert unload["passed"] is True
    assert unload["tcp_rise_m"] == pytest.approx(0.012)
    assert result.terminal is True
    kinds = [kind for kind, _ in motion.dispatched]
    assert "GRIPPER_TARGET" not in kinds
    assert ledger.owner_of_object(UPPER) == ROBOT
    assert ledger.observe(ROBOT, f"placed:{UPPER}") is None


def test_place_fails_closed_when_alignment_is_off() -> None:
    """A held brick over the wrong XY never descends."""
    documents = script()
    documents[1] = document(
        sequence=documents[1]["sequence"],
        upper_z=0.10,
        upper_xy=(0.36, 0.40),
        tcp_z=0.12,
        connection=False,
    )
    motion = Motion()
    observer = Observer(documents)
    ledger = holding_ledger()
    result = asyncio.run(handler(observer, ledger, motion).execute(direct_operation()))
    assert result.status is OperationStatus.FAILED
    assert "measured grasp transform" in result.reason
    kinds = [kind for kind, _ in motion.dispatched]
    assert "GRIPPER_TARGET" not in kinds
    assert ledger.owner_of_object(UPPER) == ROBOT


def test_interface_lateral_offset_is_corrected_before_descent() -> None:
    """A stud-scale lateral offset triggers one measured correction, then place."""

    class TrackingMotion(Motion):
        def __init__(self) -> None:
            super().__init__()
            self.corrected: list[str] = []

        def run_motion_operation(self, robot_id, operation):
            record = super().run_motion_operation(robot_id, operation)
            if "align_correct" in operation.request_id:
                self.corrected.append(operation.request_id)
            return record

    documents = script()
    # Transport lands 3.5 mm off the stud interface — beyond the 2.5 mm
    # engagement bound — and the correction move brings the held brick back
    # over it before any guarded descent.
    documents[1] = document(
        sequence=documents[1]["sequence"],
        upper_z=0.10,
        upper_xy=(0.3035, 0.3435),
        tcp_z=0.12,
        connection=False,
    )
    motion = TrackingMotion()
    observer = Observer(documents)
    ledger = holding_ledger()
    result = asyncio.run(handler(observer, ledger, motion).execute(direct_operation()))
    assert result.status is OperationStatus.CONFIRMED
    assert result.reason == "PLACE_DOWN_CONFIRMED"
    assert motion.corrected == ["place-1:transport_align_correct_0"]
    corrections = result.evidence["place_alignment_corrections"]
    assert len(corrections) == 1
    assert corrections[0]["lateral_error_m"] == pytest.approx(0.0035, abs=1e-6)
    assert (
        result.evidence["place_alignment_measured"]["lateral_error_m"]
        <= SkillParams().place_interface_align_tolerance_m
    )


def test_uncorrectable_lateral_offset_fails_closed_before_contact() -> None:
    """A held brick that never reaches stud alignment never descends."""
    documents = [
        document(
            sequence=index + 4,
            upper_z=0.10 if index else 0.12,
            upper_xy=(0.305, 0.345),
            tcp_z=0.12 if index else 0.14,
            connection=False,
        )
        for index in range(6)
    ]
    motion = Motion()
    observer = Observer(documents)
    ledger = holding_ledger()
    result = asyncio.run(handler(observer, ledger, motion).execute(direct_operation()))
    assert result.status is OperationStatus.FAILED
    assert "not aligned to the interface" in result.reason
    corrections = result.evidence["place_alignment_corrections"]
    assert len(corrections) == 2
    kinds = [kind for kind, _ in motion.dispatched]
    assert "GUARDED_TCP_POSE" not in kinds
    assert "GRIPPER_TARGET" not in kinds
    assert result.terminal is False
    assert ledger.owner_of_object(UPPER) == ROBOT


def test_descent_stop_unloads_once_and_fails_without_retry() -> None:
    """Any non-connection descent stop withdraws once and fails closed."""

    class StoppedDescentMotion(Motion):
        def __init__(self) -> None:
            super().__init__()
            self.descents = 0

        def run_motion_operation(self, robot_id, operation):
            if operation.kind is OperationKind.GUARDED_TCP_POSE and (
                ":insert_descent" in operation.request_id
            ):
                self.descents += 1
                self.dispatched.append((operation.kind.value, operation.request_id))
                return {
                    "motion_sent": True,
                    "result": {
                        "passed": True,
                        "execution": {
                            "goal_accepted": True,
                            "terminal": True,
                            "safe_to_release": True,
                            "action_status": "ABORTED",
                            "command_messages": 3,
                            "details": {
                                "contact_stop_reason": "contact_tracking_stopped",
                            },
                        },
                        "kind": operation.kind.value,
                    },
                }
            return super().run_motion_operation(robot_id, operation)

    placed_upper_z = LOWER_AABB[1][2]
    # Held relation frozen at admission: tcp_z = upper centre + 0.0144 m.
    held_tcp_z = placed_upper_z + UPPER_HEIGHT / 2.0 + 0.0144
    documents = [
        # preflight, transport alignment, contact entry, pre-descent sample
        document(sequence=50, upper_z=0.12, tcp_z=0.14, connection=False),
        document(sequence=51, upper_z=0.10, tcp_z=0.12, connection=False),
        document(
            sequence=52, upper_z=placed_upper_z, tcp_z=held_tcp_z, connection=False
        ),
        document(
            sequence=53, upper_z=placed_upper_z, tcp_z=held_tcp_z, connection=False
        ),
        # post-stop settle re-inspect (still unconnected), unload start and
        # verified clearance
        document(
            sequence=54, upper_z=placed_upper_z, tcp_z=held_tcp_z, connection=False
        ),
        document(
            sequence=55, upper_z=placed_upper_z, tcp_z=held_tcp_z, connection=False
        ),
        document(
            sequence=56,
            upper_z=placed_upper_z + 0.012,
            tcp_z=held_tcp_z + 0.012,
            connection=False,
        ),
    ]
    motion = StoppedDescentMotion()
    observer = Observer(documents)
    ledger = holding_ledger()
    result = asyncio.run(handler(observer, ledger, motion).execute(direct_operation()))
    assert result.status is OperationStatus.FAILED
    assert "did not confirm the exact connection" in result.reason
    assert "contact_tracking_stopped" in result.reason
    assert motion.descents == 1
    request_ids = [request_id for _, request_id in motion.dispatched]
    assert request_ids.count("place-1:insert_descent") == 1
    assert not any("recovery" in rid for rid in request_ids)
    assert "GRIPPER_TARGET" not in [kind for kind, _ in motion.dispatched]
    unload = result.evidence["failure_unload"]
    assert unload["passed"] is True
    assert unload["tcp_rise_m"] == pytest.approx(0.012)
    assert result.terminal is True
    assert ledger.owner_of_object(UPPER) == ROBOT
    assert ledger.observe(ROBOT, f"placed:{UPPER}") is None


def test_place_fails_closed_when_the_held_orientation_mismatches() -> None:
    """A brick held in the wrong orientation is refused before any motion."""
    documents = script()
    documents[0]["bricks"][UPPER]["pose"]["orientation_xyzw"] = [
        0.0,
        0.0,
        0.7071067811865476,
        0.7071067811865476,
    ]
    observer = Observer(documents)
    ledger = holding_ledger()
    result = asyncio.run(
        handler(observer, ledger, Motion()).execute(direct_operation())
    )
    assert result.status is OperationStatus.FAILED
    assert result.motion_sent is True
    assert "measured grasp transform" in result.reason


def test_place_fails_closed_on_unproven_release_feedback() -> None:
    """A release without measured in-window feedback never detaches."""

    class BadReleaseMotion(Motion):
        def run_motion_operation(self, robot_id, operation):
            if operation.kind is OperationKind.GRIPPER_TARGET:
                return {
                    "motion_sent": False,
                    "result": {
                        "passed": True,
                        "execution": {
                            "goal_accepted": True,
                            "terminal": True,
                            "safe_to_release": True,
                            "action_status": "SUCCEEDED",
                            "command_messages": 1,
                            "details": {
                                "measured_opening_m": 0.001,
                                "expected_opening_m": [0.05, 0.07],
                                "post_result_state_messages": 3,
                            },
                        },
                        "kind": operation.kind.value,
                    },
                }
            return super().run_motion_operation(robot_id, operation)

    motion = BadReleaseMotion()
    observer = Observer(script())
    ledger = holding_ledger()
    result = asyncio.run(handler(observer, ledger, motion).execute(direct_operation()))
    assert result.status is OperationStatus.FAILED
    assert "did not prove the commanded opening" in result.reason
    kinds = [kind for kind, _ in motion.dispatched]
    assert "ATTACH_PLANNING_SCENE" not in kinds
    assert ledger.owner_of_object(UPPER) == ROBOT


def test_executor_place_dispatch_gates_on_confirmed_carrying() -> None:
    """The executor refuses PLACE_DOWN before the handler without holding."""

    class ExplodingBackend:
        async def execute(self, operation: RobotOperation) -> None:
            raise AssertionError("the handler must not run")

    class Transport:
        async def observe(self, robot_id: str) -> dict:
            return {}

    executor = SkillOperationExecutor(
        (ROBOT,),
        Transport(),
        place_backends={ROBOT: ExplodingBackend()},
        state_ledger=OperationStateLedger((ROBOT,)),
    )
    result = asyncio.run(executor.execute(operation()))
    assert result.status is OperationStatus.FAILED
    assert result.motion_sent is False
    assert result.reason.startswith("PLACE_DOWN_CARRYING_UNCONFIRMED")


def test_world_payload_authorizes_only_the_place_contact_pair() -> None:
    """The floor stays hard; only the upper/lower pair is authorized."""
    observer = Observer(script())
    ledger = holding_ledger()
    place_handler = handler(observer, ledger)
    admission = asyncio.run(place_handler.preflight(direct_operation()))
    payload = place_handler._world_payload(
        admission,
        context(),
        admission.sample,
        include_upper=False,
        scene_suffix="plan_place",
    )
    boxes = {box["object_id"]: box for box in payload["boxes"]}
    assert set(boxes) == {LOWER, "piper_0_ground"}
    assert set(map(tuple, payload["allowed_collision_pairs"])) == {(LOWER, UPPER)}
