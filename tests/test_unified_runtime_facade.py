"""Tests for the unified transport facade's complete scheduler protocol."""

import asyncio
from types import SimpleNamespace

import pytest
from pre_phase_helpers import request, success

from rocobrick.execution.operation_state_ledger import OperationStateLedger
from rocobrick.adapters.moveit.transit import TransitContext
from rocobrick.skills.targets import RobotBaseFrame, SkillParams, TcpProvenance
from rocobrick.unified_dual_arm_runtime import (
    SharedSceneSnapshot,
    TerminalPrePhasePolicy,
)
from rocobrick.unified_runtime_facade import UnifiedDualArmRuntimeFacade

ROBOTS = ("piper_0", "piper_1")


class FakeSession:
    """Persistent-session double exposing the unified observation boundary."""

    def __init__(
        self,
        *,
        readiness_passed=True,
        version=1,
        sim_time_s=1.0,
        observed_sim_time_s=1.05,
    ):
        self._shared = {
            "scene_snapshot": {
                "session_id": "session-1",
                "manifest_sha256": "a" * 64,
                "sequence": 7,
                "sim_time_s": sim_time_s,
                "robots": {
                    robot: {"robot_id": robot, "gripper": {}, "opening_m": 0.0}
                    for robot in ROBOTS
                },
            },
            "planning_scene_version": version,
            "observed_sim_time_s": observed_sim_time_s,
        }
        self.readiness = SimpleNamespace(
            passed=readiness_passed,
            readiness_by_robot={
                robot: {"passed": readiness_passed} for robot in ROBOTS
            },
        )
        self.execution_result = None

    def observe_all(self):
        """Return the raw shared scene document."""
        return {
            **self._shared["scene_snapshot"],
            "scene_digest": "c" * 64,
        }

    def execute_coordinated_pre_phase(self, coordinated):
        """Return the scripted combined execution result."""
        if self.execution_result is None:
            raise RuntimeError("no scripted execution result")
        self.last_coordination_id = coordinated.coordination_id
        return self.execution_result

    def observe_shared_scene(self):
        """Return the one complete shared observation document."""
        return self._shared


def ledger_with_holding():
    """Return a ledger with one confirmed holding fact for piper_0."""
    item = OperationStateLedger(ROBOTS)
    item.confirm_holding("piper_0", "brick-x", {"proof": {"source": "measure"}})
    return item


def _context():
    """Return a real dataclass context for facade serialization."""
    return TransitContext(
        RobotBaseFrame((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        TcpProvenance("a" * 64, "b" * 64),
        SkillParams(),
    )


def test_facade_does_not_promote_planning_to_execution():
    """Without contexts or an execution transport no physical arrival is claimed."""
    result = asyncio.run(
        UnifiedDualArmRuntimeFacade(FakeSession()).execute_pre_phase(request())
    )
    assert not result.passed
    assert not result.started
    assert not result.arrival_confirmed
    assert not result.adoption_ready
    assert "motion successor and contexts" in (result.reason or "")
    no_transport = UnifiedDualArmRuntimeFacade(
        FakeSession(), contexts={"piper_0": _context()}, state_ledger=ledger()
    )
    denied = asyncio.run(no_transport.execute_pre_phase(request()))
    assert not denied.passed
    assert "no scripted execution result" in (denied.reason or "")


def test_terminal_cleanup_is_fail_closed_until_runtime_exists():
    """Explicit authorization alone cannot prove cleanup execution."""
    policy = TerminalPrePhasePolicy(
        "cleanup", "b" * 64, ("piper_0",), {"piper_0": {"pose": "explicit"}}
    )
    result = asyncio.run(
        UnifiedDualArmRuntimeFacade(FakeSession()).execute_pre_phase(
            request(policy=policy)
        )
    )
    assert not result.passed
    assert "cleanup" in result.reason


def test_facade_exposes_the_complete_scheduler_protocol():
    """Every pre-phase seam the scheduler requires is a callable boundary."""
    facade = UnifiedDualArmRuntimeFacade(FakeSession(), state_ledger=ledger())
    for name in (
        "observe_pre_phase",
        "execute_pre_phase",
        "validate_pre_phase",
        "cancel_pre_phase",
        "terminal_pre_phase_policy",
    ):
        assert callable(getattr(facade, name)), name
    cancellation = asyncio.run(facade.cancel_pre_phase("transition-1"))
    assert cancellation["terminal"] is False
    assert cancellation["safe"] is False
    assert facade.terminal_pre_phase_policy(()) is None


def ledger():
    """Return an empty confirmed-state ledger for both robots."""
    return OperationStateLedger(ROBOTS)


def test_observe_pre_phase_admits_one_shared_observation():
    """One shared read becomes an admitted snapshot with ledger facts."""
    facade = UnifiedDualArmRuntimeFacade(
        FakeSession(version=3), state_ledger=ledger_with_holding()
    )
    snapshot = asyncio.run(facade.observe_pre_phase(("piper_0",)))
    assert isinstance(snapshot, SharedSceneSnapshot)
    assert snapshot.scene_version == 3
    block = snapshot.document["robots"]["piper_0"]
    assert block["authorized"] is True
    assert block["terminal_safe"] is True
    assert block["holding_known"] is True
    assert block["supporting_known"] is True
    assert block["facts"] == {
        "holding:brick-x": {"proof": {"source": "measure"}}
    }
    assert snapshot.document["robots"]["piper_1"]["facts"] == {}
    assert len(snapshot.scene_digest) == 64


def test_observe_pre_phase_refuses_uncertain_physical_state():
    """A robot with unreconciled holding state never enters a pre-phase."""
    item = ledger_with_holding()
    item.invalidate_if_stale("piper_0", "brick-x", "motion lost mid-lift")
    facade = UnifiedDualArmRuntimeFacade(FakeSession(), state_ledger=item)
    with pytest.raises(RuntimeError, match="uncertain physical state"):
        asyncio.run(facade.observe_pre_phase(("piper_0", "piper_1")))


def test_observe_pre_phase_refuses_unauthorized_or_unbound_sessions():
    """Missing readiness authorization, ledger or freshness fails closed."""
    with pytest.raises(RuntimeError, match="authorization"):
        asyncio.run(
            UnifiedDualArmRuntimeFacade(
                FakeSession(readiness_passed=False), state_ledger=ledger()
            ).observe_pre_phase(("piper_0",))
        )
    with pytest.raises(RuntimeError, match="state ledger"):
        asyncio.run(
            UnifiedDualArmRuntimeFacade(FakeSession()).observe_pre_phase(("piper_0",))
        )
    with pytest.raises(ValueError, match="stale"):
        asyncio.run(
            UnifiedDualArmRuntimeFacade(
                FakeSession(sim_time_s=1.0, observed_sim_time_s=2.0),
                state_ledger=ledger(),
            ).observe_pre_phase(("piper_0",))
        )


def test_validate_pre_phase_binds_provenance_and_scene_version():
    """Validation keeps session/manifest/version binding of the request."""
    facade = UnifiedDualArmRuntimeFacade(
        FakeSession(version=1), state_ledger=ledger()
    )
    item = request()
    assert asyncio.run(facade.validate_pre_phase(item, success(item)))
    stale_version = UnifiedDualArmRuntimeFacade(
        FakeSession(version=2), state_ledger=ledger()
    )
    assert not asyncio.run(stale_version.validate_pre_phase(item, success(item)))
    other_session = UnifiedDualArmRuntimeFacade(
        FakeSession(version=1), state_ledger=ledger()
    )
    other_session._session._shared["scene_snapshot"]["session_id"] = "session-2"
    assert not asyncio.run(other_session.validate_pre_phase(item, success(item)))


def test_validate_pre_phase_rejects_unpassed_evidence():
    """Fail-closed evidence cannot validate a transition."""
    facade = UnifiedDualArmRuntimeFacade(FakeSession(), state_ledger=ledger())
    item = request()
    from rocobrick.unified_dual_arm_runtime import failed_pre_phase

    assert not asyncio.run(
        facade.validate_pre_phase(item, failed_pre_phase(item, "not implemented"))
    )


def test_execute_pre_phase_promotes_only_full_physical_evidence():
    """Adoption needs prepared+started+terminal-safe+measured arrival."""
    from rocobrick.coordinated_session import CoordinatedSessionResult

    ok = CoordinatedSessionResult(
        "transition-1",
        ROBOTS,
        {},
        {"joint_names": ["piper_0_joint1"]},
        {
            "prepared": True,
            "started": True,
            "terminal": True,
            "safe_to_release": True,
            "participants": {
                "piper_0": {"arrival_confirmed": True},
            },
        },
        True,
        participants=("piper_0",),
    )
    session = FakeSession()
    session.execution_result = ok
    facade = UnifiedDualArmRuntimeFacade(
        session, contexts={robot: _context() for robot in ROBOTS}, state_ledger=ledger()
    )
    evidence = asyncio.run(facade.execute_pre_phase(request()))
    assert evidence.passed
    assert evidence.adoption_ready
    assert evidence.arrival_confirmed
    assert session.last_coordination_id == "transition-1"

    no_arrival = CoordinatedSessionResult(
        "transition-1",
        ROBOTS,
        {},
        {"joint_names": ["piper_0_joint1"]},
        {
            "prepared": True,
            "started": True,
            "terminal": True,
            "safe_to_release": True,
            "participants": {"piper_0": {"arrival_confirmed": False}},
        },
        False,
        "arrival missed",
        participants=("piper_0",),
    )
    session.execution_result = no_arrival
    denied = asyncio.run(facade.execute_pre_phase(request()))
    assert not denied.passed
    assert not denied.adoption_ready


def test_executor_routes_the_scheduler_protocol_through_the_facade():
    """The formal executor delegates every pre-phase seam to the facade."""
    from rocobrick.execution.robot_operation_executor import SkillOperationExecutor

    session = FakeSession(version=1)
    item = ledger_with_holding()
    executor = SkillOperationExecutor(
        ROBOTS,
        SimpleNamespace(observe=lambda robot_id: None),
        pre_phase_runtime=UnifiedDualArmRuntimeFacade(
            session, contexts={"piper_0": _context()}, state_ledger=item
        ),
        state_ledger=item,
    )
    snapshot = asyncio.run(executor.observe_pre_phase(("piper_0", "piper_1")))
    assert isinstance(snapshot, SharedSceneSnapshot)
    evidence = asyncio.run(
        executor.execute_pre_phase(asyncio.run(_request_for(("piper_0",), item)))
    )
    assert not evidence.passed
    assert "scene-state schema" in (evidence.reason or "")
    assert executor.terminal_pre_phase_policy(()) is None


async def _request_for(participants, ledger_item):
    """Build one admitted request bound to a live observed snapshot."""
    from rocobrick.unified_dual_arm_runtime import SuccessorPrePhaseRequest

    from pre_phase_helpers import operation

    facade = UnifiedDualArmRuntimeFacade(FakeSession(), state_ledger=ledger_item)
    snapshot = await facade.observe_pre_phase(participants)
    previous = operation("previous-op", participants[0])
    return SuccessorPrePhaseRequest(
        "transition-live",
        (previous,),
        {"previous-op": {"terminal_safe": True}},
        operation("next-op", participants[0]),
        participants,
        ("robot:" + participants[0],),
        {"robot:" + participants[0]: "previous-op"},
        snapshot,
        None,
    )
