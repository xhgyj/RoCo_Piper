"""Prepare both arms first; final contact permission comes from actual arrival."""

import asyncio
import copy
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
from test_coordinated_handover_lifecycle import Transport, operation, program
from test_handover_scene_planner import native_provider
from test_handover_simulation import authorization
from test_readonly_handover_transport import planning_session, readonly_request

from rocobrick.adapters.moveit.handover_geometry import IsolatedHandoverGeometry
from rocobrick.adapters.moveit.handover_program import NativeCoordinatedHandoverFactory
from rocobrick.adapters.moveit.handover_scene_planner import (
    NativeHandoverScenePlanner,
    ReadOnlyPlanRejectedError,
    SessionCombinedPlanRuntime,
)
from rocobrick.adapters.moveit.readonly_combined_plan import (
    perform_readonly_combined_plan,
)
from rocobrick.execution.operation_state_ledger import OperationStateLedger
from rocobrick.skills.coordinated_handover import (
    CoordinatedHandoverBackend,
    CoordinatedHandoverPreparation,
)
from rocobrick.skills.handover_profile import EXECUTION_CAPABILITIES
from rocobrick.skills.placement_geometry import compose, inverse


def staged(p):
    """Convert a synthetic complete fixture into preparation-only permission.

    Returns:
        A synthetic preparation, never hardware qualification.
    """
    return CoordinatedHandoverPreparation(
        p.request,
        p.rendezvous,
        p.paths.receiver_open,
        p.capabilities,
        p.donor_holding_window,
        p.phase_timeout_s,
        authorization(p.request),
    )


def test_preparation_requires_explicit_simulation_authorization():
    """No staged program silently qualifies hardware."""
    p = program()
    with pytest.raises(ValueError, match="simulation authorization"):
        replace(staged(p), simulation_authorization=None)
    with pytest.raises(ValueError, match="simulation only"):
        NativeCoordinatedHandoverFactory(
            lambda _: p.request,
            lambda _: None,
            lambda _: (),
            execution_capabilities=p.capabilities,
            phase_timeout_s=2,
            staged_preparation=True,
        )


@pytest.mark.parametrize(
    "defect", ["start", "grid", "participant", "opening", "mutation"]
)
def test_preparation_keeps_start_identity_time_grid_and_mutation_checks(defect):
    """Removing final-contact gates never removes carrying-motion contracts."""
    p = staged(program())
    if defect == "mutation":
        p.request.start_state[p.request.donor.robot_id + "_joint1"] = 0.1
        assert p.current_binding() != p.binding_sha256
        return
    if defect == "opening":
        last = dict(p.receiver_open[-1])
        last[p.request.donor.robot_id + "_joint1"] = 0.1
        with pytest.raises(ValueError):
            replace(p, receiver_open=(p.receiver_open[0], last))
        return
    motion = p.rendezvous
    first = motion.trajectories[0]
    if defect == "start":
        first = replace(
            first,
            points=(replace(first.points[0], positions=(0.1,) * 6), first.points[-1]),
        )
    elif defect == "grid":
        first = replace(
            first,
            points=(first.points[0], replace(first.points[-1], time_from_start_s=1.5)),
        )
    else:
        first = motion.trajectories[1]
    with pytest.raises(ValueError):
        replace(
            p, rendezvous=replace(motion, trajectories=(first, motion.trajectories[1]))
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("continuation_fails", [True, False])
async def test_final_contact_is_compiled_only_after_paired_arrival(continuation_fails):
    """A future handler rejection must not suppress safe preparation motion."""
    p = program()
    preparation = staged(p)
    ledger = OperationStateLedger(p.request.session_robots)
    ledger.confirm(
        p.identity.donor_robot_id, "holding:brick-1", {"actual_holding": True}
    )
    transport = Transport(ledger)

    class Factory:
        async def prepare(self, op):
            return preparation

        async def complete_preparation(self, op, arrived):
            assert transport.calls == ["receiver_open", "rendezvous"]
            assert arrived is preparation
            if continuation_fails:
                raise ValueError("HANDOVER_AFTER_PREPARATION: injected local budget")
            return replace(
                p,
                rendezvous=None,
                simulation_authorization=preparation.simulation_authorization,
                preparation_binding_sha256=preparation.binding_sha256,
            )

    result = await CoordinatedHandoverBackend(Factory(), transport, ledger).execute(
        operation(p)
    )
    if continuation_fails:
        assert not result.passed and result.motion_sent
        assert result.evidence["preparation_arrived"] is True
        assert transport.calls == ["receiver_open", "rendezvous"]
        assert "HANDOVER_AFTER_PREPARATION" in result.reason
        assert ledger.owner_of_object("brick-1") == p.identity.donor_robot_id
        assert transport.cancelled
        assert result.evidence["safe_to_release"] is False
    else:
        assert result.passed, result.reason
        assert transport.calls.count("rendezvous") == 1
        assert transport.calls.count("receiver_open") == 1
        assert ledger.owner_of_object("brick-1") == p.identity.receiver_robot_id


def test_native_preparation_does_not_call_any_final_contact_gate(tmp_path, monkeypatch):
    """Real native swept geometry; synthetic MoveGroup only, no motion."""
    provider, request, _, _ = native_provider(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("contact feasibility was incorrectly required before preparation")

    monkeypatch.setattr(provider, "_prepare_paths", forbidden)
    monkeypatch.setattr(provider, "_receiver_alignment", forbidden)
    result = provider.plan_preparation(request, deadline_s=20)
    assert result is not None, provider.last_path_failure
    motion, opening = result
    assert motion.participants == (request.donor.robot_id, request.receiver.robot_id)
    assert opening[0] == request.start_state
    assert not provider._paths and not provider._preparation_endpoints


def test_native_preparation_still_rejects_interior_collision(tmp_path):
    """Safe endpoint alone never admits an unsafe carrying path."""
    provider, request, _, transport = native_provider(tmp_path)
    transport.colliding_midpoint = True
    assert provider.plan_preparation(request, deadline_s=20) is None
    assert provider.last_path_failure["phase"] == "preparation_carrying_path"


def test_staged_factory_returns_preparation_without_compiling_contact(
    tmp_path, monkeypatch
):
    """Public factory mode separates preparation from complete contact paths."""
    provider, request, _, _ = native_provider(tmp_path)
    token = authorization(request)
    factory = NativeCoordinatedHandoverFactory(
        lambda _: request,
        lambda _: provider,
        lambda _: ((0.015, 0.017), (0.015, 0.017), (0.075, 0.085)),
        execution_capabilities=EXECUTION_CAPABILITIES,
        phase_timeout_s=60,
        simulation_authorization=token,
        staged_preparation=True,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("contact compiler reached during preparation")

    monkeypatch.setattr(provider, "prepare_contact_from_arrival", forbidden)
    result = asyncio.run(factory.prepare(operation(program())))
    assert isinstance(result, CoordinatedHandoverPreparation)
    assert result.current_binding() == result.binding_sha256


def test_timeout_preserves_readonly_marker_and_original_failure():
    """No generic worker identity/capability error masks a plan timeout."""
    session, query = planning_session(), readonly_request()

    def timeout(*args, **kwargs):
        raise TimeoutError("combined MoveGroup plan timed out")

    session.coordinated_plan_runtime = lambda: SimpleNamespace(plan=timeout)
    result = perform_readonly_combined_plan(session, query)
    assert not result.passed
    assert result.lifecycle["read_only"] is True
    assert result.lifecycle["physical_execution"] is False
    assert result.lifecycle["terminal"] is False
    assert "timed out" in result.error
    runtime = SessionCombinedPlanRuntime(
        SimpleNamespace(plan_readonly_combined=lambda _: result)
    )
    with pytest.raises(ReadOnlyPlanRejectedError, match="timed out"):
        runtime.plan(query, timeout_s=query.timeout_s)


def test_native_arrival_continuation_uses_measured_joints_without_movegroup(tmp_path):
    """Actual native contact compilation binds arrival, not nominal preparation."""
    provider, request, candidate, runtime = native_provider(tmp_path)
    plan, opening = provider.plan_preparation(request, deadline_s=20)
    state = dict(opening[-1])
    for trajectory in plan.trajectories:
        state.update(zip(trajectory.joint_names, trajectory.points[-1].positions))
    snapshot = copy.deepcopy(request.scene_snapshot)
    snapshot["scene_digest"] = "a" * 64
    original = request.observation(request.donor.robot_id)
    snapshot["bricks"][request.physical_instance_id]["pose"] = asdict(
        compose(
            provider.geometry.tcp_pose(request.donor.robot_id, state),
            compose(inverse(original.tcp_pose), original.object_pose),
        )
    )
    for robot in (request.donor.robot_id, request.receiver.robot_id):
        snapshot["robots"][robot]["tcp_pose"] = asdict(
            provider.geometry.tcp_pose(robot, state)
        )
    arrived = replace(request, start_state=state, scene_snapshot=snapshot)
    geometry = IsolatedHandoverGeometry(provider.geometry.files, arrived)
    base_fingers = provider.grippers(candidate)
    contact_provider = NativeHandoverScenePlanner(
        geometry,
        provider.planner,
        grippers=lambda c: replace(base_fingers, candidate_id=c.candidate_id),
        read_binding=lambda: arrived.binding_sha256,
        clock=lambda: 0.0,
    )
    factory = NativeCoordinatedHandoverFactory(
        lambda _: arrived,
        lambda _: contact_provider,
        lambda _: ((0.015, 0.017), (0.015, 0.017), (0.075, 0.085)),
        execution_capabilities=EXECUTION_CAPABILITIES,
        phase_timeout_s=60,
        simulation_authorization=authorization(request),
        staged_preparation=True,
    )
    preparation = CoordinatedHandoverPreparation(
        request,
        plan,
        opening,
        EXECUTION_CAPABILITIES,
        (0.015, 0.017),
        60,
        authorization(request),
    )
    count = len(runtime.requests)
    completed = asyncio.run(
        factory.complete_preparation(operation(program()), preparation)
    )
    assert len(runtime.requests) == count
    assert completed.request is arrived
    assert completed.rendezvous is None
    assert completed.preparation_binding_sha256 == preparation.binding_sha256
    for robot in completed.participants:
        for i in range(1, 7):
            name = f"{robot}_joint{i}"
            assert completed.paths.approach[0][name] == state[name]
