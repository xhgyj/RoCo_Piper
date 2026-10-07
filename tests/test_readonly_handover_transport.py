"""One existing MoveGroup query without apply, action execution or ID conflict."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_persistent_ros_worker import TOKEN, FakeHandle, pump
from test_prepared_handover_execution import prepared_request
from test_prepared_worker_transport import worker_session

from rocobrick.adapters.moveit.handover_scene_planner import SessionCombinedPlanRuntime
from rocobrick.adapters.moveit.readonly_combined_plan import (
    perform_readonly_combined_plan,
)
from rocobrick.coordinated_session import CoordinatedSessionRequest
from rocobrick.execution.ros_session_worker import SessionWorker
from rocobrick.execution.ros_worker_transport import RosWorkerTransport
from rocobrick.ros_moveit_runtime import RosMoveItRobotSession


def readonly_request():
    """Build one complete twelve-joint plan-only query.

    Returns:
        The explicit captured scene/start and paired joint goals.
    """
    prepared = prepared_request()
    return CoordinatedSessionRequest(
        prepared.coordination_id,
        prepared.robots,
        prepared.operation.scene_digest,
        "dual_arm",
        True,
        10.0,
        {
            robot: {
                "joint_positions": {f"{robot}_joint{i}": 0.05 for i in range(1, 7)},
                "joint_goal_tolerance_rad": 0.002,
            }
            for robot in prepared.robots
        },
        scene_snapshot=prepared.scene_snapshot,
        scene_version=1,
        start_state=prepared.start_state,
        participants=prepared.participants,
    )


def planning_session(*, changed=False, wrong_start=False):
    """Provide only read/version/plan methods; scene writers are forbidden.

    Returns:
        A no-motion session double with one combined planning client.
    """
    versions, queries = [], []

    def version():
        versions.append(1)
        return 2 if changed and len(versions) > 1 else 1

    def plan(request, *, timeout_s):
        queries.append(request)
        return {
            "status": "SUCCEEDED",
            "joint_names": [
                f"{robot}_joint{i}"
                for robot in request.participants
                for i in range(1, 7)
            ],
            "points": [
                {
                    "time_from_start_s": 0.0,
                    "positions": [0.1 if wrong_start else 0.0] * 12,
                },
                {"time_from_start_s": 1.0, "positions": [0.05] * 12},
            ],
        }

    def forbidden(*args):
        pytest.fail("read-only planning must not apply scenes or execute goals")

    def readiness(request):
        return {
            "coordination_id": request.coordination_id,
            "participants": list(request.participants),
            "scene_digest": request.scene_digest,
            "scene_sequence": 1,
            "scene_sim_time_s": 1.1,
            "move_group_action": "/dual_arm/move_action",
            "planning_group": "dual_arm",
            "components": {
                "clock_increasing": True,
                "combined_joint_state_complete": True,
                "scene_snapshot_fresh": True,
                "scene_digest_match": True,
                "combined_move_group_ready": True,
                "controller:piper_0": True,
                "controller:piper_1": True,
            },
        }

    class PlanningSession(SimpleNamespace):
        """Match the production session's version property, not a method."""

        @property
        def unified_planning_scene_version(self):
            """Read the actual test version counter.

            Returns:
                The current read-only planning-scene version.
            """
            return version()

    return PlanningSession(
        robot_ids=("piper_0", "piper_1"),
        coordinated_readiness=readiness,
        coordinated_plan_runtime=lambda: SimpleNamespace(plan=plan),
        sync_unified_scene=forbidden,
        runtime_for=forbidden,
        queries=queries,
    )


def test_readonly_uses_one_combined_plan_without_scene_writes():
    """Actual query code has no reachable apply/controller path."""
    session, request = planning_session(), readonly_request()
    result = perform_readonly_combined_plan(session, request)
    assert result.passed is True
    assert session.queries == [request]
    assert result.lifecycle["physical_execution"] is False
    assert result.lifecycle["safe_to_release"] is False


def test_readonly_accepts_actual_production_version_property():
    """Exercise the real session descriptor, not a method-shaped fake."""
    actual = RosMoveItRobotSession.__new__(RosMoveItRobotSession)
    actual._scene_binding = SimpleNamespace(version=1, refresh_version=lambda _: 1)
    fake = planning_session()
    actual.robot_ids = fake.robot_ids
    actual.coordinated_readiness = fake.coordinated_readiness
    actual.coordinated_plan_runtime = fake.coordinated_plan_runtime
    result = perform_readonly_combined_plan(actual, readonly_request())
    assert result.passed is True
    assert len(fake.queries) == 1
    assert result.lifecycle["physical_execution"] is False


@pytest.mark.parametrize("version", [True, 1.0, "1", None])
def test_readonly_invalid_version_property_fails_before_query(version):
    """Truthy/coerced version values must not bypass integer provenance."""
    fake = planning_session()

    class InvalidSession(SimpleNamespace):
        """Return a deliberately invalid version without coercion."""

        unified_planning_scene_version = version

    invalid = InvalidSession(
        robot_ids=fake.robot_ids,
        coordinated_readiness=fake.coordinated_readiness,
        coordinated_plan_runtime=fake.coordinated_plan_runtime,
    )
    result = perform_readonly_combined_plan(invalid, readonly_request())
    assert result.passed is False
    assert fake.queries == []


@pytest.mark.parametrize(
    "change", ["scene", "start", "execution", "missing_arm", "wide_goal"]
)
def test_readonly_changed_or_incomplete_inputs_fail(change):
    """Scene/start/endpoint failures cannot become successful prepared paths."""
    session = planning_session(changed=change == "scene", wrong_start=change == "start")
    request = readonly_request()
    if change == "execution":
        request = replace(request, plan_only=False)
    elif change == "missing_arm":
        request.start_state.pop("piper_1_joint1")
    elif change == "wide_goal":
        request.targets["piper_0"]["joint_goal_tolerance_rad"] = 0.1
    result = perform_readonly_combined_plan(session, request)
    assert result.passed is False
    assert result.lifecycle["physical_execution"] is False
    if change not in ("scene", "start"):
        assert session.queries == []


def test_readonly_planning_identity_does_not_consume_execution_identity(tmp_path):
    """Planning the candidate and executing it are distinct owned lanes."""
    prepared, query = prepared_request(), readonly_request()
    execution_session = worker_session(prepared)
    plan_session = planning_session()
    execution_session.plan_readonly_combined = lambda req: (
        perform_readonly_combined_plan(plan_session, req)
    )
    worker = SessionWorker(
        tmp_path / "mailbox", TOKEN, prepared.robots, execution_session
    )
    transport = RosWorkerTransport(worker.directory, TOKEN, FakeHandle())
    transport._wait = lambda path, timeout: pump(worker, path.is_file)
    facade = SimpleNamespace(
        plan_readonly_combined=lambda req: transport.call_plan_only(
            req, tmp_path / "plan.json", 2.0, read_only=True
        )
    )
    raw = SessionCombinedPlanRuntime(facade).plan(query, timeout_s=query.timeout_s)
    assert raw["status"] == "SUCCEEDED"
    assert worker.seen_plans == {prepared.coordination_id}
    assert worker.seen_operations == set()
    executed = transport.execute_coordinated(prepared, tmp_path / "executed.json", 2.0)
    assert executed.passed is True
    pump(worker, lambda: not worker.coordinated_active)
    assert worker.owners == {}
