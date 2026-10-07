"""One existing MoveGroup query without apply, action execution or ID conflict."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_persistent_ros_worker import TOKEN, FakeHandle, pump
from test_prepared_handover_execution import prepared_request
from test_prepared_worker_transport import worker_session

from rocobrick.adapters.moveit.handover_scene_planner import (
    ReadOnlyPlanRejectedError,
    SessionCombinedPlanRuntime,
)
from rocobrick.adapters.moveit.readonly_combined_plan import (
    perform_readonly_combined_plan,
)
from rocobrick.coordinated_ros_runtime import CombinedMoveItPlanningError
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


def planning_session(*, changed=False, wrong_start=False, wrong_goal=False):
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
                {
                    "time_from_start_s": 1.0,
                    "positions": [0.1 if wrong_goal else 0.05] * 12,
                },
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


def readonly_pose_request():
    """Bind two calibrated TCP poses to the captured read-only query.

    Returns:
        A complete pose-only envelope, without a joint endpoint proposal.
    """
    joint = readonly_request()
    provenance = {"runtime_model_sha256": "a" * 64, "source_sha256": "b" * 64}
    return replace(
        joint,
        scene_snapshot={**joint.scene_snapshot, "tcp_provenance": provenance},
        targets={
            r: {
                "tcp_pose": {
                    "position": [0.0, 0.0, 0.25],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "tip_link": f"{r}_rocobrick_tcp",
                    "frame_id": f"{r}_base_link",
                    **provenance,
                    "position_tolerance_m": 0.001,
                    "orientation_tolerance_rad": 0.015,
                }
            }
            for r in joint.participants
        },
    )


def test_readonly_pose_query_admits_joint_endpoint_for_separate_native_fk():
    """Transport structure is admitted; endpoint FK still needs native proof."""
    session, query = planning_session(wrong_goal=True), readonly_pose_request()
    result = perform_readonly_combined_plan(session, query)
    assert result.passed and session.queries == [query]
    assert result.trajectory["points"][-1]["positions"] == [0.1] * 12
    assert result.lifecycle["goal_mode"] == "tcp_pose"
    assert result.lifecycle["native_fk_validation_required"] is True
    assert result.lifecycle["safe_to_release"] is False
    assert result.lifecycle["physical_execution"] is False


@pytest.mark.parametrize(
    "fault",
    [
        "frame",
        "tip",
        "hash",
        "bool",
        "quaternion",
        "wide-position",
        "wide-orientation",
        "mixed",
    ],
)
def test_readonly_pose_query_rejects_bad_calibration_or_relaxed_constraints(fault):
    """Both complete, calibrated, bounded pose goals are mandatory."""
    session, query = planning_session(), readonly_pose_request()
    target = query.targets["piper_0"]["tcp_pose"]
    if fault == "mixed":
        query.targets["piper_0"] = readonly_request().targets["piper_0"]
    elif fault == "frame":
        target["frame_id"] = "world"
    elif fault == "tip":
        target["tip_link"] = "piper_1_rocobrick_tcp"
    elif fault == "hash":
        target["source_sha256"] = "c" * 64
    elif fault == "bool":
        target["position"][0] = True
    elif fault == "quaternion":
        target["orientation_xyzw"][-1] = 0.5
    elif fault == "wide-position":
        target["position_tolerance_m"] = 0.01
    else:
        target["orientation_tolerance_rad"] = 0.1
    result = perform_readonly_combined_plan(session, query)
    assert not result.passed
    assert session.queries == []


def test_readonly_pose_query_preserves_bad_start_for_diagnostics():
    """Switching goal mode must not weaken captured-start admission."""
    session = planning_session(wrong_start=True)
    result = perform_readonly_combined_plan(session, readonly_pose_request())
    assert not result.passed
    assert "start_deviations" in result.error
    assert len(result.trajectory["joint_names"]) == 12
    assert result.trajectory["points"][0]["positions"] == [0.1] * 12


@pytest.mark.parametrize("code", [99999, -1, -27, -31])
def test_moveit_failure_code_survives_readonly_lane_without_false_internal_error(code):
    """Generic FAILURE, invalid goals and failed IK remain candidate rejections."""
    query, session = readonly_pose_request(), planning_session()

    def fail(query, *, timeout_s):
        raise CombinedMoveItPlanningError("ABORTED", code)

    session.coordinated_plan_runtime = lambda: SimpleNamespace(plan=fail)
    result = perform_readonly_combined_plan(session, query)
    assert not result.passed
    assert result.lifecycle["moveit_error_code"] == code
    assert result.lifecycle["physical_execution"] is False
    facade = SimpleNamespace(plan_readonly_combined=lambda _: result)
    with pytest.raises(ReadOnlyPlanRejectedError):
        SessionCombinedPlanRuntime(facade).plan(query, timeout_s=query.timeout_s)


def test_generated_ros_assertion_retains_readonly_identity_and_original_error():
    """ROS scalar type assertions must not become identity/capability failures."""
    query = readonly_request()
    session = planning_session()

    def fail(request, *, timeout_s):
        raise AssertionError("The 'x' field must be of type 'float'")

    session.coordinated_plan_runtime = lambda: SimpleNamespace(plan=fail)
    result = perform_readonly_combined_plan(session, query)
    assert not result.passed
    assert result.error == "The 'x' field must be of type 'float'"
    assert result.coordination_id == query.coordination_id
    assert result.participants == query.participants
    assert result.lifecycle["read_only"] is True
    assert result.lifecycle["physical_execution"] is False
    assert result.lifecycle["terminal"] is True
    assert result.lifecycle["safe_to_release"] is False

    transport = SessionCombinedPlanRuntime(
        SimpleNamespace(plan_readonly_combined=lambda _: result)
    )
    with pytest.raises(ReadOnlyPlanRejectedError, match="'x'.*float"):
        transport.plan(query, timeout_s=query.timeout_s)


@pytest.mark.parametrize("remove_tag", [False, True])
def test_pose_session_transport_requires_explicit_native_fk_boundary(remove_tag):
    """An old/unqualified worker reply cannot silently admit TCP pose goals."""
    query, session = readonly_pose_request(), planning_session()
    result = perform_readonly_combined_plan(session, query)
    if remove_tag:
        result = replace(
            result,
            lifecycle={
                k: v
                for k, v in result.lifecycle.items()
                if k != "native_fk_validation_required"
            },
        )
    facade = SimpleNamespace(plan_readonly_combined=lambda _: result)
    runtime = SessionCombinedPlanRuntime(facade)
    if remove_tag:
        with pytest.raises(ValueError, match="native FK admission boundary"):
            runtime.plan(query, timeout_s=query.timeout_s)
    else:
        assert runtime.plan(query, timeout_s=query.timeout_s)["status"] == "SUCCEEDED"


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


def test_readonly_goal_mismatch_reports_per_joint_deviations_and_raw_plan():
    """A clamped endpoint rejection names joints and keeps the raw trajectory."""
    session = planning_session(wrong_goal=True)
    result = perform_readonly_combined_plan(session, readonly_request())
    assert result.passed is False
    error = result.error or ""
    assert "read-only path does not match captured start/paired goal" in error
    assert "goal_deviations" in error
    assert "piper_0_joint1" in error
    assert result.trajectory["joint_names"][0] == "piper_0_joint1"
    assert result.trajectory["points"][-1]["positions"] == pytest.approx([0.1] * 12)
    assert result.trajectory["points"][0]["time_from_start_s"] == 0.0


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
