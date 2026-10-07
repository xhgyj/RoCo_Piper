"""Prepared trajectories reuse the owned worker without replanning/replay."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from test_persistent_ros_worker import TOKEN, FakeHandle, pump
from test_prepared_handover_execution import (
    live_scene,
    prepared_request,
    strict_fake_runtime,
)

from rocobrick.adapters.moveit.prepared_execution import perform_prepared_motion
from rocobrick.execution.prepared_motion import PREPARED_MOTION_SCHEMA
from rocobrick.execution.ros_session_worker import (
    SCHEMA,
    SessionWorker,
    read_message,
    write_atomic,
)
from rocobrick.execution.ros_worker_transport import RosWorkerTransport


def worker_session(request):
    """Provide actual admission/participant logic behind an offline worker.

    Returns:
        A scripted session with both owned controller feedback channels.
    """
    runtimes = {robot: strict_fake_runtime(robot) for robot in request.robots}
    session = SimpleNamespace(
        robot_ids=request.robots,
        observe_all=lambda: live_scene(request),
        unified_planning_scene_version=lambda: 1,
        unified_start_state=lambda: request.start_state,
        latest_shared_sim_time_s=lambda: 2.01,
        runtime_for=runtimes.__getitem__,
    )
    session.execute_prepared_motion = lambda req, **kw: perform_prepared_motion(
        session, req, **kw
    )
    session.execute_coordinated_pre_phase = lambda req: pytest.fail("must not replan")
    session.unified_plan_only = lambda req: pytest.fail("must not call MoveGroup")
    return session


def test_prepared_transport_runs_existing_worker_without_replan(tmp_path):
    """Round-trip typed path, dual ownership and physical feedback evidence."""
    request = prepared_request()
    worker = SessionWorker(
        tmp_path / "mailbox", TOKEN, request.robots, worker_session(request)
    )
    transport = RosWorkerTransport(worker.directory, TOKEN, FakeHandle())

    def wait(path, timeout):
        pump(worker, path.is_file)

    transport._wait = wait
    result = transport.execute_coordinated(request, tmp_path / "result.json", 2.0)
    pump(worker, lambda: not worker.coordinated_active)
    assert result.passed is True
    assert result.lifecycle["replanned"] is False
    assert result.lifecycle["prepared_binding_sha256"] == request.binding_sha256
    assert worker.owners == {}
    assert len(worker.seen_operations) == 1
    assert all(
        len(worker.session.runtime_for(robot).sent_goals) == 1
        for robot in request.robots
    )
    assert transport.active_coordinated == {}


def publish_prepared(worker, key, raw):
    """Publish an exact-token prepared execution envelope for rejection tests."""
    write_atomic(
        worker.directory / "requests" / f"{key}.json",
        {
            "schema": SCHEMA,
            "token": TOKEN,
            "request": {
                "schema": PREPARED_MOTION_SCHEMA,
                "request_id": key,
                "request": raw,
            },
        },
    )


def test_prepared_worker_replay_never_sends_again(tmp_path):
    """A safe first execution does not authorize reuse of its consumed identity."""
    request = prepared_request()
    worker = SessionWorker(tmp_path, TOKEN, request.robots, worker_session(request))
    publish_prepared(worker, "1" * 32, request.serialize())
    pump(worker, lambda: (tmp_path / "results" / ("1" * 32 + ".json")).is_file())
    pump(worker, lambda: not worker.coordinated_active)
    publish_prepared(worker, "2" * 32, request.serialize())
    worker.tick()
    result = read_message(tmp_path / "results" / ("2" * 32 + ".json"), TOKEN)["result"]
    assert result["passed"] is False
    assert "replay" in result["error"]
    assert all(
        len(worker.session.runtime_for(robot).sent_goals) == 1
        for robot in request.robots
    )


@pytest.mark.parametrize("change", ["hash", "robot", "missing_start"])
def test_prepared_worker_bad_binding_sends_neither_arm(tmp_path, change):
    """Malformed prepared input is rejected before any action ownership lane."""
    request = prepared_request()
    session = worker_session(request)
    worker = SessionWorker(tmp_path, TOKEN, request.robots, session)
    raw = request.serialize()
    if change == "hash":
        raw["binding_sha256"] = "f" * 64
    elif change == "robot":
        raw["robots"] = ["piper_0"]
    else:
        raw["start_state"].pop("piper_1_joint1")
    publish_prepared(worker, "1" * 32, raw)
    worker.tick()
    result = read_message(tmp_path / "results" / ("1" * 32 + ".json"), TOKEN)["result"]
    assert result["passed"] is False
    assert not worker.coordinated_active
    assert all(not session.runtime_for(robot).sent_goals for robot in request.robots)
