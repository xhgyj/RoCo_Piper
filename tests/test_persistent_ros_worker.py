"""Offline persistent worker identity, ownership and terminal evidence tests."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest
from test_dual_robot_ros_adapter import joint_operation_for, session_request

from rocobrick.execution import ros_session_worker as worker_module
from rocobrick.execution.ros_session_worker import (
    SCHEMA,
    SessionWorker,
    read_message,
    terminal_safe,
    write_atomic,
)
from rocobrick.execution.ros_worker_transport import RosWorkerTransport
from rocobrick.ros_session_entry import RosSessionResult

ROBOTS = ("piper_0", "piper_1")
TOKEN = "test-token"


def result_for(request, *, safe=True, passed=True):
    """Build a strict session result whose terminal flags are controllable.

    Returns:
        A ``RosSessionResult`` answering exactly this request.
    """
    records = tuple(
        {
            "robot_id": item.robot_id,
            "operation_id": item.operation.request_id,
            "kind": item.operation.kind.value,
            "dispatched": True,
            "motion_sent": True,
            "result": {
                "schema": "rocobrick-runtime-operation-result/v1",
                "request_id": item.operation.request_id,
                "kind": item.operation.kind.value,
                "passed": passed,
                "error": None,
                "execution": {"terminal": safe, "safe_to_release": safe},
                "evidence": {},
            },
        }
        for item in request.operations
    )
    return RosSessionResult(request.request_id, ROBOTS, {}, True, records, passed, None)


class FakeHandle:
    """Inject worker lifetime independently of motion results."""

    def __init__(self):
        """Start alive with no exit code and no delivered signals."""
        self.exit_code = None
        self.signals = []

    def poll(self):
        """Return the recorded exit code while the worker is alive."""
        return self.exit_code

    def wait(self, *, timeout_s):
        """Return the recorded exit code immediately."""
        return self.exit_code

    def send_signal(self, signum):
        """Record the signal and mark the worker killed by it."""
        self.signals.append(signum)
        self.exit_code = -signum


def pump(worker, predicate):
    """Drive worker ticks until the predicate holds or the bound expires."""
    deadline = time.monotonic() + 2
    while not predicate():
        worker.tick()
        if time.monotonic() > deadline:
            raise AssertionError("worker did not settle")
        time.sleep(0.005)


def submit(worker, key, request):
    """Publish one mailbox request under a fixed hexadecimal key."""
    write_atomic(
        worker.directory / "requests" / f"{key}.json",
        {
            "schema": SCHEMA,
            "token": TOKEN,
            "request": request.serialize(),
        },
    )


def test_worker_reuses_session_and_runs_distinct_robots_concurrently(
    tmp_path, monkeypatch
):
    """One persistent session serves two robots in parallel lanes."""
    entered = {robot: threading.Event() for robot in ROBOTS}
    release = threading.Event()
    sessions = []

    def execute(session, request, **kwargs):
        sessions.append(session)
        entered[request.operations[0].robot_id].set()
        release.wait(2)
        return result_for(request)

    monkeypatch.setattr(worker_module, "perform_session", execute)
    session = object()
    worker = SessionWorker(tmp_path, TOKEN, ROBOTS, session)
    for index, robot in enumerate(ROBOTS):
        submit(
            worker,
            f"{index + 1:032x}",
            session_request(
                joint_operation_for(robot, f"op-{index}"),
                request_id=f"req-{index}",
            ),
        )
    pump(worker, lambda: all(event.is_set() for event in entered.values()))
    assert len(worker.active) == 2
    release.set()
    pump(worker, lambda: not worker.active)
    assert sessions == [session, session]
    assert worker.owners == {}


def test_same_robot_conflict_and_replay_are_refused(tmp_path, monkeypatch):
    """A robot owning an active operation refuses overlap and replay."""
    release = threading.Event()

    def execute(session, request, **kwargs):
        release.wait(2)
        return result_for(request)

    monkeypatch.setattr(worker_module, "perform_session", execute)
    worker = SessionWorker(tmp_path, TOKEN, ROBOTS, object())
    request = session_request(joint_operation_for(ROBOTS[0], "op-1"))
    submit(worker, "1" * 32, request)
    worker.tick()
    submit(worker, "2" * 32, session_request(joint_operation_for(ROBOTS[0], "op-2")))
    worker.tick()
    refused = read_message(tmp_path / "results" / ("2" * 32 + ".json"), TOKEN)
    assert refused["result"]["operations"][0]["dispatched"] is False
    release.set()
    pump(worker, lambda: not worker.active)
    submit(worker, "3" * 32, request)
    worker.tick()
    assert (
        read_message(tmp_path / "results" / ("3" * 32 + ".json"), TOKEN)["result"][
            "passed"
        ]
        is False
    )


def test_cancellation_targets_only_owned_robot(tmp_path, monkeypatch):
    """A cancel message only touches the runtime of the owning robot."""
    releases = {robot: threading.Event() for robot in ROBOTS}
    calls = []

    class Runtime:
        def __init__(self, robot):
            """Bind this fake runtime to one robot identity."""
            self.robot = robot

        def cancel_active(self):
            """Record which robot's action channel was cancelled."""
            calls.append(self.robot)
            releases[self.robot].set()

    session = SimpleNamespace(runtime_for=lambda robot: Runtime(robot))

    def execute(session, request, **kwargs):
        releases[request.operations[0].robot_id].wait(2)
        return result_for(request)

    monkeypatch.setattr(worker_module, "perform_session", execute)
    worker = SessionWorker(tmp_path, TOKEN, ROBOTS, session)
    for index, robot in enumerate(ROBOTS):
        submit(
            worker,
            str(index + 1) * 32,
            session_request(
                joint_operation_for(robot, f"op-{index}"),
                request_id=f"req-{index}",
            ),
        )
    worker.tick()
    write_atomic(
        tmp_path / "cancels" / ("1" * 32 + ".json"),
        {
            "schema": SCHEMA,
            "token": TOKEN,
            "request_id": "req-0",
        },
    )
    pump(worker, lambda: "1" * 32 not in worker.active)
    assert calls and set(calls) == {ROBOTS[0]}
    assert "2" * 32 in worker.active
    releases[ROBOTS[1]].set()
    pump(worker, lambda: not worker.active)


def test_unsafe_terminal_retains_robot_and_blocks_new_requests(tmp_path, monkeypatch):
    """Non-terminal evidence keeps ownership and fails the orderly stop."""
    monkeypatch.setattr(
        worker_module,
        "perform_session",
        lambda session, request, **kwargs: result_for(request, safe=False),
    )
    worker = SessionWorker(tmp_path, TOKEN, ROBOTS, object())
    request = session_request(joint_operation_for(ROBOTS[0], "unsafe"))
    submit(worker, "1" * 32, request)
    worker.tick()
    pump(worker, lambda: not worker.active)
    assert worker.unsafe
    assert ROBOTS[0] in worker.owners
    worker.stopping.set()
    assert worker.run() is False


@pytest.mark.parametrize("change", ["token", "schema", "identity"])
def test_transport_rejects_foreign_or_stale_answer(tmp_path, change):
    """Token, schema and identity tampering all fail closed."""
    request = session_request(joint_operation_for(ROBOTS[0], "op"))
    result = result_for(request).serialize()
    document = {"schema": SCHEMA, "token": TOKEN, "result": result}
    if change == "identity":
        result["request_id"] = "foreign"
    else:
        document[change] = "foreign"
    path = tmp_path / "response.json"
    write_atomic(path, document)
    transport = RosWorkerTransport(tmp_path, TOKEN, FakeHandle())
    with pytest.raises(ValueError):
        transport._result(path, request)


def test_worker_death_is_not_terminal_evidence(tmp_path):
    """A dead worker is a transport failure, never safe terminal evidence."""
    handle = FakeHandle()
    handle.exit_code = 1
    transport = RosWorkerTransport(tmp_path, TOKEN, handle)
    with pytest.raises(RuntimeError, match="exited"):
        transport.wait_ready(ROBOTS, 0.1)
    request = session_request(joint_operation_for(ROBOTS[0], "op"))
    assert terminal_safe(result_for(request, safe=False)) is False


def test_transport_reuses_worker_and_requires_shutdown_attestation(
    tmp_path, monkeypatch
):
    """One worker serves many calls; orderly stop needs the attestation."""
    handle = FakeHandle()
    transport = RosWorkerTransport(tmp_path, TOKEN, handle)
    keys = []
    original = transport._wait

    def respond(path, timeout_s):
        if path.parent.name == "results":
            key = path.stem
            keys.append(key)
            message = read_message(tmp_path / "requests" / f"{key}.json", TOKEN)
            request = worker_module.RosSessionRequest.parse(message["request"])
            write_atomic(
                path,
                {
                    "schema": SCHEMA,
                    "token": TOKEN,
                    "result": result_for(request).serialize(),
                },
            )
        return original(path, timeout_s)

    monkeypatch.setattr(transport, "_wait", respond)
    for index in range(2):
        request = session_request(
            joint_operation_for(ROBOTS[0], f"op-{index}"), request_id=f"req-{index}"
        )
        assert transport.call(request, tmp_path / f"result-{index}.json", 1).passed
    assert len(set(keys)) == 2
    handle.exit_code = 0
    assert transport.stop(ROBOTS)["passed"] is False
    write_atomic(
        tmp_path / "shutdown.json",
        {
            "schema": SCHEMA,
            "token": TOKEN,
            "robots": list(ROBOTS),
            "orderly": True,
            "active_requests": [],
        },
    )
    assert transport.stop(ROBOTS)["passed"] is True


def test_production_default_uses_persistent_transport(tmp_path):
    """Production sessions default to the persistent ROS client."""
    from test_multi_robot_live_session import config_kwargs

    from rocobrick.runtime_session import MultiRobotSessionConfig

    kwargs = config_kwargs(tmp_path)
    kwargs.pop("persistent_ros_client")
    assert MultiRobotSessionConfig(**kwargs).persistent_ros_client is True
