"""Offline checks for one production combined planner and strict plan-only IPC."""

from collections.abc import Mapping
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_coordinated_ros_runtime import Factory, pose, request, runtime
from test_dual_robot_ros_adapter import FakeClock, FakeNode, fake_factories
from test_persistent_ros_worker import ROBOTS, TOKEN, FakeHandle, read_message, submit

from rocobrick.coordinated_session import CoordinatedSessionResult
from rocobrick.execution.ros_session_worker import SessionWorker
from rocobrick.execution.ros_worker_transport import RosWorkerTransport
from rocobrick.robot_runtime_profile import (
    RobotRuntimeProfile,
    assert_profiles_disjoint,
)
from rocobrick.ros_moveit_runtime import RosMoveItRobotSession


def test_production_session_registers_exactly_one_combined_movegroup():
    """Controllers are isolated but no participant owns an independent planner."""
    node = FakeNode()
    session = RosMoveItRobotSession(
        node,
        ROBOTS,
        factories=fake_factories(),
        clock=FakeClock(),
        qos_state=object(),
        qos_sensor=object(),
    )
    session.register()
    actions = node.action_names()
    assert actions.count("/dual_arm/move_action") == 1
    assert "/piper_0/move_action" not in actions
    assert "/piper_1/move_action" not in actions
    for robot in ROBOTS:
        participant = session.runtime_for(robot)
        assert participant.services.compute_fk == "/dual_arm/compute_fk"
        assert participant.services.get_planning_scene == "/dual_arm/get_planning_scene"
        assert participant._move_group_client is None
    assert_profiles_disjoint(
        tuple(RobotRuntimeProfile.unified_participant(robot) for robot in ROBOTS)
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("position", (float("nan"), 0.0, 0.0)),
        ("orientation_xyzw", (0.0, 0.0, 0.0, 2.0)),
        ("orientation_xyzw", (0.0, 0.0, 0.0, float("inf"))),
    ],
)
def test_tcp_goal_rejects_nonfinite_and_unnormalized_pose(field, value):
    """Invalid geometric targets never become MoveGroup requests."""
    target = pose("piper_0", 0.1)
    target["tcp_pose"][field] = value
    with pytest.raises(ValueError):
        runtime()._build_goal(
            request({"piper_0": target, "piper_1": pose("piper_1", -0.1)}), Factory()
        )


def plan_result(item):
    """Return explicit no-motion combined evidence."""
    return CoordinatedSessionResult(
        item.coordination_id,
        item.robots,
        {},
        {},
        {"plan_only": True, "physical_execution": False, "safe_to_release": False},
        True,
    )


def test_combined_mailbox_reuses_existing_session_and_refuses_replay(tmp_path):
    """Plan-only requests do not construct an extra ROS child/runtime."""
    calls = []
    session = SimpleNamespace(
        unified_plan_only=lambda item: calls.append(item) or plan_result(item)
    )
    worker = SessionWorker(tmp_path, TOKEN, ROBOTS, session)
    item = request({robot: pose(robot, 0.1) for robot in ROBOTS})
    submit(worker, "1" * 32, item)
    worker.tick()
    result = read_message(tmp_path / "results" / ("1" * 32 + ".json"), TOKEN)["result"]
    assert result["passed"]
    assert not result["lifecycle"]["physical_execution"]
    assert len(calls) == 1
    submit(worker, "2" * 32, item)
    worker.tick()
    assert not read_message(tmp_path / "results" / ("2" * 32 + ".json"), TOKEN)[
        "result"
    ]["passed"]


def test_combined_worker_routes_physical_execution_and_requires_safe_result(tmp_path):
    """The non-plan-only envelope calls the session execution boundary."""
    from rocobrick.execution.ros_session_worker import SCHEMA, write_atomic

    item = request({robot: pose(robot, 0.1) for robot in ROBOTS})
    lifecycle = {
        "prepared": True,
        "started": True,
        "terminal": True,
        "safe_to_release": True,
        "physical_execution": True,
        "participants": {},
    }
    scripted = CoordinatedSessionResult(
        item.coordination_id,
        item.robots,
        {},
        {"joint_names": [], "points": []},
        lifecycle,
        True,
    )
    calls = []

    class Session:
        def execute_coordinated_pre_phase(self, request):
            calls.append(request)
            return scripted

    worker = SessionWorker(tmp_path, TOKEN, ROBOTS, Session())
    submit_document = {
        "schema": SCHEMA,
        "token": TOKEN,
        "request": {
            "schema": "rocobrick-coordinated-execute/v1",
            "request_id": "execute-1",
            "request": item.serialize(),
        },
    }
    write_atomic(tmp_path / "requests" / ("1" * 32 + ".json"), submit_document)
    worker.tick()
    result = read_message(tmp_path / "results" / ("1" * 32 + ".json"), TOKEN)[
        "result"
    ]
    assert result["passed"] is True
    assert result["result"]["lifecycle"]["physical_execution"] is True
    assert len(calls) == 1


def test_combined_worker_physical_execution_without_boundary_fails_closed(tmp_path):
    """No child execution boundary means no physical motion claim."""
    from rocobrick.execution.ros_session_worker import SCHEMA, write_atomic

    item = request({robot: pose(robot, 0.1) for robot in ROBOTS})
    worker = SessionWorker(tmp_path, TOKEN, ROBOTS, object())
    write_atomic(
        tmp_path / "requests" / ("1" * 32 + ".json"),
        {
            "schema": SCHEMA,
            "token": TOKEN,
            "request": {
                "schema": "rocobrick-coordinated-execute/v1",
                "request_id": "execute-2",
                "request": item.serialize(),
            },
        },
    )
    worker.tick()
    result = read_message(tmp_path / "results" / ("1" * 32 + ".json"), TOKEN)[
        "result"
    ]
    assert result["passed"] is False
    assert "execution boundary" in result["error"]


def test_transport_rejects_plan_only_result_that_claims_motion(tmp_path, monkeypatch):
    """A plan response cannot authorize arrival or lease release."""
    from rocobrick.execution.ros_session_worker import SCHEMA, write_atomic

    transport = RosWorkerTransport(tmp_path, TOKEN, FakeHandle())
    item = request({robot: pose(robot, 0.1) for robot in ROBOTS})

    def answer(path, timeout):
        write_atomic(
            path,
            {
                "schema": SCHEMA,
                "token": TOKEN,
                "result": replace(
                    plan_result(item),
                    lifecycle={"physical_execution": True, "safe_to_release": True},
                ).serialize(),
            },
        )

    monkeypatch.setattr(transport, "_wait", answer)
    with pytest.raises(ValueError, match="physical"):
        transport.call_plan_only(item, tmp_path / "answer.json", 1)


def test_shared_mailbox_observes_once_without_robot_snapshot_merge(tmp_path):
    """A single host query supplies both robot states."""
    from rocobrick.execution.ros_session_worker import SCHEMA, write_atomic

    calls = []
    shared = {"robots": {robot: {} for robot in ROBOTS}, "sequence": 7}
    session = SimpleNamespace(
        observe_all=lambda: calls.append("once") or shared,
        unified_planning_scene_version=0,
        latest_shared_sim_time_s=lambda: 0.0,
    )
    worker = SessionWorker(tmp_path, TOKEN, ROBOTS, session)
    write_atomic(
        tmp_path / "requests" / ("1" * 32 + ".json"),
        {
            "schema": SCHEMA,
            "token": TOKEN,
            "request": {
                "schema": "rocobrick-unified-observe/v1",
                "request_id": "snapshot-1",
                "robots": list(ROBOTS),
            },
        },
    )
    worker.tick()
    response = read_message(tmp_path / "results" / ("1" * 32 + ".json"), TOKEN)[
        "result"
    ]
    assert response["scene_snapshot"] == shared
    assert calls == ["once"]


def test_shared_observation_is_bound_to_scene_version_and_clock(tmp_path):
    """One observe answer carries the planning-scene version and shared clock."""
    from rocobrick.execution.ros_session_worker import SCHEMA, write_atomic

    shared = {"robots": {robot: {} for robot in ROBOTS}, "sequence": 9}
    session = SimpleNamespace(
        observe_all=lambda: shared,
        unified_planning_scene_version=4,
        latest_shared_sim_time_s=lambda: 12.5,
    )
    worker = SessionWorker(tmp_path, TOKEN, ROBOTS, session)
    write_atomic(
        tmp_path / "requests" / ("1" * 32 + ".json"),
        {
            "schema": SCHEMA,
            "token": TOKEN,
            "request": {
                "schema": "rocobrick-unified-observe/v1",
                "request_id": "observe-1",
                "robots": list(ROBOTS),
            },
        },
    )
    worker.tick()
    response = read_message(tmp_path / "results" / ("1" * 32 + ".json"), TOKEN)[
        "result"
    ]
    assert response["passed"] is True
    assert response["planning_scene_version"] == 4
    assert response["observed_sim_time_s"] == 12.5


def test_shared_observation_fails_closed_without_version_or_clock(tmp_path):
    """A session that cannot bind version/clock answers an explicit failure."""
    from rocobrick.execution.ros_session_worker import SCHEMA, write_atomic

    session = SimpleNamespace(observe_all=lambda: {"robots": {}})
    worker = SessionWorker(tmp_path, TOKEN, ROBOTS, session)
    write_atomic(
        tmp_path / "requests" / ("1" * 32 + ".json"),
        {
            "schema": SCHEMA,
            "token": TOKEN,
            "request": {
                "schema": "rocobrick-unified-observe/v1",
                "request_id": "observe-2",
                "robots": list(ROBOTS),
            },
        },
    )
    worker.tick()
    response = read_message(tmp_path / "results" / ("1" * 32 + ".json"), TOKEN)[
        "result"
    ]
    assert response["passed"] is False
    assert "planning-scene" in response["error"]


def test_transport_observe_shared_validates_the_bound_wrapper(tmp_path, monkeypatch):
    """The host transport rejects unbound or malformed observe answers."""
    from rocobrick.execution.ros_session_worker import SCHEMA, write_atomic

    transport = RosWorkerTransport(tmp_path, TOKEN, FakeHandle())
    shared = {"robots": {robot: {"robot_id": robot} for robot in ROBOTS}}

    def answer(path, timeout):
        write_atomic(
            path,
            {
                "schema": SCHEMA,
                "token": TOKEN,
                "result": {
                    "passed": True,
                    "request_id": path.stem,
                    "scene_snapshot": shared,
                    "planning_scene_version": 2,
                    "observed_sim_time_s": 3.25,
                },
            },
        )

    monkeypatch.setattr(transport, "_wait", answer)
    observed = transport.observe_shared(ROBOTS, 1)
    assert observed["planning_scene_version"] == 2
    assert observed["observed_sim_time_s"] == 3.25
    assert observed["scene_snapshot"] == shared
    assert transport.observe_all(ROBOTS, 1) == shared

    def unbound(path, timeout):
        write_atomic(
            path,
            {
                "schema": SCHEMA,
                "token": TOKEN,
                "result": {
                    "passed": True,
                    "request_id": path.stem,
                    "scene_snapshot": shared,
                    "planning_scene_version": None,
                    "observed_sim_time_s": 3.25,
                },
            },
        )

    monkeypatch.setattr(transport, "_wait", unbound)
    with pytest.raises(ValueError, match="planning-scene version"):
        transport.observe_shared(ROBOTS, 1)


def test_session_exposes_planning_scene_version_and_shared_clock():
    """The unified session binds observe answers to live planning state."""
    node = FakeNode()
    session = RosMoveItRobotSession(
        node,
        ROBOTS,
        factories=fake_factories(),
        clock=FakeClock(),
        qos_state=object(),
        qos_sensor=object(),
    )
    session.register()
    assert session.unified_planning_scene_version == 0
    with pytest.raises(RuntimeError, match="/clock"):
        session.latest_shared_sim_time_s()


def test_unified_readiness_uses_actual_age_skew_and_clock():
    """Stale participants cannot satisfy combined readiness via non-empty state."""
    import hashlib
    import json
    from test_pick_handler import document

    shared = document()
    shared["connections"] = {}
    shared["robots"]["piper_0"]["gripper"] = {
        "gripper_joint1": 0.01,
        "gripper_joint2": -0.01,
    }
    shared["robots"]["piper_1"] = {**shared["robots"]["piper_0"], "robot_id": "piper_1"}
    shared["scene_digest"] = hashlib.sha256(
        json.dumps(shared, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    item = replace(
        request({robot: pose(robot, 0.1) for robot in ROBOTS}),
        scene_snapshot=shared,
        scene_digest=shared["scene_digest"],
    )
    ready = SimpleNamespace(
        isaac_state_complete=True,
        joint_state_complete=True,
        broadcaster_match=True,
        clock_increasing=True,
        controllers_active=True,
        arm_action_ready=True,
    )
    participant = SimpleNamespace(
        readiness_snapshot=lambda: ready,
        state_timing=lambda: {
            "isaac_age_s": 0.01,
            "joint_age_s": 0.01,
            "joint_stamp_s": 3.0,
        },
        clock_values=lambda: (2_900_000_000, 3_100_000_000),
        _action_ready=lambda _: True,
        _gripper_client=object(),
    )
    session = RosMoveItRobotSession.__new__(RosMoveItRobotSession)
    session.robot_ids = ROBOTS
    session.runtimes = {robot: participant for robot in ROBOTS}
    session._coordinated_runtime = SimpleNamespace(
        register=lambda: None, ready=lambda: True
    )
    assert session.unified_readiness(item).passed
    participant.state_timing = lambda: {
        "isaac_age_s": 10.0,
        "joint_age_s": 10.0,
        "joint_stamp_s": 3.0,
    }
    assert not session.unified_readiness(item).passed


def test_maintained_participant_uses_explicit_joint_hold_goal():
    """The hold robot is fixed by measured joint goal, not a TCP path lock."""
    support = {
        **pose("piper_1", 0.1),
        "maintain_pose": True,
        "joint_positions": {"piper_1_joint1": -0.05},
    }
    goal = runtime()._build_goal(
        replace(
            request({
                "piper_0": {"joint_positions": {"piper_0_joint1": 0.1}},
                "piper_1": support,
            }),
            start_state={"piper_0_joint1": 0.0, "piper_1_joint1": -0.05},
        ),
        Factory(),
    )
    assert [
        item.joint_name for item in goal.request.goal_constraints[0].joint_constraints
    ] == ["piper_0_joint1", "piper_1_joint1"]
    assert not goal.request.path_constraints.position_constraints
    held = goal.request.path_constraints.joint_constraints
    assert len(held) == 1 and held[0].position == -0.05


def test_joint_goal_with_maintained_pose_uses_joint_hold():
    """A combined free-space goal pins the holding arm with its measured joint."""
    moving = {"joint_positions": {"piper_0_joint1": 0.1}}
    holding = {
        **pose("piper_1", 0.1),
        "maintain_pose": True,
        "joint_positions": {"piper_1_joint1": -0.05},
    }
    goal = runtime()._build_goal(
        replace(
            request({"piper_0": moving, "piper_1": holding}),
            start_state={"piper_0_joint1": 0.0, "piper_1_joint1": -0.05},
        ),
        Factory(),
    )
    joint_names = [
        constraint.joint_name
        for constraint in goal.request.goal_constraints[0].joint_constraints
    ]
    assert joint_names == ["piper_0_joint1", "piper_1_joint1"]
    assert not goal.request.goal_constraints[0].position_constraints
    assert not goal.request.path_constraints.position_constraints


def test_unified_joint_targets_composes_ik_and_measured_holds():
    """Movers get per-arm IK solutions; holding arms keep measured joints."""
    canonical = {
        f"joint{index}": 0.02 * index
        for index in range(1, 7)
    }
    canonical.update(
        {
            "gripper_joint1": 0.01,
            "gripper_joint2": -0.01,
        }
    )
    ik_answer = {f"piper_0_joint{i}": 0.11 for i in range(1, 7)}
    calls = []

    def fake_ik(robot_id, target, wire_seed, timeout_s, extra_seeds=()):
        calls.append((robot_id, dict(wire_seed)))
        return dict(ik_answer)

    mover = SimpleNamespace(
        arm_joint_names=tuple(f"piper_0_joint{i}" for i in range(1, 7)),
        latest_joint_state=lambda: dict(canonical),
        solve_tcp_ik=fake_ik,
    )
    holder = SimpleNamespace(
        arm_joint_names=tuple(f"piper_1_joint{i}" for i in range(1, 7)),
        latest_joint_state=lambda: dict(canonical),
        solve_tcp_ik=fake_ik,
    )
    session = RosMoveItRobotSession.__new__(RosMoveItRobotSession)
    session.robot_ids = ROBOTS
    session.runtimes = {"piper_0": mover, "piper_1": holder}
    session._joint_limits_cache = {
        **{f"piper_0_joint{i}": (-1.0, 1.0) for i in range(1, 7)},
        **{f"piper_1_joint{i}": (-1.0, 1.0) for i in range(1, 7)},
    }
    item = request(
        {
            "piper_0": pose("piper_0", 0.1),
            "piper_1": {**pose("piper_1", -0.1), "maintain_pose": True},
        }
    )
    solved = session.unified_joint_targets(item)
    assert solved["piper_0"] == ik_answer
    assert calls and set(calls[0][1]) == {
        f"piper_0_{name}" for name in canonical
    }
    assert solved["piper_1"] == {
        f"piper_1_joint{i}": canonical[f"joint{i}"] for i in range(1, 7)
    }
    holder.latest_joint_state = lambda: None
    with pytest.raises(RuntimeError, match="live joint state"):
        session.unified_joint_targets(item)


def test_coordinated_entry_fails_closed_without_ik_composition():
    """A pose goal without the per-arm composition boundary never dispatches."""
    from rocobrick.coordinated_session_entry import perform_coordinated_plan

    session = SimpleNamespace(robot_ids=ROBOTS)
    item = request({robot: pose(robot, 0.1) for robot in ROBOTS})
    result = perform_coordinated_plan(session, item)
    assert not result.passed
    assert "coordinated MoveIt runtime is unavailable" in (result.error or "")


def test_coordinated_entry_composes_joint_goals_end_to_end():
    """Pose targets become one admitted joint-space combined plan."""
    from rocobrick.coordinated_session_entry import perform_coordinated_plan

    arms = {
        robot: tuple(f"{robot}_joint{index}" for index in range(1, 7))
        for robot in ROBOTS
    }
    joint_names = [joint for robot in ROBOTS for joint in arms[robot]]
    planned = {
        "joint_names": tuple(joint_names),
        "points": (
            {"time_from_start_s": 0.0, "positions": tuple(0.0 for _ in joint_names)},
            {"time_from_start_s": 1.0, "positions": tuple(0.05 for _ in joint_names)},
        ),
        "status": "SUCCEEDED",
    }
    plan_calls = []
    composed = []

    def fake_ik(robot_id, target, wire_seed, timeout_s, extra_seeds=()):
        return {joint: 0.05 for joint in arms[robot_id]}

    session = SimpleNamespace(
        robot_ids=ROBOTS,
        runtime_for=lambda robot: SimpleNamespace(
            arm_joint_names=arms[robot],
            latest_joint_state=lambda: {
                f"joint{index}": 0.0 for index in range(1, 9)
            },
            solve_tcp_ik=fake_ik,
        ),
        unified_joint_targets=lambda item: (
            composed.append(dict(item.targets))
            or {robot: {joint: 0.05 for joint in arms[robot]} for robot in ROBOTS}
        ),
        unified_start_state=lambda: {
            joint: 0.0
            for robot in ROBOTS
            for joint in (*arms[robot], f"{robot}_gripper")
        },
        coordinated_readiness=lambda item: {
            "coordination_id": item.coordination_id,
            "participants": list(ROBOTS),
            "scene_digest": item.scene_digest,
            "scene_sequence": 1,
            "scene_sim_time_s": 1.0,
            "move_group_action": "/dual_arm/move_action",
            "planning_group": "dual_arm",
            "components": {"clock_increasing": True},
        },
        sync_unified_scene=lambda item: {
            "readback_confirmed": True,
            "snapshot_digest": item.scene_digest,
            "scene_version": 1,
        },
        validate_unified_scene=lambda proof: True,
        coordinated_plan_runtime=lambda: SimpleNamespace(
            plan=lambda item, timeout_s=None: plan_calls.append(1) or planned,
        ),
    )
    item = request({robot: pose(robot, 0.1) for robot in ROBOTS})
    item = replace(item, scene_snapshot={"sequence": 1, "sim_time_s": 1.0})
    item = replace(
        item,
        physical_facts={"piper_0": {}, "piper_1": {}},
    )
    result = perform_coordinated_plan(session, item)
    assert result.passed, result.error
    assert plan_calls == [1]
    for robot in ROBOTS:
        assert isinstance(composed[0][robot]["tcp_pose"], Mapping)
        assert "joint_positions" not in composed[0][robot]
    assert result.trajectory["joint_names"] == joint_names
    assert result.lifecycle["physical_execution"] is False
    assert result.trajectory["planning_scene"]["scene_version"] == 1
