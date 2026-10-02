"""Offline tests for SessionPhaseBackend over a scripted fake session.

No ROS, no Isaac, no Docker: the session is a fake with the exact
``run_operation(request_path, result_path) -> Mapping`` seam of the real
:class:`rocobrick.runtime_session.IsaacMoveItSession`, scripted per request
kind. Every documented phase -> operation mapping and failure reason of the
backend is pinned here.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from rocobrick.isaac_backend import OPERATION_RESULT_SCHEMA
from rocobrick.models import ActionType, FailureReason, SkillType
from rocobrick.planning_scene import (
    PlanningSceneUpdate,
    box_from_aabb,
    scene_digest,
)
from rocobrick.runtime_session import RuntimeSessionError
from rocobrick.session_phase_backend import SessionPhaseBackend, SkillStepSpec
from rocobrick.skill_execution import execute_skill
from rocobrick.skill_targets import (
    RobotBaseFrame,
    SkillParams,
    TcpProvenance,
    pick_targets,
)
from rocobrick.skills import PhaseStatus, SkillPhase

EXPECTED_RUNTIME_MODEL_SHA256 = (
    "a781d52f9d55291fca9c8f2695abf9e311640d996314d272f47639d5323e2375"
)
CALIBRATION_SHA = "b" * 64
PROVENANCE = TcpProvenance(
    runtime_model_sha256=EXPECTED_RUNTIME_MODEL_SHA256,
    source_sha256=CALIBRATION_SHA,
)
BASE = RobotBaseFrame(
    position_xyz=(0.0, 0.0, 0.0), orientation_xyzw=(0.0, 0.0, 0.0, 1.0)
)
INSTANCE = "brick-test-1"
FRAME = "world"

# World-frame brick AABB (base frame == world frame for the identity base).
GROUND_MIN = (1.0, 1.0, 0.0)
GROUND_MAX = (1.032, 1.016, 0.016)
GROUND_CENTER = (1.016, 1.008, 0.008)
# The lifted brick must rise by exactly the configured lift height, so the
# fixture derives it from SkillParams instead of a stale magic number.
_LIFT = SkillParams().lift_height_m
LIFT_MIN = (1.0, 1.0, _LIFT)
LIFT_MAX = (1.032, 1.016, _LIFT + 0.016)
MOVED_MIN = (0.95, 1.0, 0.0)
MOVED_MAX = (0.982, 1.016, 0.016)


def params(**overrides: float) -> SkillParams:
    defaults = {
        name: getattr(SkillParams(), name) for name in vars(SkillParams())
    }
    return SkillParams(**{**defaults, **overrides})


def planning_scene_document() -> dict[str, object]:
    """A strictly valid PlanningSceneUpdate payload for one brick."""
    box = box_from_aabb(
        object_id=INSTANCE,
        frame_id=FRAME,
        minimum_m=GROUND_MIN,
        maximum_m=GROUND_MAX,
    )
    placeholder = PlanningSceneUpdate(
        scene_id="scene-test",
        boxes=(box,),
        expected_scene_digest="0" * 64,
    )
    update = PlanningSceneUpdate(
        scene_id="scene-test",
        boxes=(box,),
        expected_scene_digest=scene_digest(placeholder),
    )
    return asdict(update)


class FakeSession:
    """Fake READY session scripted by a per-request-document callable."""

    def __init__(self, script) -> None:
        self.script = script
        self.requests: list[dict] = []
        self.request_paths: list[Path] = []
        self.result_paths: list[Path] = []
        self.fail_run = False
        self.fail_kinds: set[str] | None = None

    def run_operation(self, request_path, result_path):
        document = json.loads(
            Path(request_path).read_text(encoding="utf-8")
        )
        self.requests.append(document)
        self.request_paths.append(Path(request_path))
        self.result_paths.append(Path(result_path))
        if self.fail_run or (
            self.fail_kinds is not None and document["kind"] in self.fail_kinds
        ):
            raise RuntimeSessionError("synthetic transport failure")
        result = self.script(document)
        Path(result_path).write_text(json.dumps(result), encoding="utf-8")
        return result


def passing_result(document: dict, execution: dict | None = None) -> dict:
    return {
        "schema": OPERATION_RESULT_SCHEMA,
        "request_id": document["request_id"],
        "kind": document["kind"],
        "passed": True,
        "execution": (
            execution
            if execution is not None
            else {"status": "SUCCEEDED", "is_success": True}
        ),
        "evidence": {"waited_ok": True, "passed": True},
        "error": None,
    }


def failing_result(document: dict, *, status: str | None = None) -> dict:
    execution: dict = {"status": status or "FAILED", "is_success": False}
    if status is None:
        execution.pop("status")
    return {
        "schema": OPERATION_RESULT_SCHEMA,
        "request_id": document["request_id"],
        "kind": document["kind"],
        "passed": False,
        "execution": execution,
        "evidence": {},
        "error": "synthetic operation failure",
    }


def scene_query_result(
    document: dict,
    *,
    sim_time_s: float,
    brick_min: tuple = GROUND_MIN,
    brick_max: tuple = GROUND_MAX,
    passed: bool = True,
    brick_ids: tuple = (INSTANCE,),
) -> dict:
    if not passed:
        return {
            "schema": OPERATION_RESULT_SCHEMA,
            "request_id": document["request_id"],
            "kind": document["kind"],
            "passed": False,
            "execution": {"motion_sent": False, "scene_state": None},
            "evidence": {},
            "error": "the scene-state query failed: missing brick",
        }
    return passing_result(
        document,
        execution={
            "motion_sent": False,
            "scene_state": {
                "sim_time_s": sim_time_s,
                "bricks": {
                    brick_id: {
                        "aabb_min": list(brick_min),
                        "aabb_max": list(brick_max),
                    }
                    for brick_id in brick_ids
                },
                "connections": {},
            },
        },
    )


def motion_script(
    *,
    gripper_results: list[bool] | None = None,
    pose_results: list[object] | None = None,
    planning_scene_result: bool = True,
) -> object:
    """Script motion results by observation order; queries use the PICK script."""
    gripper_seen = {"n": 0}
    pose_seen = {"n": 0}

    def script(document: dict) -> dict:
        kind = document["kind"]
        if kind == "SCENE_QUERY":
            return scene_query_result(
                document, sim_time_s=1.0
            )
        if kind == "APPLY_PLANNING_SCENE":
            return (
                passing_result(document)
                if planning_scene_result
                else failing_result(document)
            )
        if kind == "GRIPPER_TARGET":
            index = gripper_seen["n"]
            gripper_seen["n"] += 1
            outcomes = gripper_results if gripper_results is not None else []
            if index < len(outcomes) and outcomes[index] is False:
                return failing_result(document)
            return passing_result(document)
        if kind == "TCP_POSE":
            index = pose_seen["n"]
            pose_seen["n"] += 1
            outcomes = pose_results if pose_results is not None else []
            if index < len(outcomes):
                outcome = outcomes[index]
                if outcome == "REJECTED":
                    return failing_result(document, status="REJECTED")
                if outcome is False:
                    return failing_result(document)
            return passing_result(document)
        raise AssertionError(f"unexpected kind {kind}")

    return script


def pick_script(scene_states: list[tuple]) -> object:
    """Script successive SCENE_QUERY results as (sim_time, min, max) triples."""
    queries = {"n": 0}

    def script(document: dict) -> dict:
        if document["kind"] != "SCENE_QUERY":
            return passing_result(document)
        index = min(queries["n"], len(scene_states) - 1)
        queries["n"] += 1
        sim_time, brick_min, brick_max = scene_states[index]
        return scene_query_result(
            document,
            sim_time_s=sim_time,
            brick_min=brick_min,
            brick_max=brick_max,
        )

    return script


def make_backend(
    session: FakeSession,
    tmp_path: Path,
    *,
    skill: SkillType = SkillType.PICK,
    instance: str | None = INSTANCE,
    skill_params: SkillParams | None = None,
    planning_scene: dict | None | object = "default",
    prefix: str = "skill-0",
    timeout_s: float = 30.0,
) -> SessionPhaseBackend:
    return SessionPhaseBackend(
        session=session,
        run_dir=tmp_path / "ops",
        base=BASE,
        provenance=PROVENANCE,
        spec=SkillStepSpec(
            skill=skill,
            physical_instance_id=instance,
            params=skill_params if skill_params is not None else params(),
        ),
        planning_scene_update=(
            planning_scene_document()
            if planning_scene == "default"
            else planning_scene
        ),
        request_prefix=prefix,
        operation_timeout_s=timeout_s,
    )


def run_skill(task, backend) -> object:
    return asyncio.run(
        execute_skill(task, "run-1", backend, lambda event: None)
    )


def kinds(session: FakeSession) -> list[str]:
    return [request["kind"] for request in session.requests]


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_pick_happy_path_confirms_every_phase_and_orders_ops(tmp_path) -> None:
    session = FakeSession(
        pick_script(
            [
                (0.0, GROUND_MIN, GROUND_MAX),  # capture at reach_pregrasp
                (0.5, GROUND_MIN, GROUND_MAX),  # confirm_grasp: still
                (1.0, LIFT_MIN, LIFT_MAX),  # confirm_holding: followed
            ]
        )
    )
    backend = make_backend(session, tmp_path)
    task = SimpleTaskRef("skill-0-PICK", SkillType.PICK)
    result = run_skill(task, backend)
    assert [item.phase.phase_id for item in result.phases] == [
        "plan_grasp",
        "reach_pregrasp",
        "approach",
        "close_gripper",
        "confirm_grasp",
        "lift",
        "confirm_holding",
    ]
    assert all(
        item.status is PhaseStatus.CONFIRMED for item in result.phases
    )
    assert kinds(session) == [
        "APPLY_PLANNING_SCENE",
        "SCENE_QUERY",  # first motion phase captures the brick AABB
        "GRIPPER_TARGET",  # reach_pregrasp opens the gripper first
        "TCP_POSE",
        "TCP_POSE",  # approach
        "GRIPPER_TARGET",  # close_gripper
        "SCENE_QUERY",  # confirm_grasp
        "TCP_POSE",  # lift
        "SCENE_QUERY",  # confirm_holding
    ]
    # Every operation request and result document landed in the run dir.
    ops_dir = tmp_path / "ops"
    written = {path.name for path in ops_dir.iterdir()}
    assert any(name.endswith("-request.json") for name in written)
    assert any(name.endswith("-result.json") for name in written)
    assert len(session.request_paths) * 2 == len(written)


def test_reach_pregrasp_sends_gripper_then_admitted_tcp_pose(tmp_path) -> None:
    session = FakeSession(pick_script([(0.0, GROUND_MIN, GROUND_MAX)]))
    backend = make_backend(session, tmp_path)
    task = SimpleTaskRef("skill-0-PICK", SkillType.PICK)
    run_skill(task, backend)
    planning_scene = session.requests[0]
    capture = session.requests[1]
    gripper = session.requests[2]
    pose = session.requests[3]
    assert capture["kind"] == "SCENE_QUERY"
    assert planning_scene["payload"] == json.loads(
        json.dumps(planning_scene_document())
    )
    skill_params = params()
    targets = pick_targets(
        GROUND_MIN, GROUND_MAX, BASE, PROVENANCE, skill_params
    )
    assert gripper["payload"]["opening_m"] == pytest.approx(
        targets.reach_opening_m
    )
    assert gripper["payload"]["expected_opening_m"] == pytest.approx(
        list(targets.reach_expected_window_m)
    )
    assert gripper["payload"]["duration_s"] == pytest.approx(
        skill_params.gripper_duration_s
    )
    assert gripper["payload"]["goal_tolerance"] == pytest.approx(
        skill_params.gripper_goal_tolerance
    )
    assert pose["payload"]["frame_id"] == "base_link"
    assert pose["payload"]["tip_link"] == "rocobrick_tcp"
    assert pose["payload"]["tcp_status"] == "calibrated_project"
    assert pose["payload"]["runtime_model_sha256"] == (
        EXPECTED_RUNTIME_MODEL_SHA256
    )
    assert pose["payload"]["source_sha256"] == CALIBRATION_SHA
    assert pose["payload"]["position"] == pytest.approx(
        list(targets.pregrasp.position)
    )
    assert pose["payload"]["position_tolerance_m"] == pytest.approx(
        skill_params.position_tolerance_m
    )
    assert pose["payload"]["orientation_tolerance_rad"] == pytest.approx(
        skill_params.orientation_tolerance_rad
    )
    query = session.requests[-1]
    assert query["payload"]["brick_physical_instance_ids"] == [INSTANCE]


def test_transit_happy_path_sends_one_admitted_tcp_pose(tmp_path) -> None:
    session = FakeSession(pick_script([(0.0, GROUND_MIN, GROUND_MAX)]))
    backend = make_backend(session, tmp_path, skill=SkillType.TRANSIT)
    result = run_skill(SimpleTaskRef("skill-0-TRANSIT", SkillType.TRANSIT), backend)
    assert [item.status for item in result.phases] == [PhaseStatus.CONFIRMED]
    assert kinds(session) == ["SCENE_QUERY", "TCP_POSE"]
    skill_params = params()
    target = session.requests[-1]["payload"]
    assert target["position"] == pytest.approx(
        [
            1.016,
            1.008,
            0.008
            + skill_params.pregrasp_standoff_m
            + skill_params.transit_extra_height_m,
        ]
    )
    assert target["tip_link"] == "rocobrick_tcp"


# ---------------------------------------------------------------------------
# Failure-reason mapping
# ---------------------------------------------------------------------------


def run_pick_until_failure(tmp_path, script) -> tuple[list[str], list]:
    session = FakeSession(script)
    backend = make_backend(session, tmp_path)
    result = run_skill(SimpleTaskRef("skill-0-PICK", SkillType.PICK), backend)
    statuses = [
        (item.phase.phase_id, item.status, item.reason)
        for item in result.phases
    ]
    return kinds(session), statuses


def test_plan_grasp_without_scene_document_is_capability_unavailable(
    tmp_path,
) -> None:
    session = FakeSession(motion_script())
    backend = make_backend(session, tmp_path, planning_scene=None)
    result = run_skill(SimpleTaskRef("skill-0-PICK", SkillType.PICK), backend)
    assert session.requests == []
    failed = result.phases[0]
    assert (failed.phase.phase_id, failed.status, failed.reason) == (
        "plan_grasp",
        PhaseStatus.FAILED,
        FailureReason.CAPABILITY_UNAVAILABLE,
    )
    assert len(result.phases) == 1


def test_plan_grasp_failure_is_planning_failed(tmp_path) -> None:
    kinds_seen, statuses = run_pick_until_failure(
        tmp_path, motion_script(planning_scene_result=False)
    )
    assert kinds_seen == ["APPLY_PLANNING_SCENE"]
    assert statuses == [
        ("plan_grasp", PhaseStatus.FAILED, FailureReason.PLANNING_FAILED)
    ]


def test_reach_pregrasp_gripper_failure_is_tracking_failed(tmp_path) -> None:
    kinds_seen, statuses = run_pick_until_failure(
        tmp_path, motion_script(gripper_results=[False])
    )
    assert kinds_seen == [
        "APPLY_PLANNING_SCENE",
        "SCENE_QUERY",
        "GRIPPER_TARGET",
    ]
    assert statuses[-1] == (
        "reach_pregrasp",
        PhaseStatus.FAILED,
        FailureReason.TRACKING_FAILED,
    )


def test_reach_pregrasp_rejected_pose_is_collision_blocked(tmp_path) -> None:
    kinds_seen, statuses = run_pick_until_failure(
        tmp_path, motion_script(pose_results=["REJECTED"])
    )
    assert kinds_seen == [
        "APPLY_PLANNING_SCENE",
        "SCENE_QUERY",
        "GRIPPER_TARGET",
        "TCP_POSE",
    ]
    assert statuses[-1] == (
        "reach_pregrasp",
        PhaseStatus.FAILED,
        FailureReason.COLLISION_BLOCKED,
    )


def test_reach_pregrasp_failed_pose_is_tracking_failed(tmp_path) -> None:
    kinds_seen, statuses = run_pick_until_failure(
        tmp_path, motion_script(pose_results=[False])
    )
    assert statuses[-1] == (
        "reach_pregrasp",
        PhaseStatus.FAILED,
        FailureReason.TRACKING_FAILED,
    )


def test_approach_failure_is_tracking_failed(tmp_path) -> None:
    kinds_seen, statuses = run_pick_until_failure(
        tmp_path, motion_script(pose_results=[True, False])
    )
    assert kinds_seen == [
        "APPLY_PLANNING_SCENE",
        "SCENE_QUERY",
        "GRIPPER_TARGET",
        "TCP_POSE",
        "TCP_POSE",
    ]
    assert statuses[-1] == (
        "approach",
        PhaseStatus.FAILED,
        FailureReason.TRACKING_FAILED,
    )


def test_close_gripper_failure_is_tracking_failed(tmp_path) -> None:
    kinds_seen, statuses = run_pick_until_failure(
        tmp_path, motion_script(gripper_results=[True, False])
    )
    assert kinds_seen == [
        "APPLY_PLANNING_SCENE",
        "SCENE_QUERY",
        "GRIPPER_TARGET",
        "TCP_POSE",
        "TCP_POSE",
        "GRIPPER_TARGET",
    ]
    assert statuses[-1] == (
        "close_gripper",
        PhaseStatus.FAILED,
        FailureReason.TRACKING_FAILED,
    )


def test_confirm_grasp_stillness_failure_is_holding_failed(tmp_path) -> None:
    session = FakeSession(
        pick_script(
            [
                (0.0, GROUND_MIN, GROUND_MAX),  # capture
                (0.5, MOVED_MIN, MOVED_MAX),  # confirm_grasp: brick moved
            ]
        )
    )
    backend = make_backend(session, tmp_path)
    result = run_skill(SimpleTaskRef("skill-0-PICK", SkillType.PICK), backend)
    failed = result.phases[-1]
    assert (failed.phase.phase_id, failed.status, failed.reason) == (
        "confirm_grasp",
        PhaseStatus.FAILED,
        FailureReason.HOLDING_FAILED,
    )


def test_confirm_grasp_missing_brick_is_stale_world(tmp_path) -> None:
    queries = {"n": 0}

    def scripted(document: dict) -> dict:
        if document["kind"] == "SCENE_QUERY":
            queries["n"] += 1
            if queries["n"] == 2:
                return scene_query_result(
                    document, sim_time_s=0.5, passed=False
                )
            return scene_query_result(
                document, sim_time_s=0.0, brick_min=GROUND_MIN, brick_max=GROUND_MAX
            )
        return passing_result(document)

    kinds_seen, statuses = run_pick_until_failure(tmp_path, scripted)
    assert kinds_seen == [
        "APPLY_PLANNING_SCENE",
        "SCENE_QUERY",
        "GRIPPER_TARGET",
        "TCP_POSE",
        "TCP_POSE",
        "GRIPPER_TARGET",
        "SCENE_QUERY",
    ]
    assert statuses[-1] == (
        "confirm_grasp",
        PhaseStatus.FAILED,
        FailureReason.STALE_WORLD,
    )


def test_confirm_holding_follow_failure_is_holding_failed(tmp_path) -> None:
    session = FakeSession(
        pick_script(
            [
                (0.0, GROUND_MIN, GROUND_MAX),  # capture
                (0.5, GROUND_MIN, GROUND_MAX),  # confirm_grasp: still
                (1.0, GROUND_MIN, GROUND_MAX),  # confirm_holding: not lifted
            ]
        )
    )
    backend = make_backend(session, tmp_path)
    result = run_skill(SimpleTaskRef("skill-0-PICK", SkillType.PICK), backend)
    failed = result.phases[-1]
    assert (failed.phase.phase_id, failed.status, failed.reason) == (
        "confirm_holding",
        PhaseStatus.FAILED,
        FailureReason.HOLDING_FAILED,
    )


def test_lift_failure_is_tracking_failed(tmp_path) -> None:
    kinds_seen, statuses = run_pick_until_failure(
        tmp_path, motion_script(pose_results=[True, True, False])
    )
    assert kinds_seen[-1] == "TCP_POSE"
    assert statuses[-1] == (
        "lift",
        PhaseStatus.FAILED,
        FailureReason.TRACKING_FAILED,
    )


def test_missing_brick_at_capture_is_stale_world(tmp_path) -> None:
    def scripted(document: dict) -> dict:
        if document["kind"] == "SCENE_QUERY":
            return scene_query_result(
                document,
                sim_time_s=0.0,
                passed=False,
            )
        return passing_result(document)

    kinds_seen, statuses = run_pick_until_failure(tmp_path, scripted)
    assert kinds_seen == ["APPLY_PLANNING_SCENE", "SCENE_QUERY"]
    assert statuses[-1] == (
        "reach_pregrasp",
        PhaseStatus.FAILED,
        FailureReason.STALE_WORLD,
    )


def test_scene_query_without_the_instance_is_stale_world(tmp_path) -> None:
    def scripted(document: dict) -> dict:
        if document["kind"] == "SCENE_QUERY":
            return scene_query_result(
                document,
                sim_time_s=0.0,
                brick_min=GROUND_MIN,
                brick_max=GROUND_MAX,
                brick_ids=("somebody-else",),
            )
        return passing_result(document)

    kinds_seen, statuses = run_pick_until_failure(tmp_path, scripted)
    assert statuses[-1] == (
        "reach_pregrasp",
        PhaseStatus.FAILED,
        FailureReason.STALE_WORLD,
    )


def test_runtime_session_error_becomes_a_failed_result(tmp_path) -> None:
    session = FakeSession(pick_script([(0.0, GROUND_MIN, GROUND_MAX)]))
    session.fail_kinds = {"TCP_POSE"}
    backend = make_backend(
        session, tmp_path, skill=SkillType.TRANSIT
    )
    result = run_skill(
        SimpleTaskRef("skill-0-TRANSIT", SkillType.TRANSIT), backend
    )
    assert result.phases[0].status is PhaseStatus.FAILED
    assert result.phases[0].reason is FailureReason.TRACKING_FAILED


def test_unparsable_result_document_becomes_a_failed_result(tmp_path) -> None:
    def scripted(document: dict) -> dict:
        if document["kind"] == "SCENE_QUERY":
            return scene_query_result(document, sim_time_s=0.0)
        return {"schema": "bogus"}

    session = FakeSession(scripted)
    backend = make_backend(session, tmp_path, skill=SkillType.TRANSIT)
    result = run_skill(
        SimpleTaskRef("skill-0-TRANSIT", SkillType.TRANSIT), backend
    )
    assert result.phases[0].status is PhaseStatus.FAILED
    assert result.phases[0].reason is FailureReason.TRACKING_FAILED


def test_unknown_phase_id_is_capability_unavailable(tmp_path) -> None:
    session = FakeSession(pick_script([(0.0, GROUND_MIN, GROUND_MAX)]))
    backend = make_backend(session, tmp_path)
    result = asyncio.run(
        backend.execute_phase(
            SimpleTaskRef("t", SkillType.PICK),
            SkillPhase("bogus_phase", ActionType.PICK, "x"),
            "intent",
            "attempt",
        )
    )
    assert result.status is PhaseStatus.FAILED
    assert result.reason is FailureReason.CAPABILITY_UNAVAILABLE
    assert session.requests == []


# ---------------------------------------------------------------------------
# WAIT (hold)
# ---------------------------------------------------------------------------


def test_wait_hold_passes_when_sim_time_advances(tmp_path) -> None:
    queries = {"n": 0}

    def scripted(document: dict) -> dict:
        if document["kind"] == "SCENE_QUERY":
            queries["n"] += 1
            sim_time = 10.0 if queries["n"] == 1 else 10.6
            return scene_query_result(document, sim_time_s=sim_time)
        raise AssertionError("WAIT sends only scene queries")

    session = FakeSession(scripted)
    backend = make_backend(
        session,
        tmp_path,
        skill=SkillType.WAIT,
        instance=None,
        skill_params=params(wait_duration_s=0.5, scene_query_poll_s=0.001),
    )
    result = run_skill(SimpleTaskRef("skill-0-WAIT", SkillType.WAIT), backend)
    assert [item.status for item in result.phases] == [PhaseStatus.CONFIRMED]
    assert queries["n"] >= 2
    assert kinds(session) == ["SCENE_QUERY"] * queries["n"]
    query = session.requests[0]
    assert query["payload"]["brick_physical_instance_ids"] == [INSTANCE]
    assert query["payload"]["connection_pairs"] == []
    assert query["payload"]["include_gripper"] is False


def test_wait_hold_times_out_as_temporarily_unreachable(tmp_path) -> None:
    session = FakeSession(
        scripted_wait(sim_time=10.0)
    )
    backend = make_backend(
        session,
        tmp_path,
        skill=SkillType.WAIT,
        instance=None,
        skill_params=params(wait_duration_s=60.0, scene_query_poll_s=0.005),
        timeout_s=0.05,
    )
    result = run_skill(SimpleTaskRef("skill-0-WAIT", SkillType.WAIT), backend)
    assert result.phases[0].status is PhaseStatus.FAILED
    assert result.phases[0].reason is FailureReason.TEMPORARILY_UNREACHABLE
    assert len(session.requests) >= 2


def test_wait_initial_query_failure_is_stale_world(tmp_path) -> None:
    session = FakeSession(scripted_wait(sim_time=None, fail_first=True))
    backend = make_backend(
        session,
        tmp_path,
        skill=SkillType.WAIT,
        instance=None,
        skill_params=params(wait_duration_s=0.1, scene_query_poll_s=0.001),
    )
    result = run_skill(SimpleTaskRef("skill-0-WAIT", SkillType.WAIT), backend)
    assert result.phases[0].status is PhaseStatus.FAILED
    assert result.phases[0].reason is FailureReason.STALE_WORLD


def scripted_wait(sim_time: float | None, fail_first: bool = False):
    queries = {"n": 0}

    def script(document: dict) -> dict:
        queries["n"] += 1
        if fail_first and queries["n"] == 1:
            return scene_query_result(document, sim_time_s=0.0, passed=False)
        return scene_query_result(document, sim_time_s=float(sim_time or 0.0))

    return script


def test_wait_without_known_scene_objects_is_stale_world(tmp_path) -> None:
    session = FakeSession(scripted_wait(sim_time=1.0))
    backend = make_backend(
        session,
        tmp_path,
        skill=SkillType.WAIT,
        instance=None,
        planning_scene=None,
    )
    result = run_skill(SimpleTaskRef("skill-0-WAIT", SkillType.WAIT), backend)
    assert result.phases[0].status is PhaseStatus.FAILED
    assert result.phases[0].reason is FailureReason.STALE_WORLD
    assert session.requests == []


class SimpleTaskRef:
    """Minimal TaskRef: only task_id and skill."""

    def __init__(self, task_id: str, skill: SkillType) -> None:
        self.task_id = task_id
        self.skill = skill
