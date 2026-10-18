"""Bounded full-gripper endpoint admission without ROS or physical dispatch."""

import asyncio
from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
from test_pick_grasp_region import context, endpoint, query, region, sample
from test_pick_handler import Observer, document, operation

from rocobrick.execution.operation_state_ledger import OperationStateLedger
from rocobrick.skills.grasp_region import place_up_grasp_region
from rocobrick.skills.pick import PickHandler
from rocobrick.skills.targets import pick_targets


def verdict(*, passed=False, code="GRASP_APPROACH_WORLD_CLEARANCE", **lifecycle):
    """Return explicit plan-only evidence, never simulated arrival or holding.

    Returns:
        A planner result with a structured endpoint rejection or admitted pose.
    """
    trajectory = {
        "joint_names": ["piper_0_joint1"],
        "points": [{"positions": [0.1], "time_from_start_s": 1.0}],
    }
    if passed:
        trajectory.update(
            {
                "selection_only_discard_path": True,
                "selected_grasp_endpoints": {
                    "piper_0": {"tcp_pose": asdict(endpoint().tcp_pose)}
                },
                "grasp_endpoint_clearance": {"checked_pairs": 3},
            }
        )
    else:
        trajectory["candidate_rejection"] = {
            "code": code,
            "stage": "endpoint_geometry",
            "reason": "3.747mm < 4mm",
        }
    return SimpleNamespace(
        passed=passed,
        error=None if passed else "candidate rejected",
        trajectory=trajectory,
        lifecycle={
            "plan_only": True,
            "physical_execution": False,
            "dispatch_attempted": False,
            "prepared": False,
            "started": False,
            "terminal": True,
            **lifecycle,
        },
    )


def selection_fixture(outcomes, *, refresh=None):
    """Create a pure plan-only handler and independently captured snapshots.

    Returns:
        Handler, call arguments, audit dictionary and captured request lists.
    """
    scene = document()
    initial_region = place_up_grasp_region(sample(), context())
    scene["robots"]["piper_0"]["opening_m"] = initial_region["approach_opening_m"]
    refreshed = deepcopy(scene)
    refreshed["sequence"] += 1
    refreshed["sim_time_s"] += 0.1
    refreshed["robots"]["piper_0"]["sequence"] += 1
    refreshed["robots"]["piper_0"]["sim_time_s"] += 0.1
    requests, captures = [], []

    async def capture(participants):
        captures.append(participants)
        if refresh is not None:
            refresh(refreshed)
        return refreshed, 2, {"piper_0_joint1": 0.2}

    async def plan(request):
        requests.append(request)
        return outcomes[min(len(requests) - 1, len(outcomes) - 1)]

    handler = PickHandler(
        Observer(scene),
        OperationStateLedger(("piper_0", "piper_1")),
        session_id="session-1",
        manifest_sha256="a" * 64,
        context=context(),
        free_space_transport=SimpleNamespace(
            robot_ids=("piper_0", "piper_1"),
            plan_free_space=plan,
            free_space_context=capture,
        ),
    )
    observed = sample()
    target = pick_targets(
        observed.aabb_min,
        observed.aabb_max,
        context().base,
        context().provenance,
        context().params,
    ).precontact_boundary
    args = (operation(), scene, 1, {"piper_0_joint1": 0.1}, initial_region, target)
    return handler, args, {}, requests, captures


def test_rejected_endpoint_resamples_from_fresh_scene_without_executing():
    """A fresh accepted region, not the rejected region, reaches the handler."""
    handler, args, audit, requests, captures = selection_fixture(
        [verdict(), verdict(passed=True)]
    )
    _, _, _, proof = asyncio.run(
        handler._select_region_hover(*args, selection_evidence=audit)
    )
    assert len(requests) == 2 and len(captures) == 1
    assert requests[0].motion_id != requests[1].motion_id
    assert requests[1].scene_version == 2
    assert requests[1].start_state == {"piper_0_joint1": 0.2}
    assert requests[1].scene_snapshot is not requests[0].scene_snapshot
    assert requests[1].timeout_s <= requests[0].timeout_s <= 60.0
    assert proof["region"] is requests[1].targets["piper_0"]["grasp_region"]
    assert proof["region"] is not requests[0].targets["piper_0"]["grasp_region"]
    assert proof["region"]["floor_margin_m"] == args[4]["floor_margin_m"]
    assert audit["stop_reason"] == "accepted" and audit["accepted_attempt"] == 2
    assert len(audit["attempts"]) == 2
    assert audit["attempts"][0]["trajectory"]["points"]
    assert not handler._active_free_space


def test_all_endpoint_rejections_exhaust_four_candidates_without_motion():
    """Exhaustion preserves every rejected endpoint and never starts a hover."""
    handler, args, audit, requests, captures = selection_fixture([verdict()])
    with pytest.raises(ValueError, match="candidate limit exhausted"):
        asyncio.run(handler._select_region_hover(*args, selection_evidence=audit))
    assert len(requests) == len(audit["attempts"]) == 4
    assert len(captures) == 3
    assert audit["stop_reason"] == "candidate_limit_exhausted"
    assert all(a["lifecycle"]["dispatch_attempted"] is False for a in audit["attempts"])


@pytest.mark.parametrize("code", ["scene_snapshot_fresh", "IK_FAILED", "unknown"])
def test_other_failures_are_not_retried(code):
    """Readiness, planning and unknown failures cannot trigger blind retries."""
    handler, args, audit, requests, captures = selection_fixture([verdict(code=code)])
    with pytest.raises(ValueError, match="endpoint selection failed"):
        asyncio.run(handler._select_region_hover(*args, selection_evidence=audit))
    assert len(requests) == 1 and not captures
    assert audit["stop_reason"] == "non_retryable_failure"


@pytest.mark.parametrize(
    "field,value",
    [
        ("dispatch_attempted", True),
        ("physical_execution", True),
        ("started", True),
        ("prepared", True),
        ("terminal", False),
        ("plan_only", False),
    ],
)
def test_clearance_rejection_without_no_dispatch_proof_is_not_retried(field, value):
    """A known geometry reason never overrides uncertain physical lifecycle."""
    handler, args, audit, requests, captures = selection_fixture(
        [verdict(**{field: value})]
    )
    with pytest.raises(ValueError, match="strict no-dispatch proof"):
        asyncio.run(handler._select_region_hover(*args, selection_evidence=audit))
    assert len(requests) == 1 and not captures


def test_moved_brick_aborts_instead_of_rebinding_away_the_disturbance():
    """Subsequent fresh observations still compare against the initial brick."""

    def disturb(scene):
        scene["bricks"]["brick-a"]["pose"]["position_xyz"][0] += 0.002

    handler, args, audit, requests, _ = selection_fixture([verdict()], refresh=disturb)
    with pytest.raises(ValueError, match="brick moved"):
        asyncio.run(handler._select_region_hover(*args, selection_evidence=audit))
    assert len(requests) == 1 and audit["stop_reason"] == "object_moved"


def test_shared_selection_deadline_prevents_another_planner_call(monkeypatch):
    """Four proposals do not multiply the original selection timeout by four."""
    handler, args, audit, requests, captures = selection_fixture([verdict()])
    times = iter([0.0, 0.0, 0.0, 61.0])
    monkeypatch.setattr(
        "rocobrick.skills.pick.time", SimpleNamespace(monotonic=lambda: next(times))
    )
    with pytest.raises(ValueError, match="budget exhausted"):
        asyncio.run(handler._select_region_hover(*args, selection_evidence=audit))
    assert len(requests) == 1 and len(captures) == 1
    assert audit["stop_reason"] == "time_budget_exhausted"


@pytest.mark.parametrize(
    "message,retryable",
    [
        (
            "GRASP_APPROACH_WORLD_CLEARANCE: finger vs platform: "
            "0.003747 m < 0.004000 m",
            True,
        ),
        ("missing world hull", False),
    ],
)
def test_session_marks_only_endpoint_geometry_failure_as_retryable(
    monkeypatch, message, retryable
):
    """The real plan-only entry owns the typed rejection and dispatch evidence."""
    from rocobrick.coordinated_session_entry import perform_coordinated_plan

    item = replace(
        query(
            {
                "piper_0": {
                    "grasp_region": region(),
                    "pick_endpoint_selection_only": True,
                }
            }
        ),
        physical_facts={"piper_0": {}},
    )
    session = SimpleNamespace(
        robot_ids=item.robots,
        coordinated_readiness=lambda request: {
            "coordination_id": request.coordination_id,
            "scene_digest": request.scene_digest,
            "scene_sequence": 10,
            "scene_sim_time_s": 100.0,
            "participants": list(request.participants),
            "move_group_action": "/dual_arm/move_action",
            "planning_group": "dual_arm",
            "components": {"scene": True, "move_group": True},
        },
        sync_unified_scene=lambda request: {
            "readback_confirmed": True,
            "snapshot_digest": request.scene_digest,
        },
        validate_unified_scene=lambda proof: True,
        unified_start_state=lambda: item.start_state,
        unified_robot_description=lambda: "<robot/>",
        runtime_for=lambda robot: SimpleNamespace(arm_joint_names=(f"{robot}_joint1",)),
        coordinated_plan_runtime=lambda: SimpleNamespace(
            plan=lambda *a, **k: {
                "joint_names": ["piper_0_joint1"],
                "points": [
                    {"positions": [0.1], "time_from_start_s": 0.0},
                    {"positions": [0.15], "time_from_start_s": 1.0},
                ],
            }
        ),
    )
    monkeypatch.setattr(
        "rocobrick.skills.grasp_region.planned_grasp_endpoints", lambda *a: {}
    )

    def reject(*args):
        raise ValueError(message)

    monkeypatch.setattr(
        "rocobrick.coordinated_session_entry.grasp_approach_clearance", reject
    )
    result = perform_coordinated_plan(session, item)
    assert result.passed is False and message in result.error
    assert ("candidate_rejection" in result.trajectory) is retryable
    assert result.trajectory["points"]
    assert result.lifecycle["plan_only"] is True
    assert result.lifecycle["dispatch_attempted"] is False
    assert result.lifecycle["physical_execution"] is False
    assert result.lifecycle["terminal"] is True


def test_error_text_alone_cannot_enable_candidate_retry():
    """An untyped failure is not a geometry proposal even if text looks similar."""
    failure = verdict()
    failure.trajectory.pop("candidate_rejection")
    failure.error = "GRASP_APPROACH_WORLD_CLEARANCE: untyped failure"
    handler, args, audit, requests, captures = selection_fixture([failure])
    with pytest.raises(ValueError, match="endpoint selection failed"):
        asyncio.run(handler._select_region_hover(*args, selection_evidence=audit))
    assert len(requests) == 1 and not captures


def test_success_without_discarded_path_proof_is_not_an_executable_candidate():
    """Planner success is not authorization to execute its incidental path."""
    success = verdict(passed=True)
    success.trajectory.pop("selection_only_discard_path")
    handler, args, audit, requests, captures = selection_fixture([success])
    with pytest.raises(ValueError, match="discarded-path proof"):
        asyncio.run(handler._select_region_hover(*args, selection_evidence=audit))
    assert len(requests) == 1 and not captures


def test_missing_dispatch_evidence_is_not_assumed_false():
    """Unknown dispatch state must stop even an otherwise valid endpoint."""
    failure = verdict()
    failure.lifecycle.pop("dispatch_attempted")
    handler, args, audit, requests, captures = selection_fixture([failure])
    with pytest.raises(ValueError, match="strict no-dispatch proof"):
        asyncio.run(handler._select_region_hover(*args, selection_evidence=audit))
    assert len(requests) == 1 and not captures


def test_over_budget_success_is_not_admitted(monkeypatch):
    """Late planner completion cannot silently extend the selection deadline."""
    handler, args, audit, requests, captures = selection_fixture([verdict(passed=True)])
    times = iter([0.0, 0.0, 61.0])
    monkeypatch.setattr(
        "rocobrick.skills.pick.time", SimpleNamespace(monotonic=lambda: next(times))
    )
    with pytest.raises(ValueError, match="budget exhausted"):
        asyncio.run(handler._select_region_hover(*args, selection_evidence=audit))
    assert len(requests) == 1 and not captures
    assert audit["stop_reason"] == "time_budget_exhausted"
