from __future__ import annotations

from rocobrick.models import ActionType, Candidate, ExecutionSnapshot, Revisions, RobotSnapshot
from rocobrick.scheduler import decide
from rocobrick.task_facts import TaskFactStore


def snapshot():
    revisions = Revisions(1, 2, 3)
    robots = {
        robot: RobotSnapshot(robot, (0.0,) * 6, 1.0)
        for robot in ("piper_0", "piper_1")
    }
    return ExecutionSnapshot(revisions, robots)


def candidate(cid, task, robot, resources, **kwargs):
    return Candidate(cid, task, ActionType.TRANSPORT, robot, Revisions(1, 2, 3), frozenset(resources), True, **kwargs)


def test_same_snapshot_is_deterministic_regardless_of_input_iteration(parsed):
    plan, _ = parsed
    facts = TaskFactStore(plan, "run-1").facts
    execution = snapshot()
    candidates = (
        candidate("right", "cap@right", "piper_1", {"right-zone"}),
        candidate("left", "foundation:left", "piper_0", {"left-zone"}),
    )
    first = decide(plan, facts, execution, candidates)
    second = decide(plan, facts, execution, tuple(reversed(candidates)))
    assert first == second
    assert {item.candidate_id for item in first.selected} == {"left", "right"}


def test_shared_space_is_mutually_exclusive(parsed):
    plan, _ = parsed
    facts = TaskFactStore(plan, "run-1").facts
    candidates = (
        candidate("right", "cap@right", "piper_1", {"shared"}),
        candidate("left", "foundation:left", "piper_0", {"shared"}),
    )
    decision = decide(plan, facts, snapshot(), candidates)
    assert [item.candidate_id for item in decision.selected] == ["right"]
    assert decision.waiting["left"].value == "RESOURCE_WAIT"


def test_same_arm_receives_only_one_action(parsed):
    plan, _ = parsed
    facts = TaskFactStore(plan, "run-1").facts
    candidates = (
        candidate("a", "foundation:left", "piper_0", {"a"}),
        candidate("b", "joined/top", "piper_0", {"b"}, holding_continuation=True),
    )
    decision = decide(plan, facts, snapshot(), candidates)
    assert len(decision.selected) == 1


def test_stale_candidate_is_not_selected(parsed):
    plan, _ = parsed
    facts = TaskFactStore(plan, "run-1").facts
    stale = Candidate("stale", "foundation:left", ActionType.PICK, "piper_0", Revisions(0, 2, 3), frozenset(), True)
    decision = decide(plan, facts, snapshot(), (stale,))
    assert not decision.selected
    assert decision.waiting["stale"].value == "STALE_WORLD"


def test_controlled_stop_selects_at_most_one_intent_per_robot(parsed):
    plan, _ = parsed
    facts = TaskFactStore(plan, "run-1").facts
    execution = snapshot()
    execution = type(execution)(
        execution.revisions,
        execution.robots,
        safety_stop_requested=True,
    )
    stops = tuple(
        Candidate(
            candidate_id,
            "foundation:left",
            ActionType.CONTROLLED_STOP,
            robot,
            execution.revisions,
            frozenset(),
            True,
        )
        for candidate_id, robot in (
            ("second", "piper_0"),
            ("first", "piper_0"),
            ("right", "piper_1"),
        )
    )
    decision = decide(plan, facts, execution, stops)
    assert [(item.robot_id, item.candidate_id) for item in decision.selected] == [
        ("piper_0", "first"),
        ("piper_1", "right"),
    ]
