"""Parse tests for the ``robot-operation-plan/v1`` single-payload contract.

These tests pin the frozen wire shape (exact keys, the six skills, target
shapes, and ``robot:``/``part:``/``connection:`` resources) and the
self-contained ingress rules: grounded part/connection references, participant
and role-binding agreement, layer/schedule coverage, dependency order, and
same-layer resource exclusivity. ``source_plan_id`` is a validated field of the
payload and is never cross-checked against a second document; the committed
assembly-plan/v2 fixture only pins that literal id. Offline only; no Isaac,
ROS, Docker, or motion.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from rocobrick.robot_operation_plan import (
    RobotOperationPlanError,
    ScheduledOperation,
    parse_robot_operation_plan,
)
from rocobrick.stap_contract import parse_stap_plan

REPO_ROOT = Path(__file__).resolve().parents[1]
# Committed assembly-plan/v2 fixture whose canonical plan_id is the exact
# ``source_plan_id`` declared by the committed example1 operation-plan
# artifact. All tests stay on tracked files: no run artifact is required.
EXAMPLE1_STAP_PATH = (
    REPO_ROOT / "tests" / "fixtures" / "assembly_plans" / "example1-scene-smoke.json"
)
EXAMPLE1_PLAN_PATH = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "robot_operation_plans"
    / "example1-robot-operation-plan.json"
)


def _load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Plan builders.
# ---------------------------------------------------------------------------


def _ref(kind, *, part=None, connection=None, robot=None):
    return {
        "kind": kind,
        "physical_instance_id": part,
        "connection_id": connection,
        "robot_id": robot,
    }


def _transit(source, destination, carrying=None):
    return {
        "from": source,
        "to": destination,
        "carrying_physical_instance_id": carrying,
    }


def _place(connection, upper, lower):
    return {
        "connection_id": connection,
        "upper_physical_instance_id": upper,
        "lower_physical_instance_id": lower,
    }


def _operation(
    operation_id,
    source_task_id,
    skill,
    executor,
    *,
    participants=None,
    target=None,
    resources=None,
    evidence=(),
    parameters=None,
    role_bindings=None,
):
    participants = [executor] if participants is None else list(participants)
    if role_bindings is None:
        role_bindings = (
            {"executor": executor}
            if len(participants) == 1
            else {"executor": executor, "supporting": participants[1]}
        )
    return {
        "operation_id": operation_id,
        "source_task_id": source_task_id,
        "skill": skill,
        "executor_robot_id": executor,
        "participants": participants,
        "role_bindings": role_bindings,
        "target": target,
        "parameters": {} if parameters is None else parameters,
        "preconditions": [],
        "effects": [],
        "resources": [f"robot:{executor}"] if resources is None else list(resources),
        "evidence_requirements": list(evidence),
    }


def _schedule(*slots):
    return [[dict(entry) for entry in slot] for slot in slots]


def _plan(
    operations,
    layers,
    schedule,
    *,
    dependencies=(),
    evidence=(),
    robots=("piper_a", "piper_b"),
    plan_id="robot-op-synthetic",
    source_plan_id="assembly-plan-synthetic",
):
    return {
        "schema": "robot-operation-plan/v1",
        "plan_id": plan_id,
        "source_plan_id": source_plan_id,
        "robots": list(robots),
        "operations": list(operations),
        "dependencies": [dict(item) for item in dependencies],
        "execution_layers": [list(layer) for layer in layers],
        "planned_schedule": [list(slot) for slot in schedule],
        "evidence_requirements": [dict(item) for item in evidence],
    }


def _default_operations():
    return [
        _operation(
            "a-pick",
            "task-mid",
            "PICK",
            "piper_a",
            target={"physical_instance_id": "brick-mid"},
            resources=["robot:piper_a", "part:brick-mid"],
            evidence=["holding_confirmed"],
        ),
        _operation(
            "a-place",
            "task-mid",
            "PLACE_DOWN",
            "piper_a",
            target=_place("conn-mid-base", "brick-mid", "brick-base"),
            resources=[
                "robot:piper_a",
                "connection:conn-mid-base",
                "part:brick-mid",
                "part:brick-base",
            ],
            evidence=["connection_confirmed"],
        ),
        _operation(
            "b-pick",
            "task-top",
            "PICK",
            "piper_b",
            target={"physical_instance_id": "brick-top"},
            resources=["robot:piper_b", "part:brick-top"],
        ),
        _operation(
            "b-place",
            "task-top",
            "PLACE_DOWN",
            "piper_b",
            target=_place("conn-top-mid", "brick-top", "brick-mid"),
            resources=[
                "robot:piper_b",
                "connection:conn-top-mid",
                "part:brick-top",
                "part:brick-mid",
            ],
            evidence=["connection_confirmed"],
        ),
    ]


_DEFAULT_LAYERS = (
    ("a-pick",),
    ("a-place",),
    ("b-pick",),
    ("b-place",),
)
_DEFAULT_SCHEDULE = _schedule(
    ({"operation_id": "a-pick", "robot_id": "piper_a"},),
    ({"operation_id": "a-place", "robot_id": "piper_a"},),
    ({"operation_id": "b-pick", "robot_id": "piper_b"},),
    ({"operation_id": "b-place", "robot_id": "piper_b"},),
)


def _dependency(before, after, reason):
    return {
        "predecessor_operation_id": before,
        "successor_operation_id": after,
        "reason": reason,
    }


_DEFAULT_DEPENDENCIES = (
    _dependency("a-pick", "a-place", "place requires holding"),
    _dependency("a-place", "b-pick", "pick after support placed"),
    _dependency("b-pick", "b-place", "place requires holding"),
)


def _default_plan(**overrides):
    parameters = {
        "operations": _default_operations(),
        "layers": _DEFAULT_LAYERS,
        "schedule": _DEFAULT_SCHEDULE,
        "dependencies": _DEFAULT_DEPENDENCIES,
    }
    parameters.update(overrides)
    return _plan(
        parameters["operations"],
        parameters["layers"],
        parameters["schedule"],
        dependencies=parameters["dependencies"],
        evidence=parameters.get("evidence", ()),
        robots=parameters.get("robots", ("piper_a", "piper_b")),
        source_plan_id=parameters.get("source_plan_id", "assembly-plan-synthetic"),
    )


def _one(operation, *, layers=None, schedule=None, **overrides):
    layers = [list(layer) for layer in (layers or ([operation["operation_id"]],))]
    entry = {
        "operation_id": operation["operation_id"],
        "robot_id": operation["executor_robot_id"],
    }
    schedule = schedule or _schedule((entry,))
    return _plan([operation], layers, schedule, **overrides)


# ---------------------------------------------------------------------------
# example1 artifact binding.
# ---------------------------------------------------------------------------


def test_example1_stap_fixture_matches_the_artifact() -> None:
    stap = parse_stap_plan(_load(EXAMPLE1_STAP_PATH))
    artifact = _load(EXAMPLE1_PLAN_PATH)
    assert stap.plan_id == "assembly-plan-25a5d5cc586a1e5c65f2141d"
    assert artifact["source_plan_id"] == stap.plan_id
    assert stap.robots == ("piper_0", "piper_1")
    assert set(stap.goal_parts) == {
        "brick-4bbba49bec24496bb21e",
        "brick-768de5a17827920c48d4",
    }
    task = stap.tasks["assemble-768de5a17827920c48d4"]
    assert task.physical_instance_id == "brick-768de5a17827920c48d4"
    assert task.primary_robot == "piper_0"
    assert task.supporting_robot is None
    assert task.initial_supports == frozenset({"brick-4bbba49bec24496bb21e"})


def test_example1_artifact_parses() -> None:
    """The committed example1 artifact satisfies the self-contained contract."""
    parsed = parse_robot_operation_plan(_load(EXAMPLE1_PLAN_PATH))
    assert parsed.plan_id == "robot-operation-plan-example1-v2"
    assert parsed.source_plan_id == "assembly-plan-25a5d5cc586a1e5c65f2141d"
    assert parsed.robots == ("piper_0", "piper_1")
    assert [operation.skill for operation in parsed.operations.values()] == [
        "WAIT",
        "PICK",
        "PLACE_DOWN",
    ]
    assert parsed.operations["example1-piper0-place-down"].target.values[
        "lower_physical_instance_id"
    ] == "brick-4bbba49bec24496bb21e"


def test_example1_artifact_parses_without_a_source_plan() -> None:
    """The example1 operation set is exact without any second plan document."""
    parsed = parse_robot_operation_plan(_load(EXAMPLE1_PLAN_PATH))
    assert set(parsed.operations) == {
        "example1-piper1-wait-stable",
        "example1-piper0-pick",
        "example1-piper0-place-down",
    }


def test_example1_artifact_input_order_is_preserved_and_never_mutated() -> None:
    raw = _load(EXAMPLE1_PLAN_PATH)
    before = copy.deepcopy(raw)
    parsed = parse_robot_operation_plan(raw)
    assert raw == before
    assert list(parsed.operations) == [
        operation["operation_id"] for operation in before["operations"]
    ]
    assert parsed.execution_layers == tuple(
        tuple(layer) for layer in before["execution_layers"]
    )
    assert [
        [(entry.slot, entry.operation_id, entry.robot_id) for entry in slot]
        for slot in parsed.planned_schedule
    ] == [
        [(slot, entry["operation_id"], entry["robot_id"]) for entry in entries]
        for slot, entries in enumerate(before["planned_schedule"])
    ]


# ---------------------------------------------------------------------------
# Synthetic happy paths.
# ---------------------------------------------------------------------------


def test_synthetic_plan_parses_without_a_source_plan() -> None:
    """A self-contained payload parses with no second plan document."""
    parsed = parse_robot_operation_plan(_default_plan())
    assert list(parsed.operations) == [
        "a-pick",
        "a-place",
        "b-pick",
        "b-place",
    ]
    assert parsed.dependencies[1].predecessor_operation_id == "a-place"
    assert isinstance(parsed.planned_schedule[0][0], ScheduledOperation)
    assert parsed.planned_schedule[0][0].slot == 0


def test_supporting_robot_plan_is_accepted() -> None:
    """A plan whose task carries a supporting robot still parses formally."""
    operation = _operation(
        "a-pick",
        "task-mid",
        "PICK",
        "piper_a",
        participants=["piper_a", "piper_b"],
        role_bindings={"executor": "piper_a", "supporting": "piper_b"},
        target={"physical_instance_id": "brick-mid"},
        resources=["robot:piper_a", "robot:piper_b", "part:brick-mid"],
    )
    parsed = parse_robot_operation_plan(_one(operation))
    assert parsed.operations["a-pick"].role_bindings["supporting"] == "piper_b"


# ---------------------------------------------------------------------------
# Collaborating skills: HANDOVER, PLACE_UP, SUPPORT_TOP single payloads.
# ---------------------------------------------------------------------------


_HANDOVER_PARAMETERS = {
    "grasp_profile_id": "brick-side-grasp-v1",
    "co_hold_timeout_s": 10.0,
    "follow_samples": 4,
}


def _handover_operation(**overrides):
    operation = _operation(
        "handover",
        "task-mid",
        "HANDOVER",
        "piper_b",
        participants=["piper_a", "piper_b"],
        role_bindings={
            "executor": "piper_b",
            "donor": "piper_a",
            "receiver": "piper_b",
        },
        target={
            "physical_instance_id": "brick-mid",
            "donor_robot_id": "piper_a",
            "receiver_robot_id": "piper_b",
        },
        parameters=dict(_HANDOVER_PARAMETERS),
        resources=["robot:piper_a", "robot:piper_b", "part:brick-mid"],
    )
    operation.update(overrides)
    return operation


def _place_up_operation(**overrides):
    operation = _operation(
        "place-up",
        "task-mid",
        "PLACE_UP",
        "piper_b",
        participants=["piper_b", "piper_a"],
        role_bindings={"executor": "piper_b", "supporting": "piper_a"},
        target={
            "connection_id": "conn-mid-top",
            "upper_physical_instance_id": "brick-top",
            "lower_physical_instance_id": "brick-mid",
            "support_robot_id": "piper_a",
        },
        resources=[
            "robot:piper_b",
            "robot:piper_a",
            "connection:conn-mid-top",
            "part:brick-top",
            "part:brick-mid",
        ],
    )
    operation.update(overrides)
    return operation


def _support_top_operation(**overrides):
    operation = _operation(
        "support-top",
        "task-top",
        "SUPPORT_TOP",
        "piper_a",
        target={
            "connection_id": "conn-mid-top",
            "upper_physical_instance_id": "brick-top",
            "lower_physical_instance_id": "brick-mid",
            "placement_robot_id": "piper_b",
        },
        resources=["robot:piper_a", "connection:conn-mid-top", "part:brick-top"],
    )
    operation.update(overrides)
    return operation


def _drop_target_key(key):
    def mutate(operation):
        operation["target"].pop(key)
        return operation

    return mutate


def test_handover_plan_is_accepted() -> None:
    """A HANDOVER payload binds the receiver as executor and names both roles."""
    parsed = parse_robot_operation_plan(_one(_handover_operation()))
    operation = parsed.operations["handover"]
    assert operation.executor_robot_id == "piper_b"
    assert operation.target.values == {
        "physical_instance_id": "brick-mid",
        "donor_robot_id": "piper_a",
        "receiver_robot_id": "piper_b",
    }
    assert operation.role_bindings == {
        "executor": "piper_b",
        "donor": "piper_a",
        "receiver": "piper_b",
    }
    assert operation.parameters == _HANDOVER_PARAMETERS
    assert operation.resources == (
        "robot:piper_a",
        "robot:piper_b",
        "part:brick-mid",
    )


def test_place_up_plan_is_accepted() -> None:
    """PLACE_UP accepts the inserting executor plus the stabilising support."""
    parsed = parse_robot_operation_plan(_one(_place_up_operation()))
    operation = parsed.operations["place-up"]
    assert operation.executor_robot_id == "piper_b"
    assert operation.participants == ("piper_b", "piper_a")
    assert operation.target.values["support_robot_id"] == "piper_a"
    assert operation.resources == (
        "robot:piper_b",
        "robot:piper_a",
        "connection:conn-mid-top",
        "part:brick-top",
        "part:brick-mid",
    )


def test_support_top_plan_is_accepted() -> None:
    """SUPPORT_TOP lists only its executor; the lower part is target metadata."""
    parsed = parse_robot_operation_plan(_one(_support_top_operation()))
    operation = parsed.operations["support-top"]
    assert operation.participants == ("piper_a",)
    assert operation.target.values["placement_robot_id"] == "piper_b"
    assert operation.target.values["lower_physical_instance_id"] == "brick-mid"
    assert operation.resources == (
        "robot:piper_a",
        "connection:conn-mid-top",
        "part:brick-top",
    )


@pytest.mark.parametrize(
    ("build", "mutate", "match"),
    [
        pytest.param(
            _handover_operation,
            _drop_target_key("receiver_robot_id"),
            "keys mismatch",
            id="handover-target-missing-receiver-key",
        ),
        pytest.param(
            _handover_operation,
            lambda op: op["target"].update({"donor_robot_id": "piper_b"}),
            "different robots",
            id="handover-donor-equals-receiver",
        ),
        pytest.param(
            _handover_operation,
            lambda op: op["role_bindings"].update(
                {"donor": "piper_b", "receiver": "piper_a"}
            ),
            "target roles",
            id="handover-role-bindings-are-not-the-target-roles",
        ),
        pytest.param(
            _handover_operation,
            lambda op: op["parameters"].update({"follow_samples": 2}),
            "follow_samples",
            id="handover-follow-samples-below-three",
        ),
        pytest.param(
            _place_up_operation,
            lambda op: op["target"].update({"support_robot_id": "piper_b"}),
            "collaborating robot",
            id="place-up-support-robot-is-the-executor",
        ),
        pytest.param(
            _place_up_operation,
            lambda op: op.update(
                {
                    "participants": ["piper_b"],
                    "role_bindings": {"executor": "piper_b"},
                }
            ),
            "support robot",
            id="place-up-participants-omit-the-support-robot",
        ),
        pytest.param(
            _place_up_operation,
            lambda op: op["resources"].remove("part:brick-mid"),
            "missing",
            id="place-up-lower-part-resource-is-required",
        ),
        pytest.param(
            _support_top_operation,
            lambda op: op["target"].update({"placement_robot_id": "piper_a"}),
            "collaborating robot",
            id="support-top-placement-robot-is-the-executor",
        ),
        pytest.param(
            _support_top_operation,
            lambda op: op.update(
                {
                    "participants": ["piper_a", "piper_b"],
                    "role_bindings": {
                        "executor": "piper_a",
                        "placement": "piper_b",
                    },
                    "resources": [
                        "robot:piper_a",
                        "robot:piper_b",
                        "connection:conn-mid-top",
                        "part:brick-top",
                    ],
                }
            ),
            "placement robot",
            id="support-top-participants-include-the-placement-robot",
        ),
        pytest.param(
            _support_top_operation,
            lambda op: op["resources"].append("part:brick-mid"),
            "not referenced",
            id="support-top-lower-part-resource-is-rejected",
        ),
    ],
)
def test_rejects_collaborating_skill_violations(build, mutate, match) -> None:
    """Each collaborating skill rejects its own shape, role, or resource gap."""
    operation = build()
    mutate(operation)
    with pytest.raises(RobotOperationPlanError, match=match):
        parse_robot_operation_plan(_one(operation))


def test_rejects_pair_connection_inconsistent_across_collaborating_skills() -> None:
    """One connection id may not label two different placement pairs."""
    support = _support_top_operation()
    place = _operation(
        "place-down",
        "task-top",
        "PLACE_DOWN",
        "piper_b",
        target=_place("conn-mid-top", "brick-mid", "brick-base"),
        resources=[
            "robot:piper_b",
            "connection:conn-mid-top",
            "part:brick-mid",
            "part:brick-base",
        ],
    )
    raw = _plan(
        [support, place],
        [["support-top"], ["place-down"]],
        _schedule(
            ({"operation_id": "support-top", "robot_id": "piper_a"},),
            ({"operation_id": "place-down", "robot_id": "piper_b"},),
        ),
    )
    with pytest.raises(RobotOperationPlanError, match="bound to both"):
        parse_robot_operation_plan(raw)


# ---------------------------------------------------------------------------
# Identity and ground-ability.
# ---------------------------------------------------------------------------


def test_rejects_empty_robots() -> None:
    """A plan without any robot cannot ground an executor or a participant."""
    raw = _default_plan(robots=())
    with pytest.raises(RobotOperationPlanError, match="robots"):
        parse_robot_operation_plan(raw)


def test_rejects_duplicate_robots() -> None:
    """A robot listed twice is rejected instead of silently deduplicated."""
    raw = _default_plan(robots=("piper_a", "piper_a"))
    with pytest.raises(RobotOperationPlanError, match="duplicates"):
        parse_robot_operation_plan(raw)


def test_rejects_connection_bound_to_two_pairs() -> None:
    first = _operation(
        "p1",
        "task-mid",
        "PLACE_DOWN",
        "piper_a",
        target=_place("conn-shared", "brick-mid", "brick-base"),
        resources=[
            "robot:piper_a",
            "connection:conn-shared",
            "part:brick-mid",
            "part:brick-base",
        ],
    )
    second = _operation(
        "p2",
        "task-top",
        "PLACE_DOWN",
        "piper_b",
        target=_place("conn-shared", "brick-top", "brick-mid"),
        resources=[
            "robot:piper_b",
            "connection:conn-shared",
            "part:brick-top",
            "part:brick-mid",
        ],
    )
    raw = _plan(
        [first, second],
        [["p1"], ["p2"]],
        _schedule(
            ({"operation_id": "p1", "robot_id": "piper_a"},),
            ({"operation_id": "p2", "robot_id": "piper_b"},),
        ),
    )
    with pytest.raises(RobotOperationPlanError, match="bound to both"):
        parse_robot_operation_plan(raw)


def test_rejects_pair_labelled_by_two_connection_ids() -> None:
    first = _operation(
        "p1",
        "task-mid",
        "PLACE_DOWN",
        "piper_a",
        target=_place("conn-one", "brick-mid", "brick-base"),
        resources=[
            "robot:piper_a",
            "connection:conn-one",
            "part:brick-mid",
            "part:brick-base",
        ],
    )
    second = _operation(
        "p2",
        "task-mid",
        "PLACE_DOWN",
        "piper_a",
        target=_place("conn-two", "brick-mid", "brick-base"),
        resources=[
            "robot:piper_a",
            "connection:conn-two",
            "part:brick-mid",
            "part:brick-base",
        ],
    )
    raw = _plan(
        [first, second],
        [["p1"], ["p2"]],
        _schedule(
            ({"operation_id": "p1", "robot_id": "piper_a"},),
            ({"operation_id": "p2", "robot_id": "piper_a"},),
        ),
    )
    with pytest.raises(RobotOperationPlanError, match="labelled"):
        parse_robot_operation_plan(raw)


def test_rejects_ungrounded_connection() -> None:
    """An opaque connection cannot be introduced outside the frozen skills."""
    operation = _operation(
        "t1",
        "task-mid",
        "TRANSIT",
        "piper_a",
        target=_transit(
            _ref("robot_current"),
            _ref("connection", connection="conn-nowhere"),
        ),
        resources=["robot:piper_a"],
    )
    with pytest.raises(RobotOperationPlanError, match="unsupported"):
        parse_robot_operation_plan(_one(operation))
# ---------------------------------------------------------------------------
# WAIT is a passive single-robot observation.
# ---------------------------------------------------------------------------


def test_wait_may_run_on_a_declared_robot() -> None:
    """WAIT is a passive observation: no target, robot still declared."""
    operation = _operation(
        "w1", "task-mid", "WAIT", "piper_b", target=None, resources=["robot:piper_b"]
    )
    parsed = parse_robot_operation_plan(_one(operation))
    assert parsed.operations["w1"].executor_robot_id == "piper_b"
    assert parsed.operations["w1"].target is None


def test_rejects_wait_with_a_target() -> None:
    operation = _operation(
        "w1",
        "task-mid",
        "WAIT",
        "piper_b",
        target={"physical_instance_id": "brick-mid"},
    )
    with pytest.raises(RobotOperationPlanError, match="null"):
        parse_robot_operation_plan(_one(operation))


# ---------------------------------------------------------------------------
# Frozen shapes, resources, coverage, ordering.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda op: op.update({"skill": "TELEPORT"}),
            id="unknown-skill",
        ),
        pytest.param(
            lambda op: op["target"].update({"unexpected": 1}),
            id="transit-extra-target-key",
        ),
        pytest.param(
            lambda op: op["target"].update(
                {"to": _ref("part", part=None)}
            ),
            id="part-ref-without-id",
        ),
        pytest.param(
            lambda op: op["target"].update({"from": _ref("unknown_kind")}),
            id="unsupported-ref-kind",
        ),
        pytest.param(
            lambda op: op["target"].update(
                {"from": _ref("robot_current", part="brick-mid")}
            ),
            id="robot-current-with-part",
        ),
        pytest.param(
            lambda op: op["target"].update({"to": _ref("robot_home")}),
            id="robot-home-without-robot",
        ),
        pytest.param(
            lambda op: op["target"].update(
                {"to": _ref("robot_home", robot="piper_b")}
            ),
            id="robot-home-of-another-robot",
        ),
        pytest.param(
            lambda op: op.update({"parameters": []}),
            id="parameters-not-object",
        ),
        pytest.param(
            lambda op: op.update({"resources": ["robot:piper_a", "tool:hammer"]}),
            id="unknown-resource-kind",
        ),
        pytest.param(
            lambda op: op.update({"resources": ["robot:piper_b"]}),
            id="robot-resource-outside-participants",
        ),
        pytest.param(
            lambda op: op.update({"resources": ["robot:piper_a", "part:brick-top"]}),
            id="part-resource-not-in-target",
        ),
    ],
)
def test_single_operation_shape_and_resource_violations_are_rejected(mutate) -> None:
    operation = _operation(
        "a-transit",
        "task-mid",
        "TRANSIT",
        "piper_a",
        target=_transit(_ref("robot_current"), _ref("part", part="brick-mid")),
        resources=["robot:piper_a"],
    )
    mutate(operation)
    with pytest.raises(RobotOperationPlanError):
        parse_robot_operation_plan(_one(operation))


def test_rejects_connection_resource_not_in_target() -> None:
    operation = _operation(
        "a-pick",
        "task-mid",
        "PICK",
        "piper_a",
        target={"physical_instance_id": "brick-mid"},
        resources=["robot:piper_a", "part:brick-mid", "connection:conn-x"],
    )
    with pytest.raises(RobotOperationPlanError, match="connection"):
        parse_robot_operation_plan(_one(operation))


def test_rejects_participants_outside_the_plan() -> None:
    operation = _operation(
        "w1",
        "task-mid",
        "WAIT",
        "piper_a",
        participants=["piper_a", "piper_z"],
        role_bindings={"executor": "piper_a", "observer": "piper_z"},
    )
    with pytest.raises(RobotOperationPlanError, match="participants"):
        parse_robot_operation_plan(_one(operation))


def test_rejects_incomplete_role_bindings() -> None:
    operation = _operation(
        "w1",
        "task-mid",
        "WAIT",
        "piper_a",
        participants=["piper_a", "piper_b"],
        role_bindings={"executor": "piper_a"},
        resources=["robot:piper_a", "robot:piper_b"],
    )
    with pytest.raises(RobotOperationPlanError, match="role_bindings"):
        parse_robot_operation_plan(_one(operation))


def test_rejects_same_layer_resource_conflict() -> None:
    first = _operation(
        "a-pick",
        "task-mid",
        "PICK",
        "piper_a",
        target={"physical_instance_id": "brick-mid"},
        resources=["robot:piper_a", "part:brick-mid"],
    )
    second = _operation(
        "b-place",
        "task-top",
        "PLACE_DOWN",
        "piper_b",
        target=_place("conn-top-mid", "brick-top", "brick-mid"),
        resources=[
            "robot:piper_b",
            "connection:conn-top-mid",
            "part:brick-top",
            "part:brick-mid",
        ],
    )
    raw = _plan(
        [first, second],
        [["a-pick", "b-place"]],
        _schedule(
            (
                {"operation_id": "a-pick", "robot_id": "piper_a"},
                {"operation_id": "b-place", "robot_id": "piper_b"},
            ),
        ),
    )
    with pytest.raises(RobotOperationPlanError, match="resource conflict"):
        parse_robot_operation_plan(raw)


def test_rejects_layer_coverage_gap() -> None:
    raw = _default_plan(
        layers=(("a-transit",), ("a-pick",), ("a-place",), ("b-transit",))
    )
    with pytest.raises(RobotOperationPlanError, match="execution_layers"):
        parse_robot_operation_plan(raw)


def test_rejects_operation_listed_twice_in_layers() -> None:
    raw = _default_plan(
        layers=(
            ("a-transit",),
            ("a-transit", "a-pick", "a-place", "b-transit", "b-place"),
        )
    )
    with pytest.raises(RobotOperationPlanError, match="duplicated"):
        parse_robot_operation_plan(raw)


def test_rejects_schedule_coverage_gap() -> None:
    raw = _default_plan(schedule=_DEFAULT_SCHEDULE[:3])
    with pytest.raises(RobotOperationPlanError, match="planned_schedule"):
        parse_robot_operation_plan(raw)


def test_rejects_schedule_robot_mismatch() -> None:
    raw = _default_plan()
    raw["planned_schedule"][0] = [{"operation_id": "a-pick", "robot_id": "piper_b"}]
    with pytest.raises(RobotOperationPlanError, match="executor"):
        parse_robot_operation_plan(raw)


def test_rejects_unknown_dependency_operation() -> None:
    raw = _default_plan()
    raw["dependencies"][0]["successor_operation_id"] = "absent"
    with pytest.raises(RobotOperationPlanError, match="dependency"):
        parse_robot_operation_plan(raw)


def test_rejects_dependency_without_layer_order() -> None:
    raw = _default_plan(
        layers=(("a-place", "a-pick"), ("b-pick",), ("b-place",))
    )
    with pytest.raises(RobotOperationPlanError, match="order"):
        parse_robot_operation_plan(raw)


def test_rejects_dependency_without_schedule_slot_order() -> None:
    raw = _default_plan(
        schedule=_schedule(
            ({"operation_id": "a-place", "robot_id": "piper_a"},),
            ({"operation_id": "a-pick", "robot_id": "piper_a"},),
            ({"operation_id": "b-pick", "robot_id": "piper_b"},),
            ({"operation_id": "b-place", "robot_id": "piper_b"},),
        )
    )
    with pytest.raises(RobotOperationPlanError, match="order"):
        parse_robot_operation_plan(raw)


def test_rejects_evidence_label_not_declared_by_operation() -> None:
    raw = _default_plan(
        evidence=(),
    )
    raw["evidence_requirements"] = [
        {"operation_id": "a-pick", "evidence": "holding_confirmed"},
    ]
    parsed = parse_robot_operation_plan(raw)
    assert parsed.evidence_requirements == (("a-pick", "holding_confirmed"),)
    raw["evidence_requirements"] = [
        {"operation_id": "a-pick", "evidence": "never_declared"},
    ]
    with pytest.raises(RobotOperationPlanError, match="declared"):
        parse_robot_operation_plan(raw)


def test_rejects_evidence_for_unknown_operation() -> None:
    raw = _default_plan()
    raw["evidence_requirements"] = [{"operation_id": "absent", "evidence": "x"}]
    with pytest.raises(RobotOperationPlanError, match="unknown"):
        parse_robot_operation_plan(raw)


# ---------------------------------------------------------------------------
# source_plan typing and immutability of failing input.
# ---------------------------------------------------------------------------
def test_rejected_input_is_never_mutated_or_reordered() -> None:
    raw = _default_plan()
    raw["operations"][2]["target"]["lower_physical_instance_id"] = "brick-top"
    raw["operations"][2]["resources"] = [
        "robot:piper_a",
        "connection:conn-mid-base",
        "part:brick-mid",
        "part:brick-top",
    ]
    before = copy.deepcopy(raw)
    with pytest.raises(RobotOperationPlanError):
        parse_robot_operation_plan(raw)
    assert raw == before
