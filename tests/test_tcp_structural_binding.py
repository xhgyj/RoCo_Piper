"""Offline tests for the R3 TCP contract and structural binding.

Everything here is pure: URDF/config bytes, exact plan/task bytes, AST scans
of the two new modules, and the recorded audit artifact. No simulator, ROS,
GPU, git, or motion is touched, and no metric TCP is ever claimed.
"""

from __future__ import annotations

import ast
import json
from dataclasses import astuple
from hashlib import sha256
from pathlib import Path

import pytest

from rocobrick.stap_contract import parse_stap_plan
from rocobrick.structural_binding import (
    BindingDecision,
    StructuralBinding,
    bind_single_stap_task,
    binding_requires_tcp,
    canonical_json_digest,
)
from rocobrick.tcp_contract import (
    KIND_LEGACY_ASSUMPTION,
    KIND_LOCAL_CONFIG,
    KIND_PINNED_EXPLICIT_TRANSFORM,
    KIND_PINNED_GEOMETRY,
    KIND_PROJECT_CALIBRATION,
    TcpContract,
    TcpContractError,
    TcpStatus,
    UnresolvedTcpError,
    assess_piper_l_tcp,
    require_resolved_tcp,
)

ROOT = Path(__file__).parents[1]
ARTIFACT = ROOT / "artifacts/model-audit/piper_l_tcp_binding_contract.json"
MODEL_CONTRACT = ROOT / "artifacts/model-audit/piper_l_model_contract.json"
PLAN = ROOT / "tests/fixtures/assembly_plans/example1-compound-pick.json"
TASK_START = ROOT / "tasks/type1/example1/structure_start.json"
TASK_GOAL = ROOT / "tasks/type1/example1/structure_goal.json"
VENDOR_URDF = (
    ROOT / "isaac/vendor/agx_arm_sim/agx_arm_description/urdf/piper_l_gripper_d435.urdf"
)
ROBOT = "piper_0"

PINNED_HASH = "a" * 64
PINNED_SOURCE = "ros_ws/src/agx_arm_ros/src/agx_arm_description/agx_arm_urdf"
PINNED_SOURCE += "/piper_l/urdf/piper_l_with_gripper_description.xacro"
TCP_ITEM = {
    "id": "official-tool0",
    "kind": KIND_PINNED_EXPLICIT_TRANSFORM,
    "source": PINNED_SOURCE,
    "source_sha256": PINNED_HASH,
    "declared_tool_frame": True,
    "parent_frame": "link6",
    "child_frame": "tool0",
    "translation_m": [0.0, 0.0, 0.1],
    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
}
EVIDENCE_KINDS = (
    "pinned_explicit_transform",
    "pinned_geometry",
    "local_config",
    "historical_runtime",
    "legacy_assumption",
    "project_calibration",
)
FORBIDDEN_IMPORT_ROOTS = (
    "rclpy",
    "rospy",
    "sensor_msgs",
    "geometry_msgs",
    "moveit_msgs",
    "moveit",
    "tf2_ros",
    "pxr",
    "isaacsim",
    "omni",
    "tools",
    "numpy",
    "torch",
)


def evidence(*items: dict, pinned: dict | None = None) -> dict:
    """Build an evidence mapping from item dicts.

    Returns:
        An evidence mapping ready for ``assess_piper_l_tcp``.
    """
    return {
        "robot_model": "agilex_piper_l",
        "pinned_sources": {PINNED_SOURCE: PINNED_HASH} if pinned is None else pinned,
        "items": list(items),
    }


def item(**overrides: object) -> dict:
    """A declared pinned TCP item with fields overridden.

    Returns:
        The TCP_ITEM mapping with the given overrides applied.
    """
    return {**TCP_ITEM, **overrides}


# ---------------------------------------------------------------------------
# Pure transform and quaternion validation
# ---------------------------------------------------------------------------


def test_transform_accepts_exact_values_and_coerces_sequences():
    from rocobrick.tcp_contract import Transform

    transform = Transform(
        parent_frame="link6",
        child_frame="tool0",
        translation_m=[0.0, 0.0, 0.1],
        rotation_xyzw=[0.0, 0.0, 0.0, 1.0],
        source=PINNED_SOURCE,
        sha256=PINNED_HASH,
    )
    assert transform.translation_m == (0.0, 0.0, 0.1)
    assert transform.rotation_xyzw == (0.0, 0.0, 0.0, 1.0)
    assert transform.identity[0] == "link6"
    assert transform.identity[1] == "tool0"


def test_transform_is_frozen():
    from rocobrick.tcp_contract import Transform

    transform = Transform("link6", "tool0", (0, 0, 0.1), (0, 0, 0, 1), "s", PINNED_HASH)
    with pytest.raises(Exception):
        transform.parent_frame = "base_link"


@pytest.mark.parametrize(
    "translation",
    [(0.0, float("nan"), 0.0), (0.0, float("inf"), 0.0), (0.0, 0.0)],
)
def test_transform_rejects_non_finite_or_short_translation(translation):
    from rocobrick.tcp_contract import Transform

    with pytest.raises(TcpContractError, match="translation_m"):
        Transform("link6", "tool0", translation, (0, 0, 0, 1), "s", PINNED_HASH)


@pytest.mark.parametrize(
    "rotation",
    [
        (0.0, 0.0, 0.0, 1.1),
        (0.0, 0.0, 0.0, 0.5),
        (0.0, 0.0, 0.0, float("nan")),
        (0.0, 0.0, 0.0),
    ],
)
def test_transform_rejects_unnormalized_or_invalid_quaternion(rotation):
    from rocobrick.tcp_contract import Transform

    with pytest.raises(TcpContractError, match="rotation_xyzw"):
        Transform("link6", "tool0", (0, 0, 0), rotation, "s", PINNED_HASH)


def test_transform_rejects_identical_frames_and_bad_digest():
    from rocobrick.tcp_contract import Transform

    with pytest.raises(TcpContractError, match="distinct"):
        Transform("link6", "link6", (0, 0, 0), (0, 0, 0, 1), "s", PINNED_HASH)
    with pytest.raises(TcpContractError, match="SHA-256"):
        Transform("link6", "tool0", (0, 0, 0), (0, 0, 0, 1), "s", "XYZ")


def test_transform_rejects_non_finite_quaternion_components():
    from rocobrick.tcp_contract import Transform

    with pytest.raises(TcpContractError, match="finite"):
        Transform(
            "link6", "tool0", (0, 0, 0), (0, 0, float("inf"), 0.0), "s", PINNED_HASH
        )


def test_transform_accepts_unit_quaternion_around_x():
    import math

    from rocobrick.tcp_contract import Transform

    half = math.sqrt(0.5)
    transform = Transform(
        "gripper_base",
        "gripper_link1",
        (0.0, 0.0, 0.1358),
        (half, 0.0, 0.0, half),
        "urdf",
        PINNED_HASH,
    )
    assert transform.rotation_xyzw == (half, 0.0, 0.0, half)


# ---------------------------------------------------------------------------
# Strict TCP resolution rules
# ---------------------------------------------------------------------------


def test_status_enum_members_are_strings():
    assert {member.value for member in TcpStatus} == {
        "RESOLVED",
        "UNRESOLVED",
        "CONFLICT",
    }
    assert str(TcpStatus.UNRESOLVED) == "UNRESOLVED"


def test_empty_evidence_is_unresolved_and_requires_raises():
    contract = assess_piper_l_tcp(evidence())
    assert contract.status is TcpStatus.UNRESOLVED
    assert contract.transform is None
    assert contract.blockers
    with pytest.raises(UnresolvedTcpError):
        require_resolved_tcp(contract)


def test_local_config_offset_can_never_resolve():
    local = item(
        id="local-offset",
        kind=KIND_LOCAL_CONFIG,
        source="config/user_config.json",
        translation_m=[0.0, 0.0, -0.0053],
    )
    contract = assess_piper_l_tcp(evidence(local))
    assert contract.status is TcpStatus.UNRESOLVED
    assert contract.transform is None
    assert contract.evidence == ("local-offset",)
    with pytest.raises(UnresolvedTcpError):
        require_resolved_tcp(contract)


def test_historical_runtime_value_can_never_resolve():
    historical = item(
        id="run-tcp",
        kind="historical_runtime",
        source="runs/x/events.jsonl",
        translation_m=[0.0, 0.0, 0.1358],
    )
    contract = assess_piper_l_tcp(evidence(historical))
    assert contract.status is TcpStatus.UNRESOLVED


def test_legacy_assumption_can_never_resolve():
    legacy = item(
        id="legacy-01358",
        kind=KIND_LEGACY_ASSUMPTION,
        source="run/pregrasp_v2.py",
        translation_m=[0.0, 0.0, 0.1358],
    )
    contract = assess_piper_l_tcp(evidence(legacy))
    assert contract.status is TcpStatus.UNRESOLVED


def test_pinned_geometry_without_declaration_can_never_resolve():
    geometry = item(
        id="finger-origin", kind=KIND_PINNED_GEOMETRY, declared_tool_frame=False
    )
    contract = assess_piper_l_tcp(evidence(geometry))
    assert contract.status is TcpStatus.UNRESOLVED
    assert contract.candidate_frames == ("tool0",)


def test_single_pinned_explicit_designation_resolves():
    contract = assess_piper_l_tcp(evidence(item()))
    assert contract.status is TcpStatus.RESOLVED
    assert contract.transform is not None
    assert contract.transform.child_frame == "tool0"
    assert contract.transform.sha256 == PINNED_HASH
    assert contract.blockers == ()
    assert require_resolved_tcp(contract) is contract.transform


def test_duplicate_identical_pinned_designations_resolve_once():
    contract = assess_piper_l_tcp(evidence(item(), item()))
    assert contract.status is TcpStatus.RESOLVED
    assert contract.transform is not None


def test_source_hash_mismatch_blocks_resolution():
    contract = assess_piper_l_tcp(evidence(item(), pinned={PINNED_SOURCE: "b" * 64}))
    assert contract.status is TcpStatus.UNRESOLVED
    assert contract.transform is None
    assert any("digest does not match" in blocker for blocker in contract.blockers)


def test_conflicting_official_designations_conflict():
    first = item(id="tool-a")
    second = item(id="tool-b", translation_m=[0.0, 0.0, 0.2])
    contract = assess_piper_l_tcp(evidence(first, second))
    assert contract.status is TcpStatus.CONFLICT
    assert contract.transform is None
    assert any("disagree" in blocker for blocker in contract.blockers)
    with pytest.raises(UnresolvedTcpError):
        require_resolved_tcp(contract)


def test_local_claim_does_not_veto_official_resolution():
    local = item(
        id="local-offset",
        kind=KIND_LOCAL_CONFIG,
        source="config/user_config.json",
        translation_m=[9.0, 9.0, 9.0],
    )
    contract = assess_piper_l_tcp(evidence(item(), local))
    assert contract.status is TcpStatus.RESOLVED


def test_unlisted_pinned_source_blocks_resolution():
    contract = assess_piper_l_tcp(evidence(item(), pinned={}))
    assert contract.status is TcpStatus.UNRESOLVED
    assert contract.transform is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"kind": " gossip"},
        {"declared_tool_frame": "yes"},
        {"id": ""},
        {"source_sha256": "nope"},
    ],
)
def test_malformed_evidence_items_raise(overrides):
    with pytest.raises(TcpContractError):
        assess_piper_l_tcp(evidence(item(**overrides)))


@pytest.mark.parametrize(
    "overrides",
    [{"translation_m": [0, 0]}, {"rotation_xyzw": [0, 0, 0, 2]}, {"translation_m": ()}],
)
def test_invalid_transform_values_block_instead_of_raise(overrides):
    contract = assess_piper_l_tcp(evidence(item(**overrides)))
    assert contract.status is TcpStatus.UNRESOLVED
    assert contract.transform is None
    assert any("invalid transform" in blocker for blocker in contract.blockers)


def test_missing_robot_model_or_items_raise():
    with pytest.raises(TcpContractError):
        assess_piper_l_tcp({"pinned_sources": {}, "items": []})
    with pytest.raises(TcpContractError):
        assess_piper_l_tcp({"robot_model": "agilex_piper_l", "items": "nope"})


def test_contract_invariants():
    resolved = assess_piper_l_tcp(evidence(item()))
    with pytest.raises(TcpContractError):
        TcpContract(
            robot_model="m", status=TcpStatus.RESOLVED, transform=None,
            candidate_frames=(), evidence=(), blockers=(),
        )
    with pytest.raises(TcpContractError):
        TcpContract(
            robot_model="m", status=TcpStatus.UNRESOLVED, transform=resolved.transform,
            candidate_frames=(), evidence=(), blockers=("b",),
        )
    with pytest.raises(TcpContractError):
        TcpContract(
            robot_model="m", status=TcpStatus.UNRESOLVED, transform=None,
            candidate_frames=(), evidence=(), blockers=(),
        )
    with pytest.raises(TcpContractError):
        TcpContract(
            robot_model="m", status=TcpStatus.RESOLVED, transform=resolved.transform,
            candidate_frames=(), evidence=(), blockers=("b",),
        )


# ---------------------------------------------------------------------------
# Exact example1 structural binding
# ---------------------------------------------------------------------------


def task_bytes() -> tuple[bytes, bytes]:
    return TASK_START.read_bytes(), TASK_GOAL.read_bytes()


def test_canonical_json_digest_matches_plan_source_hashes():
    start, goal = task_bytes()
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert canonical_json_digest(json.loads(start.decode("utf-8"))) == plan["source"][
        "start_sha256"
    ]
    assert canonical_json_digest(json.loads(goal.decode("utf-8"))) == plan["source"][
        "goal_sha256"
    ]


def test_example1_binding_succeeds_with_exact_bytes():
    start, goal = task_bytes()
    plan_bytes = PLAN.read_bytes()
    decision = bind_single_stap_task(
        plan_bytes=plan_bytes, task_start_bytes=start, task_goal_bytes=goal,
        robot_id=ROBOT,
    )
    assert decision.accepted, decision.reason
    binding = decision.binding
    assert binding is not None
    assert binding.plan_id == "assembly-plan-25a5d5cc586a1e5c65f2141d"
    assert binding.task_id == "assemble-768de5a17827920c48d4"
    assert binding.physical_instance_id == "brick-768de5a17827920c48d4"
    assert binding.robot_id == ROBOT
    assert binding.brick_id == 2
    assert binding.source_goal_key == "2"
    assert binding.grid_xyz == (0, 0, 1)
    assert binding.orientation == 0
    assert binding.color == "Blue"
    assert binding.motion_geometry == "omitted"
    assert binding.plan_sha256 == sha256(plan_bytes).hexdigest()
    assert binding.task_start_sha256 == sha256(start).hexdigest()
    assert binding.task_goal_sha256 == sha256(goal).hexdigest()
    details = dict(decision.details)
    assert details["input_sha256_before"] == details["input_sha256_after"]
    assert details["motion_geometry"] == "omitted"


def test_example1_binding_matches_authoritative_parser():
    start, goal = task_bytes()
    decision = bind_single_stap_task(
        plan_bytes=PLAN.read_bytes(), task_start_bytes=start, task_goal_bytes=goal,
        robot_id=ROBOT,
    )
    plan = parse_stap_plan(json.loads(PLAN.read_text(encoding="utf-8")))
    (task,) = plan.tasks.values()
    part = plan.goal_parts[task.physical_instance_id]
    binding = decision.binding
    assert binding is not None
    assert binding.task_id == task.task_id == "assemble-768de5a17827920c48d4"
    assert binding.robot_id == task.primary_robot == ROBOT
    assert binding.grid_xyz == (part.x, part.y, part.z)
    assert binding.source_goal_key == part.source_goal_key
    assert binding.brick_id == part.brick_id
    assert binding.orientation == part.orientation_quarter_turns


def test_input_bytes_are_unchanged_by_binding():
    start, goal = task_bytes()
    plan_bytes = PLAN.read_bytes()
    before = (plan_bytes, start, goal)
    bind_single_stap_task(
        plan_bytes=plan_bytes, task_start_bytes=start, task_goal_bytes=goal,
        robot_id=ROBOT,
    )
    assert (plan_bytes, start, goal) == before
    assert PLAN.read_bytes() == before[0]


def test_binding_carries_no_metric_pose():
    start, goal = task_bytes()
    binding = bind_single_stap_task(
        plan_bytes=PLAN.read_bytes(), task_start_bytes=start, task_goal_bytes=goal,
        robot_id=ROBOT,
    ).binding
    assert binding is not None
    for value in astuple(binding):
        if isinstance(value, tuple):
            assert all(isinstance(item, int) for item in value)
        else:
            assert isinstance(value, (str, int))
    assert not any(
        field in type(binding).__dataclass_fields__ for field in ("position_m", "pose")
    )


def test_wrong_robot_is_rejected():
    start, goal = task_bytes()
    decision = bind_single_stap_task(
        plan_bytes=PLAN.read_bytes(), task_start_bytes=start, task_goal_bytes=goal,
        robot_id="piper_1",
    )
    assert not decision.accepted
    assert decision.binding is None
    assert "piper_1" in decision.reason


def test_undeclared_robot_is_rejected():
    start, goal = task_bytes()
    decision = bind_single_stap_task(
        plan_bytes=PLAN.read_bytes(), task_start_bytes=start, task_goal_bytes=goal,
        robot_id="piper_9",
    )
    assert not decision.accepted


def test_task_digest_mismatch_is_rejected():
    start, goal = task_bytes()
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    plan["source"]["goal_sha256"] = "0" * 64
    decision = bind_single_stap_task(
        plan_bytes=json.dumps(plan).encode("utf-8"), task_start_bytes=start,
        task_goal_bytes=goal, robot_id=ROBOT,
    )
    assert not decision.accepted
    assert "canonical digests" in decision.reason


def two_task_plan_bytes() -> bytes:
    moving = [
        {
            "physical_instance_id": f"brick-{index}",
            "source_goal_key": str(index),
            "brick_id": 2,
            "x": 0,
            "y": 0,
            "z": index,
            "ori": 0,
            "color": "Red",
            "task_id": f"assemble-{index}",
            "skill_type": "PLACE_DOWN",
            "primary_robot": "piper_1",
            "supporting_robot": None,
        }
        for index in (7, 9)
    ]
    value = {
        "schema": "assembly-plan/v2",
        "plan_id": "pending",
        "source": {
            "goal_sha256": "a" * 64,
            "start_sha256": "b" * 64,
        },
        "robots": [{"robot_id": "piper_1"}],
        "structural_semantics": {
            "coordinate_system": "grid",
            "position_unit": "stud/layer grid units",
            "orientation": {
                "encoding": "quarter_turn",
                "values": [0, 1],
                "one_means": "axis swap",
            },
            "motion_geometry": "omitted",
        },
        "goal_structure": {"instances": moving},
        "initial_structure": {"facts": []},
        "parts": moving,
        "dependencies": [["assemble-7", "assemble-9"]],
        "initial_dependencies": [],
        "allocation": {"piper_1": ["assemble-7", "assemble-9"]},
        "execution_layers": [["assemble-7"], ["assemble-9"]],
        "planned_schedule": [
            [{"robot": "piper_1", "task": "assemble-7"}],
            [{"robot": "piper_1", "task": "assemble-9"}],
        ],
        "specification": {
            "type": "hierarchical-ltl",
            "root_formula": "<> p1a",
            "hierarchy": [{"p0": "<> p1a"}, {"p1a": "<> assemble1"}],
        },
        "diagnostics": {},
    }
    material = {
        "source": value["source"],
        "robots": value["robots"],
        "parts": value["parts"],
        "dependencies": value["dependencies"],
        "initial_dependencies": value["initial_dependencies"],
        "allocation": value["allocation"],
        "execution_layers": value["execution_layers"],
        "planned_schedule": value["planned_schedule"],
        "hierarchy": value["specification"]["hierarchy"],
    }
    value["plan_id"] = "assembly-plan-" + sha256(
        json.dumps(
            material, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()[:24]
    return json.dumps(value).encode("utf-8")


def test_multiple_pending_tasks_are_rejected():
    start_bytes, goal_bytes = task_bytes()
    payload = json.loads(two_task_plan_bytes().decode("utf-8"))
    payload["source"] = {
        "start_sha256": canonical_json_digest(json.loads(start_bytes.decode("utf-8"))),
        "goal_sha256": canonical_json_digest(json.loads(goal_bytes.decode("utf-8"))),
    }
    material = {
        "source": payload["source"],
        "robots": payload["robots"],
        "parts": payload["parts"],
        "dependencies": payload["dependencies"],
        "initial_dependencies": payload["initial_dependencies"],
        "allocation": payload["allocation"],
        "execution_layers": payload["execution_layers"],
        "planned_schedule": payload["planned_schedule"],
        "hierarchy": payload["specification"]["hierarchy"],
    }
    payload["plan_id"] = "assembly-plan-" + sha256(
        json.dumps(
            material, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()[:24]
    decision = bind_single_stap_task(
        plan_bytes=json.dumps(payload).encode("utf-8"),
        task_start_bytes=start_bytes,
        task_goal_bytes=goal_bytes,
        robot_id="piper_1",
    )
    assert not decision.accepted
    assert "exactly one pending task" in decision.reason


def test_plan_failing_stap_validation_is_rejected():
    start, goal = task_bytes()
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    plan["plan_id"] = "tampered"
    decision = bind_single_stap_task(
        plan_bytes=json.dumps(plan).encode("utf-8"), task_start_bytes=start,
        task_goal_bytes=goal, robot_id=ROBOT,
    )
    assert not decision.accepted
    assert "STAP validation" in decision.reason


@pytest.mark.parametrize(
    "payload",
    [b"{", b"[1, 2]", b"\xff\xfe", b""],
)
def test_malformed_plan_bytes_are_rejected(payload):
    start, goal = task_bytes()
    decision = bind_single_stap_task(
        plan_bytes=payload, task_start_bytes=start, task_goal_bytes=goal,
        robot_id=ROBOT,
    )
    assert not decision.accepted


def test_malformed_or_altered_task_bytes_are_rejected():
    start, goal = task_bytes()
    tweaked = json.loads(goal.decode("utf-8"))
    tweaked["1"]["color"] = "Green"
    semantic_corruption = json.dumps(tweaked).encode("utf-8")
    for corrupt_start_bytes, corrupt_goal_bytes in (
        (b"not json", goal),
        (start, semantic_corruption),
    ):
        decision = bind_single_stap_task(
            plan_bytes=PLAN.read_bytes(), task_start_bytes=corrupt_start_bytes,
            task_goal_bytes=corrupt_goal_bytes, robot_id=ROBOT,
        )
        assert not decision.accepted


def test_whitespace_only_task_byte_changes_still_bind_canonically():
    start, goal = task_bytes()
    reformatted = json.dumps(
        json.loads(goal.decode("utf-8")), indent=4, sort_keys=True
    ).encode("utf-8")
    assert reformatted != goal
    decision = bind_single_stap_task(
        plan_bytes=PLAN.read_bytes(), task_start_bytes=start,
        task_goal_bytes=reformatted, robot_id=ROBOT,
    )
    assert decision.accepted, decision.reason
    assert decision.binding.task_goal_sha256 == sha256(reformatted).hexdigest()


def test_binding_requires_tcp_is_true_with_rationale():
    start, goal = task_bytes()
    binding = bind_single_stap_task(
        plan_bytes=PLAN.read_bytes(), task_start_bytes=start, task_goal_bytes=goal,
        robot_id=ROBOT,
    ).binding
    assert binding_requires_tcp(binding) is True
    assert binding_requires_tcp.__doc__ is not None
    assert "PLACE_DOWN" in binding_requires_tcp.__doc__
    with pytest.raises(TypeError):
        binding_requires_tcp("not a binding")


def test_structural_binding_validates_fields():
    valid = {
        "plan_id": "p", "task_id": "t", "physical_instance_id": "i",
        "brick_id": 2, "robot_id": "piper_0", "source_goal_key": "2",
        "grid_xyz": (0, 0, 1), "orientation": 0, "color": "Blue",
        "motion_geometry": "omitted", "plan_sha256": "a" * 64,
        "task_start_sha256": "b" * 64, "task_goal_sha256": "c" * 64,
    }
    assert StructuralBinding(**valid) is not None
    with pytest.raises(ValueError, match="motion_geometry"):
        StructuralBinding(**{**valid, "motion_geometry": "meters"})
    with pytest.raises(ValueError, match="orientation"):
        StructuralBinding(**{**valid, "orientation": 2})
    with pytest.raises(ValueError, match="grid_xyz"):
        StructuralBinding(**{**valid, "grid_xyz": (0, -1, 1)})
    with pytest.raises(ValueError, match="brick_id"):
        StructuralBinding(**{**valid, "brick_id": True})
    with pytest.raises(ValueError, match="SHA-256"):
        StructuralBinding(**{**valid, "plan_sha256": "zz"})


def test_binding_decision_invariants():
    with pytest.raises(ValueError):
        BindingDecision(accepted=True, reason="r", binding=None, details={})
    with pytest.raises(ValueError):
        BindingDecision(accepted=False, reason="r", binding=None, details="x")
    decision = BindingDecision(
        accepted=False, reason="r", binding=None, details={"a": 1}
    )
    assert decision.details == {"a": 1}


# ---------------------------------------------------------------------------
# Artifact truthfulness
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def model_contract() -> dict:
    return json.loads(MODEL_CONTRACT.read_text(encoding="utf-8"))


def test_artifact_declares_schema_and_offline_audit(artifact: dict):
    assert artifact["schema"] == "rocobrick-piper-l-tcp-binding-contract/v1"
    assert artifact["model"] == "agilex_piper_l"
    assert artifact["offline"] is True


def _artifact_source_path(original: str, artifact: dict) -> Path:
    """Resolve only declared audit inputs, including retired byte objects.

    Returns:
        The validated current file or immutable historical byte object.
    """
    declared = {
        entry["path"]
        for group in ("pinned_files", "local_files", "example1_files")
        for entry in artifact["inputs"][group]
    } | {entry["source"] for entry in artifact["evidence"]}
    if original not in declared:
        raise ValueError("undeclared provenance path")
    directory = ROOT / "tests/fixtures/historical_provenance"
    index = json.loads((directory / "index.json").read_text(encoding="utf-8"))
    if original in index:
        entry = index[original]
        path = directory / entry["object"]
        if path.parent != directory or path.name != entry["sha256"] + ".bin":
            raise ValueError("invalid provenance object path")
        if sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("provenance object digest mismatch")
        return path
    if original == "run/user_config.json":
        return ROOT / "tests/fixtures/run-config-older-copy.json"
    if original.startswith("runs/") or original == "run/pregrasp_v2.py":
        raise ValueError("historical provenance mapping missing")
    return ROOT / original


def test_provenance_resolver_rejects_undeclared_paths(artifact: dict):
    """Unknown provenance names must fail rather than escape the audit map."""
    with pytest.raises(ValueError, match="undeclared"):
        _artifact_source_path("../unknown", artifact)


def test_historical_evidence_hashes_match_archived_bytes(artifact: dict):
    """Keep historical source bytes independently verifiable after retirement."""
    for entry in artifact["evidence"]:
        if entry["source"].startswith("runs/"):
            assert sha256(
                _artifact_source_path(entry["source"], artifact).read_bytes()
            ).hexdigest() == entry["source_sha256"]


def test_artifact_input_hashes_match_disk(artifact: dict):
    recorded = (
        artifact["inputs"]["pinned_files"]
        + artifact["inputs"]["local_files"]
        + artifact["inputs"]["example1_files"]
    )
    assert recorded
    for entry in recorded:
        data = _artifact_source_path(entry["path"], artifact).read_bytes()
        digest = sha256(data).hexdigest()
        assert entry["sha256"] == digest


def test_artifact_commits_match_model_contract(artifact: dict, model_contract: dict):
    commits = artifact["inputs"]["repository_commits"]
    pinned = model_contract["provenance"]["pinned_repositories"]
    assert commits["agx_arm_ros"] == pinned["agx_arm_ros"]["commit"]
    assert commits["agx_arm_sim"] == pinned["agx_arm_sim"]["commit"]
    submodules = model_contract["provenance"]["submodules"]
    assert commits["agx_arm_urdf_submodule_pin"] == submodules[
        "agx_arm_urdf_in_agx_arm_ros"
    ]["pin"]
    assert commits["agx_arm_sim_agx_arm_urdf_gitlink"] == submodules[
        "agx_arm_urdf_in_agx_arm_sim"
    ]["recorded_gitlink"]


def test_artifact_evidence_items_are_all_classified(artifact: dict):
    assert artifact["evidence"]
    for entry in artifact["evidence"]:
        assert entry["kind"] in EVIDENCE_KINDS
        assert entry["classification"] == entry["kind"]
        assert entry["source"] and len(entry["source_sha256"]) == 64
        assert isinstance(entry["declared_tool_frame"], bool)


def test_artifact_classifies_legacy_offset_as_legacy_assumption(artifact: dict):
    legacy = [
        entry
        for entry in artifact["evidence"]
        if entry["id"] == "legacy-pregrasp-01358-offset"
    ]
    assert len(legacy) == 1
    assert legacy[0]["kind"] == "legacy_assumption"
    assert legacy[0]["translation_m"] == [0.0, 0.0, 0.1358]
    data = _artifact_source_path("run/pregrasp_v2.py", artifact).read_bytes()
    assert sha256(data).hexdigest() == legacy[0]["source_sha256"]


def test_artifact_compares_required_frames_and_legacy_offset(artifact: dict):
    comparisons = artifact["explicit_frame_comparisons"]
    assert {
        "link6",
        "flange_link",
        "gripper_base",
        "gripper_finger_frames",
        "legacy_01358_m_offset",
    } <= set(comparisons)
    assert comparisons["legacy_01358_m_offset"]["value_m"] == 0.1358
    assert comparisons["gripper_base"][
        "transform_from_link6_m_submodule"
    ] == [0.0, 0.0, 0.0045]
    assert comparisons["gripper_base"]["transform_from_link6_m_runtime"] == [
        0.0,
        0.0,
        0.0,
    ]


def test_artifact_frame_values_match_live_urdf_files(artifact: dict):
    import xml.etree.ElementTree as ET

    def origin(path: Path, joint_name: str) -> list[float]:
        for joint in ET.parse(path).getroot().findall("joint"):
            if joint.get("name") == joint_name:
                element = joint.find("origin")
                raw = element.get("xyz", "0 0 0") if element is not None else "0 0 0"
                return [float(v) for v in raw.split()]
        raise AssertionError(joint_name)

    xacro = ROOT / (
        "ros_ws/src/agx_arm_ros/src/agx_arm_description/agx_arm_urdf/piper_l/urdf"
        "/piper_l_with_gripper_description.xacro"
    )
    by_id = {entry["id"]: entry for entry in artifact["evidence"]}
    assert by_id["pinned-submodule-flange-joint"][
        "translation_m"
    ] == origin(xacro, "flange_joint")
    assert by_id["pinned-submodule-gripper-base-joint"]["translation_m"] == origin(
        xacro, "gripper_base_joint"
    )
    assert by_id["runtime-vendor-gripper-base-joint"]["translation_m"] == origin(
        VENDOR_URDF, "gripper_base_joint"
    )
    assert by_id["runtime-vendor-gripper-fingers"]["translation_m"] == origin(
        VENDOR_URDF, "gripper_joint1"
    ) == [0.0, 0.0, 0.1358]


def test_artifact_local_config_values_match_files(artifact: dict):
    config = json.loads((ROOT / "config/user_config.json").read_text(encoding="utf-8"))
    by_id = {entry["id"]: entry for entry in artifact["evidence"]}
    assert by_id["local-config-ee-frames"]["values"]["EE_Frames"] == config[
        "Robot_Config"
    ]["Robots"][0]["EE_Frames"] == ["link6"]
    offsets = by_id["local-config-grasp-offsets"]["values"]["Grasp_TCP_Top_Offset"]
    assert offsets == config["Manipulation_Config"]["Grasp_TCP_Top_Offset"]
    assert offsets == -0.0053
    run_config = json.loads((ROOT / "tests/fixtures/run-config-older-copy.json").read_text(encoding="utf-8"))
    assert "Grasp_TCP_Top_Offset" not in run_config.get("Manipulation_Config", {})
    assert by_id["run-config-older-copy"]["values"]["has_grasp_tcp_top_offset"] is False


def test_artifact_tcp_contract_is_resolved_via_project_calibration_only(
    artifact: dict,
):
    tcp = artifact["tcp_contract"]
    assert tcp["status"] == "RESOLVED"
    assert tcp["transform"] is not None
    assert tcp["blockers"] == []
    transform = tcp["transform"]
    assert transform["parent_frame"] == "link6"
    assert transform["child_frame"] == "rocobrick_tcp"
    assert transform["rotation_xyzw"] == [0.0, 0.0, 0.0, 1.0]
    assert transform["source"] == "artifacts/model-audit/piper_l_tcp_calibration.json"
    assert artifact["tcp_execution_allowed"] is True
    assert artifact["calibration_authority"] == "project"
    assert artifact["simulation_only"] is True
    assert artifact["hardware_validated"] is False
    assert artifact["runtime_usd_scoped"] is True
    project_items = [
        e for e in artifact["evidence"] if e["kind"] == KIND_PROJECT_CALIBRATION
    ]
    assert len(project_items) == 1
    assert project_items[0]["declared_tool_frame"] is True
    assert project_items[0]["passed"] is True
    assert not any(
        e["kind"] == "pinned_explicit_transform" and e.get("declared_tool_frame")
        for e in artifact["evidence"]
    )


def test_artifact_tcp_status_follows_the_strict_rule(artifact: dict):
    evidence_mapping = {
        "robot_model": artifact["model"],
        "pinned_sources": {
            entry["path"]: entry["sha256"]
            for entry in artifact["inputs"]["pinned_files"]
        },
        "items": artifact["evidence"],
    }
    reassessed = assess_piper_l_tcp(evidence_mapping)
    tcp = artifact["tcp_contract"]
    assert reassessed.status.value == tcp["status"]
    assert list(reassessed.candidate_frames) == tcp["candidate_frames"]
    assert list(reassessed.blockers) == tcp["blockers"]
    assert (reassessed.transform is None) == (tcp["transform"] is None)


def test_artifact_tcp_execution_allowed_iff_resolved(artifact: dict):
    resolved = artifact["tcp_contract"]["status"] == "RESOLVED"
    assert artifact["tcp_execution_allowed"] is resolved


def test_artifact_checks_are_consistent_and_passing(artifact: dict):
    assert artifact["checks"]
    assert all(value is True for value in artifact["checks"].values())
    assert artifact["passed"] is all(artifact["checks"].values()) is True
    assert artifact["checks"]["no_official_tcp_resolution_claimed"] is True
    assert artifact["checks"]["tcp_resolved_via_project_calibration_only"] is True
    assert artifact["checks"]["project_calibration_all_gates_passed"] is True
    assert artifact["checks"]["example1_plan_bytes_immutable"] is True


def test_artifact_example1_binding_matches_bytes_and_parser(artifact: dict):
    start, goal = task_bytes()
    decision = bind_single_stap_task(
        plan_bytes=PLAN.read_bytes(), task_start_bytes=start, task_goal_bytes=goal,
        robot_id=ROBOT,
    )
    assert decision.accepted
    binding = decision.binding
    recorded = artifact["example1_binding"]
    assert binding is not None
    assert recorded["accepted"] is True
    assert recorded["plan_id"] == binding.plan_id
    assert recorded["task_id"] == binding.task_id
    assert recorded["physical_instance_id"] == binding.physical_instance_id
    assert recorded["selected_robot"] == recorded["robot_id"] == ROBOT
    assert recorded["brick_id"] == binding.brick_id
    assert recorded["source_goal_key"] == binding.source_goal_key
    assert recorded["grid_xyz"] == list(binding.grid_xyz) == [0, 0, 1]
    assert recorded["orientation_quarter_turns"] == binding.orientation == 0
    assert recorded["color"] == binding.color
    assert recorded["motion_geometry"] == "omitted"
    assert recorded["plan_sha256"] == sha256(PLAN.read_bytes()).hexdigest()
    assert recorded["task_start_sha256"] == sha256(start).hexdigest()
    assert recorded["task_goal_sha256"] == sha256(goal).hexdigest()
    assert recorded["binding_requires_tcp"] is True
    assert recorded["plan_immutability"]["before_equals_after"] is True
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert recorded["canonical_task_sha256"]["start"] == plan["source"]["start_sha256"]
    assert recorded["canonical_task_sha256"]["goal"] == plan["source"]["goal_sha256"]
    assert recorded["pending_task_ids"] == ["assemble-768de5a17827920c48d4"]
    assert recorded["allocation"] == {
        "piper_0": ["assemble-768de5a17827920c48d4"],
        "piper_1": [],
    }


def test_artifact_runtime_vendor_identity_claim_holds(artifact: dict):
    runtime = ROOT / "robot_assets/piper_l/piper_l_gripper_d435.urdf"
    assert runtime.read_bytes() == VENDOR_URDF.read_bytes()
    assert artifact["checks"]["runtime_vendor_urdf_byte_identical"] is True


# ---------------------------------------------------------------------------
# Module hygiene: pure modules, no ROS/Isaac/tools/metric imports
# ---------------------------------------------------------------------------


MODULES = (
    ROOT / "src/rocobrick/tcp_contract.py",
    ROOT / "src/rocobrick/structural_binding.py",
)
ALLOWED_ROCOBRICK = {
    "rocobrick.compat",
    "rocobrick.stap_contract",
    "rocobrick.tcp_contract",
}


@pytest.mark.parametrize("module_path", MODULES, ids=lambda path: path.name)
def test_modules_import_no_ros_isaac_tools_or_metric_parser(module_path: Path):
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    for name in imported:
        root = name.split(".")[0]
        assert root not in FORBIDDEN_IMPORT_ROOTS, f"{module_path.name} imports {name}"
        if name.startswith("rocobrick."):
            assert name in ALLOWED_ROCOBRICK, f"{module_path.name} imports {name}"


def test_binding_module_never_imports_or_calls_metric_parser():
    source = (ROOT / "src/rocobrick/structural_binding.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        assert not (
            isinstance(node, ast.ImportFrom)
            and node.module == "rocobrick.contracts"
        )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "parse_plan"
    assert "parse_plan(" not in source.replace("parse_stap_plan(", "")


def test_modules_are_side_effect_free():
    for module_path in MODULES:
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        for node in tree.body:
            assert not isinstance(node, ast.Call)
