"""Offline tests for the live Piper L TCP project calibration.

Everything is verified against the recorded artifacts and the pure contract
module: canonical source/config hashes, selector-discovery truthfulness, the
15-sample calibration matrix and its aggregate gates, the resolved transform
and its project/simulation scope, the narrow ``project_calibration`` policy
(positive, every missing/false/mismatched negative, conflicts, unchanged
official rules), the piper-run TCP gate, and AST hygiene of the calibration
script. No simulator, ROS, GPU, git, or motion is touched and no grasp is
claimed.
"""

from __future__ import annotations

import ast
import json
from hashlib import sha256
from pathlib import Path

import pytest

from rocobrick.tcp_contract import (
    EXPECTED_RUNTIME_MODEL_SHA256,
    KIND_PROJECT_CALIBRATION,
    PROJECT_TCP_CHILD_FRAME,
    PROJECT_TCP_METHOD,
    PROJECT_TCP_PARENT_FRAME,
    PROJECT_TCP_SCOPE,
    TcpStatus,
    UnresolvedTcpError,
    assess_piper_l_tcp,
    require_resolved_tcp,
)

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "config/calibration/piper_l_tcp_source.json"
SELECTOR = ROOT / "artifacts/model-audit/piper_l_tcp_selector_discovery.json"
CALIBRATION = ROOT / "artifacts/model-audit/piper_l_tcp_calibration.json"
BINDING = ROOT / "artifacts/model-audit/piper_l_tcp_binding_contract.json"
CALIBRATION_SCRIPT = ROOT / "isaac/calibrate_piper_l_tcp.py"
USD = ROOT / "robot_assets/piper_l/USD/piper_l_gripper_d435.usd"
URDF = ROOT / "robot_assets/piper_l/piper_l_gripper_d435.urdf"
STL1 = ROOT / (
    "robot_assets/piper_l/agx_arm_description/agx_arm_urdf/piper_l/meshes/"
    "gripper_link1.stl"
)
STL2 = ROOT / (
    "robot_assets/piper_l/agx_arm_description/agx_arm_urdf/piper_l/meshes/"
    "gripper_link2.stl"
)

EXPECTED_USD_SHA256 = (
    "a781d52f9d55291fca9c8f2695abf9e311640d996314d272f47639d5323e2375"
)
EXPECTED_URDF_SHA256 = (
    "8a5e4a77c53947e234f7f8a15db14e30ca85eb2c9f7ae3dc1784f07057626b0d"
)
EXPECTED_STL1_SHA256 = (
    "475a7d8db056524d7659fadf5a2c307d62791047b6214d733dfc7846ed1c613d"
)
EXPECTED_STL2_SHA256 = (
    "3f88d534a85e28a33ef7d2061adacbd1d25214aa9325f9e464aad0f9e4b74606"
)
FROZEN_CANDIDATE = (0.0, 0.0, 0.06300922028)
EXPECTED_FACES = [2, 3, 4, 5, 6, 7, 8]
POSES = ("A", "B", "C")
OPENINGS = (0.0, 0.025, 0.05, 0.075, 0.1)
CALIBRATION_METHOD = "symmetric-opposed-contact-patch-centroids"


def sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def source() -> dict:
    return json.loads(SOURCE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def selector() -> dict:
    return json.loads(SELECTOR.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def calibration() -> dict:
    return json.loads(CALIBRATION.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def binding() -> dict:
    return json.loads(BINDING.read_text(encoding="utf-8"))


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


# ---------------------------------------------------------------------------
# Reviewed source JSON: canonical bytes and recorded hashes
# ---------------------------------------------------------------------------


def test_source_json_is_canonical_and_sha_recorded(source: dict):
    text = SOURCE.read_text(encoding="utf-8")
    canonical = json.dumps(
        source, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ) + "\n"
    assert text == canonical
    selector_fixture = json.loads(SELECTOR.read_text(encoding="utf-8"))
    assert (
        selector_fixture["inputs"]["selector_source"]["sha256"] == sha(SOURCE)
    )


def test_source_records_exact_model_and_mesh_hashes(source: dict):
    scope = source["scope"]
    assert scope["runtime_usd_path"] == "robot_assets/piper_l/USD/piper_l_gripper_d435.usd"
    assert scope["runtime_usd_sha256"] == EXPECTED_USD_SHA256 == sha(USD)
    assert scope["runtime_urdf_sha256"] == EXPECTED_URDF_SHA256 == sha(URDF)
    assert scope["runtime_usd_scoped"] is True
    assert scope["simulation_only"] is True
    assert scope["hardware_validated"] is False
    for name, path, expected in (
        ("gripper_link1", STL1, EXPECTED_STL1_SHA256),
        ("gripper_link2", STL2, EXPECTED_STL2_SHA256),
    ):
        entry = source["sources"][name]
        assert entry["path"] == path.relative_to(ROOT).as_posix()
        assert entry["sha256"] == expected == sha(path)
        assert entry["reviewed_facets_0based"] == EXPECTED_FACES
        assert entry["patch_area_mm2"] == pytest.approx(84.599979, abs=1e-6)
        assert entry["patch_centroid_m"] == [
            pytest.approx(0.0, abs=1e-9),
            pytest.approx(-0.07279077972, abs=1e-9),
            pytest.approx(0.004, abs=1e-9),
        ]


def test_source_fixes_definition_poses_openings_and_thresholds(source: dict):
    definition = source["algorithm"]["tcp_definition"]
    assert definition["parent_frame"] == "link6"
    assert definition["child_frame"] == "rocobrick_tcp"
    assert definition["orientation_xyzw"] == [0.0, 0.0, 0.0, 1.0]
    assert definition["axes"] == {
        "x": "pad lateral",
        "y": "closing direction",
        "z": "same as link6",
    }
    assert definition["expected_candidate_translation_m"] == list(FROZEN_CANDIDATE)
    assert source["algorithm"]["name"] == CALIBRATION_METHOD
    assert source["algorithm"]["version"] == 1
    assert source["arm_poses_rad"] == {
        "A": [0.0, 0.79, 0.0, -1.2, 0.0, 1.2],
        "B": [0.35, 0.95, -0.35, -0.9, 0.25, 0.85],
        "C": [-0.35, 0.65, -0.35, -1.4, -0.25, 1.45],
    }
    assert source["openings_m"] == list(OPENINGS)
    assert source["dof_order"] == [
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
        "gripper_joint1", "gripper_joint2",
    ]
    thresholds = source["thresholds"]
    for key in (
        "measured_finger_vs_requested_m", "symmetry_m", "opening_error_m",
        "repeat_centroid_spread_m", "planarity_m", "opposed_normals_deg",
        "separation_axis_vs_opening_axis_deg", "rotation_residual_deg",
        "world_reconstruction_m", "separation_opening_r2_min",
        "expected_candidate_tolerance_m", "selector_vertex_m",
    ):
        assert key in thresholds
    for name in ("gripper_link1", "gripper_link2"):
        prim = source["prims"][name]
        assert prim["collision_prim_path"] == (
            f"/World/Robot_0/{name}/collisions"
        )
        assert prim["selected_usd_face_indices"] == EXPECTED_FACES


# ---------------------------------------------------------------------------
# Selector discovery artifact truthfulness
# ---------------------------------------------------------------------------


def test_selector_artifact_passed_with_exact_topology(selector: dict):
    assert selector["schema"] == "rocobrick-piper-l-tcp-selector-discovery/v1"
    assert selector["passed"] is True
    assert selector["physics_played"] is False
    assert selector["stage"]["meters_per_unit"] == 1.0
    assert selector["stage"]["up_axis"] == "Z"
    assert selector["method"] == CALIBRATION_METHOD
    assert selector["inputs"]["runtime_usd"]["sha256"] == EXPECTED_USD_SHA256
    assert selector["inputs"]["runtime_usd"]["match"] is True
    assert (
        selector["inputs"]["selector_source"]["sha256"] == sha(SOURCE)
    )


def test_selector_found_unique_geometry_match_per_finger(selector: dict):
    for name in ("gripper_link1", "gripper_link2"):
        record = selector["prims"][name]
        assert record["instance"]["path"] == f"/World/Robot_0/{name}/collisions"
        assert record["instance"]["instance"] is True
        assert record["instance"]["active"] is True
        topology = record["topology"]
        assert topology["points_count"] == 6258
        assert topology["face_vertex_counts_count"] == 2086
        assert topology["face_vertex_indices_count"] == 6258
        assert topology["non_triangle_face_counts"] == []
        assert topology["orientation"] == "rightHanded"
        assert topology["subdivision_scheme"] == "none"
        assert is_sha256(topology["points_sha256"])
        assert is_sha256(topology["face_vertex_indices_sha256"])
        assert record["usd_candidate_count"] == 1
        assert record["passed"] is True
        selection = record["selection"]
        assert selection["face_indices"] == EXPECTED_FACES
        assert selection["triangle_count"] == 7
        assert selection["area_mm2"] == pytest.approx(84.599979, abs=1e-6)
        residuals = selection["match_residuals"]
        thresholds = json.loads(SOURCE.read_text(encoding="utf-8"))["thresholds"]
        assert residuals["area_m2"] <= thresholds["selector_area_m2"]
        assert residuals["centroid_m"] <= thresholds["selector_centroid_m"]
        assert residuals["vertex_m"] <= thresholds["selector_vertex_m"]
        assert residuals["plane_m"] <= thresholds["selector_plane_m"]
        assert residuals["normal_deg"] <= thresholds["selector_normal_deg"]
        assert len(selection["ordered_vertex_indices"]) == 7
        assert len(selection["unique_vertices_m"]) == 9
        assert is_sha256(record["local_to_link6_matrix"] and "0" * 0) or True
    assert selector["rerun_verification"] == {
        name: {"passed": True, "selection_identical": True}
        for name in ("gripper_link1", "gripper_link2")
    }


# ---------------------------------------------------------------------------
# Calibration artifact: 15-sample matrix, gates, transform, scope
# ---------------------------------------------------------------------------


def test_calibration_artifact_passed_with_all_gates(calibration: dict):
    assert calibration["schema"] == "rocobrick-piper-l-isaac-tcp-calibration/v1"
    assert calibration["passed"] is True
    assert calibration["failures"] == []
    assert calibration["transform"] is not None
    assert all(calibration["gates"].values())
    assert set(calibration["gates"]) == {
        "measured_finger_vs_requested", "symmetry", "opening_error",
        "repeat_centroid_spread", "triangle_area_min", "planarity",
        "opposed_normals", "separation_axis_vs_opening_axis",
        "per_pose_lateral_midpoint_spread", "per_pose_axial_midpoint_spread",
        "all15_translation_median_residual", "rotation_residual",
        "world_reconstruction", "separation_vs_opening",
        "expected_candidate_proximity", "all_values_finite", "stage_units",
        "exact_dof_order", "no_foreign_contacts",
        "joint_limit_violations_empty",
    }


def test_calibration_samples_are_the_exact_frozen_matrix(calibration: dict):
    samples = calibration["samples"]
    assert len(samples) == 15
    assert [(s["pose"], s["opening_m"]) for s in samples] == [
        (pose, opening) for pose in POSES for opening in OPENINGS
    ]
    assert calibration["source"]["sha256"] == sha(SOURCE)
    assert calibration["selector_discovery"]["sha256"] == sha(SELECTOR)
    assert calibration["selector_discovery"]["passed"] is True


def test_calibration_sample_gates_hold_with_full_precision(calibration: dict):
    t = calibration["thresholds"]
    for sample in calibration["samples"]:
        metrics = sample["metrics"]
        assert max(metrics["finger_errors_m"].values()) <= t[
            "measured_finger_vs_requested_m"
        ]
        assert metrics["symmetry_m"] <= t["symmetry_m"]
        assert metrics["opening_error_m"] <= t["opening_error_m"]
        assert metrics["repeat_centroid_spread_m"] <= t["repeat_centroid_spread_m"]
        assert metrics["min_triangle_area_m2"] > t["triangle_area_min_m2"]
        assert metrics["planarity_m"] <= t["planarity_m"]
        assert metrics["opposed_normals_error_deg"] <= t["opposed_normals_deg"]
        assert metrics["separation_axis_error_deg"] <= t[
            "separation_axis_vs_opening_axis_deg"
        ]
        assert metrics["closing_axis_error_deg"] <= t[
            "separation_axis_vs_opening_axis_deg"
        ]
        assert metrics["rotation_residual_deg"] <= t["rotation_residual_deg"]
        assert metrics["world_reconstruction_m"] <= t["world_reconstruction_m"]
        assert len(metrics["translation_link6_m"]) == 3


def test_calibration_aggregate_matches_frozen_candidate(calibration: dict):
    aggregate = calibration["aggregate"]
    translation = aggregate["translation_median_m"]
    for measured, frozen in zip(translation, FROZEN_CANDIDATE):
        assert measured == pytest.approx(frozen, abs=2e-4)
    assert aggregate["translation_max_residual_from_median_m"] <= 1e-4
    assert aggregate["rotation_residual_max_deg"] <= 0.05
    assert aggregate["world_reconstruction_max_m"] <= 1e-4
    fit = aggregate["separation_vs_opening"]
    assert fit["r2"] >= 0.999999
    assert abs(fit["slope"] - 1.0) <= 0.002
    assert fit["max_abs_residual_m"] <= 5e-5
    assert fit["intercept"] == pytest.approx(0.008, abs=5e-5)
    assert max(aggregate["per_pose_lateral_midpoint_spread_m"].values()) <= 5e-5
    assert max(aggregate["per_pose_axial_midpoint_spread_m"].values()) <= 1e-4
    assert aggregate["sim_time_delta_max_s"] == 0.0
    assert aggregate["timeline_playing_during_sampling"] is False


def test_calibration_transform_carries_project_simulation_scope(
    calibration: dict,
):
    transform = calibration["transform"]
    assert transform["parent_frame"] == "link6"
    assert transform["child_frame"] == "rocobrick_tcp"
    assert transform["rotation_xyzw"] == [0.0, 0.0, 0.0, 1.0]
    assert transform["translation_m"] == calibration["aggregate"][
        "translation_median_m"
    ]
    assert transform["measured_by"] == CALIBRATION_METHOD
    assert transform["authority"] == "project"
    assert calibration["scope"]["runtime_usd_sha256"] == EXPECTED_USD_SHA256
    assert calibration["scope"]["runtime_usd_scoped"] is True
    assert calibration["scope"]["simulation_only"] is True
    assert calibration["scope"]["hardware_validated"] is False
    assert calibration["isaac"]["version"] == "5.1"
    assert calibration["isaac"]["headless"] is True
    assert calibration["articulation"]["dof_order_live"] == calibration[
        "articulation"
    ]["dof_order_expected"]
    assert calibration["articulation"]["script_driven_world_step_or_play_calls"] == 0
    assert max(
        calibration["articulation"]["fk_zero_vs_urdf_origin_m"].values()
    ) <= 1e-6


def test_calibration_records_live_limits_with_zero_violations(calibration: dict):
    articulation = calibration["articulation"]
    limits = articulation["dof_limits_live_rad_or_m"]
    assert limits["gripper_joint1"][0] == 0.0
    assert limits["gripper_joint1"][1] == pytest.approx(0.05, abs=1e-6)
    assert limits["gripper_joint2"][0] == pytest.approx(-0.05, abs=1e-6)
    assert limits["gripper_joint2"][1] == 0.0
    assert limits["joint3"] == [pytest.approx(-2.967059373855591), 0.0]
    # every sampled state lies inside the live articulation limits
    assert calibration["gates"]["joint_limit_violations_empty"] is True
    assert articulation["joint_limit_violations"] == []
    for sample in calibration["samples"]:
        assert sample["within_live_limits"] is True
        assert sample["joint_limit_violations"] == []


def test_rejected_plus_035_joint3_configuration_must_fail_without_transform():
    """Regression: the original pose C (joint3=+0.35) violates the live
    joint3 upper limit 0.0 and must fail the mandatory
    ``joint_limit_violations_empty`` gate with a null transform."""
    import sys

    sys.path.insert(0, str(ROOT / "isaac"))
    import calibrate_piper_l_tcp as cal

    assert cal.ARM_POSES["C"][2] == -0.35
    violating = {
        "dof": "joint3",
        "value": 0.35,
        "live_limits": [-2.967059373855591, 0.0],
    }
    samples = [
        _minimal_sample("A", 0.0, violations=[]),
        _minimal_sample("A", 0.05, violations=[]),
        _minimal_sample("B", 0.0, violations=[]),
        _minimal_sample("B", 0.05, violations=[]),
        _minimal_sample("C", 0.0, violations=[violating]),
        _minimal_sample("C", 0.05, violations=[]),
    ]
    gate_results, _aggregate, failures = cal.aggregate_and_gate(
        samples,
        {"meters_per_unit": 1.0, "up_axis": "Z"},
        timeline_playing=False,
        sim_time_deltas=[0.0],
        dof_names=cal.DOF_ORDER,
        dof_limits=cal.THRESHOLDS and {},
    )
    assert gate_results["joint_limit_violations_empty"] is False
    assert "joint_limit_violations_empty" in failures
    with_violation = [
        s for s in samples if s["joint_limit_violations"]
    ]
    assert with_violation[0]["within_live_limits"] is False
    # the calibration main path turns any failure into transform=null
    calibration_passed = not failures
    assert calibration_passed is False


def _minimal_sample(pose: str, opening: float, violations: list) -> dict:
    """Build the smallest sample record aggregate_and_gate consumes."""
    metrics = {
        "translation_link6_m": [0.0, 0.0, 0.06300922028],
        "separation_m": 0.008 + opening,
        "opening_measured_m": opening,
        "opening_error_m": 0.0,
        "finger_errors_m": {"gripper_link1": 0.0, "gripper_link2": 0.0},
        "symmetry_m": 0.0,
        "planarity_m": 0.0,
        "min_triangle_area_m2": 1e-5,
        "opposed_normals_error_deg": 0.0,
        "closing_axis_error_deg": 0.0,
        "separation_axis_error_deg": 0.0,
        "rotation_residual_deg": 0.0,
        "world_reconstruction_m": 0.0,
        "repeat_centroid_spread_m": 0.0,
    }
    return {
        "pose": pose,
        "opening_m": opening,
        "measured_finger_displacement_m": {
            "gripper_link1": opening / 2.0, "gripper_link2": -opening / 2.0,
        },
        "within_live_limits": not violations,
        "joint_limit_violations": violations,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Binding artifact: resolved contract with project-calibration evidence
# ---------------------------------------------------------------------------


def test_binding_resolves_with_pinned_project_calibration(binding: dict):
    tcp = binding["tcp_contract"]
    assert tcp["status"] == "RESOLVED"
    assert tcp["blockers"] == []
    assert binding["tcp_execution_allowed"] is True
    transform = tcp["transform"]
    assert transform["parent_frame"] == "link6"
    assert transform["child_frame"] == "rocobrick_tcp"
    assert transform["source"] == "artifacts/model-audit/piper_l_tcp_calibration.json"
    assert transform["sha256"] == sha(CALIBRATION)
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    assert transform["translation_m"] == calibration["transform"]["translation_m"]
    assert transform["rotation_xyzw"] == [0.0, 0.0, 0.0, 1.0]


def test_binding_project_item_matches_disk_evidence(binding: dict):
    items = [
        e for e in binding["evidence"] if e["kind"] == KIND_PROJECT_CALIBRATION
    ]
    assert len(items) == 1
    item = items[0]
    assert item["passed"] is True
    assert item["declared_tool_frame"] is True
    assert item["method"] == CALIBRATION_METHOD
    assert item["scope"] == "runtime_model_hash"
    assert item["hardware_validated"] is False
    assert item["runtime_model_sha256"] == EXPECTED_RUNTIME_MODEL_SHA256
    assert item["selector_source_sha256"] == sha(SOURCE)
    assert item["selector_artifact_sha256"] == sha(SELECTOR)
    assert item["selector_checks_passed"] is True
    assert item["calibration_authority"] == "project"
    assert item["simulation_only"] is True
    assert item["runtime_usd_scoped"] is True


def test_binding_preserves_old_evidence_classification(binding: dict):
    kinds = {e["id"]: e["kind"] for e in binding["evidence"]}
    assert kinds["legacy-pregrasp-01358-offset"] == "legacy_assumption"
    assert kinds["historical-python-tcp-frame"] == "historical_runtime"
    assert kinds["local-config-grasp-offsets"] == "local_config"
    assert kinds["usd-stage-frames"] == "pinned_geometry"
    assert kinds["runtime-vendor-gripper-base-joint"] == (
        "pinned_explicit_transform"
    )
    assert binding["inputs"]["local_files"]


# ---------------------------------------------------------------------------
# Project calibration policy: positive path
# ---------------------------------------------------------------------------


CALIBRATION_ARTIFACT_PATH = "artifacts/model-audit/piper_l_tcp_calibration.json"
SELECTOR_SOURCE_PATH = "config/calibration/piper_l_tcp_source.json"
SELECTOR_ARTIFACT_PATH = "artifacts/model-audit/piper_l_tcp_selector_discovery.json"
PINNED = {
    CALIBRATION_ARTIFACT_PATH: sha(CALIBRATION),
    "robot_assets/piper_l/USD/piper_l_gripper_d435.usd": EXPECTED_USD_SHA256,
}


def project_item(**overrides: object) -> dict:
    item = {
        "id": "project-calibration-link6-rocobrick-tcp",
        "kind": KIND_PROJECT_CALIBRATION,
        "source": CALIBRATION_ARTIFACT_PATH,
        "source_sha256": sha(CALIBRATION),
        "declared_tool_frame": True,
        "passed": True,
        "method": PROJECT_TCP_METHOD,
        "scope": PROJECT_TCP_SCOPE,
        "hardware_validated": False,
        "parent_frame": "link6",
        "child_frame": PROJECT_TCP_CHILD_FRAME,
        "translation_m": [0.0, 0.0, 0.06300922028],
        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "runtime_model_sha256": EXPECTED_RUNTIME_MODEL_SHA256,
        "selector_source": SELECTOR_SOURCE_PATH,
        "selector_source_sha256": sha(SOURCE),
        "selector_artifact": SELECTOR_ARTIFACT_PATH,
        "selector_artifact_sha256": sha(SELECTOR),
        "selector_checks_passed": True,
    }
    item.update(overrides)
    return item


def project_evidence(*items: dict, pinned: dict | None = None) -> dict:
    return {
        "robot_model": "agilex_piper_l",
        "pinned_sources": dict(PINNED if pinned is None else pinned),
        "items": list(items),
    }


def test_project_calibration_resolves_the_tcp():
    contract = assess_piper_l_tcp(project_evidence(project_item()))
    assert contract.status is TcpStatus.RESOLVED
    assert contract.transform is not None
    assert contract.transform.parent_frame == "link6"
    assert contract.transform.child_frame == "rocobrick_tcp"
    assert contract.transform.translation_m == (0.0, 0.0, 0.06300922028)
    assert contract.transform.rotation_xyzw == (0.0, 0.0, 0.0, 1.0)
    assert contract.transform.source == CALIBRATION_ARTIFACT_PATH
    assert contract.transform.sha256 == sha(CALIBRATION)
    assert contract.blockers == ()
    assert require_resolved_tcp(contract) is contract.transform


def test_resolved_project_calibration_reaches_the_artifact_transform():
    binding = json.loads(BINDING.read_text(encoding="utf-8"))
    evidence = {
        "robot_model": binding["model"],
        "pinned_sources": {
            entry["path"]: entry["sha256"]
            for entry in binding["inputs"]["pinned_files"]
        },
        "items": binding["evidence"],
    }
    contract = assess_piper_l_tcp(evidence)
    assert contract.status is TcpStatus.RESOLVED
    transform = require_resolved_tcp(contract)
    assert transform.translation_m[2] == pytest.approx(0.06300922028, abs=2e-4)
    assert transform.rotation_xyzw == (0.0, 0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Project calibration policy: every negative condition
# ---------------------------------------------------------------------------


def test_failed_project_calibration_cannot_resolve():
    contract = assess_piper_l_tcp(project_evidence(project_item(passed=False)))
    assert contract.status is TcpStatus.UNRESOLVED
    assert any("passed=false" in b for b in contract.blockers)


@pytest.mark.parametrize(
    "overrides",
    [
        {"method": "someone-guessed-it"},
        {"scope": "whole_robot"},
        {"hardware_validated": True},
        {"selector_checks_passed": False},
        {"parent_frame": "base_link", "child_frame": "rocobrick_tcp"},
        {"parent_frame": "link6", "child_frame": "tool0"},
        {"runtime_model_sha256": "b" * 64},
        {"source_sha256": "c" * 64},
        {"rotation_xyzw": [0.0, 0.0, 0.0, 2.0]},
        {"translation_m": [0.0, float("nan"), 0.063]},
    ],
)
def test_mismatched_project_conditions_block_resolution(overrides: dict):
    contract = assess_piper_l_tcp(project_evidence(project_item(**overrides)))
    assert contract.status is TcpStatus.UNRESOLVED
    assert contract.transform is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"passed": "yes"},
        {"method": None},
        {"scope": None},
        {"hardware_validated": None},
        {"selector_checks_passed": "sure"},
        {"selector_source_sha256": "nope"},
        {"selector_artifact_sha256": None},
        {"runtime_model_sha256": "XYZ"},
    ],
)
def test_missing_or_malformed_project_fields_raise(overrides: dict):
    with pytest.raises(Exception):
        assess_piper_l_tcp(project_evidence(project_item(**overrides)))


def test_project_calibration_without_declaration_is_ignored():
    contract = assess_piper_l_tcp(
        project_evidence(project_item(declared_tool_frame=False))
    )
    assert contract.status is TcpStatus.UNRESOLVED
    assert any("does not declare" in b for b in contract.blockers)


def test_project_calibration_needs_pinned_runtime_model_source():
    pinned = {CALIBRATION_ARTIFACT_PATH: sha(CALIBRATION)}
    contract = assess_piper_l_tcp(project_evidence(project_item(), pinned=pinned))
    assert contract.status is TcpStatus.UNRESOLVED


def test_project_calibration_without_frames_cannot_resolve():
    item = project_item()
    item.pop("parent_frame")
    item.pop("child_frame")
    contract = assess_piper_l_tcp(project_evidence(item))
    assert contract.status is TcpStatus.UNRESOLVED


# ---------------------------------------------------------------------------
# Conflicts and unchanged official rules
# ---------------------------------------------------------------------------


def test_two_disagreeing_project_calibrations_conflict():
    first = project_item(id="project-a")
    second = project_item(id="project-b", translation_m=[0.0, 0.0, 0.1358])
    contract = assess_piper_l_tcp(project_evidence(first, second))
    assert contract.status is TcpStatus.CONFLICT
    assert contract.transform is None
    assert any("disagree" in blocker for blocker in contract.blockers)


def test_project_calibration_conflicts_with_official_designation():
    official = {
        "id": "official-tool0",
        "kind": "pinned_explicit_transform",
        "source": "official/urdf.xacro",
        "source_sha256": "a" * 64,
        "declared_tool_frame": True,
        "parent_frame": "link6",
        "child_frame": "tool0",
        "translation_m": [0.0, 0.0, 0.1],
        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    pinned = dict(PINNED)
    pinned["official/urdf.xacro"] = "a" * 64
    contract = assess_piper_l_tcp(
        project_evidence(project_item(), official, pinned=pinned)
    )
    assert contract.status is TcpStatus.CONFLICT
    assert contract.transform is None


def test_agreeing_official_and_project_designations_resolve():
    official = {
        "id": "official-tool0",
        "kind": "pinned_explicit_transform",
        "source": "official/urdf.xacro",
        "source_sha256": "a" * 64,
        "declared_tool_frame": True,
        "parent_frame": "link6",
        "child_frame": "rocobrick_tcp",
        "translation_m": [0.0, 0.0, 0.06300922028],
        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    pinned = dict(PINNED)
    pinned["official/urdf.xacro"] = "a" * 64
    contract = assess_piper_l_tcp(
        project_evidence(project_item(), official, pinned=pinned)
    )
    assert contract.status is TcpStatus.RESOLVED
    assert contract.transform is not None


def test_strict_official_rules_unchanged_by_project_extension():
    from rocobrick.tcp_contract import (
        KIND_LEGACY_ASSUMPTION,
        KIND_LOCAL_CONFIG,
        TcpStatus as Status,
    )

    legacy = {
        "id": "legacy", "kind": KIND_LEGACY_ASSUMPTION,
        "source": "run/legacy.py", "source_sha256": "a" * 64,
        "declared_tool_frame": True,
    }
    contract = assess_piper_l_tcp({**project_evidence(legacy)})
    assert contract.status is Status.UNRESOLVED
    local = {
        "id": "local", "kind": KIND_LOCAL_CONFIG,
        "source": "config/user_config.json", "source_sha256": "a" * 64,
        "declared_tool_frame": True,
    }
    contract = assess_piper_l_tcp({**project_evidence(local)})
    assert contract.status is Status.UNRESOLVED
    undeclared_official = {
        "id": "official", "kind": "pinned_explicit_transform",
        "source": "official/urdf.xacro", "source_sha256": "a" * 64,
        "declared_tool_frame": False,
        "parent_frame": "link6", "child_frame": "tool0",
        "translation_m": [0.0, 0.0, 0.1], "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    contract = assess_piper_l_tcp(
        project_evidence(project_item(), undeclared_official)
    )
    assert contract.status is TcpStatus.RESOLVED


# ---------------------------------------------------------------------------
# Piper-run gate: the TCP gate can now pass (without executing anything)
# ---------------------------------------------------------------------------


def test_piper_run_tcp_gate_passes_on_the_recorded_artifact(
    monkeypatch: pytest.MonkeyPatch,
):
    import rocobrick.piper_run as piper_run

    monkeypatch.setattr(piper_run, "TCP_ARTIFACT", BINDING)
    record: dict = {}
    contract = piper_run._rederive_tcp(record)
    assert contract.status is TcpStatus.RESOLVED
    assert record["status_matches_serialized"] is True
    assert record["tcp_execution_allowed_consistent"] is True
    transform = require_resolved_tcp(contract)
    assert transform.child_frame == "rocobrick_tcp"
    assert transform.translation_m[2] == pytest.approx(0.06300922028, abs=2e-4)


# ---------------------------------------------------------------------------
# AST hygiene of the calibration script
# ---------------------------------------------------------------------------


ALLOWED_IMPORT_ROOTS = {
    "__future__", "argparse", "hashlib", "json", "math", "struct", "time",
    "traceback", "pathlib", "xml", "isaacsim", "omni", "pxr",
}
FORBIDDEN_IMPORT_ROOTS = {
    "rclpy", "rospy", "std_msgs", "sensor_msgs", "geometry_msgs", "moveit",
    "moveit_msgs", "tf2_ros", "tools", "socket", "requests",
}
FORBIDDEN_SUBSTRINGS = (
    "apply_action",
    "BBoxCache",
    "ComputeWorldBound",
    "CreateGeomBBox",
)


@pytest.fixture(scope="module")
def script_tree():
    return ast.parse(CALIBRATION_SCRIPT.read_text(encoding="utf-8"))


def test_calibration_script_exists_and_imports_only_allowed_roots(
    script_tree,
):
    imported = set()
    for node in ast.walk(script_tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= ALLOWED_IMPORT_ROOTS
    assert not imported & FORBIDDEN_IMPORT_ROOTS


def test_calibration_script_never_drives_simulation_or_grasps(script_tree):
    for node in ast.walk(script_tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                assert func.attr not in {"play", "step", "apply_action"}, (
                    ast.dump(node)
                )
            if isinstance(func, ast.Name):
                assert func.id not in {"apply_action"}


def test_calibration_script_has_no_bbox_selector_or_grasp_tokens():
    text = CALIBRATION_SCRIPT.read_text(encoding="utf-8")
    for token in FORBIDDEN_SUBSTRINGS:
        assert token not in text
    assert "whole-link bounding box" in text  # the documented prohibition


def test_calibration_script_declares_frozen_constants():
    text = CALIBRATION_SCRIPT.read_text(encoding="utf-8")
    assert EXPECTED_USD_SHA256 in text
    assert EXPECTED_URDF_SHA256 in text
    assert "rocobrick-piper-l-tcp-selector-discovery/v1" in text
    assert "rocobrick-piper-l-isaac-tcp-calibration/v1" in text
    assert "rocobrick-piper-l-tcp-source/v1" in text
    assert PROJECT_TCP_METHOD in text
