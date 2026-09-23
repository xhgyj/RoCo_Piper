"""Focused tests for the strict simulation-only TCP capability gate."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path
from types import MappingProxyType

import pytest

from rocobrick.tcp_capability import TcpCapabilityDecision, authorize_tcp
from rocobrick.tcp_contract import (
    EXPECTED_RUNTIME_MODEL_SHA256,
    TcpContract,
    TcpStatus,
    Transform,
)

PROJECT_EVIDENCE = "project-calibration-link6-rocobrick-tcp"
CALIBRATION_SOURCE = "artifacts/model-audit/piper_l_tcp_calibration.json"
CALIBRATION_SHA256 = "b" * 64
CHECK_NAMES = (
    "contract_type_valid",
    "backend_is_isaac",
    "robot_model_matches",
    "contract_resolved",
    "transform_present",
    "project_calibration_evidence_present",
    "simulation_only_source_matches",
    "parent_frame_matches",
    "child_frame_matches",
    "runtime_asset_sha256_valid",
    "runtime_asset_sha256_matches",
    "calibration_artifact_sha256_valid",
    "calibration_artifact_sha256_matches",
)


def make_transform(**overrides: object) -> Transform:
    values: dict[str, object] = {
        "parent_frame": "link6",
        "child_frame": "rocobrick_tcp",
        "translation_m": (0.0, 0.0, 0.0),
        "rotation_xyzw": (0.0, 0.0, 0.0, 1.0),
        "source": CALIBRATION_SOURCE,
        "sha256": CALIBRATION_SHA256,
    }
    values.update(overrides)
    return Transform(**values)  # type: ignore[arg-type]


def make_contract(
    *,
    robot_model: str = "agilex_piper_l",
    status: TcpStatus = TcpStatus.RESOLVED,
    transform: Transform | None = None,
    evidence: tuple[str, ...] = (PROJECT_EVIDENCE,),
    blockers: tuple[str, ...] = (),
) -> TcpContract:
    if status is TcpStatus.RESOLVED and transform is None:
        transform = make_transform()
    if status is not TcpStatus.RESOLVED and not blockers:
        blockers = ("test blocker",)
    return TcpContract(
        robot_model=robot_model,
        status=status,
        transform=transform,
        candidate_frames=("link6", "rocobrick_tcp"),
        evidence=evidence,
        blockers=blockers,
    )


def authorize(
    contract: object,
    *,
    backend: object = "isaac",
    runtime_asset_sha256: object = EXPECTED_RUNTIME_MODEL_SHA256,
    calibration_artifact_sha256: object = CALIBRATION_SHA256,
) -> TcpCapabilityDecision:
    return authorize_tcp(
        contract,  # type: ignore[arg-type]
        backend=backend,  # type: ignore[arg-type]
        runtime_asset_sha256=runtime_asset_sha256,  # type: ignore[arg-type]
        calibration_artifact_sha256=calibration_artifact_sha256,  # type: ignore[arg-type]
    )


def test_authorizes_exact_resolved_project_calibration_contract():
    contract = make_contract()
    decision = authorize(contract)

    assert decision.allowed is True
    assert decision.status == "TCP_AUTHORIZED"
    assert decision.reason == "Isaac simulation-only TCP capability is authorized."
    assert decision.transform is contract.transform
    assert tuple(decision.checks) == CHECK_NAMES
    assert all(decision.checks.values())
    assert decision.allowed == all(decision.checks.values())
    assert isinstance(decision.checks, MappingProxyType)


@pytest.mark.parametrize("backend", [None, "ISAAC", " isaac", "isaac "])
def test_backend_must_match_isaac_exactly(backend: object):
    decision = authorize(make_contract(), backend=backend)

    assert decision.allowed is False
    assert decision.checks["backend_is_isaac"] is False
    assert decision.reason == "TCP capability rejected: backend_is_isaac"


class AttributeTrap:
    def __getattribute__(self, name: str) -> object:
        raise AssertionError(f"unexpected contract attribute access: {name}")


def test_non_contract_is_rejected_without_attribute_access():
    decision = authorize(AttributeTrap())

    assert decision.allowed is False
    assert decision.status == "UNRESOLVED_TCP"
    assert decision.transform is None
    assert decision.checks["contract_type_valid"] is False
    assert decision.reason == (
        "TCP capability rejected: contract_type_valid, robot_model_matches, "
        "contract_resolved, transform_present, "
        "project_calibration_evidence_present, simulation_only_source_matches, "
        "parent_frame_matches, child_frame_matches, "
        "calibration_artifact_sha256_matches"
    )


@pytest.mark.parametrize("status", [TcpStatus.UNRESOLVED, TcpStatus.CONFLICT])
def test_unresolved_and_conflict_contracts_are_rejected(status: TcpStatus):
    contract = make_contract(status=status, transform=None, blockers=("blocked",))
    decision = authorize(contract)

    assert decision.allowed is False
    assert decision.checks["contract_type_valid"] is True
    assert decision.checks["contract_resolved"] is False
    assert decision.checks["transform_present"] is False
    assert decision.checks["project_calibration_evidence_present"] is True
    assert decision.transform is None


def test_robot_model_must_match_exactly():
    decision = authorize(make_contract(robot_model="agilex_piper"))

    assert decision.checks["robot_model_matches"] is False
    assert decision.reason == "TCP capability rejected: robot_model_matches"


@pytest.mark.parametrize(
    ("evidence", "source", "failed_check"),
    [
        (("official-pinned-tcp",), CALIBRATION_SOURCE, "project_calibration_evidence_present"),
        ((PROJECT_EVIDENCE,), "official/urdf.xacro", "simulation_only_source_matches"),
        ((PROJECT_EVIDENCE,), CALIBRATION_SOURCE, "parent_frame_matches"),
        ((PROJECT_EVIDENCE,), CALIBRATION_SOURCE, "child_frame_matches"),
    ],
)
def test_project_simulation_provenance_and_frames_are_exact(
    evidence: tuple[str, ...], source: str, failed_check: str
):
    transform_overrides: dict[str, object] = {"source": source}
    if failed_check == "parent_frame_matches":
        transform_overrides["parent_frame"] = "base_link"
    if failed_check == "child_frame_matches":
        transform_overrides["child_frame"] = "tool0"
    decision = authorize(
        make_contract(
            evidence=evidence,
            transform=make_transform(**transform_overrides),
        )
    )

    assert decision.allowed is False
    assert decision.checks[failed_check] is False
    assert decision.reason == f"TCP capability rejected: {failed_check}"


def test_runtime_hash_can_be_valid_but_must_match_the_expected_asset():
    decision = authorize(make_contract(), runtime_asset_sha256="f" * 64)

    assert decision.checks["runtime_asset_sha256_valid"] is True
    assert decision.checks["runtime_asset_sha256_matches"] is False
    assert decision.reason == "TCP capability rejected: runtime_asset_sha256_matches"


@pytest.mark.parametrize(
    "runtime_hash",
    [None, 123, b"0" * 64, "", "0" * 63, "0" * 65, "F" * 64, "g" * 64],
)
def test_runtime_hash_requires_a_lowercase_64_hex_string(runtime_hash: object):
    decision = authorize(make_contract(), runtime_asset_sha256=runtime_hash)

    assert decision.checks["runtime_asset_sha256_valid"] is False
    assert decision.checks["runtime_asset_sha256_matches"] is False


def test_calibration_hash_can_be_valid_but_must_match_the_transform():
    decision = authorize(make_contract(), calibration_artifact_sha256="c" * 64)

    assert decision.checks["calibration_artifact_sha256_valid"] is True
    assert decision.checks["calibration_artifact_sha256_matches"] is False
    assert decision.reason == (
        "TCP capability rejected: calibration_artifact_sha256_matches"
    )


@pytest.mark.parametrize(
    "calibration_hash",
    [None, 123, b"0" * 64, "", "0" * 63, "0" * 65, "B" * 64, "g" * 64],
)
def test_calibration_hash_requires_a_lowercase_64_hex_string(calibration_hash: object):
    decision = authorize(make_contract(), calibration_artifact_sha256=calibration_hash)

    assert decision.checks["calibration_artifact_sha256_valid"] is False
    assert decision.checks["calibration_artifact_sha256_matches"] is False


def test_decision_is_frozen_and_checks_are_immutable():
    decision = authorize(make_contract())

    assert is_dataclass(decision)
    with pytest.raises(FrozenInstanceError):
        decision.allowed = False  # type: ignore[misc]
    with pytest.raises(TypeError):
        decision.checks["backend_is_isaac"] = False  # type: ignore[index]


def test_authorization_does_not_mutate_contract_or_transform():
    transform = make_transform()
    contract = make_contract(transform=transform)
    contract_before = contract
    transform_before = contract.transform

    decision = authorize(contract)

    assert decision.transform is transform_before
    assert contract is contract_before
    assert contract.transform is transform_before
    assert contract == contract_before
    assert transform == transform_before


@pytest.mark.parametrize(
    "decision",
    [
        authorize(make_contract()),
        authorize(make_contract(), backend="ISAAC"),
        authorize(make_contract(), runtime_asset_sha256="bad"),
    ],
)
def test_allowed_is_exactly_all_check_values(decision: TcpCapabilityDecision):
    assert decision.allowed == all(decision.checks.values())


def test_failure_reason_follows_check_insertion_order():
    decision = authorize(
        make_contract(
            robot_model="wrong-model",
            evidence=(),
            transform=make_transform(
                source="wrong-source",
                parent_frame="base_link",
                child_frame="tool0",
            ),
        ),
        backend="ISAAC",
        runtime_asset_sha256="bad",
    )

    assert decision.reason == (
        "TCP capability rejected: backend_is_isaac, robot_model_matches, "
        "project_calibration_evidence_present, simulation_only_source_matches, "
        "parent_frame_matches, child_frame_matches, runtime_asset_sha256_valid, "
        "runtime_asset_sha256_matches"
    )


def test_source_has_only_allowed_imports_and_no_wiring_or_artifact_access():
    source_path = Path(__file__).parents[1] / "src/rocobrick/tcp_capability.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    allowed_stdlib = {"collections.abc", "dataclasses", "types"}
    contract_imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            pytest.fail("plain imports are not allowed")
        if isinstance(node, ast.ImportFrom):
            assert node.module is not None
            if node.module == "rocobrick.tcp_contract":
                contract_imports.update(alias.name for alias in node.names)
            else:
                assert node.module in allowed_stdlib
    assert contract_imports == {
        "EXPECTED_RUNTIME_MODEL_SHA256",
        "TcpContract",
        "TcpStatus",
        "Transform",
    }

    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not called_names.intersection(
        {"assess", "assess_piper_l_tcp", "require", "require_resolved_tcp"}
    )
    assert "0.1358" not in source
    assert "operation" not in source.lower()
    assert "moveit" not in source.lower()
    assert "ros" not in source.lower()

    exported = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        )
    )
    assert ast.literal_eval(exported) == ["TcpCapabilityDecision", "authorize_tcp"]
