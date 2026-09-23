"""Authorize the narrowly scoped simulation-only Piper L TCP capability."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from rocobrick.tcp_contract import (
    EXPECTED_RUNTIME_MODEL_SHA256,
    TcpContract,
    TcpStatus,
    Transform,
)

__all__ = ["TcpCapabilityDecision", "authorize_tcp"]

_PROJECT_CALIBRATION_EVIDENCE = "project-calibration-link6-rocobrick-tcp"
_SIMULATION_ONLY_SOURCE = "artifacts/model-audit/piper_l_tcp_calibration.json"
_HEX_DIGITS = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class TcpCapabilityDecision:
    """The result of the strict TCP capability authorization checks."""

    allowed: bool
    status: str
    reason: str
    transform: Transform | None
    checks: Mapping[str, bool]


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX_DIGITS for character in value)
    )


def authorize_tcp(
    contract: TcpContract,
    *,
    backend: str,
    runtime_asset_sha256: str,
    calibration_artifact_sha256: str,
) -> TcpCapabilityDecision:
    """Authorize the exact simulation-only TCP capability contract."""
    contract_type_valid = issubclass(type(contract), TcpContract)
    backend_is_isaac = backend == "isaac"

    if contract_type_valid:
        robot_model_matches = contract.robot_model == "agilex_piper_l"
        contract_resolved = contract.status is TcpStatus.RESOLVED
        transform = contract.transform
        transform_present = isinstance(transform, Transform)
        project_calibration_evidence_present = (
            _PROJECT_CALIBRATION_EVIDENCE in contract.evidence
        )
        simulation_only_source_matches = (
            transform_present and transform.source == _SIMULATION_ONLY_SOURCE
        )
        parent_frame_matches = transform_present and transform.parent_frame == "link6"
        child_frame_matches = (
            transform_present and transform.child_frame == "rocobrick_tcp"
        )
    else:
        robot_model_matches = False
        contract_resolved = False
        transform = None
        transform_present = False
        project_calibration_evidence_present = False
        simulation_only_source_matches = False
        parent_frame_matches = False
        child_frame_matches = False

    runtime_asset_sha256_valid = _valid_sha256(runtime_asset_sha256)
    runtime_asset_sha256_matches = (
        runtime_asset_sha256 == EXPECTED_RUNTIME_MODEL_SHA256
    )
    calibration_artifact_sha256_valid = _valid_sha256(calibration_artifact_sha256)
    calibration_artifact_sha256_matches = (
        transform_present and calibration_artifact_sha256 == transform.sha256
    )

    checks = MappingProxyType(
        {
            "contract_type_valid": contract_type_valid,
            "backend_is_isaac": backend_is_isaac,
            "robot_model_matches": robot_model_matches,
            "contract_resolved": contract_resolved,
            "transform_present": transform_present,
            "project_calibration_evidence_present": (
                project_calibration_evidence_present
            ),
            "simulation_only_source_matches": simulation_only_source_matches,
            "parent_frame_matches": parent_frame_matches,
            "child_frame_matches": child_frame_matches,
            "runtime_asset_sha256_valid": runtime_asset_sha256_valid,
            "runtime_asset_sha256_matches": runtime_asset_sha256_matches,
            "calibration_artifact_sha256_valid": calibration_artifact_sha256_valid,
            "calibration_artifact_sha256_matches": (
                calibration_artifact_sha256_matches
            ),
        }
    )
    allowed = all(checks.values())
    if allowed:
        return TcpCapabilityDecision(
            allowed=True,
            status="TCP_AUTHORIZED",
            reason="Isaac simulation-only TCP capability is authorized.",
            transform=transform,
            checks=checks,
        )

    failed_checks = ", ".join(name for name, passed in checks.items() if not passed)
    return TcpCapabilityDecision(
        allowed=False,
        status="UNRESOLVED_TCP",
        reason="TCP capability rejected: " + failed_checks,
        transform=None,
        checks=checks,
    )
