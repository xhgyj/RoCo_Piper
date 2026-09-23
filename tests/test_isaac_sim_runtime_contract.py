"""Offline structural and policy tests for the Isaac Sim 5.1 runtime report (P0.2).

These tests never touch the network, pull images, or start containers. They pin
the exact schema, verified tag, undeclared digest, EULA terms, documented host
requirements, ROS 2 bridge facts, headless readiness signals, host facts, open
gaps, and source URLs of ``artifacts/phase0/isaac-sim-5.1-runtime.json`` so the
report stays a faithful, non-fabricated machine record.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
REPORT_PATH = ROOT / "artifacts/phase0/isaac-sim-5.1-runtime.json"

TOP_LEVEL_KEYS = {
    "artifact",
    "phase",
    "date_collected",
    "image",
    "eula",
    "host_requirements",
    "docker_run",
    "ros2",
    "headless",
    "host_facts",
    "gaps",
    "sources",
}

EXPECTED_SOURCES = {
    "container_install": (
        "https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/"
        "install_container.html"
    ),
    "requirements": (
        "https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/"
        "requirements.html"
    ),
    "ros_install": (
        "https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/"
        "install_ros.html"
    ),
    "license_additional": (
        "https://docs.isaacsim.omniverse.nvidia.com/5.1.0/common/"
        "license-isaac-sim-additional.html"
    ),
    "ngc_catalog": "https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim",
}

EXPECTED_GAP_IDS = {
    "compose_missing",
    "digest_undeclared_image_not_pulled",
    "rootless_gpu_unvalidated",
    "eula_compose_pending",
    "plain_headless_readiness_unknown",
    "shm_size_unknown",
    "dds_unvalidated",
    "eol_decision_pending",
    "assets_internet_unknown",
}

HEX64_RE = re.compile(r"\b[0-9a-f]{64}\b")
FLOATING_TAG_RE = re.compile(r":latest\b")


def _reject_non_finite(token: str) -> float:
    """Reject NaN/Infinity constants encountered while parsing JSON."""
    raise ValueError(f"non-finite JSON constant: {token}")


@pytest.fixture
def report_text() -> str:
    """Return the raw artifact text."""
    return REPORT_PATH.read_text(encoding="utf-8")


@pytest.fixture
def report(report_text: str) -> dict:
    """Return the artifact parsed with json.loads (strict, offline)."""
    return json.loads(report_text)


def test_report_loads_as_valid_json(report: dict) -> None:
    """The report must parse as a JSON object."""
    assert isinstance(report, dict)


def test_report_path_exists(report_text: str) -> None:
    """The artifact must be present at the agreed location."""
    assert REPORT_PATH.exists()
    assert report_text.strip().startswith("{")


def test_json_is_finite(report_text: str, report: dict) -> None:
    """Re-parsing must reject NaN/Infinity and re-serialization must stay finite."""
    json.loads(report_text, parse_constant=_reject_non_finite)
    json.dumps(report, allow_nan=False)


def test_top_level_keys_exact(report: dict) -> None:
    """Top level must expose exactly the agreed twelve keys."""
    assert set(report) == TOP_LEVEL_KEYS


def test_phase_and_collection_metadata(report: dict) -> None:
    """The report must be identified as the P0.2 collection of 2026-09-20."""
    assert report["artifact"] == "isaac-sim-5.1-runtime"
    assert report["phase"] == "P0.2"
    assert report["date_collected"] == "2026-09-20"
    assert "read-only" in report["host_facts"]["method"]


def test_image_identity_pins_verified_tag(report: dict) -> None:
    """The image identity must pin the verified 5.1.0 tag, never a floating one."""
    image = report["image"]
    assert image["registry"] == "nvcr.io"
    assert image["repository"] == "nvidia/isaac-sim"
    assert image["tag_verified"] == "5.1.0"
    assert image["reference"] == "nvcr.io/nvidia/isaac-sim:5.1.0"
    assert not FLOATING_TAG_RE.search(image["reference"])


def test_image_digest_is_undeclared_with_reason(report: dict) -> None:
    """The digest must stay not_declared (no NGC auth, no pull) with a reason."""
    image = report["image"]
    assert image["digest"] == "not_declared"
    assert isinstance(image["digest_reason"], str)
    assert image["digest_reason"].strip()
    assert "NGC" in image["digest_reason"]


def test_image_catalog_metadata(report: dict) -> None:
    """The catalog metadata must record signed tag, 9.9GB size, and multiarch."""
    image = report["image"]
    assert image["signed_tag"] is True
    assert image["size_compressed_gb"] == 9.9
    assert image["pulled"] is False
    assert image["multiarch"]["supported"] is True
    assert set(image["multiarch"]["platforms"]) == {"amd64", "arm64"}


def test_image_release_status_records_unsupported_banner(report: dict) -> None:
    """The release status must record that 5.1 is unsupported with newer releases."""
    status = report["image"]["release_status"]
    assert status["supported"] is False
    assert status["newer_release_available"] is True
    assert "no longer supported" in status["banner"]


def test_eula_acceptance_is_mandatory(report: dict) -> None:
    """ACCEPT_EULA=Y must be recorded as mandatory."""
    eula = report["eula"]
    assert eula["accept_env"] == "ACCEPT_EULA"
    assert eula["accept_value"] == "Y"
    assert eula["mandatory"] is True


def test_eula_license_and_redistribution(report: dict) -> None:
    """The license must forbid redistribution, allow internal use, optional privacy."""
    eula = report["eula"]
    assert eula["license_name"] == (
        "NVIDIA Isaac Sim Additional Software and Materials License"
    )
    assert eula["redistribution_allowed"] is False
    assert eula["internal_use_allowed"] is True
    assert eula["privacy_consent"]["required"] is False
    assert eula["privacy_consent"]["env"] == "PRIVACY_CONSENT"


def test_documented_host_requirements(report: dict) -> None:
    """Documented requirements must match the official 5.1.0 requirements page."""
    req = report["host_requirements"]
    assert req["os"] == ["Ubuntu 22.04", "Ubuntu 24.04"]
    assert req["driver_tested_linux"] == "580.65.06"
    assert req["gpu_recommended"] == "GeForce RTX 4080"
    assert req["gpu_vram_min_gb"] == 16
    assert req["gpu_rt_cores_required"] is True
    assert set(req["gpu_unsupported"]) == {"A100", "H100"}
    assert req["ram_gb"] == 32
    assert req["storage_gb"] == 50
    assert req["toolkit"] == "latest"


def test_docker_run_profile(report: dict) -> None:
    """The documented docker run must match gpus all, host network, rootless user."""
    run = report["docker_run"]
    assert run["image_ref"] == "nvcr.io/nvidia/isaac-sim:5.1.0"
    assert run["gpus"] == "all"
    assert run["network"] == "host"
    assert run["user"] == "1234:1234"
    assert run["rootless"] is True
    assert run["shm_size"] == "not_documented"
    assert run["environment"]["ACCEPT_EULA"] == "Y"
    assert run["gui"]["recommended"] is False
    assert len(run["cache_mounts"]) == 6


def test_ros2_bridge_extension_and_distros(report: dict) -> None:
    """The ROS bridge must be isaacsim.ros2.bridge with Humble/Jazzy on 3.11 only."""
    ros2 = report["ros2"]
    assert ros2["bridge_extension"] == "isaacsim.ros2.bridge"
    assert {"Humble", "Jazzy"} <= set(ros2["distros_supported"])
    assert ros2["python"] == "3.11 only"


def test_ros2_dds_default_and_alternative(report: dict) -> None:
    """Fast DDS must be the default and Cyclone DDS the alternative."""
    ros2 = report["ros2"]
    assert "Fast DDS" in ros2["dds_default"]
    assert "rmw_fastrtps_cpp" in ros2["dds_default"]
    assert "Cyclone DDS" in ros2["dds_alternative"]
    assert "rmw_cyclonedds_cpp" in ros2["dds_alternative"]


def test_ros2_moveit_tutorial_recorded(report: dict) -> None:
    """The MoveIt tutorial URL and path must be recorded."""
    tutorial = report["ros2"]["moveit_tutorial"]
    assert tutorial["url"] == (
        "https://docs.isaacsim.omniverse.nvidia.com/5.1.0/"
        "ros2_tutorials/tutorial_ros2_moveit.html"
    )
    assert tutorial["path"] == "../ros2_tutorials/tutorial_ros2_moveit.html"


def test_headless_run_script_and_readiness_marker(report: dict) -> None:
    """Headless readiness is documented via runheadless.sh -v and its marker."""
    headless = report["headless"]
    assert headless["run_script"] == "./runheadless.sh"
    assert headless["run_flag"] == "-v"
    assert headless["readiness_marker"] == "Isaac Sim Full Streaming App is loaded."


def test_headless_compatibility_check_marker(report: dict) -> None:
    """The compatibility check must record its script and PASSED marker."""
    check = report["headless"]["compatibility_check"]
    assert check["script"] == "./isaac-sim.compatibility_check.sh"
    assert check["marker"] == "System checking result: PASSED"


def test_plain_headless_readiness_undocumented(report: dict) -> None:
    """Plain headless readiness must stay not_documented, not invented."""
    headless = report["headless"]
    assert headless["plain_headless_readiness"] == "not_documented"
    assert headless["plain_headless_readiness_reason"].strip()


def test_host_facts_match_workstation(report: dict) -> None:
    """Host facts must record the read-only inspection of this workstation."""
    facts = report["host_facts"]
    assert facts["collected"] == "2026-09-20"
    assert facts["os_release"] == "Ubuntu 22.04.5"
    assert facts["kernel"] == "6.8.0-136-generic"
    assert facts["arch"] == "x86_64"
    assert facts["cpu_logical"] == 32
    assert facts["ram_gib"] == 125.5
    assert facts["gpu_name"] == "RTX 4090 D"
    assert facts["gpu_vram_mib"] == 24564
    assert facts["nvidia_driver"] == "580.178.04"
    assert facts["cuda_version"] == "13.0"


def test_host_container_stack_facts(report: dict) -> None:
    """Host facts must record the container stack without compose or Isaac images."""
    facts = report["host_facts"]
    assert facts["docker_version"] == "29.6.2"
    assert facts["docker_mode"] == "rootless"
    assert facts["nvidia_ctk_version"] == "1.20.0"
    assert facts["nvidia_runtime_present"] is True
    assert facts["cdi_present"] is True
    assert facts["docker_compose_available"] is False
    assert facts["isaac_sim_images_local"] == 0
    assert facts["home_free_gb"] == 164


def test_gap_ids_complete(report: dict) -> None:
    """The report must list exactly the nine agreed open gaps."""
    gaps = report["gaps"]
    ids = {gap["id"] for gap in gaps}
    assert ids == EXPECTED_GAP_IDS
    for gap in gaps:
        assert gap["detail"].strip(), gap["id"]


def test_sources_mapping_exact(report: dict) -> None:
    """The sources must map the five agreed URLs exactly."""
    assert report["sources"] == EXPECTED_SOURCES


def test_all_sources_are_https(report: dict) -> None:
    """Every source URL must use HTTPS."""
    for name, url in report["sources"].items():
        assert url.startswith("https://"), name


def test_docs_sources_pin_version_5_1_0(report: dict) -> None:
    """Every Isaac Sim docs URL must be scoped to the 5.1.0 documentation."""
    for name, url in report["sources"].items():
        if url.startswith("https://docs.isaacsim.omniverse.nvidia.com"):
            assert "/5.1.0/" in url, name


def test_no_floating_latest_tag(report_text: str) -> None:
    """The report must never reference the floating :latest image tag."""
    assert not FLOATING_TAG_RE.search(report_text)


def test_no_fabricated_digest(report_text: str, report: dict) -> None:
    """No hex digest may appear anywhere; the digest stays not_declared."""
    assert report["image"]["digest"] == "not_declared"
    assert not HEX64_RE.search(report_text)
