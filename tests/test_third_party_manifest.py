"""Offline structural and policy tests for the third-party lock (P0.1).

These tests never touch the network. They pin the exact schema, repository
commits, license source hashes, verified facts, and README policy text that
make ``third_party/manifest.lock.yaml`` the single source of truth for
pinned third-party sources.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
MANIFEST_PATH = ROOT / "third_party/manifest.lock.yaml"
README_PATH = ROOT / "third_party/README.md"

COMMIT_HEX_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
UTC_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

EXPECTED_REPOSITORIES = {
    "agx_arm_ros": {
        "url": "https://github.com/agilexrobotics/agx_arm_ros.git",
        "branch": "ros2",
        "commit": "b9ad14de2a6eb818a2d206fbd1adba48343662f7",
        "license_id": "MIT",
        "license_source_path": "LICENSE",
        "license_sha256": (
            "247944c63de1573653722ba784f96123da267105b7b48577e0c8826a24b809f6"
        ),
        "destination": "ros_ws/src/agx_arm_ros",
    },
    "agx_arm_sim": {
        "url": "https://github.com/agilexrobotics/agx_arm_sim.git",
        "branch": "master",
        "commit": "f8cd8b147c75d59e14f90fb0646770eefa268ed0",
        "license_id": "MIT",
        "license_source_path": "README.md",
        "license_sha256": (
            "b91918f2972101eee621cba9badbc90e6057b9c5269c0bb7ae00f84195d31399"
        ),
        "destination": "isaac/vendor/agx_arm_sim",
    },
}

REQUIRED_ROS_PACKAGES = {"agx_arm_description", "agx_arm_moveit", "agx_arm_ctrl"}


@pytest.fixture
def manifest() -> dict:
    """Return the manifest parsed with yaml.safe_load (strict, offline)."""
    return yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def readme_text() -> str:
    """Return the third-party README policy text."""
    return README_PATH.read_text(encoding="utf-8")


def test_manifest_loads_with_safe_load(manifest: dict) -> None:
    """The manifest must be plain YAML loadable with yaml.safe_load."""
    assert isinstance(manifest, dict)


def test_manifest_top_level_keys_exact(manifest: dict) -> None:
    """Top level must expose exactly schema/generated_at_utc/repositories."""
    assert set(manifest) == {"schema", "generated_at_utc", "repositories"}
    assert manifest["schema"] == "rocobrick-third-party-lock/v1"


def test_generated_at_utc_is_iso8601_utc(manifest: dict) -> None:
    """generated_at_utc must be an ISO-8601 UTC timestamp ending in Z."""
    stamp = manifest["generated_at_utc"]
    assert isinstance(stamp, str)
    assert UTC_TIMESTAMP_RE.match(stamp)
    datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")


def test_repositories_keys_exact(manifest: dict) -> None:
    """Repositories must contain exactly agx_arm_ros and agx_arm_sim."""
    repositories = manifest["repositories"]
    assert set(repositories) == {"agx_arm_ros", "agx_arm_sim"}


def test_repository_entry_keys_exact(manifest: dict) -> None:
    """Each entry must use exactly the agreed key set."""
    for name, entry in manifest["repositories"].items():
        assert set(entry) == {
            "url",
            "branch",
            "commit",
            "license",
            "purpose",
            "verified_facts",
            "import",
        }, name
        assert set(entry["license"]) == {"id", "source_path", "sha256"}, name
        assert set(entry["import"]) == {"method", "destination", "patches"}, name
        assert isinstance(entry["purpose"], str) and entry["purpose"].strip(), name


def test_repository_hex_formats(manifest: dict) -> None:
    """Commits are 40-char lowercase hex; license hashes 64-char lowercase hex."""
    for name, entry in manifest["repositories"].items():
        assert COMMIT_HEX_RE.match(entry["commit"]), name
        assert SHA256_HEX_RE.match(entry["license"]["sha256"]), name


@pytest.mark.parametrize("name", ["agx_arm_ros", "agx_arm_sim"])
def test_repository_pinned_values(manifest: dict, name: str) -> None:
    """Each repository pins the audited upstream URL, branch, and commit."""
    entry = manifest["repositories"][name]
    expected = EXPECTED_REPOSITORIES[name]
    assert entry["url"] == expected["url"]
    assert entry["branch"] == expected["branch"]
    assert entry["commit"] == expected["commit"]
    assert entry["license"]["id"] == expected["license_id"]
    assert entry["license"]["source_path"] == expected["license_source_path"]
    assert entry["license"]["sha256"] == expected["license_sha256"]


@pytest.mark.parametrize("name", ["agx_arm_ros", "agx_arm_sim"])
def test_repository_import_block(manifest: dict, name: str) -> None:
    """Imports are plain vcs checkouts into the agreed destinations."""
    entry = manifest["repositories"][name]
    assert entry["import"]["method"] == "vcs"
    assert entry["import"]["destination"] == EXPECTED_REPOSITORIES[name]["destination"]
    assert entry["import"]["patches"] == []


def test_ros_verified_facts(manifest: dict) -> None:
    """agx_arm_ros facts pin Humble support, packages, and gripper contract."""
    facts = manifest["repositories"]["agx_arm_ros"]["verified_facts"]
    assert facts["ros2_humble_supported"] is True
    assert REQUIRED_ROS_PACKAGES <= set(facts["packages_present"])
    # Package list ordering must be stable (lexically sorted, no churn).
    assert facts["packages_present"] == sorted(facts["packages_present"])
    gripper = facts["gripper"]
    assert gripper["command_joint"] == "gripper"
    assert len(gripper["mimic_joints"]) == 2
    assert gripper["mimic_joints"] == ["gripper_joint1", "gripper_joint2"]
    assert gripper["opening_range_m"] == [0.0, 0.1]


def test_sim_verified_facts(manifest: dict) -> None:
    """agx_arm_sim facts record the README-only MIT license and re-audit gate."""
    facts = manifest["repositories"]["agx_arm_sim"]["verified_facts"]
    assert facts["supports_piper"] is True
    assert facts["supports_piper_l"] is True
    assert facts["isaac_sim_5_1_compatibility"] == "not_declared"
    assert facts["no_standalone_license_file"] is True
    assert facts["assets_license_reaudit_phase"] == "P0.4"
    # The README-only MIT statement must never be read as USD/mesh clearance.
    assert facts["assets_license_confirmed"] is False


def test_readme_names_manifest_as_single_version_source(readme_text: str) -> None:
    """README must declare manifest.lock.yaml as the unique version source."""
    assert "manifest.lock.yaml" in readme_text
    assert "唯一版本源" in readme_text


def test_readme_forbids_floating_branch_clone(readme_text: str) -> None:
    """README must forbid floating-branch clones and demand exact checkouts."""
    assert "禁止" in readme_text
    assert "浮动 branch" in readme_text
    assert "Dockerfile" in readme_text
    for name, expected in EXPECTED_REPOSITORIES.items():
        assert expected["url"] in readme_text, name
        assert expected["commit"] in readme_text, name
    assert "git rev-parse HEAD" in readme_text


def test_readme_requires_license_hash_verification(readme_text: str) -> None:
    """README must require sha256 verification against the manifest hashes."""
    assert "sha256sum" in readme_text
    assert "license.sha256" in readme_text
    for expected in EXPECTED_REPOSITORIES.values():
        assert expected["license_sha256"] in readme_text


def test_readme_marks_legacy_vendor_reference_only(readme_text: str) -> None:
    """README must flag planner_ws copies as legacy, reference-only vendor."""
    assert "planner_ws/src" in readme_text
    assert "robot_assets" in readme_text
    assert "仅供参考" in readme_text


def test_readme_sim_license_limitation_and_reaudit(readme_text: str) -> None:
    """README must state the sim license limitation and the P0.4 re-audit."""
    assert "没有独立的 LICENSE 文件" in readme_text
    assert "no_standalone_license_file" in readme_text
    assert "assets_license_reaudit_phase: P0.4" in readme_text
    assert "不得把这条许可记录解读为 USD / mesh 资产许可已确认" in readme_text
    assert "isaac_sim_5_1_compatibility: not_declared" in readme_text


def test_readme_readonly_update_git_commands(readme_text: str) -> None:
    """README must document read-only git commands for update checks."""
    ros_remote = (
        "git ls-remote https://github.com/agilexrobotics/agx_arm_ros.git"
        " refs/heads/ros2"
    )
    sim_remote = (
        "git ls-remote https://github.com/agilexrobotics/agx_arm_sim.git"
        " refs/heads/master"
    )
    assert ros_remote in readme_text
    assert sim_remote in readme_text
    assert "git checkout" in readme_text
    assert "git show" in readme_text


def test_readme_states_no_update_scripts(readme_text: str) -> None:
    """README must state that no third-party update scripts are created."""
    assert "不创建" in readme_text
    assert "更新脚本" in readme_text


# The agx_arm_ros locked commit carries a submodule gitlink at
# src/agx_arm_description/agx_arm_urdf (verified via `git ls-tree` on the
# pinned commit). Without recursively initializing submodules the gripper
# facts / URDF / mesh cannot be re-verified from the imported checkout.
ROS_SUBMODULE_PATH = "src/agx_arm_description/agx_arm_urdf"
ROS_SUBMODULE_GITLINK = "f6642ce0d7872c686f29c99e9e10cd23d1d49313"


def _readme_ros_clone_section(readme_text: str) -> str:
    """Return the README bash block spanning the agx_arm_ros clone steps."""
    ros_marker = "git clone https://github.com/agilexrobotics/agx_arm_ros.git"
    sim_marker = "git clone https://github.com/agilexrobotics/agx_arm_sim.git"
    start = readme_text.index(ros_marker)
    end = readme_text.index(sim_marker, start)
    return readme_text[start:end]


def test_readme_ros_checkout_requires_recursive_submodules(readme_text: str) -> None:
    """The agx_arm_ros checkout block must recursively init submodules."""
    assert "git submodule update --init --recursive" in readme_text
    ros_section = _readme_ros_clone_section(readme_text)
    assert "git submodule update --init --recursive" in ros_section


def test_readme_records_ros_submodule_gitlink(readme_text: str) -> None:
    """README must record the agx_arm_urdf gitlink pinned inside agx_arm_ros."""
    assert ROS_SUBMODULE_GITLINK in readme_text
    assert ROS_SUBMODULE_PATH in readme_text
    ros_section = _readme_ros_clone_section(readme_text)
    assert ROS_SUBMODULE_GITLINK in ros_section
    # Legacy vendor stays reference-only and the P0.4 asset re-audit survives.
    assert "仅供参考" in readme_text
    assert "P0.4" in readme_text


def test_manifest_and_readme_do_not_mention_mode_b() -> None:
    """Neither the manifest nor the README may carry the internal mode marker."""
    # Built at runtime so the marker itself never appears verbatim in this file.
    marker = "Mode" + " B"
    for path in (MANIFEST_PATH, README_PATH):
        text = path.read_text(encoding="utf-8")
        assert marker not in text, path
        assert marker.lower() not in text.lower(), path
