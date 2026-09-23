from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

from tools.validate_piper_assets import validate


ROOT = Path(__file__).parents[1]
PLANNING_URDF = (
    ROOT
    / "planner_ws/src/agx_arm_description/agx_arm_urdf/piper_l/urdf/piper_l_description.urdf"
)
EXECUTION_URDF = (
    ROOT / "isaac/vendor/agx_arm_sim/agx_arm_description/urdf/piper_l_gripper_d435.urdf"
)


def test_runtime_urdf_matches_pinned_execution_asset():
    validate(ROOT / "robot_assets/piper_l/piper_l_gripper_d435.urdf", EXECUTION_URDF)


def test_widened_runtime_urdf_is_rejected(tmp_path):
    edited = tmp_path / "piper_l_gripper_d435.urdf"
    source = EXECUTION_URDF.read_text(encoding="utf-8")
    edited.write_text(
        source.replace('lower="-1.7453292"', 'lower="-2.2165681"'),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="not byte-identical"):
        validate(edited, EXECUTION_URDF)


def test_generated_dual_model_has_unique_names_and_required_groups(tmp_path):
    urdf = tmp_path / "dual.urdf"
    srdf = tmp_path / "dual.srdf"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "planner_ws/src/rocobrick_moveit_config/scripts/generate_dual_model.py"),
            "--single",
            str(PLANNING_URDF),
            "--urdf",
            str(urdf),
            "--srdf",
            str(srdf),
        ],
        check=True,
    )
    model = ET.parse(urdf).getroot()
    names = [item.get("name") for item in model if item.tag in {"link", "joint"}]
    assert len(names) == len(set(names))
    for prefix in ("piper_0_", "piper_1_"):
        assert f"{prefix}tcp_link" in names
        assert f"{prefix}d435_camera_link" in names
    groups = {item.get("name") for item in ET.parse(srdf).getroot().findall("group")}
    assert groups == {"piper_0_arm", "piper_0_gripper", "piper_1_arm", "piper_1_gripper", "dual_arms"}
    disabled = ET.parse(srdf).getroot().findall("disable_collisions")
    assert any(
        item.get("link1") == "piper_0_link1"
        and item.get("link2") == "piper_0_link2"
        for item in disabled
    )
    assert any(
        item.get("link1") == "piper_1_gripper_link1"
        and item.get("link2") == "piper_1_gripper_link2"
        for item in disabled
    )
