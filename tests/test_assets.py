from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from tools.validate_piper_assets import validate


ROOT = Path(__file__).parents[1]
OFFICIAL = (
    ROOT
    / "planner_ws/src/agx_arm_description/agx_arm_urdf/piper_l/urdf/piper_l_description.urdf"
)


def test_runtime_urdf_uses_pinned_official_joint_limits():
    validate(ROOT / "robot_assets/piper_l/piper_l_gripper_d435.urdf", OFFICIAL)


def test_generated_dual_model_has_unique_names_and_required_groups(tmp_path):
    urdf = tmp_path / "dual.urdf"
    srdf = tmp_path / "dual.srdf"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "planner_ws/src/rocobrick_moveit_config/scripts/generate_dual_model.py"),
            "--single",
            str(OFFICIAL),
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
