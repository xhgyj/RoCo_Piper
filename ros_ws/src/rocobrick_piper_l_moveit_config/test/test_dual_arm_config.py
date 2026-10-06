"""Offline tests for the combined dual-arm MoveIt model generator."""

from pathlib import Path
from types import SimpleNamespace

import yaml

from rocobrick_piper_l_moveit_config.launch.dual_arm_config import (
    DUAL_ARM_GROUP,
    DUAL_ARM_TCP_GROUP,
    combine_joint_limits,
    combine_moveit_controllers,
    combine_srdfs,
    load_base_poses,
    combine_urdfs,
)


def config(tmp_path: Path, robot_id: str) -> SimpleNamespace:
    """Create one minimal prefixed model/config fixture."""
    directory = tmp_path / robot_id
    directory.mkdir()
    (directory / "model.urdf").write_text(
        f"""<robot name=\"piper_l\">
<link name=\"{robot_id}_world\"/>
<joint name=\"{robot_id}_world_to_base_link\" type=\"fixed\"><parent link=\"{robot_id}_world\"/><child link=\"{robot_id}_base_link\"/></joint>
<link name=\"{robot_id}_base_link\"/><link name=\"{robot_id}_link6\"/><link name=\"{robot_id}_rocobrick_tcp\"/>
<joint name=\"{robot_id}_joint1\" type=\"revolute\"><parent link=\"{robot_id}_base_link\"/><child link=\"{robot_id}_link6\"/></joint>
<joint name=\"{robot_id}_tcp_joint\" type=\"fixed\"><parent link=\"{robot_id}_link6\"/><child link=\"{robot_id}_rocobrick_tcp\"/></joint>
</robot>""",
        encoding="utf-8",
    )
    (directory / "model.srdf").write_text(
        f"<robot name=\"piper_l\"><group name=\"{robot_id}_arm\"/><group name=\"{robot_id}_arm_tcp\"/></robot>",
        encoding="utf-8",
    )
    (directory / "kinematics.yaml").write_text(
        yaml.safe_dump({f"{robot_id}_arm": {"solver": "kdl"}}), encoding="utf-8"
    )
    (directory / "joint_limits.yaml").write_text(
        yaml.safe_dump({"joint_limits": {f"{robot_id}_joint1": {"max": 1}}}),
        encoding="utf-8",
    )
    (directory / "moveit_controllers.yaml").write_text(
        yaml.safe_dump(
            {
                "moveit_simple_controller_manager": {
                    "controller_names": ["arm_controller"],
                    "arm_controller": {
                        "joints": ["joint1"],
                        "action_ns": "follow_joint_trajectory",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(
        robot_id=robot_id,
        urdf=directory / "model.urdf",
        srdf=directory / "model.srdf",
        kinematics=directory / "kinematics.yaml",
        joint_limits=directory / "joint_limits.yaml",
        moveit_controllers=directory / "moveit_controllers.yaml",
    )


def test_base_pose_loader_binds_hash_and_wxyz_rotation(tmp_path):
    """Configured yaw is not replaced by a hardcoded identity rotation."""
    import hashlib
    import json
    import math
    import pytest
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps({"Robot_Config": {"Robots": [{"Name": "piper_0", "Robot_Base_Frame": {"Position": [0.25, -0.25, 0.0], "Orientation": [0.7071, 0, 0, 0.7071]}}]}}))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    xyz, rpy = load_base_poses(path, digest, ("piper_0",))["piper_0"]
    assert xyz == (0.25, -0.25, 0.0)
    assert rpy[2] == pytest.approx(math.pi / 2)
    with pytest.raises(ValueError, match="hash"):
        load_base_poses(path, "a" * 64, ("piper_0",))


def test_combined_urdf_has_one_root_and_real_base_poses(tmp_path: Path) -> None:
    """The combined model removes per-robot roots and preserves base poses."""
    configs = (config(tmp_path, "piper_0"), config(tmp_path, "piper_1"))
    xml = combine_urdfs(
        configs,
        {
            "piper_0": ((0.25, -0.25, 0.0), (0.0, 0.0, 0.0)),
            "piper_1": ((-0.25, -0.25, 0.0), (0.0, 0.0, 0.0)),
        },
    )
    assert xml.count('link name="dual_world"') == 1
    assert "piper_0_world" not in xml
    assert "piper_1_world" not in xml
    assert 'xyz="0.25 -0.25 0.0"' in xml
    assert 'xyz="-0.25 -0.25 0.0"' in xml
    assert xml.count('name="piper_0_dual_world_joint"') == 1
    assert xml.count('name="piper_1_dual_world_joint"') == 1


def test_combined_srdf_has_joint_groups(tmp_path: Path) -> None:
    """The SRDF contains both subgroups and the joint-space dual group."""
    xml = combine_srdfs((config(tmp_path, "piper_0"), config(tmp_path, "piper_1")))
    assert f'name="{DUAL_ARM_GROUP}"' in xml
    assert f'name="{DUAL_ARM_TCP_GROUP}"' in xml
    assert xml.count('name="piper_0_arm_tcp"') == 3
    assert xml.count('name="piper_1_arm_tcp"') == 3


def test_combined_limits_and_controllers_prefix_every_joint(tmp_path: Path) -> None:
    """Controller and limit documents retain robot-specific joint identity."""
    configs = (config(tmp_path, "piper_0"), config(tmp_path, "piper_1"))
    limits = combine_joint_limits(configs)
    assert set(limits["joint_limits"]) == {"piper_0_joint1", "piper_1_joint1"}
    controllers = combine_moveit_controllers(configs)
    names = controllers["moveit_simple_controller_manager"]["controller_names"]
    assert names == ["piper_0_arm_controller", "piper_1_arm_controller"]
    assert controllers["moveit_simple_controller_manager"][names[1]]["joints"] == [
        "piper_1_joint1"
    ]
