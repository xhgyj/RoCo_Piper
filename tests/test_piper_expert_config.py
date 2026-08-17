"""Regression tests for the Piper TCP and wrist-camera configuration."""

import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def test_grasp_tcp_is_fixed_at_fingertip_plane():
    """Ensure the IK frame stays at the measured finger-tip plane."""
    tree = ElementTree.parse(
        ROOT / "robot_assets/piper_l/piper_l_gripper_d435.urdf"
    )
    joint = tree.find("./joint[@name='grasp_tcp_joint']")
    assert joint is not None
    assert joint.attrib["type"] == "fixed"
    assert joint.find("parent").attrib["link"] == "link6"
    assert joint.find("child").attrib["link"] == "grasp_tcp"
    xyz = np.fromstring(joint.find("origin").attrib["xyz"], sep=" ")
    np.testing.assert_allclose(xyz, [0.0, 0.0, 0.0593], atol=1e-9)


def test_both_arms_use_runtime_wrist_cameras():
    """Ensure both cameras are mounted, unique, and face along the tool."""
    config = json.loads((ROOT / "config/user_config.json").read_text())
    robots = config["Robot_Config"]["Robots"]
    paths = set()
    for robot in robots:
        assert robot["EE_Frames"] == ["grasp_tcp"]
        camera = robot["Camera_Config"]["Wrist_Camera"]
        assert camera["Prim_Path"].startswith(camera["Mount_Prim_Path"] + "/")
        assert camera["Resolution"] == [640, 480]
        paths.add(camera["Prim_Path"])
        w, x, y, z = camera["Local_Orientation"]
        rotation = Rotation.from_quat([x, y, z, w])
        forward = rotation.apply([0.0, 0.0, -1.0])
        assert forward[2] > 0.9
    assert len(paths) == len(robots)
