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
    np.testing.assert_allclose(xyz, [0.0, 0.0, 0.1358], atol=1e-9)


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
        assert camera["Clipping_Range"][0] <= 0.01
        assert camera["Focal_Length"] > 0.0
    assert len(paths) == len(robots)


def test_both_grippers_have_stable_contact_drives():
    """Long bricks need adequate clamp stiffness without excess max force."""
    config = json.loads((ROOT / "config/user_config.json").read_text())
    for robot in config["Robot_Config"]["Robots"]:
        drive = robot["Gripper_Config"]["Drive"]
        assert drive["Max_Force"] == 10.0
        assert drive["Damping"] == 10.0
        assert drive["Stiffness"] == 500.0


def test_global_camera_is_configured_for_review():
    """Ensure a fixed scene camera is available for collection review."""
    config = json.loads((ROOT / "config/user_config.json").read_text())
    camera = config["Env_Config"]["Camera_Config"]["Global_Camera"]
    assert camera["Prim_Path"] == "/World/Global_Camera"
    assert camera["Resolution"] == [960, 540]
    assert camera["Position"][2] > camera["Target"][2]
    assert camera["Position"][1] > camera["Target"][1]
    assert camera["Clipping_Range"][0] <= 0.01


def test_storage_workspace_fits_every_brick_and_stays_outside_plate():
    """Keep the complete static pickup region valid and expert-reachable."""
    config = json.loads((ROOT / "config/user_config.json").read_text())
    environment = config["Env_Config"]
    storage = environment["Storage_Config"]
    workspace_clearance = 0.016
    longest_brick = 8 * 0.008
    required_side = longest_brick + 2 * workspace_clearance
    assert storage["Size"][0] >= required_side
    assert storage["Size"][1] >= required_side

    base_plate = config["Task_Config"]["Base_Plate"]
    plate_front = (
        base_plate["Position"][1]
        + base_plate["Dimension"][1] * 0.008 * 0.5
    )
    workspace_back = storage["Position"][1] - storage["Size"][1] * 0.5
    assert workspace_back - plate_front >= environment["Loose_Target_Clearance"]


def test_camera_poses_are_applied_in_usd_axes():
    """Prevent feeding USD (-Z-forward) quaternions through world-axis conversion."""
    source = (ROOT / "src/rocobrick/env/Env.py").read_text()
    assert source.count('camera_axes="usd"') >= 2
