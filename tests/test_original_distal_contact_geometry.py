"""Independent geometry regression for the original-mounted distal fingers."""

from __future__ import annotations

import json
import math
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from rocobrick.gripper import (
    PIPER_L_PAD_SEPARATION_INTERCEPT_M,
    pad_separation_to_logical_opening,
)
from rocobrick.skills.pick import FINGER_TIP_BELOW_TCP_M, PickHandler

ROOT = Path(__file__).resolve().parents[1]
MESHES = ROOT / "robot_assets/piper_l/agx_arm_description/agx_arm_urdf/piper_l/meshes"


def test_original_mount_and_distal_surface_agree_across_visual_and_collision():
    """The separately materialized visual pad matches the collision patch."""
    urdf = ET.parse(ROOT / "robot_assets/piper_l/piper_l_gripper_d435.urdf")
    origin = urdf.find("joint[@name='gripper_joint1']/origin")
    assert origin.attrib == {"xyz": "0 0 0.1358", "rpy": "1.5707963 0 0"}
    data = (MESHES / "gripper_link1.stl").read_bytes()
    candidates = []
    for index in range(struct.unpack_from("<I", data, 80)[0]):
        values = struct.unpack_from("<12fH", data, 84 + 50 * index)
        points = [values[3 + k * 3:6 + k * 3] for k in range(3)]
        if values[2] < -0.999 and all(abs(point[2]) < 1e-7 for point in points):
            candidates.append((index, points))
    assert [index for index, _ in candidates] == [545, 546, 547, 548]
    collision_points = [point for _, points in candidates for point in points]
    visual = ET.parse(MESHES / "dae/gripper_link1.dae")
    ns = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
    mesh = next(item for item in visual.findall(".//c:mesh", ns)
                if any(poly.attrib["count"] == "4"
                       for poly in item.findall("c:triangles", ns)))
    sources = {item.attrib["id"]: item for item in mesh.findall("c:source", ns)}
    vertices = {item.attrib["id"]: item for item in mesh.findall("c:vertices", ns)}
    independent_pads = [item for item in mesh.findall("c:triangles", ns)
                        if item.attrib["count"] == "4"]
    assert len(independent_pads) == 1
    pad = independent_pads[0]
    inputs = pad.findall("c:input", ns)
    stride = max(int(item.attrib["offset"]) for item in inputs) + 1
    vertex = next(item for item in inputs if item.attrib["semantic"] == "VERTEX")
    source = vertices[vertex.attrib["source"][1:]].find("c:input", ns)
    coordinates = [float(value) for value in sources[source.attrib["source"][1:]]
                   .find("c:float_array", ns).text.split()]
    indices = [int(value) for value in pad.find("c:p", ns).text.split()]
    offset = int(vertex.attrib["offset"])
    visual_points = [coordinates[3 * indices[i + offset]:3 * indices[i + offset] + 3]
                     for i in range(0, len(indices), stride)]
    for point in collision_points:
        assert min(math.dist(point, other) for other in visual_points) < 1e-7
    source_path = ROOT / "config/calibration/piper_l_tcp_source.json"
    source_doc = json.loads(source_path.read_text())
    centroid = source_doc["sources"]["gripper_link1"]["patch_centroid_m"]
    # Rx(+90) maps mesh y to tool z; tool +z points down in the grasp pose.
    tcp_depth = 0.1358 + centroid[1]
    assert tcp_depth == pytest.approx(0.13058958, abs=1e-8)
    assert FINGER_TIP_BELOW_TCP_M == pytest.approx(0.1358 - tcp_depth, abs=1e-6)
    assert PIPER_L_PAD_SEPARATION_INTERCEPT_M == 0.0
    assert pad_separation_to_logical_opening(0.016) == 0.016


def close_record():
    """Return a safely canceled, settled contact-limited feedback record."""
    return {"runtime": {"execution": {
        "goal_accepted": True, "terminal": True, "safe_to_release": True,
        "action_status": "CANCELED", "details": {
            "measured_opening_m": 0.016, "expected_opening_m": [0.01, 0.022],
            "post_result_state_messages": 3, "contact_limited": True,
            "contact_opening_m": 0.016, "gripper_settled": True,
            "cancel": {"terminal": True, "safe": True},
        },
    }}}


def test_contact_limited_feedback_is_not_itself_holding():
    """A safe close still requires independent physical lift evidence."""
    result = PickHandler._close_feedback(close_record())
    assert result["action_status"] == "CANCELED"
    assert result["contact_confirmed"] is False
    assert result["contact_confirmation_deferred"] == "lift_following"


@pytest.mark.parametrize("defect", ["plain_cancel", "unsafe", "unsettled", "nan",
                                    "bool", "out_of_window", "no_post_state"])
def test_contact_limited_feedback_fails_closed(defect):
    """Missing lifecycle, finite feedback or safe cancellation rejects close."""
    record = close_record()
    execution = record["runtime"]["execution"]
    details = execution["details"]
    if defect == "plain_cancel":
        details["contact_limited"] = False
    elif defect == "unsafe":
        details["cancel"]["safe"] = False
    elif defect == "unsettled":
        details["gripper_settled"] = False
    elif defect == "nan":
        details["contact_opening_m"] = math.nan
    elif defect == "bool":
        details["contact_opening_m"] = True
    elif defect == "out_of_window":
        details["contact_opening_m"] = 0.03
    else:
        details["post_result_state_messages"] = True
    with pytest.raises(ValueError):
        PickHandler._close_feedback(record)
