from pathlib import Path
import importlib.util
import xml.etree.ElementTree as ET


def _module():
    path = (
        Path(__file__).parents[1]
        / "planner_ws/src/rocobrick_moveit_config/scripts/generate_dual_model.py"
    )
    spec = importlib.util.spec_from_file_location("generate_dual_model", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dual_model_uses_bricksim_base_and_tcp_transforms(tmp_path):
    source = tmp_path / "single.urdf"
    source.write_text(
        '<robot name="piper"><link name="world"/><link name="base_link"/>'
        '<link name="link6"/><link name="flange_link"/>'
        '<joint name="world_to_base_link" type="fixed">'
        '<parent link="world"/><child link="base_link"/></joint></robot>',
        encoding="utf-8",
    )
    output = tmp_path / "dual.urdf"
    _module().generate_urdf(source, output)
    root = ET.parse(output).getroot()
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}

    for prefix, x in (("piper_0_", "0.25"), ("piper_1_", "-0.25")):
        base = joints[prefix + "world_to_base"]
        assert base.find("origin").attrib == {
            "xyz": f"{x} -0.25 0",
            "rpy": "0 0 1.57079632679",
        }
        tcp = joints[prefix + "tcp_joint"]
        assert tcp.find("parent").attrib["link"] == prefix + "link6"
        assert tcp.find("origin").attrib["xyz"] == "0 0 0.1358"
