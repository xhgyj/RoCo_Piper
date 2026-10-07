"""Offline wiring checks; real sampler behavior is covered by the C++ tests."""

import ast
from pathlib import Path
from xml.etree import ElementTree

from rocobrick.runtime_session import DUAL_CONTAINER_LAUNCH_SCRIPT

PACKAGE = Path(__file__).parents[1] / "ros_ws/src/rocobrick_piper_l_moveit_config"


def test_dual_launch_loads_clean_reference_allocator():
    """The actual MoveGroup node must receive the plugin, not only a probe."""
    tree = ast.parse((PACKAGE / "launch/dual_arm_move_group.launch.py").read_text())
    matches = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value == "constraint_samplers":
                    matches.append(ast.literal_eval(value))
    assert matches == ["rocobrick/CleanReferenceSamplerAllocator"]


def test_sampler_plugin_is_exported_and_built():
    """Pluginlib discovery and the shared library name must agree."""
    xml = ElementTree.parse(PACKAGE / "constraint_sampler_plugins.xml").getroot()
    assert xml.attrib["path"] == "rocobrick_clean_reference_sampler"
    plugin = xml.find("class")
    assert plugin.attrib == {
        "name": "rocobrick/CleanReferenceSamplerAllocator",
        "type": "rocobrick::CleanReferenceSamplerAllocator",
        "base_class_type": "constraint_samplers::ConstraintSamplerAllocator",
    }
    cmake = (PACKAGE / "CMakeLists.txt").read_text()
    export = (
        "pluginlib_export_plugin_description_file(moveit_core "
        "constraint_sampler_plugins.xml)"
    )
    assert export in cmake
    assert "add_library(rocobrick_clean_reference_sampler SHARED" in cmake
    assert "test_clean_reference_union_sampler.cpp" in cmake


def test_dual_startup_rebuilds_config_only_install_without_sampler():
    """An old install/setup.bash cannot skip the newly required compiled plugin."""
    guard = DUAL_CONTAINER_LAUNCH_SCRIPT.split("then", 1)[0]
    assert "install/setup.bash" in guard
    assert "lib/librocobrick_clean_reference_sampler.so" in guard
    assert " || " in guard
    assert "timeout 300 colcon build" in DUAL_CONTAINER_LAUNCH_SCRIPT
    assert "--symlink-install" in DUAL_CONTAINER_LAUNCH_SCRIPT
