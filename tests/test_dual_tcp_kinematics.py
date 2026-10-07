"""Combined TCP solver selection tests without launching ROS or controllers."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


def generator():
    """Load the production config generator without requiring a ROS install.

    Returns:
        The same module consumed by the combined MoveGroup launch.
    """
    path = (
        Path(__file__).parents[1]
        / "ros_ws/src/rocobrick_piper_l_moveit_config/launch/dual_arm_config.py"
    )
    name = "test_dual_tcp_config_generator"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configs(tmp_path, *, unprefixed=False):
    """Create explicit overlapping arm/TCP solver inputs for both robots.

    Returns:
        Captured per-robot YAML paths; no fallback solver is invented.
    """
    result = []
    for robot in ("piper_0", "piper_1"):
        solver = {
            "kinematics_solver": "kdl_kinematics_plugin/KDLKinematicsPlugin",
            "kinematics_solver_timeout": 0.005,
            "kinematics_solver_search_resolution": 0.005,
        }
        prefix = "" if unprefixed else robot + "_"
        path = tmp_path / (robot + ".yaml")
        path.write_text(
            yaml.safe_dump(
                {
                    prefix + "arm": solver,
                    prefix + "arm_tcp": solver,
                    prefix + "gripper": solver,
                }
            )
        )
        result.append(SimpleNamespace(robot_id=robot, kinematics=path))
    return tuple(result)


@pytest.mark.parametrize("unprefixed", [True, False])
def test_combined_ik_selects_tcp_chains_without_overlapping_flange_allocators(
    tmp_path, unprefixed
):
    """Humble's greedy compound IK must not consume TCP joints via *_arm."""
    captured = configs(tmp_path, unprefixed=unprefixed)
    originals = [c.kinematics.read_bytes() for c in captured]
    result = generator().combine_kinematics(captured)
    assert set(result) == {"piper_0_arm_tcp", "piper_1_arm_tcp"}
    assert all(v["kinematics_solver_timeout"] == 0.005 for v in result.values())
    assert [c.kinematics.read_bytes() for c in captured] == originals
    result["piper_0_arm_tcp"]["kinematics_solver_timeout"] = 9.0
    assert captured[0].kinematics.read_bytes() == originals[0]


@pytest.mark.parametrize("fault", ["missing", "empty", "not-mapping", "duplicate"])
def test_combined_ik_rejects_missing_or_invalid_tcp_solver(tmp_path, fault):
    """No implicit flange fallback may silently disable the TCP sampler."""
    captured = configs(tmp_path)
    data = yaml.safe_load(captured[0].kinematics.read_text())
    if fault == "missing":
        data.pop("piper_0_arm_tcp")
    elif fault == "empty":
        data["piper_0_arm_tcp"]["kinematics_solver"] = ""
    elif fault == "not-mapping":
        data = ["piper_0_arm_tcp"]
    else:
        data["arm_tcp"] = data["piper_0_arm_tcp"]
    captured[0].kinematics.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        generator().combine_kinematics(captured)


def test_combined_ik_rejects_empty_robot_set():
    """A dual group cannot advertise capability without robot solvers."""
    with pytest.raises(ValueError):
        generator().combine_kinematics(())
