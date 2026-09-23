import copy
from pathlib import Path

import pytest

import rocobrick.runtime_host_inputs as module
from rocobrick.isaac_scene import IsaacSceneConfig
from rocobrick.stap_contract import parse_stap_plan
from test_stap_contract import payload


class Topology:
    pass


def plan():
    return parse_stap_plan(payload())


def config():
    def robot(name, prim, pos):
        return {
            "Name": name,
            "Robot_USD_Path": "../robot_assets/piper_l/USD/piper_l_gripper_d435.usd",
            "Robot_Prim_Path": prim,
            "Joint_Order": ["j1", "j2"],
            "Robot_Base_Frame": {"Position": pos, "Orientation": [1, 0, 0, 0]},
            "Camera_Config": {"wrist": {"Prim_Path": prim + "/camera"}},
        }
    return {
        "Task_Config": {"Base_Plate": {"Dimension": [32, 32], "Position": [0, 0, 0], "Orientation": [1, 0, 0, 0], "Color": "Gray"}},
        "Robot_Config": {"Robots": [robot("piper_0", "/World/Robot_0", [0, 0, 0]), robot("piper_1", "/World/Robot_1", [1, 0, 0])]},
        "Env_Config": {"Storage_Config": {"Size": [0.2, 0.1, 0.1], "Position": [0, 0, 0.05], "Orientation": [1, 0, 0, 0]}},
    }


def install_fake(monkeypatch):
    monkeypatch.setattr(module, "build_bricksim_topologies", lambda plan, base: Topology())


def test_builds_all_runtime_inputs_and_resolves_usd(monkeypatch, tmp_path):
    install_fake(monkeypatch)
    result = module.build_runtime_host_inputs(plan(), config(), config_dir=tmp_path)
    assert isinstance(result.scene_config, IsaacSceneConfig)
    assert set(result.robots) == {"piper_0", "piper_1"}
    assert result.robots["piper_0"].usd_path == str((tmp_path / "../robot_assets/piper_l/USD/piper_l_gripper_d435.usd").resolve())
    assert result.scene_config.workspace_path == "/World/LegoWorkspace"
    with pytest.raises(TypeError):
        result.robots["extra"] = object()


@pytest.mark.parametrize("change, message", [
    (lambda c: c["Robot_Config"].update(Robots=c["Robot_Config"]["Robots"][:1]), "robot config"),
    (lambda c: c["Task_Config"]["Base_Plate"].update(Dimension=[32, 1.5]), "integral"),
    (lambda c: c["Env_Config"]["Storage_Config"].update(Prim_Path="relative"), "workspace"),
])
def test_rejects_invalid_runtime_inputs(monkeypatch, change, message):
    install_fake(monkeypatch)
    value = config()
    change(value)
    with pytest.raises(module.RuntimeHostInputError, match=message):
        module.build_runtime_host_inputs(plan(), value, config_dir=Path("/tmp"))


def test_rejects_duplicate_robot(monkeypatch):
    install_fake(monkeypatch)
    value = config()
    value["Robot_Config"]["Robots"].append(copy.deepcopy(value["Robot_Config"]["Robots"][0]))
    with pytest.raises(module.RuntimeHostInputError, match="duplicate"):
        module.build_runtime_host_inputs(plan(), value, config_dir=Path("/tmp"))


def test_rejects_camera_outside_robot(monkeypatch):
    install_fake(monkeypatch)
    value = config()
    value["Robot_Config"]["Robots"][0]["Camera_Config"]["wrist"]["Prim_Path"] = "/World/Other/camera"
    with pytest.raises(module.RuntimeHostInputError, match="outside"):
        module.build_runtime_host_inputs(plan(), value, config_dir=Path("/tmp"))


def test_rejects_unknown_robot(monkeypatch):
    install_fake(monkeypatch)
    value = config()
    value["Robot_Config"]["Robots"][0]["Name"] = "unknown"
    with pytest.raises(module.RuntimeHostInputError, match="not present"):
        module.build_runtime_host_inputs(plan(), value, config_dir=Path("/tmp"))
