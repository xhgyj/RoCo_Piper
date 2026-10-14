"""Offline contracts for shared elevated loose-brick feeding platforms."""

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_runtime_host_inputs import config, plan

from rocobrick.feeding_platforms import (
    FeedingPlatform,
    allocate_feeding_slots,
    parse_feeding_platforms,
)
from rocobrick.runtime_host_inputs import (
    RuntimeHostInputError,
    build_runtime_host_inputs,
)

ROOT = Path(__file__).parents[1]
BASES = {"piper_0": (0.24, -0.26), "piper_1": (-0.35, -0.30)}


def platform_document():
    """Return the shared two-region configuration.

    Returns:
        Two equally elevated complete platform descriptions.
    """
    return json.loads((ROOT / "config/user_config_type1_scaled2_C.json").read_text())[
        "Env_Config"
    ]["Feeding_Platforms"]


def allocate(identities, **changes):
    """Allocate small bricks with no runtime or motion calls.

    Returns:
        Original brick identities mapped to supported slots.
    """
    arguments = {
        "platforms": parse_feeding_platforms(platform_document(), tuple(BASES)),
        "footprints_xy_m": {identity: (0.032, 0.032) for identity in identities},
        "preferred_robots": dict.fromkeys(identities, "piper_0"),
        "robot_base_xy": BASES,
        "board_center_xy": (-0.048, -0.034),
        "board_half_extent_xy": (0.192, 0.192),
    }
    arguments.update(changes)
    return allocate_feeding_slots(identities, **arguments)


def test_balances_all_loose_bricks_without_changing_robot_allocation():
    """Use both platforms even when all original operations belong to one arm."""
    identities = ("brick-d", "brick-b", "brick-a", "brick-c")
    slots = allocate(identities)
    assert set(slots) == set(identities)
    assert [slot.platform.robot_id for slot in slots.values()] == [
        "piper_0",
        "piper_1",
        "piper_0",
        "piper_1",
    ]
    assert all(slot.preferred_robot_id == "piper_0" for slot in slots.values())
    assert all(slot.platform.top_z_m == 0.08 for slot in slots.values())
    assert allocate(tuple(reversed(identities))) == slots
    json.dumps({identity: slot.evidence() for identity, slot in slots.items()})


def test_single_brick_is_not_duplicated_and_moves_farther_from_base():
    """A one-brick partial plan still creates two platforms but one physical brick."""
    slots = allocate(("brick-a",))
    assert len(slots) == 1
    slot = slots["brick-a"]
    assert slot.position_xy_m == (0.35, 0.06)
    assert slot.distance_to_robot_base_m > 0.30
    assert slot.distance_to_robot_base_m > math.dist((0.252, -0.10), BASES["piper_0"])


def test_slots_support_different_footprints_and_leave_spacing():
    """Keep complete rectangular footprints on each shared platform."""
    sizes = {f"brick-{i}": (0.032, 0.064) for i in range(6)}
    slots = allocate(tuple(sizes), footprints_xy_m=sizes)
    assert len({slot.platform.robot_id for slot in slots.values()}) == 2
    for identity, slot in slots.items():
        for axis in (0, 1):
            low = slot.position_xy_m[axis] - sizes[identity][axis] / 2
            high = slot.position_xy_m[axis] + sizes[identity][axis] / 2
            assert low >= slot.platform.minimum_m[axis] + 0.01 - 1e-9
            assert high <= slot.platform.maximum_m[axis] - 0.01 + 1e-9
    values = list(slots.values())
    assert all(
        math.dist(first.position_xy_m, last.position_xy_m) >= 0.12
        for index, first in enumerate(values)
        for last in values[index + 1 :]
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("Top_Z", True),
        ("Thickness", -1),
        ("Size_XY", [0, 0.24]),
        ("Center_XY", [float("nan"), 0]),
        ("Robot_ID", "other"),
    ],
)
def test_invalid_platform_configuration_fails_closed(field, value):
    """Reject invalid support geometry before loading a scene."""
    document = platform_document()
    document[0][field] = value
    with pytest.raises(ValueError):
        parse_feeding_platforms(document, tuple(BASES))


def test_overlap_incomplete_robot_set_and_capacity_fail_closed():
    """No overlapping or unsupported fallback slots may be manufactured."""
    document = platform_document()
    with pytest.raises(ValueError, match="every configured robot"):
        parse_feeding_platforms(document[:1], tuple(BASES))
    document[1]["Center_XY"] = document[0]["Center_XY"]
    with pytest.raises(ValueError, match="overlap"):
        parse_feeding_platforms(document, tuple(BASES))
    with pytest.raises(ValueError, match="capacity"):
        allocate(("brick-a",), footprints_xy_m={"brick-a": (0.30, 0.03)})
    with pytest.raises(ValueError, match="capacity"):
        allocate(tuple(f"brick-{i}" for i in range(40)))
    platform = FeedingPlatform("piper_0", (0.0, -0.05), (0.24, 0.24), 0.08, 0.08)
    with pytest.raises(ValueError, match="baseplate"):
        allocate(("brick-a",), platforms=(platform,))


def test_runtime_inputs_elevate_only_loose_parts_and_preserve_inputs(monkeypatch):
    """Supply height and full-platform geometry without touching assembled parts."""
    import rocobrick.runtime_host_inputs as module

    value = config()
    for raw in value["Robot_Config"]["Robots"]:
        raw["Robot_Base_Frame"]["Position"] = [*BASES[raw["Name"]], 0.0]
    value["Env_Config"]["Feeding_Platforms"] = platform_document()
    before = copy.deepcopy(value)
    topology = SimpleNamespace(
        loose={"parts": [{"id": 2, "payload": {"W": 4, "L": 8, "H": 6}}]},
        instance_by_topology_id={1: "brick-old", 2: "brick-new"},
    )
    monkeypatch.setattr(module, "build_bricksim_topologies", lambda *args: topology)
    built = build_runtime_host_inputs(plan(), value, config_dir=ROOT / "config")
    scene = built.scene_config
    assert len(scene.feeding_platforms) == 2
    assert scene.initial_part_support_top_z_m == {"brick-new": 0.08}
    assert set(scene.initial_part_positions_xy) == {"brick-new"}
    assert scene.initial_part_allocation["brick-new"]["platform_robot_id"] == "piper_1"
    assert value == before
    assert plan().initially_assembled == {"brick-old"}
    value["Env_Config"]["Feeding_Platforms"][0]["Thickness"] = 0.2
    with pytest.raises(RuntimeHostInputError, match="feeding"):
        build_runtime_host_inputs(plan(), value, config_dir=ROOT / "config")


def test_platforms_are_two_whole_static_collision_boxes(monkeypatch):
    """Author full platform support, not per-brick pillars or floating bricks."""
    import sys

    from rocobrick.isaac_scene import _create_feeding_platforms

    authored = {}

    def cube(stage, path):
        authored[path] = {}
        return SimpleNamespace(
            CreateSizeAttr=lambda value: authored[path].update(size=value),
            AddTranslateOp=lambda: SimpleNamespace(
                Set=lambda value: authored[path].update(center=value)
            ),
            AddScaleOp=lambda: SimpleNamespace(
                Set=lambda value: authored[path].update(scale=value)
            ),
            CreateDisplayColorAttr=lambda value: authored[path].update(color=value),
            GetPrim=lambda: path,
        )

    def collision(path):
        return SimpleNamespace(
            CreateCollisionEnabledAttr=lambda value: authored[path].update(
                collision_enabled=value
            )
        )

    pxr = SimpleNamespace(
        Gf=SimpleNamespace(Vec3d=lambda *args: args, Vec3f=lambda *args: args),
        UsdGeom=SimpleNamespace(
            Cube=SimpleNamespace(Define=cube)
        ),
        UsdPhysics=SimpleNamespace(CollisionAPI=SimpleNamespace(Apply=collision)),
    )
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    stage = SimpleNamespace(
        GetPrimAtPath=lambda path: SimpleNamespace(IsValid=lambda: False)
    )
    platforms = parse_feeding_platforms(platform_document(), tuple(BASES))
    _create_feeding_platforms(stage, platforms)
    assert len(authored) == 2
    for platform in platforms:
        shape = authored[platform.prim_path]
        assert shape["scale"] == (0.24, 0.24, 0.08)
        assert shape["center"] == (*platform.center_xy_m, 0.04)
        assert shape["collision_enabled"] is True
        assert shape["size"] == 1.0
