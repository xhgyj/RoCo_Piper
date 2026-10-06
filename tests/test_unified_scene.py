"""Planning-scene binding and readback regression tests."""

from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
from test_coordinated_ros_runtime import pose, request

from rocobrick.unified_scene import UnifiedSceneBinding


def collision(box):
    """Return a minimal actual GetPlanningScene box object."""
    return NS(
        header=NS(frame_id=box.frame_id),
        primitives=[NS(type=1, dimensions=box.dimensions_m)],
        primitive_poses=[
            NS(
                position=NS(x=box.center_m[0], y=box.center_m[1], z=box.center_m[2]),
                orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0),
            )
        ],
        meshes=[],
        planes=[],
    )


class Transport:
    """Track scene mutations independently of planner request acceptance."""

    def __init__(self):
        self.objects = {}

    def apply_planning_scene(self, update, timeout):
        self.objects = {box.object_id: collision(box) for box in update.boxes}
        return NS(passed=True, details={"readback_confirmed": True})

    def unified_scene_fingerprint(self, timeout):
        from rocobrick.unified_scene import readback_digest

        return readback_digest(self.objects)

    def _resolve_factories(self):
        return object()

    def _read_world_collision_objects(self, timeout, factories):
        return self.objects, None


def test_scene_readback_binds_whole_world_and_detects_external_change():
    """Snapshot digest and actual planning-scene digest are distinct proofs."""
    transport = Transport()
    binding = UnifiedSceneBinding(transport)
    item = replace(
        request({robot: pose(robot, 0.1) for robot in ("piper_0", "piper_1")}),
        scene_snapshot={
            "bricks": {
                "brick": {"aabb_min": [0.0, 0.0, 0.0], "aabb_max": [0.1, 0.1, 0.1]}
            },
            "ground": {"aabb_min": [-1.0, -1.0, -0.01], "aabb_max": [1.0, 1.0, 0.0]},
        },
    )
    proof = binding.sync(item)
    assert proof["snapshot_digest"] == item.scene_digest
    assert proof["scene_digest"] != item.scene_digest
    assert binding.validate(proof)
    transport.objects["brick"].primitive_poses[0].position.x += 0.01
    assert not binding.validate(proof)


def test_holding_without_measured_attachment_fails_before_world_apply():
    """Carrying ownership alone cannot define the collision attachment."""
    item = replace(
        request({robot: pose(robot, 0.1) for robot in ("piper_0", "piper_1")}),
        scene_snapshot={
            "bricks": {
                "brick": {"aabb_min": [0.0, 0.0, 0.0], "aabb_max": [0.1, 0.1, 0.1]}
            },
            "ground": {"aabb_min": [-1.0, -1.0, -0.01], "aabb_max": [1.0, 1.0, 0.0]},
        },
        physical_facts={
            "piper_0": {"holding:brick": {"confirmed": True}},
            "piper_1": {},
        },
    )
    transport = Transport()
    with pytest.raises(ValueError, match="attachment"):
        UnifiedSceneBinding(transport).sync(item)
    assert transport.objects == {}


def test_support_requires_continuous_path_constraint():
    """Stationary upper bricks alone do not prove supporting-arm continuity."""
    item = replace(
        request({robot: pose(robot, 0.1) for robot in ("piper_0", "piper_1")}),
        scene_snapshot={
            "bricks": {
                "brick": {"aabb_min": [0.0, 0.0, 0.0], "aabb_max": [0.1, 0.1, 0.1]}
            }
        },
        physical_facts={
            "piper_0": {},
            "piper_1": {"supporting:brick": {"confirmed": True}},
        },
    )
    with pytest.raises(ValueError, match="continuous"):
        UnifiedSceneBinding(Transport()).sync(item)


def test_missing_ground_envelope_fails_before_apply():
    """A scene containing only the target brick is not a complete world."""
    binding = UnifiedSceneBinding(Transport())
    item = replace(
        request({robot: pose(robot, 0.1) for robot in ("piper_0", "piper_1")}),
        scene_snapshot={
            "bricks": {
                "brick": {"aabb_min": [0.0, 0.0, 0.0], "aabb_max": [0.1, 0.1, 0.1]}
            }
        },
    )
    with pytest.raises(ValueError, match="ground"):
        binding.sync(item)
