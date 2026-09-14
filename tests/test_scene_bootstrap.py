import asyncio

import pytest

from rocobrick.models import (
    AssemblyPlan,
    PartInstance,
    Pose,
    SceneManifest,
    SkillType,
    TaskDefinition,
)
from rocobrick.scene_bootstrap import (
    LoadedPart,
    LoadedRobot,
    RobotSpawnSpec,
    SceneBootstrapError,
    bootstrap_scene,
)


POSE = Pose("world", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))


def inputs():
    task = TaskDefinition(
        "any-id", "moving", 2, POSE,
        "base:top",
        SkillType.PLACE_DOWN, "left", None, frozenset(), 0, 0,
    )
    plan = AssemblyPlan("p", "world", "b", "r", {task.task_id: task})
    scene = SceneManifest("s", "world", {
        "moving": PartInstance("moving", 2, POSE),
        "base": PartInstance("base", 3, POSE),
    })
    spec = RobotSpawnSpec("left", "/World/left", "piper.usd", POSE, ("joint1", "joint2"))
    return plan, scene, {"left": spec}


class FakePort:
    def __init__(self):
        self.calls = []
        self.bad_pose = False
        self.parts = {}

    async def open(self):
        self.calls.append("open")

    async def spawn_robot(self, spec):
        self.calls.append(("robot", spec.robot_id))
        return LoadedRobot(spec.robot_id, spec.arm_joint_names, (0.0, 0.0))

    async def spawn_part(self, instance, brick_id, pose):
        self.calls.append(("part", instance))
        measured = Pose("world", (0.1, 0.0, 0.0), pose.orientation_xyzw) if self.bad_pose else pose
        loaded = LoadedPart(instance, f"/World/parts/{instance}", brick_id, measured)
        self.parts[instance] = loaded
        return loaded

    async def settle_and_measure(self):
        self.calls.append("settle")

    def measured_parts(self):
        return self.parts


def test_bootstrap_is_deterministic_and_creates_initial_snapshot_without_motion():
    plan, scene, specs = inputs()
    port = FakePort()
    loaded = asyncio.run(bootstrap_scene(port, plan, scene, specs, monotonic=lambda: 4.0))
    assert port.calls == ["open", ("robot", "left"), ("part", "base"), ("part", "moving"), "settle"]
    assert loaded.execution_snapshot.revisions.world == 1
    assert loaded.execution_snapshot.revisions.motion == 0
    assert loaded.execution_snapshot.robots["left"].feedback_monotonic_s == 4.0


def test_bootstrap_rejects_extra_robot_before_opening_simulation():
    plan, scene, specs = inputs()
    specs["unused"] = specs["left"]
    port = FakePort()
    with pytest.raises(SceneBootstrapError, match="exactly match"):
        asyncio.run(bootstrap_scene(port, plan, scene, specs))
    assert port.calls == []


def test_bootstrap_rejects_measured_part_pose_mismatch():
    plan, scene, specs = inputs()
    port = FakePort()
    port.bad_pose = True
    with pytest.raises(SceneBootstrapError, match="initial pose mismatch"):
        asyncio.run(bootstrap_scene(port, plan, scene, specs))
