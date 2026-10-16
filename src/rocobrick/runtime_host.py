from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from rocobrick.isaac_scene import IsaacLoadedScene, IsaacSceneConfig, load_verified_scene
from rocobrick.bricksim_topology import BrickSimTopologyBundle
from rocobrick.scene_bootstrap import RobotSpawnSpec
from rocobrick.stap_contract import StapPlan

__all__ = ["RuntimeHostBootstrapError", "LoadedRuntimeHost", "bootstrap_runtime_host", "collect_world_aabbs"]

class RuntimeHostBootstrapError(RuntimeError):
    pass

@dataclass(frozen=True)
class LoadedRuntimeHost:
    scene: IsaacLoadedScene
    physical_instance_paths: Mapping[str, str]
    physical_instance_brick_ids: Mapping[str, int]

    @property
    def app(self): return self.scene.app
    @property
    def world(self): return self.scene.world
    @property
    def robots(self): return self.scene.robots
    @property
    def measured_robots(self): return self.scene.measured_robots
    @property
    def parts(self): return self.scene.parts
    @property
    def topology_paths(self): return self.scene.topology_paths
    @property
    def collision_filter_pairs(self): return self.scene.collision_filter_pairs

    @property
    def disabled_camera_collisions(self):
        """Return the wrist-camera collision prims this session disabled."""
        return self.scene.disabled_camera_collisions

    @property
    def hidden_camera_visuals(self):
        """Return the wrist-camera link prims whose visuals this session hid."""
        return self.scene.hidden_camera_visuals

    def close(self) -> None:
        if getattr(self.app, "_rocobrick_runtime_host_closed", False):
            return
        try:
            setattr(self.app, "_rocobrick_runtime_host_closed", True)
        except Exception:
            pass
        shutdown = getattr(self.app, "shutdown", None)
        if shutdown is not None:
            shutdown()
            return
        close = getattr(self.app, "close", None)
        if close is not None:
            close()

async def bootstrap_runtime_host(plan: StapPlan, topology: BrickSimTopologyBundle, robots: Mapping[str, RobotSpawnSpec], config: IsaacSceneConfig) -> LoadedRuntimeHost:
    try:
        scene = await load_verified_scene(plan, topology, robots, config)
        paths = {}
        bricks = {}
        for instance_id, part in scene.parts.items():
            if not isinstance(instance_id, str) or not instance_id:
                raise RuntimeHostBootstrapError("invalid physical instance id")
            if instance_id in paths:
                raise RuntimeHostBootstrapError(f"duplicate physical instance: {instance_id}")
            if not isinstance(part.prim_path, str) or not part.prim_path:
                raise RuntimeHostBootstrapError(f"missing prim path: {instance_id}")
            if not isinstance(part.brick_id, int) or isinstance(part.brick_id, bool):
                raise RuntimeHostBootstrapError(f"invalid brick id: {instance_id}")
            paths[instance_id] = part.prim_path
            bricks[instance_id] = part.brick_id
        if not paths:
            raise RuntimeHostBootstrapError("empty loaded scene")
        return LoadedRuntimeHost(scene, MappingProxyType(paths), MappingProxyType(bricks))
    except RuntimeHostBootstrapError:
        raise
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except Exception as exc:
        raise RuntimeHostBootstrapError("runtime host bootstrap failed") from exc


def collect_world_aabbs(host: LoadedRuntimeHost):
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import Usd, UsdGeom
    from rocobrick.planning_scene import box_from_aabb
    if not isinstance(host, LoadedRuntimeHost):
        raise RuntimeHostBootstrapError("invalid runtime host")
    stage = get_current_stage()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    boxes = {}
    for instance_id, prim_path in host.physical_instance_paths.items():
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeHostBootstrapError(f"missing runtime prim: {instance_id}")
        aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        minimum = tuple(float(v) for v in aligned.GetMin())
        maximum = tuple(float(v) for v in aligned.GetMax())
        boxes[instance_id] = box_from_aabb(
            object_id=instance_id, frame_id="assembly_world",
            minimum_m=minimum, maximum_m=maximum,
        )
    return MappingProxyType(boxes)
