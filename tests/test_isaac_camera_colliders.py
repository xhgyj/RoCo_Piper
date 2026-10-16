"""The wrist-camera colliders are disabled by our own scene bootstrap.

The vendor stand collides as the convex hull of a 97 x 48 x 74 mm bracket; it
is what staged the offset12 support pre-phase with a 1.87 mm interpenetration.
The vendor USD must stay byte-identical (the TCP authorization artifact binds
its SHA-256), and each link's ``collisions`` subtree is instanced, so the
bootstrap materializes those subtrees into the session layer and switches
``collisionEnabled`` off there -- the same idiom ``materialize_robot_visuals``
uses for the visuals.  A launch on 2026-10-14 failed with "carries no collider
to disable" because the first implementation walked the instance proxies and
never saw the prodotypes' applied schemas; these tests pin the fixed order.
"""

from __future__ import annotations

import sys
import types

import pytest

from rocobrick.isaac_scene import (
    WRIST_CAMERA_COLLISION_LINKS,
    SceneBootstrapError,
    _disable_camera_collisions,
    _hide_camera_visuals,
)


class _CollisionAttribute:
    def __init__(self, value: bool = True) -> None:
        self.value = value

    def Get(self) -> bool:  # noqa: N802
        return self.value

    def Set(self, value: bool) -> None:  # noqa: N802
        self.value = value


class _Prim:
    def __init__(
        self, path: str, collider: bool = False, instance: bool = False
    ) -> None:
        self._path = path
        self._collider = collider
        self._instance = instance
        self.children: list[_Prim] = []
        self.collision_enabled = _CollisionAttribute() if collider else None
        self.authored_layers: list[object] = []
        self.invisible = False
        self.stage: object = None

    def IsValid(self) -> bool:  # noqa: N802
        return True

    def HasAPI(self, api: object) -> bool:  # noqa: N802
        return bool(api.HasAPI(self))

    def GetPath(self) -> str:  # noqa: N802
        return self._path

    def GetChild(self, name: str) -> _Prim:  # noqa: N802
        for child in self.children:
            if child._path.rsplit("/", 1)[-1] == name:
                return child
        return _Missing()

    def IsInstance(self) -> bool:  # noqa: N802
        return self._instance

    def SetInstanceable(self, value: bool) -> None:  # noqa: N802
        assert value is False
        self._instance = False
        self.materialized = True

    def add(self, child: _Prim) -> _Prim:
        self.children.append(child)
        return self


class _Missing(_Prim):
    def __init__(self) -> None:
        super().__init__("")

    def IsValid(self) -> bool:  # noqa: N802
        return False


class _CollisionAPI:
    def __init__(self, prim: _Prim) -> None:
        self._prim = prim

    @classmethod
    def HasAPI(cls, prim: _Prim) -> bool:  # noqa: N802
        return prim._collider

    def GetCollisionEnabledAttr(self) -> _CollisionAttribute:  # noqa: N802
        return self._prim.collision_enabled

    def CreateCollisionEnabledAttr(self, value: bool) -> _CollisionAttribute:  # noqa: N802
        if self._prim.collision_enabled is None:
            self._prim.collision_enabled = _CollisionAttribute(value)
        self._prim.collision_enabled.Set(value)
        self._prim.authored_layers.append(self._prim.stage.edit_layer)
        return self._prim.collision_enabled


class _Imageable:
    def __init__(self, prim: _Prim) -> None:
        self._prim = prim

    def MakeInvisible(self) -> None:  # noqa: N802
        self._prim.invisible = True
        self._prim.authored_layers.append(self._prim.stage.edit_layer)


class _EditContext:
    def __init__(self, stage: object, layer: object) -> None:
        self._stage = stage
        self._layer = layer

    def __enter__(self) -> _EditContext:
        self._stage.edit_layer = self._layer
        return self

    def __exit__(self, *_exc: object) -> bool:
        self._stage.edit_layer = None
        return False


class _Usd:
    EditContext = _EditContext

    @staticmethod
    def TraverseInstanceProxies() -> object:  # noqa: N802
        return object()

    @staticmethod
    def PrimRange(root: _Prim, _predicate: object = None) -> list[_Prim]:  # noqa: N802
        collected: list[_Prim] = []

        def walk(prim: _Prim) -> None:
            collected.append(prim)
            for child in prim.children:
                walk(child)

        walk(root)
        return collected


def _install_fake_pxr() -> None:
    module = types.SimpleNamespace(
        Usd=_Usd,
        UsdGeom=types.SimpleNamespace(Imageable=_Imageable),
        UsdPhysics=types.SimpleNamespace(CollisionAPI=_CollisionAPI),
    )
    sys.modules["pxr"] = module


class _Stage:
    """Indexes every prim in the tree so nested collision paths resolve."""

    def __init__(self, roots: dict[str, _Prim]) -> None:
        self._prims: dict[str, _Prim] = {}
        self.session_layer = object()
        self.edit_layer: object = None

        def index(prim: _Prim) -> None:
            prim.stage = self
            self._prims[prim.GetPath()] = prim
            for child in prim.children:
                index(child)

        for root in roots.values():
            index(root)

    def GetPrimAtPath(self, path: str) -> _Prim:  # noqa: N802
        return self._prims.get(path, _Missing())

    def GetSessionLayer(self) -> object:  # noqa: N802
        return self.session_layer


class _Robot:
    prim_path = "/World/Robot_0"


def _camera_fixture() -> tuple[_Stage, _Prim, _Prim, _Prim]:
    stand = _Prim("/World/Robot_0/camera_stand_link")
    collisions = _Prim(
        "/World/Robot_0/camera_stand_link/collisions", instance=True
    )
    collider = _Prim(
        "/World/Robot_0/camera_stand_link/collisions/realsense_mid_stand/Scene",
        collider=True,
    )
    collisions.add(collider)
    stand.add(collisions)
    camera = _Prim("/World/Robot_0/camera_link")
    camera_collisions = _Prim("/World/Robot_0/camera_link/collisions", instance=True)
    box = _Prim("/World/Robot_0/camera_link/collisions/mesh_0/box", collider=True)
    camera_collisions.add(box)
    camera.add(camera_collisions)
    gripper = _Prim("/World/Robot_0/gripper_link1")
    gripper_collisions = _Prim("/World/Robot_0/gripper_link1/collisions", instance=True)
    control = _Prim(
        "/World/Robot_0/gripper_link1/collisions/gripper_link1/node_STL_BINARY_",
        collider=True,
    )
    gripper_collisions.add(control)
    gripper.add(gripper_collisions)
    stage = _Stage(
        {
            "/World/Robot_0/camera_stand_link": stand,
            "/World/Robot_0/camera_link": camera,
            "/World/Robot_0/gripper_link1": gripper,
        }
    )
    return stage, collider, box, control


@pytest.fixture(autouse=True)
def _fake_pxr(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_pxr()
    monkeypatch.setitem(sys.modules, "pxr", sys.modules["pxr"])


def test_camera_colliders_are_materialized_then_disabled() -> None:
    """Instances are materialized first, then disabled in the session layer."""
    stage, collider, box, control = _camera_fixture()
    disabled = _disable_camera_collisions(stage, robot=_Robot())
    assert disabled == (
        "/World/Robot_0/camera_stand_link/collisions/realsense_mid_stand/Scene",
        "/World/Robot_0/camera_link/collisions/mesh_0/box",
    )
    for prim in (collider, box):
        assert prim.collision_enabled.value is False
        assert prim.authored_layers == [stage.session_layer]
    collisions_root = stage.GetPrimAtPath("/World/Robot_0/camera_stand_link/collisions")
    assert getattr(collisions_root, "materialized", False) is True
    # A control collider on another link keeps colliding.
    assert control.collision_enabled.value is True
    assert WRIST_CAMERA_COLLISION_LINKS == ("camera_stand_link", "camera_link")


def test_missing_camera_link_is_tolerated() -> None:
    """A vendor variant without the camera has nothing to disable."""
    camera = _Prim("/World/Robot_0/camera_link")
    box = _Prim("/World/Robot_0/camera_link/collisions/mesh_0/box", collider=True)
    camera.add(box)
    stage = _Stage({"/World/Robot_0/camera_link": camera})
    assert _disable_camera_collisions(stage, robot=_Robot()) == (
        "/World/Robot_0/camera_link/collisions/mesh_0/box",
    )
    assert box.collision_enabled.value is False


def test_camera_link_without_collider_fails_closed() -> None:
    """A camera link that carries no collider is a contract inconsistency."""
    camera = _Prim("/World/Robot_0/camera_link")
    camera.add(_Prim("/World/Robot_0/camera_link/visuals/d435/ID1/mesh"))
    stage = _Stage({"/World/Robot_0/camera_link": camera})
    with pytest.raises(SceneBootstrapError):
        _disable_camera_collisions(stage, robot=_Robot())


def test_camera_visuals_are_hidden_in_the_session_layer() -> None:
    """The hidden camera is not drawn, and only the session layer is edited."""
    stage, _collider, _box, _control = _camera_fixture()
    hidden = _hide_camera_visuals(stage, robot=_Robot())
    assert hidden == (
        "/World/Robot_0/camera_stand_link",
        "/World/Robot_0/camera_link",
    )
    for path in hidden:
        prim = stage.GetPrimAtPath(path)
        assert prim.invisible is True
        assert prim.authored_layers == [stage.session_layer]
    control = stage.GetPrimAtPath("/World/Robot_0/gripper_link1")
    assert control.invisible is False


def test_visual_hiding_is_a_no_op_without_camera_links() -> None:
    """A vendor variant without the camera has nothing to hide."""
    gripper = _Prim("/World/Robot_0/gripper_link1")
    stage = _Stage({"/World/Robot_0/gripper_link1": gripper})
    assert _hide_camera_visuals(stage, robot=_Robot()) == ()
    assert gripper.invisible is False


def test_only_the_colliderless_link_fails_closed() -> None:
    """Each camera link is checked on its own colliders, not the other's."""
    stand = _Prim("/World/Robot_0/camera_stand_link")
    stand.add(
        _Prim("/World/Robot_0/camera_stand_link/collisions/stand/Scene", collider=True)
    )
    camera = _Prim("/World/Robot_0/camera_link")
    camera.add(_Prim("/World/Robot_0/camera_link/visuals/d435/ID1/mesh"))
    stage = _Stage(
        {
            "/World/Robot_0/camera_stand_link": stand,
            "/World/Robot_0/camera_link": camera,
        }
    )
    with pytest.raises(SceneBootstrapError):
        _disable_camera_collisions(stage, robot=_Robot())
