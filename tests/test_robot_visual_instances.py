"""Visual-only session overlays must never change collider or body metadata."""

import sys
from types import SimpleNamespace

import pytest

from rocobrick.adapters.isaac.visual_instances import materialize_robot_visuals


def stage_fixture(monkeypatch, *, physics=False):
    """Provide only the inspected USD API, with explicit edit tracking.

    Returns:
        A fake stage and exact records of session-only instance edits.
    """
    edits = []
    context = []
    collision, rigid = object(), object()
    missing = SimpleNamespace(IsValid=lambda: False)
    visual = SimpleNamespace(
        IsValid=lambda: True,
        IsInstance=lambda: True,
        HasAPI=lambda api: physics and api is collision,
        GetPath=lambda: "/World/Robot_0/gripper_base/visuals",
        SetInstanceable=lambda value: edits.append((context[-1], value)),
    )
    body = SimpleNamespace(
        GetChild=lambda name: visual if name == "visuals" else missing
    )
    robot = SimpleNamespace(IsValid=lambda: True, GetChildren=lambda: [body])
    stage = SimpleNamespace(
        GetPrimAtPath=lambda path: robot, GetSessionLayer=lambda: "session"
    )

    class EditContext:
        """Track the destination layer of every metadata edit."""

        def __init__(self, usd_stage, layer):
            """Keep the selected overlay layer."""
            self.layer = layer

        def __enter__(self):
            """Activate the overlay layer."""
            context.append(self.layer)

        def __exit__(self, *args):
            """Restore the original edit target."""
            context.pop()

    monkeypatch.setitem(
        sys.modules,
        "pxr",
        SimpleNamespace(
            Usd=SimpleNamespace(
                PrimRange=lambda prim, predicate: [prim],
                TraverseInstanceProxies=lambda: True,
                EditContext=EditContext,
            ),
            UsdPhysics=SimpleNamespace(CollisionAPI=collision, RigidBodyAPI=rigid),
        ),
    )
    return stage, edits


def test_materialize_only_visual_instance_metadata_in_session_layer(monkeypatch):
    """No model asset, mesh geometry, collision or body setter is available."""
    stage, edits = stage_fixture(monkeypatch)
    paths = materialize_robot_visuals(stage, "/World/Robot_0")
    assert paths == ("/World/Robot_0/gripper_base/visuals",)
    assert edits == [("session", False)]


def test_visual_subtree_with_physics_is_rejected_before_any_edit(monkeypatch):
    """Physics-bearing references cannot be silently materialized as visuals."""
    stage, edits = stage_fixture(monkeypatch, physics=True)
    with pytest.raises(ValueError, match="contains physics"):
        materialize_robot_visuals(stage, "/World/Robot_0")
    assert edits == []
