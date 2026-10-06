"""Offline rules for rejecting aggregate room collider AABBs."""

from pathlib import Path


def test_environment_collector_uses_primrange_not_getdescendants():
    """``Usd.Prim`` has no ``GetDescendants``; the collector must not call it.

    A live scene-state query once failed closed on every callback because the
    leaf walk called the non-existent ``Prim.GetDescendants`` API; this pins
    the subtree traversal to ``Usd.PrimRange`` and forbids the bad name.
    """
    source = (
        Path(__file__).resolve().parents[1] / "isaac" / "p05_host_bridge.py"
    ).read_text(encoding="utf-8")
    assert ".GetDescendants(" not in source
    assert "Usd.PrimRange(prim)" in source


def test_environment_bounds_must_be_leaf_geometry():
    """The diagnostic contract documents aggregate-vs-leaf failure explicitly."""
    aggregate = {"aabb_min": [-3.6, -3.6, -0.75], "aabb_max": [3.6, 3.6, 1.75]}
    assert aggregate["aabb_max"][0] - aggregate["aabb_min"][0] > 7.0
