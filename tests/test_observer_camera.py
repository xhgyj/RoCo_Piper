"""Observer framing checks without starting Isaac or moving a robot."""

import sys
from types import SimpleNamespace

import pytest

from rocobrick.isaac_scene import _set_initial_viewport_camera


def _mock_camera_sdk(monkeypatch, *, viewport):
    calls = []
    attribute = SimpleNamespace(Set=lambda value: calls.append(("interest", value)))
    prim = SimpleNamespace(
        CreateAttribute=lambda name, kind: attribute,
    )
    camera = SimpleNamespace(
        CreateFocalLengthAttr=lambda value: calls.append(("focal", value)),
        CreateFocusDistanceAttr=lambda value: calls.append(("focus", value)),
        GetPrim=lambda: prim,
    )

    def define(stage, path):
        calls.append(("define", stage, path))
        return camera

    class CameraState:
        def __init__(self, *, camera_path, viewport):
            calls.append(("state", camera_path, viewport))

        def set_position_world(self, value, rotate):
            calls.append(("position", value, rotate))

        def set_target_world(self, value, rotate):
            calls.append(("target", value, rotate))

    monkeypatch.setitem(
        sys.modules,
        "omni.kit.viewport.utility",
        SimpleNamespace(get_active_viewport=lambda: viewport),
    )
    monkeypatch.setitem(
        sys.modules,
        "omni.kit.viewport.utility.camera_state",
        SimpleNamespace(ViewportCameraState=CameraState),
    )
    monkeypatch.setitem(
        sys.modules,
        "pxr",
        SimpleNamespace(
            Gf=SimpleNamespace(Vec3d=lambda *values: values),
            Sdf=SimpleNamespace(ValueTypeNames=SimpleNamespace(Vector3d=object())),
            UsdGeom=SimpleNamespace(Camera=SimpleNamespace(Define=define)),
        ),
    )
    return calls


def test_headed_observer_frames_pick_and_handover(monkeypatch, capsys):
    """Apply the close observer framing without accessing robot primitives."""
    monkeypatch.setenv("ROCOBRICK_HEADED", "1")
    viewport = SimpleNamespace(camera_path=None)
    calls = _mock_camera_sdk(monkeypatch, viewport=viewport)
    stage = object()

    _set_initial_viewport_camera(stage)

    path = "/World/RoCoBrickInitialCamera"
    assert viewport.camera_path == path
    assert calls == [
        ("define", stage, path),
        ("focal", 28.0),
        ("focus", 0.75),
        ("interest", (0.0, 0.0, -1.0)),
        ("state", path, viewport),
        ("position", (0.0, 0.60, 0.42), True),
        ("target", (0.0, -0.10, 0.16), True),
    ]
    assert "focal_length_mm=28.0" in capsys.readouterr().out


@pytest.mark.parametrize("headed", [None, "0"])
def test_headless_does_not_touch_camera(monkeypatch, headed):
    """Leave camera state untouched when the run is not headed."""
    if headed is None:
        monkeypatch.delenv("ROCOBRICK_HEADED", raising=False)
    else:
        monkeypatch.setenv("ROCOBRICK_HEADED", headed)
    calls = _mock_camera_sdk(monkeypatch, viewport=None)

    _set_initial_viewport_camera(object())

    assert calls == []


def test_headed_without_viewport_reports_error(monkeypatch):
    """Reject missing headed viewports instead of silently losing framing."""
    monkeypatch.setenv("ROCOBRICK_HEADED", "1")
    _mock_camera_sdk(monkeypatch, viewport=None)

    with pytest.raises(RuntimeError, match="no active viewport"):
        _set_initial_viewport_camera(object())
