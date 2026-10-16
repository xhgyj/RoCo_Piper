"""Observer framing checks without starting Isaac or moving a robot."""

import sys
from types import SimpleNamespace

import pytest

from rocobrick.isaac_scene import (
    _observer_focus_bounds,
    _observer_pose,
    _set_initial_viewport_camera,
)


def _aligned_bounds(lower, upper):
    return SimpleNamespace(
        IsEmpty=lambda: False,
        GetMin=lambda: lower,
        GetMax=lambda: upper,
    )


def _bbox_cache_factory(calls, prim_bounds):
    """Fake the USD bounds cache: path -> (lower, upper), None means empty.

    Returns:
        A stand-in for the USD bounds cache class.
    """

    def make(time_code, purposes):
        calls.append(("bbox_cache", purposes))

        def compute_world_bound(prim):
            calls.append(("world_bound", prim.path))
            bounds = (prim_bounds or {}).get(prim.path)
            if bounds is None:
                empty = SimpleNamespace(IsEmpty=lambda: True)
                return SimpleNamespace(ComputeAlignedRange=lambda: empty)
            return SimpleNamespace(ComputeAlignedRange=lambda: _aligned_bounds(*bounds))

        return SimpleNamespace(ComputeWorldBound=compute_world_bound)

    return make


def _mock_camera_sdk(monkeypatch, *, viewport, prim_bounds=None):
    calls = []
    attribute = SimpleNamespace(Set=lambda value: calls.append(("interest", value)))
    prim = SimpleNamespace(
        CreateAttribute=lambda name, kind: attribute,
    )
    camera = SimpleNamespace(
        CreateFocalLengthAttr=lambda value: calls.append(("focal", value)),
        CreateFocusDistanceAttr=lambda value: calls.append(("focus", value)),
        CreateFStopAttr=lambda value: calls.append(("fstop", value)),
        CreateClippingRangeAttr=lambda value: calls.append(("clipping", value)),
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

    posted = []
    monkeypatch.setitem(
        sys.modules,
        "carb.settings",
        SimpleNamespace(
            get_settings=lambda: SimpleNamespace(
                set_bool=lambda key, value: posted.append((key, value))
            )
        ),
    )
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
            Usd=SimpleNamespace(TimeCode=SimpleNamespace(Default=lambda: "default")),
            UsdGeom=SimpleNamespace(
                Camera=SimpleNamespace(Define=define),
                BBoxCache=_bbox_cache_factory(calls, prim_bounds),
                Tokens=SimpleNamespace(default_="default"),
            ),
        ),
    )
    return calls, posted


def _stage(valid_paths=()):
    valid = set(valid_paths)

    def get_prim_at_path(path):
        prim = SimpleNamespace(path=path)
        prim.IsValid = lambda: path in valid
        return prim

    return SimpleNamespace(GetPrimAtPath=get_prim_at_path)


def test_observer_pose_fits_the_measured_bounds():
    """Pin the fitted pose for a 0.85 x 0.5 m footprint with a 0.55 m band."""
    position, target, distance = _observer_pose((-0.5, -0.35, 0.0), (0.35, 0.15, 0.6))

    assert target == pytest.approx((-0.075, -0.1, 0.275))
    assert distance == pytest.approx(1.2565, abs=1e-3)
    assert position == pytest.approx((-0.075, 0.9474, 0.9691), abs=1e-3)


def test_observer_pose_keeps_a_high_arm_out_of_the_fit():
    """A raised arm raises the camera, not the framed height band."""
    low = _observer_pose((-0.5, -0.35, 0.0), (0.35, 0.15, 0.6))
    high = _observer_pose((-0.5, -0.35, 0.0), (0.35, 0.15, 1.4))

    assert high[2] == pytest.approx(low[2])
    assert high[1][2] == pytest.approx(low[1][2])


def test_overview_pose_includes_full_height():
    """Overview mode frames the raised arms instead of a capped height band."""
    lower, upper = (-0.8, -0.9, 0.0), (0.9, 0.4, 0.85)
    default = _observer_pose(lower, upper)
    overview = _observer_pose(lower, upper, full_height=True)
    assert overview[1][2] == pytest.approx(0.425)
    assert overview[2] > default[2]


def test_headed_overview_includes_robot_motion_envelopes(monkeypatch):
    """Base envelopes enlarge framing without touching the scene physics."""
    monkeypatch.setenv("ROCOBRICK_HEADED", "1")
    monkeypatch.setenv("ROCOBRICK_OBSERVER_OVERVIEW", "1")
    viewport = SimpleNamespace(camera_path=None)
    calls, _ = _mock_camera_sdk(
        monkeypatch,
        viewport=viewport,
        prim_bounds={
            "/World/part_0": ((-0.1, -0.1, 0.0), (0.1, 0.1, 0.2)),
        },
    )
    _set_initial_viewport_camera(
        _stage(("/World/part_0",)),
        ("/World/part_0",),
        ((-0.3, -0.3, 0.0), (0.3, -0.3, 0.0)),
    )
    expected = _observer_pose((-0.95, -0.95, 0.0), (0.95, 0.35, 0.85), full_height=True)
    by_name = {call[0]: call[1:] for call in calls}
    assert by_name["target"][0] == pytest.approx(expected[1])
    assert by_name["position"][0] == pytest.approx(expected[0])


def test_observer_focus_bounds_skips_invalid_prims(monkeypatch):
    """Only prims that resolve contribute to the fit."""
    _mock_camera_sdk(
        monkeypatch,
        viewport=SimpleNamespace(camera_path=None),
        prim_bounds={"/World/Robot_0": ((-0.5, -0.35, 0.0), (0.35, 0.15, 0.6))},
    )
    stage = _stage(valid_paths={"/World/Robot_0"})

    bounds = _observer_focus_bounds(stage, ("/World/Robot_0", "/World/Missing"))

    assert bounds == ((-0.5, -0.35, 0.0), (0.35, 0.15, 0.6), ())


def test_observer_focus_bounds_rejects_a_vendor_artefact_prim(monkeypatch):
    """The piper USD's 50 x 50 m reference plane never sizes the frame."""
    _mock_camera_sdk(
        monkeypatch,
        viewport=SimpleNamespace(camera_path=None),
        prim_bounds={
            "/World/Robot_0": ((-25.0, -25.0, 0.0), (25.0, 25.0, 0.32)),
            "/World/part_0": ((-0.1, -0.1, 0.08), (0.1, 0.1, 0.12)),
        },
    )
    stage = _stage(valid_paths={"/World/Robot_0", "/World/part_0"})

    bounds = _observer_focus_bounds(stage, ("/World/Robot_0", "/World/part_0"))

    assert bounds == (
        (-0.1, -0.1, 0.08),
        (0.1, 0.1, 0.12),
        ("/World/Robot_0",),
    )


def test_observer_focus_bounds_includes_explicit_points(monkeypatch):
    """Robot base points join the box without a measurable prim."""
    _mock_camera_sdk(
        monkeypatch,
        viewport=SimpleNamespace(camera_path=None),
        prim_bounds={"/World/part_0": ((-0.1, -0.1, 0.08), (0.1, 0.1, 0.12))},
    )
    stage = _stage(valid_paths={"/World/part_0"})

    bounds = _observer_focus_bounds(
        stage,
        ("/World/part_0",),
        ((0.24, -0.26, 0.0), (-0.35, -0.30, 0.0)),
    )

    assert bounds == (
        (-0.35, -0.30, 0.0),
        (0.24, 0.1, 0.12),
        (),
    )


def test_observer_focus_bounds_reports_an_unmeasurable_stage(monkeypatch):
    """An unmeasurable stage yields no bounds instead of a partial box."""
    _mock_camera_sdk(
        monkeypatch, viewport=SimpleNamespace(camera_path=None), prim_bounds=None
    )

    assert _observer_focus_bounds(_stage(), ("/World/Robot_0",)) is None


def test_headed_observer_frames_the_loaded_scene(monkeypatch, capsys):
    """Frame the measured focus prims and keep the viewport wiring."""
    monkeypatch.setenv("ROCOBRICK_HEADED", "1")
    viewport = SimpleNamespace(camera_path=None)
    calls, posted = _mock_camera_sdk(
        monkeypatch,
        viewport=viewport,
        prim_bounds={"/World/part_0": ((-0.5, -0.35, 0.0), (0.35, 0.15, 0.6))},
    )
    stage = _stage(valid_paths={"/World/part_0"})

    _set_initial_viewport_camera(stage, ("/World/part_0",))

    path = "/World/RoCoBrickInitialCamera"
    assert viewport.camera_path == path
    assert [call[0] for call in calls] == [
        "define",
        "bbox_cache",
        "world_bound",
        "focal",
        "focus",
        "fstop",
        "clipping",
        "interest",
        "state",
        "position",
        "target",
    ]
    by_name = {call[0]: call[1:] for call in calls}
    assert by_name["focal"] == (18.0,)
    assert by_name["focus"][0] == pytest.approx(1.2565, abs=1e-4)
    assert by_name["fstop"] == (0.0,)
    assert by_name["clipping"] == ((0.01, 100.0),)
    assert viewport.resolution == (1920, 1080)
    assert by_name["interest"] == ((0.0, 0.0, -1.0),)
    assert by_name["position"][1] is True
    assert by_name["position"][0] == pytest.approx((-0.075, 0.9474, 0.9691), abs=1e-4)
    assert by_name["target"][1] is True
    assert by_name["target"][0] == pytest.approx((-0.075, -0.1, 0.275))
    assert posted == [
        ("/rtx/post/dof/enabled", False),
        ("/rtx/post/motionblur/enabled", False),
    ]
    output = capsys.readouterr().out
    assert "focal_length_mm=18.0" in output
    assert "focus_distance_m=1.2565" in output
    assert "skipped_focus_prims=[]" in output


def test_pullback_preserves_direction_and_target(monkeypatch):
    """A modest pullback changes scale, not the chosen observation angle."""
    monkeypatch.setenv("ROCOBRICK_HEADED", "1")
    monkeypatch.setenv("ROCOBRICK_OBSERVER_PULLBACK", "1")
    viewport = SimpleNamespace(camera_path=None)
    calls, _ = _mock_camera_sdk(
        monkeypatch,
        viewport=viewport,
        prim_bounds={
            "/World/part_0": ((-0.5, -0.35, 0.0), (0.35, 0.15, 0.6)),
        },
    )
    _set_initial_viewport_camera(_stage(("/World/part_0",)), ("/World/part_0",))
    position, target, distance = _observer_pose((-0.5, -0.35, 0.0), (0.35, 0.15, 0.6))
    by_name = {call[0]: call[1:] for call in calls}
    assert by_name["target"][0] == pytest.approx(target)
    assert by_name["position"][0] == pytest.approx(
        tuple(target[i] + 1.25 * (position[i] - target[i]) for i in range(3))
    )
    assert by_name["focus"][0] == pytest.approx(1.25 * distance)


def test_headed_observer_falls_back_without_measurable_bounds(monkeypatch, capsys):
    """Keep the legacy observer pose when no focus prim can be measured."""
    monkeypatch.setenv("ROCOBRICK_HEADED", "1")
    viewport = SimpleNamespace(camera_path=None)
    calls, _posted = _mock_camera_sdk(monkeypatch, viewport=viewport, prim_bounds=None)

    _set_initial_viewport_camera(_stage(), ("/World/Missing",))

    by_name = {call[0]: call[1:] for call in calls}
    assert by_name["position"][0] == (0.0, 0.60, 0.42)
    assert by_name["target"][0] == (0.0, -0.10, 0.16)
    assert by_name["focus"][0] == pytest.approx(0.7467, abs=1e-4)
    assert "focus_distance_m=0.7467" in capsys.readouterr().out


@pytest.mark.parametrize("headed", [None, "0"])
def test_headless_does_not_touch_camera(monkeypatch, headed):
    """Leave camera state untouched when the run is not headed."""
    if headed is None:
        monkeypatch.delenv("ROCOBRICK_HEADED", raising=False)
    else:
        monkeypatch.setenv("ROCOBRICK_HEADED", headed)
    calls, posted = _mock_camera_sdk(monkeypatch, viewport=None)

    _set_initial_viewport_camera(object())

    assert calls == []
    assert posted == []


def test_headed_without_viewport_reports_error(monkeypatch):
    """Reject missing headed viewports instead of silently losing framing."""
    monkeypatch.setenv("ROCOBRICK_HEADED", "1")
    _mock_camera_sdk(monkeypatch, viewport=None, prim_bounds=None)

    with pytest.raises(RuntimeError, match="no active viewport"):
        _set_initial_viewport_camera(object())
