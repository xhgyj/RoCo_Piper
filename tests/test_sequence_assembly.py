"""Tests for the strict alternating dense1 assembly plan."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rocobrick.env import video
from rocobrick.policy import gt_assembly, multi_arm_gt_assembly, sequence_assembly
from rocobrick.policy.sequence_assembly import (
    DENSE1_TARGET_ORDER,
    DENSE1_WORKSPACE_SIZE,
    dense1_turns,
    dense1_workspace_slots,
)

ROOT = Path(__file__).resolve().parents[1]


def test_dense1_goal_and_turns_are_fixed() -> None:
    """The checked-in goal follows the support-first arm alternation."""
    goal = json.loads(
        (ROOT / "tasks/type2/dense1/structure.json").read_text(encoding="utf-8")
    )
    assert sorted(map(int, goal)) == [1, 2, 3, 4, 5, 6]
    assert DENSE1_TARGET_ORDER == (2, 3, 1, 6, 4, 5)
    assert [(turn.target_id, turn.arm_index) for turn in dense1_turns()] == [
        (2, 0),
        (3, 1),
        (1, 0),
        (6, 1),
        (4, 0),
        (5, 1),
    ]
    assembled = set()
    supports = {3: 2, 5: 4, 6: 1}
    for target_id in DENSE1_TARGET_ORDER:
        if target_id in supports:
            assert supports[target_id] in assembled
        assembled.add(target_id)


def test_dense1_workspace_slots_are_complete_and_bounded() -> None:
    """The pickup layout supplies one deterministic slot per loose part."""
    slots = dense1_workspace_slots()
    assert set(slots) == set(DENSE1_TARGET_ORDER)
    assert slots[2].yaw_degrees == 90.0
    assert all(
        abs(slot.offset_xy[0]) < DENSE1_WORKSPACE_SIZE[0] / 2
        for slot in slots.values()
    )
    assert all(
        abs(slot.offset_xy[1]) < DENSE1_WORKSPACE_SIZE[1] / 2
        for slot in slots.values()
    )


def test_resolver_can_attach_first_part_to_baseplate() -> None:
    """A Type-2 ground brick may use part zero as its first reference."""
    env = SimpleNamespace(
        topology={
            "parts": [
                {"id": 0, "payload": {"L": 32, "W": 32, "H": 1}},
                {"id": 1, "payload": {"L": 2, "W": 2, "H": 3}},
            ],
            "connections": [
                {
                    "stud_id": 0,
                    "stud_iface": 1,
                    "hole_id": 1,
                    "hole_iface": 0,
                    "offset": [4, 5],
                    "yaw": 0,
                }
            ],
        },
        pre_placed_parts={0: "/plate"},
        to_place_placed={1: "/target"},
    )
    task = multi_arm_gt_assembly.resolve_single_step_task(
        env, target_id=1, assembled_parts=env.pre_placed_parts
    )
    assert task.target_id == 1
    assert task.primary.reference_id == 0
    assert task.primary.reference_path == "/plate"


def test_sequence_promotes_only_after_home_gated_turn(monkeypatch) -> None:
    """Each target becomes a reference only after cleanup and home verification."""
    env = SimpleNamespace(
        robot_pins=[object(), object()],
        pre_placed_parts={0: "/plate"},
        to_place_placed={part_id: f"/part_{part_id}" for part_id in range(1, 7)},
    )
    events = []

    def resolve(env, target_id, assembled_parts):
        assert target_id not in assembled_parts
        events.append(("resolve", target_id, tuple(sorted(assembled_parts))))
        return SimpleNamespace(target_id=target_id)

    async def prepare(env, safe_height, task, arm_index):
        events.append(("prepare", task.target_id, arm_index))
        return SimpleNamespace(task=task, arm_index=arm_index)

    async def execute(env, prepared, runtime_config):
        events.append(("execute", prepared.task.target_id, prepared.arm_index))
        return gt_assembly.ExpertResult(True, 1, None)

    async def cleanup(env, prepared):
        events.append(("home", prepared.task.target_id, prepared.arm_index))

    monkeypatch.setattr(sequence_assembly, "arm_home_error", lambda env, arm: None)
    monkeypatch.setattr(sequence_assembly, "resolve_single_step_task", resolve)
    monkeypatch.setattr(sequence_assembly, "prepare_safe_start", prepare)
    monkeypatch.setattr(sequence_assembly, "run_gt_assembly_expert", execute)
    monkeypatch.setattr(sequence_assembly, "release_and_return_home", cleanup)
    monkeypatch.setattr(sequence_assembly, "verify_connections", lambda task: True)
    result = asyncio.run(
        sequence_assembly.run_dense1_sequence(env, SimpleNamespace())
    )
    assert result.success
    assert result.completed_targets == DENSE1_TARGET_ORDER
    assert [event[2] for event in events if event[0] == "prepare"] == [
        0,
        1,
        0,
        1,
        0,
        1,
    ]
    for target_id in DENSE1_TARGET_ORDER:
        target_events = [event[0] for event in events if event[1] == target_id]
        assert target_events == ["resolve", "prepare", "execute", "home"]


def test_video_recorder_samples_and_finalizes(monkeypatch, tmp_path: Path) -> None:
    """The recorder wraps simulation steps only while attached."""
    written = []
    closed = []

    class Writer:
        def append_data(self, frame: np.ndarray) -> None:
            written.append(frame)

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(video.imageio, "get_writer", lambda *args, **kwargs: Writer())

    class Camera:
        def get_rgb(self) -> np.ndarray:
            return np.ones((4, 6, 4), dtype=np.float32)

    class Environment:
        def __init__(self) -> None:
            self.config = {"BrickSim_Physics": {"FPS": 60}}
            self.cameras = {"Global_Camera": Camera()}
            self.steps = 0

        async def step(self) -> None:
            self.steps += 1

    env = Environment()
    original_step = env.step
    recorder = video.GlobalCameraVideoRecorder(tmp_path / "run.mp4", fps=30)
    recorder.attach(env)
    asyncio.run(env.step())
    asyncio.run(env.step())
    recorder.close()
    assert env.steps == 2
    assert len(written) == recorder.frame_count == 1
    assert written[0].shape == (4, 6, 3)
    assert written[0].dtype == np.uint8
    assert closed == [True]
    assert env.step == original_step


def test_launcher_defaults_to_visible_without_video(monkeypatch) -> None:
    """Launcher defaults do not create recording output or hide the window."""
    path = ROOT / "run/launch_dual_arm_dense1.py"
    spec = importlib.util.spec_from_file_location("dense1_launcher", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr("sys.argv", [str(path)])
    monkeypatch.setattr(module.shutil, "which", lambda name: "/bin/bricksim")
    args = module.parse_args()
    command = module.build_command(args)
    assert "--/app/window/enabled=false" not in command
    assert "--save-video" not in command
    assert args.video_view == "assembly-close"
    assert command[-4:-2] == ["--video-view", "assembly-close"]


def test_close_video_view_targets_dense1_plate() -> None:
    """The default recording view enlarges the assembly without renaming it."""
    config = {
        "Env_Config": {
            "Camera_Config": {
                "Global_Camera": {"Prim_Path": "/World/Global_Camera"}
            }
        }
    }
    video.configure_global_camera_view(config, "assembly-close")
    camera = config["Env_Config"]["Camera_Config"]["Global_Camera"]
    assert camera["Prim_Path"] == "/World/Global_Camera"
    assert camera["Resolution"] == [1280, 720]
    assert camera["Target"] == [0.0, -0.20, 0.035]
    assert camera["Focal_Length"] == 36.0


def test_dense1_worker_always_closes_kit() -> None:
    """The Isaac worker finalizes video and Kit on every exit path."""
    source = (ROOT / "run/demo_dual_arm_dense1.py").read_text(encoding="utf-8")
    assert "finally:" in source
    assert "recorder.close()" in source
    assert "await close_kit_app(env, return_code)" in source
