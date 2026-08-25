"""Video recording helpers for BrickSim workflows."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np

GLOBAL_CAMERA_VIEWS = {
    "overview": {
        "Position": [0.0, 1.05, 0.95],
        "Target": [0.0, -0.12, 0.24],
        "Focal_Length": 16.0,
        "Resolution": [960, 540],
    },
    "assembly-close": {
        "Position": [0.0, 0.42, 0.42],
        "Target": [0.0, -0.20, 0.035],
        "Focal_Length": 36.0,
        "Resolution": [1280, 720],
    },
}


def configure_global_camera_view(config: dict, view: str) -> None:
    """Apply a named global-camera view to a mutable user configuration."""
    if view not in GLOBAL_CAMERA_VIEWS:
        raise ValueError(
            f"unknown video view {view!r}; "
            f"choose from {sorted(GLOBAL_CAMERA_VIEWS)}"
        )
    camera = config["Env_Config"]["Camera_Config"]["Global_Camera"]
    for key, value in GLOBAL_CAMERA_VIEWS[view].items():
        camera[key] = list(value) if isinstance(value, list) else value


class GlobalCameraVideoRecorder:
    """Write the configured global RGB camera to an H.264 MP4 file."""

    def __init__(self, output_path: str | Path, fps: int = 30) -> None:
        """Initialize an unattached recorder for one output file."""
        if fps <= 0:
            raise ValueError("video fps must be positive")
        self.output_path = Path(output_path)
        self.fps = fps
        self.frame_count = 0
        self._step_count = 0
        self._step_stride = 1
        self._writer = None
        self._attached_env = None
        self._original_step = None

    def attach(self, env) -> None:
        """Attach to an initialized environment and begin frame sampling."""
        if self._attached_env is not None:
            raise RuntimeError("video recorder is already attached")
        if "Global_Camera" not in env.cameras:
            raise RuntimeError("Global_Camera is unavailable")
        physics_fps = int(env.config["BrickSim_Physics"]["FPS"])
        self._step_stride = max(1, round(physics_fps / self.fps))
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio.get_writer(
            self.output_path,
            fps=self.fps,
            codec="libx264",
            macro_block_size=None,
        )
        self._attached_env = env
        self._original_step = env.step

        async def recording_step() -> None:
            await self._original_step()
            self.on_step(env)

        env.step = recording_step

    def on_step(self, env) -> None:
        """Append one sampled RGB frame after a simulation update."""
        self._step_count += 1
        if self._writer is None or self._step_count % self._step_stride:
            return
        value = env.cameras["Global_Camera"].get_rgb()
        if value is None:
            return
        frame = np.asarray(value)
        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            raise RuntimeError(f"invalid global camera RGB shape: {frame.shape}")
        frame = frame[:, :, :3]
        if frame.dtype != np.uint8:
            if np.issubdtype(frame.dtype, np.floating) and frame.max(initial=0) <= 1.0:
                frame = frame * 255.0
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        self._writer.append_data(np.ascontiguousarray(frame))
        self.frame_count += 1

    def close(self) -> None:
        """Detach from the environment and finalize the MP4 container."""
        if self._attached_env is not None:
            self._attached_env.step = self._original_step
            self._attached_env = None
            self._original_step = None
        if self._writer is not None:
            self._writer.close()
            self._writer = None
