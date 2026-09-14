"""Runtime supervision that never pauses physics while planning is pending."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import time
from typing import Awaitable, Protocol, TypeVar


class PhysicsLoop(Protocol):
    async def step(self) -> None: ...
    def sample_feedback(self) -> bool: ...
    def safety_valid(self) -> bool: ...


class SafetyViolation(RuntimeError):
    pass


class NoProgressTimeout(TimeoutError):
    pass


T = TypeVar("T")


@dataclass(frozen=True)
class SupervisionMetrics:
    physics_steps: int
    feedback_samples: int
    elapsed_s: float


async def await_with_physics(
    operation: Awaitable[T],
    physics: PhysicsLoop,
    *,
    no_progress_timeout_s: float = 30.0,
    monotonic=time.monotonic,
) -> tuple[T, SupervisionMetrics]:
    """Await work while advancing physics, feedback, and safety checks."""
    task = asyncio.ensure_future(operation)
    started = monotonic()
    last_progress = started
    physics_steps = 0
    feedback_samples = 0
    try:
        while not task.done():
            await physics.step()
            physics_steps += 1
            if physics.sample_feedback():
                feedback_samples += 1
                last_progress = monotonic()
            if not physics.safety_valid():
                task.cancel()
                raise SafetyViolation("safety condition became invalid while operation was pending")
            if monotonic() - last_progress >= no_progress_timeout_s:
                task.cancel()
                raise NoProgressTimeout(f"no confirmable progress for {no_progress_timeout_s:.3f}s")
            await asyncio.sleep(0)
        result = await task
        return result, SupervisionMetrics(physics_steps, feedback_samples, monotonic() - started)
    finally:
        if not task.done():
            task.cancel()
