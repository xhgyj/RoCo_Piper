from __future__ import annotations

import asyncio

import pytest

from rocobrick.supervision import SafetyViolation, await_with_physics


class Physics:
    def __init__(self, safe=True):
        self.steps = 0
        self.safe = safe

    async def step(self):
        self.steps += 1

    def sample_feedback(self):
        return True

    def safety_valid(self):
        return self.safe


@pytest.mark.asyncio
async def test_physics_and_feedback_continue_while_planning_waits():
    physics = Physics()

    async def planning():
        for _ in range(5):
            await asyncio.sleep(0)
        return "planned"

    result, metrics = await await_with_physics(planning(), physics)
    assert result == "planned"
    assert metrics.physics_steps >= 5
    assert metrics.feedback_samples == metrics.physics_steps


@pytest.mark.asyncio
async def test_safety_failure_cancels_pending_work():
    physics = Physics(safe=False)

    async def planning():
        await asyncio.sleep(10)

    with pytest.raises(SafetyViolation):
        await await_with_physics(planning(), physics)
