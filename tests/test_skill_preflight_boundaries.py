"""Admission methods must stay separate from free-space dispatch."""

import asyncio
from types import SimpleNamespace

import pytest

from rocobrick.execution.operation_state_ledger import OperationStateLedger
from rocobrick.skills.place_up import PlaceUpHandler
from rocobrick.skills.support_top import SupportTopHandler


@pytest.mark.parametrize("handler_type", [PlaceUpHandler, SupportTopHandler])
def test_upward_skill_preflight_refuses_invalid_skill_without_observation(handler_type):
    """A refactor must not hide admission below a free-space method's return."""
    calls = []

    async def observe(robot_id):
        calls.append(robot_id)
        pytest.fail("invalid operation must be refused before observing")

    handler = handler_type(
        SimpleNamespace(observe=observe),
        OperationStateLedger(("piper_0", "piper_1")),
        session_id="test-session",
        manifest_sha256="a" * 64,
    )
    operation = SimpleNamespace(skill="WAIT", target=None)
    with pytest.raises(ValueError, match="target is required"):
        asyncio.run(handler.preflight(operation))
    assert calls == []
