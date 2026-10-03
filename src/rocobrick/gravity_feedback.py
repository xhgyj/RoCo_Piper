"""Pure bounded articulation gravity feed-forward validation."""

from __future__ import annotations

import math
from collections.abc import Sequence


def arm_gravity_efforts(
    names: Sequence[str], values: Sequence[float], limit_nm: float = 10.0
) -> tuple[float, ...]:
    """Admit gravity compensation without modifying finger forces or gains.

    Returns:
        Finite per-DOF efforts; non-arm DOFs receive exactly zero.
    """
    if len(names) != len(values) or not names:
        raise ValueError("gravity vector must exactly match articulation joints")
    result = []
    for name, raw in zip(names, values):
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (float, int))
            or not math.isfinite(raw)
        ):
            raise ValueError("gravity model returned invalid efforts")
        local = str(name).rsplit("_", 1)[-1]
        if local in {f"joint{i}" for i in range(1, 7)} and "gripper" not in str(name):
            if abs(raw) > limit_nm:
                raise ValueError("gravity model exceeds bounded arm effort")
            result.append(float(raw))
        else:
            result.append(0.0)
    return tuple(result)
