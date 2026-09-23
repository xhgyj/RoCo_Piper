"""Compatibility seam for ``enum.StrEnum`` across runtime Pythons.

The host packaging pins Python 3.11 (see ``pyproject`` ``requires-python``),
where ``enum.StrEnum`` exists. The ROS Humble runtime container ships Python
3.10, where it does not, so the formal library modules import ``StrEnum``
from here when mounted read-only into that container. When the standard
library provides ``StrEnum`` it is re-exported unchanged; otherwise a minimal
value-shaped string enum base is defined locally. No stdlib module is ever
patched. The module exports exactly one name: ``StrEnum``.
"""

from __future__ import annotations

from enum import Enum

try:
    from enum import StrEnum as StrEnum
except ImportError:  # pragma: no cover - Python 3.10 (ROS Humble container)

    class StrEnum(str, Enum):
        """Minimal stand-in for ``enum.StrEnum``: members are their values."""

        def __str__(self) -> str:
            """Render the member as its string value.

            Returns:
                The member's string value, matching ``enum.StrEnum``.
            """
            return str(self.value)


__all__ = ["StrEnum"]
