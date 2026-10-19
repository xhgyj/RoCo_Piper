"""Strict primitive validation shared by the cross-module contracts.

The helpers here are the single place where JSON shape, scalar type, finiteness
and uniqueness rules are expressed. ``bool`` is never accepted as a number and
``int`` is never accepted where a ``bool`` is required.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import TypeVar

EnumT = TypeVar("EnumT", bound=Enum)


class ContractViolationError(ValueError):
    """A contract payload is structurally invalid or internally inconsistent."""


def require_schema(value: object, expected: str, path: str) -> str:
    """Require an exact schema discriminator.

    Args:
        value: Candidate schema value.
        expected: Frozen schema identifier.
        path: Diagnostic path used in the error message.

    Returns:
        The validated schema identifier.
    """
    schema = require_text(value, f"{path}.schema")
    if schema != expected:
        raise ContractViolationError(
            f"{path}.schema must be {expected!r}, not {schema!r}"
        )
    return schema


def require_object(value: object, path: str) -> Mapping[str, object]:
    """Require one JSON object with string keys.

    Args:
        value: Candidate value.
        path: Diagnostic path used in the error message.

    Returns:
        The validated mapping.
    """
    if not isinstance(value, Mapping):
        raise ContractViolationError(f"{path} must be a JSON object")
    for key in value:
        if not isinstance(key, str):
            raise ContractViolationError(f"{path} keys must be strings")
    return value


def require_keys(
    value: Mapping[str, object], expected: frozenset[str], path: str
) -> None:
    """Require an exact key set, rejecting missing and unknown keys.

    Args:
        value: Validated mapping.
        expected: Frozen expected key set.
        path: Diagnostic path used in the error message.
    """
    actual = frozenset(value)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    raise ContractViolationError(
        f"{path} keys mismatch: missing {missing}, unexpected {unknown}"
    )


def require_array(value: object, path: str) -> tuple[object, ...]:
    """Require one JSON array (strings and mappings are rejected).

    Args:
        value: Candidate value.
        path: Diagnostic path used in the error message.

    Returns:
        The array items as a tuple.
    """
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ContractViolationError(f"{path} must be a JSON array")
    return tuple(value)


def require_text(value: object, path: str) -> str:
    """Require a non-empty string.

    Args:
        value: Candidate value.
        path: Diagnostic path used in the error message.

    Returns:
        The validated string.
    """
    if not isinstance(value, str):
        raise ContractViolationError(f"{path} must be a string")
    if not value.strip():
        raise ContractViolationError(f"{path} must be a non-empty string")
    return value


def require_optional_text(value: object, path: str) -> str | None:
    """Require a non-empty string or ``None``.

    Args:
        value: Candidate value.
        path: Diagnostic path used in the error message.

    Returns:
        The validated string, or ``None``.
    """
    if value is None:
        return None
    return require_text(value, path)


def require_int(value: object, path: str, *, minimum: int = 0) -> int:
    """Require a non-boolean integer of at least ``minimum``.

    Args:
        value: Candidate value.
        path: Diagnostic path used in the error message.
        minimum: Inclusive lower bound.

    Returns:
        The validated integer.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractViolationError(
            f"{path} must be an integer, not {type(value).__name__}"
        )
    if value < minimum:
        raise ContractViolationError(f"{path} must be >= {minimum}")
    return value


def require_float(value: object, path: str, *, minimum: float | None = None) -> float:
    """Require a finite non-boolean number of at least ``minimum``.

    Integers are accepted because JSON does not distinguish ``0`` from ``0.0``;
    booleans, non-finite values and anything else are rejected.

    Args:
        value: Candidate value.
        path: Diagnostic path used in the error message.
        minimum: Optional inclusive lower bound.

    Returns:
        The validated number as a float.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractViolationError(
            f"{path} must be a finite number, not {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ContractViolationError(f"{path} must be finite")
    if minimum is not None and number < minimum:
        raise ContractViolationError(f"{path} must be >= {minimum}")
    return number


def require_bool(value: object, path: str) -> bool:
    """Require an actual boolean, never a truthy value.

    Args:
        value: Candidate value.
        path: Diagnostic path used in the error message.

    Returns:
        The validated boolean.
    """
    if type(value) is not bool:
        raise ContractViolationError(
            f"{path} must be a boolean, not {type(value).__name__}"
        )
    return value


def require_enum(value: object, path: str, expected: type[EnumT]) -> EnumT:
    """Require an instance of one specific enum.

    Args:
        value: Candidate value.
        path: Diagnostic path used in the error message.
        expected: Enum type the value must belong to.

    Returns:
        The validated enum member.
    """
    if not isinstance(value, expected):
        raise ContractViolationError(
            f"{path} must be a {expected.__name__}, not {type(value).__name__}"
        )
    return value


def parse_enum(value: object, path: str, expected: type[EnumT]) -> EnumT:
    """Parse one enum member from its wire string.

    Args:
        value: Candidate wire value.
        path: Diagnostic path used in the error message.
        expected: Enum type to parse.

    Returns:
        The parsed enum member.
    """
    text = require_text(value, path)
    try:
        return expected(text)
    except ValueError:
        raise ContractViolationError(
            f"{path} {text!r} is not a supported {expected.__name__}"
        ) from None


def require_texts(
    value: object, path: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    """Require an array of unique non-empty strings.

    Args:
        value: Candidate value.
        path: Diagnostic path used in the error message.
        allow_empty: Whether the empty array is acceptable.

    Returns:
        The validated strings as a tuple.
    """
    items = tuple(
        require_text(item, f"{path}[]") for item in require_array(value, path)
    )
    if not items and not allow_empty:
        raise ContractViolationError(f"{path} must not be empty")
    if len(items) != len(set(items)):
        raise ContractViolationError(f"{path} must not contain duplicates")
    return items
