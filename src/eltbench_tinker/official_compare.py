from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation
from typing import Any, Sequence


def _number(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None


def values_match(actual: Any, expected: Any, rtol: float = 1e-2, atol: float = 1e-9) -> bool:
    """Mirror ELT-Bench stage-2 numeric/string comparison semantics."""
    if actual is None or expected is None:
        return actual is None and expected is None
    left, right = _number(actual), _number(expected)
    if left is not None and right is not None:
        a, b = float(left), float(right)
        return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)
    if isinstance(actual, str) and isinstance(expected, str):
        return actual.strip().casefold() == expected.strip().casefold()
    return actual == expected


def rows_match(
    actual: Sequence[Sequence[Any]],
    expected: Sequence[Sequence[Any]],
    *,
    key_indices: Sequence[int] = (),
    rtol: float = 1e-2,
    atol: float = 1e-9,
) -> bool:
    """Compare rows deterministically after key-first canonical sorting."""
    if len(actual) != len(expected):
        return False

    def sort_key(row: Sequence[Any]) -> tuple[str, ...]:
        order = list(key_indices) + [i for i in range(len(row)) if i not in key_indices]
        return tuple("<null>" if row[i] is None else str(row[i]).strip().casefold() for i in order)

    left_rows = sorted(actual, key=sort_key)
    right_rows = sorted(expected, key=sort_key)
    return all(
        len(left) == len(right)
        and all(values_match(a, b, rtol=rtol, atol=atol) for a, b in zip(left, right))
        for left, right in zip(left_rows, right_rows)
    )
