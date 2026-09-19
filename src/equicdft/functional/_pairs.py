"""Canonical ordering for unique symmetric density-component pairs."""

from typing import Tuple


def symmetric_component_pairs(
    n_types: int,
) -> Tuple[Tuple[int, int], ...]:
    """Return upper-triangular component pairs in row-major order."""

    return tuple(
        (first, second)
        for first in range(n_types)
        for second in range(first, n_types)
    )
