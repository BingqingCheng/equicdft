"""Deprecated location; use :mod:`equicdft.functional.pairwise`."""

from ._deprecation import warn_moved_module
from .functional.pairwise import (
    PairwiseReadout,
)

__all__ = [
    "PairwiseReadout",
]

warn_moved_module(__name__, "equicdft.functional.pairwise")
