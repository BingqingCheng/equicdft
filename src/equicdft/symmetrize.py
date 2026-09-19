"""Deprecated location; use :mod:`equicdft.representation.invariants`."""

from ._deprecation import warn_moved_module
from .representation.invariants import (
    CartesianBFeatures,
)

__all__ = [
    "CartesianBFeatures",
]

warn_moved_module(__name__, "equicdft.representation.invariants")
