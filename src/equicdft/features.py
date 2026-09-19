"""Deprecated location; use :mod:`equicdft.representation.cartesian`."""

from ._deprecation import warn_moved_module
from .representation.cartesian import (
    CartesianAFeatures,
)

__all__ = [
    "CartesianAFeatures",
]

warn_moved_module(__name__, "equicdft.representation.cartesian")
