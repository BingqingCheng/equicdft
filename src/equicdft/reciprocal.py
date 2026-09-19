"""Deprecated location; use :mod:`equicdft.functional.reciprocal`."""

from ._deprecation import warn_moved_module
from .functional.reciprocal import (
    ReciprocalFeatures,
)

__all__ = [
    "ReciprocalFeatures",
]

warn_moved_module(__name__, "equicdft.functional.reciprocal")
