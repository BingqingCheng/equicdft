"""Deprecated location; use :mod:`equicdft.functional.metalwall`."""

from ._deprecation import warn_moved_module
from .functional.metalwall import (
    MetalWall,
)

__all__ = [
    "MetalWall",
]

warn_moved_module(__name__, "equicdft.functional.metalwall")
