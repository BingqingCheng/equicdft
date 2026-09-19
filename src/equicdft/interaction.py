"""Deprecated location; use :mod:`equicdft.representation.message`."""

from ._deprecation import warn_moved_module
from .representation.message import (
    BChiMessage,
)

__all__ = [
    "BChiMessage",
]

warn_moved_module(__name__, "equicdft.representation.message")
