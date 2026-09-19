"""Deprecated location; use :mod:`equicdft.data.loaders`."""

from ._deprecation import warn_moved_module
from .data.loaders import (
    LoaderResult,
    make_dataloaders,
)

__all__ = [
    "LoaderResult",
    "make_dataloaders",
]

warn_moved_module(__name__, "equicdft.data.loaders")
