"""Deprecated location; use :mod:`equicdft.functional.readout`."""

from ._deprecation import warn_moved_module
from .functional.readout import (
    LocalReadout,
    BulkReadout,
    LongRangeReadout,
)

__all__ = [
    "LocalReadout",
    "BulkReadout",
    "LongRangeReadout",
]

warn_moved_module(__name__, "equicdft.functional.readout")
