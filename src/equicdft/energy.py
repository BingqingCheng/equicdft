"""Deprecated location; use :mod:`equicdft.functional.base`."""

from ._deprecation import warn_moved_module
from .functional.base import (
    FREE_ENERGY_MODES,
    ReadoutContext,
    EnergyReadout,
)

__all__ = [
    "FREE_ENERGY_MODES",
    "ReadoutContext",
    "EnergyReadout",
]

warn_moved_module(__name__, "equicdft.functional.base")
