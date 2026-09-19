"""Deprecated location; use :mod:`equicdft.functional.semilocal`."""

from ._deprecation import warn_moved_module
from .functional.semilocal import (
    LDAReadout,
    GGAReadout,
    periodic_gradient_energy_density,
)

__all__ = [
    "LDAReadout",
    "GGAReadout",
    "periodic_gradient_energy_density",
]

warn_moved_module(__name__, "equicdft.functional.semilocal")
