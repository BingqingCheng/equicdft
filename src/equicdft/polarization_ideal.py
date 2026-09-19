"""Deprecated location; use :mod:`equicdft.functional.polarization_ideal`."""

from ._deprecation import warn_moved_module
from .functional.polarization_ideal import (
    langevin,
    log_sinhc,
    inverse_langevin,
    dipole_alignment,
    FixedDipoleIdeal,
)

__all__ = [
    "langevin",
    "log_sinhc",
    "inverse_langevin",
    "dipole_alignment",
    "FixedDipoleIdeal",
]

warn_moved_module(__name__, "equicdft.functional.polarization_ideal")
