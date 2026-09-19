"""Deprecated location; use :mod:`equicdft.functional.thermodynamics`."""

from ._deprecation import warn_moved_module
from .functional.thermodynamics import (
    density_weighted_integral,
    log_dimensionless_density,
    ideal_free_energy,
    local_chemical_potential,
    equilibrium_log_density,
    beta_external_potential,
)

__all__ = [
    "density_weighted_integral",
    "log_dimensionless_density",
    "ideal_free_energy",
    "local_chemical_potential",
    "equilibrium_log_density",
    "beta_external_potential",
]

warn_moved_module(__name__, "equicdft.functional.thermodynamics")
