"""Physical constants of the unit contract shared by data and models.

``GridData`` derives every frame's ``beta = 1 / (k_B T)`` from the Boltzmann
constant it is built with, and ``GridCACEModel`` uses its own copy for the
physical energy scale ``k_B * mean_temperature``. Both default to the same
value, temperatures in kelvin and energies in electronvolts; reduced-unit
work passes ``boltzmann_constant=1.0`` to both. The model checks a batch's
``beta`` against its constant on every forward pass, so a dataset and a
model built with different constants cannot be combined silently.
"""

BOLTZMANN_CONSTANT_EV_PER_K = 8.617333262e-5
"""Boltzmann constant in eV/K (CODATA 2018, exact in SI)."""

DEFAULT_BOLTZMANN_CONSTANT = BOLTZMANN_CONSTANT_EV_PER_K
"""The value ``GridData`` and ``GridCACEModel`` use when none is given."""

__all__ = ["BOLTZMANN_CONSTANT_EV_PER_K", "DEFAULT_BOLTZMANN_CONSTANT"]
