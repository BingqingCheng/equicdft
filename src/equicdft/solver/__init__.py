"""Equilibrium density solvers for a trained functional.

``GridSolver`` is re-exported here so ``equicdft.solver`` keeps its
historical meaning as the module that offered it; the option groups of
:meth:`GridSolver.solve` live in :mod:`equicdft.solver.options`.
"""

from .options import (
    AndersonOptions,
    Constraints,
    ConvergenceOptions,
    EulerOptions,
    Initialization,
    MinimizeOptions,
    SolveOptions,
)
from .solver import GridSolver

__all__ = [
    "AndersonOptions",
    "Constraints",
    "ConvergenceOptions",
    "EulerOptions",
    "GridSolver",
    "Initialization",
    "MinimizeOptions",
    "SolveOptions",
]
