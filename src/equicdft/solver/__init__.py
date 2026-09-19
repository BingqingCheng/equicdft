"""Equilibrium density solvers for a trained functional.

``GridSolver`` is re-exported here so ``equicdft.solver`` keeps its
historical meaning as the module that offered it.
"""

from .solver import GridSolver

__all__ = ["GridSolver"]
