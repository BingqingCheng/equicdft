"""Deprecated location; use :mod:`equicdft.geometry.stencil`."""

from ._deprecation import warn_moved_module
from .geometry.stencil import (
    grid_size_array,
    grid_spacing_array,
    GridOrder,
    canonical_grid_order,
    make_stencil,
    coarsen_grid,
    get_neighbor_indices,
)

__all__ = [
    "grid_size_array",
    "grid_spacing_array",
    "GridOrder",
    "canonical_grid_order",
    "make_stencil",
    "coarsen_grid",
    "get_neighbor_indices",
]

warn_moved_module(__name__, "equicdft.geometry.stencil")
