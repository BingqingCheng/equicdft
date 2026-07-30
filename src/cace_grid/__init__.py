"""Neural density-functional models for periodic Cartesian grids."""

from .data import GridData, default_data_key
from .derivatives import compute_c1
from .features import CartesianAFeatures
from .model import GridCACEModel
from .readout import LocalFreeEnergyReadout
from .stencil import (
    coarsen_grid,
    get_neighbor_indices,
    make_stencil,
)
from .symmetrize import CartesianBFeatures

__all__ = [
    "CartesianAFeatures",
    "CartesianBFeatures",
    "GridData",
    "GridCACEModel",
    "LocalFreeEnergyReadout",
    "coarsen_grid",
    "compute_c1",
    "default_data_key",
    "get_neighbor_indices",
    "make_stencil",
]

__version__ = "0.0.1"
