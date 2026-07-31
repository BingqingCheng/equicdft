"""Neural density-functional models for periodic Cartesian grids."""

from .data import GridData, default_data_key
from .derivatives import compute_grid_derivative
from .features import CartesianAFeatures
from .loader import make_dataloaders
from .loss import Loss, TensorLoss
from .metrics import Metrics, compute_metric
from .model import GridCACEModel
from .readout import LocalFreeEnergyReadout
from .stencil import (
    coarsen_grid,
    get_neighbor_indices,
    make_stencil,
)
from .symmetrize import CartesianBFeatures
from .trainer import Trainer

__all__ = [
    "CartesianAFeatures",
    "CartesianBFeatures",
    "GridData",
    "GridCACEModel",
    "LocalFreeEnergyReadout",
    "Loss",
    "Metrics",
    "TensorLoss",
    "Trainer",
    "coarsen_grid",
    "compute_grid_derivative",
    "compute_metric",
    "default_data_key",
    "get_neighbor_indices",
    "make_dataloaders",
    "make_stencil",
]

__version__ = "0.0.1"
