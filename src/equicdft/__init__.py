"""Neural density-functional models for periodic Cartesian grids."""

from ._version import __version__
from .data import FourierResponseData, GridData, default_data_key, read_metal_sites
from .functional.base import EnergyReadout
from .representation.cartesian import CartesianAFeatures
from .representation.message import BChiMessage
from .loader import make_dataloaders
from .loss import FourierResponseLoss, Loss, LossTerm, TensorLoss
from .metrics import FourierResponseMetrics, MetricCollection, Metrics
from .functional.metalwall import MetalWall
from .model import GridCACEModel
from .functional.pairwise import PairwiseReadout
from .functional.polarization_ideal import FixedDipoleIdeal
from .functional.readout import BulkReadout, LocalReadout, LongRangeReadout
from .functional.reciprocal import ReciprocalFeatures
from .response import FourierResponse
from .functional.semilocal import GGAReadout, LDAReadout
from .serialization import (
    MODEL_FORMAT,
    MODEL_FORMAT_VERSION,
    load_model,
    read_model_config,
    save_model,
)
from .solver import GridSolver
from .stability import FourierStabilityLoss
from .representation.invariants import CartesianBFeatures
from .trainer import Trainer, TrainingStream

__all__ = [
    "__version__",
    "CartesianAFeatures",
    "CartesianBFeatures",
    "BulkReadout",
    "BChiMessage",
    "GridData",
    "GridCACEModel",
    "GridSolver",
    "GGAReadout",
    "EnergyReadout",
    "FourierResponseLoss",
    "FourierResponse",
    "FourierResponseData",
    "FourierResponseMetrics",
    "FourierStabilityLoss",
    "LocalReadout",
    "LDAReadout",
    "LongRangeReadout",
    "Loss",
    "LossTerm",
    "MODEL_FORMAT",
    "MODEL_FORMAT_VERSION",
    "MetricCollection",
    "Metrics",
    "MetalWall",
    "PairwiseReadout",
    "FixedDipoleIdeal",
    "ReciprocalFeatures",
    "TensorLoss",
    "Trainer",
    "TrainingStream",
    "default_data_key",
    "load_model",
    "make_dataloaders",
    "read_metal_sites",
    "read_model_config",
    "save_model",
]
