"""Deprecated location; use :mod:`equicdft.training.loss`."""

from ._deprecation import warn_moved_module
from .training.loss import (
    LossTerm,
    TensorLoss,
    Loss,
)
from .response.loss import FourierResponseLoss

__all__ = [
    "FourierResponseLoss",
    "LossTerm",
    "TensorLoss",
    "Loss",
]

warn_moved_module(__name__, "equicdft.training.loss")
