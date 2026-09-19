"""Deprecated location; use :mod:`equicdft.training.trainer`."""

from ._deprecation import warn_moved_module
from .training.trainer import (
    TrainingStream,
    Trainer,
)

__all__ = [
    "TrainingStream",
    "Trainer",
]

warn_moved_module(__name__, "equicdft.training.trainer")
