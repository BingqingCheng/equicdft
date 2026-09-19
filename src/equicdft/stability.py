"""Deprecated location; use :mod:`equicdft.response.stability`."""

from ._deprecation import warn_moved_module
from .response.stability import (
    FourierStabilityLoss,
)

__all__ = [
    "FourierStabilityLoss",
]

warn_moved_module(__name__, "equicdft.response.stability")
