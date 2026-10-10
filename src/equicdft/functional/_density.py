"""Local density sources shared by real- and reciprocal-space readouts."""

from typing import Optional

import torch
from torch import nn

from .base import ReadoutContext


def density_fields(
    context: ReadoutContext,
    n_types: int,
    density_mlp: Optional[nn.Module],
    n_channels: Optional[int],
    include_temperature: bool,
) -> torch.Tensor:
    """Return rho, or ``f_a = sum_i rho_i A_ia(rho / rho_scale, T)``.

    The scale is the fixed model normalization, not a field's current mean.
    Physical rho guarantees zero sources in vacuum, with all derivatives
    through the local mapping retained.
    """

    rho = context.rho
    if rho.ndim < 2 or rho.shape[-1] != n_types:
        raise ValueError(
            "rho must have shape [..., n_grid, n_types] with the "
            "configured n_types"
        )
    if density_mlp is None:
        return rho
    inputs = (
        context.pointwise_state() if include_temperature
        else context.normalized_rho()
    )
    coefficients = density_mlp(inputs).reshape(*rho.shape, n_channels)
    return torch.sum(rho[..., None] * coefficients, dim=-2)
