"""Local excess-free-energy readout."""

from numbers import Integral
from typing import Optional, Sequence

import torch
from torch import nn


class LocalFreeEnergyReadout(nn.Module):
    """Predict a local excess free energy per particle and component.

    A shared MLP maps the invariant ``B`` features at grid point
    ``g`` to one dimensionless excess free energy per particle and component,

    ``beta_a_exc[g, i] = MLP(B[g])[i]``.

    Density weighting, grid integration, and functional differentiation are
    handled by :class:`cace_grid.model.GridCACEModel`.

    Parameters
    ----------
    n_features
        Flattened input width of ``B``. If supplied, the first layer is a
        regular ``Linear`` module. If ``None``, a ``LazyLinear`` module infers
        this width from the first forward pass.
    n_types
        Number of physical density components and per-particle outputs.
    hidden_sizes
        Width of each hidden layer. An empty sequence gives a linear readout.
    Notes
    -----
    When ``n_features`` is ``None``, materialize the lazy input layer with one
    representative batch before constructing an optimizer or saving its
    initial state.

    Smooth SiLU activations keep higher functional derivatives well defined.
    """

    def __init__(
        self,
        n_types: int = 1,
        hidden_sizes: Sequence[int] = (32, 16),
        n_features: Optional[int] = None,
    ) -> None:
        super().__init__()

        if n_features is not None:
            if isinstance(n_features, bool) or not isinstance(
                n_features,
                Integral,
            ):
                raise TypeError("n_features must be a positive integer or None")
            n_features = int(n_features)
            if n_features < 1:
                raise ValueError("n_features must be a positive integer or None")

        self.n_features = n_features
        self.n_types = n_types

        layers = []
        for layer_index, n_hidden in enumerate(hidden_sizes):
            if layer_index == 0:
                linear = (
                    nn.LazyLinear(n_hidden)
                    if n_features is None
                    else nn.Linear(n_features, n_hidden)
                )
            else:
                linear = nn.Linear(hidden_sizes[layer_index - 1], n_hidden)
            layers.extend((linear, nn.SiLU()))
        if hidden_sizes:
            layers.append(nn.Linear(hidden_sizes[-1], n_types))
        else:
            layers.append(
                nn.LazyLinear(n_types)
                if n_features is None
                else nn.Linear(n_features, n_types)
            )
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        B: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``beta_a_exc`` with shape ``[..., n_grid, n_types]``."""

        # Flatten the radial, invariant, and input-channel dimensions while
        # retaining every leading configuration dimension and the grid axis.
        B_flat = B.flatten(start_dim=-3)
        return self.mlp(B_flat)
