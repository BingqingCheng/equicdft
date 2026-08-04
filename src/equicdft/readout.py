"""Local readout for invariant grid features."""

from numbers import Integral
from typing import Optional, Sequence

import torch
from torch import nn


class LocalReadout(nn.Module):
    """Map local invariant features to one output per physical component.

    A shared MLP maps a local feature vector at each grid point to one output
    per component. The readout itself is agnostic about the construction and
    physical meaning of its inputs and outputs.
    :class:`equicdft.model.GridCACEModel` supplies flattened invariant B
    features and temperature, and interprets the outputs as dimensionless
    per-particle excess free energies.

    Density weighting, grid integration, and functional differentiation are
    handled by :class:`equicdft.model.GridCACEModel`.

    Parameters
    ----------
    n_features
        Width of the complete local input vector. If supplied, the first layer
        is a regular ``Linear`` module. If ``None``, a ``LazyLinear``
        module infers this width from the first forward pass.
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
        local_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return local outputs with shape ``[..., n_grid, n_types]``."""

        return self.mlp(local_features)


class BulkReadout(nn.Module):
    """Map temperature and mean densities to a bulk free energy per particle.

    The state vector contains normalized temperature followed by one
    normalized mean density per physical component. The output contains one
    dimensionless bulk excess free energy per particle and component. The
    model combines it with the particle numbers according to

    ``beta_F_exc_bulk = sum_i N_i * beta_a_exc_bulk_i``.

    Parameters
    ----------
    n_types
        Number of physical density components.
    hidden_sizes
        Width of each hidden layer. An empty sequence gives a linear readout.
    zero_init
        If true, initialize the final layer to zero. Attaching the branch then
        leaves a pretrained local model unchanged before fine-tuning.
    """

    def __init__(
        self,
        n_types: int = 1,
        hidden_sizes: Sequence[int] = (16, 16),
        zero_init: bool = True,
    ) -> None:
        super().__init__()

        if isinstance(n_types, bool) or not isinstance(n_types, Integral):
            raise TypeError("n_types must be a positive integer")
        if int(n_types) < 1:
            raise ValueError("n_types must be a positive integer")
        if not isinstance(zero_init, bool):
            raise TypeError("zero_init must be a boolean")

        self.n_types = int(n_types)
        self.n_state_features = 1 + self.n_types

        layers = []
        input_width = self.n_state_features
        for hidden_width in hidden_sizes:
            if isinstance(hidden_width, bool) or not isinstance(
                hidden_width,
                Integral,
            ):
                raise TypeError("hidden_sizes must contain positive integers")
            hidden_width = int(hidden_width)
            if hidden_width < 1:
                raise ValueError(
                    "hidden_sizes must contain positive integers"
                )
            layers.extend((nn.Linear(input_width, hidden_width), nn.SiLU()))
            input_width = hidden_width

        final_layer = nn.Linear(input_width, self.n_types)
        if zero_init:
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)
        self.mlp = nn.Sequential(*layers)

    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        """Return ``beta_a_exc_bulk`` with shape ``[..., n_types]``."""

        if state_features.shape[-1] != self.n_state_features:
            raise ValueError(
                "state_features must end with normalized temperature and "
                "one mean density per type"
            )
        return self.mlp(state_features)


class LongRangeReadout(nn.Module):
    """Map thermodynamic state to a reciprocal quadratic kernel.

    The readout predicts one coefficient for every fixed reciprocal kernel and
    unique density-component pair. Its output energy is a linear contraction
    with the reciprocal features, preserving their quadratic density
    dependence and extensive scaling.

    Parameters
    ----------
    n_kernels
        Number of fixed radial kernels in the reciprocal representation.
    n_types
        Number of physical density components. The state vector contains
        normalized temperature followed by one mean density per component.
    hidden_sizes
        Width of each state-network hidden layer. An empty sequence gives a
        linear state dependence.
    zero_init
        If true, initialize the final coefficient layer to zero. This makes an
        attached long-range branch leave a pretrained local model unchanged.
    """

    def __init__(
        self,
        n_kernels: int,
        n_types: int = 1,
        hidden_sizes: Sequence[int] = (16, 16),
        zero_init: bool = True,
    ) -> None:
        super().__init__()

        for value, name in ((n_kernels, "n_kernels"), (n_types, "n_types")):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError("{} must be a positive integer".format(name))
            if int(value) < 1:
                raise ValueError("{} must be a positive integer".format(name))
        if not isinstance(zero_init, bool):
            raise TypeError("zero_init must be a boolean")

        self.n_kernels = int(n_kernels)
        self.n_types = int(n_types)
        self.n_type_pairs = self.n_types * (self.n_types + 1) // 2
        self.n_state_features = 1 + self.n_types

        layers = []
        input_width = self.n_state_features
        for hidden_width in hidden_sizes:
            if isinstance(hidden_width, bool) or not isinstance(
                hidden_width,
                Integral,
            ):
                raise TypeError("hidden_sizes must contain positive integers")
            hidden_width = int(hidden_width)
            if hidden_width < 1:
                raise ValueError(
                    "hidden_sizes must contain positive integers"
                )
            layers.extend((nn.Linear(input_width, hidden_width), nn.SiLU()))
            input_width = hidden_width

        output_width = self.n_kernels * self.n_type_pairs
        final_layer = nn.Linear(input_width, output_width)
        if zero_init:
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)
        self.mlp = nn.Sequential(*layers)

    def coefficients(self, state_features: torch.Tensor) -> torch.Tensor:
        """Return coefficients shaped ``[..., n_kernels, n_type_pairs]``."""

        if state_features.shape[-1] != self.n_state_features:
            raise ValueError(
                "state_features must end with normalized temperature and "
                "one mean density per type"
            )
        return self.mlp(state_features).reshape(
            *state_features.shape[:-1],
            self.n_kernels,
            self.n_type_pairs,
        )

    def forward(
        self,
        reciprocal_features: torch.Tensor,
        state_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return one dimensionless long-range free energy per field."""

        expected_trailing_shape = (self.n_kernels, self.n_type_pairs)
        if reciprocal_features.shape[-2:] != expected_trailing_shape:
            raise ValueError(
                "reciprocal_features must end with shape {}".format(
                    expected_trailing_shape
                )
            )
        if reciprocal_features.shape[:-2] != state_features.shape[:-1]:
            raise ValueError(
                "reciprocal and state features must have matching leading shapes"
            )
        coefficients = self.coefficients(state_features)
        return torch.sum(
            coefficients * reciprocal_features,
            dim=(-2, -1),
        )
