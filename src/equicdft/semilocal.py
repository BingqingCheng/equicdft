"""Pointwise and gradient-expanded free-energy building blocks."""

import math
from typing import Dict, Optional, Sequence, Union

import torch
from torch import nn

from ._argument_checks import (
    nonnegative_scalar,
    optional_positive_integer,
    positive_integer,
    positive_scalar,
)
from ._nn import (
    build_mlp,
    positive_scalar_tensor,
    validate_hidden_sizes,
)
from .energy import EnergyReadout


class LDAReadout(EnergyReadout):
    """Predict a pointwise excess free energy per particle.

    The input at every grid point contains all locally normalized component
    densities followed by normalized temperature,

    ``(rho_1 / mean_density, ..., rho_M / mean_density, T / mean_T)``.

    The shared MLP returns one reduced scalar ``a_exc_lda``. The model
    multiplies it by total local density and the voxel volume, giving

    ``E_exc_lda = DeltaV * sum_g rho_total,g * a_exc_lda,g``.

    ``E_exc_lda`` follows the free-energy convention selected by the
    containing model.

    No neighboring density enters this module, so it is a strict local-density
    approximation. Supplying the complete local component vector still permits
    composition-dependent mixture free energies.

    Parameters
    ----------
    mean_density
        Positive scalar used only to scale the local density inputs.
    n_types
        Number of physical density components.
    hidden_sizes
        Width of each hidden layer. An empty sequence gives a linear mapping.
    zero_init
        If true, initialize the final layer to zero.
    """

    def __init__(
        self,
        mean_density: Union[float, torch.Tensor],
        n_types: int = 1,
        hidden_sizes: Sequence[int] = (32, 16),
        zero_init: bool = False,
    ) -> None:
        super().__init__()

        self.n_types = positive_integer(n_types, "n_types")
        self.n_state_features = self.n_types + 1
        density_scale = positive_scalar_tensor(mean_density, "mean_density")
        self.register_buffer("mean_density", density_scale)
        self.mlp = build_mlp(
            self.n_state_features,
            hidden_sizes,
            1,
            zero_init=zero_init,
        )

    def forward(self, local_state: torch.Tensor) -> torch.Tensor:
        """Return ``a_exc_lda`` with shape ``[..., n_grid, 1]``."""

        if local_state.shape[-1] != self.n_state_features:
            raise ValueError(
                "local_state must end with all normalized component "
                "densities and normalized temperature"
            )
        return self.mlp(local_state)

    def energy(
        self,
        context: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return the integrated pointwise LDA contribution."""

        rho = context["rho"]
        normalized_density = rho / self.mean_density.to(rho)
        temperature_feature = context["normalized_temperature"][
            ..., None, None
        ].expand(*rho.shape[:-1], 1)
        local_state = torch.cat(
            (normalized_density, temperature_feature),
            dim=-1,
        )
        scalar_per_particle = self(local_state)
        if scalar_per_particle.shape != (*rho.shape[:-1], 1):
            raise ValueError(
                "LDAReadout must return one scalar per grid point"
            )
        per_particle = scalar_per_particle.expand_as(rho)
        free_energy_density = torch.sum(rho * per_particle, dim=-1)
        return context["cell_volume"] * torch.sum(
            free_energy_density,
            dim=-1,
        )


class GGAReadout(EnergyReadout):
    """Predict a positive scalar density-gradient coefficient.

    The input contains local invariant environment features followed by
    normalized temperature. The shared MLP returns one coefficient per grid
    point,

    ``kappa = minimum_coefficient + softplus(raw_coefficient)``.

    The positive transformation makes the quadratic gradient contribution
    nonnegative for every density field. ``initial_coefficient`` initializes
    the last layer to a constant, so a new model starts from a controlled weak
    gradient penalty rather than ``softplus(0)``.

    Parameters
    ----------
    hidden_sizes
        Width of each hidden layer.
    n_features
        Input width, including temperature. If omitted, the first hidden layer
        is initialized lazily from the first input. An explicit width is needed
        when ``hidden_sizes`` is empty.
    minimum_coefficient
        Nonnegative lower bound on the returned coefficient.
    initial_coefficient
        Positive constant coefficient at initialization. It must be strictly
        larger than ``minimum_coefficient``.
    """

    requires_local_features = True
    def __init__(
        self,
        hidden_sizes: Sequence[int] = (16, 16),
        n_features: Optional[int] = None,
        n_types: int = 1,
        minimum_coefficient: float = 0.0,
        initial_coefficient: float = 0.01,
    ) -> None:
        super().__init__()

        self.n_types = positive_integer(n_types, "n_types")
        n_features = optional_positive_integer(n_features, "n_features")

        minimum_coefficient = nonnegative_scalar(
            minimum_coefficient,
            "minimum_coefficient",
        )
        initial_coefficient = positive_scalar(
            initial_coefficient,
            "initial_coefficient",
        )
        if initial_coefficient <= minimum_coefficient:
            raise ValueError(
                "initial_coefficient must be finite and exceed the minimum"
            )

        validated_hidden_sizes = validate_hidden_sizes(hidden_sizes)
        if n_features is None and not validated_hidden_sizes:
            raise ValueError(
                "n_features is required when hidden_sizes is empty"
            )

        self.mlp = build_mlp(n_features, validated_hidden_sizes, 1)
        final_layer = self.mlp[-1]
        nn.init.zeros_(final_layer.weight)
        softplus_value = initial_coefficient - minimum_coefficient
        inverse_softplus = softplus_value + math.log(
            -math.expm1(-softplus_value)
        )
        nn.init.constant_(final_layer.bias, inverse_softplus)
        self.minimum_coefficient = minimum_coefficient
        self.initial_coefficient = initial_coefficient

    def forward(self, local_features: torch.Tensor) -> torch.Tensor:
        """Return ``kappa`` with shape ``[..., n_grid, 1]``."""

        return self.minimum_coefficient + torch.nn.functional.softplus(
            self.mlp(local_features)
        )

    def energy(
        self,
        context: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return the positive periodic density-gradient contribution."""

        if "grid_size" not in context:
            raise KeyError("GGA evaluation requires data['grid_size']")
        rho = context["rho"]
        coefficient = self(context["local_features"])
        if coefficient.shape != (*rho.shape[:-1], 1):
            raise ValueError(
                "GGAReadout must return one coefficient per grid point"
            )
        free_energy_density = periodic_gradient_energy_density(
            rho=rho,
            coefficient=coefficient,
            grid_size=context["grid_size"],
            grid_spacing=context["grid_spacing"],
        )
        return context["cell_volume"] * torch.sum(
            free_energy_density,
            dim=-1,
        )


def periodic_gradient_energy_density(
    rho: torch.Tensor,
    coefficient: torch.Tensor,
    grid_size: torch.Tensor,
    grid_spacing: torch.Tensor,
) -> torch.Tensor:
    """Return the nonnegative periodic bond-gradient energy density.

    For each grid point ``g`` and positive Cartesian bond ``(g, g+e_d)``, the
    discretization is

    ``0.5 * kappa_gd * sum_i ((rho[g+e_d, i] - rho[g, i]) / h_d)**2``,

    where ``kappa_gd`` is the arithmetic mean of the positive coefficients at
    the two bond endpoints. Each periodic bond is counted exactly once. The
    returned shape is ``[..., n_grid]`` and excludes the voxel volume.
    """

    if rho.ndim < 2:
        raise ValueError("rho must have shape [..., n_grid, n_types]")
    expected_coefficient_shape = (*rho.shape[:-1], 1)
    if coefficient.shape != expected_coefficient_shape:
        raise ValueError(
            "coefficient must have shape [..., n_grid, 1] matching rho"
        )

    size = torch.as_tensor(grid_size).detach()
    if size.ndim < 1 or size.shape[-1] != 3:
        raise ValueError("grid_size must end with three values")
    size = size.reshape(-1, 3)
    if not torch.all(size == size[0]).item():
        raise ValueError("all fields must use one common three-dimensional grid")
    size_values = size[0].tolist()
    if any(float(value) != int(value) or int(value) < 1 for value in size_values):
        raise ValueError("grid_size must contain three positive integers")
    nx, ny, nz = (int(value) for value in size_values)
    if nx * ny * nz != rho.shape[-2]:
        raise ValueError("grid_size is inconsistent with rho")

    spacing = torch.as_tensor(
        grid_spacing,
        device=rho.device,
        dtype=rho.dtype,
    ).reshape(-1)
    if spacing.numel() != 3 or torch.any(spacing <= 0.0).item():
        raise ValueError("grid_spacing must contain three positive values")

    leading_shape = rho.shape[:-2]
    n_types = rho.shape[-1]
    rho_grid = rho.reshape(*leading_shape, nx, ny, nz, n_types)
    coefficient_grid = coefficient.reshape(*leading_shape, nx, ny, nz, 1)
    energy_density = torch.zeros_like(rho_grid[..., 0])

    # The last four axes are x, y, z, and component. Forward periodic bonds
    # penalize the Nyquist mode that a centered finite difference would miss.
    for direction, axis in enumerate((-4, -3, -2)):
        next_rho = torch.roll(rho_grid, shifts=-1, dims=axis)
        next_coefficient = torch.roll(
            coefficient_grid,
            shifts=-1,
            dims=axis,
        )
        bond_coefficient = 0.5 * (
            coefficient_grid + next_coefficient
        )
        density_difference = (
            next_rho - rho_grid
        ) / spacing[direction]
        energy_density = energy_density + 0.5 * bond_coefficient[..., 0] * (
            density_difference.square().sum(dim=-1)
        )

    return energy_density.reshape(*leading_shape, rho.shape[-2])
