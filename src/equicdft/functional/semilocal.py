"""Pointwise and gradient-expanded free-energy building blocks."""

import math
from typing import Any, Dict, Optional, Sequence

import torch
from torch import nn

from .._validation import (
    boolean,
    nonnegative_scalar,
    optional_positive_integer,
    positive_integer,
    positive_scalar,
)
from .._config import (
    Configurable,
    make_config,
    register,
)
from .._polarization import polarization_input_name
from ..geometry.grid import grid_spacing_tensor
from .._nn import (
    build_mlp,
    validate_hidden_sizes,
)
from .base import EnergyReadout, ReadoutContext

__all__ = [
    "LDAReadout",
    "GGAReadout",
    "periodic_gradient_energy_density",
]


@register
class LDAReadout(EnergyReadout, Configurable):
    """Predict a pointwise excess free energy per particle.

    The input at every grid point contains all locally normalized component
    densities followed by normalized temperature,

    ``(rho_1 / mean_density, ..., rho_M / mean_density, T / mean_T)``,

    where ``mean_density`` is the density scale owned by the containing
    :class:`equicdft.model.GridCACEModel`.

    The shared MLP returns one reduced scalar ``a_exc_lda``. The model
    multiplies it by total local density and the voxel volume, giving

    ``E_exc_lda = DeltaV * sum_g rho_total,g * a_exc_lda,g``.

    ``E_exc_lda`` follows the free-energy convention selected by the
    containing model.

    When ``include_polarization=True``, append one local squared vector
    magnitude per component before temperature. The default raw input is
    ``(rho_a/mean_density, |P_a|^2/polarization_scale^2, T/mean_T)``.
    ``polarization_input='alignment'`` replaces P by the smooth reduced field
    ``rho P / (m (rho^2 + density_floor^2))`` and uses its separate scale.
    This is an excess contribution only: translational and orientational
    ideal entropy must still be supplied separately. Squared magnitudes keep
    derivatives smooth at P=0. For mixtures this magnitude-only branch does
    not resolve relative directions between different species' dipoles.

    No neighboring density enters this module, so it is a strict local-density
    approximation. Supplying the complete local component vector still permits
    composition-dependent mixture free energies.

    Parameters
    ----------
    n_types
        Number of physical density components.
    hidden_sizes
        Width of each hidden layer. An empty sequence gives a linear mapping.
    zero_init
        If true, initialize the final layer to zero.
    include_polarization
        Include squared polarization magnitudes using the model-owned scale.
        False preserves the density-only architecture.
    polarization_input
        ``"raw"`` uses ``|P / polarization_scale|^2``. ``"alignment"``
        uses the squared smooth molecular alignment divided by its separate
        model-owned scale.
    """

    reads_smoothed_fields = True

    def __init__(
        self,
        n_types: int = 1,
        *,
        hidden_sizes: Sequence[int] = (32, 16),
        zero_init: bool = False,
        include_polarization: bool = False,
        polarization_input: str = "raw",
    ) -> None:
        super().__init__()

        self.n_types = positive_integer(n_types, "n_types")
        self.include_polarization = boolean(include_polarization, "include_polarization")
        self.polarization_input = polarization_input_name(
            polarization_input,
            include_polarization=self.include_polarization,
        )
        self.n_state_features = self.n_types + 1
        if self.include_polarization:
            self.n_state_features += self.n_types
        self.hidden_sizes = validate_hidden_sizes(hidden_sizes)
        self.zero_init = boolean(zero_init, "zero_init")
        self.mlp = build_mlp(
            self.n_state_features,
            self.hidden_sizes,
            1,
            zero_init=self.zero_init,
        )

    def to_config(self) -> Dict[str, Any]:
        """Return the constructor arguments describing this readout."""

        return make_config(
            self,
            n_types=self.n_types,
            hidden_sizes=self.hidden_sizes,
            zero_init=self.zero_init,
            include_polarization=self.include_polarization,
            polarization_input=self.polarization_input,
        )

    @property
    def requires_dipole_density(self) -> bool:
        """Whether the LDA includes squared polarization magnitudes."""

        return self.include_polarization

    @property
    def requires_polarization_scale(self) -> bool:
        return self.include_polarization and self.polarization_input == "raw"

    @property
    def requires_alignment_scale(self) -> bool:
        return self.include_polarization and self.polarization_input == "alignment"

    def forward(self, local_state: torch.Tensor) -> torch.Tensor:
        """Return ``a_exc_lda`` with shape ``[..., n_grid, 1]``."""

        if local_state.shape[-1] != self.n_state_features:
            raise ValueError(
                "local_state must end with the configured normalized densities, "
                "optional squared dipole magnitudes, and temperature"
            )
        return self.mlp(local_state)

    def energy(self, context: ReadoutContext) -> torch.Tensor:
        """Return the integrated pointwise LDA contribution."""

        rho = context.rho
        if rho.shape[-1] != self.n_types:
            raise ValueError("rho must have shape [..., n_grid, n_types]")
        inputs = [context.normalized_rho()]
        if self.requires_dipole_density:
            if context.dipole_density is None:
                raise ValueError("dipole_density is required by this LDAReadout")
            dipole = context.dipole_density
            if dipole.shape != (*rho.shape, 3):
                raise ValueError("dipole_density must have shape [..., n_grid, n_types, 3]")
            if dipole.dtype != rho.dtype or dipole.device != rho.device:
                raise ValueError("rho and dipole_density must share dtype and device")
            if not torch.isfinite(dipole).all():
                raise ValueError("dipole_density must be finite")
            normalized_dipole = (
                context.normalized_polarization()
                if self.polarization_input == "raw"
                else context.normalized_alignment()
            )
            inputs.append(normalized_dipole.square().sum(dim=-1))
        local_state = (
            context.pointwise_state()
            if len(inputs) == 1
            else torch.cat((*inputs, context.temperature_feature()), dim=-1)
        )
        scalar_per_particle = self(local_state)
        if scalar_per_particle.shape != (*rho.shape[:-1], 1):
            raise ValueError(
                "LDAReadout must return one scalar per grid point"
            )
        per_particle = scalar_per_particle.expand_as(rho)
        return context.integrate_local(per_particle)


@register
class GGAReadout(EnergyReadout, Configurable):
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
        Input width, including temperature. When omitted, the readout stays
        unbound until :meth:`bind_input_size` is called;
        :class:`equicdft.model.GridCACEModel` does this at construction.
    minimum_coefficient
        Nonnegative lower bound on the returned coefficient.
    initial_coefficient
        Positive constant coefficient at initialization. It must be strictly
        larger than ``minimum_coefficient``.
    """

    requires_local_features = True
    reads_smoothed_fields = True

    def __init__(
        self,
        n_types: int = 1,
        *,
        hidden_sizes: Sequence[int] = (32, 16),
        n_features: Optional[int] = None,
        minimum_coefficient: float = 0.0,
        initial_coefficient: float = 0.01,
    ) -> None:
        super().__init__()

        self.n_types = positive_integer(n_types, "n_types")
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
        self.minimum_coefficient = minimum_coefficient
        self.initial_coefficient = initial_coefficient
        self.hidden_sizes = validate_hidden_sizes(hidden_sizes)
        self.n_features = None
        self.mlp = None
        n_features = optional_positive_integer(n_features, "n_features")
        if n_features is not None:
            self.bind_input_size(n_features)

    @property
    def is_bound(self) -> bool:
        return self.mlp is not None

    def bind_input_size(self, n_features: int) -> None:
        """Build the coefficient network for ``n_features`` inputs, once.

        The final layer starts at the constant ``initial_coefficient``.
        Rebinding to the same width is a no-op; a different width is an error.
        """

        n_features = positive_integer(n_features, "n_features")
        if self.is_bound:
            if n_features != self.n_features:
                raise ValueError(
                    "GGAReadout is bound to {} input features, not {}".format(
                        self.n_features, n_features,
                    )
                )
            return
        mlp = build_mlp(n_features, self.hidden_sizes, 1)
        final_layer = mlp[-1]
        nn.init.zeros_(final_layer.weight)
        softplus_value = self.initial_coefficient - self.minimum_coefficient
        inverse_softplus = softplus_value + math.log(
            -math.expm1(-softplus_value)
        )
        nn.init.constant_(final_layer.bias, inverse_softplus)
        self.n_features = n_features
        self.mlp = mlp

    def to_config(self) -> Dict[str, Any]:
        """Return the constructor arguments describing this readout.

        ``n_features`` is ``None`` while the readout is unbound.
        """

        return make_config(
            self,
            hidden_sizes=self.hidden_sizes,
            n_features=self.n_features,
            n_types=self.n_types,
            minimum_coefficient=self.minimum_coefficient,
            initial_coefficient=self.initial_coefficient,
        )

    def forward(self, local_features: torch.Tensor) -> torch.Tensor:
        """Return ``kappa`` with shape ``[..., n_grid, 1]``."""

        if not self.is_bound:
            raise RuntimeError(
                "GGAReadout has no input width yet; pass n_features or let "
                "GridCACEModel bind it"
            )
        return self.minimum_coefficient + torch.nn.functional.softplus(
            self.mlp(local_features)
        )

    def energy(self, context: ReadoutContext) -> torch.Tensor:
        """Return the positive periodic density-gradient contribution."""

        grid_size = context.require("grid_size")
        rho = context.rho
        coefficient = self(context.local_features())
        if coefficient.shape != (*rho.shape[:-1], 1):
            raise ValueError(
                "GGAReadout must return one coefficient per grid point"
            )
        free_energy_density = periodic_gradient_energy_density(
            rho=rho,
            coefficient=coefficient,
            grid_size=grid_size,
            grid_spacing=context.require("grid_spacing"),
        )
        return context.voxel_volume * torch.sum(
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

    spacing = grid_spacing_tensor(
        grid_spacing,
        device=rho.device,
        dtype=rho.dtype,
    )

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
