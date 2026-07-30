"""Local excess-free-energy readout and functional derivatives."""

from numbers import Integral
from typing import Dict, Mapping, Optional, Sequence

import torch
from torch import nn


def _positive_integer(name: str, value: int) -> int:
    """Validate a positive integer constructor argument."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("{} must be a positive integer".format(name))
    value = int(value)
    if value < 1:
        raise ValueError("{} must be a positive integer".format(name))
    return value


def compute_rms_feature_scale(
    B: torch.Tensor,
    minimum_scale: float = 1.0e-12,
) -> torch.Tensor:
    """Return one fixed RMS scale for each flattened ``B`` feature.

    ``B`` may contain one configuration or a stacked training split. All axes
    other than the three trailing representation axes are treated as samples.
    The returned tensor is detached because these are preprocessing statistics,
    not trainable parameters.
    """

    if B.ndim < 4:
        raise ValueError(
            "B must have shape "
            "[..., n_grid, n_alphas, n_B, n_channels]"
        )
    if isinstance(minimum_scale, bool):
        raise TypeError("minimum_scale must be a positive number")
    minimum_scale = float(minimum_scale)
    if minimum_scale <= 0.0:
        raise ValueError("minimum_scale must be positive")

    B_flat = B.detach().flatten(start_dim=-3)
    sample_dimensions = tuple(range(B_flat.ndim - 1))
    scale = torch.sqrt(torch.mean(B_flat.square(), dim=sample_dimensions))
    return torch.clamp(scale, min=minimum_scale)


class LocalFreeEnergyReadout(nn.Module):
    """Predict anchored local excess free energies per particle.

    At each grid point the flattened invariant ``B`` features are concatenated
    with the physical center densities and, optionally, inverse temperature.
    A shared pointwise network predicts one dimensionless excess free energy
    per particle and physical component.

    The raw prediction is evaluated at both the physical input and a vacuum
    input with ``B = 0`` and ``rho = 0`` at the same temperature. Their
    difference defines ``beta_a_exc``. The module then constructs

    ``beta_f_exc[g] = sum_i rho[g, i] * beta_a_exc[g, i]``

    and ``beta_F_exc = cell_volume * sum_g beta_f_exc[g]``.

    This enforces ``F_exc[0] = 0`` and a vanishing linear contribution at zero
    density. One per-particle output per physical component extends the same
    construction to mixtures.

    Parameters
    ----------
    n_features
        Flattened size of the radial, invariant, and channel axes of ``B``.
    n_types
        Number of physical density components and per-particle outputs.
    hidden_sizes
        Width of each hidden layer in the nonlinear branch.
    include_temperature
        If ``True``, append ``1 / temperature`` to every local input.
    add_linear
        If ``True``, add a bias-free linear branch to the nonlinear network.
    feature_scale
        Optional positive tensor with shape ``[n_features]``. ``B`` is divided
        by this fixed scale before entering the readout. Use
        :func:`compute_rms_feature_scale` on the training split to construct it.

    Notes
    -----
    Smooth SiLU activations keep higher functional derivatives well defined.
    Batch normalization and dropout are deliberately absent because they
    would make the functional depend on batch context or stochastic state.
    """

    def __init__(
        self,
        n_features: int,
        n_types: int = 1,
        hidden_sizes: Sequence[int] = (32, 16),
        include_temperature: bool = False,
        add_linear: bool = True,
        feature_scale: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.n_features = _positive_integer("n_features", n_features)
        self.n_types = _positive_integer("n_types", n_types)
        if not isinstance(include_temperature, bool):
            raise TypeError("include_temperature must be a boolean")
        if not isinstance(add_linear, bool):
            raise TypeError("add_linear must be a boolean")

        hidden_sizes = tuple(
            _positive_integer("hidden_sizes entry", size)
            for size in hidden_sizes
        )
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer")

        self.hidden_sizes = hidden_sizes
        self.include_temperature = include_temperature
        self.add_linear = add_linear
        self.n_input = (
            self.n_features
            + self.n_types
            + int(self.include_temperature)
        )

        if feature_scale is None:
            scale = torch.ones(
                self.n_features,
                dtype=torch.get_default_dtype(),
            )
            self.normalize_features = False
        else:
            scale = torch.as_tensor(
                feature_scale,
                dtype=torch.get_default_dtype(),
            ).detach().clone()
            if scale.shape != (self.n_features,):
                raise ValueError(
                    "feature_scale must have shape ({},)".format(
                        self.n_features
                    )
                )
            if not torch.all(torch.isfinite(scale)) or torch.any(scale <= 0.0):
                raise ValueError("feature_scale must contain finite positive values")
            self.normalize_features = True
        self.register_buffer("feature_scale", scale)

        layers = []
        n_in = self.n_input
        for n_hidden in hidden_sizes:
            layers.extend((nn.Linear(n_in, n_hidden), nn.SiLU()))
            n_in = n_hidden
        # The final additive bias would cancel exactly between the physical
        # and vacuum evaluations, so omitting it avoids a redundant parameter.
        layers.append(nn.Linear(n_in, self.n_types, bias=False))
        self.nonlinear = nn.Sequential(*layers)
        self.linear = (
            nn.Linear(self.n_input, self.n_types, bias=False)
            if self.add_linear
            else None
        )

    def _unanchored(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the raw linear-plus-nonlinear per-particle prediction."""

        output = self.nonlinear(inputs)
        if self.linear is not None:
            output = output + self.linear(inputs)
        return output

    def _inverse_temperature(
        self,
        temperature: torch.Tensor,
        rho: torch.Tensor,
    ) -> torch.Tensor:
        """Broadcast inverse temperature over the grid axis."""

        configuration_shape = rho.shape[:-2]
        if tuple(temperature.shape) != tuple(configuration_shape):
            raise ValueError(
                "temperature must match the configuration dimensions of rho"
            )
        if torch.any(temperature <= 0.0):
            raise ValueError("temperature must be positive")

        inverse_temperature = torch.reciprocal(
            temperature.to(dtype=rho.dtype, device=rho.device)
        )
        inverse_temperature = inverse_temperature.reshape(
            *configuration_shape,
            1,
            1,
        )
        return inverse_temperature.expand(*rho.shape[:-1], 1)

    def forward(
        self,
        B: torch.Tensor,
        data: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Return local per-particle, density, and integrated free energies."""

        if B.ndim < 4:
            raise ValueError(
                "B must have shape "
                "[..., n_grid, n_alphas, n_B, n_channels]"
            )
        B_flat = B.flatten(start_dim=-3)
        if B_flat.shape[-1] != self.n_features:
            raise ValueError(
                "flattened B has {} features but this readout expects {}".format(
                    B_flat.shape[-1],
                    self.n_features,
                )
            )

        rho = data["rho"]
        if rho.ndim < 2 or rho.shape[-1] != self.n_types:
            raise ValueError(
                "rho must have shape [..., n_grid, {}]".format(self.n_types)
            )
        if B_flat.shape[:-1] != rho.shape[:-1]:
            raise ValueError("B and rho grid/configuration shapes must match")

        scaled_B = B_flat / self.feature_scale.to(dtype=B_flat.dtype)
        physical_parts = [scaled_B, rho]
        vacuum_parts = [torch.zeros_like(B_flat), torch.zeros_like(rho)]
        if self.include_temperature:
            inverse_temperature = self._inverse_temperature(
                data["temperature"],
                rho,
            )
            physical_parts.append(inverse_temperature)
            vacuum_parts.append(inverse_temperature)

        physical_input = torch.cat(physical_parts, dim=-1)
        vacuum_input = torch.cat(vacuum_parts, dim=-1)
        beta_free_energy_per_particle = self._unanchored(
            physical_input
        ) - self._unanchored(vacuum_input)

        beta_free_energy_density = torch.sum(
            rho * beta_free_energy_per_particle,
            dim=-1,
        )

        grid_spacing = data["grid_spacing"]
        configuration_shape = rho.shape[:-2]
        if (
            grid_spacing.shape[-1] != 3
            or tuple(grid_spacing.shape[:-1]) != tuple(configuration_shape)
        ):
            raise ValueError(
                "grid_spacing must have shape [..., 3] matching rho"
            )
        cell_volume = torch.prod(grid_spacing, dim=-1)
        beta_F_exc = cell_volume * torch.sum(
            beta_free_energy_density,
            dim=-1,
        )

        return {
            "beta_free_energy_per_particle": beta_free_energy_per_particle,
            "beta_free_energy_density": beta_free_energy_density,
            "beta_F_exc": beta_F_exc,
        }


def compute_c1(
    beta_F_exc: torch.Tensor,
    rho: torch.Tensor,
    grid_spacing: torch.Tensor,
    create_graph: bool = False,
) -> torch.Tensor:
    """Return ``c1 = -(1 / cell_volume) * d(beta_F_exc) / d(rho)``.

    Set ``create_graph=True`` while training from ``c1`` labels so gradients
    of the derivative loss propagate into all model parameters.
    """

    if not isinstance(create_graph, bool):
        raise TypeError("create_graph must be a boolean")
    if not rho.requires_grad:
        raise ValueError("rho must require gradients before the model forward")
    if tuple(beta_F_exc.shape) != tuple(rho.shape[:-2]):
        raise ValueError(
            "beta_F_exc must have one scalar per rho configuration"
        )
    if (
        grid_spacing.shape[-1] != 3
        or tuple(grid_spacing.shape[:-1]) != tuple(rho.shape[:-2])
    ):
        raise ValueError("grid_spacing must have shape [..., 3] matching rho")

    derivative = torch.autograd.grad(
        beta_F_exc.sum(),
        rho,
        create_graph=create_graph,
    )[0]
    cell_volume = torch.prod(grid_spacing, dim=-1)
    while cell_volume.ndim < derivative.ndim:
        cell_volume = cell_volume.unsqueeze(-1)
    return -derivative / cell_volume
