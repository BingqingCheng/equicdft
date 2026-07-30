"""Local excess-free-energy readout."""

from typing import Dict, Mapping, Optional, Sequence

import torch
from torch import nn


class LocalFreeEnergyReadout(nn.Module):
    """Predict local excess free energies and integrate them over the grid.

    A shared MLP maps the normalized invariant ``B`` features at grid point
    ``g`` to one dimensionless excess free energy per particle and component,

    ``beta_a_exc[g, i] = MLP(B[g])[i]``.

    The density supplies the grid-point weight,

    ``beta_f_exc[g] = sum_i rho[g, i] * beta_a_exc[g, i]``,

    and the grid quadrature gives

    ``beta_F_exc = cell_volume * sum_g beta_f_exc[g]``.

    Thus an empty grid has zero excess free energy without imposing a separate
    constraint on the MLP output at ``B = 0``.

    Parameters
    ----------
    n_features
        Flattened size of the radial, invariant, and channel axes of ``B``.
    n_types
        Number of physical density components and per-particle outputs.
    hidden_sizes
        Width of each hidden layer. An empty sequence gives a linear readout.
    feature_scale
        Optional positive tensor with shape ``[n_features]``. ``B`` is divided
        by this fixed scale before entering the readout. Compute it once from
        the training split with ``compute_rms_feature_scale``.

    Notes
    -----
    Smooth SiLU activations keep higher functional derivatives well defined.
    """

    def __init__(
        self,
        n_features: int,
        n_types: int = 1,
        hidden_sizes: Sequence[int] = (32, 16),
        feature_scale: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.n_features = n_features
        self.n_types = n_types

        if feature_scale is None:
            feature_scale = torch.ones(n_features)
        scale = torch.as_tensor(
            feature_scale,
            dtype=torch.get_default_dtype(),
        ).detach().clone()
        self.register_buffer("feature_scale", scale)

        layers = []
        n_in = n_features
        for n_hidden in hidden_sizes:
            layers.extend((nn.Linear(n_in, n_hidden), nn.SiLU()))
            n_in = n_hidden
        layers.append(nn.Linear(n_in, n_types))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        B: torch.Tensor,
        data: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Return local per-particle, density, and integrated free energies."""

        B_flat = B.flatten(start_dim=-3)
        rho = data["rho"]
        beta_free_energy_per_particle = self.mlp(
            B_flat / self.feature_scale
        )

        beta_free_energy_density = torch.sum(
            rho * beta_free_energy_per_particle,
            dim=-1,
        )

        cell_volume = torch.prod(
            data["grid_spacing"],
            dim=-1,
        )
        beta_F_exc = cell_volume * torch.sum(
            beta_free_energy_density,
            dim=-1,
        )

        return {
            "beta_free_energy_per_particle": beta_free_energy_per_particle,
            "beta_free_energy_density": beta_free_energy_density,
            "beta_F_exc": beta_F_exc,
        }
