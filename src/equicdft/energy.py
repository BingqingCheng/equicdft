"""Common interface for additive free-energy readouts."""

from typing import Dict, Union

import torch
from torch import nn


def density_weighted_integral(
    rho: torch.Tensor,
    per_particle: torch.Tensor,
    voxel_volume: Union[float, torch.Tensor],
) -> torch.Tensor:
    r"""Return ``Delta V sum_(g,a) rho[g,a] f[g,a]`` per field.

    ``rho`` has shape ``[..., n_grid, n_types]``. ``per_particle`` may
    contain one value per grid point with the same shape, or a spatially
    constant value with shape ``[..., 1, n_types]``. ``voxel_volume`` is
    either scalar or has the leading field shape ``[...]``.
    """

    if rho.ndim < 2:
        raise ValueError("rho must have shape [..., n_grid, n_types]")
    if per_particle.ndim != rho.ndim:
        raise ValueError("per_particle must have the same rank as rho")
    if (
        per_particle.shape[:-2] != rho.shape[:-2]
        or per_particle.shape[-1] != rho.shape[-1]
        or per_particle.shape[-2] not in (1, rho.shape[-2])
    ):
        raise ValueError(
            "per_particle must match rho, with one or n_grid spatial values"
        )

    volume = torch.as_tensor(
        voxel_volume,
        dtype=rho.dtype,
        device=rho.device,
    )
    if volume.ndim != 0 and volume.shape != rho.shape[:-2]:
        raise ValueError(
            "voxel_volume must be scalar or match rho's leading field shape"
        )
    if (
        not torch.all(torch.isfinite(volume)).item()
        or torch.any(volume <= 0.0).item()
    ):
        raise ValueError("voxel_volume must be finite and positive")

    return volume * torch.sum(
        rho * per_particle,
        dim=(-2, -1),
    )


class EnergyReadout(nn.Module):
    """Neural readout that supplies one scalar functional contribution."""

    requires_local_features = False
    requires_state_features = False

    def energy(
        self,
        context: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return one scalar energy per complete density field."""

        raise NotImplementedError
