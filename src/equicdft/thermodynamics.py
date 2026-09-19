"""Ideal-gas thermodynamics and Euler-Lagrange relations on density grids.

Every formula that turns a density field into a dimensionless free energy or
chemical potential is defined once here. The model, the data pipeline, the
equilibrium solver, and the regression harness all call these functions, so
the discrete conventions (density floors, vacuum handling, quadrature) cannot
drift between them.

Conventions
-----------
``rho`` has shape ``[..., n_grid, n_types]`` and ``thermal_wavelength`` has
shape ``[n_types]`` (or a broadcast-compatible leading shape). ``beta`` is a
scalar or has the leading field shape ``[...]``. All returned quantities are
dimensionless: free energies carry the factor ``beta`` and chemical
potentials are ``beta*mu``.
"""

from typing import Optional, Union

import torch

__all__ = [
    "density_weighted_integral",
    "log_dimensionless_density",
    "ideal_free_energy",
    "local_chemical_potential",
    "equilibrium_log_density",
    "beta_external_potential",
]


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


def log_dimensionless_density(
    rho: torch.Tensor,
    thermal_wavelength: torch.Tensor,
) -> torch.Tensor:
    r"""Return ``log(rho * Lambda**3)`` with density floored at ``tiny``."""

    tiny = torch.finfo(rho.dtype).tiny
    return torch.log(
        torch.clamp(rho, min=tiny)
        * thermal_wavelength[..., None, :] ** 3
    )


def ideal_free_energy(
    rho: torch.Tensor,
    thermal_wavelength: torch.Tensor,
    voxel_volume: Union[float, torch.Tensor],
) -> torch.Tensor:
    r"""Return discrete dimensionless ideal-gas free energy per field.

    Vacuum grid points contribute exactly zero, matching the ``rho log rho``
    limit rather than the floored logarithm.
    """

    per_particle = torch.where(
        rho > 0.0,
        log_dimensionless_density(rho, thermal_wavelength) - 1.0,
        torch.zeros_like(rho),
    )
    return density_weighted_integral(rho, per_particle, voxel_volume)


def _field_scalar(
    value: Union[float, torch.Tensor],
    rho: torch.Tensor,
    name: str,
) -> torch.Tensor:
    """Return ``value`` broadcastable over the grid and type axes of rho."""

    tensor = torch.as_tensor(value, dtype=rho.dtype, device=rho.device)
    if tensor.ndim == 0:
        return tensor
    if tensor.shape != rho.shape[:-2]:
        raise ValueError(
            "{} must be scalar or match rho's leading field shape".format(name)
        )
    return tensor[..., None, None]


def _resolved_mask(
    rho: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Return the Boolean set of grid points that carry a chemical potential."""

    if mask is None:
        return rho > 0.0
    mask = torch.as_tensor(mask, dtype=torch.bool, device=rho.device)
    try:
        return mask.expand_as(rho)
    except RuntimeError:
        raise ValueError("mask must be broadcastable to rho") from None


def local_chemical_potential(
    rho: torch.Tensor,
    V_ext: torch.Tensor,
    beta: Union[float, torch.Tensor],
    c1: Optional[torch.Tensor],
    thermal_wavelength: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Return the dimensionless local chemical potential ``beta*mu_loc``.

    The Euler-Lagrange relation of classical DFT reads

    ``beta*mu_loc(g) = log(rho_g * Lambda**3) + beta*V_ext(g) - c1(g)``.

    At equilibrium this field is spatially constant and equal to ``beta*mu``.
    With ``c1=None`` the function returns ``c1 + beta*mu``, the training
    target that requires no chemical-potential label. Entries outside
    ``mask`` (by default the points with ``rho > 0``) are exactly zero.
    """

    if rho.ndim < 2:
        raise ValueError("rho must have shape [..., n_grid, n_types]")
    if V_ext.shape != rho.shape:
        raise ValueError("V_ext must have the same shape as rho")
    if c1 is not None and c1.shape != rho.shape:
        raise ValueError("c1 must have the same shape as rho")
    value = (
        log_dimensionless_density(rho, thermal_wavelength)
        + _field_scalar(beta, rho, "beta") * V_ext
    )
    if c1 is not None:
        value = value - c1
    return torch.where(
        _resolved_mask(rho, mask),
        value,
        torch.zeros_like(value),
    )


def equilibrium_log_density(
    V_ext: torch.Tensor,
    beta: Union[float, torch.Tensor],
    mu: torch.Tensor,
    c1: torch.Tensor,
    thermal_wavelength: torch.Tensor,
) -> torch.Tensor:
    r"""Return ``log rho`` that satisfies the Euler-Lagrange relation.

    Solving :func:`local_chemical_potential` for the density at fixed
    ``c1`` gives the grand-canonical fixed-point map

    ``log rho(g) = beta*(mu - V_ext(g)) + c1(g) - 3 log Lambda``.

    ``mu`` has shape ``[n_types]`` (energy units, not ``beta*mu``). The
    returned logarithm is unfloored: the caller decides how to clamp it.
    """

    if V_ext.ndim < 2:
        raise ValueError("V_ext must have shape [..., n_grid, n_types]")
    if c1.shape != V_ext.shape:
        raise ValueError("c1 must have the same shape as V_ext")
    return (
        _field_scalar(beta, V_ext, "beta") * (mu[None, :] - V_ext)
        + c1
        - 3.0 * torch.log(thermal_wavelength)[None, :]
    )


def beta_external_potential(
    rho: torch.Tensor,
    c1: torch.Tensor,
    thermal_wavelength: torch.Tensor,
    beta_mu: Union[float, torch.Tensor] = 0.0,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Return ``beta*V_ext`` that makes ``rho`` an equilibrium density.

    This inverts :func:`local_chemical_potential`,

    ``beta*V_ext(g) = c1(g) + beta*mu - log(rho_g * Lambda**3)``.

    ``beta_mu`` is a scalar, a per-type vector of shape ``[n_types]``, or a
    per-field-and-type tensor of shape ``[..., n_types]``; it fixes the gauge
    of the potential. Entries outside ``mask`` (by default the points with
    ``rho > 0``) are exactly zero.
    """

    if rho.ndim < 2:
        raise ValueError("rho must have shape [..., n_grid, n_types]")
    if c1.shape != rho.shape:
        raise ValueError("c1 must have the same shape as rho")
    offset = torch.as_tensor(beta_mu, dtype=rho.dtype, device=rho.device)
    if offset.ndim > 0:
        if offset.shape not in (rho.shape[-1:], rho.shape[:-2] + rho.shape[-1:]):
            raise ValueError(
                "beta_mu must be scalar, [n_types], or [..., n_types]"
            )
        offset = offset[..., None, :]
    value = c1 + offset - log_dimensionless_density(rho, thermal_wavelength)
    return torch.where(
        _resolved_mask(rho, mask),
        value,
        torch.zeros_like(value),
    )
