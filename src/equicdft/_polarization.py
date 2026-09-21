"""Shared learned-input transformations for polarization density."""

import torch

from ._species import species_vector

__all__ = [
    "polarization_input_name",
    "regularized_reduced_polarization",
]


def polarization_input_name(value, *, include_polarization):
    """Validate the vector field used by one learned polarization input."""

    if not isinstance(value, str):
        raise TypeError("polarization_input must be 'raw' or 'alignment'")
    value = value.lower()
    if value not in ("raw", "alignment"):
        raise ValueError("polarization_input must be 'raw' or 'alignment'")
    if not include_polarization and value != "raw":
        raise ValueError(
            "polarization_input='alignment' requires include_polarization=True"
        )
    return value


def regularized_reduced_polarization(
    rho,
    dipole_density,
    dipole_magnitude,
    density_floor,
):
    r"""Return a smooth extension of ``P / (m rho)`` through vacuum.

    The species-wise vector

    ``u_eps = rho * P / (m * (rho**2 + density_floor**2))``

    approaches the molecular alignment ``P/(m rho)`` away from vacuum and
    vanishes smoothly with the physical fields at vacuum.  No input is
    detached, so differentiation with respect to both rho and P includes the
    complete chain rule.
    """

    if not torch.is_tensor(rho) or rho.ndim < 2 or not rho.is_floating_point():
        raise ValueError("rho must be floating [..., grid, species]")
    if not torch.isfinite(rho).all():
        raise ValueError("rho must be finite")
    if (
        not torch.is_tensor(dipole_density)
        or dipole_density.shape != (*rho.shape, 3)
    ):
        raise ValueError(
            "dipole_density must have shape [..., grid, species, 3]"
        )
    if dipole_density.dtype != rho.dtype or dipole_density.device != rho.device:
        raise ValueError("rho and dipole_density must share dtype and device")
    if not torch.isfinite(dipole_density).all():
        raise ValueError("dipole_density must be finite")
    floor = torch.as_tensor(
        density_floor,
        dtype=rho.dtype,
        device=rho.device,
    ).reshape(-1)
    if (
        floor.numel() != 1
        or not torch.isfinite(floor).all()
        or torch.any(floor <= 0)
    ):
        raise ValueError("alignment_density_floor must be a positive scalar")
    moment = species_vector(
        dipole_magnitude,
        rho.shape[-1],
        rho,
        "dipole_magnitude",
        positive=True,
    )
    denominator = moment * (rho.square() + floor.reshape(()).square())
    return rho[..., None] * dipole_density / denominator[..., None]
