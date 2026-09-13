"""Ideal entropy of freely rotating, fixed-magnitude electric point dipoles.

Angular measure is dOmega/(4*pi). rho is number/volume and P is dipole/volume.
The finite differentiable domain is rho > 0 and |P| < m*rho. Vacuum and perfect
alignment are boundary limits, not finite states accepted by this interface.
"""

import torch
from torch import nn

from .energy import density_weighted_integral


def _langevin_ratio_series(square):
    """Small-field series for L(x)/x, expressed in x^2 for vector inputs."""
    return (
        1 / 3 - square / 45 + 2 * square**2 / 945
        - square**3 / 4725 + 2 * square**4 / 93555
    )


def langevin(x):
    """L(x) = coth(x) - 1/x, including a differentiable zero limit."""
    square = x.square()
    series = x * _langevin_ratio_series(square)
    safe = x.abs().clamp_min(0.1)
    regular = x.sign() * (1 / torch.tanh(safe) - 1 / safe)
    return torch.where(x.abs() < 0.1, series, regular)


def _langevin_prime(x):
    square = x.square()
    series = (
        1 / 3 - square / 15 + 2 * square**2 / 189
        - square**3 / 675 + 2 * square**4 / 10395
    )
    safe = x.abs().clamp_min(0.1)
    exponential = torch.exp(-2 * safe)
    regular = 1 / safe.square() - 4 * exponential / (-torch.expm1(-2 * safe)).square()
    return torch.where(x.abs() < 0.1, series, regular)


def log_sinhc(x):
    """log(sinh(x)/x), finite at zero and without large-field overflow."""
    square = x.square()
    series = (
        square / 6 - square**2 / 180 + square**3 / 2835
        - square**4 / 37800 + square**5 / 467775
    )
    safe = x.abs().clamp_min(0.1)
    regular = safe + torch.log(-torch.expm1(-2 * safe)) - torch.log(2 * safe)
    return torch.where(x.abs() < 0.1, series, regular)


def inverse_langevin(p):
    """Differentiable inverse on -1 < p < 1; Newton solves the actual L."""
    if not torch.isfinite(p).all() or torch.any(p.abs() >= 1):
        raise ValueError("inverse Langevin requires finite |p| < 1")
    # A rational initial estimate, refined rather than used as an approximation.
    x = p * (3 - p.square()) / (1 - p.square())
    for _ in range(12):
        x = x - (langevin(x) - p) / _langevin_prime(x)
    return x


def _positive_components(value, reference, name):
    result = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if result.ndim > 1 or result.numel() not in (1, reference.shape[-1]):
        raise ValueError(name + " must be scalar or one value per species")
    if not torch.isfinite(result).all() or torch.any(result <= 0):
        raise ValueError(name + " must be finite and positive")
    return result.reshape(-1)


def dipole_alignment(vector):
    """Return L(|a|)*a/|a| with smooth first and second derivatives at a=0."""
    square = vector.square().sum(dim=-1, keepdim=True)
    safe = square.clamp_min(0.01).sqrt()
    coefficient = torch.where(
        square < 0.01, _langevin_ratio_series(square), langevin(safe) / safe
    )
    return coefficient * vector


class FixedDipoleIdeal(nn.Module):
    """Ideal beta free energy and derivatives at fixed independent rho and P.

    ``dipole_magnitude`` and ``thermal_wavelength`` are positive scalars or one
    value per species, in the same dipole and length units as the fields.
    Input shapes are rho [..., grid, species], P [..., grid, species, 3].
    Density derivative has a positive sign (unlike excess c1). Polarization
    derivative is positive, with units inverse dipole; both divide out Delta V.
    """

    def __init__(self, dipole_magnitude, thermal_wavelength=1.0):
        super().__init__()
        for name, value in (
            ("dipole_magnitude", dipole_magnitude),
            ("thermal_wavelength", thermal_wavelength),
        ):
            # Preserve Python physical constants before casting to a field's
            # dtype. Tensor inputs retain their explicitly selected precision.
            if isinstance(value, torch.Tensor):
                tensor = value
            else:
                tensor = torch.as_tensor(value, dtype=torch.float64)
            if tensor.is_complex():
                raise ValueError(name + " must be real")
            if not tensor.is_floating_point():
                tensor = tensor.to(torch.get_default_dtype())
            if (tensor.ndim > 1 or not tensor.numel()
                    or not torch.isfinite(tensor).all() or torch.any(tensor <= 0)):
                raise ValueError(name + " must be a positive scalar or species vector")
            self.register_buffer(name, tensor.detach().clone())

    def forward(self, rho, dipole_density, voxel_volume):
        if (rho.ndim < 2 or min(rho.shape) < 1 or not rho.is_floating_point()
                or not torch.isfinite(rho).all() or torch.any(rho <= 0)):
            raise ValueError("rho must be finite, positive, floating [..., grid, species]")
        if dipole_density.shape != (*rho.shape, 3):
            raise ValueError("dipole_density must have shape [..., grid, species, 3]")
        if dipole_density.dtype != rho.dtype or dipole_density.device != rho.device:
            raise ValueError("rho and dipole_density must share dtype and device")
        if not torch.isfinite(dipole_density).all():
            raise ValueError("dipole_density must be finite")
        moment = _positive_components(self.dipole_magnitude, rho, "dipole_magnitude")
        wavelength = _positive_components(self.thermal_wavelength, rho, "thermal_wavelength")
        reduced = dipole_density / (rho * moment)[..., None]
        square = reduced.square().sum(dim=-1)
        if torch.any(square >= 1):
            raise ValueError("fixed dipoles require |P| < m*rho (finite interior)")
        # Express the zero-polarization branch in p^2, avoiding the undefined
        # Hessian of a bare vector norm at the origin.
        safe_p = square.clamp_min(1e-6).sqrt()
        xi = inverse_langevin(safe_p)
        entropy_series = (
            1.5 * square + 0.45 * square**2
            + 99 / 350 * square**3 + 1539 / 7000 * square**4
        )
        entropy = torch.where(
            square < 1e-6, entropy_series, xi * safe_p - log_sinhc(xi)
        )
        ratio_series = (
            3 + 9 / 5 * square + 297 / 175 * square**2 + 1539 / 875 * square**3
        )
        ratio = torch.where(square < 1e-6, ratio_series, xi / safe_p)
        log_z = ratio * square - entropy
        log_rho = torch.log(rho) + 3 * torch.log(wavelength)
        return {
            "beta_F_id": density_weighted_integral(
                rho, log_rho - 1 + entropy, voxel_volume
            ),
            "density_derivative": log_rho - log_z,
            "polarization_derivative": ratio[..., None] * reduced / moment[..., None],
        }
