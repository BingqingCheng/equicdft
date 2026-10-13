"""Ideal entropy of freely rotating, fixed-magnitude electric point dipoles.

Angular measure is dOmega/(4*pi). rho is number/volume and P is dipole/volume.
The local finite differentiable domain is rho > 0 and |P| < m*rho. The shared
grid evaluator handles exact vacuum; perfect alignment remains a boundary limit.
"""

import torch
from .._config import make_config, register
from .._species import (
    broadcast_species_field,
    species_parameter,
    species_vector,
)
from .ideal import IdealReference

from .thermodynamics import (
    log_dimensionless_density,
)

__all__ = [
    "langevin",
    "log_sinhc",
    "inverse_langevin",
    "dipole_alignment",
    "FixedDipoleIdeal",
]


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


class _InverseLangevin(torch.autograd.Function):
    """Solve the scalar inverse, differentiating its defining equation.

    The root value uses the same twelve Newton steps as the original
    implementation.  Saving the *output*, rather than a detached Newton
    intermediate, keeps implicit derivatives differentiable for Hessians and
    their parameter gradients without retaining the iteration graph.
    """

    @staticmethod
    def forward(p):
        if not torch.isfinite(p).all() or torch.any(p.abs() >= 1):
            raise ValueError("inverse Langevin requires finite |p| < 1")
        # Refine the rational estimate; do not replace the inverse with it.
        x = p * (3 - p.square()) / (1 - p.square())
        for _ in range(12):
            x = x - (langevin(x) - p) / _langevin_prime(x)
        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(output)
        ctx.save_for_forward(output)

    @staticmethod
    def backward(ctx, gradient):
        (x,) = ctx.saved_tensors
        return gradient / _langevin_prime(x)

    @staticmethod
    def jvp(ctx, tangent):
        (x,) = ctx.saved_tensors
        return tangent / _langevin_prime(x)

    @staticmethod
    def vmap(info, in_dims, p):
        # The inverse is elementwise.  Applying it to the full explicit batch
        # also leaves the domain check outside vmap's data-dependent branching.
        return _InverseLangevin.apply(p), in_dims[0]


def inverse_langevin(p):
    """Differentiable inverse on -1 < p < 1; Newton solves the actual L.

    Derivatives follow ``dx/dp = 1/L'(x)`` implicitly, including higher-order
    reverse derivatives and ``torch.func`` forward/batched transforms.
    """
    return _InverseLangevin.apply(p)


def dipole_alignment(vector):
    """Return L(|a|)*a/|a| with smooth first and second derivatives at a=0."""
    square = vector.square().sum(dim=-1, keepdim=True)
    safe = square.clamp_min(0.01).sqrt()
    coefficient = torch.where(
        square < 0.01, _langevin_ratio_series(square), langevin(safe) / safe
    )
    return coefficient * vector


@register
class FixedDipoleIdeal(IdealReference):
    """Ideal beta free energy and derivatives at fixed independent rho and P.

    ``dipole_magnitude`` and ``thermal_wavelength`` are positive scalars or one
    value per species, in the same dipole and length units as the fields.
    Input shapes are rho [..., grid, species], P [..., grid, species, 3].
    Density derivative has a positive sign (unlike excess c1). Polarization
    derivative is positive, with units inverse dipole; both divide out Delta V.
    """

    requires_dipole_density = True

    def __init__(self, dipole_magnitude, *, thermal_wavelength=1.0):
        super().__init__(thermal_wavelength=thermal_wavelength)
        self.register_buffer("dipole_magnitude", species_parameter(
            dipole_magnitude, "dipole_magnitude",
        ))

    def to_config(self):
        return make_config(self, dipole_magnitude=self.dipole_magnitude,
                           thermal_wavelength=self.thermal_wavelength)

    def local(self, rho, dipole_density=None, thermal_wavelength=None):
        if (rho.ndim < 2 or min(rho.shape) < 1 or not rho.is_floating_point()
                or not torch.isfinite(rho).all() or torch.any(rho <= 0)):
            raise ValueError("rho must be finite, positive, floating [..., grid, species]")
        if dipole_density is None or dipole_density.shape != (*rho.shape, 3):
            raise ValueError("dipole_density must have shape [..., grid, species, 3]")
        if dipole_density.dtype != rho.dtype or dipole_density.device != rho.device:
            raise ValueError("rho and dipole_density must share dtype and device")
        if not torch.isfinite(dipole_density).all():
            raise ValueError("dipole_density must be finite")
        moment = species_vector(
            self.dipole_magnitude, rho.shape[-1], rho, "dipole_magnitude",
            positive=True,
        )
        wavelength = broadcast_species_field(
            self.thermal_wavelength
            if thermal_wavelength is None
            else thermal_wavelength,
            rho, "thermal_wavelength",
        )
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
        log_rho = log_dimensionless_density(rho, wavelength)
        return {
            "energy_per_particle": log_rho - 1 + entropy,
            "ideal_density_derivative": log_rho - log_z,
            "ideal_polarization_derivative": ratio[..., None] * reduced / moment[..., None],
        }
