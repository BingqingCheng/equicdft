"""Shared intrinsic/external evaluation for scalar and fixed-dipole solvers."""

import torch

from .._species import species_vector
from ..functional.ideal import validate_density_field
from ..functional.polarization_ideal import FixedDipoleIdeal
from ..functional.thermodynamics import density_weighted_integral
from ..geometry.grid import voxel_volume
from .numerics import (
    _polarization_residual_norms,
    _resolve_accessibility_masks,
)


def fixed_dipole_inputs(solver, data):
    """Validate one coupled fixed-dipole field."""

    potential = data["V_ext"]
    field = data["E_ext"]
    if (
        not torch.is_tensor(potential)
        or potential.ndim != 2
        or not potential.is_floating_point()
        or min(potential.shape) < 1
        or not torch.isfinite(potential).all()
    ):
        raise ValueError("V_ext must be finite floating [grid, species]")
    if (
        not torch.is_tensor(field)
        or field.shape != (*potential.shape, 3)
        or field.dtype != potential.dtype
        or field.device != potential.device
        or not torch.isfinite(field).all()
    ):
        raise ValueError(
            "E_ext must match V_ext dtype/device with shape [grid, species, 3]"
        )
    beta = torch.as_tensor(
        data["beta"], dtype=potential.dtype, device=potential.device
    )
    if beta.ndim or not torch.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be a finite positive scalar")
    if hasattr(solver.model, "boltzmann_constant"):
        temperature = torch.as_tensor(
            data["temperature"],
            dtype=potential.dtype,
            device=potential.device,
        )
        k_b = solver.model.boltzmann_constant.to(potential)
        if (
            temperature.ndim
            or not torch.isfinite(temperature)
            or temperature <= 0
            or not torch.allclose(
                beta * k_b * temperature,
                beta.new_ones(()),
                atol=0,
                rtol=1.0e-6,
            )
        ):
            raise ValueError(
                "beta must agree with model k_B and data temperature"
            )
    spacing = torch.as_tensor(
        data["grid_spacing"],
        dtype=potential.dtype,
        device=potential.device,
    ).reshape(-1)
    if (
        spacing.shape != (3,)
        or not torch.isfinite(spacing).all()
        or torch.any(spacing <= 0)
    ):
        raise ValueError(
            "grid_spacing must contain three positive finite values"
        )
    if not torch.allclose(
        spacing,
        spacing[0].expand_as(spacing),
        rtol=1.0e-7,
        atol=0,
    ):
        raise ValueError("fixed-dipole solving requires cubic voxels")
    excluded, accessible = _resolve_accessibility_masks(data, potential)
    moment = species_vector(
        solver.model.ideal.dipole_magnitude,
        potential.shape[-1],
        potential,
        "dipole_magnitude",
        positive=True,
    )
    wavelength = species_vector(
        data.get("thermal_wavelength", solver.model.ideal.thermal_wavelength),
        potential.shape[-1],
        potential,
        "thermal_wavelength",
        positive=True,
    )
    return beta, voxel_volume(spacing), moment, wavelength, excluded, accessible


def _intrinsic_outputs(model, data, compute_derivatives, polarized=False):
    """Solvers use physical constraints, never the model's fitting mask."""
    kwargs = dict(compute_c1=compute_derivatives, compute_c2=False,
                  compute_polarization_derivative=polarized and compute_derivatives)
    if hasattr(model, "compute_local_mu"):
        kwargs["compute_local_mu"] = False
    if hasattr(model, "ideal_mask"):
        # Analytic Euler updates and line-search objectives use the full
        # physical ideal law, never a density-threshold fitting selection.
        kwargs["apply_ideal_mask"] = False
    return model(data, **kwargs)


def _set_density_residuals(result, residual, chemical_potential, maximum, rms):
    """Record the density stationarity residual and its two norms."""

    result.update(
        beta_mu=chemical_potential,
        density_residual=residual,
        max_density_residual=maximum,
        rms_density_residual=rms,
    )




# Inputs carried through to the result so a solve reports what it was given.
_CARRIED_KEYS = (
    "rho", "dipole_density", "V_ext", "E_ext", "mu", "temperature",
    "beta", "grid_spacing", "thermal_wavelength", "excluded_mask", "grid_center",
)


def _detached(result):
    return {
        key: value.detach() if torch.is_tensor(value) else value
        for key, value in result.items()
    }


def _started_result(data, outputs, rho, volume):
    """Collect the carried inputs, the model outputs and the particle count."""

    result = {key: data[key] for key in _CARRIED_KEYS if key in data}
    result.update(outputs)
    result["particle_numbers"] = volume[..., None] * rho.sum(dim=-2)
    return result


def _canonical_chemical_potential(weights, local_mu, tiny):
    """Return the density-weighted mean of the local chemical potential."""

    return (weights * local_mu).sum(dim=-2) / weights.sum(dim=-2).clamp_min(tiny)


def _density_residuals(rho, local_mu, chemical_potential, weights, excluded_density,
                       accessible=None):
    """Return the density stationarity residual and its maximum and rms norms.

    ``accessible`` selects the coupled solver's compact accessible sums; the
    scalar path passes ``None`` and masks instead, which keeps the leading
    batch dimension its results carry.
    """

    residual = (local_mu - chemical_potential[..., None, :]).masked_fill(
        excluded_density, 0,
    )
    active = residual if accessible is None else residual[accessible]
    maximum = active.abs().amax(dim=(-2, -1))
    rms = torch.sqrt(
        (weights * active.square()).sum(dim=(-2, -1))
        / weights.sum(dim=(-2, -1)).clamp_min(torch.finfo(rho.dtype).tiny)
    )
    return residual, maximum, rms


def _evaluate_scalar(solver, data, compute_derivatives):
    """Evaluate a density-only field, with an optional leading batch axis."""

    rho = data["rho"].detach().clone()
    data = dict(data, rho=rho)
    volume = voxel_volume(data["grid_spacing"])
    excluded, _ = _resolve_accessibility_masks(data, rho)
    validate_density_field(rho, excluded)
    excluded_density = excluded[..., None].expand_as(rho)

    outputs = _intrinsic_outputs(solver.model, data, compute_derivatives)
    result = _started_result(data, outputs, rho, volume)

    if "V_ext" not in data:
        return _detached(result)

    beta = data["beta"][..., None, None]
    external = density_weighted_integral(rho, beta * data["V_ext"], volume)
    result.update(
        beta_V_ext=external, beta_external=external,
        beta_A=outputs["beta_F"] + external,
    )
    if "mu" in data:
        beta_mu_n = density_weighted_integral(
            rho, beta * data["mu"][..., None, :], volume,
        )
        result.update(beta_mu_N=beta_mu_n, beta_Omega=result["beta_A"] - beta_mu_n)

    if compute_derivatives:
        # The summation order is the one existing minimizer trajectories were
        # recorded with, so it is spelled out rather than taken from the
        # model's density_derivative.
        local_mu = (
            outputs["ideal_density_derivative"] + beta * data["V_ext"] - outputs["c1"]
        )
        result["beta_mu_local"] = local_mu
        weights = rho.masked_fill(excluded_density, 0)
        if "mu" in data:
            chemical_potential = data["beta"][..., None] * data["mu"]
        else:
            chemical_potential = _canonical_chemical_potential(
                weights, local_mu.masked_fill(excluded_density, 0),
                torch.finfo(rho.dtype).tiny,
            )
        residual, maximum, rms = _density_residuals(
            rho, local_mu, chemical_potential, weights, excluded_density,
        )
        _set_density_residuals(result, residual, chemical_potential, maximum, rms)
        result.update(maximum_residual=maximum, rms_residual=rms)
    return _detached(result)


def _evaluate_fixed_dipole(solver, data, compute_derivatives):
    """Evaluate one coupled density and polarization field, always canonical."""

    beta, volume, moment, _, excluded, accessible = fixed_dipole_inputs(solver, data)
    potential = data["V_ext"]
    rho = torch.as_tensor(
        data["rho"], dtype=potential.dtype, device=potential.device,
    ).detach().clone()
    polarization = torch.as_tensor(
        data["dipole_density"], dtype=potential.dtype, device=potential.device,
    ).detach().clone()
    if rho.shape != potential.shape:
        raise ValueError("rho must match V_ext shape")
    if polarization.shape != (*rho.shape, 3):
        raise ValueError("dipole_density must have shape rho.shape + (3,)")
    if torch.any(polarization[excluded] != 0):
        raise ValueError("rho and dipole_density must be zero in excluded voxels")
    rho.requires_grad_(compute_derivatives)
    polarization.requires_grad_(compute_derivatives)
    data = dict(data, rho=rho, dipole_density=polarization)

    validate_density_field(
        rho, excluded, "rho and dipole_density must be zero in excluded voxels",
    )
    excluded_density = excluded[..., None].expand_as(rho)

    outputs = _intrinsic_outputs(solver.model, data, compute_derivatives, polarized=True)
    if outputs["beta_F_exc"].ndim:
        raise ValueError("model must return scalar beta_F_exc for one field")
    result = _started_result(data, outputs, rho, volume)
    result["maximum_alignment"] = (
        polarization[accessible].norm(dim=-1) / (rho[accessible] * moment)
    ).max()

    beta_field = beta[..., None, None]
    # Keep the historical coupled summation order: its native-dtype
    # objective controls line-search acceptance.
    external = beta * volume * (
        (rho * data["V_ext"]).sum()
        - (polarization * data["E_ext"]).sum()
    )
    result.update(
        beta_V_ext=density_weighted_integral(rho, beta_field * data["V_ext"], volume),
        beta_P_ext=-beta * volume * (polarization * data["E_ext"]).sum(),
        beta_external=external,
        beta_A=outputs["beta_F"] + external,
    )
    if not torch.isfinite(result["beta_A"]):
        raise FloatingPointError("nonfinite fixed-dipole objective")

    if compute_derivatives:
        local_mu = outputs["density_derivative"] + beta_field * data["V_ext"]
        result["beta_mu_local"] = local_mu
        weights = rho[accessible]
        chemical_potential = _canonical_chemical_potential(
            weights, local_mu[accessible], torch.finfo(rho.dtype).tiny,
        )
        residual, maximum, rms = _density_residuals(
            rho, local_mu, chemical_potential, weights, excluded_density, accessible,
        )
        _set_density_residuals(result, residual, chemical_potential, maximum, rms)

        p_residual = torch.where(
            accessible[:, None, None],
            outputs["polarization_derivative"] - beta * data["E_ext"],
            torch.zeros_like(polarization),
        )
        scaled = p_residual * moment[..., None]
        p_max, p_rms = _polarization_residual_norms(rho, scaled, accessible)
        result.update(
            polarization_residual=p_residual,
            scaled_polarization_residual=scaled,
            max_polarization_residual=p_max,
            rms_polarization_residual=p_rms,
        )
        result.update(
            maximum_residual=torch.maximum(maximum, p_max),
            rms_residual=torch.maximum(rms, p_rms),
        )
    return _detached(result)


@torch.no_grad()
def evaluate(solver, data, compute_derivatives=True):
    """Add external coupling and stationarity diagnostics to model outputs.

    The two field kinds are evaluated by their own functions below, which
    share the result conventions but not one body of conditionals: scalar
    evaluation supports batches and either canonical or prescribed-mu
    residuals, while fixed-dipole evaluation keeps its single-field
    canonical contract and its historical summation order.

    Bounds and projections are applied by the solve drivers, not here.
    Inputs are copied before the model enables differentiation; all returned
    tensors are detached. Use the model directly for differentiable outputs.
    """

    data = solver._move_data(data)
    if isinstance(solver.model.ideal, FixedDipoleIdeal):
        return _evaluate_fixed_dipole(solver, data, compute_derivatives)
    return _evaluate_scalar(solver, data, compute_derivatives)
