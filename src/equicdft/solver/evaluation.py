"""Shared intrinsic/external evaluation for scalar and fixed-dipole solvers."""

import torch

from ..functional.polarization_ideal import FixedDipoleIdeal, _positive_components
from ..functional.thermodynamics import density_weighted_integral
from ..geometry.grid import voxel_volume
from .numerics import (
    _component_tensor,
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
    moment = _positive_components(
        solver.model.ideal.dipole_magnitude,
        potential,
        "dipole_magnitude",
    ).expand(potential.shape[-1])
    wavelength = _component_tensor(
        data.get("thermal_wavelength", solver.model.ideal.thermal_wavelength),
        potential.shape[-1],
        potential,
        "thermal_wavelength",
    )
    if not torch.isfinite(wavelength).all() or torch.any(wavelength <= 0):
        raise ValueError("thermal_wavelength values must be finite and positive")
    return beta, voxel_volume(spacing), moment, wavelength, excluded, accessible


def _set_density_residuals(result, residual, chemical_potential, maximum, rms):
    """Keep common density diagnostics and the historical Euler names in sync."""

    result.update(
        beta_mu=chemical_potential,
        density_residual=residual,
        max_density_residual=maximum,
        rms_density_residual=rms,
        euler_lagrange_residual=residual,
        max_euler_lagrange_residual=maximum,
        rms_euler_lagrange_residual=rms,
    )


@torch.no_grad()
def evaluate(solver, data, compute_derivatives=True):
    """Add external coupling and stationarity diagnostics to model outputs.

    Scalar evaluation supports batches and either canonical or prescribed-mu
    residuals. Fixed-dipole evaluation retains its single-field canonical
    contract. Bounds/projections are applied by the solve drivers, not here.
    Inputs are copied before the model enables differentiation; all returned
    tensors are detached. Use the model directly for differentiable outputs.
    """

    data = solver._move_data(data)
    polarized = isinstance(solver.model.ideal, FixedDipoleIdeal)
    if polarized:
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
    else:
        rho = data["rho"].detach().clone()
        data = dict(data, rho=rho)
        volume = voxel_volume(data["grid_spacing"])
        excluded, accessible = _resolve_accessibility_masks(data, rho)

    if torch.any(rho < 0).item():
        raise ValueError("rho must be nonnegative")
    excluded_density = excluded[..., None].expand_as(rho)
    if torch.any(rho[excluded_density] != 0).item():
        raise ValueError(
            "rho and dipole_density must be zero in excluded voxels"
            if polarized else "rho must be zero at excluded grid points"
        )

    outputs = solver.model(
        data, compute_c1=compute_derivatives, compute_c2=False,
        compute_polarization_derivative=polarized and compute_derivatives,
    )
    result = {
        key: data[key] for key in (
            "rho", "dipole_density", "V_ext", "E_ext", "mu", "temperature",
            "beta", "grid_spacing", "thermal_wavelength", "excluded_mask", "grid_center",
        ) if key in data
    }
    result.update(outputs)
    result["particle_numbers"] = volume[..., None] * rho.sum(dim=-2)
    if polarized:
        if outputs["beta_F_exc"].ndim:
            raise ValueError("model must return scalar beta_F_exc for one field")
        result["maximum_alignment"] = (
            polarization[accessible].norm(dim=-1) / (rho[accessible] * moment)
        ).max()

    if "V_ext" in data:
        beta_value = beta if polarized else data["beta"]
        beta = beta_value[..., None, None]
        density_external = density_weighted_integral(rho, beta * data["V_ext"], volume)
        external = density_external
        if polarized:
            # Keep the historical coupled summation order: its native-dtype
            # objective controls line-search acceptance.
            external = beta_value * volume * (
                (rho * data["V_ext"]).sum()
                - (polarization * data["E_ext"]).sum()
            )
            result["beta_P_ext"] = (
                -beta_value * volume * (polarization * data["E_ext"]).sum()
            )
        result.update(
            beta_V_ext=density_external, beta_external=external,
            beta_A=outputs["beta_F"] + external,
        )
        if polarized and not torch.isfinite(result["beta_A"]):
            raise FloatingPointError("nonfinite fixed-dipole objective")
        if "mu" in data and not polarized:
            beta_mu_n = density_weighted_integral(
                rho, beta * data["mu"][..., None, :], volume,
            )
            result.update(beta_mu_N=beta_mu_n, beta_Omega=result["beta_A"] - beta_mu_n)

        if compute_derivatives:
            # Scalar ordering is retained for existing minimizer trajectories.
            local_mu = (
                outputs["density_derivative"] + beta * data["V_ext"]
                if polarized else
                outputs["ideal_density_derivative"] + beta * data["V_ext"] - outputs["c1"]
            )
            result["beta_mu_local"] = local_mu
            # Compact accessible sums retain the coupled solver's arithmetic;
            # the scalar branch also supports a leading batch dimension.
            weights = (
                rho[accessible] if polarized else rho.masked_fill(excluded_density, 0)
            )
            active_mu = (
                local_mu[accessible] if polarized
                else local_mu.masked_fill(excluded_density, 0)
            )
            if "mu" in data and not polarized:
                chemical_potential = data["beta"][..., None] * data["mu"]
            else:
                chemical_potential = (weights * active_mu).sum(dim=-2) / weights.sum(
                    dim=-2
                ).clamp_min(torch.finfo(rho.dtype).tiny)
            residual = (local_mu - chemical_potential[..., None, :]).masked_fill(
                excluded_density, 0,
            )
            active_residual = residual[accessible] if polarized else residual
            maximum = active_residual.abs().amax(dim=(-2, -1))
            rms = torch.sqrt(
                (weights * active_residual.square()).sum(dim=(-2, -1))
                / weights.sum(dim=(-2, -1)).clamp_min(torch.finfo(rho.dtype).tiny)
            )
            _set_density_residuals(result, residual, chemical_potential, maximum, rms)
            if polarized:
                p_residual = torch.where(
                    accessible[:, None, None],
                    outputs["polarization_derivative"] - beta_value * data["E_ext"],
                    torch.zeros_like(polarization),
                )
                scaled = p_residual * moment[..., None]
                p_max, p_rms = _polarization_residual_norms(rho, scaled, accessible)
                result.update(
                    polarization_residual=p_residual,
                    polarization_euler_lagrange_residual=p_residual,
                    scaled_polarization_residual=scaled,
                    max_polarization_residual=p_max,
                    rms_polarization_residual=p_rms,
                )
                maximum, rms = torch.maximum(maximum, p_max), torch.maximum(rms, p_rms)
            result.update(maximum_residual=maximum, rms_residual=rms)

    return {
        key: value.detach() if torch.is_tensor(value) else value
        for key, value in result.items()
    }
