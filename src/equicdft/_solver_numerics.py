"""Private numerical kernels used by the equilibrium grid solver."""

from typing import Dict, Optional, Sequence, Union

import torch


def _log_dimensionless_density(
    rho: torch.Tensor,
    thermal_wavelength: torch.Tensor,
) -> torch.Tensor:
    tiny = torch.finfo(rho.dtype).tiny
    return torch.log(
        torch.clamp(rho, min=tiny)
        * thermal_wavelength[..., None, :] ** 3
    )


def _component_tensor(
    values: Union[float, Sequence[float], torch.Tensor],
    n_types: int,
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor:
    values = torch.as_tensor(
        values,
        dtype=reference.dtype,
        device=reference.device,
    ).reshape(-1)
    if values.numel() == 1:
        values = values.repeat(n_types)
    if values.shape != (n_types,):
        raise ValueError("{} must contain one value per type".format(name))
    return values


def _thermodynamic_objective(
    evaluation: Dict[str, torch.Tensor],
    fixed_particle_numbers: bool,
    cell_volume: torch.Tensor,
    thermal_wavelength: torch.Tensor,
) -> torch.Tensor:
    """Return the objective with high-precision scalar accumulation."""

    rho = evaluation["rho"]
    accumulation_dtype = _accumulation_dtype(rho.dtype)
    rho = rho.to(accumulation_dtype)
    wavelength = thermal_wavelength.to(accumulation_dtype)
    log_density = _log_dimensionless_density(rho, wavelength)
    beta = evaluation["beta"].to(accumulation_dtype)
    component_density = (
        rho * (log_density - 1.0)
        + rho * beta * evaluation["V_ext"].to(accumulation_dtype)
    )
    if not fixed_particle_numbers:
        component_density = (
            component_density
            - rho
            * beta
            * evaluation["mu"].to(accumulation_dtype)[None, :]
        )
    ideal_external_objective = (
        cell_volume.to(accumulation_dtype)
        * torch.sum(component_density)
    )
    return (
        ideal_external_objective
        + evaluation["beta_F_exc"].to(accumulation_dtype)
    )


def _mirror_descent_trial(
    rho: torch.Tensor,
    functional_gradient: torch.Tensor,
    step_size: float,
    particle_numbers: Optional[torch.Tensor],
    cell_volume: torch.Tensor,
    max_log_density_change: float,
    maximum_density: Optional[torch.Tensor],
) -> torch.Tensor:
    """Return one positive exponentiated-gradient trial density."""

    log_rho = torch.log(
        torch.clamp(rho, min=torch.finfo(rho.dtype).tiny)
    )
    log_change = torch.clamp(
        -step_size * functional_gradient,
        min=-max_log_density_change,
        max=max_log_density_change,
    )
    trial_log_rho = log_rho + log_change
    if particle_numbers is None:
        trial_rho = torch.exp(trial_log_rho)
        if maximum_density is not None:
            trial_rho = torch.minimum(
                trial_rho,
                maximum_density[None, :],
            )
        return trial_rho

    accumulation_dtype = _accumulation_dtype(rho.dtype)
    trial_rho = (
        particle_numbers.to(accumulation_dtype)[None, :]
        * torch.softmax(trial_log_rho.to(accumulation_dtype), dim=0)
        / cell_volume.to(accumulation_dtype)
    )
    return _normalize_particle_numbers(
        trial_rho.to(rho.dtype),
        particle_numbers,
        cell_volume,
        maximum_density=maximum_density,
    )


def _maximum_relative_change(
    rho: torch.Tensor,
    next_rho: torch.Tensor,
) -> float:
    """Return the largest gridwise relative density update."""

    return torch.max(
        torch.abs(next_rho - rho) / torch.clamp(rho, min=1.0e-12)
    ).item()


def _normalize_particle_numbers(
    rho: torch.Tensor,
    particle_numbers: torch.Tensor,
    cell_volume: torch.Tensor,
    maximum_density: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Normalize each component, optionally subject to an upper density."""

    accumulation_dtype = _accumulation_dtype(rho.dtype)
    rho_accumulated = rho.to(accumulation_dtype)
    target_sums = (
        particle_numbers.to(accumulation_dtype)
        / cell_volume.to(accumulation_dtype)
    )

    if maximum_density is None:
        current_sums = torch.sum(rho_accumulated, dim=0)
        normalized_accumulated = (
            rho_accumulated * (target_sums / current_sums)[None, :]
        )
    else:
        caps = maximum_density.to(accumulation_dtype)
        normalized_accumulated = torch.zeros_like(rho_accumulated)
        for component in range(rho.shape[-1]):
            weights = torch.clamp(
                rho_accumulated[:, component],
                min=torch.finfo(accumulation_dtype).tiny,
            )
            active = torch.ones(
                rho.shape[0],
                dtype=torch.bool,
                device=rho.device,
            )
            remaining = target_sums[component]
            cap = caps[component]

            # Capped proportional (KL) projection: saturated entries are
            # removed and the remaining mass is redistributed proportionally.
            while torch.any(active).item():
                scale = remaining / torch.sum(weights[active])
                active_values = scale * weights[active]
                saturated_local = active_values >= cap
                active_indices = torch.nonzero(
                    active,
                    as_tuple=False,
                ).reshape(-1)
                if not torch.any(saturated_local).item():
                    normalized_accumulated[
                        active_indices,
                        component,
                    ] = active_values
                    remaining = remaining.new_zeros(())
                    break

                saturated_indices = active_indices[saturated_local]
                normalized_accumulated[
                    saturated_indices,
                    component,
                ] = cap
                active[saturated_indices] = False
                remaining = remaining - cap * saturated_indices.numel()

            if torch.abs(remaining).item() > 1.0e-10:
                raise RuntimeError(
                    "failed to normalize density under maximum_density"
                )

    normalized = normalized_accumulated.to(rho.dtype).clone()
    # Correct casting error so line-search directions remain exactly tangent
    # to the fixed-particle-number constraint.
    for component in range(normalized.shape[-1]):
        current_sum = torch.sum(
            normalized[:, component].to(accumulation_dtype)
        )
        correction = (target_sums[component] - current_sum).to(rho.dtype)
        if correction.item() > 0.0 and maximum_density is not None:
            available = (
                maximum_density[component] - normalized[:, component]
            )
            grid_index = torch.argmax(available)
            if correction > available[grid_index]:
                raise RuntimeError(
                    "float correction exceeds room below maximum_density"
                )
        else:
            grid_index = torch.argmax(normalized[:, component])
        normalized[grid_index, component] += correction
    return normalized


def _euler_residual(
    rho: torch.Tensor,
    c1: torch.Tensor,
    V_ext: torch.Tensor,
    beta: torch.Tensor,
    thermal_wavelength: torch.Tensor,
    mu: Optional[torch.Tensor],
    density_threshold: float,
    maximum_density: Optional[torch.Tensor] = None,
):
    """Return the dimensionless beta*mu residual and optional bound KKT.

    Both ``local_chemical_potential`` and ``chemical_potential`` below denote
    dimensionless ``beta*mu`` quantities, so their difference is likewise
    dimensionless.
    """

    local_chemical_potential = (
        _log_dimensionless_density(rho, thermal_wavelength)
        + beta * V_ext
        - c1
    )
    if mu is None:
        averaging_weights = rho
        if maximum_density is not None:
            cap_tolerance = 1.0e-6 * torch.maximum(
                maximum_density,
                torch.ones_like(maximum_density),
            )
            below_cap = rho < (
                maximum_density - cap_tolerance
            )[None, :]
            free_weights = rho * below_cap
            has_free_weight = torch.sum(free_weights, dim=0) > 0.0
            averaging_weights = torch.where(
                has_free_weight[None, :],
                free_weights,
                rho,
            )
        chemical_potential = torch.sum(
            averaging_weights * local_chemical_potential,
            dim=0,
        ) / torch.sum(averaging_weights, dim=0)
    else:
        chemical_potential = beta * mu
    residual = local_chemical_potential - chemical_potential[None, :]

    constrained_residual = residual
    if maximum_density is not None:
        cap_tolerance = 1.0e-6 * torch.maximum(
            maximum_density,
            torch.ones_like(maximum_density),
        )
        at_cap = rho >= (maximum_density - cap_tolerance)[None, :]
        constrained_residual = torch.where(
            at_cap,
            torch.clamp(residual, min=0.0),
            residual,
        )

    active = (
        torch.ones_like(rho, dtype=torch.bool)
        if density_threshold == 0.0
        else rho > density_threshold
    )
    if torch.any(active).item():
        max_residual = torch.max(
            torch.abs(constrained_residual[active])
        ).item()
        rms_weights = rho * active.to(rho.dtype)
        rms_residual = torch.sqrt(
            torch.sum(rms_weights * constrained_residual.square())
            / torch.sum(rms_weights)
        ).item()
    else:
        max_residual = float("inf")
        rms_residual = float("inf")
    return constrained_residual, chemical_potential, max_residual, rms_residual


def _residuals_converged(
    max_residual: float,
    rms_residual: float,
    tolerance_residual: float,
    tolerance_rms_residual: Optional[float],
) -> bool:
    """Return whether the configured maximum and RMS bounds are satisfied."""

    return bool(
        max_residual <= tolerance_residual
        and (
            tolerance_rms_residual is None
            or rms_residual <= tolerance_rms_residual
        )
    )


def _accumulation_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float16, torch.bfloat16, torch.float32):
        return torch.float64
    return dtype
