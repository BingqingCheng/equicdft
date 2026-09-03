"""Private numerical kernels used by the equilibrium grid solver."""

from typing import Dict, Optional, Sequence, Union

import torch

from .energy import (
    density_weighted_integral,
    log_dimensionless_density,
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
    voxel_volume: torch.Tensor,
    thermal_wavelength: torch.Tensor,
) -> torch.Tensor:
    """Return the objective with high-precision scalar accumulation."""

    rho = evaluation["rho"]
    accumulation_dtype = _accumulation_dtype(rho.dtype)
    rho = rho.to(accumulation_dtype)
    wavelength = thermal_wavelength.to(accumulation_dtype)
    log_density = log_dimensionless_density(rho, wavelength)
    beta = evaluation["beta"].to(accumulation_dtype)
    per_particle = (
        log_density - 1.0
        + beta * evaluation["V_ext"].to(accumulation_dtype)
    )
    if not fixed_particle_numbers:
        per_particle = (
            per_particle
            - beta * evaluation["mu"].to(accumulation_dtype)[None, :]
        )
    ideal_external_objective = density_weighted_integral(
        rho,
        per_particle,
        voxel_volume.to(accumulation_dtype),
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
    voxel_volume: torch.Tensor,
    max_log_density_change: float,
    maximum_density: Optional[torch.Tensor],
    accessible_mask: Optional[torch.Tensor] = None,
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
    else:
        accumulation_dtype = _accumulation_dtype(rho.dtype)
        trial_rho = (
            particle_numbers.to(accumulation_dtype)[None, :]
            * torch.softmax(trial_log_rho.to(accumulation_dtype), dim=0)
            / voxel_volume.to(accumulation_dtype)
        ).to(rho.dtype)
    return _project_density_constraints(
        trial_rho,
        particle_numbers,
        voxel_volume,
        maximum_density,
        accessible_mask,
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
    voxel_volume: torch.Tensor,
    maximum_density: Optional[torch.Tensor] = None,
    accessible_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Normalize each component, optionally subject to an upper density."""

    accumulation_dtype = _accumulation_dtype(rho.dtype)
    accessible = _accessible_density_mask(rho, accessible_mask)
    rho_accumulated = torch.where(
        accessible,
        rho.to(accumulation_dtype),
        torch.zeros((), dtype=accumulation_dtype, device=rho.device),
    )
    target_sums = (
        particle_numbers.to(accumulation_dtype)
        / voxel_volume.to(accumulation_dtype)
    )

    if maximum_density is None:
        current_sums = torch.sum(rho_accumulated, dim=0)
        if torch.any(current_sums <= 0.0).item():
            raise ValueError(
                "rho must have positive weight on accessible grid points"
            )
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
            active = accessible[:, component].clone()
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
            available = torch.where(
                accessible[:, component],
                maximum_density[component] - normalized[:, component],
                torch.full_like(normalized[:, component], -torch.inf),
            )
            grid_index = torch.argmax(available)
            if correction > available[grid_index]:
                raise RuntimeError(
                    "float correction exceeds room below maximum_density"
                )
        else:
            grid_index = torch.argmax(normalized[:, component])
        normalized[grid_index, component] += correction
    return torch.where(accessible, normalized, torch.zeros_like(normalized))


def _project_density_constraints(
    rho: torch.Tensor,
    particle_numbers: Optional[torch.Tensor],
    voxel_volume: torch.Tensor,
    maximum_density: Optional[torch.Tensor] = None,
    accessible_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply the solver's mask, particle-number, and upper-bound constraints."""

    accessible = _accessible_density_mask(rho, accessible_mask)
    projected = torch.where(accessible, rho, torch.zeros_like(rho))
    if particle_numbers is not None:
        return _normalize_particle_numbers(
            projected,
            particle_numbers,
            voxel_volume,
            maximum_density=maximum_density,
            accessible_mask=accessible_mask,
        )
    if maximum_density is not None:
        projected = torch.minimum(projected, maximum_density[None, :])
    return projected


def _anderson_log_density_candidate(
    log_density_history: Sequence[torch.Tensor],
    residual_history: Sequence[torch.Tensor],
    weights: torch.Tensor,
    regularization: float,
    damping: float,
) -> torch.Tensor:
    """Return a density-weighted constrained Anderson log-density trial."""

    history_size = len(log_density_history)
    if history_size != len(residual_history):
        raise ValueError("Anderson density and residual history differ")
    if history_size < 2:
        raise ValueError("Anderson acceleration requires two history entries")

    accumulation_dtype = torch.float64
    residual_matrix = torch.stack(
        residual_history,
        dim=-1,
    ).reshape(-1, history_size).to(accumulation_dtype)
    flattened_weights = weights.reshape(-1).to(accumulation_dtype)
    gram = residual_matrix.mT @ (
        flattened_weights[:, None] * residual_matrix
    )
    scale = torch.clamp(
        torch.trace(gram) / history_size,
        min=torch.finfo(accumulation_dtype).eps,
    )
    system = gram + regularization * scale * torch.eye(
        history_size,
        dtype=accumulation_dtype,
        device=gram.device,
    )
    ones = torch.ones(
        history_size,
        dtype=accumulation_dtype,
        device=gram.device,
    )
    coefficients = torch.linalg.solve(system, ones)
    denominator = torch.sum(coefficients)
    if (
        not torch.all(torch.isfinite(coefficients)).item()
        or not torch.isfinite(denominator).item()
        or torch.abs(denominator).item()
        <= torch.finfo(accumulation_dtype).eps
    ):
        raise RuntimeError("Anderson coefficient solve is singular")
    coefficients = coefficients / denominator

    candidates = (
        torch.stack(log_density_history)
        + damping * torch.stack(residual_history)
    ).to(accumulation_dtype)
    coefficient_shape = (history_size,) + (1,) * (
        candidates.ndim - 1
    )
    return torch.sum(
        coefficients.reshape(coefficient_shape) * candidates,
        dim=0,
    ).to(log_density_history[-1].dtype)


def _euler_residual(
    rho: torch.Tensor,
    c1: torch.Tensor,
    V_ext: torch.Tensor,
    beta: torch.Tensor,
    thermal_wavelength: torch.Tensor,
    mu: Optional[torch.Tensor],
    density_threshold: float,
    maximum_density: Optional[torch.Tensor] = None,
    accessible_mask: Optional[torch.Tensor] = None,
):
    """Return the dimensionless beta*mu residual and optional bound KKT.

    Both ``local_chemical_potential`` and ``chemical_potential`` below denote
    dimensionless ``beta*mu`` quantities, so their difference is likewise
    dimensionless.
    """

    accessible = _accessible_density_mask(rho, accessible_mask)
    local_chemical_potential = (
        log_dimensionless_density(rho, thermal_wavelength)
        + beta * V_ext
        - c1
    )
    local_chemical_potential = torch.where(
        accessible,
        local_chemical_potential,
        torch.zeros_like(local_chemical_potential),
    )
    if mu is None:
        averaging_weights = torch.where(
            accessible,
            rho,
            torch.zeros_like(rho),
        )
        if maximum_density is not None:
            cap_tolerance = 1.0e-6 * torch.maximum(
                maximum_density,
                torch.ones_like(maximum_density),
            )
            below_cap = accessible & (rho < (
                maximum_density - cap_tolerance
            )[None, :])
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
    residual = torch.where(accessible, residual, torch.zeros_like(residual))

    constrained_residual = residual
    if maximum_density is not None:
        cap_tolerance = 1.0e-6 * torch.maximum(
            maximum_density,
            torch.ones_like(maximum_density),
        )
        at_cap = accessible & (
            rho >= (maximum_density - cap_tolerance)[None, :]
        )
        constrained_residual = torch.where(
            at_cap,
            torch.clamp(residual, min=0.0),
            residual,
        )

    active = accessible & (
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


def _accessible_density_mask(
    rho: torch.Tensor,
    accessible_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Broadcast a shared per-grid accessibility mask over components."""

    if accessible_mask is None:
        return torch.ones_like(rho, dtype=torch.bool)
    accessible_mask = torch.as_tensor(
        accessible_mask,
        dtype=torch.bool,
        device=rho.device,
    )
    if accessible_mask.shape != rho.shape[:-1]:
        raise ValueError("accessible_mask must have shape rho.shape[:-1]")
    return accessible_mask[..., None].expand_as(rho)


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
