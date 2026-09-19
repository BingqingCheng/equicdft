"""Mirror-descent minimization of the thermodynamic objective."""

from typing import Any, Dict, Optional

import torch

from .numerics import (
    _euler_residual,
    _stationarity_field,
    _maximum_relative_change,
    _mirror_descent_trial,
    _residuals_converged,
    _thermodynamic_objective,
)
from .options import ConvergenceOptions, MinimizeOptions
from .symmetry import _HomogeneousDensityProjection

__all__ = ["minimize_density"]


def minimize_density(
    solver,
    data: Dict[str, Any],
    rho: torch.Tensor,
    V_ext: torch.Tensor,
    thermal_wavelength: torch.Tensor,
    mu: Optional[torch.Tensor],
    fixed_N: Optional[torch.Tensor],
    voxel_volume: torch.Tensor,
    maximum_density: Optional[torch.Tensor],
    accessible_mask: torch.Tensor,
    projection: Optional[_HomogeneousDensityProjection],
    convergence: ConvergenceOptions,
    options: MinimizeOptions,
) -> Dict[str, Any]:
    """Minimize the thermodynamic objective by mirror descent.

    ``solver`` supplies :meth:`GridSolver.evaluate`; the fields arrive
    validated and projected onto the constraints. Returns the driver state
    (final density, counters, objective history).
    """

    evaluate = solver.evaluate
    max_iter = convergence.max_iter
    tolerance_residual = convergence.tolerance_residual
    tolerance_rms_residual = convergence.tolerance_rms_residual
    residual_density_threshold = convergence.residual_density_threshold or 0.0
    step_size = options.step_size
    minimum_step_size = options.minimum_step_size
    line_search_factor = options.line_search_factor
    armijo_factor = options.armijo_factor
    max_log_density_change = options.max_log_density_change

    objective_history = []
    n_iter = 0
    n_evaluations = 0
    line_search_failures = 0
    final_relative_change = float("inf")

    next_step_size = step_size
    for _ in range(max_iter):
        current_data = dict(data)
        current_data["rho"] = rho.detach().clone()
        current_data["V_ext"] = V_ext
        evaluation = evaluate(current_data, compute_derivatives=True)
        n_evaluations += 1
        objective = _thermodynamic_objective(
            evaluation,
            fixed_N is not None,
            voxel_volume,
            thermal_wavelength,
                solver.model.ideal,
        ).detach()
        if not torch.isfinite(objective).item():
            raise ValueError(
                "the thermodynamic objective became non-finite"
            )
        if not objective_history:
            objective_history.append(objective.item())

        residual, _, max_residual, rms_residual = _euler_residual(
            rho, _stationarity_field(evaluation, projection), data["beta"], mu,
            residual_density_threshold, maximum_density, accessible_mask,
        )
        if _residuals_converged(
            max_residual,
            rms_residual,
            tolerance_residual,
            tolerance_rms_residual,
        ):
            break

        # Armijo needs the true objective derivative, whereas the
        # update and convergence test use the projected KKT residual
        # when an upper density bound is active.
        line_search_gradient = torch.where(
            accessible_mask[:, None],
            _stationarity_field(evaluation),
            torch.zeros_like(rho),
        )
        if fixed_N is None:
            line_search_gradient = torch.where(
                accessible_mask[:, None],
                line_search_gradient - data["beta"] * mu[None, :],
                torch.zeros_like(line_search_gradient),
            )

        accepted = False
        trial_step_size = next_step_size
        while trial_step_size >= minimum_step_size:
            trial_rho = _mirror_descent_trial(
                rho,
                residual,
                trial_step_size,
                fixed_N,
                voxel_volume,
                max_log_density_change,
                maximum_density,
                accessible_mask,
            )
            displacement = trial_rho - rho
            objective_dtype = objective.dtype
            directional_derivative = voxel_volume.to(
                objective_dtype
            ) * torch.sum(
                line_search_gradient.to(objective_dtype)
                * displacement.to(objective_dtype)
            )

            trial_data = dict(data)
            trial_data["rho"] = trial_rho.detach().clone()
            trial_data["V_ext"] = V_ext
            trial_evaluation = evaluate(
                trial_data,
                compute_derivatives=False,
            )
            n_evaluations += 1
            trial_objective = _thermodynamic_objective(
                trial_evaluation,
                fixed_N is not None,
                voxel_volume,
                thermal_wavelength,
                solver.model.ideal,
            ).detach()

            sufficient_decrease = (
                objective + armijo_factor * directional_derivative
            )
            if (
                torch.isfinite(trial_objective).item()
                and directional_derivative.item() < 0.0
                and trial_objective.item()
                <= sufficient_decrease.item()
            ):
                accepted = True
                break
            trial_step_size *= line_search_factor

        if not accepted:
            line_search_failures += 1
            break

        relative_change = _maximum_relative_change(rho, trial_rho)
        rho = trial_rho.detach()
        n_iter += 1
        final_relative_change = relative_change
        objective_history.append(trial_objective.item())
        next_step_size = min(
            step_size,
            trial_step_size / line_search_factor,
        )

    return {
        "rho": rho,
        "n_iter": n_iter,
        "n_evaluations": n_evaluations,
        "objective_history": objective_history,
        "final_relative_density_change": final_relative_change,
        "line_search_failures": line_search_failures,
    }
