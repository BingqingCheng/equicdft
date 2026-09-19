"""Damped Euler--Lagrange fixed-point iteration with optional Anderson trials."""

from typing import Any, Dict, Optional

import torch

from .numerics import (
    _anderson_log_density_candidate,
    _euler_residual,
    _stationarity_field,
    _maximum_relative_change,
    _project_density_constraints,
    _residuals_converged,
    _thermodynamic_objective,
)
from ..functional.thermodynamics import equilibrium_log_density
from .options import AndersonOptions, ConvergenceOptions, EulerOptions

__all__ = ["solve_euler"]


def solve_euler(
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
    convergence: ConvergenceOptions,
    options: EulerOptions,
) -> Dict[str, Any]:
    """Solve the Euler fixed point with optional residual backtracking.

    ``solver`` supplies :meth:`GridSolver.evaluate`; the fields arrive
    validated and projected onto the constraints. Returns the driver state
    (final density, counters, mixing and Anderson statistics).
    """

    evaluate = solver.evaluate
    max_iter = convergence.max_iter
    tolerance_residual = convergence.tolerance_residual
    tolerance_rms_residual = convergence.tolerance_rms_residual
    residual_density_threshold = convergence.residual_density_threshold or 0.0
    tolerance_change = options.tolerance_change
    mixing = options.mixing
    adaptive_mixing = options.adaptive_mixing
    minimum_mixing = options.minimum_mixing
    maximum_mixing = options.maximum_mixing
    mixing_growth = options.mixing_growth
    mixing_backtrack_factor = options.mixing_backtrack_factor
    max_log_density_change = options.max_log_density_change
    anderson = options.anderson is not None
    anderson_settings = options.anderson or AndersonOptions()
    anderson_history = anderson_settings.history
    anderson_regularization = anderson_settings.regularization
    anderson_damping = anderson_settings.damping

    objective_history = []
    n_iter = 0
    n_evaluations = 0
    mixing_backtracks = 0
    stalled = False
    anderson_attempts = 0
    anderson_accepted = 0
    anderson_rejected = 0
    anderson_resets = 0
    anderson_log_history = []
    anderson_residual_history = []
    final_relative_change = float("inf")
    current_mixing = mixing
    current_data = dict(data)
    current_data["rho"] = rho.detach().clone()
    current_data["V_ext"] = V_ext
    evaluation = evaluate(current_data, compute_derivatives=True)
    n_evaluations += 1
    objective = _thermodynamic_objective(
        evaluation,
        fixed_N is not None,
        voxel_volume,
        solver.model.ideal,
        thermal_wavelength,
    )
    objective_history.append(objective.detach().item())

    def evaluate_trial(trial_rho: torch.Tensor):
        trial_data = dict(data)
        trial_data["rho"] = trial_rho.detach().clone()
        trial_data["V_ext"] = V_ext
        trial_evaluation = evaluate(trial_data, compute_derivatives=True)
        _, _, trial_max_residual, trial_rms_residual = _euler_residual(
            trial_rho, _stationarity_field(trial_evaluation), data["beta"], mu,
            residual_density_threshold, maximum_density, accessible_mask,
        )
        return (
            trial_evaluation,
            trial_max_residual,
            trial_rms_residual,
        )

    for _ in range(max_iter):
        c1 = evaluation["c1"].detach()
        _, _, max_residual, rms_residual = _euler_residual(
            rho, _stationarity_field(evaluation), data["beta"], mu,
            residual_density_threshold, maximum_density, accessible_mask,
        )
        if _residuals_converged(
            max_residual,
            rms_residual,
            tolerance_residual,
            tolerance_rms_residual,
        ):
            break

        current_log_density = torch.log(
            torch.clamp(rho, min=torch.finfo(rho.dtype).tiny)
        )
        if fixed_N is None:
            target_log_density = equilibrium_log_density(
                V_ext, data["beta"], mu, c1, thermal_wavelength,
            )
            log_change = torch.clamp(
                target_log_density - current_log_density,
                min=-max_log_density_change,
                max=max_log_density_change,
            )
            target_rho = torch.exp(current_log_density + log_change)
            if maximum_density is not None:
                target_rho = torch.minimum(
                    target_rho,
                    maximum_density[None, :],
                )
            target_rho = target_rho.masked_fill(
                ~accessible_mask[:, None],
                0.0,
            )
        else:
            logits = (-data["beta"] * V_ext + c1).masked_fill(
                ~accessible_mask[:, None],
                -torch.inf,
            )
            target_rho = (
                fixed_N[None, :]
                * torch.softmax(logits, dim=0)
                / voxel_volume
            )

        if anderson:
            target_log_density = torch.log(
                torch.clamp(
                    target_rho,
                    min=torch.finfo(target_rho.dtype).tiny,
                )
            )
            fixed_point_residual = (
                target_log_density - current_log_density
            ).masked_fill(~accessible_mask[:, None], 0.0)
            anderson_log_history.append(current_log_density.detach())
            anderson_residual_history.append(
                fixed_point_residual.detach()
            )
            anderson_log_history = anderson_log_history[
                -anderson_history:
            ]
            anderson_residual_history = anderson_residual_history[
                -anderson_history:
            ]

        trial_mixing = current_mixing
        while True:
            next_rho = (
                (1.0 - trial_mixing) * rho
                + trial_mixing * target_rho
            )
            next_rho = _project_density_constraints(
                next_rho,
                fixed_N,
                voxel_volume,
                maximum_density,
                accessible_mask,
            )

            (
                trial_evaluation,
                trial_max_residual,
                trial_rms_residual,
            ) = evaluate_trial(next_rho)
            n_evaluations += 1

            if (
                not adaptive_mixing
                or trial_rms_residual <= rms_residual
                or trial_mixing <= minimum_mixing
            ):
                break
            trial_mixing = max(
                minimum_mixing,
                trial_mixing * mixing_backtrack_factor,
            )
            mixing_backtracks += 1

        if (
            anderson
            and len(anderson_log_history) == anderson_history
            and (n_iter + 1) % anderson_history == 0
        ):
            anderson_attempts += 1
            try:
                weights = torch.where(
                    accessible_mask[:, None],
                    rho,
                    torch.zeros_like(rho),
                )
                weights = weights / torch.sum(weights)
                candidate_log_density = _anderson_log_density_candidate(
                    anderson_log_history,
                    anderson_residual_history,
                    weights,
                    anderson_regularization,
                    anderson_damping,
                )
                log_change = torch.clamp(
                    candidate_log_density - current_log_density,
                    min=-max_log_density_change,
                    max=max_log_density_change,
                )
                candidate_rho = torch.exp(
                    current_log_density + log_change
                ).masked_fill(~accessible_mask[:, None], 0.0)
                candidate_rho = _project_density_constraints(
                    candidate_rho,
                    fixed_N,
                    voxel_volume,
                    maximum_density,
                    accessible_mask,
                )

                (
                    candidate_evaluation,
                    candidate_max_residual,
                    candidate_rms_residual,
                ) = evaluate_trial(candidate_rho)
                n_evaluations += 1
                candidate_objective = _thermodynamic_objective(
                    candidate_evaluation,
                    fixed_N is not None,
                    voxel_volume,
                    solver.model.ideal,
                    thermal_wavelength,
                )
                accept_anderson = (
                    torch.all(torch.isfinite(candidate_rho)).item()
                    and torch.isfinite(candidate_objective).item()
                    and candidate_max_residual <= trial_max_residual
                    and candidate_rms_residual <= trial_rms_residual
                )
            except (RuntimeError, ValueError):
                accept_anderson = False

            if accept_anderson:
                next_rho = candidate_rho
                trial_evaluation = candidate_evaluation
                trial_rms_residual = candidate_rms_residual
                anderson_accepted += 1
            else:
                anderson_rejected += 1
                anderson_resets += 1
                anderson_log_history = anderson_log_history[-1:]
                anderson_residual_history = anderson_residual_history[-1:]

        relative_change = _maximum_relative_change(rho, next_rho)
        rho = next_rho.detach()
        evaluation = trial_evaluation
        n_iter += 1
        final_relative_change = relative_change
        objective = _thermodynamic_objective(
            evaluation,
            fixed_N is not None,
            voxel_volume,
            solver.model.ideal,
            thermal_wavelength,
        )
        objective_history.append(objective.detach().item())

        if adaptive_mixing:
            current_mixing = min(
                maximum_mixing,
                trial_mixing * mixing_growth,
            )
        else:
            current_mixing = mixing
        if relative_change <= tolerance_change:
            stalled = n_iter < max_iter
            break

    return {
        "rho": rho,
        "n_iter": n_iter,
        "n_evaluations": n_evaluations,
        "objective_history": objective_history,
        "final_relative_density_change": final_relative_change,
        "line_search_failures": 0,
        "stalled": stalled,
        "mixing_backtracks": mixing_backtracks,
        "final_mixing": current_mixing,
        "solver_anderson": anderson,
        "anderson_attempts": anderson_attempts,
        "anderson_accepted": anderson_accepted,
        "anderson_rejected": anderson_rejected,
        "anderson_resets": anderson_resets,
    }
