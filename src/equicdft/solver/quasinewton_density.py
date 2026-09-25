"""Projected L-BFGS minimization of the density-only thermodynamic objective.

The scalar counterpart of :mod:`equicdft.solver.quasinewton`. The objective
is the canonical ``beta A`` (fixed particle numbers) or the grand potential
``beta Omega`` (fixed chemical potentials) as a function of ``l = log rho``
on the accessible voxels. The mirror-descent step of
:func:`equicdft.solver.minimize.minimize_density`, the gradient divided by
the ideal Hessian ``V rho``, is the initial inverse Hessian ``H0`` of the
two-loop recursion; the stored curvature pairs add what the excess
functional contributes, which is what lets one step address the
compressibility stiffness of the bulk and the soft modes of cavities at
once.

Constraints are an active set. The particle numbers are imposed by the
normalization of the free voxels, the density cap by projection after every
step, and a voxel resting on the cap with a gradient pointing outward is
frozen for the step (its gradient component is zeroed and it is excluded
from the curvature pairs). The chemical potential that decides whether a
capped voxel is released is the same bound-compatible multiplier the
reported residual uses. Homogeneous-axis projections keep the density on
the projected subspace by projecting the gradient.

Trials of the line search are evaluated without derivatives; only the
accepted point is differentiated. The acceptance is Armijo against the
largest of the last ``nonmonotone_window`` objectives.
"""

import math
from typing import Any, Dict, Optional

import torch

from .numerics import (
    _bound_chemical_potential,
    _euler_residual,
    _maximum_relative_change,
    _project_density_constraints,
    _residuals_converged,
    _stationarity_field,
    _thermodynamic_objective,
)
from .options import ConvergenceOptions, MinimizeOptions
from .symmetry import _HomogeneousDensityProjection

__all__ = ["minimize_density_lbfgs"]


def _log_floor(dtype, device):
    return torch.tensor(math.log(torch.finfo(dtype).tiny), dtype=dtype, device=device)


def minimize_density_lbfgs(
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
    """Minimize the density-only objective by projected L-BFGS.

    Same contract as :func:`equicdft.solver.minimize.minimize_density`:
    the fields arrive validated and projected onto the constraints, and the
    driver state (final density, counters, objective history) is returned.
    """

    evaluate = solver.evaluate
    max_iter = convergence.max_iter
    tolerance_residual = convergence.tolerance_residual
    tolerance_rms_residual = convergence.tolerance_rms_residual
    residual_density_threshold = convergence.residual_density_threshold or 0.0
    history_size = options.lbfgs_history
    window = options.nonmonotone_window
    armijo = options.armijo_factor
    shrink = options.line_search_factor
    initial_step = options.step_size
    minimum_step = options.minimum_step_size
    max_log_change = options.max_log_density_change
    beta = data["beta"]
    canonical = fixed_N is not None

    # Grid mask; indexing with it keeps the species axis, as in quasinewton.py.
    accessible = torch.as_tensor(accessible_mask, dtype=torch.bool, device=rho.device)
    log_floor = _log_floor(rho.dtype, rho.device)
    cap_tolerance = None
    if maximum_density is not None:
        cap_tolerance = (1.0e-6 * torch.maximum(maximum_density, torch.ones_like(maximum_density)))[None, :]

    # A voxel resting on the cap keeps its coordinate this far beyond it, so
    # that the projection re-pins it until the gradient asks for its release
    # (see quasinewton.py for why the kink at the bound needs the margin).
    margin = 2.0

    def pinned(rho_accessible):
        if maximum_density is None:
            return torch.zeros_like(rho_accessible, dtype=torch.bool)
        return rho_accessible >= (maximum_density[None, :] - cap_tolerance).expand_as(rho_accessible)

    def fields_from(log_density):
        """Feasible density for the coordinates, and the consistent ``l``."""

        full = torch.zeros_like(rho)
        full[accessible] = torch.exp(torch.maximum(log_density, log_floor))
        full = _project_density_constraints(
            full, fixed_N, voxel_volume, maximum_density, accessible_mask,
        )
        log_density = torch.maximum(torch.log(full[accessible]), log_floor)
        log_density = torch.where(pinned(full[accessible]), log_density + margin, log_density)
        return full, log_density

    def evaluate_at(density, derivatives):
        current = dict(data)
        current["rho"] = density.detach().clone()
        current["V_ext"] = V_ext
        return evaluate(current, compute_derivatives=derivatives)

    def objective_of(evaluation):
        value = _thermodynamic_objective(
            evaluation, canonical, voxel_volume, solver.model.ideal, thermal_wavelength,
        ).detach()
        return float(value)

    def gradient(evaluation, density, log_density):
        """Gradient of the objective in ``l`` on the active set.

        Returns the gradient (zero on frozen voxels), the free mask and the
        log density with capped voxels whose gradient points inward
        released onto the cap.
        """

        rho_accessible = density[accessible]
        local = _stationarity_field(evaluation, projection)[accessible]
        upper = pinned(rho_accessible)
        if canonical:
            # The multiplier is the free-voxel weighted mean of the local
            # chemical potential; without an interior voxel it comes from
            # the KKT bracket, as in the reported residual.
            weights = torch.where(upper, torch.zeros_like(rho_accessible), rho_accessible)
            weight_sum = weights.sum(dim=0)
            mean = (weights * local).sum(dim=0) / weight_sum.clamp_min(torch.finfo(rho.dtype).tiny)
            no_floor = torch.zeros_like(upper)
            mean = torch.where(weight_sum > 0, mean, _bound_chemical_potential(local, no_floor, upper))
        else:
            mean = beta * mu
        excess = local - mean[None, :]
        frozen = upper & (excess < 0)
        free = ~frozen
        released = upper & ~frozen
        log_density = torch.where(released, torch.log(rho_accessible), log_density)
        if canonical:
            weights = torch.where(free, rho_accessible, torch.zeros_like(rho_accessible))
            mean = (weights * local).sum(dim=0) / weights.sum(dim=0).clamp_min(torch.finfo(rho.dtype).tiny)
            excess = local - mean[None, :]
        g = voxel_volume * rho_accessible * excess
        g = torch.where(free, g, torch.zeros_like(g))
        return g, free, log_density

    def initial_inverse(density, g):
        """The mirror-descent step as ``H0 g``."""

        return g / (voxel_volume * density[accessible])

    density, log_density = fields_from(torch.log(torch.clamp(rho[accessible], min=torch.finfo(rho.dtype).tiny)))
    result = evaluate_at(density, True)
    n_evaluations = 1
    n_energy_evaluations = 0
    g, free, log_density = gradient(result, density, log_density)
    objective = objective_of(result)
    if not math.isfinite(objective):
        raise ValueError("the thermodynamic objective became non-finite")
    recent = [objective]
    objective_history = [objective]
    pairs_s, pairs_y = [], []
    n_iter = 0
    line_search_failures = 0
    final_relative_change = float("inf")
    status = "max_iter"
    gamma = 1.0

    def residual_of(evaluation):
        return _euler_residual(
            evaluation["rho"], _stationarity_field(evaluation, projection), beta, mu,
            residual_density_threshold, maximum_density, accessible_mask,
        )

    for iteration in range(max_iter + 1):
        _, _, max_residual, rms_residual = residual_of(result)
        if _residuals_converged(max_residual, rms_residual, tolerance_residual, tolerance_rms_residual):
            status = "converged"
            break
        if iteration == max_iter:
            status = "max_iter"
            break
        if not torch.any(g != 0):
            # Every free component is stationary; the residual test decides
            # whether the constrained point counts as converged.
            status = "stalled"
            break

        # Two-loop recursion with the mirror step (scaled by gamma) as H0.
        q = g.clone()
        alphas = []
        for s, y in zip(reversed(pairs_s), reversed(pairs_y)):
            rho_i = 1.0 / float((y * s).sum())
            alpha_i = rho_i * float((s * q).sum())
            q = q - alpha_i * y
            alphas.append((alpha_i, rho_i))
        r = gamma * initial_inverse(density, q)
        for (s, y), (alpha_i, rho_i) in zip(zip(pairs_s, pairs_y), reversed(alphas)):
            beta_i = rho_i * float((y * r).sum())
            r = r + (alpha_i - beta_i) * s
        direction = torch.where(free, -r, torch.zeros_like(r))
        slope = float((g * direction).sum())
        if slope >= 0.0:
            pairs_s, pairs_y = [], []
            gamma = 1.0
            direction = torch.where(free, -initial_inverse(density, g), torch.zeros_like(g))
            slope = float((g * direction).sum())
        largest = float(direction.abs().max()) if direction.numel() else 0.0
        largest_trial_change = initial_step * largest
        if largest_trial_change > max_log_change:
            scale = max_log_change / largest_trial_change
            direction = direction * scale
            slope *= scale

        reference = max(recent[-window:])
        roundoff = 32.0 * torch.finfo(rho.dtype).eps * max(1.0, abs(objective))
        step = initial_step
        accepted = None
        # With curvature pairs a step below 1e-4 of the quasi-Newton length
        # means the model is wrong, not that the scale is; the search gives
        # up early and the memory is rebuilt. The plain mirror step may
        # shrink all the way to minimum_step_size.
        shortest = max(minimum_step, 1.0e-4 * initial_step) if pairs_s else minimum_step
        while step >= shortest:
            trial_density, trial_log = fields_from(log_density + step * direction)
            try:
                energy = evaluate_at(trial_density, False)
                n_energy_evaluations += 1
                value = objective_of(energy)
                finite = math.isfinite(value)
            except (ValueError, FloatingPointError):
                finite = False
            if finite and value <= reference + armijo * step * slope + roundoff:
                accepted = (trial_density, trial_log, value)
                break
            step *= shrink
        if accepted is None:
            if pairs_s:
                # The curvature model misled the search: forget it and let
                # the next iteration try the plain mirror step.
                pairs_s, pairs_y = [], []
                gamma = 1.0
                line_search_failures += 1
                continue
            status = "line_search_failed"
            line_search_failures += 1
            break

        trial_density, trial_log, value = accepted
        evaluation = evaluate_at(trial_density, True)
        n_evaluations += 1
        new_g, new_free, trial_log = gradient(evaluation, trial_density, trial_log)
        mask = (free & new_free).to(g.dtype)
        s = (trial_log - log_density) * mask
        y = (new_g - g) * mask
        curvature = float((s * y).sum())
        if history_size > 0 and curvature > 1.0e-12 * float(s.norm()) * float(y.norm()):
            pairs_s.append(s)
            pairs_y.append(y)
            pairs_s, pairs_y = pairs_s[-history_size:], pairs_y[-history_size:]
            # Scale H0 by the Barzilai-Borwein estimate in the H0 metric.
            denominator = float((y * initial_inverse(trial_density, y)).sum())
            if denominator > 0.0:
                gamma = min(max(curvature / denominator, 1.0e-3), 1.0e3)

        final_relative_change = _maximum_relative_change(density, trial_density)
        result, density, log_density = evaluation, trial_density, trial_log
        g, free = new_g, new_free
        objective = value
        recent.append(objective)
        recent = recent[-window:]
        objective_history.append(objective)
        n_iter += 1

    return {
        "rho": density,
        "status": status,
        "n_iter": n_iter,
        "n_evaluations": n_evaluations,
        "n_energy_evaluations": n_energy_evaluations,
        "objective_history": objective_history,
        "final_relative_density_change": final_relative_change,
        "line_search_failures": line_search_failures,
        "lbfgs_pairs": len(pairs_s),
    }
