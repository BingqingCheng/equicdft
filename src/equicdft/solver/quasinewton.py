"""Projected L-BFGS minimization of the fixed-dipole free energy.

The objective is ``beta A`` as a function of ``l = log rho`` and the
conjugate field ``h`` with ``P = m rho L(h)``. The mirror-descent step of
:func:`equicdft.solver.dipoles._minimize_fixed_dipoles_conjugate_field`
is the gradient preconditioned by the ideal Hessian in these coordinates,
so it serves here as the initial inverse Hessian ``H0`` of the two-loop
recursion; the stored curvature pairs add what the excess functional
contributes, which is what lets one step address the compressibility and
dielectric stiffness of the bulk and the soft modes of cavities at once.

Constraints are handled as an active set. The particle number is imposed by
the normalization of the free voxels, the density bounds and the
polarization-fraction cap by projection after every step, and a coordinate
resting on a bound with a gradient pointing outward is frozen for the step
(its gradient component is zeroed and it is excluded from the curvature
pairs). The reported KKT residual of the other drivers is then exactly the
stationarity condition being driven to zero.

Trials of the line search are evaluated without derivatives; only the
accepted point is differentiated. The acceptance is Armijo against the
largest of the last ``nonmonotone_window`` objectives, which is the strict
test for a window of one.
"""

import math

import torch

from ..functional.polarization_ideal import (
    _langevin_prime,
    dipole_alignment,
    inverse_langevin,
    langevin,
)
from .numerics import (
    _bound_chemical_potential,
    _maximum_relative_change,
    _project_density_constraints,
    _project_vector_norm,
    _residuals_converged,
)

__all__ = ["minimize_fixed_dipoles_lbfgs"]



def _langevin_jacobian(field):
    """Return the radial and tangential eigenvalues of ``dL/dh`` and ``h_hat``."""

    norm = field.norm(dim=-1, keepdim=True)
    small = norm < 1.0e-3
    safe = norm.clamp_min(1.0e-3)
    radial = torch.where(small, torch.full_like(safe, 1.0 / 3.0), _langevin_prime(safe))
    tangential = torch.where(small, torch.full_like(safe, 1.0 / 3.0), langevin(safe) / safe)
    direction = field / safe
    return radial, tangential, direction


def _apply_langevin_jacobian(field, vector, inverse=False):
    """Apply ``dL/dh`` (or its inverse) to a vector field."""

    radial, tangential, direction = _langevin_jacobian(field)
    along = (vector * direction).sum(dim=-1, keepdim=True) * direction
    across = vector - along
    if inverse:
        return along / radial + across / tangential
    return radial * along + tangential * across


def _conjugate_field_from(polarization, rho, moment, accessible):
    alignment = polarization[accessible] / (rho[accessible] * moment)[..., None]
    norm = alignment.norm(dim=-1, keepdim=True)
    interior = norm.clamp_max(1.0 - 1.0e-12)
    return torch.where(
        norm > 0,
        inverse_langevin(interior) * alignment / norm.clamp_min(1.0e-300),
        torch.zeros_like(alignment),
    )


def _log_floor(dtype, device):
    return torch.tensor(math.log(torch.finfo(dtype).tiny), dtype=dtype, device=device)


def minimize_fixed_dipoles_lbfgs(
    solver,
    data,
    result,
    numbers,
    moment,
    volume,
    accessible,
    maximum_density,
    maximum_polarization_fraction,
    minimum_density,
    convergence,
    options,
    bounds,
):
    """Minimize ``beta A`` over ``(log rho, h)`` with projected L-BFGS.

    ``bounds`` applies the fixed-dipole KKT residuals to an evaluation (it
    is :func:`equicdft.solver.dipoles.apply_fixed_dipole_bounds` bound to
    this problem's constraints). Returns the driver state expected by
    :func:`equicdft.solver.reporting.finalize_result`.
    """

    potential = data["V_ext"]
    max_iter = convergence.max_iter
    tolerance_residual = convergence.tolerance_residual
    tolerance_rms_residual = convergence.tolerance_rms_residual
    history_size = options.lbfgs_history
    window = options.nonmonotone_window
    armijo = options.armijo_factor
    shrink = options.line_search_factor
    initial_step = options.step_size
    minimum_step = options.minimum_step_size
    max_log_change = options.max_log_density_change

    maximum_field = (
        None
        if maximum_polarization_fraction is None
        else inverse_langevin(maximum_polarization_fraction)[None, :, None]
    )
    interior = 1.0 - 64.0 * torch.finfo(moment.dtype).eps
    field_limit = inverse_langevin(moment.new_tensor(interior)).expand_as(moment)[None, :, None]
    field_bound = field_limit if maximum_field is None else maximum_field
    log_floor = _log_floor(potential.dtype, potential.device)
    floor_tolerance = None
    if minimum_density is not None:
        floor_tolerance = 1.0e-6 * torch.maximum(minimum_density, torch.ones_like(minimum_density))
    cap_tolerance = None
    if maximum_density is not None:
        cap_tolerance = (1.0e-6 * torch.maximum(maximum_density, torch.ones_like(maximum_density)))[None, :]

    # A voxel resting on a bound keeps its coordinate this far beyond the
    # bound. At the bound itself the projected mapping has a kink that every
    # change of the normalization would cross, and the gradient of the
    # smooth branch would then mispredict the objective; with the margin a
    # pinned voxel stays pinned until the gradient asks for its release.
    margin = 2.0

    def pinned_sets(rho):
        rho_accessible = rho[accessible]
        lower = torch.zeros_like(rho_accessible, dtype=torch.bool)
        upper = torch.zeros_like(rho_accessible, dtype=torch.bool)
        if minimum_density is not None:
            lower = rho_accessible <= minimum_density[accessible] + floor_tolerance[accessible]
        if maximum_density is not None:
            upper = rho_accessible >= (maximum_density[None, :] - cap_tolerance).expand_as(rho_accessible)
        return lower, upper

    def fields_from(log_density, field):
        """Feasible ``(rho, P)`` for the coordinates, and the consistent ``l``."""

        rho = torch.zeros_like(potential)
        rho[accessible] = torch.exp(torch.maximum(log_density, log_floor))
        rho = _project_density_constraints(
            rho, numbers, volume, maximum_density, accessible,
            minimum_density=minimum_density, proportional=True,
        )
        field = _project_vector_norm(field, field_bound)
        polarization = potential.new_zeros(*potential.shape, 3)
        polarization[accessible] = (rho[accessible] * moment)[..., None] * dipole_alignment(field)
        log_density = torch.maximum(torch.log(rho[accessible]), log_floor)
        lower, upper = pinned_sets(rho)
        log_density = torch.where(lower, log_density - margin, log_density)
        log_density = torch.where(upper, log_density + margin, log_density)
        return rho, polarization, log_density, field

    def gradient(evaluation, rho, field, log_density):
        """Gradient of ``beta A`` in ``(l, h)`` on the active set.

        Returns the gradient, the free mask and the log density with pinned
        voxels whose gradient points inward released onto their bound.
        """

        p_residual = evaluation["polarization_residual"][accessible]          # d beta A / dP
        rho_accessible = rho[accessible]
        alignment = dipole_alignment(field)
        # d beta A / d l at fixed h: chain rule through P = m rho L(h), then
        # the normalization removes the free-voxel weighted mean.
        local = evaluation["beta_mu_local"][accessible] + (
            moment[..., None] * alignment * p_residual
        ).sum(dim=-1)
        lower, upper = pinned_sets(rho)
        # The sign that decides whether a pinned voxel stays pinned is that
        # of the gradient, local minus the chemical potential; the multiplier
        # itself depends on the free set, so it is settled in two passes.
        tiny = torch.finfo(rho.dtype).tiny
        weights = torch.where(lower | upper, torch.zeros_like(rho_accessible), rho_accessible)
        weight_sum = weights.sum(dim=0)
        mean = (weights * local).sum(dim=0) / weight_sum.clamp_min(tiny)
        # Use the same all-bound KKT multiplier as the reported residual.
        mean = torch.where(
            weight_sum > 0, mean, _bound_chemical_potential(local, lower, upper),
        )
        excess = local - mean[None, :]
        frozen = (lower & (excess > 0)) | (upper & (excess < 0))
        free = ~frozen
        released = (lower | upper) & ~frozen
        log_density = torch.where(released, torch.log(rho_accessible), log_density)
        weights = torch.where(free, rho_accessible, torch.zeros_like(rho_accessible))
        mean = (weights * local).sum(dim=0) / weights.sum(dim=0).clamp_min(tiny)
        g_l = volume * rho_accessible * (local - mean[None, :])
        g_l = torch.where(free, g_l, torch.zeros_like(g_l))
        # d beta A / d h = V m rho (dL/dh)^T d beta A / dP.
        g_h = volume * moment[..., None] * rho_accessible[..., None] * _apply_langevin_jacobian(
            field, p_residual,
        )
        if maximum_field is not None:
            norm = field.norm(dim=-1, keepdim=True)
            at_cap = norm >= maximum_field * (1.0 - 1.0e-9)
            outward = (g_h * field).sum(dim=-1, keepdim=True) < 0     # descent would increase |h|
            radial = (g_h * field).sum(dim=-1, keepdim=True) * field / norm.clamp_min(1e-300).square()
            g_h = torch.where(at_cap & outward, g_h - radial, g_h)
        return g_l, g_h, free, log_density

    def initial_inverse(rho, field, g_l, g_h):
        """The mirror-descent step as ``H0 g``."""

        # g_h = V m rho (dL/dh)^T dA/dP, so (dL/dh)^-1 g_h / (V rho) is the
        # scaled polarization residual m dA/dP, the conjugate-field mirror step.
        d_l = g_l / (volume * rho[accessible])
        d_h = _apply_langevin_jacobian(field, g_h, inverse=True) / (
            volume * rho[accessible][..., None]
        )
        return d_l, d_h

    def pack(a, b):
        return torch.cat([a.reshape(-1), b.reshape(-1)])

    rho = result["rho"]
    polarization = result["dipole_density"]
    field = _project_vector_norm(_conjugate_field_from(polarization, rho, moment, accessible), field_bound)
    rho, polarization, log_density, field = fields_from(torch.log(rho[accessible]), field)
    n_log = log_density.numel()

    def unpack(vector):
        return vector[:n_log].reshape(log_density.shape), vector[n_log:].reshape(field.shape)

    g_l, g_h, free, log_density = gradient(result, rho, field, log_density)
    objective = float(result["beta_A"])
    recent = [objective]
    pairs_s, pairs_y = [], []
    history = []
    n_evaluations = 1
    n_energy_evaluations = 0
    n_iter = 0
    line_search_failures = 0
    density_change = polarization_change = field_change = float("inf")
    status = "max_iter"
    converged = False
    gamma = 1.0
    for iteration in range(max_iter + 1):
        history.append({
            "iteration": iteration,
            "beta_A": objective,
            "maximum_residual": float(result["maximum_residual"]),
        })
        converged = _residuals_converged(
            result["maximum_residual"], result["rms_residual"],
            tolerance_residual, tolerance_rms_residual,
        )
        if converged or iteration == max_iter:
            status = "converged" if converged else "max_iter"
            break

        g = pack(g_l, g_h)
        if not torch.any(g != 0):
            # Every free component is stationary; the residual test above
            # decides whether the constrained point counts as converged.
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
        q_l, q_h = unpack(q)
        r_l, r_h = initial_inverse(rho, field, q_l, q_h)
        r = gamma * pack(r_l, r_h)
        for (s, y), (alpha_i, rho_i) in zip(zip(pairs_s, pairs_y), reversed(alphas)):
            beta_i = rho_i * float((y * r).sum())
            r = r + (alpha_i - beta_i) * s
        direction = -r
        # Frozen coordinates take no part in the step even if the stored
        # pairs would move them.
        d_l, d_h = unpack(direction)
        d_l = torch.where(free, d_l, torch.zeros_like(d_l))
        direction = pack(d_l, d_h)
        slope = float((g * direction).sum())
        if slope >= 0.0:
            pairs_s, pairs_y = [], []
            gamma = 1.0
            d_l, d_h = initial_inverse(rho, field, g_l, g_h)
            direction = -pack(d_l, d_h)
            slope = float((g * direction).sum())
        d_l, d_h = unpack(direction)
        largest = float(d_l.abs().max()) if d_l.numel() else 0.0
        largest_trial_change = initial_step * largest
        if largest_trial_change > max_log_change:
            scale = max_log_change / largest_trial_change
            direction = direction * scale
            slope *= scale

        reference = max(recent[-window:])
        roundoff = 32.0 * torch.finfo(rho.dtype).eps * max(1.0, abs(objective))
        # ``step_size`` scales the direction; 1 is the quasi-Newton length.
        step = initial_step
        accepted = None
        x = pack(log_density, field)
        # With curvature pairs a step below 1e-4 of the quasi-Newton length
        # means the model is wrong, not that the scale is; the search gives
        # up early and the memory is rebuilt. The plain mirror step may
        # shrink all the way to minimum_step_size.
        shortest = max(minimum_step, 1.0e-4 * initial_step) if pairs_s else minimum_step
        while step >= shortest:
            trial_l, trial_h = unpack(x + step * direction)
            trial_rho, trial_p, trial_l, trial_h = fields_from(trial_l, trial_h)
            try:
                energy = solver.evaluate(
                    dict(data, rho=trial_rho, dipole_density=trial_p), compute_derivatives=False,
                )
                n_energy_evaluations += 1
                value = float(energy["beta_A"])
                finite = math.isfinite(value)
            except (ValueError, FloatingPointError):
                finite = False
            if finite and value <= reference + armijo * step * slope + roundoff:
                accepted = (trial_rho, trial_p, trial_l, trial_h, value)
                break
            step *= shrink
        if accepted is None:
            if pairs_s:
                # The curvature model misled the search: forget it and let
                # the next iteration try the plain mirror step.
                pairs_s, pairs_y = [], []
                gamma = 1.0
                line_search_failures += 1
                history[-1]["accepted_step"] = 0.0
                continue
            status = "line_search_failed"
            line_search_failures += 1
            break

        trial_rho, trial_p, trial_l, trial_h, value = accepted
        evaluation = bounds(solver.evaluate(dict(data, rho=trial_rho, dipole_density=trial_p)))
        n_evaluations += 1
        new_g_l, new_g_h, new_free, trial_l = gradient(evaluation, trial_rho, trial_h, trial_l)
        s = pack(trial_l, trial_h) - x
        y = pack(new_g_l, new_g_h) - g
        # A voxel that was or became active contributes no curvature.
        mask_l = (free & new_free).to(s.dtype)
        s_l, s_h = unpack(s)
        y_l, y_h = unpack(y)
        s = pack(s_l * mask_l, s_h)
        y = pack(y_l * mask_l, y_h)
        curvature = float((s * y).sum())
        if history_size > 0 and curvature > 1.0e-12 * float(s.norm()) * float(y.norm()):
            pairs_s.append(s)
            pairs_y.append(y)
            pairs_s, pairs_y = pairs_s[-history_size:], pairs_y[-history_size:]
            # Scale H0 by the Barzilai-Borwein estimate in the H0 metric.
            h0_y_l, h0_y_h = initial_inverse(trial_rho, trial_h, y_l * mask_l, y_h)
            denominator = float((y * pack(h0_y_l, h0_y_h)).sum())
            if denominator > 0.0:
                gamma = min(max(curvature / denominator, 1.0e-3), 1.0e3)

        density_change = _maximum_relative_change(rho, trial_rho)
        polarization_change = float((
            (trial_p - polarization).norm(dim=-1)
            / torch.clamp(trial_rho * moment, min=1.0e-12)
        )[accessible].max())
        field_change = float((trial_h - field).norm(dim=-1).max())
        history[-1]["accepted_step"] = step
        result, rho, polarization = evaluation, trial_rho, trial_p
        log_density, field = trial_l, trial_h
        g_l, g_h, free = new_g_l, new_g_h, new_free
        objective = float(result["beta_A"])
        recent.append(objective)
        recent = recent[-window:]
        n_iter += 1

    return {
        "result": result,
        "converged": converged,
        "status": status,
        "n_iter": n_iter,
        "n_evaluations": n_evaluations,
        "n_energy_evaluations": n_energy_evaluations,
        "history": history,
        "line_search_failures": line_search_failures,
        "mixing_backtracks": 0,
        "final_mixing": None,
        "lbfgs_pairs": len(pairs_s),
        "final_relative_density_change": density_change,
        "final_relative_polarization_change": polarization_change,
        "final_conjugate_field_change": field_change,
    }
