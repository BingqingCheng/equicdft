"""Coupled Euler iteration in conjugate-field coordinates.

The fixed-dipole Euler equations are iterated on ``l = log rho`` and the
conjugate field ``h`` with ``P = m rho L(h)``, where ``L`` is the Langevin
function. Both coordinates are unconstrained, so any linear combination of
iterates (a mixed step, an Anderson extrapolation) is a positive density
with an interior polarization, and no margin or clipping onto the
saturation boundary is ever needed. Particle numbers and the density bounds
are imposed by projecting the density iterate.

Anderson acceleration combines the recent iterates with a sliding window.
It is safeguarded twice: an accelerated step after which the fixed-point
residual has grown by more than ``AndersonOptions.safeguard`` discards the
history, and every step, accelerated or mixed, must not raise the free
energy ``beta A``. The second test matters near phase coexistence, where a
fixed point reached by extrapolation can be the unstable uniform state; the
mixed step is a descent step of the free energy for small enough mixing, so
a mixed step that raises it shrinks the mixing and is retried, and an
accepted step lets the mixing grow again within the configured schedule.
"""

import math

import torch

from ..functional.polarization_ideal import dipole_alignment, inverse_langevin
from .numerics import (
    _maximum_relative_change,
    _orientational_log_weight,
    _project_density_constraints,
    _project_vector_norm,
    _residuals_converged,
)

__all__ = ["solve_fixed_dipole_conjugate"]


def _conjugate_field_from(polarization, rho, moment, accessible):
    """Return ``h = L^-1(P / (m rho))`` on the accessible voxels."""

    alignment = polarization[accessible] / (rho[accessible] * moment)[..., None]
    norm = alignment.norm(dim=-1, keepdim=True)
    interior = norm.clamp_max(1.0 - 1.0e-12)
    field = torch.where(
        norm > 0,
        inverse_langevin(interior) * alignment / norm.clamp_min(1.0e-300),
        torch.zeros_like(alignment),
    )
    return field


def _log_floor(numbers, volume, accessible):
    """Smallest log density the coordinate carries.

    A Boltzmann target that underflows to zero would otherwise put ``-inf``
    into the mixing and Anderson vectors. Use the dtype's smallest normal
    density without an extra log-space margin, which would suppress valid
    dilute densities in float32. ``Constraints.minimum_density`` is the
    separate physical bound and carries a residual.
    """

    return torch.as_tensor(
        math.log(torch.finfo(numbers.dtype).tiny),
        dtype=numbers.dtype, device=numbers.device,
    )


def _field_limit(moment):
    """Largest conjugate field whose alignment is still strictly interior."""

    interior = 1.0 - 64.0 * torch.finfo(moment.dtype).eps
    return inverse_langevin(moment.new_tensor(interior)).expand_as(moment)[None, :, None]


def _fields_from(log_density, field, potential, moment, numbers, volume,
                 accessible, maximum_density, minimum_density):
    """Map coordinates to a feasible ``(rho, P)`` and the consistent ``l``."""

    floor = _log_floor(numbers, volume, accessible)
    rho = torch.zeros_like(potential)
    rho[accessible] = torch.exp(torch.maximum(log_density, floor))
    rho = _project_density_constraints(
        rho, numbers, volume, maximum_density, accessible,
        minimum_density=minimum_density, proportional=True,
    )
    polarization = potential.new_zeros(*potential.shape, 3)
    polarization[accessible] = (rho[accessible] * moment)[..., None] * dipole_alignment(field)
    return rho, polarization, torch.maximum(torch.log(rho[accessible]), floor)


def _euler_targets(result, data, beta, moment, numbers, volume, accessible,
                   maximum_density, minimum_density, maximum_field):
    """Return the Euler targets ``(l_t, h_t)`` on the accessible voxels."""

    unconstrained_field = moment[..., None] * (
        beta * data["E_ext"][accessible]
        - result["polarization_derivative_excess"][accessible]
    )
    target_field = _project_vector_norm(unconstrained_field, maximum_field)
    logits = (
        -beta * data["V_ext"][accessible]
        + result["c1"][accessible]
        + _orientational_log_weight(unconstrained_field, target_field)
    )
    target_rho = torch.zeros_like(data["V_ext"])
    target_rho[accessible] = torch.exp(
        torch.log_softmax(logits, dim=0) + torch.log(numbers / volume)
    )
    target_rho = _project_density_constraints(
        target_rho, numbers, volume, maximum_density, accessible,
        minimum_density=minimum_density, proportional=True,
    )
    floor = _log_floor(numbers, volume, accessible)
    return torch.maximum(torch.log(target_rho[accessible]), floor), target_field


def _anderson_step(iterates, residuals, damping, regularization):
    """Type-II Anderson combination of the stored iterates and residuals."""

    x = iterates[-1]
    r = residuals[-1]
    if len(iterates) < 2:
        return x + damping * r
    dX = torch.stack([iterates[i + 1] - iterates[i] for i in range(len(iterates) - 1)], dim=1)
    dR = torch.stack([residuals[i + 1] - residuals[i] for i in range(len(residuals) - 1)], dim=1)
    gram = dR.T @ dR
    scale = torch.clamp(torch.trace(gram) / gram.shape[0], min=torch.finfo(gram.dtype).tiny)
    system = gram + regularization * scale * torch.eye(
        gram.shape[0], dtype=gram.dtype, device=gram.device,
    )
    gamma = torch.linalg.solve(system, dR.T @ r)
    if not torch.isfinite(gamma).all():
        raise RuntimeError("Anderson coefficient solve is singular")
    return x + damping * r - (dX + damping * dR) @ gamma


def solve_fixed_dipole_conjugate(
    solver,
    data,
    result,
    numbers,
    moment,
    beta,
    volume,
    accessible,
    maximum_density,
    maximum_polarization_fraction,
    minimum_density,
    convergence,
    options,
):
    """Iterate the coupled Euler map in ``(log rho, h)`` coordinates."""

    potential = data["V_ext"]
    max_iter = convergence.max_iter
    tolerance_residual = convergence.tolerance_residual
    tolerance_rms_residual = convergence.tolerance_rms_residual
    mixing = options.mixing
    adaptive_mixing = options.adaptive_mixing
    minimum_mixing = options.minimum_mixing if adaptive_mixing else mixing
    maximum_mixing = options.maximum_mixing if adaptive_mixing else mixing
    mixing_growth = options.mixing_growth
    backtrack_factor = options.mixing_backtrack_factor
    anderson = options.anderson
    maximum_field = (
        None
        if maximum_polarization_fraction is None
        else inverse_langevin(maximum_polarization_fraction)[None, :, None]
    )
    # The iterate's alignment is bounded by the physical cap when there is
    # one and otherwise only by representability of the inverse Langevin.
    field_bound = _field_limit(moment) if maximum_field is None else maximum_field

    # dipoles imports this driver, so its bound helper is imported here.
    from .dipoles import apply_fixed_dipole_bounds

    def bounds(evaluation):
        return apply_fixed_dipole_bounds(
            evaluation, maximum_density, maximum_polarization_fraction, moment,
            accessible, minimum_density=minimum_density,
        )

    rho = result["rho"]
    polarization = result["dipole_density"]
    log_density = torch.maximum(
        torch.log(rho[accessible]), _log_floor(numbers, volume, accessible),
    )
    field = _conjugate_field_from(polarization, rho, moment, accessible)
    field = _project_vector_norm(field, field_bound)
    n_evaluations = 1
    n_log = log_density.numel()

    def pack(log_density, field):
        return torch.cat([log_density.reshape(-1), field.reshape(-1)])

    def unpack(vector):
        return (
            vector[:n_log].reshape(log_density.shape),
            vector[n_log:].reshape(field.shape),
        )

    def evaluate_at(vector):
        candidate_l, candidate_h = unpack(vector)
        candidate_h = _project_vector_norm(candidate_h, field_bound)
        rho, polarization, candidate_l = _fields_from(
            candidate_l, candidate_h, potential, moment, numbers, volume, accessible,
            maximum_density, minimum_density,
        )
        evaluation = bounds(solver.evaluate(dict(data, rho=rho, dipole_density=polarization)))
        if not torch.isfinite(evaluation["maximum_residual"]):
            raise FloatingPointError("nonfinite fixed-dipole residual")
        return evaluation, rho, polarization, candidate_l, candidate_h

    def _clip_step(step):
        step_l, step_h = unpack(step)
        step_l = step_l.clamp(-options.max_log_density_change, options.max_log_density_change)
        step_h = _project_vector_norm(
            step_h,
            step_h.new_tensor(options.max_conjugate_field_change).expand(1, moment.numel(), 1),
        )
        return pack(step_l, step_h)

    def _try(vector):
        try:
            return evaluate_at(vector)
        except (ValueError, FloatingPointError):
            return None

    energy_slack = 1.0e-9 * max(1.0, abs(float(result["beta_A"])))
    current_mixing = mixing
    mixing_backtracks = 0
    line_search_failures = 0
    history = []
    iterates = []
    residuals = []
    previous_norm = float("inf")
    n_iter = 0
    anderson_attempts = anderson_accepted = anderson_rejected = anderson_resets = 0
    density_change = polarization_change = field_change = float("inf")
    status = "max_iter"
    converged = False
    for iteration in range(max_iter + 1):
        history.append({
            "iteration": iteration,
            "beta_A": float(result["beta_A"]),
            "maximum_residual": float(result["maximum_residual"]),
        })
        converged = _residuals_converged(
            result["maximum_residual"], result["rms_residual"],
            tolerance_residual, tolerance_rms_residual,
        )
        if converged or iteration == max_iter:
            status = "converged" if converged else "max_iter"
            break

        target_l, target_h = _euler_targets(
            result, data, beta, moment, numbers, volume, accessible,
            maximum_density, minimum_density, maximum_field,
        )
        residual_l = target_l - log_density
        residual_h = target_h - field
        x = pack(log_density, field)
        r = pack(residual_l, residual_h)
        residual_norm = float(r.norm()) / math.sqrt(r.numel())

        accelerated = False
        if anderson is not None:
            if iterates and residual_norm > anderson.safeguard * previous_norm:
                # The accelerated step made the fixed-point residual worse:
                # forget the history and fall back to the mixed step.
                iterates, residuals = [], []
                anderson_rejected += 1
                anderson_resets += 1
            iterates.append(x)
            residuals.append(r)
            iterates = iterates[-(anderson.history + 1):]
            residuals = residuals[-(anderson.history + 1):]
            if len(iterates) >= 2:
                anderson_attempts += 1
                try:
                    proposal = _anderson_step(
                        iterates, residuals, current_mixing * anderson.damping,
                        anderson.regularization,
                    )
                    accelerated = True
                except RuntimeError:
                    iterates, residuals = [x], [r]
                    anderson_resets += 1
        previous_norm = residual_norm

        # Every step must not raise the free energy. The mixed step is a
        # descent step for small enough mixing, so a failing one shrinks the
        # mixing and is retried; an accelerated step that fails is discarded
        # (a fixed point reached by extrapolation may be a saddle, such as
        # the uniform state inside a coexistence region) and its history
        # forgotten. An accepted step lets the mixing grow again.
        energy = float(result["beta_A"])
        trial = None
        if accelerated:
            candidate = _try(x + _clip_step(proposal - x))
            n_evaluations += 1
            if candidate is not None and float(candidate[0]["beta_A"]) <= energy + energy_slack:
                trial = candidate
                anderson_accepted += 1
            else:
                iterates, residuals = [x], [r]
                anderson_rejected += 1
                anderson_resets += 1
        while trial is None:
            candidate = _try(x + _clip_step(current_mixing * r))
            n_evaluations += 1
            if candidate is not None and float(candidate[0]["beta_A"]) <= energy + energy_slack:
                trial = candidate
                break
            if current_mixing <= minimum_mixing:
                # No admissible mixing lowers the free energy: keep the
                # current state and report it, as the minimizer does.
                break
            current_mixing = max(minimum_mixing, current_mixing * backtrack_factor)
            mixing_backtracks += 1
        if trial is None:
            status = "line_search_failed"
            line_search_failures = 1
            break
        trial, rho_trial, p_trial, log_trial, field_trial = trial
        if adaptive_mixing:
            current_mixing = min(maximum_mixing, current_mixing * mixing_growth)

        density_change = _maximum_relative_change(rho, rho_trial)
        polarization_change = float((
            (p_trial - polarization).norm(dim=-1)
            / torch.clamp(rho_trial * moment, min=1.0e-12)
        )[accessible].max())
        field_change = float((field_trial - field).norm(dim=-1).max())
        history[-1]["mixing"] = current_mixing
        result, rho, polarization = trial, rho_trial, p_trial
        log_density, field = log_trial, field_trial
        n_iter += 1
        if max(density_change, field_change) <= options.tolerance_change:
            converged = _residuals_converged(
                result["maximum_residual"], result["rms_residual"],
                tolerance_residual, tolerance_rms_residual,
            )
            if converged:
                status = "converged"
            else:
                status = "max_iter" if n_iter >= max_iter else "stalled"
            history.append({
                "iteration": n_iter,
                "beta_A": float(result["beta_A"]),
                "maximum_residual": float(result["maximum_residual"]),
            })
            break

    return {
        "result": result,
        "converged": converged,
        "status": status,
        "n_iter": n_iter,
        "n_evaluations": n_evaluations,
        "history": history,
        "line_search_failures": line_search_failures,
        "mixing_backtracks": mixing_backtracks,
        "final_mixing": current_mixing,
        "solver_anderson": anderson is not None,
        "anderson_attempts": anderson_attempts,
        "anderson_accepted": anderson_accepted,
        "anderson_rejected": anderson_rejected,
        "anderson_resets": anderson_resets,
        "final_relative_density_change": density_change,
        "final_relative_polarization_change": polarization_change,
        "final_conjugate_field_change": field_change,
    }
