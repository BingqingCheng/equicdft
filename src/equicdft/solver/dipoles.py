"""Fixed-dipole (coupled density and polarization) numerical update drivers.

These drivers mirror :mod:`equicdft.solver.euler` and
:mod:`equicdft.solver.minimize` for the coupled ``(rho, P)`` problem of a
fixed-magnitude dipole fluid. They keep their own update maps, acceptance
tests and residual bookkeeping: the orientational mirror map and the
coupled KKT projections are not the scalar ones, so the two families share
the option groups, the constraint helpers and the result conventions
rather than one loop.
"""

import torch

from .._species import species_vector
from ..functional.polarization_ideal import (
    dipole_alignment,
    inverse_langevin,
    log_sinhc,
)
from .numerics import (
    _canonical_density_residual,
    _fixed_dipole_bound_kkt,
    _maximum_relative_change,
    _orientational_log_weight,
    _polarization_residual_norms,
    _project_density_constraints,
    _project_vector_norm,
    _residuals_converged,
)

from .conjugate import solve_fixed_dipole_conjugate
from .quasinewton import minimize_fixed_dipoles_lbfgs
from .evaluation import fixed_dipole_inputs, _set_density_residuals
from .reporting import finalize_result

__all__ = ["solve_fixed_dipoles"]


def apply_fixed_dipole_bounds(
    result,
    maximum_density,
    maximum_polarization_fraction,
    moment,
    accessible,
    fixed_field=None,
    minimum_density=None,
):
    """Apply canonical density and polarization-bound KKT residuals."""

    if (
        maximum_density is None
        and minimum_density is None
        and maximum_polarization_fraction is None
        and fixed_field is None
    ):
        return result
    if fixed_field == "dipole_density":
        residual, chemical_potential, maximum, rms = (
            _canonical_density_residual(
                result["rho"],
                result["beta_mu_local"],
                0.0,
                maximum_density,
                accessible,
                minimum_density=minimum_density,
            )
        )
        _set_density_residuals(result, residual, chemical_potential, maximum, rms)
        result["full_maximum_residual"] = torch.maximum(
            result["max_polarization_residual"],
            result["max_polarization_residual"].new_tensor(maximum),
        )
        result["full_rms_residual"] = torch.maximum(
            result["rms_polarization_residual"],
            result["rms_polarization_residual"].new_tensor(rms),
        )
        result["maximum_residual"] = result["rho"].new_tensor(maximum)
        result["rms_residual"] = result["rho"].new_tensor(rms)
        return result
    unconstrained_polarization = result[
        "scaled_polarization_residual"
    ]
    effective_mu, constrained_polarization, multiplier = (
        _fixed_dipole_bound_kkt(
            result["rho"],
            result["dipole_density"],
            result["beta_mu_local"],
            unconstrained_polarization,
            moment,
            maximum_polarization_fraction,
            accessible,
        )
    )

    residual, chemical_potential, maximum, rms = (
        _canonical_density_residual(
            result["rho"],
            effective_mu,
            0.0,
            maximum_density,
            accessible,
            minimum_density=minimum_density,
        )
    )
    _set_density_residuals(result, residual, chemical_potential, maximum, rms)

    max_polarization, rms_polarization = _polarization_residual_norms(
        result["rho"], constrained_polarization, accessible
    )
    result["unconstrained_scaled_polarization_residual"] = (
        unconstrained_polarization
    )
    result["scaled_polarization_residual"] = constrained_polarization
    result["polarization_kkt_residual"] = (
        constrained_polarization / moment[..., None]
    )
    result["polarization_bound_multiplier"] = multiplier
    result["max_polarization_residual"] = max_polarization
    result["rms_polarization_residual"] = rms_polarization
    full_maximum = torch.maximum(
        max_polarization.new_tensor(maximum),
        max_polarization,
    )
    full_rms = torch.maximum(
        rms_polarization.new_tensor(rms),
        rms_polarization,
    )
    result["full_maximum_residual"] = full_maximum
    result["full_rms_residual"] = full_rms
    if fixed_field == "rho":
        result["maximum_residual"] = max_polarization
        result["rms_residual"] = rms_polarization
    else:
        result["maximum_residual"] = full_maximum
        result["rms_residual"] = full_rms
    return result


@torch.enable_grad()
def solve_fixed_dipoles(solver, data, method, options):
    """Solve fixed-dipole Euler equations at fixed particle numbers."""

    convergence = options.convergence
    euler = options.euler
    minimize = options.minimize
    constraints = options.constraints
    initial = options.initial
    particle_numbers = constraints.particle_numbers
    if particle_numbers is None:
        raise ValueError(
            "fixed-dipole solving currently requires particle_numbers"
        )
    if (
        method == "euler"
        and euler.anderson is not None
        and euler.fixed_dipole_coordinate != "conjugate_field"
    ):
        raise ValueError(
            "Anderson acceleration with fixed dipoles requires "
            "EulerOptions(fixed_dipole_coordinate='conjugate_field')"
        )
    if convergence.residual_density_threshold not in (None, 0, 0.0):
        raise ValueError(
            "residual_density_threshold is not supported with fixed dipoles"
        )
    fixed_field = constraints.fixed_field
    if fixed_field is not None and method != "minimize":
        raise ValueError("fixed_field is supported only with method='minimize'")
    initial_rho = initial.rho
    initial_polarization = initial.polarization
    beta_multiplier = initial.beta_multiplier
    max_iter = convergence.max_iter
    tolerance_residual = convergence.tolerance_residual
    tolerance_rms_residual = convergence.tolerance_rms_residual
    maximum_density = constraints.maximum_density
    maximum_polarization_fraction = constraints.maximum_polarization_fraction
    step_size = minimize.step_size
    minimum_step_size = minimize.minimum_step_size
    line_search_factor = minimize.line_search_factor
    armijo_factor = minimize.armijo_factor
    minimize_coordinate = minimize.fixed_dipole_coordinate
    tolerance_change = euler.tolerance_change
    mixing = euler.mixing
    adaptive_mixing = euler.adaptive_mixing
    minimum_mixing = euler.minimum_mixing
    maximum_mixing = euler.maximum_mixing
    mixing_growth = euler.mixing_growth
    mixing_backtrack_factor = euler.mixing_backtrack_factor
    euler_coordinate = euler.fixed_dipole_coordinate

    data = solver._move_data(data)
    beta, volume, moment, wavelength, excluded, accessible = (
        fixed_dipole_inputs(solver, data)
    )
    potential = data["V_ext"]
    numbers = species_vector(
        particle_numbers, potential.shape[-1], potential, "particle_numbers",
        positive=True,
    )
    density_cap = None
    if maximum_density is not None:
        density_cap = species_vector(
            maximum_density,
            potential.shape[-1],
            potential,
            "maximum_density",
        )
        if (
            not torch.all(torch.isfinite(density_cap)).item()
            or torch.any(density_cap <= 0.0).item()
        ):
            raise ValueError(
                "maximum_density values must be finite and positive"
            )
        maximum_numbers = volume * accessible.sum() * density_cap
        if torch.any(numbers > maximum_numbers).item():
            raise ValueError(
                "particle_numbers are infeasible under maximum_density"
            )
    density_floor = None
    if constraints.minimum_density is not None:
        density_floor = species_vector(
            constraints.minimum_density,
            potential.shape[-1],
            potential,
            "minimum_density",
        )
        if (
            not torch.all(torch.isfinite(density_floor)).item()
            or torch.any(density_floor <= 0.0).item()
        ):
            raise ValueError("minimum_density values must be finite and positive")
        if density_cap is not None and torch.any(
            density_floor >= density_cap
        ).item():
            raise ValueError("minimum_density must lie below maximum_density")
        minimum_numbers = volume * accessible.sum() * density_floor
        if torch.any(numbers <= minimum_numbers).item():
            raise ValueError(
                "particle_numbers are infeasible under minimum_density"
            )
    polarization_fraction_cap = None
    if maximum_polarization_fraction is not None:
        polarization_fraction_cap = species_vector(
            maximum_polarization_fraction,
            potential.shape[-1],
            potential,
            "maximum_polarization_fraction",
        )
        if (
            not torch.all(torch.isfinite(polarization_fraction_cap)).item()
            or torch.any(polarization_fraction_cap <= 0.0).item()
            or torch.any(polarization_fraction_cap >= 1.0).item()
        ):
            raise ValueError(
                "maximum_polarization_fraction values must be finite "
                "and lie in (0, 1)"
            )
    rho0 = initial_rho if initial_rho is not None else data.get("rho")
    p0 = (
        initial_polarization
        if initial_polarization is not None
        else data.get("dipole_density")
    )
    if fixed_field == "rho" and rho0 is None:
        raise ValueError("fixed_field='rho' requires an initial density")
    if fixed_field == "dipole_density" and p0 is None:
        raise ValueError(
            "fixed_field='dipole_density' requires an initial polarization"
        )
    if rho0 is None:
        initial_alignment = (
            beta_multiplier
            * beta
            * moment[..., None]
            * data["E_ext"][accessible]
        )
        logits = (
            -beta_multiplier * beta * potential[accessible]
            + log_sinhc(initial_alignment.norm(dim=-1))
        )
        rho = torch.zeros_like(potential)
        rho[accessible] = (
            torch.softmax(logits, dim=0) * numbers / volume
        )
        polarization = potential.new_zeros(*potential.shape, 3)
        polarization[accessible] = (
            rho[accessible] * moment
        )[..., None] * dipole_alignment(initial_alignment)
    else:
        rho = torch.as_tensor(
            rho0, dtype=potential.dtype, device=potential.device
        ).detach().clone()
        if rho.shape != potential.shape:
            raise ValueError("initial_rho must match V_ext shape")
        polarization = (
            potential.new_zeros(*potential.shape, 3)
            if p0 is None
            else torch.as_tensor(
                p0, dtype=potential.dtype, device=potential.device
            ).detach().clone()
        )
    if rho0 is None and p0 is not None:
        polarization = torch.as_tensor(
            p0, dtype=potential.dtype, device=potential.device
        ).detach().clone()
    if rho.shape != potential.shape:
        raise ValueError("initial_rho must match V_ext shape")
    if polarization.shape != (*potential.shape, 3):
        raise ValueError(
            "initial_polarization must have shape V_ext.shape + (3,)"
        )
    if (
        not torch.isfinite(rho).all()
        or torch.any(rho[accessible] <= 0)
    ):
        raise ValueError(
            "initial_rho must be finite and positive in accessible voxels"
        )
    if not torch.isfinite(polarization).all():
        raise ValueError("initial_polarization must be finite")
    if torch.any(rho[excluded] != 0) or torch.any(polarization[excluded] != 0):
        raise ValueError(
            "initial fields must be zero in excluded voxels"
        )
    count_tolerance = 100 * torch.finfo(potential.dtype).eps
    if not torch.allclose(
        volume * rho.sum(dim=0),
        numbers,
        rtol=count_tolerance,
        atol=0,
    ):
        raise ValueError(
            "initial_rho must have the requested particle_numbers"
        )
    # One pointwise lower bound serves both a user floor and the fixed
    # polarization's interior requirement |P| < m rho.
    minimum_density = None
    if density_floor is not None:
        minimum_density = torch.zeros_like(rho)
        minimum_density[accessible] = density_floor
    if fixed_field == "dipole_density":
        interior_fraction = (
            polarization_fraction_cap
            if polarization_fraction_cap is not None
            else 1.0 - 16.0 * torch.finfo(rho.dtype).eps
        )
        interior_density = torch.zeros_like(rho)
        interior_density[accessible] = (
            polarization[accessible].norm(dim=-1)
            / (moment * interior_fraction)
        )
        minimum_density = (
            interior_density
            if minimum_density is None
            else torch.maximum(minimum_density, interior_density)
        )
        rho = _project_density_constraints(
            rho,
            numbers,
            volume,
            density_cap,
            accessible,
            minimum_density=minimum_density,
        )
    elif fixed_field == "rho":
        if density_cap is not None and torch.any(
            rho[accessible] > density_cap
        ).item():
            raise ValueError(
                "fixed initial density exceeds maximum_density"
            )
        if minimum_density is not None and torch.any(
            rho[accessible] < minimum_density[accessible]
        ).item():
            raise ValueError(
                "fixed initial density lies below minimum_density"
            )
    else:
        projected_rho = _project_density_constraints(
            rho,
            numbers,
            volume,
            density_cap,
            accessible,
            minimum_density=minimum_density,
        )
        polarization[accessible] *= (
            projected_rho[accessible] / rho[accessible]
        )[..., None]
        rho = projected_rho
    if (
        polarization_fraction_cap is not None
        and fixed_field != "dipole_density"
    ):
        allowed_norm = (
            rho[accessible]
            * moment
            * polarization_fraction_cap
        )[..., None]
        polarization[accessible] = _project_vector_norm(
            polarization[accessible], allowed_norm
        )
    # Only the conjugate-field minimizer reads this: the alignment minimizer
    # takes it as an unused argument and Euler is never passed it. Computing
    # it eagerly would also evaluate the inverse Langevin on a saturated
    # start for paths that have no use for the result.
    conjugate_field = None
    if method == "minimize" and minimize_coordinate == "conjugate_field":
        initial_ideal = solver.model.ideal.local(
            rho[accessible],
            polarization[accessible],
            thermal_wavelength=wavelength,
        )
        conjugate_field = (
            initial_ideal["ideal_polarization_derivative"] * moment[..., None]
        ).detach()

    was_training = solver.model.training
    solver.model.eval()
    try:
        result = solver.evaluate(
            dict(data, rho=rho, dipole_density=polarization)
        )
        result = apply_fixed_dipole_bounds(
            result,
            density_cap,
            polarization_fraction_cap,
            moment,
            accessible,
            fixed_field=fixed_field,
            minimum_density=minimum_density,
        )
        if method == "minimize" and minimize.lbfgs_history > 0:
            if fixed_field is not None:
                raise ValueError("lbfgs_history is not supported with fixed_field")
            state = minimize_fixed_dipoles_lbfgs(
                solver, data, result, numbers, moment, volume, accessible,
                density_cap, polarization_fraction_cap, minimum_density,
                convergence, minimize,
                lambda evaluation: apply_fixed_dipole_bounds(
                    evaluation, density_cap, polarization_fraction_cap, moment,
                    accessible, minimum_density=minimum_density,
                ),
            )
        elif method == "minimize":
            arguments = (
                solver, data, result, conjugate_field, numbers, moment, volume,
                accessible, density_cap, polarization_fraction_cap, max_iter,
                tolerance_residual, tolerance_rms_residual, step_size,
                minimum_step_size, line_search_factor, armijo_factor,
                fixed_field, minimum_density,
            )
            if minimize_coordinate == "alignment":
                state = _minimize_fixed_dipoles_alignment(*arguments)
            else:
                state = _minimize_fixed_dipoles_conjugate_field(*arguments)
        elif euler_coordinate == "conjugate_field":
            state = solve_fixed_dipole_conjugate(
                solver, data, result, numbers, moment, beta, volume, accessible,
                density_cap, polarization_fraction_cap, minimum_density,
                convergence, euler,
            )
        else:
            state = solve_fixed_dipole_euler(solver,
                data,
                result,
                numbers,
                moment,
                beta,
                volume,
                accessible,
                density_cap,
                polarization_fraction_cap,
                minimum_density,
                max_iter,
                tolerance_residual,
                tolerance_rms_residual,
                tolerance_change,
                mixing,
                adaptive_mixing,
                minimum_mixing,
                maximum_mixing,
                mixing_growth,
                mixing_backtrack_factor,
                euler_coordinate,
            )
    finally:
        solver.model.train(was_training)
    result = state["result"]
    result.update(
        solver_maximum_polarization_fraction=(
            None
            if polarization_fraction_cap is None
            else polarization_fraction_cap.detach()
        ),
        solver_fixed_field=fixed_field,
        particle_number_error=result["particle_numbers"] - numbers,
    )
    if density_floor is not None:
        result["solver_minimum_density"] = density_floor.detach()
    if method == "minimize" and minimize_coordinate != "conjugate_field":
        result["solver_fixed_dipole_coordinate"] = minimize_coordinate
    if method == "minimize" and minimize.lbfgs_history > 0:
        result["solver_lbfgs_history"] = minimize.lbfgs_history
    if method == "euler" and euler_coordinate != "polarization":
        result["solver_fixed_dipole_coordinate"] = euler_coordinate
    state["objective_history"] = [item["beta_A"] for item in state["history"]]
    return finalize_result(
        result, state, method=method, beta_multiplier=beta_multiplier,
        converged=state["converged"],
    )


def _minimize_fixed_dipoles_conjugate_field(
    solver,
    data,
    result,
    conjugate_field,
    numbers,
    moment,
    volume,
    accessible,
    maximum_density,
    maximum_polarization_fraction,
    max_iter,
    tolerance_residual,
    tolerance_rms_residual,
    step_size,
    minimum_step_size,
    line_search_factor,
    armijo_factor,
    fixed_field,
    minimum_density,
):
    history = []
    n_evaluations = 1
    density_change = polarization_change = float("inf")
    status = "max_iter"
    converged = False
    n_iter = 0
    maximum_field = (
        None
        if maximum_polarization_fraction is None
        else inverse_langevin(maximum_polarization_fraction)[None, :, None]
    )
    for iteration in range(max_iter + 1):
        history.append(
            {
                "iteration": iteration,
                "beta_A": float(result["beta_A"]),
                "maximum_residual": float(result["maximum_residual"]),
            }
        )
        converged = _residuals_converged(
            result["maximum_residual"], result["rms_residual"],
            tolerance_residual, tolerance_rms_residual,
        )
        if converged or iteration == max_iter:
            status = "converged" if converged else "max_iter"
            break
        rho = result["rho"]
        polarization = result["dipole_density"]
        log_z = log_sinhc(conjugate_field.norm(dim=-1))
        trial_step = step_size
        accepted = False
        while trial_step >= minimum_step_size:
            if fixed_field == "dipole_density":
                trial_field = conjugate_field
                log_weights = (
                    rho[accessible].log()
                    - trial_step * result["density_residual"][accessible]
                )
                trial_rho = torch.zeros_like(rho)
                trial_rho[accessible] = (
                    torch.softmax(log_weights, dim=0) * numbers / volume
                )
                trial_rho = _project_density_constraints(
                    trial_rho,
                    numbers,
                    volume,
                    maximum_density,
                    accessible,
                    minimum_density=minimum_density,
                )
                trial_p = polarization.clone()
            else:
                trial_field = (
                    conjugate_field
                    - trial_step
                    * result["scaled_polarization_residual"][accessible]
                )
                trial_field = _project_vector_norm(
                    trial_field, maximum_field
                )
                if fixed_field == "rho":
                    trial_rho = rho.clone()
                else:
                    log_weights = (
                        rho[accessible].log()
                        - trial_step * result["density_residual"][accessible]
                        + log_sinhc(trial_field.norm(dim=-1))
                        - log_z
                    )
                    trial_rho = torch.zeros_like(rho)
                    trial_rho[accessible] = (
                        torch.softmax(log_weights, dim=0) * numbers / volume
                    )
                    trial_rho = _project_density_constraints(
                        trial_rho,
                        numbers,
                        volume,
                        maximum_density,
                        accessible,
                        minimum_density=minimum_density,
                    )
                trial_p = torch.zeros_like(polarization)
                trial_p[accessible] = (
                    trial_rho[accessible] * moment
                )[..., None] * dipole_alignment(trial_field)
            interior = (
                torch.isfinite(trial_rho).all()
                and torch.all(trial_rho[accessible] > 0)
                and torch.isfinite(trial_p).all()
                and torch.all(
                    trial_p[accessible].norm(dim=-1)
                    < trial_rho[accessible] * moment
                )
            )
            if interior:
                trial = solver.evaluate(
                    dict(data, rho=trial_rho, dipole_density=trial_p)
                )
                trial = apply_fixed_dipole_bounds(
                    trial,
                    maximum_density,
                    maximum_polarization_fraction,
                    moment,
                    accessible,
                    fixed_field=fixed_field,
                    minimum_density=minimum_density,
                )
                n_evaluations += 1
                rho_delta = trial_rho - rho
                p_delta = trial_p - polarization
                linear_change = volume * (
                    (
                        0.0
                        if fixed_field == "rho"
                        else (
                            result["beta_mu_local"] * rho_delta
                        ).sum()
                    )
                    + (
                        0.0
                        if fixed_field == "dipole_density"
                        else (
                            result["polarization_residual"] * p_delta
                        ).sum()
                    )
                )
                epsilon = torch.finfo(rho.dtype).eps
                roundoff = 32 * epsilon * max(
                    1.0, abs(float(result["beta_A"]))
                )
                energy_bound = (
                    result["beta_A"]
                    + armijo_factor * linear_change
                    + roundoff
                )
                accepted = bool(
                    torch.isfinite(trial["maximum_residual"])
                    and linear_change <= 0
                    and trial["beta_A"] <= energy_bound
                    and (
                        linear_change < -roundoff
                        or trial["maximum_residual"]
                        < result["maximum_residual"]
                    )
                )
                if accepted:
                    break
            trial_step *= line_search_factor
        if not accepted:
            status = "line_search_failed"
            break
        density_change = _maximum_relative_change(rho, trial_rho)
        polarization_change = float(
            (
                (trial_p - polarization).norm(dim=-1)
                / torch.clamp(trial_rho * moment, min=1.0e-12)
            )[accessible].max()
        )
        history[-1]["accepted_step"] = trial_step
        result = trial
        conjugate_field = trial_field
        n_iter += 1
    return {
        "result": result,
        "converged": converged,
        "status": status,
        "n_iter": n_iter,
        "n_evaluations": n_evaluations,
        "history": history,
        "line_search_failures": int(status == "line_search_failed"),
        "mixing_backtracks": 0,
        "final_mixing": None,
        "final_relative_density_change": density_change,
        "final_relative_polarization_change": polarization_change,
    }


def _physical_alignment(rho, polarization, moment, accessible):
    """Return physical ``u=P/(m rho)`` with zero excluded voxels."""

    alignment = torch.zeros_like(polarization)
    alignment[accessible] = (
        polarization[accessible]
        / (rho[accessible] * moment)[..., None]
    )
    return alignment


def _alignment_density_residual(
    result,
    alignment,
    moment,
    maximum_density,
    accessible,
    minimum_density=None,
):
    r"""Return the canonical density direction at fixed physical alignment.

    With ``P=m rho u``, the local derivative transforms as

    ``dA/drho|u = dA/drho|P + m u . dA/dP``.

    The physical polarization derivative is deliberately unconstrained here:
    a polarization-fraction bound is independent of density in ``u``
    coordinates, so its KKT multiplier must not enter the density direction.
    """

    local_mu = result["beta_mu_local"] + (
        moment[..., None]
        * alignment
        * result["polarization_residual"]
    ).sum(dim=-1)
    residual, _, _, _ = _canonical_density_residual(
        result["rho"],
        local_mu,
        0.0,
        maximum_density,
        accessible,
        minimum_density=minimum_density,
    )
    return residual


def _alignment_log_density_trial(
    rho,
    density_residual,
    trial_step,
    numbers,
    volume,
    maximum_density,
    accessible,
    minimum_density=None,
):
    """Take one positivity-preserving canonical log-density step."""

    trial_rho = torch.zeros_like(rho)
    log_weights = (
        rho[accessible].log()
        - trial_step * density_residual[accessible]
    )
    trial_rho[accessible] = (
        torch.softmax(log_weights, dim=0) * numbers / volume
    )
    return _project_density_constraints(
        trial_rho,
        numbers,
        volume,
        maximum_density,
        accessible,
        minimum_density=minimum_density,
    )


def _minimize_fixed_dipoles_alignment(
    solver,
    data,
    result,
    _conjugate_field,
    numbers,
    moment,
    volume,
    accessible,
    maximum_density,
    maximum_polarization_fraction,
    max_iter,
    tolerance_residual,
    tolerance_rms_residual,
    step_size,
    minimum_step_size,
    line_search_factor,
    armijo_factor,
    fixed_field,
    minimum_density,
):
    """Minimize by direct steps in density and physical alignment ``u``.

    The density direction is the exact fixed-``u`` chain-rule derivative and
    uses the positivity-preserving canonical log-density step. Alignment is
    updated with its exact physical-coordinate gradient ``rho*m*dA/dP`` and
    projected onto its Euclidean ball. Armijo acceptance uses the exact
    physical ``(rho, P)`` directional derivative and convergence retains the
    existing physical KKT residual.
    """

    history = []
    n_evaluations = 1
    density_change = polarization_change = alignment_change = float("inf")
    status = "max_iter"
    converged = False
    n_iter = 0
    # A user polarization-fraction cap is a constraint whose boundary is a
    # legitimate solution, so a trial is projected onto it. The interior
    # margin below is not a constraint but a numerical guard: the inverse
    # Langevin becomes singular as |u| approaches one, so an uncapped trial
    # that would cross the margin is rejected and backtracked instead of
    # being clipped onto it.
    maximum_alignment = (
        None
        if maximum_polarization_fraction is None
        else maximum_polarization_fraction[None, :, None]
    )
    interior_margin = 1.0 - max(1.0e-8, 64.0 * torch.finfo(moment.dtype).eps)

    for iteration in range(max_iter + 1):
        history.append(
            {
                "iteration": iteration,
                "beta_A": float(result["beta_A"]),
                "maximum_residual": float(result["maximum_residual"]),
            }
        )
        converged = _residuals_converged(
            result["maximum_residual"], result["rms_residual"],
            tolerance_residual, tolerance_rms_residual,
        )
        if converged or iteration == max_iter:
            status = "converged" if converged else "max_iter"
            break

        rho = result["rho"]
        polarization = result["dipole_density"]
        alignment = _physical_alignment(
            rho, polarization, moment, accessible
        )
        # Both search directions are loop-invariant: only the step length
        # changes between trials, so they are built once per outer iteration
        # rather than once per backtracking trial.
        alignment_direction = None
        density_residual = None
        if fixed_field != "dipole_density":
            alignment_direction = (
                rho[accessible][..., None]
                * result["scaled_polarization_residual"][accessible]
            )
            if fixed_field != "rho":
                density_residual = _alignment_density_residual(
                    result,
                    alignment,
                    moment,
                    maximum_density,
                    accessible,
                    minimum_density=minimum_density,
                )
        # The alignment gradient diverges like ``1/(1-|u|)`` as the fluid
        # saturates, so near saturation every step down to an absolute
        # ``minimum_step_size`` still overshoots the interior margin and the
        # line search fails without moving at all. The floor is therefore
        # measured relative to the direction magnitude, so that it bounds the
        # smallest admissible move in ``u`` rather than the raw multiplier.
        # For a direction of order one this is the absolute floor unchanged.
        direction_scale = 1.0
        if alignment_direction is not None and alignment_direction.numel():
            largest = alignment_direction.abs().max()
            if bool(torch.isfinite(largest)):
                direction_scale = max(1.0, float(largest))
        smallest_step = minimum_step_size / direction_scale
        trial_step = step_size
        accepted = False
        while trial_step >= smallest_step:
            if fixed_field == "dipole_density":
                trial_u = alignment
                trial_rho = _alignment_log_density_trial(
                    rho,
                    result["density_residual"],
                    trial_step,
                    numbers,
                    volume,
                    maximum_density,
                    accessible,
                    minimum_density=minimum_density,
                )
                trial_p = polarization.clone()
            else:
                trial_u = alignment.clone()
                trial_u[accessible] = (
                    alignment[accessible]
                    - trial_step * alignment_direction
                )
                if maximum_alignment is not None:
                    trial_u[accessible] = _project_vector_norm(
                        trial_u[accessible], maximum_alignment
                    )
                if fixed_field == "rho":
                    trial_rho = rho.clone()
                else:
                    trial_rho = _alignment_log_density_trial(
                        rho,
                        density_residual,
                        trial_step,
                        numbers,
                        volume,
                        maximum_density,
                        accessible,
                        minimum_density=minimum_density,
                    )
                trial_p = torch.zeros_like(polarization)
                trial_p[accessible] = (
                    trial_rho[accessible] * moment
                )[..., None] * trial_u[accessible]

            trial_alignment = _physical_alignment(
                trial_rho, trial_p, moment, accessible
            )
            if (
                maximum_alignment is None
                and torch.any(
                    trial_alignment[accessible].norm(dim=-1)
                    >= interior_margin
                )
            ):
                trial_step *= line_search_factor
                continue
            interior = (
                torch.isfinite(trial_rho).all()
                and torch.all(trial_rho[accessible] > 0)
                and torch.isfinite(trial_p).all()
                and torch.all(
                    trial_p[accessible].norm(dim=-1)
                    < trial_rho[accessible] * moment
                )
            )
            if interior:
                trial = solver.evaluate(
                    dict(data, rho=trial_rho, dipole_density=trial_p)
                )
                trial = apply_fixed_dipole_bounds(
                    trial,
                    maximum_density,
                    maximum_polarization_fraction,
                    moment,
                    accessible,
                    fixed_field=fixed_field,
                    minimum_density=minimum_density,
                )
                n_evaluations += 1
                rho_delta = trial_rho - rho
                p_delta = trial_p - polarization
                linear_change = volume * (
                    (
                        0.0
                        if fixed_field == "rho"
                        else (result["beta_mu_local"] * rho_delta).sum()
                    )
                    + (
                        0.0
                        if fixed_field == "dipole_density"
                        else (result["polarization_residual"] * p_delta).sum()
                    )
                )
                epsilon = torch.finfo(rho.dtype).eps
                roundoff = 32 * epsilon * max(
                    1.0, abs(float(result["beta_A"]))
                )
                energy_bound = (
                    result["beta_A"]
                    + armijo_factor * linear_change
                    + roundoff
                )
                accepted = bool(
                    torch.isfinite(trial["maximum_residual"])
                    and linear_change <= 0
                    and trial["beta_A"] <= energy_bound
                    and (
                        linear_change < -roundoff
                        or trial["maximum_residual"]
                        < result["maximum_residual"]
                    )
                )
                if accepted:
                    break
            trial_step *= line_search_factor

        if not accepted:
            status = "line_search_failed"
            break
        density_change = _maximum_relative_change(rho, trial_rho)
        polarization_change = float(
            (
                (trial_p - polarization).norm(dim=-1)
                / torch.clamp(trial_rho * moment, min=1.0e-12)
            )[accessible].max()
        )
        alignment_change = float(
            (trial_alignment - alignment)[accessible].norm(dim=-1).max()
        )
        history[-1]["accepted_step"] = trial_step
        result = trial
        n_iter += 1

    return {
        "result": result,
        "converged": converged,
        "status": status,
        "n_iter": n_iter,
        "n_evaluations": n_evaluations,
        "history": history,
        "line_search_failures": int(status == "line_search_failed"),
        "mixing_backtracks": 0,
        "final_mixing": None,
        "final_relative_density_change": density_change,
        "final_relative_polarization_change": polarization_change,
        "final_relative_alignment_change": alignment_change,
    }


def solve_fixed_dipole_euler(
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
    max_iter,
    tolerance_residual,
    tolerance_rms_residual,
    tolerance_change,
    mixing,
    adaptive_mixing,
    minimum_mixing,
    maximum_mixing,
    mixing_growth,
    mixing_backtrack_factor,
    fixed_dipole_coordinate,
):
    history = []
    n_evaluations = 1
    n_iter = 0
    mixing_backtracks = 0
    current_mixing = mixing
    density_change = polarization_change = coordinate_change = float("inf")
    status = "max_iter"
    converged = False
    maximum_field = (
        None
        if maximum_polarization_fraction is None
        else inverse_langevin(maximum_polarization_fraction)[None, :, None]
    )
    for iteration in range(max_iter + 1):
        history.append(
            {
                "iteration": iteration,
                "beta_A": float(result["beta_A"]),
                "maximum_residual": float(result["maximum_residual"]),
            }
        )
        converged = _residuals_converged(
            result["maximum_residual"], result["rms_residual"],
            tolerance_residual, tolerance_rms_residual,
        )
        if converged or iteration == max_iter:
            status = "converged" if converged else "max_iter"
            break
        rho = result["rho"]
        polarization = result["dipole_density"]
        unconstrained_field = moment[..., None] * (
            beta * data["E_ext"]
            - result["polarization_derivative_excess"]
        )
        target_field = _project_vector_norm(
            unconstrained_field, maximum_field
        )
        logits = (
            -beta * data["V_ext"]
            + result["c1"]
            + _orientational_log_weight(unconstrained_field, target_field)
        )
        target_rho = torch.zeros_like(rho)
        target_rho[accessible] = (
            torch.softmax(logits[accessible], dim=0) * numbers / volume
        )
        target_rho = _project_density_constraints(
            target_rho,
            numbers,
            volume,
            maximum_density,
            accessible,
            minimum_density=minimum_density,
            proportional=True,
        )
        target_u = torch.zeros_like(polarization)
        target_u[accessible] = dipole_alignment(target_field[accessible])
        if fixed_dipole_coordinate == "alignment":
            current_u = _physical_alignment(
                rho, polarization, moment, accessible
            )
        else:
            target_p = (
                target_rho * moment
            )[..., None] * target_u

        trial_mixing = current_mixing
        while True:
            trial_rho = (
                (1 - trial_mixing) * rho
                + trial_mixing * target_rho
            )
            if fixed_dipole_coordinate == "alignment":
                trial_u = (
                    (1 - trial_mixing) * current_u
                    + trial_mixing * target_u
                )
                trial_p = (
                    trial_rho * moment
                )[..., None] * trial_u
            else:
                trial_p = (
                    (1 - trial_mixing) * polarization
                    + trial_mixing * target_p
                )
            trial = solver.evaluate(
                dict(data, rho=trial_rho, dipole_density=trial_p)
            )
            trial = apply_fixed_dipole_bounds(
                trial,
                maximum_density,
                maximum_polarization_fraction,
                moment,
                accessible,
                minimum_density=minimum_density,
            )
            n_evaluations += 1
            if (
                not adaptive_mixing
                or trial["rms_residual"] <= result["rms_residual"]
                or trial_mixing <= minimum_mixing
            ):
                break
            trial_mixing = max(
                minimum_mixing,
                trial_mixing * mixing_backtrack_factor,
            )
            mixing_backtracks += 1
        density_change = _maximum_relative_change(rho, trial_rho)
        polarization_change = float(
            (
                (trial_p - polarization).norm(dim=-1)
                / torch.clamp(trial_rho * moment, min=1.0e-12)
            )[accessible].max()
        )
        if fixed_dipole_coordinate == "alignment":
            alignment_change = float(
                (trial_u - current_u)[accessible].norm(dim=-1).max()
            )
            coordinate_change = alignment_change
        else:
            coordinate_change = polarization_change
        history[-1]["mixing"] = trial_mixing
        result = trial
        n_iter += 1
        current_mixing = (
            min(maximum_mixing, trial_mixing * mixing_growth)
            if adaptive_mixing
            else mixing
        )
        if max(density_change, coordinate_change) <= tolerance_change:
            converged = _residuals_converged(
                result["maximum_residual"], result["rms_residual"],
                tolerance_residual, tolerance_rms_residual,
            )
            if converged:
                status = "converged"
            else:
                status = "max_iter" if n_iter >= max_iter else "stalled"
            history.append(
                {
                    "iteration": n_iter,
                    "beta_A": float(result["beta_A"]),
                    "maximum_residual": float(
                        result["maximum_residual"]
                    ),
                }
            )
            break
    state = {
        "result": result,
        "converged": converged,
        "status": status,
        "n_iter": n_iter,
        "n_evaluations": n_evaluations,
        "history": history,
        "line_search_failures": 0,
        "mixing_backtracks": mixing_backtracks,
        "final_mixing": current_mixing,
        "final_relative_density_change": density_change,
        "final_relative_polarization_change": polarization_change,
    }
    if fixed_dipole_coordinate == "alignment":
        state["final_relative_alignment_change"] = coordinate_change
    return state
