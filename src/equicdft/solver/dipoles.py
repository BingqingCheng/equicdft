"""Fixed-dipole (coupled density and polarization) evaluation and solvers.

These drivers mirror :mod:`equicdft.solver.euler` and
:mod:`equicdft.solver.minimize` for the coupled ``(rho, P)`` problem of a
fixed-magnitude dipole fluid. They keep their own update maps, acceptance
tests and residual bookkeeping: the orientational mirror map and the
coupled KKT projections are not the scalar ones, so the two families share
the option groups, the constraint helpers and the result conventions
rather than one loop.
"""

import torch

from ..functional.polarization_ideal import (
    _positive_components,
    dipole_alignment,
    inverse_langevin,
    log_sinhc,
)
from ..geometry.grid import voxel_volume
from .numerics import (
    _canonical_density_residual,
    _component_tensor,
    _fixed_dipole_bound_kkt,
    _maximum_relative_change,
    _polarization_residual_norms,
    _project_density_constraints,
    _project_vector_norm,
    _resolve_accessibility_masks,
)

__all__ = ["evaluate_fixed_dipoles", "solve_fixed_dipoles"]

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
    if solver.model is not None and hasattr(solver.model, "boltzmann_constant"):
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


@torch.enable_grad()
def evaluate_fixed_dipoles(solver, data, compute_derivatives=True):
    """Evaluate ideal, excess and external terms for independent rho/P."""

    data = solver._move_data(data)
    beta, volume, moment, wavelength, excluded, accessible = (
        fixed_dipole_inputs(solver, data)
    )
    potential = data["V_ext"]
    rho = torch.as_tensor(
        data["rho"], dtype=potential.dtype, device=potential.device
    ).detach().clone()
    polarization = torch.as_tensor(
        data["dipole_density"],
        dtype=potential.dtype,
        device=potential.device,
    ).detach().clone()
    if rho.shape != potential.shape:
        raise ValueError("rho must match V_ext shape")
    if polarization.shape != (*rho.shape, 3):
        raise ValueError("dipole_density must have shape rho.shape + (3,)")
    if torch.any(rho[excluded] != 0) or torch.any(polarization[excluded] != 0):
        raise ValueError(
            "rho and dipole_density must be zero in excluded voxels"
        )
    rho.requires_grad_(compute_derivatives)
    polarization.requires_grad_(compute_derivatives)

    model_data = dict(data, rho=rho, dipole_density=polarization)
    model_outputs = solver.model(
        model_data, compute_c1=compute_derivatives, compute_c2=False,
        compute_polarization_derivative=compute_derivatives,
    )
    excess = model_outputs["beta_F_exc"]
    if excess.ndim:
        raise ValueError("model must return scalar beta_F_exc for one field")
    external = beta * volume * (
        (rho * potential).sum()
        - (polarization * data["E_ext"]).sum()
    )
    beta_f = model_outputs["beta_F"]
    beta_a = beta_f + external
    if not torch.isfinite(beta_a):
        raise FloatingPointError("nonfinite fixed-dipole objective")

    result = {
        key: data[key]
        for key in (
            "V_ext",
            "E_ext",
            "temperature",
            "beta",
            "grid_spacing",
            "thermal_wavelength",
            "excluded_mask",
        )
        if key in data
    }
    result.update(model_outputs)
    result.update(
        rho=rho,
        dipole_density=polarization,
        beta_F_id=model_outputs["beta_F_id"],
        beta_F_exc=excess,
        beta_F=beta_f,
        beta_external=external,
        beta_A=beta_a,
        particle_numbers=volume * rho.sum(dim=0),
        maximum_alignment=(
            polarization[accessible].norm(dim=-1)
            / (rho[accessible] * moment)
        ).max(),
    )
    if compute_derivatives:
        c1 = model_outputs["c1"]
        polarization_derivative = model_outputs["polarization_derivative"]
        ideal_density = model_outputs["ideal_density_derivative"]
        ideal_polarization = model_outputs["ideal_polarization_derivative"]
        local_mu = model_outputs["density_derivative"] + beta * potential
        weights = rho[accessible]
        chemical_potential = (
            weights * local_mu[accessible]
        ).sum(dim=0) / weights.sum(dim=0)
        density_residual = torch.where(
            accessible[:, None],
            local_mu - chemical_potential,
            torch.zeros_like(local_mu),
        )
        polarization_residual = torch.where(
            accessible[:, None, None],
            model_outputs["total_polarization_derivative"]
            - beta * data["E_ext"],
            torch.zeros_like(polarization),
        )
        scaled_polarization_residual = (
            polarization_residual * moment[..., None]
        )
        active_rho = rho[accessible]
        active_density_residual = density_residual[accessible]
        max_density = active_density_residual.abs().max()
        rms_density = torch.sqrt(
            (active_rho * active_density_residual.square()).sum()
            / active_rho.sum()
        )
        max_polarization, rms_polarization = (
            _polarization_residual_norms(
                rho,
                scaled_polarization_residual,
                accessible,
            )
        )
        maximum = torch.maximum(max_density, max_polarization)
        rms = torch.maximum(rms_density, rms_polarization)
        result.update(
            c1=c1,
            polarization_derivative=polarization_derivative,
            ideal_density_derivative=ideal_density,
            ideal_polarization_derivative=ideal_polarization,
            beta_mu_local=local_mu,
            beta_mu=chemical_potential,
            euler_lagrange_residual=density_residual,
            density_residual=density_residual,
            polarization_euler_lagrange_residual=polarization_residual,
            polarization_residual=polarization_residual,
            scaled_polarization_residual=scaled_polarization_residual,
            max_euler_lagrange_residual=max_density,
            rms_euler_lagrange_residual=rms_density,
            max_polarization_residual=max_polarization,
            rms_polarization_residual=rms_polarization,
            maximum_residual=maximum,
            rms_residual=rms,
        )
    return {
        key: value.detach() if torch.is_tensor(value) else value
        for key, value in result.items()
    }


def fixed_dipole_converged(
    solver, result, tolerance_residual, tolerance_rms_residual
):
    return bool(
        result["maximum_residual"] <= tolerance_residual
        and (
            tolerance_rms_residual is None
            or result["rms_residual"] <= tolerance_rms_residual
        )
    )


def apply_fixed_dipole_bounds(solver,
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
        result["beta_mu"] = chemical_potential
        result["euler_lagrange_residual"] = residual
        result["density_residual"] = residual
        result["max_euler_lagrange_residual"] = maximum
        result["rms_euler_lagrange_residual"] = rms
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
        )
    )
    result["beta_mu"] = chemical_potential
    result["euler_lagrange_residual"] = residual
    result["density_residual"] = residual
    result["max_euler_lagrange_residual"] = maximum
    result["rms_euler_lagrange_residual"] = rms

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
    if euler.anderson is not None:
        raise ValueError(
            "Anderson acceleration is not yet supported with fixed dipoles"
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
    tolerance_change = euler.tolerance_change
    mixing = euler.mixing
    adaptive_mixing = euler.adaptive_mixing
    minimum_mixing = euler.minimum_mixing
    maximum_mixing = euler.maximum_mixing
    mixing_growth = euler.mixing_growth
    mixing_backtrack_factor = euler.mixing_backtrack_factor

    data = solver._move_data(data)
    beta, volume, moment, wavelength, excluded, accessible = (
        fixed_dipole_inputs(solver, data)
    )
    potential = data["V_ext"]
    numbers = _positive_components(
        particle_numbers, potential, "particle_numbers"
    ).expand(potential.shape[-1])
    density_cap = None
    if maximum_density is not None:
        density_cap = _component_tensor(
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
    polarization_fraction_cap = None
    if maximum_polarization_fraction is not None:
        polarization_fraction_cap = _component_tensor(
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
    minimum_density = None
    if fixed_field == "dipole_density":
        interior_fraction = (
            polarization_fraction_cap
            if polarization_fraction_cap is not None
            else 1.0 - 16.0 * torch.finfo(rho.dtype).eps
        )
        minimum_density = torch.zeros_like(rho)
        minimum_density[accessible] = (
            polarization[accessible].norm(dim=-1)
            / (moment * interior_fraction)
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
    else:
        projected_rho = _project_density_constraints(
            rho,
            numbers,
            volume,
            density_cap,
            accessible,
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
    initial_ideal = solver.model.ideal(
        rho[accessible],
        polarization[accessible],
        volume,
        thermal_wavelength=wavelength,
    )
    alignment = (
        initial_ideal["polarization_derivative"] * moment[..., None]
    ).detach()

    was_training = solver.model.training if solver.model is not None else None
    if solver.model is not None:
        solver.model.eval()
    try:
        result = solver.evaluate(
            dict(data, rho=rho, dipole_density=polarization)
        )
        result = apply_fixed_dipole_bounds(solver,
            result,
            density_cap,
            polarization_fraction_cap,
            moment,
            accessible,
            fixed_field=fixed_field,
            minimum_density=minimum_density,
        )
        if method == "minimize":
            state = minimize_fixed_dipoles(solver,
                data,
                result,
                alignment,
                numbers,
                moment,
                volume,
                accessible,
                density_cap,
                polarization_fraction_cap,
                max_iter,
                tolerance_residual,
                tolerance_rms_residual,
                step_size,
                minimum_step_size,
                line_search_factor,
                armijo_factor,
                fixed_field,
                minimum_density,
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
            )
    finally:
        if solver.model is not None:
            solver.model.train(was_training)
    result = state["result"]
    result.update(
        converged=state["converged"],
        status=state["status"],
        solver_method=method,
        solver_beta_multiplier=beta_multiplier,
        solver_maximum_polarization_fraction=(
            None
            if polarization_fraction_cap is None
            else polarization_fraction_cap.detach()
        ),
        solver_fixed_field=fixed_field,
        n_iter=state["n_iter"],
        iterations=state["n_iter"],
        n_evaluations=state["n_evaluations"],
        history=state["history"],
        objective_history=[item["beta_A"] for item in state["history"]],
        line_search_failures=state["line_search_failures"],
        mixing_backtracks=state["mixing_backtracks"],
        final_mixing=state["final_mixing"],
        final_relative_density_change=state["density_change"],
        final_relative_polarization_change=state["polarization_change"],
        solver_anderson=False,
        anderson_attempts=0,
        anderson_accepted=0,
        anderson_rejected=0,
        anderson_resets=0,
    )
    result["particle_number_error"] = result["particle_numbers"] - numbers
    return result


def minimize_fixed_dipoles(
    solver,
    data,
    result,
    alignment,
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
    maximum_alignment = (
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
        converged = fixed_dipole_converged(solver,
            result, tolerance_residual, tolerance_rms_residual
        )
        if converged or iteration == max_iter:
            status = "converged" if converged else "max_iter"
            break
        rho = result["rho"]
        polarization = result["dipole_density"]
        log_z = log_sinhc(alignment.norm(dim=-1))
        trial_step = step_size
        accepted = False
        while trial_step >= minimum_step_size:
            if fixed_field == "dipole_density":
                trial_alignment = alignment
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
                trial_alignment = (
                    alignment
                    - trial_step
                    * result["scaled_polarization_residual"][accessible]
                )
                trial_alignment = _project_vector_norm(
                    trial_alignment, maximum_alignment
                )
                if fixed_field == "rho":
                    trial_rho = rho.clone()
                else:
                    log_weights = (
                        rho[accessible].log()
                        - trial_step * result["density_residual"][accessible]
                        + log_sinhc(trial_alignment.norm(dim=-1))
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
                    )
                trial_p = torch.zeros_like(polarization)
                trial_p[accessible] = (
                    trial_rho[accessible] * moment
                )[..., None] * dipole_alignment(trial_alignment)
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
                trial = apply_fixed_dipole_bounds(solver,
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
        alignment = trial_alignment
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
        "density_change": density_change,
        "polarization_change": polarization_change,
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
):
    history = []
    n_evaluations = 1
    n_iter = 0
    mixing_backtracks = 0
    current_mixing = mixing
    density_change = polarization_change = float("inf")
    status = "max_iter"
    converged = False
    maximum_alignment = (
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
        converged = fixed_dipole_converged(solver,
            result, tolerance_residual, tolerance_rms_residual
        )
        if converged or iteration == max_iter:
            status = "converged" if converged else "max_iter"
            break
        rho = result["rho"]
        polarization = result["dipole_density"]
        target_alignment = moment[..., None] * (
            beta * data["E_ext"]
            - result["polarization_derivative"]
        )
        target_alignment = _project_vector_norm(
            target_alignment, maximum_alignment
        )
        logits = (
            -beta * data["V_ext"]
            + result["c1"]
            + log_sinhc(target_alignment.norm(dim=-1))
        )
        target_rho = torch.zeros_like(rho)
        target_p = torch.zeros_like(polarization)
        target_rho[accessible] = (
            torch.softmax(logits[accessible], dim=0) * numbers / volume
        )
        target_rho = _project_density_constraints(
            target_rho,
            numbers,
            volume,
            maximum_density,
            accessible,
        )
        target_p[accessible] = (
            target_rho[accessible] * moment
        )[..., None] * dipole_alignment(target_alignment[accessible])

        trial_mixing = current_mixing
        while True:
            trial_rho = (
                (1 - trial_mixing) * rho
                + trial_mixing * target_rho
            )
            trial_p = (
                (1 - trial_mixing) * polarization
                + trial_mixing * target_p
            )
            trial = solver.evaluate(
                dict(data, rho=trial_rho, dipole_density=trial_p)
            )
            trial = apply_fixed_dipole_bounds(solver,
                trial,
                maximum_density,
                maximum_polarization_fraction,
                moment,
                accessible,
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
        history[-1]["mixing"] = trial_mixing
        result = trial
        n_iter += 1
        current_mixing = (
            min(maximum_mixing, trial_mixing * mixing_growth)
            if adaptive_mixing
            else mixing
        )
        if max(density_change, polarization_change) <= tolerance_change:
            converged = fixed_dipole_converged(solver,
                result, tolerance_residual, tolerance_rms_residual
            )
            status = "converged" if converged else "max_iter"
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
    return {
        "result": result,
        "converged": converged,
        "status": status,
        "n_iter": n_iter,
        "n_evaluations": n_evaluations,
        "history": history,
        "line_search_failures": 0,
        "mixing_backtracks": mixing_backtracks,
        "final_mixing": current_mixing,
        "density_change": density_change,
        "polarization_change": polarization_change,
    }
