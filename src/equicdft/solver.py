"""Evaluate learned grid functionals and minimize thermodynamic objectives."""

from typing import Any, Dict, Optional, Sequence, Union

import torch
from torch import nn


class GridSolver:
    """Evaluate prescribed densities or solve for an equilibrium density.

    The wrapped model supplies the intrinsic excess functional. This class
    adds the exact ideal-gas, external-potential, and chemical-potential terms.

    ``evaluate(data)`` requires ``rho`` and returns all quantities supported by
    the available fields. ``solve(data)`` requires ``V_ext`` and either ``mu``
    or fixed ``particle_numbers``. Equilibrium solving supports one complete,
    unbatched field; prescribed-density evaluation also supports batches.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        self.model = model
        self.device = (
            _module_device(model)
            if device is None
            else torch.device(device)
        )
        self.model.to(self.device)

    def evaluate(
        self,
        data: Dict[str, Any],
        compute_c1: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate a supplied density and its available thermodynamics."""

        data = self._move_data(data)
        rho = data["rho"]
        if torch.any(rho < 0.0).item():
            raise ValueError("rho must be nonnegative")

        outputs = self.model(data, compute_c1=compute_c1)
        result = {
            key: data[key]
            for key in ("rho", "V_ext", "mu", "temperature", "beta")
            if key in data
        }
        result.update(outputs)

        cell_volume = torch.prod(data["grid_spacing"], dim=-1)
        thermal_wavelength = data.get(
            "thermal_wavelength",
            torch.ones(
                rho.shape[-1],
                dtype=rho.dtype,
                device=rho.device,
            ),
        )
        log_density = _log_dimensionless_density(
            rho,
            thermal_wavelength,
        )
        ideal_density = torch.where(
            rho > 0.0,
            rho * (log_density - 1.0),
            torch.zeros_like(rho),
        )
        result["beta_F_id"] = cell_volume * torch.sum(
            ideal_density,
            dim=(-2, -1),
        )
        result["beta_F"] = result["beta_F_id"] + result["beta_F_exc"]

        if "V_ext" in data:
            beta = data["beta"][..., None, None]
            result["beta_V_ext"] = cell_volume * torch.sum(
                rho * beta * data["V_ext"],
                dim=(-2, -1),
            )

        if "V_ext" in data and "mu" in data:
            beta = data["beta"][..., None, None]
            beta_mu_N = cell_volume * torch.sum(
                rho * beta * data["mu"][..., None, :],
                dim=(-2, -1),
            )
            result["beta_mu_N"] = beta_mu_N
            result["beta_Omega"] = (
                result["beta_F"] + result["beta_V_ext"] - beta_mu_N
            )
            if "local_chemical_potential" in result:
                result["euler_lagrange_residual"] = (
                    result["local_chemical_potential"]
                    - beta * data["mu"][..., None, :]
                )

        return result

    def solve(
        self,
        data: Dict[str, Any],
        initial_rho: Optional[torch.Tensor] = None,
        particle_numbers: Optional[
            Union[float, Sequence[float], torch.Tensor]
        ] = None,
        method: str = "minimize",
        max_iter: int = 200,
        tolerance_residual: float = 1.0e-4,
        tolerance_rms_residual: Optional[float] = None,
        tolerance_change: float = 1.0e-7,
        step_size: float = 1.0,
        minimum_step_size: float = 1.0e-8,
        line_search_factor: float = 0.5,
        armijo_factor: float = 1.0e-4,
        mixing: float = 0.05,
        adaptive_mixing: bool = True,
        minimum_mixing: float = 0.005,
        maximum_mixing: float = 0.2,
        mixing_growth: float = 1.1,
        mixing_backtrack_factor: float = 0.5,
        continuation_steps: int = 5,
        max_log_density_change: float = 2.0,
        residual_density_threshold: Optional[float] = None,
        maximum_density: Optional[
            Union[float, Sequence[float], torch.Tensor]
        ] = None,
    ) -> Dict[str, Any]:
        """Minimize the thermodynamic functional to obtain equilibrium.

        ``method="minimize"`` performs positivity-preserving mirror descent
        with an Armijo line search on the actual free energy (fixed particle
        numbers) or grand potential (known chemical potential). Every
        accepted step lowers that thermodynamic objective. Fixed particle
        numbers are imposed by exact normalization of every trial density.

        ``maximum_density`` optionally imposes one upper density bound per
        component. At fixed particle number, every update is projected onto
        the intersection of the particle-number constraint and this box
        constraint. The reported residual then uses the corresponding KKT
        condition: a capped grid point may have a negative unconstrained
        residual because increasing its density is forbidden.

        ``method="euler"`` uses a damped Euler--Lagrange fixed-point
        iteration. By default, its mixing is increased after an improving
        density-weighted RMS residual and backtracked after a worsening
        trial. Set ``adaptive_mixing=False`` to use a fixed mixing value.
        Both methods can introduce the external field by continuation and
        use the physical projected functional gradient as a convergence
        diagnostic.

        ``tolerance_residual`` bounds the largest active-grid residual. When
        ``tolerance_rms_residual`` is supplied, its density-weighted RMS bound
        must also be satisfied. ``residual_density_threshold`` excludes
        statistically unresolved low-density voxels from both diagnostics.
        """

        if method not in ("minimize", "euler"):
            raise ValueError("method must be 'minimize' or 'euler'")
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")
        if (
            not isinstance(continuation_steps, int)
            or continuation_steps < 0
        ):
            raise ValueError(
                "continuation_steps must be a nonnegative integer"
            )
        tolerance_residual = float(tolerance_residual)
        if tolerance_rms_residual is not None:
            tolerance_rms_residual = float(tolerance_rms_residual)
        tolerance_change = float(tolerance_change)
        step_size = float(step_size)
        minimum_step_size = float(minimum_step_size)
        line_search_factor = float(line_search_factor)
        armijo_factor = float(armijo_factor)
        mixing = float(mixing)
        minimum_mixing = float(minimum_mixing)
        maximum_mixing = float(maximum_mixing)
        mixing_growth = float(mixing_growth)
        mixing_backtrack_factor = float(mixing_backtrack_factor)
        max_log_density_change = float(max_log_density_change)
        if tolerance_residual <= 0.0:
            raise ValueError("tolerance_residual must be positive")
        if (
            tolerance_rms_residual is not None
            and tolerance_rms_residual <= 0.0
        ):
            raise ValueError("tolerance_rms_residual must be positive")
        if tolerance_change <= 0.0:
            raise ValueError("tolerance_change must be positive")
        if step_size <= 0.0:
            raise ValueError("step_size must be positive")
        if minimum_step_size <= 0.0 or minimum_step_size > step_size:
            raise ValueError(
                "minimum_step_size must be positive and no larger than "
                "step_size"
            )
        if not 0.0 < line_search_factor < 1.0:
            raise ValueError(
                "line_search_factor must be in the interval (0, 1)"
            )
        if not 0.0 <= armijo_factor < 1.0:
            raise ValueError("armijo_factor must be in the interval [0, 1)")
        if not 0.0 < mixing <= 1.0:
            raise ValueError("mixing must be in the interval (0, 1]")
        if not isinstance(adaptive_mixing, bool):
            raise TypeError("adaptive_mixing must be a boolean")
        if adaptive_mixing:
            if not 0.0 < minimum_mixing <= mixing:
                raise ValueError(
                    "minimum_mixing must be positive and no larger than mixing"
                )
            if not mixing <= maximum_mixing <= 1.0:
                raise ValueError(
                    "maximum_mixing must be at least mixing and no larger than one"
                )
            if mixing_growth < 1.0:
                raise ValueError("mixing_growth must be at least one")
            if not 0.0 < mixing_backtrack_factor < 1.0:
                raise ValueError(
                    "mixing_backtrack_factor must lie in the interval (0, 1)"
                )
        if max_log_density_change <= 0.0:
            raise ValueError("max_log_density_change must be positive")

        data = self._move_data(data)
        V_ext = data["V_ext"]
        if V_ext.ndim != 2:
            raise ValueError("solve currently accepts one unbatched field")

        n_types = V_ext.shape[-1]
        thermal_wavelength = _component_tensor(
            data.get("thermal_wavelength", 1.0),
            n_types,
            V_ext,
            "thermal_wavelength",
        )
        cell_volume = torch.prod(data["grid_spacing"])

        density_cap = None
        if maximum_density is not None:
            density_cap = _component_tensor(
                maximum_density,
                n_types,
                V_ext,
                "maximum_density",
            )
            if (
                not torch.all(torch.isfinite(density_cap)).item()
                or torch.any(density_cap <= 0.0).item()
            ):
                raise ValueError(
                    "maximum_density values must be finite and positive"
                )

        fixed_N = None
        if particle_numbers is None:
            mu = _component_tensor(data["mu"], n_types, V_ext, "mu")
        else:
            fixed_N = _component_tensor(
                particle_numbers,
                n_types,
                V_ext,
                "particle_numbers",
            )
            if torch.any(fixed_N <= 0.0).item():
                raise ValueError("particle_numbers must be positive")
            if density_cap is not None:
                maximum_particle_numbers = (
                    cell_volume * V_ext.shape[0] * density_cap
                )
                if torch.any(fixed_N > maximum_particle_numbers).item():
                    raise ValueError(
                        "particle_numbers are infeasible under maximum_density"
                    )
            mu = None

        if residual_density_threshold is None:
            residual_density_threshold = 0.0
        else:
            residual_density_threshold = float(residual_density_threshold)
        if residual_density_threshold < 0.0:
            raise ValueError(
                "residual_density_threshold must be nonnegative"
            )

        if initial_rho is None:
            initial_rho = data.get("rho")
        if initial_rho is not None:
            initial_rho = torch.as_tensor(
                initial_rho,
                dtype=V_ext.dtype,
                device=V_ext.device,
            )
            if initial_rho.shape != V_ext.shape:
                raise ValueError("initial_rho must have the same shape as V_ext")
            if torch.any(initial_rho <= 0.0).item():
                raise ValueError("initial_rho must be positive")
            rho = initial_rho.detach().clone()
        elif fixed_N is None:
            rho = torch.exp(data["beta"] * mu)[None, :].expand_as(V_ext)
            rho = rho / thermal_wavelength[None, :] ** 3
        else:
            uniform_density = fixed_N / (cell_volume * V_ext.shape[0])
            rho = uniform_density[None, :].expand_as(V_ext).clone()

        if fixed_N is not None:
            rho = _normalize_particle_numbers(
                rho,
                fixed_N,
                cell_volume,
                maximum_density=density_cap,
            )
        elif density_cap is not None:
            rho = torch.minimum(rho, density_cap[None, :])

        was_training = self.model.training
        self.model.eval()
        try:
            if method == "minimize":
                state = self._minimize(
                    data=data,
                    rho=rho,
                    V_ext=V_ext,
                    thermal_wavelength=thermal_wavelength,
                    mu=mu,
                    fixed_N=fixed_N,
                    cell_volume=cell_volume,
                    max_iter=max_iter,
                    tolerance_residual=tolerance_residual,
                    tolerance_rms_residual=tolerance_rms_residual,
                    step_size=step_size,
                    minimum_step_size=minimum_step_size,
                    line_search_factor=line_search_factor,
                    armijo_factor=armijo_factor,
                    continuation_steps=continuation_steps,
                    max_log_density_change=max_log_density_change,
                    residual_density_threshold=residual_density_threshold,
                    maximum_density=density_cap,
                )
            else:
                state = self._solve_euler(
                    data=data,
                    rho=rho,
                    V_ext=V_ext,
                    thermal_wavelength=thermal_wavelength,
                    mu=mu,
                    fixed_N=fixed_N,
                    cell_volume=cell_volume,
                    max_iter=max_iter,
                    tolerance_residual=tolerance_residual,
                    tolerance_rms_residual=tolerance_rms_residual,
                    tolerance_change=tolerance_change,
                    mixing=mixing,
                    adaptive_mixing=adaptive_mixing,
                    minimum_mixing=minimum_mixing,
                    maximum_mixing=maximum_mixing,
                    mixing_growth=mixing_growth,
                    mixing_backtrack_factor=mixing_backtrack_factor,
                    continuation_steps=continuation_steps,
                    max_log_density_change=max_log_density_change,
                    residual_density_threshold=residual_density_threshold,
                    maximum_density=density_cap,
                )

            final_data = dict(data)
            final_data["rho"] = state["rho"].detach().clone()
            result = self.evaluate(final_data, compute_c1=True)
        finally:
            self.model.train(was_training)

        residual, chemical_potential, max_residual, rms_residual = (
            _euler_residual(
                result["rho"],
                result["c1"],
                V_ext,
                data["beta"],
                thermal_wavelength,
                mu,
                residual_density_threshold,
                density_cap,
            )
        )
        result["euler_lagrange_residual"] = residual
        result["equilibrium_chemical_potential"] = chemical_potential
        result["max_euler_lagrange_residual"] = max_residual
        result["rms_euler_lagrange_residual"] = rms_residual
        result["converged"] = _residuals_converged(
            max_residual,
            rms_residual,
            tolerance_residual,
            tolerance_rms_residual,
        )
        result["solver_method"] = method
        result["n_iter"] = state["n_iter"]
        result["n_evaluations"] = state["n_evaluations"]
        result["objective_history"] = state["stage_objective_history"][-1]
        result["stage_objective_history"] = state[
            "stage_objective_history"
        ]
        result["final_relative_density_change"] = state[
            "final_relative_density_change"
        ]
        result["line_search_failures"] = state["line_search_failures"]
        result["mixing_backtracks"] = state.get("mixing_backtracks", 0)
        result["final_mixing"] = state.get("final_mixing")
        return result

    def _minimize(
        self,
        data: Dict[str, Any],
        rho: torch.Tensor,
        V_ext: torch.Tensor,
        thermal_wavelength: torch.Tensor,
        mu: Optional[torch.Tensor],
        fixed_N: Optional[torch.Tensor],
        cell_volume: torch.Tensor,
        max_iter: int,
        tolerance_residual: float,
        tolerance_rms_residual: Optional[float],
        step_size: float,
        minimum_step_size: float,
        line_search_factor: float,
        armijo_factor: float,
        continuation_steps: int,
        max_log_density_change: float,
        residual_density_threshold: float,
        maximum_density: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        """Minimize each continued thermodynamic objective by mirror descent."""

        stage_histories = []
        n_iter = 0
        n_evaluations = 0
        line_search_failures = 0
        final_relative_change = float("inf")

        for stage_V_ext in _continued_fields(V_ext, continuation_steps):
            stage_history = []
            next_step_size = step_size

            for _ in range(max_iter):
                current_data = dict(data)
                current_data["rho"] = rho.detach().clone()
                current_data["V_ext"] = stage_V_ext
                evaluation = self.evaluate(current_data, compute_c1=True)
                n_evaluations += 1
                objective = _thermodynamic_objective(
                    evaluation,
                    fixed_N is not None,
                    cell_volume,
                    thermal_wavelength,
                ).detach()
                if not torch.isfinite(objective).item():
                    raise ValueError(
                        "the thermodynamic objective became non-finite"
                    )
                if not stage_history:
                    stage_history.append(objective.item())

                (
                    residual,
                    chemical_potential,
                    max_residual,
                    rms_residual,
                ) = _euler_residual(
                    rho,
                    evaluation["c1"].detach(),
                    stage_V_ext,
                    data["beta"],
                    thermal_wavelength,
                    mu,
                    residual_density_threshold,
                    maximum_density,
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
                line_search_gradient = (
                    _log_dimensionless_density(rho, thermal_wavelength)
                    + data["beta"] * stage_V_ext
                    - evaluation["c1"].detach()
                )
                if fixed_N is None:
                    line_search_gradient = (
                        line_search_gradient - data["beta"] * mu[None, :]
                    )

                accepted = False
                trial_step_size = next_step_size
                while trial_step_size >= minimum_step_size:
                    trial_rho = _mirror_descent_trial(
                        rho,
                        residual,
                        trial_step_size,
                        fixed_N,
                        cell_volume,
                        max_log_density_change,
                        maximum_density,
                    )
                    displacement = trial_rho - rho
                    objective_dtype = objective.dtype
                    directional_derivative = cell_volume.to(
                        objective_dtype
                    ) * torch.sum(
                        line_search_gradient.to(objective_dtype)
                        * displacement.to(objective_dtype)
                    )

                    trial_data = dict(data)
                    trial_data["rho"] = trial_rho.detach().clone()
                    trial_data["V_ext"] = stage_V_ext
                    trial_evaluation = self.evaluate(
                        trial_data,
                        compute_c1=False,
                    )
                    n_evaluations += 1
                    trial_objective = _thermodynamic_objective(
                        trial_evaluation,
                        fixed_N is not None,
                        cell_volume,
                        thermal_wavelength,
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
                stage_history.append(trial_objective.item())
                next_step_size = min(
                    step_size,
                    trial_step_size / line_search_factor,
                )

            stage_histories.append(stage_history)

        return {
            "rho": rho,
            "n_iter": n_iter,
            "n_evaluations": n_evaluations,
            "stage_objective_history": stage_histories,
            "final_relative_density_change": final_relative_change,
            "line_search_failures": line_search_failures,
        }

    def _solve_euler(
        self,
        data: Dict[str, Any],
        rho: torch.Tensor,
        V_ext: torch.Tensor,
        thermal_wavelength: torch.Tensor,
        mu: Optional[torch.Tensor],
        fixed_N: Optional[torch.Tensor],
        cell_volume: torch.Tensor,
        max_iter: int,
        tolerance_residual: float,
        tolerance_rms_residual: Optional[float],
        tolerance_change: float,
        mixing: float,
        adaptive_mixing: bool,
        minimum_mixing: float,
        maximum_mixing: float,
        mixing_growth: float,
        mixing_backtrack_factor: float,
        continuation_steps: int,
        max_log_density_change: float,
        residual_density_threshold: float,
        maximum_density: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        """Solve the Euler fixed point with optional residual backtracking."""

        stage_histories = []
        n_iter = 0
        n_evaluations = 0
        mixing_backtracks = 0
        final_relative_change = float("inf")
        current_mixing = mixing

        for stage_V_ext in _continued_fields(V_ext, continuation_steps):
            current_data = dict(data)
            current_data["rho"] = rho.detach().clone()
            current_data["V_ext"] = stage_V_ext
            evaluation = self.evaluate(current_data, compute_c1=True)
            n_evaluations += 1
            objective = _thermodynamic_objective(
                evaluation,
                fixed_N is not None,
                cell_volume,
                thermal_wavelength,
            )
            stage_history = [objective.detach().item()]

            for _ in range(max_iter):
                c1 = evaluation["c1"].detach()
                _, _, max_residual, rms_residual = _euler_residual(
                    rho,
                    c1,
                    stage_V_ext,
                    data["beta"],
                    thermal_wavelength,
                    mu,
                    residual_density_threshold,
                    maximum_density,
                )
                if _residuals_converged(
                    max_residual,
                    rms_residual,
                    tolerance_residual,
                    tolerance_rms_residual,
                ):
                    break

                if fixed_N is None:
                    current_log_density = torch.log(
                        torch.clamp(
                            rho,
                            min=torch.finfo(rho.dtype).tiny,
                        )
                    )
                    target_log_density = (
                        data["beta"] * (mu[None, :] - stage_V_ext)
                        + c1
                        - 3.0 * torch.log(thermal_wavelength)[None, :]
                    )
                    log_change = torch.clamp(
                        target_log_density - current_log_density,
                        min=-max_log_density_change,
                        max=max_log_density_change,
                    )
                    target_rho = torch.exp(
                        current_log_density + log_change
                    )
                    if maximum_density is not None:
                        target_rho = torch.minimum(
                            target_rho,
                            maximum_density[None, :],
                        )
                else:
                    logits = -data["beta"] * stage_V_ext + c1
                    target_rho = (
                        fixed_N[None, :]
                        * torch.softmax(logits, dim=0)
                        / cell_volume
                    )

                trial_mixing = current_mixing
                while True:
                    next_rho = (
                        (1.0 - trial_mixing) * rho
                        + trial_mixing * target_rho
                    )
                    if fixed_N is not None:
                        next_rho = _normalize_particle_numbers(
                            next_rho,
                            fixed_N,
                            cell_volume,
                            maximum_density=maximum_density,
                        )
                    elif maximum_density is not None:
                        next_rho = torch.minimum(
                            next_rho,
                            maximum_density[None, :],
                        )

                    trial_data = dict(data)
                    trial_data["rho"] = next_rho.detach().clone()
                    trial_data["V_ext"] = stage_V_ext
                    trial_evaluation = self.evaluate(
                        trial_data,
                        compute_c1=True,
                    )
                    n_evaluations += 1
                    _, _, _, trial_rms_residual = _euler_residual(
                        next_rho,
                        trial_evaluation["c1"].detach(),
                        stage_V_ext,
                        data["beta"],
                        thermal_wavelength,
                        mu,
                        residual_density_threshold,
                        maximum_density,
                    )

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

                relative_change = _maximum_relative_change(rho, next_rho)
                rho = next_rho.detach()
                evaluation = trial_evaluation
                n_iter += 1
                final_relative_change = relative_change
                objective = _thermodynamic_objective(
                    evaluation,
                    fixed_N is not None,
                    cell_volume,
                    thermal_wavelength,
                )
                stage_history.append(objective.detach().item())

                if adaptive_mixing:
                    current_mixing = min(
                        maximum_mixing,
                        trial_mixing * mixing_growth,
                    )
                else:
                    current_mixing = mixing
                if relative_change <= tolerance_change:
                    break

            stage_histories.append(stage_history)

        return {
            "rho": rho,
            "n_iter": n_iter,
            "n_evaluations": n_evaluations,
            "stage_objective_history": stage_histories,
            "final_relative_density_change": final_relative_change,
            "line_search_failures": 0,
            "mixing_backtracks": mixing_backtracks,
            "final_mixing": current_mixing,
        }

    def _move_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in data.items()
        }


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


def _continued_fields(
    V_ext: torch.Tensor,
    continuation_steps: int,
):
    """Yield external fields from zero to the requested full field."""

    if continuation_steps == 0:
        return (V_ext,)
    fractions = torch.linspace(
        0.0,
        1.0,
        steps=continuation_steps + 1,
        dtype=V_ext.dtype,
        device=V_ext.device,
    )
    return tuple(fraction * V_ext for fraction in fractions)


def _thermodynamic_objective(
    evaluation: Dict[str, torch.Tensor],
    fixed_particle_numbers: bool,
    cell_volume: torch.Tensor,
    thermal_wavelength: torch.Tensor,
) -> torch.Tensor:
    """Return the objective with high-precision scalar accumulation."""

    rho = evaluation["rho"]
    accumulation_dtype = (
        torch.float64
        if rho.dtype in (torch.float16, torch.bfloat16, torch.float32)
        else rho.dtype
    )
    rho = rho.to(accumulation_dtype)
    wavelength = thermal_wavelength.to(accumulation_dtype)
    log_density = _log_dimensionless_density(rho, wavelength)
    component_density = (
        rho * (log_density - 1.0)
        + rho
        * evaluation["beta"].to(accumulation_dtype)
        * evaluation["V_ext"].to(accumulation_dtype)
    )
    if not fixed_particle_numbers:
        component_density = component_density - (
            rho
            * evaluation["beta"].to(accumulation_dtype)
            * evaluation["mu"].to(accumulation_dtype)[None, :]
        )
    objective_density = (
        torch.sum(component_density, dim=-1)
        + evaluation["beta_free_energy_density"].to(accumulation_dtype)
    )
    return cell_volume.to(accumulation_dtype) * torch.sum(objective_density)


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
    accumulation_dtype = (
        torch.float64
        if rho.dtype in (torch.float16, torch.bfloat16, torch.float32)
        else rho.dtype
    )
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

    accumulation_dtype = (
        torch.float64
        if rho.dtype in (torch.float16, torch.bfloat16, torch.float32)
        else rho.dtype
    )
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

            # This is the capped proportional (KL) projection. Saturated
            # entries are removed and the remaining particle number is
            # redistributed proportionally over unsaturated entries.
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

    normalized = normalized_accumulated.to(rho.dtype)

    # Casting a normalized density back to float32 can leave an O(n_grid)
    # summation error. Correct it on the largest entry of each component so
    # line-search directions remain tangent to the fixed-N constraint.
    normalized = normalized.clone()
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
    """Return the physical residual, including an optional upper-bound KKT."""

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

    # At rho == maximum_density, a negative residual points toward increasing
    # rho and is blocked by the upper constraint; only a positive residual is
    # a KKT violation. Interior points retain the ordinary equality residual.
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


def _module_device(module: nn.Module) -> torch.device:
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(module.buffers(), None)
    return torch.device("cpu") if buffer is None else buffer.device
