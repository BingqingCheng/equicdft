"""Evaluate learned grid functionals and minimize thermodynamic objectives."""

import math
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from ._solver_numerics import (
    _component_tensor,
    _euler_residual,
    _log_dimensionless_density,
    _maximum_relative_change,
    _mirror_descent_trial,
    _normalize_particle_numbers,
    _residuals_converged,
    _thermodynamic_objective,
)


class GridSolver:
    """Evaluate prescribed densities or solve for an equilibrium density.

    The wrapped model supplies the intrinsic excess functional. This class
    adds the exact ideal-gas, external-potential, and chemical-potential terms.

    ``evaluate(data)`` requires ``rho`` and returns all quantities supported by
    the available fields. ``solve(data)`` requires ``V_ext`` and either ``mu``
    or fixed ``particle_numbers``. Equilibrium solving supports one complete,
    unbatched field; prescribed-density evaluation also supports batches.
    An optional Boolean ``mask`` has shape ``[..., n_grid]``; true entries are
    hard exclusions whose density is fixed to zero and omitted from residuals.
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
        excluded_mask, _ = _hard_wall_masks(data, rho)
        excluded_density = excluded_mask[..., None].expand_as(rho)
        if torch.any(rho[excluded_density] != 0.0).item():
            raise ValueError("rho must be zero at masked grid points")

        outputs = self.model(data, compute_c1=compute_c1)
        result = {
            key: data[key]
            for key in (
                "rho",
                "V_ext",
                "mu",
                "temperature",
                "beta",
                "mask",
            )
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
                residual = (
                    result["local_chemical_potential"]
                    - beta * data["mu"][..., None, :]
                )
                result["euler_lagrange_residual"] = residual.masked_fill(
                    excluded_density,
                    0.0,
                )

        return result

    def solve(
        self,
        data: Dict[str, Any],
        initial_rho: Optional[torch.Tensor] = None,
        particle_numbers: Optional[
            Union[float, Sequence[float], torch.Tensor]
        ] = None,
        method: str = "euler",
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
        max_log_density_change: float = 2.0,
        residual_density_threshold: Optional[float] = None,
        maximum_density: Optional[
            Union[float, Sequence[float], torch.Tensor]
        ] = None,
        beta_multiplier: float = 0.0,
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

        A true entry in ``data["mask"]`` is an inaccessible grid point,
        mathematically equivalent to an infinite external potential. Masked
        densities remain exactly zero, fixed particle numbers are normalized
        over accessible points, and masked residuals do not enter convergence.
        The mask does not alter the periodic neighborhood topology.

        When no density is supplied through ``initial_rho`` or ``data["rho"]``,
        the initial profile is proportional to
        ``exp(-beta_multiplier * beta * V_ext)``. The default
        ``beta_multiplier=0`` exactly recovers the former uniform
        initialization, including its chemical-potential-dependent amplitude.
        ``beta_multiplier=1`` is the physical ideal-gas field profile, and
        intermediate values smoothly temper its spatial modulation. An
        explicitly supplied density always takes precedence.

        The default ``method="euler"`` uses a damped Euler--Lagrange fixed-point
        iteration. By default, its mixing is increased after an improving
        density-weighted RMS residual and backtracked after a worsening
        trial. Set ``adaptive_mixing=False`` to use a fixed mixing value.
        Both methods act directly on the supplied external field and use the
        physical projected functional gradient as a convergence diagnostic.

        ``tolerance_residual`` bounds the largest active-grid residual. When
        ``tolerance_rms_residual`` is supplied, its density-weighted RMS bound
        must also be satisfied. ``residual_density_threshold`` excludes
        statistically unresolved low-density voxels from both diagnostics.
        """

        if method not in ("minimize", "euler"):
            raise ValueError("method must be 'minimize' or 'euler'")
        beta_multiplier = float(beta_multiplier)
        if not math.isfinite(beta_multiplier) or beta_multiplier < 0.0:
            raise ValueError(
                "beta_multiplier must be finite and nonnegative"
            )
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")
        tolerance_residual = float(tolerance_residual)
        if tolerance_rms_residual is not None:
            tolerance_rms_residual = float(tolerance_rms_residual)
        max_log_density_change = float(max_log_density_change)
        if tolerance_residual <= 0.0:
            raise ValueError("tolerance_residual must be positive")
        if (
            tolerance_rms_residual is not None
            and tolerance_rms_residual <= 0.0
        ):
            raise ValueError("tolerance_rms_residual must be positive")
        if max_log_density_change <= 0.0:
            raise ValueError("max_log_density_change must be positive")
        if method == "minimize":
            step_size = float(step_size)
            minimum_step_size = float(minimum_step_size)
            line_search_factor = float(line_search_factor)
            armijo_factor = float(armijo_factor)
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
                raise ValueError(
                    "armijo_factor must be in the interval [0, 1)"
                )
        else:
            tolerance_change = float(tolerance_change)
            mixing = float(mixing)
            minimum_mixing = float(minimum_mixing)
            maximum_mixing = float(maximum_mixing)
            mixing_growth = float(mixing_growth)
            mixing_backtrack_factor = float(mixing_backtrack_factor)
            if tolerance_change <= 0.0:
                raise ValueError("tolerance_change must be positive")
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

        data = self._move_data(data)
        V_ext = data["V_ext"]
        if V_ext.ndim != 2:
            raise ValueError("solve currently accepts one unbatched field")
        excluded_mask, accessible_mask = _hard_wall_masks(data, V_ext)

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
                    cell_volume * accessible_mask.sum() * density_cap
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

        if initial_rho is None and "rho" in data:
            initial_rho = data["rho"]
        if initial_rho is not None:
            initial_rho = torch.as_tensor(
                initial_rho,
                dtype=V_ext.dtype,
                device=V_ext.device,
            )
            if initial_rho.shape != V_ext.shape:
                raise ValueError("initial_rho must have the same shape as V_ext")
            if torch.any(initial_rho[accessible_mask] <= 0.0).item():
                raise ValueError(
                    "initial_rho must be positive on accessible grid points"
                )
            rho = initial_rho.detach().clone().masked_fill(
                excluded_mask[:, None],
                0.0,
            )
        elif fixed_N is None:
            log_rho = data["beta"] * (
                mu[None, :] - beta_multiplier * V_ext
            )
            rho = torch.exp(log_rho) / thermal_wavelength[None, :] ** 3
            rho = rho.masked_fill(excluded_mask[:, None], 0.0)
        else:
            initial_logits = (
                -beta_multiplier * data["beta"] * V_ext
            ).masked_fill(excluded_mask[:, None], -torch.inf)
            boltzmann_weights = torch.softmax(
                initial_logits,
                dim=0,
            )
            rho = torch.where(
                accessible_mask[:, None],
                torch.clamp(
                    boltzmann_weights,
                    min=torch.finfo(V_ext.dtype).tiny,
                ),
                torch.zeros_like(boltzmann_weights),
            )

        if fixed_N is not None:
            rho = _normalize_particle_numbers(
                rho,
                fixed_N,
                cell_volume,
                maximum_density=density_cap,
                accessible_mask=accessible_mask,
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
                    max_log_density_change=max_log_density_change,
                    residual_density_threshold=residual_density_threshold,
                    maximum_density=density_cap,
                    accessible_mask=accessible_mask,
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
                    max_log_density_change=max_log_density_change,
                    residual_density_threshold=residual_density_threshold,
                    maximum_density=density_cap,
                    accessible_mask=accessible_mask,
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
                accessible_mask,
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
        result["solver_beta_multiplier"] = beta_multiplier
        result["n_iter"] = state["n_iter"]
        result["n_evaluations"] = state["n_evaluations"]
        result["objective_history"] = state["objective_history"]
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
        max_log_density_change: float,
        residual_density_threshold: float,
        maximum_density: Optional[torch.Tensor],
        accessible_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        """Minimize the thermodynamic objective by mirror descent."""

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
            if not objective_history:
                objective_history.append(objective.item())

            residual, _, max_residual, rms_residual = _euler_residual(
                rho,
                evaluation["c1"].detach(),
                V_ext,
                data["beta"],
                thermal_wavelength,
                mu,
                residual_density_threshold,
                maximum_density,
                accessible_mask,
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
                + data["beta"] * V_ext
                - evaluation["c1"].detach()
            )
            if fixed_N is None:
                line_search_gradient = (
                    line_search_gradient - data["beta"] * mu[None, :]
                )
            line_search_gradient = torch.where(
                accessible_mask[:, None],
                line_search_gradient,
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
                    cell_volume,
                    max_log_density_change,
                    maximum_density,
                    accessible_mask,
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
                trial_data["V_ext"] = V_ext
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
        max_log_density_change: float,
        residual_density_threshold: float,
        maximum_density: Optional[torch.Tensor],
        accessible_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        """Solve the Euler fixed point with optional residual backtracking."""

        objective_history = []
        n_iter = 0
        n_evaluations = 0
        mixing_backtracks = 0
        final_relative_change = float("inf")
        current_mixing = mixing
        current_data = dict(data)
        current_data["rho"] = rho.detach().clone()
        current_data["V_ext"] = V_ext
        evaluation = self.evaluate(current_data, compute_c1=True)
        n_evaluations += 1
        objective = _thermodynamic_objective(
            evaluation,
            fixed_N is not None,
            cell_volume,
            thermal_wavelength,
        )
        objective_history.append(objective.detach().item())

        for _ in range(max_iter):
            c1 = evaluation["c1"].detach()
            _, _, max_residual, rms_residual = _euler_residual(
                rho,
                c1,
                V_ext,
                data["beta"],
                thermal_wavelength,
                mu,
                residual_density_threshold,
                maximum_density,
                accessible_mask,
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
                target_log_density = (
                    data["beta"] * (mu[None, :] - V_ext)
                    + c1
                    - 3.0 * torch.log(thermal_wavelength)[None, :]
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
                        accessible_mask=accessible_mask,
                    )
                elif maximum_density is not None:
                    next_rho = torch.minimum(
                        next_rho,
                        maximum_density[None, :],
                    )

                trial_data = dict(data)
                trial_data["rho"] = next_rho.detach().clone()
                trial_data["V_ext"] = V_ext
                trial_evaluation = self.evaluate(
                    trial_data,
                    compute_c1=True,
                )
                n_evaluations += 1
                _, _, _, trial_rms_residual = _euler_residual(
                    next_rho,
                    trial_evaluation["c1"].detach(),
                    V_ext,
                    data["beta"],
                    thermal_wavelength,
                    mu,
                    residual_density_threshold,
                    maximum_density,
                    accessible_mask,
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
            objective_history.append(objective.detach().item())

            if adaptive_mixing:
                current_mixing = min(
                    maximum_mixing,
                    trial_mixing * mixing_growth,
                )
            else:
                current_mixing = mixing
            if relative_change <= tolerance_change:
                break

        return {
            "rho": rho,
            "n_iter": n_iter,
            "n_evaluations": n_evaluations,
            "objective_history": objective_history,
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


def _module_device(module: nn.Module) -> torch.device:
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(module.buffers(), None)
    return torch.device("cpu") if buffer is None else buffer.device


def _hard_wall_masks(
    data: Dict[str, Any],
    field: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return validated excluded and accessible masks for a grid field."""

    expected_shape = field.shape[:-1]
    mask = data.get("mask")
    if mask is None:
        excluded = torch.zeros(
            expected_shape,
            dtype=torch.bool,
            device=field.device,
        )
    else:
        if not torch.is_tensor(mask) or mask.dtype != torch.bool:
            raise TypeError("mask must be a Boolean tensor")
        if mask.shape != expected_shape:
            raise ValueError("mask must have shape field.shape[:-1]")
        excluded = mask.to(device=field.device)
    if torch.any(torch.all(excluded, dim=-1)).item():
        raise ValueError("mask must leave at least one accessible grid point")
    return excluded, ~excluded
