"""Evaluate learned grid functionals and minimize thermodynamic objectives."""

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from ._argument_checks import (
    boolean,
    finite_scalar,
    nonnegative_scalar,
    positive_integer,
    positive_scalar,
)
from ._grid import voxel_volume
from ._solver_symmetry import _HomogeneousDensityProjection, _normalize_homogeneous_axes
from ._solver_numerics import (
    _anderson_log_density_candidate,
    _canonical_density_residual,
    _component_tensor,
    _euler_residual,
    _fixed_dipole_bound_kkt,
    _maximum_relative_change,
    _mirror_descent_trial,
    _polarization_residual_norms,
    _project_density_constraints,
    _project_vector_norm,
    _residuals_converged,
    _thermodynamic_objective,
)
from .energy import (
    density_weighted_integral,
    ideal_free_energy,
    log_dimensionless_density,
)
from .polarization_ideal import (
    FixedDipoleIdeal,
    _positive_components,
    dipole_alignment,
    inverse_langevin,
    log_sinhc,
)


class GridSolver:
    """Evaluate or minimize scalar-density and fixed-dipole functionals.

    The wrapped model supplies the intrinsic excess functional. This class
    adds the appropriate ideal, external-field, and chemical-potential terms.
    Supplying ``dipole_magnitude`` explicitly enables the freely rotating,
    fixed-dipole ideal reference and coupled ``(rho, dipole_density)`` solve.

    ``evaluate(data)`` requires ``rho`` and returns all quantities supported by
    the available fields. ``solve(data)`` requires ``V_ext`` and either ``mu``
    or fixed ``particle_numbers``. Fixed-dipole solving additionally requires
    ``E_ext`` and fixed particle numbers. Equilibrium solving supports one
    complete, unbatched field; scalar prescribed-density evaluation also
    supports batches.
    An optional Boolean ``excluded_mask`` has shape ``[..., n_grid]``; true
    entries are hard exclusions whose density is fixed to zero and omitted
    from residuals. Electrode coordinates do not infer exclusions; supply the
    entire inaccessible volume explicitly in ``excluded_mask``.
    """

    def __init__(
        self,
        model: Optional[nn.Module],
        device: Optional[Union[str, torch.device]] = None,
        *,
        dipole_magnitude: Optional[
            Union[float, Sequence[float], torch.Tensor]
        ] = None,
    ) -> None:
        if model is not None and not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module or None")
        if model is None and dipole_magnitude is None:
            raise TypeError(
                "model may be None only when dipole_magnitude is supplied"
            )
        if (
            model is not None
            and getattr(model, "requires_dipole_density", False)
            and dipole_magnitude is None
        ):
            raise ValueError(
                "dipole-density models require an explicit dipole_magnitude"
            )
        self.model = model
        self.fixed_dipole_ideal = (
            None
            if dipole_magnitude is None
            else FixedDipoleIdeal(dipole_magnitude)
        )
        self.device = (
            (_module_device(model) if model is not None else None)
            if device is None
            else torch.device(device)
        )
        if self.model is not None and self.device is not None:
            self.model.to(self.device)
        if self.fixed_dipole_ideal is not None and self.device is not None:
            self.fixed_dipole_ideal.to(self.device)

    def evaluate(
        self,
        data: Dict[str, Any],
        compute_c1: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate a supplied density and its available thermodynamics."""

        if self.fixed_dipole_ideal is not None:
            return self._evaluate_fixed_dipoles(data, compute_c1)

        data = self._move_data(data)
        rho = data["rho"]
        if torch.any(rho < 0.0).item():
            raise ValueError("rho must be nonnegative")
        excluded_mask, _ = _resolve_accessibility_masks(data, rho)
        excluded_density = excluded_mask[..., None].expand_as(rho)
        if torch.any(rho[excluded_density] != 0.0).item():
            raise ValueError("rho must be zero at excluded grid points")

        outputs = self.model(data, compute_c1=compute_c1)
        result = {
            key: data[key]
            for key in (
                "rho",
                "V_ext",
                "mu",
                "temperature",
                "beta",
                "excluded_mask",
                "grid_center",
            )
            if key in data
        }
        result.update(outputs)

        volume_element = voxel_volume(data["grid_spacing"])
        thermal_wavelength = data.get(
            "thermal_wavelength",
            torch.ones(
                rho.shape[-1],
                dtype=rho.dtype,
                device=rho.device,
            ),
        )
        result["beta_F_id"] = ideal_free_energy(
            rho,
            thermal_wavelength,
            volume_element,
        )
        result["beta_F"] = result["beta_F_id"] + result["beta_F_exc"]

        if "V_ext" in data:
            beta = data["beta"][..., None, None]
            result["beta_V_ext"] = density_weighted_integral(
                rho,
                beta * data["V_ext"],
                volume_element,
            )

        if "V_ext" in data and "mu" in data:
            beta = data["beta"][..., None, None]
            beta_mu_N = density_weighted_integral(
                rho,
                beta * data["mu"][..., None, :],
                volume_element,
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
        method: Optional[str] = None,
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
        anderson: bool = False,
        anderson_history: int = 5,
        anderson_regularization: float = 1.0e-8,
        anderson_damping: float = 1.0,
        max_log_density_change: float = 2.0,
        residual_density_threshold: Optional[float] = None,
        maximum_density: Optional[
            Union[float, Sequence[float], torch.Tensor]
        ] = None,
        maximum_polarization_fraction: Optional[
            Union[float, Sequence[float], torch.Tensor]
        ] = None,
        beta_multiplier: float = 0.0,
        initial_polarization: Optional[torch.Tensor] = None,
        fixed_field: Optional[str] = None,
        homogeneous_axes: Optional[Union[str, Sequence[Union[int, str]]]] = None,
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

        ``maximum_polarization_fraction`` optionally imposes the local bound
        ``|P_i| <= q_max,i * m_i * rho_i``, with one dimensionless value in
        ``(0, 1)`` per component. This is a density-aware restriction on the
        orientational alignment rather than an absolute dipole-density cap.
        Coupled fixed-dipole updates are projected onto the bound and their
        reported residual uses the corresponding KKT condition. The option is
        unavailable for scalar-density solving.

        A true entry in ``data["excluded_mask"]`` is an inaccessible grid
        point, mathematically equivalent to an infinite external potential.
        Excluded densities remain exactly zero, fixed particle numbers are
        normalized over accessible points, and excluded residuals do not enter
        convergence. The exclusion mask does not alter the periodic
        neighborhood topology.

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
        ``anderson=True`` additionally forms a log-density Anderson trial from
        recent fixed-point residuals. The accelerated trial is projected onto
        the same accessibility, density-bound, and particle-number constraints
        and is accepted only when neither its maximum nor its density-weighted
        RMS physical Euler residual exceeds that of the scalar-mixed fallback.
        A trial is attempted after each full history window. Rejected or
        numerically singular trials leave the established scalar-mixing step
        unchanged and reset the history.
        Anderson mixing acts directly on the supplied external field and uses
        the physical projected functional gradient as a convergence
        diagnostic.

        ``tolerance_residual`` bounds the largest active-grid residual. When
        ``tolerance_rms_residual`` is supplied, its density-weighted RMS bound
        must also be satisfied. ``residual_density_threshold`` excludes
        statistically unresolved low-density voxels from both diagnostics.

        ``homogeneous_axes='x'`` makes liquid density constant along x;
        ``homogeneous_axes=['x', 'y']`` leaves only z dependence. Any distinct
        selection of 'x', 'y', 'z' is accepted; None or an empty sequence is
        unrestricted.
        Names refer to grid axes, not component channels. The original
        ``homogeneous_axes=(0, 1)`` integer form remains supported (x=0, y=1,
        z=2); names and indices may be mixed. This opt-in constraint requires
        a complete rectangular grid whose accessibility is invariant along
        those axes.
        The full model (including metal response) and energy are unchanged;
        the gradient is averaged before each update. Standard residual and
        convergence keys then describe the constrained problem. Additional
        ``full_*euler_lagrange_residual`` and ``full_converged`` results report
        unrestricted stationarity. Available only for density-only minimize.

        With an explicit ``dipole_magnitude``, both methods optimize density
        and dipole density. ``method="euler"`` uses the coupled orientational
        Euler expression with simultaneous adaptive mixing. The minimizer uses
        the fixed-dipole entropy mirror map and an Armijo line search, preserving
        ``rho > 0`` and ``|P| < m*rho``. When ``method`` is omitted, scalar
        solving defaults to Euler and fixed-dipole solving to minimization.

        For fixed-dipole minimization, ``fixed_field="dipole_density"`` holds
        the supplied initial polarization fixed and minimizes only over
        density. ``fixed_field="rho"`` holds the supplied initial density
        fixed and minimizes only over polarization. Convergence then uses only
        the residual of the relaxed field; the full coupled residual remains
        available as ``full_maximum_residual`` and ``full_rms_residual``.
        """

        if method is None:
            method = (
                "minimize"
                if self.fixed_dipole_ideal is not None
                else "euler"
            )
        if method not in ("minimize", "euler"):
            raise ValueError("method must be 'minimize' or 'euler'")
        homogeneous_axes = _normalize_homogeneous_axes(homogeneous_axes)
        if homogeneous_axes and method != "minimize":
            raise ValueError("homogeneous_axes is currently available only with method='minimize'")
        if self.fixed_dipole_ideal is not None:
            if homogeneous_axes:
                raise ValueError("homogeneous_axes is not supported with fixed dipoles")
            return self._solve_fixed_dipoles(
                data=data,
                initial_rho=initial_rho,
                initial_polarization=initial_polarization,
                particle_numbers=particle_numbers,
                method=method,
                max_iter=max_iter,
                tolerance_residual=tolerance_residual,
                tolerance_rms_residual=tolerance_rms_residual,
                tolerance_change=tolerance_change,
                step_size=step_size,
                minimum_step_size=minimum_step_size,
                line_search_factor=line_search_factor,
                armijo_factor=armijo_factor,
                mixing=mixing,
                adaptive_mixing=adaptive_mixing,
                minimum_mixing=minimum_mixing,
                maximum_mixing=maximum_mixing,
                mixing_growth=mixing_growth,
                mixing_backtrack_factor=mixing_backtrack_factor,
                anderson=anderson,
                residual_density_threshold=residual_density_threshold,
                maximum_density=maximum_density,
                maximum_polarization_fraction=maximum_polarization_fraction,
                beta_multiplier=beta_multiplier,
                fixed_field=fixed_field,
            )
        if fixed_field is not None:
            raise ValueError("fixed_field requires an explicit dipole_magnitude")
        if initial_polarization is not None:
            raise ValueError(
                "initial_polarization requires an explicit dipole_magnitude"
            )
        if maximum_polarization_fraction is not None:
            raise ValueError(
                "maximum_polarization_fraction requires an explicit "
                "dipole_magnitude"
            )
        beta_multiplier = nonnegative_scalar(
            beta_multiplier,
            "beta_multiplier",
        )
        max_iter = positive_integer(max_iter, "max_iter")
        tolerance_residual = positive_scalar(
            tolerance_residual,
            "tolerance_residual",
        )
        if tolerance_rms_residual is not None:
            tolerance_rms_residual = positive_scalar(
                tolerance_rms_residual,
                "tolerance_rms_residual",
            )
        max_log_density_change = positive_scalar(
            max_log_density_change,
            "max_log_density_change",
        )
        if method == "minimize":
            step_size = positive_scalar(step_size, "step_size")
            minimum_step_size = positive_scalar(
                minimum_step_size,
                "minimum_step_size",
            )
            line_search_factor = finite_scalar(
                line_search_factor,
                "line_search_factor",
            )
            armijo_factor = finite_scalar(armijo_factor, "armijo_factor")
            if minimum_step_size > step_size:
                raise ValueError(
                    "minimum_step_size must be no larger than step_size"
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
            tolerance_change = positive_scalar(
                tolerance_change,
                "tolerance_change",
            )
            mixing = positive_scalar(mixing, "mixing")
            minimum_mixing = positive_scalar(
                minimum_mixing,
                "minimum_mixing",
            )
            maximum_mixing = positive_scalar(
                maximum_mixing,
                "maximum_mixing",
            )
            mixing_growth = positive_scalar(
                mixing_growth,
                "mixing_growth",
            )
            mixing_backtrack_factor = positive_scalar(
                mixing_backtrack_factor,
                "mixing_backtrack_factor",
            )
            if mixing > 1.0:
                raise ValueError("mixing must be in the interval (0, 1]")
            adaptive_mixing = boolean(adaptive_mixing, "adaptive_mixing")
            anderson = boolean(anderson, "anderson")
            if adaptive_mixing:
                if minimum_mixing > mixing:
                    raise ValueError(
                        "minimum_mixing must be no larger than mixing"
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
            if anderson:
                anderson_history = positive_integer(
                    anderson_history,
                    "anderson_history",
                )
                if anderson_history < 2:
                    raise ValueError("anderson_history must be at least two")
                anderson_regularization = nonnegative_scalar(
                    anderson_regularization,
                    "anderson_regularization",
                )
                anderson_damping = positive_scalar(
                    anderson_damping,
                    "anderson_damping",
                )
                if anderson_damping > 1.0:
                    raise ValueError(
                        "anderson_damping must be no larger than one"
                    )

        data = self._move_data(data)
        V_ext = data["V_ext"]
        if V_ext.ndim != 2:
            raise ValueError("solve currently accepts one unbatched field")
        excluded_mask, accessible_mask = _resolve_accessibility_masks(
            data,
            V_ext,
        )
        projection = (
            _HomogeneousDensityProjection(data, homogeneous_axes, accessible_mask)
            if homogeneous_axes else None
        )

        n_types = V_ext.shape[-1]
        thermal_wavelength = _component_tensor(
            data.get("thermal_wavelength", 1.0),
            n_types,
            V_ext,
            "thermal_wavelength",
        )
        volume_element = voxel_volume(data["grid_spacing"])

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
                    volume_element * accessible_mask.sum() * density_cap
                )
                if torch.any(fixed_N > maximum_particle_numbers).item():
                    raise ValueError(
                        "particle_numbers are infeasible under maximum_density"
                    )
            mu = None

        if residual_density_threshold is None:
            residual_density_threshold = 0.0
        else:
            residual_density_threshold = nonnegative_scalar(
                residual_density_threshold,
                "residual_density_threshold",
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

        if projection is not None:
            rho = projection.average(rho)
        rho = _project_density_constraints(
            rho,
            fixed_N,
            volume_element,
            density_cap,
            accessible_mask,
        )

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
                    voxel_volume=volume_element,
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
                    projection=projection,
                )
            else:
                state = self._solve_euler(
                    data=data,
                    rho=rho,
                    V_ext=V_ext,
                    thermal_wavelength=thermal_wavelength,
                    mu=mu,
                    fixed_N=fixed_N,
                    voxel_volume=volume_element,
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
                    anderson=anderson,
                    anderson_history=anderson_history,
                    anderson_regularization=anderson_regularization,
                    anderson_damping=anderson_damping,
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
        if projection is not None:
            result["full_euler_lagrange_residual"] = residual
            result["full_max_euler_lagrange_residual"] = max_residual
            result["full_rms_euler_lagrange_residual"] = rms_residual
            result["full_converged"] = _residuals_converged(
                max_residual, rms_residual, tolerance_residual, tolerance_rms_residual,
            )
            residual, chemical_potential, max_residual, rms_residual = _euler_residual(
                result["rho"], projection.average(result["c1"]),
                projection.average(V_ext), data["beta"], thermal_wavelength,
                mu, residual_density_threshold, density_cap, accessible_mask,
            )
            result["solver_homogeneous_axes"] = list(homogeneous_axes)
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
        result["solver_anderson"] = state.get("solver_anderson", False)
        result["anderson_attempts"] = state.get("anderson_attempts", 0)
        result["anderson_accepted"] = state.get("anderson_accepted", 0)
        result["anderson_rejected"] = state.get("anderson_rejected", 0)
        result["anderson_resets"] = state.get("anderson_resets", 0)
        return result

    def _minimize(
        self,
        data: Dict[str, Any],
        rho: torch.Tensor,
        V_ext: torch.Tensor,
        thermal_wavelength: torch.Tensor,
        mu: Optional[torch.Tensor],
        fixed_N: Optional[torch.Tensor],
        voxel_volume: torch.Tensor,
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
        projection: Optional[_HomogeneousDensityProjection] = None,
    ) -> Dict[str, Any]:
        """Minimize the thermodynamic objective by mirror descent."""

        objective_history = []
        n_iter = 0
        n_evaluations = 0
        line_search_failures = 0
        final_relative_change = float("inf")

        projected_V_ext = V_ext if projection is None else projection.average(V_ext)
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
                voxel_volume,
                thermal_wavelength,
            ).detach()
            if not torch.isfinite(objective).item():
                raise ValueError(
                    "the thermodynamic objective became non-finite"
                )
            if not objective_history:
                objective_history.append(objective.item())

            projected_c1 = evaluation["c1"].detach()
            if projection is not None:
                projected_c1 = projection.average(projected_c1)
            residual, _, max_residual, rms_residual = _euler_residual(
                rho,
                projected_c1,
                projected_V_ext,
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
                log_dimensionless_density(rho, thermal_wavelength)
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
                trial_evaluation = self.evaluate(
                    trial_data,
                    compute_c1=False,
                )
                n_evaluations += 1
                trial_objective = _thermodynamic_objective(
                    trial_evaluation,
                    fixed_N is not None,
                    voxel_volume,
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
        voxel_volume: torch.Tensor,
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
        anderson: bool,
        anderson_history: int,
        anderson_regularization: float,
        anderson_damping: float,
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
        evaluation = self.evaluate(current_data, compute_c1=True)
        n_evaluations += 1
        objective = _thermodynamic_objective(
            evaluation,
            fixed_N is not None,
            voxel_volume,
            thermal_wavelength,
        )
        objective_history.append(objective.detach().item())

        def evaluate_trial(trial_rho: torch.Tensor):
            trial_data = dict(data)
            trial_data["rho"] = trial_rho.detach().clone()
            trial_data["V_ext"] = V_ext
            trial_evaluation = self.evaluate(trial_data, compute_c1=True)
            _, _, trial_max_residual, trial_rms_residual = _euler_residual(
                trial_rho,
                trial_evaluation["c1"].detach(),
                V_ext,
                data["beta"],
                thermal_wavelength,
                mu,
                residual_density_threshold,
                maximum_density,
                accessible_mask,
            )
            return (
                trial_evaluation,
                trial_max_residual,
                trial_rms_residual,
            )

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
            "solver_anderson": anderson,
            "anderson_attempts": anderson_attempts,
            "anderson_accepted": anderson_accepted,
            "anderson_rejected": anderson_rejected,
            "anderson_resets": anderson_resets,
        }

    def _fixed_dipole_inputs(self, data):
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
        if self.model is not None and hasattr(self.model, "boltzmann_constant"):
            temperature = torch.as_tensor(
                data["temperature"],
                dtype=potential.dtype,
                device=potential.device,
            )
            k_b = self.model.boltzmann_constant.to(potential)
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
            self.fixed_dipole_ideal.dipole_magnitude,
            potential,
            "dipole_magnitude",
        ).expand(potential.shape[-1])
        wavelength = _component_tensor(
            data.get("thermal_wavelength", 1.0),
            potential.shape[-1],
            potential,
            "thermal_wavelength",
        )
        if not torch.isfinite(wavelength).all() or torch.any(wavelength <= 0):
            raise ValueError("thermal_wavelength values must be finite and positive")
        return beta, voxel_volume(spacing), moment, wavelength, excluded, accessible

    @torch.enable_grad()
    def _evaluate_fixed_dipoles(self, data, compute_derivatives=True):
        """Evaluate ideal, excess and external terms for independent rho/P."""

        data = self._move_data(data)
        beta, volume, moment, wavelength, excluded, accessible = (
            self._fixed_dipole_inputs(data)
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

        ideal = self.fixed_dipole_ideal(
            rho[accessible],
            polarization[accessible],
            volume,
            thermal_wavelength=wavelength,
        )
        excess = rho.new_zeros(())
        if self.model is not None:
            model_data = dict(data, rho=rho, dipole_density=polarization)
            model_outputs = self.model(
                model_data,
                compute_c1=False,
                compute_c2=False,
                compute_polarization_derivative=False,
            )
            excess = model_outputs["beta_F_exc"]
            if excess.ndim:
                raise ValueError(
                    "model must return scalar beta_F_exc for one field"
                )
        external = beta * volume * (
            (rho * potential).sum()
            - (polarization * data["E_ext"]).sum()
        )
        beta_f = ideal["beta_F_id"] + excess
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
        result.update(
            rho=rho,
            dipole_density=polarization,
            beta_F_id=ideal["beta_F_id"],
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
            if excess.requires_grad:
                derivative_rho, derivative_p = torch.autograd.grad(
                    excess,
                    (rho, polarization),
                    allow_unused=True,
                )
            else:
                derivative_rho = derivative_p = None
            derivative_rho = (
                torch.zeros_like(rho)
                if derivative_rho is None
                else derivative_rho
            )
            derivative_p = (
                torch.zeros_like(polarization)
                if derivative_p is None
                else derivative_p
            )
            c1 = -derivative_rho / volume
            polarization_derivative = derivative_p / volume
            ideal_density = torch.zeros_like(rho)
            ideal_polarization = torch.zeros_like(polarization)
            ideal_density[accessible] = ideal["density_derivative"]
            ideal_polarization[accessible] = ideal[
                "polarization_derivative"
            ]
            local_mu = ideal_density - c1 + beta * potential
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
                ideal_polarization
                + polarization_derivative
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
                local_chemical_potential=local_mu,
                equilibrium_chemical_potential=chemical_potential,
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

    def _fixed_dipole_converged(
        self, result, tolerance_residual, tolerance_rms_residual
    ):
        return bool(
            result["maximum_residual"] <= tolerance_residual
            and (
                tolerance_rms_residual is None
                or result["rms_residual"] <= tolerance_rms_residual
            )
        )

    def _apply_fixed_dipole_bounds(
        self,
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
                    result["local_chemical_potential"],
                    0.0,
                    maximum_density,
                    accessible,
                    minimum_density=minimum_density,
                )
            )
            result["equilibrium_chemical_potential"] = chemical_potential
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
                result["local_chemical_potential"],
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
        result["equilibrium_chemical_potential"] = chemical_potential
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
    def _solve_fixed_dipoles(
        self,
        data,
        initial_rho,
        initial_polarization,
        particle_numbers,
        method,
        max_iter,
        tolerance_residual,
        tolerance_rms_residual,
        tolerance_change,
        step_size,
        minimum_step_size,
        line_search_factor,
        armijo_factor,
        mixing,
        adaptive_mixing,
        minimum_mixing,
        maximum_mixing,
        mixing_growth,
        mixing_backtrack_factor,
        anderson,
        residual_density_threshold,
        maximum_density,
        maximum_polarization_fraction,
        beta_multiplier,
        fixed_field,
    ):
        """Solve fixed-dipole Euler equations at fixed particle numbers."""

        if particle_numbers is None:
            raise ValueError(
                "fixed-dipole solving currently requires particle_numbers"
            )
        if anderson:
            raise ValueError(
                "Anderson acceleration is not yet supported with fixed dipoles"
            )
        if residual_density_threshold not in (None, 0, 0.0):
            raise ValueError(
                "residual_density_threshold is not supported with fixed dipoles"
            )
        if fixed_field not in (None, "rho", "dipole_density"):
            raise ValueError(
                "fixed_field must be None, 'rho', or 'dipole_density'"
            )
        if fixed_field is not None and method != "minimize":
            raise ValueError("fixed_field is supported only with method='minimize'")
        max_iter = positive_integer(max_iter, "max_iter")
        tolerance_residual = positive_scalar(
            tolerance_residual, "tolerance_residual"
        )
        if tolerance_rms_residual is not None:
            tolerance_rms_residual = positive_scalar(
                tolerance_rms_residual, "tolerance_rms_residual"
            )
        beta_multiplier = nonnegative_scalar(beta_multiplier, "beta_multiplier")
        if method == "minimize":
            step_size = positive_scalar(step_size, "step_size")
            minimum_step_size = positive_scalar(
                minimum_step_size, "minimum_step_size"
            )
            line_search_factor = finite_scalar(
                line_search_factor, "line_search_factor"
            )
            armijo_factor = finite_scalar(armijo_factor, "armijo_factor")
            if minimum_step_size > step_size:
                raise ValueError(
                    "minimum_step_size must be no larger than step_size"
                )
            if not 0 < line_search_factor < 1:
                raise ValueError("line_search_factor must lie in (0, 1)")
            if not 0 <= armijo_factor < 1:
                raise ValueError("armijo_factor must lie in [0, 1)")
        else:
            tolerance_change = positive_scalar(
                tolerance_change, "tolerance_change"
            )
            mixing = positive_scalar(mixing, "mixing")
            minimum_mixing = positive_scalar(
                minimum_mixing, "minimum_mixing"
            )
            maximum_mixing = positive_scalar(
                maximum_mixing, "maximum_mixing"
            )
            mixing_growth = positive_scalar(mixing_growth, "mixing_growth")
            mixing_backtrack_factor = positive_scalar(
                mixing_backtrack_factor, "mixing_backtrack_factor"
            )
            adaptive_mixing = boolean(adaptive_mixing, "adaptive_mixing")
            if not 0 < mixing <= 1:
                raise ValueError("mixing must lie in (0, 1]")
            if not minimum_mixing <= mixing <= maximum_mixing <= 1:
                raise ValueError(
                    "require minimum_mixing <= mixing <= maximum_mixing <= 1"
                )
            if mixing_growth < 1:
                raise ValueError("mixing_growth must be at least one")
            if not 0 < mixing_backtrack_factor < 1:
                raise ValueError("mixing_backtrack_factor must lie in (0, 1)")

        data = self._move_data(data)
        beta, volume, moment, wavelength, excluded, accessible = (
            self._fixed_dipole_inputs(data)
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
        initial_ideal = self.fixed_dipole_ideal(
            rho[accessible],
            polarization[accessible],
            volume,
            thermal_wavelength=wavelength,
        )
        alignment = (
            initial_ideal["polarization_derivative"] * moment[..., None]
        ).detach()

        was_training = self.model.training if self.model is not None else None
        if self.model is not None:
            self.model.eval()
        try:
            result = self.evaluate(
                dict(data, rho=rho, dipole_density=polarization)
            )
            result = self._apply_fixed_dipole_bounds(
                result,
                density_cap,
                polarization_fraction_cap,
                moment,
                accessible,
                fixed_field=fixed_field,
                minimum_density=minimum_density,
            )
            if method == "minimize":
                state = self._minimize_fixed_dipoles(
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
                state = self._solve_fixed_dipole_euler(
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
            if self.model is not None:
                self.model.train(was_training)
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

    def _minimize_fixed_dipoles(
        self,
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
            converged = self._fixed_dipole_converged(
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
                    trial = self.evaluate(
                        dict(data, rho=trial_rho, dipole_density=trial_p)
                    )
                    trial = self._apply_fixed_dipole_bounds(
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
                                result["local_chemical_potential"] * rho_delta
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

    def _solve_fixed_dipole_euler(
        self,
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
            converged = self._fixed_dipole_converged(
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
                trial = self.evaluate(
                    dict(data, rho=trial_rho, dipole_density=trial_p)
                )
                trial = self._apply_fixed_dipole_bounds(
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
                converged = self._fixed_dipole_converged(
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

    def _move_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.device is None:
            return dict(data)
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


def _resolve_accessibility_masks(
    data: Dict[str, Any],
    field: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return validated excluded and accessible masks for a grid field."""

    expected_shape = field.shape[:-1]
    excluded_mask = data.get("excluded_mask")
    if excluded_mask is None:
        excluded = torch.zeros(
            expected_shape,
            dtype=torch.bool,
            device=field.device,
        )
    else:
        if (
            not torch.is_tensor(excluded_mask)
            or excluded_mask.dtype != torch.bool
        ):
            raise TypeError("excluded_mask must be a Boolean tensor")
        if excluded_mask.shape != expected_shape:
            raise ValueError(
                "excluded_mask must have shape field.shape[:-1]"
            )
        excluded = excluded_mask.to(device=field.device)
    if torch.any(torch.all(excluded, dim=-1)).item():
        raise ValueError(
            "exclusion masks must leave at least one accessible grid point"
        )
    return excluded, ~excluded
