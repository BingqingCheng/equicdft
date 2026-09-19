"""Evaluate learned grid functionals and minimize thermodynamic objectives."""

from typing import (
    Any,
    Dict,
    Optional,
    Sequence,
    Union,
)
from .._deprecation import keyword_only

import torch
from torch import nn

from .._validation import nonnegative_scalar
from ..geometry.grid import voxel_volume
from .symmetry import (
    _HomogeneousDensityProjection,
    _normalize_homogeneous_axes,
)
from .dipoles import evaluate_fixed_dipoles, solve_fixed_dipoles
from .euler import solve_euler
from .results import aliased_result
from .minimize import minimize_density
from .options import (
    Constraints,
    ConvergenceOptions,
    EulerOptions,
    Initialization,
    MinimizeOptions,
    resolve_options,
)
from .numerics import (
    _resolve_accessibility_masks,
    _component_tensor,
    _euler_residual,
    _project_density_constraints,
    _residuals_converged,
)
from .._device import GeometryDeviceCache, module_device, move_batch
from ..functional.thermodynamics import (
    density_weighted_integral,
    ideal_free_energy,
)
from ..functional.polarization_ideal import FixedDipoleIdeal

__all__ = [
    "GridSolver",
]


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

    @keyword_only(1, ("model", "device"))
    def __init__(
        self,
        model: Optional[nn.Module],
        *,
        device: Optional[Union[str, torch.device]] = None,
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
            (module_device(model) if model is not None else None)
            if device is None
            else torch.device(device)
        )
        self._geometry_cache = GeometryDeviceCache()
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
            return evaluate_fixed_dipoles(self, data, compute_c1)

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
            if "beta_mu_local" in result:
                residual = (
                    result["beta_mu_local"]
                    - beta * data["mu"][..., None, :]
                )
                result["euler_lagrange_residual"] = residual.masked_fill(
                    excluded_density,
                    0.0,
                )

        return aliased_result(result)

    def solve(
        self,
        data: Dict[str, Any],
        *,
        method: Optional[str] = None,
        convergence: Optional[ConvergenceOptions] = None,
        euler: Optional[EulerOptions] = None,
        minimize: Optional[MinimizeOptions] = None,
        constraints: Optional[Constraints] = None,
        initial: Optional[Initialization] = None,
        **flat_options: Any,
    ) -> Dict[str, Any]:
        """Minimize the thermodynamic functional to obtain equilibrium.

        Options come in validated groups, ``convergence``
        (:class:`ConvergenceOptions`), ``euler`` (:class:`EulerOptions`,
        with ``anderson=AndersonOptions(...)`` for acceleration), ``minimize``
        (:class:`MinimizeOptions`), ``constraints`` (:class:`Constraints`:
        particle numbers, density and polarization caps, homogeneous axes,
        fixed field) and ``initial`` (:class:`Initialization`: starting
        fields and ``beta_multiplier``). Every field can equally be passed
        as a flat keyword (``max_iter=``, ``mixing=``, ``anderson=True`` with
        ``anderson_history=``, ``particle_numbers=``, ``initial_rho=``, ...);
        a field set both ways is an error. The flat keywords are described
        below by their field names.

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

        options = resolve_options(
            {
                "convergence": convergence,
                "euler": euler,
                "minimize": minimize,
                "constraints": constraints,
                "initial": initial,
            },
            flat_options,
        )
        convergence = options.convergence
        euler = options.euler
        minimize = options.minimize
        constraints = options.constraints
        initial = options.initial

        if method is None:
            method = (
                "minimize"
                if self.fixed_dipole_ideal is not None
                else "euler"
            )
        if method not in ("minimize", "euler"):
            raise ValueError("method must be 'minimize' or 'euler'")
        homogeneous_axes = _normalize_homogeneous_axes(constraints.homogeneous_axes)
        if homogeneous_axes and method != "minimize":
            raise ValueError("homogeneous_axes is currently available only with method='minimize'")
        if self.fixed_dipole_ideal is not None:
            if homogeneous_axes:
                raise ValueError("homogeneous_axes is not supported with fixed dipoles")
            return solve_fixed_dipoles(self, data, method, options)
        if constraints.fixed_field is not None:
            raise ValueError("fixed_field requires an explicit dipole_magnitude")
        if initial.polarization is not None:
            raise ValueError(
                "initial_polarization requires an explicit dipole_magnitude"
            )
        if constraints.maximum_polarization_fraction is not None:
            raise ValueError(
                "maximum_polarization_fraction requires an explicit "
                "dipole_magnitude"
            )

        # Local names of the option fields the scalar path reads here; the
        # drivers read their own groups.
        initial_rho = initial.rho
        particle_numbers = constraints.particle_numbers
        beta_multiplier = initial.beta_multiplier
        tolerance_residual = convergence.tolerance_residual
        tolerance_rms_residual = convergence.tolerance_rms_residual
        residual_density_threshold = convergence.residual_density_threshold
        maximum_density = constraints.maximum_density

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
                state = minimize_density(
                    self, data, rho, V_ext, thermal_wavelength, mu, fixed_N,
                    volume_element, density_cap, accessible_mask, projection,
                    convergence, minimize,
                )
            else:
                state = solve_euler(
                    self, data, rho, V_ext, thermal_wavelength, mu, fixed_N,
                    volume_element, density_cap, accessible_mask,
                    convergence, euler,
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
        result["beta_mu"] = chemical_potential
        result["max_euler_lagrange_residual"] = max_residual
        result["rms_euler_lagrange_residual"] = rms_residual
        result["converged"] = _residuals_converged(
            max_residual,
            rms_residual,
            tolerance_residual,
            tolerance_rms_residual,
        )
        # The coupled fixed-dipole path reports its overall residuals under
        # these names; for a density-only solve they are the density ones.
        result["maximum_residual"] = max_residual
        result["rms_residual"] = rms_residual
        if result["converged"]:
            result["status"] = "converged"
        elif state.get("line_search_failures"):
            result["status"] = "line_search_failed"
        elif state.get("stalled"):
            result["status"] = "stalled"
        else:
            result["status"] = "max_iter"
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
        return aliased_result(result)
    def _move_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return move_batch(data, self.device, self._geometry_cache)
