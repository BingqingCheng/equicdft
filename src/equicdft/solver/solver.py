"""Evaluate learned grid functionals and minimize thermodynamic objectives."""

from typing import (
    Any,
    Dict,
    NamedTuple,
    Optional,
    Union,
)

import torch
from torch import nn

from .._validation import nonnegative_scalar
from ..geometry.grid import voxel_volume
from .symmetry import (
    _HomogeneousDensityProjection,
    _normalize_homogeneous_axes,
)
from .dipoles import solve_fixed_dipoles
from .evaluation import evaluate, _set_density_residuals
from .reporting import finalize_result
from .euler import solve_euler
from .minimize import minimize_density
from .options import (
    Constraints,
    ConvergenceOptions,
    EulerOptions,
    Initialization,
    MinimizeOptions,
    SolveOptions,
)
from .numerics import (
    _resolve_accessibility_masks,
    _euler_residual,
    _stationarity_field,
    _project_density_constraints,
    _residuals_converged,
)
from .._species import species_vector
from .._device import GeometryDeviceCache, module_device, move_batch
from ..functional.polarization_ideal import FixedDipoleIdeal

__all__ = [
    "GridSolver",
]


def _resolve_method(method, polarized, constraints):
    """Return the driver name and its normalized homogeneous axes.

    The default follows the model: a fixed-dipole functional minimizes, a
    scalar one iterates the Euler-Lagrange condition.
    """

    if method is None:
        method = "minimize" if polarized else "euler"
    if method not in ("minimize", "euler"):
        raise ValueError("method must be 'minimize' or 'euler'")
    homogeneous_axes = _normalize_homogeneous_axes(constraints.homogeneous_axes)
    if homogeneous_axes and method != "minimize":
        raise ValueError(
            "homogeneous_axes is currently available only with method='minimize'"
        )
    if homogeneous_axes and polarized:
        raise ValueError("homogeneous_axes is not supported with fixed dipoles")
    return method, homogeneous_axes


def _reject_fixed_dipole_only_options(constraints, initial):
    """Reject options that only mean something for a coupled rho/P solve."""

    if constraints.fixed_field is not None:
        raise ValueError("fixed_field requires model.ideal=FixedDipoleIdeal(...)")
    if initial.polarization is not None:
        raise ValueError(
            "initial_polarization requires model.ideal=FixedDipoleIdeal(...)"
        )
    if constraints.maximum_polarization_fraction is not None:
        raise ValueError(
            "maximum_polarization_fraction requires "
            "model.ideal=FixedDipoleIdeal(...)"
        )


class _ScalarProblem(NamedTuple):
    """Everything the scalar drivers need, validated once before the loop."""

    data: Dict[str, Any]
    V_ext: torch.Tensor
    rho: torch.Tensor
    mu: Optional[torch.Tensor]
    fixed_N: Optional[torch.Tensor]
    density_cap: Optional[torch.Tensor]
    thermal_wavelength: torch.Tensor
    volume_element: torch.Tensor
    excluded_mask: torch.Tensor
    accessible_mask: torch.Tensor
    projection: Optional[_HomogeneousDensityProjection]
    residual_density_threshold: float


def _prepare_scalar_problem(
    solver, data, constraints, initial, convergence, homogeneous_axes,
) -> _ScalarProblem:
    """Validate one density-only problem and build its starting profile.

    Either a chemical potential or a particle number is resolved, never
    both, and the starting density is the supplied one, the ideal-gas
    profile of the external field, or its normalized Boltzmann weights.
    """

    initial_rho = initial.rho
    particle_numbers = constraints.particle_numbers
    beta_multiplier = initial.beta_multiplier
    residual_density_threshold = convergence.residual_density_threshold
    maximum_density = constraints.maximum_density

    data = solver._move_data(data)
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
    thermal_wavelength = species_vector(
        data.get("thermal_wavelength", solver.model.ideal.thermal_wavelength),
        n_types,
        V_ext,
        "thermal_wavelength",
    )
    volume_element = voxel_volume(data["grid_spacing"])

    density_cap = None
    if maximum_density is not None:
        density_cap = species_vector(
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
        mu = species_vector(data["mu"], n_types, V_ext, "mu")
    else:
        fixed_N = species_vector(
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
    return _ScalarProblem(
        data=data,
        V_ext=V_ext,
        rho=rho,
        mu=mu,
        fixed_N=fixed_N,
        density_cap=density_cap,
        thermal_wavelength=thermal_wavelength,
        volume_element=volume_element,
        excluded_mask=excluded_mask,
        accessible_mask=accessible_mask,
        projection=projection,
        residual_density_threshold=residual_density_threshold,
    )


class GridSolver:
    """Evaluate or minimize scalar-density and fixed-dipole functionals.

    The model supplies the complete intrinsic functional and its ideal reference.
    This class adds external fields, ensemble constraints and numerical updates.
    A model with ``FixedDipoleIdeal`` enables coupled rho/P solving.

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

    When no density is supplied through ``initial.rho`` or ``data["rho"]``,
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
    ``euler=EulerOptions(anderson=AndersonOptions(...))`` additionally forms
    a log-density Anderson trial from
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
    ``full_*density_residual`` and ``full_converged`` results report
    unrestricted stationarity. Available only for density-only minimize.

    With ``model.ideal=FixedDipoleIdeal(...)``, both methods optimize density
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

    def __init__(
        self,
        model: nn.Module,
        *,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        if not isinstance(model, nn.Module) or getattr(model, "ideal", None) is None:
            raise TypeError("GridSolver requires a model with an explicit ideal reference")
        self.model = model
        self.device = module_device(model) if device is None else torch.device(device)
        self._geometry_cache = GeometryDeviceCache()
        if self.device is not None:
            self.model.to(self.device)

    def evaluate(
        self,
        data: Dict[str, Any],
        compute_derivatives: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate intrinsic energy, external coupling and stationarity.

        compute_derivatives=False skips differentiated outputs independently
        of model defaults. Analytic ideal-reference diagnostics may remain.
        Solver evaluation never requests c2.
        beta_external includes all external coupling; beta_A adds it to beta_F.
        Density residuals use the prescribed chemical potential when present
        for scalar fields, otherwise the canonical weighted mean.
        Fixed-dipole evaluation remains canonical and requires V_ext and E_ext.
        Bounds and homogeneous projections are applied only by solve.
        All returned tensors are detached and caller field tensors are not
        marked for gradients. Use the model for differentiable calculations.
        """

        return evaluate(self, data, compute_derivatives)

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
    ) -> Dict[str, Any]:
        """Minimize the thermodynamic functional to obtain equilibrium.

        ``method`` selects the driver; it defaults to Euler for a scalar
        model and to minimization for a fixed-dipole one. Options come in
        the validated groups of :mod:`equicdft.solver.options`. Method-specific
        groups require the corresponding method. What each method does, and
        what every constraint and initialization means, is described in this
        class's own docstring.
        """

        polarized = isinstance(self.model.ideal, FixedDipoleIdeal)
        options = SolveOptions(
            convergence=ConvergenceOptions() if convergence is None else convergence,
            euler=EulerOptions() if euler is None else euler,
            minimize=MinimizeOptions() if minimize is None else minimize,
            constraints=Constraints() if constraints is None else constraints,
            initial=Initialization() if initial is None else initial,
        )
        method, homogeneous_axes = _resolve_method(method, polarized, options.constraints)
        if method == "euler" and minimize is not None:
            raise ValueError("minimize options require method='minimize'")
        if method == "minimize" and euler is not None:
            raise ValueError("euler options require method='euler'")
        convergence = options.convergence
        euler = options.euler
        minimize = options.minimize
        constraints = options.constraints
        initial = options.initial

        if polarized:
            return solve_fixed_dipoles(self, data, method, options)
        _reject_fixed_dipole_only_options(constraints, initial)

        problem = _prepare_scalar_problem(
            self, data, constraints, initial, convergence, homogeneous_axes,
        )
        data = problem.data
        V_ext = problem.V_ext
        rho = problem.rho
        mu = problem.mu
        fixed_N = problem.fixed_N
        density_cap = problem.density_cap
        thermal_wavelength = problem.thermal_wavelength
        volume_element = problem.volume_element
        accessible_mask = problem.accessible_mask
        projection = problem.projection
        residual_density_threshold = problem.residual_density_threshold
        beta_multiplier = initial.beta_multiplier
        tolerance_residual = convergence.tolerance_residual
        tolerance_rms_residual = convergence.tolerance_rms_residual

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
            result = self.evaluate(final_data, compute_derivatives=True)
        finally:
            self.model.train(was_training)

        residual, chemical_potential, max_residual, rms_residual = (
            _euler_residual(
                result["rho"], _stationarity_field(result), data["beta"], mu,
                residual_density_threshold, density_cap, accessible_mask,
            )
        )
        if projection is not None:
            result["full_density_residual"] = residual
            result["full_max_density_residual"] = max_residual
            result["full_rms_density_residual"] = rms_residual
            result["full_converged"] = _residuals_converged(
                max_residual, rms_residual, tolerance_residual, tolerance_rms_residual,
            )
            residual, chemical_potential, max_residual, rms_residual = _euler_residual(
                result["rho"], _stationarity_field(result, projection), data["beta"], mu,
                residual_density_threshold, density_cap, accessible_mask,
            )
            result["solver_homogeneous_axes"] = list(homogeneous_axes)
        _set_density_residuals(
            result, residual, chemical_potential, max_residual, rms_residual,
        )
        converged = _residuals_converged(
            max_residual,
            rms_residual,
            tolerance_residual,
            tolerance_rms_residual,
        )
        # The coupled fixed-dipole path reports its overall residuals under
        # these names; for a density-only solve they are the density ones.
        result["maximum_residual"] = max_residual
        result["rms_residual"] = rms_residual
        return finalize_result(
            result, state, method=method, beta_multiplier=beta_multiplier,
            converged=converged,
        )

    def _move_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return move_batch(data, self.device, self._geometry_cache)
