"""Option groups of :meth:`equicdft.GridSolver.solve`.

Each group is a frozen dataclass that validates itself once, so the solver
paths (scalar and fixed-dipole, Euler and minimize) share one definition of
every option and its admissible range.

The groups are ``convergence`` (:class:`ConvergenceOptions`), ``euler``
(:class:`EulerOptions`, with ``anderson=AndersonOptions(...)`` for
acceleration), ``minimize`` (:class:`MinimizeOptions`), ``constraints``
(:class:`Constraints`: particle numbers, density and polarization caps,
homogeneous axes, fixed field) and ``initial`` (:class:`Initialization`:
starting fields and ``beta_multiplier``).

``solve`` accepts only these keyword groups. Each field is documented below;
method-specific options may only accompany the corresponding method.
"""

from dataclasses import dataclass, fields
from typing import Any, Optional, Sequence, Union

import torch

from .._validation import (
    boolean,
    finite_scalar,
    nonnegative_integer,
    nonnegative_scalar,
    positive_integer,
    positive_scalar,
)

__all__ = [
    "AndersonOptions",
    "Constraints",
    "ConvergenceOptions",
    "EulerOptions",
    "Initialization",
    "MinimizeOptions",
    "SolveOptions",
]

ComponentValues = Union[float, Sequence[float], torch.Tensor]


def _set(instance: Any, name: str, value: Any) -> None:
    object.__setattr__(instance, name, value)


@dataclass(frozen=True)
class AndersonOptions:
    """Anderson acceleration of the Euler fixed point.

    ``history`` recent residual pairs form each trial (at least two),
    ``regularization`` is added to the least-squares normal matrix and
    ``damping`` in ``(0, 1]`` scales the accelerated step. The scalar driver
    forms a log-density trial after every full history window and keeps it
    only when it beats the mixed step. The fixed-dipole conjugate-field
    driver instead accelerates every step with a sliding window and uses
    ``safeguard`` (larger than one): an accelerated step after which the
    fixed-point residual has grown by more than that factor discards the
    history and is followed by a plain mixed step.
    """

    history: int = 5
    regularization: float = 1.0e-8
    damping: float = 1.0
    safeguard: float = 5.0

    def __post_init__(self) -> None:
        _set(self, "history", positive_integer(self.history, "anderson_history"))
        if self.history < 2:
            raise ValueError("anderson_history must be at least two")
        _set(
            self, "regularization",
            nonnegative_scalar(self.regularization, "anderson_regularization"),
        )
        _set(self, "damping", positive_scalar(self.damping, "anderson_damping"))
        if self.damping > 1.0:
            raise ValueError("anderson_damping must be no larger than one")
        _set(self, "safeguard", positive_scalar(self.safeguard, "anderson_safeguard"))
        if self.safeguard <= 1.0:
            raise ValueError("anderson_safeguard must be larger than one")


@dataclass(frozen=True)
class EulerOptions:
    """Damped Euler--Lagrange fixed-point iteration.

    The mixing parameter grows by ``mixing_growth`` after an improving step
    and shrinks by ``mixing_backtrack_factor`` after a worsening trial while
    ``adaptive_mixing`` is set, staying within ``[minimum_mixing,
    maximum_mixing]``. ``tolerance_change`` stops the iteration when the
    relative density change falls below it; ``max_log_density_change``
    bounds one update in log density. ``anderson`` enables acceleration.
    For fixed dipoles, ``fixed_dipole_coordinate`` selects the mixed
    variables. The default ``"conjugate_field"`` iterates on ``log rho`` and
    the conjugate field ``h`` with ``P = m rho L(h)``, both unconstrained,
    and is the only coordinate that accepts ``anderson`` for fixed dipoles;
    its mixing follows the same schedule but grows after a step that lowered
    the free energy and shrinks when a step would raise it.
    ``"polarization"`` interpolates the density and dipole density ``P``
    directly; ``"alignment"`` interpolates the physical molecular alignment
    ``u=P/(m rho)`` and reconstructs ``P=m rho u``.
    ``max_conjugate_field_change`` bounds one update of ``h``; the alignment
    itself is bounded only by ``maximum_polarization_fraction``.
    """

    mixing: float = 0.05
    adaptive_mixing: bool = True
    minimum_mixing: float = 0.005
    maximum_mixing: float = 0.2
    mixing_growth: float = 1.1
    mixing_backtrack_factor: float = 0.5
    tolerance_change: float = 1.0e-7
    max_log_density_change: float = 2.0
    anderson: Optional[AndersonOptions] = None
    fixed_dipole_coordinate: str = "conjugate_field"
    max_conjugate_field_change: float = 5.0

    def __post_init__(self) -> None:
        _set(self, "tolerance_change", positive_scalar(self.tolerance_change, "tolerance_change"))
        _set(self, "mixing", positive_scalar(self.mixing, "mixing"))
        _set(self, "minimum_mixing", positive_scalar(self.minimum_mixing, "minimum_mixing"))
        _set(self, "maximum_mixing", positive_scalar(self.maximum_mixing, "maximum_mixing"))
        _set(self, "mixing_growth", positive_scalar(self.mixing_growth, "mixing_growth"))
        _set(
            self, "mixing_backtrack_factor",
            positive_scalar(self.mixing_backtrack_factor, "mixing_backtrack_factor"),
        )
        _set(
            self, "max_log_density_change",
            positive_scalar(self.max_log_density_change, "max_log_density_change"),
        )
        if self.mixing > 1.0:
            raise ValueError("mixing must be in the interval (0, 1]")
        _set(self, "adaptive_mixing", boolean(self.adaptive_mixing, "adaptive_mixing"))
        if self.adaptive_mixing:
            if self.minimum_mixing > self.mixing:
                raise ValueError("minimum_mixing must be no larger than mixing")
            if not self.mixing <= self.maximum_mixing <= 1.0:
                raise ValueError(
                    "maximum_mixing must be at least mixing and no larger than one"
                )
            if self.mixing_growth < 1.0:
                raise ValueError("mixing_growth must be at least one")
            if not 0.0 < self.mixing_backtrack_factor < 1.0:
                raise ValueError(
                    "mixing_backtrack_factor must lie in the interval (0, 1)"
                )
        if self.anderson is not None and not isinstance(self.anderson, AndersonOptions):
            raise TypeError("anderson must be an AndersonOptions or None")
        if not isinstance(self.fixed_dipole_coordinate, str):
            raise TypeError(
                "fixed_dipole_coordinate must be 'polarization', "
                "'alignment' or 'conjugate_field'"
            )
        if self.fixed_dipole_coordinate not in (
            "polarization",
            "alignment",
            "conjugate_field",
        ):
            raise ValueError(
                "fixed_dipole_coordinate must be 'polarization', "
                "'alignment' or 'conjugate_field'"
            )
        _set(
            self, "max_conjugate_field_change",
            positive_scalar(self.max_conjugate_field_change, "max_conjugate_field_change"),
        )


@dataclass(frozen=True)
class MinimizeOptions:
    """Mirror-descent minimization with an Armijo line search.

    ``step_size`` starts every line search and is reduced by
    ``line_search_factor`` down to ``minimum_step_size``; ``armijo_factor``
    is the sufficient-decrease constant; ``max_log_density_change`` bounds
    one update in log density. ``fixed_dipole_coordinate`` selects the
    orientational coordinate only for fixed-dipole minimization:
    ``"conjugate_field"`` preserves the established entropy mirror map,
    while ``"alignment"`` updates the physical molecular alignment
    ``u=P/(m rho)`` directly. Both choices retain the positivity-preserving
    canonical log-density update.

    ``lbfgs_history`` larger than zero switches a fixed-dipole minimization
    (``"conjugate_field"`` coordinate, no ``fixed_field``) to projected
    L-BFGS with that many curvature pairs, the mirror-descent step as the
    initial inverse Hessian, energy-only line-search trials and the bounds
    handled as an active set. ``nonmonotone_window`` is the number of recent
    objectives whose maximum the Armijo test is taken against; one is the
    strict monotone test.
    """

    step_size: float = 1.0
    minimum_step_size: float = 1.0e-8
    line_search_factor: float = 0.5
    armijo_factor: float = 1.0e-4
    max_log_density_change: float = 2.0
    fixed_dipole_coordinate: str = "conjugate_field"
    lbfgs_history: int = 0
    nonmonotone_window: int = 1

    def __post_init__(self) -> None:
        _set(self, "step_size", positive_scalar(self.step_size, "step_size"))
        _set(self, "minimum_step_size", positive_scalar(self.minimum_step_size, "minimum_step_size"))
        _set(self, "line_search_factor", finite_scalar(self.line_search_factor, "line_search_factor"))
        _set(self, "armijo_factor", finite_scalar(self.armijo_factor, "armijo_factor"))
        _set(
            self, "max_log_density_change",
            positive_scalar(self.max_log_density_change, "max_log_density_change"),
        )
        if not isinstance(self.fixed_dipole_coordinate, str):
            raise TypeError(
                "fixed_dipole_coordinate must be 'conjugate_field' or "
                "'alignment'"
            )
        if self.fixed_dipole_coordinate not in (
            "conjugate_field",
            "alignment",
        ):
            raise ValueError(
                "fixed_dipole_coordinate must be 'conjugate_field' or "
                "'alignment'"
            )
        _set(self, "lbfgs_history", nonnegative_integer(self.lbfgs_history, "lbfgs_history"))
        _set(self, "nonmonotone_window", positive_integer(self.nonmonotone_window, "nonmonotone_window"))
        if self.lbfgs_history > 0 and self.fixed_dipole_coordinate != "conjugate_field":
            raise ValueError(
                "lbfgs_history requires fixed_dipole_coordinate='conjugate_field'"
            )
        if self.minimum_step_size > self.step_size:
            raise ValueError("minimum_step_size must be no larger than step_size")
        if not 0.0 < self.line_search_factor < 1.0:
            raise ValueError("line_search_factor must be in the interval (0, 1)")
        if not 0.0 <= self.armijo_factor < 1.0:
            raise ValueError("armijo_factor must be in the interval [0, 1)")


@dataclass(frozen=True)
class ConvergenceOptions:
    """Iteration budget and residual tolerances.

    ``tolerance_residual`` bounds the largest active-grid Euler residual;
    ``tolerance_rms_residual``, when given, must also be met by the
    density-weighted RMS residual. ``residual_density_threshold`` excludes
    voxels below that density from both diagnostics (``None`` keeps all).
    """

    max_iter: int = 200
    tolerance_residual: float = 1.0e-4
    tolerance_rms_residual: Optional[float] = None
    residual_density_threshold: Optional[float] = None

    def __post_init__(self) -> None:
        _set(self, "max_iter", positive_integer(self.max_iter, "max_iter"))
        _set(self, "tolerance_residual", positive_scalar(self.tolerance_residual, "tolerance_residual"))
        if self.tolerance_rms_residual is not None:
            _set(
                self, "tolerance_rms_residual",
                positive_scalar(self.tolerance_rms_residual, "tolerance_rms_residual"),
            )
        if self.residual_density_threshold is not None:
            _set(
                self, "residual_density_threshold",
                nonnegative_scalar(self.residual_density_threshold, "residual_density_threshold"),
            )


@dataclass(frozen=True)
class Constraints:
    """Thermodynamic ensemble and box constraints of one solve.

    ``particle_numbers`` (one per component) selects the canonical ensemble;
    ``None`` solves at the chemical potential ``data["mu"]``.
    ``maximum_density`` caps every component, ``minimum_density`` floors
    every accessible voxel of a fixed-dipole solve (a voxel resting on the
    floor with an outward residual counts as stationary, like a capped one),
    ``maximum_polarization_fraction`` bounds ``|P| <= q_max * m * rho`` for
    fixed-dipole solves, ``homogeneous_axes`` makes the density constant
    along grid axes (minimize only) and ``fixed_field`` holds ``"rho"`` or
    ``"dipole_density"`` fixed in a fixed-dipole minimization. Shapes and
    feasibility are checked against the data by the solver.
    """

    particle_numbers: Optional[ComponentValues] = None
    maximum_density: Optional[ComponentValues] = None
    minimum_density: Optional[ComponentValues] = None
    maximum_polarization_fraction: Optional[ComponentValues] = None
    homogeneous_axes: Optional[Union[str, Sequence[Union[int, str]]]] = None
    fixed_field: Optional[str] = None

    def __post_init__(self) -> None:
        if self.fixed_field not in (None, "rho", "dipole_density"):
            raise ValueError("fixed_field must be None, 'rho', or 'dipole_density'")


@dataclass(frozen=True)
class Initialization:
    """Starting fields of one solve.

    ``rho`` and ``polarization`` override ``data["rho"]`` and
    ``data["dipole_density"]``; without any, the density starts
    proportional to ``exp(-beta_multiplier * beta * V_ext)``.
    """

    rho: Optional[torch.Tensor] = None
    polarization: Optional[torch.Tensor] = None
    beta_multiplier: float = 0.0

    def __post_init__(self) -> None:
        _set(self, "beta_multiplier", nonnegative_scalar(self.beta_multiplier, "beta_multiplier"))


@dataclass(frozen=True)
class SolveOptions:
    """All option groups of one solve, with validated defaults."""

    convergence: ConvergenceOptions = ConvergenceOptions()
    euler: EulerOptions = EulerOptions()
    minimize: MinimizeOptions = MinimizeOptions()
    constraints: Constraints = Constraints()
    initial: Initialization = Initialization()

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            expected = field.type
            if not isinstance(value, expected):
                raise TypeError(
                    "{} must be a {}".format(field.name, expected.__name__)
                )
