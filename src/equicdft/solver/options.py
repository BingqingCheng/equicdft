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

``solve`` accepts them as keyword arguments and, equivalently, every field
as a flat keyword (``max_iter=``, ``mixing=``, ``anderson=True`` with
``anderson_history=``, ``particle_numbers=``, ``initial_rho=``, ...), named
after the field itself and listed in :data:`FLAT_KEYWORDS`;
:func:`resolve_options` merges the two forms and rejects a field set both
ways. Each field is documented on its group below.
"""

from dataclasses import dataclass, fields, replace
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch

from .._validation import (
    boolean,
    finite_scalar,
    nonnegative_scalar,
    positive_integer,
    positive_scalar,
)

__all__ = [
    "AndersonOptions",
    "Constraints",
    "ConvergenceOptions",
    "EulerOptions",
    "FLAT_KEYWORDS",
    "Initialization",
    "MinimizeOptions",
    "SolveOptions",
    "resolve_options",
]

ComponentValues = Union[float, Sequence[float], torch.Tensor]


def _set(instance: Any, name: str, value: Any) -> None:
    object.__setattr__(instance, name, value)


@dataclass(frozen=True)
class AndersonOptions:
    """Anderson acceleration of the Euler fixed point (log-density trials).

    ``history`` recent residual pairs form each trial (at least two),
    ``regularization`` is added to the least-squares normal matrix and
    ``damping`` in ``(0, 1]`` scales the accelerated step.
    """

    history: int = 5
    regularization: float = 1.0e-8
    damping: float = 1.0

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


@dataclass(frozen=True)
class EulerOptions:
    """Damped Euler--Lagrange fixed-point iteration.

    The mixing parameter grows by ``mixing_growth`` after an improving step
    and shrinks by ``mixing_backtrack_factor`` after a worsening trial while
    ``adaptive_mixing`` is set, staying within ``[minimum_mixing,
    maximum_mixing]``. ``tolerance_change`` stops the iteration when the
    relative density change falls below it; ``max_log_density_change``
    bounds one update in log density. ``anderson`` enables acceleration.
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


@dataclass(frozen=True)
class MinimizeOptions:
    """Mirror-descent minimization with an Armijo line search.

    ``step_size`` starts every line search and is reduced by
    ``line_search_factor`` down to ``minimum_step_size``; ``armijo_factor``
    is the sufficient-decrease constant; ``max_log_density_change`` bounds
    one update in log density.
    """

    step_size: float = 1.0
    minimum_step_size: float = 1.0e-8
    line_search_factor: float = 0.5
    armijo_factor: float = 1.0e-4
    max_log_density_change: float = 2.0

    def __post_init__(self) -> None:
        _set(self, "step_size", positive_scalar(self.step_size, "step_size"))
        _set(self, "minimum_step_size", positive_scalar(self.minimum_step_size, "minimum_step_size"))
        _set(self, "line_search_factor", finite_scalar(self.line_search_factor, "line_search_factor"))
        _set(self, "armijo_factor", finite_scalar(self.armijo_factor, "armijo_factor"))
        _set(
            self, "max_log_density_change",
            positive_scalar(self.max_log_density_change, "max_log_density_change"),
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
    ``maximum_density`` caps every component, ``maximum_polarization_fraction``
    bounds ``|P| <= q_max * m * rho`` for fixed-dipole solves,
    ``homogeneous_axes`` makes the density constant along grid axes
    (minimize only) and ``fixed_field`` holds ``"rho"`` or
    ``"dipole_density"`` fixed in a fixed-dipole minimization. Shapes and
    feasibility are checked against the data by the solver.
    """

    particle_numbers: Optional[ComponentValues] = None
    maximum_density: Optional[ComponentValues] = None
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


# flat keyword -> (group attribute of SolveOptions, field name)
FLAT_KEYWORDS: Dict[str, Tuple[str, str]] = {
    "max_iter": ("convergence", "max_iter"),
    "tolerance_residual": ("convergence", "tolerance_residual"),
    "tolerance_rms_residual": ("convergence", "tolerance_rms_residual"),
    "residual_density_threshold": ("convergence", "residual_density_threshold"),
    "mixing": ("euler", "mixing"),
    "adaptive_mixing": ("euler", "adaptive_mixing"),
    "minimum_mixing": ("euler", "minimum_mixing"),
    "maximum_mixing": ("euler", "maximum_mixing"),
    "mixing_growth": ("euler", "mixing_growth"),
    "mixing_backtrack_factor": ("euler", "mixing_backtrack_factor"),
    "tolerance_change": ("euler", "tolerance_change"),
    "step_size": ("minimize", "step_size"),
    "minimum_step_size": ("minimize", "minimum_step_size"),
    "line_search_factor": ("minimize", "line_search_factor"),
    "armijo_factor": ("minimize", "armijo_factor"),
    "particle_numbers": ("constraints", "particle_numbers"),
    "maximum_density": ("constraints", "maximum_density"),
    "maximum_polarization_fraction": ("constraints", "maximum_polarization_fraction"),
    "homogeneous_axes": ("constraints", "homogeneous_axes"),
    "fixed_field": ("constraints", "fixed_field"),
    "initial_rho": ("initial", "rho"),
    "initial_polarization": ("initial", "polarization"),
    "beta_multiplier": ("initial", "beta_multiplier"),
}
# max_log_density_change bounds one update of either driver.
_SHARED_KEYWORDS = ("max_log_density_change",)
_ANDERSON_KEYWORDS = {
    "anderson_history": "history",
    "anderson_regularization": "regularization",
    "anderson_damping": "damping",
}


def _is_explicit(value: Any, default_value: Any) -> bool:
    if torch.is_tensor(value) or torch.is_tensor(default_value):
        return value is not default_value
    return value != default_value


def resolve_options(
    groups: Dict[str, Any],
    flat: Dict[str, Any],
) -> SolveOptions:
    """Merge option groups and flat keywords into one :class:`SolveOptions`.

    ``groups`` maps ``convergence``, ``euler``, ``minimize``, ``constraints``
    and ``initial`` to option objects (or ``None``); ``flat`` holds the
    flat keyword form of :meth:`GridSolver.solve`, including ``anderson``
    (a bool or an :class:`AndersonOptions`) and its ``anderson_*`` fields.
    A flat keyword may not repeat a field also set through its group.
    """

    unknown = sorted(
        key for key in flat
        if key not in FLAT_KEYWORDS
        and key not in _SHARED_KEYWORDS
        and key not in _ANDERSON_KEYWORDS
        and key != "anderson"
    )
    if unknown:
        raise TypeError(
            "solve() got unexpected keyword argument(s): {}".format(", ".join(unknown))
        )
    updates: Dict[str, Dict[str, Any]] = {name: {} for name in ("convergence", "euler", "minimize", "constraints", "initial")}
    for key, value in flat.items():
        if key in FLAT_KEYWORDS:
            group, field = FLAT_KEYWORDS[key]
            updates[group][field] = value
    for key in _SHARED_KEYWORDS:
        if key in flat:
            updates["euler"][key] = flat[key]
            updates["minimize"][key] = flat[key]
    anderson = flat.get("anderson", None)
    anderson_fields = {
        _ANDERSON_KEYWORDS[key]: flat[key] for key in _ANDERSON_KEYWORDS if key in flat
    }
    if isinstance(anderson, AndersonOptions):
        if anderson_fields:
            raise TypeError("anderson_* keywords cannot accompany an AndersonOptions")
        updates["euler"]["anderson"] = anderson
    elif anderson is not None:
        if boolean(anderson, "anderson"):
            updates["euler"]["anderson"] = AndersonOptions(**anderson_fields)
        else:
            # Anderson fields without the switch were ignored historically.
            updates["euler"]["anderson"] = None

    resolved = {}
    for name, default in (
        ("convergence", ConvergenceOptions),
        ("euler", EulerOptions),
        ("minimize", MinimizeOptions),
        ("constraints", Constraints),
        ("initial", Initialization),
    ):
        group = groups.get(name)
        changes = updates[name]
        if group is None:
            resolved[name] = default(**changes)
            continue
        if not isinstance(group, default):
            raise TypeError("{} must be a {}".format(name, default.__name__))
        if changes:
            defaults = default()
            explicit = {
                field.name for field in fields(group)
                if _is_explicit(getattr(group, field.name), getattr(defaults, field.name))
            }
            clash = sorted(set(changes) & explicit)
            if clash:
                raise TypeError(
                    "{} set both through {} and by keyword: {}".format(
                        ", ".join(clash), name, ", ".join(clash),
                    )
                )
            group = replace(group, **changes)
        resolved[name] = group
    return SolveOptions(**resolved)
