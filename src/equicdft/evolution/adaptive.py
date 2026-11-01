"""Adaptive timesteps for the deterministic density evolution.

Each step is taken twice from the same stored rate: as one full step and as
two half steps, with one model evaluation at the midpoint. Two estimates of
the local error are formed from the integrator's own updates, so both see the
semi-implicit stabilizer. The step-doubling difference ``rho_full -
rho_half`` is the error of the half-step result, which is the one kept (the
full step errs by about twice as much); it tracks the error whenever halving
the step changes the result, which includes the backward-Euler damping of
stiff modes. The defect ``dt/2 R(end) - (rho_end - rho_start)`` of each half
step, summed over the two, compares the displacement a half step produced
with the one its endpoint rate implies; it vanishes for a linear problem
whose stabilizer matches its curvature and exposes the slowed kinetics of a
mode whose stabilizer is much stiffer than its physics, where both step
sizes freeze the mode alike and the step-doubling difference stays small.
The larger of the two, relative to the density, is compared with the
tolerance. The rate at the accepted state is needed to start the next step
anyway, and the free energies from that pass and from the midpoint pass must
not rise beyond the configured allowances. An accepted step therefore costs
two model evaluations; a rejected step costs none or one when a density turns
non-positive or a masked implicit solve does not converge, one or two when
the free energy rises or the error estimate exceeds the tolerance.
"""

from dataclasses import dataclass
import math
import numbers

import torch

from ..geometry.grid import grid_spacing_tensor

from .dynamics import _metadata
from .integrator import ConvergenceError, DensityError, SemiImplicitIntegrator, _Grid, _positive_scalar, _with_mask
from .field import _FieldEvaluator, _breakpoints, _time, _work, _same_fields

_SAFETY = 0.9        # fraction of the step the error estimate would allow
_MAX_SHRINK = 0.1    # smallest factor applied after an accuracy rejection


def _positive_count(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError("{} must be a positive integer.".format(name))
    return int(value)


@dataclass(frozen=True)
class StepControl:
    """Bounds and tolerances of the adaptive step.

    ``dt_min`` and ``dt_max`` bound the step. A step that would have to fall
    below ``dt_min`` to satisfy the other criteria stops the run with the last
    accepted state attached to the error as ``error.trajectory``; only the
    final step is shortened freely to land on ``t_end``.
    ``max_relative_change`` caps ``dt * max|R|/rho`` over accessible voxels
    before a step is tried; below one it keeps an explicit step positive by
    construction. ``error_tolerance`` is the largest accepted estimate of the
    local error relative to the density (see the module docstring), taken as
    a maximum over accessible voxels. The following step is scaled by
    ``0.9 sqrt(error_tolerance / error)``, by at most ``growth`` after an
    accepted step (with the proposal floored at ``dt_min`` before applying
    the independent bounds) and by at least one tenth after a rejected one; ``shrink``
    is the factor after a rejection that carries no estimate (positivity,
    free energy, an unconverged masked implicit solve). ``energy_increase`` is the absolute increase of ``beta A``
    tolerated across a step or half step, in units of ``k_B T`` and
    independent of the zero of the free energy; the default rejects every
    increase beyond roundoff. ``energy_roundoff`` is the roundoff allowance
    relative to ``|beta A|``: the rounding error of the summed free energy
    scales with its magnitude, so this allowance does too. The default is a
    few thousand float64 ulps and is not a physical tolerance; float32 models
    need about ``1e-5``. ``stability_safety`` multiplies the linear
    explicit-stability bound that caps the step of an explicit integrator;
    the semi-implicit step has no such cap.

    ``density_absolute_tolerance`` adds an absolute density-error allowance:
    |local error| <= error_tolerance * rho + density_absolute_tolerance.
    Its default zero preserves purely relative control. The same mixed scale
    is used for the preliminary change bound; with a nonzero absolute term
    that bound alone no longer proves positivity. Integrator density checks
    remain mandatory; FluxLimitedEulerIntegrator bounds outgoing transfers.
    The error scale never changes the density and is not a density floor.
    """

    dt_min: float
    dt_max: float
    max_relative_change: float = 0.05
    error_tolerance: float = 1e-3
    growth: float = 1.25
    shrink: float = 0.5
    energy_increase: float = 0.0
    energy_roundoff: float = 1e-12
    stability_safety: float = 0.9
    density_absolute_tolerance: float = 0.0

    def __post_init__(self):
        for name in ("dt_min", "dt_max", "max_relative_change", "error_tolerance", "growth", "shrink", "stability_safety"):
            _positive_scalar(getattr(self, name), name)
        if self.dt_min > self.dt_max:
            raise ValueError("dt_min must not exceed dt_max.")
        if not self.growth >= 1 or not 0 < self.shrink < 1:
            raise ValueError("growth must be >= 1 and 0 < shrink < 1.")
        for name in ("energy_increase", "energy_roundoff", "density_absolute_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Real) or not 0 <= value < math.inf:
                raise ValueError("{} must be finite and nonnegative.".format(name))


class _Rejected(Exception):
    """A trial step failed; ``factor`` scales the step for the retry."""

    def __init__(self, count, reason, factor):
        super().__init__(reason)
        self.count = count
        self.reason = reason
        self.factor = factor


def _rejection(failure, control):
    """A non-positive density or an unconverged implicit solve rejects the step by ``shrink``."""
    if isinstance(failure, ConvergenceError):
        return _Rejected("rejected_solve", "implicit solve", control.shrink)
    return _Rejected("rejected_positivity", "positivity", control.shrink)


def _explicit_limit(integrator, state, grid):
    """Linear explicit-stability bound of the field; walls are dropped for the estimate.

    The bound is evaluated for the periodic homogeneous reference at the mean
    density of the free voxels (the integrator's default reference; frozen
    voxels are kept out of it). Blocking faces only removes couplings from
    the discrete Laplacian, which cannot raise its largest eigenvalue, so for
    a uniform mobility the periodic bound is the conservative one; the energy
    and positivity checks remain the safeguard against the nonlinear response
    near the walls.
    """
    if not grid.has_excluded:
        return integrator.stability_limit(state)
    pool = grid.free if bool(grid.free.any()) else ~grid.excluded
    periodic = {key: value for key, value in state.items() if key != "excluded_mask"}
    periodic["rho"] = state["rho"][pool].mean(0).expand_as(state["rho"]).clone()
    return integrator.stability_limit(periodic)


class AdaptiveDynamics:
    """Advance a density field to ``t_end`` with error-controlled steps."""

    def __init__(self, integrator, control, field=None, *, mask=None):
        if getattr(integrator, "stochastic", False):
            raise ValueError("AdaptiveDynamics is deterministic-only; use fixed-step Dynamics for noise.")
        if not isinstance(control, StepControl):
            raise TypeError("control must be a StepControl.")
        self.integrator = integrator
        self.control = control
        _breakpoints(field)
        self.field = field
        self._field_at = _FieldEvaluator(field)
        self.mask = mask
        self.last_work = None

    def run(self, data, t_end, save_every=1, max_steps=1000000, *, start_time=0.0):
        """Record the initial state, every ``save_every`` accepted steps, and the final state.

        Besides the ``Dynamics`` fields the result carries ``dt`` and the
        error estimate ``error`` per frame, the counts of accepted steps,
        rejections and model evaluations, and the step cap ``dt_cap``.
        """
        t_end = _positive_scalar(t_end, "t_end")
        start_time = _time(start_time)
        if t_end <= start_time:
            raise ValueError("t_end must exceed start_time (both are absolute times).")
        points = _breakpoints(self.field)
        save_every = _positive_count(save_every, "save_every")
        max_steps = _positive_count(max_steps, "max_steps")
        integrator, control = self.integrator, self.control
        data = _with_mask(data, self.mask, self.integrator.mask)
        self._field_at = _FieldEvaluator(self.field)
        state = self._field_at(start_time, dict(data, rho=data["rho"].detach().clone()))
        initial_state = state
        grid = _Grid(state)
        accessible = ~grid.excluded
        active = grid.free
        cap = control.dt_max
        if not isinstance(integrator, SemiImplicitIntegrator):
            cap = min(cap, control.stability_safety * _explicit_limit(integrator, state, grid))
        records = {key: [] for key in ("time", "dt", "error", "rho", "beta_A", "particle_numbers", "min_density")}
        work = state["rho"].new_zeros(())
        if self.field is not None:
            records.update(V_ext=[], beta_work=[])
        if "metal_external_field" in state:
            records["metal_external_field"] = []
        counts = {"accepted": 0, "rejected_positivity": 0, "rejected_solve": 0, "rejected_accuracy": 0,
                  "rejected_energy": 0, "model_evaluations": 0}

        def record(time, dt, error, energy):
            rho = state["rho"].detach()
            values = (time, dt, error, rho.cpu().clone(), energy.detach().cpu(),
                      (grid.volume * rho.sum(0)).cpu(), rho[accessible].min().cpu())
            for key, value in zip(records, values):
                records[key].append(value)
            if self.field is not None:
                records["V_ext"].append(state["V_ext"].cpu().clone())
                records["beta_work"].append(work.cpu().clone())
            if "metal_external_field" in records:
                records["metal_external_field"].append(state["metal_external_field"].cpu().clone())

        def assemble(completed):
            scalars = ("time", "dt", "error")
            result = {key: torch.stack(values) for key, values in records.items() if key not in scalars}
            for key in scalars:
                result[key] = torch.tensor(records[key], dtype=torch.float64)
            result["metadata"] = _metadata(integrator, initial_state, completed=completed, t_end=t_end, save_every=save_every,
                                           start_time=start_time,
                                           dt_cap=cap, control=control.__dict__, **counts)
            if self.field is not None:
                result["metadata"]["field"] = self.field.metadata()
            return result

        rate = integrator.rhs(state)
        counts["model_evaluations"] += 1
        energy = integrator.last_beta_A
        dipole = integrator.last_metal_dipole
        time = start_time
        dt = min(max(integrator.dt, control.dt_min), cap)
        record(time, math.nan, math.nan, energy)
        try:
            while time < t_end and counts["accepted"] < max_steps:
                controlled_rate = self._control_rate(state, rate, min(dt, cap))
                change = float((controlled_rate.abs() / self._density_scale(state["rho"])).masked_fill(~active[:, None], 0).max())
                bound = min(dt, cap, control.max_relative_change / change) if change > 0 else min(dt, cap)
                if bound < control.dt_min:
                    raise ValueError(
                        "The step bounds require dt = {:.3g} below dt_min = {:.3g} at t = {:.6g}; "
                        "the last accepted state is attached as error.trajectory.".format(bound, control.dt_min, time))
                next_stop = next((t for t in points if time < t <= t_end), t_end)
                dt = min(bound, next_stop - time)
                endpoint = next_stop if dt == next_stop - time else time + dt
                # Check the right-limit field before accepting a trial.
                right = self._field_at(endpoint, state)
                while True:
                    try:
                        trial, new_rate, new_energy, error = self._trial(
                            state, rate, dt, active, energy, counts, time=time, endpoint=endpoint,
                            start_dipole=dipole)
                        break
                    except _Rejected as rejection:
                        counts[rejection.count] += 1
                        dt = self._retry(dt, rejection)
                        endpoint = time + dt
                        right = self._field_at(endpoint, state)
                new_work = work
                if self.field is not None:
                    right = dict(right, rho=trial["rho"])
                    left_dipole = integrator.last_metal_dipole
                    field_changed = not _same_fields(trial, right)
                    if field_changed:
                        new_rate = integrator.rhs(right)
                        new_energy = integrator.last_beta_A
                        counts["model_evaluations"] += 1
                    new_work = work + self.last_work + _work(trial, right, integrator.model,
                        start_dipole=left_dipole, end_dipole=integrator.last_metal_dipole)
                    # Equal endpoint fields can be distinct cloned tensors.
                    # Keep the state that owns the reused rate/face fluxes;
                    # only a real field jump requires adopting the right state.
                    if field_changed:
                        trial = right
                rate, energy, work = new_rate, new_energy, new_work
                dipole = integrator.last_metal_dipole
                state = trial
                time = endpoint
                counts["accepted"] += 1
                if counts["accepted"] % save_every == 0 or time >= t_end:
                    record(time, dt, error, energy)
                factor = control.growth if error == 0 else min(control.growth, _SAFETY * math.sqrt(control.error_tolerance / error))
                # This is an accuracy recommendation, not a hard upper bound.
                # The next iteration still enforces the cap and relative-change bound.
                dt = max(dt * factor, control.dt_min)
        except ValueError as failure:
            if records["time"][-1] != time:
                record(time, math.nan, math.nan, energy)
            failure.trajectory = assemble(completed=False)
            raise
        if records["time"][-1] != time:
            record(time, math.nan, math.nan, energy)
        return assemble(completed=time >= t_end)

    def _trial(self, state, rate, dt, accessible, energy, counts, time=0.0, endpoint=None,
               start_dipole=None):
        """Try one step as a full step and two half steps from the stored rate.

        Returns the half-step state, its rate and free energy, and the error
        estimate; raises ``_Rejected`` when a density turns non-positive, the
        work-corrected energy rises beyond allowances at the midpoint or end, or
        the estimate exceeds the tolerance. Checks that need only the midpoint
        evaluation run before the second one.
        """
        integrator, control = self.integrator, self.control
        allowance = control.energy_increase + control.energy_roundoff * abs(float(energy))
        energy_limit = float(energy) + allowance
        half = dt / 2
        endpoint = time + dt if endpoint is None else endpoint
        try:
            full = integrator.advance(state, rate, dt)
            midpoint = dict(state, rho=integrator.advance(state, rate, half))
        except (DensityError, ConvergenceError) as failure:
            raise _rejection(failure, control) from None
        midpoint = self._field_at(time + half, midpoint)
        midpoint_rate = integrator.rhs(midpoint)
        counts["model_evaluations"] += 1
        midpoint_energy = float(integrator.last_beta_A)
        midpoint_dipole = integrator.last_metal_dipole
        work_first = float(_work(state, midpoint, integrator.model,
            start_dipole=start_dipole, end_dipole=midpoint_dipole)) if self.field is not None else 0.0
        energy_limit += work_first
        if midpoint_energy > energy_limit:
            raise _Rejected("rejected_energy", "free energy", control.shrink)
        # Check the second half against its own start, as well as the whole step.
        midpoint_allowance = control.energy_increase + control.energy_roundoff * abs(midpoint_energy)
        energy_limit = min(energy_limit, midpoint_energy + midpoint_allowance)
        try:
            trial = dict(state, rho=integrator.advance(midpoint, midpoint_rate, half))
        except (DensityError, ConvergenceError) as failure:
            raise _rejection(failure, control) from None
        rho = trial["rho"]
        defect = (half * self._control_rate(midpoint, midpoint_rate, half)
                  - (midpoint["rho"] - state["rho"])).abs()
        error = float((torch.maximum((full - rho).abs(), defect) / self._density_scale(rho)).masked_fill(~accessible[:, None], 0).max())
        if error > control.error_tolerance:
            raise _Rejected("rejected_accuracy", "accuracy", self._factor(error))
        trial = self._field_at(endpoint, trial, left=True)
        trial_rate = integrator.rhs(trial)
        counts["model_evaluations"] += 1
        work_second = float(_work(midpoint, trial, integrator.model,
            start_dipole=midpoint_dipole, end_dipole=integrator.last_metal_dipole)) if self.field is not None else 0.0
        energy_limit += work_second
        if self.field is not None:
            full_work = _work(state, dict(trial, rho=full), integrator.model)
            # Dimensionless work error per particle, using the density error tolerance.
            volume = grid_spacing_tensor(state["grid_spacing"], dtype=state["rho"].dtype,
                                         device=state["rho"].device).prod()
            particles = float(volume * state["rho"].sum())
            error = max(error, abs(float(full_work) - work_first - work_second) / max(1.0, particles))
        defect += (half * self._control_rate(trial, trial_rate, half)
                   - (rho - midpoint["rho"])).abs()
        error = max(error, float((defect / self._density_scale(rho)).masked_fill(~accessible[:, None], 0).max()))
        if error > control.error_tolerance:
            raise _Rejected("rejected_accuracy", "accuracy", self._factor(error))
        if float(integrator.last_beta_A) > energy_limit:
            raise _Rejected("rejected_energy", "free energy", control.shrink)
        self.last_work = state["rho"].new_tensor(work_first + work_second)
        return trial, trial_rate, integrator.last_beta_A, error

    def _control_rate(self, state, rate, dt):
        method = getattr(self.integrator, "control_rate", None)
        return rate if method is None else method(state, rate, dt)

    def _density_scale(self, rho):
        # error <= rtol means |density error| <= rtol*rho + atol.
        # This only scales errors; it never replaces the physical density.
        return rho + self.control.density_absolute_tolerance / self.control.error_tolerance

    def _factor(self, error):
        return max(_MAX_SHRINK, _SAFETY * math.sqrt(self.control.error_tolerance / error))

    def _retry(self, dt, rejection):
        if dt <= self.control.dt_min:
            raise ValueError("Step rejected ({}) at dt = {:.3g} <= dt_min; the last accepted state is attached as "
                             "error.trajectory.".format(rejection.reason, dt))
        return max(dt * rejection.factor, self.control.dt_min)
