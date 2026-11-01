"""Run density trajectories without retaining autograd graphs."""

from numbers import Integral
from pathlib import Path

import torch

from .integrator import _Grid, _with_mask
from .mobility import ConstantMobility, DensityMobility
from .field import _FieldEvaluator, _breakpoints, _time, _work, _same_fields


def _metadata(integrator, data, **extra):
    """Describe a run: integrator, mobility, geometry and the caller's extras."""
    metadata = {
        "dt": integrator.dt, "method": type(integrator).__name__,
        "deterministic": not getattr(integrator, "stochastic", False),
        "face_average": integrator.face_average,
        "mobility": {"class": type(integrator.mobility).__name__},
    }
    metadata.update(extra)
    if hasattr(integrator, "outflow_fraction"):
        metadata["outflow_fraction"] = integrator.outflow_fraction
        metadata["density_floor"] = integrator.density_floor
        metadata["last_limiter"] = integrator.last_limiter
    if getattr(integrator, "stochastic", False):
        metadata.update(seed=integrator.seed, noise="unfiltered conserved face flux",
                        stochastic_convention="Ito", rng_device=str(data["rho"].device),
                        rho_min=integrator.rho_min,
                        density_floor_correction=("species-wise headroom mass restore"
                                                  if integrator.rho_min is not None else None))
    for key in ("grid_size", "grid_spacing", "grid_positions", "grid_center", "excluded_mask", "mask", "temperature", "beta", "V_ext", "metal_external_field"):
        if key in data:
            metadata[key] = torch.as_tensor(data[key]).detach().cpu().clone()
    if isinstance(integrator.mobility, ConstantMobility):
        metadata["mobility"]["value"] = integrator.mobility.value.detach().cpu().clone()
    elif isinstance(integrator.mobility, DensityMobility):
        metadata["mobility"]["diffusivity"] = integrator.mobility.diffusivity.detach().cpu().clone()
    return metadata


class Dynamics:
    """Advance a density field and record compact CPU snapshots and diagnostics."""

    def __init__(self, integrator, field=None, *, mask=None):
        self.integrator = integrator
        _breakpoints(field)
        self.field = field
        self.mask = mask

    def run(self, data, n_steps, save_every=1, *, start_time=0.0):
        """Record the initial state, sampling intervals, and the final state.

        If a step is rejected, the frames recorded so far plus the last valid
        state are attached to the raised error as ``error.trajectory`` (with
        ``metadata["completed"] = False``), so a long run is not lost.
        Times are absolute. Declared field breakpoints split nominal steps;
        save_every still counts nominal steps. beta_work starts at zero here.
        """
        if isinstance(n_steps, bool) or not isinstance(n_steps, Integral) or n_steps < 0:
            raise ValueError("n_steps must be a nonnegative integer.")
        if isinstance(save_every, bool) or not isinstance(save_every, Integral) or save_every <= 0:
            raise ValueError("save_every must be a positive integer.")
        start_time = _time(start_time)
        points = _breakpoints(self.field)
        data = _with_mask(data, self.mask, self.integrator.mask)
        field_at = _FieldEvaluator(self.field)
        state = field_at(start_time, dict(data, rho=data["rho"].detach().clone()))
        initial_state = state
        grid = _Grid(state)
        records = {key: [] for key in ("time", "rho", "beta_A", "particle_numbers", "min_density")}
        work = state["rho"].new_zeros(())
        pending_work = None
        if self.field is not None:
            records.update(V_ext=[], beta_work=[])
        if "metal_external_field" in state:
            records["metal_external_field"] = []
        if getattr(self.integrator, "stochastic", False):
            records["rng_state"] = []
        floor_totals = {}
        if getattr(self.integrator, "rho_min", None) is not None:
            for key in ("floor_clamped_voxels", "floor_added_particles",
                        "floor_l1_particles", "floor_max_density_change"):
                dtype = torch.long if key == "floor_clamped_voxels" else grid.rho.dtype
                floor_totals[key] = torch.zeros(grid.rho.shape[1], device=grid.rho.device, dtype=dtype)
                records[key] = []
        progress = {"done": 0, "time": start_time, "recorded": start_time, "pending": False}
        dt = self.integrator.dt

        def record():
            # The free energy of this state comes free with the next step's
            # derivative pass (integrator.last_beta_A); fill it in then.
            rho = state["rho"].detach()
            values = (progress["time"], rho.cpu().clone(), None,
                      (grid.volume * rho.sum(0)).cpu(), rho[~grid.excluded].min().cpu())
            for key, value in zip(records, values):
                records[key].append(value)
            if "rng_state" in records:
                records["rng_state"].append(self.integrator.get_rng_state(state))
            for key, total in floor_totals.items():
                records[key].append(total.detach().cpu().clone())
            if self.field is not None:
                records["V_ext"].append(state["V_ext"].cpu().clone())
                records["beta_work"].append(work.cpu().clone())
            if "metal_external_field" in records:
                records["metal_external_field"].append(state["metal_external_field"].cpu().clone())
            progress["recorded"] = progress["time"]
            progress["pending"] = True

        def settle(energy):
            records["beta_A"][-1] = energy.detach().cpu()
            if self.field is not None:
                records["beta_work"][-1] = work.cpu().clone()
            progress["pending"] = False

        def settle_work():
            # A smooth step's endpoint is evaluated by the next step (or the
            # final energy pass). Reuse its dipole before replacing the state.
            nonlocal work, pending_work
            if pending_work is not None:
                start, start_dipole = pending_work
                work = work + _work(start, state, self.integrator.model,
                    start_dipole=start_dipole, end_dipole=self.integrator.last_metal_dipole)
                pending_work = None

        def assemble(completed):
            result = {key: torch.stack(values) for key, values in records.items() if key != "time"}
            result["time"] = torch.tensor(records["time"], dtype=torch.float64)
            result["metadata"] = _metadata(self.integrator, initial_state, n_steps=n_steps, save_every=save_every,
                                           start_time=start_time, dt=dt,
                                           completed=completed, steps_completed=progress["done"])
            if self.field is not None:
                result["metadata"]["field"] = self.field.metadata()
            return result

        record()
        try:
            for step in range(1, n_steps + 1):
                if self.integrator.dt != dt:
                    raise ValueError("dt changed during Dynamics.run; use AdaptiveDynamics for variable steps.")
                target = start_time + step * dt
                if target <= progress["time"]:
                    raise ValueError("dt is too small to advance the absolute simulation time.")
                endpoints = [t for t in points if progress["time"] < t < target] + [target]
                for end in endpoints:
                    # Validate both limits before consuming a stochastic draw.
                    left = field_at(end, state, left=True)
                    right = field_at(end, state) if end in points else left
                    self.integrator.dt = (end - progress["time"] if len(endpoints) > 1 else dt)
                    rho = self.integrator.step(state)
                    settle_work()
                    if progress["pending"]:
                        settle(self.integrator.last_beta_A)
                    for key, total in floor_totals.items():
                        value = self.integrator.last_floor_correction[key]
                        floor_totals[key] = (torch.maximum(total, value)
                                             if key == "floor_max_density_change" else total + value)
                    if self.field is not None:
                        left, right = dict(left, rho=rho), dict(right, rho=rho)
                        if _same_fields(left, right):
                            pending_work = (state, self.integrator.last_metal_dipole)
                        else:
                            # A discontinuity has a distinct left endpoint;
                            # retain the exact two-segment midpoint calculation.
                            work = work + _work(state, left, self.integrator.model) + _work(left, right, self.integrator.model)
                        state = right
                    else:
                        state = dict(state, rho=rho)
                    progress["time"] = end
                self.integrator.dt = dt
                progress["done"] = step
                if step % save_every == 0 or step == n_steps:
                    record()
        except ValueError as error:
            if progress["recorded"] != progress["time"]:
                record()
            if progress["pending"]:
                try:
                    energy = self.integrator.beta_A(state)
                    settle_work()
                    settle(energy)
                except ValueError:
                    # Preserve the failed trajectory even when its last
                    # energy cannot be evaluated; never mark it completed.
                    if pending_work is not None:
                        work = state["rho"].new_tensor(float("nan"))
                    settle(state["rho"].new_tensor(float("nan")))
            error.trajectory = assemble(completed=False)
            raise
        finally:
            self.integrator.dt = dt
        if progress["pending"]:
            energy = self.integrator.beta_A(state)
            settle_work()
            settle(energy)
        return assemble(completed=True)

    @staticmethod
    def save(trajectory, path):
        """Save a trajectory without overwriting an existing file."""
        path = Path(path)
        stream = path.open("xb")
        try:
            with stream:
                torch.save(trajectory, stream)
        except Exception:
            path.unlink()
            raise
        return path
