"""Run deterministic density trajectories without retaining autograd graphs."""

from numbers import Integral
from pathlib import Path

import torch

from .integrator import _Grid
from .mobility import ConstantMobility


def _metadata(integrator, data, **extra):
    """Describe a run: integrator, mobility, geometry and the caller's extras."""
    metadata = {
        "dt": integrator.dt, "method": type(integrator).__name__, "deterministic": True,
        "mobility": {"class": type(integrator.mobility).__name__},
    }
    metadata.update(extra)
    for key in ("grid_size", "grid_spacing", "grid_positions", "grid_center", "excluded_mask", "temperature", "beta", "V_ext"):
        if key in data:
            metadata[key] = torch.as_tensor(data[key]).detach().cpu().clone()
    if isinstance(integrator.mobility, ConstantMobility):
        metadata["mobility"]["value"] = integrator.mobility.value.detach().cpu().clone()
    return metadata


class Dynamics:
    """Advance a density field and record compact CPU snapshots and diagnostics."""

    def __init__(self, integrator):
        self.integrator = integrator

    def run(self, data, n_steps, save_every=1):
        """Record the initial state, sampling intervals, and the final state.

        If a step is rejected, the frames recorded so far plus the last valid
        state are attached to the raised error as ``error.trajectory`` (with
        ``metadata["completed"] = False``), so a long run is not lost.
        """
        if isinstance(n_steps, bool) or not isinstance(n_steps, Integral) or n_steps < 0:
            raise ValueError("n_steps must be a nonnegative integer.")
        if isinstance(save_every, bool) or not isinstance(save_every, Integral) or save_every <= 0:
            raise ValueError("save_every must be a positive integer.")
        state = dict(data, rho=data["rho"].detach().clone())
        grid = _Grid(state)
        records = {key: [] for key in ("time", "rho", "beta_A", "particle_numbers", "min_density")}
        progress = {"done": 0, "recorded": 0, "pending": None}
        dt = self.integrator.dt

        def record(step):
            # The free energy of this state comes free with the next step's
            # derivative pass (integrator.last_beta_A); fill it in then.
            rho = state["rho"].detach()
            values = (step * dt, rho.cpu().clone(), None,
                      (grid.volume * rho.sum(0)).cpu(), rho[~grid.excluded].min().cpu())
            for key, value in zip(records, values):
                records[key].append(value)
            progress["recorded"] = progress["pending"] = step

        def settle(energy):
            records["beta_A"][-1] = energy.detach().cpu()
            progress["pending"] = None

        def assemble(completed):
            result = {key: torch.stack(values) for key, values in records.items() if key != "time"}
            result["time"] = torch.tensor(records["time"], dtype=torch.float64)
            result["metadata"] = _metadata(self.integrator, data, n_steps=n_steps, save_every=save_every,
                                           completed=completed, steps_completed=progress["done"])
            return result

        record(0)
        try:
            for step in range(1, n_steps + 1):
                if self.integrator.dt != dt:
                    raise ValueError("dt changed during Dynamics.run; use AdaptiveDynamics for variable steps.")
                state["rho"] = self.integrator.step(state)
                if progress["pending"] == step - 1:
                    settle(self.integrator.last_beta_A)
                progress["done"] = step
                if step % save_every == 0 or step == n_steps:
                    record(step)
        except ValueError as error:
            if progress["recorded"] != progress["done"]:
                record(progress["done"])
            if progress["pending"] is not None:
                try:
                    settle(self.integrator.beta_A(state))
                except ValueError:
                    # Preserve the failed trajectory even when its last
                    # energy cannot be evaluated; never mark it completed.
                    settle(state["rho"].new_tensor(float("nan")))
            error.trajectory = assemble(completed=False)
            raise
        if progress["pending"] is not None:
            settle(self.integrator.beta_A(state))
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
