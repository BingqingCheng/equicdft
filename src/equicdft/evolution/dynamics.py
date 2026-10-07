"""Run deterministic density trajectories without retaining autograd graphs."""

from numbers import Integral
from pathlib import Path

import torch

from .integrator import _Grid
from .mobility import ConstantMobility


class Dynamics:
    """Advance a density field and record compact CPU snapshots and diagnostics."""

    def __init__(self, integrator):
        self.integrator = integrator

    def run(self, data, n_steps, save_every=1):
        """Record the initial state, sampling intervals, and the final state."""
        if isinstance(n_steps, bool) or not isinstance(n_steps, Integral) or n_steps < 0:
            raise ValueError("n_steps must be a nonnegative integer.")
        if isinstance(save_every, bool) or not isinstance(save_every, Integral) or save_every <= 0:
            raise ValueError("save_every must be a positive integer.")
        state = dict(data, rho=data["rho"].detach().clone())
        grid = _Grid(state)
        records = {key: [] for key in ("time", "rho", "beta_A", "particle_numbers", "min_density")}

        def record(step):
            rho = state["rho"].detach()
            records["time"].append(step * self.integrator.dt)
            records["rho"].append(rho.cpu().clone())
            records["beta_A"].append(self.integrator.beta_A(state).cpu())
            records["particle_numbers"].append((grid.volume * rho.sum(0)).cpu())
            records["min_density"].append(rho[~grid.excluded].min().cpu())

        record(0)
        for step in range(1, n_steps + 1):
            state["rho"] = self.integrator.step(state)
            if step % save_every == 0 or step == n_steps:
                record(step)
        result = {key: torch.stack(values) for key, values in records.items() if key != "time"}
        result["time"] = torch.tensor(records["time"], dtype=torch.float64)
        result["metadata"] = {
            "dt": self.integrator.dt, "n_steps": n_steps, "save_every": save_every,
            "method": type(self.integrator).__name__, "deterministic": True,
            "mobility": {"class": type(self.integrator.mobility).__name__},
        }
        for key in ("grid_size", "grid_spacing", "grid_positions", "grid_center", "excluded_mask", "temperature", "beta", "V_ext"):
            if key in data:
                result["metadata"][key] = torch.as_tensor(data[key]).detach().cpu().clone()
        if isinstance(self.integrator.mobility, ConstantMobility):
            result["metadata"]["mobility"]["value"] = self.integrator.mobility.value.detach().cpu().clone()
        return result

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
