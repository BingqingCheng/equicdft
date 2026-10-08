"""Conserved Euler--Maruyama dynamics with validated face mobility laws."""

import math
from numbers import Integral

import torch

from .integrator import DensityError, EulerIntegrator, _positive_scalar
from .mobility import ConstantMobility, DensityMobility


class StochasticEulerIntegrator(EulerIntegrator):
    """Add unfiltered thermal face fluxes to the deterministic Euler drift.

    Constant species matrices and diagonal DensityMobility are supported. The
    integrated face flux has covariance ``2 dt M / (beta * voxel_volume)``.
    Energy may increase. Optional ``rho_min`` floors trial densities and
    restores each species' mass using the density above that floor. This is
    numerical regularization, not exact sampling of the original process.
    Without a floor, invalid densities stop the run. No draws are retried.
    """

    stochastic = True

    def __init__(self, model, mobility, dt, *, seed, rho_min=None, mask=None):
        if type(mobility) not in (ConstantMobility, DensityMobility):
            raise TypeError("Stochastic evolution requires ConstantMobility or DensityMobility.")
        if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63).")
        super().__init__(model, mobility, dt, mask=mask)
        self.seed = int(seed)
        self.rho_min = None if rho_min is None else _positive_scalar(rho_min, "rho_min")
        self.last_floor_correction = None
        self._generator = None
        self._factor_key = None
        self._factor = None

    def _rng(self, data):
        device = data["rho"].device
        if self._generator is None:
            self._generator = torch.Generator(device=device).manual_seed(self.seed)
        elif self._generator.device != device:
            raise ValueError("Stochastic integrator device changed; construct a new integrator.")
        return self._generator

    def get_rng_state(self, data):
        """CPU checkpoint for the random stream at the supplied density state."""
        return self._rng(data).get_state().cpu().clone()

    def set_rng_state(self, data, state):
        """Restore a checkpoint on the same device/backend as the original run.

        The copy matters: a row of a trajectory's stacked ``rng_state`` is a
        view with a storage offset, which ``Generator.set_state`` must not see.
        """
        self._rng(data).set_state(state.detach().cpu().clone())

    def _constant_factor(self, data, grid):
        value = self.mobility.value
        key = (id(value), value._version, grid.rho.device, grid.rho.dtype, grid.rho.shape[-1])
        if key != self._factor_key:
            matrix, _ = self._mobility_matrices(data, grid)
            eigenvalues, vectors = torch.linalg.eigh((matrix + matrix.T) / 2)
            # ConstantMobility validates PSD; remove roundoff only, with no jitter.
            self._factor = vectors * eigenvalues.clamp_min(0).sqrt().unsqueeze(0)
            self._factor_key = key
        return self._factor

    def _noise_increment(self, data, grid):
        if isinstance(self.mobility, ConstantMobility):
            factor = self._constant_factor(data, grid)
            diagonal = None
        else:
            matrices, _ = self._mobility_matrices(data, grid)
            diagonal = matrices.diagonal(dim1=-2, dim2=-1)
        generator = self._rng(data)
        increment = torch.zeros_like(grid.dense(grid.rho))
        amplitude = math.sqrt(2 * self.dt / (grid.beta * float(grid.volume)))
        for axis, spacing in enumerate(grid.spacing):
            normal = torch.randn(increment.shape, device=increment.device,
                                 dtype=increment.dtype, generator=generator)
            if diagonal is None:
                flux = amplitude * (normal @ factor.T)
            else:
                # Same arithmetic face mobility as the deterministic drift.
                face = (diagonal + torch.roll(diagonal, -1, axis)) / 2
                flux = amplitude * face.sqrt() * normal
            increment += grid.divergence(flux, axis)
        return grid.flat(increment)

    def advance(self, data, rate, dt=None):
        raise ValueError("Use step(data) for stochastic increments; advance is deterministic-only.")

    def _apply_density_floor_region(self, rho, grid, rows):
        """Floor accessible cells, preserving species sums without a uniform shift."""
        trial = rho[rows]
        if not torch.isfinite(trial).all():
            raise DensityError("Trial densities must be finite; the floor cannot repair NaN or Inf.")
        below = trial < self.rho_min
        stats = {
            "floor_clamped_voxels": below.sum(0),
            "floor_added_particles": trial.new_zeros(trial.shape[1]),
            "floor_l1_particles": trial.new_zeros(trial.shape[1]),
            "floor_max_density_change": trial.new_zeros(trial.shape[1]),
        }
        if not below.any():
            return rho, stats  # Bitwise unchanged when the guard does not act.
        floored = trial.clamp_min(self.rho_min)
        headroom = floored - self.rho_min
        target = (grid.rho.detach()[rows] - self.rho_min).sum(0)
        available = headroom.sum(0)
        if not torch.isfinite(available).all() or torch.any((available == 0) & (target > 0)):
            raise DensityError("Density floor cannot restore the species particle numbers.")
        scale = target / available.clamp_min(torch.finfo(trial.dtype).tiny)
        corrected = self.rho_min + headroom * scale
        # Leave species without floor hits untouched, including immobile ones.
        corrected = torch.where(below.any(0), corrected, trial)
        change = corrected - trial
        stats.update(
            floor_added_particles=grid.volume * (floored - trial).sum(0),
            floor_l1_particles=grid.volume * change.abs().sum(0),
            floor_max_density_change=change.abs().amax(0),
        )
        result = rho.clone()
        result[rows] = corrected
        return result, stats

    def _apply_density_floor(self, rho, grid):
        """Repair only free cells, separately within each no-flux region."""
        totals = {key: rho.new_zeros(rho.shape[1], dtype=torch.long if key == "floor_clamped_voxels" else rho.dtype)
                  for key in ("floor_clamped_voxels", "floor_added_particles",
                              "floor_l1_particles", "floor_max_density_change")}
        for rows in grid.free_regions():
            rho, stats = self._apply_density_floor_region(rho, grid, rows)
            for key, value in stats.items():
                totals[key] = torch.maximum(totals[key], value) if key == "floor_max_density_change" else totals[key] + value
        return rho, totals

    def step(self, data):
        grid = self._grid_for(data)
        if self.rho_min is not None and torch.any(grid.rho[grid.free] < self.rho_min):
            raise DensityError("Initial unfrozen densities must be at least rho_min.")
        rate = self._rhs(data, grid)
        rng_state = self.get_rng_state(data)
        try:
            rho = grid.restore_frozen(grid.rho.detach() + self.dt * rate + self._noise_increment(data, grid))
            stats = None
            if self.rho_min is not None:
                rho, stats = self._apply_density_floor(rho, grid)
            grid.check_density(rho)
        except Exception:
            # The failed trajectory ends at the previous state and RNG point.
            # Do not automatically retry with another draw or a smaller step.
            self.set_rng_state(data, rng_state)
            raise
        self.last_floor_correction = stats
        self._last_output = (rho, rho._version)
        return rho
