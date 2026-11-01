"""Conservative deterministic Euler update with a common face-flux limiter."""
import math
import torch
from .integrator import EulerIntegrator, SemiImplicitIntegrator, _fingerprint


class FluxLimitedEulerIntegrator(EulerIntegrator):
    """Limit outgoing transfers instead of clipping the updated density.

    Each voxel/species donates at most ``outflow_fraction`` of its density
    per step. The minimum donor factor over all species scales the *whole*
    face flux vector, preserving its PSD mobility direction and semidiscrete
    dissipation sign. Adjacent voxels see opposite transfers. Blocked faces
    stay blocked. Finite-step energy acceptance and accuracy still require
    a driver such as AdaptiveDynamics.

    ``density_floor`` optionally replaces the machine-scale donor reserve.
    It must be chosen explicitly in the density units of the input. Inputs
    below that floor are not raised; supply a separately validated conservative
    initialization if a hard floor is required. Excluded zeros and frozen
    densities remain untouched. A positive floor also enables constrained
    finite-step rates for adaptive change and defect estimates. This is a
    numerical regularization, not the original unconstrained equation.

    Above the donor reserve, the correction becomes inactive as dt tends to
    zero. It changes no density, potential or mobility law before the RHS
    evaluation. It is not a stochastic-noise limiter.
    """

    preserve_face_fluxes = True

    def __init__(self, model, mobility, dt, *, outflow_fraction=.9, mask=None,
                 face_average="arithmetic", density_floor=0.):
        if isinstance(outflow_fraction, bool) or not 0 < outflow_fraction < 1:
            raise ValueError("outflow_fraction must be strictly between zero and one.")
        super().__init__(model, mobility, dt, mask=mask, face_average=face_average)
        self.outflow_fraction = float(outflow_fraction)
        if isinstance(density_floor, bool) or not math.isfinite(density_floor) or density_floor < 0:
            raise ValueError("density_floor must be finite and nonnegative.")
        self.density_floor = float(density_floor)
        self.last_limiter = None

    def control_rate(self, data, rate, dt):
        """Rate of the actual constrained update, for adaptive error control.

        Only enabled with an explicit physical floor. Default-zero behavior
        retains the original unconstrained-rate controller.
        """
        if self.density_floor == 0:
            return rate
        return (self.advance(data, rate, dt)-data['rho'])/dt

    def _advance(self, grid, rate, dt):
        expected = (id(grid.rho), grid.rho._version, grid.signature, _fingerprint(grid.potential))
        if (getattr(rate, "_face_state", None) != expected
                or getattr(rate, "_face_rate_version", None) != rate._version):
            raise ValueError("Flux-limited advance needs the unmodified RHS tensor from this density and field state.")
        return self._limited_flux_update(grid, rate._face_fluxes, dt)

    def _limited_flux_update(self, grid, fluxes, dt):
        """Apply equal-and-opposite, common-species face-transfer budgets."""
        rho = grid.dense(grid.rho.detach())
        outgoing = torch.zeros_like(rho)
        for axis, (flux, spacing) in enumerate(zip(fluxes, grid.spacing)):
            outgoing += (flux.clamp_min(0)+torch.roll((-flux).clamp_min(0), 1, axis))/spacing
        transfer = dt*outgoing
        # No reciprocal of a tiny density and no positive-density floor.
        denominator = torch.where(transfer > 0, transfer, torch.ones_like(transfer))
        # Retain the smallest *normal* floating-point density when donating.
        # Repeated depletion otherwise reaches subnormals and rounds to zero,
        # even with a fractional outflow budget. This never raises a density:
        # inputs already below tiny simply cannot donate until replenished.
        reserve = max(self.density_floor, torch.finfo(rho.dtype).tiny)
        available = (rho - reserve).clamp_min(0)
        factors = torch.minimum(torch.ones_like(rho), self.outflow_fraction*available/denominator)
        factors = torch.where(transfer > 0, factors, torch.ones_like(factors))
        limited_rate = torch.zeros_like(rho)
        count = 0
        minimum = rho.new_ones(())
        for axis, flux in enumerate(fluxes):
            donor = torch.where(flux >= 0, factors, torch.roll(factors, -1, axis))
            donor = torch.where(flux == 0, torch.ones_like(donor), donor)
            factor = donor.amin(-1, keepdim=True)
            limited_rate += grid.divergence(factor*flux, axis)
            count += int((factor < 1).sum())
            minimum = torch.minimum(minimum, factor.min())
        result = grid.restore_frozen(grid.rho.detach()+dt*grid.flat(limited_rate))
        grid.check_density(result)
        self.last_limiter = dict(limited_faces=count, minimum_factor=float(minimum))
        self._last_output = (result, result._version)
        return result


class FluxLimitedSemiImplicitIntegrator(SemiImplicitIntegrator):
    """Existing IMEX stabilizer followed by conservative face-flux limiting.

    Reconstruct J_eff = J(rho) - M_ref grad(H_ref delta_rho)/beta from
    the unconstrained implicit increment, then apply the same donor budgets
    as FluxLimitedEulerIntegrator. This preserves region/species numbers and
    the floor without clipping. With inactive budgets it recovers the IMEX
    update to linear-solve accuracy. Limiting changes that update and does
    not establish unconditional stability or energy descent: adaptive energy,
    defect and temporal-refinement checks remain necessary.
    """

    preserve_face_fluxes = True
    _limited_flux_update = FluxLimitedEulerIntegrator._limited_flux_update

    def __init__(self, model, mobility, dt, reference_density=None, reference_curvature=None,
                 *, mask=None, face_average="arithmetic", density_floor=0., outflow_fraction=.9):
        if isinstance(outflow_fraction, bool) or not 0 < outflow_fraction < 1:
            raise ValueError("outflow_fraction must be strictly between zero and one.")
        if isinstance(density_floor, bool) or not math.isfinite(density_floor) or density_floor < 0:
            raise ValueError("density_floor must be finite and nonnegative.")
        super().__init__(model, mobility, dt, reference_density, reference_curvature,
                         mask=mask, face_average=face_average)
        self.density_floor = float(density_floor)
        self.outflow_fraction = float(outflow_fraction)
        self.last_limiter = None

    def control_rate(self, data, rate, dt):
        # Use the constrained *explicit* endpoint rate for the defect. Applying
        # implicit smoothing here could hide slowed physical kinetics.
        if self.density_floor == 0:
            return rate
        grid = self._grid_for(data)
        return (FluxLimitedEulerIntegrator._advance(self, grid, rate, dt)-data['rho'])/dt

    def _advance(self, grid, rate, dt):
        expected = (id(grid.rho), grid.rho._version, grid.signature, _fingerprint(grid.potential))
        if (getattr(rate, "_face_state", None) != expected
                or getattr(rate, "_face_rate_version", None) != rate._version):
            raise ValueError("Flux-limited advance needs the unmodified RHS tensor from this density and field state.")
        increment = self._implicit_increment(grid, rate, dt)
        modes = torch.fft.fftn(grid.dense(increment), dim=(0, 1, 2))
        potential = torch.fft.ifftn((self._curvature @ modes.unsqueeze(-1)).squeeze(-1),
                                   dim=(0, 1, 2)).real
        fluxes = []
        for axis, (flux, spacing) in enumerate(zip(rate._face_fluxes, grid.spacing)):
            gradient = (torch.roll(potential, -1, axis)-potential)/spacing
            corrected = flux - (gradient @ self._reference_mobility.T)/grid.beta
            if grid.has_blocked:
                corrected = corrected.masked_fill(grid.blocked[axis], 0)
            fluxes.append(corrected)
        return self._limited_flux_update(grid, fluxes, dt)
