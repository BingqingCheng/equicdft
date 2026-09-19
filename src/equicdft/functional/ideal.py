"""Model-owned ideal references, independent of learned excess readouts."""

import torch
from torch import nn

from .._config import Configurable, make_config, register
from .._species import broadcast_species_field, species_parameter
from .thermodynamics import log_dimensionless_density


def validate_density_field(rho, excluded_mask=None, exclusion_message=None):
    """Check a density grid and its exclusions once, for every consumer.

    The ideal reference cannot evaluate a negative or nonfinite density, and
    an excluded voxel is an exact hard wall, so both rules are enforced here
    rather than separately by every caller that hands a field to a model.
    ``exclusion_message`` lets a caller that also carries a polarization say
    so, since the two fields are excluded together.
    """

    if (rho.ndim < 2 or not rho.is_floating_point()
            or not torch.isfinite(rho).all() or torch.any(rho < 0)):
        raise ValueError("rho must be finite nonnegative [..., grid, species]")
    if excluded_mask is None:
        return
    if excluded_mask.dtype != torch.bool or excluded_mask.shape != rho.shape[:-1]:
        raise ValueError("excluded_mask must be Boolean [..., grid]")
    if torch.any(rho.masked_select(excluded_mask[..., None]) != 0):
        raise ValueError(
            exclusion_message or "rho must be zero at excluded grid points"
        )


class IdealReference(nn.Module, Configurable):
    """Intrinsic ideal beta energy and continuum derivatives on a grid.

    Exact vacuum has zero energy. Derivatives there are returned as zero
    placeholders with ``ideal_derivative_mask=False``: they are not finite
    thermodynamic derivatives. Exclusions and statistical masks are distinct.

    ``thermal_wavelength`` is the stored default, not a unit contract. A
    batch that carries its own ``thermal_wavelength`` overrides it, because
    ``Lambda = h / sqrt(2*pi*m*k_B*T)`` varies with temperature and species
    mass and one batch may hold several thermodynamic states. This is the
    opposite of the Boltzmann constant, which fixes the unit system and is
    checked against every batch's ``beta`` in
    :meth:`GridCACEModel._check_unit_contract`. Build data with
    ``grid_info=model.grid_info`` to carry the model's own value.
    """

    requires_dipole_density = False

    def __init__(self, *, thermal_wavelength=1.0):
        super().__init__()
        self.register_buffer("thermal_wavelength", species_parameter(
            thermal_wavelength, "thermal_wavelength",
        ))

    def local(self, rho, dipole_density=None, thermal_wavelength=None):
        """Return beta energy per particle and local ideal derivatives.

        This is the finite-interior physics interface. The common evaluate
        wrapper handles vacuum/exclusions and integrates over the grid.
        Derivatives are continuum derivatives (no voxel-volume factor).
        """
        raise NotImplementedError

    def evaluate(self, data, voxel_volume):
        """Evaluate full grids, preserving batches and zero-density boundaries."""
        rho = data["rho"]
        validate_density_field(rho, data.get("excluded_mask"))
        active = rho > 0
        if self.requires_dipole_density:
            p = data["dipole_density"]
            if p.shape != (*rho.shape, 3):
                raise ValueError("dipole_density must have shape rho.shape + (3,)")
            if torch.any((~active)[..., None] & (p != 0)):
                raise ValueError("dipole_density must be zero in vacuum")
        # Safe placeholders avoid evaluating logarithms/inverse Langevin at
        # vacuum; their local energy/derivatives are removed before integration.
        safe_rho = torch.where(active, rho, torch.ones_like(rho))
        # A batch wavelength wins over the stored one on purpose, unlike the
        # Boltzmann constant: Lambda = h / sqrt(2*pi*m*k_B*T) depends on the
        # temperature and the species mass, so one batch may legitimately
        # hold several, while k_B fixes the unit system and may not vary.
        # The buffer is therefore the default for data that carries none.
        local = self.local(
            safe_rho,
            data.get("dipole_density"),
            data.get("thermal_wavelength"),
        )
        density_energy = safe_rho * local.pop("energy_per_particle")
        volume = torch.as_tensor(voxel_volume, dtype=rho.dtype, device=rho.device)
        result = {
            "beta_F_id": (density_energy * active).sum(dim=(-2, -1)) * volume,
            "ideal_derivative_mask": active,
        }
        for key, value in local.items():
            mask = active[..., None] if value.ndim == rho.ndim + 1 else active
            result[key] = torch.where(mask, value, torch.zeros_like(value))
        return result


@register
class IdealGas(IdealReference):
    """Translational ideal gas, with angular measure absent/normalized to one."""

    def to_config(self):
        return make_config(self, thermal_wavelength=self.thermal_wavelength)

    def local(self, rho, dipole_density=None, thermal_wavelength=None):
        wavelength = broadcast_species_field(
            self.thermal_wavelength if thermal_wavelength is None else thermal_wavelength,
            rho, "thermal_wavelength",
        )
        log_rho = log_dimensionless_density(rho, wavelength)
        return {
            "energy_per_particle": log_rho - 1,
            "ideal_density_derivative": log_rho,
        }
