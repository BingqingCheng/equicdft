"""Training-only, diagonal-uncertainty perturbations in physical field units."""

from dataclasses import dataclass
from typing import Any, Dict

import torch

from .._species import species_vector
from .._validation import boolean, nonnegative_scalar
from ..functional.polarization_ideal import FixedDipoleIdeal

__all__ = ["NoiseOptions", "add_density_noise", "add_field_noise"]


@dataclass(frozen=True)
class NoiseOptions:
    """Training-time field noise of one trainer or training stream.

    ``density`` and ``polarization`` add independent Gaussian draws scaled
    by the batch ``rho_std`` and ``dipole_density_std`` before each training
    forward; validation stays clean. ``density_floor`` is a nonnegative
    lower density bound of the noisy field (excluded cells stay zero,
    particle numbers are not renormalized). For polarized batches, the trainer
    supplies ``model.ideal``; standalone callers pass ``ideal=FixedDipoleIdeal(...)``
    to ``apply``. The ideal reference supplies the molecular moment needed for
    the local bound. External-field targets remain unchanged by augmentation.
    """

    density: bool = False
    density_floor: float = 0.0
    polarization: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "density", boolean(self.density, "density_noise"))
        object.__setattr__(
            self, "density_floor",
            nonnegative_scalar(self.density_floor, "density_noise_floor"),
        )
        object.__setattr__(
            self, "polarization", boolean(self.polarization, "polarization_noise"),
        )

    @property
    def enabled(self) -> bool:
        return self.density or self.polarization

    def apply(self, batch: Dict[str, Any], *, ideal=None) -> Dict[str, Any]:
        """Return ``batch`` with the configured noise (unchanged when disabled)."""

        if not self.enabled:
            return batch
        return add_field_noise(
            batch,
            density_noise=self.density,
            density_noise_floor=self.density_floor,
            polarization_noise=self.polarization,
            ideal=ideal,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "density": self.density,
            "density_floor": self.density_floor,
            "polarization": self.polarization,
        }

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "NoiseOptions":
        return cls(**values)


_MAX_NOISE_ATTEMPTS = 1000


def _noise_std(batch, key, reference, enabled):
    if not enabled:
        return None
    if key not in batch:
        raise KeyError("noise augmentation requires " + key + " in training batches")
    std = batch[key].detach()
    if std.is_complex():
        raise ValueError(key + " must be real")
    std = std.to(reference)
    if std.shape != reference.shape:
        raise ValueError(key + " must have the same shape as its field")
    if not torch.all(torch.isfinite(std) & (std >= 0)).item():
        raise ValueError(key + " must be finite and nonnegative")
    excluded = batch.get("excluded_mask")
    if excluded is not None:
        if excluded.shape != batch["rho"].shape[:-1]:
            raise ValueError("excluded_mask must match rho's field/grid shape")
        if torch.any(std[excluded.bool()] != 0).item():
            raise ValueError(key + " must be zero at excluded grid points")
    return std


@torch.no_grad()
def add_density_noise(batch, floor):
    """Scalar-fluid SEM proposal and floor, without N normalization."""
    if "dipole_density" in batch:
        raise ValueError("polarized density noise requires add_field_noise and an ideal reference")
    rho = batch["rho"].detach()
    std = _noise_std(batch, "rho_std", rho, True)
    noisy = (rho + std * torch.randn_like(rho)).clamp_min(floor)
    excluded = batch.get("excluded_mask")
    if excluded is not None:
        excluded = excluded.bool().unsqueeze(-1).expand_as(rho)
        noisy = noisy.masked_fill(excluded, 0.0)
    if not torch.isfinite(noisy).all():
        raise ValueError("density noise produced nonfinite rho")
    return dict(batch, rho=noisy)


def _dipole_interior(rho, polar, moment):
    """Test the same finite rho/P domain for original fields and proposals."""
    bound = rho * moment
    reduced = polar / torch.where(bound > 0, bound, 1.0)[..., None]
    return ((rho > 0) & (bound > 0) & torch.isfinite(bound)
            & (reduced.square().sum(-1) < 1))


def _sample_polarized_fields(rho, polar, rho_std, polar_std, active, moment, floor):
    """Redraw only rejected species/voxels, always around the original mean."""
    noisy_rho, noisy_polar = rho.clone(), polar.clone()
    pending = torch.zeros_like(active)
    if rho_std is not None:
        pending |= (rho_std != 0) | (rho < floor)
    if polar_std is not None:
        pending |= (polar_std != 0).any(dim=-1)
    pending &= active
    moments = moment.expand_as(rho)

    for _ in range(_MAX_NOISE_ATTEMPTS):
        if not pending.any():
            return noisy_rho, noisy_polar
        proposed_rho, proposed_polar = rho[pending], polar[pending]
        if rho_std is not None:
            proposed_rho = (
                proposed_rho + rho_std[pending] * torch.randn_like(proposed_rho)
            ).clamp_min(floor)
        if polar_std is not None:
            proposed_polar = (
                proposed_polar + polar_std[pending] * torch.randn_like(proposed_polar)
            )
        accepted = _dipole_interior(proposed_rho, proposed_polar, moments[pending])
        # Convert the compact acceptance decisions back to the full grid.
        accepted_cells = pending.clone()
        accepted_cells[pending] = accepted
        noisy_rho[accepted_cells] = proposed_rho[accepted]
        noisy_polar[accepted_cells] = proposed_polar[accepted]
        pending &= ~accepted_cells

    if pending.any():
        raise RuntimeError(
            "SEM proposals could not satisfy rho>0 and |P|<m*rho after "
            f"{_MAX_NOISE_ATTEMPTS} attempts"
        )
    return noisy_rho, noisy_polar


@torch.no_grad()
def add_field_noise(batch, *, density_noise=False, density_noise_floor=0.0,
                    polarization_noise=False, ideal=None):
    """Apply optional SEM noise, preserving a polarized example's ideal domain.

    Polarized proposals use independent Cartesian/component Gaussian draws,
    conditioned on rho>0 and |P|<m*rho by whole (rho,P)-pair rejection. Original
    invalid/excluded voxels remain unchanged. No N or mean-P projection is made.
    SEM is the proposal sigma, not the variance after truncation. All targets
    remain untouched: external fields supervise total intrinsic derivatives.
    """
    if not density_noise and not polarization_noise:
        return batch
    if "dipole_density" not in batch:
        if polarization_noise:
            raise KeyError("polarization_noise=True requires dipole_density")
        return add_density_noise(batch, density_noise_floor)
    if not isinstance(ideal, FixedDipoleIdeal):
        raise ValueError("polarized noise requires a FixedDipoleIdeal reference")
    rho = batch["rho"].detach()
    polar = batch["dipole_density"].detach()
    if rho.ndim < 2 or not rho.is_floating_point():
        raise ValueError("rho must be floating [..., grid, species]")
    if polar.shape != (*rho.shape, 3):
        raise ValueError("dipole_density must have shape [..., grid, species, 3]")
    if polar.dtype != rho.dtype or polar.device != rho.device:
        raise ValueError("rho and dipole_density must share dtype and device")
    if (not torch.isfinite(rho).all() or torch.any(rho < 0)
            or not torch.isfinite(polar).all()):
        raise ValueError("noise requires finite fields and nonnegative rho")
    moment = species_vector(
        ideal.dipole_magnitude, rho.shape[-1], rho, "dipole_magnitude",
        positive=True,
    )
    rho_std = _noise_std(batch, "rho_std", rho, density_noise)
    polar_std = _noise_std(batch, "dipole_density_std", polar, polarization_noise)
    valid = batch.get("valid")
    if valid is not None and valid.shape != rho.shape[:-1]:
        raise ValueError("valid must match rho's field/grid shape")
    active = torch.ones_like(rho, dtype=torch.bool)
    if valid is not None:
        active &= valid.bool()[..., None]
    excluded = batch.get("excluded_mask")
    if excluded is not None:
        active &= ~excluded.bool()[..., None]
    # Exact vacuum is unchanged, even without an explicit mask.
    active &= ~((rho == 0) & (polar == 0).all(dim=-1))
    if torch.any(active & ~_dipole_interior(rho, polar, moment)):
        raise ValueError("unmasked polarized noise inputs require rho>0 and |P|<m*rho")

    noisy_rho, noisy_polar = _sample_polarized_fields(
        rho, polar, rho_std, polar_std, active, moment, density_noise_floor,
    )
    return dict(batch, rho=noisy_rho, dipole_density=noisy_polar)
