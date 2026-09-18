"""Training-only, diagonal-uncertainty perturbations in physical field units."""

import torch

from .energy import log_dimensionless_density
from .polarization_ideal import FixedDipoleIdeal, _positive_components


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
    """Legacy scalar-fluid SEM proposal and floor, without N normalization."""
    if "dipole_density" in batch:
        raise ValueError("polarized density noise requires add_field_noise and dipole_magnitude")
    rho = batch["rho"].detach()
    std = _noise_std(batch, "rho_std", rho, True)
    noisy = (rho + std * torch.randn_like(rho)).clamp_min(floor)
    excluded = batch.get("excluded_mask")
    if excluded is not None:
        excluded = excluded.bool().unsqueeze(-1).expand_as(rho)
        noisy = noisy.masked_fill(excluded, 0.0)
    if not torch.isfinite(noisy).all():
        raise ValueError("density noise produced nonfinite rho")
    result = dict(batch, rho=noisy)
    if "c1_plus_beta_mu" in batch or "c1" in batch:
        target = (log_dimensionless_density(noisy, batch["thermal_wavelength"])
                  + batch["beta"][..., None, None] * batch["V_ext"])
        if excluded is not None:
            target = target.masked_fill(excluded, 0.0)
        if "c1_plus_beta_mu" in batch:
            result["c1_plus_beta_mu"] = target
        if "c1" in batch:
            c1 = target - batch["beta_mu"][..., None, :]
            result["c1"] = c1 if excluded is None else c1.masked_fill(excluded, 0.0)
    return result


def _target_delta(target, delta, valid, name):
    """Accept full-grid targets or targets packed by the original voxel mask."""
    if target.shape == delta.shape:
        return delta
    if valid is not None:
        selected = delta[valid]
        # Collation adds batch axes to each case's packed target. Never
        # reinterpret per-species/component axes as a different target.
        component_shape = delta.shape[valid.ndim:]
        if (target.shape[-len(component_shape):] == component_shape
                and target.numel() == selected.numel()):
            return selected.reshape_as(target)
    raise ValueError(name + " must match its full field or original valid-mask selection")


def _refresh_polarized_targets(batch, result, active, moment):
    """Change only the known ideal parts; external fields and gauges stay fixed."""
    scalar_keys = ("c1_plus_beta_mu", "c1", "target_c1")
    vector_keys = ("polarization_derivative", "target_P_derivative")
    if not any(key in batch for key in scalar_keys + vector_keys):
        return
    ideal = FixedDipoleIdeal(moment)

    def response(fields):
        # Non-target vacuum/boundary cells must not enter the finite-interior
        # ideal formula. The thermal wavelength cancels in the differences.
        rho = torch.where(active, fields["rho"], torch.ones_like(fields["rho"]))
        polar = torch.where(active[..., None], fields["dipole_density"], 0.0)
        return ideal(rho, polar, 1.0)

    before, after = response(batch), response(result)
    valid = batch.get("valid")
    if valid is not None:
        valid = valid.bool()
    for keys, derivative, sign in (
        (scalar_keys, "density_derivative", 1),
        (vector_keys, "polarization_derivative", -1),
    ):
        delta = sign * (after[derivative] - before[derivative])
        for key in keys:
            if key in batch:
                result[key] = batch[key] + _target_delta(batch[key], delta, valid, key)


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
                    polarization_noise=False, dipole_magnitude=None):
    """Apply optional SEM noise, preserving a polarized example's ideal domain.

    Polarized proposals use independent Cartesian/component Gaussian draws,
    conditioned on rho>0 and |P|<m*rho by whole (rho,P)-pair rejection. Original
    invalid/excluded voxels remain unchanged. No N or mean-P projection is made.
    SEM is the proposal sigma, not the variance after truncation. Unknown custom
    targets are untouched; applications must derive them from the live fields.
    """
    if not density_noise and not polarization_noise:
        return batch
    if "dipole_density" not in batch:
        if polarization_noise:
            raise KeyError("polarization_noise=True requires dipole_density")
        return add_density_noise(batch, density_noise_floor)
    if dipole_magnitude is None:
        raise ValueError("polarized noise requires noise_dipole_magnitude")
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
    moment = _positive_components(dipole_magnitude, rho, "noise_dipole_magnitude")
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
    result = dict(batch, rho=noisy_rho, dipole_density=noisy_polar)
    _refresh_polarized_targets(batch, result, active, moment)
    return result
