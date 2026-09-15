"""Shared numerical helpers for projected periodic Fourier curvatures."""

import math
from typing import Dict, Sequence, Tuple, Union

import torch
from torch import nn

from ._argument_checks import finite_scalar
from ._grid import voxel_volume
from .energy import ideal_free_energy
from .polarization_ideal import FixedDipoleIdeal


def integer_mode_tensor(
    supplied_modes: object,
    name: str = "modes",
) -> torch.Tensor:
    """Return exact integer reciprocal modes without rounding or truncation."""

    modes = torch.as_tensor(supplied_modes)
    if modes.dtype == torch.bool or torch.is_complex(modes):
        raise TypeError("{} must contain real integer values".format(name))
    if not torch.all(torch.isfinite(modes)).item():
        raise ValueError("{} must be finite".format(name))
    integer_modes = modes.to(torch.long)
    if not torch.equal(modes, integer_modes.to(modes.dtype)):
        raise ValueError("{} must contain integers".format(name))
    return integer_modes


def mode_triplets(
    supplied_modes: object,
    name: str = "modes",
) -> torch.Tensor:
    """Return a nonempty collection of distinct, nonzero integer triplets."""

    modes = integer_mode_tensor(supplied_modes, name)
    if modes.ndim != 2 or modes.shape[0] == 0 or modes.shape[1] != 3:
        raise ValueError("{} must have shape [n_modes, 3]".format(name))
    if torch.any(torch.all(modes == 0, dim=-1)).item():
        raise ValueError("{} must not contain the zero mode".format(name))
    if torch.unique(modes, dim=0).shape[0] != modes.shape[0]:
        raise ValueError("{} must not contain duplicates".format(name))
    return modes


def expand_mode_amplitudes(amplitude, rho, modes, n_directions, n_phases=2):
    """Broadcast one amplitude per field/mode to its phases and probes.

    Keep the scalar path scalar to preserve existing numerical behavior.
    Explicit tensors must have shape [field, mode]; amplitudes are not learned.
    """
    amplitude = _validated_mode_amplitudes(amplitude, rho, modes)
    if not torch.is_tensor(amplitude):
        return amplitude
    return amplitude.repeat_interleave(n_phases * n_directions, dim=1)


def _validated_mode_amplitudes(amplitude, reference, modes):
    """Return a scalar or detached [field, mode] amplitude."""

    if not torch.is_tensor(amplitude):
        value = finite_scalar(amplitude, "relative_amplitude")
        if not 0.0 < value < 1.0:
            raise ValueError("relative_amplitude must lie in (0, 1)")
        return value
    if amplitude.shape != modes.shape[:2]:
        raise ValueError("relative_amplitude tensor must have shape [field, mode]")
    if (
        amplitude.dtype == torch.bool or torch.is_complex(amplitude)
        or not torch.all(torch.isfinite(amplitude) & (amplitude > 0) & (amplitude < 1))
    ):
        raise ValueError("relative_amplitude tensor values must lie in (0, 1)")
    return amplitude.detach().to(reference)


def _validated_validity_mask(mask, reference):
    """Return a true-is-valid [field, grid] mask on reference's device."""

    if mask is None:
        return None
    if not torch.is_tensor(mask) or mask.dtype != torch.bool:
        raise TypeError("validity mask must be a Boolean tensor")
    if mask.shape != reference.shape[:2]:
        raise ValueError("validity mask must have shape rho.shape[:-1]")
    if torch.any(~torch.any(mask, dim=-1)).item():
        raise ValueError("validity mask must retain a voxel in every field")
    return mask.to(device=reference.device)


def real_mode_shape(modes, mode_phases):
    """A supplied phase per wave selects one summed spatial pattern."""
    return (modes.shape[1], 2) if mode_phases is None else (1, 1)


def projected_fourier_curvature(
    model: nn.Module,
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    rho: torch.Tensor,
    directions: torch.Tensor,
    valid_directions: torch.Tensor,
    mean_densities: torch.Tensor,
    relative_amplitude: Union[float, torch.Tensor],
    perturbations_per_forward: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate normalized total intrinsic curvature along fixed directions."""

    second_difference, perturbation_norm = _symmetric_energy_difference(
        model=model,
        outputs=outputs,
        batch=batch,
        rho=rho,
        directions=directions,
        relative_amplitude=relative_amplitude,
        perturbations_per_forward=perturbations_per_forward,
    )
    curvature = (
        mean_densities
        * second_difference
        / torch.clamp(perturbation_norm, min=1.0e-12)
    )
    valid = valid_directions & (perturbation_norm > 1.0e-12)
    return curvature, valid


def _symmetric_energy_difference(
    model: nn.Module,
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    rho: torch.Tensor,
    directions: torch.Tensor,
    relative_amplitude: Union[float, torch.Tensor],
    perturbations_per_forward: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return central energy differences and squared perturbation norms."""

    scale = (
        relative_amplitude[..., None, None]
        if torch.is_tensor(relative_amplitude) else relative_amplitude
    )
    delta_rho = scale * directions
    rho_plus = rho[:, None, :, :] + delta_rho
    rho_minus = rho[:, None, :, :] - delta_rho
    if torch.any(rho_plus < -1.0e-7).item() or torch.any(
        rho_minus < -1.0e-7
    ).item():
        raise RuntimeError("Fourier perturbation produced negative density")

    n_directions = directions.shape[1]
    perturbed_rho = torch.stack((rho_plus, rho_minus), dim=2).flatten(1, 2)
    volume_element = voxel_volume(batch["grid_spacing"].to(rho))
    # These modes conserve each component's particle number, so the
    # thermal-wavelength term is linear and has zero projected curvature.
    thermal_wavelength = rho.new_ones(rho.shape[-1])
    reference_energy = (
        ideal_free_energy(rho, thermal_wavelength, volume_element)
        + outputs["beta_F_exc"]
    )
    chunk_size = perturbations_per_forward or perturbed_rho.shape[1]
    energy_chunks = []
    for start in range(0, perturbed_rho.shape[1], chunk_size):
        chunk = perturbed_rho[:, start:start + chunk_size]
        perturbed_outputs = _energy_only_model(
            model,
            _expand_batch(batch, chunk),
        )
        if "beta_F_exc" not in perturbed_outputs:
            raise KeyError("model outputs are missing 'beta_F_exc'")
        energy_chunks.append(
            ideal_free_energy(
                chunk,
                thermal_wavelength,
                volume_element[:, None].expand(-1, chunk.shape[1]),
            )
            + perturbed_outputs["beta_F_exc"]
        )
    perturbed_energy = torch.cat(energy_chunks, dim=1).reshape(
        rho.shape[0],
        n_directions,
        2,
    )
    second_difference = (
        perturbed_energy.sum(dim=2) - 2.0 * reference_energy[:, None]
    )
    perturbation_norm = volume_element[:, None] * torch.sum(
        delta_rho.square(),
        dim=(-2, -1),
    )
    return second_difference, perturbation_norm


def fourier_curvature_matrix(
    model: nn.Module,
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    rho: torch.Tensor,
    modes: torch.Tensor,
    relative_amplitude: Union[float, torch.Tensor],
    perturbations_per_forward: int = None,
    mode_phases: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Return the ideal-metric component Hessian for every real mode.

    The returned matrix has shape ``[field, mode, phase, type, type]``.
    For a homogeneous mixture it is the dimensionless inverse OZ response
    ``I - sqrt(R) c(k) sqrt(R)``. Around an inhomogeneous density it is the
    component Hessian projected onto the selected real Fourier perturbations.
    With ``mode_phases`` supplied, waves are summed before projection and the
    output has shape [field, 1, 1, type, type]; it includes cross-wavevector
    contributions but does not reconstruct the complete spatial Hessian. Pair
    polarization adds numerical cancellation, so matrix applications should
    check convergence with respect to ``relative_amplitude``.
    """

    component_directions, active = component_fourier_directions(
        batch,
        rho,
        modes,
        mode_phases=mode_phases,
    )
    n_fields, n_real_modes, _, n_types = component_directions.shape
    identity = torch.eye(n_types, device=rho.device, dtype=rho.dtype)
    pair_indices = torch.triu_indices(
        n_types,
        n_types,
        offset=1,
        device=rho.device,
    )
    pair_weights = (
        identity[pair_indices[0]] + identity[pair_indices[1]]
    )
    weights = torch.cat((identity, pair_weights), dim=0)
    n_patterns, n_phases = real_mode_shape(modes, mode_phases)
    amplitude = expand_mode_amplitudes(
        relative_amplitude, rho, modes[:, :n_patterns], weights.shape[0], n_phases,
    )
    directions = (
        component_directions[:, :, None, :, :]
        * weights[None, None, :, None, :]
    ).flatten(start_dim=1, end_dim=2)

    second_difference, _ = _symmetric_energy_difference(
        model=model,
        outputs=outputs,
        batch=batch,
        rho=rho,
        directions=directions,
        relative_amplitude=amplitude,
        perturbations_per_forward=perturbations_per_forward,
    )
    scale_squared = (
        amplitude.reshape(n_fields, n_real_modes, weights.shape[0]).square()
        if torch.is_tensor(amplitude) else amplitude**2
    )
    quadratic_forms = second_difference.reshape(
        n_fields,
        n_real_modes,
        weights.shape[0],
    ) / scale_squared
    diagonal = quadratic_forms[..., :n_types]
    matrix = torch.diag_embed(diagonal)
    pair_values = 0.5 * (
        quadratic_forms[..., n_types:]
        - diagonal[..., pair_indices[0]]
        - diagonal[..., pair_indices[1]]
    )
    matrix[..., pair_indices[0], pair_indices[1]] = pair_values
    matrix[..., pair_indices[1], pair_indices[0]] = pair_values

    volume_element = voxel_volume(batch["grid_spacing"].to(rho))
    density = rho[:, None, :, :]
    ideal_integrand = torch.where(
        density > 0.0,
        component_directions.square()
        / torch.clamp(density, min=torch.finfo(rho.dtype).tiny),
        torch.zeros_like(component_directions),
    )
    ideal_norm = (
        volume_element[:, None, None] * ideal_integrand.sum(dim=-2)
    )
    active = active & (ideal_norm > 1.0e-12)
    scale = torch.sqrt(
        ideal_norm[..., :, None] * ideal_norm[..., None, :]
    )
    matrix = matrix / torch.clamp(scale, min=1.0e-12)
    active_matrix = active[..., :, None] & active[..., None, :]
    matrix = torch.where(active_matrix, matrix, torch.zeros_like(matrix))

    shape = (n_fields, n_patterns, n_phases)
    return (
        matrix.reshape(*shape, n_types, n_types),
        active.reshape(*shape, n_types),
    )


def polarization_fourier_curvature(
    model: nn.Module,
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    rho: torch.Tensor,
    modes: torch.Tensor,
    dipole_magnitude: torch.Tensor,
    relative_amplitude: Union[float, torch.Tensor],
    polarization_directions: torch.Tensor,
    perturbations_per_forward: int = None,
    validity_mask: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Return one ideal-normalized polarization curvature per field.

    The selected basis displacement applies one shared Cartesian alignment
    change to every species, scaled locally by ``m_a*rho_a``. The result has
    shape ``[field, mode, phase]`` independent of species count. Density is
    held fixed, including for the zero wavevector.
    """

    if "dipole_density" not in batch:
        raise KeyError("batch is missing 'dipole_density'")
    polarization = batch["dipole_density"]
    if polarization.shape != rho.shape + (3,):
        raise ValueError("dipole_density must have shape rho.shape + (3,)")
    if polarization.dtype != rho.dtype or polarization.device != rho.device:
        raise ValueError("rho and dipole_density must share dtype and device")
    validity_mask = _validated_validity_mask(validity_mask, rho)
    if validity_mask is not None:
        polarization = polarization.masked_fill(
            ~validity_mask[..., None, None], 0.0,
        )
    if not torch.all(torch.isfinite(polarization)).item():
        raise ValueError("dipole_density must be finite")
    zero_density = rho == 0.0
    if torch.any(zero_density[..., None] & (polarization != 0.0)).item():
        raise ValueError("dipole_density must be zero where rho is zero")

    ideal = FixedDipoleIdeal(dipole_magnitude).to(rho)
    moment = ideal.dipole_magnitude.reshape(-1)
    if moment.numel() not in (1, rho.shape[-1]):
        raise ValueError("dipole_magnitude must contain one value per density type")

    waves, valid_wave = _real_fourier_waves(batch, rho, modes)
    if validity_mask is not None:
        waves = waves * validity_mask[:, None, :]
    wave_norm = torch.amax(torch.abs(waves), dim=-1)
    valid_wave = wave_norm > 1.0e-5
    waves = waves / torch.clamp(wave_norm[..., None], min=1.0e-12)
    selected_direction = _unit_polarization_directions(
        polarization_directions, rho,
    )

    ideal_rho = torch.where(zero_density, torch.ones_like(rho), rho)
    alignment_scale = rho * moment
    ideal_alignment_scale = ideal_rho * moment
    directions = (
        waves[:, :, :, None, None]
        * alignment_scale[:, None, :, :, None]
        * selected_direction[:, None, None, None, :]
    )

    reduced = polarization / ideal_alignment_scale[..., None]
    used_amplitude = _safe_polarization_amplitudes(
        relative_amplitude,
        rho,
        modes,
        waves,
        reduced,
        selected_direction,
        valid_wave,
    )
    total_second, ideal_second = _polarization_energy_differences(
        model=model,
        outputs=outputs,
        batch=batch,
        rho=rho,
        ideal_rho=ideal_rho,
        polarization=polarization,
        directions=directions,
        ideal=ideal,
        relative_amplitude=used_amplitude,
        perturbations_per_forward=perturbations_per_forward,
    )
    ideal_denominator = ideal_second.detach()
    ideal_scale = torch.clamp(
        torch.amax(torch.abs(ideal_denominator), dim=1, keepdim=True),
        min=1.0,
    )
    ideal_valid = ideal_denominator > (
        100.0 * torch.finfo(rho.dtype).eps * ideal_scale
    )
    valid = valid_wave & ideal_valid
    curvature = total_second / torch.clamp(
        ideal_denominator, min=torch.finfo(rho.dtype).tiny
    )
    curvature = torch.where(valid, curvature, torch.zeros_like(curvature))
    return (
        curvature.reshape(rho.shape[0], modes.shape[1], 2),
        valid.reshape(rho.shape[0], modes.shape[1], 2),
    )


def _unit_polarization_directions(directions, reference):
    """Validate and normalize one Cartesian direction per field."""

    directions = torch.as_tensor(directions, device=reference.device)
    if (
        directions.dtype == torch.bool
        or torch.is_complex(directions)
        or directions.shape != (reference.shape[0], 3)
        or not torch.all(torch.isfinite(directions)).item()
    ):
        raise ValueError(
            "polarization_directions must be finite with shape [field, 3]"
        )
    directions = directions.to(reference)
    norm = torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
    if torch.any(norm == 0.0).item():
        raise ValueError("polarization_directions must be nonzero")
    return directions / norm


def _safe_polarization_amplitudes(
    amplitude,
    rho,
    modes,
    waves,
    reduced_polarization,
    direction,
    valid_wave,
):
    """Reduce amplitudes so both probes remain inside the dipole sphere."""

    reduced_norm = torch.linalg.vector_norm(reduced_polarization, dim=-1)
    if torch.any(reduced_norm >= 1.0).item():
        raise ValueError("fixed dipoles require |P| < m*rho")
    parallel = torch.abs(torch.sum(
        reduced_polarization * direction[:, None, None, :], dim=-1,
    ))
    directional_limit = torch.sqrt(torch.clamp(
        parallel.square() + 1.0 - reduced_norm.square(), min=0.0,
    )) - parallel
    absolute_wave = torch.abs(waves)[..., None]
    local_limit = torch.where(
        absolute_wave > 1.0e-12,
        directional_limit[:, None] / absolute_wave,
        torch.inf,
    )
    requested = _validated_mode_amplitudes(amplitude, rho, modes)
    if not torch.is_tensor(requested):
        requested = rho.new_full(modes.shape[:2], requested)
    requested = requested.repeat_interleave(2, dim=1)
    selected = torch.minimum(
        requested,
        0.9 * torch.amin(local_limit, dim=(-2, -1)),
    )
    if torch.any(valid_wave & (selected <= 0.0)).item():
        raise ValueError("no finite-interior polarization displacement is available")
    return selected


def _polarization_energy_differences(
    model,
    outputs,
    batch,
    rho,
    ideal_rho,
    polarization,
    directions,
    ideal,
    relative_amplitude,
    perturbations_per_forward=None,
):
    """Return total and exact-ideal central differences for P probes."""

    delta = relative_amplitude[..., None, None, None] * directions
    plus = polarization[:, None] + delta
    minus = polarization[:, None] - delta
    perturbed = torch.stack((plus, minus), dim=2).flatten(1, 2)
    n_perturbations = perturbed.shape[1]
    expanded_rho = rho[:, None].expand(-1, n_perturbations, -1, -1)
    expanded_ideal_rho = ideal_rho[:, None].expand(
        -1, n_perturbations, -1, -1
    )
    volume = voxel_volume(batch["grid_spacing"].to(rho))
    reference_ideal = _orientation_ideal_energy(
        ideal, ideal_rho, polarization, volume
    )

    ideal_chunks = []
    excess_chunks = []
    chunk_size = perturbations_per_forward or n_perturbations
    for start in range(0, n_perturbations, chunk_size):
        chunk_p = perturbed[:, start:start + chunk_size]
        chunk_rho = expanded_rho[:, start:start + chunk_size]
        chunk_ideal_rho = expanded_ideal_rho[:, start:start + chunk_size]
        expanded_batch = _expand_batch(batch, chunk_rho)
        expanded_batch["dipole_density"] = chunk_p
        perturbed_outputs = _energy_only_model(model, expanded_batch)
        if "beta_F_exc" not in perturbed_outputs:
            raise KeyError("model outputs are missing 'beta_F_exc'")
        ideal_energy = _orientation_ideal_energy(
            ideal,
            chunk_ideal_rho,
            chunk_p,
            volume[:, None].expand(-1, chunk_p.shape[1]),
        )
        ideal_chunks.append(ideal_energy)
        excess_chunks.append(perturbed_outputs["beta_F_exc"])

    perturbed_ideal = torch.cat(ideal_chunks, dim=1).reshape(
        rho.shape[0], directions.shape[1], 2
    )
    perturbed_excess = torch.cat(excess_chunks, dim=1).reshape(
        rho.shape[0], directions.shape[1], 2
    )
    ideal_second = perturbed_ideal.sum(dim=2) - 2.0 * reference_ideal[:, None]
    excess_second = (
        perturbed_excess.sum(dim=2) - 2.0 * outputs["beta_F_exc"][:, None]
    )
    return ideal_second + excess_second, ideal_second


def _orientation_ideal_energy(ideal, rho, polarization, volume):
    """Return only the fixed-dipole orientational ideal contribution."""

    total = ideal(rho, polarization, volume)["beta_F_id"]
    wavelength = rho.new_ones(rho.shape[-1])
    return total - ideal_free_energy(rho, wavelength, volume)


def _energy_only_model(model, batch):
    """Evaluate energy without optional response derivatives."""

    kwargs = {"compute_c1": False}
    if hasattr(model, "compute_polarization_derivative"):
        kwargs["compute_polarization_derivative"] = False
    return model(batch, **kwargs)


def fourier_directions(
    batch: Dict[str, torch.Tensor],
    rho: torch.Tensor,
    modes: torch.Tensor,
    mixture_weights: torch.Tensor,
    mode_phases: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return fixed-number Fourier directions, validity, and density scale."""

    n_fields, n_grid, n_types = rho.shape
    component_directions, valid_component = component_fourier_directions(
        batch,
        rho,
        modes,
        mode_phases=mode_phases,
    )
    total_density = torch.sum(rho, dim=-2)
    component_present = total_density > 1.0e-12

    mixture_weights = mixture_weights.to(rho)
    if mixture_weights.ndim != 2 or mixture_weights.shape[1] != n_types:
        raise ValueError(
            "mixture weights must have shape [n_directions, n_types]"
        )
    active_components = torch.abs(mixture_weights) > 0.0
    n_patterns, n_phases = real_mode_shape(modes, mode_phases)
    valid_by_mode = valid_component.reshape(
        n_fields,
        n_patterns,
        n_phases,
        n_types,
    ).any(dim=2)
    required_components = (
        component_present & active_components.any(dim=0)[None, :]
    )
    if torch.any(required_components[:, None, :] & ~valid_by_mode).item():
        raise ValueError(
            "a requested mode aliases to a constant for a present component"
        )

    directions = (
        component_directions[:, :, None, :, :]
        * mixture_weights[None, None, :, None, :]
    ).flatten(start_dim=1, end_dim=2)
    valid = (
        valid_component[:, :, None, :]
        & active_components[None, None, :, :]
    ).any(dim=-1).flatten(start_dim=1, end_dim=2)

    component_mean_densities = total_density / n_grid
    squared_weights = mixture_weights.square()
    effective_density = torch.einsum(
        "mc,bc->bm",
        squared_weights,
        component_mean_densities.square(),
    ) / torch.clamp(
        torch.einsum(
            "mc,bc->bm",
            squared_weights,
            component_mean_densities,
        ),
        min=1.0e-12,
    )
    mean_densities = effective_density[:, None, :].expand(
        -1,
        component_directions.shape[1],
        -1,
    ).flatten(start_dim=1, end_dim=2)
    return directions.detach(), valid, mean_densities.detach()


def component_fourier_directions(
    batch: Dict[str, torch.Tensor],
    rho: torch.Tensor,
    modes: torch.Tensor,
    mode_phases: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return one fixed-number real Fourier direction per component."""

    n_fields, n_grid, n_types = rho.shape
    waves, _ = _real_fourier_waves(
        batch,
        rho,
        modes,
        mode_phases=mode_phases,
    )

    total_density = torch.sum(rho, dim=-2)
    component_present = total_density > 1.0e-12
    weighted_mean = torch.sum(
        rho[:, None, :, :] * waves[..., None],
        dim=-2,
    ) / torch.clamp(total_density[:, None, :], min=1.0e-12)
    relative_direction = waves[..., None] - weighted_mean[:, :, None, :]
    relative_norm = torch.amax(torch.abs(relative_direction), dim=-2)
    valid_component = (
        (relative_norm > 1.0e-5) & component_present[:, None, :]
    )
    relative_direction = relative_direction / torch.clamp(
        relative_norm[:, :, None, :],
        min=1.0e-12,
    )
    component_directions = rho[:, None, :, :] * relative_direction
    return component_directions.detach(), valid_component


def _real_fourier_waves(
    batch: Dict[str, torch.Tensor],
    reference: torch.Tensor,
    modes: torch.Tensor,
    mode_phases: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return cosine/sine waves and a mask for nonzero real waves."""

    n_fields, n_grid = reference.shape[:2]
    positions = batch["grid_positions"].to(reference)
    grid_size = batch["grid_size"].to(reference)
    if positions.shape != (n_fields, n_grid, 3):
        raise ValueError(
            "grid_positions must have shape [n_fields, n_grid, 3]"
        )
    if grid_size.shape != (n_fields, 3):
        raise ValueError("grid_size must have shape [n_fields, 3]")
    if modes.ndim != 3 or modes.shape[0] != n_fields or modes.shape[2] != 3:
        raise ValueError("modes must have shape [n_fields, n_modes, 3]")

    phase = 2.0 * torch.pi * torch.sum(
        positions[:, None, :, :]
        * modes.to(reference)[:, :, None, :]
        / grid_size[:, None, None, :],
        dim=-1,
    )
    # A self-conjugate grid mode has an identically zero sine phase. Enforce
    # this algebraically: float32 sin(m*pi) roundoff at Nyquist corners can
    # otherwise be normalized into a spurious finite perturbation.
    self_conjugate = torch.all(
        torch.remainder(2 * modes, batch["grid_size"].to(modes)[:, None, :]) == 0,
        dim=-1,
    )
    sine = torch.sin(phase).masked_fill(self_conjugate[:, :, None], 0.0)
    if mode_phases is None:
        waves = torch.stack((torch.cos(phase), sine), dim=2)
        waves = waves.flatten(start_dim=1, end_dim=2)
    else:
        if (
            not torch.is_tensor(mode_phases)
            or mode_phases.shape != modes.shape[:2]
            or mode_phases.dtype == torch.bool
            or torch.is_complex(mode_phases)
            or not torch.all(torch.isfinite(mode_phases)).item()
        ):
            raise ValueError("mode_phases must be a finite real [field, mode] tensor")
        offsets = mode_phases.detach().to(reference)[..., None]
        # Sum BEFORE projecting/normalizing. Keep exact zero sine at Nyquist
        # rather than amplifying sin(m*pi) roundoff into a spurious wave.
        waves = (
            torch.cos(phase) * torch.cos(offsets) - sine * torch.sin(offsets)
        ).sum(dim=1, keepdim=True)

    valid = torch.amax(torch.abs(waves), dim=-1) > 1.0e-5
    return waves, valid


def average_fourier_phases(
    curvature: torch.Tensor,
    valid: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Average valid cosine and sine curvatures for each target."""

    if curvature.ndim != 4 or curvature.shape[2] != 2:
        raise ValueError(
            "curvature must have shape [field, mode, phase, direction]"
        )
    if valid.shape != curvature.shape:
        raise ValueError("valid and curvature must have the same shape")
    phase_count = valid.sum(dim=2)
    prediction = torch.sum(curvature * valid.to(curvature), dim=2)
    prediction = prediction / torch.clamp(phase_count, min=1)
    return prediction, phase_count > 0


def normalized_directions(
    directions: Sequence[Sequence[float]],
) -> torch.Tensor:
    """Return finite nonzero direction rows with common scales removed."""

    value = torch.as_tensor(directions, dtype=torch.get_default_dtype())
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError("directions must have shape [n_directions, n_types]")
    if not torch.all(torch.isfinite(value)).item():
        raise ValueError("directions must be finite")
    scale = torch.amax(torch.abs(value), dim=-1, keepdim=True)
    if torch.any(scale == 0.0).item():
        raise ValueError("every direction must contain a nonzero value")
    return (value / scale).detach().clone()


def validate_response(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    n_types: int,
    require_uniform: bool = False,
) -> torch.Tensor:
    """Return a validated density for a projected Fourier response."""

    if "beta_F_exc" not in outputs:
        raise KeyError("model outputs are missing 'beta_F_exc'")
    for key in (
        "rho",
        "grid_positions",
        "grid_size",
        "grid_spacing",
        "temperature",
    ):
        if key not in batch:
            raise KeyError("batch is missing '{}'".format(key))

    rho = batch["rho"]
    if rho.ndim != 3:
        raise ValueError("rho must have shape [n_fields, n_grid, n_types]")
    if rho.shape[-1] != n_types:
        raise ValueError("directions must contain one value per density type")
    if not torch.all(torch.isfinite(rho)).item():
        raise ValueError("rho must be finite")
    if torch.any(rho < 0.0).item():
        raise ValueError("rho must be nonnegative")
    if outputs["beta_F_exc"].shape != rho.shape[:-2]:
        raise ValueError("beta_F_exc must contain one value per field")
    if require_uniform:
        spatial_range = torch.amax(rho, dim=-2) - torch.amin(rho, dim=-2)
        spatial_scale = torch.clamp(torch.amax(rho, dim=-2), min=1.0)
        if torch.any(spatial_range > 1.0e-7 * spatial_scale).item():
            raise ValueError("Fourier response rho must be spatially uniform")
        if "excluded_mask" in batch and torch.any(
            batch["excluded_mask"]
        ).item():
            raise ValueError(
                "homogeneous Fourier response batches must not exclude grid points"
            )
    return rho


def validate_uniform_response(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    n_types: int,
) -> torch.Tensor:
    """Return a validated homogeneous, periodic, unmasked response density."""

    return validate_response(
        outputs,
        batch,
        n_types=n_types,
        require_uniform=True,
    )


def validate_explicit_modes(
    batch: Dict[str, torch.Tensor],
    rho: torch.Tensor,
    supplied_modes: torch.Tensor,
    mode_domain: str = "sphere",
) -> torch.Tensor:
    """Validate and canonicalize per-field integer response modes."""

    modes = torch.as_tensor(supplied_modes)
    n_fields = rho.shape[0]
    if (
        modes.ndim != 3
        or modes.shape[0] != n_fields
        or modes.shape[1] == 0
        or modes.shape[2] != 3
    ):
        raise ValueError(
            "response modes must have shape [n_fields, n_modes, 3]"
        )
    grid_size, grid_spacing = _validated_grid(
        batch,
        n_fields,
        n_grid=rho.shape[1],
    )
    canonical_by_field = []
    for field in range(n_fields):
        size = tuple(grid_size[field].tolist())
        spacing = tuple(grid_spacing[field].tolist())
        canonical_by_field.append(
            canonical_mode_triplets(
                modes[field],
                size,
                spacing,
                name="response modes",
                mode_domain=mode_domain,
            )
        )
    return torch.stack(canonical_by_field).to(device=rho.device)


def feasible_modes(
    grid_size: Sequence[int],
    grid_spacing: Sequence[float],
    mode_domain: str = "sphere",
) -> Sequence[Tuple[int, int, int]]:
    """Return unique real modes in the selected grid-representable domain."""

    mode_domain = validated_mode_domain(mode_domain)
    size_tensor, spacing_tensor = _validated_grid_geometry(
        grid_size,
        grid_spacing,
    )
    grid_size = tuple(size_tensor.tolist())
    grid_spacing = tuple(spacing_tensor.tolist())

    half_sizes = [size // 2 for size in grid_size]
    feasible = set()
    for nx in range(-half_sizes[0], half_sizes[0] + 1):
        for ny in range(-half_sizes[1], half_sizes[1] + 1):
            for nz in range(-half_sizes[2], half_sizes[2] + 1):
                mode = (nx, ny, nz)
                if mode != (0, 0, 0) and _mode_is_feasible(
                    mode,
                    grid_size,
                    grid_spacing,
                    mode_domain,
                ):
                    feasible.add(canonical_grid_mode(mode, grid_size))
    return sorted(feasible)


def wavevector_magnitude(
    mode: Sequence[int],
    box_lengths: Sequence[float],
) -> float:
    """Return the physical reciprocal-space magnitude of an integer mode."""

    return math.sqrt(
        sum(
            (2.0 * math.pi * component / length) ** 2
            for component, length in zip(mode, box_lengths)
        )
    )


def canonical_grid_mode(
    mode: Sequence[int],
    grid_size: Sequence[int],
) -> Tuple[int, int, int]:
    """Remove Nyquist-sign and global-sign duplicates of a real Fourier mode."""

    canonical = []
    for component, size in zip(mode, grid_size):
        if size % 2 == 0 and abs(component) == size // 2:
            component = abs(component)
        canonical.append(component)
    # Nyquist components are their own negatives on an even grid. Select
    # the sign with a positive first non-Nyquist component; the two signs
    # coincide when every nonzero component is Nyquist.
    opposite = [
        component
        if size % 2 == 0 and component == size // 2
        else -component
        for component, size in zip(canonical, grid_size)
    ]
    return max(tuple(canonical), tuple(opposite))


def validated_mode_domain(mode_domain: str) -> str:
    """Return the requested Nyquist-domain convention."""

    if mode_domain not in ("sphere", "cube"):
        raise ValueError("mode_domain must be 'sphere' or 'cube'")
    return mode_domain


def _validated_grid(
    batch: Dict[str, torch.Tensor],
    n_fields: int,
    n_grid: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return validated CPU grid sizes and spacings."""

    grid_size = batch["grid_size"].detach().cpu()
    grid_spacing = batch["grid_spacing"].detach().cpu()
    if grid_size.shape != (n_fields, 3):
        raise ValueError("grid_size must have shape [n_fields, 3]")
    if grid_spacing.shape != (n_fields, 3):
        raise ValueError("grid_spacing must have shape [n_fields, 3]")
    geometries = [
        _validated_grid_geometry(grid_size[field], grid_spacing[field])
        for field in range(n_fields)
    ]
    integer_grid_size = torch.stack([item[0] for item in geometries])
    grid_spacing = torch.stack([item[1] for item in geometries])
    if n_grid is not None and torch.any(
        torch.prod(integer_grid_size, dim=-1) != n_grid
    ).item():
        raise ValueError("grid_size product must match the number of grid points")
    return integer_grid_size, grid_spacing


def _validated_grid_geometry(
    grid_size: object,
    grid_spacing: object,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return one positive integer grid size and finite positive spacing."""

    size = torch.as_tensor(grid_size)
    spacing = torch.as_tensor(grid_spacing)
    if size.dtype == torch.bool or torch.is_complex(size):
        raise TypeError("grid_size must contain real integer values")
    if size.shape != (3,):
        raise ValueError("grid_size must contain three positive integers")
    if not torch.all(torch.isfinite(size)).item():
        raise ValueError("grid_size must be finite")
    integer_size = size.to(torch.long)
    if not torch.equal(size, integer_size.to(size.dtype)):
        raise ValueError("grid_size must contain integers")
    if torch.any(integer_size <= 0).item():
        raise ValueError("grid_size must contain three positive integers")

    if spacing.dtype == torch.bool or torch.is_complex(spacing):
        raise TypeError("grid_spacing must contain real values")
    if spacing.shape != (3,):
        raise ValueError("grid_spacing must contain three positive values")
    if (
        not torch.all(torch.isfinite(spacing)).item()
        or torch.any(spacing <= 0.0).item()
    ):
        raise ValueError("grid_spacing must contain three finite positive values")
    return integer_size, spacing


def _mode_is_feasible(
    mode: Sequence[int],
    grid_size: Sequence[int],
    grid_spacing: Sequence[float],
    mode_domain: str = "sphere",
) -> bool:
    """Return whether a mode is on-grid and inside the requested domain."""

    if any(
        abs(component) > size // 2
        for component, size in zip(mode, grid_size)
    ):
        return False
    if mode_domain == "cube":
        return True
    box_lengths = tuple(
        size * spacing for size, spacing in zip(grid_size, grid_spacing)
    )
    isotropic_nyquist = min(math.pi / spacing for spacing in grid_spacing)
    return (
        wavevector_magnitude(mode, box_lengths) ** 2
        <= isotropic_nyquist**2 * (1.0 + 1.0e-12)
    )


def canonical_mode_triplets(
    supplied_modes: object,
    grid_size: Sequence[int],
    grid_spacing: Sequence[float],
    name: str = "modes",
    mode_domain: str = "sphere",
) -> torch.Tensor:
    """Validate and canonicalize explicit modes for one periodic grid."""

    mode_domain = validated_mode_domain(mode_domain)
    modes = mode_triplets(supplied_modes, name)
    size, spacing = _validated_grid_geometry(grid_size, grid_spacing)
    grid_size = tuple(size.tolist())
    grid_spacing = tuple(spacing.tolist())
    selected = [
        canonical_grid_mode(mode, grid_size)
        for mode in modes.detach().cpu().tolist()
    ]
    if len(set(selected)) != len(selected):
        raise ValueError("{} contain equivalent Fourier directions".format(name))
    if any(
        not _mode_is_feasible(mode, grid_size, grid_spacing, mode_domain)
        for mode in selected
    ):
        domain_name = (
            "isotropic Nyquist sphere"
            if mode_domain == "sphere"
            else "componentwise Nyquist cube"
        )
        raise ValueError("a requested mode lies outside the " + domain_name)
    return torch.tensor(selected, dtype=torch.long, device=modes.device)


def _expand_batch(
    batch: Dict[str, torch.Tensor],
    rho: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Insert the perturbation axis into field-wise batch tensors."""

    n_fields, n_perturbations = rho.shape[:2]
    expanded = {}
    for key, value in batch.items():
        if key == "rho":
            expanded[key] = rho
        elif (
            torch.is_tensor(value)
            and value.ndim > 0
            and value.shape[0] == n_fields
        ):
            expanded[key] = value.detach().unsqueeze(1).expand(
                n_fields,
                n_perturbations,
                *value.shape[1:]
            )
        else:
            expanded[key] = value
    return expanded
