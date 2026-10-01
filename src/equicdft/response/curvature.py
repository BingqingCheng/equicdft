"""Excess finite-difference curvature plus declared ideal stability references."""

from typing import (
    Dict,
    Optional,
    Tuple,
    Union,
)
import torch
from torch import nn
from .._validation import finite_scalar
from ..geometry.grid import voxel_volume
from ..functional.thermodynamics import ideal_free_energy
from ..functional.polarization_ideal import FixedDipoleIdeal
from .modes import (
    _real_fourier_waves,
    _validated_probe_amplitudes,
    component_fourier_directions,
    expand_probe_amplitudes,
    probe_shape,
)

__all__ = [
    "projected_fourier_curvature",
    "fourier_curvature_matrix",
    "polarization_fourier_curvature",
]


def projected_fourier_curvature(
    model: nn.Module,
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    rho: torch.Tensor,
    directions: torch.Tensor,
    valid_directions: torch.Tensor,
    mean_densities: torch.Tensor,
    relative_amplitude: Union[float, torch.Tensor],
    perturbations_per_forward: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate excess plus translational ideal curvature along fixed directions."""

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
    perturbations_per_forward: Optional[int] = None,
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
    # These probes conserve each component's particle number, so the
    # thermal-wavelength term is linear and has zero projected curvature.
    thermal_wavelength = rho.new_ones(rho.shape[-1])
    if "local_energy_mask" in outputs:
        # A field loss may have supplied a masked fitting forward. Reference
        # and displaced energies must use the same full excess functional.
        outputs = _energy_only_model(model, dict(batch, rho=rho))
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
    wavevector_indices: torch.Tensor,
    relative_amplitude: Union[float, torch.Tensor],
    perturbations_per_forward: Optional[int] = None,
    wavevector_phases: Optional[torch.Tensor] = None,
    density_power: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Return the ideal-metric component Hessian for every real probe.

    The returned matrix has shape
    ``[field, wavevector, phase, type, type]``.
    For a homogeneous mixture it is the dimensionless inverse OZ response
    ``I - sqrt(R) c(k) sqrt(R)``. Around an inhomogeneous density it is the
    component Hessian projected onto the selected real Fourier perturbations.
    With ``wavevector_phases`` supplied, waves are summed before projection and the
    output has shape [field, 1, 1, type, type]; it includes cross-wavevector
    contributions but does not reconstruct the complete spatial Hessian. Pair
    polarization adds numerical cancellation, so matrix applications should
    check convergence with respect to ``relative_amplitude``.
    ``density_power = p > 0`` weights every component wave by
    ``(rho_total / max rho_total)^p`` (see
    :func:`~equicdft.response.modes.component_fourier_directions`); the
    ideal-metric normalization is unchanged, so 1 remains the ideal gas.
    """

    component_directions, active = component_fourier_directions(
        batch,
        rho,
        wavevector_indices,
        wavevector_phases=wavevector_phases,
        density_power=density_power,
    )
    n_fields, n_real_probes, _, n_types = component_directions.shape
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
    n_patterns, n_phases = probe_shape(wavevector_indices, wavevector_phases)
    amplitude = expand_probe_amplitudes(
        relative_amplitude,
        rho,
        wavevector_indices[:, :n_patterns],
        weights.shape[0],
        n_phases,
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
        amplitude.reshape(n_fields, n_real_probes, weights.shape[0]).square()
        if torch.is_tensor(amplitude) else amplitude**2
    )
    quadratic_forms = second_difference.reshape(
        n_fields,
        n_real_probes,
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
    wavevector_indices: torch.Tensor,
    relative_amplitude: Union[float, torch.Tensor],
    polarization_directions: torch.Tensor,
    perturbations_per_forward: Optional[int] = None,
    wavevector_phases: Optional[torch.Tensor] = None,
    density_power: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Return excess plus quadratic-alignment ideal curvature per field.

    The moment comes from model.ideal; exact orientational entropy is not
    evaluated, so probes may cross its local physical domain boundary.

    The selected basis displacement applies one shared Cartesian alignment
    change to every species, scaled locally by ``m_a*rho_a``. The result has
    shape ``[field, wavevector, phase]`` independent of species count. With
    ``wavevector_phases`` supplied, all selected waves are summed into one
    pattern and the result has shape ``[field, 1, 1]``. Density is held fixed,
    including for the zero wavevector.

    ``polarization_directions`` is either one unit vector per field,
    ``[field, 3]``, shared by every wave, or one per field and wavevector,
    ``[field, n_wavevectors, 3]``, so that each wave can be polarized
    relative to its own wavevector (transverse or longitudinal probes). In
    the summed pattern each wave carries its own direction, so a sum of
    transverse waves stays divergence-free.

    ``density_power = p > 0`` multiplies every alignment wave by
    ``(rho_total / max rho_total)^p`` per field before normalization, which
    localizes the probe in the densest voxels. The returned curvature is
    still total over ideal second variation, so it measures the stiffness of
    the dense regions instead of the field average (a box-wide wave passes a
    floor while a dense layer is locally unstable). The weighting is not
    divergence-free where the density gradient has a component along the
    wave's polarization; it is exact for layers normal to the wavevector.
    """

    if "dipole_density" not in batch:
        raise KeyError("batch is missing 'dipole_density'")
    polarization = batch["dipole_density"]
    if polarization.shape != rho.shape + (3,):
        raise ValueError("dipole_density must have shape rho.shape + (3,)")
    if polarization.dtype != rho.dtype or polarization.device != rho.device:
        raise ValueError("rho and dipole_density must share dtype and device")
    if not torch.all(torch.isfinite(polarization)).item():
        raise ValueError("dipole_density must be finite")
    zero_density = rho == 0.0
    if torch.any(zero_density[..., None] & (polarization != 0.0)).item():
        raise ValueError("dipole_density must be zero where rho is zero")

    ideal = getattr(model, "ideal", None)
    if not isinstance(ideal, FixedDipoleIdeal):
        raise TypeError("polarization stability requires model.ideal=FixedDipoleIdeal(...)")
    moment = ideal.dipole_magnitude.to(rho).reshape(-1)
    if moment.numel() not in (1, rho.shape[-1]):
        raise ValueError("dipole_magnitude must contain one value per density type")

    n_wavevectors = wavevector_indices.shape[1]
    selected_direction = _unit_polarization_directions(
        polarization_directions, rho, n_wavevectors,
    )
    vector_waves, valid_wave = _real_fourier_vector_waves(
        batch,
        rho,
        wavevector_indices,
        selected_direction,
        wavevector_phases=wavevector_phases,
    )
    density_power = finite_scalar(density_power, "density_power")
    if density_power < 0.0:
        raise ValueError("density_power must be nonnegative")
    if density_power > 0.0:
        total = rho.sum(dim=-1)
        relative = total / torch.clamp(
            torch.amax(total, dim=-1, keepdim=True), min=torch.finfo(rho.dtype).tiny,
        )
        vector_waves = vector_waves * (relative ** density_power)[:, None, :, None]
    wave_norm = torch.amax(
        torch.linalg.vector_norm(vector_waves, dim=-1), dim=-1,
    )
    vector_waves = vector_waves / torch.clamp(
        wave_norm[..., None, None], min=1.0e-12,
    )

    alignment_scale = rho * moment
    directions = (
        vector_waves[:, :, :, None, :]
        * alignment_scale[:, None, :, :, None]
    )

    n_patterns, n_phases = probe_shape(wavevector_indices, wavevector_phases)
    amplitude_indices = (
        wavevector_indices
        if wavevector_phases is None
        else wavevector_indices[:, :1]
    )
    used_amplitude = _validated_probe_amplitudes(
        relative_amplitude,
        rho,
        amplitude_indices,
    )
    if not torch.is_tensor(used_amplitude):
        used_amplitude = rho.new_full(
            amplitude_indices.shape[:2], used_amplitude
        )
    used_amplitude = used_amplitude.repeat_interleave(n_phases, dim=1)
    excess_second = _polarization_excess_energy_difference(
        model=model,
        outputs=outputs,
        batch=batch,
        rho=rho,
        polarization=polarization,
        directions=directions,
        relative_amplitude=used_amplitude,
        perturbations_per_forward=perturbations_per_forward,
    )
    volume = voxel_volume(batch["grid_spacing"].to(rho))
    alignment_change = used_amplitude[..., None, None] * vector_waves
    ideal_second = (
        3.0
        * volume[:, None]
        * torch.sum(
            rho.sum(dim=-1)[:, None] * alignment_change.square().sum(dim=-1),
            dim=-1,
        )
    )
    total_second = ideal_second + excess_second
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
        curvature.reshape(rho.shape[0], n_patterns, n_phases),
        valid.reshape(rho.shape[0], n_patterns, n_phases),
    )


def _unit_polarization_directions(directions, reference, n_wavevectors):
    """Validate and normalize directions to ``[field, n_wavevectors, 3]``."""

    directions = torch.as_tensor(directions, device=reference.device)
    n_fields = reference.shape[0]
    if (
        directions.dtype == torch.bool
        or torch.is_complex(directions)
        or directions.shape not in ((n_fields, 3), (n_fields, n_wavevectors, 3))
        or not torch.all(torch.isfinite(directions)).item()
    ):
        raise ValueError(
            "polarization_directions must be finite with shape [field, 3] "
            "or [field, n_wavevectors, 3]"
        )
    directions = directions.to(reference)
    if directions.ndim == 2:
        directions = directions[:, None, :].expand(n_fields, n_wavevectors, 3)
    norm = torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
    if torch.any(norm == 0.0).item():
        raise ValueError("polarization_directions must be nonzero")
    return directions / norm


def _real_fourier_vector_waves(
    batch, reference, wavevector_indices, directions, wavevector_phases=None,
):
    """Return vector wave patterns ``[field, n_probe, n_grid, 3]`` and validity.

    Every wave is multiplied by its own unit direction. Without phases the
    probes are the cosine and sine of every wavevector in turn; with phases
    the shifted waves are summed into one pattern per field before any
    normalization, so a sum of transverse waves remains transverse.
    """

    if wavevector_phases is None:
        waves, valid = _real_fourier_waves(batch, reference, wavevector_indices)
        per_probe = directions.repeat_interleave(2, dim=1)
        return waves[..., None] * per_probe[:, :, None, :], valid
    n_waves = wavevector_indices.shape[1]
    patterns = []
    for index in range(n_waves):
        single, _ = _real_fourier_waves(
            batch, reference, wavevector_indices[:, index : index + 1],
            wavevector_phases=wavevector_phases[:, index : index + 1],
        )
        patterns.append(single[..., None] * directions[:, index : index + 1, None, :])
    summed = torch.stack(patterns, dim=0).sum(dim=0)
    valid = torch.amax(torch.linalg.vector_norm(summed, dim=-1), dim=-1) > 1.0e-5
    return summed, valid


def _polarization_excess_energy_difference(
    model,
    outputs,
    batch,
    rho,
    polarization,
    directions,
    relative_amplitude,
    perturbations_per_forward=None,
):
    """Return excess-energy central differences for polarization probes."""

    if "local_energy_mask" in outputs:
        outputs = _energy_only_model(
            model, dict(batch, rho=rho, dipole_density=polarization),
        )

    delta = relative_amplitude[..., None, None, None] * directions
    plus = polarization[:, None] + delta
    minus = polarization[:, None] - delta
    perturbed = torch.stack((plus, minus), dim=2).flatten(1, 2)
    n_perturbations = perturbed.shape[1]
    expanded_rho = rho[:, None].expand(-1, n_perturbations, -1, -1)

    excess_chunks = []
    chunk_size = perturbations_per_forward or n_perturbations
    for start in range(0, n_perturbations, chunk_size):
        chunk_p = perturbed[:, start:start + chunk_size]
        chunk_rho = expanded_rho[:, start:start + chunk_size]
        expanded_batch = _expand_batch(batch, chunk_rho)
        expanded_batch["dipole_density"] = chunk_p
        perturbed_outputs = _energy_only_model(model, expanded_batch)
        if "beta_F_exc" not in perturbed_outputs:
            raise KeyError("model outputs are missing 'beta_F_exc'")
        excess_chunks.append(perturbed_outputs["beta_F_exc"])

    perturbed_excess = torch.cat(excess_chunks, dim=1).reshape(
        rho.shape[0], directions.shape[1], 2
    )
    return (
        perturbed_excess.sum(dim=2) - 2.0 * outputs["beta_F_exc"][:, None]
    )


def _energy_only_model(model, batch):
    """Evaluate energy without optional response derivatives."""

    kwargs = {"compute_c1": False}
    if hasattr(model, "voxel_weights"):
        kwargs["use_voxel_mask"] = False
    if hasattr(model, "compute_local_mu"):
        kwargs["compute_local_mu"] = False
    if hasattr(model, "ideal"):
        kwargs["compute_ideal"] = False
    if hasattr(model, "compute_polarization_derivative"):
        kwargs["compute_polarization_derivative"] = False
    return model(batch, **kwargs)


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
