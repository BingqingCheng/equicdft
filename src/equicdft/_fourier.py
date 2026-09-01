"""Shared numerical helpers for projected periodic Fourier curvatures."""

import math
from typing import Dict, Sequence, Tuple

import torch
from torch import nn

from ._grid import voxel_volume
from .energy import density_weighted_integral


def projected_fourier_curvature(
    model: nn.Module,
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    rho: torch.Tensor,
    directions: torch.Tensor,
    valid_directions: torch.Tensor,
    mean_densities: torch.Tensor,
    relative_amplitude: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate normalized total intrinsic curvature along fixed directions."""

    delta_rho = relative_amplitude * directions
    rho_plus = rho[:, None, :, :] + delta_rho
    rho_minus = rho[:, None, :, :] - delta_rho
    if torch.any(rho_plus < -1.0e-7).item() or torch.any(
        rho_minus < -1.0e-7
    ).item():
        raise RuntimeError("Fourier perturbation produced negative density")

    n_directions = directions.shape[1]
    perturbed_rho = torch.cat((rho_plus, rho_minus), dim=1)
    perturbed_outputs = model(
        _expand_batch(batch, perturbed_rho),
        compute_c1=False,
    )
    if "beta_F_exc" not in perturbed_outputs:
        raise KeyError("model outputs are missing 'beta_F_exc'")

    volume_element = voxel_volume(batch["grid_spacing"].to(rho))
    reference_energy = (
        _ideal_free_energy(rho, volume_element) + outputs["beta_F_exc"]
    )
    perturbed_energy = (
        _ideal_free_energy(
            perturbed_rho,
            volume_element[:, None].expand(-1, 2 * n_directions),
        )
        + perturbed_outputs["beta_F_exc"]
    )
    second_difference = (
        perturbed_energy[:, :n_directions]
        + perturbed_energy[:, n_directions:]
        - 2.0 * reference_energy[:, None]
    )
    perturbation_norm = volume_element[:, None] * torch.sum(
        delta_rho.square(),
        dim=(-2, -1),
    )
    curvature = (
        mean_densities
        * second_difference
        / torch.clamp(perturbation_norm, min=1.0e-12)
    )
    valid = valid_directions & (perturbation_norm > 1.0e-12)
    return curvature, valid


def fourier_directions(
    batch: Dict[str, torch.Tensor],
    rho: torch.Tensor,
    modes: torch.Tensor,
    mixture_weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return fixed-number Fourier directions, validity, and density scale."""

    n_fields, n_grid, n_types = rho.shape
    positions = batch["grid_positions"].to(rho)
    grid_size = batch["grid_size"].to(rho)
    if positions.shape != (n_fields, n_grid, 3):
        raise ValueError(
            "grid_positions must have shape [n_fields, n_grid, 3]"
        )
    if grid_size.shape != (n_fields, 3):
        raise ValueError("grid_size must have shape [n_fields, 3]")
    if modes.ndim != 3 or modes.shape[0] != n_fields or modes.shape[2] != 3:
        raise ValueError("modes must have shape [n_fields, n_modes, 3]")
    if mixture_weights.ndim != 2 or mixture_weights.shape[1] != n_types:
        raise ValueError(
            "mixture weights must have shape [n_directions, n_types]"
        )

    phase = 2.0 * torch.pi * torch.sum(
        positions[:, None, :, :]
        * modes.to(rho)[:, :, None, :]
        / grid_size[:, None, None, :],
        dim=-1,
    )
    waves = torch.stack((torch.cos(phase), torch.sin(phase)), dim=2)
    waves = waves.flatten(start_dim=1, end_dim=2)

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

    mixture_weights = mixture_weights.to(rho)
    active_components = torch.abs(mixture_weights) > 0.0
    valid_by_mode = valid_component.reshape(
        n_fields,
        modes.shape[1],
        2,
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
        waves.shape[1],
        -1,
    ).flatten(start_dim=1, end_dim=2)
    return directions.detach(), valid, mean_densities.detach()


def average_fourier_phases(
    curvature: torch.Tensor,
    valid: torch.Tensor,
    n_modes: int,
    n_directions: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Average valid cosine and sine curvatures for each target."""

    shape = (curvature.shape[0], n_modes, 2, n_directions)
    curvature = curvature.reshape(shape)
    valid = valid.reshape(shape)
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


def validate_uniform_response(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    n_types: int,
) -> torch.Tensor:
    """Return a validated homogeneous, periodic, unmasked response density."""

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
    spatial_range = torch.amax(rho, dim=-2) - torch.amin(rho, dim=-2)
    spatial_scale = torch.clamp(torch.amax(rho, dim=-2), min=1.0)
    if torch.any(spatial_range > 1.0e-7 * spatial_scale).item():
        raise ValueError("Fourier response rho must be spatially uniform")
    if "excluded_mask" in batch and torch.any(batch["excluded_mask"]).item():
        raise ValueError("Fourier response batches must not exclude grid points")
    return rho


def validate_explicit_modes(
    batch: Dict[str, torch.Tensor],
    rho: torch.Tensor,
    supplied_modes: torch.Tensor,
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
    integer_modes = modes.to(torch.long)
    if not torch.equal(modes, integer_modes.to(modes.dtype)):
        raise ValueError("response modes must contain integers")
    if torch.any(torch.all(integer_modes == 0, dim=-1)).item():
        raise ValueError("response modes must not contain the zero mode")

    grid_size, grid_spacing = _validated_grid(batch, n_fields)
    canonical_by_field = []
    for field in range(n_fields):
        size = tuple(grid_size[field].tolist())
        spacing = tuple(grid_spacing[field].tolist())
        selected = [
            canonical_grid_mode(mode, size)
            for mode in integer_modes[field].detach().cpu().tolist()
        ]
        if len(set(selected)) != len(selected):
            raise ValueError("response modes contain equivalent directions")
        if any(not _mode_is_feasible(mode, size, spacing) for mode in selected):
            raise ValueError(
                "a response mode lies outside the isotropic Nyquist sphere"
            )
        canonical_by_field.append(torch.tensor(selected, dtype=torch.long))
    return torch.stack(canonical_by_field).to(device=rho.device)


def feasible_modes(
    grid_size: Sequence[int],
    grid_spacing: Sequence[float],
) -> Sequence[Tuple[int, int, int]]:
    """Return unique modes inside the physical isotropic Nyquist sphere."""

    if len(grid_size) != 3 or any(size <= 0 for size in grid_size):
        raise ValueError("grid_size must contain three positive integers")
    if (
        len(grid_spacing) != 3
        or any(not math.isfinite(spacing) for spacing in grid_spacing)
        or any(spacing <= 0.0 for spacing in grid_spacing)
    ):
        raise ValueError("grid_spacing must contain three positive values")

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
    first_nonzero = next(component for component in canonical if component != 0)
    if first_nonzero < 0:
        canonical = [-component for component in canonical]
    return tuple(canonical)


def _validated_grid(
    batch: Dict[str, torch.Tensor],
    n_fields: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return validated CPU grid sizes and spacings."""

    grid_size = batch["grid_size"].detach().cpu()
    grid_spacing = batch["grid_spacing"].detach().cpu()
    if grid_size.shape != (n_fields, 3):
        raise ValueError("grid_size must have shape [n_fields, 3]")
    if grid_spacing.shape != (n_fields, 3):
        raise ValueError("grid_spacing must have shape [n_fields, 3]")
    integer_grid_size = grid_size.to(torch.long)
    if not torch.equal(grid_size, integer_grid_size.to(grid_size.dtype)):
        raise ValueError("grid_size must contain integers")
    return integer_grid_size, grid_spacing


def _mode_is_feasible(
    mode: Sequence[int],
    grid_size: Sequence[int],
    grid_spacing: Sequence[float],
) -> bool:
    """Return whether a mode lies on-grid and inside the isotropic sphere."""

    if any(
        abs(component) > size // 2
        for component, size in zip(mode, grid_size)
    ):
        return False
    box_lengths = tuple(
        size * spacing for size, spacing in zip(grid_size, grid_spacing)
    )
    isotropic_nyquist = min(math.pi / spacing for spacing in grid_spacing)
    return (
        wavevector_magnitude(mode, box_lengths) ** 2
        <= isotropic_nyquist**2 * (1.0 + 1.0e-12)
    )


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


def _ideal_free_energy(
    rho: torch.Tensor,
    volume_element: torch.Tensor,
) -> torch.Tensor:
    """Return discrete ``beta*F_id``; omitted linear terms cancel."""

    positive = rho > 0.0
    safe_density = torch.where(positive, rho, torch.ones_like(rho))
    per_particle = torch.where(
        positive,
        torch.log(safe_density) - 1.0,
        torch.zeros_like(rho),
    )
    return density_weighted_integral(rho, per_particle, volume_element)
