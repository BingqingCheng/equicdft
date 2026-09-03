"""Shared numerical helpers for projected periodic Fourier curvatures."""

import math
from typing import Dict, Sequence, Tuple

import torch
from torch import nn

from ._grid import voxel_volume
from .energy import ideal_free_energy


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


def projected_fourier_curvature(
    model: nn.Module,
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    rho: torch.Tensor,
    directions: torch.Tensor,
    valid_directions: torch.Tensor,
    mean_densities: torch.Tensor,
    relative_amplitude: float,
    perturbations_per_forward: int = None,
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
        perturbed_outputs = model(
            _expand_batch(batch, chunk),
            compute_c1=False,
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
            )
        )
    return torch.stack(canonical_by_field).to(device=rho.device)


def feasible_modes(
    grid_size: Sequence[int],
    grid_spacing: Sequence[float],
) -> Sequence[Tuple[int, int, int]]:
    """Return unique modes inside the physical isotropic Nyquist sphere."""

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


def canonical_mode_triplets(
    supplied_modes: object,
    grid_size: Sequence[int],
    grid_spacing: Sequence[float],
    name: str = "modes",
) -> torch.Tensor:
    """Validate and canonicalize explicit modes for one periodic grid."""

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
        not _mode_is_feasible(mode, grid_size, grid_spacing)
        for mode in selected
    ):
        raise ValueError(
            "a requested mode lies outside the isotropic Nyquist sphere"
        )
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
