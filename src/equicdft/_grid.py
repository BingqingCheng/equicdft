"""Small tensor helpers for regular three-dimensional grids."""

import math
from typing import Optional

import torch


def gather_neighbors(
    values: torch.Tensor,
    neighbor_index: torch.Tensor,
) -> torch.Tensor:
    """Gather periodic neighborhoods of a grid-aligned tensor.

    ``values`` has shape ``[..., n_grid, *feature_shape]`` and the integer
    neighbor table has shape ``[..., n_grid, n_neighbors]``. The returned
    tensor has shape ``[..., n_grid, n_neighbors, *feature_shape]``.
    """

    if neighbor_index.ndim < 2:
        raise ValueError(
            "neighbor_index must have shape [..., n_grid, n_neighbors]"
        )
    if neighbor_index.dtype != torch.long:
        raise TypeError("neighbor_index must have dtype torch.long")

    n_leading = neighbor_index.ndim - 2
    if values.ndim <= n_leading:
        raise ValueError(
            "values must have shape [..., n_grid, *feature_shape]"
        )
    leading_shape = neighbor_index.shape[:-2]
    if values.shape[:n_leading] != leading_shape:
        raise ValueError("values and neighbor_index leading shapes must match")
    n_grid = neighbor_index.shape[-2]
    if values.shape[n_leading] != n_grid:
        raise ValueError("values and neighbor_index grid sizes must match")

    feature_shape = values.shape[n_leading + 1 :]
    n_features = math.prod(feature_shape) if feature_shape else 1
    n_neighbors = neighbor_index.shape[-1]

    # Flatten leading field dimensions and trailing feature dimensions around
    # the grid axis. torch.gather then selects the grid axis independently for
    # every field and feature.
    values_flat = values.reshape(-1, n_grid, n_features)
    index_flat = neighbor_index.reshape(-1, n_grid * n_neighbors)
    gather_index = index_flat.unsqueeze(-1).expand(-1, -1, n_features)
    local_values = torch.gather(values_flat, dim=1, index=gather_index)
    return local_values.reshape(
        *leading_shape,
        n_grid,
        n_neighbors,
        *feature_shape,
    )


def grid_spacing_tensor(
    grid_spacing: object,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return finite positive Cartesian spacing with shape ``[3]``.

    A scalar spacing is expanded to the three Cartesian directions. The
    returned tensor is detached and cloned because grid spacing is fixed
    geometry rather than a differentiable model input.
    """

    if dtype is None:
        dtype = torch.get_default_dtype()
    spacing = torch.as_tensor(
        grid_spacing,
        dtype=dtype,
        device=device,
    ).detach().clone().reshape(-1)
    if spacing.numel() == 1:
        spacing = spacing.repeat(3)
    if spacing.shape != (3,):
        raise ValueError("grid_spacing must contain one or three values")
    if (
        not torch.all(torch.isfinite(spacing)).item()
        or torch.any(spacing <= 0.0).item()
    ):
        raise ValueError("grid_spacing values must be finite and positive")
    return spacing


def voxel_volume(grid_spacing: torch.Tensor) -> torch.Tensor:
    r"""Return the regular-grid quadrature weight ``Delta V``.

    ``grid_spacing`` must have shape ``[..., 3]``. The result has shape
    ``[...]`` and is ``h_x h_y h_z``; for cubic voxels this is ``h**3``.
    """

    if not torch.is_tensor(grid_spacing):
        raise TypeError("grid_spacing must be a torch.Tensor")
    if grid_spacing.ndim < 1 or grid_spacing.shape[-1] != 3:
        raise ValueError("grid_spacing must end with three Cartesian values")
    if (
        not torch.all(torch.isfinite(grid_spacing)).item()
        or torch.any(grid_spacing <= 0.0).item()
    ):
        raise ValueError("grid_spacing values must be finite and positive")
    return torch.prod(grid_spacing, dim=-1)


def require_matching_grid_spacing(
    grid_spacing: torch.Tensor,
    reference: torch.Tensor,
) -> None:
    """Reject input spacing inconsistent with fixed model geometry."""

    if not torch.is_tensor(grid_spacing):
        raise TypeError("input grid_spacing must be a torch.Tensor")
    if grid_spacing.ndim < 1 or grid_spacing.shape[-1] != 3:
        raise ValueError(
            "input grid_spacing must end with three Cartesian values"
        )
    comparison = grid_spacing.to(dtype=reference.dtype)
    expected = reference.to(device=grid_spacing.device).expand_as(comparison)
    if not torch.allclose(comparison, expected):
        raise ValueError(
            "input grid_spacing does not match the trained model"
        )
