"""Small tensor helpers for regular three-dimensional grids."""

from typing import Optional

import torch


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
