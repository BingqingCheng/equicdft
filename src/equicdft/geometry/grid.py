"""Tensor validators and index helpers for regular three-dimensional grids."""

import math
from typing import Optional, Tuple

import torch


def _grid_center_origin(grid_center, grid_positions, grid_spacing):
    """Validate physical centers = origin + integer indices * spacing.

    Geometry is fixed and evaluated in float64. Permit file roundoff (1e-8
    length units) and roundoff from the supplied coordinate/spacing dtypes,
    not a distorted or rotated grid. Leading batch dimensions may broadcast.
    """
    spacing = torch.as_tensor(grid_spacing)
    centers = torch.as_tensor(grid_center, device=spacing.device)
    if centers.requires_grad:
        raise ValueError("grid_center must be fixed geometry")
    if centers.dtype == torch.bool or centers.is_complex():
        raise ValueError("grid_center must contain real finite coordinates")
    indices = torch.as_tensor(grid_positions, device=centers.device)
    if centers.ndim < 2 or centers.shape[-2:] != indices.shape[-2:]:
        raise ValueError("grid_center must have shape [..., n_grid, 3]")
    if centers.shape[-1] != 3 or centers.shape[-2] == 0:
        raise ValueError("grid_center must have shape [..., n_grid, 3]")
    if not torch.all(torch.isfinite(centers)).item():
        raise ValueError("grid_center must contain real finite coordinates")
    # Python float lists carry double precision even under a float32 default.
    center_eps = (torch.finfo(centers.dtype).eps if centers.is_floating_point()
                  and hasattr(grid_center, "dtype") else torch.finfo(torch.float64).eps)
    spacing_eps = torch.finfo(spacing.dtype).eps if spacing.is_floating_point() else 0.0
    steps = indices.double() * spacing.double()[..., None, :]
    centers = torch.as_tensor(grid_center, dtype=torch.float64, device=centers.device)
    offsets = centers - steps
    origin = offsets[..., 0, :]
    tolerance = 1e-8 + 8 * (center_eps * centers.abs().max()
                            + spacing_eps * steps.abs().max())
    if torch.any((offsets - origin[..., None, :]).abs() > tolerance).item():
        raise ValueError("grid_center must match grid_positions and grid_spacing on a regular grid")
    return origin


def common_grid_size(
    grid_size: torch.Tensor,
    leading_shape: torch.Size,
) -> Tuple[int, int, int]:
    """Validate and return one regular-grid shape shared by a batch."""

    sizes = torch.as_tensor(grid_size).detach().reshape(-1, 3)
    if sizes.shape[0] not in (1, math.prod(leading_shape)):
        raise ValueError("grid_size leading shape must match rho")
    rounded = torch.round(sizes).to(dtype=torch.long)
    if not torch.allclose(
        sizes.to(dtype=torch.float64),
        rounded.to(dtype=torch.float64),
    ):
        raise ValueError("grid_size values must be integers")
    if torch.any(rounded <= 0).item():
        raise ValueError("grid_size values must be positive")
    if not torch.all(rounded == rounded[0]).item():
        raise ValueError("all fields in one batch must share grid_size")
    return tuple(int(value) for value in rounded[0].cpu().tolist())


def _integer_tensor(
    values: torch.Tensor,
    name: str,
    device: torch.device,
) -> torch.Tensor:
    """Return integer-valued geometry as a long tensor."""

    tensor = torch.as_tensor(values, device=device)
    if tensor.dtype == torch.bool or tensor.is_complex():
        raise TypeError(f"{name} must be integer-valued")
    if not tensor.is_floating_point():
        return tensor.to(dtype=torch.long)
    rounded = torch.round(tensor).to(dtype=torch.long)
    if not torch.allclose(
        tensor.to(dtype=torch.float64),
        rounded.to(dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-8,
    ):
        raise ValueError(f"{name} must be integer-valued")
    return rounded


def _ravel_grid_indices(
    positions: torch.Tensor,
    grid_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """Return C-order flat indices for three-dimensional coordinates."""

    return (
        (positions[..., 0] * grid_shape[1] + positions[..., 1])
        * grid_shape[2]
        + positions[..., 2]
    )


def _flat_grid_positions(
    grid_positions: torch.Tensor,
    grid_shape: Tuple[int, int, int],
    leading_shape: torch.Size,
    device: torch.device,
) -> Tuple[torch.Tensor, bool]:
    """Return validated C-order indices and whether rows are canonical."""

    n_grid = math.prod(grid_shape)
    positions = _integer_tensor(grid_positions, "grid_positions", device)
    if positions.shape == (n_grid, 3):
        positions = positions.reshape(1, n_grid, 3).expand(
            math.prod(leading_shape) if leading_shape else 1,
            -1,
            -1,
        )
    elif positions.shape == (*leading_shape, n_grid, 3):
        positions = positions.reshape(-1, n_grid, 3)
    else:
        raise ValueError(
            "grid_positions must have shape [n_grid, 3] or "
            "[..., n_grid, 3] matching values"
        )
    sizes = positions.new_tensor(grid_shape)
    if torch.any(positions < 0).item() or torch.any(positions >= sizes).item():
        raise ValueError("grid_positions are outside grid_size")
    flat = _ravel_grid_indices(positions, grid_shape)
    canonical = torch.arange(n_grid, device=device).expand_as(flat)
    rows_are_canonical = torch.equal(flat, canonical)
    if (
        not rows_are_canonical
        and not torch.equal(torch.sort(flat, dim=-1).values, canonical)
    ):
        raise ValueError(
            "grid_positions must contain one complete regular grid"
        )
    return flat, rows_are_canonical


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
