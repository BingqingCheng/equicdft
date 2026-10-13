"""Spectral derivatives of periodic grid fields."""

import math
from typing import Sequence, Union

import torch

from .grid import common_grid_size, grid_spacing_tensor

__all__ = ["spectral_laplacian"]


def spectral_laplacian(
    field: torch.Tensor,
    grid_size: Sequence[int],
    grid_spacing: Union[float, Sequence[float], torch.Tensor],
) -> torch.Tensor:
    r"""Return the periodic Laplacian of ``field`` by Fourier differentiation.

    ``field`` has shape ``[..., n_grid, n]`` on the flattened canonical grid
    (last grid axis fastest, as the products and the reciprocal features use
    it); the result has the same shape, dtype and device. Each Fourier
    component is multiplied by ``-|k|^2``, so the result is exact for fields
    that are band-limited on the grid (plane waves, Gaussians much wider than
    the spacing) and is the Laplacian of the sampled interpolant otherwise.
    For a potential with content above the grid Nyquist, the analytic
    Laplacian evaluated at the voxel centres is the better input.
    """

    if not torch.is_tensor(field) or not field.is_floating_point():
        raise TypeError("field must be a floating tensor")
    if field.ndim < 2:
        raise ValueError("field must have shape [..., n_grid, n]")
    nx, ny, nz = common_grid_size(grid_size, field.shape[:-2])
    if nx * ny * nz != field.shape[-2]:
        raise ValueError("grid_size product does not match the field's grid axis")
    spacing = grid_spacing_tensor(grid_spacing, dtype=field.dtype, device=field.device)
    axes = [
        2.0 * math.pi * torch.fft.fftfreq(
            n, d=float(spacing[axis].item()), dtype=field.dtype, device=field.device,
        )
        for axis, n in enumerate((nx, ny, nz))
    ]
    kx, ky, kz = torch.meshgrid(*axes, indexing="ij")
    squared = kx.square() + ky.square() + kz.square()
    grid = field.reshape(*field.shape[:-2], nx, ny, nz, field.shape[-1])
    spectrum = torch.fft.fftn(grid, dim=(-4, -3, -2))
    laplacian = torch.fft.ifftn(-squared[..., None] * spectrum, dim=(-4, -3, -2)).real
    return laplacian.reshape(field.shape)
