"""Spectral derivatives and filters of periodic grid fields."""

import math
from typing import Optional, Sequence, Union

import torch

from .._validation import positive_scalar
from .grid import _flat_grid_positions, common_grid_size, grid_spacing_tensor

__all__ = ["nonnegative_average", "spectral_laplacian"]


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


def nonnegative_average(
    field: torch.Tensor,
    grid_size: Sequence[int],
    grid_spacing: Union[float, Sequence[float], torch.Tensor],
    sigma: float,
    *,
    grid_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Average a periodic grid field with a nonnegative kernel of width ``sigma``.

    The kernel is a product of nearest-neighbour steps. A step of second
    moment ``s**2`` replaces every value by ``1 - 2 sum_i a_i`` times itself
    plus ``a_i`` times each of its two neighbours along axis ``i``, with
    ``a_i = s**2 / (2 h_i**2)``, and has the transfer function

    ``t(k) = 1 - 2 s**2 sum_i sin(k_i h_i / 2)**2 / h_i**2``.

    No step is wider than ``s_max = (2 sum_i h_i**-2)**-0.5``, ``h / sqrt(6)``
    on a cubic grid, the width at which ``t`` vanishes at the cube corner and
    the centre weight is one half. ``sigma <= s_max`` takes one step; a wider
    ``sigma`` takes whole steps of ``s_max`` and one remainder, so that the
    second moments add up to ``sigma**2``. The kernel is therefore
    nonnegative, sums to one, is symmetric with the symmetry of the grid,
    reaches one voxel per step (its weights beyond are exact zeros) and has
    the second moment ``sigma**2`` along every axis. Its transfer ``T``, the
    product of the step transfers, lies in ``[0, 1]``, agrees with
    ``exp(-sigma**2 |k|**2 / 2)`` to second order in ``k``, decreases
    continuously with ``sigma`` at every wavevector and vanishes at the cube
    corner once ``sigma >= s_max``. At ``sigma = h / sqrt(6)`` it is 2/3 at an
    axis Nyquist mode and 1/3 at a [110] zone-edge mode.

    Nonnegative weights make the average preserve every convex set of field
    values: a nonnegative density stays nonnegative, and fields with
    ``|P| <= m rho`` at every voxel keep ``|G P| <= G |P| <= m G rho``, so an
    alignment formed from the averaged fields never exceeds the largest voxel
    alignment in the kernel's reach. A positive Fourier multiplier does not
    guarantee this: the zone-restricted Gaussian ``exp(-sigma**2 |k|**2 / 2)``
    has negative real-space weights, and on a 2 A grid with ``sigma = 0.6`` A
    it turns a 0.001 A^-3 voxel two steps from a 0.09 A^-3 slab into
    1.05e-4 A^-3 while nearly doubling its polarization. Positivity bounds
    the attainable suppression: every nonnegative kernel with second moment
    ``sigma**2`` has ``T >= 1 - 2 sigma**2 / h**2`` at an axis Nyquist mode
    and ``T >= 1 - 6 sigma**2 / h**2`` at the cube corner, and the one-step
    kernel attains both bounds.

    ``field`` has shape ``[..., n_grid, n]``; flatten vector or species
    components into the last axis. Rows follow ``grid_positions`` when given
    (any complete order, ``[n_grid, 3]`` or ``[..., n_grid, 3]``) and the
    canonical order of ``grid_size`` (last axis fastest) otherwise. The result
    has the field's shape, dtype and device and is differentiable; the
    average is its own adjoint.
    """

    if not torch.is_tensor(field) or not field.is_floating_point():
        raise TypeError("field must be a real floating tensor")
    if field.ndim < 2:
        raise ValueError("field must have shape [..., n_grid, n]")
    sigma = positive_scalar(sigma, "sigma")
    leading = field.shape[:-2]
    shape = common_grid_size(grid_size, leading)
    n_grid, n_values = field.shape[-2:]
    if math.prod(shape) != n_grid:
        raise ValueError("grid_size product does not match the field's grid axis")
    spacing = grid_spacing_tensor(grid_spacing, dtype=torch.float64, device="cpu")
    inverse_squares = [1.0 / float(value) ** 2 for value in spacing]
    widest = 0.5 / sum(inverse_squares)
    # The tolerance keeps sigma = s_max at one step despite rounding in
    # sigma**2; the remainder is capped so that no step overshoots s_max.
    steps = max(1, math.ceil(sigma ** 2 / widest - 1.0e-12))
    moments = [widest] * (steps - 1)
    moments.append(min(widest, sigma ** 2 - (steps - 1) * widest))

    n_fields = math.prod(leading) if leading else 1
    values = field.reshape(n_fields, n_grid, n_values)
    rows = None
    if grid_positions is not None:
        flat, canonical = _flat_grid_positions(
            grid_positions, shape, leading, field.device,
        )
        if not canonical:
            rows = flat.reshape(n_fields, n_grid)[..., None].expand_as(values)
            values = torch.zeros_like(values).scatter(1, rows, values)
    grid = values.reshape(n_fields, *shape, n_values)
    for moment in moments:
        weights = [0.5 * moment * value for value in inverse_squares]
        averaged = (1.0 - 2.0 * sum(weights)) * grid
        for axis, weight in enumerate(weights, start=1):
            averaged = averaged + weight * (
                torch.roll(grid, 1, dims=axis) + torch.roll(grid, -1, dims=axis)
            )
        grid = averaged
    averaged = grid.reshape(n_fields, n_grid, n_values)
    if rows is not None:
        averaged = torch.gather(averaged, 1, rows)
    return averaged.reshape(field.shape)
