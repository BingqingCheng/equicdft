"""Periodic-neighborhood helper for regular density grids.

This module plays the same role as ``cace.data.neighborhood``: it contains the
geometry operation used by the data class, while storing no dataset state.
"""

from itertools import product
from typing import Sequence, Tuple, Union

import numpy as np


def get_local_density(
    rho: np.ndarray,
    grid_positions: np.ndarray,
    grid_spacing: Union[float, Sequence[float], np.ndarray],
    cutoff: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gather one periodic local density environment around every grid point.

    For central grid point ``m`` and local displacement ``k``, this function
    constructs

    ``local_density[m, k] = rho[(g_m + s_k) mod grid_size]``,

    where ``g_m = grid_positions[m]`` and
    ``s_k = local_density_positions[k]``.

    Parameters
    ----------
    rho
        Scalar density field with shape ``[n_grid]``.
    grid_positions
        Zero-based integer coordinates with shape ``[n_grid, 3]``.
    grid_spacing
        Physical spacing along the three grid axes.
    cutoff
        Physical radius of the local environment.

    Returns
    -------
    local_density
        Density environments with shape ``[n_grid, n_neighbors]``.
    local_density_positions
        Shared relative integer displacements with shape ``[n_neighbors, 3]``.

    The center is always the first entry in ``local_density_positions``.
    Periodic wrapping is applied independently along all three grid axes.
    """

    rho = np.asarray(rho)
    grid_positions = np.asarray(grid_positions)
    spacing = np.asarray(grid_spacing, dtype=float).reshape(-1)

    if rho.ndim != 1:
        raise ValueError("rho must be a one-dimensional scalar field")
    if grid_positions.shape != (rho.shape[0], 3):
        raise ValueError("grid_positions must have shape [n_grid, 3]")
    if spacing.size == 1:
        spacing = np.repeat(spacing, 3)
    if spacing.size != 3 or np.any(spacing <= 0.0):
        raise ValueError("grid_spacing must contain three positive values")
    if cutoff < 0.0:
        raise ValueError("cutoff must be nonnegative")

    # Grid coordinates must be exact integer lattice points. Keeping them as
    # integer offsets avoids floating-point ambiguity in periodic wrapping.
    rounded_positions = np.rint(grid_positions).astype(np.int64)
    if not np.allclose(grid_positions, rounded_positions, atol=1.0e-8, rtol=0.0):
        raise ValueError("grid_positions must be integer-valued")
    if np.any(rounded_positions.min(axis=0) != 0):
        raise ValueError("grid_positions must use zero-based indexing")

    grid_size = rounded_positions.max(axis=0) + 1
    n_grid = int(np.prod(grid_size))
    if rho.shape[0] != n_grid:
        raise ValueError("grid_positions do not contain one complete regular grid")

    # Map the input row order onto canonical C-order flat indices. The helper
    # remains correct even if its caller supplies shuffled grid rows.
    flat_positions = np.ravel_multi_index(
        rounded_positions.T, tuple(grid_size), order="C"
    )
    if np.unique(flat_positions).size != n_grid:
        raise ValueError("grid_positions contain duplicate grid points")

    # Below half the shortest periodic box length, each relative offset selects
    # one unique periodic image.
    if cutoff >= 0.5 * np.min(grid_size * spacing):
        raise ValueError("cutoff must be smaller than half the shortest box length")

    # Enumerate a rectangular integer bounding box, then keep only offsets
    # whose physical vectors lie inside the spherical cutoff.
    maximum_offsets = np.ceil(cutoff / spacing).astype(int)
    candidates = []
    for offset in product(
        range(-maximum_offsets[0], maximum_offsets[0] + 1),
        range(-maximum_offsets[1], maximum_offsets[1] + 1),
        range(-maximum_offsets[2], maximum_offsets[2] + 1),
    ):
        physical_vector = np.asarray(offset, dtype=float) * spacing
        squared_distance = float(np.dot(physical_vector, physical_vector))
        if squared_distance <= (cutoff + 1.0e-12) ** 2:
            candidates.append((offset, squared_distance))

    def ordering(item):
        """Put the center first and otherwise use a deterministic shell order."""

        offset, squared_distance = item
        x, y, z = offset
        return (
            round(squared_distance, 14),
            -abs(z),
            -z,
            -abs(y),
            -y,
            -abs(x),
            -x,
        )

    # Since the center has zero squared distance, it is guaranteed to be k=0.
    candidates.sort(key=ordering)
    local_density_positions = np.asarray(
        [item[0] for item in candidates], dtype=np.int64
    )

    # Invert the canonical-index mapping: given a wrapped grid coordinate,
    # recover the row in the caller's rho array that stores its density.
    row_from_flat_position = np.empty(n_grid, dtype=np.int64)
    row_from_flat_position[flat_positions] = np.arange(n_grid)

    # Add every shared relative offset to every center. `remainder` implements
    # periodic boundary conditions, e.g. [0,0,0] + [0,0,-1] -> [0,0,Nz-1].
    neighbor_positions = np.remainder(
        rounded_positions[:, None, :] + local_density_positions[None, :, :],
        grid_size[None, None, :],
    )
    # Convert wrapped neighbor coordinates to input row indices, then gather
    # the complete [n_grid, n_neighbors] local-density matrix at once.
    neighbor_flat_positions = np.ravel_multi_index(
        neighbor_positions.reshape(-1, 3).T, tuple(grid_size), order="C"
    ).reshape(n_grid, -1)
    neighbor_rows = row_from_flat_position[neighbor_flat_positions]
    local_density = rho[neighbor_rows]

    return local_density, local_density_positions
