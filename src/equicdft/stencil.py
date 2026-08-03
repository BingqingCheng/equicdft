"""Geometry helpers for periodic neighborhoods on regular density grids."""

from itertools import product
from numbers import Integral
from typing import Sequence, Tuple, Union

import numpy as np


def make_stencil(cutoff_grid: int = 3) -> np.ndarray:
    """Return canonically ordered integer offsets inside a spherical cutoff.

    The cutoff is measured in grid steps and is inclusive: an offset ``s`` is
    retained when ``s[0]**2 + s[1]**2 + s[2]**2 <= cutoff_grid**2``. Offsets
    related by axis permutations and independent sign flips are contiguous.

    Parameters
    ----------
    cutoff_grid
        Nonnegative integer cutoff in grid steps. The default is three.

    Returns
    -------
    numpy.ndarray
        Relative integer displacements with shape ``[n_neighbors, 3]``.
    """

    if isinstance(cutoff_grid, bool) or not isinstance(cutoff_grid, Integral):
        raise TypeError("cutoff_grid must be a nonnegative integer")
    cutoff_grid = int(cutoff_grid)
    if cutoff_grid < 0:
        raise ValueError("cutoff_grid must be a nonnegative integer")

    squared_cutoff = cutoff_grid**2
    offsets = [
        offset
        for offset in product(
            range(-cutoff_grid, cutoff_grid + 1),
            range(-cutoff_grid, cutoff_grid + 1),
            range(-cutoff_grid, cutoff_grid + 1),
        )
        if sum(component**2 for component in offset) <= squared_cutoff
    ]

    def ordering(offset):
        """Group offsets by cubic-symmetry orbit, then orient canonically."""

        absolute_offset = tuple(abs(component) for component in offset)

        # Signed permutations of an offset belong to the same cubic orbit and
        # therefore share the sorted absolute-coordinate triple. Squared
        # distance orders the shells; the orbit label additionally separates
        # inequivalent offsets such as (3, 0, 0) and (2, 2, 1), which both have
        # x^2 + y^2 + z^2 = 9.
        squared_distance = sum(component**2 for component in offset)
        orbit = tuple(sorted(absolute_offset))

        # Lexicographic absolute coordinates produce z-, y-, then x-oriented
        # axial vectors. Positive signs precede their mirrored counterparts.
        signs = tuple(0 if component >= 0 else 1 for component in offset)
        return squared_distance, orbit, absolute_offset, signs

    offsets.sort(key=ordering)
    return np.asarray(offsets, dtype=np.int64)


def coarsen_grid(
    values: np.ndarray,
    grid_positions: np.ndarray,
    grid_spacing: Union[float, Sequence[float], np.ndarray],
    target_grid_spacing: Union[float, Sequence[float], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Block-average scalar fields onto a commensurate coarser grid.

    Parameters
    ----------
    values
        One or more fields with shape ``[n_grid]`` or ``[n_grid, n_fields]``.
    grid_positions
        Zero-based integer source-grid coordinates with shape ``[n_grid, 3]``.
    grid_spacing
        Physical source-grid spacing along the three axes.
    target_grid_spacing
        Requested physical spacing. Each target/source spacing ratio must be a
        positive integer and must divide the corresponding grid size.

    Returns
    -------
    coarse_values
        Block-averaged fields in canonical C order.
    coarse_positions
        Zero-based integer coordinates of the coarse grid.
    target_spacing
        Normalized three-component target spacing.
    """

    values = np.asarray(values)
    grid_positions = np.asarray(grid_positions)
    source_spacing = np.asarray(grid_spacing, dtype=float).reshape(-1)
    target_spacing = np.asarray(target_grid_spacing, dtype=float).reshape(-1)

    scalar_field = values.ndim == 1
    if scalar_field:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError("values must have shape [n_grid] or [n_grid, n_fields]")
    if grid_positions.shape != (values.shape[0], 3):
        raise ValueError("grid_positions must have shape [n_grid, 3]")

    if source_spacing.size == 1:
        source_spacing = np.repeat(source_spacing, 3)
    if target_spacing.size == 1:
        target_spacing = np.repeat(target_spacing, 3)
    if source_spacing.size != 3 or np.any(source_spacing <= 0.0):
        raise ValueError("grid_spacing must contain three positive values")
    if target_spacing.size != 3 or np.any(target_spacing <= 0.0):
        raise ValueError("target_grid_spacing must contain three positive values")

    rounded_positions = np.rint(grid_positions).astype(np.int64)
    if not np.allclose(grid_positions, rounded_positions, atol=1.0e-8, rtol=0.0):
        raise ValueError("grid_positions must be integer-valued")
    if np.any(rounded_positions.min(axis=0) != 0):
        raise ValueError("grid_positions must use zero-based indexing")

    source_shape = rounded_positions.max(axis=0) + 1
    n_grid = int(np.prod(source_shape))
    if values.shape[0] != n_grid:
        raise ValueError("grid_positions do not contain one complete regular grid")

    flat_positions = np.ravel_multi_index(
        rounded_positions.T, tuple(source_shape), order="C"
    )
    if np.unique(flat_positions).size != n_grid:
        raise ValueError("grid_positions contain duplicate grid points")
    order = np.argsort(flat_positions)
    if not np.array_equal(flat_positions[order], np.arange(n_grid)):
        raise ValueError("grid_positions do not cover the complete grid")

    factor_float = target_spacing / source_spacing
    factors = np.rint(factor_float).astype(np.int64)
    if np.any(factors < 1) or not np.allclose(
        factor_float, factors, atol=1.0e-10, rtol=1.0e-10
    ):
        raise ValueError(
            "target_grid_spacing must be an integer multiple of grid_spacing"
        )
    if np.any(source_shape % factors != 0):
        raise ValueError("coarsening factors must divide the source grid size")

    coarse_shape = source_shape // factors
    n_fields = values.shape[1]
    values_grid = values[order].reshape(
        source_shape[0], source_shape[1], source_shape[2], n_fields
    )
    blocked = values_grid.reshape(
        coarse_shape[0],
        factors[0],
        coarse_shape[1],
        factors[1],
        coarse_shape[2],
        factors[2],
        n_fields,
    )
    coarse_values = blocked.mean(axis=(1, 3, 5)).reshape(-1, n_fields)
    coarse_positions = np.indices(
        tuple(coarse_shape), dtype=np.int64
    ).reshape(3, -1).T

    if scalar_field:
        coarse_values = coarse_values[:, 0]
    return coarse_values, coarse_positions, target_spacing


def get_neighbor_indices(
    grid_positions: np.ndarray,
    cutoff_grid: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return periodic neighbor rows for every regular-grid point.

    This helper stores only geometry. The returned integer matrix is placed in
    :class:`equicdft.data.GridData` and used to gather from the live PyTorch
    ``rho`` tensor during a model forward pass. That keeps all overlapping
    local environments connected to ``rho`` in the differentiation graph.

    The stencil lives on the infinitely repeated periodic lattice. Therefore,
    when the cutoff exceeds half a box length, distinct relative offsets may
    wrap onto the same stored voxel. They remain separate stencil entries
    because they represent different periodic images at different relative
    positions and consequently carry different Cartesian/radial weights.

    Returns
    -------
    neighbor_indices
        Row indices with shape ``[n_grid, n_neighbors]``.
    local_density_positions
        Shared relative integer displacements with shape ``[n_neighbors, 3]``.
    """

    grid_positions = np.asarray(grid_positions)
    if grid_positions.ndim != 2 or grid_positions.shape[1] != 3:
        raise ValueError("grid_positions must have shape [n_grid, 3]")

    local_density_positions = make_stencil(cutoff_grid)

    # Grid coordinates must be exact integer lattice points. Keeping them as
    # integer offsets avoids floating-point ambiguity in periodic wrapping.
    rounded_positions = np.rint(grid_positions).astype(np.int64)
    if not np.allclose(grid_positions, rounded_positions, atol=1.0e-8, rtol=0.0):
        raise ValueError("grid_positions must be integer-valued")
    if np.any(rounded_positions.min(axis=0) != 0):
        raise ValueError("grid_positions must use zero-based indexing")

    grid_size = rounded_positions.max(axis=0) + 1
    n_grid = int(np.prod(grid_size))
    if rounded_positions.shape[0] != n_grid:
        raise ValueError("grid_positions do not contain one complete regular grid")

    # Map the input row order onto canonical C-order flat indices. The helper
    # remains correct even if its caller supplies shuffled grid rows.
    flat_positions = np.ravel_multi_index(
        rounded_positions.T, tuple(grid_size), order="C"
    )
    if np.unique(flat_positions).size != n_grid:
        raise ValueError("grid_positions contain duplicate grid points")

    # Invert the canonical-index mapping: given a wrapped grid coordinate,
    # recover the row in the caller's density array that stores its value.
    row_from_flat_position = np.empty(n_grid, dtype=np.int64)
    row_from_flat_position[flat_positions] = np.arange(n_grid)

    # Add every shared relative offset to every center. `remainder` implements
    # periodic boundary conditions, e.g. [0,0,0] + [0,0,-1] -> [0,0,Nz-1].
    neighbor_positions = np.remainder(
        rounded_positions[:, None, :] + local_density_positions[None, :, :],
        grid_size[None, None, :],
    )
    neighbor_flat_positions = np.ravel_multi_index(
        neighbor_positions.reshape(-1, 3).T, tuple(grid_size), order="C"
    ).reshape(n_grid, -1)
    neighbor_indices = row_from_flat_position[neighbor_flat_positions]

    return neighbor_indices, local_density_positions
