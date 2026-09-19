"""Geometry helpers for periodic neighborhoods on regular density grids.

This module is the single NumPy-side authority on what a valid regular grid
is: :func:`grid_size_array` and :func:`grid_spacing_array` coerce serialized
metadata, and :func:`canonical_grid_order` validates a complete zero-based
integer grid and returns its canonical C ordering. The torch-side helpers in
``equicdft._grid`` validate live tensors during model evaluation.
"""

from itertools import product
from typing import NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

from ._argument_checks import nonnegative_integer


def grid_size_array(value: object, name: str = "grid_size") -> np.ndarray:
    """Coerce serialized metadata to three positive integer grid dimensions."""

    try:
        raw = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        raise ValueError("{} must contain three values".format(name)) from None
    if raw.size != 3:
        raise ValueError("{} must contain three values".format(name))
    grid_size = np.rint(raw).astype(np.int64)
    if not np.allclose(raw, grid_size) or np.any(grid_size <= 0):
        raise ValueError("{} values must be positive integers".format(name))
    return grid_size


def grid_spacing_array(
    value: object,
    name: str = "grid_spacing",
) -> np.ndarray:
    """Coerce serialized metadata to one finite positive spacing per axis."""

    try:
        spacing = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        raise ValueError(
            "{} must contain one or three finite positive values".format(name)
        ) from None
    if spacing.size == 1:
        spacing = np.repeat(spacing, 3)
    if (
        spacing.size != 3
        or not np.all(np.isfinite(spacing))
        or np.any(spacing <= 0.0)
    ):
        raise ValueError(
            "{} must contain one or three finite positive values".format(name)
        )
    return spacing


class GridOrder(NamedTuple):
    """Canonical C ordering of one complete regular grid.

    ``positions`` are the validated integer coordinates in the caller's row
    order, ``grid_size`` the three grid dimensions, ``flat_index`` the C-order
    flat index of every row, and ``order`` the permutation that sorts the rows
    into canonical C order (``positions[order]`` enumerates the grid with the
    last axis varying fastest).
    """

    positions: np.ndarray
    grid_size: np.ndarray
    flat_index: np.ndarray
    order: np.ndarray


def canonical_grid_order(
    grid_positions: object,
    grid_size: Optional[object] = None,
) -> GridOrder:
    """Validate one complete zero-based regular grid and return its ordering.

    ``grid_positions`` must be integer-valued (within ``1e-8``), zero-based
    along every axis, and must enumerate every point of the grid exactly
    once. ``grid_size`` is inferred from the largest coordinates when it is
    omitted; when supplied it must agree with them.
    """

    positions = np.asarray(grid_positions)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("grid_positions must have shape [n_grid, 3]")
    rounded = np.rint(positions).astype(np.int64)
    if not np.allclose(positions, rounded, atol=1.0e-8, rtol=0.0):
        raise ValueError("grid_positions must be integer-valued")
    if positions.shape[0] == 0 or np.any(rounded.min(axis=0) != 0):
        raise ValueError("grid_positions must use zero-based indexing")

    inferred_size = rounded.max(axis=0) + 1
    if grid_size is None:
        size = inferred_size
    else:
        size = grid_size_array(grid_size)
        if np.any(inferred_size != size):
            raise ValueError("grid_positions are inconsistent with grid_size")
    n_grid = int(np.prod(size))
    if rounded.shape[0] != n_grid:
        raise ValueError(
            "grid_positions do not contain one complete regular grid"
        )

    flat_index = np.ravel_multi_index(rounded.T, tuple(size), order="C")
    if np.unique(flat_index).size != n_grid:
        raise ValueError("grid_positions contain duplicate grid points")
    order = np.argsort(flat_index)
    if not np.array_equal(flat_index[order], np.arange(n_grid)):
        raise ValueError("grid_positions do not cover the complete grid")
    return GridOrder(
        positions=rounded,
        grid_size=size,
        flat_index=flat_index,
        order=order,
    )


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

    cutoff_grid = nonnegative_integer(cutoff_grid, "cutoff_grid")

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

    scalar_field = values.ndim == 1
    if scalar_field:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError("values must have shape [n_grid] or [n_grid, n_fields]")
    if grid_positions.shape != (values.shape[0], 3):
        raise ValueError("grid_positions must have shape [n_grid, 3]")

    source_spacing = grid_spacing_array(grid_spacing)
    target_spacing = grid_spacing_array(
        target_grid_spacing,
        "target_grid_spacing",
    )
    ordering = canonical_grid_order(grid_positions)
    source_shape = ordering.grid_size
    order = ordering.order

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

    local_density_positions = make_stencil(cutoff_grid)

    # Grid coordinates must be exact integer lattice points. Keeping them as
    # integer offsets avoids floating-point ambiguity in periodic wrapping.
    # The helper remains correct even if its caller supplies shuffled rows.
    ordering = canonical_grid_order(grid_positions)
    rounded_positions = ordering.positions
    grid_size = ordering.grid_size
    n_grid = int(np.prod(grid_size))

    # Invert the canonical-index mapping: given a wrapped grid coordinate,
    # recover the row in the caller's density array that stores its value.
    row_from_flat_position = np.empty(n_grid, dtype=np.int64)
    row_from_flat_position[ordering.flat_index] = np.arange(n_grid)

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
