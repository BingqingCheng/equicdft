"""Convert EXTXYZ density fields into Grid-CACE input dictionaries.

The role of :class:`GridData` is analogous to ``AtomicData`` in CACE: one
object represents one complete configuration. Here the nodes are regular grid
points rather than atoms, and their scalar node feature is the density ``rho``.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from ase import Atoms
from ase.io import read

from .stencil import coarsen_grid, get_neighbor_indices


# Canonical GridData field -> source field in the EXTXYZ/ASE Atoms object.
# As in CACE AtomicData, callers can override any subset through `data_key`.
default_data_key = {
    "temperature": "T",
    "mu": "mu",
    "grid_spacing": "grid_spacing",
    "grid_size": "grid_size",
    "grid_indexing": "grid_indexing",
    "grid_positions": "positions",
    "V_ext": "V_ext",
    "rho": "density",
}


class GridData(dict):
    """Dictionary-like data for one complete periodic density configuration.

    Each object contains the following fields; ``mu`` is included only when
    supplied by the source frame::

        temperature                 scalar
        mu                          scalar (optional)
        n_types                     scalar
        grid_spacing                [3]
        index                       [n_grid]
        grid_positions              [n_grid, 3]
        V_ext                       [n_grid, n_types]
        rho                         [n_grid, n_types]
        local_density_index         [n_grid, n_neighbors]
        local_density_positions     [n_neighbors, 3]

    ``local_density_index[m, k]`` selects the density at relative integer
    displacement ``local_density_positions[k]`` from central point ``m``. The
    first displacement is always ``[0, 0, 0]``. Model forwards gather from the
    live ``rho`` tensor so functional derivatives retain the full graph.
    """

    def __init__(self, **data: torch.Tensor) -> None:
        super().__init__(data)

    @classmethod
    def from_xyz(
        cls,
        path: Union[str, Path],
        cutoff_grid: int = 3,
        index: str = ":",
        data_key: Optional[Dict[str, str]] = None,
        target_grid_spacing: Optional[
            Union[float, Sequence[float], np.ndarray]
        ] = None,
    ) -> List["GridData"]:
        """Read EXTXYZ frames and convert each frame to one ``GridData``.

        Parameters
        ----------
        path
            EXTXYZ file containing density and external-potential fields.
        cutoff_grid
            Inclusive spherical cutoff in integer grid steps. The default is
            three, so retained offsets satisfy ``x^2 + y^2 + z^2 <= 9``.
        index
            ASE frame selection. ``":"`` reads every frame.
        data_key
            Optional mapping from canonical GridData names to EXTXYZ field
            names. Supplied entries override :data:`default_data_key`.
        target_grid_spacing
            If provided, block-average ``rho`` and ``V_ext`` onto this
            commensurate coarser spacing before constructing local environments.

        Returns
        -------
        list of GridData
            One complete grid configuration per selected EXTXYZ frame.
        """

        keys = default_data_key.copy()
        if data_key is not None:
            unknown_keys = set(data_key) - set(default_data_key)
            if unknown_keys:
                raise KeyError(
                    "unknown GridData data_key entries: {}".format(
                        sorted(unknown_keys)
                    )
                )
            keys.update(data_key)

        # ASE returns either one Atoms object or a list, depending on `index`.
        # Normalize both cases to a list so this method has one return type.
        configurations = read(str(Path(path).expanduser()), index=index)
        if not isinstance(configurations, list):
            configurations = [configurations]
        return [
            cls(
                **_process_atoms(
                    atoms,
                    cutoff_grid,
                    keys,
                    target_grid_spacing=target_grid_spacing,
                )
            )
            for atoms in configurations
        ]


def _get_source_value(atoms: Atoms, source_key: str) -> Optional[Any]:
    """Read one mapped source field from an ASE Atoms object."""

    if source_key == "positions":
        return atoms.get_positions()
    if source_key in atoms.info:
        return atoms.info[source_key]
    if source_key in atoms.arrays:
        return atoms.arrays[source_key]
    return None


def _required_source_value(atoms: Atoms, source_key: str, field: str) -> Any:
    value = _get_source_value(atoms, source_key)
    if value is None:
        raise ValueError(
            "frame is missing '{}' for GridData field '{}'".format(source_key, field)
        )
    return value


def _process_atoms(
    atoms: Atoms,
    cutoff_grid: int,
    data_key: Dict[str, str],
    target_grid_spacing: Optional[
        Union[float, Sequence[float], np.ndarray]
    ] = None,
) -> Dict[str, torch.Tensor]:
    """Convert one ASE frame to the tensor dictionary stored by GridData."""

    # In these EXTXYZ files, ASE `positions` are zero-based integer grid
    # coordinates. `grid_center` contains physical coordinates and `grid_id`
    # is only a scalar running index, so neither is needed here.
    positions = np.asarray(
        _required_source_value(
            atoms, data_key["grid_positions"], "grid_positions"
        ),
        dtype=float,
    )
    grid_positions = np.rint(positions).astype(np.int64)
    if not np.allclose(positions, grid_positions, atol=1.0e-8, rtol=0.0):
        raise ValueError("EXTXYZ positions must contain integer grid coordinates")

    indexing_value = _get_source_value(atoms, data_key["grid_indexing"])
    indexing = str(
        "zero_based" if indexing_value is None else indexing_value
    ).lower()
    if indexing in ("one_based", "one-based", "1_based"):
        grid_positions -= 1
    if np.any(grid_positions.min(axis=0) != 0):
        raise ValueError("grid positions must use zero-based indexing")

    # Prefer the declared grid size, but allow it to be inferred from a
    # complete zero-based regular grid.
    grid_size_value = _get_source_value(atoms, data_key["grid_size"])
    if grid_size_value is not None:
        grid_size = np.asarray(grid_size_value, dtype=np.int64).reshape(3)
    else:
        grid_size = grid_positions.max(axis=0) + 1

    n_grid = int(np.prod(grid_size))
    if len(atoms) != n_grid:
        raise ValueError("frame does not contain one complete regular grid")
    if np.any(grid_positions.max(axis=0) + 1 != grid_size):
        raise ValueError("grid positions are inconsistent with grid_size")

    # Canonical C-order indexing is
    #     index(ix, iy, iz) = (ix * Ny + iy) * Nz + iz,
    # so z is the fastest-running coordinate. Sorting here makes the tensor
    # layout independent of the row order in the EXTXYZ file.
    flat_positions = np.ravel_multi_index(
        grid_positions.T, tuple(grid_size), order="C"
    )
    if np.unique(flat_positions).size != n_grid:
        raise ValueError("grid positions contain duplicate grid points")
    order = np.argsort(flat_positions)
    if not np.array_equal(flat_positions[order], np.arange(n_grid)):
        raise ValueError("grid positions do not cover the complete grid")

    # Apply exactly the same canonical permutation to the coordinates and both
    # scalar fields.
    grid_positions = grid_positions[order]
    rho = np.asarray(
        _required_source_value(atoms, data_key["rho"], "rho"), dtype=float
    )[order]
    V_ext = np.asarray(
        _required_source_value(atoms, data_key["V_ext"], "V_ext"), dtype=float
    )[order]
    if rho.ndim not in (1, 2) or V_ext.ndim not in (1, 2):
        raise ValueError(
            "rho and V_ext must have shape [n_grid] or [n_grid, n_types]"
        )
    if rho.ndim == 1:
        rho = rho[:, None]
    if V_ext.ndim == 1:
        V_ext = V_ext[:, None]
    if rho.shape[1] != V_ext.shape[1]:
        raise ValueError("rho and V_ext must contain the same number of columns")
    n_types = rho.shape[1]

    grid_spacing = np.asarray(
        _required_source_value(
            atoms, data_key["grid_spacing"], "grid_spacing"
        ),
        dtype=float,
    ).reshape(-1)
    if grid_spacing.size == 1:
        grid_spacing = np.repeat(grid_spacing, 3)
    if grid_spacing.size != 3:
        raise ValueError("grid_spacing must contain one or three values")

    # Optionally average both scalar fields over non-overlapping regular-grid
    # blocks. The local environments are built only after this transformation.
    if target_grid_spacing is not None:
        fields, grid_positions, grid_spacing = coarsen_grid(
            values=np.column_stack((rho, V_ext)),
            grid_positions=grid_positions,
            grid_spacing=grid_spacing,
            target_grid_spacing=target_grid_spacing,
        )
        rho = fields[:, :n_types]
        V_ext = fields[:, n_types:]
        n_grid = grid_positions.shape[0]

    # Store only the geometry needed to construct all overlapping periodic
    # environments. Model forwards use these indices to gather from live rho.
    local_density_index, local_density_positions = get_neighbor_indices(
        grid_positions=grid_positions,
        cutoff_grid=cutoff_grid,
    )

    # Match CACE's AtomicData convention by honoring PyTorch's current default
    # floating dtype and using int64 tensors for indices and grid coordinates.
    dtype = torch.get_default_dtype()
    data = {
        "temperature": torch.tensor(
            float(
                _required_source_value(
                    atoms, data_key["temperature"], "temperature"
                )
            ),
            dtype=dtype,
        ),
        "n_types": torch.tensor(n_types, dtype=torch.long),
        "grid_spacing": torch.tensor(grid_spacing, dtype=dtype),
        "index": torch.arange(n_grid, dtype=torch.long),
        "grid_positions": torch.tensor(grid_positions, dtype=torch.long),
        "V_ext": torch.tensor(V_ext, dtype=dtype),
        "rho": torch.tensor(rho, dtype=dtype),
        "local_density_index": torch.tensor(
            local_density_index, dtype=torch.long
        ),
        "local_density_positions": torch.tensor(
            local_density_positions, dtype=torch.long
        ),
    }

    # Chemical potential is useful metadata for grand-canonical frames but is
    # not part of the density field itself and may be absent, for example for
    # canonical simulations. Do not assign an artificial value when missing.
    mu = _get_source_value(atoms, data_key["mu"])
    if mu is not None:
        data["mu"] = torch.tensor(float(mu), dtype=dtype)
    return data
