"""Convert EXTXYZ density fields into model input dictionaries.

One :class:`GridData` object represents one complete periodic density field.
It stores physical fields and the regular-grid geometry needed by the model.
"""

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch
from ase import Atoms
from ase.io import read

from .stencil import coarsen_grid


# Default to temperatures in kelvin and energies in electronvolts. Reduced-unit
# datasets can recover beta = 1 / T by passing boltzmann_constant=1.0.
DEFAULT_BOLTZMANN_CONSTANT = 8.617333262e-5


# Canonical GridData field -> source field in the EXTXYZ/ASE Atoms object.
# Callers can override any subset of source names through `data_key`.
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


GRID_INFO_KEYS = {
    "cutoff_grid",
    "grid_spacing",
    "n_types",
    "boltzmann_constant",
    "thermal_wavelength",
}


def _normalize_xyz_paths(
    path: Union[
        str,
        Path,
        Sequence[Union[str, Path]],
    ],
) -> List[Path]:
    """Return a validated nonempty list of expanded EXTXYZ paths."""

    if isinstance(path, (str, Path)):
        return [Path(path).expanduser()]
    if not isinstance(path, Sequence):
        raise TypeError(
            "path must be a path-like value or a sequence of path-like values"
        )

    values = list(path)
    if not values:
        raise ValueError("path sequence must not be empty")
    if any(not isinstance(value, (str, Path)) for value in values):
        raise TypeError(
            "every item in path must be a string or pathlib.Path"
        )
    return [Path(value).expanduser() for value in values]


class GridData(dict):
    """Dictionary-like data for one complete periodic density configuration.

    Every object contains the grid geometry, temperature, and beta. Density,
    external-potential, chemical-potential, and reference fields are included
    only when their source quantities are available::

        temperature                 scalar
        beta                        scalar
        mu                          [n_types] (optional)
        beta_mu                     [n_types] (if mu exists)
        n_types                     scalar
        thermal_wavelength          [n_types] (if V_ext exists or built directly)
        grid_spacing                [3]
        index                       [n_grid]
        grid_positions              [n_grid, 3]
        V_ext                       [n_grid, n_types] (optional)
        rho                         [n_grid, n_types] (optional)
        c1_plus_beta_mu             [n_grid, n_types] (if rho is positive and
                                    V_ext exists)
        c1                          [n_grid, n_types] (if rho is positive and
                                    V_ext and mu exist)

    EXTXYZ records require at least one of ``rho`` and ``V_ext``. A grid built
    directly with :meth:`from_dict` may initially contain neither so that an
    external field can be assigned afterward.

    Density neighborhoods are evaluated inside the model by periodic 3D
    convolution. They are therefore not materialized or stored per frame.
    """

    def __init__(self, **data: torch.Tensor) -> None:
        super().__init__(data)

    @classmethod
    def from_xyz(
        cls,
        path: Union[
            str,
            Path,
            Sequence[Union[str, Path]],
        ],
        cutoff_grid: int = 3,
        index: str = ":",
        data_key: Optional[Dict[str, str]] = None,
        target_grid_spacing: Optional[
            Union[float, Sequence[float], np.ndarray]
        ] = None,
        boltzmann_constant: float = DEFAULT_BOLTZMANN_CONSTANT,
        thermal_wavelength: Union[
            float, Sequence[float], np.ndarray
        ] = 1.0,
        grid_info: Optional[Mapping[str, Any]] = None,
    ) -> List["GridData"]:
        """Read EXTXYZ frames and convert each frame to one ``GridData``.

        Parameters
        ----------
        path
            One EXTXYZ path, or an ordered sequence of EXTXYZ paths. Every
            file must contain a regular grid, temperature, and at least one of
            density or external potential. Chemical-potential metadata is
            optional. With multiple paths, frames are returned in path order
            and then in frame order within each file.
        cutoff_grid
            Inclusive spherical cutoff in integer grid steps. The default is
            three, so retained offsets satisfy ``x^2 + y^2 + z^2 <= 9``.
        index
            ASE frame selection. ``":"`` reads every frame.
        data_key
            Optional mapping from canonical GridData names to EXTXYZ field
            names. Supplied entries override :data:`default_data_key`.
        target_grid_spacing
            If provided, block-average all available grid fields onto this
            commensurate coarser spacing before constructing local
            environments.
        boltzmann_constant
            Boltzmann constant in energy per temperature. The default is in
            eV/K. Use ``1.0`` when temperature is already expressed in energy
            units, as in reduced Lennard-Jones data.
        thermal_wavelength
            Thermal de Broglie wavelength in the length unit reciprocal to
            that used by ``rho``. Supply one positive value or one value per
            component. It is stored when an external potential is available
            and is used both for equilibrium solving and direct-correlation
            references. The default is one.
        grid_info
            Optional model metadata dictionary containing ``cutoff_grid``,
            ``grid_spacing``, ``n_types``, ``boltzmann_constant``, and
            ``thermal_wavelength``. When supplied, it replaces the matching
            individual arguments and validates every processed frame against
            the trained grid spacing and component count.

        Returns
        -------
        list of GridData
            One complete grid configuration per selected EXTXYZ frame.
        """

        resolved_grid_info = None
        if grid_info is not None:
            resolved_grid_info = _normalize_grid_info(grid_info)
            cutoff_grid = resolved_grid_info["cutoff_grid"]
            boltzmann_constant = resolved_grid_info["boltzmann_constant"]
            thermal_wavelength = resolved_grid_info["thermal_wavelength"]

        boltzmann_constant = _positive_scalar(
            boltzmann_constant,
            "boltzmann_constant",
        )

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

        paths = _normalize_xyz_paths(path)

        # ASE returns either one Atoms object or a list, depending on `index`.
        # Normalize each read and append it in supplied path order. The same
        # frame selection is applied independently to every file.
        configurations: List[Atoms] = []
        for xyz_path in paths:
            selected = read(str(xyz_path), index=index)
            if isinstance(selected, list):
                configurations.extend(selected)
            else:
                configurations.append(selected)
        data = [
            cls(
                **_process_atoms(
                    atoms,
                    cutoff_grid,
                    keys,
                    target_grid_spacing=target_grid_spacing,
                    boltzmann_constant=boltzmann_constant,
                    thermal_wavelength=thermal_wavelength,
                )
            )
            for atoms in configurations
        ]
        # Default PyTorch collation requires identical dictionary keys. If a
        # selected frame contains an empty voxel, omit logarithmic pointwise
        # targets from the complete selection; the masked local-chemical-
        # potential loss uses rho and V_ext directly for every frame.
        if any("c1_plus_beta_mu" not in frame for frame in data):
            for frame in data:
                frame.pop("c1_plus_beta_mu", None)
                frame.pop("c1", None)
        if resolved_grid_info is not None:
            for frame in data:
                _validate_frame_grid_info(frame, resolved_grid_info)
        return data

    @classmethod
    def from_dict(
        cls,
        values: Dict[str, Any],
        cutoff_grid: int = 3,
        boltzmann_constant: float = DEFAULT_BOLTZMANN_CONSTANT,
        thermal_wavelength: Union[
            float, Sequence[float], np.ndarray
        ] = 1.0,
        grid_info: Optional[Mapping[str, Any]] = None,
    ) -> "GridData":
        """Build one regular periodic grid without reading an EXTXYZ file.

        ``values`` requires the three spatial dimensions in ``grid_size`` and
        either ``temperature`` or ``T``. It also requires the separate scalar
        ``n_types`` and ``grid_spacing`` unless they are supplied through
        ``grid_info``. Density, external potential, and chemical potential
        fields can be assigned to the returned dictionary afterward.
        """

        if not isinstance(values, dict):
            raise TypeError("values must be a dictionary")
        values = values.copy()
        if grid_info is not None:
            resolved_grid_info = _normalize_grid_info(grid_info)
            if "grid_spacing" in values:
                supplied_spacing = np.asarray(
                    values["grid_spacing"],
                    dtype=float,
                ).reshape(-1)
                if supplied_spacing.size == 1:
                    supplied_spacing = np.repeat(supplied_spacing, 3)
                if (
                    supplied_spacing.shape != (3,)
                    or not np.allclose(
                        supplied_spacing,
                        resolved_grid_info["grid_spacing"],
                    )
                ):
                    raise ValueError(
                        "values grid_spacing does not match grid_info"
                    )
            if "n_types" in values:
                supplied_n_types = np.asarray(
                    values["n_types"],
                    dtype=float,
                ).reshape(-1)
                if (
                    supplied_n_types.size != 1
                    or not np.isclose(
                        supplied_n_types[0],
                        resolved_grid_info["n_types"],
                    )
                ):
                    raise ValueError("values n_types does not match grid_info")
            values["grid_spacing"] = resolved_grid_info["grid_spacing"]
            values["n_types"] = resolved_grid_info["n_types"]
            cutoff_grid = resolved_grid_info["cutoff_grid"]
            boltzmann_constant = resolved_grid_info["boltzmann_constant"]
            thermal_wavelength = resolved_grid_info["thermal_wavelength"]
        allowed_keys = {
            "grid_size",
            "grid_spacing",
            "temperature",
            "T",
            "n_types",
        }
        unknown_keys = set(values) - allowed_keys
        if unknown_keys:
            raise KeyError(
                "unknown GridData.from_dict entries: {}".format(
                    sorted(unknown_keys)
                )
            )
        boltzmann_constant = _positive_scalar(
            boltzmann_constant,
            "boltzmann_constant",
        )
        if "grid_size" not in values:
            raise ValueError("values is missing required field 'grid_size'")
        if "grid_spacing" not in values:
            raise ValueError(
                "values is missing required field 'grid_spacing'"
            )
        if "n_types" not in values:
            raise ValueError("values is missing required field 'n_types'")
        temperature_value = values.get("temperature", values.get("T"))
        if temperature_value is None:
            raise ValueError(
                "values is missing required field 'temperature'"
            )

        raw_grid_size = np.asarray(
            values["grid_size"],
            dtype=float,
        ).reshape(-1)
        if raw_grid_size.size != 3:
            raise ValueError("grid_size must contain three values")
        grid_size = np.rint(raw_grid_size).astype(np.int64)
        if not np.allclose(raw_grid_size, grid_size) or np.any(
            grid_size <= 0
        ):
            raise ValueError("grid_size values must be positive integers")

        grid_spacing = np.asarray(
            values["grid_spacing"],
            dtype=float,
        ).reshape(-1)
        if grid_spacing.size == 1:
            grid_spacing = np.repeat(grid_spacing, 3)
        if (
            grid_spacing.size != 3
            or not np.all(np.isfinite(grid_spacing))
            or np.any(grid_spacing <= 0.0)
        ):
            raise ValueError(
                "grid_spacing must contain one or three finite positive values"
            )

        temperature = _positive_scalar(temperature_value, "temperature")
        raw_n_types = np.asarray(
            values["n_types"],
            dtype=float,
        ).reshape(-1)
        if raw_n_types.size != 1:
            raise ValueError("n_types must be a positive integer")
        n_types = int(np.rint(raw_n_types[0]))
        if not np.isclose(raw_n_types[0], n_types) or n_types < 1:
            raise ValueError("n_types must be a positive integer")

        thermal_wavelength_values = np.asarray(
            thermal_wavelength,
            dtype=float,
        ).reshape(-1)
        if thermal_wavelength_values.size == 1:
            thermal_wavelength_values = np.repeat(
                thermal_wavelength_values,
                n_types,
            )
        if (
            thermal_wavelength_values.size != n_types
            or not np.all(np.isfinite(thermal_wavelength_values))
            or np.any(thermal_wavelength_values <= 0.0)
        ):
            raise ValueError(
                "thermal_wavelength must contain one positive value per type"
            )

        grid_positions = np.indices(
            tuple(grid_size),
            dtype=np.int64,
        ).reshape(3, -1).T
        n_grid = grid_positions.shape[0]
        dtype = torch.get_default_dtype()
        return cls(
            temperature=torch.tensor(temperature, dtype=dtype),
            beta=torch.tensor(
                1.0 / (boltzmann_constant * temperature),
                dtype=dtype,
            ),
            n_types=torch.tensor(n_types, dtype=torch.long),
            grid_size=torch.tensor(grid_size, dtype=torch.long),
            thermal_wavelength=torch.tensor(
                thermal_wavelength_values,
                dtype=dtype,
            ),
            grid_spacing=torch.tensor(grid_spacing, dtype=dtype),
            index=torch.arange(n_grid, dtype=torch.long),
            grid_positions=torch.tensor(grid_positions, dtype=torch.long),
        )


def _normalize_grid_info(grid_info: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate model grid metadata and return CPU-native values."""

    if not isinstance(grid_info, Mapping):
        raise TypeError("grid_info must be a mapping")
    missing_keys = GRID_INFO_KEYS - set(grid_info)
    if missing_keys:
        raise KeyError(
            "grid_info is missing required entries: {}".format(
                sorted(missing_keys)
            )
        )
    unknown_keys = set(grid_info) - GRID_INFO_KEYS
    if unknown_keys:
        raise KeyError(
            "unknown grid_info entries: {}".format(sorted(unknown_keys))
        )

    raw_cutoff_grid = np.asarray(
        grid_info["cutoff_grid"],
        dtype=float,
    ).reshape(-1)
    if raw_cutoff_grid.size != 1:
        raise ValueError("grid_info cutoff_grid must be a nonnegative integer")
    cutoff_grid = int(np.rint(raw_cutoff_grid[0]))
    if not np.isclose(raw_cutoff_grid[0], cutoff_grid) or cutoff_grid < 0:
        raise ValueError("grid_info cutoff_grid must be a nonnegative integer")

    grid_spacing = np.asarray(
        grid_info["grid_spacing"],
        dtype=float,
    ).reshape(-1)
    if grid_spacing.size == 1:
        grid_spacing = np.repeat(grid_spacing, 3)
    if (
        grid_spacing.shape != (3,)
        or not np.all(np.isfinite(grid_spacing))
        or np.any(grid_spacing <= 0.0)
    ):
        raise ValueError(
            "grid_info grid_spacing must contain three positive values"
        )

    raw_n_types = np.asarray(
        grid_info["n_types"],
        dtype=float,
    ).reshape(-1)
    if raw_n_types.size != 1:
        raise ValueError("grid_info n_types must be a positive integer")
    n_types = int(np.rint(raw_n_types[0]))
    if not np.isclose(raw_n_types[0], n_types) or n_types < 1:
        raise ValueError("grid_info n_types must be a positive integer")

    boltzmann_constant = _positive_scalar(
        grid_info["boltzmann_constant"],
        "grid_info boltzmann_constant",
    )
    thermal_wavelength = np.asarray(
        grid_info["thermal_wavelength"],
        dtype=float,
    ).reshape(-1)
    if thermal_wavelength.size == 1:
        thermal_wavelength = np.repeat(thermal_wavelength, n_types)
    if (
        thermal_wavelength.shape != (n_types,)
        or not np.all(np.isfinite(thermal_wavelength))
        or np.any(thermal_wavelength <= 0.0)
    ):
        raise ValueError(
            "grid_info thermal_wavelength must contain one positive value "
            "per type"
        )

    return {
        "cutoff_grid": cutoff_grid,
        "grid_spacing": grid_spacing.tolist(),
        "n_types": n_types,
        "boltzmann_constant": boltzmann_constant,
        "thermal_wavelength": thermal_wavelength.tolist(),
    }


def _validate_frame_grid_info(
    frame: "GridData",
    grid_info: Mapping[str, Any],
) -> None:
    """Check that one processed EXTXYZ frame matches trained grid metadata."""

    if int(frame["n_types"].item()) != grid_info["n_types"]:
        raise ValueError("frame n_types does not match grid_info")
    frame_spacing = frame["grid_spacing"].detach().cpu().numpy()
    if not np.allclose(frame_spacing, grid_info["grid_spacing"]):
        raise ValueError("frame grid_spacing does not match grid_info")


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
    boltzmann_constant: float = DEFAULT_BOLTZMANN_CONSTANT,
    thermal_wavelength: Union[
        float, Sequence[float], np.ndarray
    ] = 1.0,
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

    # Apply exactly the same canonical permutation to the coordinates and all
    # available grid fields. At least one physical field is needed to infer
    # the number of components, but rho itself is optional for an equilibrium
    # inference record that supplies only V_ext.
    grid_positions = grid_positions[order]
    rho_value = _get_source_value(atoms, data_key["rho"])
    rho = None
    if rho_value is not None:
        rho = np.asarray(rho_value, dtype=float)[order]
        if rho.ndim not in (1, 2):
            raise ValueError(
                "rho must have shape [n_grid] or [n_grid, n_types]"
            )
        if rho.ndim == 1:
            rho = rho[:, None]
        if not np.all(np.isfinite(rho)):
            raise ValueError("rho must contain only finite values")
        if np.any(rho < 0.0):
            raise ValueError("rho must contain only nonnegative values")

    V_ext_value = _get_source_value(atoms, data_key["V_ext"])
    V_ext = None
    if V_ext_value is not None:
        V_ext = np.asarray(V_ext_value, dtype=float)[order]
        if V_ext.ndim not in (1, 2):
            raise ValueError(
                "V_ext must have shape [n_grid] or [n_grid, n_types]"
            )
        if V_ext.ndim == 1:
            V_ext = V_ext[:, None]
    if rho is None and V_ext is None:
        raise ValueError("frame must contain at least one of rho or V_ext")

    n_types = rho.shape[1] if rho is not None else V_ext.shape[1]
    if rho is not None and V_ext is not None:
        if rho.shape[1] != V_ext.shape[1]:
            raise ValueError(
                "rho and V_ext must contain the same number of columns"
            )

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

    # Optionally average all available fields over non-overlapping regular-grid
    # blocks. The local environments are built only after this transformation.
    if target_grid_spacing is not None:
        fields_to_coarsen = []
        if rho is not None:
            fields_to_coarsen.append(rho)
        if V_ext is not None:
            fields_to_coarsen.append(V_ext)
        fields, grid_positions, grid_spacing = coarsen_grid(
            values=np.column_stack(fields_to_coarsen),
            grid_positions=grid_positions,
            grid_spacing=grid_spacing,
            target_grid_spacing=target_grid_spacing,
        )
        field_start = 0
        if rho is not None:
            rho = fields[:, field_start : field_start + n_types]
            field_start += n_types
        if V_ext is not None:
            V_ext = fields[:, field_start : field_start + n_types]
        n_grid = grid_positions.shape[0]
        grid_size = grid_positions.max(axis=0) + 1

    # Temperature is required even for an initial fixed-temperature fit so the
    # data retain their thermodynamic state and unit conversion explicitly.
    temperature = _positive_scalar(
        _required_source_value(
            atoms,
            data_key["temperature"],
            "temperature",
        ),
        "temperature",
    )
    beta = 1.0 / (boltzmann_constant * temperature)

    # Chemical potential is optional metadata. Normalize it to one value per
    # physical density component even if the source stores a scalar.
    mu_value = _get_source_value(atoms, data_key["mu"])
    mu_values = None
    if mu_value is not None:
        mu_values = np.asarray(mu_value, dtype=float).reshape(-1)
        if mu_values.size == 1:
            mu_values = np.repeat(mu_values, n_types)
        if mu_values.size != n_types:
            raise ValueError("mu must contain one value or one value per type")
        if np.all(np.isnan(mu_values)):
            # EXTXYZ writers commonly use mu=nan as an explicit placeholder
            # for canonical data. Treat this exactly like absent optional mu.
            mu_values = None
        elif not np.all(np.isfinite(mu_values)):
            raise ValueError("mu values must be finite")

    # The thermal wavelength is required when an external field defines an
    # equilibrium inference problem. A reference direct-correlation field is
    # constructed only when both rho and V_ext are available. rho and the
    # thermal wavelength use reciprocal length units so rho*Lambda^3 is
    # dimensionless.
    thermal_wavelength_values = None
    c1_plus_beta_mu = None
    if V_ext is not None:
        thermal_wavelength_values = np.asarray(
            thermal_wavelength,
            dtype=float,
        ).reshape(-1)
        if thermal_wavelength_values.size == 1:
            thermal_wavelength_values = np.repeat(
                thermal_wavelength_values,
                n_types,
            )
        if thermal_wavelength_values.size != n_types:
            raise ValueError(
                "thermal_wavelength must contain one value or one value per type"
            )
        if (
            not np.all(np.isfinite(thermal_wavelength_values))
            or np.any(thermal_wavelength_values <= 0.0)
        ):
            raise ValueError(
                "thermal_wavelength values must be finite and positive"
            )
    # Precompute conventional pointwise targets only when the logarithm is
    # defined everywhere. The model's chemical-potential weights mask empty
    # voxels in weighted objectives, so zero-density records remain valid.
    if (
        rho is not None
        and V_ext is not None
        and np.all(rho > 0.0)
    ):
        c1_plus_beta_mu = (
            np.log(rho * thermal_wavelength_values[None, :] ** 3)
            + beta * V_ext
        )

    # Use PyTorch's current default floating dtype for physical values and
    # int64 tensors for indices and integer grid coordinates.
    dtype = torch.get_default_dtype()
    data = {
        "temperature": torch.tensor(temperature, dtype=dtype),
        "beta": torch.tensor(beta, dtype=dtype),
        "n_types": torch.tensor(n_types, dtype=torch.long),
        "grid_size": torch.tensor(grid_size, dtype=torch.long),
        "grid_spacing": torch.tensor(grid_spacing, dtype=dtype),
        "index": torch.arange(n_grid, dtype=torch.long),
        "grid_positions": torch.tensor(grid_positions, dtype=torch.long),
    }

    if rho is not None:
        data["rho"] = torch.tensor(rho, dtype=dtype)
    if V_ext is not None:
        data["V_ext"] = torch.tensor(V_ext, dtype=dtype)
    if mu_values is not None:
        data["mu"] = torch.tensor(mu_values, dtype=dtype)
        # The local chemical potential produced by the model is dimensionless,
        # so beta_mu is the directly compatible supervised target. Retain mu
        # separately in its original energy units for thermodynamic use.
        data["beta_mu"] = torch.tensor(beta * mu_values, dtype=dtype)
    if thermal_wavelength_values is not None:
        data["thermal_wavelength"] = torch.tensor(
            thermal_wavelength_values,
            dtype=dtype,
        )
    if c1_plus_beta_mu is not None:
        data["c1_plus_beta_mu"] = torch.tensor(
            c1_plus_beta_mu,
            dtype=dtype,
        )
    if c1_plus_beta_mu is not None and mu_values is not None:
        data["c1"] = torch.tensor(
            c1_plus_beta_mu - beta * mu_values[None, :],
            dtype=dtype,
        )
    return data


def _positive_scalar(value: Any, name: str) -> float:
    """Return one finite positive scalar with a field-specific error."""

    values = np.asarray(value, dtype=float).reshape(-1)
    if values.size != 1:
        raise ValueError("{} must be a scalar".format(name))
    scalar = float(values[0])
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError("{} must be finite and positive".format(name))
    return scalar
