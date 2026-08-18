"""Public construction interface for complete periodic density fields."""

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
from ase import Atoms
from ase.io import read

from ._data_helpers import (
    build_grid_data,
    harmonize_optional_targets,
    normalize_grid_info,
    normalize_grid_size,
    normalize_grid_spacing,
    positive_integer,
    positive_scalar,
    process_atoms,
    validate_frame_grid_info,
)


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
    "mask": "mask",
}


def _normalize_xyz_paths(
    path: Union[str, Path, Sequence[Union[str, Path]]],
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

    Every object contains the grid geometry, temperature, beta, and
    neighborhood fields. Density, external-potential, chemical-potential, and
    reference fields are included only when their source quantities are
    available::

        temperature                 scalar
        beta                        scalar, 1 / (k_B*T)
        mu                          [n_types] (optional, energy units)
        beta_mu                     [n_types] (optional, dimensionless beta*mu)
        n_types                     scalar
        thermal_wavelength          [n_types] (when needed for inference)
        grid_spacing                [3]
        index                       [n_grid]
        grid_positions              [n_grid, 3]
        V_ext                       [n_grid, n_types] (optional)
        rho                         [n_grid, n_types] (optional)
        mask                        [n_grid] bool; true grid points are excluded
        c1_plus_beta_mu             [n_grid, n_types] (optional, dimensionless)
        c1                          [n_grid, n_types] (optional)
        local_density_index         [n_grid, n_neighbors]
        local_density_positions     [n_neighbors, 3]

    EXTXYZ records require at least one of ``rho`` and ``V_ext``. A grid built
    directly with :meth:`from_dict` may initially contain neither. Model
    forwards gather from the live ``rho`` tensor through
    ``local_density_index``, preserving the functional-derivative graph.
    """

    @classmethod
    def from_xyz(
        cls,
        path: Union[str, Path, Sequence[Union[str, Path]]],
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
        """Read selected EXTXYZ frames into complete periodic grid records.

        ``path`` may contain one path or an ordered sequence. ``data_key``
        maps canonical field names to EXTXYZ names. If
        ``target_grid_spacing`` is supplied, available fields are block-
        averaged before neighborhoods are constructed. ``grid_info`` may be
        supplied from a trained model to configure and validate its grid and
        unit metadata.
        """

        resolved_grid_info = None
        if grid_info is not None:
            resolved_grid_info = normalize_grid_info(grid_info)
            cutoff_grid = resolved_grid_info["cutoff_grid"]
            boltzmann_constant = resolved_grid_info["boltzmann_constant"]
            thermal_wavelength = resolved_grid_info["thermal_wavelength"]
        boltzmann_constant = positive_scalar(
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

        configurations: List[Atoms] = []
        for xyz_path in _normalize_xyz_paths(path):
            selected = read(str(xyz_path), index=index)
            if isinstance(selected, list):
                configurations.extend(selected)
            else:
                configurations.append(selected)

        # Complete fields normally share their immutable neighborhood tensors.
        geometry_cache: Dict[Any, Any] = {}
        data = [
            cls(
                **process_atoms(
                    atoms=atoms,
                    cutoff_grid=cutoff_grid,
                    data_key=keys,
                    target_grid_spacing=target_grid_spacing,
                    boltzmann_constant=boltzmann_constant,
                    thermal_wavelength=thermal_wavelength,
                    geometry_cache=geometry_cache,
                )
            )
            for atoms in configurations
        ]
        harmonize_optional_targets(data)
        if resolved_grid_info is not None:
            for frame in data:
                validate_frame_grid_info(frame, resolved_grid_info)
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
        """Build one empty regular periodic grid from explicit metadata.

        ``values`` requires ``grid_size``, ``grid_spacing``, ``n_types``, and
        either ``temperature`` or ``T``. Matching model metadata may instead
        be supplied through ``grid_info``. Density and external-potential
        fields can be assigned to the returned dictionary afterward.
        """

        if not isinstance(values, dict):
            raise TypeError("values must be a dictionary")
        values = values.copy()
        if grid_info is not None:
            resolved = normalize_grid_info(grid_info)
            if "grid_spacing" in values:
                supplied_spacing = normalize_grid_spacing(
                    values["grid_spacing"]
                )
                if not np.allclose(
                    supplied_spacing,
                    resolved["grid_spacing"],
                ):
                    raise ValueError(
                        "values grid_spacing does not match grid_info"
                    )
            if "n_types" in values:
                supplied_n_types = positive_integer(
                    values["n_types"],
                    "n_types",
                )
                if supplied_n_types != resolved["n_types"]:
                    raise ValueError(
                        "values n_types does not match grid_info"
                    )
            values["grid_spacing"] = resolved["grid_spacing"]
            values["n_types"] = resolved["n_types"]
            cutoff_grid = resolved["cutoff_grid"]
            boltzmann_constant = resolved["boltzmann_constant"]
            thermal_wavelength = resolved["thermal_wavelength"]

        allowed_keys = {
            "grid_size",
            "grid_spacing",
            "temperature",
            "T",
            "n_types",
            "mask",
        }
        unknown_keys = set(values) - allowed_keys
        if unknown_keys:
            raise KeyError(
                "unknown GridData.from_dict entries: {}".format(
                    sorted(unknown_keys)
                )
            )
        for required in ("grid_size", "grid_spacing", "n_types"):
            if required not in values:
                raise ValueError(
                    "values is missing required field '{}'".format(required)
                )
        temperature = values.get("temperature", values.get("T"))
        if temperature is None:
            raise ValueError(
                "values is missing required field 'temperature'"
            )

        grid_size = normalize_grid_size(values["grid_size"])
        grid_spacing = normalize_grid_spacing(values["grid_spacing"])
        temperature = positive_scalar(temperature, "temperature")
        n_types = positive_integer(values["n_types"], "n_types")
        grid_positions = np.indices(
            tuple(grid_size),
            dtype=np.int64,
        ).reshape(3, -1).T
        return cls(
            **build_grid_data(
                grid_positions=grid_positions,
                grid_size=grid_size,
                grid_spacing=grid_spacing,
                temperature=temperature,
                n_types=n_types,
                cutoff_grid=cutoff_grid,
                boltzmann_constant=boltzmann_constant,
                thermal_wavelength=thermal_wavelength,
                mask=values.get("mask"),
            )
        )
