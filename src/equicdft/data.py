"""Public construction interface for complete periodic density fields."""

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch
from ase import Atoms
from ase.io import read
from torch.utils.data import Dataset

from ._argument_checks import (
    nonempty_string,
    nonnegative_integer,
    positive_scalar,
)
from ._data_helpers import (
    _metadata_grid_size,
    _metadata_grid_spacing,
    _metadata_positive_integer,
    _metadata_positive_scalar,
    build_grid_data,
    harmonize_optional_targets,
    normalize_grid_info,
    process_atoms,
    validate_frame_grid_info,
)
from ._fourier import canonical_mode_triplets, integer_mode_tensor


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
    "excluded_mask": "excluded_mask",
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
        excluded_mask               [n_grid] bool; true grid points are excluded
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
        cutoff_grid = nonnegative_integer(cutoff_grid, "cutoff_grid")
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
                supplied_spacing = _metadata_grid_spacing(
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
                supplied_n_types = _metadata_positive_integer(
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

        cutoff_grid = nonnegative_integer(cutoff_grid, "cutoff_grid")
        boltzmann_constant = positive_scalar(
            boltzmann_constant,
            "boltzmann_constant",
        )

        allowed_keys = {
            "grid_size",
            "grid_spacing",
            "temperature",
            "T",
            "n_types",
            "excluded_mask",
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

        grid_size = _metadata_grid_size(values["grid_size"])
        grid_spacing = _metadata_grid_spacing(values["grid_spacing"])
        temperature = _metadata_positive_scalar(temperature, "temperature")
        n_types = _metadata_positive_integer(values["n_types"], "n_types")
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
                excluded_mask=values.get("excluded_mask"),
            )
        )


class FourierResponseData(Dataset):
    """Homogeneous grid fields paired with projected Fourier curvatures.

    ``template`` supplies one periodic grid geometry. ``density`` contains
    the component densities for each response item, while ``modes`` and
    ``curvature`` contain its integer reciprocal modes and projected response
    targets.  The dataset is deliberately agnostic to component names,
    directions, thermodynamic units, and how response targets were obtained.

    A single mode per item may be supplied as ``[n_items, 3]``; otherwise
    modes have shape ``[n_items, n_modes, 3]``. Curvature, scale, and weight
    have shape ``[n_items, n_modes, n_directions]``. Optional ``indices``
    selects an existing, externally defined split without regenerating it.
    """

    def __init__(
        self,
        template: Mapping[str, Any],
        density: Any,
        modes: Any,
        curvature: Any,
        scale: Optional[Any] = None,
        weight: Optional[Any] = None,
        indices: Optional[Sequence[int]] = None,
        modes_key: str = "fourier_modes",
        target_key: str = "fourier_curvature",
        scale_key: str = "fourier_scale",
        weights_key: str = "fourier_weight",
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        if not isinstance(template, Mapping):
            raise TypeError("template must be a mapping")
        self.template = dict(template)
        required_template_keys = {
            "index",
            "n_types",
            "grid_size",
            "grid_spacing",
        }
        missing_template_keys = required_template_keys - set(self.template)
        if missing_template_keys:
            raise ValueError(
                "template is missing keys: {}".format(
                    sorted(missing_template_keys)
                )
            )
        self.modes_key = nonempty_string(modes_key, "modes_key")
        self.target_key = nonempty_string(target_key, "target_key")
        self.scale_key = nonempty_string(scale_key, "scale_key")
        self.weights_key = nonempty_string(weights_key, "weights_key")

        if dtype is None:
            dtype = torch.get_default_dtype()
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise TypeError("dtype must be a floating-point torch dtype")
        self.density = torch.as_tensor(density, dtype=dtype)
        self.modes = integer_mode_tensor(modes)
        self.curvature = torch.as_tensor(curvature, dtype=dtype)
        if self.modes.ndim == 2:
            self.modes = self.modes.unsqueeze(1)
        if self.curvature.ndim == 2:
            self.curvature = self.curvature.unsqueeze(1)

        if self.density.ndim != 2:
            raise ValueError("density must have shape [n_items, n_types]")
        n_items, n_types = self.density.shape
        if int(torch.as_tensor(self.template["n_types"]).item()) != n_types:
            raise ValueError("density n_types does not match template")
        if self.modes.shape[:1] != (n_items,) or self.modes.ndim != 3:
            raise ValueError("modes must have shape [n_items, n_modes, 3]")
        if self.modes.shape[-1] != 3:
            raise ValueError("modes must contain three integer components")
        self.modes = torch.stack(
            [
                canonical_mode_triplets(
                    item_modes,
                    self.template["grid_size"],
                    self.template["grid_spacing"],
                )
                for item_modes in self.modes
            ]
        )
        expected = (n_items, self.modes.shape[1])
        if self.curvature.ndim != 3 or self.curvature.shape[:2] != expected:
            raise ValueError(
                "curvature must have shape [n_items, n_modes, n_directions]"
            )

        self.scale = self._optional_response_tensor(
            scale,
            "scale",
            positive=True,
        )
        self.weight = self._optional_response_tensor(
            weight,
            "weight",
            nonnegative=True,
        )
        if not torch.all(torch.isfinite(self.density)).item():
            raise ValueError("density must be finite")
        if torch.any(self.density <= 0.0).item():
            raise ValueError("density must be positive")
        if not torch.all(torch.isfinite(self.curvature)).item():
            raise ValueError("curvature must be finite")

        if indices is None:
            self.indices = tuple(range(n_items))
        else:
            values = tuple(indices)
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                for value in values
            ):
                raise TypeError("indices must contain only integers")
            if any(value < 0 or value >= n_items for value in values):
                raise IndexError("indices are outside the response data")
            self.indices = tuple(int(value) for value in values)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        frame = dict(self.template)
        n_grid = int(torch.as_tensor(frame["index"]).numel())
        frame["rho"] = self.density[index].expand(n_grid, -1).clone()
        frame[self.modes_key] = self.modes[index]
        frame[self.target_key] = self.curvature[index]
        if self.scale is not None:
            frame[self.scale_key] = self.scale[index]
        if self.weight is not None:
            frame[self.weights_key] = self.weight[index]
        return frame

    def _optional_response_tensor(
        self,
        value: Optional[Any],
        name: str,
        positive: bool = False,
        nonnegative: bool = False,
    ) -> Optional[torch.Tensor]:
        if value is None:
            return None
        tensor = torch.as_tensor(value, dtype=self.curvature.dtype)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(1)
        if tensor.shape != self.curvature.shape:
            raise ValueError("{} must have the curvature shape".format(name))
        if not torch.all(torch.isfinite(tensor)).item():
            raise ValueError("{} must be finite".format(name))
        if positive and torch.any(tensor <= 0.0).item():
            raise ValueError("{} must be positive".format(name))
        if nonnegative and torch.any(tensor < 0.0).item():
            raise ValueError("{} must be nonnegative".format(name))
        return tensor
