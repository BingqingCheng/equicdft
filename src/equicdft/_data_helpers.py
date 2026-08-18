"""Private parsing, validation, and construction helpers for GridData."""

from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch
from ase import Atoms

from .stencil import coarsen_grid, get_neighbor_indices


GRID_INFO_KEYS = {
    "cutoff_grid",
    "grid_spacing",
    "n_types",
    "boltzmann_constant",
    "thermal_wavelength",
}


def normalize_grid_info(grid_info: Mapping[str, Any]) -> Dict[str, Any]:
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

    cutoff_grid = nonnegative_integer(
        grid_info["cutoff_grid"],
        "grid_info cutoff_grid",
    )
    grid_spacing = normalize_grid_spacing(
        grid_info["grid_spacing"],
        "grid_info grid_spacing",
    )
    n_types = positive_integer(
        grid_info["n_types"],
        "grid_info n_types",
    )
    boltzmann_constant = positive_scalar(
        grid_info["boltzmann_constant"],
        "grid_info boltzmann_constant",
    )
    thermal_wavelength = per_type_positive_values(
        grid_info["thermal_wavelength"],
        n_types,
        "grid_info thermal_wavelength",
    )
    return {
        "cutoff_grid": cutoff_grid,
        "grid_spacing": grid_spacing.tolist(),
        "n_types": n_types,
        "boltzmann_constant": boltzmann_constant,
        "thermal_wavelength": thermal_wavelength.tolist(),
    }


def validate_frame_grid_info(
    frame: Dict[str, torch.Tensor],
    grid_info: Mapping[str, Any],
) -> None:
    """Check that one processed EXTXYZ frame matches trained grid metadata."""

    if int(frame["n_types"].item()) != grid_info["n_types"]:
        raise ValueError("frame n_types does not match grid_info")
    frame_spacing = frame["grid_spacing"].detach().cpu().numpy()
    if not np.allclose(frame_spacing, grid_info["grid_spacing"]):
        raise ValueError("frame grid_spacing does not match grid_info")


def process_atoms(
    atoms: Atoms,
    cutoff_grid: int,
    data_key: Dict[str, str],
    target_grid_spacing: Optional[Any],
    boltzmann_constant: float,
    thermal_wavelength: Any,
    geometry_cache: Optional[Dict[Any, Any]],
) -> Dict[str, torch.Tensor]:
    """Extract and canonicalize one EXTXYZ frame, then build its tensors."""

    positions = np.asarray(
        _required_source_value(
            atoms,
            data_key["grid_positions"],
            "grid_positions",
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

    grid_size_value = _get_source_value(atoms, data_key["grid_size"])
    if grid_size_value is None:
        grid_size = grid_positions.max(axis=0) + 1
    else:
        grid_size = normalize_grid_size(grid_size_value)
    n_grid = int(np.prod(grid_size))
    if len(atoms) != n_grid:
        raise ValueError("frame does not contain one complete regular grid")
    if np.any(grid_positions.max(axis=0) + 1 != grid_size):
        raise ValueError("grid positions are inconsistent with grid_size")

    flat_positions = np.ravel_multi_index(
        grid_positions.T,
        tuple(grid_size),
        order="C",
    )
    if np.unique(flat_positions).size != n_grid:
        raise ValueError("grid positions contain duplicate grid points")
    order = np.argsort(flat_positions)
    if not np.array_equal(flat_positions[order], np.arange(n_grid)):
        raise ValueError("grid positions do not cover the complete grid")
    grid_positions = grid_positions[order]

    rho = _ordered_optional_field(atoms, data_key["rho"], order, "rho")
    mask = _ordered_optional_mask(atoms, data_key["mask"], order)
    V_ext = _ordered_optional_field(
        atoms,
        data_key["V_ext"],
        order,
        "V_ext",
    )
    if rho is None and V_ext is None:
        raise ValueError("frame must contain at least one of rho or V_ext")
    n_types = rho.shape[1] if rho is not None else V_ext.shape[1]
    if rho is not None and V_ext is not None:
        if rho.shape[1] != V_ext.shape[1]:
            raise ValueError(
                "rho and V_ext must contain the same number of columns"
            )

    grid_spacing = normalize_grid_spacing(
        _required_source_value(
            atoms,
            data_key["grid_spacing"],
            "grid_spacing",
        )
    )
    if target_grid_spacing is not None:
        fields_to_coarsen = [
            field for field in (rho, V_ext) if field is not None
        ]
        if mask is not None:
            fields_to_coarsen.append(mask[:, None].astype(float))
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
            field_start += n_types
        if mask is not None:
            coarse_mask_fraction = fields[:, field_start]
            uniform_mask = np.isclose(coarse_mask_fraction, 0.0) | np.isclose(
                coarse_mask_fraction,
                1.0,
            )
            if not np.all(uniform_mask):
                raise ValueError(
                    "coarsening would create partially accessible grid "
                    "points; supply a hard-wall mask on the target grid"
                )
            mask = np.isclose(coarse_mask_fraction, 1.0)
        grid_size = grid_positions.max(axis=0) + 1
        if mask is None:
            mask = np.zeros(len(grid_positions), dtype=bool)

    temperature = positive_scalar(
        _required_source_value(
            atoms,
            data_key["temperature"],
            "temperature",
        ),
        "temperature",
    )
    return build_grid_data(
        grid_positions=grid_positions,
        grid_size=grid_size,
        grid_spacing=grid_spacing,
        temperature=temperature,
        n_types=n_types,
        cutoff_grid=cutoff_grid,
        boltzmann_constant=boltzmann_constant,
        thermal_wavelength=thermal_wavelength,
        rho=rho,
        V_ext=V_ext,
        mu=_get_source_value(atoms, data_key["mu"]),
        mask=mask,
        geometry_cache=geometry_cache,
        include_thermal_wavelength=V_ext is not None,
    )


def build_grid_data(
    grid_positions: Any,
    grid_size: Any,
    grid_spacing: Any,
    temperature: Any,
    n_types: Any,
    cutoff_grid: Any,
    boltzmann_constant: Any,
    thermal_wavelength: Any,
    rho: Optional[Any] = None,
    V_ext: Optional[Any] = None,
    mu: Optional[Any] = None,
    mask: Optional[Any] = None,
    geometry_cache: Optional[Dict[Any, Any]] = None,
    include_thermal_wavelength: bool = True,
) -> Dict[str, torch.Tensor]:
    """Build the canonical tensor dictionary from normalized grid fields."""

    grid_size_values = normalize_grid_size(grid_size)
    grid_spacing_values = normalize_grid_spacing(grid_spacing)
    temperature_value = positive_scalar(temperature, "temperature")
    n_type_values = positive_integer(n_types, "n_types")
    cutoff_value = nonnegative_integer(cutoff_grid, "cutoff_grid")
    boltzmann_value = positive_scalar(
        boltzmann_constant,
        "boltzmann_constant",
    )

    positions = np.asarray(grid_positions, dtype=np.int64)
    n_grid = int(np.prod(grid_size_values))
    if positions.shape != (n_grid, 3):
        raise ValueError(
            "grid_positions must have shape [prod(grid_size), 3]"
        )
    rho_values = _normalize_grid_field(
        rho,
        n_grid,
        n_type_values,
        "rho",
        nonnegative=True,
    )
    V_ext_values = _normalize_grid_field(
        V_ext,
        n_grid,
        n_type_values,
        "V_ext",
    )
    mu_values = _normalize_optional_mu(mu, n_type_values)
    mask_values = _normalize_mask(mask, n_grid)
    if rho_values is not None and np.any(
        np.abs(rho_values[mask_values]) > 1.0e-12
    ):
        raise ValueError("rho must be zero at masked grid points")
    if rho_values is not None:
        rho_values = rho_values.copy()
        rho_values[mask_values] = 0.0
    wavelength_values = None
    if include_thermal_wavelength:
        wavelength_values = per_type_positive_values(
            thermal_wavelength,
            n_type_values,
            "thermal_wavelength",
        )

    local_density_index, local_density_positions = _geometry_tensors(
        grid_positions=positions,
        grid_size=grid_size_values,
        cutoff_grid=cutoff_value,
        geometry_cache=geometry_cache,
    )
    beta = 1.0 / (boltzmann_value * temperature_value)
    dtype = torch.get_default_dtype()
    data = {
        "temperature": torch.tensor(temperature_value, dtype=dtype),
        "beta": torch.tensor(beta, dtype=dtype),
        "n_types": torch.tensor(n_type_values, dtype=torch.long),
        "grid_size": torch.tensor(grid_size_values, dtype=torch.long),
        "grid_spacing": torch.tensor(grid_spacing_values, dtype=dtype),
        "index": torch.arange(n_grid, dtype=torch.long),
        "grid_positions": torch.tensor(positions, dtype=torch.long),
        "mask": torch.tensor(mask_values, dtype=torch.bool),
        "local_density_index": local_density_index,
        "local_density_positions": local_density_positions,
    }
    if rho_values is not None:
        data["rho"] = torch.tensor(rho_values, dtype=dtype)
    if V_ext_values is not None:
        data["V_ext"] = torch.tensor(V_ext_values, dtype=dtype)
    if wavelength_values is not None:
        data["thermal_wavelength"] = torch.tensor(
            wavelength_values,
            dtype=dtype,
        )
    _add_thermodynamic_fields(
        data=data,
        rho=rho_values,
        V_ext=V_ext_values,
        mu=mu_values,
        beta=beta,
        thermal_wavelength=wavelength_values,
        mask=mask_values,
    )
    return data


def harmonize_optional_targets(data: List[Dict[str, torch.Tensor]]) -> None:
    """Make optional target keys compatible with default PyTorch collation."""

    if any("c1_plus_beta_mu" not in frame for frame in data):
        for frame in data:
            frame.pop("c1_plus_beta_mu", None)
            frame.pop("c1", None)
    elif any("c1" not in frame for frame in data):
        for frame in data:
            frame.pop("c1", None)

    has_beta_mu = ["beta_mu" in frame for frame in data]
    if not any(has_beta_mu) or all(has_beta_mu):
        return
    for frame, available in zip(data, has_beta_mu):
        if available:
            continue
        n_types = int(frame["n_types"].item())
        missing_mu = torch.full(
            (n_types,),
            float("nan"),
            dtype=frame["temperature"].dtype,
        )
        frame["mu"] = missing_mu.clone()
        frame["beta_mu"] = missing_mu


def normalize_grid_size(value: Any) -> np.ndarray:
    """Return three positive integer grid dimensions."""

    raw = np.asarray(value, dtype=float).reshape(-1)
    if raw.size != 3:
        raise ValueError("grid_size must contain three values")
    grid_size = np.rint(raw).astype(np.int64)
    if not np.allclose(raw, grid_size) or np.any(grid_size <= 0):
        raise ValueError("grid_size values must be positive integers")
    return grid_size


def normalize_grid_spacing(
    value: Any,
    name: str = "grid_spacing",
) -> np.ndarray:
    """Return one positive physical spacing for each Cartesian axis."""

    spacing = np.asarray(value, dtype=float).reshape(-1)
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


def positive_integer(value: Any, name: str) -> int:
    """Return one positive integer with a field-specific error."""

    integer = nonnegative_integer(value, name)
    if integer < 1:
        raise ValueError("{} must be a positive integer".format(name))
    return integer


def nonnegative_integer(value: Any, name: str) -> int:
    """Return one nonnegative integer with a field-specific error."""

    values = np.asarray(value, dtype=float).reshape(-1)
    if values.size != 1:
        raise ValueError("{} must be a nonnegative integer".format(name))
    integer = int(np.rint(values[0]))
    if not np.isclose(values[0], integer) or integer < 0:
        raise ValueError("{} must be a nonnegative integer".format(name))
    return integer


def positive_scalar(value: Any, name: str) -> float:
    """Return one finite positive scalar with a field-specific error."""

    values = np.asarray(value, dtype=float).reshape(-1)
    if values.size != 1:
        raise ValueError("{} must be a scalar".format(name))
    scalar = float(values[0])
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError("{} must be finite and positive".format(name))
    return scalar


def per_type_positive_values(
    value: Any,
    n_types: int,
    name: str,
) -> np.ndarray:
    """Return one finite positive value per physical component."""

    values = np.asarray(value, dtype=float).reshape(-1)
    if values.size == 1:
        values = np.repeat(values, n_types)
    if (
        values.size != n_types
        or not np.all(np.isfinite(values))
        or np.any(values <= 0.0)
    ):
        raise ValueError(
            "{} must contain one positive value per type".format(name)
        )
    return values


def _get_source_value(atoms: Atoms, source_key: str) -> Optional[Any]:
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
            "frame is missing '{}' for GridData field '{}'".format(
                source_key,
                field,
            )
        )
    return value


def _ordered_optional_field(
    atoms: Atoms,
    source_key: str,
    order: np.ndarray,
    name: str,
) -> Optional[np.ndarray]:
    value = _get_source_value(atoms, source_key)
    if value is None:
        return None
    field = np.asarray(value, dtype=float)[order]
    if field.ndim == 1:
        field = field[:, None]
    if field.ndim != 2:
        raise ValueError(
            "{} must have shape [n_grid] or [n_grid, n_types]".format(name)
        )
    return field


def _ordered_optional_mask(
    atoms: Atoms,
    source_key: str,
    order: np.ndarray,
) -> Optional[np.ndarray]:
    """Return an optional Boolean hard-wall mask in canonical grid order."""

    value = _get_source_value(atoms, source_key)
    if value is None:
        return None
    return _normalize_mask(np.asarray(value)[order], len(order))


def _normalize_grid_field(
    value: Optional[Any],
    n_grid: int,
    n_types: int,
    name: str,
    nonnegative: bool = False,
) -> Optional[np.ndarray]:
    if value is None:
        return None
    field = np.asarray(value, dtype=float)
    if field.ndim == 1:
        field = field[:, None]
    if field.shape != (n_grid, n_types):
        raise ValueError(
            "{} must have shape [n_grid, n_types]".format(name)
        )
    if not np.all(np.isfinite(field)):
        raise ValueError("{} must contain only finite values".format(name))
    if nonnegative and np.any(field < 0.0):
        raise ValueError("{} must contain only nonnegative values".format(name))
    return field


def _normalize_mask(value: Optional[Any], n_grid: int) -> np.ndarray:
    """Return a Boolean exclusion mask with one value per grid point."""

    if value is None:
        return np.zeros(n_grid, dtype=bool)
    mask = np.asarray(value)
    if mask.ndim == 2 and mask.shape[1] == 1:
        mask = mask[:, 0]
    if mask.shape != (n_grid,):
        raise ValueError("mask must have shape [n_grid]")
    if mask.dtype != np.bool_:
        try:
            numeric_mask = np.asarray(mask, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("mask values must be Boolean or binary")
        if (
            not np.all(np.isfinite(numeric_mask))
            or not np.all((numeric_mask == 0.0) | (numeric_mask == 1.0))
        ):
            raise ValueError("mask values must be Boolean or binary")
        mask = numeric_mask.astype(bool)
    else:
        mask = mask.astype(bool, copy=False)
    if np.all(mask):
        raise ValueError("mask must leave at least one accessible grid point")
    return mask


def _normalize_optional_mu(
    value: Optional[Any],
    n_types: int,
) -> Optional[np.ndarray]:
    if value is None:
        return None
    mu = np.asarray(value, dtype=float).reshape(-1)
    if mu.size == 1:
        mu = np.repeat(mu, n_types)
    if mu.size != n_types:
        raise ValueError("mu must contain one value or one value per type")
    if np.all(np.isnan(mu)):
        return None
    if not np.all(np.isfinite(mu)):
        raise ValueError("mu values must be finite")
    return mu


def _geometry_tensors(
    grid_positions: np.ndarray,
    grid_size: np.ndarray,
    cutoff_grid: int,
    geometry_cache: Optional[Dict[Any, Any]],
) -> Any:
    geometry_key = (cutoff_grid,) + tuple(int(value) for value in grid_size)
    cached = (
        None if geometry_cache is None else geometry_cache.get(geometry_key)
    )
    if cached is not None:
        return cached
    local_density_index, local_density_positions = get_neighbor_indices(
        grid_positions=grid_positions,
        cutoff_grid=cutoff_grid,
    )
    geometry = (
        torch.tensor(local_density_index, dtype=torch.long),
        torch.tensor(local_density_positions, dtype=torch.long),
    )
    if geometry_cache is not None:
        geometry_cache[geometry_key] = geometry
    return geometry


def _add_thermodynamic_fields(
    data: Dict[str, torch.Tensor],
    rho: Optional[np.ndarray],
    V_ext: Optional[np.ndarray],
    mu: Optional[np.ndarray],
    beta: float,
    thermal_wavelength: Optional[np.ndarray],
    mask: np.ndarray,
) -> None:
    dtype = torch.get_default_dtype()
    if mu is not None:
        data["mu"] = torch.tensor(mu, dtype=dtype)
        data["beta_mu"] = torch.tensor(beta * mu, dtype=dtype)
    if (
        rho is None
        or V_ext is None
        or thermal_wavelength is None
        or not np.all(rho[~mask] > 0.0)
    ):
        return
    safe_rho = rho.copy()
    safe_rho[mask] = 1.0
    c1_plus_beta_mu = (
        np.log(safe_rho * thermal_wavelength[None, :] ** 3) + beta * V_ext
    )
    c1_plus_beta_mu[mask] = 0.0
    data["c1_plus_beta_mu"] = torch.tensor(
        c1_plus_beta_mu,
        dtype=dtype,
    )
    if mu is not None:
        c1 = c1_plus_beta_mu - beta * mu[None, :]
        c1[mask] = 0.0
        data["c1"] = torch.tensor(
            c1,
            dtype=dtype,
        )
