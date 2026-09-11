"""Small shared validation helpers for fixed-charge metal group metadata."""

from typing import Any, Dict, Optional, Sequence

import torch


def normalize_metal_field(field=None, origin=None, *, dtype=None, device=None,
                          batch_shape=()) -> Dict[str, torch.Tensor]:
    """Optional metal-only field and wrapping origin, shared or per frame.

    Vectors are Cartesian [3], in energy/(e*length) and length respectively.
    The origin is relative to grid index zero; it defaults to that grid point.
    """
    if field is None:
        if origin is not None:
            raise ValueError("metal_field_origin requires metal_external_field")
        return {}
    leading = tuple(batch_shape)
    result = {}
    for name, value in (("metal_external_field", field),
                        ("metal_field_origin", (0.0, 0.0, 0.0) if origin is None else origin)):
        raw = torch.as_tensor(value, device=device)
        if raw.dtype == torch.bool or raw.is_complex():
            raise ValueError(name + " must contain real finite values")
        vector = torch.as_tensor(value, dtype=dtype or torch.get_default_dtype(),
                                 device=device)
        if vector.shape not in ((3,), (*leading, 3)):
            raise ValueError(name + " must have shape [3] or [..., 3] matching the field")
        if not torch.all(torch.isfinite(vector)).item():
            raise ValueError(name + " must contain real finite values")
        result[name] = vector.expand(*leading, 3)
    return result


def _integer_tensor(value: Any, name: str, device=None) -> torch.Tensor:
    values = torch.as_tensor(value, device=device)
    if values.dtype == torch.bool or values.is_complex():
        raise ValueError("{} must contain integers".format(name))
    if values.is_floating_point() and (
        not torch.all(torch.isfinite(values)).item()
        or not torch.all(values == values.round()).item()
    ):
        raise ValueError("{} must contain finite integers".format(name))
    integers = values.to(dtype=torch.long)
    if not torch.equal(integers.to(values.dtype), values):
        raise ValueError("{} must contain representable int64 integers".format(name))
    return integers


def metal_mask_tensor(value: Any, n_grid: int, device=None) -> torch.Tensor:
    """Return integer group labels; every negative label means nonmetal."""

    mask = _integer_tensor(value, "metal_mask", device)
    if mask.ndim < 1 or mask.shape[-1] != n_grid:
        raise ValueError("metal_mask must have shape [..., n_grid]")
    return mask


def normalize_metal_metadata(
    metal_mask: Any,
    metal_group_ids: Any = None,
    metal_total_charge: Any = None,
    metal_charge_units: Any = None,
    *,
    n_grid: int,
    dtype=None,
    device=None,
    batch_shape: Optional[Sequence[int]] = None,
    metal_positions: Any = None,
    metal_site_groups: Any = None,
) -> Dict[str, Any]:
    """Validate metal groups without changing labels, charge order, or charge.

    Group IDs and totals have shape ``[n_groups]`` or ``[..., n_groups]``.
    Single-group scalar metadata is accepted. Unbatched metadata may be
    shared by a batch, but each frame must contain exactly its declared IDs.
    Charge totals are in elementary-charge units, explicitly labeled ``e``.
    Explicit fixed positions have shape ``[n_sites, 3]`` or
    ``[..., n_sites, 3]`` and remain float64, relative to grid index zero.
    Their nonnegative ``metal_site_groups`` replace grid-mask group labels;
    they never imply an accessibility mask. The two representations cannot
    both contain charge sites.
    """

    explicit = metal_positions is not None or metal_site_groups is not None
    geometry = {}
    if explicit:
        if metal_positions is None or metal_site_groups is None:
            raise ValueError("metal_positions and metal_site_groups are required together")
        if torch.is_tensor(metal_positions) and metal_positions.requires_grad:
            raise ValueError("metal_positions must be fixed geometry (requires_grad=False)")
        raw_positions = torch.as_tensor(metal_positions, device=device)
        if raw_positions.dtype == torch.bool or raw_positions.is_complex():
            raise ValueError("metal_positions must contain real finite coordinates")
        # Geometry remains float64 even for a lower-precision density field.
        # Converting an already rounded tensor cannot recover lost precision.
        positions = torch.as_tensor(metal_positions, dtype=torch.float64, device=device)
        if positions.ndim < 2 or positions.shape[-1] != 3 or positions.shape[-2] == 0:
            raise ValueError("metal_positions must have shape [..., n_sites, 3] with n_sites > 0")
        if not torch.all(torch.isfinite(positions)).item():
            raise ValueError("metal_positions must contain real finite coordinates")
        leading = tuple(positions.shape[:-2]) if batch_shape is None else tuple(batch_shape)
        if positions.shape[:-2] not in ((), leading):
            raise ValueError("metal_positions batch shape does not match the field")
        n_sites = positions.shape[-2]
        labels = _integer_tensor(metal_site_groups, "metal_site_groups", device)
        if labels.ndim < 1 or labels.shape[-1] != n_sites:
            raise ValueError("metal_site_groups must have shape [..., n_sites]")
        if labels.shape[:-1] not in ((), leading):
            raise ValueError("metal_site_groups batch shape does not match the field")
        if torch.any(labels < 0).item():
            raise ValueError("metal_site_groups must be nonnegative")
        mask = labels.expand(*leading, n_sites)
        geometry = {"metal_positions": positions.expand(*leading, n_sites, 3),
                    "metal_site_groups": mask}
        if metal_mask is not None:
            grid_mask = metal_mask_tensor(metal_mask, n_grid, device)
            if grid_mask.shape[:-1] not in ((), leading):
                raise ValueError("metal_mask batch shape does not match the field")
            if torch.any(grid_mask >= 0).item():
                raise ValueError("metal_positions cannot coexist with nonnegative metal_mask entries")
            geometry["metal_mask"] = grid_mask.expand(*leading, n_grid)
        label_count = n_sites
        has_metal = True
    elif metal_mask is None:
        if any(value is not None for value in (
            metal_group_ids, metal_total_charge, metal_charge_units,
        )):
            raise ValueError("metal metadata requires metal_mask")
        return {}

    else:
        mask = metal_mask_tensor(metal_mask, n_grid, device)
        leading = tuple(mask.shape[:-1]) if batch_shape is None else tuple(batch_shape)
        if mask.shape[:-1] not in ((), leading):
            raise ValueError("metal_mask batch shape does not match the field")
        mask = mask.expand(*leading, n_grid)
        geometry = {"metal_mask": mask}
        label_count = n_grid
        has_metal = torch.any(mask >= 0).item()
    supplied = (metal_group_ids, metal_total_charge, metal_charge_units)
    if not has_metal and all(value is None for value in supplied):
        return {"metal_mask": mask}
    if any(value is None for value in supplied):
        raise ValueError(
            "metal requires explicit metal_group_ids, metal_total_charge, "
            "and metal_charge_units='e'"
        )
    units = metal_charge_units
    if isinstance(units, (list, tuple)):
        valid_units = bool(units) and all(value == "e" for value in units)
    else:
        valid_units = isinstance(units, str) and units == "e"
    if not valid_units:
        raise ValueError("metal_charge_units must be 'e'")

    ids = _integer_tensor(metal_group_ids, "metal_group_ids", mask.device)
    charge_values = torch.as_tensor(metal_total_charge, device=mask.device)
    if charge_values.dtype == torch.bool or charge_values.is_complex():
        raise ValueError("metal_total_charge must contain real finite values")
    totals = torch.as_tensor(
        metal_total_charge, dtype=dtype or torch.get_default_dtype(),
        device=mask.device,
    )
    if ids.ndim == 0:
        ids = ids.reshape(1)
    if totals.ndim == 0:
        totals = totals.reshape(1)
    if ids.shape[:-1] not in ((), leading):
        raise ValueError("metal_group_ids batch shape does not match the field")
    if totals.shape[:-1] not in ((), leading):
        raise ValueError("metal_total_charge batch shape does not match the field")
    n_groups = ids.shape[-1]
    if totals.shape[-1] != n_groups:
        raise ValueError("metal_total_charge must have one value per metal_group_id")
    ids = ids.expand(*leading, n_groups)
    totals = totals.expand(*leading, n_groups)
    if not torch.all(torch.isfinite(totals)).item():
        raise ValueError("metal_total_charge must contain real finite values")
    if torch.any(ids < 0).item():
        raise ValueError("metal_group_ids must be nonnegative")

    n_fields = mask.numel() // label_count
    for field_mask, field_ids in zip(
        mask.reshape(n_fields, label_count), ids.reshape(n_fields, n_groups),
    ):
        unique_ids = torch.unique(field_ids)
        if unique_ids.numel() != n_groups:
            raise ValueError("metal_group_ids must be unique in every field")
        present = torch.unique(field_mask[field_mask >= 0])
        if not torch.equal(present, unique_ids):
            raise ValueError(
                "metal_group_ids must exactly cover the groups in metal geometry"
            )
    return {
        **geometry,
        "metal_group_ids": ids,
        "metal_total_charge": totals,
        "metal_charge_units": "e",
    }
