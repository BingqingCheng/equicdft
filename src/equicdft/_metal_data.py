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
) -> Dict[str, Any]:
    """Validate metal groups without changing labels, charge order, or charge.

    Group IDs and totals have shape ``[n_groups]`` or ``[..., n_groups]``.
    Single-group scalar metadata is accepted. Unbatched metadata may be
    shared by a batch, but each frame must contain exactly its declared IDs.
    Charge totals are in elementary-charge units, explicitly labeled ``e``.
    """

    if metal_mask is None:
        if any(value is not None for value in (
            metal_group_ids, metal_total_charge, metal_charge_units,
        )):
            raise ValueError("metal metadata requires metal_mask")
        return {}

    mask = metal_mask_tensor(metal_mask, n_grid, device)
    leading = tuple(mask.shape[:-1]) if batch_shape is None else tuple(batch_shape)
    if mask.shape[:-1] not in ((), leading):
        raise ValueError("metal_mask batch shape does not match the field")
    mask = mask.expand(*leading, n_grid)
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

    n_fields = mask.numel() // n_grid
    for field_mask, field_ids in zip(
        mask.reshape(n_fields, n_grid), ids.reshape(n_fields, n_groups),
    ):
        unique_ids = torch.unique(field_ids)
        if unique_ids.numel() != n_groups:
            raise ValueError("metal_group_ids must be unique in every field")
        present = torch.unique(field_mask[field_mask >= 0])
        if not torch.equal(present, unique_ids):
            raise ValueError(
                "metal_group_ids must exactly cover the groups in metal_mask"
            )
    return {
        "metal_mask": mask,
        "metal_group_ids": ids,
        "metal_total_charge": totals,
        "metal_charge_units": "e",
    }
