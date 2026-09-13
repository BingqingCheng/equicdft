"""Validation for one fixed set of electrode sites and charge constraints."""

from typing import Any, Dict, Mapping

import torch


METAL_SITE_KEYS = frozenset({
    "metal_positions", "metal_site_groups", "metal_group_ids",
    "metal_total_charge", "metal_charge_units",
})
METAL_DATA_KEYS = METAL_SITE_KEYS | {
    "metal_external_field", "metal_field_origin",
}


def _integer_tensor(value: Any, name: str, device=None) -> torch.Tensor:
    if torch.is_tensor(value) and value.requires_grad:
        raise ValueError(name + " must be fixed")
    values = torch.as_tensor(value, device=device)
    if values.dtype == torch.bool or values.is_complex():
        raise ValueError(name + " must contain integers")
    if values.is_floating_point() and (
        not torch.all(torch.isfinite(values)).item()
        or not torch.all(values == values.round()).item()
    ):
        raise ValueError(name + " must contain finite integers")
    integers = values.to(dtype=torch.long)
    if not torch.equal(integers.to(values.dtype), values):
        raise ValueError(name + " must contain representable int64 integers")
    return integers


def normalize_metal_sites(
    metal_group_ids: Any,
    metal_total_charge: Any,
    metal_charge_units: Any,
    *,
    metal_positions: Any,
    metal_site_groups: Any,
    dtype=torch.float64,
    device=None,
) -> Dict[str, Any]:
    """Return validated tensors for one immutable electrode geometry."""

    values = {
        "metal_positions": metal_positions,
        "metal_site_groups": metal_site_groups,
        "metal_group_ids": metal_group_ids,
        "metal_total_charge": metal_total_charge,
        "metal_charge_units": metal_charge_units,
    }
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise ValueError("metal sites require: " + ", ".join(missing))

    if torch.is_tensor(metal_positions) and metal_positions.requires_grad:
        raise ValueError("metal_positions must be fixed geometry")
    raw_positions = torch.as_tensor(metal_positions, device=device)
    if raw_positions.dtype == torch.bool or raw_positions.is_complex():
        raise ValueError("metal_positions must contain real finite coordinates")
    positions = torch.as_tensor(
        metal_positions, dtype=torch.float64, device=device,
    )
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
        raise ValueError("metal_positions must have shape [n_sites, 3] with n_sites > 0")
    if not torch.all(torch.isfinite(positions)).item():
        raise ValueError("metal_positions must contain real finite coordinates")

    labels = _integer_tensor(metal_site_groups, "metal_site_groups", device)
    if labels.shape != (positions.shape[0],):
        raise ValueError("metal_site_groups must have shape [n_sites]")
    if torch.any(labels < 0).item():
        raise ValueError("metal_site_groups must be nonnegative")

    if metal_charge_units != "e":
        raise ValueError("metal_charge_units must be 'e'")
    ids = _integer_tensor(metal_group_ids, "metal_group_ids", device)
    if ids.ndim == 0:
        ids = ids.reshape(1)
    if ids.ndim != 1 or ids.numel() == 0:
        raise ValueError("metal_group_ids must be a nonempty one-dimensional array")
    if torch.any(ids < 0).item():
        raise ValueError("metal_group_ids must be nonnegative")
    if torch.unique(ids).numel() != ids.numel():
        raise ValueError("metal_group_ids must be unique")

    if torch.is_tensor(metal_total_charge) and metal_total_charge.requires_grad:
        raise ValueError("metal_total_charge must be fixed")
    raw_totals = torch.as_tensor(metal_total_charge, device=device)
    if raw_totals.dtype == torch.bool or raw_totals.is_complex():
        raise ValueError("metal_total_charge must contain real finite values")
    totals = torch.as_tensor(metal_total_charge, dtype=dtype, device=device)
    if totals.ndim == 0:
        totals = totals.reshape(1)
    if totals.shape != ids.shape:
        raise ValueError("metal_total_charge must have one value per metal_group_id")
    if not torch.all(torch.isfinite(totals)).item():
        raise ValueError("metal_total_charge must contain real finite values")
    if not torch.equal(torch.unique(labels), torch.sort(ids).values):
        raise ValueError("metal_group_ids must exactly cover the groups in metal geometry")

    return {
        "metal_positions": positions,
        "metal_site_groups": labels,
        "metal_group_ids": ids,
        "metal_total_charge": totals,
        "metal_charge_units": "e",
    }


def normalize_metal_site_mapping(
    metal_sites: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate the complete mapping accepted by :class:`MetalWall`."""

    if not isinstance(metal_sites, Mapping):
        raise TypeError("metal_sites must be a mapping")
    missing = METAL_SITE_KEYS - metal_sites.keys()
    unknown = metal_sites.keys() - METAL_SITE_KEYS
    if missing:
        raise ValueError(
            "metal_sites is missing: " + ", ".join(sorted(missing))
        )
    if unknown:
        raise ValueError(
            "unknown metal_sites entries: " + ", ".join(sorted(unknown))
        )
    return normalize_metal_sites(
        metal_sites["metal_group_ids"],
        metal_sites["metal_total_charge"],
        metal_sites["metal_charge_units"],
        metal_positions=metal_sites["metal_positions"],
        metal_site_groups=metal_sites["metal_site_groups"],
    )
