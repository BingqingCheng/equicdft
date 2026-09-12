"""Configure modules from the independent electrostatics oracle case records.

The reference cases keep electrode inputs beside liquid fields for their
independent Fourier calculations. This adapter configures MetalWall from those
records, then passes only liquid/grid state to production code. Public API
tests do not use it.
"""

import torch
from equicdft import MetalWall
from equicdft._metal_data import normalize_metal_sites
from equicdft.metalwall import _field_vector


METAL_SITE_KEYS = (
    "metal_positions", "metal_site_groups", "metal_group_ids",
    "metal_total_charge", "metal_charge_units",
)
METAL_FIELD_KEYS = ("metal_external_field", "metal_field_origin")


def metal_sites(data):
    return {key: data[key] for key in METAL_SITE_KEYS}


def liquid_data(data):
    return {
        key: value for key, value in data.items()
        if key not in METAL_SITE_KEYS + METAL_FIELD_KEYS
    }


def _run(call, data, *args, **kwargs):
    module = call if isinstance(call, torch.nn.Module) else call.__self__.model
    for wall in module.modules():
        if isinstance(wall, MetalWall):
            if all(key in data for key in METAL_SITE_KEYS):
                sites = normalize_metal_sites(
                    data["metal_group_ids"], data["metal_total_charge"],
                    data["metal_charge_units"],
                    metal_positions=data["metal_positions"],
                    metal_site_groups=data["metal_site_groups"],
                )
                changed = False
                for name in METAL_SITE_KEYS[:-1]:
                    target = getattr(wall, name)
                    value = sites[name].to(target)
                    if target.shape != value.shape or not torch.equal(target, value):
                        setattr(wall, name, value)
                        changed = True
                if changed:
                    wall._cache_key = wall._cache = None
            for name in ("external_field", "field_origin"):
                value = data.get("metal_" + name, (0., 0., 0.))
                setattr(wall, name, _field_vector(value, name).to(wall.liquid_charges))
    return call(liquid_data(data), *args, **kwargs)
