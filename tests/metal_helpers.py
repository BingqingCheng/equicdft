"""Configure modules from the independent electrostatics oracle case records.

The reference cases keep electrode inputs beside liquid fields for their
independent Fourier calculations. This adapter configures MetalWall from those
records, then passes only liquid/grid state to production code. Public API
tests do not use it.
"""

import torch
from equicdft import LongRangeReadout, MetalWall, ReciprocalFeatures
from equicdft.data.metal import normalize_metal_sites
from equicdft.functional.base import ReadoutContext
from equicdft.functional.metalwall import _field_vector


METAL_SITE_KEYS = (
    "metal_positions", "metal_site_groups", "metal_group_ids",
    "metal_total_charge", "metal_charge_units",
)
METAL_FIELD_KEYS = ("metal_external_field",)


def liquid_coulomb(charges=(1., -1.), amplitude=1.7, exponent=.2):
    """One fixed charge-factorized Coulomb readout for metal test fixtures."""
    readout = LongRangeReadout(
        n_kernels=1, n_types=len(charges), charges=charges,
        coulomb_amplitude=amplitude,
        features=ReciprocalFeatures(
            kernel="coulomb", n_types=len(charges),
            radial_exponents=(exponent,),
        ),
    )
    # Oracle comparisons use exact float64 literals rather than a float32
    # construction followed by dtype promotion.
    readout.coulomb_amplitude = torch.tensor(amplitude, dtype=torch.float64)
    return readout


def metal_sites(data):
    return {key: data[key] for key in METAL_SITE_KEYS}


def liquid_data(data):
    return {
        key: value for key, value in data.items()
        if key not in METAL_SITE_KEYS + METAL_FIELD_KEYS
    }


def _run(call, data, *args, **kwargs):
    if isinstance(call, torch.nn.Module):
        module = call
    else:
        owner = call.__self__
        module = getattr(owner, "model", owner)
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
            value = data.get("metal_external_field", (0., 0., 0.))
            field = _field_vector(value, "external_field").to(
                wall.liquid_coulomb.charges
            )
            if torch.any(field != 0).item() and wall.metal_group_ids.numel() != 1:
                raise ValueError(
                    "a nonzero external_field currently requires exactly one "
                    "electrode group"
                )
            wall.external_field = field
    clean = liquid_data(data)
    if isinstance(module, MetalWall):
        return call(wall_context(clean), *args, **kwargs)
    return call(clean, *args, **kwargs)


def wall_context(data):
    """Build the readout context a model would hand to a standalone wall.

    The unit contract is the reduced one of the oracle cases: ``beta`` from
    the data (default one), unit reference energy, and ``T / T_ref = 1``
    while keeping the temperature's differentiation graph.
    """

    rho = data["rho"]
    scalar_dtype = rho.dtype if rho.is_floating_point() else torch.float64
    temperature = torch.as_tensor(
        data.get("temperature", 1.0), dtype=scalar_dtype, device=rho.device,
    )
    normalized = temperature / temperature.detach()
    return ReadoutContext(
        rho=rho,
        normalized_temperature=normalized.expand(rho.shape[:-2]),
        voxel_volume=torch.as_tensor(
            data["grid_spacing"], dtype=scalar_dtype, device=rho.device,
        ).prod(),
        grid_spacing=torch.as_tensor(
            data["grid_spacing"], dtype=scalar_dtype, device=rho.device,
        ),
        grid_size=data.get("grid_size"),
        beta=torch.as_tensor(
            data.get("beta", 1.), dtype=scalar_dtype, device=rho.device,
        ),
        reference_energy=torch.ones((), dtype=scalar_dtype, device=rho.device),
        free_energy_mode="beta",
        dipole_density=data.get("dipole_density"),
        grid_center=data.get("grid_center"),
        excluded_mask=data.get("excluded_mask"),
    )
