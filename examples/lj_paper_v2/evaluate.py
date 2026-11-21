"""Evaluate the revised-paper LJ functional on one held-out density field."""

import argparse
import json
from pathlib import Path

import torch

from equicdft import GridData, load_model
from equicdft.functional.thermodynamics import beta_external_potential


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    model = load_model(HERE / "model.pt").to(args.device).eval().requires_grad_(False)
    field = GridData.from_xyz(
        HERE / "field.extxyz", index="0", grid_info=model.grid_info,
        include_local_density_index=model.requires_local_density_index,
    )[0]
    field = {key: value.to(args.device) for key, value in field.items()}
    output = model(
        field, compute_c1=True, compute_c2=False,
        compute_ideal=False, compute_local_mu=False,
    )

    # Canonical data determine V_ext only up to a spatially constant offset.
    rho = field["rho"].detach()
    active = rho > 1e-3
    raw = beta_external_potential(rho, output["c1"].detach(), model.thermal_wavelength)
    target = field["V_ext"] / field["temperature"]  # LJ convention: k_B = 1.
    offset = (target[active] - raw[active]).mean()
    rmse = (raw[active] + offset - target[active]).square().mean().sqrt()
    if not torch.isfinite(output["c1"]).all() or not torch.isfinite(rmse):
        raise RuntimeError("Nonfinite model prediction")
    print(json.dumps({
        "beta_F_exc": float(output["beta_F_exc"].detach()),
        "c1_shape": list(output["c1"].shape),
        "particle_number": float(rho.sum() * field["grid_spacing"].prod()),
        "gauge_aligned_beta_V_RMSE": float(rmse),
    }, indent=2))


if __name__ == "__main__":
    main()
