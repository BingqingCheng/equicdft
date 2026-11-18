#!/usr/bin/env python3
"""Evaluate beta*F_exc and c1 on the first EXTXYZ density field.

The input must already be on --spacing; no resampling or solve is performed.
Use a versioned save_model archive, not an optimizer/training checkpoint.
"""

import argparse
from pathlib import Path

import torch

from equicdft import GridData, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("field", type=Path)
    parser.add_argument("--spacing", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--center-normalization", choices=("strict", "reference"), default="strict")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new --output; existing files are not overwritten.")

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    model = load_model(args.model, map_location="cpu").double().eval().requires_grad_(False)
    model = model.regrid(
        args.spacing,
        center_normalization=args.center_normalization,
        convolution_backend="fft",
    ).to(args.device)

    field = GridData.from_xyz(
        args.field, index="0", grid_info=model.grid_info,
        include_local_density_index=False,
    )[0]
    field = {key: value.to(args.device) for key, value in field.items()}
    result = model(
        field, compute_c1=True, compute_c2=False,
        compute_local_mu=False, compute_ideal=False,
    )

    # c1 = -(1/Delta V) * d(beta*F_exc)/d(rho); shape [n_grid, n_types].
    saved = {key: result[key].detach().cpu() for key in ("beta_F_exc", "c1")}
    for key in ("grid_positions", "grid_size", "grid_spacing", "temperature"):
        saved[key] = field[key].cpu()
    saved.update(model_path=str(args.model.resolve()), field_path=str(args.field.resolve()),
                 frame_index=0, center_normalization=args.center_normalization)
    with args.output.open("xb") as handle:
        torch.save(saved, handle)
    print("beta F_exc = {:.10g}".format(saved["beta_F_exc"].item()))
    print("Saved c1 and grid metadata to", args.output.resolve())


if __name__ == "__main__":
    main()
