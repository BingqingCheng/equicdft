#!/usr/bin/env python3
"""Evolve the retained LJ model for ten deterministic steps from a sine wave."""

import argparse
import hashlib
from pathlib import Path

import torch

from equicdft import GridData
from equicdft.conversion import load_converted_model
from equicdft.evolution import (
    ConstantMobility, Dynamics, EulerIntegrator, SemiImplicitIntegrator,
)


MODEL_PATH = Path(__file__).resolve().parents[2] / "tests/regression/lj_paper_v1/model.pt"

# LJ reduced units. Mobility and timestep are illustrative, not MD-calibrated.
GRID_SIZE = (16, 16, 16)
TEMPERATURE = 1.5
MEAN_DENSITY = 0.4
AMPLITUDE = 0.05
MOBILITY = 0.01
DT = 0.01
N_STEPS = 10


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("euler", "semi-implicit"), default="semi-implicit")
    parser.add_argument("--output", type=Path, default=Path("lj-evolution.pt"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new --output path; existing trajectories are not overwritten.")

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    model = load_converted_model(MODEL_PATH).double().eval()
    data = GridData.from_dict(
        {"grid_size": GRID_SIZE, "temperature": TEMPERATURE},
        grid_info=model.grid_info,
        include_local_density_index=model.requires_local_density_index,
    )
    data["grid_center"] = (data["grid_positions"].double() + 0.5) * data["grid_spacing"]
    length_x = GRID_SIZE[0] * data["grid_spacing"][0]
    phase = 2 * torch.pi * data["grid_center"][:, 0] / length_x
    data["rho"] = (MEAN_DENSITY + AMPLITUDE * phase.sin())[:, None]
    data["V_ext"] = torch.zeros_like(data["rho"])

    method = {"euler": EulerIntegrator, "semi-implicit": SemiImplicitIntegrator}[args.method]
    integrator = method(model, ConstantMobility(MOBILITY), dt=DT)
    rates = integrator.relaxation_rates(data)
    print("explicit stability limit {:.3f}; slowest linear relaxation time {:.1f}; run covers {:.1%} of it".format(
        integrator.stability_limit(data), 1 / float(rates.flatten()[1:].min()),
        DT * N_STEPS * float(rates.flatten()[1:].min())))
    trajectory = Dynamics(integrator).run(data, n_steps=N_STEPS, save_every=1)
    trajectory["metadata"].update(
        model_path=str(MODEL_PATH),
        model_sha256=hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        initialization={"mean_density": MEAN_DENSITY, "amplitude": AMPLITUDE,
                        "formula": "rho(x) = mean_density + amplitude*sin(2*pi*x/Lx)"},
    )
    Dynamics.save(trajectory, args.output)

    print("step    time           beta_A      particle_number      min_density")
    for step, (time, energy, number, minimum) in enumerate(zip(
        trajectory["time"], trajectory["beta_A"],
        trajectory["particle_numbers"][:, 0], trajectory["min_density"],
    )):
        print("{:4d}  {:6.3f}  {:15.8f}  {:19.10f}  {:15.10f}".format(
            step, float(time), float(energy), float(number), float(minimum),
        ))
    print("Saved {} frames to {}".format(len(trajectory["time"]), args.output.resolve()))


if __name__ == "__main__":
    main()
