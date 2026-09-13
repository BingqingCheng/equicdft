"""Evaluate an untrained scalar/dipole-density functional on a synthetic field.

Run from the repository root: PYTHONPATH=src python examples/polarization_density/example.py
This checks the API and gradient path; it is not a polar-fluid training fit.
"""

import torch

from equicdft import (
    GridCACEModel, GridData, LocalReadout,
    PolarizationAFeatures, PolarizationBFeatures,
)


def main():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(7)
    a_features = PolarizationAFeatures(
        mean_density=0.7,
        dipole_density_scale=0.1,  # dipole moment per volume, in the data's units
        cutoff_grid=1,
        max_power=2,
        radial_exponents=(0.125,),
        trainable_radial_exponents=True,
    )
    b_features = PolarizationBFeatures(a_features, max_product_order=3)
    model = GridCACEModel(
        a_features=a_features,
        b_features=b_features,
        readout=[LocalReadout(n_features=b_features.n_features + 1, hidden_sizes=(16, 16))],
        grid_spacing=0.5,
        boltzmann_constant=1.0,
        compute_c1=True,
        compute_polarization_derivative=True,
    )
    # Orthorhombic cells are allowed; each voxel must be a cube.
    data = GridData.from_dict(
        {"grid_size": [4, 4, 6], "temperature": 1.5},
        grid_info=model.grid_info,
    )
    n_grid = data["grid_positions"].shape[0]
    data["rho"] = 0.7 + 0.05 * torch.rand(n_grid, 1)
    data["dipole_density"] = 0.01 * torch.randn(n_grid, 1, 3)

    outputs = model(data)
    for name in ("beta_F_exc", "c1", "polarization_derivative"):
        assert torch.isfinite(outputs[name]).all()
        print("{}: shape {}".format(name, tuple(outputs[name].shape)))
    # Both responses remain differentiable with respect to model parameters
    # in training mode. Physical targets/losses require an ideal orientational
    # functional and the external electric-field convention first.
    outputs["polarization_derivative"].square().sum().backward()
    # A constant per-particle offset disappears from the P derivative, so
    # parameters controlling only that offset need not receive a gradient.
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    print("Finite parameter gradients through the polarization derivative.")


if __name__ == "__main__":
    main()
