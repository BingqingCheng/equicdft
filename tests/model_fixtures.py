"""Small model and field builders shared by the serialization tests.

Each builder returns a fresh module; tests materialize lazy readouts with one
forward pass before describing or saving the model.
"""

import numpy as np
import torch

from equicdft import (
    BChiMessage,
    BulkReadout,
    CartesianAFeatures,
    CartesianBFeatures,
    GGAReadout,
    GridCACEModel,
    LDAReadout,
    LocalReadout,
    LongRangeReadout,
    PairwiseReadout,
    ReciprocalFeatures,
)
from equicdft.stencil import get_neighbor_indices


def grid_data(
    shape=(6, 6, 6),
    cutoff_grid=1,
    n_types=1,
    seed=0,
    grid_spacing=1.0,
):
    generator = torch.Generator().manual_seed(seed)
    positions = np.indices(shape, dtype=int).reshape(3, -1).T
    neighbor_indices, _ = get_neighbor_indices(
        positions,
        cutoff_grid=cutoff_grid,
    )
    n_grid = int(np.prod(shape))
    return {
        "rho": torch.rand(n_grid, n_types, generator=generator) + 0.1,
        "V_ext": torch.rand(n_grid, n_types, generator=generator),
        "grid_positions": torch.tensor(positions, dtype=torch.long),
        "local_density_index": torch.tensor(
            neighbor_indices,
            dtype=torch.long,
        ),
        "grid_spacing": torch.full((3,), float(grid_spacing)),
        "grid_size": torch.tensor(shape),
        "temperature": torch.tensor(1.5),
        "beta": torch.tensor(1.0 / 1.5),
    }


def example_model(**overrides):
    """Build the compact LDA + invariant model used by the LJ example."""

    n_types = overrides.pop("n_types", 1)
    separate_center = overrides.pop("separate_center", True)
    a_features = CartesianAFeatures(
        mean_density=0.4,
        cutoff_grid=1,
        max_power=2,
        radial_basis="none",
        n_radial_channels=1,
        separate_center=separate_center,
        n_types=n_types,
    )
    b_features = CartesianBFeatures(max_power=2, max_product_order=2)
    readouts = [
        LDAReadout(mean_density=0.4, n_types=n_types, hidden_sizes=(4,)),
        LocalReadout(n_types=n_types, hidden_sizes=(4,)),
    ]
    arguments = dict(
        a_features=a_features,
        b_features=b_features,
        readout=readouts,
        grid_spacing=1.0,
        mean_temperature=1.2,
        boltzmann_constant=1.0,
        thermal_wavelength=1.0,
        compute_c1=True,
        compute_local_mu=True,
        rho_min=1.0e-3,
    )
    arguments.update(overrides)
    return GridCACEModel(**arguments)


def gaussian_message_model(
    radial_basis="gaussian",
    backend="gather",
    radial_transform=True,
):
    a_features = CartesianAFeatures(
        mean_density=0.7,
        cutoff_grid=1,
        max_power=1,
        radial_basis="gaussian",
        radial_exponents=(0.125, 0.5),
        radial_centers=(0.0, 1.0),
        trainable_radial_exponents=True,
        n_radial_channels=2 if radial_transform else None,
        n_types=2,
        n_channels=2,
        density_transform=((0.5, 0.5), (-0.5, 0.5)),
        trainable_density_transform=False,
        convolution_backend=backend,
    )
    b_features = CartesianBFeatures(max_power=1, max_product_order=2)
    if radial_basis == "gaussian":
        message = BChiMessage(
            n_invariant_features=b_features.n_features,
            n_radial_channels=a_features.n_radial_channels,
            n_channels=a_features.n_output_channels,
            hidden_sizes=(4,),
            radial_exponents=(0.25, 0.75),
            trainable_radial_exponents=True,
            radial_centers=(0.0, 0.5),
            trainable_radial_centers=True,
            convolution_backend=backend,
        )
    elif radial_basis == "shared":
        message = BChiMessage(
            n_invariant_features=b_features.n_features,
            n_radial_channels=a_features.n_radial_channels,
            n_channels=a_features.n_output_channels,
            hidden_sizes=(4,),
            convolution_backend=backend,
        )
    else:
        raise ValueError(radial_basis)
    return GridCACEModel(
        a_features=a_features,
        b_features=b_features,
        readout=[LocalReadout(n_types=2, hidden_sizes=(4,))],
        grid_spacing=(0.5, 0.5, 0.5),
        mean_temperature=1.0,
        message_layers=[message],
        free_energy_mode="physical",
    )


def bessel_message_model():
    a_features = CartesianAFeatures(
        mean_density=0.5,
        cutoff_grid=2,
        max_power=1,
        radial_basis="bessel",
        n_radial_functions=3,
        n_radial_channels=2,
        n_types=1,
    )
    b_features = CartesianBFeatures(max_power=1, max_product_order=2)
    message = BChiMessage(
        n_invariant_features=b_features.n_features,
        n_radial_channels=2,
        n_channels=1,
        hidden_sizes=(4,),
        radial_basis="bessel",
        n_radial_functions=3,
    )
    return GridCACEModel(
        a_features=a_features,
        b_features=b_features,
        readout=[
            LocalReadout(n_types=1, hidden_sizes=(4,)),
            GGAReadout(hidden_sizes=(3,), n_types=1),
        ],
        grid_spacing=1.0,
        message_layers=[message],
    )


def long_range_model(charges=None, coulomb_amplitude=None):
    reciprocal = ReciprocalFeatures(
        kernel="coulomb",
        radial_exponents=(0.3,),
        n_types=2,
    )
    readouts = [
        LDAReadout(mean_density=0.4, n_types=2, hidden_sizes=(3,)),
        BulkReadout(n_types=2, hidden_sizes=(3,), zero_init=False),
        PairwiseReadout(cutoff_grid=2, n_types=2, hidden_sizes=(3,)),
        LongRangeReadout(
            n_kernels=1,
            n_types=2,
            hidden_sizes=(3,),
            charges=charges,
            coulomb_amplitude=coulomb_amplitude,
            features=reciprocal,
        ),
    ]
    return GridCACEModel(
        a_features=None,
        b_features=None,
        readout=readouts,
        grid_spacing=0.5,
        mean_temperature=1.3,
    )
