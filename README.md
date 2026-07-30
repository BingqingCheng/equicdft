# cace_grid

`cace_grid` develops Cartesian symmetry-adapted models for classical
density-functional theory on periodic three-dimensional grids.

One `GridData` object represents one complete density configuration and
contains the grid fields together with periodic neighborhood indices for every
grid point.

Compute one mean density per frame before constructing the model:

```bash
python scripts/mean_density.py density.extxyz
```

The script prints a JSON list. Select the training-frame entries and average
them to obtain the scalar `mean_density` used below.

The same calculation can be called from a training script:

```python
from scripts.mean_density import compute_mean_densities

frame_mean_densities = compute_mean_densities(
    "density.extxyz",
    index="0:50",  # training frames only
)
mean_density = sum(frame_mean_densities) / len(frame_mean_densities)
```

```python
from cace_grid import (
    CartesianAFeatures,
    CartesianBFeatures,
    GridCACEModel,
    GridData,
    LocalFreeEnergyReadout,
)

dataset = GridData.from_xyz(
    "density.extxyz",
    cutoff_grid=3,
    boltzmann_constant=1.0,  # reduced units; default is eV/K
)
data = dataset[0]
n_types = int(data["n_types"].item())
mean_density = 0.7  # precomputed from the training frames

a_features = CartesianAFeatures(
    mean_density=mean_density,
    cutoff_grid=3,
    max_power=4,
    n_radial_channels=4,
    trainable_radial_exponents=False,
    n_types=n_types,
)
b_features = CartesianBFeatures(max_power=4, max_product_order=3)
readout = LocalFreeEnergyReadout(
    n_types=n_types,
)
model = GridCACEModel(
    a_features=a_features,
    b_features=b_features,
    readout=readout,
    compute_c1=True,
)

outputs = model(data)
beta_F_exc = outputs["beta_F_exc"]
c1 = outputs["c1"]
```

`GridData` stores `beta = 1 / (k_B T)` and constructs the equilibrium
reference field `c1_plus_beta_mu = log(rho * thermal_wavelength**3) +
beta * V_ext`. When `mu` is present, it also stores `c1 =
c1_plus_beta_mu - beta * mu`. The default Boltzmann constant is
`8.617333262e-5` eV/K and the default thermal wavelength is one. Pass
`boltzmann_constant=1.0` for reduced-unit data.

For a multicomponent density field, an optional learned, bias-free channel map
can mix the physical component channels of `A` before symmetrization:

```python
a_features = CartesianAFeatures(
    mean_density=mean_density,
    cutoff_grid=3,
    max_power=4,
    n_radial_channels=4,
    n_types=n_types,
    n_channels=4,
)
```

`CartesianAFeatures` applies the same mixing matrix over all grid points,
radial channels, and Cartesian components, so it commutes with cubic rotations
and reflections. The nonlinear products in `B` then contain cross-component
correlations. For a one-component field, leave `n_channels=None`; no mixing
module or mixing parameters are created.

A fine regular grid can be block-averaged in memory before its local
environments are constructed:

```python
dataset = GridData.from_xyz(
    "density-grid-0.25.extxyz",
    cutoff_grid=3,
    target_grid_spacing=0.5,
)
```
