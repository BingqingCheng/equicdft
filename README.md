# cace_grid

`cace_grid` develops Cartesian atomic-cluster-expansion models for classical
density-functional theory on periodic three-dimensional grids.

The initial data interface follows CACE's `AtomicData` pattern. One `GridData`
object represents one complete density configuration and contains the grid
fields together with a periodic local-density environment for every grid point.

```python
from cace_grid import CartesianAFeatures, CartesianBFeatures, GridData

dataset = GridData.from_xyz("density.extxyz", cutoff_grid=3)
data = dataset[0]

a_features = CartesianAFeatures(
    cutoff_grid=3,
    max_power=4,
    n_alphas=4,
    trainable_alphas=False,
)
A = a_features(data)
b_features = CartesianBFeatures(max_power=4, max_nu=3)
B = b_features(A)

print(data["rho"].shape)
print(data["local_density"].shape)
print(data["local_density_positions"].shape)
print(A.shape)
print(B.shape)
```

For a multicomponent density field, an optional learned, bias-free channel map
can mix the physical component channels of `A` before symmetrization:

```python
from cace_grid import AChannelMixing

channel_mixing = AChannelMixing(
    n_types=data["n_types"].item(),
    n_channels=4,
)
A_mixed = channel_mixing(A)
B = b_features(A_mixed)
```

The same mixing matrix is shared over grid points, radial channels, and
Cartesian components, so it commutes with cubic rotations and reflections.
The nonlinear products in `B` then contain cross-component correlations. For a
one-component system, pass `A` directly to `CartesianBFeatures` as above.

A fine regular grid can be block-averaged in memory before its local
environments are constructed:

```python
dataset = GridData.from_xyz(
    "density-grid-0.25.extxyz",
    cutoff_grid=3,
    target_grid_spacing=0.5,
)
```
