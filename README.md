# cace_grid

`cace_grid` develops Cartesian atomic-cluster-expansion models for classical
density-functional theory on periodic three-dimensional grids.

The initial data interface follows CACE's `AtomicData` pattern. One `GridData`
object represents one complete density configuration and contains the grid
fields together with a periodic local-density environment for every grid point.

```python
from cace_grid import GridData

dataset = GridData.from_xyz("density.extxyz", cutoff=2.0)
data = dataset[0]

print(data["rho"].shape)
print(data["local_density"].shape)
print(data["local_density_positions"].shape)
```

A fine regular grid can be block-averaged in memory before its local
environments are constructed:

```python
dataset = GridData.from_xyz(
    "density-grid-0.25.extxyz",
    cutoff=2.0,
    target_grid_spacing=0.5,
)
```
