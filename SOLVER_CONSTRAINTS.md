# Homogeneous density constraints

`GridSolver.solve(..., method="minimize", homogeneous_axes=["x", "y"])`
minimizes over densities constant along x and y (only z dependence remains).

```python
solver.solve(data, method="minimize", particle_numbers=N, homogeneous_axes="x")
solver.solve(data, method="minimize", particle_numbers=N, homogeneous_axes="y")
solver.solve(data, method="minimize", particle_numbers=N, homogeneous_axes=["x", "y"])
solver.solve(data, method="minimize", particle_numbers=N, homogeneous_axes=["x", "z"])
solver.solve(data, method="minimize", particle_numbers=N, homogeneous_axes=["x", "y", "z"])
```

Each named axis is a direction along which density is **constant**, not a
direction in which it may vary. Names map to grid axes x=0, y=1, z=2, not
species or metal coordinates. A single lowercase name or a sequence of distinct
names is accepted; None/empty list/tuple means unrestricted. Invalid or duplicate
names are rejected. The original `homogeneous_axes=(0, 1)` integer form remains
supported; names and indices can be mixed, for example `["x", 1]`. Duplicate
axes after translation (such as `["x", 0]`) are rejected. There is no separate
`restrict` keyword.
`solver_homogeneous_axes` continues to report integer indices, for compatibility.

The option is general across component counts, axis choices, canonical and
grand-canonical ensembles. It currently applies only to minimization.

The complete rectangular grid is identified by grid_size and integer
grid_positions, including arbitrary row order. Accessibility must be constant
along the constrained axes; incompatible masks are rejected, not averaged or
weakened. Projection reorders rows into the rectangular grid, averages selected
axes with broadcasting, and restores the original row order. Initial densities
are averaged; c1 is averaged at every update and fixed V_ext is averaged once
before the minimization loop to construct the constrained Euler gradient.
Positivity, exact fixed particle numbers and per-species upper bounds use the
existing projection. Armijo still checks the full, unmodified objective.

The model and metal-site geometry are not reduced to one dimension. Electrode
charges relax in 3D on every evaluation. Only liquid-density variations are
restricted. This can give a constrained stationary density while a lateral
drive remains in the unrestricted problem.

With nonempty homogeneous_axes, existing residual/converged keys describe the
constrained problem. Additional full_euler_lagrange_residual,
full_max_euler_lagrange_residual, full_rms_euler_lagrange_residual and
full_converged preserve unrestricted diagnostics. solver_homogeneous_axes
records the constraint. An opt-in constrained result must not be labeled a
full-dimensional equilibrium solely because converged is true.

For spatially varying external fields the correct ideal-gas reference is
rho proportional to exp(-beta*average(V_ext)), not average(exp(-beta*V_ext)).
Independent constant species shifts remain permitted canonical gauges.

Tests cover analytic multi-component and grand-canonical solutions, arbitrary
row order/axes, bounds and gauges, invalid masks/coordinates, default
compatibility, and a relaxed-electrode directional derivative. This is a
general software capability; choosing planar liquid density for an atomically
or grid-discrete metal interface is an exploratory application approximation.

Names and indices select the same projection. Reshape/mean averaging can differ
from the earlier grouped sums at floating-point roundoff. Existing fixed-number
roundoff correction can introduce tiny
single-voxel symmetry deviations; this interface update does not fix that
separate numerical issue or change the declared EDL acceptance thresholds.
