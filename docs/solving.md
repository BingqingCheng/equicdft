# Equilibrium solves: choosing a method and reading the result

`GridSolver` wraps a model with external fields, ensemble constraints and
numerical updates. This page is the operational companion to the class
docstring: which method to use, what each result key means, and which option to
reach for when a solve does not converge. `GridSolver.solve.__doc__` and
`equicdft.solver.options` remain the exhaustive reference for every field and
its admissible range.

```python
from equicdft import GridSolver
from equicdft.solver import Constraints, ConvergenceOptions

result = GridSolver(model).solve(
    data,
    method="euler",
    constraints=Constraints(particle_numbers=[n_particles]),
    convergence=ConvergenceOptions(max_iter=200, tolerance_residual=1.0e-4),
)
print(result["status"], result["n_iter"], result["maximum_residual"])
```

`solve` requires `V_ext` and either `data["mu"]` or fixed
`constraints.particle_numbers`. `evaluate` requires only `rho` and returns
every quantity the available fields support.

## Which method

| | `method="euler"` | `method="minimize"` |
|---|---|---|
| update | damped Euler--Lagrange fixed point | positivity-preserving mirror descent with an Armijo line search |
| monotone | no | yes: every accepted step lowers the free energy or grand potential |
| default for | scalar models | fixed-dipole models (`FixedDipoleIdeal`) |
| supports `homogeneous_axes` | no | yes, density-only |
| typical use | fast when it converges | when Euler oscillates, or when a monotone objective is needed |

Omitting `method` picks the default for the model. Passing a method-specific
option group with the other method raises, so `euler=EulerOptions(...)` and
`method="minimize"` is an error rather than a silently ignored setting.

## The five option groups

`solve` accepts only these keyword groups, each a frozen dataclass that
validates itself once:

| group | governs |
|---|---|
| `convergence=ConvergenceOptions(...)` | iteration budget and residual tolerances |
| `euler=EulerOptions(...)` | mixing schedule, `fixed_dipole_coordinate`, and optional `anderson=AndersonOptions(...)` |
| `minimize=MinimizeOptions(...)` | step size, Armijo line search, and `fixed_dipole_coordinate` |
| `constraints=Constraints(...)` | ensemble, density and polarization caps, homogeneous axes, fixed field |
| `initial=Initialization(...)` | starting `rho`, `polarization` and `beta_multiplier` |

## The orientational coordinate for fixed dipoles

`fixed_dipole_coordinate` selects which orientational variable a fixed-dipole
solve updates. A scalar model rejects any non-default value rather than
ignoring it, so the option never silently does nothing. `"alignment"` works
in the physical molecular alignment `u = P/(m rho)` and reconstructs
`P = m rho u`, which keeps the update in the variable the saturation bound is
written in.
The alternative is the established behaviour of each method: Euler mixes `P`
directly, and the minimizer steps the conjugate field of the entropy mirror
map. Both choices keep the positivity-preserving canonical log-density step
and the same physical KKT residual, so they are two routes to one stationary
point rather than two problems.

The accepted values differ by group, because the established behaviour they
name differs:

| group | default | alternative |
|---|---|---|
| `EulerOptions` | `"polarization"` | `"alignment"` |
| `MinimizeOptions` | `"conjugate_field"` | `"alignment"` |

A string valid for one group raises `ValueError` on the other, so
`MinimizeOptions(fixed_dipole_coordinate="polarization")` is an error rather
than a synonym for its default. A solve that used anything but its default
reports the choice back as `solver_fixed_dipole_coordinate`, and an alignment
solve also reports `final_relative_alignment_change`, the largest change in
`u` over the last accepted step — the orientational counterpart to
`final_relative_density_change`.

## Reading `status`

Every scalar result carries `status`, `converged`, `n_iter`,
`maximum_residual` and `rms_residual`. The status distinguishes four
outcomes, and each one points at a different option.

**`converged`** — the residual bounds in `ConvergenceOptions` were met.
`maximum_residual` is the largest active-grid Euler residual and must fall
below `tolerance_residual`; `rms_residual` is its density-weighted root mean
square, bounded by `tolerance_rms_residual` when that is supplied.
`residual_density_threshold` drops voxels below a density from both
diagnostics, which keeps statistically unresolved dilute regions from setting
the convergence criterion.

**`stalled`** — the iteration stopped changing before it met the residual
bound, and before the budget ran out. For Euler this means the relative
density change fell below `tolerance_change` while `maximum_residual` was
still above `tolerance_residual`. It usually means the tolerance is tighter
than the fixed point can reach with the current mixing, not that the physics
is wrong:

```python
loose = GridSolver(model).solve(
    data,
    method="euler",
    constraints=Constraints(particle_numbers=[n_particles]),
    convergence=ConvergenceOptions(max_iter=2000, tolerance_residual=1.0e-4),
)

tight = GridSolver(model).solve(
    data,
    method="euler",
    constraints=Constraints(particle_numbers=[n_particles]),
    convergence=ConvergenceOptions(max_iter=2000, tolerance_residual=1.0e-12),
)
```

On one small test field the first converges in 44 iterations at a maximum
residual of 8.9e-05, while the second stalls after 80 with the residual down
to 2.8e-06 and still six orders above the bound it was given. The physics is
the same; only the stopping rule differs.

Loosen `tolerance_residual` to a value the fixed point can reach, or lower
`euler=EulerOptions(tolerance_change=...)` so the iteration is allowed to keep
taking small steps. Compare `final_relative_density_change` against
`tolerance_change` to see which bound bit.

**`max_iter`** — the budget was exhausted with the residual still decreasing.
Raise `max_iter`, or make each iteration more effective: `mixing` larger for a
slow monotone approach, `anderson=AndersonOptions(history=5)` for an
oscillating one. `mixing_backtracks` counts how often the adaptive schedule
had to retreat; a large count on a short run means the steps are too
aggressive rather than too timid.

**`line_search_failed`** — a minimize step could not find a sufficient
decrease before `minimum_step_size`. Usually the objective is locally much
stiffer than `step_size` assumes. Lower `step_size`, or lower
`line_search_factor` below its default 0.5 so the search contracts faster.
`armijo_factor` sets how much decrease counts as sufficient; its default
1e-4 is permissive already, so lowering it further rarely helps.
`line_search_failures` counts the occurrences.

## When the iteration oscillates

`EulerOptions` grows the mixing by `mixing_growth` after an improving step and
shrinks it by `mixing_backtrack_factor` after a worsening one, staying inside
`[minimum_mixing, maximum_mixing]`. That adaptation is on by default; set
`adaptive_mixing=False` to hold `mixing` fixed, which makes a run reproducible
step for step at the cost of robustness.

`max_log_density_change` bounds one update in log density and exists to stop a
single step from emptying or filling a voxel. It appears in both
`EulerOptions` and `MinimizeOptions` and rarely needs changing; reaching it
often is a sign that `mixing` or `step_size` is too large.

Anderson acceleration forms a trial from recent residual pairs and is accepted
only when neither its maximum nor its RMS residual exceeds the scalar-mixed
fallback, so it cannot make a step worse. `anderson_attempts`,
`anderson_accepted`, `anderson_rejected` and `anderson_resets` report how it
fared; all-rejected means the history window is not informative and the
acceleration is only costing evaluations.

## Constraints

`particle_numbers` selects the canonical ensemble; omitting it solves at
`data["mu"]`. `maximum_density` caps each component, and at fixed particle
number every update is projected onto the intersection of the cap and the
normalization — so a capped voxel may report a negative unconstrained
residual, because raising its density is forbidden. That is the KKT condition,
not a failure.

`maximum_polarization_fraction` bounds `|P| <= q_max * m * rho` per component
for fixed-dipole solves, a density-aware restriction on alignment rather than
an absolute cap; it is unavailable for scalar solving.

`homogeneous_axes="x"` or `["x", "y"]` makes the density constant along those
grid axes for a density-only minimize; `full_max_density_residual` and
`full_converged` then report unrestricted stationarity alongside the
constrained ones. `fixed_field="rho"` or `"dipole_density"` holds one field of
a fixed-dipole minimization fixed and relaxes only the other, with the coupled
residual still available as `full_maximum_residual` and `full_rms_residual`.

`excluded_mask` in the data marks inaccessible voxels: their density stays
exactly zero, particle numbers are normalized over the accessible points, and
their residuals do not enter convergence.

## Starting point

Without `initial.rho` or `data["rho"]`, the density starts proportional to
`exp(-beta_multiplier * beta * V_ext)`. The default `beta_multiplier=0` is a
uniform start; `1.0` is the ideal-gas profile, and intermediate values temper
it. A supplied density always wins. For a scan over state points, feeding the
previous solution through `initial=Initialization(rho=previous["rho"])` is
usually worth more than any tolerance tuning.

## The ideal reference during a solve

A solve evaluates the full physical intrinsic functional. It never sets
`use_voxel_mask`, so a fitting cutoff such as `rho_min` cannot restrict the
ideal law behind an Euler update or a line-search objective — switching
energy terms on at a density threshold would put a discontinuity in the
objective being minimized. A training forward of the same model does apply
that selection, so the two can report different `beta_F_id` for one field;
[`model.md`](model.md#selecting-energy-centers-for-fitting) says which is
which.
