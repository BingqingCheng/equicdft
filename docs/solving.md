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
| `minimize=MinimizeOptions(...)` | step size, Armijo line search, `fixed_dipole_coordinate`, and `lbfgs_history` |
| `constraints=Constraints(...)` | ensemble, density bounds and polarization cap, homogeneous axes, fixed field |
| `initial=Initialization(...)` | starting `rho`, `polarization` and `beta_multiplier` |

## The orientational coordinate for fixed dipoles

`fixed_dipole_coordinate` selects which orientational variable a fixed-dipole
solve updates. A scalar model rejects any non-default value rather than
ignoring it, so the option never silently does nothing. `"alignment"` works
in the physical molecular alignment `u = P/(m rho)` and reconstructs
`P = m rho u`, which keeps the update in the variable the saturation bound is
written in.
The default for both methods is the conjugate field of the orientational
entropy. Euler can instead mix `P` directly. All choices keep the
positivity-preserving canonical log-density step and the same physical KKT
residual, so they are routes to one stationary point rather than different
problems.

The accepted values differ by group, because the established behaviour they
name differs:

| group | default | alternatives |
|---|---|---|
| `EulerOptions` | `"conjugate_field"` | `"polarization"`, `"alignment"` |
| `MinimizeOptions` | `"conjugate_field"` | `"alignment"` |

A string valid for one group raises `ValueError` on the other, so
`MinimizeOptions(fixed_dipole_coordinate="polarization")` is an error rather
than a synonym for its default. The conjugate-field Euler driver and
non-default minimizer coordinates report the choice back as
`solver_fixed_dipole_coordinate`, and an alignment
solve also reports `final_relative_alignment_change`, the largest change in
`u` over the last accepted step — the orientational counterpart to
`final_relative_density_change`.

The default `EulerOptions(fixed_dipole_coordinate="conjugate_field")` carries
the density as `log rho` and the polarization as the conjugate field
`h = L^-1(u)` with `P = m rho L(h)`, so that any linear combination of
iterates is a positive density with an interior polarization; particle
numbers and the density bounds are imposed by projection. It is the only
fixed-dipole path that accepts `anderson=AndersonOptions(...)`, which combines
the recent iterates with a sliding window (`history=40` is a good start), and
two safeguards keep the acceleration honest. An accelerated step after which
the fixed-point residual has grown by more than `AndersonOptions.safeguard`
discards the history, and every step, accelerated or mixed, must not raise
`beta_A`. The second test is what lets the driver cross a coexistence region
instead of settling on the unstable uniform fixed point. When it fires on a
mixed step the mixing shrinks by `mixing_backtrack_factor` and the step is
retried, and an accepted step lets it grow again by `mixing_growth`, so
`mixing` is the starting value of a schedule rather than a fixed number;
`mixing_backtracks` and `final_mixing` report how it fared. When even
`minimum_mixing` (or the fixed `mixing`, with `adaptive_mixing=False`) would
raise the free energy, the driver keeps the current state and stops with
`status="line_search_failed"`, exactly as the minimizer does; lower `mixing`
or turn the schedule on. `max_conjugate_field_change` bounds one update of
`h`; the alignment itself is limited only by `maximum_polarization_fraction`,
whose KKT residual the result reports, never by a hidden numerical margin.
The result carries `final_conjugate_field_change` beside the density and
polarization changes.

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

### Quasi-Newton minimization

`MinimizeOptions(lbfgs_history=50)` replaces the mirror-descent update of a
fixed-dipole minimization by projected L-BFGS in the same `(log rho, h)`
coordinates. The mirror-descent step is the gradient preconditioned by the
ideal Hessian in those coordinates, so it is the initial inverse Hessian of
the two-loop recursion, scaled each iteration by the Barzilai–Borwein
estimate in that metric; the stored pairs add the curvature of the excess
functional, which is what lets one step act on the stiff bulk modes and the
soft cavity modes together. Line-search trials are evaluated without
derivatives and only the accepted point is differentiated, so
`n_energy_evaluations` counts the cheap trials and `n_evaluations` the full
ones. The density bounds and the polarization cap are an active set: a
coordinate resting on a bound with an outward gradient is frozen for the
step and excluded from the curvature pairs, and the KKT residual the result
reports is exactly the stationarity condition being driven to zero. Whether
a pinned density voxel is released is decided against the chemical
potential of the free voxels; when every voxel of a species rests on a
bound, the multiplier is taken from the KKT bracket between the cap and the
floor values instead, so a constant shift of the potential never changes
the outcome. `step_size` scales the direction of every line search (1 is
the quasi-Newton length) and is reduced by `line_search_factor` as for
mirror descent.
`nonmonotone_window` larger than one accepts a step if `beta_A` falls below
the largest of that many recent values instead of the last one, which lets
long quasi-Newton steps through on a landscape with a shallow valley while
still ruling out the saddles a residual method can settle on. A line search
that fails with pairs in memory forgets them and retries with the mirror
step; only a failure of that step ends the solve with
`status="line_search_failed"`. The option requires the `"conjugate_field"`
coordinate and no `fixed_field`; `lbfgs_pairs` reports the pairs in memory
at the end.

The same option selects projected L-BFGS in `log rho` for a density-only
solve, canonical or grand canonical, with the same construction: the mirror
step `g / (V rho)` as the initial inverse Hessian, energy-only trials, the
density cap as the active set with the free-voxel chemical potential (or the
KKT bracket when no voxel is free) deciding releases, and homogeneous axes
handled by projecting the gradient. Convergence is judged by the same
residual as the other scalar drivers, so the tolerances carry over unchanged.
For a soft wall or another problem spanning many orders of magnitude in
density, `lbfgs_curvature_density_threshold` can restrict the quasi-Newton
curvature correction to populated density components. The threshold uses the
same component-density units as `rho`. Components at or below it still move
and retain the local mirror-descent direction; they are not frozen or floored.
They neither contribute to nor receive the quasi-Newton curvature correction,
although canonical normalization still couples all coordinates. The default
`None` (and the no-op value zero) keeps the original all-density L-BFGS
behavior. This option is separate from `residual_density_threshold`, which
affects diagnostics rather than steps.
On the LJ regression frame (4096 voxels, one species, `maximum_density`
inactive) it reaches a residual of `1e-4` in about 25 full evaluations where
Anderson-accelerated Euler needs about 70 and mirror descent stalls in its
line search near `1e-2`; with an active cap (about 200 voxels pinned) the
count is 20. The quasi-Newton path is preferable whenever the excess
functional is stiff relative to the ideal term, which is the usual case for
a dense liquid; on a nearly ideal gas the plain mirror step converges in a
few iterations and the curvature pairs only add bookkeeping. The result
carries `solver_lbfgs_history`, `n_energy_evaluations` and `lbfgs_pairs` as
for the fixed-dipole driver.

## Constraints

`particle_numbers` selects the canonical ensemble; omitting it solves at
`data["mu"]`. `maximum_density` caps each component, and at fixed particle
number every update is projected onto the intersection of the cap and the
normalization — so a capped voxel may report a negative unconstrained
residual, because raising its density is forbidden. That is the KKT condition,
not a failure.

`minimum_density` floors every accessible voxel of a fixed-dipole solve in the
same way: a voxel resting on the floor with an outward residual counts as
stationary, and `solver_minimum_density` records the floor. Use it when a
learned functional is not trusted at vanishing density; a floor around one
hundredth of the bulk density keeps cavities and vacuum regions from carrying
the whole residual budget while the rest of the fluid equilibrates.

`maximum_polarization_fraction` bounds `|P| <= q_max * m * rho` per component
for fixed-dipole solves, a density-aware restriction on alignment rather than
an absolute cap; it is unavailable for scalar solving.

`homogeneous_axes="x"` or `["x", "y"]` makes the density constant along those
grid axes for a density-only minimize; `full_max_density_residual` and
`full_converged` then report unrestricted stationarity alongside the
constrained ones. Projected L-BFGS evaluates every line-search trial on that
homogeneous manifold and projects its gradients, directions and curvature
pairs into the corresponding tangent space. `fixed_field="rho"` or
`"dipole_density"` holds one field of a fixed-dipole minimization fixed and
relaxes only the other, with the coupled residual still available as
`full_maximum_residual` and `full_rms_residual`.

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
