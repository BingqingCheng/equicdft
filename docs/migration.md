# Migrating existing models, checkpoints and code

What changed for material that predates the current release, collected here
so the README can describe the package as it is rather than as it was.
`equicdft/serialization.py` holds the authoritative per-version record of the
model file format; this page covers the parts that need a decision from you.

## Model files and checkpoints

Model files are migrated in memory when they load, so nothing needs to be
rewritten on disk:

- Density-only format 1 and 2 files load directly. Polarized files from those
  formats require an explicit ideal reference, as described below.
- Format 3 moved the ideal reference into the model. **Polarized files older
  than format 3** do not record the physical molecular moment, and it is not
  inferred from descriptor scales. Loading one requires the value from the
  original run: `load_model(path, ideal=FixedDipoleIdeal(m))`, where `m` is
  the molecular dipole magnitude in the original run's dipole units, or
  `python -m equicdft.convert --dipole-magnitude m ...`. Supply the original
  thermal wavelength to `FixedDipoleIdeal` as well if it differs from 1.
- Format 4 discards the retired `compute_local_mu` and `rho_min` settings.
  Fitted parameters and intrinsic thermodynamics are unchanged. If your fit
  relied on that density cutoff, set it explicitly in the application's loss
  mask and in the solver's residual settings; the model no longer carries it.

Training checkpoints written at version 2 and later resume through
`Trainer.load_checkpoint`. A checkpoint whose `FourierStabilityLoss` saved
its own `dipole_magnitude` is accepted when the value agrees with
`model.ideal`, and the saved copy is then dropped: the moment lives in the
model alone.

Whole-object `torch.save(model)` files from the flat 0.0.x layout still
convert with `python -m equicdft.convert`, which remaps the old module paths.
The flat module paths themselves (`equicdft.features`, `equicdft.readout`,
`equicdft.trainer`, ...) no longer import.

## Chemical potentials moved out of the model

The model no longer computes masked or averaged chemical potentials. Fits use
the intrinsic `density_derivative` against `-beta * V_ext`, with explicit
loss weights and `TensorLoss(center_axis=-2)`; `Metrics(center_axis=-2)`
applies the same canonical gauge projection. Masks and mean removal belong to
those losses and metrics. `GridSolver.evaluate` supplies `beta_mu_local` when
external fields are given.

## Solver evaluation

`solver.evaluate(data, compute_derivatives=True)` replaces the old
`compute_c1` keyword, and overrides the model's derivative defaults;
`compute_derivatives=False` is energy-only. Analytic ideal-reference
diagnostics may still be returned, and `c2` is never requested by solver
evaluation. Direct model calls keep their own `compute_c1` keyword.

Solver evaluation returns detached tensors and does not mark caller fields
for gradients; use the model directly for training or higher derivatives.

`GridSolver(model, dipole_magnitude=m)` is gone: build the model with
`ideal=FixedDipoleIdeal(m)` and call `GridSolver(model)`.

## Renamed result keys

| Retired | Current |
|---|---|
| `euler_lagrange_residual` | `density_residual` |
| `max_euler_lagrange_residual` | `max_density_residual` |
| `rms_euler_lagrange_residual` | `rms_density_residual` |
| `full_*euler_lagrange_residual` | `full_*density_residual` |
| `polarization_euler_lagrange_residual` | `polarization_residual` |
| `iterations` (coupled solve) | `n_iter` |
| `mu_loc`, `local_chemical_potential` | `beta_mu_local` |
| `average_chemical_potential` | `beta_mu_average` (application-supplied) |
| `equilibrium_chemical_potential` | `beta_mu` |

The old spellings carried the same values and are no longer returned.
`density_derivative_excess` is exactly `-c1`; both names are returned, and
new code that handles total and excess responses together should prefer the
`*_derivative*` pair.

## Constructors

Options are keyword-only after the structural arguments, and the 0.0.x
positional order raises `TypeError`. `hidden_sizes` defaults to `(32, 16)`
everywhere. `FourierStabilityLoss` no longer takes `dipole_magnitude`; it
reads the moment from `model.ideal`.
