# Migrating existing models, checkpoints and code

What changed for material that predates the current release, collected here
so the README can describe the package as it is rather than as it was.
`equicdft/serialization.py` holds the authoritative per-version record of the
model file format; this page covers the parts that need a decision from you.

## Model files

Normal `load_model`, `read_model_config` and `build_model` accept only the
current model format (4). Older files must be converted explicitly using the
installed, optional-use `equicdft.conversion` package:

```bash
python -m equicdft.conversion old_model.pt converted_model.pt
```

This accepts trusted whole-object `torch.save(model)` pickles and older
versioned model files. A distinct destination is required: conversion never
overwrites the source. Existing output files require `--overwrite`.
Only convert files you trust; whole-object conversion uses Python pickle.

- Density-only format-1/2 models need no extra physical metadata.
- Format 3 introduced model-owned ideal references. Older polarized models
  require `--dipole-magnitude` using the original molecular moment in the
  original dipole units, and `--thermal-wavelength` if it differs from 1.
  Descriptor normalization scales are not used to guess physical moments.
- Conversion to format 4 discards retired `compute_local_mu` and `rho_min`
  settings without changing fitted parameters. Preserve the original masks
  explicitly in application losses and solver residual settings.

The Python API is `equicdft.conversion.convert_model(source, destination, ...)`.
It accepts `ideal=FixedDipoleIdeal(...)` for missing physical metadata and
`verification_data=...` for explicit forward checks (required when verifying
whole-object models with electrode sites). Whole-object conversion verifies
old/upgraded and rebuilt excess predictions by default. Versioned conversion
strictly loads migrated state; file conversion checks a current-payload
round trip, and explicit verification data adds a forward comparison.
`load_converted_model(source, ...)` performs explicit conversion in memory
without writing a file. Already-current files need no conversion.

Historical module remapping and attribute repair live in
`equicdft.conversion.legacy`; versioned migrations live in
`equicdft.conversion.formats`. Normal package import, model loading, training
and solving do not import conversion. The old `equicdft.legacy` and
`equicdft.convert` paths are removed.

## Training checkpoints

`Trainer.load_checkpoint` and `load_stream_checkpoint` resume only current
training checkpoints (version 4). Optimizer, scheduler, history, noise settings
and recorded random states are restored normally. Legacy training-checkpoint
resume and conversion are not supported. Converting a fitted model does not
recover its optimizer or training history; use it as initialization for a new
fit if needed. Model-format and training-checkpoint versions are separate.

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

## Solver options and stopping statuses

`solve` now accepts only grouped options from `equicdft.solver`:

- `ConvergenceOptions`: iteration limit and residual tolerances.
- `EulerOptions`: mixing, change threshold and optional `AndersonOptions`.
- `MinimizeOptions`: step size and line-search settings.
- `Constraints`: particle counts, bounds, homogeneous axes and fixed field.
- `Initialization`: `rho`, `polarization` and `beta_multiplier`.

Flat keywords such as `max_iter`, `particle_numbers`, `initial_rho` and
`anderson=True` are removed. For example, replace `initial_rho=rho` with
`initial=Initialization(rho=rho)`. Pass `euler` only with Euler and `minimize`
only with minimization; the other method's group raises instead of being ignored.
If `method` is omitted, it is Euler for density-only models and minimize for
fixed-dipole models.

All drivers use `converged`, `stalled`, `max_iter` and `line_search_failed`.
`stalled` means an early small-change stop without satisfying the residual
criterion; `max_iter` means the iteration budget was exhausted. This corrects
the previous coupled-Euler label for a small-change stop and the scalar-Euler
label when that stop coincides with the budget. Updates and trajectories are
unchanged.

## External-field targets and SEM noise

`GridData` loads physical fields and metadata, not ideal-dependent `c1` or
`c1_plus_beta_mu` targets. Fit total `density_derivative` against `-beta*V_ext`
with the same canonical gauge projection and original density mask. With known
absolute chemical potentials, use `beta*(mu - V_ext)` without mean removal.
For dipoles, fit total `polarization_derivative` against `beta*E_ext`, retaining
the original weights and physical moment scaling inside the squared residual.
Model `c1`, `c2` and excess-derivative outputs remain available.

Noise augmentation changes only `rho` and/or `dipole_density`, never targets.
It no longer refreshes targets by recognizing their dictionary names. Using
the total-derivative targets above gives the same squared objective and
parameter gradients as explicitly recomputing the old ideal-dependent targets
on the same noisy fields. Do not continue an old noisy excess-target fit by
merely renaming its target: switch both the prediction and the target formula.

`NoiseOptions` contains only `density`, `polarization` and `density_floor`;
the separate `dipole_magnitude` option is removed. The trainer uses
`model.ideal`. Standalone polarized augmentation must pass `ideal` explicitly
to `NoiseOptions.apply` or `add_field_noise`. Scalar Gaussian-plus-floor and
polarized whole-pair rejection sampling, masks and RNG draw order are unchanged.

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
