# Migrating existing models, checkpoints and code

What changed for material that predates the current release, collected here
so the README can describe the package as it is rather than as it was.
`equicdft/serialization.py` holds the authoritative per-version record of the
model file format; this page covers the parts that need a decision from you.

## Model files

Normal `load_model`, `read_model_config` and `build_model` accept only the
current model format (5). Older files must be converted explicitly using the
installed, optional-use `equicdft.conversion` package:

```bash
python -m equicdft.conversion old_model.pt converted_model.pt
```

This accepts trusted whole-object `torch.save(model)` pickles and older
versioned model files. A distinct destination is required: conversion never
overwrites the source. Existing output files require `--overwrite`.
Only convert files you trust, including files believed to be versioned. The
converter tries restricted loading first, but an unpickling rejection falls
back to the historical object loader, which can execute code from the file.
Normal `load_model` uses restricted loading without this fallback.

- Density-only format-1/2 models need no extra physical metadata.
- Format 3 introduced model-owned ideal references. Older polarized models
  require `--dipole-magnitude` using the original molecular moment in the
  original dipole units, and `--thermal-wavelength` if it differs from 1.
  Descriptor normalization scales are not used to guess physical moments.
- Conversion carries `compute_local_mu` and `rho_min` forward
  without changing fitted parameters: both are optional diagnostics on the
  current model again, computing the same quantities from the same mask, so
  a converted functional evaluates the chemical potential its fit used. A
  file written while they were briefly absent takes the current defaults.

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

## Model-owned polarization normalization (format 5)

Pass one explicit positive `polarization_scale` to `GridCACEModel`. It normalizes
polarized Cartesian A features, their message carriers and polarization LDA
inputs. `CartesianAFeatures` no longer accepts `dipole_density_scale`;
`LDAReadout(include_polarization=True)` replaces its old scale argument.
Standalone A evaluation takes `a_features(data, polarization_scale=scale)`;
standalone LDA gets the scale from `ReadoutContext`.

`make_dataloaders(..., compute_polarization_std=True)` returns the optional
`polarization_std` statistic over training **and validation**, excluding test
data. It is the square root of one third of the trace of the population
covariance, pooling retained voxel/species vectors. Invalid/excluded voxels
are omitted; no density cutoff is applied. The model divides P by the chosen
scale without subtracting any mean. The user may instead supply any other
positive reference with the same units, including molecular moment times
reference density. A zero dataset standard deviation requires another explicit
positive choice, not an automatic floor.

Model files now use format 5; convert older files explicitly. Conversion keeps
the historical feature scale (or the first polarized LDA scale without polarized
features). If an old LDA used a different scale, its first-layer squared-P
weights are rescaled by `(new_scale / old_scale)**2` to preserve its function.
Matching scales leave fitted weights unchanged. Existing data and models are
never renormalized using newly computed dataset statistics. Density-only
architectures are unchanged. Training-checkpoint format is separate; historical
trainer resume is not restored. Analytical physics and reciprocal kernels keep
physical P and their original units.

## Model-provided local chemical potential

The explicit physical fitting interface is restored as an opt-in:
`GridCACEModel(..., compute_local_mu=True, rho_min=...)`. The model returns
`beta_mu_local` (total density derivative plus beta V), its weighted
per-field/per-species `beta_mu_average`, and
`voxel_weights`. Both chemical potentials are dimensionless beta mu.
Use those keys with ordinary `TensorLoss` and `Metrics`; no custom loss or
trainer output transform is needed. The intrinsic energy remains independent
of V. Local-mu evaluation requests the total density derivative automatically.

The defaults are `compute_local_mu=False`, `rho_min=0`. Settings are saved in
current model configurations; older files without them acquire these defaults
during explicit conversion, which carries recorded settings through, so a
converted fitted functional keeps the cutoff its fit used. Model-format
version and tensor state are unchanged by these optional settings.

`model(data, compute_local_mu=False)` overrides the saved setting for intrinsic
evaluation. Missing V raises when diagnostics are requested, as does
`compute_ideal=False`. Solvers and internal Fourier energy probes disable these
fitting diagnostics independently of the saved defaults.

The cutoff and optional `valid`/exclusion masks define weights unless explicit
input `voxel_weights` provide the fixed selection. SEM training now applies
noise before selecting voxels, then recalculates local and mean chemical
potentials on that realized selection. Keep independent support in `valid`
and exclusions, not a clean-density cutoff or cached `voxel_weights`.
Explicit weights remain overrides; selecting noisy vacuum explicitly raises.
Every field/species must retain positive weight for the spatial mean.

The equivalent centered `density_derivative` versus `-beta*V_ext` formulation
remains supported, with the same mask and gauge. Its external-field RMSE/sigma
and R² normalization differs from comparison against mean chemical potentials.

## Solver evaluation

### Explicit voxel selection during fitting

`model(data)` now evaluates the full physical intrinsic functional. Use
`model(data, use_voxel_mask=True)` for fitting: omit ideal/LDA/CACE-centered
energy outside `model.voxel_weights(data) > 0`, retaining original rho/P in
all descriptors and full Coulomb/Gaussian LR. Masked centers can have nonzero
excess derivatives through neighboring or LR contributions. Bulk, pairwise
and GGA readouts remain full; this is not a generic mask on every readout.

Rename the input/output/loss key `beta_mu_weights` to `voxel_weights` and use
the single public `model.voxel_weights(data)` method instead of
`beta_mu_weights()`, `density_weights()` or `ideal_mask()`. Nonnegative
fractional weights are preserved for averages/losses; only their positive
support selects local energy. Empty selections are valid except when computing
a chemical-potential average. `beta_mu_local` and `beta_mu_average` are unchanged.

The old `apply_ideal_mask` switch is removed. `use_voxel_mask` defaults to
False and is not a serialized model setting. Merely supplying voxel weights
does not mask energy. `Trainer` defaults to True for models exposing the weight
method; SEM density cutoffs follow the noisy realization. Direct fitting, scoring and parity calls
must now opt in explicitly. A `TrainingStream` can override this default via
`model_kwargs={"use_voxel_mask": False}`.

For SEM migration, remove cached clean `voxel_weights` and
`polarization_weights`; both field `TensorLoss` terms can use output
`voxel_weights`, which expands over polarization components. A legacy
`valid` that already includes the clean density cutoff is not independent
support. Rebuild that cache with full-support fields/targets and a separate
support mask; renaming the old mask does not fix it. This is a changed noise
objective and requires a separately identified run, not silent resumption.

This deliberately changes masked *excess* energy and its derivatives compared
with the former ideal-only mask. It is a new fitting objective, not a
loss-preserving script rename; do not resume an old campaign silently. Full
physical evaluation and solves retain their previous functional. Converted
models retain their parameters and saved rho_min; no model format/state
migration is required. Historical executed scripts must retain their pinned
source, with application API changes made in a new copy.

Solvers use full energy. Fourier stability retains its existing ideal reference
formulas and full excess energy, recomputing the full central energy if supplied
a masked forward. Standalone `IdealReference.evaluate(..., mask=None)` is
unchanged, and still accepts its optional Boolean ideal-only mask.

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
| `average_chemical_potential` | `beta_mu_average` |
| `equilibrium_chemical_potential` | `beta_mu` (solver) |

The old spellings carried the same values and are no longer returned. The
model's optional `beta_mu_local` and `beta_mu_average` are the 0.0.x
quantities under current names, computed from the model's own mask; the
solver's `beta_mu_local` is a separate solver result.
`density_derivative_excess` is exactly `-c1`; both names are returned, and
new code that handles total and excess responses together should prefer the
`*_derivative*` pair.

## Constructors

Options are keyword-only after the structural arguments, and the 0.0.x
positional order raises `TypeError`. `hidden_sizes` defaults to `(32, 16)`
everywhere. `FourierStabilityLoss` no longer takes `dipole_magnitude`; it
reads the moment from `model.ideal`.
