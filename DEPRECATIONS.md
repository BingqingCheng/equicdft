# Deprecations

Every deprecated path warns with `equicdft._deprecation.EquicdftDeprecationWarning`
(a `DeprecationWarning` subclass, silent by default outside tests; make it an
error with `warnings.simplefilter("error", EquicdftDeprecationWarning)`).
Deprecated in 0.1.0, removed in 0.2.0 unless noted.

## Module paths (0.1.0)

| old path | new home |
|---|---|
| `equicdft.stencil` | `equicdft.geometry.stencil` |
| `equicdft.features` | `equicdft.representation.cartesian` |
| `equicdft.symmetrize` | `equicdft.representation.invariants` |
| `equicdft.interaction` | `equicdft.representation.message` |
| `equicdft.energy` | `equicdft.functional.base` |
| `equicdft.thermodynamics` | `equicdft.functional.thermodynamics` |
| `equicdft.readout` | `equicdft.functional.readout` |
| `equicdft.semilocal` | `equicdft.functional.semilocal` |
| `equicdft.pairwise` | `equicdft.functional.pairwise` |
| `equicdft.reciprocal` | `equicdft.functional.reciprocal` |
| `equicdft.metalwall` | `equicdft.functional.metalwall` |
| `equicdft.polarization_ideal` | `equicdft.functional.polarization_ideal` |
| `equicdft.stability` | `equicdft.response.stability` |
| `equicdft.loss` | `equicdft.training.loss` (and `equicdft.response.loss` for `FourierResponseLoss`) |
| `equicdft.metrics` | `equicdft.training.metrics` (and `equicdft.response.metrics` for `FourierResponseMetrics`) |
| `equicdft.trainer` | `equicdft.training.trainer` |
| `equicdft.loader` | `equicdft.data.loaders` |
| `scripts/mean_density.py` | `python -m equicdft.data.mean_density` / `equicdft-mean-density` |

`equicdft.data`, `equicdft.response` and `equicdft.solver` became packages
that re-export their former contents; those paths are not deprecated.
Private modules (`equicdft._grid`, `equicdft._fourier`, ...) moved without
shims. Whole-object pickles that recorded any old path keep loading through
`equicdft.legacy`.

## Output keys (0.1.0)

Model outputs and solver results carry canonical keys with the `beta_`
prefix of the other dimensionless quantities; the old names remain as
aliases of the same tensors and warn when read through `[]` or `.get`.

| old key | canonical key |
|---|---|
| `local_chemical_potential` | `beta_mu_local` |
| `average_chemical_potential` | `beta_mu_average` |
| `chemical_potential_weights` | `beta_mu_weights` |
| `equilibrium_chemical_potential` (solver) | `beta_mu` |

## Changed defaults (0.1.0)

These change freshly built models only; saved models record every
constructor argument.

- `GridCACEModel(boltzmann_constant=...)` defaults to
  `equicdft.units.DEFAULT_BOLTZMANN_CONSTANT` (8.617333262e-5 eV/K, the
  `GridData` default) instead of `1.0`; the model also rejects a batch whose
  `beta` was built with another constant. Reduced-unit work passes
  `boltzmann_constant=1.0` to both `GridData` and `GridCACEModel`.
- `BulkReadout`, `LongRangeReadout` and `GGAReadout` default to
  `hidden_sizes=(32, 16)` like every other readout, instead of `(16, 16)`.

## Keyword-only constructor arguments (0.1.0)

Constructors take only their structural arguments positionally and every
option by keyword: `LocalReadout(n_types, ...)`, `BulkReadout(n_types, ...)`,
`LDAReadout(n_types, ...)`, `GGAReadout(...)` (all keyword; `n_types` is
listed first), `LongRangeReadout(n_kernels, n_types, ...)`,
`PairwiseReadout(cutoff_grid, n_types, ...)`,
`MetalWall(liquid_coulomb, metal_sites, metal_sigma, ...)`,
`ReciprocalFeatures(radial_exponents, ...)`, `FixedDipoleIdeal(dipole_magnitude, ...)`,
`CartesianAFeatures(max_power, mean_density, ...)`,
`CartesianBFeatures(max_power, max_product_order, ...)`,
`BChiMessage(n_invariant_features, n_radial_channels, n_channels, ...)`,
`GridCACEModel(a_features, b_features, readout, ...)`,
`TensorLoss(name, prediction_key, target_key, ...)`, `Metrics(target_key, ...)`,
`FourierResponseLoss(directions, ...)`, `FourierResponseMetrics(direction_names, ...)`,
`GridSolver(model, ...)` and `Trainer(model, loss, ...)`. Calls that pass
further arguments positionally still work in the 0.0.x order and warn; in
0.2.0 they raise `TypeError`.

## Trainer noise settings (0.1.0)

`Trainer(noise=NoiseOptions(density=..., density_floor=..., polarization=...,
dipole_magnitude=...))` and `TrainingStream(..., noise=NoiseOptions(...))`
replace the four keywords `density_noise`, `density_noise_floor`,
`polarization_noise` and `noise_dipole_magnitude`, which still work with a
warning, as do the `Trainer.density_noise`-style attributes (read
`Trainer.noise`). Checkpoints are written as version 3 (`noise`,
`stream_noise`); version-2 checkpoints still load.
