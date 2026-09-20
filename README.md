# equicdft

`equicdft` learns excess Helmholtz free-energy functionals of classical fluids
from equilibrium density fields on periodic three-dimensional grids. The model
is energy-first: it predicts one scalar functional for a complete field, then
uses automatic differentiation to obtain direct correlations and equilibrium
conditions.

The package is research software under active development. The example below
is the smallest complete training workflow; it uses the same model construction
and local-chemical-potential objective as the current Lennard--Jones fits.

Three longer documents live in `docs/`: `docs/polarization.md` for fixed-dipole
fluids and the coupled density/polarization solve, `docs/metal-wall.md` for
electrode boundary conditions, and `docs/experimental.md` for the stability
regularizers and other features still being evaluated. `docs/migration.md`
records what changed for existing models and data files.

## Method in one page

For grid-cell volume $\Delta V$, density $\rho_g$, and a learned local
excess free energy per particle $a_g^{\mathrm{exc}}$, the default beta-energy
convention used by the model and the included example is

$$
\beta F_{\mathrm{exc}}[\rho,T]
= \Delta V\sum_g \rho_g\,\beta a_g^{\mathrm{exc}}.
$$

Each grid point is represented by the density in a periodic integer-spherical
neighborhood. Cartesian density moments are symmetrized under the 48 signed
axis permutations of the cubic grid, giving invariants of the lattice point
group. A separate center-density channel and the invariant neighborhood
features are passed to a local neural readout. An optional LDA readout supplies
the density-only baseline; all enabled readout energies are summed before any
derivative is taken.

An optional finite-range pairwise readout learns one isotropic kernel for each
unique density-component pair. Its neural network receives normalized grid
distance and temperature, a quintic envelope smoothly cuts the kernel off,
and the zero-displacement self term is excluded. The periodic pair sum is
evaluated by FFT convolution without constructing gathered pair densities.

The first direct correlation follows from the discrete functional derivative

$$
c_g^{(1)} = -\frac{1}{\Delta V}
\frac{\partial\,\beta F_{\mathrm{exc}}}{\partial \rho_g}.
$$

For an equilibrium field in an external potential, the dimensionless local
chemical potential is

$$
\beta\mu_g^{\mathrm{loc}}
= \log(\rho_g\Lambda^3)+\beta V_g^{\mathrm{ext}}-c_g^{(1)}.
$$

Equilibrium requires this quantity to be spatially constant. NVT data therefore
need no chemical-potential labels: the loss is the masked spatial variance of
$\beta\mu_g^{\mathrm{loc}}$ about its per-field, per-species mean.
Construct the model with `compute_local_mu=True` and a chosen `rho_min` to
return `beta_mu_local`, `beta_avg_mu`, and
`beta_mu_weights`. Both chemical potentials are dimensionless
**beta mu**, not energies. The default cutoff is
zero; the LJ example uses `rho_min=1e-3` in reduced density units.

```python
from equicdft import Loss, Metrics, TensorLoss

loss = Loss(terms=[
    TensorLoss(
        name="beta_mu_local",
        prediction_key="beta_mu_local",
        target_key="beta_avg_mu",
        weights_key="beta_mu_weights",
    )
])
metrics = [Metrics(
    name="local beta mu",
    prediction_key="beta_mu_local",
    target_key="beta_avg_mu",
    selection_mask_key="beta_mu_weights",
    metric_keys=("mae", "rmse", "rmse_percent", "r2"),
)]
```

Weights select `rho > rho_min`, intersected with accessibility and an optional
Boolean `valid` voxel mask. Explicit input `beta_mu_weights` replace
the cutoff selection, enabling fixed masks or nonnegative statistical weights;
every field/species must retain positive weight. The local output is zero
outside this mask. With SEM augmentation, the trainer freezes the original
weights and the model recomputes both chemical potentials on the noisy fields.

RMSE is the dimensionless chemical-potential residual. `rmse_percent` is
100 times RMSE divided by the standard deviation of the target means over
selected entries; R² uses the same target variation. These are undefined for a
constant target, not normalized by external-field variation. The equivalent
centered-total-derivative loss remains available for applications needing it.

The model owns its ideal reference (`IdealGas` by default for scalar models).
It returns `beta_F_id`, `beta_F_exc`, their sum `beta_F`, and requested total
intrinsic derivatives. `c1` and `c2` retain their excess-only definitions.
`GridSolver(model)` adds external fields and ensemble constraints; it does not
construct another ideal functional. `model(data, compute_ideal=False, compute_local_mu=False)` evaluates
only the excess functional, for example for an excess-response probe.

One complete density field is one dataset item. Fields are never split into
independent grid-point samples because the scalar functional and its derivative
couple overlapping neighborhoods.

## Installation

From a clone of this repository:

```bash
python -m pip install -e .
```

The core dependencies are PyTorch, NumPy, and ASE. Python 3.8 or newer is
supported.

## Five-minute example

The repository contains five reduced-unit Lennard--Jones density fields on
$16^3$ grids. Run a two-epoch smoke test from the repository root:

```bash
python examples/lj_nvt/train.py --epochs 2
```

The pedagogical default is 20 epochs:

```bash
python examples/lj_nvt/train.py
```

The script automatically uses a CUDA device when available. It writes
checkpoints, `history.csv`, a plain training log, `run_config.json`, and a
directly loadable `model.pt` under `examples/lj_nvt/fit/`. Restarting the same
command resumes from `last.pt`; pass `--fresh` to start a new fit.

The mini dataset is intentionally too small for a quantitatively transferable
functional. It exists to verify data loading, functional differentiation,
training, checkpoint restart, and GPU/CPU execution. See
[`examples/lj_nvt/DATASET.md`](examples/lj_nvt/DATASET.md) for provenance.

## Current training recipe

The annotated example exposes the important options while using these defaults:

| Setting | Value |
|---|---:|
| grid spacing | $0.5\sigma$ |
| local cutoff | 3 grid cells, $\lvert\mathbf q\rvert^2\leq 3^2$ |
| Cartesian degree | $p=3$ |
| invariant product order | $\nu=2$ |
| distance damping | none (equal-weight polynomial moments) |
| center treatment | separate normalized center density |
| energy readouts | LDA + local invariant readout |
| MLP hidden widths | 32, 16 with SiLU activations |
| density scale | train+validation mean density |
| temperature scale | train+validation mean temperature |
| free-energy convention | directly learn $\beta F_{\mathrm{exc}}$ |
| density mask | $\rho>10^{-3}$ |
| batch size | 2 complete fields |
| optimizer | Adam, learning rate $10^{-4}$ |
| scheduler | ReduceLROnPlateau, factor 0.5, patience 3 |
| objective | spatial constancy of $\beta\mu^{\mathrm{loc}}$ |

The data split acts on complete fields and is reproducible from the seed. For
scientific benchmarks, thermodynamic states intended for testing should be
placed in a separately constructed test dataset rather than left to the random
validation split.

## Data format

`GridData.from_xyz` reads one EXTXYZ frame per complete regular grid. The
default field names and requirements are:

| EXTXYZ location | Name | Required | Meaning |
|---|---|---:|---|
| frame metadata | `T` | yes | temperature |
| frame metadata | `grid_spacing` | yes | one or three physical grid spacings |
| frame metadata | `grid_size` | no | $(N_x,N_y,N_z)$; inferred from positions if absent |
| frame metadata | `grid_indexing` | no | `zero_based` (default) or `one_based` |
| frame metadata | `mu` | no | reservoir chemical potential per component |
| per-grid array | `density` | conditional | density component(s) |
| per-grid array | `V_ext` | conditional | external-potential component(s) |
| per-grid array | `excluded_mask` | no | Boolean hard exclusion; `True` means inaccessible |

Each EXTXYZ frame has the usual three-part structure: the number of grid
points, one metadata/property line, and one record per grid point. An
abbreviated one-component frame looks like

```text
4096
Lattice="16 0 0 0 16 0 0 0 16" Properties=species:S:1:pos:R:3:density:R:1:V_ext:R:1 T=1.2 grid_size="16 16 16" grid_spacing="0.5 0.5 0.5" grid_indexing=zero_based pbc="T T T"
X  0  0  0   0.0214  -0.340
X  0  0  1   0.0189  -0.281
...
```

Here `...` only abbreviates the remaining records: an actual frame must contain
exactly $N_xN_yN_z$ distinct sites. The three `pos` values are integer grid
indices, not physical Cartesian coordinates. They must cover the complete
regular grid; input row order is arbitrary because `GridData` sorts sites into
C order (the last index varies fastest). Physical displacements are obtained by
multiplying these indices by `grid_spacing`. The ASE species and `Lattice`
fields are container bookkeeping and are not used to build neighborhoods.

At least one of `density` and `V_ext` must be present. Training the equilibrium
local-chemical-potential objective requires both. A density-only frame can be
used for intrinsic-functional evaluation, while a potential-only grid is the
starting point for an equilibrium solve. `mu` may be omitted (or stored as
`nan`) for NVT data.

For a mixture with $M$ components, declare `density:R:M` and `V_ext:R:M` in
the `Properties` field and place the component columns consecutively in each
record. When both arrays are present they must have the same width; this width
defines `n_types`. A scalar field is normalized internally to `[n_grid, 1]`,
and a mixture to `[n_grid, n_types]`.

All numerical units must be mutually consistent. `GridData` computes
$\beta=1/(k_BT)$ from `T` and the `boltzmann_constant` argument, while
`thermal_wavelength` fixes the ideal-gas logarithm. `GridData` and
`GridCACEModel` share one default, `equicdft.units.DEFAULT_BOLTZMANN_CONSTANT`
(eV/K, temperatures in kelvin), and the model checks every batch's $\beta$
against its own constant, so a dataset and a model built with different
constants raise instead of mixing units. The bundled example uses
Lennard--Jones reduced units with $k_B=\Lambda=1$ on both sides.

```python
from equicdft import GridData

fields = GridData.from_xyz(
    "fields.extxyz",
    cutoff_grid=3,
    boltzmann_constant=1.0,   # Lennard--Jones reduced units
    thermal_wavelength=1.0,
)
```

`GridData.from_xyz` also accepts a list of files. Custom source names can be
provided through `data_key`, and `GridData.from_dict` constructs an empty
regular grid for inference.

Every processed field contains a Boolean `excluded_mask` tensor with shape
`[n_grid]`. Missing exclusion masks default to all `False`. A `True` value
represents an exact hard wall: the corresponding density must be zero for every
component. This geometric constraint is distinct from the selection mask used
to omit noisy targets from metrics and from the numerical weights used by the
local-chemical-potential loss. An exclusion mask may be coarsened only when
every coarse block is either entirely accessible or entirely excluded;
partially accessible coarse voxels require an explicit accessible-volume
representation and are rejected.

## Inference and equilibrium solution

A saved model can be loaded without reconstructing the architecture:

```python
import torch
from equicdft import GridData, load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model("examples/lj_nvt/fit/model.pt", map_location=device)

field = GridData.from_xyz(
    "examples/lj_nvt/mini_lj_nvt.extxyz",
    index="0",
    grid_info=model.grid_info,
)[0]
field = {
    key: value.to(device) if torch.is_tensor(value) else value
    for key, value in field.items()
}
outputs = model(field)
print(outputs["beta_F_exc"], outputs["c1"])
```

`save_model` and `load_model` use a versioned file that stores the model's
structure (`model.to_config()`, a plain JSON-compatible dictionary) next to
its fitted `state_dict`. The file contains no Python object references, so
it loads under PyTorch's default `weights_only` unpickler, and
`read_model_config` inspects the architecture without building it. Model
files in older formats, including `torch.save(model)` pickles, are converted explicitly:

```bash
python -m equicdft.conversion old_model.pt converted_model.pt
```

The converter rebuilds the model, strictly transfers migrated fitted state,
and verifies whole-object forward equivalence by default. A distinct output
path is required; the source is never overwritten. Only convert trusted files.
Normal loading accepts only current-format models. Training resume likewise
accepts only current-format checkpoints; legacy checkpoint resume is unsupported.

Application checkpoints containing a configuration and `state_dict` are not
whole-object files: rebuild them with their original application builder,
load the state strictly, and then call `save_model`. Inference model files
do not replace training-resume checkpoints with optimizer/scheduler state.

`GridSolver` provides two complementary operations:

- infer the external potential, up to the chemical-potential gauge, from a
  supplied equilibrium density;
- solve an equilibrium density at fixed particle number or fixed chemical
  potential using Euler iteration or direct free-energy minimization.

`solver.evaluate(data, compute_derivatives=True)` returns intrinsic and external
energies, local chemical potentials and density residuals;
`compute_derivatives=False` is energy-only and overrides the model's
derivative defaults. It returns detached tensors and does not mark caller
fields for gradients, so training and higher derivatives go through the model
directly, which reports `density_derivative` for the total intrinsic response
and `density_derivative_excess = -c1` for its excess part. With
`compute_ideal=False` only excess responses are returned; total-response names
are never reused.

`beta_external` is the external coupling (`beta_V_ext` for density-only data),
`beta_A = beta_F + beta_external`, and `beta_Omega` subtracts `beta_mu_N` when
a chemical potential is supplied. The density stationarity residual and its
norms are `density_residual`, `max_density_residual` and
`rms_density_residual`, with `polarization_residual` the coupled counterpart.
Renamed and retired keys are listed in `docs/migration.md`.
Prescribed-density evaluation reports unconstrained stationarity; a solve
reports the residuals after its bounds and homogeneous projections.

The solver validates the grid against `model.grid_info`. Its options come in
validated groups from `equicdft.solver` (`ConvergenceOptions`, `EulerOptions`
with `AndersonOptions`, `MinimizeOptions`, `Constraints`, `Initialization`),
and only these groups are accepted. For example,
`solver.solve(data, method="euler", constraints=Constraints(particle_numbers=[N]),
convergence=ConvergenceOptions(max_iter=300), euler=EulerOptions(mixing=0.1))`.
Pass `euler` only for Euler and `minimize` only for minimization; mismatched
groups raise an error. See the docstring for the full set. Both equilibrium algorithms keep excluded
densities exactly zero, normalize fixed particle numbers over accessible grid
points only, and omit excluded points from Euler--Lagrange convergence
diagnostics. This treats the exclusion mask as an infinite external potential
on the existing periodic grid; it does not replace periodic stencils or
reciprocal kernels with nonperiodic boundary conditions.

## Package map

- `equicdft.geometry`: stencils, grid index arithmetic, periodic convolutions
- `equicdft.data`: `GridData` fields, EXTXYZ parsing, loaders, metal sites,
  the `equicdft-mean-density` console script
- `equicdft.representation`: Cartesian moments (`CartesianAFeatures`), cubic
  invariants (`CartesianBFeatures`), message passing (`BChiMessage`)
- `equicdft.functional`: the readout interface and thermodynamics, local,
  state-dependent, pairwise, reciprocal-space and metal-wall readouts, the
  fixed-dipole ideal gas
- `model.py`, `derivatives.py`: scalar functional and automatic derivatives
- `equicdft.response`: Fourier probe modes, curvature, stability and response
  losses, metrics and data
- `equicdft.training`: loss terms, streaming metrics, the trainer, noise
- `equicdft.solver`: forward thermodynamics and equilibrium density solution
- `serialization.py`: current-format model saving/loading
- `equicdft.conversion`: explicit old-model conversion; not imported by the runtime

Whole-object `torch.save(model)` files written under the flat 0.0.x layout
still convert with `python -m equicdft.conversion`: `equicdft.conversion.legacy` maps the
old module paths to the current ones while unpickling.

Run the test suite with:

```bash
python -m unittest discover -s tests -t . -p "test_*.py"
```

The `tests/regression/lj_paper_v1` harness freezes one held-out `T=1.5` field and
the released model. It doubles as a regression test for gauge-aligned
density-to-external-potential inference and the fixed-particle-number reverse
solve:

```bash
python -m tests.regression.lj_paper_v1.forward
python -m tests.regression.lj_paper_v1.solve
```

## Scope

Implemented capabilities include multicomponent density tensors, optional LDA,
GGA, finite-range pairwise, and reciprocal-space readouts, first direct
correlations, selected rows of the second direct correlation, supervised or
latent chemical-potential training, and fixed $N$/fixed $\mu$ equilibrium
solves. The compact example uses the validated short-range LDA +
Cartesian-invariant model and does not enable GGA, pairwise, reciprocal-space,
or direct $c^{(2)}$ supervision.

Readout constructors share one convention: the structural arguments
(`n_types`, and `cutoff_grid` for `PairwiseReadout`) are positional, every
option is a keyword, and `hidden_sizes` defaults to `(32, 16)` everywhere.
`zero_init` defaults encode the readout's role and are deliberately not
uniform: the additive corrections `BulkReadout`, `LongRangeReadout` and
`PairwiseReadout` start at zero contribution (`zero_init=True`), so a model
initially equals its local part, while the primary `LDAReadout` starts from
random weights (`zero_init=False`); `LocalReadout` and `GGAReadout` have no
such option (`GGAReadout` controls its start through `initial_coefficient`).
