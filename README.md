# equicdft

`equicdft` learns excess Helmholtz free-energy functionals of classical fluids
from equilibrium density fields on periodic three-dimensional grids. The model
is energy-first: it predicts one scalar functional for a complete field, then
uses automatic differentiation to obtain direct correlations and equilibrium
conditions.

The package is research software under active development. The example below
is the smallest complete training workflow; it uses the same model construction
and local-chemical-potential objective as the current Lennard--Jones fits.

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
$\beta\mu_g^{\mathrm{loc}}$ about its per-field mean. If a reservoir chemical
potential is known, the same tensor loss can instead target `beta_mu`.

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
`thermal_wavelength` fixes the ideal-gas logarithm. The bundled example uses
Lennard--Jones reduced units with $k_B=\Lambda=1$.

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

A saved full model can be loaded without reconstructing the architecture:

```python
import torch
from equicdft import GridData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("examples/lj_nvt/fit/model.pt", map_location=device)
model = model.to(device).eval()

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

`GridSolver` provides two complementary operations:

- infer the external potential, up to the chemical-potential gauge, from a
  supplied equilibrium density;
- solve an equilibrium density at fixed particle number or fixed chemical
  potential using Euler iteration or direct free-energy minimization.

The solver validates the grid against `model.grid_info`. See its docstring and
the tests for the full option set. Both equilibrium algorithms keep excluded
densities exactly zero, normalize fixed particle numbers over accessible grid
points only, and omit excluded points from Euler--Lagrange convergence
diagnostics. This treats the exclusion mask as an infinite external potential
on the existing periodic grid; it does not replace periodic stencils or
reciprocal kernels with nonperiodic boundary conditions.

## Package map

- `data.py`, `stencil.py`: complete fields and periodic neighborhoods
- `features.py`, `symmetrize.py`: Cartesian moments and cubic invariants
- `readout.py`, `lda.py`, `gga.py`: composable free-energy contributions
- `model.py`, `derivatives.py`: scalar functional and automatic derivatives
- `loss.py`: composable training objectives
- `metrics.py`, `trainer.py`: fitting, reporting, and restart state
- `solver.py`: forward thermodynamics and equilibrium density solution
- `reciprocal.py`: optional reciprocal-space features and readout support

Run the test suite with:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

The `lj_paper_v1_regression` example freezes one held-out `T=1.5` field and
the released model. It doubles as a regression test for gauge-aligned
density-to-external-potential inference and the fixed-particle-number reverse
solve:

```bash
python -m examples.lj_paper_v1_regression.forward
python -m examples.lj_paper_v1_regression.solve
```

## Scope

Implemented capabilities include multicomponent density tensors, optional LDA,
GGA, and reciprocal-space readouts, first direct correlations, selected rows of
the second direct correlation, supervised or latent chemical-potential
training, and fixed $N$/fixed $\mu$ equilibrium solves. The compact example
uses the validated short-range LDA + Cartesian-invariant model and does not
enable GGA, reciprocal-space terms, or direct $c^{(2)}$ supervision.
