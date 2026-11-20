# Vanilla local-density-patch baseline

`DensityPatchReadout` feeds an ordered local density patch directly into a shared
SiLU MLP. No A/B descriptors, LDA, radial basis, rotation averaging, or augmentation
are needed. It retains the scalar-free-energy formulation and translation symmetry,
but does **not** enforce lattice rotations or reflections. This is an implemented
baseline architecture, not yet evidence of its performance relative to the fitted
LDA+CACE models.

For grid center g the network sees

\[
 x_g = \big[\{\rho_{g+q,a}/\rho_{\rm ref}\}_{q,a},\ T/T_{\rm ref}\big],
 \qquad |q|^2\leq q_{\rm cut}^2.
\]

Offsets use [the canonical spherical stencil](../src/equicdft/geometry/stencil.py),
including the center exactly once; species vary fastest within each offset. At
cutoff 4 there are 257 offsets, giving 258 inputs for a single species. The default
hidden sizes `(32, 16)` then give 8,833 parameters. Periodic indices may repeat when
the stencil spans more than half a box, following the existing stencil convention.

The MLP returns one per-particle value per species and integrates

\[
 E_\theta[\rho,T] = \Delta V\sum_{g,a}\rho_{g,a}\,\mathrm{MLP}_\theta(x_g)_a.
\]

The containing model's `free_energy_mode` is unchanged: `"beta"` interprets E as
beta F_exc; `"physical"` interprets it as F_exc/(k_B T_ref) and applies T_ref/T to
obtain beta F_exc. Both normalization scales belong to the model. The density used
for integration is physical number density, not the normalized network input.

```python
from equicdft import DensityPatchReadout, GridCACEModel, GridData

domain = "sphere"  # Change only this to "cube" for a cubic patch.
model = GridCACEModel(
    readout=[DensityPatchReadout(cutoff_grid=4, domain=domain, hidden_sizes=(32, 16))],
    grid_spacing=0.5,
    mean_density=0.4,       # replace with the training-data density scale
    mean_temperature=1.2,  # replace with the training-data temperature scale
    boltzmann_constant=1.0,  # LJ reduced units, not a universal constant
    free_energy_mode="beta",
    compute_c1=True,
    compute_local_mu=True,
    rho_min=1e-3,
)
frames = GridData.from_xyz("train.extxyz", grid_info=model.grid_info)
```

The same `GridData.from_xyz` call works for both domains with its default
`include_local_density_index=True`: sphere uses that table, while cube ignores
it and constructs its own geometry. Setting it to `False` is an optional
memory optimization for cube, not a second domain selector. Patches gather
from live rho, not detached precomputed density arrays; all overlapping environments
therefore contribute to c1 and c2. The usual model conversions remain
`c1 = -(1/DeltaV) d(beta F_exc)/d rho` and
`c2 = (1/DeltaV) d(c1_reference)/d rho` for a selected reference grid/species.
The existing local-chemical-potential loss, loaders, trainer, serialization, and
solver require no special baseline path. No external potential or chemical
potential is an MLP input; those quantities enter the equilibrium loss as before.

This architecture has a fixed input width and direction-specific weights. Changing
grid resolution is not a justified automatic transfer operation for a fitted patch
network. Extra rotation augmentation or symmetrization would be a separate baseline,
not part of this implementation. For a future comparison, the requested control is
raw-patch-only versus the complete production LDA+CACE model, not CACE-only.
Gathered patches occupy `[..., n_grid, n_neighbors, n_types]`; memory therefore
grows with patch size and batch size rather than a compressed descriptor width.

## Domain selector

`DensityPatchReadout(domain="cube", cutoff_grid=4)` instead reads every integer
offset in `[-4,4]^3`: 729 densities plus temperature, 23,937 parameters for the
same `(32,16)` hidden widths. The default remains `domain="sphere"`, preserving
old configurations and spherical weights. Cube offsets are lexicographic,
with z varying fastest, and include the center exactly once. A cube at this
cutoff keeps the sphere's axial reach but extends to `sqrt(3)*cutoff_grid` at
its corners; it is not a matched-radius or matched-parameter comparison.

Cube geometry is built from `grid_size` and `grid_positions`; a spherical
`local_density_index` is neither needed nor reinterpreted. The periodic index
table is cached for the current shape/device, supports shuffled rows, and is
not serialized as fitted state. Live density gathering and the scalar/c1/c2
conventions are unchanged. Cube inputs can use
`GridData.from_xyz(..., include_local_density_index=False)` to avoid storing
an unused spherical table. No rotational symmetrization, extra LDA or grid
resolution-transfer rule is added by this flag.

Choose the domain when constructing a model. It is saved in the readout's
configuration. Changing the domain changes the input width, so do not mutate
`readout.domain` on an already trained model or reuse spherical weights for
a cubic network.

## Software validation

[The focused tests](../tests/functional/test_patch.py) cover patch ordering and
normalization, species/batch layouts, the integrated scalar, analytic and numerical
c1/c2 factors, backpropagation of the local-mu loss, translation symmetry, lack of
imposed axis-swap symmetry, serialization, model-owned grid metadata, and vacuum.
They also provide a CUDA test that is skipped when CUDA is unavailable.

```bash
python -m unittest tests.functional.test_patch -v
```

No fit or scientific performance benchmark is implied by these software tests.

### Implementation check, 19 November 2026

The new readout/export, this note and the focused test are additive changes to
the existing working tree at base commit `26585dbab814a4e211cc7ed7cb9a19a7bc096eaa`;
pre-existing changes are not part of this baseline implementation. No derivative,
data, trainer, solver, energy convention or existing readout interface was changed.

Validated on CPU with Python 3.8 and PyTorch `2.1.0.dev20230503`:

- New focused suite: 12 passed, one CUDA test skipped (CUDA unavailable).
- The combined command below ran 150 tests: 148 passed, one CUDA skip and one
  unrelated error-message assertion failure. The failure is
  `tests.functional.test_energy.TestDensityWeightedIntegral.test_invalid_shapes_and_volume_are_rejected`:
  it expects `"leading field shape"`, while the unchanged integration helper
  raises `"match leading fields"`. Neither file was modified for the patch readout.
- Both frozen LJ-paper-v1 forward and fixed-N solve regression tests passed.
- `git diff --check` passed. At this implementation-check stage, no training,
  checkpoint replacement, commit or push had been performed.

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src \
/Users/tc/miniconda3/envs/asap/bin/python -m unittest \
  tests.functional.test_patch tests.functional.test_readout \
  tests.functional.test_semilocal tests.functional.test_energy \
  tests.model.test_model tests.core.test_config tests.core.test_documented_api \
  tests.io.test_serialization tests.regression.test_lj_paper_v1
```

A subsequent BRC GPU pilot passed all 13 focused tests, including CUDA, and
verified optimization, checkpoint restart and finite saved-model energy/c1/local-mu
outputs after reload under Python 3.11, PyTorch 2.9.1+cu126 and an NVIDIA L40S.
Training campaigns and their data/checkpoints are kept outside this code repository.
Those exploratory fits do not by themselves establish a converged accuracy or
data-efficiency advantage for either architecture.

Before publication, the same 150-test command was rerun in a clean checkout of
remote main (`26cbe8f`) with only this vanilla patch: **149 passed, one CUDA skip,
no failures**. The historical local error-message mismatch above is not present
on that target base. This also verifies that the readout does not depend on the
unrelated, uncommitted working-tree changes.

### Cubic domain API publication check, 19 November 2026

The optional `domain="sphere"` / `domain="cube"` selector is published as a
focused extension to remote main `800c56d`. The sphere remains the default;
old versioned sphere configurations and parameter shapes are preserved. No
energy, derivative-normalization, dataset, trainer or solver convention changes.
The only affected files are the patch readout, its tests and this document.

The combined command above now runs **156 tests: 155 passed, one CUDA skip**
on the isolated remote-main base, including the retained LJ forward and fixed-N
solve regressions. Tests cover the single-flag common model/data API, cube
ordering and periodic wrapping, shuffled/batched fields, analytic and numerical
c1/c2, loss backpropagation and serialization. The same cubic implementation
also passed prior L40S GPU pilots, including training/restart and finite
energy/c1/c2/local-mu reload checks. No running HPC campaign is modified by
this publication, and no model-quality conclusion follows from the unit tests.
