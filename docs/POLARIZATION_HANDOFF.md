# Polarization-density implementation: handoff

Verified 12 September 2026. Scope: the optional particle-/dipole-density
extension, not the earlier LJ fitting campaigns or a complete polar-fluid cDFT.

Latest optional extension: [shared charge/dipole Coulomb](POLARIZATION_COULOMB.md)
generalizes the existing reciprocal source, with mixed interactions and
unchanged charge-only defaults. This implementation does not start a new fit.

Subsequent optional extensions: [squared-P LDA](POLARIZATION_LDA.md) and
[one joint-invariant Bχ message](POLARIZATION_MESSAGE.md). The latter passes
polarization-dependent scalar invariant information, not explicit vector
messages. The historical baseline description below remains unchanged.

Subsequent implementation: the user approved the fixed-dipole ideal reference
and coupled canonical solver. See [the new implementation record](POLARIZATION_IDEAL_SOLVER.md)
for this addition and its validation. The original handoff below records the
earlier baseline and its then-open gates.

## 1. Current state and where to resume

**Implemented and tested:** an excess free-energy model of independent scalar
number density and electric dipole density, with invariant descriptors and
variational derivatives. **Exploratory:** physical usefulness of this descriptor
family. No polar-fluid dataset, trained checkpoint, or production benchmark has
been created by this task. No polar fitting or simulation campaign is running
from this implementation work.

- Repository: [BingqingCheng/equicdft](https://github.com/BingqingCheng/equicdft).
- Side branch: [`feature/polarization-density`](https://github.com/BingqingCheng/equicdft/tree/feature/polarization-density).
- Implementation and cleanup: `c7d7f03` (10 September 2026).
- Theory write-up: `ea70cf0` (11 September 2026).
- Verified remote branch head on 12 September:
  `ea70cf0491b879217a13cfe3d2a1c35aedb00245`, matching local HEAD.
- Base: local main `e401a47` at branch creation. This was not necessarily the
  remote main revision. The polarization implementation has not been merged
  into local main; preserve the side-branch workflow unless asked otherwise.
- Local worktree:
  `/Users/tc/IST-Cheng Dropbox/Bingqing Cheng/cDFT/.worktrees/equicdft-polarization`.

The worktree was clean before adding this handoff and its navigation links.
This handoff is a subsequent documentation addition, not part of `ea70cf0`.
No commit or push is implied by creating it.

Start with the [theory note](POLARIZATION_DENSITY.md), then the
[runnable synthetic example](../examples/polarization_density/example.py).
The [implementation record](../examples/polarization_density/README.md)
contains equations, limitations and the earlier cleanup comparison.

## 2. The scientific contract to preserve

1. **Polarization is an electric polar vector.** `dipole_density` means dipole
   moment per volume, not orientation per particle or three scalar species.
   Transform positions and vectors together, including reflections:
   $\rho'(R\mathbf r)=\rho(\mathbf r)$ and
   $\mathbf P'(R\mathbf r)=R\mathbf P(\mathbf r)$.
2. **Keep spatial and dipole indices distinct.** A dipole moment of spatial
   power $\ell$ has tensor rank $\ell+1$. For example $q_xP_y$ and $q_yP_x$
   are distinct. Only the spatial indices are symmetric.
3. **One scalar energy generates every response.** Sum enabled readout energies
   before differentiating. At fixed temperature and cubic voxel volume,

   $$
   c^{(1)}=-\frac{1}{\Delta V}
   \left.\frac{\partial\beta F_{\mathrm{exc}}}{\partial\rho}\right|_P,
   \qquad
   \mathbf g^P=+\frac{1}{\Delta V}
   \left.\frac{\partial\beta F_{\mathrm{exc}}}{\partial\mathbf P}\right|_\rho.
   $$

   `polarization_derivative` is $\mathbf g^P$: an **excess** response with a
   positive sign, not the external electric field. Existing `c2` remains a
   selected density–density derivative row at fixed polarization.
4. **Do not reuse scalar ideal thermodynamics silently.** Polarized records
   omit automatic scalar-fluid `c1`/`c1_plus_beta_mu` targets. Polarization models
   reject `compute_local_mu=True` and the existing `GridSolver`. Their missing
   orientational ideal entropy is a scientific issue, not a convenience check
   to delete.
5. **Tensor invariance and lattice covariance are distinct.** Delta contractions
   are algebraically O(3)-invariant. Exact field transformations on the fixed
   lattice are commensurate signed permutations; arbitrary rotations need
   resampling. Cubic voxels are required, not a cubic box. Axis swaps of an
   orthorhombic box also exchange its dimensions.

These conventions are defined in the [theory note](POLARIZATION_DENSITY.md)
and tested in the linked source/tests below. The fixed-magnitude orientational
ideal reference and equilibrium loss in that note are **proposals**, not code
that is already present.

## 3. Public API and data flow

### Data

`GridData.from_xyz(...)` accepts the optional key
`data_key={"dipole_density": "source_column_name"}`. EXTXYZ vectors are
flattened by species: `type0_x,type0_y,type0_z,type1_x,...`. Species count still
comes from `rho` or `V_ext`, never from counting Cartesian components.

| Quantity | Shape, with optional leading batch dimensions |
|---|---|
| `rho` | `[..., n_grid, n_types]` |
| `dipole_density` | `[..., n_grid, n_types, 3]` |
| `local_density_index` | `[..., n_grid, n_neighbors]` |
| invariant features | `[..., n_grid, n_features]` |
| `beta_F_exc` | one scalar per field |
| `c1` / `polarization_derivative` | same shape as `rho` / `dipole_density` |

`from_dict` is still geometry-first: attach `rho` afterward. It optionally
accepts a canonical dipole tensor without severing its autograd graph.
`grid_info=model.grid_info` can configure matching geometry/unit conventions
for either constructor. `temperature` is required; use explicit physical units
rather than copying the example's LJ-like constants.

Componentwise block averaging preserves the integrated dipole vector.
`excluded_mask=True` requires exactly zero polarization; checks occur before
averaging so opposing invalid values cannot cancel undetected. Absent
polarization remains absent, rather than being filled with zeros. Batch records
for a polar model must consistently contain the necessary fields.

### Descriptors and readout

```python
PolarizationFeatures(
    mean_density, dipole_density_scale,
    cutoff_grid=3, max_power=2, max_product_order=3,
    radial_exponents=(0.125,), trainable_radial_exponents=True,
    n_types=1, dipole_reversal_symmetry=False,
)
PolarizationReadout(features, hidden_sizes=(32, 16))
```

- `mean_density` and `dipole_density_scale` are fixed positive scalar scales.
  The latter is the same for all xyz components; neither mean-vector nor
  per-axis normalization is appropriate.
- The shared weights are
  $w_n(\mathbf q)=e^{-\alpha_n|\mathbf q|^2}/\sum_{\mathbf q'}e^{-\alpha_n|\mathbf q'|^2}$.
  Positive trainable exponents use logarithmic parameters. The cutoff is
  inclusive and measured in integer grid steps.
- Supported spatial powers are 0–2; product orders are 1–3. Radial/species
  channels combine independently. This explicit invariant set is finite,
  non-minimal and **not complete**. Channel counts grow cubically at order
  three; no large-system performance study exists.
- The Gaussian stencil includes the center; raw center rho/P are additionally
  appended. Center vectors participate in cross contractions with neighbors.
- `dipole_reversal_symmetry=True` removes odd-P products. This is an optional
  **global** fixed-position reversal, not implied by spatial inversion and not
  a separate reversal symmetry for each species.
- The readout owns the joint descriptor and appends normalized temperature.
  It integrates $\Delta V\sum_{g,t}\rho_{gt}a_{gt}$ and can coexist with scalar
  CACE/LDA readouts. Its density-only features mean that it does not necessarily
  vanish at zero polarization; the additive decomposition is not unique.

### Model

Add `PolarizationReadout(...)` to the normal `readout` list. For a standalone
polarization readout, set `a_features=None, b_features=None`. Combined local
representations must share a cutoff. Explicit neighbor gathering is currently
required; no polarization FFT backend or vector message passing is added.

`compute_polarization_derivative=False` is the model-constructor default and
may be overridden per forward call. Density, polarization or both responses
use one requested-field list and one autodiff call. Gradients are enabled for
requested responses even inside `torch.no_grad()`, and higher-order graphs
are retained during training. A P-independent invariant selection returns
shape-correct zero P derivatives, including subsequent derivatives.

Energy conventions are unchanged:

- Default `free_energy_mode="beta"`: readout sum is $\beta F_{\mathrm{exc}}$.
- `free_energy_mode="physical"`: sum is $F_{\mathrm{exc}}/(k_BT_*)$;
  multiply by $T_*/T$ before taking the same beta-scaled derivatives.
- Normalized descriptor averages contain no $\Delta V$; energy integration
  introduces it once and functional derivatives divide it out.

## 4. Source map

| Responsibility | Authoritative files |
|---|---|
| Optional field parsing, ordering, coarsening, validation | [data.py](../src/equicdft/data.py), [_data_helpers.py](../src/equicdft/_data_helpers.py) |
| Shared moments and explicit invariant list | [polarization_features.py](../src/equicdft/polarization_features.py) |
| MLP and density-weighted integration | [readout.py](../src/equicdft/readout.py), [energy.py](../src/equicdft/energy.py) |
| Aggregation, response flags, autodiff and geometry checks | [model.py](../src/equicdft/model.py) |
| Guard against incomplete scalar thermodynamics | [solver.py](../src/equicdft/solver.py) |
| Data tests | [test_polarization_data.py](../tests/test_polarization_data.py) |
| Tensor and symmetry tests | [test_polarization_features.py](../tests/test_polarization_features.py) |
| Variational, serialization and integration tests | [test_polarization_model.py](../tests/test_polarization_model.py) |

The existing scalar A/B interfaces and model compatibility fallbacks were
preserved. No code, models or results on the main checkout were replaced.

## 5. Validation and restart commands

Rechecked on 12 September at `ea70cf0`: **464/464 tests passed**, including
**33 polarization tests** (13 data, 8 descriptor, 12 model) and the retained
LJ forward/reverse regression. The synthetic example passed with outputs
`beta_F_exc: ()`, `c1: (96,1)`, `polarization_derivative: (96,1,3)` and finite
parameter gradients through the P derivative.

Environment: CPU, Python 3.11.14, PyTorch 2.8.0, ASE 3.26.0, executable
`/Users/tc/miniconda3/envs/xtal/bin/python`. CUDA and the package's minimum
supported dependency versions were not revalidated for this feature.

Run from the worktree/repository root with that environment active:

```bash
PYTHONPATH=src python examples/polarization_density/example.py
PYTHONPATH=src python -m unittest discover -s tests -p 'test_polarization_*.py'
PYTHONPATH=src python -m unittest discover -s tests -p 'test_lj_paper_v1_regression.py'
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests
```

Tests cover all 48 lattice operations, independent orthogonal tensor rotations,
translations, direct neighbor sums, species/batch ordering, reversal symmetry,
finite differences in both energy modes at two spacings, mixed-Hessian
reciprocity, zeros, saved models and the scalar regression. The full suite emits
an existing tensor-to-scalar warning from `test_fourier_amplitudes.py`; it is
not a polarization failure.

The earlier cleanup record reports agreement of 136 before/after output,
gradient and feature-name records, with maximum scaled difference
$1.99\times10^{-14}$. This historical comparison is documented in the example
README; its temporary baseline is not a portable retained regression fixture.
The direct-neighbor-sum test is the permanent independent moment check.

## 6. Decision history and superseded approaches

| Stage / origin | Decision and refinement | Evidence or reason |
|---|---|---|
| User hypothesis | Add particle and dipole densities while preserving invariant local energies. | Preserve a single scalar functional rather than independent response heads. |
| Descriptor design and review | Retain the independent dipole index and use explicit low-order delta contractions. | Scalar-species processing of Px/Py/Pz would impose the wrong transformation law; magnitude-only input loses relative orientation. These were rejected designs, not failed fitted models. |
| Symmetry review | Do not assume magnetic parity or global P reversal; use one vector scale. | Inversion acts on geometry and dipoles together; fixed-position reversal is a separate microscopic assumption. |
| Integration review | Use the existing additive `EnergyReadout` interface, with explicit thermodynamic guards. | Avoid replacing the scalar workflow or silently applying its incomplete ideal reference. |
| Data round-trip test | Add lookup in stored ASE calculator results for reserved property names. | A custom `polarization` column initially failed because ASE relocated it out of `atoms.arrays`. The retained data test covers the fix. |
| User-requested readability cleanup | One shared basis evaluation, direct axis order, one autodiff path; no NumPy→Torch→NumPy round trip. | Preserve the same invariant list and outputs while making the calculation easier to inspect. |
| 10 September | Implementation and cleanup committed as `c7d7f03`. | Source and regression tests. |
| 11 September | Theory note committed as `ea70cf0`; both commits pushed as a side branch. | Git history and remote branch verification. |
| 12 September | Handoff consultations and full regression rerun. | Current handoff validation and consultation record. |

What helped, based on retained tests and review: explicit shapes, independent
tensor-index checks, narrow ownership of modules, and an isolated worktree.
The main near-friction risks were confusing vector components with species,
equating inversion with P reversal, and assuming scalar ideal thermodynamics
already solved the polar problem. The initial separate rho/P gradient paths
and dense axis manipulation were simplified, not scientifically superseded.

## 7. Unresolved work and recommended next step

**Best next step: agree on the microscopic polar model and its thermodynamic
contract before generating targets or starting a fit.** Specify the molecular
reference position and dipole definition, fixed versus induced moment,
units, external coupling, resolved fields, and ideal orientational reference.
The fixed-magnitude example in the theory note is a candidate, not an adopted
system choice.

Then the smallest useful validation would be a noninteracting dipole fluid in
a known field: recover its analytic orientational response before introducing
learned excess terms. This is a recommended next experiment, not an authorized
or running calculation.

Still missing or unvalidated:

- Orientational ideal functional and its density/P derivatives; physical
  realizability constraints such as $|P_t|\leq m_t\rho_t$ when applicable.
- Explicit electric-field input convention, coupled equilibrium targets/loss
  and coupled rho/P minimization or Euler solver.
- Coupled Hessian/response observables. Density-only Fourier stability cannot
  establish stability against arbitrary polarization or mixed perturbations.
- Vector long-range electrostatics, higher orientational moments, vector MP,
  large-channel performance, GPU checks and polar-fluid predictive accuracy.

Do not treat the implemented local descriptor as a dielectric model, remove
the solver guards to start training, or infer a polar dataset/checkpoint from
the scalar LJ regression. Any simulation/training campaign needs new explicit
physical inputs and user authorization.

## 8. Consultation and artifact record

All three relevant implementation agents were consulted on 12 September before
writing this handoff:

- `polar_data` (Hubble): optional data interface, ASE round-trip issue,
  coarsening, exclusions and target suppression.
- `polar_model` (Fermat): readout/model integration, signs, normalization,
  graph lifetime, guards and scalar compatibility.
- `polar_descriptor_review` (Kierkegaard): invariance, parity, tensor indices,
  contraction scope, design provenance and independent direct-sum review.

All responded; none was unavailable. Their conclusions were checked against
the current source, tests, Git history, remote revision and the fresh root-run
regression/example. No unresolved disagreement was found. Earlier broad LJ/HPC
tasks are outside this handoff's scope; their status was not audited here.

The primary portable documents are this handoff, the
[theory note](POLARIZATION_DENSITY.md) and the
[example record](../examples/polarization_density/README.md). Local generated
rendering artifacts are additionally available under
`/Users/tc/IST-Cheng Dropbox/Bingqing Cheng/cDFT/output/polarization_density/`
(`POLARIZATION_DENSITY.html`, previews, `render.cjs`, `render-check.json`,
and a pinned MathJax bundle). They were not pushed to Git and are not needed
to run or test the model. Use the Markdown as authoritative; do not rely on
the temporary localhost preview server for future access.
