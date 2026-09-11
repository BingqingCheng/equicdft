# Optional particle- and dipole-density functional

Status: **exploratory implementation**, not a trained or validated polar-fluid
model. Branch `feature/polarization-density` starts from local main `e401a47`.
The existing scalar-density representation and saved LJ model remain usable.

Run the small synthetic example from the repository root:

```bash
PYTHONPATH=src python examples/polarization_density/example.py
```

## Reading the implementation

The data flow is `fields → Cartesian moments → invariant features → energy → derivatives`:

1. `GridData` reads and orders `rho` and `dipole_density` on the same grid.
2. `PolarizationFeatures._moments` gathers both fields and sums them against
   **one shared Gaussian/Cartesian basis**. The dipole index stays separate.
3. `_contract` applies the explicit invariant list. Its channel ordering is
   recorded by `feature_names`.
4. `PolarizationReadout` maps these scalars and temperature to per-particle
   free energies, then integrates with density and voxel volume.
5. `GridCACEModel` sums the readout energies and makes **one autodiff call**
   for the requested density and/or polarization responses.

## Fields and tensor moments

The new input is an **electric polar-vector density**
$\mathbf P_t(\mathbf r)$, with units of dipole moment per volume. It is not
the mean dipole per particle. The fields transform jointly as

$$
\rho'_t(R\mathbf r)=\rho_t(\mathbf r),\qquad
\mathbf P'_t(R\mathbf r)=R\mathbf P_t(\mathbf r).
$$

The implementation assumes cubic voxels, $\Delta V=\Delta L^3$, but does not
require equal numbers of voxels along each box dimension. Integer offsets
$\mathbf q$ obey $|\mathbf q|^2\leq q_{\rm cut}^2$. On a cubic lattice the
exact field symmetries are the 48 signed axis permutations (transforming the
box as well when appropriate); arbitrary rotations need resampling.

[`PolarizationFeatures`](../../src/equicdft/polarization_features.py) reuses the
existing Cartesian moment and Gaussian implementation. At central grid $g$,

$$
w_n(\mathbf q)=\frac{e^{-\alpha_n|\mathbf q|^2}}
 {\sum_{\mathbf q'}e^{-\alpha_n|\mathbf q'|^2}},\qquad
M^{\rho}_{nt,\ell}(g)=\sum_{\mathbf q}w_n(\mathbf q)
 \frac{\rho_t(g+\mathbf q)}{\rho_*}\,\mathbf q^{\otimes\ell},
$$

$$
M^{P}_{nt,\ell}(g)=\sum_{\mathbf q}w_n(\mathbf q)
 \mathbf q^{\otimes\ell}\otimes\frac{\mathbf P_t(g+\mathbf q)}{P_*}.
$$

The positive, explicitly supplied $P_*$ is shared across all Cartesian
components. Normalizing by a mean polarization would be inappropriate when
that mean is zero. The $\alpha_n$ are shared by scalar and vector moments and
can be trained in logarithmic form. There is no $\Delta V$ factor in these
normalized averages; quadrature appears only in the energy integration.

For spatial powers through two, write
$s=M^\rho_0$, $\mathbf v=M^\rho_1$, $Q=M^\rho_2$,
$\mathbf p=M^P_0$, $D=M^P_1$, and $H=M^P_2$. The dipole index is independent
of the spatial indices: $D_{ij}\ne D_{ji}$ in general, and only the first two
indices of $H_{ijk}$ are symmetric. Unsmeared center density and center
polarization are additional $s$ and $\mathbf p$ channels.

The finite descriptor list contains delta contractions such as

$$
\mathbf p_a\!\cdot\!\mathbf p_b,\quad
\mathbf v_a\!\cdot\!\mathbf p_b,\quad
\operatorname{tr}D_a,\quad D_a\!:\!D_b,\quad
\operatorname{tr}(D_aD_b),\quad
\mathbf p_a^TQ_b\mathbf p_c,\quad
\mathbf v_a^TD_b\mathbf p_c,\quad
Q_{a,ij}H_{b,ijk}p_{c,k}.
$$

The full explicit list is `_CONTRACTIONS` in the feature module. The labels
$a,b,c$ independently range over radial/species channels; `feature_names`
records their flattened order. Center vectors participate in cross contractions,
not just through their magnitudes. These contractions are invariant under
orthogonal transformations of the tensors, including reflections. They are a
finite, intentionally non-minimal set, **not a complete tensor invariant basis**.
`max_power` supports 0--2 and `max_product_order` supports 1--3; the latter
counts moment factors before the nonlinear MLP, not the final functional's
polynomial degree. Channel combinations grow cubically at order three.

`dipole_reversal_symmetry=True` removes contractions with an odd number of
dipole factors. This enforces the additional symmetry
$F[\rho,-\mathbf P]=F[\rho,\mathbf P]$ at fixed positions. It is **off by
default**: it is not implied by spatial inversion and need not hold for a
general polar molecular fluid. Magnetic axial-vector parity is not used.

## Readout and functional derivatives

[`PolarizationReadout`](../../src/equicdft/readout.py) owns the new features,
appends normalized temperature and outputs one local free energy per particle
and species. It can be the only readout or an additional member of the existing
model readout list alongside scalar CACE, LDA or other energy contributions.
All local representations using the shared neighbor table need the same cutoff.
In the default `free_energy_mode="beta"`, its contribution is

$$
\beta F_{{\rm exc},P}=\Delta V\sum_{g,t}
 \rho_{gt}\,a_{\theta,t}(B^P_g,T/T_*).
$$

For `free_energy_mode="physical"`, the summed readout instead represents
$F_{\rm exc}/(k_B T_*)$. The model multiplies this reduced sum by $k_B T_*$
to obtain physical energy and by $T_*/T$ to obtain $\beta F_{\rm exc}$
before differentiation, exactly as for the existing scalar readouts.
The model's optional `compute_polarization_derivative=True` gives

$$
c^{(1)}_{gt}=-\frac{1}{\Delta V}
 \left.\frac{\partial\beta F_{\rm exc}}{\partial\rho_{gt}}\right|_{P},
\qquad
\texttt{polarization\_derivative}_{gtj}=\frac{1}{\Delta V}
 \left.\frac{\partial\beta F_{\rm exc}}{\partial P_{gtj}}\right|_{\rho}.
$$

The latter uses the **positive** derivative convention. It is not by itself
the external electric field. Responses are taken from the same total energy
graph; there is no separate vector prediction head. Existing `c2` remains a
density--density derivative at fixed polarization, not the full block Hessian.

For batch dimensions denoted `...`:

- `rho`: `[..., n_grid, n_types]`.
- `dipole_density`: `[..., n_grid, n_types, 3]`.
- Invariant features: `[..., n_grid, n_features]`.
- `beta_F_exc`: `[...]`; `c1`: same shape as `rho`.
- `polarization_derivative`: same shape as `dipole_density`.

## Data and scope boundaries

[`GridData`](../../src/equicdft/data.py) optionally reads `dipole_density` from
EXTXYZ as `3*n_types` columns, ordered `type0_x,type0_y,type0_z,type1_x,...`.
The source name can be changed with
`data_key={"dipole_density": "your_column_name"}`. `from_dict` accepts the
canonical `[n_grid,n_types,3]` tensor, retaining its graph. Alternatively attach
the live tensor to the returned dictionary, as in the example. Loader/device
handling uses the existing tensor-dictionary path.

Coarsening averages each Cartesian component, preserving
$\Delta V\sum_g\mathbf P_g$. Excluded voxels must have exactly zero dipole
density; invalid values are rejected before averaging. Scalar datasets without
this field keep their previous behavior.

This implements **excess free energy and derivatives**, not a complete polar
cDFT. In particular:

- The orientational ideal free energy, molecular dipole magnitude, and external
  electric coupling/sign/unit conventions are not specified. A fixed molecular
  moment $\mu_0$ would impose $|\mathbf P_t|\leq\mu_{0,t}\rho_t$; no unprovided
  bound is guessed or silently enforced here.
- Scalar-fluid equilibrium `c1` targets are not auto-generated for polarized
  records. `beta` and supplied `beta_mu` remain metadata. `compute_local_mu`
  and `GridSolver` reject polarization readouts, since their scalar ideal-gas
  thermodynamics would be incomplete. Evaluate the model directly instead.
- No dipolar solver, vector message passing, vector long-range electrostatics,
  or polar-fluid training is added. Number-density-only Fourier stability is
  not a test of the coupled density/polarization Hessian.
- The contraction symmetry assumes an isotropic, parity-even intrinsic
  functional; fixed material axes or chiral terms need additional design.

## Validation record

On 2026-09-10, the full unit-test discovery passed **463/463 tests**, including
**32 new polarization tests** and the retained LJ forward/reverse regression.
The synthetic example also passed. These were CPU checks with Python 3.11
and PyTorch 2.8 in the local `xtal` environment; no CUDA run or polar-fluid
fit was performed. An independent review additionally compared all spatial
and dipole indices against direct neighbor sums for two batches, two species
and three Gaussian channels.

The subsequent readability cleanup retained the same public constructor,
feature ordering, invariant list and normalization. It removed duplicate basis
evaluation, intermediate axis transpositions, separate first-derivative code
paths, and an unnecessary NumPy-to-tensor-to-NumPy conversion. Explicit loops
and moment names replace densely nested expressions; this is a readability
refactor, not a smaller descriptor set.

Before/after comparison of 136 output, gradient and feature-name records
covered all supported power/order combinations, both reversal settings, and
both free-energy modes with/without scalar readouts. Maximum absolute change
was $1.82\times10^{-11}$; maximum change scaled by
$\max(1,\|\mathrm{reference}\|_\infty)$ was $1.99\times10^{-14}$, consistent
with changed floating-point summation order. A permanent direct-neighbor-sum
test now checks the moment construction and single basis evaluation. After
cleanup, **464/464 tests passed**, including **33 polarization tests** and the
LJ regression; the example still passed. No physical assumptions were changed.

The new tests check all 48 lattice transformations, continuous orthogonal
tensor contractions, translations, independent Cartesian indices, species and
batch shapes, dipole reversal, zero-field behavior, trainable-radial gradients,
data coarsening and exclusion, saved-model round trips, and finite-difference
responses in both energy modes at two voxel spacings. Mixed density/polarization
derivative reciprocity is checked separately. Tests use double precision with
explicit tolerances; no fitting run is launched.

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_polarization_*.py'
PYTHONPATH=src python -m unittest discover -s tests -p 'test_lj_paper_v1_regression.py'
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests
```
