# Polarizable metal electrodes

`MetalWall` relaxes Gaussian electrode charges in response to liquid density,
subject to a fixed total charge for each declared electrode group. It supports
grid-selected sites or independent Cartesian coordinates in an orthorhombic,
three-dimensionally periodic cell. Metal charge is separate from fluid species.

**Experimental:** this is a general software capability, not a validated
electrolyte or electrode material model. Choose charges, geometry, widths,
dielectric-dependent Coulomb amplitude and numerical resolution explicitly.
Independent fixed electrode potentials, open/slab boundaries, fitted metal
hardness and microscopic image/correlation free energy are not implemented.

## 1. Attach electrodes to a liquid model

This example uses independent sites. Numbers illustrate the API, not a
physical parameterization; coordinates and energies are Coulomb-reduced.

```python
from equicdft import GridData, MetalWall, MetalElectrodeReadout, read_metal_sites

# The liquid frame supplies densities, temperature, V_ext and excluded_mask.
data = GridData.from_xyz(
    "liquid.extxyz", boltzmann_constant=1.0,
    include_local_density_index=False,
)[0]
data.update(read_metal_sites(
    "metal.extxyz",
    origin=(0.125, 0.125, 0.125),  # physical position of liquid grid index zero
))
wall = MetalWall(
    charges=[1.0, -1.0],     # liquid valencies; any number of species is supported
    sigma=0.35,              # metal Gaussian standard deviation, length units
    liquid_sigma=0.0,        # unsmeared liquid voxel source
    coulomb_amplitude=1.0,   # physical Coulomb coefficient, not beta-scaled
    boundary="periodic",
)
state = wall(data)
q_sites = state["metal_site_q"]  # integrated e/site, not charge density

# Include this readout in GridCACEModel's readout list at construction.
electrode_readout = MetalElectrodeReadout(wall, contribution="correction")
```

`correction` is appropriate when the fitted liquid functional already includes
its full liquid electrostatics. See [energy accounting](#3-energy-and-normalization)
before selecting `total`. Keep the actual reduced temperature in the input;
`boltzmann_constant=1` is not an instruction to replace temperature by one.

## 2. Geometry, exclusions and charge constraints

Choose **one** site representation. Both use `metal_group_ids`,
`metal_total_charge` and `metal_charge_units="e"`. Group totals are integrated
charges, not potentials or densities. Every present group must be declared
exactly once; missing charge does not mean zero. A zero-total group can still
polarize. Charges on individual sites may have either sign.

### Explicit coordinates

| Field | Contract |
|---|---|
| `metal_positions` | `[M,3]` or `[...,M,3]` Cartesian positions in the liquid grid's length units, **relative to grid index zero**. Not integer grid indices. |
| `metal_site_groups` | `[M]` or `[...,M]` nonnegative integer labels; IDs need not be contiguous. |
| `metal_group_ids`, `metal_total_charge` | Matching group labels and integrated totals, in declared output order. |
| `excluded_mask` | Independent Boolean liquid-voxel exclusion. Sites do not infer a solid volume or exclude neighboring voxels. |

A separate metal EXTXYZ can supply both coordinates and constraints:

```text
2
Properties=species:S:1:pos:R:3:metal_site_groups:I:1 metal_group_ids="0 3" metal_total_charge="0.1 -0.1" metal_charge_units=e
X 0.2 0.4 0.7 0
X 1.6 0.4 0.7 3
```

`read_metal_sites` subtracts the supplied `origin` without wrapping sites,
converting units or changing the electric field. Its default origin is zero;
chemical symbols do not set electrode groups. Plain XYZ files can instead use
`read_metal_sites(path, site_groups=0, group_ids=[0], total_charge=[0.0],
charge_units="e")`. Scalar `site_groups` assigns every site to that group;
explicit arguments override the corresponding frame metadata.

Positions are stored in float64; conversion cannot recover precision lost in
float32 input. Shared geometry can broadcast over a batch; per-frame geometry
requires matching site counts. Empty/nonfinite/coincident periodic sites,
position tensors requiring gradients, and explicit sites combined with an
active `metal_mask` are rejected. An all-negative legacy mask is harmless.
Site positions are fixed: position derivatives are not supported.

`GridData.from_dict` accepts the site arrays; liquid EXTXYZ frames can also
hold them as frame metadata, not per-voxel columns. Explicit-site grid
coarsening is rejected: supply a deliberately matched grid/origin instead.

### Grid-selected sites

Add `metal_mask:I:1` to the liquid EXTXYZ `Properties` declaration. Negative
values mean nonmetal; nonnegative values are electrode group IDs, including
zero. For example, the comment-line fragment

```text
metal_group_ids="0 3" metal_total_charge="0.1 -0.1" metal_charge_units="e"
```

assigns totals to groups 0 and 3. `GridData.from_xyz` / `from_dict` retain these
fields; scalar metadata normalize to one-element arrays. Ordinary `data_key`
overrides apply. Batched frames require matching metadata keys and group counts,
although labels/totals may differ. Categorical metal-mask coarsening is rejected.

Liquid accessibility is the complement of the union of `excluded_mask` and
`metal_mask >= 0`. Densities must be exactly zero there and remain zero during
inverse solves. Subsample charge sites if desired, but keep the full solid
region in `excluded_mask`. Metal charge has no ideal-gas entropy and does not
count toward fluid particle numbers.

### Neutrality and ensemble

For metal-containing periodic cells, combined liquid plus electrode charge
must be zero. Supply fixed fluid particle numbers compatible with all group
totals. The check does not redistribute charge or add a background.
Ordinary fixed-chemical-potential updates do not enforce this extra neutrality
condition; general charge-constrained grand-canonical metalwall updates are
not implemented. Fixed group totals already determine metal charge, so
neutrality is a consistency check, not another dependent constraint row.

## 3. Energy and normalization

Let `DeltaV` be the product of all three liquid grid spacings and `z_i` the
liquid valencies. The distinction between densities and integrated charges is:

```text
liquid_charge_density[g] = sum_i z_i * rho[g,i]       # e / length^3
q_liquid[g]              = DeltaV * liquid_charge_density[g]  # e / voxel
sum(q_m in group a)      = metal_total_charge[a]     # e; no DeltaV factor
```

The metal unknowns are integrated Gaussian coefficients, not a voxel density.
Dividing grid-mode `metal_q` by `DeltaV` gives coefficient density, not the
spatially Gaussian-smeared metal density. Do not add another voxel factor to
the charge-coefficient energy quadratic.

For liquid potential `b` at metal sites and metal interaction matrix `A`,

```text
U_Coulomb = U_ll + q.T @ b + 0.5 * q.T @ A @ q
U_field  = sum_m q_m * v_m
```

| Readout contribution | Energy added to the liquid functional |
|---|---|
| `correction` (default) | `q.T @ b + 0.5*q.T @ A @ q + U_field` |
| `total` | `U_ll + q.T @ b + 0.5*q.T @ A @ q + U_field` |

Use `total` only if the liquid residual was defined against that exact `U_ll`.
Adding it to an already complete liquid functional double-counts electrostatics.
`correction` preserves the existing SR/LR split, including a damped liquid LR
term and its learned residual. Diagnostic `coulomb_energy` always reports the
full Coulomb-only sum, **not** the chosen readout contribution. Neither it nor
the other physical-energy diagnostics should be added again to `beta_F_exc`.

`coulomb_amplitude` has units energy × length / e². Unlike the beta-energy
amplitude in `LongRangeReadout`, it is physical: do not insert an extra `1/T`.
`GridCACEModel` supplies beta/energy conventions to the adapter. Beta mode
multiplies by beta; physical mode divides by `reference_energy=k_B*mean_temperature`
before the model's existing conversion. Standalone contexts need the same scale.

Density differentiation passes through the charge solve. Second derivatives
include induced electrode response, not a frozen-charge Hessian. Model `c1/c2`
therefore depend on electrode geometry and constraints; they are not bulk
correlations. A mean-density quadratic still misses microscopic image and
correlation effects: zero mean liquid charge does not imply zero ion image
attraction. This closure requires separate physical benchmarking.

## 4. Applied constant field

Fixed-charge groups are not independently voltage-controlled electrodes.
To allow charge transfer between two metal regions at fixed **combined** charge,
assign their sites to one group (for example group 0 with total charge 0).
Apply a consistent field to both liquid and metal:

- Liquid: include the species field potentials in `V_ext`, with walls/other
  external potentials. MetalWall never modifies `V_ext` or places induced,
  density-dependent metal potentials in that input.
- Metal: supply `metal_external_field` and optionally `metal_field_origin`.

Both are Cartesian vectors, shared `[3]` or per-frame `[...,3]`. Missing field
means zero; missing origin means zero. An origin without a field is rejected;
the loader fills absent fields with zeros in mixed biased/unbiased batches.

The field has units **energy / (e × length)**, not beta energy and not
automatically V/Angstrom. In Coulomb-reduced units use `E*=e*sigma*E/E_C`.
`metal_field_origin` is the wrapping-interval center relative to grid index
zero. For either explicit site coordinates or `r_m = grid_index*spacing`,

```text
d_m    = r_m - metal_field_origin
r_wrap = d_m - L * round(d_m/L)   # componentwise
v_m    = -E dot r_wrap
```

The XYZ reader's `origin` converts coordinate frames; `metal_field_origin`
selects the field branch. Neither changes Coulomb separations. If physical
grid centers are `(g+1/2)*dx` and the desired wrapping center is the physical
cell origin, `metal_field_origin=-dx/2`.

This is a periodic sawtooth potential, not a globally periodic linear potential.
Keep branch cuts away from metal sites. Moving a cut across sites can change
the voltage condition and is not generally a gauge shift. For electrodes near
opposite ends of `[0,Lz)`, the top-side image is shifted by `-Lz`; shifting
the bottom image by `+Lz` is equivalent for zero combined metal charge.
The corresponding finite-field convention is
`E_z=(V_bottom-V_top)/Lz`, using the full periodic length, not the slit gap.

Both readout modes include `U_field` exactly once. Do not add another
independent-electrode voltage-source term for this same construction.

## 5. Solver constraints

`homogeneous_axes` restricts **liquid density variations** during minimization.
The option is general, including systems without electrodes. Examples assume
an existing `GridSolver` and desired species counts `N`:

```python
solver.solve(data, method="minimize", particle_numbers=N, homogeneous_axes="x")
solver.solve(data, method="minimize", particle_numbers=N, homogeneous_axes="y")
solver.solve(data, method="minimize", particle_numbers=N, homogeneous_axes=["x", "y"])
solver.solve(data, method="minimize", particle_numbers=N, homogeneous_axes=["x", "z"])
solver.solve(data, method="minimize", particle_numbers=N, homogeneous_axes=["x", "y", "z"])
```

Each selected axis is a direction in which density is **constant**. Thus
`["x","y"]` leaves only z dependence. Names denote grid axes (x=0, y=1, z=2),
not species or electrode coordinates. None/empty list/tuple is unrestricted.
Legacy integer axes and mixtures such as `["x",1]` work; invalid or duplicate
axes, including `["x",0]`, are rejected. There is no separate `restrict` keyword.
This option applies only to `method="minimize"`.

The grid must be complete and rectangular, described by `grid_size` and
integer `grid_positions`; arbitrary row order is supported. Accessibility must
be constant along restricted axes: incompatible masks are rejected, not averaged.
The solver averages initial density and each `c1`, averages fixed `V_ext` once,
and restores original row order after projection. Existing positivity, fixed
species counts and per-species upper bounds still apply. Armijo tests the
full, unmodified objective. For the ideal gas with varying external fields,
the constrained solution is proportional to `exp(-beta*average(V_ext))`, not
`average(exp(-beta*V_ext))`. Independent constant species shifts remain valid
canonical gauges.

The model and electrode geometry remain three-dimensional; charges relax on
every evaluation. Homogeneous-density projection supports arbitrary component
counts and both canonical/grand-canonical solvers, but does not remove the
metalwall neutrality limitation described above.

With nonempty constraints, ordinary residual/convergence keys refer to the
**constrained** problem. Inspect unrestricted diagnostics separately:

```text
full_euler_lagrange_residual
full_max_euler_lagrange_residual
full_rms_euler_lagrange_residual
full_converged
solver_homogeneous_axes            # normalized integer indices
```

A constrained `converged=True` is not proof of full-dimensional equilibrium.
Choosing planar liquid density near discrete electrodes is an application
approximation, not a metalwall requirement. Averaging and fixed-number roundoff
correction can leave tiny single-voxel symmetry deviations; this option does
not waive any declared numerical acceptance threshold.

## 6. Outputs

The following diagnostics come from `wall(data)`, and pass through model
evaluation, `GridSolver.evaluate` and the final `GridSolver.solve` result.
They correspond to the density of that evaluation, not a previous iteration.

| Key | Meaning |
|---|---|
| `liquid_charge_density`, `q_liquid` | Full-grid liquid charge density and integrated voxel charges. |
| `metal_site_q`, `metal_positions`, `metal_site_groups` | **Explicit mode:** site coefficients `[...,M]`, coordinates `[...,M,3]`, labels `[...,M]`. |
| `metal_q`, `q_mw` | **Grid mode only:** integrated full-grid metal coefficients (zero elsewhere) and `q_liquid + metal_q`. Explicit mode has no fake grid-charge output. |
| `metal_charge`, `metal_potential` | Group totals in e and solved potentials in energy/e, in declared group order. |
| `coulomb_liquid_energy`, `coulomb_cross_energy`, `coulomb_metal_energy` | Physical `U_ll`, `q.T@b`, `q.T@A@q/2`. |
| `coulomb_energy` | Sum of those three Coulomb terms. |
| `metal_external_energy` | Physical imposed field energy; zero without field, separate from `coulomb_energy`. |
| `charge_residual`, `potential_residual` | Maximum absolute group-total and site-equipotential errors per field. |

Energy-only evaluation respects `torch.no_grad()`; derivative/training calls
retain charge-response gradients. Readouts expose auxiliary outputs through
`EnergyReadout.energy_and_outputs(context)`; scalar-only readouts retain the
default `(energy(context), {})`. Auxiliary keys must not collide with reserved
model outputs or another readout. Float32/float64 require device FFT/LU support;
no additional dependency is introduced.

## 7. Numerical electrostatics

The charge basis has form factor `g_sigma(k)=exp(-sigma²*k²/2)`. Use
`K(k)=4*pi*coulomb_amplitude/k²` on the full FFT domain, omitting k=0.
Liquid/liquid, liquid/metal and metal/metal kernels multiply K by
`g_liquid²`, `g_liquid*g_metal` and `g_metal²`, respectively. Widths describe
individual Gaussian standard deviations, not a combined damping-kernel width.
The metal Gaussian self term is retained; the mean-density liquid quadratic
has no point-particle self subtraction. Even `liquid_sigma=0` is a finite-grid
spectral source, not an exact point-particle Ewald sum. Check grid/cutoff and
Gaussian-width convergence before scientific use.

The inverse-FFT potential includes `1/DeltaV`; combined with FFT normalization
this gives the reciprocal-sum factor `1/cell_volume`. Grid-mode matrix entries
sample `G=real(ifftn(K_mm))/DeltaV` at periodic site separations; the self term
is `G[0,0,0]`. With group incidence matrix `C[m,a]`, solve

```text
[ A   C ] [ q      ] = [ -b-v ]
[ C.T 0 ] [ lambda ]   [   Q  ]
```

Here `b` is the liquid **scalar potential** conjugate to metal charge, not
the vector electric field; `v=0` without an imposed field. Group potentials
are `-lambda`, including the imposed potential. Charges vary across sites
while each group is equipotential.

Nonfinite, singular or inaccurate solves raise errors; no hidden diagonal
jitter repairs singular geometry. With dtype precision `eps`, neutrality
allows `tolerance + 64*eps*(1+sum(abs(q_liquid))+sum(abs(Q)))` in e.
Charge residuals allow `(tolerance+64*eps)*(1+max(abs(Q)))`; potential residuals
allow `(tolerance+64*eps)*(1+max(abs(b+v))+max(abs(Aq)))`. Check tolerances in
physical units. Without metal the ordinary liquid LR removed-zero-mode
convention remains; the metal-containing neutrality check is not applied.

### Explicit-site spectral continuation and cost

Off-grid sites use the same reciprocal domain, kernels, self term and charge
solve. Each coordinate splits into an integer index and a fractional offset.
One shifted inverse FFT per shared offset samples liquid potential, with its
exact density-autograd adjoint. Matrix blocks use the **difference** of offsets,
not a product of separately interpolated fields. On even axes the Nyquist
phase is `cos(k_Nyquist*delta)`, equivalent to half-weighted positive/negative
endpoints. This real symmetric continuation matches the on-grid operator;
it is not real-space interpolation or nearest-grid assignment.

Only float64-roundoff-equivalent offsets are grouped, within
`32*eps64*max(grid_size)` per axis in grid units. The first actual offset is
retained, not rounded to a mesh. Original coordinates remain in outputs,
cache identity and field work; float32 input does not permit a looser grouping.
Check reciprocal-cutoff convergence at fixed physical positions.

The latest exact geometry/operator LU is cached; geometry, constraints ordering,
grid/spacing, widths, amplitude, dtype or device changes invalidate it.
Runtime caches are not serialized; charges/RHS are always recomputed and
differentiable. Reuse `Aq` for residuals and metal energy. Grid lookup stores
O(G+M²), not one full grid per metal site. For C distinct fractional offsets,
shifted phases need O(C*G), matrix construction O(C²) grid transforms, and the
dense constrained factorization O((M+groups)²) memory with cubic work.
Regular layers may have few offsets; arbitrary irregular sites can be costly.
This is not a scalable large-electrode solver.

## 8. Validation

```sh
PYTHONPATH=src python -m unittest discover -s tests -p 'test_metal*.py'
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
PYTHONPATH=src python -m examples.lj_paper_v1_regression.forward
PYTHONPATH=src python -m examples.lj_paper_v1_regression.solve
```

Tests cover independent Fourier/KKT references, grid/off-grid equivalence,
energy and first/second density derivatives, masks/counts/group constraints,
field/gauge behavior, metadata/XYZ precision/batching, output propagation,
cache behavior, and inverse recovery from two initializations. Homogeneous-axis
tests also cover analytic mixture/grand-canonical solutions, reordered grids,
axis names/indices, bounds, invalid masks and electrode directional derivatives.
The retained LJ checkpoint and tolerances are unchanged. Passing software
tests is not evidence of converged EDLs or accurate physical electrode response.
