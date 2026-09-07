# CACE-style metal grids (experimental)

`MetalWall` implements fixed-total-charge electrodes on an orthorhombic,
three-dimensionally periodic density grid. It follows CACE's small
`forward(data)` / cached interaction-matrix / charge-update pattern, followed
by Coulomb-energy evaluation. It does not introduce an electrode species into
the fluid functional.

This is a general software capability, not an accepted electrolyte benchmark.
No electrode material, geometry, charge, dielectric or Gaussian width is chosen
implicitly. The module does not implement independent electrode voltages, open/slab
boundaries, fitted metal hardness, or microscopic image/correlation free energy.

## Data contract

EXTXYZ uses the existing integer grid-coordinate and density conventions. Add
`metal_mask:I:1` to the `Properties` declaration and one integer per grid row:

- Negative: not metal. All negative values have the same nonmetal meaning.
- Nonnegative: electrode group ID; zero is a valid ID. IDs need not be contiguous.

Put group totals in the frame comment line, for example this header fragment:

```text
metal_group_ids="0 3" metal_total_charge="0.1 -0.1" metal_charge_units="e"
```

The first charge belongs to group 0, the second to group 3. Values are
**integrated total charges in elementary-charge units**, not densities or
potentials. Every present group must be specified exactly once. Missing does
not mean zero; explicit zero means a net-neutral, still-polarizable electrode.

`GridData.from_xyz` and `GridData.from_dict` retain these four fields. Single
group scalars normalize to one-element arrays. `data_key` supports ordinary
field-name overrides. Frames in a batch must have matching metadata keys and
group counts, as for other fixed-shape grid data; group IDs and values may
differ by frame. Categorical metal-mask coarsening is rejected: supply the
mask and densities on the requested grid directly.

`excluded_mask` remains a separate Boolean field for insulating walls/other
exclusions. The fluid solver uses its union with `metal_mask >= 0`, while
preserving both input fields. Fluid densities must be exactly zero in metal
and stay zero through inverse iterations. Species counts are normalized over
the accessible domain. The electrode charges are unconstrained in sign and do
not enter the ideal-gas entropy or fluid count constraints.

Use fixed fluid particle numbers compatible with the electrode totals for
inverse calculations. Ordinary fixed-chemical-potential updates do not enforce
the extra liquid-charge condition and can fail the periodic neutrality check;
general charge-constrained grand-canonical updates are not implemented here.

### Optional constant field (CACE convention)

To allow charge transfer between two metal regions while constraining their
combined charge to zero, simply label all selected sites `metal_mask=0` and use
`metal_group_ids=0 metal_total_charge=0 metal_charge_units="e"`. No new constraint
mode is needed. Subsample metal sites independently of the liquid grid;
keep the entire inaccessible region in `excluded_mask`.

Optional EXTXYZ comment-line fields (illustrative values, not an RPM setup):

```text
metal_external_field="0 0 -0.1" metal_field_origin="0 0 -0.25"
```

Both vectors have three Cartesian components; programmatic batches accept
shared `[3]` or per-field `[..., 3]` values. Missing field means zero; missing
origin defaults to zero. An origin without a field is rejected. The loader
fills absent fields with zero when batching biased and unbiased frames.

- `metal_external_field` is **energy/(e * coordinate_unit)**, not beta-scaled
  and not automatically V/Angstrom. In Coulomb-reduced units supply
  `E* = e*sigma*E/E_C`; retain the actual reduced temperature.
- `metal_field_origin` is the center of the field-coordinate wrapping interval,
  in coordinate units **relative to grid index zero**. For canonical integer
  grid indices `g`, spacing `dx`, and cell lengths `L`, the metal coupling uses
  `r = g*dx - origin; r_wrap = r - L*round(r/L)` componentwise, as in CACE.
  If physical grid centers are `(g+1/2)*dx` and the desired wrapping center is
  the physical cell origin, supply `origin=-dx/2`. This metadata does not move
  charge centers in the Coulomb/FFT calculation.
- Place each active field component's branch cut away from the metal sites.
  This is a sawtooth field-potential convention for periodic cells, not a
  globally single-valued periodic linear potential. Moving a branch cut across
  sites can change the voltage condition; it is not generally a gauge shift.

The imposed metal potential is `v_m = -E dot r_wrap,m`. The charge RHS becomes
`[-b-v; Q]` and `metal_potential=-lambda` includes that imposed potential.
For electrodes near opposite ends of a `[0,Lz)` cell, this uses the top-side
image shifted by `-Lz`. It is equivalent to shifting the bottom side by `+Lz`
under zero combined metal charge. In the corresponding LAMMPS finite-field
convention, `E_z=(V_bottom-V_top)/Lz`, using the full periodic length, not the gap.

Supply the liquid field separately through species-dependent `V_ext`, together
with walls/other prescribed potentials. MetalWall never applies the liquid
field, changes `V_ext`, or stores the density-dependent induced metal potential
in it. Both readout modes add **exactly once** the metal field energy
`metal_external_energy = sum_m q_m v_m` alongside their selected Coulomb terms.
This is necessary for variational density derivatives. No extra independent
electrode-voltage work should be added for this same finite-field construction.

## Minimal use

The numbers below are software-example parameters, not a physical electrode
parameterization. In Coulomb-reduced coordinates/energy, the physical Coulomb
amplitude is 1; temperature remains the actual reduced temperature in the
input file, with `boltzmann_constant=1`.

```python
from equicdft import GridData, MetalWall, MetalElectrodeReadout

data = GridData.from_xyz(
    "frame.extxyz", boltzmann_constant=1.0,
    include_local_density_index=False,
)[0]
wall = MetalWall(
    charges=[1.0, -1.0],  # arbitrary number of liquid species is supported
    sigma=0.35,           # metal Gaussian STANDARD DEVIATION, coordinate units
    liquid_sigma=0.0,     # unsmeared liquid grid source
    coulomb_amplitude=1.0,  # physical energy * length, per elementary charge²
    boundary="periodic",  # explicitly choose the supported boundary
)
state = wall(data)
q_combined = state["q_mw"]
physical_coulomb_energy = state["coulomb_energy"]

# Attach to the existing GridCACEModel's readout list at construction:
electrode_readout = MetalElectrodeReadout(wall, contribution="correction")
```

Once that readout is attached, ordinary model evaluation returns the same
charge fields and physical diagnostics without a second charge solve:

```python
outputs = model(data, compute_c1=False)
q_metal = outputs["metal_q"]                    # [..., G], integrated e/site
rho_q_liquid = outputs["liquid_charge_density"]  # [..., G], e/coordinate_unit^3
q_liquid = outputs["q_liquid"]                  # [..., G], integrated e/voxel
```

These outputs also pass through `GridSolver.evaluate` and the final result
of `GridSolver.solve`. They correspond to the density of that evaluation,
not a cached density from a preceding solver iteration. Energy-only calls
respect `torch.no_grad()`; training calls retain charge-response gradients.
The generic `EnergyReadout.energy_and_outputs(context)` interface defaults
to `(energy(context), {})`, preserving scalar-only readouts. Auxiliary output
names must not duplicate another readout's keys or reserved model outputs.

The module supports float32 and float64 on devices supporting the corresponding
FFT and LU operations. Software validation currently covers CPU in the recorded
environment; GPU behavior is not separately benchmarked. No new dependency is
required.

## Electrostatics and units

From fluid number densities, form voxel charges

```
q_liquid[g] = DeltaV * sum_i charges[i] * rho[g,i].
```

`liquid_charge_density[g] = sum_i charges[i] * rho[g,i]` contains **no** voxel
factor. Thus `sum(q_liquid) = DeltaV * sum(liquid_charge_density)`. In contrast,
the metal unknowns are already integrated charge coefficients:
`sum(metal_q[metal_mask == a]) = metal_total_charge[a]`, without multiplying by
`DeltaV`. No extra voxel factor belongs in the charge-coefficient energy
quadratic. The inverse FFT potential includes `1/DeltaV`; together with FFT
normalization this gives the Fourier-sum factor `1/cell_volume`. Use the
product of all three spacings, including anisotropic grids. Dividing
`metal_q` by voxel volume would give a coefficient-density representation, not
the spatial Gaussian-smeared metal charge density.

Metal unknowns `q_m` already represent integrated charges. Use normalized
Gaussian basis form factors `g_sigma(k) = exp(-sigma² k²/2)` and
`K(k) = 4 pi * coulomb_amplitude / k²` over the full FFT grid, with `k=0` omitted.
The liquid–liquid, liquid–metal and metal–metal blocks multiply `K` by,
respectively,

```
g_liquid², g_liquid * g_metal, g_metal².
```

The finite Gaussian metal self interaction is retained. The liquid term is a
mean-density quadratic energy, with no point-particle self subtraction. This
is a finite-grid spectral representation; grid and Gaussian-width convergence
must be checked before scientific use. `liquid_sigma=0` does not make the
finite FFT grid an exact point-particle Ewald sum.

For CACE's common kernel `exp(-sigma_CACE² k²/2)`, equal basis widths
`sigma=liquid_sigma=sigma_CACE/sqrt(2)` give the same damping exponent, after
the separate unit/normalization conversion. Do not copy LES normalization
constants or identify a kernel damping parameter with a basis width silently.

For metal-containing cells the combined electrode + fluid charge must be zero.
The check allows `tolerance + 64*eps*(1+sum(abs(q_liquid))+sum(abs(Q)))` in e,
where `eps` is the dtype machine precision. It does not redistribute fluid
charge or provide an undeclared background. Without any metal, the liquid-only
quadratic retains the ordinary removed-zero-mode convention of existing LR
features. In particular, neutrality is a consistency check when each group
total is fixed, not an extra dependent constraint row.

## Charge solve and outputs

Construct `A` with unit-charge probes, and `C[m,a]=1` for metal site `m` in
electrode `a`. For electrolyte potential `b` conjugate to the metal Gaussian
charges, solve

```
[ A  C ] [ q      ] = [ -b ]
[ Cᵀ 0 ] [ lambda ]   [  Q ]
```

Each electrode potential is the solved value `-lambda[a]`. The sites within
an electrode are equipotential, but their charges may be nonuniform. No
independent electrode voltage or voltage-source work is imposed in this mode.
The word "field" in the CACE precedent denotes this scalar charge derivative;
the vector electric field would be `-grad(phi)`.

The returned dictionary includes:

| Key | Meaning |
| --- | --- |
| `liquid_charge_density` | `sum_i z_i rho_i`, full-grid liquid charge density in e/coordinate_unit³; zero on metal. |
| `q_liquid`, `metal_q`, `q_mw` | Full-grid integrated coefficients; `metal_q` is zero outside metal and `q_mw=q_liquid+metal_q`. Widths remain specified by the module and mask. |
| `metal_charge`, `metal_potential` | Total charge in e and potential in energy/e, in header group order. |
| `coulomb_liquid_energy`, `coulomb_cross_energy`, `coulomb_metal_energy` | `U_ll`, `qᵀb`, `qᵀAq/2` in physical energy units. |
| `coulomb_energy` | Sum of those three terms. |
| `metal_external_energy` | Imposed field work `-sum_m q_m E dot r_wrap,m`, physical energy; zero if no field. Not part of `coulomb_energy`. |
| `charge_residual`, `potential_residual` | Maximum absolute group-charge and site-equipotential errors per field. |

Charge residual acceptance uses `(tolerance+64*eps)*(1+max(abs(Q)))`;
potential residual acceptance uses `(tolerance+64*eps)` times
`1+max(abs(b+v))+max(abs(Aq))` (`v=0` without imposed field).
Nonfinite or inaccurate solutions raise an error.
Singular geometry/width combinations are rejected, never repaired by hidden
diagonal jitter. Tighten/check tolerances in the chosen physical units.

Cache only the latest exact geometry/operator's LU factorization. Changes in
grid size, cell spacing, mask, group order, widths, amplitude, dtype or device
invalidate it. Charges and right-hand sides are recomputed and remain
differentiable. Runtime caches are not serialized. Dense probes require
O(n_grid*n_metal) temporary storage; factorization requires O(n_metal²) memory
and O(n_metal³) work. This first implementation targets small electrode tests,
not large all-metal-volume grids.

## Integration without double counting

`MetalElectrodeReadout(..., contribution="correction")` adds **only** `qᵀb +
qᵀAq/2` to an existing complete liquid functional. This preserves that model's
liquid SR/LR partition, including a damped LR term and its learned residual.
It computes the full combined Coulomb energy for diagnostics nonetheless.
These model-visible `coulomb_*_energy` outputs always stay in physical energy
units. They are not extra terms to sum into `beta_F_exc`, nor is diagnostic
`coulomb_energy` identical to the selected correction contribution.

`contribution="total"` instead returns `U_ll + qᵀb + qᵀAq/2`. Use it only with
a liquid residual defined for that exact explicit liquid block; appending it
to an already complete liquid functional double-counts electrostatics.
With an imposed field, both expressions additionally include
`metal_external_energy`; `coulomb_energy` retains its original Coulomb-only meaning.

The core amplitude is **physical**, unlike the beta-energy amplitude of the
existing `LongRangeReadout`. `GridCACEModel` supplies its own `beta` and energy
convention to this adapter. Beta mode multiplies physical energy by `beta`;
physical mode divides by `reference_energy=k_B*mean_temperature` before the
model's existing conversion. Do not pass a Bjerrum-length/`1/T` amplitude and
then apply beta a second time. The adapter's standalone context requires the
same explicit scale.

Density derivatives pass through the charge solve; the second derivative
contains the induced response (the constrained Schur complement), not a frozen
charge Hessian. Density `c1`/`c2` from a model including this term are
electrode-dependent functional derivatives, not bulk reference correlations.
Mean-density polarization alone misses microscopic image self/correlation
effects: zero mean liquid charge can coexist with microscopic ion image
attraction. It is an exploratory closure until separately benchmarked.

## Validation and provenance

```sh
PYTHONPATH=src python -m unittest discover -s tests -p 'test_metal*.py'
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
PYTHONPATH=src python -m examples.lj_paper_v1_regression.forward
PYTHONPATH=src python -m examples.lj_paper_v1_regression.solve
```

Tests cover independent dense Fourier/KKT references, energy components,
per-group constraints, positive fluid densities and exact masked zeros,
autograd first/second derivatives, metadata and batching, unit conversion,
cache invalidation/serialization and inverse recovery from baseline and
perturbed starts. The original compact LJ checkpoint and tolerances are
unchanged. These are software tests, not constant-charge electrode MD or
continuum-limit image-charge validation.

CACE structural precedent: `cace/modules/metalwall.py` and `metalwall_qeq.py`
at `a0536bf1940371821d84807d9572be69b4a6b20a` in the
[shared CACE repository](https://github.com/BingqingCheng/cace/tree/a0536bf1940371821d84807d9572be69b4a6b20a/cace/modules).
See the [LAMMPS electrode documentation](https://docs.lammps.org/latest/fix_electrode.html)
for the distinction between fixed total electrode charge and fixed potential.
