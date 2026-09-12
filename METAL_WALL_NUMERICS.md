# MetalWall numerical reference

For setup and parameter meanings, see [the API guide](METAL_WALL.md).
This reference describes the electrode-only periodic operator.

## Charge normalization and energy

Each liquid voxel supplies integrated charge
`q_liquid = DeltaV * sum_i(liquid_charges[i] * rho_i)`.
Each metal coefficient is an integrated Gaussian charge in e, not a density.
The Gaussian widths `metal_sigma` and `liquid_sigma` are individual standard
deviations in length units. Zero liquid width means an unsmeared center source
on a finite reciprocal grid, not exact point-particle Ewald or box averaging.

With liquid potential `b` at sites, metal interaction matrix `A`, imposed
metal potential `v`, and optimized site charges `q`:

```text
U_electrode = q.T @ b + q.T @ A @ q / 2 + q.T @ v
```

No liquid–liquid energy is evaluated. The complete liquid functional owns it;
the solver owns liquid external-potential work. Electrode energy diagnostics
remain physical energies. The readout multiplies by beta for beta free-energy
mode; in physical mode it first divides by the model reference energy before
the model's existing conversion.

Charge relaxation remains differentiable in liquid density, including second
derivatives. Geometry, Gaussian widths and field configuration are fixed.
The mean-density quadratic does not include microscopic image/correlation
free energy. Metal-containing c1/c2 are not bulk correlation functions.

## Coordinates and imposed field

The grid origin is derived from physical `grid_center`, or zero if absent.
Only Fourier sampling subtracts it from physical `metal_positions`.
The constructor `field_origin` independently sets the physical metal-field
wrapping center:

```text
d = metal_positions - field_origin
r_wrap = d - L * round(d / L)
v = -external_field dot r_wrap
```

This potential enters both the charge solve and metal field work. The liquid
field must be supplied once in `V_ext`, with a consistently chosen branch.
The finite-field construction is not an independently fixed-potential ensemble.
Moving the sawtooth cut through metal sites can change the voltage condition.

## Reciprocal operator and constrained charge solve

The charge basis has form factor `g_sigma(k)=exp(-sigma²*k²/2)`. Use
`K(k)=4*pi*coulomb_amplitude/k²` on the full FFT domain, omitting k=0.
Liquid/metal and metal/metal kernels multiply K by
`g_liquid*g_metal` and `g_metal²`, respectively. Widths describe
individual Gaussian standard deviations, not a combined damping-kernel width.
The metal Gaussian self term is retained. Even `liquid_sigma=0` is a finite-grid
spectral source, not an exact point-particle Ewald sum. Check grid/cutoff and
Gaussian-width convergence before scientific use.

The inverse-FFT potential includes `1/DeltaV`; combined with FFT normalization
this gives the reciprocal-sum factor `1/cell_volume`. For sites on grid points,
matrix entries sample `G=real(ifftn(K_mm))/DeltaV` at periodic site separations; the self term
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
physical units. Without metal the electrode contribution is zero; the
metal-containing neutrality check is not applied. The liquid functional owns
its own electrostatic convention.

## Explicit-site spectral continuation and cost

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
differentiable. Reuse `Aq` for residuals and metal energy. On-grid sites need
O(G+M²) storage, not one full grid per metal site. For C distinct fractional offsets,
shifted phases need O(C*G), matrix construction O(C²) grid transforms, and the
dense constrained factorization O((M+groups)²) memory with cubic work.
Regular layers may have few offsets; arbitrary irregular sites can be costly.
This is not a scalable large-electrode solver.

## Validation

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
tests is not evidence of converged equilibria or accurate physical electrode response.
