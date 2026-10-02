# Experimental capabilities

The top-level [`README.md`](../README.md) documents the compact, established
workflow and model-file API. This file
collects opt-in model and training capabilities developed for ongoing
experiments. They are tested software interfaces, but their scientific value
must be established for each physical system and dataset.

The examples below describe general EquiCDFT behavior. They do not prescribe
an electrolyte model, charge convention, density basis, cutoff, or response
target.

## Compatibility principles

- Omitting every option described here retains the existing model paths.
- Models saved with `save_model` record every option in their configuration
  and load unchanged; whole-object `torch.save(model)` files from before that
  format are converted once with `python -m equicdft.conversion`, which fills
  in the defaults such files implied and verifies the result.
- Optional energies are summed before functional differentiation.
- Radial transforms act before invariant products; density transforms act
  before neighborhood gathering and Cartesian moments.
- Fourier perturbations preserve each physical component's particle number.
- A passing software regression is not evidence that an experimental feature
  improves a scientific benchmark.

## Density-channel transforms

`CartesianAFeatures` can transform physical density components pointwise
before constructing either neighbor moments or separated-center descriptors:

```python
a_features = CartesianAFeatures(
    mean_density=mean_density,
    cutoff_grid=6,
    max_power=3,
    n_types=2,
    density_transform=(
        (0.5, 0.5),
        (-0.5, 0.5),
    ),
    trainable_density_transform=False,
)
```

For weights `W[q,t]`, the descriptor density is

```text
rho_descriptor[..., q] = sum_t W[q,t] rho_physical[..., t].
```

The transform has no bias, so an empty voxel stays empty. It may be square or
rectangular and may contain negative weights. If `n_channels` is omitted, its
value is inferred from the number of matrix rows. Supplying `n_channels`
without explicit weights retains the learned Xavier-initialized transform;
omitting both keeps the physical density channels unchanged.

The same scalar `mean_density` normalizes all resulting descriptor channels.
Consequently, the magnitude convention in a prescribed matrix is part of the
model definition.

## Radial representations and pre-invariant transforms

`CartesianAFeatures.radial_basis` supports three representations:

| value | primitive radial values | conditioning |
| --- | --- | --- |
| `"none"` | one equal-weight channel | discrete unit sum |
| `"gaussian"` | `exp[-alpha_n (r-u_n)^2]` | each channel has discrete unit sum |
| `"bessel"` | `sqrt(2/Rc) sin(n pi r/Rc)/r` | degree-wise discrete Gram whitening |

Here `r`, the Gaussian center `u_n`, and cutoff `Rc` are in grid units.
Gaussian exponents therefore have units of inverse squared grid spacing. For
physical grid spacing `Delta`, the Bessel mode has physical radial wave number
`n pi/(Rc Delta)`. Bessel functions use their finite analytic value at `r=0`
and are exactly zero at and beyond `Rc`.

### Gaussian channels

```python
a_features = CartesianAFeatures(
    mean_density=mean_density,
    cutoff_grid=6,
    max_power=3,
    radial_basis="gaussian",
    radial_exponents=(0.015625, 0.03125, 0.0625, 0.125),
    radial_centers=(0.0, 1.0, 2.0, 3.0),
    trainable_radial_exponents=True,
    trainable_radial_centers=True,
    n_types=n_types,
)
```

Positive trainable exponents are represented logarithmically. Trainable
centers are direct unconstrained parameters after initialization.

### Conditioned Bessel channels

```python
a_features = CartesianAFeatures(
    mean_density=mean_density,
    cutoff_grid=6,
    max_power=3,
    radial_basis="bessel",
    n_radial_functions=6,  # fixed primitive count N
    n_radial_channels=4,   # transformed output count M <= N
    coordinate_scaling="none",
    separate_center=True,
    n_types=n_types,
)
```

The primitive radial-Cartesian products are whitened on the exact integer
stencil separately for every total Cartesian degree. Rank-deficient choices
are rejected. Because whitening is degree-specific, the uniform degree scale
introduced by `coordinate_scaling="cutoff"` cancels for the Bessel basis up
to numerical roundoff.

### Learned radial transform

For Gaussian or Bessel primitives, explicitly setting
`n_radial_channels=M` adds one bias-free `N x M` transform for every total
Cartesian degree. Each matrix is shared by all density channels and all
monomials of that degree and is initialized as a rectangular identity. The
transform is applied to Cartesian `A` moments before nonlinear invariant
products are formed, allowing those products to contain cross-primitive
radial terms.

Omitting `n_radial_channels` retains all primitive functions without a
transform. The `"none"` basis accepts only its compatibility width of one.

### FFT Cartesian A construction

The initial Cartesian density moments normally gather every local
neighborhood explicitly. Complete regular periodic grids can instead apply
the same stencil as a circular FFT cross-correlation:

```python
a_features = CartesianAFeatures(
    mean_density=mean_density,
    cutoff_grid=6,
    max_power=3,
    convolution_backend="fft",
    # radial and density-transform arguments omitted here
)
```

The pointwise density transform, when configured, is applied before this
contraction. Its transformed density channels are Fourier transformed once
and shared across all radial and Cartesian stencil channels. `"gather"`
remains the default and legacy behavior. Selecting FFT adds no parameters or
`state_dict` keys, but requires complete `grid_positions` and `grid_size`.

### Omitting explicit neighborhood indices

When the initial Cartesian features and every message layer use FFT or
`conv3d`, their input records do not need the explicit
`[n_grid, n_neighbors]` gather table. It may be omitted at construction:

```python
frames = GridData.from_xyz(
    paths,
    grid_info=model.grid_info,
    include_local_density_index=False,
)
assert not model.requires_local_density_index
```

The default remains `True`. Gather-based features or messages report
`requires_local_density_index=True` and raise a clear error if the table is
missing. Stencil offsets remain in each record; only the large explicit
per-grid-point index table is skipped. This option changes neither model
parameters nor serialized model state.

## Equivariant message passing

`BChiMessage` converts invariant `B` features into a new equivariant `A`
field. At layer `l`, an invariant neural gate is convolved with a
radial-Cartesian stencil and symmetrized again:

```text
B_l -> invariant gate -> A_(l+1) -> B_(l+1).
```

The gate uses `h(B)-h(0)`, which makes a zero invariant field produce a zero
message while preserving the gate Jacobian at zero. `GridCACEModel` retains
and concatenates `B_0, B_1, ...` for the local readout.

```python
from equicdft import BChiMessage, GridCACEModel

message = BChiMessage(
    n_invariant_features=b_features.n_features,
    n_radial_channels=a_features.n_radial_channels,
    n_channels=a_features.n_output_channels,
    hidden_sizes=(32, 16),
)

model = GridCACEModel(
    a_features=a_features,
    b_features=b_features,
    message_layers=[message],
    readout=readouts,
    # remaining thermodynamic arguments omitted here
)
```

The message radial choices are:

- `radial_basis=None` with no `radial_exponents`: share the complete initial
  `CartesianAFeatures` stencil, including its transform.
- Explicit `radial_exponents`: use a message-owned Gaussian basis with the
  same Cartesian geometry, bypassing the initial radial transform.
- `radial_basis="bessel"`: build a message-owned conditioned Bessel basis and
  a distinct learned transform.

An independent Bessel message is configured as:

```python
message = BChiMessage(
    n_invariant_features=b_features.n_features,
    n_radial_channels=4,
    n_channels=a_features.n_output_channels,
    hidden_sizes=(32, 16),
    radial_basis="bessel",
    n_radial_functions=6,
)
```

Its fixed basis inherits the cutoff, center mask, Cartesian monomials, and
stencil geometry from `a_features`, but its degree-dependent `N -> M`
transform is independent. Multiple message layers own distinct neural and
radial-transform parameters.

### Low-memory periodic convolution

The established message contraction explicitly gathers a tensor with shape
`[..., G, J, N, C]`. Large regular grids and stencils can instead select a
mathematically equivalent periodic convolution (up to floating-point
operation order):

```python
message = BChiMessage(
    n_invariant_features=b_features.n_features,
    n_radial_channels=4,
    n_channels=a_features.n_output_channels,
    hidden_sizes=(32, 16),
    convolution_backend="fft",  # or "conv3d"
)
```

This execution option changes neither learned parameters nor `state_dict`
keys. It is part of the model configuration, so `save_model` records it and
`load_model` restores it; converted legacy whole-object files that predate
the option report `"gather"`. `"conv3d"` scatters the spherical stencil into its enclosing dense Cartesian
kernel and applies grouped cross-correlation. `"fft"` scatters onto the
periodic grid and applies the same cross-correlation in reciprocal space;
duplicate or periodically aliased offsets are summed. Both avoid the explicit
neighbor tensor. These backends require a complete regular periodic grid.
The FFT path is intended for real float32 or float64 fields; mixed-precision
FFT support is device- and grid-size-dependent and is not assumed here.

### Enabling FFT throughout a model

Select FFT independently for the initial Cartesian features and for every
message layer. For example, a one-message model uses:

```python
a_features = CartesianAFeatures(
    mean_density=mean_density,
    cutoff_grid=6,
    max_power=3,
    convolution_backend="fft",
    # radial and density-transform arguments omitted here
)

message = BChiMessage(
    n_invariant_features=b_features.n_features,
    n_radial_channels=a_features.n_radial_channels,
    n_channels=a_features.n_output_channels,
    hidden_sizes=(32, 16),
    convolution_backend="fft",
    # optional independent radial-basis arguments omitted here
)

model = GridCACEModel(
    a_features=a_features,
    b_features=b_features,
    message_layers=[message],
    readout=readouts,
    # remaining thermodynamic arguments omitted here
)
```

If all local contractions use FFT, the large gather table may also be omitted:

```python
assert not model.requires_local_density_index
frames = GridData.from_xyz(
    paths,
    grid_info=model.grid_info,
    include_local_density_index=False,
)
```

For a model saved with the gather backend, the execution method can be
changed without modifying its learned state, either on the loaded object or
in its configuration before rebuilding:

```python
model = load_model(path)
model.a_features.convolution_backend = "fft"
for message in model.message_layers:
    message.convolution_backend = "fft"
```

Do this only when inputs provide a complete regular periodic grid. A model
reconstructed from a `state_dict` should instead set the backend explicitly in
the constructors, because `convolution_backend` is a configuration entry and
deliberately not a learned or serialized state tensor.

Gather and FFT evaluate the same periodic cross-correlation. Their outputs are
mathematically equivalent, but need not be bitwise identical because the FFT
changes floating-point summation order. Validate energies, functional
derivatives, parameter gradients, and application-level tolerances before
adopting FFT for a new dtype, device, grid, or model family. Performance also
depends on grid size and runtime: benchmark the complete workload rather than
assuming the large-grid speedup transfers to a small model.

## Charge-factorized reciprocal readouts

`LongRangeReadout` can constrain pair coefficients to a known charge-product
form instead of learning every species pair independently:

```python
reciprocal = ReciprocalFeatures(
    kernel="coulomb",
    radial_exponents=(alpha,),
    n_types=n_types,
)

fixed_lr = LongRangeReadout(
    n_kernels=1,
    n_types=n_types,
    charges=charges,
    coulomb_amplitude=amplitude,
    features=reciprocal,
)
```

The coefficients are `C_ij = amplitude * charge_i * charge_j`. Omitting
`coulomb_amplitude` learns one shared state-dependent amplitude. Omitting
`charges` retains freely learned pair coefficients. Charge factorization
currently requires one reciprocal kernel.

For `kernel="coulomb"`, the implemented kernel is
`4 pi exp(-alpha k^2)/k^2` at nonzero wavevectors. It is a Gaussian-damped
Coulomb kernel, so choosing `alpha` also defines an LR/SR partition and is not
merely a numerical setting.

## Fixed MSA reference for the restricted primitive model

`MSAReferenceReadout` adds a parameter-free functional built from the
analytic direct correlation functions of the restricted primitive model,
so that a learned excess only has to represent correlations beyond the
mean-spherical closure:

```python
reference = MSAReferenceReadout(
    n_types,
    charges=charges,
    diameter=diameter,
    coulomb_amplitude=amplitude,
    long_range_alpha=alpha,
)
```

The functional is $\beta F_{\rm ref} = \tfrac12\sum_{ij}\iint \rho_i K_{ij} \rho_j$
with $K_{ij} = -c_{ij}$ and

$$
c_{ij}(r) = c^{\rm PY}_{\rm HS}(r) + z_i z_j\, c^{\rm MSA}_{\rm el}(r),
\qquad
c^{\rm MSA}_{\rm el}(r) =
\begin{cases}
-\dfrac{A}{\sigma}\left[2B - B^2\, r/\sigma\right], & r<\sigma,\\[6pt]
-\dfrac{A}{r}, & r>\sigma,
\end{cases}
\qquad
B = \frac{x}{x+1+\sqrt{1+2x}},\ x = \kappa_D\sigma,
$$

the Percus-Yevick hard-sphere function for the number channel and the
Waisman-Lebowitz solution for the charge channel ($B$ is the
cancellation-free form of $(x+1-\sqrt{1+2x})/x$). `coulomb_amplitude` is the
Bjerrum length $A$ in coordinate units of the beta free energy; the readout
is defined for `free_energy_mode="beta"` at the reference temperature and
rejects other modes or temperatures.

**Coulomb branch.** The bare tail $-A z_i z_j/r$ is carried by a
charge-factorized damped Coulomb `LongRangeReadout` with kernel
$4\pi e^{-\alpha k^2}/k^2$, and the reference subtracts the matching
$A\,\mathrm{erf}(r/2\sqrt{\alpha})/r$ from its charge kernel, leaving the
short-ranged $A\,\mathrm{erfc}(r/2\sqrt{\alpha})/r$ outside the core and the
polynomial correction inside it, so that the pair is exactly the MSA whatever
`alpha` is (`0` for the exact kernel). By default
(`include_long_range=True`) the readout owns that branch, built from the same
charges, amplitude and exponent and reported through `wrapped_readouts()`,
and rejects any other charged Coulomb `LongRangeReadout` in the model, at top
level or wrapped (by a `MetalWall`, say). With `include_long_range=False` the
model must contain exactly one such readout, at top level or wrapped, with
the same charges, a fixed equal `coulomb_amplitude`, one Coulomb kernel with
exponent `long_range_alpha` and no polarization source; this is the
composition for electrode calculations, `MetalWall(liquid_coulomb=...)`
beside the reference. The check is the readout's own
`validate_readout_graph`, which `GridCACEModel` calls for every top-level
readout at construction; a mismatch would silently give something other
than the MSA.

**Reference state.** The bulk state enters only through
$\kappa_D^2 = 4\pi A\sum_i z_i^2\rho_i$ and the packing fraction, and by
default it is each field's own: the mean density of every component over
the accessible voxels (`excluded_mask` false, or all voxels), a
differentiable functional of the density. The bulk reference must be
electroneutral. This permits arbitrary local charge separation at fixed
global neutrality. A finite liquid carrying net charge compensated by
electrodes must instead supply an explicit neutral `reference_density`.
The readout is then the Ramakrishnan–Yussouff expansion about that state,

$$
\beta F_{\rm ref}[\rho] = V_{\rm acc}\,\beta f_{\rm bulk}(\bar\rho)
+ \tfrac12\sum_{ij}\iint \delta\rho_i\,K_{ij}(|\mathbf r-\mathbf r'|;\bar\rho)\,\delta\rho_j ,
\qquad \delta\rho = \rho - \bar\rho \ \text{on accessible voxels},
$$

with $\beta f_{\rm bulk}$ the bulk excess free energy density whose Hessian
is $-\hat c_{ij}(0)$: the scaled-particle (PY-compressibility) hard-sphere
free energy $\rho[-\ln(1-\eta) + 3\eta/(1-\eta) + 3\eta^2/2(1-\eta)^2]$ plus
$\tfrac12\hat K_Z(0)\rho_Z^2$ for the charge channel. The homogeneous mode is
carried by the integrated bulk free energy and the state-dependent kernel
acts on mean-zero fluctuations only, so this is one variational functional:
at a homogeneous state the uniform ($k=0$) curvature is exactly the MSA/PY
$-\hat c(0)$, continuous with the $k\to0$ limit of the kernel, the autograd
derivative equals the finite-difference slope along a uniform density
change, and energies of different densities and compositions are mutually
consistent. A set of fields at different densities gets the MSA of each
density and a wall field that of its accessible region;
`reference_density=(rho_1, ...)` fixes one state for every field instead and
makes the readout the plain quadratic form $\tfrac12\rho K(\rho_{\rm ref})\rho$
including its homogeneous mode. With a mask the expansion is about the flat
accessible profile: the reference does not describe the excluded region
itself, and wall structure beyond quadratic order is left to other terms. A
field whose reference packing fraction reaches one is rejected. Vacuum and
zero Coulomb amplitude use the exact zero-screening limit, without an
artificial density or screening floor.

**Evaluation.** The kernels are applied in reciprocal space on the canonical
grid order (rows are reordered through `grid_positions` when they are not
canonical). Inside the core both functions are polynomials in $r/\sigma$, so
their transforms are state-dependent coefficients times the
density-independent transforms of $1$, $r/\sigma$ and $(r/\sigma)^3$ on the
core (plus the fixed erf/erfc part), integrated once per distinct grid
wavenumber and cached per grid geometry; per-field states therefore cost
nothing, and contact discontinuities enter as band-limited transforms rather
than shell samples.

A model made of the ideal gas and this reference has the MSA response,
$1/S_{ZZ}(k) = 1 - \rho\,\hat c_{\rm el}(k)$ and
$1/S_{NN}(k) = 1 - \rho\,\hat c^{\rm PY}_{\rm HS}(k)$, which is positive at
every wavevector; `FourierResponse` on a homogeneous batch reproduces both to
better than $10^{-3}$, and the uniform curvature reproduces the PY
compressibility $(1+2\eta)^2/(1-\eta)^4$. `include_hard_sphere=False` keeps
only the electrostatic correction (kernel and bulk term), for use under a
separate hard-sphere functional. This is a state-conditioned
quadratic reference that reproduces the bulk second derivative, not a
nonlinear hard-sphere functional: packing and confinement structure beyond
quadratic order are left to other terms. The closure linearizes the Coulomb
coupling outside the core, so at strong coupling it underestimates ion
pairing; that residual, not the ordering peak itself, is what a learned
readout on top of it has to supply.

## Fundamental measure theory for hard spheres

`FMTReadout` adds fundamental measure theory as a fixed, parameter-free
hard-sphere functional, evaluated with analytic weight spectra on the FFT
grid:

```python
hard_spheres = FMTReadout(
    n_types,
    radii=radii,
    version="white_bear_i",
    third_term="q3",
)
```

The excess free energy is a local function of weighted densities,

$$
\beta F_{\rm HS}[\rho] = \int \Phi\big(\{n_\alpha(\mathbf r)\}\big)\,d\mathbf r,
\qquad
n_\alpha(\mathbf r) = \sum_i \int \rho_i(\mathbf r')\,\omega^{(i)}_\alpha(\mathbf r-\mathbf r')\,d\mathbf r',
$$

with the weights of a sphere of radius $R_i$: $\omega_3 = \Theta(R-r)$,
$\omega_2 = \delta(R-r)$, $\omega_1 = \omega_2/4\pi R$, $\omega_0 =
\omega_2/4\pi R^2$, $\boldsymbol\omega_2 = \hat{\mathbf r}\,\delta(R-r) =
-\nabla\omega_3$, $\boldsymbol\omega_1 = \boldsymbol\omega_2/4\pi R$, the
tensor weight $\overleftrightarrow{\omega}_2 = \hat{\mathbf r}\hat{\mathbf
r}\,\delta(R-r)$ when the tensor third term is selected, and

$$
\Phi = -\,n_0\ln(1-n_3)
+ \frac{n_1 n_2 - \mathbf n_1\!\cdot\mathbf n_2}{1-n_3}
+ P\big(n_2,\mathbf n_2,\overleftrightarrow{n}_2\big)\,\phi_3(n_3),
\qquad
\phi_3 = \begin{cases}
\dfrac{1}{24\pi(1-n_3)^2} & \text{Rosenfeld},\\[8pt]
\dfrac{n_3 + (1-n_3)^2\ln(1-n_3)}{36\pi\,n_3^2\,(1-n_3)^2} & \text{White Bear mark I}.
\end{cases}
$$

`version` selects $\phi_3$: `"rosenfeld"` reproduces the scaled-particle
(Percus-Yevick compressibility) equation of state in the bulk, `"white_bear_i"`
(default; `"white_bear"` is accepted as the same thing) is the factor of the
White Bear functional of Roth, Evans, Lang and Kahl (2002), the mark I form
rather than the later mark II, with the Carnahan–Starling bulk equation of
state. `third_term` selects the polynomial $P$, which equals $n_2^3$ in any
homogeneous fluid, so that the three choices share the bulk free energy and
the bulk second derivative and differ only away from homogeneity. The two
are independent choices and only some combinations are published
functionals — (`rosenfeld`, `original`) is Rosenfeld 1989, (`rosenfeld`,
`q3`) the 1997 antisymmetrized functional, (`rosenfeld`, `tensor`) Tarazona
2000, (`white_bear_i`, `original`) White Bear mark I as published,
(`white_bear_i`, `tensor`) the White Bear tensor functional used for
freezing; the default (`white_bear_i`, `q3`) is the White Bear I bulk
factor with the q3 closure, chosen for its bounded third term and its
equation of state, and is not itself a named functional:

- `"q3"` (default): $P = n_2^3\,(1-\xi^2)^3$ with $\xi = |\mathbf n_2|/n_2$,
  the "antisymmetrized" third term of Rosenfeld, Schmidt, Löwen and Tarazona
  (1997). The exponent 3 keeps the original term through order $\xi^2$, so
  the Percus-Yevick pair correlations are unchanged; the factor vanishes
  where $|\mathbf n_2| = n_2$, which makes a single zero-dimensional cavity
  exact; and $P \ge 0$ wherever $n_2 \ge 0$. The authors presented it as a
  provisional interpolation (its crystal density peaks come out too narrow),
  not as a stability device.
- `"tensor"`: $P = \tfrac92\big[\mathbf n_2\!\cdot\overleftrightarrow{n}_2\!\cdot\mathbf n_2
  - n_2\,\mathbf n_2\!\cdot\mathbf n_2 - \mathrm{Tr}\,\overleftrightarrow{n}_2^{\,3}
  + n_2\,\mathrm{Tr}\,\overleftrightarrow{n}_2^{\,2}\big]$, Tarazona's
  tensorial form (2000), whose dimensional crossover extends to overlapping
  cavities and lost dimensions; it is the usual choice for freezing but has
  no global lower bound (Lutsko 2020), so it is the second choice rather
  than the default.
- `"original"`: $P = n_2^3 - 3\,n_2\,\mathbf n_2\!\cdot\mathbf n_2$,
  Rosenfeld's 1989 scalar-vector form. It has no lower bound (a single
  sphere-sized density peak drives $\beta F_{\rm HS}$ to $-\infty$ as the
  peak narrows) and it does not reproduce the zero-dimensional limit; it is
  kept for comparison with the literature, not for production.

None of this is a statement about the grid. The weights are sharp, so for a
density the grid resolves the weighted densities are exactly translation
covariant, and the energy of a peak at least one spacing wide is the same
to a percent on a grid point and between grid points, at $0.25\sigma$ and
$0.125\sigma$ spacing alike and for every third term. A peak narrower than
the spacing is a different matter: it is not represented on the grid at
all — its samples change with its position, and the ideal-gas term alone
changes by $N\ln 8$ between the two alignments — and its FMT energy
depends on the alignment as well (a $0.05\sigma$ peak of 0.95 particles at
$0.25\sigma$ spacing: 0.8 between grid points, 6.5 on a grid point with
`"q3"`), because the sharp weights ring around it. Ringing produces
weighted densities no density could have, and those are what let $\Phi$
change sign; the readout therefore enforces the continuum bounds $n_3^{(i)}
\ge 0$, $n_2^{(i)} \ge 0$, $|\mathbf n_2^{(i)}| \le n_2^{(i)}$ on the
per-species weighted densities, exactly where they are violated and nowhere
else (a smooth field is untouched to the bit): $n_3^{(i)}$ is clamped at
zero, where $n_2^{(i)} < 0$ the whole shell family of that species — scalar,
vector and tensor — is set to zero, as every moment of a shell of no mass
is, so that $\mathrm{Tr}\,\overleftrightarrow{n}_2 = n_2$ survives, and
elsewhere $\mathbf n_2^{(i)}$ is shortened to length $n_2^{(i)}$. With them $\Phi \ge 0$
pointwise for `"q3"` on any grid, so an unresolved peak costs energy at
either alignment and cannot open a direction of descent for a minimizer.
The bounds are active at the few-percent level next to a hard wall too
(the Gibbs overshoot of the sharp weights at a density step), which
changes the slab energy by $10^{-4}$ and is why they are enforced in both
`packing_guard` modes rather than treated as an error. A negative input
density is rejected, as the ideal-gas and MSA readouts reject it, rather
than being absorbed by the bounds. Reference densities should be smoothed
to at least the grid spacing, as the Gaussian-binned data are.

The weights are applied through their analytic transforms,
$\hat\omega_3 = 4\pi(\sin kR - kR\cos kR)/k^3$, $\hat\omega_2 = 4\pi R
\sin(kR)/k$, $\hat{\boldsymbol\omega}_2 = -i\mathbf k\,\hat\omega_3$,
$\hat{\overleftrightarrow{\omega}}_2 = 4\pi R^2\big[\delta_{ij}\,j_1(kR)/kR -
\hat k_i\hat k_j\,j_2(kR)\big]$, so they are exactly isotropic and exactly
normalized (their $k\to0$ values are the fundamental measures, the tensor
weight's $\tfrac13\,n_2\,\delta_{ij}$) and involve no stencil; the vector
weights use the real spectral derivative that zeroes each even-axis Nyquist
component, as the Coulomb source does. Grid rows are reordered through
`grid_positions` when they are not canonical. The functional is entropic: in
`free_energy_mode="beta"` the readout returns $\beta F_{\rm HS}$, in
`"physical"` mode $\beta F_{\rm HS}\,T/T_{\rm ref}$ so that the model's
conversion recovers the same $\beta F_{\rm HS}$ at every temperature.

What the tests pin down: a homogeneous field gives the scaled-particle
free energy for Rosenfeld and the Carnahan–Starling one for White Bear
mark I to machine precision for every third term, with $c_1 =
-\beta\mu_{\rm ex}$; the weighted densities of a uniform mixture are the
fundamental measures; the Rosenfeld second derivative on the number
direction is the Percus-Yevick $1-\rho\hat c(k)$ at every grid wavevector
below Nyquist, at every density and for every third term (the density
dependence is automatic, unlike a quadratic reference), and the uniform
($k=0$) curvature is each version's compressibility; the low-density limit
is the second virial coefficient; a Gaussian peak one and one-and-a-half
spacings wide gives the same energy on a grid point and between grid
points to 1 % at both spacings for every third term; peaks of 0.4 and 0.2
spacings keep $\Phi \ge 0$ pointwise and $\beta F_{\rm HS} \ge 0$ with
`"q3"` at both alignments while `"original"` goes negative and keeps
falling; the guard leaves a smooth random field bit-identical, tensor
included, and keeps $\mathrm{Tr}\,\overleftrightarrow{n}_2 = n_2$ to
$10^{-12}$ where it is active; the packing clamp returns $n_3$ itself in
single precision down to $10^{-8}$;
and a slab between hard walls solved at fixed $\mu$ shows the
contact enhancement, the depletion minimum and the bulk density mid-slab,
with the extrapolated contact density within a few percent of $\beta P$ at
$0.125\sigma$ spacing. At the Nyquist wavevector the vector terms vanish by
the derivative convention, so the response there is not the PY value; at
$0.25\sigma$ spacing the delta-shell weight is two cells wide and contact
quantities are a few percent less accurate than at $0.125\sigma$.

`maximum_packing` (default 0.95) regularizes the free-volume terms: with
`packing_guard="clamp"` the local $n_3$ is replaced by the smooth
$n_3 - s\,\mathrm{softplus}((n_3-n_{\max})/s)$, $s = 0.01$, which is
$n_3$ itself (exactly, in single precision too) more than a few $s$ below
$n_{\max}$ and saturates at $n_{\max}$ above it, so
that a noisy reference density or an early solver iterate cannot reach the
singularity; `"raise"` rejects such a field instead. The readout is
charge-blind and owns packing only; beside an `MSAReferenceReadout` the
latter must be built with `include_hard_sphere=False`, which the readout's
`validate_readout_graph` enforces, and a model may hold one `FMTReadout`.
Excluded (wall) voxels need no special treatment: the density is zero
there, $\Phi$ is integrated over the whole grid, and the weighted densities
reaching into the wall from the fluid are how the functional feels it.

## Shared Fourier-response evaluator

`FourierResponse` evaluates projected curvature of the total intrinsic
dimensionless free energy `beta(F_id+F_exc)` using symmetric finite
differences around a supplied density field. Integer reciprocal-lattice
indices and component-space directions are supplied at evaluation time.

For every physical component, the perturbation is projected to preserve that
component's particle number. Cosine and sine phases are evaluated separately.
`require_uniform=True` additionally enforces a homogeneous, periodic,
unmasked reference state.

```python
response = FourierResponse(
    relative_amplitude=0.01,
    perturbations_per_forward=4,
    require_uniform=True,
)
curvature_by_phase, valid = response(
    model,
    batch,
    wavevector_indices=integer_wavevector_indices,
    directions=component_directions,
)
```

The result has shape `[field, wavevector, phase, direction]`. This evaluator
is the shared numerical core used by both stability regularization and
supervised response fitting.

For multicomponent stability, the same evaluator can reconstruct the complete
physical-component curvature matrix:

```python
curvature_matrix, active_components = response.matrix(
    model,
    batch,
    wavevector_indices=integer_wavevector_indices,
)
```

The matrix has shape `[field, wavevector, phase, type, type]`. It is normalized
in the ideal-gas metric, so the ideal contribution is the identity. For a
homogeneous mixture it is the dimensionless inverse OZ matrix
`I-sqrt(R)c(k)sqrt(R)`. Around an inhomogeneous field it is a projected
component Hessian; it is not the complete position-dependent OZ operator.

## Fourier-stability loss

`FourierStabilityLoss` applies a squared hinge when projected curvature falls
below `minimum_curvature`:

```python
stability = FourierStabilityLoss(
    random_wavevectors_per_field=1,
    wavevector_treatment="independent",
    wavevector_range=(k_min, k_max),
    relative_amplitude=0.01,
    minimum_curvature=0.0,
    component_treatment="charge",
    charges=(1.0, -1.0),
    weight=1.0,
    training_only=True,
)
```

The complete multicomponent stability test is selected explicitly with:

```python
matrix_stability = FourierStabilityLoss(
    random_wavevectors_per_field=1,
    wavevector_treatment="independent",
    wavevector_range=(k_min, k_max),
    relative_amplitude=0.05,
    minimum_curvature=0.0,
    component_treatment="full_matrix",
    perturbations_per_forward=4,
    weight=1.0,
    training_only=True,
)
```

Explicit `wavevector_indices=((nx,ny,nz),...)` and random sampling are mutually
exclusive. Random candidates lie in the physical isotropic Nyquist sphere.
The optional inclusive `wavevector_range=(k_min,k_max)` is expressed in
reciprocal units implied by `grid_spacing`: if positions are measured in
`sigma`, its units are `1/sigma`. It is not measured in grid-index units.

`random_wavevectors_per_field` controls how many reciprocal triplets are
selected; `wavevector_treatment` controls how those triplets become probes:

- `"independent"` evaluates the cosine and sine of every triplet independently.
  Thus `random_wavevectors_per_field=12` gives up to 24 nonzero curvature
  probes per field.
- `"superposition"` draws one phase for every triplet, sums the waves, and
  evaluates one composite spatial direction per field. Thus
  `random_wavevectors_per_field=12` still gives one curvature probe containing
  12 wavevectors.

The default is `wavevector_treatment="independent"`. Use `"superposition"`
explicitly when a random-phase superposition is intended.

Mixture directions are:

- `"independent"`: perturb and average individual physical components;
- `"total_density"`: perturb all components in phase;
- `"charge"`: perturb with explicit charge weights.
- `"full_matrix"`: reconstruct the ideal-metric physical-component Hessian
  and penalize every eigenvalue below `minimum_curvature`.

For an equal-density symmetric binary mixture, `(1,1)` and `(1,-1)` are the
number and charge directions. That special interpretation must not be assumed
for a general mixture. Positive curvature along selected directions does not
guarantee that the coupled matrix is positive definite, so `"full_matrix"` is
the recommended general mixture stability check. The directional treatments
remain useful as cheaper targeted regularizers.

The matrix is a finite-difference estimate with truncation error of order
`relative_amplitude**2`. Taking an extremely small amplitude can instead
amplify energy roundoff, especially in float32, and the off-diagonal
polarization adds another subtraction. Check amplitude convergence for the
actual model and grid; a smaller amplitude is not automatically more accurate.
For inhomogeneous fields this check covers the physical-component subspace of
each selected cosine or sine perturbation, but not cosine-sine or
cross-wavevector Hessian blocks.

For `n` components the exact matrix reconstruction uses `n(n+1)/2` component
directions. Cosine and sine phases and symmetric positive/negative differences
therefore require `2n(n+1)` perturbed fields per selected wavevector. Use
`perturbations_per_forward` to limit peak memory. The loss gives every active
eigenvalue equal weight, matching the existing equal weighting of valid scalar
directions.

Stability loss prevents selected negative curvatures; it does not determine
the correct positive response and is not direct structure-factor training.

### Density-weighted density probes

A box-wide density wave measures the compressibility averaged over the
field, so a functional that is unstable inside a dense layer can pass the
floor while the rest of the field dominates the average. With
`probe_density_power=p > 0` every component is displaced by
$\rho_a\,g\,(w - c_a)$ with $g = (\rho/\max\rho)^p$ ($\rho$ the total
density, per field) and $c_a$ the $\rho_a g$-weighted mean of the wave $w$,
so every particle number stays exact and the largest relative change is
still `relative_amplitude`. The mean is divided by the component's actual
weighted mass, so this holds for dilute components too; a component with no
weighted mass in the field's dtype (absent, or only where the weight
underflows) is inactive for that probe rather than an error. The directional curvature is then measured in the
ideal metric of the displacement,

$$
K = \frac{\delta^2(\beta F)}{\Delta V \sum_i \delta\rho_i^2/\rho_i},
$$

one for an ideal gas anywhere in the field (the default normalization by the
mean density is exact only for a homogeneous field; both agree there).
`"full_matrix"` already uses this metric and takes the same weighting. The
default `p = 0` is the original probe and normalization.

```python
FourierStabilityLoss(
    random_wavevectors_per_field=2,
    relative_amplitude=0.01,
    minimum_curvature=0.0,
    probe_density_power=2.0,
    weight=1.0,  # relative to the data loss; tune per campaign
    name="dense_layer_density_stability",
)
```

Weighting also moves the probe away from liquid-vapor interfaces, where a
box-wide wave partly translates the interface, a physical near-zero mode
that a floor would wrongly penalize. It tests the local curvature near the
sampled states only. A free-energy well far above the sampled densities, such
as a single voxel that a reverse solve drives to its upper density bound, is
invisible to any small-amplitude probe at the data; that needs a
finite-excursion check or a fixed high-density reference.

### Fixed-dipole polarization stability

Density and polarization stability are separate loss terms. The density term
keeps the existing fixed-particle-number construction. The polarization term
holds density fixed and perturbs the dimensionless local alignment fraction
along one random direction per field:

$$
\mathbf u_f=\frac{\mathbf z_f}{|\mathbf z_f|},
\qquad \mathbf z_f\sim\mathcal N(\mathbf 0,I_3),
$$

$$
\delta\mathbf P_{fa}(\mathbf r)
=\epsilon_P m_a\rho_{fa}(\mathbf r)
\phi_{\mathbf k}(\mathbf r)\mathbf u_f.
$$

Here $m_a$ is the fixed molecular dipole magnitude of species $a$, not a loss
weight. One direction $\mathbf u_f$ is shared by every selected wavevector and
phase in field $f$. With `wavevector_treatment="independent"`, each cosine and
sine is an independent curvature direction. With
`wavevector_treatment="superposition"`, the selected nonzero wavevectors form
one random-phase composite direction. Repeated training calls sample the full
polarization sphere without evaluating three coordinate-axis probes on every
call.

For each valid phase, the normalized curvature is

$$
\kappa_P=
\frac{
\Delta_P^2 F_{\rm id,0}
+F_{\rm exc}[\rho,\mathbf P+\delta\mathbf P]
+F_{\rm exc}[\rho,\mathbf P-\delta\mathbf P]
-2F_{\rm exc}[\rho,\mathbf P]
}{
\Delta_P^2 F_{\rm id,0}
},
$$

where the excess-energy second difference is combined with the isotropic
small-alignment fixed-dipole ideal curvature

$$
\Delta_P^2 F_{\rm id,0}
=3\,\Delta V\sum_{a,\mathbf r}
\rho_a(\mathbf r)
\left|\frac{\delta\mathbf P_a(\mathbf r)}
{m_a\rho_a(\mathbf r)}\right|^2.
$$

This ideal reference gives $\kappa_P=1$ when the learned excess functional is
zero. The polarization loss is

$$
L_{P,\mathrm{stab}}
=\frac{w_P}{N_{\rm valid}}
\sum_{f,\mathbf k,s}
\max(0,\kappa_{\min}-\kappa_{P,f\mathbf k s})^2,
$$

where $s$ labels cosine or sine, $w_P$ is the statistical loss weight, and
`minimum_curvature` supplies $\kappa_{\min}$. The Fourier-stability probe does
not impose the local fixed-dipole inequality
$|\mathbf P_a|<m_a\rho_a$ and uses the requested $\epsilon_P$ unchanged. This
is deliberate for gridded molecular reference fields, which can reach the
sampling bound locally. The exact ideal functional and equilibrium solver
retain their physical-domain checks; only this stochastic regularizer omits
the constraint.

The zero-wavevector polarization probe is meaningful because total
polarization is not conserved. It is included by default; its cosine is the
constant wave and its identically zero sine partner is discarded. It remains
a separate probe when nonzero waves are superposed. Density stability excludes
the zero wavevector because each species particle number is fixed.

```python
w_rho_stability, w_P_stability = 1.0, 1.0  # relative to the data loss
loss = Loss([
    FourierStabilityLoss(
        variable="rho",
        random_wavevectors_per_field=1,
        wavevector_treatment="independent",
        relative_amplitude=0.02,
        minimum_curvature=0.0,
        weight=w_rho_stability,
        name="rho_fourier_stability",
    ),
    FourierStabilityLoss(
        variable="dipole_density",
        random_wavevectors_per_field=1,
        wavevector_treatment="independent",
        include_zero_wavevector=True,
        relative_amplitude=0.02,
        minimum_curvature=0.0,
        weight=w_P_stability,
        name="polarization_fourier_stability",
    ),
])
```

The two positive directional tests implement the working approximation that
density and polarization fluctuations can be regularized independently. They
do not test the mixed density--polarization Hessian block. One random polarization
direction per field also does not prove that every Cartesian direction is
positive on a particular call; it is a stochastic stability regularizer whose
spherical coverage accumulates during training.

For a dipolar liquid the random direction is not enough. Its longitudinal
stiffness is dominated by the Coulomb term, $4\pi\beta C \approx 19$ ideal
units for water, so the curvature projected on a random direction is positive
unless that direction is within a few degrees of transverse, and a functional
whose transverse stiffness has gone negative passes the random probe.
`polarization_direction="transverse"` draws, for every wave, a random
direction in the plane perpendicular to its own wavevector $2\pi\mathbf n/L$
(the uniform wave keeps a random direction), and `"longitudinal"` uses the
wavevector itself; both work with independent and superposed waves, a sum of
transverse waves staying divergence-free. A floor of 0.1--0.2 on the
transverse probe is the guard to train a learned dielectric kernel under
(`polarization.md`, section 7).

```python
FourierStabilityLoss(
    variable="dipole_density",
    random_wavevectors_per_field=2,
    polarization_direction="transverse",
    relative_amplitude=0.02,
    minimum_curvature=0.15,
    weight=1.0,  # relative to the data loss (the water campaigns use 20)
    name="transverse_polarization_stability",
)
```

The probe's curvature is a ratio of total to ideal second variation over
the whole field, so a box-wide wave can pass the floor while a dense
region, a contact layer or a dense slab, is locally unstable: the bulk
dominates the average. `probe_density_power=p` weights every probe wave by
$(\rho/\max\rho)^p$ per field, concentrating it in the densest voxels. On a
2 Å grid `p = 2` is a good choice: much larger powers localize the wave
below the widths of smooth kernels, which then no longer act on it, and
hide the softening again. The weighted wave is divergence-free only where
the density gradient is perpendicular to the wave's polarization, as for
layers normal to the wavevector; elsewhere it picks up some longitudinal
stiffness. It is a detector for dense-layer instability, to be combined
with a density-scaled dielectric kernel (`polarization.md`, section 7).

### Coupled density-polarization probes

In a polarized state the two single-variable probes can both pass while a
wave that moves density and polarization together lowers the free energy:
the Hessian blocks are positive but the density-polarization cross term is
large enough to make the joint matrix indefinite. `variable="coupled"`
evaluates that matrix. For every real wave each species $a$ contributes a
density channel, the fixed-number displacement of the density probe, and a
polarization channel $\delta\mathbf P_a = m_a\rho_a\,w\,\mathbf n$. The
second variation of excess plus ideal reference is assembled from diagonal
and pair differences, exactly as the density `"full_matrix"` treatment, and
each channel is normalized by its unpolarized ideal metric
($\sum\delta\rho^2/\rho$ and $3\sum|\delta\mathbf P|^2/(m^2\rho)$), so an
unpolarized ideal gas gives the identity. Every eigenvalue below
`minimum_curvature` is penalized.

```python
FourierStabilityLoss(
    variable="coupled",
    random_wavevectors_per_field=2,
    polarization_direction="parallel",
    relative_amplitude=0.01,
    minimum_curvature=0.0,
    probe_density_power=0.0,
    weight=1.0,  # relative to the data loss; tune per campaign
    name="coupled_density_polarization_stability",
)
```

- `component_treatment="independent"` keeps one 2×2 block per species;
  `"full_matrix"` keeps every cross-species block (2n×2n). Charge and
  total-density treatments are not defined here.
- `polarization_direction="parallel"` displaces polarization along the local
  base polarization of each species, the alignment-magnitude mode through
  which density couples; `"random"`, `"transverse"` and `"longitudinal"` are
  as for the polarization probe. The zero wavevector is excluded because the
  density channels conserve particle number.
- The ideal reference enters as its analytic Hessian at the base state
  (translational $\delta\rho^2/\rho$ plus orientational
  $A(\hat{\mathbf P}\cdot\delta\mathbf P - m p\,\delta\rho)^2 + B|\delta\mathbf P_\perp|^2$
  with $A = s''(p)/(m^2\rho)$, $B = s'(p)/(p\,m^2\rho)$), one regime per
  voxel shared by every probe row. It is therefore convex and independent of
  the probe amplitude; only the excess is finite-differenced.
- `alignment_reference="exact"` (default) uses the Langevin orientational
  entropy of `FixedDipoleIdeal`. Its coefficients diverge at saturation
  ($A \sim (1-p)^{-2}$, $B \sim (1-p)^{-1}$); they are evaluated at
  $\min(p, 1-\sqrt{\epsilon})$ of the dtype ($p \approx 0.99966$ in float32,
  $1 - 1.5\times10^{-8}$ in float64), so they stay continuous and
  nondecreasing and are held at their boundary value in voxels at or beyond
  saturation, which gridded fields can reach.
  Polarization is required to vanish where density does. `"quadratic"` uses
  $3|\mathbf P|^2/(2m^2\rho)$ everywhere, the polarization probe's reference;
  it underestimates the stiffness of strongly aligned states and flags
  stable bulk (in uniform water at alignment 0.8 its lowest eigenvalue was
  −0.96 where the exact entropy gives +4.0), so use it only for weakly
  polarized fields.
- `probe_density_power` weights both channels as in the single-variable
  probes.

Every real probe (the cosine and the sine of each wavevector) costs $6n$
energy evaluations with independent blocks and $2n(2n+1)$ with the full
matrix, six for one species either way; `perturbations_per_forward` limits
memory.

## Supervised homogeneous Fourier response

`FourierResponseLoss` compares projected homogeneous curvature with explicit
targets. For a one-component fluid, or a true symmetry eigenchannel under the
same normalization, the target is `K(k)=1/S(k)`. For a general mixture, first
construct and invert the full response matrix, then project its inverse using
the same component-space direction convention as the loss.

EquiCDFT deliberately does not read an MD-specific structure-factor format,
interpolate shells, extrapolate, smooth, invert response matrices, or propagate
measurement uncertainty. Those operations belong in a versioned scientific
data-preparation workflow.

```python
response_loss = FourierResponseLoss(
    directions=((1.0, 1.0), (1.0, -1.0)),
    wavevector_indices_key="fourier_wavevector_indices",
    target_key="fourier_curvature",
    scale_key="fourier_scale",
    weights_key="fourier_weight",
    relative_amplitude=0.01,
    perturbations_per_forward=4,
    weight=1.0,
)
```

`FourierResponseData` constructs homogeneous fields from a grid template and
explicit tensors:

```python
response_data = FourierResponseData(
    template=grid_template,
    density=density_by_item,       # [item, component]
    wavevector_indices=wavevector_indices_by_item,
                                    # [item, wavevector, 3]
    curvature=target_by_item,       # [item, wavevector, direction]
    scale=scale_by_item,           # optional, same target shape
    weight=weight_by_item,         # optional, same target shape
    indices=frozen_train_indices,
)
```

Every response item must be uniform, periodic, positive, and unmasked.
Wavevector triplets are integer reciprocal-lattice indices inside the
isotropic Nyquist sphere. Valid sine and cosine estimates are averaged before
applying the loss.

`FourierResponseMetrics` reuses the tensors already computed by the loss and
reports curvature RMSE, scaled curvature RMSE, nonpositive counts, and
diagonal `S=1/K` diagnostics. The latter are not a substitute for inverting a
coupled response matrix.

## Bulk density-polarization response

`BulkResponseLoss` fits the complete second variation of a homogeneous,
unpolarized state to equilibrium spectra: number `S(k)`, longitudinal and
transverse dipole spectra and the density-dipole cross spectrum, plus the
total-dipole fluctuation at `k = 0`. Where `FourierResponseLoss` projects
finite-difference curvatures onto fixed component directions,
`BulkResponseLoss` uses the exact Hessian symbol
(`equicdft.response.bulk.bulk_response_symbol`): the gradient of `beta_F_exc`
is differentiated once more at one voxel per channel, the rows are Fourier
summed, and the ideal gas contributes the identity in the normalized
coordinates

$$
\frac{\delta\rho_a}{\sqrt{\rho_a/\Delta V}},\qquad
\frac{\sqrt3\,\delta\mathbf P_a}{m_a\sqrt{\rho_a/\Delta V}},
$$

in which the covariance of Fourier modes $e^{-i\mathbf k\cdot\mathbf r}$ is
$K(\mathbf k)^{-1}$. The symbol stays differentiable with respect to the model
parameters; on a 16³ grid one evaluation with about a hundred wavevectors
costs a few seconds on a CPU. The model must have
`ideal=FixedDipoleIdeal(...)`.

For an isotropic, inversion-symmetric fluid each wavevector reduces
(`reduce_bulk_symbol`) to four curvature blocks per species pair: density
$K_{\rho\rho}$, longitudinal $K_{LL}$, transverse $K_{TT}$ (mean of the two
transverse directions) and the imaginary cross block $\mathrm{Im}\,K_{\rho L}$.
The largest coupling that this reduction drops (density or longitudinal with
transverse, which a grid functional can produce off the axes) is reported as
`anisotropy`.

A bin's reference is the arithmetic mean of the member spectra, which for
members at different $|\mathbf k|$ is not the inverse of the mean curvature.
With `bin_average="covariance"` (default) the member blocks are inverted to
covariances, averaged over the bin and inverted back, so the prediction is
the same statistic as the reference for any spread of $|\mathbf k|$ within a
bin. A bin with any non-positive-definite member has no covariance; it falls
back to the arithmetic curvature mean and is listed in the
`unstable_bins` entry of the result (the curvature loss still drives it
toward stability). `bin_average="curvature"` always averages curvatures;
for bins that are a cubic orbit of one lattice vector the two modes agree
exactly. Bins of one cubic orbit, or one vector each, are recommended.

Members are padded with `member_mask = False`; padded entries never enter the
symbol or the average. An unmasked member at $\mathbf k = 0$ is an error:
$k = 0$ enters only through the zero-wavevector term. Models with no learned
excess term, or with excess terms that do not couple to polarization, give
the ideal identity for the missing blocks.

Reference spectra follow the simulation definitions
($S_{ab} = \langle n_a n_b^*\rangle/\sqrt{N_aN_b}$,
$G_{L,T} = 3\langle p p^*\rangle/(\sqrt{N_aN_b}\,m_am_b)$ per component,
$C_{ab} = \mathrm{Im}\langle n_a p_{L,b}^*\rangle/(\sqrt{N_aN_b}\,m_b)$) and are
converted to the same blocks by inverting the longitudinal covariance
$[[S, i\sqrt3 C], [-i\sqrt3 C^T, G_L]]$ and $G_T$ (`spectra_to_curvature`;
`curvature_to_spectra` is its inverse). The comparison is in curvature,
$K = S^{-1}$: it is defined for an unstable model and is what the solver sees.
Per element, prediction and target are divided by the reference SEM
propagated linearly to that element, combined in quadrature with
`relative_floor` times the element's size, and passed to `loss_fn` (smooth L1
by default), as in `FourierResponseLoss`.

At `k = 0` a canonical reference has no number fluctuation, so only the
polarization block enters: its isotropic average is compared with the inverse
of the total-dipole fluctuation $g_K$ (for one species
$\varepsilon - 1 = 4\pi\beta\rho m^2 g_K / 3$ in tin-foil boundary conditions).

```python
bulk_response = BulkResponseLoss(
    wavevector_indices_key="response_wavevector_indices",  # [field, bin, member, 3]
    member_mask_key="response_member_mask",                # [field, bin, member]
    spectra_key="response_spectra",                        # [field, bin, 4, n, n]: S, G_L, G_T, C
    spectra_sem_key="response_spectra_sem",                # same shape (optional)
    zero_wavevector_key="response_gk",                     # [field, n, n] (optional)
    zero_wavevector_sem_key="response_gk_sem",
    relative_floor=0.02,
    weight=1.0,  # relative to the data loss; tune per campaign
)
```

The reference should be computed from simulation frames binned exactly as
the training fields are: a voxel functional reproduces the spectra of
voxel-averaged fields, including the assignment window and aliasing, not the
point-particle spectra. Without that, use only wavevectors well below the
grid's Nyquist limit. Bulk response constrains the homogeneous unpolarized
state only; polarized and dense states still need the stability probes.

## Joint field and response training

`TrainingStream` keeps different datasets, losses, metrics, and model-forward
options explicit while sharing one optimizer update:

```python
field_stream = TrainingStream(
    name="field",
    train_loader=field_train_loader,
    valid_loader=field_valid_loader,
    loss=field_loss,
    metrics=field_metrics,
    batches_per_step=1,
)

response_stream = TrainingStream(
    name="response",
    train_loader=response_train_loader,
    valid_loader=response_valid_loader,
    loss=Loss([response_loss]),
    metrics=(FourierResponseMetrics(("number", "charge")),),
    batches_per_step=1,
    model_kwargs={"compute_c1": False, "compute_local_mu": False},
    cycle=True,
)

history = trainer.fit_streams(
    (field_stream, response_stream),
    epochs=epochs,
    stream_weights={"field": 1.0, "response": response_weight},
    record_initial_validation=True,
)
```

Non-cycling streams define epoch length and are consumed once. A cycling
stream restarts as needed and contributes `batches_per_step` batches to every
optimizer update. Losses are averaged within each stream and then multiplied
by the stream weight. Validation is single-pass for every stream, and the
weighted validation total controls scheduling and ordinary `best.pt`
selection. `record_initial_validation=True` records an explicit epoch-zero
baseline before optimization.

Scientific splits must be frozen outside the trainer. In particular, response
states used for fitting or model selection are not independent held-out
structure-factor benchmarks.

## Safeguarded Anderson Euler mixing

The fixed-particle-number Euler solver can optionally propose Anderson-mixed
updates in log-density space:

```python
from equicdft.solver import AndersonOptions, EulerOptions

result = solver.solve(
    field,
    method="euler",
    euler=EulerOptions(anderson=AndersonOptions(
        history=5, regularization=1.0e-8, damping=1.0,
    )),
)
```

Each proposal is accepted only through the solver's existing safeguards;
rejected proposals fall back to the baseline update and reset history when
required. The result records attempts, accepted/rejected proposals, and
resets. Anderson acceleration changes the numerical path, not the free-energy
functional or convergence criterion.

## Implementation map

Paths are relative to `src/equicdft/`.

- `geometry/grid.py`, `geometry/convolution.py`: regular-grid validation,
  neighborhood gathers, and the optional periodic stencil convolution
- `representation/radial.py`: radial formulas, validation, discrete
  conditioning, and pre-invariant transforms
- `representation/cartesian.py`: Cartesian feature ownership, density
  transforms, and radial dispatch
- `representation/message.py`: invariant-to-equivariant message layers
- `representation/local.py`: the A/B/message pipeline shared by the model's
  top-level features and every `CACEBranch`
- `_polarization.py`: the smooth molecular-alignment field shared by the
  polarized learned inputs
- `response/modes.py`, `response/response.py`, `response/curvature.py`: shared
  wavevector validation, perturbations, and projected curvature
- `response/stability.py`: curvature-positivity regularization
- `training/loss.py`, `training/metrics.py`, `response/data.py`: supervised
  response loss, diagnostics, and homogeneous response items
- `training/trainer.py`: explicit multi-stream optimization and epoch-zero
  validation
- `functional/reciprocal.py`, `functional/readout.py`: reciprocal kernels and
  coefficient controls
- `functional/msa.py`: the fixed MSA / Percus-Yevick reference of the
  restricted primitive model
- `functional/fmt.py`: fundamental measure theory with analytic weight
  spectra

The [README's package map](../README.md) describes the layout as a whole.

Before promoting any experimental option into a default workflow, run focused
tests, the complete test suite, and the compact LJ forward/inverse regression;
then evaluate system-appropriate held-out observables without using them for
model selection.
