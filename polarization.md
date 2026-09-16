# Density and polarization features: API and mathematics

`CartesianAFeatures` and `CartesianBFeatures` support both density-only and joint density–polarization models:

```python
include_polarization=False  # Density only
include_polarization=True   # Density and polarization
```

A features collect local Cartesian moments. B features combine those moments into scalar invariants. An ordinary `LocalReadout` maps the invariants to an excess free-energy contribution.

## 1. Minimal API example

The reference scales, grid spacing and thermodynamic constants below are supplied by the application.

```python
from equicdft import (
    CartesianAFeatures,
    CartesianBFeatures,
    GridCACEModel,
    LocalReadout,
)

a_features = CartesianAFeatures(
    mean_density=rho_ref,
    dipole_density_scale=P_ref,
    include_polarization=True,
    separate_center=True,
    n_types=1,
    cutoff_grid=3,
    max_power=1,
    radial_basis="gaussian",
    radial_exponents=(0.125,),
    trainable_radial_exponents=True,
)

b_features = CartesianBFeatures(
    max_power=1,
    max_product_order=2,
    include_polarization=True,
    separate_center=True,
)

short_range = LocalReadout(
    hidden_sizes=(16, 8),
)

model = GridCACEModel(
    a_features=a_features,
    b_features=b_features,
    readout=[short_range],
    grid_spacing=grid_spacing,
    mean_temperature=temperature_ref,
    boltzmann_constant=k_B,
    compute_c1=True,
    compute_polarization_derivative=True,
)
```

`LocalReadout` infers its input dimension on the first forward pass. Run one representative batch before constructing the optimizer or saving the initialized model:

```python
output = model(data)
```

The input fields are:

```python
data["rho"]             # [..., n_grid, n_types]
data["dipole_density"]  # [..., n_grid, n_types, 3]
```

Both must use the same floating-point dtype and device. Construct the remaining periodic-grid metadata with the existing `GridData` interface using `model.grid_info`.

The polarization toggle is fixed when constructing the model. With it enabled, every input must provide `dipole_density`.

## 2. What polarization density means

Write the molecular number density as $\rho(\mathbf r)$ and the electric dipole density as $\mathbf P(\mathbf r)$.

For identical fixed-magnitude molecular dipoles,

$$
\mathbf P(\mathbf r)
=
m\,\rho(\mathbf r)\,
\langle\hat{\boldsymbol\mu}\rangle_{\mathbf r}.
$$

Thus $\mathbf P$ is **dipole moment per volume**, not mean orientation per molecule.

The descriptors use fixed reference scales:

$$
\tilde\rho=\frac{\rho}{\rho_{\rm ref}},
\qquad
\tilde{\mathbf P}=\frac{\mathbf P}{P_{\rm ref}}.
$$

A possible choice for fixed-magnitude dipoles is $P_{\rm ref}=m\rho_{\rm ref}$. The reference scales are not local fields: the construction does **not** divide $\mathbf P$ by the local density.

All three vector components use the same scale. No mean is subtracted.

## 3. A features: shared spatial moments

Let $\mathbf q_j$ denote an integer voxel offset from the evaluation point, and let

$$
\alpha=(\alpha_x,\alpha_y,\alpha_z),
\qquad
\mathbf q_j^\alpha
=q_{jx}^{\alpha_x}q_{jy}^{\alpha_y}q_{jz}^{\alpha_z}.
$$

`max_power` bounds the total spatial degree $|\alpha|=\alpha_x+\alpha_y+\alpha_z$.

For a Gaussian radial channel with zero radial center, the normalized weights are

$$
w_n(\mathbf q_j)=
\frac{\exp[-a_n|\mathbf q_j|^2]}
{\sum_{k\in\mathcal N}\exp[-a_n|\mathbf q_k|^2]}.
$$

The density and polarization moments use **the same weights, stencil and monomials**:

$$
A^\rho_{n,\alpha}(\mathbf r)
=
\sum_{j\in\mathcal N}
w_n(\mathbf q_j)\,
\mathbf q_j^\alpha\,
\tilde\rho(\mathbf r+h\mathbf q_j),
$$

$$
A^P_{n,\alpha,a}(\mathbf r)
=
\sum_{j\in\mathcal N}
w_n(\mathbf q_j)\,
\mathbf q_j^\alpha\,
\tilde P_a(\mathbf r+h\mathbf q_j).
$$

Here $h$ is the grid spacing, and $a\in\{x,y,z\}$ is the independent polarization-vector index. These normalized local averages contain no voxel-volume factor.

For `max_power=1`, useful notation is

$$
s=\sum_j w_j\tilde\rho_j,
\qquad
v_i=\sum_j w_jq_{ji}\tilde\rho_j,
$$

$$
p_a=\sum_j w_j\tilde P_{j,a},
\qquad
D_{ia}=\sum_j w_jq_{ji}\tilde P_{j,a}.
$$

The lower-case $\mathbf p$ denotes a smoothed polarization moment, whereas $\mathbf P$ is the input field.

Under a cubic rotation or reflection $R$, evaluated at corresponding transformed grid points,

$$
s'=s,\qquad
\mathbf v'=R\mathbf v,\qquad
\mathbf p'=R\mathbf p,\qquad
D'=RDR^\mathsf T.
$$

The spatial index and the polarization index both transform. Polarization components are therefore not treated as three additional scalar species.

### Center handling

With `separate_center=True`:

- The center voxel is excluded from the neighbor sums and their normalization.
- Normalized center $\rho$ and $\mathbf P$ are appended as separate A components.
- Those components participate in B symmetrization together with the neighbor moments.

Raw center-vector components are never passed directly to the scalar readout.

## 4. B features: generated scalar invariants

B forms products of A components and averages them over the cubic point group $O_h$, comprising all 48 signed axis permutations.

For a representative product $I=(i_1,\ldots,i_\nu)$,

$$
B_I(A)
=
\frac{1}{48}
\sum_{R\in O_h}
\prod_{\ell=1}^{\nu}
[D(R)A]_{i_\ell},
$$

where $D(R)$ denotes the representation acting on all A components, including their spatial and polarization indices.

`max_product_order` specifies the largest number $\nu$ of A factors in a product. It is distinct from `max_power`, which specifies the spatial degree of each moment.

The implementation generates sparse signed-orbit recipes:

- Symmetry-related products contribute to one invariant.
- Products whose group average vanishes are omitted.
- Equivalent products are not emitted repeatedly.

There is no hand-written polarization contraction list.

Examples of generated features include

$$
s,\qquad
\frac{D_{xx}+D_{yy}+D_{zz}}{3},\qquad
\frac{\mathbf v\cdot\mathbf v}{3},\qquad
\frac{\mathbf v\cdot\mathbf p}{3},\qquad
\frac{|\mathbf p|^2}{3}.
$$

The factors such as $1/3$ arise from orbit averaging, rather than unnormalized tensor contraction.

Center components allow additional invariants such as

$$
s\,\tilde\rho_{\rm center},
\qquad
\frac{\mathbf p\cdot\tilde{\mathbf P}_{\rm center}}{3}.
$$

### Cubic symmetry versus continuous rotations

These features are invariant under $O_h$, not necessarily under arbitrary continuous $O(3)$ rotations.

For example, cubic symmetry permits separate features for

$$
\sum_i D_{ii}^2
\quad\text{and}\quad
\sum_{i\ne j}D_{ij}^2.
$$

Their distinction need not survive an arbitrary rotation of the coordinate axes.

Spatial inversion also does not imply invariance under changing only $\mathbf P\rightarrow-\mathbf P$ while leaving the spatial field unchanged. Mixed terms such as $\mathbf v\cdot\mathbf p$ are retained.

## 5. Channels and readout

The feature layouts are

```text
A: [..., n_grid, n_radial_channels, n_A_components, n_channels]
B: [..., n_grid, n_radial_channels, n_B_invariants, n_channels]
```

B forms products independently within each radial/species channel. The readout combines their flattened outputs.

An optional species-channel transform is applied identically to $\rho$ and each Cartesian component of $\mathbf P$. It mixes species, not vector axes.

`GridCACEModel` flattens the B features and appends $T/T_{\rm ref}$. `LocalReadout` can infer the resulting input dimension lazily.

With separate-center polarization features, center information is already inside B and is not appended again.

## 6. Joint field-gated message passing

Use the existing `BChiMessage` with its polarization toggle matching A and B:

```python
from equicdft import BChiMessage

message = BChiMessage(
    n_invariant_features=b_features.n_features,
    n_radial_channels=a_features.n_radial_channels,
    n_channels=a_features.n_output_channels,
    hidden_sizes=(16, 8),
    include_polarization=True,
    convolution_backend=a_features.convolution_backend,
)
# Pass message_layers=[message] to GridCACEModel in the example above.
```

At layer $t$, one neural map of the full joint invariant vector produces two
scalar gates per radial/channel pair:

$$
a_i^t=h_\rho^t(B_i^t)-h_\rho^t(0),\qquad
b_i^t=h_P^t(B_i^t)-h_P^t(0).
$$

These gates multiply the original normalized fields before stencil aggregation:

$$
u_i^t=a_i^t\tilde\rho_i,\qquad
\mathbf w_i^t=b_i^t\tilde{\mathbf P}_i.
$$

One polarization gate is shared across x/y/z, so the gated polarization remains
a polar vector. Species mixing and reference scales are exactly those of the
initial A features. Each layer uses the original field carriers, not products
of all preceding gates. The physical input fields are never overwritten.

The existing radial–Cartesian convolution aggregates $u$ and $\mathbf w$ into
the same joint A layout. With `separate_center=True`, their gated center values
are appended within each field block. The same joint B constructor then gives
the next invariant level. The lazy readout receives flattened
`[B0, B1, ..., T/T_ref]`; no separate message symmetrizer is needed.

Messages add no voxel-volume factor, extra density normalization or update to
the physical fields used by energy integration, LDA, Coulomb or ideal terms.
Shared and message-owned radial bases and the gather/conv3d/FFT message backends
use the existing stencil machinery. Omitting message radial settings shares
the initial basis; supplying them gives the message its own basis.

The default `BChiMessage(include_polarization=False)` remains the original
scalar gate-only message, without multiplication by rho. Its parameter layout
and operation are unchanged. Joint field gating is an opt-in extension, not
an identical reproduction of the historical scalar-only polarization message.

## 7. Gaussian polarization response

The existing reciprocal features and long-range readout also support direct
polarization coupling:

```python
from equicdft import LongRangeReadout, ReciprocalFeatures

# Supply Gaussian widths in the same length units as grid_spacing.
polarization_features = ReciprocalFeatures(
    radial_exponents=[0.5 * sigma**2 for sigma in gaussian_widths],
    kernel="gaussian",
    variable="dipole_density",
    n_types=1,
)
polarization_lr = LongRangeReadout(
    features=polarization_features,
    hidden_sizes=(16, 16),
)
# Include polarization_lr in GridCACEModel's readout list, alongside the
# desired local and Coulomb branches. The A features above provide rho_ref
# for the existing state normalization. No neighbor list is needed by LR.
```

`variable="rho"` remains the default and preserves the density-only behavior.
Kernel and species counts are inferred from `features` at construction, not
lazily on the first batch. Explicit counts remain supported and must match.
Standalone coefficient contraction without `features` requires `n_kernels`
and defaults to one species unless `n_types` is supplied.
For `variable="dipole_density"`, the input is the physical
`data["dipole_density"]`, not a descriptor-normalized field. No extra charge,
molecular moment or reference-scale factor is applied. `LongRangeReadout`
automatically requires polarization from its features; do not supply
`charges`, `coulomb_amplitude` or the Coulomb-only `include_polarization` flag.

For one species in beta-free-energy mode, this contribution is

$$
\beta F_{G,P}=\frac{1}{2V}\sum_{\mathbf k}\sum_n
c_n(T,\bar\rho)e^{-\alpha_n k^2}
\widehat{\mathbf P}(\mathbf k)^*\cdot\widehat{\mathbf P}(\mathbf k),
\qquad \alpha_n=\frac{\sigma_n^2}{2},
$$

where $\widehat{\mathbf P}=\Delta V\sum_g\mathbf P_g
e^{-i\mathbf k\cdot\mathbf r_g}$. For mixtures, the features use the real
part of $\widehat{\mathbf P}_a^*\cdot\widehat{\mathbf P}_b$, with a factor
of two for off-diagonal species pairs. All three vector components share each
coefficient. With P in charge/length squared, the features have units
charge squared/length and $c_n$ has units length/charge squared.

Unlike density-fluctuation and Coulomb features, this branch retains the
uniform mode: no polarization mean is subtracted and the Gaussian kernel
equals one at $k=0$. It therefore responds to uniform, longitudinal and
transverse polarization. The coefficients are predicted from normalized
temperature and mean species densities by the existing state MLP; derivatives
through the mean density are retained. The default zero initialization leaves
an existing model unchanged until this branch is trained.

The physical Coulomb branch still uses the source
$q_a\widehat\rho_a-i\mathbf k_D\cdot\widehat{\mathbf P}_a$ through
`include_polarization=True` with Coulomb features and explicit charges.
It remains separate and its zero-mode convention is unchanged. The Gaussian
branch also acts longitudinally: it is a learned residual response fitted
jointly with the fixed Coulomb contribution, not another fixed electrostatic
term. Density and polarization Gaussian readouts have independent coefficients.

The exact ideal orientational term remains necessary. Gaussian coefficients
are signed, like the existing density LR coefficients; neither this branch nor
its presence alongside the ideal term guarantees positive total curvature.
Broad Gaussians weaken at high wavevectors. Widths, regularization and any
stability constraints remain application choices, not implicit API defaults.

### Distinct longitudinal and transverse response

To add a separate finite-wavelength longitudinal correction, construct the
same features with `include_divergence=True`:

```python
polarization_features = ReciprocalFeatures(
    radial_exponents=[0.5 * sigma**2 for sigma in gaussian_widths],
    variable="dipole_density",
    kernel="gaussian",
    include_divergence=True,
)
polarization_lr = LongRangeReadout(
    features=polarization_features,
)
```

For N radial exponents, the output has 2N kernel channels: all N vector
dot-product features first, followed by N divergence-pair features in the same
radial order. The state network supplies independent signed coefficients for
both blocks. For one species,

$$
\beta F_{G,P}=\frac{1}{2V}\sum_{\mathbf k}
\left[A(k)|\widehat{\mathbf P}|^2
+B(k)|\mathbf k_D\cdot\widehat{\mathbf P}|^2\right],
$$

$$
A(k)=\sum_n a_n e^{-\alpha_n k^2},\qquad
B(k)=\sum_n b_n e^{-\alpha_n k^2}.
$$

For resolved modes away from Nyquist, $\mathbf k_D=\mathbf k$ and the added
transverse and longitudinal stiffnesses are $G_T=A$ and $G_L=A+k^2B$.
Their difference vanishes at $k=0$, preserving a common isotropic uniform
response and leaving the asymptotic dipolar interaction in the fixed Coulomb
branch. Setting the B coefficients to zero recovers the unsplit Gaussian
functional. This is a residual-response parameterization, not a positivity
constraint or a replacement for the ideal term.

The divergence uses the same real spectral derivative as Coulomb: each
even-axis Nyquist derivative component is zero. The radial Gaussian still
uses the ordinary FFT wavevectors. B therefore vanishes on uniform and pure
Nyquist patterns; A retains them. On mixed Nyquist modes the tensor is
$A I+B\mathbf k_D\mathbf k_D$, not the continuum projector constructed from
ordinary $\mathbf k$.

No width, molecular-moment, beta or descriptor-reference factor is inserted
into these features. The B features have units charge squared/length cubed,
so $b_n$ has units length cubed/charge squared in beta-free-energy mode.
The A units remain those stated above. Mixture B features use the same
unique species pairs and off-diagonal factor of two as A. The default
`include_divergence=False` preserves existing Gaussian-P, density and Coulomb
models; this option is only available for direct Gaussian polarization.

### Density-only use of the same API

```python
density_features = ReciprocalFeatures(
    radial_exponents=[0.5 * sigma**2 for sigma in gaussian_widths],
    variable="rho",  # Default; no polarization input is required.
    kernel="gaussian",
    n_types=1,
)
density_lr = LongRangeReadout(features=density_features, hidden_sizes=(16, 16))
```

For one species, write the physical density fluctuation as
$\delta\rho=\rho-\bar\rho$ and its continuum-normalized discrete Fourier
transform as $\widehat{\delta\rho}=\Delta V\,\mathrm{FFT}(\delta\rho)$.
The Gaussian features and their energy contribution are

$$
X_n=\frac{1}{2V}\sum_{\mathbf k\ne0}
e^{-\alpha_n k^2}|\widehat{\delta\rho}(\mathbf k)|^2,
\qquad
\beta F_{G,\rho}=\sum_n c_n X_n.
$$

Here $V$ is the cell volume, $\alpha_n=\sigma_n^2/2$ is fixed, and the
readout learns signed coefficients $c_n$ from normalized temperature and
mean density. The Fourier input is not divided by a reference density.
For number density in inverse volume, $X_n$ has inverse-volume units and
$c_n$ has volume units in beta-free-energy mode. Uniform density contributes
zero to this branch; retain the appropriate local/bulk and ideal terms.
This differs from direct polarization mode, which retains the uniform mode.

For mixtures, set `n_types` on the feature module. The readout infers it and
learns coefficients for every radial channel and unique species pair, with
off-diagonal pairs counted twice. `include_divergence` is not applicable to
scalar density. Fixed-charge Coulomb remains a separate option using
`kernel="coulomb"`, charges and the desired Coulomb amplitude; the Gaussian
density example above is not itself an electrostatic kernel.

## Training-only SEM noise augmentation

Load componentwise uncertainties on the final training grid:

```python
data = GridData.from_xyz(
    path,
    data_key={"dipole_density_std": "dipole_density_sem"},
    # Usual grid and unit arguments.
)
trainer = Trainer(
    model=model,
    loss=loss,
    polarization_noise=True,
    noise_dipole_magnitude=m,
)
```

Here `m` is the positive, fixed **single-particle dipole magnitude**, in the
units of P/rho; it may be scalar or one value per species. It is not mass,
a loss weight, or the descriptor reference scale. `dipole_density_std` has
exactly the same shape as P: `[n_grid, n_types, 3]`, or its batched form.
The loader requires finite nonnegative values and preserves their units.
For SEM input, no additional beta, m, reference-density or square-root-of-
sample-count factor is applied.

Each training visit proposes `P' = P + dipole_density_std * randn_like(P)`.
Draws are independent over voxels, species and Cartesian components. Proposals
violating `rho > 0` or `|P| < m*rho` are rejected and redrawn as whole vectors,
not componentwise clipped. After 1,000 unsuccessful attempts the update fails
explicitly. If density augmentation is also enabled with `density_noise=True`
and `rho_std` (e.g. mapped to `density_sem`), the entire `(rho', P')` proposal
is redrawn together; `density_noise_floor` defaults to zero. The scalar-fluid
density-noise API retains its original Gaussian-plus-floor behavior.

The optional original `valid` voxel mask is kept fixed, and its false cells
are unchanged. Excluded cells stay zero and must have zero SEM. Exact vacuum
is unchanged. Unmasked nonvacuum inputs must already satisfy the ideal domain.
No particle-number normalization or subtraction of mean polarization is applied.

Both ideal-derived targets change when either field changes. The trainer
refreshes recognized scalar targets `c1_plus_beta_mu`, `c1`, and `target_c1`
by adding the change in `FixedDipoleIdeal.density_derivative`; it refreshes
`polarization_derivative` and `target_P_derivative` by subtracting the change
in the ideal polarization derivative. The latter names must denote **excess**
derivative targets `beta*E_ext - ideal_P`, not electric-field values themselves.
This preserves external fields, chemical-potential/gauge offsets and thermal-
wavelength conventions. Targets may be full-grid tensors or packed using the
original `valid` mask. Custom target names are untouched and must be computed
from the live perturbed fields by the application.

All flags default off. Validation, explicit evaluation and lazy initialization
remain clean. Ordinary and per-`TrainingStream` settings and PyTorch RNG states
are restored by checkpoints; older checkpoints without flags restore noise off.
Turning noise on for a new fit therefore requires an explicit configuration,
not resuming an old clean checkpoint unchanged.

This is diagonal-SEM input regularization, not covariance-aware sampling or
inverse-variance loss weighting. Truncation changes the proposal distribution
near the physical bound. SEM components are not a polar vector; general rotated
covariances cannot be recovered from three diagonal values. The loader refuses
uncertainty coarsening without covariance information. Noise augmentation alone
does not guarantee a stable or more accurate learned free energy.
