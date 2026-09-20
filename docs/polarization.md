# Density and polarization features: API and mathematics

`CartesianAFeatures` and `CartesianBFeatures` support both density-only and joint density–polarization models:

Use `include_polarization=False` for density only and
`include_polarization=True` for density and polarization.

A features collect local Cartesian moments. B features combine those moments into scalar invariants. An ordinary `LocalReadout` maps the invariants to an excess free-energy contribution.

## 1. Minimal API example

Each Python block below is a standalone helper definition, with its own
imports and explicit inputs. Defining a helper does not load data or start
training. Call it with your application's physical parameters; no dataset,
model, loss or units are assumed to exist in the Python session.

The first helper constructs a one-species short-range excess functional plus
its fixed-dipole ideal reference.
Its required inputs are:

- `density_reference`: a positive number-density scale, in inverse volume.
- `dipole_density_reference`: a positive dipole-density scale, in dipole moment
  per volume. It is not the molecular dipole magnitude.
- `dipole_magnitude`: the physical fixed molecular dipole moment, not a descriptor scale.
- `grid_spacing`: voxel spacing, in your chosen length unit.
- `temperature_reference`: a positive reference temperature.
- `boltzmann_constant`: Boltzmann's constant in your energy/temperature units.

For fixed-magnitude dipoles, one possible normalization is
`dipole_density_reference = dipole_magnitude * density_reference`.
Reference scales are application choices, not universal API defaults.

```python
from equicdft import (
    CartesianAFeatures,
    CartesianBFeatures,
    GridCACEModel,
    LocalReadout,
    FixedDipoleIdeal,
)

def build_short_range_model(
    *,
    density_reference,
    dipole_density_reference,
    dipole_magnitude,
    grid_spacing,
    temperature_reference,
    boltzmann_constant,
):
    a_features = CartesianAFeatures(
        mean_density=density_reference,
        dipole_density_scale=dipole_density_reference,
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
    return GridCACEModel(
        a_features=a_features,
        b_features=b_features,
        readout=[LocalReadout(hidden_sizes=(16, 8))],
        ideal=FixedDipoleIdeal(dipole_magnitude),
        grid_spacing=grid_spacing,
        mean_temperature=temperature_reference,
        boltzmann_constant=boltzmann_constant,
        compute_c1=True,
        compute_polarization_derivative=True,
    )
```

`GridCACEModel` binds `LocalReadout` to the width of the local feature vector
at construction, so the model is complete (and can be optimized or saved)
as soon as it is built. The model includes ideal and short-range excess terms;
Coulomb and other long-range contributions must be added when required by the system.

The input fields are:

```text
data["rho"]             # [..., n_grid, n_types]
data["dipole_density"]  # [..., n_grid, n_types, 3]
```

Both must use the same floating-point dtype and device. Construct the remaining periodic-grid metadata with the existing `GridData` interface using `model.grid_info`.

The polarization toggle is fixed when constructing the model. With it enabled, every input must provide `dipole_density`.

## 2. What polarization density means

Write the molecular number density as $\rho(\mathbf{r})$ and the electric dipole density as $\mathbf{P}(\mathbf{r})$.

For identical fixed-magnitude molecular dipoles,

$$
\mathbf{P}(\mathbf{r})
=
m\,\rho(\mathbf{r})\,
\langle\hat{\boldsymbol{\mu}}\rangle_{\mathbf{r}}.
$$

Here $m$ is the fixed single-molecule dipole magnitude, and the angle brackets
denote the local mean unit orientation. Thus $\mathbf{P}$ is **dipole moment
per volume**, not mean orientation per molecule.

The descriptors use fixed reference scales:

$$
\tilde{\rho}=\frac{\rho}{\rho_{\mathrm{ref}}},
\qquad
\tilde{\mathbf{P}}=\frac{\mathbf{P}}{P_{\mathrm{ref}}}.
$$

A possible choice for fixed-magnitude dipoles is $P_{\mathrm{ref}}=m\rho_{\mathrm{ref}}$. The reference scales are not local fields: the construction does **not** divide $\mathbf{P}$ by the local density.

All three vector components use the same scale. No mean is subtracted.

## 3. A features: shared spatial moments

Let $\mathbf{q}_j$ denote an integer voxel offset from the evaluation point, and let

$$
\alpha=(\alpha_x,\alpha_y,\alpha_z),
\qquad
\mathbf{q}_j^\alpha
=q_{jx}^{\alpha_x}q_{jy}^{\alpha_y}q_{jz}^{\alpha_z}.
$$

`max_power` bounds the total spatial degree $|\alpha|=\alpha_x+\alpha_y+\alpha_z$.

For a Gaussian radial channel with zero radial center, the normalized weights are

$$
w_n(\mathbf{q}_j)=
\frac{\exp[-a_n|\mathbf{q}_j|^2]}
{\sum_{k\in\mathcal{N}}\exp[-a_n|\mathbf{q}_k|^2]}.
$$

The density and polarization moments use **the same weights, stencil and monomials**:

$$
A^\rho_{n,\alpha}(\mathbf{r})
=
\sum_{j\in\mathcal{N}}
w_n(\mathbf{q}_j)\,
\mathbf{q}_j^\alpha\,
\tilde{\rho}(\mathbf{r}+h\mathbf{q}_j),
$$

$$
A^P_{n,\alpha,a}(\mathbf{r})
=
\sum_{j\in\mathcal{N}}
w_n(\mathbf{q}_j)\,
\mathbf{q}_j^\alpha\,
\tilde{P}_a(\mathbf{r}+h\mathbf{q}_j).
$$

Here $h$ is the grid spacing, and $a\in\{x,y,z\}$ is the independent polarization-vector index. These normalized local averages contain no voxel-volume factor.

For `max_power=1`, useful notation is

$$
s=\sum_j w_j\tilde{\rho}_j,
\qquad
v_i=\sum_j w_jq_{ji}\tilde{\rho}_j,
$$

$$
p_a=\sum_j w_j\tilde{P}_{j,a},
\qquad
D_{ia}=\sum_j w_jq_{ji}\tilde{P}_{j,a}.
$$

The lower-case $\mathbf{p}$ denotes a smoothed polarization moment, whereas $\mathbf{P}$ is the input field.

Under a cubic rotation or reflection $R$, evaluated at corresponding transformed grid points,

$$
s'=s,\qquad
\mathbf{v}'=R\mathbf{v},\qquad
\mathbf{p}'=R\mathbf{p},\qquad
D'=RDR^{\mathsf{T}}.
$$

The spatial index and the polarization index both transform. Polarization components are therefore not treated as three additional scalar species.

### Center handling

With `separate_center=True`:

- The center voxel is excluded from the neighbor sums and their normalization.
- Normalized center $\rho$ and $\mathbf{P}$ are appended as separate A components.
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
\frac{\mathbf{v}\cdot\mathbf{v}}{3},
$$

$$
\frac{\mathbf{v}\cdot\mathbf{p}}{3},\qquad
\frac{|\mathbf{p}|^2}{3}.
$$

The factors such as $1/3$ arise from orbit averaging, rather than unnormalized tensor contraction.

Center components allow additional invariants such as

$$
s\,\tilde{\rho}_{\mathrm{center}},
\qquad
\frac{\mathbf{p}\cdot\tilde{\mathbf{P}}_{\mathrm{center}}}{3}.
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

Spatial inversion also does not imply invariance under changing only $\mathbf{P}\rightarrow-\mathbf{P}$ while leaving the spatial field unchanged. Mixed terms such as $\mathbf{v}\cdot\mathbf{p}$ are retained.

## 5. Channels and readout

The feature layouts are

```text
A: [..., n_grid, n_radial_channels, n_A_components, n_channels]
B: [..., n_grid, n_radial_channels, n_B_invariants, n_channels]
```

B forms products independently within each radial/species channel. The readout combines their flattened outputs.

An optional species-channel transform is applied identically to $\rho$ and each Cartesian component of $\mathbf{P}$. It mixes species, not vector axes.

`GridCACEModel` flattens the B features and appends $T/T_{\mathrm{ref}}$. `LocalReadout` can infer the resulting input dimension lazily.

With separate-center polarization features, center information is already inside B and is not appended again.

## 6. Joint field-gated message passing

Use the existing `BChiMessage` with its polarization toggle matching A and B:

The inputs below are the already-constructed joint A and B feature modules.
Pass the returned message in `message_layers` when constructing `GridCACEModel`.

```python
from equicdft import BChiMessage

def build_polarization_message(a_features, b_features):
    return BChiMessage(
        n_invariant_features=b_features.n_features,
        n_radial_channels=a_features.n_radial_channels,
        n_channels=a_features.n_output_channels,
        hidden_sizes=(16, 8),
        include_polarization=True,
        convolution_backend=a_features.convolution_backend,
    )
```

At layer $t$, one neural map of the full joint invariant vector produces two
scalar gates per radial/channel pair:

$$
a_i^t=h_\rho^t(B_i^t)-h_\rho^t(0),\qquad
b_i^t=h_P^t(B_i^t)-h_P^t(0).
$$

These gates multiply the original normalized fields before stencil aggregation:

$$
u_i^t=a_i^t\tilde{\rho}_i,\qquad
\mathbf{w}_i^t=b_i^t\tilde{\mathbf{P}}_i.
$$

One polarization gate is shared across x/y/z, so the gated polarization remains
a polar vector. Species mixing and reference scales are exactly those of the
initial A features. Each layer uses the original field carriers, not products
of all preceding gates. The physical input fields are never overwritten.

The existing radial–Cartesian convolution aggregates $u$ and $\mathbf{w}$ into
the same joint A layout. With `separate_center=True`, their gated center values
are appended within each field block. The same joint B constructor then gives
the next invariant level. The local readout, bound by the model to the width
of `[B0, B1, ..., T/T_ref]`, receives that flattened vector; no separate
message symmetrizer is needed.

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
polarization coupling. `gaussian_widths` below is a nonempty sequence of
positive widths in the same length units as `grid_spacing`; `n_types` is the
number of species, not the number of Cartesian components. These widths are
fixed. The readout learns their coefficients.

```python
from equicdft import LongRangeReadout, ReciprocalFeatures

def build_polarization_lr(
    gaussian_widths, *, n_types=1, include_divergence=False
):
    features = ReciprocalFeatures(
        radial_exponents=[0.5 * sigma**2 for sigma in gaussian_widths],
        kernel="gaussian",
        variable="dipole_density",
        n_types=n_types,
        include_divergence=include_divergence,
    )
    return LongRangeReadout(features=features, hidden_sizes=(16, 16))
```

Include the returned readout in `GridCACEModel`'s `readout` list alongside
the desired local and Coulomb branches. The model also needs a density
reference for state normalization, supplied by its A features or another
compatible readout. This LR branch itself needs no neighbor list.

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
\beta F_{G,P}=\frac{1}{2V}\sum_{\mathbf{k}}\sum_n
c_n(T,\bar{\rho})e^{-\alpha_n k^2}
\widehat{\mathbf{P}}(\mathbf{k})^*\cdot\widehat{\mathbf{P}}(\mathbf{k}).
$$

Here $\alpha_n=\sigma_n^2/2$, $V$ is the cell volume and $\Delta V$ is the voxel volume. The Fourier
normalization is

$$
\widehat{\mathbf{P}}(\mathbf{k})
=\Delta V\sum_g\mathbf{P}_g e^{-i\mathbf{k}\cdot\mathbf{r}_g}.
$$

For mixtures, the features use the real
part of $\widehat{\mathbf{P}}_a^*\cdot\widehat{\mathbf{P}}_b$, with a factor
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
$q_a\widehat{\rho}_a-i\mathbf{k}_D\cdot\widehat{\mathbf{P}}_a$ through
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

To add a separate finite-wavelength longitudinal correction, pass
`include_divergence=True` to `build_polarization_lr` above. It forwards that
option to `ReciprocalFeatures`; no other construction changes are needed.

For N radial exponents, the output has 2N kernel channels: all N vector
dot-product features first, followed by N divergence-pair features in the same
radial order. The state network supplies independent signed coefficients for
both blocks. For one species,

$$
\beta F_{G,P}=\frac{1}{2V}\sum_{\mathbf{k}}
\left[A(k)|\widehat{\mathbf{P}}|^2
+B(k)|\mathbf{k}_D\cdot\widehat{\mathbf{P}}|^2\right],
$$

$$
A(k)=\sum_n a_n e^{-\alpha_n k^2},\qquad
B(k)=\sum_n b_n e^{-\alpha_n k^2}.
$$

For resolved modes away from Nyquist, $\mathbf{k}_D=\mathbf{k}$ and the added
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
$A I+B\mathbf{k}_D\mathbf{k}_D^{\mathsf{T}}$, not the continuum projector constructed from
ordinary $\mathbf{k}$.

No width, molecular-moment, beta or descriptor-reference factor is inserted
into these features. The B features have units charge squared/length cubed,
so $b_n$ has units length cubed/charge squared in beta-free-energy mode.
The A units remain those stated above. Mixture B features use the same
unique species pairs and off-diagonal factor of two as A. The default
`include_divergence=False` preserves existing Gaussian-P, density and Coulomb
models; this option is only available for direct Gaussian polarization.

### Density-only use of the same API

This independent helper accepts the same physical Gaussian widths and species
count, but requires only density fields.

```python
from equicdft import LongRangeReadout, ReciprocalFeatures


def build_density_lr(gaussian_widths, *, n_types=1):
    features = ReciprocalFeatures(
        radial_exponents=[0.5 * sigma**2 for sigma in gaussian_widths],
        variable="rho",  # Default; no polarization input is required.
        kernel="gaussian",
        n_types=n_types,
    )
    return LongRangeReadout(features=features, hidden_sizes=(16, 16))
```

For one species, write the physical density fluctuation as
$\delta\rho=\rho-\bar{\rho}$ and its continuum-normalized discrete Fourier
transform as $\widehat{\delta\rho}=\Delta V\,\mathrm{FFT}(\delta\rho)$.
The Gaussian features and their energy contribution are

$$
X_n=\frac{1}{2V}\sum_{\mathbf{k}\ne0}
e^{-\alpha_n k^2}|\widehat{\delta\rho}(\mathbf{k})|^2,
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

## 8. Training-only SEM noise augmentation

The loader helper takes an ExtXYZ `path` and a constructed `model`, using
`model.grid_info` for grid and unit conventions. It expects the usual density,
polarization and grid properties plus `density_sem` and `dipole_density_sem`,
both on the final training grid. It returns a list of frames, not a data loader.

The trainer helper takes that model, an application-defined `equicdft.Loss`,
and obtains the physical `dipole_magnitude` from `model.ideal`. This is the fixed single-particle
dipole moment in units of polarization density divided by number density
(for example, e Å if density is in Å⁻³ and polarization density in e Å⁻²).
It is **not** mass, a loss weight, or the descriptor normalization scale.
`learning_rate` is also explicit; these helpers do not choose a training
objective, dataset split or training duration.

```python
from equicdft import GridData, NoiseOptions, Trainer


def load_sem_frames(path, model):
    return GridData.from_xyz(
        path,
        grid_info=model.grid_info,
        include_local_density_index=model.requires_local_density_index,
        data_key={
            "rho_std": "density_sem",
            "dipole_density_std": "dipole_density_sem",
        },
    )


def build_sem_trainer(model, loss, *, learning_rate):
    return Trainer(
        model=model,
        loss=loss,
        optimizer_args={"lr": learning_rate},
        noise=NoiseOptions(
            density=True,
            polarization=True,
        ),
    )
```

This example enables both density and polarization noise. For polarization
noise alone, use `NoiseOptions(polarization=True)` and omit the density-SEM mapping if
that property is absent. `dipole_magnitude` must be positive and may be a
scalar or one value per species; it is the quantity denoted by $m$ in the
equations. It must come from the physical particle model, not from fitting.

The loader assumes that density and polarization describe the same species.
Raw site-resolved files (for example, separate O/H densities but one molecular
polarization field) require application-side species selection first; merely
renaming SEM properties does not perform that conversion.

`dipole_density_std` has
exactly the same shape as P: `[n_grid, n_types, 3]`, or its batched form.
The loader requires finite nonnegative values and preserves their units.
For SEM input, no additional beta, m, reference-density or square-root-of-
sample-count factor is applied.

Each training visit proposes `P' = P + dipole_density_std * randn_like(P)`.
Draws are independent over voxels, species and Cartesian components. Proposals
violating `rho > 0` or `|P| < m*rho` are rejected and redrawn as whole vectors,
not componentwise clipped. After 1,000 unsuccessful attempts the update fails
explicitly. If density augmentation is also enabled with `density=True`
and `rho_std` (e.g. mapped to `density_sem`), the entire `(rho', P')` proposal
is redrawn together; `density_floor` defaults to zero. The scalar-fluid
density-noise API retains its original Gaussian-plus-floor behavior.

The optional original `valid` voxel mask is kept fixed, and its false cells
are unchanged. Excluded cells stay zero and must have zero SEM. Exact vacuum
is unchanged. Unmasked nonvacuum inputs must already satisfy the ideal domain.
No particle-number normalization or subtraction of mean polarization is applied.

Augmentation changes only the input fields; **every target stays unchanged**.
Fit `density_derivative` against `-beta*V_ext` with the canonical gauge
projection, and `polarization_derivative` against `beta*E_ext`. If absolute
chemical potentials are known, the density target is `beta*(mu - V_ext)`
without the canonical projection. These total derivatives already include
the model's ideal response to the noisy fields. Keep the original masks and
loss scales, including any physical molecular moment multiplying the P
residual inside its square. No target-name detection or refresh is performed.
The loader no longer constructs `c1` or `c1_plus_beta_mu` targets. Model
outputs `c1`, `c2` and explicit excess derivatives remain available for
direct-correlation and response analysis.

`NoiseOptions` stores only `density`, `polarization` and `density_floor`.
Standalone augmentation of polarized fields must also receive the ideal
reference, for example `options.apply(batch, ideal=model.ideal)`; the trainer
supplies it automatically. There is no second dipole-magnitude setting.

All flags default off. Validation and explicit evaluation remain clean. Ordinary and per-`TrainingStream` settings and PyTorch RNG states
are restored by current-format checkpoints. Legacy checkpoint resume is unsupported.
Turning noise on for a new fit therefore requires an explicit configuration,
not resuming an old clean checkpoint unchanged.

This is diagonal-SEM input regularization, not covariance-aware sampling or
inverse-variance loss weighting. Truncation changes the proposal distribution
near the physical bound. SEM components are not a polar vector; general rotated
covariances cannot be recovered from three diagonal values. The loader refuses
uncertainty coarsening without covariance information. Noise augmentation alone
does not guarantee a stable or more accurate learned free energy.

## 9. Model-owned ideal energy, field fitting and solving

`GridCACEModel` owns the full intrinsic functional. Its readouts still compute
only excess contributions. Choose `ideal=IdealGas(...)` for scalar density or
`ideal=FixedDipoleIdeal(dipole_magnitude, ...)` for density and fixed dipoles.
The latter already includes translational entropy: do not add another ideal gas.
Both accept `thermal_wavelength`, a positive scalar or per-species vector in
the chosen length units, defaulting to one.

For each species, define $u=|\mathbf{P}|/(m\rho)$ and
$x=L^{-1}(u)$, where $L(x)=\coth x-1/x$. With normalized angular measure,

$$
\beta f_{\mathrm{id}}
=\rho\left[\ln(\rho\Lambda^3)-1
+ux-\ln\left(\frac{\sinh x}{x}\right)\right].
$$

The orientational part vanishes at $\mathbf{P}=0$, recovering the scalar
ideal gas. Its domain is $\rho>0$ and $|\mathbf{P}|<m\rho$.
Exact vacuum with zero polarization has zero energy; returned ideal derivatives
there are zero placeholders, marked by `ideal_derivative_mask=False`, not
finite physical vacuum derivatives. `excluded_mask` requires zero fields;
a statistical training mask does not change the free energy.

For normal evaluation, the model returns:

| Output | Meaning |
| --- | --- |
| `beta_F_id`, `beta_F_exc`, `beta_F` | Ideal, excess and total intrinsic beta free energies |
| `ideal_density_derivative` | Ideal derivative with respect to density at fixed polarization |
| `ideal_polarization_derivative` | Ideal derivative with respect to polarization at fixed density |
| `density_derivative` | Total density derivative, `ideal_density_derivative - c1` |
| `density_derivative_excess` | Excess density derivative, `-c1` |
| `polarization_derivative` | Ideal plus excess polarization derivative |
| `polarization_derivative_excess` | Excess polarization derivative |

Total density derivatives require `compute_c1=True`; total polarization
derivatives require `compute_polarization_derivative=True`. The existing `c1`
and `c2` retain their **excess-only** definitions. When `compute_ideal=False`,
the total derivative keys are absent; only excess responses are returned.
Derivatives use continuum normalization, including division by voxel volume.
External fields and chemical potentials are not part of the intrinsic model.

Migration: the former excess-only `polarization_derivative` is now
`polarization_derivative_excess`; the former `total_polarization_derivative`
is now `polarization_derivative`. Update loss prediction keys, target names
and analysis scripts explicitly. Coupled Euler updates use the excess response;
stationarity residuals use the total response. Fitted tensors are unchanged.

Both ideal references expose `local(rho, dipole_density=None,
thermal_wavelength=None)` for `energy_per_particle` (beta units) and the
`ideal_*_derivative` values. Their shared `evaluate(data, voxel_volume)`
handles vacuum/exclusions and integrates rho times the local per-particle
energy. The dipole-only callable interface and scalar-only `per_particle`
method have been removed. Scalar minimization retains higher-precision
accumulation using this common local interface.

`GridSolver(model)` obtains the same ideal reference and total derivatives
from the model; there is no solver-side molecular-moment argument. It adds
the external coupling $\int(\rho V_{\mathrm{ext}}-\mathbf{P}\cdot\mathbf{E}_{\mathrm{ext}})d\mathbf{r}$
and applies ensemble constraints. For fixed particle numbers, equilibrium obeys

$$
\frac{\delta\beta F}{\delta\rho}+\beta V_{\mathrm{ext}}=\beta\mu,
\qquad
\frac{\delta\beta F}{\delta\mathbf{P}}=\beta\mathbf{E}_{\mathrm{ext}}.
$$

Solver evaluation uses `solver.evaluate(data, compute_derivatives=True)` for
both scalar and coupled fields. This solver keyword replaces `compute_c1`;
direct model calls keep their separate derivative flags. Set it to `False`
to skip differentiated outputs; analytic ideal-reference diagnostics may
remain. The solver never requests `c2` and does not change model defaults.

Both paths return `beta_external` and `beta_A = beta_F + beta_external`.
For polarized fields, `beta_V_ext` is the density coupling and `beta_P_ext`
is the **negative** electric-field coupling; their sum is `beta_external`
up to floating-point summation order. Density diagnostics are named
`density_residual`, `max_density_residual` and `rms_density_residual`, with
the historical Euler names retained as aliases. Coupled convergence also
includes the molecular-moment-scaled polarization residual. A bounded solve
reports its projected/KKT norms; the raw `polarization_residual` remains
available separately from `polarization_kkt_residual`.

Evaluation applies no density/polarization bounds or homogeneous projections;
those belong to `solve`. Scalar evaluation supports batches and either a
prescribed chemical potential or canonical mean removal. Fixed-dipole
evaluation remains canonical, unbatched, and requires both `V_ext` and
`E_ext`. The scalar and coupled numerical update kernels remain separate.
Solver evaluation returns detached numerical tensors for both systems, without
marking caller fields for gradients. `compute_derivatives=True` still computes
the requested derivatives internally; it does not retain their autograd graph.
For training and differentiable observables, including higher derivatives, use
the model directly. Solves do not retain a graph through their iterations.

Canonical fitting can therefore use two ordinary `TensorLoss` terms, without
a script-defined Euler loss. Construct the model with `compute_local_mu=True`
and the desired `rho_min`. Its local chemical potential includes the total
density derivative, including ideal orientation entropy. The following
standalone helper uses a fixed density mask and the model's independent
mean chemical potential for each density field/species.
Polarization is not mean-centered. Pass **one unbatched frame** to the data
helper before batching. `beta` is inverse energy; V is energy per particle,
and E is energy per dipole moment.

```python
from equicdft import FixedDipoleIdeal, Loss, TensorLoss


def prepare_field_targets(frame, model):
    result = dict(frame)
    result["beta_E_target"] = frame["beta"] * frame["E_ext"]
    mask = model.beta_mu_weights(frame)
    result["beta_mu_weights"] = mask
    result["polarization_weights"] = mask[..., None].expand_as(
        frame["dipole_density"]
    )
    return result


def build_joint_field_loss(model, *, density_weight=1.0, polarization_weight=1.0):
    if not isinstance(model.ideal, FixedDipoleIdeal):
        raise TypeError("joint field fitting requires FixedDipoleIdeal")
    if not model.compute_local_mu:
        raise ValueError("construct the model with compute_local_mu=True")
    moment = model.ideal.dipole_magnitude.reshape(-1, 1)
    return Loss([
        TensorLoss(
            "beta_mu_local", "beta_mu_local", "beta_mu_average",
            weights_key="beta_mu_weights", weight=density_weight,
        ),
        TensorLoss(
            "polarization", "polarization_derivative", "beta_E_target",
            residual_scale=moment, weights_key="polarization_weights",
            weight=polarization_weight,
        ),
    ])
```

Explicitly, with binary mask $w$, $C_w$ subtracting the weighted spatial
mean separately for each case/species, and $D_\rho,D_P$ denoting total
intrinsic derivatives, this helper computes

$$
\mathcal{L}_\rho=\lambda_\rho
\frac{\sum w\,[C_w(D_\rho+\beta V_{\mathrm{ext}})]^2}{\sum w},
\qquad
\mathcal{L}_P=\lambda_P
\frac{\sum_{a=1}^3\sum w\,[m(D_{P,a}-\beta E_{\mathrm{ext},a})]^2}
{3\sum w}.
$$

Sums include batch, voxel and species indices. Cases are weighted by their
number of included entries, not independently normalized per case. The defaults
are $\lambda_\rho=\lambda_P=1$. The physical per-species moment $m$ is inside
the square to make both residuals dimensionless; it is not a statistical task
weight. No sigma normalization or inverse-SEM weighting is applied.
External fields and original masks stay fixed under SEM augmentation. The model
reevaluates total derivatives, `beta_mu_local` and
`beta_mu_average` on the perturbed fields. Both named chemical
potentials represent dimensionless beta mu. The density term is exactly the
spatial-constancy objective above, not a new custom loss. Solver chemical
potentials use their own physical constraints and are independent of this
fitting mask.

### Excess-only evaluation and old models

Use `model(data, compute_ideal=False, compute_local_mu=False)` when only the excess functional is
needed. Fourier stability probes use this path so their intentionally relaxed
polarization perturbations do not trigger the exact ideal-domain constraint;
their separate quadratic ideal reference is unchanged. A polarization model
built without an ideal reference remains excess-only and cannot be solved.

New model files store the ideal configuration and parameters. Older model
formats require explicit conversion. Pre-format-3 polarization files require
a `FixedDipoleIdeal` with the original physical moment and wavelength:
descriptor normalization scales are not a reliable source of these parameters.

```python
from equicdft import FixedDipoleIdeal, load_model
from equicdft.conversion import convert_model


def migrate_polarization_model(source, destination, *, dipole_magnitude,
                               thermal_wavelength=1.0):
    convert_model(
        source, destination,
        ideal=FixedDipoleIdeal(
            dipole_magnitude, thermal_wavelength=thermal_wavelength,
        ),
    )
    return load_model(destination)
```

This helper accepts trusted whole-object and old versioned files that lack
an ideal reference, and requires a distinct destination to preserve the source.
The command `python -m equicdft.conversion` also accepts
`--dipole-magnitude` and `--thermal-wavelength`. Do not supply an ideal
override for a model that already stores its ideal reference.

Historical conversion resets `compute_local_mu=False` and `rho_min=0`, without
changing fitted parameters. Explicitly enable the current diagnostics and
choose their cutoff when using a converted model for fitting. The solver's
`residual_density_threshold` remains separate. Converting a model does not reconstruct its
training objective or automatically migrate an application script.

## 10. Explicit Fourier stability references

`FourierStabilityLoss` obtains the fixed molecular moment exclusively from
`model.ideal`. Do not pass a second `dipole_magnitude` to the loss or to
`polarization_fourier_curvature`. Polarization probes require the model to
own a `FixedDipoleIdeal`, even though they do not evaluate its exact entropy.

The two regularizers retain their original ideal **reference approximations**:

| Variable | `ideal_reference` label | Ideal reference added to excess curvature |
| --- | --- | --- |
| `rho` | `translational_ideal_gas` | Translational ideal gas, without orientational entropy |
| `dipole_density` | `quadratic_alignment` | Weak-alignment fixed-dipole quadratic term |

The label is a read-only property, appears in the module representation, and
is returned in `term.evaluate(...)["ideal_reference"]` and the corresponding
`Loss.evaluate` details. It labels the existing formula; it is not an option
that silently changes the objective.

For species a the quadratic polarization reference is

$$
\beta f_{\mathrm{ref},P}
=\frac{3|\mathbf{P}_a|^2}{2m_a^2\rho_a}.
$$

The displacement is still
$\delta\mathbf{P}_a=\epsilon m_a\rho_a w\mathbf{u}$, with real spatial wave
$w$ and one random unit direction $\mathbf{u}$ per field. The ideal second
difference is therefore $3\Delta V\sum_{g,a}\rho_{g,a}\epsilon^2w_g^2$.
The exact nonzero-alignment orientational curvature is not substituted.
Density probes likewise do not include the density curvature of orientational
entropy. These decoupled tests are not the exact full coupled Hessian at
nonzero polarization and do not certify its positive definiteness.

The following standalone helper constructs both terms. The caller chooses
wavevectors, amplitude and weights; no new fitting defaults are proposed here.
Add the returned terms to the same `Loss` as the field-fitting terms.

```python
from equicdft import FourierStabilityLoss


def build_stability_terms(*, wavevector_indices, relative_amplitude,
                          density_weight, polarization_weight):
    return [
        FourierStabilityLoss(
            name="density_stability", variable="rho",
            wavevector_indices=wavevector_indices,
            relative_amplitude=relative_amplitude, weight=density_weight,
        ),
        FourierStabilityLoss(
            name="polarization_stability", variable="dipole_density",
            wavevector_indices=wavevector_indices,
            relative_amplitude=relative_amplitude, weight=polarization_weight,
        ),
    ]
```

Each weight multiplies the mean squared hinge of normalized curvature below
the existing threshold (zero by default), over valid probes. Polarization
includes its uniform mode by default; density conserves particle numbers.
The existing density masking, amplitudes, random draws, normalization and
relaxed local polarization bound are unchanged. Perturbed model calls use
`compute_ideal=False` so the exact ideal-domain restriction is not reintroduced.

Current checkpoints have no separate Fourier moment buffer. Legacy training
checkpoint resume is not supported. Existing run scripts still need the explicit
API migration; fitted model conversion is independent of training state.
