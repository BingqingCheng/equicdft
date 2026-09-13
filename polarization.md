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
