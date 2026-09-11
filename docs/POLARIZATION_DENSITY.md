# Particle and polarization densities in neural cDFT

Theory and implementation · 11 September 2026

This note describes the exploratory implementation in `equicdft`, branch
`feature/polarization-density`, commit `c7d7f03`. It learns one scalar excess
free energy from a number-density field and an electric dipole-density field.
Its derivatives follow from that same energy. The descriptor and derivative
machinery is implemented and tested; a complete polar-fluid equilibrium model
is not yet implemented.

## 1. What are the two density fields?

For species $t$, define the ensemble-averaged microscopic fields

$$
\rho_t(\mathbf r)=\left\langle\sum_{a\in t}
\delta(\mathbf r-\mathbf r_a)\right\rangle,
\qquad
\mathbf P_t(\mathbf r)=\left\langle\sum_{a\in t}
\mathbf m_a\,\delta(\mathbf r-\mathbf r_a)\right\rangle.
$$

Here $\mathbf r_a$ is a chosen molecular reference position and $\mathbf m_a$
is its electric dipole moment. Number density has units of inverse volume;
polarization density has units of dipole moment per volume. On a voxel, the
corresponding estimates are the mean particle count and mean vector sum of
dipoles, each divided by the voxel volume. The reference positions, dipole
definition, units and averaging convention must be specified by the dataset.

In particular, $\mathbf P$ is **not** a unit orientation or the mean dipole per
particle. Where $\rho_t>0$, the latter is $\mathbf P_t/\rho_t$. Zero mean
polarization does not imply an empty voxel: oppositely oriented molecules can
cancel. The three components of $\mathbf P$ are not three scalar species.

For a fixed molecular dipole magnitude $m_t$, physically realizable fields obey

$$
|\mathbf P_t(\mathbf r)|\leq m_t\rho_t(\mathbf r).
$$

This bound is system-specific and is not imposed by the current code, because
no molecular moment magnitude is supplied. These two fields also do not
resolve every possible molecular ordering: higher orientational moments may
be needed for, for example, nematic order at zero polarization. Molecular DFT
provides a precedent for using number and polarization fields, but the
multipolar polarization used for distributed molecular charges is not
automatically identical to the point-dipole field defined here.
[Jeanmairet et al. (2016)](https://arxiv.org/abs/1601.06535).

## 2. The functional and its conjugate fields

The intrinsic functional is written

$$
F[\rho,\mathbf P,T]
=F_{\mathrm{id}}[\rho,\mathbf P,T]
+F_{\mathrm{exc},\theta}[\rho,\mathbf P,T].
$$

The learned object is $F_{\mathrm{exc},\theta}$, not an independently fitted
chemical potential or electric field. For a point-dipole model, choose the
external coupling convention

$$
\Omega=F_{\mathrm{id}}+F_{\mathrm{exc},\theta}
+\sum_t\int d\mathbf r\,
\left[(V_t^{\mathrm{ext}}-\mu_t)\rho_t
-\mathbf E_t^{\mathrm{ext}}\cdot\mathbf P_t\right].
$$

$V_t^{\mathrm{ext}}$ is energy per particle; $\mathbf E_t^{\mathrm{ext}}$ is
the field conjugate to the chosen dipole, in consistent energy/dipole units.
The minus sign is part of this stated convention. General orientational
external potentials need not reduce to this linear dipole coupling.

At fixed temperature, with $\beta=(k_BT)^{-1}$, define

$$
c_t^{(1)}(\mathbf r)
=-\left.\frac{\delta\beta F_{\mathrm{exc},\theta}}
{\delta\rho_t(\mathbf r)}\right|_{\mathbf P},
\qquad
\mathbf g_t^P(\mathbf r)
=\left.\frac{\delta\beta F_{\mathrm{exc},\theta}}
{\delta\mathbf P_t(\mathbf r)}\right|_\rho.
$$

The polarization response deliberately has a **positive** sign. It is the
excess contribution to the restoring field, not the complete external field.
For an interior equilibrium state, independent variation of the two fields gives

$$
\begin{aligned}
\beta\mu_t^{\mathrm{loc}}
&=\left.\frac{\delta\beta F_{\mathrm{id}}}{\delta\rho_t}\right|_{\mathbf P}
-c_t^{(1)}+\beta V_t^{\mathrm{ext}}
=\beta\mu_t,\\
\mathbf R_t^P
&=\left.\frac{\delta\beta F_{\mathrm{id}}}{\delta\mathbf P_t}\right|_\rho
+\mathbf g_t^P-\beta\mathbf E_t^{\mathrm{ext}}
=\mathbf 0.
\end{aligned}
$$

The first condition is equal local chemical potential: no particle transfer
can reduce the constrained free energy. The second is local orientational
balance: no polarization variation can reduce it. In NVT, $\mu_t$ is the
Lagrange multiplier enforcing particle number. Polarization need not be
spatially constant or conserved. On active realizability bounds, stationarity
must instead be interpreted with the appropriate constraint conditions.

### Why the scalar ideal-gas formula is insufficient

Adding particles **at fixed polarization** changes the polarization per
particle and therefore the orientational entropy. Thus
$\delta\beta F_{\mathrm{id}}/\delta\rho$ is generally not just
$\ln(\rho\Lambda^3)$.

As an illustrative reference, consider freely rotating fixed-magnitude point
dipoles, and minimize their local ideal orientational entropy at prescribed
$\rho$ and $\mathbf P$. For one species, let
$p=|\mathbf P|/(m\rho)<1$, let $\xi$ solve
$p=\coth\xi-1/\xi$, and set $Z(\xi)=\sinh\xi/\xi$. Using normalized
angular measure, the entropy-minimizing orientation distribution is
proportional to $\exp(\xi\widehat{\mathbf P}\cdot\mathbf u)$, giving

$$
\beta f_{\mathrm{id}}
=\rho\left[\ln(\rho\Lambda^3)-1
+\xi p-\ln Z(\xi)\right].
$$

Its two derivatives are

$$
\left.\frac{\partial\beta f_{\mathrm{id}}}{\partial\rho}\right|_{\mathbf P}
=\ln(\rho\Lambda^3)-\ln Z(\xi),
\qquad
\left.\frac{\partial\beta f_{\mathrm{id}}}{\partial\mathbf P}\right|_\rho
=\frac{\xi}{m}\widehat{\mathbf P}.
$$

Both orientational corrections have regular zero-polarization limits. This is
a possible **future reference**, not a term present in the implementation and
not an assertion that fixed-magnitude point dipoles suffice for every molecular
fluid. The ideal/excess split must be defined consistently with the selected
resolved fields.

## 3. Symmetry: rotate positions and dipoles together

For an orthogonal transformation $R$ and a translation $\mathbf a$,

$$
\rho'_t(R\mathbf r+\mathbf a)=\rho_t(\mathbf r),
\qquad
\mathbf P'_t(R\mathbf r+\mathbf a)=R\mathbf P_t(\mathbf r).
$$

An isotropic, parity-even intrinsic free energy must be unchanged under this
joint transformation. This includes reflections: an electric dipole is a
**polar vector**, not a magnetic axial vector. A scalar readout must therefore
see invariant combinations of geometry and polarization, rather than raw
$P_x,P_y,P_z$ values or three independently processed scalar channels.

The tensor contractions below are orthogonally invariant as algebraic
operations. Sampling is a separate issue. For a cubic lattice the 48 signed
axis permutations preserve the lattice exactly: $3!$ axis permutations times
$2^3$ independent reflections. Arbitrary rotations require resampling.
For a rectangular periodic box, an axis exchange also exchanges its dimensions;
only the box-preserving subgroup acts within that fixed box.

Spatial inversion is also different from reversing every dipole **without
moving the density field**. The optional assumption

$$
F[\rho,-\mathbf P]=F[\rho,\mathbf P]
$$

is controlled by `dipole_reversal_symmetry`. It is off by default. For example,
a density–polarization coupling $\mathbf v\cdot\mathbf p$ is invariant under
joint spatial inversion but odd under polarization-only reversal. Such
couplings should not be discarded without a symmetry argument for the chosen
system. Nonvanishing density–polarization coupling is discussed explicitly in
[Jeanmairet et al. (2016)](https://arxiv.org/html/1601.06535v1).

## 4. Cartesian moments of the two fields

Let $g$ label a central voxel and $\mathbf q$ an integer offset in the
inclusive stencil $|\mathbf q|^2\leq q_{\mathrm{cut}}^2$. Each voxel is cubic,
$\Delta V=(\Delta L)^3$; the total box need not be a cube. The current radial
weights are shared by the scalar and vector fields:

$$
w_n(\mathbf q)=
\frac{\exp[-\alpha_n|\mathbf q|^2]}
{\sum_{\mathbf q'}\exp[-\alpha_n|\mathbf q'|^2]},
\qquad \alpha_n=\exp(\eta_n)>0
\quad\text{when trainable}.
$$

The exponents are in inverse squared **grid units**. Fixed zero exponents are
also permitted by the underlying Gaussian implementation, giving uniform
weights. Scalar normalization $\rho_*>0$ and dipole-density normalization
$P_*>0$ are fixed inputs. A single $P_*$ scales every Cartesian component;
normalization by the usually vanishing mean vector would be inappropriate.

The moments are

$$
\begin{aligned}
M^{\rho}_{nt,\ell}(g)
&=\sum_{\mathbf q}w_n(\mathbf q)
\frac{\rho_t(g+\mathbf q)}{\rho_*}\,
\mathbf q^{\otimes\ell},\\
M^{P}_{nt,\ell}(g)
&=\sum_{\mathbf q}w_n(\mathbf q)
\mathbf q^{\otimes\ell}\otimes
\frac{\mathbf P_t(g+\mathbf q)}{P_*}.
\end{aligned}
$$

The density moment has rank $\ell$; the dipole moment has rank $\ell+1$.
Only the spatial indices are symmetrized. For instance,

$$
D_{ij}=\sum_{\mathbf q}w_n(\mathbf q)q_iP_{t,j}(g+\mathbf q)/P_*,
\qquad D_{xy}\ne D_{yx}\ \text{in general}.
$$

This distinction is the essential extension beyond scalar-density moments.
The spatial factor $q_xP_y$ cannot be merged with $q_yP_x$. Generalized ACE
also treats vector-valued degrees of freedom alongside spatial coordinates;
the present implementation uses a small explicit Cartesian contraction list,
not that work's full expansion machinery.
[Drautz (2020)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.102.024104).

For the implemented spatial powers $\ell=0,1,2$, the code uses:

| Symbol | Field moment | Tensor rank |
|---|---|---:|
| $s$ | $M^\rho_0$ | 0 |
| $\mathbf v$ | $M^\rho_1$ | 1 |
| $Q$ | $M^\rho_2$ | 2 |
| $\mathbf p$ | $M^P_0$ | 1 |
| $D$ | $M^P_1$ | 2 |
| $H$ | $M^P_2$, with $H_{ijk}=H_{jik}$ | 3 |

The two inequivalent traces $u_k=H_{iik}$ and $w_i=H_{ijj}$ are retained.
Radial and species indices form one channel index, ordered radial first and
species second. Unsmeared center $\rho_t/\rho_*$ and $\mathbf P_t/P_*$ are
appended to the $s$ and $\mathbf p$ channel lists. The Gaussian stencil itself
still includes the center; these extra channels give direct access to its
values without smoothing.

## 5. From tensor moments to a scalar local free energy

Contracting every Cartesian index produces scalars. Representative features,
with independent channel labels $a,b,c$, are

$$
\begin{aligned}
&\operatorname{tr}D_a,\quad
\mathbf p_a\cdot\mathbf p_b,\quad
\mathbf v_a\cdot\mathbf p_b,\\
&D_a:D_b,\quad\operatorname{tr}(D_aD_b),\quad H_a:H_b,\\
&\mathbf p_a^TQ_b\mathbf p_c,\quad
\mathbf v_a^TD_b\mathbf p_c,\quad
Q_{a,ij}H_{b,ijk}p_{c,k}.
\end{aligned}
$$

Repeated Cartesian indices are summed; the channel labels are not summed away
but enumerate distinct features. For a vector pair, for example,
$(R\mathbf p_a)\cdot(R\mathbf p_b)
=\mathbf p_a^TR^TR\mathbf p_b=\mathbf p_a\cdot\mathbf p_b$.
The same cancellation of paired rotation matrices proves invariance of the
higher contractions. Center polarization enters cross contractions with the
neighborhood, retaining relative alignment rather than only its magnitude.

The current list contains up to three moment factors. `max_product_order`
counts these factors **before** the MLP; it does not bound the polynomial
degree of the final nonlinear functional. This is a finite, deliberately
non-minimal descriptor set, not a complete invariant basis. It differs from
the scalar branch's cubic-group symmetrization: here explicit tensor
contractions enforce the orthogonal tensor symmetry directly.

With invariant vector $B_g^P$ and normalized temperature $\tau=T/T_*$, the
default reduced-energy readout supplies

$$
\beta F_{\mathrm{exc},P}
=\Delta V\sum_{g,t}\rho_{gt}\,
a_{\theta,t}(B_g^P,\tau).
$$

The density prefactor makes the local contribution zero in an empty voxel,
provided its descriptor/readout is finite. It does not remove that voxel from
neighbor environments. The normalized local averages contain **no** voxel
volume; $\Delta V$ enters once through the energy quadrature.

Other readouts may be added before differentiation, for example a scalar CACE
or LDA contribution. The polarization readout itself includes density-only
features, so this is an additive model decomposition, not a unique physical
separation: it need not vanish when $\mathbf P=0$. A complete long-range
dipolar electrostatic functional is not supplied by the local descriptors.

## 6. Autodiff and a future equilibrium learning objective

On the discrete grid the implemented outputs are

$$
c^{(1)}_{gt}=-\frac{1}{\Delta V}
\left.\frac{\partial\beta F_{\mathrm{exc}}}{\partial\rho_{gt}}\right|_P,
\qquad
g^P_{gtj}=\frac{1}{\Delta V}
\left.\frac{\partial\beta F_{\mathrm{exc}}}{\partial P_{gtj}}\right|_\rho.
$$

Both inputs remain live tensors, so overlapping environments and all readouts
contribute to these derivatives. The model makes one autodiff call for the
requested fields and retains the graph during training. In
`free_energy_mode="beta"`, readout energies sum directly to $\beta F_{\rm exc}$.
In `free_energy_mode="physical"`, the sum represents $F_{\rm exc}/(k_BT_*)$;
the model multiplies it by $T_*/T$ before calculating these same responses.

Mixed derivatives obey the reciprocity relation

$$
\frac{\partial c^{(1)}_{gt}}{\partial P_{hsj}}
=-\frac{\partial g^P_{hsj}}{\partial\rho_{gt}}
$$

for equal voxel volumes and a twice-differentiable energy. Existing `c2`
returns one selected density–density derivative row **at fixed polarization**.
It is not the full coupled response matrix. Coupled stability concerns the
second variation of the **total** intrinsic functional in both fields,
including mixed density–polarization variations; checking density perturbations
alone does not establish it.

Once an ideal orientational reference and external-field convention are fixed,
a natural extension of the existing equilibrium loss would be

$$
\begin{aligned}
\overline{\widetilde\mu}_t
&=\frac{\sum_g w_{gt}\widetilde\mu_{gt}}{\sum_g w_{gt}},
\qquad \widetilde\mu_{gt}=\beta\mu^{\mathrm{loc}}_{gt},\\
\mathcal L_\mu&=\frac{\sum_{g,t}w_{gt}
(\widetilde\mu_{gt}-\widetilde\mu_t^{\mathrm{target}})^2}
{\sum_{g,t}w_{gt}},\\
\mathcal L_P&=\frac{\sum_{g,t}w_{gt}\|m_t\mathbf R^P_{gt}\|^2}
{\sum_{g,t}w_{gt}},\qquad
\mathcal L=\lambda_\mu\mathcal L_\mu+\lambda_P\mathcal L_P.
\end{aligned}
$$

Here $w_{gt}$ is a sampling/accessibility weight and
$\widetilde\mu_t^{\mathrm{target}}$ is known $\beta\mu_t$ or the inferred
$\overline{\widetilde\mu}_t$. The molecular dipole magnitude $m_t$ makes the
polarization residual dimensionless; a general molecular model can instead
use an explicitly chosen dipole-unit scale. Each contributing species must
have nonzero total weight.

This loss is a theoretical proposal, **not an implemented polar-fluid training
protocol**. The code deliberately does not apply the scalar-fluid
$\ln(\rho\Lambda^3)$ target formula to polarized data.

## 7. Implementation and use

The calculation follows a short sequence:

1. [`data.py`](../src/equicdft/data.py) reads and orders both fields. Canonical
   shapes are `rho[..., n_grid, n_types]` and
   `dipole_density[..., n_grid, n_types, 3]`. EXTXYZ uses type-major flattened
   vector columns: `type0_x,type0_y,type0_z,type1_x,...`. Source names are
   configurable with `data_key`.
2. [`polarization_features.py`](../src/equicdft/polarization_features.py) gathers
   both fields, evaluates one shared Gaussian/Cartesian basis, forms the
   moments, and applies `_CONTRACTIONS`. `feature_names` records the output
   order. Positive trainable radial exponents remain inside the graph.
3. [`readout.py`](../src/equicdft/readout.py) implements `PolarizationReadout`:
   invariants plus normalized temperature go into an MLP; the result is
   integrated with number density and voxel volume.
4. [`model.py`](../src/equicdft/model.py) sums all readout energies and obtains
   the optional `c1` and `polarization_derivative` outputs through autodiff.
   Their shapes match `rho` and `dipole_density`, respectively. One scalar
   energy is returned per complete field, not per independent environment.

For example:

```python
features = PolarizationFeatures(
    mean_density=0.7,          # fixed number-density scale
    dipole_density_scale=0.1,  # fixed dipole moment / volume scale
    cutoff_grid=3,
    max_power=2,
    max_product_order=3,
    radial_exponents=(0.125,),
    trainable_radial_exponents=True,
    n_types=1,
    dipole_reversal_symmetry=False,
)

model = GridCACEModel(
    a_features=None,
    b_features=None,
    readout=[PolarizationReadout(features)],
    grid_spacing=0.5,
    compute_c1=True,
    compute_polarization_derivative=True,
)

outputs = model(data)  # data includes rho, dipole_density, temperature and geometry
```

The literal temperature key is `temperature`, as in `GridData`. Numbers in this
snippet are illustrative scales, not recommended polar-fluid parameters.
The [complete runnable example](../examples/polarization_density/example.py)
constructs the grid, supplies synthetic fields and checks gradient propagation.

Coarsening averages vector components, preserving
$\Delta V\sum_g\mathbf P_g$. Excluded voxels must contain zero polarization;
invalid values are rejected before coarsening so cancellation cannot conceal
them. Models combining scalar and polarization local representations require
the same stencil cutoff. Polarization models require equal spacing along all
three axes, but permit rectangular boxes.

## 8. What is established, and what remains open?

At commit `c7d7f03`, 464 CPU tests pass, including 33 polarization tests and
the retained scalar LJ forward/reverse regression. The tests cover lattice
symmetry, orthogonal tensor contractions, direct neighbor sums, batching,
species channels, derivative signs and voxel factors, mixed derivatives,
trainable radials, coarsening, exclusions and saved-model round trips. This
establishes the tested numerical properties, not accuracy for a polar liquid.

The orientational ideal functional, realizability enforcement, external
electric-field data/units, coupled equilibrium loss, coupled solver and vector
long-range physics remain to be specified or implemented. Consequently,
`compute_local_mu` and the existing `GridSolver` reject polarization models;
the model can currently be evaluated directly for excess energies and
derivatives. No polar-fluid model has been fitted in this branch.

The central design principle is simple: **keep scalar and vector fields
distinct while constructing moments, make the local energy scalar by tensor
contraction, and derive every response from that one energy.**
