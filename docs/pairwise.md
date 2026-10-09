# Finite-range pairwise density functionals

`PairwiseReadout` is a general, optional excess-energy term. It does not
replace the ideal reference or prescribe a Coulomb decomposition. Its
scientific accuracy and stability remain exploratory until benchmarked for
the chosen system.

## Existing quadratic form (unchanged default)

```python
pair = PairwiseReadout(cutoff_grid=12, n_types=n_types)
```

The distance MLP takes `(|q| / cutoff_grid, T / mean_temperature)` and
returns one kernel for each unique symmetric physical-component pair. With
voxel volume `dv`, the energy is

```text
E_pair = dv^2 / 2 sum_(g,q!=0,ij) rho_i(g) K_ij(q,T) rho_j(g+q).
```

The distance is in grid cells, not physical coordinates. Only
`0 < |q| < cutoff_grid` contributes. Every kernel is multiplied by the
quintic envelope `s(x)=1-10x^3+15x^4-6x^5` for `0 <= x < 1`, zero outside;
its value and first two derivatives vanish at the cutoff. The physical range
is `cutoff_grid * grid_spacing` for an isotropic grid. The cutoff is independent
of CACE and must not exceed half the smallest periodic grid dimension.

The homogeneous Fourier mode is retained. The self displacement `q=0` is
removed for every component pair, including unlike components. Kernels are
applied by periodic FFT convolution, with the MLP evaluated only once per
distinct radial shell. The output follows the containing model's beta or
physical free-energy convention, without another temperature conversion.

## Optional nonlinear endpoint-density dependence

```python
pair = PairwiseReadout(
    cutoff_grid=12,
    n_types=n_types,
    hidden_sizes=(32, 16),           # distance/temperature network
    density_channels=4,               # illustrative local feature count M
    density_hidden_sizes=(16, 16),    # endpoint-density network
    zero_init=True,
)
model = GridCACEModel(
    readout=[pair],
    mean_density=density_scale,       # fixed normalization from training
    grid_spacing=grid_spacing,
)
```

`density_channels=M` enables the new form; `None` keeps the existing form.
The count and hidden widths above are example choices, not physical defaults.
No assumption about species number, charge, diameter or composition is made.

At each voxel, the SiLU density MLP reads the complete local species vector
`rho(g)/density_scale` and outputs `A_ia(g)` with shape `[n_types, M]`.
The model supplies one fixed positive `mean_density` normalization. The
readout does **not** compute a spatial average, accessible-volume average,
reference composition, or pair-average density. It constructs

```text
f_a(g) = sum_i rho_i(g) A_ia(rho(g)/density_scale).
E_pair = dv^2 / 2 sum_(g,q!=0,ab) f_a(g) W_ab(q,T) f_b(g+q).
```

The distance MLP now outputs `M*(M+1)/2` symmetric feature-pair kernels
`W_ab`, with the same distance/temperature inputs and cutoff envelope. The
local feature functions are shared between endpoints. The equivalent
physical-component kernel is

```text
K_ij(q;u,v,T) = sum_ab A_ia(u/density_scale) W_ab(q,T) A_jb(v/density_scale).
K_ij(q;u,v,T) = K_ji(q;v,u,T).
```

Thus both endpoint densities affect a pair. This is a finite separable
representation, not an arbitrary neural network evaluated for every voxel
pair. `M` is a density-feature count, not a radial-basis count. Signed
features and kernels are allowed; symmetry does not impose positive
curvature or guarantee solver stability. No special RPM charge-inversion
symmetry is imposed by this general interface.

Every transformed field vanishes when all species densities at that voxel
vanish, even when the density MLP has biases. There is no division by local
density, logarithm, density clipping, or field-dependent normalization.

### Temperature in the density features

```python
pair = PairwiseReadout(
    cutoff_grid=12,
    n_types=n_types,
    density_channels=4,
    density_temperature=True,         # density MLP also reads T / mean_T
)
```

By default `A` depends on density alone and all temperature dependence sits
in `W_ab(q,T)`, so density and temperature enter the endpoint response
separably. With `density_temperature=True` the density MLP reads
`[rho(g)/density_scale, T/mean_temperature]` and `A_ia` depends on both, so
`K_ij(q;u,v,T)` couples them in the endpoint response, as screened or
packing-dependent effective interactions do. The vacuum property is
unchanged, since every field still carries a factor of physical `rho`. The
option requires `density_channels`, and is written to the configuration only
when enabled, so existing configurations and model files are unaffected.

### Differentiation and efficiency

The scalar energy is differentiated as a whole. Density derivatives through
`A` are retained for chemical potentials, `c2`, and curvature-based training;
there is no detach or frozen-density kernel. In this mode `c2` is generally
**not** minus the endpoint kernel alone. Removing self-pair energies also
does **not** force the on-site Hessian to vanish: a nonlinear endpoint can
have local second derivatives while interacting with other voxels.

Only a pointwise MLP and the existing FFT path are needed. Storage does not
contain a `[n_grid, n_neighbors, ...]` pair-density tensor. Convolution
sources number `M`, and symmetric kernel channels number `M*(M+1)/2`;
increasing `M` therefore increases the cost. This extension permits
density-dependent nonlocal curvature, but does not by itself ensure MSA
agreement or accurate strongly inhomogeneous predictions.

### Compatibility and initialization

Omitting `density_channels` preserves the old MLP dimensions, parameter
names, RNG initialization, energy path and configuration fields. Old
configuration/state models load strictly without migration. Enabled models
record `density_channels` and `density_hidden_sizes`, and save both MLPs in
the normal model/checkpoint state.

`zero_init=True` zeroes the last distance-network layer, not the density MLP.
The initial pair energy and all density derivatives are zero, while the
distance output can receive gradients on the first update. Density-network
gradients become active once the distance kernel leaves zero. Enabling this
architecture changes parameter shapes, so a quadratic state does not load
into it directly; the warm start below converts one.

### Warm start from a trained quadratic readout

```python
model = load_model("quadratic.pt")        # must carry mean_density
k = 0                                     # index of the PairwiseReadout
model.readout[k] = PairwiseReadout.from_quadratic(
    model.readout[k],
    density_channels=4,
    density_hidden_sizes=(16, 16),
    density_temperature=False,
)
```

For `density_channels >= n_types` the density-aware readout contains the
quadratic one. `PairwiseReadout.from_quadratic` builds a readout with the
source's cutoff, species count, distance widths and `zero_init` flag, in
its dtype and on its device, and `warm_start_from(quadratic)` fills it: the
distance network is copied, physical pair `(i, j)` becomes feature pair
`(i, j)`, every kernel involving an extra feature is zero, and the last
density-network layer is set to `A_ia = delta_ia` for the first `n_types`
features at every density and temperature. Energies, chemical potentials
and `c2` of the new readout equal the quadratic ones to numerical roundoff,
so fine-tuning starts from the trained model instead of
from zero, and the density network receives gradient on the first update.

Extra features (`a >= n_types`) keep their random initialization. They
contribute nothing while their kernels are zero, but give those kernels a
nonzero gradient on the first update; with both factors zero neither would
ever move. The hidden layers of the density network receive gradient from
the second update on. `warm_start_from` requires a quadratic source with
matching `cutoff_grid`, `n_types` and `hidden_sizes`, and
`density_channels >= n_types` on the target.

Replacing `model.readout[k]` keeps every other parameter; the model
configuration follows the new readout and `save_model` stores it as usual.
A quadratic model built without `mean_density` is rebuilt from its
configuration with `mean_density` set before the readout is replaced.

## Verification

`tests/functional/test_pairwise_density.py` checks the FFT energy, density
gradient and Hessian-vector product against an explicit physical-species
endpoint sum, with and without temperature-coupled features; endpoint
exchange and cubic-grid symmetries; general species and feature counts;
batched fields and temperatures; vacuum derivatives; backpropagation from
a curvature loss; analytically density-dependent `c2`; zero initialization;
the warm start (energies, `c1` and `c2` against the quadratic readout for
`M = n_types` and `M > n_types`, gradient flow into every parameter,
in-model replacement); and strict configuration/model-file round trips.
Existing quadratic pairwise tests and the compact LJ production regression
remain compatibility gates.
