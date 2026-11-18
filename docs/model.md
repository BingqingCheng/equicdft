# The model: ideal reference, outputs and independent branches

`GridCACEModel` owns the complete intrinsic functional. Its readouts compute
only excess contributions; the model adds one ideal reference, differentiates
the sum, and applies its own unit contract and normalization scales. This page
covers the parts of that contract that apply to every model, scalar or
polarized: which ideal reference to choose, how a fitting selection can
restrict which energy centers are evaluated, what a forward returns, and how
to compose several independent CACE pipelines in one functional.

The fixed-dipole entropy, the alignment inputs and the coupled
density/polarization solve are in [`polarization.md`](polarization.md).
Equilibrium solving is in [`solving.md`](solving.md); running a fit is in
[`training.md`](training.md).

Explicit scalar-model resolution transfer and the opt-in reference-center
normalization policy are documented in [`regridding.md`](regridding.md).

## Choosing the ideal reference

Choose `ideal=IdealGas(...)` for a scalar density or
`ideal=FixedDipoleIdeal(dipole_magnitude, ...)` for density with fixed
dipoles. The latter already includes translational entropy: do not add
another ideal gas. Both accept `thermal_wavelength`, a positive scalar or
per-species vector in the chosen length units, defaulting to one. Unlike the
Boltzmann constant, a batch that carries its own `thermal_wavelength`
overrides the stored one, because the thermal wavelength depends on the
temperature and the species mass while the Boltzmann constant fixes the unit
system.

A model built without an ideal reference is excess-only: it evaluates
`beta_F_exc` and its derivatives but cannot be solved.

## Selecting energy centers for fitting

Ordinary `model(data)` evaluates the **full physical intrinsic functional**.
For fitting, use `model(data, use_voxel_mask=True)`. This selects ideal,
LDA and CACE-centered energy contributions using
`model.voxel_weights(data) > 0`: normally `rho > model.rho_min`,
intersected with `valid` and nonexcluded voxels. Explicit
`data["voxel_weights"]` replace the cutoff with their positive support.
Weights have rho's shape, including species. Their magnitudes weight fitting
averages and losses, **not energy**. `valid` and `excluded_mask` are per voxel.

The original rho/P arrays remain unchanged in descriptors, messages, LDA
inputs and all long-range sources. Masked voxels still influence CACE
contributions centered on retained neighbors, with gradients intact. Coulomb
and Gaussian LR always use the full fields. Bulk, pairwise and GGA readouts
are not selected by this ideal/LDA/CACE policy.

Masked entries contribute zero whole ideal energy (translation and orientation)
and zero ideal derivatives. The fixed-dipole domain condition applies to
selected positive-density entries. A finite but unphysical P at an omitted
center is ignored by the ideal, not projected out of other inputs. Global
finite-field, shape/dtype and hard-wall density checks still apply.
An empty energy selection is allowed; chemical-potential averages require
positive weight for every field/species.

The returned `local_energy_mask` identifies a masked fitting forward;
`ideal_derivative_mask` additionally excludes exact vacuum. Total/excess
derivatives at omitted centers can be nonzero through neighboring and LR
contributions. They are derivatives of the selected scalar energy with its
selection held fixed, not post-hoc masked derivatives.

`Trainer` requests `use_voxel_mask=True` for models exposing
`voxel_weights`, for both training and validation. SEM augmentation precedes
selection: the density cutoff, local chemical potentials and their weighted
mean are recomputed from the perturbed fields, with unchanged independent
support/exclusion masks. A `TrainingStream` can explicitly set
`model_kwargs={"use_voxel_mask": False}` for full-functional fitting.
Standalone augmentation callers should not freeze cutoff weights. `valid`
must encode support independently of the clean density cutoff; old caches
with a baked-in cutoff cannot recover upward crossings. Explicit input
weights still override the cutoff.
Merely putting weights in the input does not mask an ordinary model forward.

`GridSolver` uses full physical evaluation, ignoring fitting selections and
retaining strict physical ideal/domain checks (including zero rho/P at hard
walls). It never minimizes an energy with density-threshold switches.

Standalone `ideal.evaluate(data, voxel_volume, mask=mask)` retains its
optional Boolean ideal-only mask. Its default `mask=None` and
`ideal.local(...)` remain full physical ideal evaluation.

For normal evaluation, the model returns:

| Output | Meaning |
| --- | --- |
| `beta_F_id`, `beta_F_exc`, `beta_F` | Ideal, excess and total intrinsic energy; full by default, selected local terms with `use_voxel_mask=True` |
| `ideal_density_derivative` | Ideal derivative with respect to density at fixed polarization |
| `ideal_polarization_derivative` | Ideal derivative with respect to polarization at fixed density |
| `density_derivative` | Total density derivative, `ideal_density_derivative - c1` |
| `density_derivative_excess` | Excess density derivative, `-c1` |
| `polarization_derivative` | Ideal plus excess polarization derivative |
| `polarization_derivative_excess` | Excess polarization derivative |

Selected rows of the second direct correlation are returned when the model
is built with `compute_c2=True` and a `c2_reference` index is passed to the
forward. Total density derivatives require `compute_c1=True`; total polarization
derivatives require `compute_polarization_derivative=True`. The existing `c1`
and `c2` retain their **excess-only** definitions. When `compute_ideal=False`,
the total derivative keys are absent; only excess responses are returned.
Derivatives use continuum normalization, including division by voxel volume.
External fields and chemical potentials are not part of the intrinsic model.

Both ideal references expose `local(rho, dipole_density=None,
thermal_wavelength=None)` for `energy_per_particle` (beta units) and the
`ideal_*_derivative` values. Their shared `evaluate(data, voxel_volume)`
handles vacuum/exclusions and integrates rho times the local per-particle
energy. The dipole-only callable interface and scalar-only `per_particle`
method have been removed. Scalar minimization retains higher-precision
accumulation using this common local interface.

## Excess-only evaluation

Use `model(data, compute_ideal=False, compute_local_mu=False)` when only the
excess functional is needed. Fourier stability probes use this path, so their
intentionally relaxed perturbations do not trigger the exact ideal-domain
constraint.

## Independent additive CACE branches

Use `CACEBranch` when the density-only and joint density/polarization terms
need **independent** A features, B features, message layers and neural readouts:

$$
\beta F = \beta F_{\mathrm{id}}
 + \beta F_{\rho}[\rho]
 + \beta F_{\mathrm{joint}}[\rho,\mathbf P]
 + \beta F_{\mathrm{LDA}}[\rho,\mathbf P]
 + \beta F_{\mathrm{LR}}[\rho,\mathbf P].
$$

Each branch returns only its scalar **excess** energy. The parent sums all
excess readouts before differentiating, and adds the single model-owned ideal
contribution. Density derivatives therefore include both branches; the
density-only branch has exactly zero polarization derivative. The global LDA,
Coulomb and any other LR terms are listed once, outside the branches.

This builder is complete; physical reference values and any global LR
readouts are explicit caller inputs. The example architecture uses p2/v3,
one Gaussian radial channel and no messages by default. `radial_exponent`
is in grid-coordinate units: for physical Gaussian width sigma and cubic
spacing h, it is h²/(2 sigma²). These defaults are architecture choices,
not universal physical constants.

```python
from equicdft import (
    BChiMessage, CACEBranch, CartesianAFeatures, CartesianBFeatures,
    FixedDipoleIdeal, GridCACEModel, LDAReadout, LocalReadout,
)


def build_additive_model(
    *, density_reference, polarization_scale, dipole_magnitude,
    grid_spacing, temperature_reference, boltzmann_constant,
    global_readouts=(), n_types=1, cutoff_grid=3,
    max_power=2, max_product_order=3, radial_exponent=0.125,
    message_count=0, hidden_sizes=(16, 8),
):
    def branch(include_polarization):
        a = CartesianAFeatures(
            n_types=n_types, mean_density=density_reference,
            cutoff_grid=cutoff_grid, max_power=max_power,
            include_polarization=include_polarization, separate_center=True,
            coordinate_scaling="cutoff", convolution_backend="fft",
            radial_basis="gaussian", radial_exponents=(radial_exponent,),
            radial_centers=(0.0,), trainable_radial_exponents=True,
            trainable_radial_centers=False,
        )
        b = CartesianBFeatures(
            max_power=max_power, max_product_order=max_product_order,
            include_polarization=include_polarization, separate_center=True,
        )
        messages = [BChiMessage(
            b.n_features, a.n_radial_channels, a.n_output_channels,
            hidden_sizes=hidden_sizes, include_polarization=include_polarization,
            convolution_backend="fft",
        ) for _ in range(message_count)]
        return CACEBranch(
            a, b, LocalReadout(n_types=n_types, hidden_sizes=hidden_sizes),
            message_layers=messages,
        )

    return GridCACEModel(
        readout=[
            branch(False), branch(True),
            LDAReadout(n_types=n_types, include_polarization=True,
                       hidden_sizes=hidden_sizes, zero_init=True),
            *global_readouts,
        ],
        ideal=FixedDipoleIdeal(dipole_magnitude),
        mean_density=density_reference, polarization_scale=polarization_scale,
        grid_spacing=grid_spacing, mean_temperature=temperature_reference,
        boltzmann_constant=boltzmann_constant,
        compute_c1=True, compute_polarization_derivative=True,
    )
```

Use ordinary global `LongRangeReadout` instances in `global_readouts`; do not
wrap them or the ideal term in a CACE branch. To fit local chemical potentials,
also enable the model's `compute_local_mu` and choose `rho_min` as described
in the fitting section. The branch API changes neither losses nor physical
units. There is no automatic coupling between branch parameters: construct
fresh A/B/message/readout modules for each branch.

All branches currently share one grid cutoff, species count and density
normalization; inconsistent settings raise at model construction. The single
model `polarization_scale` is passed to every polarized branch and LDA. Only
gather-based pipelines need explicit neighbor indices. Local readouts are
bound to their own feature widths at construction, before making an optimizer.
A `LocalReadout(feature_scale=..., feature_offset=...)` subtracts one value
and divides by one positive value per input feature before the network; the fitting guide's
[standardization section](training.md#standardizing-local-readout-inputs)
derives that scale from the training data.

The original `GridCACEModel(a_features=..., b_features=..., readout=[...])`
syntax remains supported, including its state-dictionary paths. Its ordinary
local readouts share one feature vector; it may coexist with independent
`CACEBranch` readouts. `model.has_local_features` and `model.n_local_features`
describe that top-level shared pipeline, not the branch pipelines.
Existing format-5 models need no conversion; older formats still use the
explicit converter. New branch models use the same config/state format and
require a package version that knows `CACEBranch`.

The joint branch also contains pure-density invariants. This decomposition
does **not** enforce `F_joint(rho, 0) = 0` and is not a unique separation of
physical contributions; only the total energy and its derivatives are fitted.

## Quadrupole-trace coupling to the applied potential

A point-dipole model couples its molecules to an applied electrostatic
potential through `-P . E_ext` only. A real rigid molecule carries its
charges at finite displacements `d_a` from the site the density is binned
on, and the orientation-independent part of its energy beyond the dipole
term is

    (Tr M / 6) laplacian(phi_ext),   Tr M = sum_a q_a |d_a|^2,

the trace of the second charge moment (SPC/E about the oxygen: 0.8476 e A^2).
It depends on where the molecule is, not on how it points, so it is a number
potential: `V_ext,total = V_ext + (Tr M / 6) laplacian(phi_ext)`. For the
potentials used in MD training sets (Gaussians of width 1.5-3 A, sinusoids)
this term is 5-20 % of the dipole coupling and dominates the density response
of strongly charge-coupled cases; it is also the Bethe-potential physics of a
water interface.

`GridCACEModel(second_moment=...)` switches it on, with one trace per species
in `e * length^2`. The batch must then carry `laplacian_phi_ext`, the
Laplacian of the applied potential at the voxel centres with shape
`[..., n_grid, 1]` or `[..., n_grid, n_types]` (zero for a uniform field).
`model.external_potential(data)` returns the effective potential; it feeds
the chemical-potential diagnostic `beta_mu_local`, and `GridSolver`
resolves it once on its inputs (`model.resolve_external_potential`, which
keeps the number-only potential under the private `_V_ext_number` entry
that also marks the batch as resolved), so objectives, residuals and the
recorded `V_ext` all include the term. No public batch key switches the
coupling off; a loader may carry the products' `V_ext_number` column. The learned
functional and its second variation are unchanged: the term is linear in the
density. Store the analytic Laplacian in the data products; for band-limited
potentials `equicdft.geometry.spectral.spectral_laplacian` differentiates a
sampled potential exactly, but a Gaussian narrower than the grid spacing is
not band-limited and should be differentiated analytically.

With the coupling on, the bulk-response targets for the learned part are the
responses to the dipole and traceless-quadrupole couplings only: compute them
from the site charge density with the trace part removed,
`rho_q(k) + (k^2 Tr M / 6) n(k)`, rather than from the full charge density.
Without it, use the full site-charge cross spectra; see `docs/polarization.md`.

## Averaged inputs for the learned terms

`GridCACEModel(learned_input_smoothing=sigma)` evaluates every learned local
readout on periodic averages of the fields instead of the voxel fields
themselves:

    beta F_exc[rho, P] = F_learned[G * rho, G * P] + F_fixed[rho, P].

`G` is the nonnegative nearest-neighbour kernel of
`equicdft.geometry.spectral.nonnegative_average`, with second moment
`sigma^2` along every axis; `sigma` is in the length unit of `grid_spacing`.
The learned readouts are `CACEBranch` pipelines, readouts of the shared
features (`LocalReadout`, `GGAReadout`), `LDAReadout`, `PairwiseReadout` and a
`LongRangeReadout` with learned `density_channels` sources; each declares
`reads_smoothed_fields`, and a readout that consumes the local features
always reads them. They see the averaged fields everywhere, including the
density that weights their per-particle energies, so their whole
contribution is a functional of `G * rho` and `G * P`. Averaging only the
descriptors would not do: with the voxel density as the prefactor, a term of
the form `2 a'(G * rho) T(k)` survives in the curvature, with the amplitude
`T` rather than `T^2`.

The ideal reference, the external coupling (including the quadrupole-trace
term), `LongRangeReadout` on the physical fields (Coulomb, Gaussian and
dielectric kernels, fixed references), `BulkReadout`, FMT, MSA and
metal-wall terms keep the voxel fields, and so do the fitting selection
(`voxel_weights`, `use_voxel_mask`) and the chemical-potential diagnostic.
Every derivative (`c1`, the `c2` row, the polarization derivative) is taken
with respect to the voxel fields; the average is its own adjoint, so the
learned part of `c1` is `G * c1_learned` evaluated at the averaged fields.
Solvers, losses, bulk-response targets and feature standardization need no
change, because they all evaluate the model.

At a homogeneous state, the learned second variation is multiplied by
`T(k)^2`, where `T` is the kernel's transfer function. At a nonuniform state,
the learned Hessian is `G^T H_learned[G x] G`, where `x` collects the voxel
density and polarization fields; Fourier modes can mix. Averaging removes
exact null modes and attenuates other learned contributions, but does not
guarantee stability or prevent excessive stiffness of the total functional.
For example, at `sigma = h / sqrt(6)` the corner is removed, while the
learned curvature retains `4/9` at an axis Nyquist mode and `1/9` at a [110]
edge. Full-grid and relevant nonuniform-state stability checks remain needed.

### The kernel

One step of second moment `s^2` replaces every value by `1 - 2 sum_i a_i`
times itself plus `a_i = s^2 / (2 h_i^2)` times each of its two neighbours
along axis `i`; its transfer function is

    t(k) = 1 - 2 s^2 sum_i sin^2(k_i h_i / 2) / h_i^2.

No step is wider than `s_max = h / sqrt(6)` (on a cubic grid; `(2 sum_i
h_i^-2)^(-1/2)` in general), where `t` vanishes at the cube corner and the
centre weight is one half. A `sigma` up to `s_max` is one step; a wider one
is whole steps of `s_max` and a remainder, the second moments adding up to
`sigma^2`. `T`, the product of the step transfers, lies in [0, 1], equals
`exp(-sigma^2 |k|^2 / 2)` to second order in `k`, decreases continuously
with `sigma` at every wavevector and is zero at the cube corner for every
`sigma >= s_max`. The weights reach one voxel per step and are exact zeros
beyond.

The weights are nonnegative by design, because the alignment inputs assume
admissible fields. A nonnegative average keeps a nonnegative density
nonnegative and keeps `|G P| <= G |P| <= m G rho` wherever `|P| <= m rho`
holds voxel by voxel, so an alignment formed from the averaged fields stays
within the range of the voxel alignments it averages. A positive Fourier
multiplier does not guarantee this. The zone-restricted Gaussian
`exp(-sigma^2 |k|^2 / 2)` has negative real-space weights, and the alignment
`u = rho P / (m (rho^2 + floor^2))` amplifies the cancellation: across a
slab of 0.09 and 0.001 A^-3 with alignment 0.8 in opposite directions on a
2 A grid, `sigma = 0.6` A leaves 1.05e-4 A^-3 at the second dilute voxel
with nearly twice its polarization, alignment 14.4. The density floor
prevents a division by zero there, not an alignment above one, and raising
it would change valid low-density inputs as well. The Gaussian residual
smoothing of `TensorLoss(residual_smoothing=...)` is unaffected: a residual
weighting needs only a positive multiplier, an input average needs positive
weights.

Positivity limits the suppression at the zone boundary. Every nonnegative
kernel with second moment `sigma^2` has `T >= 1 - 2 sigma^2 / h^2` at an
axis Nyquist mode, `T >= 1 - 4 sigma^2 / h^2` at a [110] edge mode and
`T >= 1 - 6 sigma^2 / h^2` at the cube corner, and one step attains all
three. `sigma = h / sqrt(6)` (0.82 A on a 2 A grid) is therefore the natural
choice: the learned curvature keeps 0.85 of its value at |k| = 0.5 A^-1
along an axis, 0.58 at 1.0, 0.44 at the axis Nyquist and 0.11 at the [110]
edge, and none at the cube corner. Beyond second order the transfer is
anisotropic: at |k| = 1.0 A^-1 the factor is 0.58 along an axis and 0.49
along [111]. A wider `sigma` suppresses the zone boundary further at the
cost of the low-`k` response; `sigma = h / sqrt(12)` (0.58 A for 2 A) is the
second moment of the voxel itself.

Choose the fixed terms accordingly: the voxel scale must keep positive
curvature without the learned part. `IdealGas` and `FixedDipoleIdeal` have it
in every mode. A reference whose curvature vanishes in some mode, such as one
combined ideal for several site densities (no curvature in the composition
mode), leaves that mode flat at the zone boundary unless a fixed term, for
example an intramolecular reference, supplies it. With site charges the
Coulomb damping is part of the same choice: `exp(-alpha k^2)` with
`alpha = sigma_q^2` is the Coulomb energy of Gaussian-averaged charges, and a
damping much stronger than the charge structure of the reference system
forces the learned terms to restore the missing stiffness at the wavevectors
they can still reach.

The density and every polarization component are averaged together, six
shifted additions per step. The data must carry `grid_size`; rows may follow
any complete order given by `grid_positions`.

With polarization, every Cartesian component of `P` is averaged by the same
kernel, which has the symmetry of the grid, so the cubic equivariance of the
features is unchanged. Alignment inputs are formed from the averaged fields,
`u = (G rho)(G P) / (m ((G rho)^2 + floor^2))`, not by averaging `u`, so at a
homogeneous state the density, cross and polarization blocks of the learned
curvature all scale by `T(k)^2`. The fixed-dipole ideal keeps the Langevin
curvature of the voxel fields in every polarization mode. The slab above is
a regression in `tests/model/test_learned_input_smoothing.py`: the learned readouts see
densities within the input range and alignments of at most 0.8, and the
derivatives at the interface match finite differences of the energy.

The option is part of the configuration and has no state entry; `None`, the
default, evaluates every readout on the voxel fields exactly as before.
Changing it on a fitted model changes the functional the parameters were
fitted for.
