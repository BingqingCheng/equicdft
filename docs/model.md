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
