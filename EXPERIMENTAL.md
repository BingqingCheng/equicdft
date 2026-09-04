# Experimental capabilities

The top-level [`README.md`](README.md) intentionally matches the current
`main` branch and documents the compact, established workflow. This file
collects opt-in model and training capabilities developed for ongoing
experiments. They are tested software interfaces, but their scientific value
must be established for each physical system and dataset.

The examples below describe general EquiCDFT behavior. They do not prescribe
an electrolyte model, charge convention, density basis, cutoff, or response
target.

## Compatibility principles

- Omitting every option described here retains the existing model paths.
- Existing models without message passing, density transformation, radial
  transforms, or Fourier losses load without an external migration helper.
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
keys. A reconstructed model must select it explicitly; whole-object saves
retain it, while legacy whole objects use the class default `"gather"`.
`"conv3d"` scatters the spherical stencil into its enclosing dense Cartesian
kernel and applies grouped cross-correlation. `"fft"` scatters onto the
periodic grid and applies the same cross-correlation in reciprocal space;
duplicate or periodically aliased offsets are summed. Both avoid the explicit
neighbor tensor. These backends require a complete regular periodic grid.
The FFT path is intended for real float32 or float64 fields; mixed-precision
FFT support is device- and grid-size-dependent and is not assumed here.

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
`charges` retains freely learned pair coefficients. Charge-factorized mode
currently requires one reciprocal kernel.

For `kernel="coulomb"`, the implemented kernel is
`4 pi exp(-alpha k^2)/k^2` at nonzero modes. It is a Gaussian-damped Coulomb
kernel, so choosing `alpha` also defines an LR/SR partition and is not merely
a numerical setting.

## Shared Fourier-response evaluator

`FourierResponse` evaluates projected curvature of the total intrinsic
dimensionless free energy `beta(F_id+F_exc)` using symmetric finite
differences around a supplied density field. Integer reciprocal-grid modes and
component-space directions are supplied at evaluation time.

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
    modes=integer_modes,
    directions=component_directions,
)
```

The result has shape `[field, mode, phase, direction]`. This evaluator is the
shared numerical core used by both stability regularization and supervised
response fitting.

For multicomponent stability, the same evaluator can reconstruct the complete
physical-component curvature matrix:

```python
curvature_matrix, active_components = response.matrix(
    model,
    batch,
    modes=integer_modes,
)
```

The matrix has shape `[field, mode, phase, type, type]`. It is normalized in
the ideal-gas metric, so the ideal contribution is the identity. For a
homogeneous mixture it is the dimensionless inverse OZ matrix
`I-sqrt(R)c(k)sqrt(R)`. Around an inhomogeneous field it is a projected
component Hessian; it is not the complete position-dependent OZ operator.

## Fourier-stability loss

`FourierStabilityLoss` applies a squared hinge when projected curvature falls
below `minimum_curvature`:

```python
stability = FourierStabilityLoss(
    random_modes_per_field=1,
    wavevector_range=(k_min, k_max),
    relative_amplitude=0.01,
    minimum_curvature=0.0,
    mixture_mode="charge",
    charges=(1.0, -1.0),
    weight=1.0,
    training_only=True,
)
```

The complete multicomponent stability test is selected explicitly with:

```python
matrix_stability = FourierStabilityLoss(
    random_modes_per_field=1,
    wavevector_range=(k_min, k_max),
    relative_amplitude=0.05,
    minimum_curvature=0.0,
    mixture_mode="full_matrix",
    perturbations_per_forward=4,
    weight=1.0,
    training_only=True,
)
```

Explicit `modes=((nx,ny,nz),...)` and random sampling are mutually exclusive.
Random candidates lie in the physical isotropic Nyquist sphere. The optional
inclusive `wavevector_range=(k_min,k_max)` is expressed in reciprocal units
implied by `grid_spacing`: if positions are measured in `sigma`, its units are
`1/sigma`. It is not measured in grid-index units.

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
the recommended general mixture stability check. Existing modes remain useful
as cheaper targeted regularizers.

The matrix is a finite-difference estimate with truncation error of order
`relative_amplitude**2`. Taking an extremely small amplitude can instead
amplify energy roundoff, especially in float32, and the off-diagonal
polarization adds another subtraction. Check amplitude convergence for the
actual model and grid; a smaller amplitude is not automatically more accurate.
For inhomogeneous fields this check covers the physical-component subspace of
each selected cosine or sine perturbation, but not cosine-sine or inter-mode
Hessian blocks.

For `n` components the exact matrix reconstruction uses `n(n+1)/2` component
directions. Cosine and sine phases and symmetric positive/negative differences
therefore require `2n(n+1)` perturbed fields per selected wavevector. Use
`perturbations_per_forward` to limit peak memory. The loss gives every active
eigenvalue equal weight, matching the existing equal weighting of valid scalar
directions.

Stability loss prevents selected negative curvatures; it does not determine
the correct positive response and is not direct structure-factor training.

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
    modes_key="fourier_modes",
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
    modes=modes_by_item,           # [item, mode, 3]
    curvature=target_by_item,      # [item, mode, direction]
    scale=scale_by_item,           # optional, same target shape
    weight=weight_by_item,         # optional, same target shape
    indices=frozen_train_indices,
)
```

Every response item must be uniform, periodic, positive, and unmasked. Mode
triplets are integer reciprocal-grid indices inside the isotropic Nyquist
sphere. Valid sine and cosine estimates are averaged before applying the loss.

`FourierResponseMetrics` reuses the tensors already computed by the loss and
reports curvature RMSE, scaled curvature RMSE, nonpositive counts, and
diagonal `S=1/K` diagnostics. The latter are not a substitute for inverting a
coupled response matrix.

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
    model_kwargs={"compute_c1": False},
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
result = solver.solve(
    field,
    method="euler",
    anderson=True,
    anderson_history=5,
    anderson_regularization=1.0e-8,
    anderson_damping=1.0,
)
```

Each proposal is accepted only through the solver's existing safeguards;
rejected proposals fall back to the baseline update and reset history when
required. The result records attempts, accepted/rejected proposals, and
resets. Anderson acceleration changes the numerical path, not the free-energy
functional or convergence criterion.

## Implementation map

- `_grid.py`: regular-grid validation, neighborhood gathers, and the optional
  periodic stencil convolution
- `_radial.py`: radial formulas, validation, discrete conditioning, and
  pre-invariant transforms
- `features.py`: Cartesian feature ownership, density transforms, and radial
  dispatch
- `interaction.py`: invariant-to-equivariant message layers
- `_fourier.py`, `response.py`: shared mode validation, perturbations, and
  projected curvature
- `stability.py`: curvature-positivity regularization
- `loss.py`, `metrics.py`, `data.py`: supervised response loss, diagnostics,
  and homogeneous response items
- `trainer.py`: explicit multi-stream optimization and epoch-zero validation
- `reciprocal.py`, `readout.py`: reciprocal kernels and coefficient controls

Before promoting any experimental option into a default workflow, run focused
tests, the complete test suite, and the compact LJ forward/inverse regression;
then evaluate system-appropriate held-out observables without using them for
model selection.
