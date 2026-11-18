# Explicit grid-resolution transfer

`GridCACEModel.regrid` returns an independent model for another isotropic
grid. It does not resample a density or external field, solve an equilibrium,
or retrain the model. Supply input fields and their grid metadata at the new
spacing; the usual forward grid-mismatch check remains active.
All three supplied spacings must be exactly equal. Near-isotropic triples
are rejected rather than silently changed or saved with ambiguous geometry.

```python
# Start from the trusted model on its original training/reference grid.
strict = model.regrid(0.125)  # ordinary center-excluded Gaussian weights

# Explicitly opt into the exploratory reference-normalization compensation.
controlled = model.regrid(
    0.125,
    center_normalization="reference",
    convolution_backend="fft",
)
```

Strict center exclusion remains the default. An omitted backend preserves the
existing backend; an explicit `"gather"` or `"fft"` selects that implementation
for both the base descriptor and message convolutions. Regridding preserves
physical local cutoff and Gaussian widths, Cartesian-moment units, density
normalizations, learned-input smoothing lengths, all readout and ideal
parameters, device/dtype, training mode and parameter `requires_grad` flags.
Only local radial exponents change their numerical grid-unit representation.
Long-range reciprocal radial parameters are not rescaled.

## What reference normalization means

For primitive Gaussian channel `n`, let `p0_n(h)` be the central normalized
weight the center-inclusive stencil would have. The compensated primitive
basis is

```text
gamma_n = (1 - p0_n(h)) / (1 - p0_n(h_reference))
basis'[j,n,k] = gamma_n * basis[j,n,k]
                + (1 - gamma_n) * delta[j,center] * delta[k,constant].
```

Each primitive channel has its own factor. The same spatial kernel acts on
each density component independently, so two Gaussian channels with two
species need neither a shared factor nor species mixing. The center term
uses the existing descriptor carrier, including any model-owned density
smoothing or linear component transform. The separate central-density input
and independent pointwise, ideal and long-range terms are unchanged.

The normal representation path constructs the compensated basis before
convolution. Shared message kernels reuse it once; independent Gaussian
message kernels use their own live exponents. No runtime method replacement
or post-hoc rescaling of derivatives is used. Both target and reference
normalizers remain differentiable with respect to the current radial
parameters. The reference geometry is not a frozen `gamma` or `p0` snapshot.
Exact reference geometry takes the original arithmetic path unchanged.
This is exact operator/state identity; nondeterministic GPU accumulation can
still differ at roundoff between repeated evaluations. Bitwise CUDA checks
require deterministic PyTorch operations and the corresponding BLAS setup.

This is an alternative off-grid functional, not a proven continuum limit or
a guarantee of improved accuracy. Its scalar central coefficient can be
negative. Log-space normalizers avoid subtracting `p0` from one for very
narrow Gaussians, but large signed weights can still be ill-conditioned;
unrepresentable factors raise rather than being silently clamped. The
modified scalar functional is differentiated consistently for all outputs.

## Checkpoints and repeated transfers

The first transfer records the source grid as its original reference. Start
from a trusted native checkpoint: an old manually modified model does not
contain enough metadata to reconstruct its training convention automatically.
Unrecorded source geometry or moment rescaling is rejected.

The original spacing, stencil offsets/mask and coordinate convention remain
in the feature configuration. `stencil_radius` and `monomial_scale` record the
current noninteger active cutoff and moment units. They survive `save_model`
and `load_model`; reference tensors are reconstructed as device-aware buffers.
Old checkpoints need no extra state keys and retain the old defaults.

A training checkpoint containing `model_config` and `model_state_dict` is
not a `save_model` archive. Restore its model through the training workflow
before calling `regrid`, or export that unchanged model with `save_model`.
Do not pass the whole optimizer/training checkpoint to `load_model`.

```python
from equicdft import save_model, load_model

save_model(controlled, "controlled.pt")  # choose a new output, not the original
restored = load_model("controlled.pt")
coarse = restored.regrid(0.5, center_normalization="reference")
```

Repeated transfers retain the original reference, not the intermediate grid,
and regenerate the basis rather than compounding previous corrections. Each
call defaults to `"strict"`; pass `"reference"` again when that is intended.
Ordinary floating-point roundoff can accompany chained exponent unit changes.

## Supported scope and rejection

The initial implementation supports scalar top-level Cartesian descriptors
with any number of untransformed, fixed-zero-centered Gaussian primitives,
or the equal-weight `"none"` limit, arbitrary density-component count, and
shared or independent Gaussian messages. Supported readouts are local CACE,
LDA, bulk and long-range, with an optional scalar ideal-gas reference.

Regridding explicitly rejects anisotropic spacings, empty noncentral stencils,
shifted/trainable radial centers, Bessel or transformed radial bases,
polarization, independent CACE branches and other grid-dependent readouts.
These existing model capabilities remain available outside `regrid`; a
Gaussian center-mass rule must not be applied silently to a signed or
conditioned radial basis.

Focused tests live in
[`test_center_normalization.py`](../tests/representation/test_center_normalization.py)
and [`test_regrid.py`](../tests/model/test_regrid.py). The original LJ
production integrity check remains in
[`tests/regression/lj_paper_v1`](../tests/regression/lj_paper_v1/README.md).
