# One joint-invariant polarization message

User-approved September12,2026. Optional `PolarizationReadout(..., message=...)`
reuses `BChiMessage` and `CartesianBFeatures`; default omission preserves old
state-dict names, energy and derivatives. No change to ideal thermodynamics.

The complete joint invariant vector B0 is packed as [G,1,Q,1]. One scalar
gate h(B0)-h(0) is aggregated with a radial Cartesian stencil, giving A1;
CartesianBFeatures contracts A1 into B1. The final energy MLP receives
[B0,B1,T/T_ref]. The message is not added to raw rho/P, and carries no
explicit vector channel. Dependence on P enters through B0 and remains in
the full differentiation graph. No extra volume or physical normalization
is applied to the dimensionless message. The model still sums scalar energies
before taking both functional derivatives.

The current optional interface supports one latent scalar/radial channel
and one layer, with gather execution. It accepts arbitrary physical species
counts in the underlying joint descriptor. The message must consume all
joint invariants; mismatched dimensions/backend are rejected. Independent
Gaussian or Bessel bases follow BChiMessage's existing interface; sharing
the initial basis requires one initial radial channel.

```python
message = BChiMessage(features.n_features, 1, 1, hidden_sizes=(16,8),
                      radial_exponents=(.125,), trainable_radial_exponents=True)
readout = PolarizationReadout(features, hidden_sizes=(16,8), message=message)
```

The water trial uses max_power1 and product_order2: B1 has three entries,
the scalar moment, its square, and the average of the three squared vector
moment components. Twelve B0 entries plus three B1 entries plus temperature
give16 readout inputs. The gate has353 parameters plus one radial exponent;
the enlarged readout adds48 parameters. With the existing579-parameter LDA
model this totals981. The message gate's final additive bias cancels in
h(B0)-h(0), following the existing BChi definition.

Two nested cutoff3/6Å stencils give descriptor reach12Å; this is not the
maximum reach of a derivative of the integrated energy. Fixed-lattice
symmetry is the48 signed axis permutations. Explicit vector transport or
arbitrary continuous-grid rotational covariance is not claimed.

Validation:495 shared tests pass, including6 new tests of all48 energy and
response symmetries, finite differences in both free-energy modes, mixed
Hessian reciprocity, two-stencil P dependence, multi-species batches,
empty-voxel backward gradients, strict checkpoint roundtrip, zero-message
recovery and rejected configurations. Both compact LJ regressions pass.
17 water application tests pass, including981 parameters,189 masked voxels,
MP gradients on the sparse case, and reproduction of120 retained per-case
channel scores across the two cutoff2 models and no-MP LDA model.

Commands use the established float64 CPU xtal Python environment:
`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m unittest discover -s tests`.
Changes are uncommitted. The application retains a base-commit plus complete
runtime patch identity for the requested fit; no code copy or new repository.
