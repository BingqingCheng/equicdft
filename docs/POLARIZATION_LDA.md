# Optional squared-polarization LDA

Implemented September 12, 2026, on `feature/polarization-density`, starting
from `6d5c6f9a3502a36d8d5347ec8124b1ed8ea0f5c2`. This change is uncommitted.
Scope: a reusable local excess-energy readout, not a new fitted water model.

## API and definition

The existing density-only API remains unchanged. Opt in with a fixed positive
dipole-density scale (dipole moment per volume, not dipole per molecule):

```python
from equicdft import GridCACEModel, LDAReadout

lda = LDAReadout(
    mean_density=rho_reference,
    dipole_density_scale=P_reference,
    n_types=1,
    hidden_sizes=(16, 8),
    zero_init=True,
)
model = GridCACEModel(
    a_features=None, b_features=None, readout=[lda],
    compute_c1=True, compute_polarization_derivative=True,
)
```

The per-voxel input contains, in order, all normalized species densities,
all squared normalized species dipole magnitudes, and normalized temperature:

```
(rho_1/rho_ref, ..., rho_M/rho_ref,
 |P_1|^2/P_ref^2, ..., |P_M|^2/P_ref^2, T/T_ref)
```

The MLP returns one scalar per-particle local contribution. Its energy is
`DeltaV * sum_g rho_total,g * a_LDA,g`, using the same beta/physical energy
conventions as all other readouts. Both responses come from differentiating
the sum of all enabled excess energies. There is no division by local rho,
norm square root, clipping, density floor, fitted gauge or physical constant
inside this branch. The P dependence is smooth at P=0; its local P gradient
there is zero, while its local Hessian can be nonzero and finite.

This is **excess only**. `FixedDipoleIdeal` supplies translational and
orientational ideal entropy separately in `PolarizationSolver` or in the
application's Euler targets. A zero-initialized LDA contributes zero energy
and derivatives, not another copy of the ideal entropy.

For water the chosen scales are rho_ref=rho_bar and P_ref=m*rho_bar.
These are application choices, not library constants. General mixtures are
supported without assuming two species. This magnitude-only LDA does not
resolve angles between species' local vectors: it is invariant under their
independent rotations/reversals. Cross-directional physics needs additional
invariants or another readout. Adding an LDA to a neighborhood readout with
local descriptors is an overlapping, non-unique excess decomposition.

## Geometry and compatibility

The new LDA requires dipole density but **no neighbor index or cutoff**.
`GridCACEModel` now distinguishes these two capabilities. Standalone polarized
LDA has `cutoff_grid=0` and `requires_local_density_index=False`; adding a
`PolarizationReadout` uses that readout's cutoff in either list order.
Conflicting neighborhood cutoffs are still rejected. Existing cubic-voxel,
scalar-solver and scalar-chemical-potential guards for polar models remain.

Omitting `dipole_density_scale` preserves density-only parameter/state-dict
layout, input width, and behavior. The optional scale is a registered buffer
only when enabled. The dipole requirement property falls back to false for
older serialized density-only readouts without this buffer. Earlier
polarization readouts retain their neighbor requirement through the metadata
fallback. No new public class or import is needed.

## Changes and validation

- `src/equicdft/semilocal.py`: optional LDA invariant and validation.
- `src/equicdft/model.py`: separate dipole requirement from stencil requirement.
- `src/equicdft/readout.py`: explicit neighborhood metadata on PolarizationReadout.
- `tests/test_polarization_lda.py`: 12 focused tests.

On CPU, Python 3.11.14, PyTorch 2.8.0, ASE 3.26.0:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/tc/miniconda3/envs/xtal/bin/python -m unittest discover -s tests -p test_polarization_lda.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/tc/miniconda3/envs/xtal/bin/python -m unittest discover -s tests
```

**12/12 focused tests and 489/489 full-suite tests pass**, including both
compact LJ production regressions at unchanged thresholds. Focused coverage:
three-species input ordering and energy; batch independence; strict locality;
analytic zero-P Hessian and empty voxels; density/P finite differences in both
energy modes at two voxel volumes; mixed-Hessian reciprocity; rotations,
reflections and reversal; derivative-loss parameter gradients; additive
readouts in either order; serialization/density-only layout; invalid inputs;
zero-LDA equality to ideal density and polarization equilibrium. Existing
Fourier test code emits its pre-existing tensor-to-scalar warning.

Water application integration additionally passes 15 tests, including unchanged
80 retained per-case Euler scores and backward-only validation on observed
water fields. No new water fit, checkpoint, MD or HPC run was generated.
No commit, push or merge was performed. Scientific accuracy of a fitted
LDA-augmented model remains untested.
