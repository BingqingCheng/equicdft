# Deprecations

Every deprecated path warns with `equicdft._deprecation.EquicdftDeprecationWarning`
(a `DeprecationWarning` subclass, silent by default outside tests; make it an
error with `warnings.simplefilter("error", EquicdftDeprecationWarning)`).
Deprecated in 0.1.0, removed in 0.2.0 unless noted.

## Module paths (0.1.0)

| old path | new home |
|---|---|
| `equicdft.stencil` | `equicdft.geometry.stencil` |
| `equicdft.features` | `equicdft.representation.cartesian` |
| `equicdft.symmetrize` | `equicdft.representation.invariants` |
| `equicdft.interaction` | `equicdft.representation.message` |
| `equicdft.energy` | `equicdft.functional.base` |
| `equicdft.thermodynamics` | `equicdft.functional.thermodynamics` |
| `equicdft.readout` | `equicdft.functional.readout` |
| `equicdft.semilocal` | `equicdft.functional.semilocal` |
| `equicdft.pairwise` | `equicdft.functional.pairwise` |
| `equicdft.reciprocal` | `equicdft.functional.reciprocal` |
| `equicdft.metalwall` | `equicdft.functional.metalwall` |
| `equicdft.polarization_ideal` | `equicdft.functional.polarization_ideal` |
| `equicdft.stability` | `equicdft.response.stability` |
| `equicdft.loss` | `equicdft.training.loss` (and `equicdft.response.loss` for `FourierResponseLoss`) |
| `equicdft.metrics` | `equicdft.training.metrics` (and `equicdft.response.metrics` for `FourierResponseMetrics`) |
| `equicdft.trainer` | `equicdft.training.trainer` |
| `equicdft.loader` | `equicdft.data.loaders` |
| `scripts/mean_density.py` | `python -m equicdft.data.mean_density` / `equicdft-mean-density` |

`equicdft.data`, `equicdft.response` and `equicdft.solver` became packages
that re-export their former contents; those paths are not deprecated.
Private modules (`equicdft._grid`, `equicdft._fourier`, ...) moved without
shims. Whole-object pickles that recorded any old path keep loading through
`equicdft.legacy`.
