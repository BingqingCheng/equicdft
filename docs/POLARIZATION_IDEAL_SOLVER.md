# Fixed-dipole ideal functional and coupled canonical solver

Implemented and validated on 12 September 2026 following the user's approval
of the noninteracting benchmark. Numerical implementation is established for
the tests below; use for an interacting polar liquid remains exploratory.

## Scope and physical contract

Each modeled species has an explicitly supplied, positive, fixed molecular
dipole magnitude `m`. The resolved fields are number density `rho` and electric
dipole density `dipole_density`, with shapes `[..., grid, species]` and
`[..., grid, species, 3]`. Angular measure is normalized as `dOmega/(4*pi)`.
The external interaction is exactly `integral(rho*V_ext - P.E_ext)`.
`V_ext` is energy/particle, `E_ext` is energy/dipole, and `beta` is inverse
energy. Thus eV, e Angstrom and V/Angstrom are one consistent choice.

The ideal functional accepts the finite interior `rho > 0`, `|P| < m*rho`.
It rejects vacuum, perfect alignment, and violations of that domain explicitly;
these require boundary-limit treatment beyond the finite derivatives here.
The solver supports one unbatched, fully accessible grid with cubic voxels and
any number of fixed-dipole species. Zero particle numbers, hard exclusions,
grand-canonical solving and variable/induced dipole magnitudes are not supported.
These are explicit physical/interface limits, not guessed fluid parameters.

The optional excess model supplies one `beta_F_exc` through the existing
`GridCACEModel` energy-only API. The model and its input tensors must already
use compatible devices, units, and precision. When the model exposes `k_B`,
the solver checks `beta*k_B*temperature = 1`. The ideal reference's thermal
wavelength is explicitly owned by the solver; at fixed species counts changing
it shifts chemical potentials and a constant energy, but not equilibrium fields.
The solver preserves model training mode and parameter gradients.

This adds no SPC/E adapter or fitted water model. The accepted SPC/E data uses
charge-site coupling `sum_a q_a phi(r_a)`; replacing that by an oxygen-anchored
point-dipole coupling requires its own approximation audit. The scalar
`GridSolver` and `compute_local_mu` guards remain appropriate for that API.

## Ideal entropy and independent derivatives

For `p = |P|/(m*rho)`, `xi = L^-1(p)`, `L(x)=coth(x)-1/x`, and
`Z(x)=sinh(x)/x`, the implemented dimensionless free energy is

```text
beta F_id = DeltaV sum_(g,t) rho [log(rho Lambda^3) - 1 + xi*p - log Z(xi)]
d(beta f_id)/d rho |_P = log(rho Lambda^3) - log Z(xi)
d(beta f_id)/d P   |_rho = (xi/m) P_hat
```

`FixedDipoleIdeal` returns `beta_F_id`, `density_derivative`, and
`polarization_derivative`. Both derivatives have positive signs and are
functional derivatives with voxel volume divided out. These are distinct from
the excess model's negative density derivative `c1`.

Stable Taylor branches cover zero/weak polarization, using squared vector
magnitudes so that the zero-vector Hessian is finite. A rational estimate
initializes twelve differentiable Newton steps for the inverse Langevin
function; the rational approximation itself is not used as the final inverse.
Exponentially scaled expressions avoid overflow for strong fields. Independent
quadrature tests include dimensionless fields through 100; the inverse and
finite-entropy tests reach `p=0.9999`. Double precision is recommended for
strict residuals. GPU and float32 accuracy have not been established here.

## Canonical mirror descent

`PolarizationSolver.evaluate` differentiates the total discrete objective
with respect to independent density and polarization tensors. It subtracts
one unweighted spatial mean per species from the density derivative, giving
the canonical residual `R_rho`. The polarization residual is `R_P`; the
reported dimensionless version is `m*R_P`. Convergence requires the maximum
absolute component across both residual fields to pass `tolerance_residual`.
Particle-number errors and maximum `|P|/(m*rho)` are reported separately.

For an auxiliary alignment vector `a` with `P=m*rho*L(|a|)*a_hat`, each trial is

```text
a_trial = a - step*m*R_P
log w_trial = log rho - step*R_rho + log Z(|a_trial|) - log Z(|a|)
rho_trial = N_t * softmax_grid(log w_trial) / DeltaV
P_trial = m_t * rho_trial * L(|a_trial|) * a_trial_hat
```

This follows from multiplying the molecular distribution
`rho*exp(a.u)/Z(|a|)` by the exponential of minus its functional gradient.
It enforces species counts and realizability at every accepted iterate.
A backtracking line search requires descent of the total energy with an Armijo
factor of `1e-4`. The energy comparison allows 32 machine epsilons scaled by
the current objective magnitude; when the predicted energy change is below
that scale, the physical residual must also decrease. This never changes the
requested convergence tolerance. Nonfinite/inadmissible field trials backtrack
rather than being clipped. The return status is `converged`, `max_iter`, or
`line_search_failed`, with iteration history and the last admissible fields.

The default step of one reaches the exact noninteracting equilibrium in one
update (unless the start is already converged). This is an algebraic property
of the ideal entropy mirror map; there is no noninteracting shortcut in the
solver. The same update handles the excess-model tests, which require multiple
iterations. It is a minimization method, not a proof of global convergence for
an arbitrary learned nonconvex functional.

The initially tested L-BFGS parametrization stalled in several float64 cases at
residuals approximately `1e-7` to `5e-7` because objective changes reached
roundoff. This numerical observation motivated the entropy mirror map;
the original `1e-7` regression threshold was retained. No L-BFGS fallback is
part of the delivered solver.

## API and reproduction

```python
import torch
from equicdft import PolarizationSolver

torch.set_default_dtype(torch.float64)
data = {
    "V_ext": torch.zeros(64, 1),          # energy per particle
    "E_ext": torch.ones(64, 1, 3)*0.2,   # energy per dipole
    "beta": torch.tensor(1.4),
    "grid_spacing": torch.tensor([0.5, 0.5, 0.5]),
}
solver = PolarizationSolver(dipole_magnitude=1.7, thermal_wavelength=1.0)
result = solver.solve(data, particle_numbers=[4.0], tolerance_residual=1e-8)
assert result["converged"]
```

A model-backed solve additionally requires the model's ordinary grid and
temperature metadata. Pass `model=model` to the solver constructor. Explicit
`initial_rho` and `initial_polarization` override data fields; otherwise the
solver uses data fields if present, or uniform density/zero polarization.
An initial density must already integrate to the requested species counts.

Run from the polarization worktree root with
`/Users/tc/miniconda3/envs/xtal/bin/python`:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python examples/polarization_density/noninteracting.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_polarization_*.py'
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests
```

## Validation record

Baseline: branch `feature/polarization-density`, commit
`ea70cf0491b879217a13cfe3d2a1c35aedb00245`, plus the local additions described
here. Existing uncommitted handoff/navigation documentation was preserved.
No commit, merge, push, MD campaign, or model fitting was performed.
CPU environment: Python 3.11.14, PyTorch 2.8.0, NumPy 1.26.4, ASE 3.26.0.

- **477/477 full-suite tests passed**, including the compact LJ forward/reverse
  regression; the baseline was 464 tests.
- **46/46 polarization tests passed**, including 13 new test methods.
- New coverage: independent angular quadrature, finite differences and autodiff
  at two voxel volumes, mixed Hessian reciprocity, zero-polarization Hessian,
  rotations, batching, multiple species, weak/strong fields, unit changes,
  canonical gauge, invalid states, and explicit unconverged statuses.
- A manufactured positive quadratic excess tests actual backtracking and
  multi-iteration convergence; a manufactured equilibrium using the existing
  neural polarization model tests its energy interface. Neither is a fit.
- The existing tensor-to-scalar warning in `test_fourier_amplitudes.py` is
  unchanged. `git diff --check` passed.

The standalone benchmark uses an `8x6x4` grid, spacing `0.8`, `beta=1.4`,
`m=1.7`, `N=60`, float64, and initialization seed 37. It tests zero, scalar,
uniform electric, varying electric, coupled, weak (`1e-5` field scale), and
strong (`20` scale) forcing. All **14/14 case/start combinations** pass a
`1e-8` residual criterion and `1e-7` scaled field-error criterion. The exact
discrete reference uses `rho proportional exp(-beta V)*Z(beta m|E|)` and
the Langevin polarization; a separate angular quadrature evaluates its weak
field branch to avoid subtraction cancellation.

| Maximum over all 14 solves | Observed |
|---|---:|
| Dimensionless physical residual | 1.64e-13 |
| Density error / mean density | 4.08e-14 |
| Polarization error / (m * mean density) | 2.83e-14 |
| Absolute particle-number error | 9.95e-14 |
| Alignment fraction | 0.967310 |

All nontrivial ideal solves took one accepted mirror step; the initial uniform
zero-field state took zero steps. Both initializations agree within the
declared tolerances. The benchmark prints complete per-case metrics as JSON.

Next scientific gate: audit the SPC/E external coupling and molecular field
mapping before defining water equilibrium targets or fitting a functional.

## Subsequent readability cleanup (12 September)

The solver now obtains its initial conjugate alignment directly from the
already evaluated ideal polarization derivative, `a = m*d(beta f_id)/dP`.
This removes a redundant inverse-Langevin evaluation and a duplicate truncated
series. Scalar Langevin evaluation and vector alignment share their small-field
polynomial. The Armijo/roundoff acceptance predicate is a named helper, while
the main loop retains the explicit coupled field update. Input validation also
rejects complex moment/wavelength tensors before any conversion to real dtype.
Public APIs, physical conventions, and numerical thresholds are unchanged.

All 477 tests passed again, including the added complex-input assertions in
the existing validation test. A deterministic before/after comparison covered
154 energy, first/second-derivative and benchmark records, with maximum
difference scaled by `max(1, |reference|)` of `3.64e-16`. Both initializations
of all seven noninteracting cases still pass. The temporary comparison used
seed 912, eight voxels/two species, alignment fractions 0, 1e-5, 0.2 and 0.95,
voxel volumes 0.125 and 8, followed by the retained standalone benchmark.
The permanent scientific regression fixtures remain the tests and example.

Architectural recommendation: preserve the common `evaluate(data)` and
`solve(data, particle_numbers=...)` pattern, with explicit thermodynamic
reference selection. Keep scalar and coupled polarization update strategies
separate for now: the scalar solver additionally implements density caps,
exclusions, grand-canonical solving and Anderson acceleration. A future shared
entry point can delegate to the appropriate strategy once matching capability
and regression contracts are defined. Presence of a dipole observable in data
alone should not silently select a different ideal reference or solver.
