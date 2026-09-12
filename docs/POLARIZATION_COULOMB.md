# Shared charge and dipole Coulomb functional

Implemented September 12, 2026, at the user's request to follow the shared
charge/dipole evaluator in LES. General capability; its use for coarse-grained
SPC/E water remains exploratory. No new water fit is included in this change.

## One kernel, generalized source

`ReciprocalFeatures.forward` now accepts optional `dipole_density` and
`charges`. `LongRangeReadout(include_polarization=True)` connects these to the
existing model context. There is no separate dipolar kernel, solver, or LES
runtime dependency. The inspected reference is LES commit
`fadfb0ca3b89ed371e5de6b383f4425defabd4a8`, `src/les/module/ewald.py`,
`compute_potential_triclinic`; untracked alternative LES files were not used
or changed.

Let a label physical components, rho_a be number/volume, P_a be electric
dipole/volume, and q_a be the molecular/ionic net charge in compatible units.
The continuum-normalized FFT and component source are

```
rho_hat_a = DeltaV * FFT(rho_a - mean(rho_a))
P_hat_a   = DeltaV * FFT(P_a - mean(P_a))
S_a(k)    = q_a*rho_hat_a(k) - i*k_D.P_hat_a(k)
S(k)      = sum_a S_a(k)
F_LR      = C/(2V) * sum_(k != 0) [4*pi*exp(-alpha*k^2)/k^2] * |S(k)|^2
```

The existing unique-component-pair contraction, including off-diagonal
multiplicity two, evaluates the final square. In polarization mode charges
are inside the source: the readout applies a single shared amplitude to all
pair features, not another q_a*q_b factor. Thus a neutral component with q=0
can still polarize, and charge/dipole cross terms are included automatically.
P is already an electric dipole density: do not multiply by q or a molecular
moment again. Shapes remain rho `[..., G, n_types]`, P
`[..., G, n_types, 3]`, including arbitrary leading batch dimensions.

LES uses exp(+ikr), hence its +i*k.mu. Our FFT uses exp(-ikr), hence -i*k.P.
LES's Gaussian sigma maps to alpha=sigma^2/2 in this kernel. Its half-space
mode factor is equivalent to our full-grid sum with 1/(2V). The independent
test uses explicit positive-phase sums, rather than the FFT implementation.
Finite grid bandwidth differs from LES's particle-mode cutoff.

### Zero and Nyquist modes

The shared k=0 omission is unchanged: no macroscopic surface penalty is added
to the 3D-periodic conducting-boundary functional. Uniform polarization is
not penalized by this LR branch. Local and transverse correlations still
belong to the residual functional.

For odd grid axes k_D=k. For an even axis, its Nyquist component in k_D is
zero (the real trigonometric spectral-derivative convention). Other components
on that plane remain active. This preserves Hermitian symmetry, real bound
charge, and signed-permutation lattice covariance. The radial kernel still
uses the original k, including Nyquist: charge-only behavior is unchanged.
A polarization varying only along its own Nyquist direction has zero
resolved divergence. This is a stated numerical resolution limitation,
not physical absence of molecular dipole interactions. Resolve shorter
wavelengths with a finer grid rather than interpreting that mode physically.

### Units, self energy, and residual

For rho in Angstrom^-3, q in elementary-charge units and P in e/Angstrom^2,
C=14.39964547842567 eV Angstrom/e^2. In beta-free-energy mode set the fixed
amplitude to beta*C at the specified temperature. In physical mode set it to
C/(k_B*T_ref); GridCACEModel multiplies by T_ref/T before differentiating.
The tests compare these conventions at two temperatures. There is no implicit
epsilon_r, beta, dipole-magnitude factor, or descriptor normalization in the
source. Fixed-amplitude readouts require no learned mean-density scale;
state-dependent fitted amplitudes retain the established state features.

This is a Gaussian-smoothed continuum Coulomb energy, not a complete bare
particle Ewald sum. LES's particle self-energy subtraction must not be
reinterpreted as a subtraction of each voxel's squared mean dipole. No such
subtraction is added. Define the learned excess residual by

```
F = F_id[rho,P] + F_LR[rho,P] + F_residual[rho,P].
```

Both translational and orientational ideal terms remain unchanged. The
residual includes the chosen short-range/contact convention and molecular
correlations. Include the LR derivative in the total training prediction
(or subtract it from the residual target, not both). Do not attach LR to an
old total-excess fit and claim a no-double-counting decomposition.

The water dataset's oxygen-anchored point-dipole P does not encode the exact
charge-site SPC/E charge density at molecular wavelengths. This extension
does not resolve that approximation or establish improved water predictions.

## Usage

The caller must select and record the LR smoothing scale separately from
the local descriptor cutoff. No water smoothing scale or new fit is selected
by this implementation. A fixed-temperature neutral polar component is built
as follows, where alpha_lr and beta are explicit caller inputs:

```python
features = ReciprocalFeatures(
    radial_exponents=(alpha_lr,), kernel="coulomb", n_types=1,
)
lr = LongRangeReadout(
    n_kernels=1, n_types=1, charges=(0.0,),
    coulomb_amplitude=beta * 14.39964547842567,
    features=features, include_polarization=True,
)
# Add lr to the existing readout list before differentiating the total energy.
# Supply data["dipole_density"] and data["grid_size"].
```

For mixed components supply their net charges and per-component P; use zero P
for nonpolar components. Omit include_polarization to retain the old API,
state-dictionary layout, and charge-only coefficient convention. Older
whole-module checkpoints lacking the new attribute default to charge-only.
Reciprocal features support orthogonal anisotropic spacings; GridCACEModel's
existing polarized-model cubic-voxel restriction remains unchanged.

The FFT adds three component transforms when polarization is enabled. Energy
and all responses share autograd, including charge/P mixed Hessians and
response-loss differentiation of an optional learned amplitude. No changes
to the coupled solver or local neural architecture were required.

## Validation

`tests/test_polarization_coulomb.py` contains 14 focused tests: old checkpoint
and charge-only limits, an independent LES-style all-mode sum and derivatives,
analytic longitudinal/transverse modes and restoring sign, mixed-term sign
and exact source cancellation, k=0 and Nyquist, batch/multiple/neutral species,
both-field finite differences at two voxel sizes and energy modes, mixed
Hessian reciprocity, physical temperature conversion, first/second derivative
checks at P=0, LDA additivity, lattice covariance/translation/extensivity,
learned-amplitude response gradients, constructed coupled equilibrium from
uniform and perturbed starts, and invalid inputs.

From the shared worktree, the validation command is:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src MPLCONFIGDIR=/private/tmp/water-polarization-mpl-cache-v2 /Users/tc/miniconda3/envs/xtal/bin/python -c 'import torch, unittest, sys; torch.set_num_threads(1); result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.discover("tests")); sys.exit(not result.wasSuccessful())'
```

The full suite includes the compact LJ production regression. Exact run
outcomes and source identities are retained in the application workspace's
`water-polarization-grid2A-pilot-v1/COULOMB_IMPLEMENTATION.md`.
