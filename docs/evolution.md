# Density evolution

## Conservative floor-aware semi-implicit steps

`FluxLimitedSemiImplicitIntegrator` combines `SemiImplicitIntegrator` with
the donor-budget mechanism of `FluxLimitedEulerIntegrator`. Supply an explicit
`density_floor` in input-density units; excluded zeros and frozen densities
are unchanged. Initialization must separately meet that floor conservatively.
The unconstrained IMEX increment defines corrected face fluxes; common-species
face factors limit outgoing transfers before applying their divergence.
There is no post-step clipping or particle redistribution across blocked faces.
Inactive limiting recovers the original semi-implicit result to solve accuracy.
Active limiting need not preserve implicit stability or energy descent, so use
adaptive energy/defect checks and temporal refinement. The defect uses the
floor-constrained explicit endpoint rate, not the implicitly smoothed rate.
For excluded domains supply a separate homogeneous-liquid reference curvature.
This method has no explicit-Euler timestep cap, but not every larger step is
accurate or acceptable. It supports general species matrices and is deterministic.

`equicdft.evolution` advances conserved species densities using an equilibrium
functional and a separate mobility. Deterministic and stochastic
dynamics run at a fixed uniform temperature on periodic regular grids.

```python
from equicdft.evolution import (
    ConstantMobility, Dynamics, EulerIntegrator, SemiImplicitIntegrator,
)

mobility = ConstantMobility(0.01)
integrator = SemiImplicitIntegrator(model, mobility, dt=0.001)
# EulerIntegrator(model, mobility, dt=0.001) has the same interface.
dynamics = Dynamics(integrator)
trajectory = dynamics.run(data, n_steps=1000, save_every=10)
dynamics.save(trajectory, "trajectory.pt")
```

The numerical values above use the length, energy and time units you choose.
The density model must provide the full intrinsic `density_derivative`,
including the ideal contribution. Each run evolves one field with `rho` shaped
`[n_grid, n_species]`. Integrators temporarily evaluate model and mobility
modules in evaluation mode, then restore their original mode flags.
The evolving fields are species number densities; polarization evolution is
not included.

### Strong density depletion

For deterministic evolution with a local density-dependent mobility,
`EulerIntegrator(..., face_average="harmonic-density")` evaluates the mobility
at the componentwise harmonic mean of the two densities bordering each face.
The default `"arithmetic"` still averages the two voxel mobility matrices;
constant mobilities are identical under either choice. The new option does
not floor, freeze or remove low-density voxels. It is a spatial quadrature
choice, with second-order consistency for smooth positive densities; sharply
varying profiles still need spatial-convergence checks.

Both choices use one shared flux per face, conserving every species. A
symmetric positive-semidefinite face matrix gives non-positive instantaneous
free-energy change at fixed external potential. Time integration must still
enforce positivity and energy acceptance. `AdaptiveDynamics` retains its
relative-rate, error and work-corrected-energy checks without modification.
Direct Euler updates avoid subtracting Fourier reconstruction roundoff from
densities far below machine precision relative to the bulk. The homogeneous
Fourier stabilizer is not positivity preserving in such inhomogeneous fields.

`SemiImplicitIntegrator` also accepts the face option, but this does not remove
that limitation. The option is not available through
`StochasticEulerIntegrator`: its noise and thermal drift are derived for
arithmetic faces and cannot be reused unchanged. Trajectory metadata records
`face_average`. No existing default is changed.

For extreme depletion or composition imbalance, use
`FluxLimitedEulerIntegrator(..., face_average="harmonic-density")` with an
explicitly chosen `StepControl(density_absolute_tolerance=...)`. The latter
changes the error test to `abs(error) <= error_tolerance*rho + absolute_tolerance`.
Its default is zero, and it never changes the physical density. A positive
absolute tolerance should be refined together with the relative tolerance.

The flux-limited integrator budgets each voxel's outgoing transfers, retaining
at least `1-outflow_fraction` of its initial density in that step (default
outflow_fraction0.9). The donor budget also retains the smallest normal
floating-point value (`torch.finfo(dtype).tiny`) to prevent repeated depletion
from underflowing to zero. This does not raise densities: an input below that
machine-scale value cannot donate until replenished. It scales each entire
species flux vector by a common
nonnegative face factor. Thus each species remains conservative and the
face's instantaneous dissipation sign is preserved even with cross coupling.
Finite-step energy is still checked by the driver. The limiter is inactive
for sufficiently small steps at fixed densities above the machine reserve; it neither clips
densities nor redistributes particles globally. `advance` requires the
unmodified rate tensor returned by `rhs` for that same state, including the
external field. Stored face fluxes belong to each rate, so adaptive retries
can safely reuse start-state rates after midpoint evaluations.

## Mobility

### Optional deterministic density floor

`FluxLimitedEulerIntegrator(..., density_floor=...)` uses a nonnegative donor
reserve in the input density units. The default zero retains the machine-scale
reserve and the original adaptive controller. A positive floor is an explicit
numerical regularization: each donor can export only a fraction of its excess
above that floor. It never clips the result or creates particles. Inputs below
the floor are not automatically raised; applications must specify a separate
conservative initialization if they require all accessible densities above it.
Excluded zeros and frozen values remain untouched by integration.

With a positive floor, adaptive change and defect tests use the integrator's
actual constrained finite-step rate. Full-versus-two-half-step error and
work-corrected energy checks remain active. `dt_min` remains a failure threshold,
not permission to force a step that fails these checks. Test sensitivity to
both the floor and temporal tolerances before adopting this regularization.
It is not available as a stochastic-noise prescription.

### Mobility convention

The supplied mobility is the physical coefficient in

$$
\partial_t\rho_a=-\nabla\cdot\mathbf j_a,\qquad
\mathbf j_a=-\sum_b M_{ab}(\boldsymbol\rho,T)\nabla\mu_b.
$$

Here $a,b$ label species, $\mu_b$ is chemical potential in energy units,
and $M_{ab}$ couples the flux of species $a$ to the gradient for species $b$.
Mobility has units of number divided by energy, length and time. Its value
therefore fixes the time scale; the fitted equilibrium functional alone does
not determine it.

`ConstantMobility(value)` accepts a scalar, a species vector, or a species
matrix. A scalar gives the same diagonal mobility for all species; a vector
gives a separate diagonal value for each species; a matrix also permits
cross-species coupling. The matrix must be symmetric positive semidefinite to
roundoff. A matrix estimated from simulation data satisfies that only up to
statistical noise; `ConstantMobility.from_estimate(matrix, tolerance=1e-6)`
symmetrizes it and projects it onto positive semidefinite matrices, and raises
if the required change exceeds `tolerance` times the largest entry. The
tolerance must be a finite nonnegative scalar.

A zero-frequency mobility can be calibrated from equilibrium MD through the
integrated density correlation. With $S(k)$ the static structure factor and
$\tau_c(k)=\int_0^\infty C(k,t)/C(k,0)\,dt$ the signed integrated correlation
time of the density mode,

$$
M_0(k)=\frac{\rho_0 S(k)}{k_BT\,k^2\tau_c(k)},
\qquad
\mathbf M_0(k)=\frac{V}{k_BT\,k^2}\,
\mathbf C(k,0)\Big[\int_0^\infty\mathbf C(k,t)\,dt\Big]^{-1}\mathbf C(k,0)
$$

for one and for several species. This is the effective zero-frequency
mobility of the density mode, not a tagged-particle diffusion coefficient;
using its small-$k$ value as a local constant is the modelling step discussed
below.

`DensityMobility(diffusivity)` supplies the diagonal Einstein law
`M_a = beta * diffusivity_a * rho_a`. Supply a nonnegative diffusivity scalar
or one value per species, in length squared per time. It uses `data['beta']`
for inverse thermal energy, so no separate Boltzmann constant is needed.
It works with both deterministic and stochastic integrators. Diffusivity is a
user-supplied dynamical parameter, not automatically measured self-diffusion.

`LocalMobility(function)` calls `function(rho, temperature)` to evaluate a
mobility from the current species-density vector at each voxel and the fixed
temperature. The function may return diagonal values with the same shape as
`rho`, full matrices with shape `rho.shape + (n_species,)`, or a shared
scalar, species vector, or matrix. For example, returning `prefactor * rho`
gives independent mobilities proportional to each species density. Local-shaped
outputs take priority when a shape also matches a shared matrix. The function
does not receive `beta`; an Einstein mobility $\rho D/(k_BT)$ in eV/K units
is `lambda rho, T: rho * D / (k_B * T)` with the data's Boltzmann constant,
or `rho * D * beta` with `beta` captured from the data. A custom
mobility module implements `forward(data)` and returns
a tensor with shape `[..., n_grid, n_species, n_species]`, on the same device
and with the same dtype as `data["rho"]`. Its output is checked for symmetry
and positive semidefiniteness at every step unless the module sets the class
attribute `validated = True`, as the two built-in classes do.

A local, instantaneous mobility is a working dynamical approximation. A
mobility extracted from MD at finite wavevector or with memory requires a
separate justification before being used as a local function.

## One timestep

Both integrators expose `step(data)`, which returns the new density tensor,
`rhs(data)`, which evaluates its instantaneous time derivative, and
`advance(data, rate, dt=None)`, which applies a rate from `rhs` over a step
without evaluating the model again. `integrator.dt` may be reassigned
between steps; `Dynamics.run` takes a fixed step and rejects a change made
during a run, while `AdaptiveDynamics` below varies the step itself.
The driver may split a nominal fixed step at a declared field breakpoint.
The equilibrium model returns a dimensionless chemical-potential derivative,
so the driving field is

$$
u_a=\beta\mu_a
=\texttt{density\_derivative}_a+\beta V_{\mathrm{ext},a},\qquad
\mathbf j_a=-k_BT\sum_b M_{ab}\nabla u_b.
$$

The model already applies voxel-volume normalization to its derivative; do
not divide it by the voxel volume again. `V_ext` uses the model data's energy
units, and `beta` supplies $1/(k_BT)$.

## Freezing voxel densities

Use a Boolean `mask` with one entry per voxel, in the same row order as `rho`.
`True` freezes **all species** in that voxel at their supplied values:

```python
mask = data["grid_center"][:, 0] < 2.0
integrator = EulerIntegrator(model, mobility, dt=0.01, mask=mask)
trajectory = Dynamics(integrator).run(data, n_steps=100)
```

The same `mask=` argument works with `SemiImplicitIntegrator`,
`StochasticEulerIntegrator`, `Dynamics`, and `AdaptiveDynamics`. Alternatively,
put it in `data["mask"]`. A driver's explicit mask overrides the data mask;
the data mask overrides an integrator's default. No caller-owned data is changed.
`GridData.from_dict` and `GridData.from_xyz` also read an optional `mask` field
(Boolean or binary 0/1); EXTXYZ property names can be mapped with
`data_key={"mask": "frozen"}`. Coarsening rejects blocks mixing frozen and free
voxels rather than silently changing the boundary.

Frozen voxels are no-flux boundaries: deterministic and random face fluxes
are blocked on both sides, preserving particle numbers in the remaining
connected regions. Their densities still enter the free-energy functional
and can affect neighboring chemical potentials. They are **not reservoirs**.
They need not be zero: unlike `excluded_mask`, they keep their original
positive densities (a frozen voxel without a finite positive density is
rejected when the mask is first seen; an empty voxel belongs in
`excluded_mask`). All-frozen masks are allowed. Trajectory metadata stores
the resolved mask, which must be reused when restarting. The homogeneous
reference of the semi-implicit stabilizer and of `stability_limit` is the
mean over free voxels, so a frozen slab at a different density does not
shift it; `min_density` still scans every non-excluded voxel, frozen ones
included.

Mask validation, topology and flux blocking are shared by the integrators.
Masked semi-implicit stepping solves the restricted IMEX system
$(I+\Delta tB_{\rm masked})\,\delta\boldsymbol\rho=\Delta t\,\mathcal R$ with
restarted GMRES, right-preconditioned by the periodic propagator
$(I+\Delta tB)^{-1}$ restricted to free rows and projected to zero species
sums within each connected free region. This keeps region particle numbers
at roundoff and frozen values bitwise unchanged, rather than resetting cells
after an unconstrained FFT update. The relative residual tolerance is
$\max(10^{-12},32\epsilon_{\rm dtype})$, where $\epsilon_{\rm dtype}$ is machine
precision. Each normal iteration uses four FFTs plus the face stencils;
the iteration count depends on the mask, curvature and timestep. The
preconditioner helps stiff slab cases but does not guarantee convergence
for arbitrary masks. A solve that still fails raises
`ConvergenceError` (a `ValueError`); `AdaptiveDynamics` treats it as a
rejection and shrinks the step. Unmasked/empty-mask runs retain the original
FFT fast path. Optional stochastic density-floor repair touches only free
voxels and preserves the species numbers of each connected free region,
whether excluded or frozen voxels separate the regions.

## Time-dependent external fields

Pass `field=` to either trajectory driver. The field returns the **complete**
species-resolved `V_ext`, replacing rather than adding to `data['V_ext']`.
Omit it to keep the existing static-field behavior.

For a conducting electrode, ramp its field with the liquid potential:

```python
field = RampedField.from_electric_field(
    V_wall, data["grid_center"], charges, duration=20.0,
    final_field=[0, 0, E_z],
)
trajectory = AdaptiveDynamics(integrator, control, field=field).run(data, t_end=100.)
```

Here `E_z` is the final field in model energy per charge per length. The model
must already contain `MetalWall`. Its geometry and charge constraints stay
fixed; electrode charges equilibrate at each density/field evaluation. This
assumes instantaneous electronic response, not finite-rate electrode dynamics.
The applied electrode field is stored in each state as `metal_external_field`;
it overrides the wall's constructor field without changing its buffers or
geometry cache. Custom fields can return a dictionary with `V_ext` and
`metal_external_field` instead of a potential tensor.
They must return the same keys throughout a trajectory. The helper constructs
`V_ext,i = V_wall,i - q_i E dot r` and the matching electrode field; it accepts
an optional `initial_field`, and `metal=False` selects a liquid-only ramp.
It does not clip the potential. For manually supplied potentials, keeping the
liquid/electrode signs, units and time dependence consistent is the caller's job.
The model and data must use the same fixed temperature and beta convention.

A uniform field represented by a periodic liquid potential has a branch cut.
Place that cut across excluded/frozen no-flux faces, not an active transport
connection. MetalWall separately keeps the electrode on its compact periodic
branch. A periodic field without such blocked transport is a different driven
problem, not the scalar-potential setup implemented here.

Both drivers include liquid work and electrode work
`-beta * p_metal dot dE` in `beta_work`, on the same periodic coordinate branch
as the charge solve. Both drivers reuse detached endpoint dipoles from
existing model evaluations: `-beta * (p_start + p_end) dot dE / 2`.
The independent full-step work estimate still uses an electrode evaluation,
because that density has no model output. Fixed-step evolution completes each
smooth step's work at the next model evaluation, just like its free energy;
the final state and failed-run records are settled before returning.
Field jumps and uncached custom models retain the midpoint calculation. For linear induced charges
the midpoint equals the endpoint trapezoid
and gives the exact energy change for a field jump at fixed density. Rejected
steps do not retain trial fields or accumulate work. Trajectories include the
actual electrode field for restart and diagnostics.
These identities assume fixed geometry, charge constraints and temperature.

For semi-implicit evolution with zero-density `excluded_mask` voxels, supply
`reference_curvature` from the homogeneous **liquid** model before adding the
metal. The excluded/frozen no-flux solve uses the shared restricted operator;
the nonlinear density rate still includes the full electrode response. Do not
infer a homogeneous Fourier stabilizer from an inhomogeneous electrode Hessian.

```python
from equicdft.evolution import RampedField

field = RampedField(V_initial, V_final, duration=10.0)
trajectory = Dynamics(integrator, field=field).run(
    data, n_steps=1000, save_every=10, start_time=0.0,
)
# The same field works with deterministic adaptive evolution:
# AdaptiveDynamics(integrator, control, field=field).run(data, t_end=10.0)
```

`RampedField` holds `initial` before `start_time` (default zero), interpolates
linearly over `duration`, then holds `final`. Inputs have shape
`[n_grid, n_species]` and use energy units, not beta-scaled potentials.
They are copied at construction and converted to the density's device/dtype
on evaluation.

For another prescribed protocol, subclass `ExternalField`:

```python
import math
from equicdft.evolution import ExternalField

class OscillatingField(ExternalField):
    def __init__(self, amplitude, omega):
        self.amplitude = amplitude.detach().clone()
        self.omega = omega

    def __call__(self, time, data):
        return self.amplitude.to(data['rho']) * math.sin(self.omega * time)
```

Custom fields must return a
finite tensor matching `rho`'s shape, row order, dtype and device. Evaluation
must not mutate data or internal state, depend on the evolving density, or
draw random numbers: adaptive trials can repeat or revisit a time. This is a
prescribed-field interface, not a feedback controller. Override `metadata()`
to record custom parameters, and retain the protocol source with scientific runs.

For sudden switches, declare a strictly increasing `breakpoints` sequence of
absolute times. The field must be right-continuous at each switch. Both drivers
land on those times, evolve up to the left limit, then change the field at
fixed density. Fixed-step runs split nominal steps internally, keeping the
nominal output cadence and restoring `integrator.dt`; stochastic substeps
draw independent increments at their actual durations. Adaptive steps use
the field at the start, midpoint and endpoint, without carrying a rejected
trial's field into the accepted state. Choose a timestep or `dt_max` that
resolves smooth oscillations; error estimation cannot detect an arbitrarily
undersampled protocol.

The energy of a driven system need not decrease. Adaptive acceptance instead
checks the change in `beta_A` minus external work, across each half step and
the full step. Smooth-protocol work is estimated by

$$
\beta\Delta W=\beta\Delta V\sum_{g,a}
\frac{\rho^{\rm start}_{a,g}+\rho^{\rm end}_{a,g}}2
\left(V^{\rm end}_{a,g}-V^{\rm start}_{a,g}\right).
$$

The full-step versus two-half-step work difference, divided by
`max(1, total_particle_number)`, also contributes to `error` and is bounded by
`error_tolerance`. Work at a field jump is exactly
`beta * voxel_volume * sum(rho * delta_V)`, with no density update.
Stochastic runs record work but never impose energy-descent acceptance.

Driven trajectories additionally save `V_ext` and cumulative `beta_work` at
each recorded frame, plus field metadata. Work starts at zero for each run.
For a restart, restore the density (and stochastic RNG state), reuse the same
protocol and pass the saved absolute time as `start_time`; adaptive `t_end`
is an absolute endpoint too. Add the previous final work when joining records.
An adaptive restart may choose different subsequent steps; stochastic restart
reproducibility requires the same backend and subsequent step/substep schedule.
These protocols do not update metal-module parameters, temperature or geometry.

## Model evaluation and spatial discretization

The integrators call the model as `GridCACEModel.forward` is called:
`model(data, compute_c1=..., compute_c2=False, compute_ideal=True,
compute_local_mu=False, compute_polarization_derivative=False,
use_voxel_mask=False)`. A custom model must accept these keywords (for
example through `**kwargs`) and return `beta_F` and, when `compute_c1=True`,
the full `density_derivative`. The semi-implicit stabilizer additionally
differentiates `beta_F` twice.

`relaxation_rates(data)` returns the linear relaxation rate of every Fourier
mode about the homogeneous reference (the eigenvalues of $B(\mathbf k)$
defined below), and `stability_limit(data)` returns the explicit-Euler bound
$2/\max_{\mathbf k}\lambda(\mathbf k)$; $1/\min_{\mathbf k\ne0}\lambda$ is the
slowest linear relaxation time, which sets the duration a run needs. Both are
linear estimates about the reference; a field far from it, or a strongly
density-dependent mobility, may need a smaller explicit step.

The spatial update uses shared fluxes across neighboring voxel faces. For
spacing $h$ along one grid direction,

$$
M_{g+1/2}=\frac{M_g+M_{g+1}}2,\qquad
j_{a,g+1/2}=-k_BT\sum_b M_{ab,g+1/2}
                  \frac{u_{b,g+1}-u_{b,g}}h.
$$

The same face flux enters adjacent voxels with opposite signs, conserving
each species' integrated particle number. Faces adjoining an excluded voxel
carry zero flux in the Euler integrator.

`EulerIntegrator(model, mobility, dt)` uses

$$
\boldsymbol\rho^{n+1}
=\boldsymbol\rho^n-\Delta t\,\operatorname{div}_h\boldsymbol j^n.
$$

It is explicit and first order in time. Choose a timestep small enough to
resolve relaxation and preserve positive accessible densities. An invalid
trial step raises an error; densities are not clipped or renormalized.

`SemiImplicitIntegrator(model, mobility, dt)` treats an approximate linear
relaxation implicitly while evaluating the full nonlinear response explicitly:

$$
(I+\Delta t B)\boldsymbol\rho^{n+1}
=\boldsymbol\rho^n+\Delta t
 [\mathcal R(\boldsymbol\rho^n)+B\boldsymbol\rho^n],
$$

where $\mathcal R$ is the same conserved face-flux response as in Euler.
Adding and subtracting $B\boldsymbol\rho$ preserves the underlying equation.
This is also first order in time, with a species-matrix solve at each Fourier
mode.

At its first step, the integrator constructs $B$ from a homogeneous reference
density, its mobility, and the full intrinsic free-energy curvature. Writing
$H_\beta$ for the response of $\beta\mu$ to density,

$$
B(\mathbf k)=k_BT\,q_h^2(\mathbf k)M_{\mathrm{ref}}
 H_{\beta,+}(\mathbf k),\qquad
q_h^2(\mathbf k)=\sum_i\frac{4\sin^2(k_i h_i/2)}{h_i^2}.
$$

Only negative eigenvalues of the reference curvature are clipped when forming
$H_{\beta,+}$. The full model response remains in $\mathcal R$, so this does
not remove physically unstable directions from the evolution.

`reference_density` optionally supplies a positive species vector; otherwise
the first field's species means over free voxels are used (excluded voxels
hold zeros and frozen voxels may hold anything). A smaller reference strengthens the
stabilizer (the ideal curvature grows as $1/\rho$), which buys robustness at
low densities at the price of extra first-order damping error: the step is
reduced by $(I+\Delta tB)^{-1}$, so a stronger $B$ slows the computed
relaxation at fixed $\Delta t$. `reference_curvature` optionally
supplies a `beta`-unit species matrix or Fourier matrices of shape
`[*grid_size, n_species, n_species]`, avoiding automatic Hessian construction.
Supplied Fourier matrices must be Hermitian in species indices and satisfy
$H_\beta(-\mathbf k)=H_\beta(\mathbf k)^*$, so the update remains real.
Automatic construction requires twice differentiable model energy. Its
anchor-row Hessian gives the exact bulk Fourier curvature for a
translation-invariant intrinsic model. Otherwise it provides an approximate
stabilizer; the compensating $B\boldsymbol\rho$ term still preserves the full
response. The reference operator is built once and reused. Assigning a new
positive finite `integrator.dt` rebuilds only the propagator on the next step,
not the reference operator. A later field
with a different grid, temperature or reference density is rejected, and a
new integrator is also needed after changing model parameters or the
mobility. Excluded voxels require an explicitly supplied homogeneous liquid
`reference_curvature`; the restricted no-flux solve then keeps them empty.
The implicit part relaxes a stiffness constraint but does not guarantee
positivity or energy decrease at arbitrary timesteps.

## Stochastic dynamics

Use a separate integrator to include thermal fluctuations; deterministic calls
above are unchanged:

```python
from equicdft.evolution import StochasticEulerIntegrator, DensityMobility

integrator = StochasticEulerIntegrator(model, ConstantMobility(0.1), dt=1e-4, seed=42)
trajectory = Dynamics(integrator).run(data, n_steps=1000, save_every=10)
# Alternatively: mobility=DensityMobility(diffusivity=0.1)
```

The value and timestep are illustrative, not a calibrated physical choice.
This fixed-step Euler--Maruyama method supports constant scalar, diagonal or
full symmetric positive-semidefinite species mobility, and `DensityMobility`.
It uses the same model
derivative, external potential and shared faces as deterministic Euler. Each
face receives one integrated random flux

$$
\Delta J_f^R=\sqrt{\frac{2\Delta t}{\beta\Delta V}}\,B_fz_f,
\qquad B_fB_f^{\mathsf T}=M_f,\qquad z_f\sim\mathcal N(0,I),
$$

where delta V is the voxel volume. Its discrete divergence is added to the
deterministic density increment. Independent face draws conserve every species
by cancellation, including at periodic boundaries. Excluded voxels block both
fluxes. Matrix square roots preserve cross-species covariance and null modes;
no diagonal jitter is added. Thermal energy is read from `1/data['beta']`,
not a separate noise temperature. There is no spatial filter or adjustable
noise scale. `rhs` remains the deterministic drift; use `step`, not `advance`,
for stochastic updates. General `LocalMobility` is not supported yet because
its discrete thermal drift must be specified consistently.

For `DensityMobility`, both drift and noise use the arithmetic face value
`M_{a,f} = beta D_a (rho_{a,L} + rho_{a,R})/2`, evaluated at the current state
(Ito). No extra thermal drift is needed for this specific discretization:
writing `G` for the face gradient, the density-space mobility is
`K = G^T M_face G / (beta delta V)`. The Ito correction would be
`sum_j partial K_ij / partial rho_j`. At each face its two endpoint
derivatives are equal, `partial_L M_face = partial_R M_face = beta D_a/2`,
while the two gradient coefficients have opposite signs, so they cancel.
This also holds with fixed blocked faces and independent species with different
diffusivities. It does not justify omitting thermal drift for arbitrary
nonlinear mobility. The shared-face covariance matches this same `K`.

Energy increases are physical and are recorded, not rejected. By default a
nonpositive or nonfinite trial stops with the last valid trajectory and RNG
checkpoint. An optional minimum-density guard is enabled explicitly:

```python
integrator = StochasticEulerIntegrator(model, mobility, dt=1e-5,
                                      seed=42, rho_min=1e-4)
```

After the drift and noise update, accessible densities below `rho_min` are
floored. The added mass is removed in proportion to each voxel's density above
the floor, separately for each species. Excluded voxels remain zero; species
without floor hits are unchanged. Initial accessible densities must already
meet the floor. Nonfinite trials still stop. No new random numbers are drawn
for the correction.

This is an explicit numerical regularization: species particle numbers are
preserved, but the original noise statistics and equilibrium distribution need
not be. With the guard enabled, saved trajectories include cumulative per-species
`floor_clamped_voxels` (hits summed over all steps), `floor_added_particles`
(mass added by flooring before removal), `floor_l1_particles` (absolute mass
redistributed), and `floor_max_density_change` (largest single-cell correction
so far). These totals start at zero for each `Dynamics.run`, including restarts.
`rho_min` and the correction rule are recorded in metadata. Floor/timestep
sensitivity and stationary fluctuation statistics require separate validation.

Do not repeatedly
redraw failed steps or select successful seeds: this biases the process.
`AdaptiveDynamics` refuses stochastic integrators; deterministic step-doubling
and energy-descent gates are not a stochastic controller.

Every saved frame includes `rng_state`. To resume on the same backend/device:

```python
state = dict(data, rho=trajectory['rho'][-1].to(data['rho']))
integrator.set_rng_state(state, trajectory['rng_state'][-1])
continued = Dynamics(integrator).run(state, n_steps=1000, save_every=10)
```

Use the same model, mobility, geometry and timestep, and offset the returned
time axis when joining trajectories. Cross-device bitwise reproducibility is
not promised. Rewinding the RNG after a failed trial preserves the checkpoint;
it is not an adaptive Brownian-bridge algorithm or permission to retry until
a trajectory succeeds.

Validate the stationary covariance and correlation times as dt decreases:
Euler--Maruyama has finite-step sampling bias. For a stable homogeneous
quadratic functional, volume-normalized modes approach
`C(k,0) = inverse(H_beta(k))/V` for nonzero k. The relaxation uses the same
finite-difference `q_h^2` as deterministic Euler. Fine grids with far below
one particle per voxel may make Gaussian fluctuating density inappropriate;
a learned mean-density functional is not automatically the coarse-grained
free energy for stochastic trajectories. Agreement with MD and nucleation
rates requires separate scientific validation of this field interpretation.

## Adaptive timesteps

`AdaptiveDynamics(integrator, control).run(data, t_end, save_every=1)` takes
every step twice from the same stored rate, as one full step and as two half
steps with one model evaluation at the midpoint, and keeps the half-step
result. With `StepControl(dt_min, dt_max, ...)` the step is bounded before it
is tried by `dt_max`, by `max_relative_change` $\delta$ through
$\Delta t\le\delta\min_g\rho_g/|R_g|$ over accessible voxels (which keeps an
explicit step positive by construction), and for an explicit integrator by
`stability_safety` times the linear stability bound; on a grid with excluded
voxels that bound is evaluated for the periodic reference at the mean
accessible density, which blocked faces cannot raise. Two estimates of the
local error relative to the density are then formed from the integrator's own
updates, and the larger must not exceed `error_tolerance` $\epsilon$:

$$
e_1=\max_g\frac{|\rho^{\rm full}_g-\rho^{\rm half}_g|}{\rho^{\rm half}_g},\qquad
e_2=\max_g\frac{1}{\rho^{\rm half}_g}\sum_{s=1}^{2}
\Bigl|\tfrac{\Delta t}{2}R(\boldsymbol\rho^{(s)})_g-\bigl(\rho^{(s)}_g-\rho^{(s-1)}_g\bigr)\Bigr| .
$$

$e_1$ is the step-doubling difference, the error of the half-step result for
a first-order method (the full step errs by about twice as much); it includes
the backward-Euler damping error of the semi-implicit step on stiff modes, so
it responds whenever halving the step changes the result. $e_2$ sums the
defect of each half step, $s=1,2$ with $\boldsymbol\rho^{(0)}$ the starting
state: the displacement the half step produced against the one its endpoint
rate implies. It vanishes for a linear problem whose stabilizer matches its
curvature, and it catches the case $e_1$ misses: a mode whose stabilizer is
much stiffer than its physics is frozen alike by both step sizes, so the two
results agree while the mode moves far more slowly than it should. On a
linear test with the stabilizer set to 100 times the true curvature the
former rate-difference estimate was 90-160 times below the exact one-step
error; the estimate above is within a factor 0.7-4 of it in every stiffness
regime tested (`tests/evolution/test_adaptive.py`), and the trajectory error
of that over-stabilized problem now falls with the tolerance where it
previously did not.

After an accepted step the next step is scaled by
$0.9\sqrt{\epsilon/e}$, at most by `growth`; after a rejection for accuracy
by the same factor but at least by one tenth; after a rejection for
positivity or free energy by `shrink`. An accepted step's next proposal is
floored at `dt_min` before the independent stability and relative-change
bounds are applied; this does not override either bound or allow a rejected
step at `dt_min`. For a static field, the free energy must not rise across either half step or
the whole step by more than
`energy_increase` + `energy_roundoff` $\cdot|\beta A|$: the first is an
absolute tolerance in $k_BT$, independent of the zero of the free energy
(default zero); the second is a roundoff allowance, relative because the
rounding error of a summed free energy scales with its magnitude (default
$10^{-12}$, a few thousand float64 ulps; float32 models need about
$10^{-5}$). Set `energy_roundoff=0` for acceptance decisions that are exactly
independent of the zero of the free energy. A step that would have to fall
below `dt_min` to satisfy the bounds, or that is still rejected at `dt_min`,
stops the run with the last accepted state attached as `error.trajectory`;
the final step and steps landing on field breakpoints may be shortened to
land exactly on their endpoint. Driven-field acceptance subtracts external
work as described above.

The model evaluation at the accepted state starts the next step, so an
accepted step normally costs two model evaluations, twice the fixed-step cost;
a field jump also refreshes the endpoint derivative. A step
rejected for positivity or for an unconverged masked implicit solve costs
none or one, for accuracy or free energy one or two. The result has the
`Dynamics` fields plus `dt` and the error estimate `error` per frame, the
counts of accepted steps, rejections by cause (`rejected_positivity`,
`rejected_solve`, `rejected_accuracy`, `rejected_energy`) and model
evaluations, and the step cap `dt_cap` in the metadata.

```python
control = StepControl(dt_min=1e-4, dt_max=50.0, max_relative_change=0.05, error_tolerance=1e-3)
trajectory = AdaptiveDynamics(SemiImplicitIntegrator(model, mobility, dt=0.01), control).run(data, t_end=300.0, save_every=10)
```

Where it pays: approaching equilibrium, or across transients with widely
different rates, the semi-implicit integrator reaches a given time with a
controlled local error in a fraction of the fixed-step evaluations (an LJ
relaxation to ten relaxation times takes 100-130 model evaluations with the
error estimate held at $10^{-2}$-$10^{-3}$, against 150-600 at fixed steps
without any error control). For a time-accurate trajectory over a single
relaxation time a well-chosen fixed step is as efficient or better, because
the local error of a first-order method is then roughly uniform in time and
every adaptive step costs two evaluations (an LJ square-wave transient to
$t=60$ reached the accuracy of 120 fixed semi-implicit steps with 260
adaptive evaluations); the explicit integrator gains nothing, since its
stability bound already fixes its step and the doubled cost remains.
With a loose tolerance the controller is still bounded by `max_relative_change`,
so it reaches equilibrium quickly but with the slowed kinetics of the
over-stabilized modes; a tight tolerance holds those modes to their physical
rates at the cost of smaller steps. The remaining lever for time accuracy is
a second-order step, not a different step controller.

## Trajectories

`Dynamics(integrator).run(data, n_steps, save_every=1)` records the initial
state, every requested sampling interval, and the final state. The returned
dictionary contains `time`, `rho`, `beta_A`, `particle_numbers`, and
`min_density`. `beta_A` includes the intrinsic functional and the applied
external-potential energy. `min_density` refers to accessible voxels.

Recorded tensors are detached and moved to the CPU; the evolving state stays
on the model's device. `Dynamics.save(trajectory, path)` writes a PyTorch file
without overwriting an existing file. The trajectory includes grid geometry,
temperature, `beta`, timestep, method, and mobility metadata, and
`metadata["completed"]` with `metadata["steps_completed"]`. Identify the
model checkpoint separately when retaining a scientific result. The free
energy of a recorded frame is taken from the next step's derivative pass
(`integrator.last_beta_A`), so recording every step costs no additional model
evaluation; only the final frame needs one more forward pass.

If a step is rejected, the error still propagates, but the frames recorded so
far and the last valid density are attached to it as `error.trajectory` with
`metadata["completed"] = False`; restart from its final frame with a smaller
timestep rather than repeating the run.
If the last frame's energy cannot be evaluated, it is recorded as `NaN` only
in that failed trajectory; completed trajectories require finite energies.

Inspect particle-number conservation and density
positivity, and repeat with a smaller timestep to establish temporal
convergence. For static external fields, the continuous-time evolution
dissipates free energy; a finite numerical timestep still needs validation.

## Cost and devices

One explicit step is one model evaluation with its density derivative plus a
few tens of elementwise kernels on the grid; one semi-implicit step adds two
forward and one inverse FFT and a multiplication by the stored propagator
$(I+\Delta tB)^{-1}$, which is inverted when a step size is first used; the
two most recent step sizes stay cached, so an adaptive step and its half do
not rebuild it.
Grid geometry and walls are validated when first seen and reused while the same
tensors are passed unchanged. `V_ext` is validated separately on each call,
so a changed potential does not rebuild geometry or reuse a stale field.
Non-tensor containers such as lists and arrays
are revalidated on each call. A density returned by `step` is not re-checked
unless modified in place or the geometry changes, and a `ConstantMobility`
is applied as one shared matrix rather than one matrix per voxel. The model evaluation therefore
dominates for any grid large enough to occupy a GPU.

On a GPU, keep `rho`, `V_ext` and the geometry on the model's device; frames
are copied to the CPU only when recorded, so `save_every` sets the transfer
cost. Float32 can reduce evaluation cost, but its speed and accuracy need
checking on the target device. Conserved particle numbers then drift at
float32 roundoff, which the recorded
`particle_numbers` make visible. For many small grids (parameter sweeps,
ensembles of initial conditions) the kernel launch overhead of one field at a
time dominates, and evolving a batch of fields in one call would be the next
structural gain; the integrators currently evolve one field per call.
