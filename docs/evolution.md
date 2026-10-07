# Density evolution

`equicdft.evolution` advances conserved species densities using an equilibrium
functional and a separate mobility. The initial implementation is deterministic,
at a fixed uniform temperature, on periodic regular grids.

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

## Mobility

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
the first field's species means are used. A smaller reference strengthens the
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
mobility. This initial semi-implicit method requires a
fully accessible periodic grid; nonempty `excluded_mask` is rejected.
The implicit part relaxes a stiffness constraint but does not guarantee
positivity or energy decrease at arbitrary timesteps.

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
step at `dt_min`. The free energy must not rise across either half step or
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
only the final step is shortened freely to land exactly on `t_end`.

The model evaluation at the accepted state starts the next step, so an
accepted step costs two model evaluations, twice the fixed-step cost; a step
rejected for positivity costs none or one, for accuracy or free energy one or
two. The result has the `Dynamics` fields plus `dt` and the error estimate
`error` per frame, the counts of accepted steps, rejections by cause and
model evaluations, and the step cap `dt_cap` in the metadata.

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
Grid geometry,
walls and `V_ext` are validated when first seen and reused while the same
tensors are passed unchanged. Non-tensor containers such as lists and arrays
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
