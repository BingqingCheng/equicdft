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
cross-species coupling. The matrix must be symmetric positive semidefinite.

`LocalMobility(function)` calls `function(rho, temperature)` to evaluate a
mobility from the current species-density vector at each voxel and the fixed
temperature. The function may return diagonal values with the same shape as
`rho`, full matrices with shape `rho.shape + (n_species,)`, or a shared
scalar, species vector, or matrix. For example, returning `prefactor * rho`
gives independent mobilities proportional to each species density. Local-shaped
outputs take priority when a shape also matches a shared matrix. A custom
mobility module implements `forward(data)` and returns
a tensor with shape `[..., n_grid, n_species, n_species]`, on the same device
and with the same dtype as `data["rho"]`.

A local, instantaneous mobility is a working dynamical approximation. A
mobility extracted from MD at finite wavevector or with memory requires a
separate justification before being used as a local function.

## One timestep

Both integrators expose `step(data)`, which returns the new density tensor,
and `rhs(data)`, which evaluates its instantaneous time derivative.
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
the first field's species means are used. `reference_curvature` optionally
supplies a `beta`-unit species matrix or Fourier matrices of shape
`[*grid_size, n_species, n_species]`, avoiding automatic Hessian construction.
Supplied Fourier matrices must be Hermitian in species indices and satisfy
$H_\beta(-\mathbf k)=H_\beta(\mathbf k)^*$, so the update remains real.
Automatic construction requires twice differentiable model energy. Its
anchor-row Hessian gives the exact bulk Fourier curvature for a
translation-invariant intrinsic model. Otherwise it provides an approximate
stabilizer; the compensating $B\boldsymbol\rho$ term still preserves the full
response. The reference operator is reused: create a new integrator after changing model
parameters or the mobility. This initial semi-implicit method requires a
fully accessible periodic grid; nonempty `excluded_mask` is rejected.
The implicit part relaxes a stiffness constraint but does not guarantee
positivity or energy decrease at arbitrary timesteps.

## Trajectories

`Dynamics(integrator).run(data, n_steps, save_every=1)` records the initial
state, every requested sampling interval, and the final state. The returned
dictionary contains `time`, `rho`, `beta_A`, `particle_numbers`, and
`min_density`. `beta_A` includes the intrinsic functional and the applied
external-potential energy. `min_density` refers to accessible voxels.

Recorded tensors are detached and moved to the CPU; the evolving state stays
on the model's device. `Dynamics.save(trajectory, path)` writes a PyTorch file
without overwriting an existing file. The trajectory includes grid geometry,
temperature, `beta`, timestep, method, and mobility metadata. Identify the
model checkpoint separately when retaining a scientific result.

Inspect particle-number conservation and density
positivity, and repeat with a smaller timestep to establish temporal
convergence. For static external fields, the continuous-time evolution
dissipates free energy; a finite numerical timestep still needs validation.
