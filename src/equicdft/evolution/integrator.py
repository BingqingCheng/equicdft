"""Deterministic, conservative timesteps for species density fields."""

import math
from contextlib import contextmanager

import torch

from ..geometry.grid import common_grid_size, _flat_grid_positions, grid_spacing_tensor
from .mobility import ConstantMobility, _validate_matrix
from .field import _metal_context, _metal_dipole


@contextmanager
def _evaluation(model):
    """Evaluate a frozen functional without changing its caller-owned mode."""
    modes = [(module, module.training) for module in model.modules()] if isinstance(model, torch.nn.Module) else []
    if modes:
        model.eval()
    try:
        yield
    finally:
        for module, training in modes:
            module.training = training


def _positive_scalar(value, name):
    tensor = torch.as_tensor(value)
    if tensor.numel() != 1 or tensor.is_complex() or tensor.dtype == torch.bool:
        raise ValueError("{} must be a positive scalar.".format(name))
    result = float(torch.as_tensor(value, dtype=torch.float64).detach())
    if not math.isfinite(result) or result <= 0:
        raise ValueError("{} must be finite and positive.".format(name))
    return result


_GEOMETRY_KEYS = ("grid_size", "grid_positions", "grid_spacing", "excluded_mask", "mask", "beta", "temperature")


def _with_mask(data, mask=None, default=None):
    """Resolve one voxel mask: driver argument, data, then integrator default."""
    mask = data.get("mask", default) if mask is None else mask
    return data if mask is None else dict(data, mask=mask)


class DensityError(ValueError):
    """A density is non-finite, non-positive, or nonzero in an excluded voxel."""


def _fingerprint(value):
    """Cache tensors and immutable scalars; rebuild for other input types."""
    if torch.is_tensor(value):
        return ("tensor", id(value), value._version)
    if value is None or isinstance(value, (bool, int, float, str)):
        return ("value", value)
    if isinstance(value, tuple) and all(isinstance(item, (bool, int, float, str)) for item in value):
        return ("value", value)
    # Lists/arrays (including containers holding them) have no version counter.
    return ("uncached", object())


class _Grid:
    """One complete regular grid, retaining the input row order.

    The static geometry (shape, row order, spacing, walls)
    is validated once and reused while the same data tensors are passed in
    unchanged; ``cached`` is the previous grid of the same integrator. Density
    checks are skipped only for unchanged outputs under unchanged geometry.
    """

    def __init__(self, data, cached=None, trusted=False):
        self.rho = data["rho"]
        if self.rho.ndim != 2 or not self.rho.is_floating_point() or not self.rho.shape[1]:
            raise ValueError("Evolution requires rho shaped [n_grid, n_species].")
        self.signature = (tuple(_fingerprint(data.get(key)) for key in _GEOMETRY_KEYS)
                          + (self.rho.shape, self.rho.dtype, self.rho.device))
        # Holding the tensors keeps their ids from being reused while cached.
        self.anchors = tuple(data.get(key) for key in _GEOMETRY_KEYS)
        reused = cached is not None and cached.signature == self.signature
        if reused:
            self.geometry = cached.geometry
        else:
            self.geometry = self._build(data)
        self.__dict__.update(self.geometry)
        potential = data.get("V_ext")
        self.potential = None
        if potential is not None:
            self.potential = torch.as_tensor(potential, dtype=self.rho.dtype, device=self.rho.device)
            if self.potential.shape != self.rho.shape or not torch.isfinite(self.potential).all():
                raise ValueError("V_ext must be finite and have the same shape as rho.")
        if not trusted or not reused:
            self.check_density(self.rho)

    def _build(self, data):
        rho = self.rho
        shape = common_grid_size(data["grid_size"], torch.Size())
        indices, canonical = _flat_grid_positions(data["grid_positions"], shape, torch.Size(), rho.device)
        indices = indices[0]
        if rho.shape[0] != math.prod(shape):
            raise ValueError("rho must cover the complete regular grid.")
        spacing = grid_spacing_tensor(data["grid_spacing"], dtype=rho.dtype, device=rho.device)
        excluded = torch.as_tensor(
            data.get("excluded_mask", torch.zeros(rho.shape[0], dtype=torch.bool)), device=rho.device,
        )
        if excluded.dtype != torch.bool or excluded.shape != rho.shape[:1]:
            raise ValueError("excluded_mask must be a boolean [n_grid] tensor.")
        if excluded.all():
            raise ValueError("At least one voxel must be accessible.")
        frozen = torch.as_tensor(
            data.get("mask", torch.zeros(rho.shape[0], dtype=torch.bool)), device=rho.device,
        )
        if frozen.dtype != torch.bool or frozen.shape != rho.shape[:1]:
            raise ValueError("mask must be a boolean [n_grid] tensor; True freezes all species.")
        held = rho[frozen & ~excluded]
        if held.numel() and not bool(torch.all(torch.isfinite(held) & (held > 0))):
            raise ValueError("Frozen voxels keep finite positive densities; use excluded_mask for empty voxels.")
        inactive = excluded | frozen
        geometry = {
            "shape": shape, "indices": indices, "canonical": canonical,
            "spacing": spacing, "volume": spacing.prod(),
            "beta": _positive_scalar(data["beta"], "beta"),
            "temperature": _positive_scalar(data["temperature"], "temperature"),
            "excluded": excluded, "has_excluded": bool(excluded.any()),
            "frozen": frozen, "has_frozen": bool(frozen.any()),
            "free": ~inactive, "has_blocked": bool(inactive.any()),
            "blocked": None,
            "_regions": None,
        }
        if geometry["has_blocked"]:
            self.__dict__.update(geometry)
            dense = self.dense(inactive)
            geometry["blocked"] = [(dense | torch.roll(dense, -1, axis))[..., None] for axis in range(3)]
        return geometry

    def dense(self, values):
        if not self.canonical:
            values = torch.zeros_like(values).index_copy(0, self.indices, values)
        return values.reshape(self.shape + values.shape[1:])

    def flat(self, values):
        values = values.reshape((-1,) + values.shape[3:])
        return values if self.canonical else values[self.indices]

    def check_density(self, rho):
        accessible = rho.masked_fill(self.excluded[:, None], 1.0) if self.has_excluded else rho
        low, high = (value.item() for value in torch.aminmax(accessible))
        if not (0.0 < low and high < math.inf):  # a NaN fails both comparisons
            raise DensityError("Accessible densities must be finite and positive; reduce the timestep.")
        if self.has_excluded and torch.any(rho[self.excluded] != 0):
            raise DensityError("Excluded voxels must have exactly zero density.")

    def external(self):
        """Return the validated external potential, zero when none was given."""
        return torch.zeros_like(self.rho) if self.potential is None else self.potential

    def divergence(self, flux, axis):
        """Shared conservative face update, including excluded and frozen voxels."""
        if self.has_blocked:
            flux = flux.masked_fill(self.blocked[axis], 0)
        return -(flux - torch.roll(flux, 1, axis)) / self.spacing[axis]

    def restore_frozen(self, rho):
        """Keep original frozen values bitwise, without mass redistribution."""
        return torch.where(self.frozen[:, None], self.rho.detach(), rho) if self.has_frozen else rho

    def free_regions(self):
        """Connected free regions (no blocked face between members), for the density-floor repair.

        Excluded and frozen voxels both split the grid; a region's species
        numbers are conserved by every flux, so a repair must act per region.
        """
        if self._regions is None:
            if not self.has_blocked:
                self._regions = [torch.nonzero(self.free).flatten()]
            else:
                # Static topology: label on CPU once, retaining arbitrary input row order.
                free = self.dense(self.free).cpu().reshape(-1).tolist()
                nx, ny, nz = self.shape
                regions = []
                for start, active in enumerate(free):
                    if not active:
                        continue
                    free[start] = False
                    stack, cells = [start], []
                    while stack:
                        cell = stack.pop()
                        cells.append(cell)
                        x, rem = divmod(cell, ny * nz)
                        y, z = divmod(rem, nz)
                        neighbours = [((x+d) % nx)*ny*nz+y*nz+z for d in (-1, 1)]
                        neighbours += [x*ny*nz+((y+d) % ny)*nz+z for d in (-1, 1)]
                        neighbours += [x*ny*nz+y*nz+(z+d) % nz for d in (-1, 1)]
                        for neighbour in neighbours:
                            if free[neighbour]:
                                free[neighbour] = False
                                stack.append(neighbour)
                    selected = torch.zeros(math.prod(self.shape), dtype=torch.bool, device=self.rho.device)
                    selected[cells] = True
                    rows = self.flat(selected.reshape(self.shape))
                    regions.append(torch.nonzero(rows).flatten())
                self._regions = regions
            self.geometry["_regions"] = self._regions
        return self._regions

    def project_free_increment(self, increment):
        """Restrict a correction to free cells, conserving each region's species sums."""
        result = increment.masked_fill(~self.free[:, None], 0)
        for rows in self.free_regions():
            result[rows] -= result[rows].mean(0)
        return result


class EulerIntegrator:
    """Explicit first-order step for j = -M grad(mu), with shared face fluxes.

    ``model`` returns the full intrinsic beta-free-energy derivative. Mobility
    is physical, so its coefficient against grad(beta*mu) is ``M / beta``.
    Excluded voxels have zero density and block every adjoining face. Frozen
    ``mask`` voxels also block faces but retain their supplied densities.
    """

    def __init__(self, model, mobility, dt, *, mask=None):
        self.model = model
        self.mobility = mobility
        self.mask = mask
        self.dt = dt
        self.last_beta_A = None
        self.last_metal_dipole = None
        self._grid = None
        self._last_output = None

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        self._dt = _positive_scalar(value, "dt")

    def _grid_for(self, data):
        """Validate ``data``, reusing the previous geometry when it is unchanged.

        A density this integrator itself returned from ``step`` was checked
        then and is not checked again unless it or the geometry changes.
        """
        output, version = self._last_output if self._last_output is not None else (None, None)
        trusted = data["rho"] is output and output._version == version
        self._grid = _Grid(_with_mask(data, default=self.mask), self._grid, trusted=trusted)
        return self._grid

    def _model_output(self, data, derivatives):
        if getattr(self.model, "requires_dipole_density", False) or "dipole_density" in data:
            raise ValueError("Evolution currently supports species densities, not dipole density.")
        state = dict(data, rho=data["rho"].detach().clone())
        self.last_metal_dipole = None
        if "metal_external_field" in data:
            _metal_context(self.model, data)
        with _evaluation(self.model), torch.enable_grad():
            output = self.model(
                state, compute_c1=derivatives, compute_c2=False, compute_ideal=True,
                compute_local_mu=False, compute_polarization_derivative=False,
                use_voxel_mask=False,
            )
        if "metal_external_field" in data:
            self.last_metal_dipole = _metal_dipole(self.model, output)
        return output

    def _mobility_matrices(self, data, grid):
        """Return per-voxel matrices, or one shared matrix for a uniform mobility."""
        with _evaluation(self.mobility), torch.no_grad():
            matrices = self.mobility(data).detach()
        if (matrices.shape != grid.rho.shape + (grid.rho.shape[-1],)
                or matrices.dtype != grid.rho.dtype or matrices.device != grid.rho.device):
            raise ValueError("Mobility must return finite matrices matching rho's shape, dtype and device.")
        if not getattr(self.mobility, "validated", False):
            # Built-in mobilities validate their own output; custom modules are checked here.
            if not torch.isfinite(matrices).all():
                raise ValueError("Mobility must return finite matrices matching rho's shape, dtype and device.")
            _validate_matrix(matrices)
        if isinstance(self.mobility, ConstantMobility):
            return matrices[0], True
        return grid.dense(matrices), False

    def _rhs(self, data, grid):
        self.last_beta_A = None
        output = self._model_output(data, derivatives=True)
        derivative = output["density_derivative"].detach()
        if derivative.shape != grid.rho.shape or not torch.isfinite(derivative).all():
            raise ValueError("The model must return a finite full density_derivative.")
        external = grid.external()
        energy = (output["beta_F"].sum() + grid.beta * grid.volume * (grid.rho * external).sum()).detach()
        if not torch.isfinite(energy):
            raise ValueError("The model free energy must be finite.")
        self.last_beta_A = energy
        u = grid.dense(derivative + grid.beta * external)
        matrices, uniform = self._mobility_matrices(data, grid)
        rate = torch.zeros_like(u)
        for axis, spacing in enumerate(grid.spacing):
            gradient = (torch.roll(u, -1, axis) - u) / spacing
            if uniform:
                flux = -(gradient @ matrices.transpose(-1, -2)) / grid.beta
            else:
                face_mobility = (matrices + torch.roll(matrices, -1, axis)) / 2
                flux = -(face_mobility @ gradient.unsqueeze(-1)).squeeze(-1) / grid.beta
            rate += grid.divergence(flux, axis)
        return grid.flat(rate).detach()

    def rhs(self, data):
        """Return drho/dt without altering data or retaining an autograd graph."""
        return self._rhs(data, self._grid_for(data))

    def beta_A(self, data):
        """Return intrinsic beta F plus the static external-potential energy.

        After a ``step`` or ``rhs`` call, ``last_beta_A`` already holds this
        value for the state that call started from, at no extra model cost.
        """
        grid = self._grid_for(data)
        intrinsic = self._model_output(data, derivatives=False)["beta_F"].sum()
        result = intrinsic + grid.beta * grid.volume * (grid.rho * grid.external()).sum()
        if not torch.isfinite(result):
            raise ValueError("The model free energy must be finite.")
        return result.detach()

    def _advance(self, grid, rate, dt):
        rho = grid.restore_frozen(grid.rho.detach() + dt * rate)
        grid.check_density(rho)
        self._last_output = (rho, rho._version)
        return rho

    def advance(self, data, rate, dt=None):
        """Apply a rate from ``rhs(data)`` over ``dt`` (default ``self.dt``).

        The model is not evaluated again, so a step can be retried with a
        smaller ``dt`` from the same rate; invalid results raise as in ``step``.
        """
        dt = self.dt if dt is None else _positive_scalar(dt, "dt")
        return self._advance(self._grid_for(data), rate, dt)

    def step(self, data):
        """Return one trial density; invalid steps raise instead of clipping."""
        grid = self._grid_for(data)
        return self._advance(grid, self._rhs(data, grid), self.dt)

    def relaxation_rates(self, data, reference_density=None):
        """Return the linear relaxation rate of every Fourier mode.

        The rates are the eigenvalues of the homogeneous reference operator
        ``k_B T q_h^2 M_ref H_+(k)`` (the semi-implicit stabilizer), shaped
        ``grid_size + (n_species,)``; ``1 / rates[rates > 0].min()`` is the
        slowest linear relaxation time. The reference is the species means
        over free voxels unless ``reference_density`` is given; frozen voxels
        only block faces, which cannot raise the largest rate.
        """
        grid = self._grid_for(data)
        if grid.has_excluded:
            raise ValueError("Relaxation rates require a fully accessible periodic grid.")
        reference = _reference_density(grid, reference_density)
        operator = _reference_operator(self.model, self.mobility, data, grid, reference)
        return torch.linalg.eigvals(operator).real

    def stability_limit(self, data, reference_density=None):
        """Return the explicit-Euler bound ``2 / max(relaxation_rates)``.

        This is the linear estimate about the homogeneous reference; a field
        far from it, or a density-dependent mobility, may need a smaller step.
        """
        return float(2 / self.relaxation_rates(data, reference_density).max())


def _reference_density(grid, reference_density):
    """Return the validated homogeneous reference species vector.

    The default is the mean over free voxels: excluded voxels hold zeros and
    frozen voxels may hold any density, and neither evolves.
    """
    rho = grid.rho
    if reference_density is None:
        # An all-frozen field has nothing to stabilize; any positive reference serves.
        pool = grid.free if bool(grid.free.any()) else ~grid.excluded
        reference = rho[pool].mean(0)
    else:
        reference = torch.as_tensor(reference_density, dtype=rho.dtype, device=rho.device)
    if reference.shape != (rho.shape[-1],) or not torch.isfinite(reference).all() or torch.any(reference <= 0):
        raise ValueError("reference_density must be a finite positive species vector.")
    return reference.detach()


def _reference_curvature(model, data, grid):
    """FFT of p Hessian rows, not an n_grid-by-n_grid dense Hessian."""
    if getattr(model, "requires_dipole_density", False) or "dipole_density" in data:
        raise ValueError("Evolution currently supports species densities, not dipole density.")
    rho = data["rho"].detach().clone().requires_grad_(True)
    state = dict(data, rho=rho)
    anchor = int(torch.nonzero(grid.indices == 0)[0, 0])
    with _evaluation(model), torch.enable_grad():
        energy = model(
            state, compute_c1=False, compute_c2=False, compute_ideal=True,
            compute_local_mu=False, compute_polarization_derivative=False,
            use_voxel_mask=False,
        )["beta_F"].sum()
        mu = torch.autograd.grad(energy, rho, create_graph=True)[0] / grid.volume
        rows = []
        for species in range(rho.shape[-1]):
            value = mu[anchor, species]
            row = (torch.autograd.grad(value, rho, retain_graph=True, allow_unused=True)[0]
                   if value.requires_grad else None)
            rows.append(torch.zeros_like(rho) if row is None else row)
        jacobian = grid.dense(torch.stack(rows, dim=1).detach())
    # A row anchored at the origin is H(-r); conjugate its FFT to get H(k).
    # The discrete Jacobian already contains the voxel quadrature weight.
    return torch.fft.fftn(jacobian, dim=(0, 1, 2)).conj()


def _reference_operator(model, mobility, data, grid, reference, reference_curvature=None, *, parts=False):
    """Return ``k_B T q_h^2 M_ref H_+(k)`` for every mode, shape ``grid_size + (p, p)``."""
    n_species = grid.rho.shape[-1]
    reference_data = dict(data, rho=reference.expand_as(grid.rho).clone())
    with _evaluation(mobility), torch.no_grad():
        matrices = mobility(reference_data).detach()
    if (matrices.shape != grid.rho.shape + (n_species,)
            or matrices.dtype != grid.rho.dtype or matrices.device != grid.rho.device
            or not torch.isfinite(matrices).all()):
        raise ValueError("Reference mobility must provide finite matrices matching rho's shape, dtype and device.")
    _validate_matrix(matrices)
    if not torch.allclose(matrices, matrices[0].expand_as(matrices)):
        raise ValueError("Reference mobility must be uniform on a homogeneous field.")
    if reference_curvature is None:
        curvature = _reference_curvature(model, reference_data, grid)
    else:
        curvature = torch.as_tensor(reference_curvature, device=grid.rho.device)
        curvature = curvature.to(torch.complex128 if grid.rho.dtype == torch.float64 else torch.complex64)
        if curvature.shape == (n_species, n_species):
            curvature = curvature.expand(grid.shape + (n_species, n_species))
        if curvature.shape != grid.shape + (n_species, n_species):
            raise ValueError("reference_curvature must be a species matrix or full Fourier matrices.")
    if not torch.isfinite(curvature).all():
        raise ValueError("Reference curvature must be finite.")
    adjoint = curvature.transpose(-1, -2).conj()
    if reference_curvature is not None and not torch.allclose(curvature, adjoint, rtol=1e-5, atol=1e-7):
        raise ValueError("Reference curvature must be Hermitian.")
    negative_modes = curvature
    for axis, size in enumerate(grid.shape):
        reverse = (-torch.arange(size, device=curvature.device)) % size
        negative_modes = negative_modes.index_select(axis, reverse)
    if not torch.allclose(negative_modes, curvature.conj(), rtol=1e-5, atol=1e-7):
        raise ValueError("Reference curvature must satisfy H(-k) = conj(H(k)) for real fields.")
    eigenvalues, vectors = torch.linalg.eigh((curvature + adjoint) / 2)
    positive = (vectors * eigenvalues.clamp_min(0).unsqueeze(-2)) @ vectors.transpose(-1, -2).conj()
    q2 = grid.rho.new_zeros(grid.shape)
    for axis, (size, spacing) in enumerate(zip(grid.shape, grid.spacing)):
        modes = torch.arange(size, dtype=grid.rho.dtype, device=grid.rho.device)
        component = 4 * torch.sin(math.pi * modes / size).square() / spacing.square()
        shape = [1, 1, 1]
        shape[axis] = size
        q2 += component.reshape(shape)
    operator = q2[..., None, None] * (matrices[0].to(positive.dtype) @ positive) / grid.beta
    return (operator, positive, matrices[0]) if parts else operator


_SOLVE_TOLERANCE = 1e-12   # float64 target; lower precision uses a roundoff floor
_SOLVE_RESTART = 40
_SOLVE_CYCLES = 5


class ConvergenceError(ValueError):
    """An implicit solve did not reach its tolerance; a smaller step usually will."""


def _gmres(action, rhs, precondition):
    """Right-preconditioned restarted GMRES with double reorthogonalization.

    Solves ``action(x) = rhs`` through ``x = precondition(y)``. The masked
    IMEX operator need not be symmetric for coupled species, so CG is
    unsuitable. The least-squares residual decides when a candidate is
    formed and its true residual checked, so an iteration applies the
    operator once. The preconditioner preserves zero increments on frozen
    rows and zero species sums in each connected free region.
    """
    solution = torch.zeros_like(rhs)
    norm = torch.linalg.vector_norm(rhs)
    if norm == 0:
        return solution
    limit = max(_SOLVE_TOLERANCE, 32 * torch.finfo(rhs.dtype).eps) * norm
    for _ in range(_SOLVE_CYCLES):
        residual = rhs - action(solution)
        length = torch.linalg.vector_norm(residual)
        if length <= limit:
            return solution
        basis = [residual / length]
        hessenberg = rhs.new_zeros((_SOLVE_RESTART + 1, _SOLVE_RESTART))
        target = rhs.new_zeros(_SOLVE_RESTART + 1)
        target[0] = length
        for j in range(_SOLVE_RESTART):
            vector = action(precondition(basis[j]))
            # Reorthogonalize to retain accuracy for stiff/non-normal mixtures.
            for _ in range(2):
                for i, direction in enumerate(basis):
                    coefficient = (direction * vector).sum()
                    hessenberg[i, j] += coefficient
                    vector = vector - coefficient * direction
            next_length = torch.linalg.vector_norm(vector)
            hessenberg[j + 1, j] = next_length
            weights = torch.linalg.lstsq(hessenberg[:j + 2, :j + 1], target[:j + 2, None]).solution[:, 0]
            estimate = torch.linalg.vector_norm(target[:j + 2] - hessenberg[:j + 2, :j + 1] @ weights)
            exhausted = next_length <= torch.finfo(rhs.dtype).tiny or j == _SOLVE_RESTART - 1
            if estimate <= limit or exhausted:
                candidate = solution + precondition(sum(weight * direction for weight, direction in zip(weights, basis)))
                if torch.linalg.vector_norm(rhs - action(candidate)) <= limit:
                    return candidate
                if exhausted:
                    solution = candidate
                    break
            basis.append(vector / next_length)
    raise ConvergenceError("Masked implicit solve did not converge; reduce the timestep.")


class SemiImplicitIntegrator(EulerIntegrator):
    """First-order IMEX step with a frozen homogeneous Fourier stabilizer.

    The full nonlinear response stays explicit. Only the positive part of the
    reference beta-free-energy curvature enters the implicit operator; negative
    physical curvature is not removed from the evolution. Without a frozen
    mask the solve is diagonal in Fourier space. With a mask, a matrix-free
    solve uses the same curvature and shared blocked-face fluxes, with the
    periodic propagator as its preconditioner. With excluded voxels, supply
    ``reference_curvature`` from a separate homogeneous liquid calculation;
    an electrode/wall geometry is not a homogeneous Hessian reference.
    """

    def __init__(self, model, mobility, dt, reference_density=None, reference_curvature=None, *, mask=None):
        super().__init__(model, mobility, dt, mask=mask)
        self.reference_density = reference_density
        self.reference_curvature = reference_curvature
        self._operator = None
        self._propagators = {}   # dt -> (I + dt B)^-1 for the two most recent step sizes
        self._signature = None
        self._reference = None

    def _prepare(self, data, grid):
        """Build the reference operator on first use and guard its reuse."""
        if grid.has_excluded and self.reference_curvature is None:
            raise ValueError("SemiImplicitIntegrator requires a fully accessible periodic grid "
                             "or an explicit homogeneous reference_curvature for excluded voxels.")
        signature = (grid.shape, tuple(grid.spacing.tolist()), grid.beta, grid.temperature,
                     grid.rho.shape[-1], grid.rho.dtype, grid.rho.device)
        reference = _reference_density(grid, self.reference_density)
        if self._operator is None:
            self._operator, self._curvature, self._reference_mobility = _reference_operator(
                self.model, self.mobility, data, grid, reference, self.reference_curvature, parts=True,
            )
            self._signature = signature
            self._reference = reference
        elif signature != self._signature or not torch.allclose(reference, self._reference, rtol=1e-6, atol=0.):
            raise ValueError("Grid, reference density or temperature changed; construct a new semi-implicit integrator.")

    def _propagator_for(self, dt):
        """Return (I + dt B)^-1, keeping the two most recent step sizes.

        A fixed-step run reuses one propagator; step doubling alternates
        between a step and its half without rebuilding either.
        """
        propagator = self._propagators.get(dt)
        if propagator is None:
            if self._operator.shape[-1] == 1:
                propagator = 1 / (1 + dt * self._operator)
            else:
                identity = torch.eye(self._operator.shape[-1], dtype=self._operator.dtype, device=self._operator.device)
                propagator = torch.linalg.inv(identity + dt * self._operator)
            if len(self._propagators) >= 2:
                del self._propagators[next(iter(self._propagators))]
            self._propagators[dt] = propagator
        return propagator

    def _advance(self, grid, rate, dt):
        propagator = self._propagator_for(dt)
        if grid.has_frozen or grid.has_excluded:
            # Solve the same IMEX increment with blocked faces, not a periodic
            # solve followed by clipping (which leaks mass into frozen voxels).
            # Restrict the periodic preconditioner to conservative free-region
            # increments: zeroing frozen rows alone would leak region mass.

            def action(increment):
                modes = torch.fft.fftn(grid.dense(increment), dim=(0, 1, 2))
                potential = torch.fft.ifftn(
                    (self._curvature @ modes.unsqueeze(-1)).squeeze(-1), dim=(0, 1, 2),
                ).real
                response = torch.zeros_like(potential)
                for axis, spacing in enumerate(grid.spacing):
                    gradient = (torch.roll(potential, -1, axis) - potential) / spacing
                    flux = -(gradient @ self._reference_mobility.T) / grid.beta
                    response += grid.divergence(flux, axis)
                return increment - dt * grid.flat(response)

            def precondition(vector):
                modes = torch.fft.fftn(grid.dense(vector), dim=(0, 1, 2))
                smoothed = torch.fft.ifftn((propagator @ modes.unsqueeze(-1)).squeeze(-1), dim=(0, 1, 2)).real
                return grid.project_free_increment(grid.flat(smoothed))

            increment = _gmres(action, dt * rate, precondition)
            rho = torch.where(grid.free[:, None], grid.rho.detach() + increment, grid.rho.detach())
            grid.check_density(rho)
            self._last_output = (rho, rho._version)
            return rho
        rho_hat = torch.fft.fftn(grid.dense(grid.rho.detach()), dim=(0, 1, 2))
        rate_hat = torch.fft.fftn(grid.dense(rate), dim=(0, 1, 2))
        # (I + dt B) rho' = rho + dt [R + B rho]  is  rho' = rho + dt (I + dt B)^-1 R.
        updated = rho_hat + dt * (propagator @ rate_hat.unsqueeze(-1)).squeeze(-1)
        updated[0, 0, 0] = rho_hat[0, 0, 0]
        rho = grid.flat(torch.fft.ifftn(updated, dim=(0, 1, 2)).real)
        grid.check_density(rho)
        self._last_output = (rho, rho._version)
        return rho

    def advance(self, data, rate, dt=None):
        dt = self.dt if dt is None else _positive_scalar(dt, "dt")
        grid = self._grid_for(data)
        self._prepare(data, grid)
        return self._advance(grid, rate, dt)

    def step(self, data):
        grid = self._grid_for(data)
        self._prepare(data, grid)
        return self._advance(grid, self._rhs(data, grid), self.dt)
