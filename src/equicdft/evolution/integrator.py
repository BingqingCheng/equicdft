"""Deterministic, conservative timesteps for species density fields."""

import math
from contextlib import contextmanager

import torch

from ..geometry.grid import common_grid_size, _flat_grid_positions, grid_spacing_tensor
from .mobility import ConstantMobility, _validate_matrix


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


_GEOMETRY_KEYS = ("grid_size", "grid_positions", "grid_spacing", "excluded_mask", "V_ext", "beta", "temperature")


def _fingerprint(value):
    """Cache tensors and immutable scalars; rebuild for other input types."""
    if torch.is_tensor(value):
        return ("tensor", id(value), value._version)
    if value is None or isinstance(value, (bool, int, float, str)):
        return ("value", value)
    # Lists/arrays (including containers holding them) have no version counter.
    return ("uncached", object())


class _Grid:
    """One complete regular grid, retaining the input row order.

    The static geometry (shape, row order, spacing, walls, external potential)
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
        potential = data.get("V_ext")
        if potential is not None:
            potential = torch.as_tensor(potential, dtype=rho.dtype, device=rho.device)
            if potential.shape != rho.shape or not torch.isfinite(potential).all():
                raise ValueError("V_ext must be finite and have the same shape as rho.")
        geometry = {
            "shape": shape, "indices": indices, "canonical": canonical,
            "spacing": spacing, "volume": spacing.prod(),
            "beta": _positive_scalar(data["beta"], "beta"),
            "temperature": _positive_scalar(data["temperature"], "temperature"),
            "excluded": excluded, "has_excluded": bool(excluded.any()),
            "potential": potential, "blocked": None,
        }
        if geometry["has_excluded"]:
            self.__dict__.update(geometry)
            dense = self.dense(excluded)
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
            raise ValueError("Accessible densities must be finite and positive; reduce the timestep.")
        if self.has_excluded and torch.any(rho[self.excluded] != 0):
            raise ValueError("Excluded voxels must have exactly zero density.")

    def external(self):
        """Return the validated external potential, zero when none was given."""
        return torch.zeros_like(self.rho) if self.potential is None else self.potential


class EulerIntegrator:
    """Explicit first-order step for j = -M grad(mu), with shared face fluxes.

    ``model`` returns the full intrinsic beta-free-energy derivative. Mobility
    is physical, so its coefficient against grad(beta*mu) is ``M / beta``.
    Excluded voxels have zero density and block every adjoining face.
    """

    def __init__(self, model, mobility, dt):
        self.model = model
        self.mobility = mobility
        self.dt = dt
        self.last_beta_A = None
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
        self._grid = _Grid(data, self._grid, trusted=trusted)
        return self._grid

    def _model_output(self, data, derivatives):
        if getattr(self.model, "requires_dipole_density", False) or "dipole_density" in data:
            raise ValueError("Evolution currently supports species densities, not dipole density.")
        state = dict(data, rho=data["rho"].detach().clone())
        with _evaluation(self.model), torch.enable_grad():
            return self.model(
                state, compute_c1=derivatives, compute_c2=False, compute_ideal=True,
                compute_local_mu=False, compute_polarization_derivative=False,
                use_voxel_mask=False,
            )

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
            if grid.has_excluded:
                flux = flux.masked_fill(grid.blocked[axis], 0)
            rate -= (flux - torch.roll(flux, 1, axis)) / spacing
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

    def step(self, data):
        """Return one trial density; invalid steps raise instead of clipping."""
        grid = self._grid_for(data)
        rho = grid.rho.detach() + self.dt * self._rhs(data, grid)
        grid.check_density(rho)
        self._last_output = (rho, rho._version)
        return rho

    def relaxation_rates(self, data, reference_density=None):
        """Return the linear relaxation rate of every Fourier mode.

        The rates are the eigenvalues of the homogeneous reference operator
        ``k_B T q_h^2 M_ref H_+(k)`` (the semi-implicit stabilizer), shaped
        ``grid_size + (n_species,)``; ``1 / rates[rates > 0].min()`` is the
        slowest linear relaxation time. The reference is the field's species
        means unless ``reference_density`` is given.
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
    """Return the validated homogeneous reference species vector."""
    rho = grid.rho
    reference = (rho.mean(0) if reference_density is None else
                 torch.as_tensor(reference_density, dtype=rho.dtype, device=rho.device))
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


def _reference_operator(model, mobility, data, grid, reference, reference_curvature=None):
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
    return q2[..., None, None] * (matrices[0].to(positive.dtype) @ positive) / grid.beta


class SemiImplicitIntegrator(EulerIntegrator):
    """First-order IMEX step with a frozen homogeneous Fourier stabilizer.

    The full nonlinear response stays explicit. Only the positive part of the
    reference beta-free-energy curvature enters the implicit operator; negative
    physical curvature is not removed from the evolution. This FFT step requires
    a fully accessible periodic grid, unlike Euler's blocked-face update.
    """

    def __init__(self, model, mobility, dt, reference_density=None, reference_curvature=None):
        super().__init__(model, mobility, dt)
        self.reference_density = reference_density
        self.reference_curvature = reference_curvature
        self._operator = None
        self._propagator = None
        self._propagator_dt = None
        self._signature = None
        self._reference = None

    def step(self, data):
        grid = self._grid_for(data)
        if grid.has_excluded:
            raise ValueError("SemiImplicitIntegrator requires a fully accessible periodic grid.")
        signature = (grid.shape, tuple(grid.spacing.tolist()), grid.beta, grid.temperature,
                     grid.rho.shape[-1], grid.rho.dtype, grid.rho.device)
        reference = _reference_density(grid, self.reference_density)
        if self._operator is None:
            self._operator = _reference_operator(
                self.model, self.mobility, data, grid, reference, self.reference_curvature,
            )
            self._signature = signature
            self._reference = reference
        elif signature != self._signature or not torch.allclose(reference, self._reference, rtol=1e-6, atol=0.):
            raise ValueError("Grid, reference density or temperature changed; construct a new semi-implicit integrator.")
        if self._propagator_dt != self.dt:
            identity = torch.eye(grid.rho.shape[-1], dtype=self._operator.dtype, device=self._operator.device)
            self._propagator = torch.linalg.inv(identity + self.dt * self._operator)
            self._propagator_dt = self.dt
        rho_hat = torch.fft.fftn(grid.dense(grid.rho.detach()), dim=(0, 1, 2))
        rate_hat = torch.fft.fftn(grid.dense(self._rhs(data, grid)), dim=(0, 1, 2))
        # (I + dt B) rho' = rho + dt [R + B rho]  is  rho' = rho + dt (I + dt B)^-1 R.
        updated = rho_hat + self.dt * (self._propagator @ rate_hat.unsqueeze(-1)).squeeze(-1)
        updated[0, 0, 0] = rho_hat[0, 0, 0]
        rho = grid.flat(torch.fft.ifftn(updated, dim=(0, 1, 2)).real)
        grid.check_density(rho)
        self._last_output = (rho, rho._version)
        return rho
