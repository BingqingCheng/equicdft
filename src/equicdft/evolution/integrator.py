"""Deterministic, conservative timesteps for species density fields."""

import math
from contextlib import contextmanager

import torch

from ..geometry.grid import common_grid_size, _flat_grid_positions, grid_spacing_tensor
from .mobility import _validate_matrix


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


class _Grid:
    """One complete regular grid, retaining the input row order."""

    def __init__(self, data):
        self.rho = data["rho"]
        if self.rho.ndim != 2 or not self.rho.is_floating_point() or not self.rho.shape[1]:
            raise ValueError("Evolution requires rho shaped [n_grid, n_species].")
        self.shape = common_grid_size(data["grid_size"], torch.Size())
        self.indices, self.canonical = _flat_grid_positions(
            data["grid_positions"], self.shape, torch.Size(), self.rho.device
        )
        self.indices = self.indices[0]
        if self.rho.shape[0] != math.prod(self.shape):
            raise ValueError("rho must cover the complete regular grid.")
        self.spacing = grid_spacing_tensor(
            data["grid_spacing"], dtype=self.rho.dtype, device=self.rho.device
        )
        self.volume = self.spacing.prod()
        self.beta = _positive_scalar(data["beta"], "beta")
        self.temperature = _positive_scalar(data["temperature"], "temperature")
        self.excluded = torch.as_tensor(
            data.get("excluded_mask", torch.zeros(self.rho.shape[0], dtype=torch.bool)),
            device=self.rho.device,
        )
        if self.excluded.dtype != torch.bool or self.excluded.shape != self.rho.shape[:1]:
            raise ValueError("excluded_mask must be a boolean [n_grid] tensor.")
        if self.excluded.all():
            raise ValueError("At least one voxel must be accessible.")
        self.check_density(self.rho)

    def dense(self, values):
        if not self.canonical:
            values = torch.zeros_like(values).index_copy(0, self.indices, values)
        return values.reshape(self.shape + values.shape[1:])

    def flat(self, values):
        values = values.reshape((-1,) + values.shape[3:])
        return values if self.canonical else values[self.indices]

    def check_density(self, rho):
        if not torch.isfinite(rho).all() or torch.any(rho[~self.excluded] <= 0):
            raise ValueError("Accessible densities must be finite and positive; reduce the timestep.")
        if torch.any(rho[self.excluded] != 0):
            raise ValueError("Excluded voxels must have exactly zero density.")

    def external(self, data):
        potential = torch.as_tensor(
            data.get("V_ext", torch.zeros_like(self.rho)),
            dtype=self.rho.dtype, device=self.rho.device,
        )
        if potential.shape != self.rho.shape or not torch.isfinite(potential).all():
            raise ValueError("V_ext must be finite and have the same shape as rho.")
        return potential


class EulerIntegrator:
    """Explicit first-order step for j = -M grad(mu), with shared face fluxes.

    ``model`` returns the full intrinsic beta-free-energy derivative. Mobility
    is physical, so its coefficient against grad(beta*mu) is ``M / beta``.
    Excluded voxels have zero density and block every adjoining face.
    """

    def __init__(self, model, mobility, dt):
        self.model = model
        self.mobility = mobility
        self.dt = _positive_scalar(dt, "dt")

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

    def _rhs(self, data, grid):
        output = self._model_output(data, derivatives=True)
        derivative = output["density_derivative"].detach()
        if derivative.shape != grid.rho.shape or not torch.isfinite(derivative).all():
            raise ValueError("The model must return a finite full density_derivative.")
        u = grid.dense(derivative + grid.beta * grid.external(data))
        with _evaluation(self.mobility), torch.no_grad():
            matrices = self.mobility(data).detach()
        if (matrices.shape != grid.rho.shape + (grid.rho.shape[-1],)
                or matrices.dtype != grid.rho.dtype or matrices.device != grid.rho.device
                or not torch.isfinite(matrices).all()):
            raise ValueError("Mobility must return finite matrices matching rho's shape, dtype and device.")
        _validate_matrix(matrices)
        matrices = grid.dense(matrices)
        excluded = grid.dense(grid.excluded)
        rate = torch.zeros_like(u)
        for axis, spacing in enumerate(grid.spacing):
            gradient = (torch.roll(u, -1, axis) - u) / spacing
            face_mobility = (matrices + torch.roll(matrices, -1, axis)) / 2
            flux = -(face_mobility @ gradient.unsqueeze(-1)).squeeze(-1) / grid.beta
            flux = flux.masked_fill((excluded | torch.roll(excluded, -1, axis))[..., None], 0)
            rate -= (flux - torch.roll(flux, 1, axis)) / spacing
        return grid.flat(rate).detach()

    def rhs(self, data):
        """Return drho/dt without altering data or retaining an autograd graph."""
        return self._rhs(data, _Grid(data))

    def beta_A(self, data):
        """Return intrinsic beta F plus the static external-potential energy."""
        grid = _Grid(data)
        intrinsic = self._model_output(data, derivatives=False)["beta_F"].sum()
        result = intrinsic + grid.beta * grid.volume * (grid.rho * grid.external(data)).sum()
        if not torch.isfinite(result):
            raise ValueError("The model free energy must be finite.")
        return result.detach()

    def step(self, data):
        """Return one trial density; invalid steps raise instead of clipping."""
        grid = _Grid(data)
        rho = grid.rho.detach() + self.dt * self._rhs(data, grid)
        grid.check_density(rho)
        return rho.detach()


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
        self._signature = None

    def _curvature(self, data, grid):
        """FFT of p Hessian rows, not an n_grid-by-n_grid dense Hessian."""
        if getattr(self.model, "requires_dipole_density", False) or "dipole_density" in data:
            raise ValueError("Evolution currently supports species densities, not dipole density.")
        rho = data["rho"].detach().clone().requires_grad_(True)
        state = dict(data, rho=rho)
        anchor = int(torch.nonzero(grid.indices == 0)[0, 0])
        with _evaluation(self.model), torch.enable_grad():
            energy = self.model(
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

    def _build_operator(self, data, grid):
        n_species = grid.rho.shape[-1]
        reference = (grid.rho.mean(0) if self.reference_density is None else
                     torch.as_tensor(self.reference_density, dtype=grid.rho.dtype, device=grid.rho.device))
        if reference.shape != (n_species,) or not torch.isfinite(reference).all() or torch.any(reference <= 0):
            raise ValueError("reference_density must be a finite positive species vector.")
        reference_data = dict(data, rho=reference.detach().expand_as(grid.rho).clone())
        with _evaluation(self.mobility), torch.no_grad():
            mobility = self.mobility(reference_data).detach()
        if (mobility.shape != grid.rho.shape + (n_species,)
                or mobility.dtype != grid.rho.dtype or mobility.device != grid.rho.device
                or not torch.isfinite(mobility).all()):
            raise ValueError("Reference mobility must provide finite matrices matching rho's shape, dtype and device.")
        _validate_matrix(mobility)
        if not torch.allclose(mobility, mobility[0].expand_as(mobility)):
            raise ValueError("Reference mobility must be uniform on a homogeneous field.")
        if self.reference_curvature is None:
            curvature = self._curvature(reference_data, grid)
        else:
            curvature = torch.as_tensor(self.reference_curvature, device=grid.rho.device)
            curvature = curvature.to(torch.complex128 if grid.rho.dtype == torch.float64 else torch.complex64)
            if curvature.shape == (n_species, n_species):
                curvature = curvature.expand(grid.shape + (n_species, n_species))
            if curvature.shape != grid.shape + (n_species, n_species):
                raise ValueError("reference_curvature must be a species matrix or full Fourier matrices.")
        if not torch.isfinite(curvature).all():
            raise ValueError("Reference curvature must be finite.")
        adjoint = curvature.transpose(-1, -2).conj()
        if self.reference_curvature is not None and not torch.allclose(curvature, adjoint, rtol=1e-5, atol=1e-7):
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
        self._operator = q2[..., None, None] * (mobility[0].to(positive.dtype) @ positive) / grid.beta

    def step(self, data):
        grid = _Grid(data)
        if grid.excluded.any():
            raise ValueError("SemiImplicitIntegrator requires a fully accessible periodic grid.")
        signature = (grid.shape, tuple(grid.spacing.tolist()), grid.beta, grid.temperature,
                     grid.rho.shape[-1], grid.rho.dtype, grid.rho.device)
        if self._operator is None:
            self._build_operator(data, grid)
            self._signature = signature
        elif signature != self._signature:
            raise ValueError("Grid or temperature changed; construct a new semi-implicit integrator.")
        rho_hat = torch.fft.fftn(grid.dense(grid.rho.detach()), dim=(0, 1, 2))
        rate_hat = torch.fft.fftn(grid.dense(self._rhs(data, grid)), dim=(0, 1, 2))
        identity = torch.eye(grid.rho.shape[-1], dtype=rho_hat.dtype, device=rho_hat.device)
        right = rho_hat + self.dt * (rate_hat + (self._operator @ rho_hat.unsqueeze(-1)).squeeze(-1))
        updated = torch.linalg.solve(identity + self.dt * self._operator, right.unsqueeze(-1)).squeeze(-1)
        updated[0, 0, 0] = rho_hat[0, 0, 0]
        rho = grid.flat(torch.fft.ifftn(updated, dim=(0, 1, 2)).real)
        grid.check_density(rho)
        return rho.detach()
