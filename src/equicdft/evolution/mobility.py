"""Local species mobilities for conserved density evolution.

The physical mobility is defined by ``j = -M grad(mu)``. Each call returns
a matrix at every voxel, with shape ``rho.shape + (n_species,)``.
"""

from abc import ABC, abstractmethod
import math

import torch
from torch import nn


def _as_float(value, *, device=None, dtype=None):
    value = torch.as_tensor(value, device=device)
    if value.is_complex() or value.dtype == torch.bool:
        raise ValueError("Mobility must contain real numerical values.")
    if dtype is not None:
        value = value.to(dtype=dtype)
    elif not value.is_floating_point():
        value = value.to(dtype=torch.get_default_dtype())
    if not torch.isfinite(value).all():
        raise ValueError("Mobility must be finite.")
    return value


def _validate_diagonal(value):
    if torch.any(value < 0):
        raise ValueError("Diagonal mobility must be nonnegative.")


def _validate_matrix(value):
    # Only roundoff-sized violations are allowed; the tolerance scales with M.
    detached = value.detach()
    scale = detached.abs().amax(dim=(-2, -1))
    tolerance = 32 * value.shape[-1] * torch.finfo(value.dtype).eps * scale
    asymmetry = (detached - detached.transpose(-2, -1)).abs().amax(dim=(-2, -1))
    if torch.any(asymmetry > tolerance):
        raise ValueError("Mobility matrices must be symmetric.")
    checked = detached if value.dtype in (torch.float32, torch.float64) else detached.float()
    eigenvalues = torch.linalg.eigvalsh((checked + checked.transpose(-2, -1)) / 2)
    if torch.any(eigenvalues[..., 0] < -tolerance):
        raise ValueError("Mobility matrices must be positive semidefinite.")


def _density(data):
    rho = data["rho"]
    if not rho.is_floating_point() or rho.ndim < 2 or rho.shape[-1] == 0:
        raise ValueError("rho must be a floating tensor with voxel and species axes.")
    return rho


def _shared_matrix(value, rho):
    n_species = rho.shape[-1]
    if value.ndim == 0:
        _validate_diagonal(value)
        matrix = value * torch.eye(n_species, device=rho.device, dtype=rho.dtype)
    elif value.shape == (n_species,):
        _validate_diagonal(value)
        matrix = torch.diag(value)
    elif value.shape == (n_species, n_species):
        _validate_matrix(value)
        matrix = value
    else:
        raise ValueError("Shared mobility must be a scalar, species vector, or species matrix.")
    return matrix.expand(rho.shape + (n_species,))


class Mobility(nn.Module, ABC):
    """Return physical species mobility matrices from the current field data.

    ``validated`` declares that ``forward`` already guarantees finite,
    symmetric, positive semidefinite matrices; the integrators then skip
    their own per-step eigenvalue check. Custom modules leave it ``False``.
    """

    validated = False

    @abstractmethod
    def forward(self, data):
        """Return M with shape ``[..., n_grid, n_species, n_species]``."""
        raise NotImplementedError


class ConstantMobility(Mobility):
    """A shared scalar, diagonal species vector, or symmetric PSD matrix.

    A scalar means the same diagonal mobility for every species. A species
    vector specifies independent diagonal mobilities. A matrix also permits
    coupling between species; its off-diagonal entries may be negative.
    """

    validated = True

    def __init__(self, value):
        super().__init__()
        value = _as_float(value).detach().clone()
        if value.ndim <= 1:
            if value.numel() == 0:
                raise ValueError("Mobility must not be empty.")
            _validate_diagonal(value)
        elif value.ndim == 2 and value.shape[0] == value.shape[1] and value.numel():
            _validate_matrix(value)
        else:
            raise ValueError("Constant mobility must be a scalar, vector, or square matrix.")
        self.register_buffer("value", value)

    @classmethod
    def from_estimate(cls, value, *, tolerance=1e-6):
        """Build from a measured matrix: symmetrize and project onto PSD matrices.

        A mobility matrix estimated from simulation data is symmetric and
        positive semidefinite only up to statistical noise, which the strict
        constructor rejects. The adjustment is bounded by ``tolerance`` times
        the largest entry, so a genuinely asymmetric or indefinite estimate
        still raises instead of being silently repaired.
        """
        scalar = torch.as_tensor(tolerance)
        if scalar.numel() != 1 or scalar.is_complex() or scalar.dtype == torch.bool:
            raise ValueError("tolerance must be a finite nonnegative scalar.")
        tolerance = float(torch.as_tensor(tolerance, dtype=torch.float64).detach())
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance must be a finite nonnegative scalar.")
        estimate = _as_float(value).detach().clone()
        if estimate.ndim != 2 or estimate.shape[0] != estimate.shape[1] or not estimate.numel():
            return cls(estimate)
        eigenvalues, vectors = torch.linalg.eigh((estimate + estimate.T) / 2)
        projected = (vectors * eigenvalues.clamp_min(0)) @ vectors.T
        projected = (projected + projected.T) / 2
        if (projected - estimate).abs().max() > tolerance * estimate.abs().max():
            raise ValueError("Estimate is not a symmetric positive semidefinite matrix within the tolerance.")
        return cls(projected)

    def forward(self, data):
        rho = _density(data)
        value = self.value.to(device=rho.device, dtype=rho.dtype)
        return _shared_matrix(value, rho)


class LocalMobility(Mobility):
    """Evaluate ``function(rho, temperature)`` on the live density field.

    The function receives all species densities at each voxel. It can return
    diagonal coefficients shaped like ``rho``, full matrices shaped like
    ``rho.shape + (n_species,)``, or a shared scalar, vector, or matrix.
    Local shapes take precedence when a shape also matches a shared matrix.
    An ``nn.Module`` function is registered as a child module normally.
    """

    validated = True

    def __init__(self, function):
        super().__init__()
        if not callable(function):
            raise TypeError("Local mobility requires a callable function(rho, temperature).")
        self.function = function

    def forward(self, data):
        rho = _density(data)
        value = _as_float(
            self.function(rho, data["temperature"]), device=rho.device, dtype=rho.dtype
        )
        if value.shape == rho.shape:
            _validate_diagonal(value)
            return torch.diag_embed(value)
        if value.shape == rho.shape + (rho.shape[-1],):
            _validate_matrix(value)
            return value
        return _shared_matrix(value, rho)
