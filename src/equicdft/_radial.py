"""Radial functions and conditioning for Cartesian grid features."""

import math
from typing import Tuple

import torch
from torch import nn

from ._argument_checks import positive_integer


def gaussian_radial_values(
    squared_distances: torch.Tensor,
    radial_exponents: torch.Tensor,
    radial_centers: torch.Tensor,
    neighbor_mask: torch.Tensor,
) -> torch.Tensor:
    """Return unit-sum Gaussian weights with shape ``[J, N]``."""

    distances = torch.sqrt(squared_distances)
    centered_squared_distances = (
        squared_distances[:, None]
        - 2.0 * distances[:, None] * radial_centers[None, :]
        + radial_centers[None, :].square()
    ).clamp_min(0.0)
    radial_values = torch.exp(
        -centered_squared_distances * radial_exponents[None, :]
    )
    radial_values = radial_values * neighbor_mask[:, None]
    return radial_values / torch.clamp(
        torch.sum(radial_values, dim=0, keepdim=True),
        min=torch.finfo(radial_values.dtype).tiny,
    )


def bessel_radial_values(
    squared_distances: torch.Tensor,
    n_radial_functions: int,
    cutoff_grid: int,
    neighbor_mask: torch.Tensor,
) -> torch.Tensor:
    r"""Return fixed spherical-Bessel-like values with shape ``[J, N]``.

    For integer-grid radius ``r`` and cutoff ``R_c=cutoff_grid``, function
    ``n`` is

    ``sqrt(2/R_c) * sin(n*pi*r/R_c) / r``, for ``n=1,...,N``.

    The normalized ``torch.sinc`` expression evaluates the finite ``r=0``
    limit without a special division. Values at the inclusive stencil boundary
    are set exactly to zero.
    """

    n_radial_functions = positive_integer(
        n_radial_functions,
        "n_radial_functions",
    )
    cutoff_grid = positive_integer(cutoff_grid, "cutoff_grid")
    distances = torch.sqrt(squared_distances)
    modes = torch.arange(
        1,
        n_radial_functions + 1,
        dtype=distances.dtype,
        device=distances.device,
    )
    scaled_distances = distances[:, None] * modes[None, :] / float(
        cutoff_grid
    )
    frequencies = modes * math.pi / float(cutoff_grid)
    radial_values = (
        math.sqrt(2.0 / float(cutoff_grid))
        * frequencies[None, :]
        * torch.sinc(scaled_distances)
    )
    support = neighbor_mask & (distances < float(cutoff_grid))
    return radial_values * support[:, None]


def whiten_radial_cartesian_basis(
    radial_values: torch.Tensor,
    monomial_values: torch.Tensor,
    powers: torch.Tensor,
    neighbor_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Condition radial channels on the exact Cartesian stencil.

    A separate discrete Gram matrix is formed for every total Cartesian
    degree. The returned basis has shape ``[J, N, K]`` and uses neighborhood-
    average scaling. The returned eigenvalues have shape ``[P+1, N]`` and are
    useful for diagnostics.
    """

    if radial_values.ndim != 2:
        raise ValueError("radial_values must have shape [J, N]")
    if monomial_values.ndim != 2:
        raise ValueError("monomial_values must have shape [J, K]")
    if radial_values.shape[0] != monomial_values.shape[0]:
        raise ValueError("radial and monomial stencil sizes must match")
    if powers.ndim != 2 or powers.shape[0] != monomial_values.shape[1]:
        raise ValueError("powers must have shape [K, 3]")
    if powers.shape[1] != 3:
        raise ValueError("powers must have shape [K, 3]")
    if neighbor_mask.shape != radial_values.shape[:1]:
        raise ValueError("neighbor_mask must have shape [J]")

    n_active = int(neighbor_mask.sum().item())
    if n_active == 0:
        raise ValueError("Bessel radial basis has no active stencil points")

    primitive_basis = (
        radial_values[:, :, None] * monomial_values[:, None, :]
    )
    total_degrees = powers.sum(dim=-1)
    max_power = int(total_degrees.max().item())
    conditioned_blocks = []
    eigenvalue_blocks = []

    for degree in range(max_power + 1):
        component_mask = total_degrees == degree
        block = primitive_basis[:, :, component_mask]
        samples = block[neighbor_mask].permute(0, 2, 1).reshape(
            -1,
            radial_values.shape[1],
        )
        samples64 = samples.to(dtype=torch.float64)
        n_components = int(component_mask.sum().item())
        gram = samples64.transpose(0, 1).matmul(samples64)
        gram = gram / float(n_components)
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)

        largest = eigenvalues[-1]
        if largest <= 0.0:
            raise ValueError(
                "radial basis has a nonpositive Gram spectrum for "
                "Cartesian degree {}".format(degree)
            )
        relative_minimum = eigenvalues[0] / largest
        if relative_minimum <= 1.0e-10:
            raise ValueError(
                "radial basis is rank deficient for Cartesian degree {}: "
                "lambda_min/lambda_max={:.3e}".format(
                    degree,
                    relative_minimum.item(),
                )
            )

        inverse_sqrt = eigenvectors.matmul(
            torch.diag(eigenvalues.rsqrt())
        ).matmul(eigenvectors.transpose(0, 1))
        inverse_sqrt = inverse_sqrt.to(
            device=block.device,
            dtype=block.dtype,
        )
        conditioned = torch.einsum("jnk,nm->jmk", block, inverse_sqrt)
        conditioned_blocks.append((component_mask, conditioned))
        eigenvalue_blocks.append(eigenvalues.to(dtype=block.dtype))

    result = torch.empty_like(primitive_basis)
    for component_mask, conditioned in conditioned_blocks:
        result[:, :, component_mask] = conditioned
    result = result / math.sqrt(float(n_active))
    return result, torch.stack(eigenvalue_blocks)


class _RadialTransform(nn.Module):
    """Bias-free radial mixing shared within each Cartesian degree."""

    def __init__(
        self,
        max_power: int,
        n_radial_functions: int,
        n_radial_channels: int,
    ) -> None:
        super().__init__()

        self.max_power = int(max_power)
        self.n_radial_functions = positive_integer(
            n_radial_functions,
            "n_radial_functions",
        )
        self.n_radial_channels = positive_integer(
            n_radial_channels,
            "n_radial_channels",
        )
        if self.n_radial_channels > self.n_radial_functions:
            raise ValueError(
                "n_radial_channels must not exceed "
                "n_radial_functions"
            )

        initial = torch.zeros(
            self.max_power + 1,
            self.n_radial_functions,
            self.n_radial_channels,
            dtype=torch.get_default_dtype(),
        )
        diagonal = torch.arange(self.n_radial_channels)
        initial[:, diagonal, diagonal] = 1.0
        self.weight = nn.Parameter(initial)

    def forward(
        self,
        basis: torch.Tensor,
        powers: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``[J, M, K]`` from primitive ``[J, N, K]`` values."""

        if basis.ndim != 3:
            raise ValueError("basis must have shape [J, N, K]")
        if basis.shape[1] != self.n_radial_functions:
            raise ValueError("basis radial-function count does not match")
        if powers.ndim != 2 or powers.shape != (basis.shape[2], 3):
            raise ValueError("powers must have shape [K, 3]")
        total_degrees = powers.sum(dim=-1)
        if int(total_degrees.max().item()) > self.max_power:
            raise ValueError("powers exceed the configured max_power")
        weights = self.weight.index_select(0, total_degrees)
        return torch.einsum("jnk,knm->jmk", basis, weights)
