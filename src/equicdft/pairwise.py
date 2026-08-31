"""Finite-range isotropic pair contributions on periodic density grids."""

import math
from typing import Dict, Sequence

import torch

from ._argument_checks import boolean, positive_integer
from ._grid import common_grid_size
from ._nn import build_mlp
from .energy import EnergyReadout
from .stencil import make_stencil


def _smooth_cutoff_envelope(distance_fraction: torch.Tensor) -> torch.Tensor:
    """Return a quintic cutoff with zero value and two derivatives at one."""

    x = distance_fraction
    polynomial = 1.0 - 10.0 * x.pow(3) + 15.0 * x.pow(4) - 6.0 * x.pow(5)
    return torch.where(
        (x >= 0.0) & (x < 1.0),
        polynomial,
        torch.zeros_like(x),
    )


class PairwiseReadout(EnergyReadout):
    r"""Learn an isotropic finite-range quadratic density functional.

    For periodic grid displacement ``q`` and component pair ``ij``, the
    readout constructs

    ``K_ij(q, T) = s(|q| / cutoff_grid)``
    ``              * NN_ij(|q| / cutoff_grid, T / mean_T)``

    and evaluates

    ``E_pair = DeltaV**2 / 2 * sum_(g,q!=0,ij)``
    ``         rho_i(g) K_ij(q,T) rho_j(g+q)``.

    The neural network is evaluated once per distinct radial shell. The full
    pair sum is then evaluated by periodic FFT convolution rather than by
    materializing all ``[n_grid, n_pair_neighbors, n_types]`` densities.

    ``q=0`` is absent for every component pair, while the homogeneous Fourier
    mode is retained. The output follows the free-energy convention of the
    containing :class:`equicdft.model.GridCACEModel`.

    Parameters
    ----------
    cutoff_grid
        Spherical cutoff radius in integer grid steps. Pair displacements
        satisfy ``0 < |q| < cutoff_grid``. This cutoff is independent of the
        CACE descriptor cutoff.
    n_types
        Number of physical density components. The network returns one kernel
        value for every unique symmetric component pair ``i <= j``.
    hidden_sizes
        Width of each distance-network hidden layer. An empty sequence gives
        a linear function of normalized distance and temperature.
    zero_init
        If true, initialize the final network layer to zero, so attaching this
        readout leaves an existing model unchanged before fine-tuning.

    Notes
    -----
    Distances and the cutoff are measured in grid units, matching the CACE
    stencil convention. The network distance input is dimensionless. For an
    isotropic grid with spacing ``h``, the physical cutoff is
    ``cutoff_grid * h``.
    """

    def __init__(
        self,
        cutoff_grid: int,
        n_types: int = 1,
        hidden_sizes: Sequence[int] = (32, 16),
        zero_init: bool = True,
    ) -> None:
        super().__init__()

        self.cutoff_grid = positive_integer(cutoff_grid, "cutoff_grid")
        if self.cutoff_grid < 2:
            raise ValueError("cutoff_grid must be at least two")
        self.n_types = positive_integer(n_types, "n_types")
        self.type_pairs = tuple(
            (first, second)
            for first in range(self.n_types)
            for second in range(first, self.n_types)
        )
        self.n_type_pairs = len(self.type_pairs)
        zero_init = boolean(zero_init, "zero_init")

        offsets = torch.from_numpy(make_stencil(self.cutoff_grid))
        squared_distances = torch.sum(offsets.square(), dim=1)
        active = (squared_distances > 0) & (
            squared_distances < self.cutoff_grid**2
        )
        offsets = offsets[active]
        squared_distances = squared_distances[active]
        if offsets.shape[0] == 0:
            raise ValueError("cutoff_grid contains no nonzero pair offsets")

        shell_squared_distances, offset_shell_index = torch.unique(
            squared_distances,
            sorted=True,
            return_inverse=True,
        )
        normalized_shell_distances = torch.sqrt(
            shell_squared_distances.to(dtype=torch.get_default_dtype())
        ) / float(self.cutoff_grid)
        shell_envelope = _smooth_cutoff_envelope(
            normalized_shell_distances
        )

        self.n_offsets = int(offsets.shape[0])
        self.n_shells = int(shell_squared_distances.shape[0])
        self.mlp = build_mlp(
            input_size=2,
            hidden_sizes=hidden_sizes,
            output_size=self.n_type_pairs,
            zero_init=zero_init,
        )
        self.register_buffer("offsets", offsets, persistent=False)
        self.register_buffer(
            "offset_shell_index",
            offset_shell_index,
            persistent=False,
        )
        self.register_buffer(
            "normalized_shell_distances",
            normalized_shell_distances,
            persistent=False,
        )
        self.register_buffer(
            "shell_envelope",
            shell_envelope,
            persistent=False,
        )

    def shell_kernel_values(
        self,
        normalized_temperature: torch.Tensor,
    ) -> torch.Tensor:
        """Return envelope-weighted kernels for each radial shell and pair.

        The result has shape ``[..., n_shells, n_type_pairs]``.
        """

        if not torch.is_tensor(normalized_temperature):
            raise TypeError("normalized_temperature must be a torch.Tensor")
        if not torch.all(torch.isfinite(normalized_temperature)).item():
            raise ValueError("normalized_temperature must be finite")

        distances = self.normalized_shell_distances.to(
            normalized_temperature
        )
        leading_shape = normalized_temperature.shape
        distance_feature = distances.reshape(
            *((1,) * len(leading_shape)),
            self.n_shells,
        ).expand(*leading_shape, self.n_shells)
        temperature_feature = normalized_temperature[..., None].expand_as(
            distance_feature
        )
        features = torch.stack(
            (distance_feature, temperature_feature),
            dim=-1,
        )
        envelope = self.shell_envelope.to(normalized_temperature).reshape(
            *((1,) * len(leading_shape)),
            self.n_shells,
            1,
        )
        return envelope * self.mlp(features)

    def real_space_kernel(
        self,
        normalized_temperature: torch.Tensor,
        grid_size: torch.Tensor,
    ) -> torch.Tensor:
        """Return periodic kernels shaped ``[..., pair, nx, ny, nz]``."""

        leading_shape = normalized_temperature.shape
        nx, ny, nz = common_grid_size(grid_size, leading_shape)
        if 2 * self.cutoff_grid > min(nx, ny, nz):
            raise ValueError(
                "pair cutoff must not exceed half the smallest grid size"
            )

        shell_values = self.shell_kernel_values(normalized_temperature)
        offset_values = shell_values.index_select(
            -2,
            self.offset_shell_index.to(device=shell_values.device),
        )
        offsets = self.offsets.to(device=shell_values.device)
        wrapped = torch.remainder(
            offsets,
            offsets.new_tensor((nx, ny, nz)),
        )
        flat_index = (
            (wrapped[:, 0] * ny + wrapped[:, 1]) * nz + wrapped[:, 2]
        )

        n_fields = math.prod(leading_shape)
        source = offset_values.reshape(
            n_fields,
            self.n_offsets,
            self.n_type_pairs,
        ).permute(0, 2, 1)
        scatter_index = flat_index.reshape(1, 1, self.n_offsets).expand(
            n_fields,
            self.n_type_pairs,
            self.n_offsets,
        )
        kernel = torch.zeros(
            (n_fields, self.n_type_pairs, nx * ny * nz),
            dtype=source.dtype,
            device=source.device,
        ).scatter_add(2, scatter_index, source)
        return kernel.reshape(
            *leading_shape,
            self.n_type_pairs,
            nx,
            ny,
            nz,
        )

    def _periodic_convolution(
        self,
        rho: torch.Tensor,
        kernel: torch.Tensor,
    ) -> torch.Tensor:
        """Convolve component densities with symmetric periodic kernels."""

        nx, ny, nz = kernel.shape[-3:]
        if nx * ny * nz != rho.shape[-2]:
            raise ValueError("grid_size product does not match rho n_grid")

        rho_grid = rho.reshape(
            *rho.shape[:-2],
            nx,
            ny,
            nz,
            self.n_types,
        ).movedim(-1, -4)
        rho_hat = torch.fft.rfftn(rho_grid, dim=(-3, -2, -1))
        kernel_hat = torch.fft.rfftn(kernel, dim=(-3, -2, -1))
        rho_hat_by_type = rho_hat.unbind(dim=-4)
        kernel_hat_by_pair = kernel_hat.unbind(dim=-4)
        potential_hat = [
            torch.zeros_like(rho_hat_by_type[0])
            for _ in range(self.n_types)
        ]

        for pair_index, (first, second) in enumerate(self.type_pairs):
            pair_kernel = kernel_hat_by_pair[pair_index]
            potential_hat[first] = (
                potential_hat[first]
                + pair_kernel * rho_hat_by_type[second]
            )
            if first != second:
                potential_hat[second] = (
                    potential_hat[second]
                    + pair_kernel * rho_hat_by_type[first]
                )

        return torch.fft.irfftn(
            torch.stack(potential_hat, dim=-4),
            s=(nx, ny, nz),
            dim=(-3, -2, -1),
        ).movedim(-4, -1).reshape_as(rho)

    def energy(self, context: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the periodic pair contribution for a complete field."""

        if "grid_size" not in context:
            raise KeyError("pairwise evaluation requires data['grid_size']")
        rho = context["rho"]
        if rho.ndim < 2 or rho.shape[-1] != self.n_types:
            raise ValueError(
                "rho must have shape [..., n_grid, n_types] with the "
                "configured n_types"
            )
        normalized_temperature = context["normalized_temperature"]
        if normalized_temperature.shape != rho.shape[:-2]:
            raise ValueError(
                "normalized_temperature must match rho's leading shape"
            )
        kernel = self.real_space_kernel(
            normalized_temperature,
            context["grid_size"],
        )
        potential = self._periodic_convolution(rho, kernel)
        volume_element = context["voxel_volume"].to(rho)
        return (
            0.5
            * volume_element.square()
            * torch.sum(rho * potential, dim=(-2, -1))
        )
