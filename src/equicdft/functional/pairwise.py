"""Finite-range isotropic pair contributions on periodic density grids."""

import math
from typing import Any, Dict, Optional, Sequence

import torch

from .._validation import boolean, optional_positive_integer, positive_integer
from ._pairs import symmetric_component_pairs
from .._config import Configurable, make_config, register
from ..geometry.grid import common_grid_size
from .._nn import build_mlp, validate_hidden_sizes
from .base import EnergyReadout, ReadoutContext
from ..geometry.stencil import make_stencil

__all__ = [
    "PairwiseReadout",
]


def _smooth_cutoff_envelope(distance_fraction: torch.Tensor) -> torch.Tensor:
    """Return a quintic cutoff with zero value and two derivatives at one."""

    x = distance_fraction
    polynomial = 1.0 - 10.0 * x.pow(3) + 15.0 * x.pow(4) - 6.0 * x.pow(5)
    return torch.where(
        (x >= 0.0) & (x < 1.0),
        polynomial,
        torch.zeros_like(x),
    )


@register
class PairwiseReadout(EnergyReadout, Configurable):
    r"""Learn isotropic finite-range interactions between density fields.

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

    With ``density_channels=M``, replace the physical convolution sources by
    ``f_a(g) = sum_i rho_i(g) A_ia(rho(g) / mean_density)``. A pointwise MLP
    supplies the ``n_types * M`` entries of ``A`` from every local component
    density; ``mean_density`` is the fixed model normalization, never a
    field's current spatial average. With ``density_temperature=True`` the
    same MLP also reads ``T / mean_T``, so ``A`` couples density and
    temperature instead of leaving all temperature dependence to the kernel.
    The distance MLP supplies a symmetric ``M * M`` kernel ``W_ab``,
    including the same cutoff envelope. The energy is then
    ``DeltaV**2 / 2 * sum_(g,q!=0,ab) f_a(g) W_ab(q,T) f_b(g+q)``.
    Equivalently, the physical endpoint kernel is
    ``K_ij(q; u,v,T) = sum_ab A_ia(u) W_ab(q,T) A_jb(v)``, which obeys
    ``K_ij(q; u,v,T) = K_ji(q; v,u,T)``. It depends on both endpoint densities
    without gathering voxel pairs, and is generally no longer quadratic in
    physical rho. All density derivatives include derivatives of ``A``.

    A density-aware readout with ``M >= n_types`` contains every quadratic
    readout of the same cutoff and distance network: :meth:`warm_start_from`
    and :meth:`from_quadratic` set it up to reproduce a trained quadratic
    readout exactly before fine-tuning.

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
        Only the distance network is zeroed; density features remain learnable.
    density_channels
        Optional positive number of nonlinear local density fields to
        convolve. ``None`` preserves the quadratic physical-component model,
        its parameter shapes and its saved configuration. No species count,
        charge, composition or global reference density is assumed.
    density_hidden_sizes
        Hidden widths of the local density MLP when ``density_channels`` is
        supplied. These are density features, not radial-basis channels.
    density_temperature
        If true, the local density MLP also receives the normalized
        temperature ``T / mean_T`` at every voxel. Requires
        ``density_channels``. The vacuum limit is unchanged, since every
        transformed field still carries a factor of physical rho.

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
        n_types: int,
        *,
        hidden_sizes: Sequence[int] = (32, 16),
        zero_init: bool = True,
        density_channels: Optional[int] = None,
        density_hidden_sizes: Sequence[int] = (16, 16),
        density_temperature: bool = False,
    ) -> None:
        super().__init__()

        self.cutoff_grid = positive_integer(cutoff_grid, "cutoff_grid")
        if self.cutoff_grid < 2:
            raise ValueError("cutoff_grid must be at least two")
        self.n_types = positive_integer(n_types, "n_types")
        self.type_pairs = symmetric_component_pairs(self.n_types)
        self.n_type_pairs = len(self.type_pairs)
        self.density_channels = optional_positive_integer(
            density_channels, "density_channels",
        )
        self.n_kernel_channels = self.density_channels or self.n_types
        self.kernel_pairs = symmetric_component_pairs(self.n_kernel_channels)
        self.n_kernel_pairs = len(self.kernel_pairs)
        self.density_hidden_sizes = validate_hidden_sizes(density_hidden_sizes)
        self.density_temperature = boolean(
            density_temperature, "density_temperature",
        )
        if self.density_temperature and self.density_channels is None:
            raise ValueError("density_temperature requires density_channels")
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
        self.hidden_sizes = validate_hidden_sizes(hidden_sizes)
        self.zero_init = zero_init
        self.mlp = build_mlp(
            input_size=2,
            hidden_sizes=self.hidden_sizes,
            output_size=self.n_kernel_pairs,
            zero_init=self.zero_init,
        )
        self.density_mlp = None if self.density_channels is None else build_mlp(
            input_size=self.n_types + int(self.density_temperature),
            hidden_sizes=self.density_hidden_sizes,
            output_size=self.n_types * self.density_channels,
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

    def to_config(self) -> Dict[str, Any]:
        """Return the constructor arguments describing this readout."""

        config = make_config(
            self,
            cutoff_grid=self.cutoff_grid,
            n_types=self.n_types,
            hidden_sizes=self.hidden_sizes,
            zero_init=self.zero_init,
        )
        if self.density_channels is not None:
            config.update(
                density_channels=self.density_channels,
                density_hidden_sizes=list(self.density_hidden_sizes),
            )
            if self.density_temperature:
                config["density_temperature"] = True
        return config

    @classmethod
    def from_quadratic(
        cls,
        quadratic: "PairwiseReadout",
        density_channels: int,
        *,
        density_hidden_sizes: Sequence[int] = (16, 16),
        density_temperature: bool = False,
    ) -> "PairwiseReadout":
        """Build a density-aware readout that starts as a trained quadratic one.

        The result takes the cutoff, species count, distance network widths
        and ``zero_init`` flag of ``quadratic``, lives on its device with its
        floating dtype, and is initialized by :meth:`warm_start_from`, so it
        evaluates to the same energies before fine-tuning. Replacing the
        readout inside a model, ``model.readout[k] = ...``, keeps every other
        parameter; the model must supply ``mean_density``.
        """

        if not isinstance(quadratic, PairwiseReadout):
            raise TypeError("quadratic must be a PairwiseReadout")
        readout = cls(
            quadratic.cutoff_grid,
            quadratic.n_types,
            hidden_sizes=quadratic.hidden_sizes,
            zero_init=quadratic.zero_init,
            density_channels=density_channels,
            density_hidden_sizes=density_hidden_sizes,
            density_temperature=density_temperature,
        )
        reference = quadratic.mlp[-1].weight
        readout.to(dtype=reference.dtype, device=reference.device)
        return readout.warm_start_from(quadratic)

    def warm_start_from(
        self,
        quadratic: "PairwiseReadout",
    ) -> "PairwiseReadout":
        """Set this readout to reproduce a quadratic readout exactly.

        ``quadratic`` must use ``density_channels=None`` with this readout's
        ``cutoff_grid``, ``n_types`` and ``hidden_sizes``, and this readout
        needs ``density_channels >= n_types``. The distance network is
        copied: physical pair ``(i, j)`` becomes feature pair ``(i, j)`` and
        every kernel involving a feature ``a >= n_types`` is set to zero. The
        last density-network layer is set to ``A_ia = delta_ia`` for the first
        ``n_types`` features, independent of density and temperature, so the
        convolution sources equal physical rho and energies, chemical
        potentials and ``c2`` match ``quadratic`` to rounding.

        Features beyond ``n_types`` keep their random initialization. They
        contribute nothing while their kernels are zero, but they give those
        kernels a nonzero gradient; with both factors zero neither would
        ever leave zero. Returns ``self``.
        """

        if not isinstance(quadratic, PairwiseReadout):
            raise TypeError("quadratic must be a PairwiseReadout")
        if quadratic.density_channels is not None:
            raise ValueError(
                "warm_start_from copies a quadratic readout; the source "
                "has density_channels"
            )
        if self.density_channels is None:
            raise ValueError(
                "warm_start_from requires density_channels; a quadratic "
                "readout loads the state dictionary directly"
            )
        if self.density_channels < self.n_types:
            raise ValueError(
                "density_channels must be at least n_types to warm start "
                "from a quadratic readout"
            )
        for name in ("cutoff_grid", "n_types", "hidden_sizes"):
            if getattr(self, name) != getattr(quadratic, name):
                raise ValueError(
                    "{} must match the quadratic readout".format(name)
                )

        kernel_index = {pair: index for index, pair in enumerate(self.kernel_pairs)}
        rows = [kernel_index[pair] for pair in quadratic.type_pairs]
        with torch.no_grad():
            # The shell distances and envelope are fixed by cutoff_grid but
            # carry the rounding of the dtype they were built in; copying
            # them evaluates the source's kernel, not a re-rounded one.
            for name in ("normalized_shell_distances", "shell_envelope"):
                getattr(self, name).copy_(getattr(quadratic, name))
            for target, source in zip(
                self.mlp[:-1].parameters(), quadratic.mlp[:-1].parameters(),
            ):
                target.copy_(source)
            final, source = self.mlp[-1], quadratic.mlp[-1]
            final.weight.zero_()
            final.bias.zero_()
            final.weight[rows] = source.weight.to(final.weight)
            final.bias[rows] = source.bias.to(final.bias)
            # Output index i * M + a holds A_ia; the first n_types features
            # become the identity, the rest keep their current values.
            output = self.density_mlp[-1]
            weight = output.weight.view(self.n_types, self.density_channels, -1)
            bias = output.bias.view(self.n_types, self.density_channels)
            weight[:, : self.n_types].zero_()
            bias[:, : self.n_types].copy_(
                torch.eye(self.n_types, dtype=bias.dtype, device=bias.device)
            )
        return self

    def density_fields(self, context: ReadoutContext) -> torch.Tensor:
        """Return physical rho, or nonlinear endpoint fields with zero vacuum.

        The MLP reads all local species densities, plus the normalized
        temperature with ``density_temperature``. Multiplication by physical
        rho makes every transformed field zero in an empty voxel without
        clipping, dividing by local rho, or discarding derivative terms. In
        the quadratic case the result is ``context.rho`` itself, not a copy.
        """

        rho = context.rho
        if rho.shape[-1] != self.n_types:
            raise ValueError(
                "rho must have shape [..., n_grid, n_types] with the "
                "configured n_types"
            )
        if self.density_mlp is None:
            return rho
        inputs = (
            context.pointwise_state() if self.density_temperature
            else context.normalized_rho()
        )
        coefficients = self.density_mlp(inputs).reshape(
            *rho.shape, self.density_channels,
        )
        return torch.sum(rho[..., None] * coefficients, dim=-2)

    def shell_kernel_values(
        self,
        normalized_temperature: torch.Tensor,
    ) -> torch.Tensor:
        """Return envelope-weighted kernels for each radial shell and pair.

        The result has shape ``[..., n_shells, n_kernel_pairs]``. Kernel
        channels are physical species by default, density features otherwise.
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
            self.n_kernel_pairs,
        ).permute(0, 2, 1)
        scatter_index = flat_index.reshape(1, 1, self.n_offsets).expand(
            n_fields,
            self.n_kernel_pairs,
            self.n_offsets,
        )
        kernel = torch.zeros(
            (n_fields, self.n_kernel_pairs, nx * ny * nz),
            dtype=source.dtype,
            device=source.device,
        ).scatter_add(2, scatter_index, source)
        return kernel.reshape(
            *leading_shape,
            self.n_kernel_pairs,
            nx,
            ny,
            nz,
        )

    def _periodic_convolution(
        self,
        rho: torch.Tensor,
        kernel: torch.Tensor,
    ) -> torch.Tensor:
        """Convolve physical or nonlinear density fields with symmetric kernels."""

        nx, ny, nz = kernel.shape[-3:]
        if nx * ny * nz != rho.shape[-2]:
            raise ValueError("grid_size product does not match rho n_grid")

        rho_grid = rho.reshape(
            *rho.shape[:-2],
            nx,
            ny,
            nz,
            self.n_kernel_channels,
        ).movedim(-1, -4)
        rho_hat = torch.fft.rfftn(rho_grid, dim=(-3, -2, -1))
        kernel_hat = torch.fft.rfftn(kernel, dim=(-3, -2, -1))
        rho_hat_by_type = rho_hat.unbind(dim=-4)
        kernel_hat_by_pair = kernel_hat.unbind(dim=-4)
        potential_hat = [
            torch.zeros_like(rho_hat_by_type[0])
            for _ in range(self.n_kernel_channels)
        ]

        for pair_index, (first, second) in enumerate(self.kernel_pairs):
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

    def energy(self, context: ReadoutContext) -> torch.Tensor:
        """Return the periodic pair contribution for a complete field."""

        grid_size = context.require("grid_size")
        fields = self.density_fields(context)
        kernel = self.real_space_kernel(
            context.normalized_temperature,
            grid_size,
        )
        potential = self._periodic_convolution(fields, kernel)
        volume_element = context.voxel_volume.to(fields)
        return (
            0.5
            * volume_element.square()
            * torch.sum(fields * potential, dim=(-2, -1))
        )
