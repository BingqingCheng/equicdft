"""Cartesian moment features for density fields on regular periodic grids.

The angular basis uses integer stencil coordinates. Gaussian radial functions
are evaluated using squared integer-grid distances without an additional
kernel normalization.
"""

import math
from numbers import Integral
from typing import Mapping, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F

from .stencil import make_stencil


def _common_grid_size(
    grid_size: torch.Tensor,
    leading_shape: torch.Size,
) -> tuple:
    """Return the single regular-grid shape shared by one dense batch."""

    sizes = torch.as_tensor(grid_size).detach().reshape(-1, 3)
    n_fields = math.prod(leading_shape) if leading_shape else 1
    if sizes.shape[0] not in (1, n_fields):
        raise ValueError("grid_size leading shape must match rho")
    rounded = torch.round(sizes).to(dtype=torch.long)
    if not torch.allclose(
        sizes.to(dtype=torch.float64),
        rounded.to(dtype=torch.float64),
    ):
        raise ValueError("grid_size values must be integers")
    if torch.any(rounded <= 0).item():
        raise ValueError("grid_size values must be positive")
    if not torch.all(rounded == rounded[0]).item():
        raise ValueError("all fields in one batch must share grid_size")
    return tuple(int(value) for value in rounded[0].cpu().tolist())


def _periodic_extend_3d(
    values: torch.Tensor,
    padding: int,
) -> torch.Tensor:
    """Periodically extend ``[batch, channels, nx, ny, nz]`` on every axis.

    Explicit modular indices, rather than ``F.pad(mode='circular')``, retain
    the repeated-image stencil semantics when ``padding`` is as large as a box
    dimension.
    """

    if padding == 0:
        return values
    extended = values
    for dimension in range(2, 5):
        size = values.shape[dimension]
        index = torch.arange(
            -padding,
            size + padding,
            device=values.device,
        ).remainder(size)
        extended = extended.index_select(dimension, index)
    return extended


def _make_powers(max_power: int) -> torch.Tensor:
    """Enumerate ``(a, b, c)`` by increasing total Cartesian power.

    For example, ``max_power=2`` gives the monomials
    ``1, x, y, z, x^2, xy, xz, y^2, yz, z^2``.
    """

    if isinstance(max_power, bool) or not isinstance(max_power, Integral):
        raise TypeError("max_power must be a nonnegative integer")
    max_power = int(max_power)
    if max_power < 0:
        raise ValueError("max_power must be a nonnegative integer")

    powers = []
    for total_power in range(max_power + 1):
        for x_power in range(total_power, -1, -1):
            remaining_power = total_power - x_power
            for y_power in range(remaining_power, -1, -1):
                z_power = remaining_power - y_power
                powers.append((x_power, y_power, z_power))
    return torch.tensor(powers, dtype=torch.long)


class CartesianAFeatures(nn.Module):
    """Compute normalized Cartesian ``A`` features by periodic convolution.

    For central grid point ``g``, Gaussian channel ``n``, monomial
    ``k = (a, b, c)``, and component ``t``, the module computes

    ``A[g, n, k, t] = sum_j (rho[g, j, t] / mean_density)``
    ``* w[n, j] * x_j^a * y_j^b * z_j^c``,

    where ``j`` runs over the canonically ordered integer stencil and
    ``w[n, j] = exp(-alpha[n] * r_grid[j]^2)``. Grid-volume quadrature is
    deliberately left to the free-energy readout rather than included in
    these local descriptors.

    Parameters
    ----------
    cutoff_grid
        Inclusive spherical cutoff in integer grid steps. The default is
        three, matching :class:`equicdft.data.GridData`.
    max_power
        Maximum total Cartesian power ``a + b + c``.
    n_radial_channels
        Number of Gaussian radial channels. Their positive initial decay
        coefficients are logarithmically spaced from 0.5 to 4.0 in inverse
        squared grid units.
    trainable_radial_exponents
        If ``True``, optimize the Gaussian decay coefficients. They are stored
        in logarithmic form so that the resulting ``alpha`` values stay
        positive. If ``False``, they remain fixed model buffers.
    n_types
        Number of physical density components in the input field.
    n_channels
        Number of latent output channels after learned component mixing.
        ``None`` retains the physical component channels. Mixing is available
        only when ``n_types`` is greater than one.
    mean_density
        Positive scalar used for scale-only density normalization. For fitting,
        set this to the precomputed mean density of the training split. It is
        stored as a fixed buffer; no mean is subtracted, so zero density
        remains zero.

    The input grid may have any positive rectangular shape. All fields in one
    dense batch must share that shape, while different batches may use
    different shapes.
    """

    def __init__(
        self,
        max_power: int,
        mean_density: Union[float, torch.Tensor],
        cutoff_grid: int = 3,
        n_radial_channels: int = 4,
        trainable_radial_exponents: bool = False,
        n_types: int = 1,
        n_channels: Optional[int] = None,
    ) -> None:
        super().__init__()

        if isinstance(n_radial_channels, bool) or not isinstance(
            n_radial_channels,
            Integral,
        ):
            raise TypeError("n_radial_channels must be a positive integer")
        n_radial_channels = int(n_radial_channels)
        if n_radial_channels < 1:
            raise ValueError("n_radial_channels must be a positive integer")
        if not isinstance(trainable_radial_exponents, bool):
            raise TypeError("trainable_radial_exponents must be a boolean")
        if isinstance(n_types, bool) or not isinstance(n_types, Integral):
            raise TypeError("n_types must be a positive integer")
        n_types = int(n_types)
        if n_types < 1:
            raise ValueError("n_types must be a positive integer")

        if n_channels is not None:
            if isinstance(n_channels, bool) or not isinstance(
                n_channels,
                Integral,
            ):
                raise TypeError("n_channels must be a positive integer or None")
            n_channels = int(n_channels)
            if n_channels < 1:
                raise ValueError("n_channels must be a positive integer or None")
            if n_types == 1:
                raise ValueError(
                    "channel mixing is disabled for one-component density fields"
                )

        mean_density_tensor = torch.as_tensor(
            mean_density,
            dtype=torch.get_default_dtype(),
        ).detach().clone()
        if mean_density_tensor.numel() != 1:
            raise ValueError("mean_density must be a positive scalar")
        mean_density_tensor = mean_density_tensor.reshape(())
        if not torch.isfinite(mean_density_tensor).item():
            raise ValueError("mean_density must be finite")
        if mean_density_tensor.item() <= 0.0:
            raise ValueError("mean_density must be positive")

        # The ordered stencil is retained as compact model geometry and is also
        # useful for inspecting the exact basis represented by the convolution.
        local_density_positions = torch.from_numpy(make_stencil(cutoff_grid))
        powers = _make_powers(max_power)

        # Evaluate every Cartesian monomial once on the shared integer
        # stencil. These values are independent of the central grid point and
        # of the density configuration.
        positions = local_density_positions.to(dtype=torch.get_default_dtype())
        squared_distances = torch.sum(positions**2, dim=1)
        monomial_values = torch.ones(
            (positions.shape[0], powers.shape[0]),
            dtype=torch.get_default_dtype(),
        )
        for axis in range(3):
            monomial_values = monomial_values * positions[:, axis, None].pow(
                powers[None, :, axis]
            )

        # Gaussian channel n uses the radial weight
        # R_n(q) = exp(-alpha_n * |q|**2).
        # For four radial channels, the alpha values are [0.5, 1, 2, 4].
        initial_radial_exponents = 2.0 ** torch.linspace(
            -1.0,
            2.0,
            steps=n_radial_channels,
            dtype=torch.get_default_dtype(),
        )
        log_radial_exponents = torch.log(initial_radial_exponents)
        if trainable_radial_exponents:
            self.log_radial_exponents = nn.Parameter(log_radial_exponents)
        else:
            self.register_buffer(
                "log_radial_exponents",
                log_radial_exponents,
            )

        self.cutoff_grid = int(cutoff_grid)
        self.max_power = int(max_power)
        self.n_radial_channels = n_radial_channels
        self.trainable_radial_exponents = trainable_radial_exponents
        self.n_types = n_types
        self.n_channels = n_channels
        self.n_output_channels = n_types if n_channels is None else n_channels
        self.channel_mixing = (
            None
            if n_channels is None
            else _AChannelMixing(
                n_types=n_types,
                n_channels=n_channels,
            )
        )
        self.register_buffer("local_density_positions", local_density_positions)
        self.register_buffer("squared_distances", squared_distances)
        self.register_buffer("powers", powers)
        self.register_buffer("monomial_values", monomial_values)
        self.register_buffer("mean_density", mean_density_tensor)

        # Build dense cubic coordinate tables once. Values outside the spherical
        # cutoff are masked to zero when constructing each convolution kernel.
        # PyTorch conv3d is a cross-correlation, so kernel coordinate q directly
        # multiplies rho(r + q) after symmetric periodic extension.
        coordinate = torch.arange(
            -self.cutoff_grid,
            self.cutoff_grid + 1,
            dtype=torch.get_default_dtype(),
        )
        qx, qy, qz = torch.meshgrid(
            coordinate,
            coordinate,
            coordinate,
            indexing="ij",
        )
        kernel_positions = torch.stack((qx, qy, qz), dim=-1)
        kernel_squared_distances = torch.sum(kernel_positions**2, dim=-1)
        kernel_monomials = torch.ones(
            (*kernel_squared_distances.shape, powers.shape[0]),
            dtype=torch.get_default_dtype(),
        )
        for axis in range(3):
            kernel_monomials = kernel_monomials * kernel_positions[
                ..., axis, None
            ].pow(powers[None, None, None, :, axis])
        kernel_mask = kernel_squared_distances <= self.cutoff_grid**2
        self.register_buffer(
            "kernel_squared_distances",
            kernel_squared_distances,
        )
        self.register_buffer("kernel_monomials", kernel_monomials)
        self.register_buffer("kernel_mask", kernel_mask)
        if self.trainable_radial_exponents:
            self.register_buffer(
                "fixed_convolution_kernel",
                None,
                persistent=False,
            )
        else:
            self.register_buffer(
                "fixed_convolution_kernel",
                self._build_convolution_kernel(),
                persistent=False,
            )

    @property
    def radial_exponents(self) -> torch.Tensor:
        """Positive Gaussian decay coefficients in inverse grid units squared."""

        return torch.exp(self.log_radial_exponents)

    def _apply(self, function):
        """Move/cast buffers and regenerate a fixed kernel at full precision."""

        result = super()._apply(function)
        if not self.trainable_radial_exponents:
            self.fixed_convolution_kernel = self._build_convolution_kernel()
        return result

    def _build_convolution_kernel(self) -> torch.Tensor:
        """Construct grouped-convolution weights for all basis terms."""

        radial_values = torch.exp(
            -self.radial_exponents[:, None, None, None]
            * self.kernel_squared_distances[None, ...]
        )
        basis_values = (
            radial_values[..., None]
            * self.kernel_monomials[None, ...]
            * self.kernel_mask[None, ..., None]
        )
        # [n_radial, n_monomial, kx, ky, kz]
        basis_values = basis_values.permute(0, 4, 1, 2, 3)
        # Group t owns one identical bank of radial/monomial output filters and
        # reads only input density channel t.
        return basis_values.unsqueeze(0).expand(
            self.n_types,
            -1,
            -1,
            -1,
            -1,
            -1,
        ).reshape(
            self.n_types
            * self.n_radial_channels
            * self.powers.shape[0],
            1,
            *basis_values.shape[-3:],
        )

    def _convolution_kernel(self) -> torch.Tensor:
        """Return cached fixed weights or rebuild trainable radial weights."""

        if self.fixed_convolution_kernel is not None:
            return self.fixed_convolution_kernel
        return self._build_convolution_kernel()

    def forward(self, data: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return density-weighted Cartesian ``A`` features.

        Parameters
        ----------
        data
            GridData-like mapping containing:

            ``rho``
                Live density tensor with shape ``[..., n_grid, n_types]``.
            ``grid_size``
                Three grid dimensions with shape ``[3]`` or
                ``[..., 3]``. Every field in one dense batch must have the same
                dimensions.
        Returns
        -------
        torch.Tensor
            ``A`` with shape
            ``[..., n_grid, n_radial_channels, n_monomials, n_output_channels]``.
            The final dimension equals ``n_types`` when mixing is disabled and
            ``n_channels`` when mixing is enabled.
        """

        if "rho" not in data:
            raise KeyError("data is missing required field 'rho'")
        if "grid_size" not in data:
            raise KeyError("data is missing required field 'grid_size'")
        rho = data["rho"]
        if rho.ndim < 2:
            raise ValueError("rho must have shape [..., n_grid, n_types]")
        if rho.shape[-1] != self.n_types:
            raise ValueError(
                "rho has {} type channels but CartesianAFeatures expects {}".format(
                    rho.shape[-1],
                    self.n_types,
                )
            )
        leading_shape = rho.shape[:-2]
        grid_size = _common_grid_size(data["grid_size"], leading_shape)
        n_grid = math.prod(grid_size)
        if rho.shape[-2] != n_grid:
            raise ValueError(
                "grid_size product does not match rho n_grid"
            )

        rho_3d = rho.reshape(-1, *grid_size, self.n_types).permute(
            0,
            4,
            1,
            2,
            3,
        )
        rho_3d = _periodic_extend_3d(rho_3d, self.cutoff_grid)
        features = F.conv3d(
            rho_3d / self.mean_density,
            self._convolution_kernel(),
            groups=self.n_types,
        )
        n_monomials = self.powers.shape[0]
        features = features.reshape(
            -1,
            self.n_types,
            self.n_radial_channels,
            n_monomials,
            *grid_size,
        ).permute(0, 4, 5, 6, 2, 3, 1)
        features = features.reshape(
            *leading_shape,
            n_grid,
            self.n_radial_channels,
            n_monomials,
            self.n_types,
        )
        if self.channel_mixing is not None:
            features = self.channel_mixing(features)
        return features


class _AChannelMixing(nn.Module):
    """Internal learned map from physical A channels to latent channels.

    For each grid point, radial channel, and Cartesian component, this module
    applies the same learned linear map

    ``A_mixed[..., q] = sum_t weight[q, t] * A[..., t]``.

    Here ``t`` labels the physical components of the density field and ``q``
    labels latent channels. Mixing only the final channel axis means that the
    operation commutes with rotations and reflections of the Cartesian
    component axis. Consequently, ``A_mixed`` can be passed directly to
    :class:`equicdft.symmetrize.CartesianBFeatures`.

    Notes
    -----
    The map intentionally has no additive bias. A constant bias applied to
    odd Cartesian components would not transform equivariantly under axis
    reflections.
    """

    def __init__(self, n_types: int, n_channels: int) -> None:
        super().__init__()

        for name, value in (
            ("n_types", n_types),
            ("n_channels", n_channels),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError("{} must be a positive integer".format(name))
            if int(value) < 1:
                raise ValueError("{} must be a positive integer".format(name))

        self.n_types = int(n_types)
        self.n_channels = int(n_channels)
        self.weight = nn.Parameter(
            torch.empty(
                self.n_channels,
                self.n_types,
                dtype=torch.get_default_dtype(),
            )
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        """Return ``A`` with its final physical-type axis linearly mixed."""

        if A.ndim < 4:
            raise ValueError(
                "A must have shape "
                "[..., n_grid, n_radial_channels, n_monomials, n_types]"
            )
        if A.shape[-1] != self.n_types:
            raise ValueError(
                "A has {} type channels but this module expects {}".format(
                    A.shape[-1],
                    self.n_types,
                )
            )

        return torch.einsum("...t,qt->...q", A, self.weight)
