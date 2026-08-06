"""Cartesian moment features for density fields on fixed integer grids.

The angular basis uses integer stencil coordinates. The optional Gaussian
radial basis is evaluated using squared integer-grid distances without an
additional kernel normalization. A purely polynomial representation can
instead use one undamped radial channel.
"""

from numbers import Integral
from typing import Mapping, Optional, Sequence, Union

import torch
from torch import nn

from .stencil import make_stencil


def _gather_local_density(
    rho: torch.Tensor,
    local_density_index: torch.Tensor,
) -> torch.Tensor:
    """Gather periodic environments from a live, possibly batched ``rho``."""

    if rho.ndim < 2:
        raise ValueError("rho must have shape [..., n_grid, n_types]")
    if local_density_index.ndim != rho.ndim:
        raise ValueError(
            "local_density_index must have shape "
            "[..., n_grid, n_neighbors]"
        )
    if local_density_index.shape[:-2] != rho.shape[:-2]:
        raise ValueError("rho and local_density_index leading shapes must match")
    if local_density_index.shape[-2] != rho.shape[-2]:
        raise ValueError("rho and local_density_index grid sizes must match")
    if local_density_index.dtype != torch.long:
        raise TypeError("local_density_index must have dtype torch.long")

    leading_shape = rho.shape[:-2]
    n_grid = rho.shape[-2]
    n_types = rho.shape[-1]
    n_neighbors = local_density_index.shape[-1]

    # Flatten only leading configuration/batch dimensions. torch.gather then
    # selects the grid axis independently for every configuration and type.
    rho_flat = rho.reshape(-1, n_grid, n_types)
    index_flat = local_density_index.reshape(-1, n_grid * n_neighbors)
    gather_index = index_flat.unsqueeze(-1).expand(-1, -1, n_types)
    local_density = torch.gather(rho_flat, dim=1, index=gather_index)
    return local_density.reshape(
        *leading_shape,
        n_grid,
        n_neighbors,
        n_types,
    )


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
    """Compute normalized Cartesian ``A`` features on a fixed stencil.

    For central grid point ``g``, radial channel ``n``, monomial
    ``k = (a, b, c)``, and component ``t``, the module computes

    ``A[g, n, k, t] = sum_j (rho[g, j, t] / mean_density)``
    ``* w[n, j] * x_j^a * y_j^b * z_j^c``,

    where ``j`` runs over the canonically ordered integer stencil. With
    ``radial_basis="gaussian"``,
    ``w[n, j] = exp(-alpha[n] * r_grid[j]^2)``. With
    ``radial_basis="none"``, there is one channel with equal weight on all
    included neighbors. When ``separate_center=True``, the central point is
    excluded from every A channel and supplied directly to the local readout
    by the model.
    Grid-volume quadrature is deliberately left to the free-energy readout
    rather than included in these local descriptors.

    Parameters
    ----------
    cutoff_grid
        Inclusive spherical cutoff in integer grid steps. The default is
        three, matching :class:`equicdft.data.GridData`.
    max_power
        Maximum total Cartesian power ``a + b + c``.
    radial_basis
        Radial representation. ``"gaussian"`` uses Gaussian decay channels;
        ``"none"`` uses one constant, discrete-sum-normalized channel.
    n_radial_channels
        Number of Gaussian radial channels. Their positive initial decay
        coefficients are logarithmically spaced from 0.5 to 4.0 in inverse
        squared grid units. Must equal one when ``radial_basis="none"``.
    radial_exponents
        Optional positive initial decay coefficients ``alpha_n`` for the
        Gaussian weights ``exp(-alpha_n * |q|**2)``. When supplied, its length
        must equal ``n_radial_channels``. ``None`` retains the default
        logarithmic sequence from 0.5 to 4.0.
    trainable_radial_exponents
        If ``True``, optimize the Gaussian decay coefficients. They are stored
        in logarithmic form so that the resulting ``alpha`` values stay
        positive. If ``False``, they remain fixed model buffers.
    coordinate_scaling
        Cartesian-coordinate convention used in the monomials. ``"none"``
        uses the raw integer stencil offsets. ``"cutoff"`` divides each
        coordinate by ``cutoff_grid``, keeping polynomial moments similarly
        scaled when comparing different cutoffs. Gaussian radial distances
        remain in raw squared grid units.
    separate_center
        If ``True``, remove the zero offset from all neighbor channels. The
        model then concatenates the normalized central density to the
        invariant neighbor features exactly once.
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

    Notes
    -----
    The stencil and all monomials are model constants. They are constructed
    once using the same canonical ordering as ``GridData`` and registered as
    buffers, so they follow the module across devices without being optimized.
    """

    def __init__(
        self,
        max_power: int,
        mean_density: Union[float, torch.Tensor],
        cutoff_grid: int = 3,
        radial_basis: str = "none",
        n_radial_channels: int = 1,
        radial_exponents: Optional[Sequence[float]] = None,
        trainable_radial_exponents: bool = False,
        coordinate_scaling: str = "none",
        separate_center: bool = True,
        n_types: int = 1,
        n_channels: Optional[int] = None,
    ) -> None:
        super().__init__()

        if not isinstance(radial_basis, str):
            raise TypeError("radial_basis must be 'gaussian' or 'none'")
        radial_basis = radial_basis.lower()
        if radial_basis not in ("gaussian", "none"):
            raise ValueError("radial_basis must be 'gaussian' or 'none'")

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
        if not isinstance(coordinate_scaling, str):
            raise TypeError("coordinate_scaling must be 'none' or 'cutoff'")
        coordinate_scaling = coordinate_scaling.lower()
        if coordinate_scaling not in ("none", "cutoff"):
            raise ValueError("coordinate_scaling must be 'none' or 'cutoff'")
        if not isinstance(separate_center, bool):
            raise TypeError("separate_center must be a boolean")
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

        # Both GridData and this module call make_stencil, so neighbor j in
        # local_density always corresponds to row j of the monomial table.
        local_density_positions = torch.from_numpy(make_stencil(cutoff_grid))
        powers = _make_powers(max_power)

        # Evaluate every Cartesian monomial once on the shared integer
        # stencil. These values are independent of the central grid point and
        # of the density configuration.
        positions = local_density_positions.to(dtype=torch.get_default_dtype())
        squared_distances = torch.sum(positions**2, dim=1)
        monomial_positions = positions
        if coordinate_scaling == "cutoff" and int(cutoff_grid) > 0:
            monomial_positions = positions / float(cutoff_grid)
        monomial_values = torch.ones(
            (positions.shape[0], powers.shape[0]),
            dtype=torch.get_default_dtype(),
        )
        for axis in range(3):
            monomial_values = monomial_values * monomial_positions[
                :, axis, None
            ].pow(powers[None, :, axis])

        # Gaussian channel n uses R_n(q) = exp(-alpha_n * |q|**2). The
        # explicit option makes the effective range a fitting choice rather
        # than coupling it implicitly to the integer stencil cutoff. The
        # undamped polynomial representation has exactly one radial channel
        # and no radial parameters. Its constant weight is normalized by the
        # stencil size so products of scalar moments remain well conditioned.
        center_mask = squared_distances == 0
        neighbor_mask = ~center_mask if separate_center else torch.ones_like(
            center_mask
        )

        if radial_basis == "gaussian":
            if radial_exponents is None:
                initial_radial_exponents = 2.0 ** torch.linspace(
                    -1.0,
                    2.0,
                    steps=n_radial_channels,
                    dtype=torch.get_default_dtype(),
                )
            else:
                initial_radial_exponents = torch.as_tensor(
                    radial_exponents,
                    dtype=torch.get_default_dtype(),
                ).detach().clone()
                if initial_radial_exponents.ndim != 1:
                    raise ValueError("radial_exponents must be one-dimensional")
                if initial_radial_exponents.numel() != n_radial_channels:
                    raise ValueError(
                        "radial_exponents must contain n_radial_channels values"
                    )
                if not torch.all(
                    torch.isfinite(initial_radial_exponents)
                ).item():
                    raise ValueError("radial_exponents must be finite")
                if not torch.all(initial_radial_exponents > 0.0).item():
                    raise ValueError("radial_exponents must be positive")
            log_radial_exponents = torch.log(initial_radial_exponents)
            if trainable_radial_exponents:
                self.log_radial_exponents = nn.Parameter(log_radial_exponents)
            else:
                self.register_buffer(
                    "log_radial_exponents",
                    log_radial_exponents,
                )
        else:
            if n_radial_channels != 1:
                raise ValueError(
                    "n_radial_channels must equal one when radial_basis='none'"
                )
            if radial_exponents is not None:
                raise ValueError(
                    "radial_exponents are unavailable when radial_basis='none'"
                )
            if trainable_radial_exponents:
                raise ValueError(
                    "trainable_radial_exponents requires radial_basis='gaussian'"
                )

        self.cutoff_grid = int(cutoff_grid)
        self.max_power = int(max_power)
        self.radial_basis = radial_basis
        self.n_radial_channels = n_radial_channels
        self.trainable_radial_exponents = trainable_radial_exponents
        self.coordinate_scaling = coordinate_scaling
        self.separate_center = separate_center
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
        # This deterministic constructor product is not fitted state. Keeping
        # it out of state_dict lets older checkpoints load strictly.
        self.register_buffer("neighbor_mask", neighbor_mask, persistent=False)

    @property
    def radial_exponents(self) -> Optional[torch.Tensor]:
        """Positive Gaussian decay coefficients, or ``None`` when undamped."""

        if self.radial_basis == "none":
            return None
        return torch.exp(self.log_radial_exponents)

    def forward(self, data: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return density-weighted Cartesian ``A`` features.

        Parameters
        ----------
        data
            GridData-like mapping containing:

            ``rho``
                Live density tensor with shape ``[..., n_grid, n_types]``.
            ``local_density_index``
                Periodic neighbor rows with shape
                ``[..., n_grid, n_neighbors]``. Local environments are
                gathered from ``rho`` inside this forward pass so that
                automatic differentiation includes overlapping neighbors.
        Returns
        -------
        torch.Tensor
            ``A`` with shape
            ``[..., n_grid, n_radial_channels, n_monomials, n_output_channels]``.
            The final dimension equals ``n_types`` when mixing is disabled and
            ``n_channels`` when mixing is enabled.
        """

        local_density = _gather_local_density(
            data["rho"],
            data["local_density_index"],
        )
        if local_density.shape[-1] != self.n_types:
            raise ValueError(
                "rho has {} type channels but CartesianAFeatures expects {}".format(
                    local_density.shape[-1],
                    self.n_types,
                )
            )
        if local_density.shape[-2] != self.monomial_values.shape[0]:
            raise ValueError(
                "local_density neighbor count does not match cutoff_grid={}".format(
                    self.cutoff_grid
                )
            )

        # Gaussian values are recomputed because alpha may be trainable. The
        # undamped representation deliberately assigns equal relative radial
        # weight to the central and noncentral stencil points. Dividing by the
        # stencil size makes its scalar moment a neighborhood average.
        if self.radial_basis == "gaussian":
            radial_values = torch.exp(
                -self.squared_distances[:, None]
                * self.radial_exponents[None, :]
            )
            if getattr(self, "separate_center", False):
                radial_values = radial_values * self.neighbor_mask[:, None]
        else:
            radial_values = self.squared_distances.new_ones(
                (self.squared_distances.shape[0], 1)
            )
            if getattr(self, "separate_center", False):
                radial_values = radial_values * self.neighbor_mask[:, None]
            radial_values = radial_values / torch.clamp(
                torch.sum(radial_values),
                min=1.0,
            )
        basis_values = (
            radial_values[:, :, None] * self.monomial_values[:, None, :]
        )

        # Contract only the neighbor axis. Grid points, radial channels,
        # monomials, component channels, and leading batch dimensions remain
        # separate.
        features = torch.einsum(
            "...gjt,jnk->...gnkt",
            local_density / self.mean_density,
            basis_values,
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
