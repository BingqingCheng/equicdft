"""Cartesian moment features for density fields on fixed integer grids.

The angular basis uses integer stencil coordinates. Optional Gaussian factors
damp distant stencil points and provide one feature channel per exponent.
"""

from typing import Mapping, Optional, Sequence, Union

import torch
from torch import nn

from ._argument_checks import (
    boolean,
    nonnegative_integer,
    optional_positive_integer,
    positive_integer,
)
from .stencil import make_stencil


DEFAULT_RADIAL_EXPONENTS = (0.125,)


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

    max_power = nonnegative_integer(max_power, "max_power")

    powers = []
    for total_power in range(max_power + 1):
        for x_power in range(total_power, -1, -1):
            remaining_power = total_power - x_power
            for y_power in range(remaining_power, -1, -1):
                z_power = remaining_power - y_power
                powers.append((x_power, y_power, z_power))
    return torch.tensor(powers, dtype=torch.long)


def _prepare_radial_exponents(
    radial_basis: str,
    radial_exponents: Optional[Union[Sequence[float], torch.Tensor]],
    trainable: bool,
) -> torch.Tensor:
    """Validate and return the initial radial damping exponents."""

    if radial_basis == "none":
        if radial_exponents is not None:
            raise ValueError(
                "radial_exponents are unavailable when radial_basis='none'"
            )
        if trainable:
            raise ValueError(
                "trainable_radial_exponents requires radial_basis='gaussian'"
            )
        return torch.zeros(1, dtype=torch.get_default_dtype())

    if radial_exponents is None:
        radial_exponents = DEFAULT_RADIAL_EXPONENTS
    if isinstance(radial_exponents, bool):
        raise TypeError("radial_exponents must be a one-dimensional sequence")

    exponents = torch.as_tensor(
        radial_exponents,
        dtype=torch.get_default_dtype(),
    ).detach().clone()
    if exponents.ndim != 1:
        raise ValueError("radial_exponents must be one-dimensional")
    if exponents.numel() == 0:
        raise ValueError("radial_exponents must not be empty")
    if not torch.all(torch.isfinite(exponents)).item():
        raise ValueError("radial_exponents must be finite")
    if not torch.all(exponents >= 0.0).item():
        raise ValueError("radial_exponents must be nonnegative")
    if trainable and not torch.all(exponents > 0.0).item():
        raise ValueError("trainable_radial_exponents must be positive")
    return exponents


class CartesianAFeatures(nn.Module):
    """Compute normalized Cartesian ``A`` features on a fixed stencil.

    For central grid point ``g``, radial channel ``n``, monomial
    ``k = (a, b, c)``, and component ``t``, the module computes

    ``A[g, n, k, t] = sum_j (rho[g, j, t] / mean_density)``
    ``* w[n, j] * x_j^a * y_j^b * z_j^c``,

    where ``j`` runs over the canonically ordered integer stencil. With
    ``radial_basis="gaussian"``, the normalized weights are

    ``w[n, j] = exp(-alpha[n] * |q_j|^2)``
    ``/ sum_i exp(-alpha[n] * |q_i|^2)``.

    With ``radial_basis="none"``, the module uses the same expression with one
    fixed exponent ``alpha[0]=0``, exactly recovering the uniform normalized
    polynomial moment. When ``separate_center=True``, the central point is
    excluded from every normalized channel and supplied directly to the local
    readout by the model.
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
        ``"none"`` selects uniform normalized weights. ``"gaussian"``
        selects normalized exponential damping channels.
    radial_exponents
        Optional nonempty sequence of nonnegative damping coefficients
        ``alpha_n`` in inverse squared grid units. Its length determines the
        number of channels. For ``radial_basis="gaussian"``, ``None`` uses
        one channel with ``alpha=0.125``. It is unavailable for
        ``radial_basis="none"``, which always uses one fixed zero exponent.
    n_radial_channels
        Compatibility argument for the retained example script. Only ``1``
        with ``radial_basis="none"`` is accepted; Gaussian channel count is
        determined by ``radial_exponents``.
    trainable_radial_exponents
        If ``True``, optimize positive ``radial_exponents`` in logarithmic
        form. Zero exponents are allowed only when the list is fixed.
    coordinate_scaling
        Cartesian-coordinate convention used in the monomials. ``"none"``
        uses the raw integer stencil offsets. ``"cutoff"`` divides each
        coordinate by ``cutoff_grid``, keeping polynomial moments similarly
        scaled when comparing different cutoffs. The damping distance remains
        in raw squared grid units.
    separate_center
        If ``True``, remove the zero offset from all neighbor moments. The
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
        radial_exponents: Optional[
            Union[Sequence[float], torch.Tensor]
        ] = None,
        n_radial_channels: Optional[int] = None,
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
        if n_radial_channels is not None:
            n_radial_channels = positive_integer(
                n_radial_channels,
                "n_radial_channels",
            )
            if radial_basis != "none" or n_radial_channels != 1:
                raise ValueError(
                    "n_radial_channels is retained only as 1 with "
                    "radial_basis='none'"
                )
        trainable_radial_exponents = boolean(
            trainable_radial_exponents,
            "trainable_radial_exponents",
        )
        if not isinstance(coordinate_scaling, str):
            raise TypeError("coordinate_scaling must be 'none' or 'cutoff'")
        coordinate_scaling = coordinate_scaling.lower()
        if coordinate_scaling not in ("none", "cutoff"):
            raise ValueError("coordinate_scaling must be 'none' or 'cutoff'")
        separate_center = boolean(separate_center, "separate_center")
        n_types = positive_integer(n_types, "n_types")

        n_channels = optional_positive_integer(n_channels, "n_channels")
        if n_channels is not None:
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

        # Each factor exp(-alpha_n * |q|**2) changes only the relative emphasis
        # of stencil points. Unit-sum normalization keeps every scalar moment
        # on the same scale as the undamped neighborhood average. Trainable
        # positive exponents are stored logarithmically.
        center_mask = squared_distances == 0
        neighbor_mask = ~center_mask if separate_center else torch.ones_like(
            center_mask
        )
        initial_radial_exponents = _prepare_radial_exponents(
            radial_basis,
            radial_exponents,
            trainable_radial_exponents,
        )

        if radial_basis == "gaussian" and trainable_radial_exponents:
            self.log_radial_exponents = nn.Parameter(
                torch.log(initial_radial_exponents)
            )
        else:
            self.register_buffer(
                "fixed_radial_exponents",
                initial_radial_exponents,
            )
        self.cutoff_grid = int(cutoff_grid)
        self.max_power = int(max_power)
        self.radial_basis = radial_basis
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
    def radial_exponents(self) -> torch.Tensor:
        """Damping coefficients for all radial channels.

        The zero fallback keeps the retained undamped regression example
        loadable; its checkpoint predates explicit radial-basis attributes.
        """

        if getattr(self, "trainable_radial_exponents", False):
            return torch.exp(self.log_radial_exponents)
        stored = self._buffers.get("fixed_radial_exponents")
        if stored is not None:
            return stored
        if getattr(self, "radial_basis", "none") == "none":
            return self.squared_distances.new_zeros(1)
        raise RuntimeError("this Gaussian radial checkpoint is incompatible")

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

        # This is the only radial path. alpha=0 produces the uniform polynomial
        # channel, while positive alpha values damp distant stencil points.
        radial_values = torch.exp(
            -self.squared_distances[:, None]
            * self.radial_exponents[None, :]
        )
        if getattr(self, "separate_center", False):
            radial_values = radial_values * self.neighbor_mask[:, None]
        radial_values = radial_values / torch.clamp(
            torch.sum(radial_values, dim=0, keepdim=True),
            min=torch.finfo(radial_values.dtype).tiny,
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
    applies the same learned linear map.

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

        self.n_types = positive_integer(n_types, "n_types")
        self.n_channels = positive_integer(n_channels, "n_channels")
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
