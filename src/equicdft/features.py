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
from ._grid import gather_neighbors
from .stencil import make_stencil


DEFAULT_RADIAL_EXPONENTS = (0.125,)


def _cartesian_stencil_basis(
    squared_distances: torch.Tensor,
    monomial_values: torch.Tensor,
    radial_exponents: torch.Tensor,
    radial_centers: torch.Tensor,
    neighbor_mask: torch.Tensor,
) -> torch.Tensor:
    """Return normalized radial-Cartesian values with shape ``[J, N, K]``."""

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
    radial_values = radial_values / torch.clamp(
        torch.sum(radial_values, dim=0, keepdim=True),
        min=torch.finfo(radial_values.dtype).tiny,
    )
    return radial_values[:, :, None] * monomial_values[:, None, :]


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


def prepare_radial_exponents(
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


def prepare_radial_centers(
    radial_basis: str,
    radial_centers: Optional[Union[Sequence[float], torch.Tensor]],
    n_radial_channels: int,
) -> torch.Tensor:
    """Validate and return Gaussian centers ``u`` in grid units."""

    if radial_basis == "none":
        if radial_centers is not None:
            raise ValueError(
                "radial_centers are unavailable when radial_basis='none'"
            )
        return torch.zeros(1, dtype=torch.get_default_dtype())

    if radial_centers is None:
        return torch.zeros(
            n_radial_channels,
            dtype=torch.get_default_dtype(),
        )
    if isinstance(radial_centers, bool):
        raise TypeError("radial_centers must be a one-dimensional sequence")

    centers = torch.as_tensor(
        radial_centers,
        dtype=torch.get_default_dtype(),
    ).detach().clone()
    if centers.ndim != 1:
        raise ValueError("radial_centers must be one-dimensional")
    if centers.numel() != n_radial_channels:
        raise ValueError(
            "radial_centers length must match radial_exponents length"
        )
    if not torch.all(torch.isfinite(centers)).item():
        raise ValueError("radial_centers must be finite")
    if not torch.all(centers >= 0.0).item():
        raise ValueError("radial_centers must be nonnegative")
    return centers


class CartesianAFeatures(nn.Module):
    """Compute normalized Cartesian ``A`` features on a fixed stencil.

    For central grid point ``g``, radial channel ``n``, monomial
    ``k = (a, b, c)``, and component ``t``, the module computes

    ``A[g, n, k, t] = sum_j (rho[g, j, t] / mean_density)``
    ``* w[n, j] * x_j^a * y_j^b * z_j^c``,

    where ``j`` runs over the canonically ordered integer stencil. With
    ``radial_basis="gaussian"``, the normalized weights are

    ``w[n, j] = exp(-alpha[n] * (|q_j| - u[n])^2)``
    ``/ sum_i exp(-alpha[n] * (|q_i| - u[n])^2)``.

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
    radial_centers
        Optional Gaussian centers ``u_n`` in grid units. Its length must
        match ``radial_exponents``. ``None`` uses zero for every channel,
        exactly retaining the original zero-centered Gaussian basis. It is
        unavailable for ``radial_basis="none"``. A center ``u`` corresponds
        to physical radius ``u * grid_spacing``; ``coordinate_scaling`` does
        not alter this radial convention.
    trainable_radial_centers
        If ``True``, optimize the Gaussian centers directly. The default is
        ``False``. Initial centers must be nonnegative, but their optimized
        values are unconstrained.
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
        Number of output density channels after a pointwise linear transform.
        When ``density_transform`` is omitted, this retains the
        original Xavier-initialized learned transform. ``None`` retains the
        physical density channels unless explicit transform weights are
        supplied.
    density_transform
        Optional matrix with shape ``[n_channels, n_types]``. Before any
        neighborhood gathering or Cartesian-moment construction, physical
        density channels are replaced by ``rho_transformed[..., q] =
        sum_t weights[q, t] * rho[..., t]``. The transformed densities replace
        the physical channels in both the neighborhood moments and direct
        center descriptors. Negative and rectangular transforms are allowed.
        The number of rows determines ``n_channels`` when it is omitted.
    trainable_density_transform
        Whether a configured density transform is optimized. The default is
        ``True``, retaining the existing learned ``n_channels`` behavior.
        Set it to ``False`` with explicit ``density_transform`` weights for a
        fixed physical basis such as number and charge densities.
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
        radial_centers: Optional[
            Union[Sequence[float], torch.Tensor]
        ] = None,
        trainable_radial_centers: bool = False,
        density_transform: Optional[
            Union[Sequence[Sequence[float]], torch.Tensor]
        ] = None,
        trainable_density_transform: bool = True,
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
        trainable_radial_centers = boolean(
            trainable_radial_centers,
            "trainable_radial_centers",
        )
        if radial_basis == "none" and trainable_radial_centers:
            raise ValueError(
                "trainable_radial_centers requires radial_basis='gaussian'"
            )
        if not isinstance(coordinate_scaling, str):
            raise TypeError("coordinate_scaling must be 'none' or 'cutoff'")
        coordinate_scaling = coordinate_scaling.lower()
        if coordinate_scaling not in ("none", "cutoff"):
            raise ValueError("coordinate_scaling must be 'none' or 'cutoff'")
        separate_center = boolean(separate_center, "separate_center")
        n_types = positive_integer(n_types, "n_types")
        trainable_density_transform = boolean(
            trainable_density_transform,
            "trainable_density_transform",
        )
        if density_transform is None:
            transform_weights = None
        else:
            transform_weights = torch.as_tensor(
                density_transform,
                dtype=torch.get_default_dtype(),
            ).detach().clone()
            if transform_weights.ndim != 2:
                raise ValueError(
                    "density_transform must be a two-dimensional "
                    "matrix"
                )
            if transform_weights.shape[0] == 0:
                raise ValueError(
                    "density_transform must contain at least one row"
                )
            if transform_weights.shape[1] != n_types:
                raise ValueError(
                    "density_transform must contain one column per "
                    "physical density type"
                )
            if not torch.all(torch.isfinite(transform_weights)).item():
                raise ValueError("density_transform must be finite")
        n_channels = optional_positive_integer(n_channels, "n_channels")
        if transform_weights is not None:
            inferred_channels = int(transform_weights.shape[0])
            if n_channels is None:
                n_channels = inferred_channels
            elif n_channels != inferred_channels:
                raise ValueError(
                    "n_channels must equal the number of rows in "
                    "density_transform"
                )
        if n_channels is not None and n_types == 1:
            raise ValueError(
                "density transforms are disabled for one-component fields"
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

        # Each factor exp(-alpha_n * (|q| - u_n)**2) changes only the relative
        # emphasis of stencil points. Unit-sum normalization keeps every scalar
        # moment on the same scale as the undamped neighborhood average.
        # Trainable positive exponents are stored logarithmically; centers are
        # fixed architecture parameters.
        center_mask = squared_distances == 0
        neighbor_mask = ~center_mask if separate_center else torch.ones_like(
            center_mask
        )
        initial_radial_exponents = prepare_radial_exponents(
            radial_basis,
            radial_exponents,
            trainable_radial_exponents,
        )
        initial_radial_centers = prepare_radial_centers(
            radial_basis,
            radial_centers,
            int(initial_radial_exponents.numel()),
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
        if radial_basis == "gaussian" and trainable_radial_centers:
            self.learned_radial_centers = nn.Parameter(
                initial_radial_centers
            )
        elif radial_basis == "gaussian":
            self.register_buffer(
                "fixed_radial_centers",
                initial_radial_centers,
                persistent=False,
            )
        self.cutoff_grid = int(cutoff_grid)
        self.max_power = int(max_power)
        self.radial_basis = radial_basis
        self.trainable_radial_exponents = trainable_radial_exponents
        self.trainable_radial_centers = trainable_radial_centers
        self.coordinate_scaling = coordinate_scaling
        self.separate_center = separate_center
        self.n_types = n_types
        self.n_channels = n_channels
        self.trainable_density_transform = trainable_density_transform
        self.n_radial_channels = int(initial_radial_exponents.numel())
        self.n_output_channels = n_types if n_channels is None else n_channels
        self.density_transform = (
            None
            if n_channels is None
            else _DensityMixing(
                n_types=n_types,
                n_channels=n_channels,
                weights=transform_weights,
                trainable=trainable_density_transform,
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

    @property
    def radial_centers(self) -> torch.Tensor:
        """Gaussian centers, with a zero fallback for older checkpoints."""

        if getattr(self, "trainable_radial_centers", False):
            return self.learned_radial_centers
        stored = self._buffers.get("fixed_radial_centers")
        if stored is not None:
            return stored
        return torch.zeros_like(self.radial_exponents)

    def stencil_basis(
        self,
        radial_exponents: Optional[torch.Tensor] = None,
        radial_centers: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return a radial-Cartesian basis with shape ``[J, N, K]``.

        ``J`` is the number of stencil points, ``N`` the number of radial
        channels, and ``K`` the number of Cartesian monomials. By default the
        module's own exponents are used. Message layers may supply independent
        exponents while reusing this fixed stencil geometry.
        """

        if radial_exponents is None:
            radial_exponents = self.radial_exponents
        if radial_centers is None:
            radial_centers = self.radial_centers
        if radial_centers.shape != radial_exponents.shape:
            raise ValueError(
                "radial_centers shape must match radial_exponents shape"
            )
        return _cartesian_stencil_basis(
            self.squared_distances,
            self.monomial_values,
            radial_exponents,
            radial_centers,
            self.stencil_neighbor_mask(),
        )

    def stencil_neighbor_mask(self) -> torch.Tensor:
        """Return the center-inclusion mask, including legacy fallback."""

        stored = self._buffers.get("neighbor_mask")
        if stored is not None:
            return stored
        if getattr(self, "separate_center", False):
            return self.squared_distances != 0
        return torch.ones_like(self.squared_distances, dtype=torch.bool)

    def transform_density(self, rho: torch.Tensor) -> torch.Tensor:
        """Return the physical or configured transformed density channels.

        The transform is pointwise and remains connected to ``rho`` for
        functional derivatives.
        """

        if rho.ndim < 2 or rho.shape[-1] != self.n_types:
            raise ValueError(
                "rho must have shape [..., n_grid, n_types] with the "
                "configured number of physical density types"
            )
        if self.n_channels is None:
            return rho
        return self.density_transform(rho)

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

        descriptor_density = self.transform_density(data["rho"])
        local_density = gather_neighbors(
            descriptor_density,
            data["local_density_index"],
        )
        if local_density.shape[-2] != self.monomial_values.shape[0]:
            raise ValueError(
                "local_density neighbor count does not match cutoff_grid={}".format(
                    self.cutoff_grid
                )
            )

        # Contract only the neighbor axis. Grid points, radial channels,
        # monomials, component channels, and leading batch dimensions remain
        # separate.
        features = torch.einsum(
            "...gjt,jnk->...gnkt",
            local_density / self.mean_density,
            self.stencil_basis(),
        )
        return features


class _DensityMixing(nn.Module):
    """Internal pointwise map from physical to descriptor densities.

    ``rho_mixed[..., q] = sum_t weight[q, t] * rho[..., t]``.

    Here ``t`` labels physical density components and ``q`` labels descriptor
    density channels. The map acts independently at every spatial point, so
    spatial rotations and reflections are unaffected.

    Notes
    -----
    The map intentionally has no additive bias, so an empty voxel remains
    empty after the transform.
    """

    def __init__(
        self,
        n_types: int,
        n_channels: int,
        weights: Optional[torch.Tensor] = None,
        trainable: bool = True,
    ) -> None:
        super().__init__()

        self.n_types = positive_integer(n_types, "n_types")
        self.n_channels = positive_integer(n_channels, "n_channels")
        if weights is None:
            initial_weights = torch.empty(
                self.n_channels,
                self.n_types,
                dtype=torch.get_default_dtype(),
            )
            nn.init.xavier_uniform_(initial_weights)
        else:
            initial_weights = weights.detach().clone()
        self.weight = nn.Parameter(initial_weights, requires_grad=trainable)

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        """Return ``rho`` with its physical-type axis linearly transformed."""

        if rho.ndim < 2:
            raise ValueError(
                "rho must have shape [..., n_grid, n_types]"
            )
        if rho.shape[-1] != self.n_types:
            raise ValueError(
                "rho has {} type channels but this module expects {}".format(
                    rho.shape[-1],
                    self.n_types,
                )
            )

        return torch.einsum("...t,qt->...q", rho, self.weight)
