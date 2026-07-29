"""Cartesian moment features for density fields on fixed integer grids.

The angular basis uses the dimensionless integer stencil coordinates
``(x, y, z)`` rather than physical distances. Gaussian radial functions are
also evaluated using the squared integer-grid distance. The physical grid
spacing enters only through the cell-volume factor used to turn the neighbor
sum into a discrete spatial integral.
"""

from numbers import Integral
from typing import Mapping

import torch
from torch import nn

from .stencil import make_stencil


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
    """Compute density-weighted Cartesian ``A`` features on a fixed stencil.

    For central grid point ``g``, Gaussian channel ``n``, monomial
    ``k = (a, b, c)``, and component ``t``, the module computes

    ``A[g, n, k, t] = cell_volume * sum_j rho[g, j, t]``
    ``* exp(-alpha[n] * r_grid[j]^2) * x_j^a y_j^b z_j^c``.

    Here ``j`` runs over the canonically ordered integer stencil. The monomial
    coordinates and ``r_grid^2 = x^2 + y^2 + z^2`` use integer grid units;
    ``cell_volume = hx * hy * hz`` is the physical volume represented by one
    density-grid point.

    Parameters
    ----------
    cutoff_grid
        Inclusive spherical cutoff in integer grid steps. The default is
        three, matching :class:`cace_grid.data.GridData`.
    max_power
        Maximum total Cartesian power ``a + b + c``.
    n_alphas
        Number of Gaussian radial channels. Their positive initial decay
        coefficients are logarithmically spaced from 0.25 to 2.0 in inverse
        squared grid units.
    trainable_alphas
        If ``True``, optimize the Gaussian decay coefficients. They are stored
        in logarithmic form so that the resulting ``alpha`` values stay
        positive. If ``False``, they remain fixed model buffers.

    Notes
    -----
    The stencil and all monomials are model constants. They are constructed
    once using the same canonical ordering as ``GridData`` and registered as
    buffers, so they follow the module across devices without being optimized.
    """

    def __init__(
        self,
        max_power: int,
        cutoff_grid: int = 3,
        n_alphas: int = 4,
        trainable_alphas: bool = False,
    ) -> None:
        super().__init__()

        if isinstance(n_alphas, bool) or not isinstance(n_alphas, Integral):
            raise TypeError("n_alphas must be a positive integer")
        n_alphas = int(n_alphas)
        if n_alphas < 1:
            raise ValueError("n_alphas must be a positive integer")
        if not isinstance(trainable_alphas, bool):
            raise TypeError("trainable_alphas must be a boolean")

        # Both GridData and this module call make_stencil, so neighbor j in
        # local_density always corresponds to row j of the monomial table.
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

        # A logarithmic sequence resolves both broad and center-dominated
        # environments. For n_alphas=4 this gives [0.25, 0.5, 1.0, 2.0].
        initial_alphas = 2.0 ** torch.linspace(
            -2.0,
            1.0,
            steps=n_alphas,
            dtype=torch.get_default_dtype(),
        )
        log_alphas = torch.log(initial_alphas)
        if trainable_alphas:
            self.log_alphas = nn.Parameter(log_alphas)
        else:
            self.register_buffer("log_alphas", log_alphas)

        self.cutoff_grid = int(cutoff_grid)
        self.max_power = int(max_power)
        self.n_alphas = n_alphas
        self.trainable_alphas = trainable_alphas
        self.register_buffer("local_density_positions", local_density_positions)
        self.register_buffer("squared_distances", squared_distances)
        self.register_buffer("powers", powers)
        self.register_buffer("monomial_values", monomial_values)

    @property
    def alphas(self) -> torch.Tensor:
        """Positive Gaussian decay coefficients in inverse grid units squared."""

        return torch.exp(self.log_alphas)

    def forward(self, data: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return density-weighted Cartesian ``A`` features.

        Parameters
        ----------
        data
            GridData-like mapping containing:

            ``local_density``
                Tensor with shape
                ``[..., n_grid, n_neighbors, n_types]``. Any leading
                dimensions, such as the training-batch dimension, are
                preserved.
            ``grid_spacing``
                Physical grid spacing ``(hx, hy, hz)`` with shape ``[3]`` for
                one configuration or ``[..., 3]`` for batched data. It is not
                used to rescale the integer monomials. Its product supplies
                the cell-volume quadrature weight.

        Returns
        -------
        torch.Tensor
            ``A`` with shape
            ``[..., n_grid, n_alphas, n_monomials, n_types]``.
        """

        local_density = data["local_density"]
        if local_density.ndim < 3:
            raise ValueError(
                "local_density must have shape "
                "[..., n_grid, n_neighbors, n_types]"
            )
        if local_density.shape[-2] != self.monomial_values.shape[0]:
            raise ValueError(
                "local_density neighbor count does not match cutoff_grid={}".format(
                    self.cutoff_grid
                )
            )

        # The radial values are recomputed on every forward pass because the
        # alpha values may be trainable. Distances themselves are fixed model
        # geometry and are therefore stored once as a buffer.
        radial_values = torch.exp(
            -self.squared_distances[:, None] * self.alphas[None, :]
        )
        basis_values = (
            radial_values[:, :, None] * self.monomial_values[:, None, :]
        )

        # Contract only the neighbor axis. Grid points, radial channels,
        # monomials, component channels, and leading batch dimensions remain
        # separate.
        features = torch.einsum(
            "...gjt,jnk->...gnkt",
            local_density,
            basis_values,
        )

        # grid_spacing stores physical lengths, not grid dimensions. Its
        # product is the volume dV represented by one grid point. For batched
        # input this initially has only the batch dimensions; trailing
        # singleton axes make it broadcast over grid, radial, monomial, and
        # type dimensions.
        grid_spacing = data["grid_spacing"]
        if grid_spacing.shape[-1] != 3:
            raise ValueError("grid_spacing must have shape [..., 3]")
        cell_volume = torch.prod(grid_spacing, dim=-1)
        while cell_volume.ndim < features.ndim:
            cell_volume = cell_volume.unsqueeze(-1)
        return cell_volume * features


class AChannelMixing(nn.Module):
    """Mix physical component channels of ``A`` into latent channels.

    For each grid point, radial channel, and Cartesian component, this module
    applies the same learned linear map

    ``A_mixed[..., q] = sum_t weight[q, t] * A[..., t]``.

    Here ``t`` labels the physical components of the density field and ``q``
    labels latent channels. Mixing only the final channel axis means that the
    operation commutes with rotations and reflections of the Cartesian
    component axis. Consequently, ``A_mixed`` can be passed directly to
    :class:`cace_grid.symmetrize.CartesianBFeatures`.

    Parameters
    ----------
    n_types
        Number of physical density/component channels in ``A``.
    n_channels
        Number of latent channels produced by the learned mixing matrix.

    Notes
    -----
    The map intentionally has no additive bias. A constant bias applied to
    odd Cartesian components would not transform equivariantly under axis
    reflections. For a one-component system this module is optional; the
    original ``A`` tensor can be symmetrized directly.
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
                "[..., n_grid, n_alphas, n_monomials, n_types]"
            )
        if A.shape[-1] != self.n_types:
            raise ValueError(
                "A has {} type channels but this module expects {}".format(
                    A.shape[-1],
                    self.n_types,
                )
            )

        return torch.einsum("...t,qt->...q", A, self.weight)
