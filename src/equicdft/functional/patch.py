"""Unconstrained neural readout of an ordered local density patch."""

from typing import Any, Dict, Sequence

import torch

from .._config import Configurable, make_config, register
from .._nn import build_mlp, validate_hidden_sizes
from .._validation import nonnegative_integer, positive_integer
from ..geometry.convolution import gather_neighbors
from ..geometry.grid import (
    _flat_grid_positions, _ravel_grid_indices, common_grid_size,
)
from ..geometry.stencil import make_stencil
from .base import EnergyReadout, ReadoutContext

__all__ = ["DensityPatchReadout"]


@register
class DensityPatchReadout(EnergyReadout, Configurable):
    """Map raw periodic density patches and temperature to local energies.

    The MLP input at grid point g is the flattened list
    ``[rho[g+q, a] / mean_density]`` followed by ``T / mean_temperature``.
    With ``domain="sphere"`` (the default), offsets q follow
    :func:`make_stencil`: the center occurs once, and integer offsets with
    ``|q|^2 <= cutoff_grid^2`` are retained in canonical stencil order.
    With ``domain="cube"``, all ``|q_x|, |q_y|, |q_z| <= cutoff_grid`` are
    retained in lexicographic order (z varies fastest). The center still
    occurs once. Species vary fastest within each offset. At cutoff 4 a
    one-component sphere/cube has 257/729 density inputs, plus temperature.

    Unlike an invariant descriptor, the ordered patch gives each direction
    its own network weights. Translation symmetry follows from sharing the
    MLP over all centers; rotations and reflections are not enforced.

    The output has one per-particle value per species. Its integral is
    ``DeltaV * sum_g,a rho[g,a] * output[g,a]``. The containing model selects
    whether this is beta F_exc or F_exc/(k_B T_ref). Density normalization
    uses the model-owned scale, while integration uses physical density.
    Gathering never detaches rho, so derivatives include every overlapping
    patch. Smooth SiLU hidden layers permit c1 and higher derivatives.

    This readout needs no A/B features or separate LDA contribution. The
    sphere uses the model's shared neighbor table; the cube builds its own
    periodic geometry from grid_size and grid_positions. Only geometry is
    cached, never densities. cutoff_grid is the axial reach in both cases.
    The patch width is fixed: changing resolution is not an automatic
    resolution-transfer rule for this baseline.
    """

    shares_grid_stencil = True
    reads_smoothed_fields = True

    def __init__(
        self,
        n_types: int = 1,
        *,
        cutoff_grid: int = 4,
        domain: str = "sphere",
        hidden_sizes: Sequence[int] = (32, 16),
    ) -> None:
        super().__init__()
        self.n_types = positive_integer(n_types, "n_types")
        self.cutoff_grid = nonnegative_integer(cutoff_grid, "cutoff_grid")
        if domain not in ("sphere", "cube"):
            raise ValueError("domain must be 'sphere' or 'cube'")
        self.domain = domain
        self.hidden_sizes = validate_hidden_sizes(hidden_sizes)
        self.n_neighbors = (len(make_stencil(self.cutoff_grid))
                            if domain == "sphere" else (2*self.cutoff_grid+1)**3)
        self.n_features = self.n_neighbors * self.n_types + 1
        self.mlp = build_mlp(
            self.n_features, self.hidden_sizes, self.n_types,
        )
        # Rebuilt for a new grid shape/device; not part of fitted weights.
        self.register_buffer("_cube_index", torch.empty(0, dtype=torch.long), persistent=False)
        self._cube_shape = None

    @property
    def requires_local_density_index(self) -> bool:
        """Only the spherical patch reads the shared spherical table."""

        return self.domain == "sphere"

    def to_config(self) -> Dict[str, Any]:
        """Return constructor arguments; fitted weights live in state_dict."""

        return make_config(
            self,
            n_types=self.n_types,
            cutoff_grid=self.cutoff_grid,
            domain=self.domain,
            hidden_sizes=self.hidden_sizes,
        )

    def _cube_neighbor_indices(self, context: ReadoutContext) -> torch.Tensor:
        """Return periodic cube rows, including correct shuffled-row lookup."""

        rho = context.rho
        shape = common_grid_size(context.require("grid_size"), rho.shape[:-2])
        flat, canonical = _flat_grid_positions(
            context.require("grid_positions"), shape, rho.shape[:-2], rho.device,
        )
        if self._cube_shape != shape or self._cube_index.device != rho.device:
            coordinates = [torch.arange(n, device=rho.device) for n in shape]
            centers = torch.stack(torch.meshgrid(*coordinates, indexing="ij"), dim=-1).reshape(-1, 3)
            axis = torch.arange(-self.cutoff_grid, self.cutoff_grid+1, device=rho.device)
            offsets = torch.cartesian_prod(axis, axis, axis)
            neighbors = (centers[:, None, :] + offsets).remainder(centers.new_tensor(shape))
            self._cube_index = _ravel_grid_indices(neighbors, shape)
            self._cube_shape = shape
        if canonical:
            return self._cube_index
        # flat maps input rows -> canonical positions; argsort inverts it.
        row_from_position = flat.argsort(dim=-1)
        neighbors = self._cube_index[flat].reshape(flat.shape[0], -1)
        return row_from_position.gather(1, neighbors).reshape(
            *rho.shape[:-2], rho.shape[-2], self.n_neighbors,
        )

    def forward(self, local_inputs: torch.Tensor) -> torch.Tensor:
        """Map ``[..., n_grid, n_features]`` to ``[..., n_grid, n_types]``."""

        if local_inputs.shape[-1] != self.n_features:
            raise ValueError("local_inputs must end with n_features entries")
        return self.mlp(local_inputs)

    def energy(self, context: ReadoutContext) -> torch.Tensor:
        """Gather live density patches and return one scalar per field."""

        rho = context.rho  # [..., n_grid, n_types], physical number density.
        if rho.ndim < 2 or rho.shape[-1] != self.n_types:
            raise ValueError("rho must have shape [..., n_grid, n_types]")
        index = (context.require("local_density_index") if self.domain == "sphere"
                 else self._cube_neighbor_indices(context))
        if index.shape[-2:] != (rho.shape[-2], self.n_neighbors):
            raise ValueError(
                "local_density_index must end with [n_grid, n_neighbors] "
                "for this cutoff_grid"
            )
        # A shared table can be used by a batch; loader-stacked tables already
        # have rho's leading field dimensions. Neither form copies densities.
        if index.ndim == 2:
            index = index.expand(*rho.shape[:-2], -1, -1)
        patch = gather_neighbors(context.normalized_rho(), index)
        inputs = torch.cat(
            (patch.flatten(start_dim=-2), context.temperature_feature()),
            dim=-1,
        )
        return context.integrate_local(self(inputs))
