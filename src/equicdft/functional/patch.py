"""Unconstrained neural readout of an ordered local density patch."""

from typing import Any, Dict, Sequence

import torch

from .._config import Configurable, make_config, register
from .._nn import build_mlp, validate_hidden_sizes
from .._validation import nonnegative_integer, positive_integer
from ..geometry.convolution import gather_neighbors
from ..geometry.stencil import make_stencil
from .base import EnergyReadout, ReadoutContext

__all__ = ["DensityPatchReadout"]


@register
class DensityPatchReadout(EnergyReadout, Configurable):
    """Map raw periodic density patches and temperature to local energies.

    The MLP input at grid point g is the flattened list
    ``[rho[g+q, a] / mean_density]`` followed by ``T / mean_temperature``.
    Offsets q follow :func:`make_stencil`: the center occurs once, and all
    integer offsets with ``|q|^2 <= cutoff_grid^2`` are retained. Species a
    vary fastest within each offset. At cutoff 4 a one-component patch has
    257 density inputs, plus one temperature input.

    Unlike an invariant descriptor, the ordered patch gives each direction
    its own network weights. Translation symmetry follows from sharing the
    MLP over all centers; rotations and reflections are not enforced.

    The output has one per-particle value per species. Its integral is
    ``DeltaV * sum_g,a rho[g,a] * output[g,a]``. The containing model selects
    whether this is beta F_exc or F_exc/(k_B T_ref). Density normalization
    uses the model-owned scale, while integration uses physical density.
    Gathering never detaches rho, so derivatives include every overlapping
    patch. Smooth SiLU hidden layers permit c1 and higher derivatives.

    This readout needs no A/B features or separate LDA contribution. It uses
    the model's shared neighbor table and requires matching cutoff_grid.
    The patch width is fixed: changing resolution is not an automatic
    resolution-transfer rule for this baseline.
    """

    requires_local_density_index = True
    shares_grid_stencil = True
    reads_smoothed_fields = True

    def __init__(
        self,
        n_types: int = 1,
        *,
        cutoff_grid: int = 4,
        hidden_sizes: Sequence[int] = (32, 16),
    ) -> None:
        super().__init__()
        self.n_types = positive_integer(n_types, "n_types")
        self.cutoff_grid = nonnegative_integer(cutoff_grid, "cutoff_grid")
        self.hidden_sizes = validate_hidden_sizes(hidden_sizes)
        self.n_neighbors = len(make_stencil(self.cutoff_grid))
        self.n_features = self.n_neighbors * self.n_types + 1
        self.mlp = build_mlp(
            self.n_features, self.hidden_sizes, self.n_types,
        )

    def to_config(self) -> Dict[str, Any]:
        """Return constructor arguments; fitted weights live in state_dict."""

        return make_config(
            self,
            n_types=self.n_types,
            cutoff_grid=self.cutoff_grid,
            hidden_sizes=self.hidden_sizes,
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
        index = context.require("local_density_index")
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
