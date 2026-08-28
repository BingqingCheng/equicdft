"""Finite-range message passing between invariant grid environments."""

from typing import Sequence

import torch
from torch import nn

from ._argument_checks import positive_integer
from ._grid import gather_neighbors
from ._nn import build_mlp


class BChiMessage(nn.Module):
    r"""Convert invariant ``B`` features into the next equivariant ``A``.

    At layer ``t``, a shared neural map first constructs scalar gates at every
    grid point,

    ``g_i^t = h_t(B_i^t) - h_t(0)``.

    The gates are then convolved with the same radial-Cartesian stencil basis
    used for the initial density features,

    ``A_i^(t+1)[n,k,c] = sum_j Phi[j,n,k] g_(i+j)^t[n,c]``.

    Here ``n`` labels radial channels, ``k`` Cartesian monomials, and ``c``
    latent channels. The gate is invariant under the signed axis permutations
    of the cubic lattice, while ``Phi`` is equivariant. Consequently the next
    ``A`` transforms exactly like the initial Cartesian moments. Subtracting
    ``h_t(0)`` makes an identically zero input produce an identically zero
    message without constraining the ordinary neural-network initialization.

    This module returns the message itself: the model update is exactly
    ``A^(t+1) = M^t``. Earlier invariant ``B`` levels are retained separately
    by :class:`equicdft.model.GridCACEModel` for the final readout.

    Parameters
    ----------
    n_invariant_features
        Number of invariant features in the ``B`` axis.
    n_radial_channels
        Number of radial channels shared with ``CartesianAFeatures``.
    n_channels
        Number of physical or latent density channels.
    hidden_sizes
        Width of each gate-network hidden layer. An empty sequence gives a
        linear gate.
    """

    def __init__(
        self,
        n_invariant_features: int,
        n_radial_channels: int,
        n_channels: int,
        hidden_sizes: Sequence[int] = (32, 16),
    ) -> None:
        super().__init__()

        self.n_invariant_features = positive_integer(
            n_invariant_features,
            "n_invariant_features",
        )
        self.n_radial_channels = positive_integer(
            n_radial_channels,
            "n_radial_channels",
        )
        self.n_channels = positive_integer(n_channels, "n_channels")
        self.n_input_features = (
            self.n_radial_channels
            * self.n_invariant_features
            * self.n_channels
        )
        self.n_output_features = self.n_radial_channels * self.n_channels
        self.mlp = build_mlp(
            input_size=self.n_input_features,
            hidden_sizes=hidden_sizes,
            output_size=self.n_output_features,
        )

    def forward(
        self,
        B: torch.Tensor,
        local_density_index: torch.Tensor,
        stencil_basis: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``A^(t+1)`` with shape ``[..., G, N, K, C]``.

        ``B`` must have shape ``[..., G, N, Q, C]`` and ``stencil_basis`` must
        have shape ``[J, N, K]``. The periodic neighbor table has shape
        ``[..., G, J]``.
        """

        if B.ndim < 4:
            raise ValueError(
                "B must have shape "
                "[..., n_grid, n_radial_channels, "
                "n_invariant_features, n_channels]"
            )
        expected_tail = (
            self.n_radial_channels,
            self.n_invariant_features,
            self.n_channels,
        )
        if B.shape[-3:] != expected_tail:
            raise ValueError(
                "B trailing shape must be {}, got {}".format(
                    expected_tail,
                    tuple(B.shape[-3:]),
                )
            )
        if stencil_basis.ndim != 3:
            raise ValueError("stencil_basis must have shape [J, N, K]")
        if stencil_basis.shape[1] != self.n_radial_channels:
            raise ValueError(
                "stencil_basis radial channels do not match this message layer"
            )
        if stencil_basis.shape[0] != local_density_index.shape[-1]:
            raise ValueError(
                "stencil_basis neighbor count does not match local_density_index"
            )

        flat_B = B.flatten(start_dim=-3)
        zero_B = flat_B.new_zeros(self.n_input_features)
        gates = self.mlp(flat_B) - self.mlp(zero_B)
        gates = gates.reshape(
            *B.shape[:-3],
            self.n_radial_channels,
            self.n_channels,
        )
        local_gates = gather_neighbors(gates, local_density_index)

        return torch.einsum(
            "...gjnc,jnk->...gnkc",
            local_gates,
            stencil_basis,
        )
