"""Finite-range message passing between invariant grid environments."""

from typing import Optional, Sequence, Union

import torch
from torch import nn

from ._argument_checks import boolean, positive_integer
from ._grid import gather_neighbors
from ._nn import build_mlp
from .features import prepare_radial_centers, prepare_radial_exponents


class BChiMessage(nn.Module):
    r"""Convert invariant ``B`` features into the next equivariant ``A``.

    At layer ``t``, a shared neural map first constructs scalar gates at every
    grid point,

    ``g_i^t = h_t(B_i^t) - h_t(0)``.

    The gates are then convolved with a radial-Cartesian stencil basis,

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
    radial_exponents
        Optional damping exponents owned by this message layer. ``None``
        retains the initial ``CartesianAFeatures`` basis exactly. Supplying a
        sequence gives this layer an independent radial basis while retaining
        the same fixed stencil geometry and Cartesian monomials.
    trainable_radial_exponents
        Optimize the layer-owned positive exponents in logarithmic form.
    radial_centers
        Optional layer-owned Gaussian centers in grid units. ``None`` uses
        zero centers when ``radial_exponents`` are supplied and otherwise
        shares the complete initial basis.
    trainable_radial_centers
        Optimize layer-owned centers directly. The default is ``False``.
    """

    def __init__(
        self,
        n_invariant_features: int,
        n_radial_channels: int,
        n_channels: int,
        hidden_sizes: Sequence[int] = (32, 16),
        radial_exponents: Optional[
            Union[Sequence[float], torch.Tensor]
        ] = None,
        trainable_radial_exponents: bool = False,
        radial_centers: Optional[
            Union[Sequence[float], torch.Tensor]
        ] = None,
        trainable_radial_centers: bool = False,
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
        trainable_radial_exponents = boolean(
            trainable_radial_exponents,
            "trainable_radial_exponents",
        )
        trainable_radial_centers = boolean(
            trainable_radial_centers,
            "trainable_radial_centers",
        )
        if radial_exponents is None:
            if trainable_radial_exponents:
                raise ValueError(
                    "trainable_radial_exponents requires radial_exponents"
                )
            if radial_centers is not None or trainable_radial_centers:
                raise ValueError(
                    "radial_centers require layer-owned radial_exponents"
                )
            self.independent_radial_basis = False
        else:
            initial_radial_exponents = prepare_radial_exponents(
                "gaussian",
                radial_exponents,
                trainable_radial_exponents,
            )
            if initial_radial_exponents.numel() != self.n_radial_channels:
                raise ValueError(
                    "radial_exponents length must match n_radial_channels"
                )
            initial_radial_centers = prepare_radial_centers(
                "gaussian",
                radial_centers,
                self.n_radial_channels,
            )
            if trainable_radial_exponents:
                self.log_radial_exponents = nn.Parameter(
                    torch.log(initial_radial_exponents)
                )
            else:
                self.register_buffer(
                    "fixed_radial_exponents",
                    initial_radial_exponents,
                )
            if trainable_radial_centers:
                self.learned_radial_centers = nn.Parameter(
                    initial_radial_centers
                )
            else:
                self.register_buffer(
                    "fixed_radial_centers",
                    initial_radial_centers,
                    persistent=False,
                )
            self.independent_radial_basis = True
        self.trainable_radial_exponents = trainable_radial_exponents
        self.trainable_radial_centers = trainable_radial_centers
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

    @property
    def radial_exponents(self) -> Optional[torch.Tensor]:
        """Return layer-owned exponents, or ``None`` for the shared basis."""

        if not getattr(self, "independent_radial_basis", False):
            return None
        if getattr(self, "trainable_radial_exponents", False):
            return torch.exp(self.log_radial_exponents)
        return self.fixed_radial_exponents

    @property
    def radial_centers(self) -> Optional[torch.Tensor]:
        """Return layer-owned centers, or ``None`` for the shared basis."""

        if not getattr(self, "independent_radial_basis", False):
            return None
        if getattr(self, "trainable_radial_centers", False):
            return self.learned_radial_centers
        stored = self._buffers.get("fixed_radial_centers")
        if stored is not None:
            return stored
        return torch.zeros_like(self.radial_exponents)

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
