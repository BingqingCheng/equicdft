"""Finite-range message passing between invariant grid environments."""

from typing import Any, Dict, Optional, Sequence, Union

import torch
from torch import nn

from ._argument_checks import boolean, positive_integer
from ._config import Configurable, make_config, register
from ._grid import gather_neighbors, periodic_stencil_convolution
from ._nn import build_mlp, validate_hidden_sizes
from ._radial import (
    _RadialTransform,
    prepare_radial_centers,
    prepare_radial_exponents,
)
from .features import CartesianAFeatures


@register
class BChiMessage(nn.Module, Configurable):
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
    radial_basis
        Optional explicit radial-basis selection. ``None`` preserves the
        established behavior: the initial ``CartesianAFeatures`` basis is
        shared when ``radial_exponents`` is absent, and an independent
        Gaussian basis is used when exponents are supplied. Explicit
        ``"shared"``, ``"gaussian"``, and ``"bessel"`` selections are also
        accepted; explicit Gaussian mode requires exponents. Bessel mode gives
        this message an independently transformed, conditioned basis on the
        initial feature geometry.
    n_radial_functions
        Number of fixed primitive Bessel functions before the message-owned
        transform. Required only when ``radial_basis="bessel"``.
    convolution_backend
        ``"gather"`` retains the established explicit-neighborhood
        contraction. ``"conv3d"`` evaluates the same periodic stencil as a
        grouped dense convolution, and ``"fft"`` evaluates it as a circular
        Fourier cross-correlation. Neither alternative materializes
        ``[G, J, N, C]``.
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
        radial_basis: Optional[str] = None,
        n_radial_functions: Optional[int] = None,
        convolution_backend: str = "gather",
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
        if radial_basis is None:
            resolved_radial_basis = (
                "shared" if radial_exponents is None else "gaussian"
            )
        else:
            if not isinstance(radial_basis, str):
                raise TypeError(
                    "radial_basis must be 'bessel', 'gaussian', "
                    "'shared', or None"
                )
            resolved_radial_basis = radial_basis.lower()
            if resolved_radial_basis not in (
                "bessel",
                "gaussian",
                "shared",
            ):
                raise ValueError(
                    "radial_basis must be 'bessel', 'gaussian', "
                    "'shared', or None"
                )

        if resolved_radial_basis == "shared":
            if radial_exponents is not None:
                raise ValueError(
                    "radial_exponents are unavailable when "
                    "radial_basis='shared'"
                )
            if trainable_radial_exponents:
                raise ValueError(
                    "trainable_radial_exponents requires radial_exponents"
                )
            if radial_centers is not None or trainable_radial_centers:
                raise ValueError(
                    "radial_centers require layer-owned radial_exponents"
                )
        elif resolved_radial_basis == "gaussian":
            if radial_exponents is None:
                raise ValueError(
                    "radial_basis='gaussian' requires radial_exponents"
                )
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
        else:
            if radial_exponents is not None or trainable_radial_exponents:
                raise ValueError(
                    "Gaussian radial exponents are unavailable when "
                    "radial_basis='bessel'"
                )
            if radial_centers is not None or trainable_radial_centers:
                raise ValueError(
                    "Gaussian radial centers are unavailable when "
                    "radial_basis='bessel'"
                )
            if n_radial_functions is None:
                raise ValueError(
                    "n_radial_functions is required when "
                    "radial_basis='bessel'"
                )
            self.n_radial_functions = positive_integer(
                n_radial_functions,
                "n_radial_functions",
            )
            if self.n_radial_channels > self.n_radial_functions:
                raise ValueError(
                    "n_radial_channels must not exceed "
                    "n_radial_functions"
                )
        if (
            resolved_radial_basis != "bessel"
            and n_radial_functions is not None
        ):
            raise ValueError(
                "n_radial_functions requires radial_basis='bessel'"
            )
        self.radial_basis = resolved_radial_basis
        # A Bessel basis is conditioned on the feature geometry of the
        # containing model, so its buffers and transform are bound later by
        # GridCACEModel through _bind_bessel_basis. The slots exist from
        # construction; None marks an unbound layer.
        self.radial_transform = None
        self.register_buffer("fixed_bessel_stencil_basis", None)
        self.register_buffer("bessel_gram_eigenvalues", None)
        self._bessel_geometry_signature = None
        if not isinstance(convolution_backend, str):
            raise TypeError(
                "convolution_backend must be 'gather', 'conv3d', or 'fft'"
            )
        convolution_backend = convolution_backend.lower()
        if convolution_backend not in ("gather", "conv3d", "fft"):
            raise ValueError(
                "convolution_backend must be 'gather', 'conv3d', or 'fft'"
            )
        self.convolution_backend = convolution_backend
        self.trainable_radial_exponents = trainable_radial_exponents
        self.trainable_radial_centers = trainable_radial_centers
        self.hidden_sizes = validate_hidden_sizes(hidden_sizes)
        self.n_input_features = (
            self.n_radial_channels
            * self.n_invariant_features
            * self.n_channels
        )
        self.n_output_features = self.n_radial_channels * self.n_channels
        self.mlp = build_mlp(
            input_size=self.n_input_features,
            hidden_sizes=self.hidden_sizes,
            output_size=self.n_output_features,
        )

    def to_config(self) -> Dict[str, Any]:
        """Return the constructor arguments describing this layer.

        The radial basis is recorded explicitly. Gaussian exponents are
        recorded because their count is structural; fixed Gaussian centers
        are recorded because they are not persistent state, while trainable
        centers are restored from the ``state_dict``. A Bessel basis bound by
        the containing model is rebuilt when the model is reconstructed.
        """

        gaussian = self.radial_basis == "gaussian"
        bessel = self.radial_basis == "bessel"
        return make_config(
            self,
            n_invariant_features=self.n_invariant_features,
            n_radial_channels=self.n_radial_channels,
            n_channels=self.n_channels,
            hidden_sizes=self.hidden_sizes,
            radial_exponents=self.radial_exponents if gaussian else None,
            trainable_radial_exponents=self.trainable_radial_exponents,
            radial_centers=(
                self.radial_centers
                if gaussian and not self.trainable_radial_centers
                else None
            ),
            trainable_radial_centers=self.trainable_radial_centers,
            radial_basis=self.radial_basis,
            n_radial_functions=self.n_radial_functions if bessel else None,
            convolution_backend=self.convolution_backend,
        )

    @property
    def radial_exponents(self) -> Optional[torch.Tensor]:
        """Return Gaussian exponents, or ``None`` for shared/Bessel bases."""

        if self.radial_basis != "gaussian":
            return None
        if self.trainable_radial_exponents:
            return torch.exp(self.log_radial_exponents)
        return self.fixed_radial_exponents

    @property
    def radial_centers(self) -> Optional[torch.Tensor]:
        """Return Gaussian centers, or ``None`` for shared/Bessel bases."""

        if self.radial_basis != "gaussian":
            return None
        if self.trainable_radial_centers:
            return self.learned_radial_centers
        return self.fixed_radial_centers

    @property
    def bessel_basis_is_bound(self) -> bool:
        """Whether a Bessel layer has received its feature geometry."""

        return (
            self.fixed_bessel_stencil_basis is not None
            and self.radial_transform is not None
        )

    def _bind_bessel_basis(self, a_features: nn.Module) -> None:
        """Bind an independent Bessel basis to the model's grid geometry."""

        if self.radial_basis != "bessel":
            return
        if not isinstance(a_features, CartesianAFeatures):
            raise TypeError(
                "Bessel message layers require CartesianAFeatures geometry"
            )
        geometry_signature = (
            int(a_features.cutoff_grid),
            int(a_features.max_power),
            str(a_features.coordinate_scaling),
            bool(a_features.separate_center),
        )
        if self.bessel_basis_is_bound:
            if self._bessel_geometry_signature != geometry_signature:
                raise ValueError(
                    "Bessel message layer is already bound to an "
                    "incompatible feature geometry"
                )
            return
        basis, eigenvalues = a_features._bessel_stencil_basis(
            self.n_radial_functions
        )
        self.register_buffer("fixed_bessel_stencil_basis", basis)
        self.register_buffer("bessel_gram_eigenvalues", eigenvalues)
        self.radial_transform = _RadialTransform(
            max_power=a_features.max_power,
            n_radial_functions=self.n_radial_functions,
            n_radial_channels=self.n_radial_channels,
        ).to(device=basis.device, dtype=basis.dtype)
        self._bessel_geometry_signature = geometry_signature

    def _stencil_basis(
        self,
        a_features: nn.Module,
        shared_basis: torch.Tensor,
    ) -> torch.Tensor:
        """Return the shared, Gaussian, or message-owned Bessel stencil."""

        if self.radial_basis == "shared":
            return shared_basis
        if self.radial_basis == "gaussian":
            return a_features.stencil_basis(
                self.radial_exponents,
                self.radial_centers,
            )
        if not self.bessel_basis_is_bound:
            raise RuntimeError(
                "Bessel message layer has not been bound to feature geometry"
            )
        return self.radial_transform(
            self.fixed_bessel_stencil_basis,
            a_features.powers,
        )

    def _apply_stencil(
        self,
        gates: torch.Tensor,
        local_density_index: Optional[torch.Tensor],
        stencil_basis: torch.Tensor,
        *,
        grid_positions: Optional[torch.Tensor],
        grid_size: Optional[torch.Tensor],
        stencil_positions: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply the configured periodic stencil contraction."""

        if self.convolution_backend in ("conv3d", "fft"):
            if (
                grid_positions is None
                or grid_size is None
                or stencil_positions is None
            ):
                raise ValueError(
                    f"{self.convolution_backend} messages require "
                    "grid_positions, grid_size, and stencil_positions"
                )
            return periodic_stencil_convolution(
                gates,
                stencil_basis,
                grid_positions,
                grid_size,
                stencil_positions,
                backend=self.convolution_backend,
            )
        if self.convolution_backend != "gather":
            raise RuntimeError(
                "convolution_backend must be 'gather', 'conv3d', or 'fft'"
            )
        if local_density_index is None:
            raise ValueError(
                "gather messages require local_density_index; reload the "
                "data with include_local_density_index=True or use a "
                "conv3d/fft message backend"
            )
        if stencil_basis.shape[0] != local_density_index.shape[-1]:
            raise ValueError(
                "stencil_basis neighbor count does not match "
                "local_density_index"
            )
        local_gates = gather_neighbors(gates, local_density_index)
        return torch.einsum(
            "...gjnc,jnk->...gnkc",
            local_gates,
            stencil_basis,
        )

    def forward(
        self,
        B: torch.Tensor,
        local_density_index: Optional[torch.Tensor],
        stencil_basis: torch.Tensor,
        *,
        grid_positions: Optional[torch.Tensor] = None,
        grid_size: Optional[torch.Tensor] = None,
        stencil_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return ``A^(t+1)`` with shape ``[..., G, N, K, C]``.

        ``B`` must have shape ``[..., G, N, Q, C]`` and ``stencil_basis`` must
        have shape ``[J, N, K]``. The periodic neighbor table has shape
        ``[..., G, J]``. The ``"conv3d"`` and ``"fft"`` backends require the
        complete grid coordinates, grid size, and matching stencil offsets;
        for those backends ``local_density_index`` may be ``None``.
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
        flat_B = B.flatten(start_dim=-3)
        zero_B = flat_B.new_zeros(self.n_input_features)
        gates = self.mlp(flat_B) - self.mlp(zero_B)
        gates = gates.reshape(
            *B.shape[:-3],
            self.n_radial_channels,
            self.n_channels,
        )
        return self._apply_stencil(
            gates,
            local_density_index,
            stencil_basis,
            grid_positions=grid_positions,
            grid_size=grid_size,
            stencil_positions=stencil_positions,
        )
