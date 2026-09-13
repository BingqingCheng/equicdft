"""Local readout for invariant grid features."""

from typing import Dict, Optional, Sequence

import torch

from ._argument_checks import (
    boolean,
    finite_scalar,
    optional_positive_integer,
    positive_integer,
)
from ._component_pairs import symmetric_component_pairs
from ._nn import build_mlp
from .energy import EnergyReadout, density_weighted_integral
from .interaction import BChiMessage
from .polarization_features import PolarizationFeatures
from .reciprocal import ReciprocalFeatures
from .symmetrize import CartesianBFeatures


class LocalReadout(EnergyReadout):
    """Map local invariant features to one output per physical component.

    A shared MLP maps a local feature vector at each grid point to one output
    per component. The readout itself is agnostic about the construction and
    physical meaning of its inputs and outputs.
    :class:`equicdft.model.GridCACEModel` supplies flattened invariant B
    features and temperature, and interprets the outputs as reduced
    per-particle excess free energies according to its model-level convention.

    Its :meth:`energy` method performs density weighting and grid integration;
    :class:`equicdft.model.GridCACEModel` only sums scalar readout energies and
    differentiates their total.

    Parameters
    ----------
    n_features
        Width of the complete local input vector. If supplied, the first layer
        is a regular ``Linear`` module. If ``None``, a ``LazyLinear``
        module infers this width from the first forward pass.
    n_types
        Number of physical density components and per-particle outputs.
    hidden_sizes
        Width of each hidden layer. An empty sequence gives a linear readout.
    Notes
    -----
    When ``n_features`` is ``None``, materialize the lazy input layer with one
    representative batch before constructing an optimizer or saving its
    initial state.

    Smooth SiLU activations keep higher functional derivatives well defined.
    """

    requires_local_features = True

    def __init__(
        self,
        n_types: int = 1,
        hidden_sizes: Sequence[int] = (32, 16),
        n_features: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.n_features = optional_positive_integer(n_features, "n_features")
        self.n_types = positive_integer(n_types, "n_types")
        self.mlp = build_mlp(
            self.n_features,
            hidden_sizes,
            self.n_types,
        )

    def forward(
        self,
        local_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return local outputs with shape ``[..., n_grid, n_types]``."""

        return self.mlp(local_features)

    def energy(
        self,
        context: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return the integrated density-weighted local contribution."""

        rho = context["rho"]
        per_particle = self(context["local_features"])
        if per_particle.shape != rho.shape:
            raise ValueError(
                "LocalReadout must return one value per grid and type"
            )
        return density_weighted_integral(
            rho,
            per_particle,
            context["voxel_volume"],
        )


class PolarizationReadout(EnergyReadout):
    """Integrate a local excess free energy from scalar/vector invariants.

    ``features`` constructs invariant combinations of the live number-density
    and electric dipole-density fields. A shared MLP receives these invariants
    followed by normalized temperature and returns one reduced free energy
    per particle and component. No orientational ideal entropy or external
    electric-field coupling is included here.

    Compatibility path for existing checkpoints and scalar-invariant messages.
    New zero-message models should pass PolarizationAFeatures and
    PolarizationBFeatures to GridCACEModel and use ordinary LocalReadout.

    Optional ``message`` adds one B-chi aggregation of the complete joint
    invariant vector B0. A single scalar latent gate h(B0)-h(0) is convolved
    into Cartesian moments and contracted by CartesianBFeatures into B1.
    The readout receives [B0, B1, T/T_ref]. This passes polarization-dependent
    invariant information, not explicit vector-valued messages. The descriptor
    receptive radius doubles; the strictly local LDA is unaffected.
    """

    requires_dipole_density = True
    requires_local_density_index = True

    def __init__(
        self,
        features: PolarizationFeatures,
        hidden_sizes: Sequence[int] = (32, 16),
        message: Optional[BChiMessage] = None,
    ) -> None:
        super().__init__()
        if not isinstance(features, PolarizationFeatures):
            raise TypeError("features must be PolarizationFeatures")
        self.features = features
        width = features.n_features + 1
        if message is not None:
            if not isinstance(message, BChiMessage):
                raise TypeError("message must be a BChiMessage")
            if (message.n_radial_channels, message.n_invariant_features,
                    message.n_channels) != (1, features.n_features, 1):
                raise ValueError("joint message requires one radial/latent channel and all joint invariants")
            if message.convolution_backend != "gather":
                raise ValueError("polarization messages currently require the gather backend")
            if (message._radial_basis_kind() == "shared"
                    and features.radial_exponents.numel() != 1):
                raise ValueError("shared joint-message basis requires one radial channel")
            message._bind_bessel_basis(features.scalar_features)
            self.message = message
            self.message_invariants = CartesianBFeatures(
                features.max_power, features.max_product_order)
            width += self.message_invariants.n_features
        self.mlp = build_mlp(
            width,
            hidden_sizes,
            features.n_types,
        )

    @property
    def n_types(self) -> int:
        """Number of physical number-density/dipole-density components."""

        return self.features.n_types

    @property
    def mean_density(self) -> torch.Tensor:
        """Fixed number-density normalization used by the descriptor."""

        return self.features.mean_density

    @property
    def cutoff_grid(self) -> int:
        """Inclusive stencil cutoff, measured in grid steps."""

        return self.features.cutoff_grid

    def forward(self, context: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return per-particle outputs with shape ``[..., n_grid, n_types]``."""

        invariants = self.features(context)
        message = getattr(self, "message", None)
        if message is not None:
            geometry = self.features.scalar_features
            basis = message._stencil_basis(geometry, geometry.stencil_basis())
            moments = message(invariants.unsqueeze(-2).unsqueeze(-1),
                              context["local_density_index"], basis)
            next_invariants = self.message_invariants(moments).flatten(start_dim=-3)
            invariants = torch.cat((invariants, next_invariants), dim=-1)
        temperature = context["normalized_temperature"][..., None, None]
        temperature = temperature.expand(*invariants.shape[:-1], 1)
        return self.mlp(torch.cat((invariants, temperature), dim=-1))

    def energy(self, context: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return ``Delta V * sum_(g,a) rho[g,a] * a_exc[g,a]``."""

        return density_weighted_integral(
            context["rho"],
            self(context),
            context["voxel_volume"],
        )


class BulkReadout(EnergyReadout):
    """Map temperature and mean densities to a bulk free energy per particle.

    The state vector contains normalized temperature followed by one
    normalized mean density per physical component. The output contains one
    reduced bulk excess free energy per particle and component. The model
    combines it with the particle numbers according to

    ``E_exc_bulk = sum_i N_i * a_exc_bulk_i``,

    where ``E_exc_bulk`` follows the free-energy convention selected by the
    containing model.

    Parameters
    ----------
    n_types
        Number of physical density components.
    hidden_sizes
        Width of each hidden layer. An empty sequence gives a linear readout.
    zero_init
        If true, initialize the final layer to zero. Attaching the branch then
        leaves a pretrained local model unchanged before fine-tuning.
    """

    requires_state_features = True

    def __init__(
        self,
        n_types: int = 1,
        hidden_sizes: Sequence[int] = (16, 16),
        zero_init: bool = True,
    ) -> None:
        super().__init__()

        self.n_types = positive_integer(n_types, "n_types")
        self.n_state_features = 1 + self.n_types
        self.mlp = build_mlp(
            self.n_state_features,
            hidden_sizes,
            self.n_types,
            zero_init=zero_init,
        )

    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        """Return ``a_exc_bulk`` with shape ``[..., n_types]``."""

        if state_features.shape[-1] != self.n_state_features:
            raise ValueError(
                "state_features must end with normalized temperature and "
                "one mean density per type"
            )
        return self.mlp(state_features)

    def energy(
        self,
        context: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return the extensive homogeneous free-energy contribution."""

        rho = context["rho"]
        per_particle = self(context["state_features"])
        if per_particle.shape != rho.mean(dim=-2).shape:
            raise ValueError(
                "BulkReadout must return one value per field and type"
            )
        return density_weighted_integral(
            rho,
            per_particle.unsqueeze(-2),
            context["voxel_volume"],
        )


class LongRangeReadout(EnergyReadout):
    """Map thermodynamic state to a reciprocal quadratic kernel.

    The readout predicts one coefficient for every fixed reciprocal kernel and
    unique density-component pair. Its output energy is a linear contraction
    with the reciprocal features, preserving their quadratic density
    dependence and extensive scaling.

    Parameters
    ----------
    n_kernels
        Number of fixed radial kernels in the reciprocal representation.
    n_types
        Number of physical density components. The state vector contains
        normalized temperature followed by one mean density per component.
    hidden_sizes
        Width of each state-network hidden layer. An empty sequence gives a
        linear state dependence.
    zero_init
        If true, initialize the final coefficient layer to zero. This makes an
        attached long-range branch leave a pretrained local model unchanged.
    charges
        Optional fixed charge or valency of each density component. When
        supplied, pair coefficients are constrained to ``A * q_i * q_j``.
        With no ``coulomb_amplitude``, the state network learns the single
        shared amplitude ``A``. Charge-factorized mode currently requires one
        reciprocal kernel.
    coulomb_amplitude
        Optional fixed scalar ``A`` multiplying the charge products. It
        requires ``charges``. When supplied, the long-range coefficients have
        no trainable parameters. In a beta-free-energy Coulomb functional this
        is the Bjerrum length expressed in the coordinate units.
    features
        Reciprocal feature module used by :meth:`energy`. It may be omitted
        when the readout is used only as a standalone coefficient contraction.
    include_polarization
        If true, require ``dipole_density`` and use the same Coulomb kernel
        on ``q_a rho_hat_a - i k.P_hat_a``. Requires explicit ``charges``
        and Coulomb ``features``. Pair features already contain the charges,
        so every pair coefficient is the shared amplitude A, not A*q_i*q_j.
        With ``free_energy_mode="beta"``, use A=beta*C in compatible physical
        units at fixed temperature. For variable temperatures, use the
        model's physical mode with A=C/(k_B*T_ref). No dielectric scaling,
        dipole-magnitude factor, or particle self subtraction is implicit.
    """

    requires_local_density_index = False

    @property
    def requires_state_features(self) -> bool:
        return getattr(self, "coulomb_amplitude", None) is None

    @property
    def requires_dipole_density(self) -> bool:
        # Old serialized charge-only readouts have no include_polarization.
        return getattr(self, "include_polarization", False)

    def __init__(
        self,
        n_kernels: int,
        n_types: int = 1,
        hidden_sizes: Sequence[int] = (16, 16),
        zero_init: bool = True,
        charges: Optional[Sequence[float]] = None,
        coulomb_amplitude: Optional[float] = None,
        features: Optional[ReciprocalFeatures] = None,
        include_polarization: bool = False,
    ) -> None:
        super().__init__()

        self.include_polarization = boolean(
            include_polarization, "include_polarization",
        )
        if self.include_polarization and (
            charges is None
            or not isinstance(features, ReciprocalFeatures)
            or features.kernel != "coulomb"
        ):
            raise ValueError(
                "include_polarization requires charges and Coulomb features"
            )
        self.n_kernels = positive_integer(n_kernels, "n_kernels")
        self.n_types = positive_integer(n_types, "n_types")
        type_pairs = symmetric_component_pairs(self.n_types)
        self.n_type_pairs = len(type_pairs)
        self.n_state_features = 1 + self.n_types

        if charges is None:
            if coulomb_amplitude is not None:
                raise ValueError("coulomb_amplitude requires charges")
            charge_tensor = None
            pair_charge_products = None
        else:
            charge_tensor = torch.as_tensor(
                charges,
                dtype=torch.get_default_dtype(),
            ).detach().clone().reshape(-1)
            if charge_tensor.shape != (self.n_types,):
                raise ValueError("charges must contain one value per type")
            if not torch.all(torch.isfinite(charge_tensor)).item():
                raise ValueError("charges must be finite")
            if self.n_kernels != 1:
                raise ValueError(
                    "charge-factorized long-range mode requires n_kernels=1"
                )
            pair_charge_products = torch.tensor(
                [
                    charge_tensor[first] * charge_tensor[second]
                    for first, second in type_pairs
                ],
                dtype=charge_tensor.dtype,
            )

        if coulomb_amplitude is not None:
            coulomb_amplitude = finite_scalar(
                coulomb_amplitude,
                "coulomb_amplitude",
            )
            amplitude_tensor = torch.tensor(
                coulomb_amplitude,
                dtype=torch.get_default_dtype(),
            )
        else:
            amplitude_tensor = None

        self.register_buffer("charges", charge_tensor)
        self.register_buffer("pair_charge_products", pair_charge_products)
        self.register_buffer("coulomb_amplitude", amplitude_tensor)
        if features is not None:
            if not isinstance(features, ReciprocalFeatures):
                raise TypeError("features must be ReciprocalFeatures or None")
            if features.n_types != self.n_types:
                raise ValueError("features and readout n_types differ")
            if features.n_kernels != self.n_kernels:
                raise ValueError("features and readout kernel counts differ")
        self.features = features

        if coulomb_amplitude is not None:
            self.mlp = None
        else:
            output_width = (
                self.n_kernels * self.n_type_pairs
                if charges is None
                else 1
            )
            self.mlp = build_mlp(
                self.n_state_features,
                hidden_sizes,
                output_width,
                zero_init=zero_init,
            )

    def coefficients(self, state_features: torch.Tensor) -> torch.Tensor:
        """Return coefficients shaped ``[..., n_kernels, n_type_pairs]``."""

        if state_features.shape[-1] != self.n_state_features:
            raise ValueError(
                "state_features must end with normalized temperature and "
                "one mean density per type"
            )
        charges = getattr(self, "charges", None)
        if charges is None:
            return self.mlp(state_features).reshape(
                *state_features.shape[:-1],
                self.n_kernels,
                self.n_type_pairs,
            )

        amplitude = getattr(self, "coulomb_amplitude", None)
        if amplitude is None:
            amplitude = self.mlp(state_features).reshape(
                *state_features.shape[:-1],
                1,
                1,
            )
        else:
            amplitude = amplitude.to(state_features).expand(
                *state_features.shape[:-1],
                1,
                1,
            )
        pair_weights = self.pair_charge_products.to(state_features)
        if self.requires_dipole_density:
            # q is already in each source; applying q_i*q_j here would both
            # double-count charge factors and erase neutral-molecule dipoles.
            pair_weights = torch.ones_like(pair_weights)
        return amplitude * pair_weights.view(
            1,
            self.n_type_pairs,
        )

    def forward(
        self,
        reciprocal_features: torch.Tensor,
        state_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return one reduced long-range contribution per field."""

        expected_trailing_shape = (self.n_kernels, self.n_type_pairs)
        if reciprocal_features.shape[-2:] != expected_trailing_shape:
            raise ValueError(
                "reciprocal_features must end with shape {}".format(
                    expected_trailing_shape
                )
            )
        if reciprocal_features.shape[:-2] != state_features.shape[:-1]:
            raise ValueError(
                "reciprocal and state features must have matching leading shapes"
            )
        coefficients = self.coefficients(state_features)
        return torch.sum(
            coefficients * reciprocal_features,
            dim=(-2, -1),
        )

    def energy(
        self,
        context: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return the reciprocal-space contribution for a complete field."""

        if self.features is None:
            raise ValueError(
                "LongRangeReadout requires ReciprocalFeatures for model use"
            )
        if "grid_size" not in context:
            raise KeyError(
                "long-range evaluation requires data['grid_size']"
            )
        source_options = {}
        if self.requires_dipole_density:
            source_options = {
                "dipole_density": context["dipole_density"],
                "charges": self.charges,
            }
        reciprocal_features = self.features(
            rho=context["rho"],
            grid_size=context["grid_size"],
            grid_spacing=context["grid_spacing"],
            **source_options,
        )
        if self.requires_state_features:
            state_features = context["state_features"]
        else:
            # Fixed electrostatics needs no learned density normalization.
            state_features = context["rho"].new_zeros(
                *context["rho"].shape[:-2], self.n_state_features,
            )
        energy = self(reciprocal_features, state_features)
        if energy.shape != context["rho"].shape[:-2]:
            raise ValueError(
                "LongRangeReadout must return one scalar per field"
            )
        return energy
