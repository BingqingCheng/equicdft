"""Local readout for invariant grid features."""

from typing import Dict, Optional, Sequence

import torch

from ._argument_checks import (
    finite_scalar,
    optional_positive_integer,
    positive_integer,
)
from ._nn import build_mlp
from .energy import EnergyReadout
from .reciprocal import ReciprocalFeatures


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
        free_energy_density = torch.sum(rho * per_particle, dim=-1)
        return context["cell_volume"] * torch.sum(
            free_energy_density,
            dim=-1,
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
        particle_numbers = context["cell_volume"] * torch.sum(rho, dim=-2)
        return torch.sum(particle_numbers * per_particle, dim=-1)


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
    """

    requires_state_features = True

    def __init__(
        self,
        n_kernels: int,
        n_types: int = 1,
        hidden_sizes: Sequence[int] = (16, 16),
        zero_init: bool = True,
        charges: Optional[Sequence[float]] = None,
        coulomb_amplitude: Optional[float] = None,
        features: Optional[ReciprocalFeatures] = None,
    ) -> None:
        super().__init__()

        self.n_kernels = positive_integer(n_kernels, "n_kernels")
        self.n_types = positive_integer(n_types, "n_types")
        self.n_type_pairs = self.n_types * (self.n_types + 1) // 2
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
                    for first in range(self.n_types)
                    for second in range(first, self.n_types)
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
        return amplitude * self.pair_charge_products.to(state_features).view(
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
        reciprocal_features = self.features(
            rho=context["rho"],
            grid_size=context["grid_size"],
            grid_spacing=context["grid_spacing"],
        )
        energy = self(reciprocal_features, context["state_features"])
        if energy.shape != context["rho"].shape[:-2]:
            raise ValueError(
                "LongRangeReadout must return one scalar per field"
            )
        return energy
