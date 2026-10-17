"""Energy readouts for grid features."""

from typing import Any, Dict, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

import torch

from .._validation import (
    boolean,
    finite_scalar,
    optional_positive_integer,
    positive_integer,
)
from ._pairs import symmetric_component_pairs
from ._density import density_fields
from .._config import (
    Configurable,
    build_optional,
    constructor_arguments,
    make_config,
    optional_buffer_value,
    optional_config,
    register,
)
from .._device import module_device
from .._nn import build_mlp, validate_hidden_sizes
from .base import EnergyReadout, ReadoutContext
from .reciprocal import ReciprocalFeatures
from .thermodynamics import density_weighted_integral

__all__ = [
    "LocalReadout",
    "BulkReadout",
    "LongRangeReadout",
]


class _LiquidCoulombResult(NamedTuple):
    """Coulomb quantities needed to couple a liquid to external charges."""

    energy: torch.Tensor
    potential: torch.Tensor
    amplitude: torch.Tensor
    total_charge: torch.Tensor


@register
class LocalReadout(EnergyReadout, Configurable):
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
        Width of the complete local input vector. When omitted, the readout
        stays unbound until :meth:`bind_input_size` is called;
        :class:`equicdft.model.GridCACEModel` does this at construction with
        the width its feature modules produce.
    n_types
        Number of physical density components and per-particle outputs.
    hidden_sizes
        Width of each hidden layer. An empty sequence gives a linear readout.
    feature_scale
        Optional positive divisor per input feature. The network then sees
        ``local_features / feature_scale``. Set it to the population standard
        deviation of each feature over the training voxels
        (:func:`equicdft.training.local_feature_scale` computes it) to give
        every invariant unit spread: the Cartesian invariants of one
        pipeline span orders of magnitude (the smeared density and the
        centre channel dominate the gradient-type moments), so an unscaled
        network first fits along the largest ones. The scale is a fixed
        diagonal preconditioner that belongs to the fitted weights: it is a
        persistent buffer, so the state dict carries it and a scaled state
        refuses to load into an unscaled readout; the configuration records
        it as well. Its length must equal the bound input width, and it
        binds an unbound readout to that width.
    feature_offset
        Optional value per input feature subtracted before the division,
        ``(local_features - feature_offset) / feature_scale``; the population
        mean centres every feature. Requires ``feature_scale``. A constant
        shift is absorbed exactly by the first layer's bias, so centring
        changes conditioning, not what the readout can represent: it keeps
        inputs with a large mean-to-spread ratio (the smeared density
        channels) from entering every hidden unit with the same large
        offset.

    Notes
    -----
    Smooth SiLU activations keep higher functional derivatives well defined.
    """

    requires_local_features = True
    reads_smoothed_fields = True

    def __init__(
        self,
        n_types: int = 1,
        *,
        hidden_sizes: Sequence[int] = (32, 16),
        n_features: Optional[int] = None,
        feature_scale: Optional[Sequence[float]] = None,
        feature_offset: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()

        self.n_types = positive_integer(n_types, "n_types")
        self.hidden_sizes = validate_hidden_sizes(hidden_sizes)
        self.n_features = None
        self.mlp = None
        # Persistent: the weights are only meaningful together with the
        # scale, so it travels with them. A None buffer has no state entry,
        # which keeps unscaled readouts and their checkpoints as they were.
        self.register_buffer("feature_scale", None)
        self.register_buffer("feature_offset", None)
        n_features = optional_positive_integer(n_features, "n_features")
        if feature_offset is not None and feature_scale is None:
            raise ValueError("feature_offset requires a feature_scale")
        if feature_scale is not None:
            scale, offset = self._validated_transform(feature_scale, feature_offset)
            if n_features is None:
                n_features = scale.numel()
            elif scale.numel() != n_features:
                raise ValueError(
                    "feature_scale must contain one value per input feature "
                    "({}), got {}".format(n_features, scale.numel())
                )
            self._store_transform(scale, offset)
        if n_features is not None:
            self.bind_input_size(n_features)

    @property
    def _scale_buffer(self) -> Optional[torch.Tensor]:
        # Legacy whole-object pickles predate the buffer; treat them as unscaled.
        return self._buffers.get("feature_scale")

    @property
    def _offset_buffer(self) -> Optional[torch.Tensor]:
        return self._buffers.get("feature_offset")

    @staticmethod
    def _validated_transform(
        feature_scale: Any,
        feature_offset: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        scale = torch.as_tensor(feature_scale, dtype=torch.float64)
        scale = scale.detach().cpu().reshape(-1).clone()
        if scale.numel() == 0 or not bool(
            torch.isfinite(scale).all() and (scale > 0).all()
        ):
            raise ValueError("feature_scale must contain finite positive values")
        if feature_offset is None:
            return scale, None
        offset = torch.as_tensor(feature_offset, dtype=torch.float64)
        offset = offset.detach().cpu().reshape(-1).clone()
        if offset.numel() != scale.numel() or not bool(torch.isfinite(offset).all()):
            raise ValueError(
                "feature_offset must contain one finite value per feature_scale entry"
            )
        return scale, offset

    def _store_transform(
        self,
        scale: Optional[torch.Tensor],
        offset: Optional[torch.Tensor],
    ) -> None:
        if scale is None:
            self.feature_scale = None
            self.feature_offset = None
            return
        # Same dtype and device as the weights they belong to (the default
        # dtype before binding; bind_input_size builds the network in that
        # dtype too), so the function and the recorded values agree exactly.
        weight = self.mlp[0].weight if self.is_bound else None
        dtype = torch.get_default_dtype() if weight is None else weight.dtype
        device = module_device(self)
        self.feature_scale = scale.to(device=device, dtype=dtype)
        self.feature_offset = None if offset is None else offset.to(device=device, dtype=dtype)

    @property
    def is_bound(self) -> bool:
        return self.mlp is not None

    def bind_input_size(self, n_features: int) -> None:
        """Build the network for ``n_features`` inputs, once.

        Rebinding to the same width is a no-op; a different width is an
        error, because the fitted parameters would no longer fit.
        """

        n_features = positive_integer(n_features, "n_features")
        if self.is_bound:
            if n_features != self.n_features:
                raise ValueError(
                    "{} is bound to {} input features, not {}".format(
                        type(self).__name__, self.n_features, n_features,
                    )
                )
            return
        scale = self._scale_buffer
        if scale is not None and scale.numel() != n_features:
            raise ValueError(
                "{} carries a feature_scale for {} input features, not {}".format(
                    type(self).__name__, scale.numel(), n_features,
                )
            )
        self.n_features = n_features
        self.mlp = build_mlp(n_features, self.hidden_sizes, self.n_types)

    def set_feature_scale(
        self,
        feature_scale: Optional[Sequence[float]],
        feature_offset: Optional[Sequence[float]] = None,
        *,
        keep_function: bool = False,
    ) -> None:
        """Replace the input scale (and offset) of a bound readout.

        With ``keep_function=True`` the first linear layer is reparametrized
        so the readout computes the same function as before: the weights
        absorb the ratio of the new and the old scale and the bias absorbs
        the shift of the offset. This standardizes a fitted readout without
        changing the model. Otherwise the weights are left alone and the
        function changes, which is the intended use before a fit: a freshly
        initialized network then starts from unit-spread (and, with an
        offset, zero-mean) inputs. ``None`` removes scale and offset.
        """

        if not self.is_bound:
            raise RuntimeError(
                "{} has no input width yet; bind it before setting a "
                "feature_scale".format(type(self).__name__)
            )
        keep_function = boolean(keep_function, "keep_function")
        if feature_scale is None:
            if feature_offset is not None:
                raise ValueError("feature_offset requires a feature_scale")
            new_scale = new_offset = None
        else:
            new_scale, new_offset = self._validated_transform(feature_scale, feature_offset)
            if new_scale.numel() != self.n_features:
                raise ValueError(
                    "feature_scale must contain one value per input feature "
                    "({}), got {}".format(self.n_features, new_scale.numel())
                )
        if keep_function:
            first = self.mlp[0]
            weight = first.weight
            ones = torch.ones(self.n_features, dtype=weight.dtype, device=weight.device)
            zeros = torch.zeros_like(ones)
            old_scale = ones if self._scale_buffer is None else self._scale_buffer.to(weight)
            old_offset = zeros if self._offset_buffer is None else self._offset_buffer.to(weight)
            scale = ones if new_scale is None else new_scale.to(weight)
            offset = zeros if new_offset is None else new_offset.to(weight)
            # W (x - o)/s + b == W' (x - o')/s' + b'  with  W' = W s'/s and
            # b' = b + W (o' - o)/s.
            with torch.no_grad():
                first.bias.add_(weight @ ((offset - old_offset) / old_scale))
                weight.mul_((scale / old_scale)[None, :])
        self._store_transform(new_scale, new_offset)

    def to_config(self) -> Dict[str, Any]:
        """Return the constructor arguments describing this readout.

        ``n_features`` is ``None`` while the readout is unbound.
        """

        return make_config(
            self,
            n_types=self.n_types,
            hidden_sizes=self.hidden_sizes,
            n_features=self.n_features,
            feature_scale=optional_buffer_value(self._scale_buffer),
            feature_offset=optional_buffer_value(self._offset_buffer),
        )

    def forward(
        self,
        local_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return local outputs with shape ``[..., n_grid, n_types]``."""

        if not self.is_bound:
            raise RuntimeError(
                "{} has no input width yet; pass n_features or let "
                "GridCACEModel bind it".format(type(self).__name__)
            )
        scale = self._scale_buffer
        if scale is not None:
            offset = self._offset_buffer
            if offset is not None:
                local_features = local_features - offset.to(local_features)
            local_features = local_features / scale.to(local_features)
        return self.mlp(local_features)

    def energy(self, context: ReadoutContext) -> torch.Tensor:
        """Return the integrated density-weighted local contribution."""

        rho = context.rho
        per_particle = self(context.local_features())
        if per_particle.shape != rho.shape:
            raise ValueError(
                "LocalReadout must return one value per grid and type"
            )
        return context.integrate_local(per_particle)


@register
class BulkReadout(EnergyReadout, Configurable):
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

    def __init__(
        self,
        n_types: int = 1,
        *,
        hidden_sizes: Sequence[int] = (32, 16),
        zero_init: bool = True,
    ) -> None:
        super().__init__()

        self.n_types = positive_integer(n_types, "n_types")
        self.n_state_features = 1 + self.n_types
        self.hidden_sizes = validate_hidden_sizes(hidden_sizes)
        self.zero_init = boolean(zero_init, "zero_init")
        self.mlp = build_mlp(
            self.n_state_features,
            self.hidden_sizes,
            self.n_types,
            zero_init=self.zero_init,
        )

    def to_config(self) -> Dict[str, Any]:
        """Return the constructor arguments describing this readout."""

        return make_config(
            self,
            n_types=self.n_types,
            hidden_sizes=self.hidden_sizes,
            zero_init=self.zero_init,
        )

    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        """Return ``a_exc_bulk`` with shape ``[..., n_types]``."""

        if state_features.shape[-1] != self.n_state_features:
            raise ValueError(
                "state_features must end with normalized temperature and "
                "one mean density per type"
            )
        return self.mlp(state_features)

    def energy(self, context: ReadoutContext) -> torch.Tensor:
        """Return the extensive homogeneous free-energy contribution."""

        rho = context.rho
        per_particle = self(context.state_features())
        if per_particle.shape != rho.mean(dim=-2).shape:
            raise ValueError(
                "BulkReadout must return one value per field and type"
            )
        return density_weighted_integral(
            rho,
            per_particle.unsqueeze(-2),
            context.voxel_volume,
        )


@register
class LongRangeReadout(EnergyReadout, Configurable):
    """Map thermodynamic state to a reciprocal quadratic kernel.

    The readout predicts one coefficient for every fixed reciprocal kernel and
    unique species pair. Its output energy is a linear contraction with the
    reciprocal features, preserving extensive scaling. Coefficients are
    signed, not positivity constrained. With ``density_channels``, a local
    density transformation makes the energy nonquadratic in physical rho.

    Parameters
    ----------
    n_kernels
        Number of kernel channels. Inferred from ``features`` when omitted;
        required for standalone use without a feature module. Direct Gaussian
        polarization with divergence has two channels per radial exponent.
    n_types
        Number of physical density components. Inferred from ``features`` when
        omitted, or defaults to one without a feature module. Explicit counts
        must match the feature module. Normally the state vector contains
        normalized temperature followed by one mean density per component;
        with ``density_channels`` it contains only normalized temperature.
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
        ``ReciprocalFeatures(variable="dipole_density", kernel="gaussian")``
        selects direct vector-dot-product features, including uniform and
        transverse polarization. This mode requires no ``include_polarization``
        flag and disallows charges / Coulomb amplitudes. Its coefficients are
        independent of any separate density or Coulomb readout. The existing
        ideal orientational free energy is not replaced.
        With ``features.include_divergence=True``, the coefficient order is
        all vector kernels followed by all divergence kernels. These have
        different physical units; no width or reference scaling is implicit.
    include_polarization
        If true, require ``dipole_density`` and use the same Coulomb kernel
        on ``q_a rho_hat_a - i k.P_hat_a``. Requires explicit ``charges``
        and Coulomb ``features``. Pair features already contain the charges,
        so every pair coefficient is the shared amplitude A, not A*q_i*q_j.
        With ``free_energy_mode="beta"``, use A=beta*C in compatible physical
        units at fixed temperature. For variable temperatures, use the
        model's physical mode with A=C/(k_B*T_ref). No dielectric scaling,
        dipole-magnitude factor, or particle self subtraction is implicit.
    coefficient_scale
        Optional fixed positive factor per kernel channel (one scalar is
        broadcast) multiplying the state network's output, so that the
        network works in convenient units while the contraction keeps the
        physical ones of the features. Direct polarization features of a
        dense dipolar fluid need coefficients of order the ideal alignment
        stiffness ``3 / (rho m^2)``, hundreds to thousands in ``length /
        charge^2``; a zero-initialized head trained at ordinary learning
        rates never reaches such values unless the scale is supplied.
        Unavailable in the charge-factorized and fixed-amplitude modes.
    initial_coefficients
        Optional fixed reference coefficients per kernel channel (or per
        kernel and type pair), in the features' physical units and added to
        the scaled network output. With ``zero_init`` the readout starts
        exactly at these values and the network learns the remainder, which
        turns a known linear response, such as a dielectric kernel fitted
        to polarization structure factors, into a fixed reference.
        Unavailable in the charge-factorized and fixed-amplitude modes.
    density_channels
        Optional number M of nonlinear local density sources for scalar
        Gaussian features. Uses ``f_a = sum_i rho_i A_ia(rho / mean_density)``
        before the FFT. Kernel coefficients then depend only on normalized
        temperature, with M*(M+1)/2 symmetric pairs per kernel. The transformed
        source mean and k=0 kernel are removed; every homogeneous density has
        zero Gaussian energy. Requires Gaussian density ``features`` and no
        fixed charges. ``None`` preserves the existing quadratic branch,
        global state dependence, parameter shapes and saved configuration.
    density_hidden_sizes
        Hidden widths of the local density MLP. The normalization is the
        fixed model ``mean_density``, not the current field average.
    density_temperature
        Also supply normalized temperature to the local density MLP. Requires
        ``density_channels``. Physical rho multiplies the MLP output, so
        transformed fields vanish in vacuum without clipping or division.
    """

    @property
    def requires_dipole_density(self) -> bool:
        features = self.features
        return (
            self.include_polarization
            or (features is not None and features.requires_dipole_density)
        )

    @property
    def reads_smoothed_fields(self) -> bool:
        """Whether this readout's sources are a learned local function of rho.

        Learned ``density_channels`` sources read the model's learned-input
        fields; kernels on the physical fields (Coulomb, Gaussian, dielectric,
        fixed references) always read the voxel fields.
        """

        return getattr(self, "density_channels", None) is not None

    def __init__(
        self,
        n_kernels: Optional[int] = None,
        n_types: Optional[int] = None,
        *,
        hidden_sizes: Sequence[int] = (32, 16),
        zero_init: bool = True,
        charges: Optional[Sequence[float]] = None,
        coulomb_amplitude: Optional[float] = None,
        features: Optional[ReciprocalFeatures] = None,
        include_polarization: bool = False,
        coefficient_scale: Optional[Union[float, Sequence[float]]] = None,
        initial_coefficients: Optional[Sequence[float]] = None,
        density_channels: Optional[int] = None,
        density_hidden_sizes: Sequence[int] = (16, 16),
        density_temperature: bool = False,
    ) -> None:
        super().__init__()

        if features is not None and not isinstance(features, ReciprocalFeatures):
            raise TypeError("features must be ReciprocalFeatures or None")
        if n_kernels is None:
            if features is None:
                raise ValueError("n_kernels is required without features")
            n_kernels = features.n_kernels
        if n_types is None:
            n_types = features.n_types if features is not None else 1

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
        self.density_channels = optional_positive_integer(
            density_channels, "density_channels",
        )
        self.n_kernel_channels = self.density_channels or self.n_types
        self.n_kernel_pairs = len(symmetric_component_pairs(self.n_kernel_channels))
        self.density_hidden_sizes = validate_hidden_sizes(density_hidden_sizes)
        self.density_temperature = boolean(
            density_temperature, "density_temperature",
        )
        if self.density_temperature and self.density_channels is None:
            raise ValueError("density_temperature requires density_channels")
        if self.density_channels is not None and (
            features is None or features.kernel != "gaussian"
            or features.requires_dipole_density or charges is not None
        ):
            raise ValueError(
                "density_channels requires scalar Gaussian features without charges"
            )
        self.n_state_features = (
            1 if self.density_channels is not None else 1 + self.n_types
        )
        self.hidden_sizes = validate_hidden_sizes(hidden_sizes)
        self.zero_init = boolean(zero_init, "zero_init")

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
            if features.n_types != self.n_types:
                raise ValueError("features and readout n_types differ")
            if features.n_kernels != self.n_kernels:
                raise ValueError("features and readout kernel counts differ")
            if features.requires_dipole_density and charges is not None:
                raise ValueError("direct polarization features do not use charges")
        self.features = features

        if (coefficient_scale is not None or initial_coefficients is not None) and (
            charges is not None or coulomb_amplitude is not None
        ):
            raise ValueError(
                "coefficient_scale and initial_coefficients require the free "
                "kernel-coefficient mode (no charges or coulomb_amplitude)"
            )
        scale_tensor = None
        if coefficient_scale is not None:
            scale_tensor = torch.as_tensor(
                coefficient_scale, dtype=torch.get_default_dtype(),
            ).detach().clone().reshape(-1)
            if scale_tensor.numel() == 1:
                scale_tensor = scale_tensor.expand(self.n_kernels).clone()
            if (
                scale_tensor.shape != (self.n_kernels,)
                or not torch.all(torch.isfinite(scale_tensor)).item()
                or torch.any(scale_tensor <= 0.0).item()
            ):
                raise ValueError(
                    "coefficient_scale must be one finite positive value or one "
                    "per kernel channel"
                )
        initial_tensor = None
        if initial_coefficients is not None:
            initial_tensor = torch.as_tensor(
                initial_coefficients, dtype=torch.get_default_dtype(),
            ).detach().clone()
            if initial_tensor.numel() == self.n_kernels:
                initial_tensor = initial_tensor.reshape(self.n_kernels, 1).expand(
                    self.n_kernels, self.n_kernel_pairs,
                ).clone()
            if (
                initial_tensor.shape != (self.n_kernels, self.n_kernel_pairs)
                or not torch.all(torch.isfinite(initial_tensor)).item()
            ):
                raise ValueError(
                    "initial_coefficients must contain one finite value per kernel "
                    "channel or one per kernel channel and source pair"
                )
        # Non-persistent: both live in the config, and checkpoints written
        # before they existed keep loading strictly.
        self.register_buffer("coefficient_scale", scale_tensor, persistent=False)
        self.register_buffer("initial_coefficients", initial_tensor, persistent=False)

        if coulomb_amplitude is not None:
            self.mlp = None
        else:
            output_width = (
                self.n_kernels * self.n_kernel_pairs
                if charges is None
                else 1
            )
            self.mlp = build_mlp(
                self.n_state_features,
                self.hidden_sizes,
                output_width,
                zero_init=self.zero_init,
            )
        self.density_mlp = None if self.density_channels is None else build_mlp(
            input_size=self.n_types + int(self.density_temperature),
            hidden_sizes=self.density_hidden_sizes,
            output_size=self.n_types * self.density_channels,
        )

    def to_config(self) -> Dict[str, Any]:
        """Return the constructor arguments describing this readout."""

        config = make_config(
            self,
            n_kernels=self.n_kernels,
            n_types=self.n_types,
            hidden_sizes=self.hidden_sizes,
            zero_init=self.zero_init,
            charges=optional_buffer_value(self.charges),
            coulomb_amplitude=optional_buffer_value(self.coulomb_amplitude),
            features=optional_config(self.features),
            include_polarization=self.include_polarization,
            coefficient_scale=optional_buffer_value(self.coefficient_scale),
            initial_coefficients=optional_buffer_value(self.initial_coefficients),
        )
        if self.density_channels is not None:
            config.update(
                density_channels=self.density_channels,
                density_hidden_sizes=list(self.density_hidden_sizes),
            )
            if self.density_temperature:
                config["density_temperature"] = True
        return config

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "LongRangeReadout":
        """Rebuild the readout together with its reciprocal features."""

        arguments = constructor_arguments(cls, config)
        arguments["features"] = build_optional(arguments.get("features"))
        return cls(**arguments)

    def coefficients(self, state_features: torch.Tensor) -> torch.Tensor:
        """Return coefficients shaped ``[..., n_kernels, n_kernel_pairs]``."""

        if state_features.shape[-1] != self.n_state_features:
            raise ValueError(
                "state_features must have width {}".format(self.n_state_features)
            )
        if self.charges is None:
            coefficients = self.mlp(state_features).reshape(
                *state_features.shape[:-1],
                self.n_kernels,
                self.n_kernel_pairs,
            )
            if self.coefficient_scale is not None:
                coefficients = coefficients * self.coefficient_scale.to(
                    coefficients
                )[:, None]
            if self.initial_coefficients is not None:
                coefficients = coefficients + self.initial_coefficients.to(
                    coefficients
                )
            return coefficients

        amplitude = self.amplitude(state_features)[..., None, None]
        pair_weights = self.pair_charge_products.to(state_features)
        if self.include_polarization:
            # q is already in each source; applying q_i*q_j here would both
            # double-count charge factors and erase neutral-molecule dipoles.
            pair_weights = torch.ones_like(pair_weights)
        return amplitude * pair_weights.view(
            1,
            self.n_type_pairs,
        )

    def amplitude(self, state_features: torch.Tensor) -> torch.Tensor:
        """Return the shared scalar in charge-factorized mode."""

        if self.charges is None:
            raise ValueError("a scalar amplitude requires fixed component charges")
        if state_features.shape[-1] != self.n_state_features:
            raise ValueError(
                "state_features must end with normalized temperature and "
                "one mean density per type"
            )
        amplitude = self.coulomb_amplitude
        if amplitude is None:
            return self.mlp(state_features).reshape(state_features.shape[:-1])
        target_shape = state_features.shape[:-1]
        if not target_shape:
            return amplitude.to(state_features)
        return amplitude.to(state_features).expand(*target_shape)

    @property
    def provides_coulomb_potential(self) -> bool:
        """Whether this readout can evaluate its Coulomb potential in space."""

        features = self.features
        return (
            self.charges is not None
            and features is not None
            and features.kernel == "coulomb"
            and self.n_kernels == 1
        )

    def _coulomb_state(self, context: ReadoutContext):
        """Return the unit-amplitude Coulomb state and configured amplitude."""

        if not self.provides_coulomb_potential:
            raise ValueError(
                "a reusable Coulomb potential requires fixed charges and one "
                "Coulomb kernel"
            )
        state = self.features._coulomb_evaluation(
            rho=context.rho,
            grid_size=context.require("grid_size"),
            grid_spacing=context.require("grid_spacing"),
            charges=self.charges,
            dipole_density=(
                context.require("dipole_density")
                if self.requires_dipole_density else None
            ),
        )
        expected_shape = context.rho.shape[:-2]
        amplitude = self.amplitude(self._state_input(context))
        if state.base_energy.shape != expected_shape or amplitude.shape != expected_shape:
            raise ValueError("LongRangeReadout must return one scalar per field")
        return state, amplitude

    def _state_input(self, context: ReadoutContext) -> torch.Tensor:
        """Return the state vector, or a placeholder for fixed amplitudes.

        With a fixed ``coulomb_amplitude`` the coefficients have no state
        dependence, so the model's density scale is not needed at all.
        """

        if self.density_channels is not None:
            return context.normalized_temperature[..., None]
        if self.coulomb_amplitude is None:
            return context.state_features()
        return context.rho.new_zeros(
            *context.rho.shape[:-2], self.n_state_features,
        )

    def coulomb_at(
        self,
        context: ReadoutContext,
        positions: torch.Tensor,
        target_sigma: float = 0.0,
    ) -> _LiquidCoulombResult:
        """Evaluate the liquid Coulomb state at physical positions.

        The source includes bound charge when include_polarization is enabled.
        The potential includes the configured amplitude and the sum of the
        damped long-range kernel and its full-Coulomb complement.
        ``target_sigma`` is the Gaussian width of the sampling sites; zero
        evaluates point sites.
        """

        state, amplitude = self._coulomb_state(context)
        potential = state.potential_at(
            positions,
            grid_center=context.grid_center,
            target_sigma=target_sigma,
        )
        return _LiquidCoulombResult(
            energy=amplitude * state.base_energy,
            potential=amplitude[..., None] * potential,
            amplitude=amplitude,
            total_charge=state.total_charge,
        )

    def forward(
        self,
        reciprocal_features: torch.Tensor,
        state_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return one reduced long-range contribution per field."""

        expected_trailing_shape = (self.n_kernels, self.n_kernel_pairs)
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

    def density_fields(self, context: ReadoutContext) -> torch.Tensor:
        """Return physical rho, or local nonlinear sources for Gaussian kernels."""

        return density_fields(
            context, self.n_types, self.density_mlp, self.density_channels,
            self.density_temperature,
        )

    def energy(self, context: ReadoutContext) -> torch.Tensor:
        """Return the reciprocal-space contribution for a complete field."""

        if self.features is None:
            raise ValueError(
                "LongRangeReadout requires ReciprocalFeatures for model use"
            )
        source_options = {}
        if self.requires_dipole_density:
            source_options["dipole_density"] = context.require("dipole_density")
        if self.include_polarization:
            source_options["charges"] = self.charges
        if (context.bulk_second_variation and type(self.features) is ReciprocalFeatures
                and getattr(self.features.forward, "__func__", None) is ReciprocalFeatures.forward):
            source_options["bulk_second_variation"] = True
        if self.density_channels is not None:
            reciprocal_features = self.features._density_features(
                self.density_fields(context), context.require("grid_size"),
                context.require("grid_spacing"),
            )
        else:
            reciprocal_features = self.features(
                rho=context.rho,
                grid_size=context.require("grid_size"),
                grid_spacing=context.require("grid_spacing"),
                **source_options,
            )
        energy = self(reciprocal_features, self._state_input(context))
        if energy.shape != context.rho.shape[:-2]:
            raise ValueError(
                "LongRangeReadout must return one scalar per field"
            )
        return energy
