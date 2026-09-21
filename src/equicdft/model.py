"""Collected grid density-functional model."""

from contextlib import nullcontext
from numbers import Integral
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from ._validation import (
    boolean,
    nonnegative_scalar,
    optional_boolean,
    positive_scalar,
)
from ._config import (
    Configurable,
    build,
    build_optional,
    constructor_arguments,
    make_config,
    optional_config,
    register,
    scalar_value,
)
from .geometry.grid import (
    grid_spacing_tensor,
    require_matching_grid_spacing,
    voxel_volume as compute_voxel_volume,
)
from .derivatives import compute_grid_derivative
from .functional.base import EnergyReadout, ReadoutContext
from .functional.cace import CACEBranch
from .representation.message import BChiMessage
from .representation.local import (
    local_feature_width,
    local_invariants,
    needs_neighbor_indices,
    validate_features,
)
from .data.metal import METAL_DATA_KEYS
from ._nn import positive_scalar_tensor
from ._species import species_vector
from .functional.ideal import IdealReference, IdealGas
from .units import DEFAULT_BOLTZMANN_CONSTANT


__all__ = [
    "GridCACEModel",
    "RESERVED_OUTPUT_KEYS",
]


# Model-owned output names, including opt-in fitting diagnostics. A readout's
# extra outputs from ``energy_and_outputs`` must never shadow these.
RESERVED_OUTPUT_KEYS = frozenset((
    "F_exc",
    "beta_F_exc",
    "beta_F_id",
    "beta_F",
    "c1",
    "c2",
    "density_derivative",
    "density_derivative_excess",
    "ideal_density_derivative",
    "ideal_derivative_mask",
    "polarization_derivative",
    "polarization_derivative_excess",
    "ideal_polarization_derivative",
    "beta_mu_local",
    "beta_mu_average",
    "voxel_weights",
    "local_energy_mask",
))


def _normalize_readouts(readout: Sequence[EnergyReadout]) -> list:
    """Return the readouts as a list, rejecting anything counted twice.

    A readout that wraps another evaluates it already, so listing both the
    wrapper and its inner readout would add that energy to the functional
    twice.
    """

    if isinstance(readout, EnergyReadout):
        raise TypeError(
            "readout must be a sequence of EnergyReadout modules"
        )
    try:
        readouts = list(readout)
    except TypeError:
        raise TypeError(
            "readout must be a sequence of EnergyReadout modules"
        )
    if not all(isinstance(module, EnergyReadout) for module in readouts):
        raise TypeError("every readout must be an EnergyReadout module")
    if len({id(item) for item in readouts}) != len(readouts):
        raise ValueError(
            "readout lists the same module twice; its energy would be "
            "counted twice"
        )
    wrapped = {
        id(inner): (item, inner)
        for item in readouts
        for inner in item.wrapped_readouts()
    }
    for item in readouts:
        if id(item) in wrapped:
            wrapper, _ = wrapped[id(item)]
            raise ValueError(
                "{} already evaluates its wrapped {}; listing both counts "
                "that energy twice".format(
                    type(wrapper).__name__, type(item).__name__,
                )
            )
    inner_readouts = [inner for item in readouts for inner in item.wrapped_readouts()]
    if len({id(item) for item in inner_readouts}) != len(inner_readouts):
        raise ValueError("readout wrappers must not share the same inner readout")
    return readouts


def _resolve_polarization(
    ideal: Optional["IdealReference"],
    a_features: Optional[nn.Module],
    readouts: Sequence[EnergyReadout],
    compute_polarization_derivative: bool,
) -> bool:
    """Return whether dipole density is an independent field of this model."""

    if ideal is not None and not isinstance(ideal, IdealReference):
        raise TypeError("ideal must be an IdealReference")
    has_polarization = (ideal is not None and ideal.requires_dipole_density) or (
        a_features is not None and a_features.requires_dipole_density
    ) or any(item.requires_dipole_density for item in readouts)
    if has_polarization and ideal is not None and not ideal.requires_dipole_density:
        raise ValueError("polarization requires a polarization-capable ideal reference")
    if compute_polarization_derivative and not has_polarization:
        raise ValueError(
            "compute_polarization_derivative requires a dipole-density ideal, "
            "readout or feature module"
        )
    return has_polarization


def _shared_stencil_cutoffs(
    a_features: Optional[nn.Module],
    readouts: Sequence[EnergyReadout],
) -> list:
    """Return ``(label, cutoff_grid)`` for every user of the model's stencil.

    A readout that carries its own independent cutoff, such as
    ``PairwiseReadout``, is not one: it builds its own offsets and never
    indexes the shared neighbor list.
    """

    cutoffs = [(type(item).__name__, item.cutoff_grid) for item in readouts
               if item.shares_grid_stencil]
    if a_features is not None:
        cutoffs.insert(0, ("a_features", a_features.cutoff_grid))
    return cutoffs


def _check_cutoff_and_feature_requirements(
    a_features: Optional[nn.Module],
    readouts: Sequence[EnergyReadout],
) -> None:
    """Reject disagreeing neighborhoods and missing feature modules."""

    # Neighborhood requirements are independent of which fields are read.
    cutoffs = _shared_stencil_cutoffs(a_features, readouts)
    if cutoffs and any(value != cutoffs[0][1] for _, value in cutoffs):
        raise ValueError(
            "modules built on the model's stencil must use the same "
            "cutoff_grid; got " + ", ".join(
                "{}={}".format(label, value) for label, value in cutoffs
            )
        )
    if (
        any(item.requires_local_features for item in readouts)
        and a_features is None
    ):
        raise ValueError(
            "a_features and b_features are required by a configured readout"
        )


def _resolve_n_types(
    a_features: Optional[nn.Module],
    readouts: Sequence[EnergyReadout],
    n_types: Optional[int],
) -> int:
    """Return the one component count the features and readouts agree on."""

    if a_features is not None:
        if n_types is not None and n_types != a_features.n_types:
            raise ValueError("n_types must match a_features")
        n_types = a_features.n_types
    else:
        typed_readouts = [
            item for item in readouts if item.n_types is not None
        ]
        if typed_readouts:
            if n_types is not None and n_types != typed_readouts[0].n_types:
                raise ValueError("n_types must match readouts")
            n_types = typed_readouts[0].n_types
        elif n_types is None:
            raise ValueError("provide n_types for an ideal-only model")
    if isinstance(n_types, bool) or not isinstance(n_types, Integral) or n_types < 1:
        raise ValueError("n_types must be a positive integer")
    for item in readouts:
        if item.n_types is not None and item.n_types != n_types:
            raise ValueError("all readouts must use the same n_types")
    return n_types


@register
class GridCACEModel(nn.Module, Configurable):
    """Combine free-energy readouts and differentiate their scalar sum.

    Independent local pipelines can be supplied as ``CACEBranch`` entries in
    ``readout``. Each owns its A/B/messages/LocalReadout; the model owns one
    ideal reference, physical fields, unit contract and normalization scales.
    The existing top-level A/B/message arguments remain the shared-feature
    shorthand for ordinary local readouts and may coexist with branches.

    ``ideal`` owns the ideal reference, separate from the excess readouts.
    Scalar models default to IdealGas; polarized models require an explicit
    FixedDipoleIdeal to evaluate total thermodynamics or solve. Without one
    they remain excess-only models. ``compute_ideal=False`` explicitly skips
    ideal evaluation (including its physical-domain restrictions).
    With an ideal reference, forward returns beta_F_id and beta_F, and adds
    ideal derivatives to the requested excess derivatives under
    density_derivative and polarization_derivative. The corresponding
    *_derivative_excess outputs and c1/c2 remain excess-only. Total outputs
    are absent when ideal evaluation is disabled; their meaning never changes.

    ``density_derivative_excess`` is exactly ``-c1``, the same tensor under
    the systematic name that pairs with ``density_derivative``. Both are
    returned because ``c1`` is the direct-correlation name the literature
    and the fitted models use; new code that handles total and excess
    responses together should prefer the ``*_derivative*`` names.

    Every configured :class:`EnergyReadout` receives a shared context and
    returns one reduced free-energy contribution per field. With the default
    ``free_energy_mode="beta"``, the sum is interpreted directly as
    ``beta_F_exc``. With ``free_energy_mode="physical"``, the sum is
    interpreted as ``F_exc / (k_B * mean_temperature)`` and converted to
    ``beta_F_exc`` using the input temperature. The resulting
    ``beta_F_exc`` is differentiated with respect to the complete density to
    obtain
    ``c1 = -delta(beta_F_exc)/delta(rho)`` and, optionally, one selected row
    of ``c2 = delta(c1)/delta(rho)``. Constructor response flags provide
    defaults that individual forward calls may override. Readouts may also
    return named observables through ``energy_and_outputs``; these are passed
    through unchanged and are not additional energy contributions.

    With polarization features or readouts, ``dipole_density`` is an independent polar
    vector field. Density derivatives hold that field fixed. The optional
    ``polarization_derivative_excess`` output is the positive functional derivative
    ``delta(beta_F_exc)/delta(dipole_density)``, not an external electric
    field or a complete orientational chemical potential. Polarization
    descriptors require cubic voxels; the box dimensions may differ.

    ``boltzmann_constant`` and ``mean_temperature`` fix the unit contract:
    ``reference_energy = k_B * mean_temperature`` scales physical readouts and
    ``free_energy_mode="physical"`` outputs, and every batch's ``beta`` must
    equal ``1 / (k_B * temperature)`` under this constant or the forward pass
    raises. The default constant is the one ``GridData`` uses
    (:data:`equicdft.units.DEFAULT_BOLTZMANN_CONSTANT`, eV/K); reduced-unit
    data and models both pass ``boltzmann_constant=1.0``.

    ``mean_density`` is the one density scale shared by every normalized
    density input (feature moments, LDA inputs, bulk and long-range state
    vectors). It defaults to the scale of ``a_features`` and must agree with
    it when both are given; a model without local features that contains
    density-normalizing readouts must receive it explicitly.

    ``polarization_scale`` is a user-specified positive scalar in dipole-density
    units, shared by polarized A features, messages and LDA. It is required for
    those learned inputs, stored as a fixed buffer and never subtracted from P.
    Ideal, Coulomb, external coupling and reciprocal kernels use physical P.

    Alignment learned inputs use a separate dimensionless ``alignment_scale``
    and positive ``alignment_density_floor``. They evaluate the smooth field
    ``rho P / (m (rho^2 + alignment_density_floor^2))`` inside the autograd
    graph, with ``m`` taken from the fixed-dipole ideal. Raw learned inputs
    continue to use ``polarization_scale`` unchanged.

    Ordinary forwards evaluate the full physical intrinsic functional.
    ``use_voxel_mask=True`` selects ideal/LDA/CACE energy centers using the
    positive support of :meth:`voxel_weights`. All descriptor inputs and LR
    terms remain unmodified; derivatives at omitted centers may be nonzero
    through neighboring or long-range contributions. Solvers use full energy.

    ``compute_local_mu=True`` additionally returns dimensionless beta mu as
    ``beta_mu_local = density_derivative + beta * V_ext``, its
    weighted spatial ``beta_mu_average`` per field/species, and
    ``voxel_weights``. ``rho_min`` selects rho > rho_min unless
    explicit weights are supplied. See :meth:`voxel_weights`.
    These are fitting diagnostics, not extra intrinsic energy terms; V_ext
    and a full ideal reference are required only when requesting them.
    """

    def __init__(
        self,
        a_features: Optional[nn.Module] = None,
        b_features: Optional[nn.Module] = None,
        readout: Sequence[EnergyReadout] = (),
        *,
        grid_spacing: Union[float, Sequence[float], torch.Tensor],
        ideal: Optional[IdealReference] = None,
        n_types: Optional[int] = None,
        mean_temperature: Union[float, torch.Tensor] = 1.0,
        boltzmann_constant: float = DEFAULT_BOLTZMANN_CONSTANT,
        thermal_wavelength: Optional[Union[
            float,
            Sequence[float],
            torch.Tensor,
        ]] = None,
        compute_c1: bool = True,
        compute_c2: bool = False,
        compute_local_mu: bool = False,
        rho_min: float = 0.0,
        free_energy_mode: str = "beta",
        message_layers: Optional[Sequence[BChiMessage]] = None,
        compute_polarization_derivative: bool = False,
        mean_density: Optional[Union[float, torch.Tensor]] = None,
        polarization_scale: Optional[Union[float, torch.Tensor]] = None,
        alignment_scale: Optional[Union[float, torch.Tensor]] = None,
        alignment_density_floor: Optional[Union[float, torch.Tensor]] = None,
    ) -> None:
        super().__init__()

        if ideal is not None and thermal_wavelength is not None:
            raise ValueError("set thermal_wavelength on ideal, not on both ideal and model")
        if thermal_wavelength is None:
            thermal_wavelength = 1.0

        compute_c1 = boolean(compute_c1, "compute_c1")
        compute_c2 = boolean(compute_c2, "compute_c2")
        compute_local_mu = boolean(compute_local_mu, "compute_local_mu")
        rho_min = nonnegative_scalar(rho_min, "rho_min")
        compute_polarization_derivative = boolean(
            compute_polarization_derivative,
            "compute_polarization_derivative",
        )
        if free_energy_mode not in ("beta", "physical"):
            raise ValueError(
                "free_energy_mode must be 'beta' or 'physical'"
            )

        messages = validate_features(a_features, b_features, message_layers)
        readouts = _normalize_readouts(readout)
        has_polarization = _resolve_polarization(
            ideal, a_features, readouts, compute_polarization_derivative,
        )
        _check_cutoff_and_feature_requirements(a_features, readouts)
        n_types = _resolve_n_types(a_features, readouts, n_types)
        polarization_scale = (None if polarization_scale is None else
                              positive_scalar_tensor(polarization_scale, "polarization_scale"))
        needs_polarization_scale = (
            a_features is not None
            and getattr(
                a_features,
                "requires_polarization_scale",
                a_features.include_polarization,
            )
        ) or any(item.requires_polarization_scale for item in readouts)
        if needs_polarization_scale and polarization_scale is None:
            raise ValueError("polarized learned inputs require model polarization_scale")
        alignment_scale = (
            None if alignment_scale is None
            else positive_scalar_tensor(alignment_scale, "alignment_scale")
        )
        alignment_density_floor = (
            None if alignment_density_floor is None
            else positive_scalar_tensor(
                alignment_density_floor, "alignment_density_floor"
            )
        )
        needs_alignment_scale = (
            a_features is not None
            and getattr(a_features, "requires_alignment_scale", False)
        ) or any(item.requires_alignment_scale for item in readouts)
        if needs_alignment_scale:
            if alignment_scale is None or alignment_density_floor is None:
                raise ValueError(
                    "alignment learned inputs require model alignment_scale "
                    "and alignment_density_floor"
                )
            if ideal is None or getattr(ideal, "dipole_magnitude", None) is None:
                raise ValueError(
                    "alignment learned inputs require an ideal with dipole_magnitude"
                )
        density_scale = None
        if mean_density is not None:
            density_scale = positive_scalar_tensor(mean_density, "mean_density")
        feature_modules = ([a_features] if a_features is not None else []) + [
            item.a_features for item in readouts if isinstance(item, CACEBranch)
        ]
        for feature in feature_modules:
            feature_scale = feature.mean_density.detach().reshape(()).to(
                dtype=torch.get_default_dtype(), device="cpu",
            )
            if density_scale is None:
                density_scale = feature_scale.clone()
            elif not torch.allclose(density_scale, feature_scale):
                raise ValueError(
                    "mean_density must equal a_features.mean_density in every branch; the "
                    "model owns one density scale for every normalized input"
                )

        if a_features is not None:
            # Every readout that consumes the local feature vector is given
            # its input width now, so all parameters exist with their final
            # shapes before optimizers, checkpoints or model files see them.
            n_local_features = local_feature_width(
                a_features, b_features, len(messages),
            )
            reference = a_features.mean_density
            for item in readouts:
                if not item.requires_local_features:
                    continue
                freshly_built = not item.is_bound
                item.bind_input_size(n_local_features)
                if freshly_built:
                    item.to(dtype=reference.dtype, device=reference.device)

        self._n_types = int(n_types)
        self.ideal = ideal if ideal is not None else (
            None if has_polarization else IdealGas(thermal_wavelength=thermal_wavelength)
        )
        self.a_features = a_features
        self.b_features = b_features
        self.message_layers = nn.ModuleList(messages)
        self.readout = nn.ModuleList(readouts)
        self.compute_c1 = compute_c1 or compute_c2
        self.compute_c2 = compute_c2
        self.compute_local_mu = compute_local_mu
        self.rho_min = rho_min
        self.compute_polarization_derivative = compute_polarization_derivative
        self.free_energy_mode = free_energy_mode

        spacing = grid_spacing_tensor(
            grid_spacing,
            dtype=torch.get_default_dtype(),
        )
        if has_polarization and not torch.allclose(
            spacing, spacing[0].expand_as(spacing)
        ):
            raise ValueError(
                "dipole-density models require equal grid_spacing along all "
                "three axes (cubic voxels, not necessarily a cubic box)"
            )

        mean_temperature_tensor = torch.as_tensor(
            mean_temperature,
            dtype=torch.get_default_dtype(),
        ).detach().clone().reshape(-1)
        if mean_temperature_tensor.numel() != 1:
            raise ValueError("mean_temperature must be a positive scalar")
        mean_temperature_tensor = mean_temperature_tensor.reshape(())
        if (
            not torch.isfinite(mean_temperature_tensor).item()
            or mean_temperature_tensor.item() <= 0.0
        ):
            raise ValueError("mean_temperature must be a positive scalar")

        boltzmann_constant = positive_scalar(
            boltzmann_constant,
            "boltzmann_constant",
        )
        boltzmann_constant_tensor = torch.as_tensor(
            boltzmann_constant,
            dtype=torch.get_default_dtype(),
        )

        self.register_buffer("grid_spacing", spacing)
        self.register_buffer("mean_density", density_scale)
        self.register_buffer("polarization_scale", polarization_scale)
        self.register_buffer("alignment_scale", alignment_scale)
        self.register_buffer("alignment_density_floor", alignment_density_floor)
        self.register_buffer("mean_temperature", mean_temperature_tensor)
        self.register_buffer(
            "boltzmann_constant",
            boltzmann_constant_tensor,
        )
        # The ideal reference validated its own constants in its constructor.
        # A default one built from this model's ``thermal_wavelength`` is
        # widened to one value per component, which is the shape model files
        # and ``grid_info`` have always carried; an explicitly supplied
        # reference keeps the shape it was given and is only checked here.
        if self.ideal is not None:
            if ideal is None:
                self.ideal.thermal_wavelength = species_vector(
                    self.ideal.thermal_wavelength,
                    self.n_types,
                    spacing,
                    "thermal_wavelength",
                    positive=True,
                )
            for name in ("thermal_wavelength", "dipole_magnitude"):
                value = getattr(self.ideal, name, None)
                if value is not None and value.numel() not in (1, self.n_types):
                    raise ValueError(name + " must be scalar or one value per type")

    def to_config(self) -> Dict[str, Any]:
        """Return the nested constructor arguments describing this model.

        The configuration records structure only: feature, message, and
        readout configurations plus the fixed thermodynamic metadata. Fitted
        parameters and buffers are restored separately from the
        ``state_dict``. See :mod:`equicdft.serialization` for the file format
        that stores both together.
        """

        return make_config(
            self,
            a_features=optional_config(self.a_features),
            b_features=optional_config(self.b_features),
            readout=[optional_config(item) for item in self.readout],
            grid_spacing=self.grid_spacing,
            mean_temperature=scalar_value(self.mean_temperature),
            boltzmann_constant=scalar_value(self.boltzmann_constant),
            ideal=optional_config(self.ideal),
            n_types=self.n_types,
            compute_c1=self.compute_c1,
            compute_c2=self.compute_c2,
            compute_local_mu=self.compute_local_mu,
            rho_min=self.rho_min,
            compute_polarization_derivative=self.compute_polarization_derivative,
            free_energy_mode=self.free_energy_mode,
            message_layers=[
                optional_config(item) for item in self.message_layers
            ],
            mean_density=(
                None if self.mean_density is None
                else scalar_value(self.mean_density)
            ),
            polarization_scale=(None if self.polarization_scale is None
                                else scalar_value(self.polarization_scale)),
            alignment_scale=(None if self.alignment_scale is None
                             else scalar_value(self.alignment_scale)),
            alignment_density_floor=(
                None if self.alignment_density_floor is None
                else scalar_value(self.alignment_density_floor)
            ),
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "GridCACEModel":
        """Rebuild the model and every configured submodule."""

        arguments = constructor_arguments(cls, config)
        arguments["ideal"] = build_optional(arguments.get("ideal"))
        arguments["a_features"] = build_optional(arguments.get("a_features"))
        arguments["b_features"] = build_optional(arguments.get("b_features"))
        readouts = arguments.get("readout")
        if not isinstance(readouts, (list, tuple)):
            raise TypeError("model configuration 'readout' must be a list")
        arguments["readout"] = [build(item) for item in readouts]
        messages = arguments.get("message_layers") or []
        if not isinstance(messages, (list, tuple)):
            raise TypeError(
                "model configuration 'message_layers' must be a list"
            )
        arguments["message_layers"] = [build(item) for item in messages]
        return cls(**arguments)

    @property
    def n_local_features(self) -> int:
        """Width of the shared top-level feature vector (not branch widths)."""

        if not self.has_local_features:
            return 0
        return local_feature_width(
            self.a_features, self.b_features, len(self.message_layers),
        )

    @property
    def cutoff_grid(self) -> int:
        """Integer stencil cutoff used by the local representation."""

        cutoffs = _shared_stencil_cutoffs(
            self.a_features if self.has_local_features else None, self.readout,
        )
        return cutoffs[0][1] if cutoffs else 0

    @property
    def n_types(self) -> int:
        """Number of physical density components accepted by the model."""

        if self.has_local_features:
            return self.a_features.n_types
        for item in self.readout:
            if item.n_types is not None:
                return item.n_types
        return self._n_types

    @property
    def has_local_features(self) -> bool:
        """Whether the shared top-level A/B pipeline is configured."""

        return self.a_features is not None

    @property
    def requires_local_density_index(self) -> bool:
        """Whether any configured local operator uses explicit gathering."""

        if any(item.requires_local_density_index for item in self.readout):
            return True
        if not self.has_local_features:
            return False
        return needs_neighbor_indices(self.a_features, self.message_layers)

    @property
    def requires_dipole_density(self) -> bool:
        """Whether configured features or readouts depend on polarization."""

        return (self.ideal is not None and self.ideal.requires_dipole_density) or (
            self.has_local_features and self.a_features.requires_dipole_density
        ) or any(item.requires_dipole_density for item in self.readout)

    @property
    def thermal_wavelength(self):
        """The model-owned ideal reference wavelength."""
        if self.ideal is None:
            raise ValueError("attach an explicit ideal reference first")
        return self.ideal.thermal_wavelength

    @property
    def voxel_volume(self) -> torch.Tensor:
        """Quadrature volume represented by one grid point."""

        return compute_voxel_volume(self.grid_spacing)

    @property
    def reference_energy(self) -> torch.Tensor:
        """Energy scale ``k_B * mean_temperature`` used in physical mode."""

        return self.boltzmann_constant * self.mean_temperature

    @property
    def grid_info(self) -> Dict[str, Any]:
        """Return the grid and thermodynamic metadata needed for inference."""

        return {
            "cutoff_grid": self.cutoff_grid,
            "grid_spacing": self.grid_spacing.detach().cpu().tolist(),
            "n_types": self.n_types,
            "boltzmann_constant": (
                self.boltzmann_constant.detach().cpu().item()
            ),
            # An excess-only model (a polarized one with no explicit ideal
            # reference) defines no wavelength, and the property below says
            # so. Data still needs the entry, and this model never reads it,
            # so the neutral value is written rather than a guess at the
            # physical one: unlike a model file, nothing fitted depends on it.
            "thermal_wavelength": (
                self.thermal_wavelength.detach().cpu().expand(self.n_types).tolist()
                if self.ideal is not None else [1.0] * self.n_types
            ),
        }

    def _check_unit_contract(
        self,
        batch_beta: torch.Tensor,
        model_beta: torch.Tensor,
    ) -> None:
        """Reject a batch whose ``beta`` was built with another Boltzmann constant.

        Data carries ``beta = 1 / (k_B T)`` from the constant it was built
        with; the model derives the same quantity from its own constant. The
        two agree to rounding when the constants match and differ by orders
        of magnitude when kelvin/eV data meets a reduced-unit model or vice
        versa, which used to pass silently.
        """

        batch_beta = torch.as_tensor(batch_beta).detach()
        expected = model_beta.detach().to(batch_beta)
        if batch_beta.shape != expected.shape:
            try:
                batch_beta = batch_beta.expand_as(expected)
            except RuntimeError:
                raise ValueError(
                    "beta must be scalar for one field or match the batch shape"
                ) from None
        if torch.allclose(batch_beta, expected, rtol=1.0e-4, atol=0.0):
            return
        ratio = (
            expected.reshape(-1)[0].double() / batch_beta.reshape(-1)[0].double()
        ).item()
        model_constant = self.boltzmann_constant.item()
        raise ValueError(
            "the batch beta does not match this model's unit contract: the "
            "data implies k_B = {:.6g} but the model uses boltzmann_constant "
            "= {:.6g}; build the model with the dataset's boltzmann_constant "
            "or the data with grid_info=model.grid_info".format(
                ratio * model_constant, model_constant,
            )
        )

    def _validate_c2_reference(
        self,
        c2_reference: Tuple[int, int],
        rho: torch.Tensor,
    ) -> Tuple[int, int]:
        """Validate and return ``(reference_grid, reference_type)``."""

        if not isinstance(c2_reference, (tuple, list)) or len(c2_reference) != 2:
            raise TypeError(
                "c2_reference must contain (grid_index, type_index)"
            )
        if any(
            isinstance(index, bool) or not isinstance(index, Integral)
            for index in c2_reference
        ):
            raise TypeError("c2_reference indices must be integers")

        reference_grid, reference_type = map(int, c2_reference)
        if not 0 <= reference_grid < rho.shape[-2]:
            raise IndexError("c2 reference grid index is out of bounds")
        if not 0 <= reference_type < rho.shape[-1]:
            raise IndexError("c2 reference type index is out of bounds")
        return reference_grid, reference_type

    def _local_invariant_features(
        self,
        data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return flattened and concatenated ``[B0, B1, ...]`` features."""

        return local_invariants(
            self.a_features, self.b_features, self.message_layers, data,
            self.polarization_scale,
            self.alignment_scale,
            getattr(self.ideal, "dipole_magnitude", None),
            self.alignment_density_floor,
        )

    def voxel_weights(self, data):
        """Return detached nonnegative fitting weights [..., grid, species].

        Select rho > rho_min unless explicit ``voxel_weights`` are supplied;
        intersect with valid and nonexcluded voxels. Preserve explicit weight
        magnitudes for fitting averages and losses. Only their positive support
        selects ideal/LDA/CACE energy when ``use_voxel_mask=True``. Descriptor
        inputs and long-range terms always retain the complete physical fields.
        Empty selections are allowed here, but not for a chemical-potential
        average. Freezing weights before SEM does not freeze the noisy field's
        chemical-potential values or their weighted mean.
        """
        rho = data["rho"]
        if "voxel_weights" in data:
            weights = torch.as_tensor(data["voxel_weights"], device=rho.device)
            if (weights.shape != rho.shape or weights.is_complex()
                    or not torch.isfinite(weights).all() or torch.any(weights < 0)):
                raise ValueError("voxel_weights must be finite nonnegative with rho's shape")
            weights = weights.detach().to(rho)
        else:
            weights = (rho.detach() > self.rho_min).to(rho)
        for key in ("valid", "excluded_mask"):
            if key not in data:
                continue
            mask = data[key]
            if (not torch.is_tensor(mask) or mask.dtype != torch.bool
                    or mask.shape != rho.shape[:-1]):
                raise ValueError(key + " must be Boolean with rho's field/grid shape")
            active = ~mask if key == "excluded_mask" else mask
            weights = weights.masked_fill(~active[..., None], 0.)
        return weights

    def _chemical_potential_outputs(self, data, outputs, beta):
        """Add beta mu = delta(beta F)/delta rho + beta V and its spatial mean."""
        rho = data["rho"]
        potential = data["V_ext"]
        if (not torch.is_tensor(potential) or potential.shape != rho.shape
                or not potential.is_floating_point()):
            raise ValueError("V_ext must be floating with rho's shape")
        weights = self.voxel_weights(data)
        if torch.any(weights.sum(dim=-2) <= 0):
            raise ValueError("chemical-potential mask must retain positive weight for every field/species")
        active = weights > 0
        if torch.any(active & ~outputs["ideal_derivative_mask"]):
            raise ValueError("chemical-potential weights select vacuum; use a positive SEM density_floor")
        potential = potential.to(rho)
        if not torch.isfinite(potential[active]).all():
            raise ValueError("V_ext must be finite on the chemical-potential mask")
        # Mask before arithmetic so excluded infinite external potentials do
        # not contaminate either the weighted mean or its autograd graph.
        local = torch.where(active, outputs["density_derivative"], 0.)
        local = local + beta[..., None, None] * torch.where(active, potential, 0.)
        if not torch.isfinite(local[active]).all():
            raise ValueError("local chemical potential must be finite on its mask")
        outputs.update(
            beta_mu_local=local,
            voxel_weights=weights,
            beta_mu_average=(weights * local).sum(dim=-2) / weights.sum(dim=-2),
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        compute_c1: Optional[bool] = None,
        compute_c2: Optional[bool] = None,
        c2_reference: Tuple[int, int] = (0, 0),
        compute_polarization_derivative: Optional[bool] = None,
        compute_ideal: Optional[bool] = None,
        compute_local_mu: Optional[bool] = None,
        use_voxel_mask: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Return the full intrinsic free energy and requested responses.

        ``use_voxel_mask=True`` selects ideal/LDA/CACE energy centers using
        positive :meth:`voxel_weights`. Descriptor inputs and LR remain full.
        The default ignores fitting selections in every energy contribution.
        """

        if METAL_DATA_KEYS & data.keys():
            raise ValueError(
                "Set electrode geometry, constraints, and field on MetalWall, not in data"
            )

        use_voxel_mask = boolean(use_voxel_mask, "use_voxel_mask")
        compute_local_mu = optional_boolean(compute_local_mu, "compute_local_mu")
        if compute_local_mu is None:
            compute_local_mu = self.compute_local_mu
        if compute_local_mu and "V_ext" not in data:
            raise ValueError("compute_local_mu=True requires V_ext")
        compute_c2 = optional_boolean(compute_c2, "compute_c2")
        if compute_c2 is None:
            compute_c2 = self.compute_c2
        compute_c1 = optional_boolean(compute_c1, "compute_c1")
        if compute_c1 is None:
            compute_c1 = self.compute_c1
        compute_c1 = compute_c1 or compute_c2 or compute_local_mu
        compute_polarization_derivative = optional_boolean(
            compute_polarization_derivative,
            "compute_polarization_derivative",
        )
        if compute_polarization_derivative is None:
            compute_polarization_derivative = self.compute_polarization_derivative
        has_polarization = self.requires_dipole_density
        compute_ideal = optional_boolean(compute_ideal, "compute_ideal")
        if compute_ideal is None:
            compute_ideal = self.ideal is not None
        if compute_local_mu and not compute_ideal:
            raise ValueError("compute_local_mu requires the total derivative and an ideal reference (compute_ideal=True)")
        if compute_polarization_derivative and not has_polarization:
            raise ValueError(
                "compute_polarization_derivative requires a dipole-density readout"
            )
        if compute_ideal and self.ideal is None:
            raise ValueError("full evaluation requires an explicit ideal reference; "
                             "use FixedDipoleIdeal with the physical dipole magnitude "
                             "or compute_ideal=False for excess-only evaluation")

        # A requested functional derivative requires gradient tracking even
        # when surrounding evaluation code uses torch.no_grad(). Energy-only
        # evaluation respects the caller's existing gradient context.
        gradient_context = (
            torch.enable_grad()
            if compute_c1 or compute_polarization_derivative
            else nullcontext()
        )
        with gradient_context:
            if "grid_spacing" in data:
                require_matching_grid_spacing(
                    data["grid_spacing"],
                    self.grid_spacing,
                )
            if has_polarization and "dipole_density" not in data:
                raise ValueError("dipole_density is required by this model")
            # The response fields passed by the caller are marked as requiring
            # gradients in place. This is part of the contract: after a
            # forward pass, callers differentiate any output (including
            # readout observables) with respect to the very tensors they
            # supplied, e.g. torch.autograd.grad(outputs["c1"].sum(),
            # data["rho"]). Detached copies would silently break that.
            response_fields = []
            if compute_c1:
                response_fields.append(data["rho"])
            if compute_polarization_derivative:
                response_fields.append(data["dipole_density"])
            for field in response_fields:
                field.requires_grad_(True)

            context = ReadoutContext.from_data(
                data,
                grid_spacing=self.grid_spacing,
                mean_temperature=self.mean_temperature,
                boltzmann_constant=self.boltzmann_constant,
                mean_density=(
                    None if self.mean_density is None
                    else self.mean_density.to(data["rho"])
                ),
                free_energy_mode=self.free_energy_mode,
                polarization_scale=(None if self.polarization_scale is None
                                    else self.polarization_scale.to(data["rho"])),
                alignment_scale=(None if self.alignment_scale is None
                                 else self.alignment_scale.to(data["rho"])),
                alignment_density_floor=(
                    None if self.alignment_density_floor is None
                    else self.alignment_density_floor.to(data["rho"])
                ),
                dipole_magnitude=getattr(self.ideal, "dipole_magnitude", None),
            )
            if "beta" in data:
                self._check_unit_contract(data["beta"], context.beta)
            rho = context.rho
            weights = self.voxel_weights(data) if use_voxel_mask else None
            context.local_energy_mask = weights > 0 if weights is not None else None
            normalized_temperature = context.normalized_temperature
            volume_element = context.voxel_volume
            if any(item.requires_local_features for item in self.readout):
                context.invariants = self._local_invariant_features(data)
                if (
                    self.a_features.separate_center
                    and not self.a_features.include_polarization
                ):
                    # Polarized features already carry their center slots
                    # inside the invariant block.
                    context.center_density = (
                        self.a_features.transform_density(rho)
                        / context.require("mean_density")
                    )

            readout_energies = []
            readout_outputs = {}
            for item in self.readout:
                energy, extra_outputs = item.energy_and_outputs(context)
                if energy.shape != rho.shape[:-2]:
                    raise ValueError(
                        "every readout must return one scalar energy per field"
                    )
                readout_energies.append(energy)
                conflicts = set(extra_outputs) & (
                    RESERVED_OUTPUT_KEYS | set(readout_outputs)
                )
                if conflicts:
                    raise ValueError(
                        "readout output names must be unique and not reserved: "
                        + ", ".join(sorted(conflicts))
                    )
                readout_outputs.update(extra_outputs)

            # Summing the scalar readouts before conversion and
            # differentiation makes every enabled contribution part of one
            # variational functional.
            readout_energy = (readout_energies[0] if readout_energies
                               else 0.0 * rho.sum(dim=(-2, -1)))
            for energy in readout_energies[1:]:
                readout_energy = readout_energy + energy

            if self.free_energy_mode == "physical":
                # The network represents F_exc / (k_B*T_ref). Therefore
                # beta*F_exc = readout_energy * T_ref/T. This explicit known
                # factor leaves the model to learn the physical free energy's
                # remaining temperature dependence.
                F_exc = self.reference_energy.to(rho) * readout_energy
                beta_F_exc = readout_energy / normalized_temperature
                outputs = {
                    "F_exc": F_exc,
                    "beta_F_exc": beta_F_exc,
                }
            else:
                beta_F_exc = readout_energy
                outputs = {"beta_F_exc": beta_F_exc}
            outputs.update(readout_outputs)
            if context.local_energy_mask is not None:
                outputs["local_energy_mask"] = context.local_energy_mask

            if response_fields:
                # One backward pass differentiates the same scalar energy
                # with respect to the requested independent grid fields.
                # Summing batch energies does not couple independent fields.
                derivatives = torch.autograd.grad(
                    beta_F_exc.sum(),
                    response_fields,
                    create_graph=(self.training or compute_c2),
                    allow_unused=True,
                )

            if compute_polarization_derivative:
                # Positive beta-free-energy response to P, with shape
                # [..., n_grid, n_types, 3]; this is not the external field.
                polarization_derivative = derivatives[-1]
                if polarization_derivative is None:
                    # A low-order invariant selection may be independent of
                    # P. Its exact zero response can still enter a loss.
                    polarization_derivative = torch.zeros_like(
                        data["dipole_density"]
                    )
                    if self.training or compute_c2:
                        polarization_derivative = (
                            0.0 * data["dipole_density"] + 0.0 * beta_F_exc.sum()
                        )
                outputs["polarization_derivative_excess"] = (
                    polarization_derivative / volume_element
                )
            if compute_c1:
                # c1 has rho's shape [..., n_grid, n_types]. The minus sign
                # and division by Delta V convert its discrete derivative
                # into -delta(beta_F_exc)/delta(rho), holding P fixed.
                derivative_rho = derivatives[0]
                if derivative_rho is None:
                    derivative_rho = 0.0 * rho + 0.0 * beta_F_exc.sum()
                outputs["c1"] = -derivative_rho / volume_element
                outputs["density_derivative_excess"] = -outputs["c1"]

                if compute_c2:
                    reference_grid, reference_type = (
                        self._validate_c2_reference(
                            c2_reference,
                            data["rho"],
                        )
                    )
                    # Selecting one c1 value and differentiating it with
                    # respect to the complete density produces one row of the
                    # second direct-correlation matrix. The sum only reduces
                    # independent leading batch entries to the scalar required
                    # by compute_grid_derivative. Dividing by Delta V converts
                    # the discrete derivative to the continuum convention;
                    # there is no additional minus sign because
                    # c2 = delta(c1) / delta(rho).
                    selected_c1 = outputs["c1"][
                        ..., reference_grid, reference_type
                    ].sum()
                    outputs["c2"] = compute_grid_derivative(
                        selected_c1,
                        data["rho"],
                        create_graph=self.training,
                        allow_unused=True,
                    ) / volume_element
                    if not self.training:
                        # The c1 graph was needed only as an intermediate for
                        # c2. Match the graph-free evaluation semantics of a
                        # c1-only forward call before returning it.
                        outputs["c1"] = outputs["c1"].detach()
                        outputs["density_derivative_excess"] = outputs["density_derivative_excess"].detach()
                        if compute_polarization_derivative:
                            outputs["polarization_derivative_excess"] = outputs[
                                "polarization_derivative_excess"
                            ].detach()
            if compute_ideal:
                outputs.update(self.ideal.evaluate(
                    data, volume_element,
                    mask=context.local_energy_mask,
                ))
                outputs["beta_F"] = outputs["beta_F_id"] + outputs["beta_F_exc"]
                if compute_c1:
                    outputs["density_derivative"] = (
                        outputs["ideal_density_derivative"] - outputs["c1"]
                    )
                if compute_polarization_derivative:
                    outputs["polarization_derivative"] = (
                        outputs["ideal_polarization_derivative"]
                        + outputs["polarization_derivative_excess"]
                    )
            if compute_local_mu:
                self._chemical_potential_outputs(data, outputs, context.beta)
            elif weights is not None:
                outputs["voxel_weights"] = weights

        return outputs
