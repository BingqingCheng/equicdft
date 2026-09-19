"""Collected grid density-functional model."""

from contextlib import nullcontext
from numbers import Integral
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from ._validation import (
    boolean,
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
from .representation.message import BChiMessage
from .data.metal import METAL_DATA_KEYS
from ._nn import positive_scalar_tensor
from .functional.ideal import IdealReference, IdealGas
from .units import DEFAULT_BOLTZMANN_CONSTANT


__all__ = [
    "GridCACEModel",
]


# The model relies on these members of its feature modules. Declaring the
# contract here lets custom modules fail at construction with a clear message
# instead of being silently treated as density-only or center-free features.
# ``a_features(data)`` must return ``[..., n_grid, n_radial_channels,
# n_monomials, n_output_channels]`` and ``b_features(A)`` must return
# ``[..., n_grid, n_radial_channels, n_features, n_output_channels]``; the
# model flattens the last three axes into the local feature vector.
_A_FEATURE_CONTRACT = (
    "n_types",
    "cutoff_grid",
    "mean_density",
    "n_radial_channels",
    "n_output_channels",
    "separate_center",
    "convolution_backend",
    "include_polarization",
    "requires_dipole_density",
    "transform_density",
)
# Message layers additionally share the feature module's stencil geometry.
_A_FEATURE_MESSAGE_CONTRACT = (
    "stencil_basis",
    "local_density_positions",
    "polarization_fields",
    "bessel_stencil_basis",
    "max_power",
    "powers",
    "coordinate_scaling",
)
_B_FEATURE_CONTRACT = ("n_features", "validate_a_features")


def _validate_feature_modules(
    a_features: Optional[nn.Module],
    b_features: Optional[nn.Module],
    with_messages: bool = False,
) -> None:
    """Check that the feature pair satisfies the model's module contract."""

    if (a_features is None) != (b_features is None):
        raise ValueError(
            "a_features and b_features must be supplied together"
        )
    if a_features is None:
        return
    a_contract = _A_FEATURE_CONTRACT
    if with_messages:
        a_contract = a_contract + _A_FEATURE_MESSAGE_CONTRACT
    for module, contract, label in (
        (a_features, a_contract, "a_features"),
        (b_features, _B_FEATURE_CONTRACT, "b_features"),
    ):
        missing = [name for name in contract if not hasattr(module, name)]
        if missing:
            raise TypeError(
                "{} ({}) does not provide {}".format(
                    label, type(module).__name__, ", ".join(missing),
                )
            )
    b_features.validate_a_features(a_features)


@register
class GridCACEModel(nn.Module, Configurable):
    """Combine free-energy readouts and differentiate their scalar sum.

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
    """

    def __init__(
        self,
        a_features: Optional[nn.Module],
        b_features: Optional[nn.Module],
        readout: Sequence[EnergyReadout],
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
        free_energy_mode: str = "beta",
        message_layers: Optional[Sequence[BChiMessage]] = None,
        compute_polarization_derivative: bool = False,
        mean_density: Optional[Union[float, torch.Tensor]] = None,
    ) -> None:
        super().__init__()

        if ideal is not None and thermal_wavelength is not None:
            raise ValueError("set thermal_wavelength on ideal, not on both ideal and model")
        if thermal_wavelength is None:
            thermal_wavelength = 1.0

        compute_c1 = boolean(compute_c1, "compute_c1")
        compute_c2 = boolean(compute_c2, "compute_c2")
        compute_polarization_derivative = boolean(
            compute_polarization_derivative,
            "compute_polarization_derivative",
        )
        if free_energy_mode not in ("beta", "physical"):
            raise ValueError(
                "free_energy_mode must be 'beta' or 'physical'"
            )

        if message_layers is None:
            messages = []
        elif isinstance(message_layers, BChiMessage):
            raise TypeError(
                "message_layers must be a sequence of BChiMessage modules"
            )
        else:
            messages = list(message_layers)
        if not all(isinstance(module, BChiMessage) for module in messages):
            raise TypeError("every message layer must be a BChiMessage module")
        if messages and a_features is None:
            raise ValueError(
                "a_features and b_features are required by message_layers"
            )
        _validate_feature_modules(
            a_features, b_features, with_messages=bool(messages),
        )
        for module in messages:
            if module.include_polarization != a_features.include_polarization:
                raise ValueError("message layers and A/B include_polarization must match")
            expected = (
                a_features.n_radial_channels,
                b_features.n_features,
                a_features.n_output_channels,
            )
            actual = (
                module.n_radial_channels,
                module.n_invariant_features,
                module.n_channels,
            )
            if actual != expected:
                raise ValueError(
                    "message layer dimensions {} do not match local feature "
                    "dimensions {}".format(actual, expected)
                )
            module.bind_geometry(a_features)
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
        # Neighborhood requirements are independent of which fields are read.
        cutoffs = [item.cutoff_grid for item in readouts
                   if item.requires_local_density_index]
        if a_features is not None:
            cutoffs.append(a_features.cutoff_grid)
        if cutoffs and any(value != cutoffs[0] for value in cutoffs):
            raise ValueError(
                "local and polarization features must use the same cutoff_grid"
            )
        if (
            any(item.requires_local_features for item in readouts)
            and a_features is None
        ):
            raise ValueError(
                "a_features and b_features are required by a configured readout"
            )
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

        density_scale = None
        if mean_density is not None:
            density_scale = positive_scalar_tensor(mean_density, "mean_density")
        if a_features is not None:
            feature_scale = a_features.mean_density.detach().reshape(()).to(
                dtype=torch.get_default_dtype(), device="cpu",
            )
            if density_scale is None:
                density_scale = feature_scale.clone()
            elif not torch.allclose(density_scale, feature_scale):
                raise ValueError(
                    "mean_density must equal a_features.mean_density; the "
                    "model owns one density scale for every normalized input"
                )

        if a_features is not None:
            # Every readout that consumes the local feature vector is given
            # its input width now, so all parameters exist with their final
            # shapes before optimizers, checkpoints or model files see them.
            n_local_features = self._local_feature_width(
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

        thermal_wavelength_tensor = torch.as_tensor(
            thermal_wavelength,
            dtype=torch.get_default_dtype(),
        ).detach().clone().reshape(-1)
        if thermal_wavelength_tensor.numel() == 1:
            thermal_wavelength_tensor = thermal_wavelength_tensor.repeat(
                self.n_types
            )
        if thermal_wavelength_tensor.shape != (self.n_types,):
            raise ValueError(
                "thermal_wavelength must contain one value or one per type"
            )
        if (
            not torch.all(torch.isfinite(thermal_wavelength_tensor)).item()
            or torch.any(thermal_wavelength_tensor <= 0.0).item()
        ):
            raise ValueError(
                "thermal_wavelength values must be finite and positive"
            )

        self.register_buffer("grid_spacing", spacing)
        self.register_buffer("mean_density", density_scale)
        self.register_buffer("mean_temperature", mean_temperature_tensor)
        self.register_buffer(
            "boltzmann_constant",
            boltzmann_constant_tensor,
        )
        if ideal is None and self.ideal is not None:
            self.ideal.thermal_wavelength = thermal_wavelength_tensor
        # The ideal reference owns the physical wavelength; the property below
        # remains a read-only convenience for grid metadata and conversion.
        if self.ideal is not None:
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
            compute_polarization_derivative=self.compute_polarization_derivative,
            free_energy_mode=self.free_energy_mode,
            message_layers=[
                optional_config(item) for item in self.message_layers
            ],
            mean_density=(
                None if self.mean_density is None
                else scalar_value(self.mean_density)
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

    @staticmethod
    def _local_feature_width(
        a_features: nn.Module,
        b_features: nn.Module,
        n_messages: int,
    ) -> int:
        """Return the width of the local feature vector the readouts receive.

        The vector is ``[center_density?, B_0, ..., B_n_messages, T/T_ref]``:
        the normalized central channels when the feature module separates
        the center from its neighbor moments (polarized features keep their
        center slots inside the invariants instead), one flattened invariant
        block per message level, and the temperature feature.
        """

        invariant_width = (
            a_features.n_radial_channels
            * b_features.n_features
            * a_features.n_output_channels
        )
        center_width = (
            a_features.n_output_channels
            if a_features.separate_center and not a_features.include_polarization
            else 0
        )
        return center_width + (1 + n_messages) * invariant_width + 1

    @property
    def n_local_features(self) -> int:
        """Width of the local feature vector, or zero without local features."""

        if not self.has_local_features:
            return 0
        return self._local_feature_width(
            self.a_features, self.b_features, len(self.message_layers),
        )

    @property
    def cutoff_grid(self) -> int:
        """Integer stencil cutoff used by the local representation."""

        if self.has_local_features:
            return self.a_features.cutoff_grid
        for item in self.readout:
            if item.requires_local_density_index:
                return item.cutoff_grid
        return 0

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
        """Whether local invariant environment features are configured."""

        return self.a_features is not None

    @property
    def requires_local_density_index(self) -> bool:
        """Whether any configured local operator uses explicit gathering."""

        if any(item.requires_local_density_index for item in self.readout):
            return True
        if not self.has_local_features:
            return False
        if self.a_features.convolution_backend == "gather":
            return True
        return any(
            message.convolution_backend == "gather"
            for message in self.message_layers
        )

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
            "thermal_wavelength": (
                self.thermal_wavelength.detach().cpu().tolist()
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

        # B has the contract layout [..., n_grid, n_radial, n_B, n_channels];
        # the three feature axes are flattened into one vector per point.
        grid_rank = data["rho"].ndim - 1
        A = self.a_features(data)
        B = self.b_features(A)
        levels = [B.flatten(start_dim=grid_rank)]

        if len(self.message_layers) > 0:
            shared_basis = self.a_features.stencil_basis()
            message_fields = {}
            if self.a_features.include_polarization:
                fields, scales = self.a_features.polarization_fields(data)
                message_fields = {
                    "normalized_fields": fields / scales,
                    "separate_center": self.a_features.separate_center,
                }
            for message in self.message_layers:
                stencil_basis = message.stencil_basis(
                    self.a_features,
                    shared_basis,
                )
                A = message(
                    B,
                    data.get("local_density_index"),
                    stencil_basis,
                    grid_positions=data.get("grid_positions"),
                    grid_size=data.get("grid_size"),
                    stencil_positions=self.a_features.local_density_positions,
                    **message_fields,
                )
                B = self.b_features(A)
                levels.append(B.flatten(start_dim=grid_rank))
        return torch.cat(levels, dim=-1)

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        compute_c1: Optional[bool] = None,
        compute_c2: Optional[bool] = None,
        c2_reference: Tuple[int, int] = (0, 0),
        compute_polarization_derivative: Optional[bool] = None,
        compute_ideal: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return the collected free-energy and requested response outputs."""

        if METAL_DATA_KEYS & data.keys():
            raise ValueError(
                "Set electrode geometry, constraints, and field on MetalWall, not in data"
            )

        compute_c2 = optional_boolean(compute_c2, "compute_c2")
        if compute_c2 is None:
            compute_c2 = self.compute_c2
        compute_c1 = optional_boolean(compute_c1, "compute_c1")
        if compute_c1 is None:
            compute_c1 = self.compute_c1
        compute_c1 = compute_c1 or compute_c2
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
            )
            if "beta" in data:
                self._check_unit_contract(data["beta"], context.beta)
            rho = context.rho
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
            reserved_outputs = {
                "F_exc", "beta_F_exc", "c1", "c2",
                "beta_mu_local", "beta_mu_average", "beta_mu_weights",
                "polarization_derivative_excess", "beta_F_id", "beta_F",
                "ideal_density_derivative", "ideal_polarization_derivative",
                "density_derivative", "density_derivative_excess", "polarization_derivative",
                "ideal_derivative_mask",
            }
            for item in self.readout:
                energy, extra_outputs = item.energy_and_outputs(context)
                if energy.shape != rho.shape[:-2]:
                    raise ValueError(
                        "every readout must return one scalar energy per field"
                    )
                readout_energies.append(energy)
                conflicts = set(extra_outputs) & (
                    reserved_outputs | set(readout_outputs)
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
                outputs.update(self.ideal.evaluate(data, volume_element))
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

        return outputs
