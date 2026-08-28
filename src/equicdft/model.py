"""Collected grid density-functional model."""

from contextlib import nullcontext
from numbers import Integral
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from ._argument_checks import (
    boolean,
    nonnegative_scalar,
    optional_boolean,
    positive_scalar,
)
from ._grid import (
    grid_spacing_tensor,
    require_matching_grid_spacing,
    voxel_volume as compute_voxel_volume,
)
from .derivatives import compute_grid_derivative
from .energy import EnergyReadout
from .features import CartesianAFeatures
from .interaction import BChiMessage
from .symmetrize import CartesianBFeatures


class GridCACEModel(nn.Module):
    """Combine free-energy readouts and differentiate their scalar sum.

    Every configured :class:`EnergyReadout` receives a shared context and
    returns one reduced free-energy contribution per field. With the default
    ``free_energy_mode="beta"``, the sum is interpreted directly as
    ``beta_F_exc``. With ``free_energy_mode="physical"``, the sum is
    interpreted as ``F_exc / (k_B * mean_temperature)`` and converted to
    ``beta_F_exc`` using the input temperature. The resulting
    ``beta_F_exc`` is differentiated with respect to the complete density to
    obtain
    ``c1 = -delta(beta_F_exc)/delta(rho)`` and, optionally, one selected row
    of ``c2 = delta(c1)/delta(rho)``. If ``V_ext`` is available, the model may
    also return the local and spatially averaged dimensionless chemical
    potentials. Specifically, the outputs ``local_chemical_potential`` and
    ``average_chemical_potential`` represent ``beta * mu_local`` and
    ``beta * mu_average``; despite their established key names, they are not
    energy-valued chemical potentials. Constructor response flags provide
    defaults that individual forward calls may override.
    """

    def __init__(
        self,
        a_features: Optional[CartesianAFeatures],
        b_features: Optional[CartesianBFeatures],
        readout: Sequence[EnergyReadout],
        grid_spacing: Union[float, Sequence[float], torch.Tensor],
        mean_temperature: Union[float, torch.Tensor] = 1.0,
        boltzmann_constant: float = 1.0,
        thermal_wavelength: Union[
            float,
            Sequence[float],
            torch.Tensor,
        ] = 1.0,
        compute_c1: bool = True,
        compute_c2: bool = False,
        compute_local_mu: bool = False,
        rho_min: float = 0.0,
        free_energy_mode: str = "beta",
        message_layers: Optional[Sequence[BChiMessage]] = None,
    ) -> None:
        super().__init__()

        compute_c1 = boolean(compute_c1, "compute_c1")
        compute_c2 = boolean(compute_c2, "compute_c2")
        compute_local_mu = boolean(compute_local_mu, "compute_local_mu")
        if compute_local_mu and not (compute_c1 or compute_c2):
            raise ValueError("compute_local_mu requires compute_c1=True")
        if free_energy_mode not in ("beta", "physical"):
            raise ValueError(
                "free_energy_mode must be 'beta' or 'physical'"
            )

        if (a_features is None) != (b_features is None):
            raise ValueError(
                "a_features and b_features must be supplied together"
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
        for module in messages:
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
        if not readouts:
            raise ValueError("readout must contain at least one energy readout")
        if not all(isinstance(module, EnergyReadout) for module in readouts):
            raise TypeError("every readout must be an EnergyReadout module")
        if (
            any(item.requires_local_features for item in readouts)
            and a_features is None
        ):
            raise ValueError(
                "a_features and b_features are required by a configured readout"
            )
        if a_features is not None:
            n_types = a_features.n_types
        else:
            typed_readouts = [
                item for item in readouts if hasattr(item, "n_types")
            ]
            if not typed_readouts:
                raise ValueError("n_types cannot be inferred from the readouts")
            n_types = typed_readouts[0].n_types
        for item in readouts:
            if hasattr(item, "n_types") and item.n_types != n_types:
                raise ValueError("all readouts must use the same n_types")
            if (
                a_features is not None
                and hasattr(item, "mean_density")
                and not torch.allclose(
                    a_features.mean_density,
                    item.mean_density.to(a_features.mean_density),
                )
            ):
                raise ValueError(
                    "readouts and local features must use the same mean_density"
                )
        rho_min = nonnegative_scalar(rho_min, "rho_min")

        self.a_features = a_features
        self.b_features = b_features
        self.message_layers = nn.ModuleList(messages)
        self.readout = nn.ModuleList(readouts)
        self.compute_c1 = compute_c1 or compute_c2
        self.compute_c2 = compute_c2
        self.compute_local_mu = compute_local_mu
        self.rho_min = rho_min
        self.free_energy_mode = free_energy_mode

        spacing = grid_spacing_tensor(
            grid_spacing,
            dtype=torch.get_default_dtype(),
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
        self.register_buffer("mean_temperature", mean_temperature_tensor)
        self.register_buffer(
            "boltzmann_constant",
            boltzmann_constant_tensor,
        )
        self.register_buffer(
            "thermal_wavelength",
            thermal_wavelength_tensor,
        )

    @property
    def cutoff_grid(self) -> int:
        """Integer stencil cutoff used by the local representation."""

        return self.a_features.cutoff_grid if self.has_local_features else 0

    @property
    def n_types(self) -> int:
        """Number of physical density components accepted by the model."""

        if self.has_local_features:
            return self.a_features.n_types
        for item in self.readout:
            if hasattr(item, "n_types"):
                return item.n_types
        raise RuntimeError("n_types is unavailable")

    @property
    def mean_density(self) -> torch.Tensor:
        """Density scale fitted from the development data."""

        if self.has_local_features:
            return self.a_features.mean_density
        for item in self.readout:
            if hasattr(item, "mean_density"):
                return item.mean_density
        raise RuntimeError("mean_density is unavailable")

    @property
    def has_local_features(self) -> bool:
        """Whether local invariant environment features are configured."""

        return getattr(self, "a_features", None) is not None

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
            ),
        }

    def _compute_local_chemical_potential(
        self,
        data: Dict[str, torch.Tensor],
        c1: torch.Tensor,
    ) -> torch.Tensor:
        """Return dimensionless ``beta*mu_local`` at every grid point.

        The returned quantity is
        ``log(rho*Lambda**3) + beta*V_ext - c1``. It is stored under the
        established ``local_chemical_potential`` output key but includes the
        factor ``beta = 1 / (k_B*T)``.
        """

        rho = data["rho"].detach()
        beta = data["beta"].detach().to(rho)
        wavelength = self.thermal_wavelength.to(rho)
        positive_density = rho > 0.0
        safe_rho = torch.where(positive_density, rho, torch.ones_like(rho))
        local_chemical_potential = (
            torch.log(safe_rho * wavelength.pow(3))
            + beta[..., None, None] * data["V_ext"].detach()
            - c1
        )
        return torch.where(
            positive_density,
            local_chemical_potential,
            torch.zeros_like(local_chemical_potential),
        )

    def _weight_mask(self, rho: torch.Tensor) -> torch.Tensor:
        """Return hard averaging weights from the physical density."""

        return (rho.detach() > self.rho_min).to(dtype=rho.dtype)

    def average_chemical_potential(
        self,
        local_chemical_potential: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Return dimensionless ``beta*mu_average`` for each component.

        This is the weighted spatial average of ``beta*mu_local``. The result
        is stored under the established ``average_chemical_potential`` output
        key and therefore also includes the factor ``beta``.
        """

        weights = weights.detach().to(local_chemical_potential)
        total_weight = weights.sum(dim=-2)
        if torch.any(total_weight <= 0.0).item():
            raise ValueError(
                "every field and component must have positive total weight"
            )
        return (
            (weights * local_chemical_potential).sum(dim=-2)
            / total_weight
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

        A = self.a_features(data)
        B = self.b_features(A)
        levels = [B.flatten(start_dim=-3)]

        # getattr keeps full-model checkpoints saved before message passing
        # loadable as ordinary zero-message models.
        messages = getattr(self, "message_layers", ())
        if messages:
            shared_basis = self.a_features.stencil_basis()
            for message in messages:
                radial_exponents = message.radial_exponents
                stencil_basis = (
                    shared_basis
                    if radial_exponents is None
                    else self.a_features.stencil_basis(radial_exponents)
                )
                A = message(
                    B,
                    data["local_density_index"],
                    stencil_basis,
                )
                B = self.b_features(A)
                levels.append(B.flatten(start_dim=-3))
        return torch.cat(levels, dim=-1)

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        compute_c1: Optional[bool] = None,
        compute_c2: Optional[bool] = None,
        c2_reference: Tuple[int, int] = (0, 0),
    ) -> Dict[str, torch.Tensor]:
        """Return the collected free-energy and requested response outputs."""

        compute_c2 = optional_boolean(compute_c2, "compute_c2")
        if compute_c2 is None:
            compute_c2 = self.compute_c2
        compute_c1 = optional_boolean(compute_c1, "compute_c1")
        if compute_c1 is None:
            compute_c1 = self.compute_c1
        compute_c1 = compute_c1 or compute_c2

        # A requested functional derivative requires gradient tracking even
        # when surrounding evaluation code uses torch.no_grad(). Energy-only
        # evaluation respects the caller's existing gradient context.
        gradient_context = (
            torch.enable_grad() if compute_c1 else nullcontext()
        )
        with gradient_context:
            if "grid_spacing" in data:
                require_matching_grid_spacing(
                    data["grid_spacing"],
                    self.grid_spacing,
                )
            if compute_c1:
                data["rho"].requires_grad_(True)

            rho = data["rho"]
            temperature = data["temperature"].to(
                device=rho.device,
                dtype=rho.dtype,
            )
            if temperature.shape != rho.shape[:-2]:
                raise ValueError(
                    "temperature must be scalar for one field or have the "
                    "same leading batch shape as rho"
                )
            normalized_temperature = temperature / self.mean_temperature.to(
                device=rho.device,
                dtype=rho.dtype,
            )

            local_features = None
            if any(
                item.requires_local_features for item in self.readout
            ):
                B_flat = self._local_invariant_features(data)
                temperature_feature = normalized_temperature[
                    ..., None, None
                ].expand(*B_flat.shape[:-1], 1)
                feature_blocks = []
                if getattr(self.a_features, "separate_center", False):
                    feature_blocks.append(
                        rho
                        / self.mean_density.to(
                            device=rho.device,
                            dtype=rho.dtype,
                        )
                    )
                feature_blocks.extend((B_flat, temperature_feature))
                local_features = torch.cat(feature_blocks, dim=-1)

            # Construct the shared global state only when requested. Mean
            # density retains its connection to rho, so energy readouts using
            # this state remain part of the functional derivative.
            state_features = None
            if any(
                item.requires_state_features for item in self.readout
            ):
                mean_density = data["rho"].mean(dim=-2)
                mean_density_feature = (
                    mean_density
                    / self.mean_density.to(
                        device=data["rho"].device,
                        dtype=data["rho"].dtype,
                    )
                )
                state_features = torch.cat(
                    (
                        normalized_temperature[..., None],
                        mean_density_feature,
                    ),
                    dim=-1,
                )

            volume_element = self.voxel_volume.to(
                device=rho.device,
                dtype=rho.dtype,
            )
            context = {
                "rho": rho,
                "normalized_temperature": normalized_temperature,
                "voxel_volume": volume_element,
                "grid_spacing": self.grid_spacing.to(rho),
            }
            if local_features is not None:
                context["local_features"] = local_features
            if state_features is not None:
                context["state_features"] = state_features
            if "grid_size" in data:
                context["grid_size"] = data["grid_size"]

            readout_energies = []
            for item in self.readout:
                energy = item.energy(context)
                if energy.shape != rho.shape[:-2]:
                    raise ValueError(
                        "every readout must return one scalar energy per field"
                    )
                readout_energies.append(energy)

            # Summing the scalar readouts before conversion and
            # differentiation makes every enabled contribution part of one
            # variational functional.
            readout_energy = readout_energies[0]
            for energy in readout_energies[1:]:
                readout_energy = readout_energy + energy

            # getattr preserves full-model checkpoints saved before the mode
            # flag existed; their readouts used the beta-F convention.
            if getattr(self, "free_energy_mode", "beta") == "physical":
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

            if compute_c1:
                # beta_F_exc_derivative and c1 have the same shape as rho:
                #     [..., n_grid, n_types].
                # The generic helper returns the derivative with respect to a
                # discrete rho value; the minus sign and division by Delta V
                # produce the continuum-normalized first direct correlation.
                # Summing independent batch outputs supplies the scalar that
                # torch.autograd requires without embedding batch semantics in
                # the derivative helper.
                beta_F_exc_derivative = compute_grid_derivative(
                    beta_F_exc.sum(),
                    data["rho"],
                    create_graph=(self.training or compute_c2),
                )
                outputs["c1"] = (
                    -beta_F_exc_derivative
                    / volume_element
                )

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

                # Chemical-potential response is controlled by one model flag.
                # Both returned chemical-potential fields are dimensionless
                # beta*mu quantities. V_ext remains optional for intrinsic-
                # functional evaluation.
                if (
                    self.compute_local_mu
                    and "V_ext" in data
                ):
                    local_chemical_potential = (
                        self._compute_local_chemical_potential(
                            data=data,
                            c1=outputs["c1"],
                        )
                    )
                    outputs["local_chemical_potential"] = (
                        local_chemical_potential
                    )
                    weights = self._weight_mask(data["rho"])
                    outputs["chemical_potential_weights"] = weights
                    outputs["average_chemical_potential"] = (
                        self.average_chemical_potential(
                            local_chemical_potential,
                            weights,
                        )
                    )

        return outputs
