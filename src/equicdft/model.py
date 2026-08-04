"""Collected grid-CACE density-functional model."""

import math
from contextlib import nullcontext
from numbers import Integral, Real
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from .derivatives import compute_grid_derivative
from .features import CartesianAFeatures
from .readout import BulkReadout, LocalReadout, LongRangeReadout
from .reciprocal import ReciprocalFeatures
from .symmetrize import CartesianBFeatures


class GridCACEModel(nn.Module):
    """Collect the complete density-to-free-energy computational graph.

    The model applies the registered modules in the order

    ``rho -> A -> B -> (B, T) -> beta_F_exc_local``

    with the optional homogeneous branch

    ``(mean(rho), T) -> beta_F_exc_bulk``

    with the optional reciprocal branch

    ``rho -> reciprocal features -> beta_F_exc_long_range``.

    The scalar energy contributions are added before the response calculation,

    ``beta_F_exc -> c1 -> c2``,

    and, when an external field is supplied,

    ``(rho, V_ext, T, c1) -> local_chemical_potential``.

    It returns the canonical physical outputs as one dictionary. Density is
    marked as differentiable before the representation is built so ``c1``
    includes every overlapping local environment.

    During training, the graph used to construct response outputs is retained
    so a derivative-level loss can be differentiated with respect to model
    parameters. A requested ``c2`` row also retains the ``c1`` graph long
    enough to take the second density derivative. In evaluation mode, a
    ``c1``-only calculation avoids this higher-order graph. Setting both
    response flags to ``False`` leaves ``rho`` unchanged. Constructor values
    are defaults that an individual forward call can override.

    In the shapes below, ``...`` denotes optional leading batch dimensions,
    ``g`` runs over ``n_grid`` grid points, and ``i`` runs over ``n_types``
    density components. The returned dictionary contains:

    ``beta_free_energy_per_particle``
        Shape ``[..., n_grid, n_types]``. Dimensionless per-particle excess
        free energies
        ``beta_a_exc[..., grid, type]`` predicted by the local readout.
    ``beta_free_energy_density``
        Shape ``[..., n_grid]``. Dimensionless excess free-energy density
        ``sum_i rho[..., grid, i] * beta_a_exc[..., grid, i]`` at each grid
        point, before multiplication by the voxel volume.
    ``beta_F_exc``
        Shape ``[...]`` (a scalar without batching). Dimensionless excess free
        energy for each complete density field. It is the sum of the local,
        optional bulk, and optional long-range contributions.
    ``beta_F_exc_local``
        Shape ``[...]``. Integral of ``beta_free_energy_density`` over the
        grid. This key is present when a bulk or reciprocal branch is
        configured.
    ``beta_bulk_free_energy_per_particle``
        Shape ``[..., n_types]``. Homogeneous dimensionless excess free energy
        per particle predicted from normalized temperature and mean densities.
        This key is present when a bulk branch is configured.
    ``beta_F_exc_bulk``
        Shape ``[...]``. Extensive homogeneous contribution
        ``sum_i N_i beta_a_exc_bulk_i``. This key is present when a bulk branch
        is configured.
    ``beta_F_exc_long_range``
        Shape ``[...]``. Nonlocal reciprocal-space contribution for each
        complete density field. This key is present when a reciprocal branch
        is configured.
    ``c1``
        Shape ``[..., n_grid, n_types]``. First direct correlation
        ``-delta(beta_F_exc) / delta(rho)`` in the continuum
        functional-derivative convention. This key is present only when
        ``compute_c1=True``.
    ``c2``
        Shape ``[..., n_grid, n_types]``. One row of the second direct
        correlation, obtained by differentiating the selected
        ``c1[..., reference_grid, reference_type]`` with respect to the
        complete density field. This key is present only when
        ``compute_c2=True``. For a homogeneous fluid, translational symmetry
        makes this row a function of relative grid displacement.
    ``local_chemical_potential``
        Shape ``[..., n_grid, n_types]``. Dimensionless local chemical
        potential obtained from the Euler--Lagrange equation. This key is
        returned when ``compute_local_mu=True``, ``compute_c1`` is enabled,
        and the input contains ``V_ext``.
    ``average_chemical_potential``
        Shape ``[..., n_types]``. Hard-mask-weighted spatial average of
        ``local_chemical_potential`` with weights ``rho > rho_min``. It is
        returned together with the local field when ``compute_local_mu=True``.
    ``chemical_potential_weights``
        Shape ``[..., n_grid, n_types]``. Detached hard-mask weights used for
        both the average and the local-chemical-potential loss.

    Parameters
    ----------
    a_features
        Module that constructs local Cartesian density moments.
    b_features
        Module that contracts the Cartesian moments into invariant features.
    readout
        Module that predicts the per-particle excess free energy from the
        flattened local ``B`` features and normalized scalar temperature.
    bulk_readout
        Optional module that maps normalized temperature and mean densities to
        one homogeneous per-particle excess free energy per component. The
        model multiplies these values by the corresponding particle numbers
        and adds the resulting scalar to the local and long-range energies.
    long_range_features
        Optional module that constructs translation-invariant reciprocal
        quadratic features from the complete density field.
    long_range_readout
        Optional module that maps temperature and mean density to reciprocal
        kernel coefficients and contracts them with ``long_range_features``.
        Supply both long-range modules or neither.
    grid_spacing
        One value for an isotropic grid or three Cartesian spacings. The
        discretization is fixed by training and stored with the model.
    mean_temperature
        Positive temperature scale computed from the development data. The
        readout receives ``temperature / mean_temperature``; physical
        thermodynamic expressions continue to use the unscaled temperature.
    boltzmann_constant
        Boltzmann constant in the energy and temperature units used to train
        the model. It is stored for inference thermodynamics.
    thermal_wavelength
        One positive value or one value per density component. It fixes the
        ideal-gas convention used to reconstruct external potentials.
    compute_c1
        If ``True``, initialize density gradients and include ``c1`` in the
        collected outputs. Disable it for energy-only evaluation.
    compute_c2
        If ``True``, include one selected row of ``c2`` in the collected
        outputs. Computing ``c2`` requires ``c1`` and therefore enables it
        automatically. The reference grid and component may be selected on
        each forward call.
    compute_local_mu
        If ``True``, include both ``local_chemical_potential`` and its
        hard-mask-weighted ``average_chemical_potential``. This requires
        ``c1``, enabled by either response flag.
    rho_min
        Nonnegative density threshold defining the hard weights used by
        :meth:`average_chemical_potential`.
    """

    def __init__(
        self,
        a_features: CartesianAFeatures,
        b_features: CartesianBFeatures,
        readout: LocalReadout,
        grid_spacing: Union[float, Sequence[float], torch.Tensor],
        mean_temperature: Union[float, torch.Tensor] = 1.0,
        boltzmann_constant: float = 1.0,
        thermal_wavelength: Union[
            float,
            Sequence[float],
            torch.Tensor,
        ] = 1.0,
        long_range_features: Optional[ReciprocalFeatures] = None,
        long_range_readout: Optional[LongRangeReadout] = None,
        compute_c1: bool = True,
        compute_c2: bool = False,
        compute_local_mu: bool = False,
        rho_min: float = 0.0,
        bulk_readout: Optional[BulkReadout] = None,
    ) -> None:
        super().__init__()

        if not isinstance(compute_c1, bool):
            raise TypeError("compute_c1 must be a boolean")
        if not isinstance(compute_c2, bool):
            raise TypeError("compute_c2 must be a boolean")
        if not isinstance(compute_local_mu, bool):
            raise TypeError("compute_local_mu must be a boolean")
        if compute_local_mu and not (compute_c1 or compute_c2):
            raise ValueError("compute_local_mu requires compute_c1=True")
        if bulk_readout is not None:
            if not isinstance(bulk_readout, BulkReadout):
                raise TypeError("bulk_readout must be BulkReadout or None")
            if bulk_readout.n_types != a_features.n_types:
                raise ValueError(
                    "bulk_readout n_types does not match local features"
                )
        if (long_range_features is None) != (long_range_readout is None):
            raise ValueError(
                "long_range_features and long_range_readout must be supplied "
                "together"
            )
        if long_range_features is not None:
            if not isinstance(long_range_features, ReciprocalFeatures):
                raise TypeError(
                    "long_range_features must be ReciprocalFeatures or None"
                )
            if not isinstance(long_range_readout, LongRangeReadout):
                raise TypeError(
                    "long_range_readout must be LongRangeReadout or None"
                )
            if long_range_features.n_types != a_features.n_types:
                raise ValueError(
                    "long_range_features n_types does not match local features"
                )
            if long_range_readout.n_types != a_features.n_types:
                raise ValueError(
                    "long_range_readout n_types does not match local features"
                )
            if (
                long_range_readout.n_kernels
                != long_range_features.n_kernels
            ):
                raise ValueError(
                    "long-range feature and readout kernel counts differ"
                )
        try:
            rho_min = float(rho_min)
        except (TypeError, ValueError):
            raise ValueError("rho_min must be a finite nonnegative scalar")
        if not math.isfinite(rho_min) or rho_min < 0.0:
            raise ValueError("rho_min must be a finite nonnegative scalar")

        self.a_features = a_features
        self.b_features = b_features
        self.readout = readout
        self.bulk_readout = bulk_readout
        self.long_range_features = long_range_features
        self.long_range_readout = long_range_readout
        self.compute_c1 = compute_c1 or compute_c2
        self.compute_c2 = compute_c2
        self.compute_local_mu = compute_local_mu
        self.rho_min = rho_min

        grid_spacing_tensor = torch.as_tensor(
            grid_spacing,
            dtype=torch.get_default_dtype(),
        ).detach().clone().reshape(-1)
        if grid_spacing_tensor.numel() == 1:
            grid_spacing_tensor = grid_spacing_tensor.repeat(3)
        if grid_spacing_tensor.shape != (3,):
            raise ValueError(
                "grid_spacing must contain one or three values"
            )
        if (
            not torch.all(torch.isfinite(grid_spacing_tensor)).item()
            or torch.any(grid_spacing_tensor <= 0.0).item()
        ):
            raise ValueError("grid_spacing values must be finite and positive")

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

        if isinstance(boltzmann_constant, bool) or not isinstance(
            boltzmann_constant,
            Real,
        ):
            raise TypeError("boltzmann_constant must be a positive scalar")
        boltzmann_constant_tensor = torch.as_tensor(
            boltzmann_constant,
            dtype=torch.get_default_dtype(),
        )
        if (
            not torch.isfinite(boltzmann_constant_tensor).item()
            or boltzmann_constant_tensor.item() <= 0.0
        ):
            raise ValueError("boltzmann_constant must be finite and positive")

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

        self.register_buffer("grid_spacing", grid_spacing_tensor)
        self.register_buffer("mean_temperature", mean_temperature_tensor)
        self.register_buffer(
            "boltzmann_constant",
            boltzmann_constant_tensor,
        )
        self.register_buffer(
            "thermal_wavelength",
            thermal_wavelength_tensor,
        )

        self.required_derivatives = ["rho"] if self.compute_c1 else []
        self.model_outputs = [
            "beta_free_energy_per_particle",
            "beta_free_energy_density",
        ]
        if self.has_bulk or self.has_long_range:
            self.model_outputs.append("beta_F_exc_local")
        if self.has_bulk:
            self.model_outputs.extend(
                (
                    "beta_bulk_free_energy_per_particle",
                    "beta_F_exc_bulk",
                )
            )
        if self.has_long_range:
            self.model_outputs.append("beta_F_exc_long_range")
        self.model_outputs.append("beta_F_exc")
        if self.compute_c1:
            self.model_outputs.append("c1")
        if compute_c2:
            self.model_outputs.append("c2")
        if compute_local_mu:
            self.model_outputs.extend(
                (
                    "local_chemical_potential",
                    "average_chemical_potential",
                    "chemical_potential_weights",
                )
            )

    @property
    def cutoff_grid(self) -> int:
        """Integer stencil cutoff used by the local representation."""

        return self.a_features.cutoff_grid

    @property
    def n_types(self) -> int:
        """Number of physical density components accepted by the model."""

        return self.a_features.n_types

    @property
    def mean_density(self) -> torch.Tensor:
        """Density scale fitted from the development data."""

        return self.a_features.mean_density

    @property
    def has_long_range(self) -> bool:
        """Whether the model contains the optional reciprocal branch."""

        return self.long_range_features is not None

    @property
    def has_bulk(self) -> bool:
        """Whether the model contains the optional homogeneous branch."""

        return getattr(self, "bulk_readout", None) is not None

    @property
    def cell_volume(self) -> torch.Tensor:
        """Volume represented by one grid point."""

        return torch.prod(self.grid_spacing)

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

    def compute_local_chemical_potential(
        self,
        data: Dict[str, torch.Tensor],
        c1: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``log(rho*Lambda**3) + beta*V_ext - c1``."""

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

    def weight_mask(self, rho: torch.Tensor) -> torch.Tensor:
        """Return hard averaging weights from the physical density."""

        return (rho.detach() > self.rho_min).to(dtype=rho.dtype)

    def average_chemical_potential(
        self,
        local_chemical_potential: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Return the componentwise weighted average over grid points."""

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

    def _validate_grid_spacing(self, data: Dict[str, torch.Tensor]) -> None:
        """Reject grids inconsistent with the trained discretization."""

        if "grid_spacing" not in data:
            return
        input_spacing = data["grid_spacing"]
        if input_spacing.shape[-1:] != (3,):
            raise ValueError(
                "input grid_spacing must have three Cartesian values"
            )
        expected = self.grid_spacing.to(
            device=input_spacing.device,
            dtype=input_spacing.dtype,
        ).expand_as(input_spacing)
        if not torch.allclose(input_spacing, expected):
            raise ValueError(
                "input grid_spacing does not match the trained model"
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

    def initialize_derivatives(
        self,
        data: Dict[str, torch.Tensor],
        compute_c1: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """Enable gradients for the fields required by response outputs."""

        if compute_c1 is None:
            compute_c1 = self.compute_c1
        required_derivatives = ["rho"] if compute_c1 else []
        for key in required_derivatives:
            data[key].requires_grad_(True)
        return data

    def extract_outputs(
        self,
        data: Dict[str, torch.Tensor],
        compute_c1: Optional[bool] = None,
        compute_c2: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return only the canonical model outputs."""

        if compute_c2 is None:
            compute_c2 = self.compute_c2
        if compute_c1 is None:
            compute_c1 = self.compute_c1
        compute_c1 = compute_c1 or compute_c2
        requested_outputs = [
            "beta_free_energy_per_particle",
            "beta_free_energy_density",
        ]
        if self.has_bulk or self.has_long_range:
            requested_outputs.append("beta_F_exc_local")
        if self.has_bulk:
            requested_outputs.extend(
                (
                    "beta_bulk_free_energy_per_particle",
                    "beta_F_exc_bulk",
                )
            )
        if self.has_long_range:
            requested_outputs.append("beta_F_exc_long_range")
        requested_outputs.append("beta_F_exc")
        if compute_c1:
            requested_outputs.append("c1")
            if compute_c2:
                requested_outputs.append("c2")
            if self.compute_local_mu:
                requested_outputs.extend(
                    (
                        "local_chemical_potential",
                        "average_chemical_potential",
                        "chemical_potential_weights",
                    )
                )
        return {
            key: data[key]
            for key in requested_outputs
            if key in data
        }

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        compute_c1: Optional[bool] = None,
        compute_c2: Optional[bool] = None,
        c2_reference: Tuple[int, int] = (0, 0),
    ) -> Dict[str, torch.Tensor]:
        """Return the collected free-energy and requested response outputs."""

        if compute_c2 is None:
            compute_c2 = self.compute_c2
        if not isinstance(compute_c2, bool):
            raise TypeError("compute_c2 must be a boolean or None")
        if compute_c1 is None:
            compute_c1 = self.compute_c1
        if not isinstance(compute_c1, bool):
            raise TypeError("compute_c1 must be a boolean or None")
        compute_c1 = compute_c1 or compute_c2

        # A requested functional derivative requires gradient tracking even
        # when surrounding evaluation code uses torch.no_grad(). Energy-only
        # evaluation respects the caller's existing gradient context.
        gradient_context = (
            torch.enable_grad() if compute_c1 else nullcontext()
        )
        with gradient_context:
            self._validate_grid_spacing(data)
            data = self.initialize_derivatives(data, compute_c1=compute_c1)

            # A[..., g, n, k, c] contains Cartesian density moment k in radial
            # channel n and descriptor channel c around grid point g. Channel
            # c is physical without mixing and latent when mixing is enabled.
            A = self.a_features(data)

            # B[..., g, n, gamma, c] contains cubic-invariant contraction gamma
            # formed from the Cartesian components of A.
            B = self.b_features(A)

            # Flatten the radial, invariant, and channel axes into one local
            # feature vector [..., n_grid, n_B]. The scale-only normalized
            # temperature T/<T> has shape [...] and is broadcast over the
            # grid before being appended once, giving [..., n_grid, n_B + 1].
            B_flat = B.flatten(start_dim=-3)
            temperature = data["temperature"].to(
                device=B_flat.device,
                dtype=B_flat.dtype,
            )
            if temperature.shape != B_flat.shape[:-2]:
                raise ValueError(
                    "temperature must be scalar for one field or have the "
                    "same leading batch shape as B"
                )
            normalized_temperature = temperature / self.mean_temperature.to(
                device=B_flat.device,
                dtype=B_flat.dtype,
            )
            temperature_feature = normalized_temperature[
                ..., None, None
            ].expand(
                *B_flat.shape[:-1],
                1,
            )
            local_features = torch.cat(
                (B_flat, temperature_feature),
                dim=-1,
            )

            # beta_free_energy_per_particle has shape
            #     [..., n_grid, n_types].
            # Entry [..., g, i] is the dimensionless excess free energy
            # assigned to one particle of type i at grid point g.
            beta_free_energy_per_particle = self.readout(local_features)

            # beta_free_energy_density has shape [..., n_grid]. Entry [..., g]
            # is beta*f_exc at grid point g. The density is still in physical
            # number-density units, and no voxel volume has been applied.
            beta_free_energy_density = torch.sum(
                data["rho"] * beta_free_energy_per_particle,
                dim=-1,
            )

            # cell_volume is a scalar fixed by the training-grid
            # discretization. beta_F_exc has shape [...] (or scalar shape []
            # without a batch) and stores one dimensionless excess free energy
            # per field.
            cell_volume = self.cell_volume.to(
                device=beta_free_energy_density.device,
                dtype=beta_free_energy_density.dtype,
            )
            beta_F_exc_local = cell_volume * torch.sum(
                beta_free_energy_density,
                dim=-1,
            )

            # The bulk and reciprocal branches use the same global state:
            # normalized temperature and one normalized mean density per
            # component. Mean density retains its connection to rho so both
            # optional energies contribute to functional derivatives.
            if self.has_bulk or self.has_long_range:
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

            if self.has_bulk:
                # beta_bulk_free_energy_per_particle has shape
                # [..., n_types]. Multiplication by particle_numbers makes the
                # bulk contribution extensive and exactly zero in vacuum.
                beta_bulk_free_energy_per_particle = self.bulk_readout(
                    state_features
                )
                if beta_bulk_free_energy_per_particle.shape != mean_density.shape:
                    raise ValueError(
                        "bulk readout must return one value per field and type"
                    )
                particle_numbers = cell_volume * torch.sum(
                    data["rho"],
                    dim=-2,
                )
                beta_F_exc_bulk = torch.sum(
                    particle_numbers * beta_bulk_free_energy_per_particle,
                    dim=-1,
                )
                if beta_F_exc_bulk.shape != beta_F_exc_local.shape:
                    raise ValueError(
                        "bulk readout must return one scalar energy per field"
                    )
            else:
                beta_F_exc_bulk = torch.zeros_like(beta_F_exc_local)

            if self.has_long_range:
                if "grid_size" not in data:
                    raise KeyError(
                        "long-range evaluation requires data['grid_size']"
                    )
                reciprocal_features = self.long_range_features(
                    rho=data["rho"],
                    grid_size=data["grid_size"],
                    grid_spacing=self.grid_spacing,
                )
                beta_F_exc_long_range = self.long_range_readout(
                    reciprocal_features,
                    state_features,
                )
                if beta_F_exc_long_range.shape != beta_F_exc_local.shape:
                    raise ValueError(
                        "long-range readout must return one scalar per field"
                    )
            else:
                beta_F_exc_long_range = torch.zeros_like(beta_F_exc_local)

            beta_F_exc = (
                beta_F_exc_local
                + beta_F_exc_bulk
                + beta_F_exc_long_range
            )

            outputs = {
                "beta_free_energy_per_particle": (
                    beta_free_energy_per_particle
                ),
                "beta_free_energy_density": beta_free_energy_density,
                "beta_F_exc": beta_F_exc,
            }
            if self.has_bulk or self.has_long_range:
                outputs["beta_F_exc_local"] = beta_F_exc_local
            if self.has_bulk:
                outputs["beta_bulk_free_energy_per_particle"] = (
                    beta_bulk_free_energy_per_particle
                )
                outputs["beta_F_exc_bulk"] = beta_F_exc_bulk
            if self.has_long_range:
                outputs["beta_F_exc_long_range"] = beta_F_exc_long_range

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
                    / cell_volume
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
                    ) / cell_volume
                    if not self.training:
                        # The c1 graph was needed only as an intermediate for
                        # c2. Match the graph-free evaluation semantics of a
                        # c1-only forward call before returning it.
                        outputs["c1"] = outputs["c1"].detach()

                # Chemical-potential response is controlled by one model flag.
                # V_ext remains optional for intrinsic-functional evaluation.
                if (
                    self.compute_local_mu
                    and "V_ext" in data
                ):
                    local_chemical_potential = (
                        self.compute_local_chemical_potential(
                            data=data,
                            c1=outputs["c1"],
                        )
                    )
                    outputs["local_chemical_potential"] = (
                        local_chemical_potential
                    )
                    weights = self.weight_mask(data["rho"])
                    outputs["chemical_potential_weights"] = weights
                    outputs["average_chemical_potential"] = (
                        self.average_chemical_potential(
                            local_chemical_potential,
                            weights,
                        )
                    )

        return self.extract_outputs(
            outputs,
            compute_c1=compute_c1,
            compute_c2=compute_c2,
        )
