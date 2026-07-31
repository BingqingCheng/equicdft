"""Collected grid-CACE density-functional model."""

from contextlib import nullcontext
from numbers import Real
from typing import Dict, Optional, Sequence, Union

import torch
from torch import nn

from .derivatives import compute_grid_derivative
from .features import CartesianAFeatures
from .readout import LocalReadout
from .symmetrize import CartesianBFeatures


class GridCACEModel(nn.Module):
    """Collect the complete density-to-free-energy computational graph.

    The model applies the registered modules in the order

    ``rho -> A -> B -> (B, T) -> beta_F_exc -> c1``

    and returns the canonical physical outputs as one dictionary. Density is
    marked as differentiable before the representation is built so ``c1``
    includes every overlapping local environment.

    During training, the graph used to construct ``c1`` is retained so a
    derivative-level loss can be differentiated with respect to model
    parameters. In evaluation mode, ``c1`` is produced without constructing
    the higher-order graph. Setting ``compute_c1=False`` omits the derivative
    output and leaves ``rho`` unchanged. The constructor value is the default;
    an individual forward call can override it without mutating the model.

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
        energy for each complete density field, obtained by multiplying the
        grid sum of
        ``beta_free_energy_density`` by ``Delta V``.
    ``c1``
        Shape ``[..., n_grid, n_types]``. First direct correlation
        ``-delta(beta_F_exc) / delta(rho)`` in the continuum
        functional-derivative convention. This key is present only when
        ``compute_c1=True``.

    Parameters
    ----------
    a_features
        Module that constructs local Cartesian density moments.
    b_features
        Module that contracts the Cartesian moments into invariant features.
    readout
        Module that predicts the per-particle excess free energy from the
        flattened local ``B`` features and scalar temperature.
    grid_spacing
        One value for an isotropic grid or three Cartesian spacings. The
        discretization is fixed by training and stored with the model.
    boltzmann_constant
        Boltzmann constant in the energy and temperature units used to train
        the model. It is stored for inference thermodynamics.
    thermal_wavelength
        One positive value or one value per density component. It fixes the
        ideal-gas convention used to reconstruct external potentials.
    compute_c1
        If ``True``, initialize density gradients and include ``c1`` in the
        collected outputs. Disable it for energy-only evaluation.
    """

    def __init__(
        self,
        a_features: CartesianAFeatures,
        b_features: CartesianBFeatures,
        readout: LocalReadout,
        grid_spacing: Union[float, Sequence[float], torch.Tensor],
        boltzmann_constant: float = 1.0,
        thermal_wavelength: Union[
            float,
            Sequence[float],
            torch.Tensor,
        ] = 1.0,
        compute_c1: bool = True,
    ) -> None:
        super().__init__()

        if not isinstance(compute_c1, bool):
            raise TypeError("compute_c1 must be a boolean")

        self.a_features = a_features
        self.b_features = b_features
        self.readout = readout
        self.compute_c1 = compute_c1

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
        self.register_buffer(
            "boltzmann_constant",
            boltzmann_constant_tensor,
        )
        self.register_buffer(
            "thermal_wavelength",
            thermal_wavelength_tensor,
        )

        self.required_derivatives = ["rho"] if compute_c1 else []
        self.model_outputs = [
            "beta_free_energy_per_particle",
            "beta_free_energy_density",
            "beta_F_exc",
        ]
        if compute_c1:
            self.model_outputs.append("c1")

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
    def cell_volume(self) -> torch.Tensor:
        """Volume represented by one grid point."""

        return torch.prod(self.grid_spacing)

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
    ) -> Dict[str, torch.Tensor]:
        """Return only the canonical model outputs."""

        if compute_c1 is None:
            compute_c1 = self.compute_c1
        model_outputs = [
            "beta_free_energy_per_particle",
            "beta_free_energy_density",
            "beta_F_exc",
        ]
        if compute_c1:
            model_outputs.append("c1")
        return {key: data[key] for key in model_outputs}

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        compute_c1: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return the collected free-energy and requested response outputs."""

        if compute_c1 is None:
            compute_c1 = self.compute_c1
        if not isinstance(compute_c1, bool):
            raise TypeError("compute_c1 must be a boolean or None")

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
            # feature vector [..., n_grid, n_B]. Temperature has shape [...]
            # and is broadcast over the grid before being appended exactly
            # once, giving [..., n_grid, n_B + 1].
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
            temperature_feature = temperature[..., None, None].expand(
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
            beta_F_exc = cell_volume * torch.sum(
                beta_free_energy_density,
                dim=-1,
            )

            outputs = {
                "beta_free_energy_per_particle": (
                    beta_free_energy_per_particle
                ),
                "beta_free_energy_density": beta_free_energy_density,
                "beta_F_exc": beta_F_exc,
            }

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
                    create_graph=self.training,
                )
                outputs["c1"] = (
                    -beta_F_exc_derivative
                    / cell_volume
                )

        return self.extract_outputs(outputs, compute_c1=compute_c1)
