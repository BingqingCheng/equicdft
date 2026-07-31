"""Collected grid-CACE density-functional model."""

from contextlib import nullcontext
from typing import Dict

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
    output and leaves ``rho`` unchanged.

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
    compute_c1
        If ``True``, initialize density gradients and include ``c1`` in the
        collected outputs. Disable it for energy-only evaluation.
    """

    def __init__(
        self,
        a_features: CartesianAFeatures,
        b_features: CartesianBFeatures,
        readout: LocalReadout,
        compute_c1: bool = True,
    ) -> None:
        super().__init__()

        if not isinstance(compute_c1, bool):
            raise TypeError("compute_c1 must be a boolean")

        self.a_features = a_features
        self.b_features = b_features
        self.readout = readout
        self.compute_c1 = compute_c1

        self.required_derivatives = ["rho"] if compute_c1 else []
        self.model_outputs = [
            "beta_free_energy_per_particle",
            "beta_free_energy_density",
            "beta_F_exc",
        ]
        if compute_c1:
            self.model_outputs.append("c1")

    def initialize_derivatives(
        self,
        data: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Enable gradients for the fields required by response outputs."""

        for key in self.required_derivatives:
            data[key].requires_grad_(True)
        return data

    def extract_outputs(
        self,
        data: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Return only the canonical model outputs."""

        return {key: data[key] for key in self.model_outputs}

    def forward(
        self,
        data: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Return the collected free-energy and requested response outputs."""

        # A requested functional derivative requires gradient tracking even
        # when surrounding evaluation code uses torch.no_grad(). Energy-only
        # evaluation respects the caller's existing gradient context.
        gradient_context = (
            torch.enable_grad() if self.compute_c1 else nullcontext()
        )
        with gradient_context:
            data = self.initialize_derivatives(data)

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

            # grid_spacing has shape [..., 3], so cell_volume has shape [...].
            # beta_F_exc also has shape [...] (or scalar shape [] without a
            # batch) and stores one dimensionless excess free energy per field.
            cell_volume = torch.prod(data["grid_spacing"], dim=-1)
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

            if self.compute_c1:
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
                    / cell_volume[..., None, None]
                )

        return self.extract_outputs(outputs)
