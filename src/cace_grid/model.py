"""Collected grid-CACE density-functional model."""

from contextlib import nullcontext
from typing import Dict

import torch
from torch import nn

from .derivatives import compute_c1
from .features import CartesianAFeatures
from .readout import LocalFreeEnergyReadout
from .symmetrize import CartesianBFeatures


class GridCACEModel(nn.Module):
    """Collect the complete density-to-free-energy computational graph.

    The model applies the registered modules in the order

    ``rho -> A -> B -> beta_F_exc -> c1``

    and returns the canonical physical outputs as one dictionary. Density is
    marked as differentiable before the representation is built so ``c1``
    includes every overlapping local environment.

    During training, the graph used to construct ``c1`` is retained so a
    derivative-level loss can be differentiated with respect to model
    parameters. In evaluation mode, ``c1`` is produced without constructing
    the higher-order graph. Setting ``compute_c1=False`` omits the derivative
    output and leaves ``rho`` unchanged.

    The returned dictionary contains:

    ``beta_free_energy_per_particle``
        Dimensionless per-particle excess free energies
        ``beta_a_exc[..., grid, type]`` predicted by the local readout.
    ``beta_free_energy_density``
        Dimensionless excess free-energy density
        ``sum_i rho[..., grid, i] * beta_a_exc[..., grid, i]`` at each grid
        point, before multiplication by the voxel volume.
    ``beta_F_exc``
        Dimensionless excess free energy for each complete density field,
        obtained by multiplying the grid sum of
        ``beta_free_energy_density`` by ``Delta V``.
    ``c1``
        First direct correlation ``-delta(beta_F_exc) / delta(rho)`` in the
        continuum functional-derivative convention. This key is present only
        when ``compute_c1=True``.

    Parameters
    ----------
    a_features
        Module that constructs local Cartesian density moments.
    b_features
        Module that contracts the Cartesian moments into invariant features.
    readout
        Module that predicts the per-particle excess free energy from ``B``.
    compute_c1
        If ``True``, initialize density gradients and include ``c1`` in the
        collected outputs. Disable it for energy-only evaluation.
    """

    def __init__(
        self,
        a_features: CartesianAFeatures,
        b_features: CartesianBFeatures,
        readout: LocalFreeEnergyReadout,
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

            # beta_free_energy_per_particle[..., g, i] is the dimensionless
            # excess free energy assigned to one particle of type i at grid g.
            beta_free_energy_per_particle = self.readout(B)

            # beta_free_energy_density[..., g] is beta*f_exc at grid g. The
            # density is still in physical number-density units, and no voxel
            # volume has been applied at this stage.
            beta_free_energy_density = torch.sum(
                data["rho"] * beta_free_energy_per_particle,
                dim=-1,
            )

            # beta_F_exc[...] is the scalar dimensionless excess free energy
            # of each complete field. cell_volume is Delta V = hx*hy*hz.
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
                # c1[..., g, i] is the continuum-normalized first direct
                # correlation. A later c2 branch should differentiate this
                # field in derivatives.py and append another collected output
                # here, without changing the local readout.
                outputs["c1"] = compute_c1(
                    beta_F_exc,
                    data["rho"],
                    data["grid_spacing"],
                    create_graph=self.training,
                )

        return self.extract_outputs(outputs)
