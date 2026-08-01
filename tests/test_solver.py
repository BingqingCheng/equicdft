import unittest

import torch
from torch import nn

from equicdft import GridCACEModel, GridData, GridSolver


class _DensityFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.cutoff_grid = 0
        self.n_types = 1
        self.register_buffer("mean_density", torch.tensor(1.0))

    def forward(self, data):
        return data["rho"].unsqueeze(-2).unsqueeze(-2)


class _IdentityModule(nn.Module):
    def forward(self, values):
        return values


class _DensityReadout(nn.Module):
    def forward(self, local_features):
        return local_features[..., :1]


class TestGridSolver(unittest.TestCase):
    def _make_model(self):
        # beta_F_exc = sum_g rho_g^2 and c1 = -2 rho for unit voxels.
        return GridCACEModel(
            a_features=_DensityFeatures(),
            b_features=_IdentityModule(),
            readout=_DensityReadout(),
            grid_spacing=1.0,
            boltzmann_constant=1.0,
            thermal_wavelength=1.0,
            compute_c1=True,
        )

    def _make_data(self, target_rho=None, include_mu=True):
        if target_rho is None:
            target_rho = torch.tensor(
                [[0.2], [0.3], [0.4], [0.5]],
                dtype=torch.float64,
            )
        mu = torch.tensor([0.0], dtype=target_rho.dtype)
        V_ext = mu[None, :] - torch.log(target_rho) - 2.0 * target_rho
        data = GridData.from_dict(
            {
                "grid_size": [4, 1, 1],
                "n_types": 1,
                "grid_spacing": 1.0,
                "temperature": 1.0,
            },
            cutoff_grid=0,
            boltzmann_constant=1.0,
        )
        data["V_ext"] = V_ext
        if include_mu:
            data["mu"] = mu
        return data

    def test_evaluate_collects_functional_and_thermodynamics(self):
        rho = torch.tensor(
            [[0.2], [0.3], [0.4], [0.5]],
            dtype=torch.float64,
        )
        data = self._make_data(rho)
        data["rho"] = rho

        result = GridSolver(self._make_model()).evaluate(data)

        self.assertTrue(torch.allclose(result["c1"], -2.0 * rho))
        self.assertTrue(
            torch.allclose(result["beta_F_exc"], torch.sum(rho.square()))
        )
        expected_F_id = torch.sum(rho * (torch.log(rho) - 1.0))
        self.assertTrue(torch.allclose(result["beta_F_id"], expected_F_id))
        self.assertTrue(
            torch.allclose(
                result["euler_lagrange_residual"],
                torch.zeros_like(rho),
                atol=1.0e-12,
            )
        )

    def test_grand_canonical_solve_recovers_stationary_density(self):
        target_rho = torch.tensor(
            [[0.2], [0.3], [0.4], [0.5]],
            dtype=torch.float64,
        )

        result = GridSolver(self._make_model()).solve(
            self._make_data(target_rho),
            max_iter=200,
            tolerance_grad=1.0e-9,
            tolerance_change=1.0e-12,
        )

        self.assertTrue(
            torch.allclose(result["rho"], target_rho, atol=2.0e-6, rtol=0.0)
        )
        self.assertLess(
            torch.max(torch.abs(result["euler_lagrange_residual"])).item(),
            1.0e-5,
        )

    def test_fixed_particle_number_solve_enforces_constraint(self):
        data = self._make_data(include_mu=False)
        data["V_ext"].zero_()

        result = GridSolver(self._make_model()).solve(
            data,
            particle_numbers=[2.0],
        )

        expected_rho = torch.full(
            (4, 1),
            0.5,
            dtype=torch.float64,
        )
        self.assertTrue(torch.allclose(result["rho"], expected_rho))
        self.assertAlmostEqual(result["rho"].sum().item(), 2.0)
        self.assertLess(
            torch.max(torch.abs(result["euler_lagrange_residual"])).item(),
            1.0e-12,
        )


if __name__ == "__main__":
    unittest.main()
