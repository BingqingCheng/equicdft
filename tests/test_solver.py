import unittest

import torch
from torch import nn

from equicdft import GridCACEModel, GridData, GridSolver
from equicdft.solver import _euler_residual, _residuals_converged


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
    def assert_objective_histories_nonincreasing(self, histories):
        """Check monotonic descent separately for every continued field."""

        for history in histories:
            for previous, current in zip(history, history[1:]):
                self.assertLessEqual(current, previous + 1.0e-12)

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
            compute_local_mu=True,
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
            continuation_steps=0,
            tolerance_residual=1.0e-8,
        )

        self.assertTrue(
            torch.allclose(result["rho"], target_rho, atol=2.0e-6, rtol=0.0)
        )
        self.assertLess(
            torch.max(torch.abs(result["euler_lagrange_residual"])).item(),
            1.0e-7,
        )
        self.assertLess(result["max_euler_lagrange_residual"], 1.0e-7)
        self.assertEqual(result["solver_method"], "minimize")
        self.assertEqual(result["line_search_failures"], 0)
        self.assert_objective_histories_nonincreasing(
            result["stage_objective_history"]
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
        self.assertTrue(result["converged"])
        self.assert_objective_histories_nonincreasing(
            result["stage_objective_history"]
        )

    def test_fixed_particle_solver_recovers_from_concentrated_initial_density(
        self,
    ):
        data = self._make_data(include_mu=False)
        data["V_ext"].zero_()
        initial_rho = torch.tensor(
            [[1.9997], [0.0001], [0.0001], [0.0001]],
            dtype=torch.float64,
        )

        result = GridSolver(self._make_model()).solve(
            data,
            initial_rho=initial_rho,
            particle_numbers=[2.0],
            max_iter=200,
            continuation_steps=1,
            tolerance_residual=1.0e-7,
        )

        expected_rho = torch.full(
            (4, 1),
            0.5,
            dtype=torch.float64,
        )
        self.assertTrue(
            torch.allclose(result["rho"], expected_rho, atol=1.0e-7)
        )
        self.assertTrue(result["converged"])
        self.assertLess(result["max_euler_lagrange_residual"], 1.0e-7)
        self.assertTrue(
            any(
                history[-1] < history[0]
                for history in result["stage_objective_history"]
            )
        )
        self.assert_objective_histories_nonincreasing(
            result["stage_objective_history"]
        )

    def test_euler_method_remains_available(self):
        target_rho = torch.tensor(
            [[0.2], [0.3], [0.4], [0.5]],
            dtype=torch.float64,
        )

        result = GridSolver(self._make_model()).solve(
            self._make_data(target_rho),
            method="euler",
            max_iter=200,
            mixing=0.2,
            adaptive_mixing=False,
            tolerance_residual=1.0e-9,
            tolerance_change=1.0e-12,
        )

        self.assertEqual(result["solver_method"], "euler")
        self.assertTrue(result["converged"])
        self.assertTrue(
            torch.allclose(result["rho"], target_rho, atol=2.0e-6, rtol=0.0)
        )

    def test_default_adaptive_euler_mixing_is_bounded_and_converges(self):
        target_rho = torch.tensor(
            [[0.2], [0.3], [0.4], [0.5]],
            dtype=torch.float64,
        )

        result = GridSolver(self._make_model()).solve(
            self._make_data(target_rho),
            method="euler",
            max_iter=200,
            mixing=0.05,
            minimum_mixing=0.01,
            maximum_mixing=0.4,
            mixing_growth=1.2,
            tolerance_residual=1.0e-8,
            tolerance_rms_residual=1.0e-8,
            tolerance_change=1.0e-12,
        )

        self.assertTrue(result["converged"])
        self.assertGreaterEqual(result["final_mixing"], 0.01)
        self.assertLessEqual(result["final_mixing"], 0.4)
        self.assertGreaterEqual(result["mixing_backtracks"], 0)
        self.assertTrue(
            torch.allclose(result["rho"], target_rho, atol=2.0e-6, rtol=0.0)
        )

    def test_anderson_acceleration_reduces_fixed_point_evaluations(self):
        target_rho = torch.tensor(
            [[0.2], [0.3], [0.4], [0.5]],
            dtype=torch.float64,
        )
        settings = {
            "method": "euler",
            "max_iter": 200,
            "mixing": 0.05,
            "continuation_steps": 0,
            "tolerance_residual": 1.0e-8,
            "tolerance_rms_residual": 1.0e-8,
            "tolerance_change": 1.0e-12,
        }

        baseline = GridSolver(self._make_model()).solve(
            self._make_data(target_rho),
            **settings,
        )
        accelerated = GridSolver(self._make_model()).solve(
            self._make_data(target_rho),
            anderson_depth=4,
            **settings,
        )

        self.assertTrue(accelerated["converged"])
        self.assertGreater(accelerated["anderson_steps"], 0)
        self.assertLess(
            accelerated["n_evaluations"], baseline["n_evaluations"]
        )
        self.assertTrue(
            torch.allclose(
                accelerated["rho"],
                target_rho,
                atol=2.0e-6,
                rtol=0.0,
            )
        )

    def test_adaptive_continuation_accepts_direct_ideal_gas_probe(self):
        data = self._make_data(include_mu=False)
        data["V_ext"].zero_()

        result = GridSolver(self._make_model()).solve(
            data,
            particle_numbers=[2.0],
            method="euler",
            adaptive_continuation=True,
            continuation_probe_iterations=5,
            continuation_steps=5,
            tolerance_residual=1.0e-8,
            tolerance_rms_residual=1.0e-8,
        )

        self.assertTrue(result["converged"])
        self.assertTrue(result["adaptive_continuation_probe"])
        self.assertFalse(result["adaptive_continuation_fallback"])
        self.assertEqual(result["n_evaluations"], 1)
        self.assertTrue(
            torch.allclose(
                result["rho"],
                torch.full((4, 1), 0.5, dtype=torch.float64),
            )
        )

    def test_fixed_particle_number_density_cap_is_enforced(self):
        data = self._make_data(include_mu=False)
        data["V_ext"] = torch.tensor(
            [[-10.0], [0.0], [0.0], [0.0]],
            dtype=torch.float64,
        )

        result = GridSolver(self._make_model()).solve(
            data,
            particle_numbers=[2.0],
            maximum_density=0.6,
            continuation_steps=0,
            max_iter=500,
            tolerance_residual=1.0e-7,
        )

        self.assertLessEqual(result["rho"].max().item(), 0.6 + 1.0e-12)
        self.assertAlmostEqual(result["rho"].sum().item(), 2.0, places=12)
        self.assertAlmostEqual(result["rho"][0, 0].item(), 0.6, places=10)
        self.assertTrue(result["converged"])

    def test_infeasible_fixed_particle_number_density_cap_is_rejected(self):
        data = self._make_data(include_mu=False)

        with self.assertRaisesRegex(ValueError, "infeasible"):
            GridSolver(self._make_model()).solve(
                data,
                particle_numbers=[2.0],
                maximum_density=0.4,
            )

    def test_solver_method_is_validated(self):
        with self.assertRaisesRegex(ValueError, "method"):
            GridSolver(self._make_model()).solve(
                self._make_data(),
                method="unknown",
            )

    def test_zero_density_is_included_in_default_physical_residual(self):
        rho = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
        residual, _, max_residual, _ = _euler_residual(
            rho=rho,
            c1=torch.zeros_like(rho),
            V_ext=torch.zeros_like(rho),
            beta=torch.tensor(1.0, dtype=rho.dtype),
            thermal_wavelength=torch.tensor([1.0], dtype=rho.dtype),
            mu=None,
            density_threshold=0.0,
        )

        self.assertTrue(torch.isfinite(residual).all())
        self.assertGreater(max_residual, 100.0)

    def test_density_threshold_excludes_unresolved_voxels_from_diagnostics(
        self,
    ):
        rho = torch.tensor([[1.0], [1.0e-8]], dtype=torch.float64)
        _, _, max_residual, rms_residual = _euler_residual(
            rho=rho,
            c1=torch.zeros_like(rho),
            V_ext=torch.zeros_like(rho),
            beta=torch.tensor(1.0, dtype=rho.dtype),
            thermal_wavelength=torch.tensor([1.0], dtype=rho.dtype),
            mu=None,
            density_threshold=1.0e-3,
        )

        self.assertLess(max_residual, 1.0e-5)
        self.assertLess(rms_residual, 1.0e-5)

    def test_maximum_and_rms_residual_tolerances_are_both_required(self):
        self.assertTrue(_residuals_converged(0.02, 0.005, 0.03, 0.01))
        self.assertFalse(_residuals_converged(0.04, 0.005, 0.03, 0.01))
        self.assertFalse(_residuals_converged(0.02, 0.02, 0.03, 0.01))


if __name__ == "__main__":
    unittest.main()
