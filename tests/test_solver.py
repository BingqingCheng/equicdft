import unittest

import torch
from torch import nn

from equicdft import GridCACEModel, GridData, GridSolver, LocalReadout
from equicdft._solver_numerics import (
    _anderson_log_density_candidate as numerical_anderson_candidate,
    _project_density_constraints,
)
from equicdft.solver import (
    _anderson_log_density_candidate as solver_anderson_candidate,
    _euler_residual,
    _residuals_converged,
)


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


class _DensityReadout(LocalReadout):
    def forward(self, local_features):
        return local_features[..., :1]


class _IdealGasModel(nn.Module):
    """Minimal zero-excess model for testing solver initialization."""

    def __init__(self):
        super().__init__()
        self.register_buffer("zero", torch.tensor(0.0))

    def forward(self, data, compute_c1=True):
        rho = data["rho"]
        outputs = {"beta_F_exc": self.zero.to(rho) * torch.sum(rho)}
        if compute_c1:
            outputs["c1"] = torch.zeros_like(rho)
        return outputs


class TestGridSolver(unittest.TestCase):
    def assert_objective_history_nonincreasing(self, history):
        """Check that every accepted minimization step lowers the objective."""

        for previous, current in zip(history, history[1:]):
            self.assertLessEqual(current, previous + 1.0e-12)

    def _make_model(self):
        # beta_F_exc = sum_g rho_g^2 and c1 = -2 rho for unit voxels.
        return GridCACEModel(
            a_features=_DensityFeatures(),
            b_features=_IdentityModule(),
            readout=[_DensityReadout()],
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

    def test_constraint_projection_is_shared_by_solver_updates(self):
        rho = torch.tensor(
            [[4.0, 1.0], [9.0, 9.0], [1.0, 4.0]],
            dtype=torch.float64,
        )
        accessible = torch.tensor([True, False, True])
        particle_numbers = torch.tensor([1.0, 2.0], dtype=torch.float64)
        maximum_density = torch.tensor([1.5, 3.0], dtype=torch.float64)

        canonical = _project_density_constraints(
            rho,
            particle_numbers,
            torch.tensor(0.5, dtype=torch.float64),
            maximum_density,
            accessible,
        )
        grand_canonical = _project_density_constraints(
            rho,
            None,
            torch.tensor(0.5, dtype=torch.float64),
            maximum_density,
            accessible,
        )

        self.assertTrue(torch.equal(canonical[1], torch.zeros(2)))
        self.assertTrue(torch.equal(grand_canonical[1], torch.zeros(2)))
        self.assertTrue(
            torch.allclose(0.5 * canonical.sum(dim=0), particle_numbers)
        )
        self.assertTrue(torch.all(canonical <= maximum_density[None, :]))
        self.assertTrue(
            torch.equal(
                grand_canonical,
                torch.tensor(
                    [[1.5, 1.0], [0.0, 0.0], [1.0, 3.0]],
                    dtype=torch.float64,
                ),
            )
        )

    def test_solver_preserves_anderson_private_import(self):
        self.assertIs(solver_anderson_candidate, numerical_anderson_candidate)

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
            method="minimize",
            max_iter=200,
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
        self.assertLessEqual(result["line_search_failures"], 1)
        self.assert_objective_history_nonincreasing(
            result["objective_history"]
        )

    def test_fixed_particle_number_solve_enforces_constraint(self):
        data = self._make_data(include_mu=False)
        data["V_ext"].zero_()

        result = GridSolver(self._make_model()).solve(
            data,
            particle_numbers=[2.0],
            method="minimize",
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
        self.assert_objective_history_nonincreasing(
            result["objective_history"]
        )

    def test_hard_wall_fixed_n_is_enforced_by_both_solvers(self):
        expected = torch.tensor(
            [[0.0], [0.5], [0.5], [0.0]],
            dtype=torch.float64,
        )
        for method in ("euler", "minimize"):
            with self.subTest(method=method):
                data = self._make_data(include_mu=False)
                data["V_ext"].zero_()
                data["excluded_mask"] = torch.tensor(
                    [True, False, False, True]
                )
                result = GridSolver(self._make_model()).solve(
                    data,
                    initial_rho=torch.ones_like(data["V_ext"]),
                    particle_numbers=[1.0],
                    method=method,
                    max_iter=200,
                    tolerance_residual=1.0e-8,
                )

                self.assertTrue(
                    torch.equal(
                        result["excluded_mask"],
                        data["excluded_mask"],
                    )
                )
                self.assertTrue(torch.allclose(result["rho"], expected))
                self.assertAlmostEqual(result["rho"].sum().item(), 1.0)
                self.assertTrue(
                    torch.all(
                        result["euler_lagrange_residual"][
                            data["excluded_mask"]
                        ]
                        == 0.0
                    )
                )
                self.assertTrue(result["converged"])

    def test_hard_wall_grand_canonical_is_enforced_by_both_solvers(self):
        expected = torch.tensor(
            [[0.0], [1.0], [1.0], [0.0]],
            dtype=torch.float64,
        )
        for method in ("euler", "minimize"):
            with self.subTest(method=method):
                data = self._make_data()
                data["V_ext"].zero_()
                data["excluded_mask"] = torch.tensor(
                    [True, False, False, True]
                )
                result = GridSolver(_IdealGasModel()).solve(
                    data,
                    method=method,
                    max_iter=20,
                    tolerance_residual=1.0e-10,
                )

                self.assertTrue(torch.allclose(result["rho"], expected))
                self.assertTrue(result["converged"])

    def test_evaluate_rejects_density_inside_hard_wall(self):
        data = self._make_data()
        data["excluded_mask"] = torch.tensor(
            [True, False, False, False]
        )
        data["rho"] = torch.ones_like(data["V_ext"])

        with self.assertRaisesRegex(ValueError, "excluded"):
            GridSolver(self._make_model()).evaluate(data)

    def test_beta_multiplier_initialization_is_exact_for_fixed_n_ideal_gas(
        self,
    ):
        data = self._make_data(include_mu=False)
        data["V_ext"] = torch.tensor(
            [[-1.0], [0.0], [0.5], [1.0]],
            dtype=torch.float64,
        )
        expected = 2.0 * torch.softmax(-data["V_ext"], dim=0)

        ideal = GridSolver(_IdealGasModel()).solve(
            data,
            particle_numbers=[2.0],
            beta_multiplier=1.0,
            max_iter=1,
            tolerance_residual=1.0e-10,
        )
        uniform = GridSolver(_IdealGasModel()).solve(
            data,
            particle_numbers=[2.0],
            max_iter=1,
            tolerance_residual=1.0e-10,
            adaptive_mixing=False,
        )

        self.assertEqual(ideal["solver_beta_multiplier"], 1.0)
        self.assertEqual(ideal["n_iter"], 0)
        self.assertTrue(torch.allclose(ideal["rho"], expected))
        self.assertEqual(uniform["solver_beta_multiplier"], 0.0)
        self.assertEqual(uniform["n_iter"], 1)
        self.assertFalse(torch.allclose(uniform["rho"], expected))

        tempered_expected = 2.0 * torch.softmax(
            -0.5 * data["V_ext"], dim=0
        )
        tempered = GridSolver(_IdealGasModel()).solve(
            data,
            particle_numbers=[2.0],
            beta_multiplier=0.5,
            max_iter=1,
            tolerance_residual=10.0,
        )
        self.assertEqual(tempered["solver_beta_multiplier"], 0.5)
        self.assertTrue(torch.allclose(tempered["rho"], tempered_expected))

    def test_solver_beta_multiplier_is_validated(self):
        for invalid_value in (-1.0, float("inf"), float("nan")):
            with self.subTest(invalid_value=invalid_value):
                with self.assertRaisesRegex(ValueError, "beta_multiplier"):
                    GridSolver(self._make_model()).solve(
                        self._make_data(),
                        beta_multiplier=invalid_value,
                    )

    def test_explicit_density_overrides_beta_profile(self):
        initial_rho = torch.full((4, 1), 0.5, dtype=torch.float64)
        result = GridSolver(self._make_model()).solve(
            self._make_data(include_mu=False),
            initial_rho=initial_rho,
            particle_numbers=[2.0],
            beta_multiplier=1.0,
            max_iter=1,
            tolerance_residual=10.0,
        )
        self.assertTrue(torch.allclose(result["rho"], initial_rho))

    def test_known_mu_beta_multiplier_only_scales_external_field(self):
        data = self._make_data()
        data["mu"] = torch.tensor([0.7], dtype=torch.float64)
        expected = torch.exp(
            data["beta"]
            * (data["mu"][None, :] - 0.5 * data["V_ext"])
        )
        result = GridSolver(_IdealGasModel()).solve(
            data,
            beta_multiplier=0.5,
            max_iter=1,
            tolerance_residual=100.0,
        )
        self.assertTrue(torch.allclose(result["rho"], expected))

        uniform = GridSolver(_IdealGasModel()).solve(
            data,
            beta_multiplier=0.0,
            max_iter=1,
            tolerance_residual=100.0,
        )
        old_uniform = torch.exp(
            data["beta"] * data["mu"]
        )[None, :].expand_as(data["V_ext"])
        self.assertTrue(torch.allclose(uniform["rho"], old_uniform))

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
            method="minimize",
            max_iter=200,
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
        self.assertLess(
            result["objective_history"][-1],
            result["objective_history"][0],
        )
        self.assert_objective_history_nonincreasing(
            result["objective_history"]
        )

    def test_euler_is_the_default_method(self):
        target_rho = torch.tensor(
            [[0.2], [0.3], [0.4], [0.5]],
            dtype=torch.float64,
        )

        result = GridSolver(self._make_model()).solve(
            self._make_data(target_rho),
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

    def test_anderson_accelerates_euler_and_preserves_fixed_n(self):
        target_rho = torch.tensor(
            [[0.2], [0.3], [0.4], [0.5]],
            dtype=torch.float64,
        )
        data = self._make_data(target_rho, include_mu=False)
        particle_numbers = [target_rho.sum().item()]
        options = {
            "particle_numbers": particle_numbers,
            "method": "euler",
            "max_iter": 200,
            "mixing": 0.05,
            "minimum_mixing": 0.01,
            "maximum_mixing": 0.4,
            "mixing_growth": 1.2,
            "tolerance_residual": 1.0e-8,
            "tolerance_rms_residual": 1.0e-8,
            "tolerance_change": 1.0e-12,
        }

        scalar = GridSolver(self._make_model()).solve(data, **options)
        accelerated = GridSolver(self._make_model()).solve(
            data,
            anderson=True,
            **options,
        )

        self.assertTrue(scalar["converged"])
        self.assertTrue(accelerated["converged"])
        self.assertLess(accelerated["n_iter"], scalar["n_iter"])
        self.assertLess(
            accelerated["n_evaluations"],
            scalar["n_evaluations"],
        )
        self.assertTrue(torch.all(accelerated["rho"] > 0.0))
        self.assertAlmostEqual(
            accelerated["rho"].sum().item(),
            particle_numbers[0],
            places=12,
        )
        self.assertTrue(accelerated["solver_anderson"])
        self.assertGreater(accelerated["anderson_attempts"], 0)
        self.assertEqual(
            accelerated["anderson_attempts"],
            accelerated["anderson_accepted"]
            + accelerated["anderson_rejected"],
        )
        self.assertEqual(
            accelerated["anderson_resets"],
            accelerated["anderson_rejected"],
        )

    def test_anderson_preserves_excluded_mask_and_density_cap(self):
        data = self._make_data(include_mu=False)
        data["V_ext"] = torch.tensor(
            [[-3.0], [-0.5], [0.5], [3.0]],
            dtype=torch.float64,
        )
        data["excluded_mask"] = torch.tensor(
            [True, False, False, False]
        )

        result = GridSolver(self._make_model()).solve(
            data,
            initial_rho=torch.tensor(
                [[0.0], [0.2], [0.5], [0.3]],
                dtype=torch.float64,
            ),
            particle_numbers=[1.0],
            method="euler",
            anderson=True,
            maximum_density=0.6,
            max_iter=200,
            tolerance_residual=1.0e-8,
            tolerance_rms_residual=1.0e-8,
            tolerance_change=1.0e-12,
        )

        self.assertEqual(result["rho"][0, 0].item(), 0.0)
        self.assertLessEqual(result["rho"].max().item(), 0.6 + 1.0e-12)
        self.assertAlmostEqual(result["rho"].sum().item(), 1.0, places=12)
        self.assertTrue(torch.all(result["rho"][1:] > 0.0))

    def test_anderson_options_are_validated_when_enabled(self):
        invalid_options = (
            {"anderson_history": 1},
            {"anderson_regularization": -1.0},
            {"anderson_damping": 0.0},
            {"anderson_damping": 1.1},
        )
        for options in invalid_options:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    GridSolver(self._make_model()).solve(
                        self._make_data(),
                        anderson=True,
                        **options,
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
            method="minimize",
            maximum_density=0.6,
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

    def test_density_cap_feasibility_uses_accessible_volume(self):
        data = self._make_data(include_mu=False)
        data["excluded_mask"] = torch.tensor(
            [True, False, False, True]
        )

        with self.assertRaisesRegex(ValueError, "infeasible"):
            GridSolver(self._make_model()).solve(
                data,
                particle_numbers=[1.0],
                maximum_density=0.4,
            )

    def test_solver_method_is_validated(self):
        with self.assertRaisesRegex(ValueError, "method"):
            GridSolver(self._make_model()).solve(
                self._make_data(),
                method="unknown",
            )

    def test_only_selected_method_options_are_validated(self):
        GridSolver(self._make_model()).solve(
            self._make_data(),
            method="euler",
            step_size="unused",
            max_iter=1,
        )
        GridSolver(self._make_model()).solve(
            self._make_data(),
            method="minimize",
            mixing="unused",
            max_iter=1,
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

    def test_hard_wall_is_excluded_from_residual_diagnostics(self):
        rho = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
        residual, chemical_potential, max_residual, rms_residual = (
            _euler_residual(
                rho=rho,
                c1=torch.zeros_like(rho),
                V_ext=torch.zeros_like(rho),
                beta=torch.tensor(1.0, dtype=rho.dtype),
                thermal_wavelength=torch.tensor([1.0], dtype=rho.dtype),
                mu=None,
                density_threshold=0.0,
                accessible_mask=torch.tensor([True, False]),
            )
        )

        self.assertTrue(torch.equal(residual, torch.zeros_like(rho)))
        self.assertTrue(torch.equal(chemical_potential, torch.zeros(1)))
        self.assertEqual(max_residual, 0.0)
        self.assertEqual(rms_residual, 0.0)

    def test_maximum_and_rms_residual_tolerances_are_both_required(self):
        self.assertTrue(_residuals_converged(0.02, 0.005, 0.03, 0.01))
        self.assertFalse(_residuals_converged(0.04, 0.005, 0.03, 0.01))
        self.assertFalse(_residuals_converged(0.02, 0.02, 0.03, 0.01))


if __name__ == "__main__":
    unittest.main()
