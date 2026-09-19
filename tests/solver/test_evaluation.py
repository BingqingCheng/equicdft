"""Common solver energy/derivative contract, independent of update algorithm."""

import unittest
from unittest.mock import patch

import torch

from equicdft import FixedDipoleIdeal, GridSolver
from tests.ideal_helpers import IdealOnly
from tests.model_fixtures import example_model, grid_data


class TestSolverEvaluation(unittest.TestCase):
    def data(self, polarized=False, batch=False):
        rho = torch.tensor([[.2, .3], [.4, .1], [.1, .2], [.3, .4]], dtype=torch.float64)
        data = dict(rho=rho, V_ext=rho * 2 - .4,
                    beta=rho.new_tensor(1.3), grid_spacing=rho.new_full((3,), .7))
        if polarized:
            data.update(dipole_density=rho[..., None] * rho.new_tensor([.04, -.03, .02]),
                        E_ext=rho[..., None] * rho.new_tensor([.2, -.5, .7]))
        if batch:
            for key in ("rho", "V_ext", "beta"):
                data[key] = torch.stack((data[key], 1.2 * data[key]))
        return data

    def solver(self, polarized=False):
        return GridSolver(IdealOnly(FixedDipoleIdeal([.7, 1.2]) if polarized else None))

    def assert_canonical_residual_names(self, result):
        """One name per residual: the Euler-Lagrange spellings are retired."""

        for name in ("density_residual", "max_density_residual", "rms_density_residual"):
            self.assertIn(name, result)
        for retired in ("euler_lagrange_residual", "max_euler_lagrange_residual",
                        "rms_euler_lagrange_residual",
                        "polarization_euler_lagrange_residual"):
            self.assertNotIn(retired, result)

    def test_scalar_energies_and_canonical_residuals_single_and_batch(self):
        for batch in (False, True):
            with self.subTest(batch=batch):
                data = self.data(batch=batch)
                result = self.solver().evaluate(data)
                rho, beta, volume = data["rho"], data["beta"], data["grid_spacing"].prod()
                external = volume * (rho * beta[..., None, None] * data["V_ext"]).sum((-2, -1))
                torch.testing.assert_close(result["beta_external"], external)
                torch.testing.assert_close(result["beta_external"], result["beta_V_ext"], rtol=0, atol=0)
                torch.testing.assert_close(result["beta_A"], result["beta_F"] + external)
                torch.testing.assert_close(result["particle_numbers"], volume * rho.sum(-2))
                torch.testing.assert_close((rho * result["density_residual"]).sum(-2),
                                           torch.zeros_like(rho.sum(-2)), atol=1e-15, rtol=0)
                self.assert_canonical_residual_names(result)

    def test_prescribed_mu_batch(self):
        data = self.data(batch=True)
        data["mu"] = data["rho"].new_tensor([[.3, -.1], [.5, .2]])
        result = self.solver().evaluate(data)
        expected = result["beta_mu_local"] - (data["beta"][:, None] * data["mu"])[:, None, :]
        torch.testing.assert_close(result["density_residual"], expected)
        torch.testing.assert_close(result["beta_Omega"], result["beta_A"] - result["beta_mu_N"])
        self.assert_canonical_residual_names(result)

    def test_polarization_coupling_and_derivatives(self):
        data = self.data(polarized=True)
        result = self.solver(True).evaluate(data)
        volume, beta = data["grid_spacing"].prod(), data["beta"]
        external = beta * volume * ((data["rho"] * data["V_ext"]).sum()
                                   - (data["dipole_density"] * data["E_ext"]).sum())
        torch.testing.assert_close(result["beta_external"], external, rtol=0, atol=0)
        torch.testing.assert_close(result["beta_external"], result["beta_V_ext"] + result["beta_P_ext"])
        torch.testing.assert_close(result["polarization_residual"],
                                   result["polarization_derivative"] - beta * data["E_ext"])
        torch.testing.assert_close(result["maximum_residual"],
                                   torch.maximum(result["max_density_residual"], result["max_polarization_residual"]))
        self.assert_canonical_residual_names(result)

    def test_zero_polarization_matches_scalar_ideal(self):
        data = self.data(polarized=True)
        data["dipole_density"].zero_()
        data["E_ext"].zero_()
        scalar, coupled = self.solver().evaluate(data), self.solver(True).evaluate(data)
        for key in ("beta_F", "beta_A", "beta_mu_local", "density_residual",
                    "max_density_residual", "rms_density_residual", "particle_numbers"):
            torch.testing.assert_close(scalar[key], coupled[key], atol=1e-15, rtol=0)

    def test_single_model_call_and_explicit_derivative_requests(self):
        for polarized in (False, True):
            solver, data = self.solver(polarized), self.data(polarized)
            for derivatives in (False, True):
                with self.subTest(polarized=polarized, derivatives=derivatives):
                    with patch.object(solver.model, "forward", wraps=solver.model.forward) as forward:
                        result = solver.evaluate(data, compute_derivatives=derivatives)
                    forward.assert_called_once()
                    self.assertEqual(forward.call_args.kwargs, dict(
                        compute_c1=derivatives, compute_c2=False,
                        compute_polarization_derivative=polarized and derivatives,
                    ))
                    self.assertEqual("c1" in result, derivatives)
                    self.assertEqual("density_residual" in result, derivatives)
                    self.assertEqual("polarization_residual" in result, polarized and derivatives)
                    self.assertNotIn("c2", result)
                    self.assertFalse(data["rho"].requires_grad)

    def test_energy_only_overrides_model_derivative_defaults(self):
        model = example_model(compute_c1=True, compute_c2=True)
        solver = GridSolver(model)
        for derivatives in (False, True):
            result = solver.evaluate(grid_data(shape=(3, 3, 3)), compute_derivatives=derivatives)
            self.assertEqual("c1" in result, derivatives)
            self.assertNotIn("c2", result)
        self.assertTrue(model.compute_c1)
        self.assertTrue(model.compute_c2)

    def test_exclusions_are_omitted_from_both_residuals(self):
        for polarized in (False, True):
            data = self.data(polarized)
            data["excluded_mask"] = torch.tensor([True, False, False, False])
            data["rho"][0] = 0
            if polarized:
                data["dipole_density"][0] = 0
            result = self.solver(polarized).evaluate(data)
            self.assertTrue((result["density_residual"][0] == 0).all())
            torch.testing.assert_close(result["max_density_residual"], result["density_residual"][1:].abs().max())
            if polarized:
                self.assertTrue((result["polarization_residual"][0] == 0).all())

    def test_solve_aliases_survive_bounds_and_fixed_fields(self):
        for polarized in (False, True):
            for method in ("euler", "minimize"):
                fixed_fields = (None, "rho", "dipole_density") if polarized and method == "minimize" else (None,)
                for fixed in fixed_fields:
                    with self.subTest(polarized=polarized, method=method, fixed=fixed):
                        data = self.data(polarized)
                        numbers = (data["rho"].sum(0) * data["grid_spacing"].prod()).tolist()
                        options = dict(particle_numbers=numbers, maximum_density=.4, max_iter=2)
                        if polarized:
                            options.update(maximum_polarization_fraction=.1, fixed_field=fixed)
                            if fixed is not None:
                                # The constrained initialization must have the prescribed N.
                                rho = data["rho"].clone()
                                options.update(initial_rho=rho, initial_polarization=torch.zeros_like(data["dipole_density"]))
                        result = self.solver(polarized).solve(data, method=method, **options)
                        self.assert_canonical_residual_names(result)
                        self.assertIn("beta_A", result)
