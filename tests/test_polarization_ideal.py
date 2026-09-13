"""Independent thermodynamic and canonical equilibrium tests for fixed dipoles."""

import unittest

import numpy as np
import torch

from equicdft import FixedDipoleIdeal, PolarizationSolver
from equicdft.polarization_ideal import inverse_langevin, langevin, log_sinhc


def exact_solution(data, numbers, moments):
    # Independent NumPy closed form, deliberately not calling library helpers.
    beta = float(data["beta"])
    moment = np.asarray(moments)
    h_vector = beta*data["E_ext"].numpy()*moment[..., None]
    h = np.linalg.norm(h_vector, axis=-1)
    safe = np.where(h == 0, 1, h)
    log_z = np.where(h == 0, 0, safe + np.log(-np.expm1(-2*safe)) - np.log(2*safe))
    mean = np.where(h == 0, 0, 1/np.tanh(safe) - 1/safe)
    weak = (h > 0) & (h < 1e-3)
    nodes, angular_weights = np.polynomial.legendre.leggauss(64)
    boltzmann = np.exp(h[weak, None]*nodes)
    mean[weak] = (boltzmann @ (angular_weights*nodes))/(boltzmann @ angular_weights)
    log_weights = -beta*data["V_ext"].numpy() + log_z
    weights = np.exp(log_weights - log_weights.max(axis=0))
    volume = float(data["grid_spacing"].prod())
    rho = np.asarray(numbers)*weights/(volume*weights.sum(axis=0))
    polarization = (rho*moment*mean/safe)[..., None]*h_vector
    return torch.from_numpy(rho), torch.from_numpy(polarization)


class TestFixedDipoleIdeal(unittest.TestCase):
    def setUp(self):
        self.old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(19)

    def tearDown(self):
        torch.set_default_dtype(self.old_dtype)

    def test_angular_quadrature(self):
        # Integrate the microscopic Boltzmann weight over cos(theta), rather
        # than verifying one implementation against the same special function.
        nodes, weights = np.polynomial.legendre.leggauss(160)
        h = torch.tensor([0., 1e-7, 0.05, 0.1, 0.8, 3., 20., 100.])
        boltzmann = np.exp(h.numpy()[:, None]*nodes)
        z = boltzmann @ (weights/2)
        mean = (boltzmann @ (weights*nodes/2))/z
        torch.testing.assert_close(log_sinhc(h), torch.from_numpy(np.log(z)), rtol=2e-12, atol=5e-13)
        torch.testing.assert_close(langevin(h), torch.from_numpy(mean), rtol=2e-12, atol=5e-13)
        torch.testing.assert_close(inverse_langevin(torch.from_numpy(mean)), h, rtol=1e-10, atol=2e-12)

    def test_independent_derivatives_and_mixed_hessian(self):
        ideal = FixedDipoleIdeal([0.7, 1.9], [1.2, 0.8])
        for volume in (0.125, 8.0):
            rho = (0.5 + torch.rand(3, 2)).requires_grad_()
            polar = (0.03*torch.randn(3, 2, 3)).requires_grad_()
            result = ideal(rho, polar, volume)
            dr, dp = torch.autograd.grad(result["beta_F_id"], (rho, polar), create_graph=True)
            torch.testing.assert_close(dr/volume, result["density_derivative"], atol=2e-12, rtol=2e-11)
            torch.testing.assert_close(dp/volume, result["polarization_derivative"], atol=2e-12, rtol=2e-11)
            self.assertTrue(torch.autograd.gradcheck(lambda r, p: ideal(r, p, volume)["beta_F_id"], (rho, polar), eps=1e-6, atol=2e-8, rtol=2e-6))
            mixed_r = torch.autograd.grad(dr[0, 0], polar, retain_graph=True)[0][0, 0, 1]
            mixed_p = torch.autograd.grad(dp[0, 0, 1], rho)[0][0, 0]
            torch.testing.assert_close(mixed_r, mixed_p, atol=1e-12, rtol=1e-10)

    def test_zero_polarization_hessian_and_scalar_limit(self):
        rho = torch.tensor([[0.4]], requires_grad=True)
        polar = torch.zeros(1, 1, 3, requires_grad=True)
        ideal = FixedDipoleIdeal(1.7, 1.3)
        result = ideal(rho, polar, 8.)
        expected = 8*rho.sum()*(torch.log(rho[0, 0]*1.3**3) - 1)
        torch.testing.assert_close(result["beta_F_id"], expected)
        hessian = torch.autograd.functional.hessian(lambda p: ideal(rho, p, 8.)["beta_F_id"], polar).reshape(3, 3)
        torch.testing.assert_close(hessian, 8*3/(1.7**2*0.4)*torch.eye(3), atol=1e-12, rtol=1e-12)
        self.assertTrue(torch.autograd.gradgradcheck(lambda r, p: ideal(r, p, 8.)["beta_F_id"], (rho, polar), atol=2e-7))

    def test_near_saturation_and_inverse_derivative(self):
        p = torch.tensor([0., 1e-8, 1e-3, 0.2, 0.8, 0.99, 0.9999], requires_grad=True)
        xi = inverse_langevin(p)
        torch.testing.assert_close(langevin(xi), p, rtol=2e-12, atol=2e-14)
        self.assertTrue(torch.autograd.gradcheck(inverse_langevin, (p[:-1],), eps=1e-7, atol=1e-5, rtol=1e-5))
        polar = torch.zeros(1, 1, 3)
        polar[..., 0] = 0.9999
        self.assertTrue(torch.isfinite(FixedDipoleIdeal(1.)(torch.ones(1, 1), polar, 1.)["beta_F_id"]))

    def test_batch_and_rotation(self):
        ideal = FixedDipoleIdeal([1., 2.])
        rho = 0.5 + torch.rand(2, 5, 2)
        polar = 0.05*torch.randn(2, 5, 2, 3)
        rotation, _ = torch.linalg.qr(torch.randn(3, 3))
        original = ideal(rho, polar, torch.tensor([0.125, 8.]))
        rotated = ideal(rho, polar @ rotation.T, torch.tensor([0.125, 8.]))
        torch.testing.assert_close(original["beta_F_id"], rotated["beta_F_id"], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(original["polarization_derivative"] @ rotation.T, rotated["polarization_derivative"], rtol=1e-11, atol=1e-12)
        for batch, volume in enumerate((0.125, 8.)):
            torch.testing.assert_close(original["beta_F_id"][batch], ideal(rho[batch], polar[batch], volume)["beta_F_id"])

    def test_invalid_states(self):
        torch.set_default_dtype(torch.float32)
        precise = FixedDipoleIdeal(1.7)
        self.assertEqual(float(precise.dipole_magnitude), 1.7)
        torch.set_default_dtype(torch.float64)
        ideal = FixedDipoleIdeal(1.)
        rho = torch.ones(2, 1)
        for value in (1., 1.1, float("nan")):
            polar = torch.zeros(2, 1, 3)
            polar[0, 0, 0] = value
            with self.assertRaises(ValueError):
                ideal(rho, polar, 1.)
        with self.assertRaises(ValueError):
            ideal(torch.zeros_like(rho), torch.zeros(2, 1, 3), 1.)
        with self.assertRaises(ValueError):
            FixedDipoleIdeal(-1.)
        with self.assertRaisesRegex(ValueError, "must be real"):
            FixedDipoleIdeal(torch.tensor(1.7 + 0.2j))
        with self.assertRaisesRegex(ValueError, "must be real"):
            FixedDipoleIdeal(1.7, thermal_wavelength=torch.tensor(1.0 + 0.2j))


class TestPolarizationSolver(unittest.TestCase):
    def setUp(self):
        self.old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(29)

    def tearDown(self):
        torch.set_default_dtype(self.old_dtype)

    @staticmethod
    def case(amplitude=1., spacing=0.5):
        xyz = torch.cartesian_prod(torch.arange(4), torch.arange(3), torch.arange(2))
        phase = 2*torch.pi*(xyz + 0.5)/torch.tensor([4, 3, 2])
        potential = torch.stack((0.7*phase[:, 0].cos(), -0.4*phase[:, 1].sin()), dim=-1)
        # Gradient of a sum of periodic sine modes, with an oblique constant
        # offset, so all Cartesian components enter and species may differ.
        field = amplitude*(phase.cos() + torch.tensor([0.1, 0.3, -0.2]))
        return {"V_ext": potential, "E_ext": torch.stack((field, -0.6*field), dim=1),
                "beta": torch.tensor(1.4), "grid_spacing": torch.full((3,), spacing)}

    def check_solution(self, data, numbers, moments, perturbed=False):
        solver = PolarizationSolver(moments, thermal_wavelength=[0.8, 1.3])
        kwargs = {}
        if perturbed:
            rho = torch.exp(0.4*torch.randn_like(data["V_ext"]))
            rho *= torch.as_tensor(numbers)/(data["grid_spacing"].prod()*rho.sum(0))
            polar = 0.1*torch.randn(*rho.shape, 3)
            polar /= 1 + polar.norm(dim=-1, keepdim=True)
            kwargs = {"initial_rho": rho, "initial_polarization": polar*(rho*torch.as_tensor(moments))[..., None]}
        result = solver.solve(data, numbers, max_iter=350, tolerance_residual=1e-7, **kwargs)
        self.assertTrue(result["converged"], str(result["maximum_residual"]))
        expected_rho, expected_p = exact_solution(data, numbers, moments)
        scale = expected_rho.mean(0)
        torch.testing.assert_close(result["rho"]/scale, expected_rho/scale, rtol=0, atol=3e-7)
        torch.testing.assert_close(result["dipole_density"]/(scale*torch.as_tensor(moments))[..., None], expected_p/(scale*torch.as_tensor(moments))[..., None], rtol=0, atol=3e-7)
        torch.testing.assert_close(result["particle_numbers"], torch.as_tensor(numbers), rtol=2e-14, atol=0)
        self.assertLess(float(result["maximum_alignment"]), 1.)
        energies = np.array([item["beta_A"] for item in result["history"]])
        self.assertLessEqual(np.diff(energies).max(initial=0), 1e-11*max(1., abs(energies[0])))
        return result

    def test_zero_scalar_uniform_and_coupled_fields_both_starts(self):
        for kind in ("zero", "scalar", "uniform", "electric", "coupled"):
            data = self.case()
            if kind in ("zero", "uniform", "electric"):
                data["V_ext"].zero_()
            if kind in ("zero", "scalar"):
                data["E_ext"].zero_()
            if kind == "uniform":
                data["E_ext"][:] = torch.tensor([0.5, -1., 0.7])
            for perturbed in (False, True):
                with self.subTest(kind=kind, perturbed=perturbed):
                    self.check_solution(data, [3., 4.], [0.7, 1.9], perturbed)

    def test_weak_and_strong_fields(self):
        for amplitude in (1e-5, 12.):
            self.check_solution(self.case(amplitude), [3., 4.], [0.7, 1.9], True)

    def test_units_and_voxel_volume(self):
        original = self.case()
        first = self.check_solution(original, [3., 4.], [0.7, 1.9])
        # Change length units by 4, energy by 2.3, and dipole by 5.
        changed = dict(original, grid_spacing=original["grid_spacing"]*4,
                       beta=original["beta"]/2.3, V_ext=original["V_ext"]*2.3,
                       E_ext=original["E_ext"]*2.3/5)
        second = self.check_solution(changed, [3., 4.], [3.5, 9.5])
        torch.testing.assert_close(first["rho"], second["rho"]*64, atol=2e-7, rtol=0)
        torch.testing.assert_close(first["dipole_density"], second["dipole_density"]*64/5, atol=2e-7, rtol=0)

    def test_canonical_gauge_and_unconverged_status(self):
        data = self.case()
        rho, polar = exact_solution(data, [3., 4.], [0.7, 1.9])
        solver = PolarizationSolver([0.7, 1.9])
        first = solver.evaluate(dict(data, rho=rho, dipole_density=polar))
        shift = torch.tensor([2., -3.])
        second = solver.evaluate(dict(data, V_ext=data["V_ext"]+shift, rho=rho, dipole_density=polar))
        torch.testing.assert_close(first["density_residual"], second["density_residual"], atol=1e-12, rtol=0)
        torch.testing.assert_close(second["beta_mu"]-first["beta_mu"], data["beta"]*shift)
        result = solver.solve(data, [3., 4.], max_iter=1, tolerance_residual=1e-12, step_size=0.1)
        self.assertFalse(result["converged"])
        self.assertEqual(result["status"], "max_iter")

    def test_validation(self):
        solver = PolarizationSolver([0.7, 1.9])
        data = self.case()
        for changes in ({"beta": torch.tensor(-1.)}, {"E_ext": torch.zeros(24, 3)},
                        {"excluded_mask": torch.ones(24, dtype=torch.bool)},
                        {"grid_spacing": torch.tensor([1., 2., 1.])}):
            with self.assertRaises(ValueError):
                solver.solve(dict(data, **changes), [3., 4.])
        with self.assertRaises(ValueError):
            solver.solve(data, [3., 4.], initial_rho=torch.ones(24, 2))

    def test_excess_energy_and_line_search(self):
        class Quadratic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.strength = torch.nn.Parameter(torch.tensor(12.))

            def forward(self, data, **kwargs):
                energy = 0.5*self.strength*data["grid_spacing"].prod()*(
                    data["rho"].square().sum() + data["dipole_density"].square().sum())
                return {"beta_F_exc": energy}

        data = self.case()
        model = Quadratic()
        model.strength.grad = torch.tensor(7.)
        target_rho = 0.4 + 0.1*torch.rand_like(data["V_ext"])
        target_p = 0.04*torch.randn(*target_rho.shape, 3)
        moments = [0.7, 1.9]
        ideal = FixedDipoleIdeal(moments)(target_rho, target_p, data["grid_spacing"].prod())
        data["V_ext"] = -(ideal["density_derivative"] + model.strength.detach()*target_rho)/data["beta"]
        data["E_ext"] = (ideal["polarization_derivative"] + model.strength.detach()*target_p)/data["beta"]
        numbers = target_rho.sum(0)*data["grid_spacing"].prod()
        result = PolarizationSolver(moments, model).solve(data, numbers)
        self.assertTrue(result["converged"], str(result["maximum_residual"]))
        torch.testing.assert_close(result["rho"], target_rho, rtol=0, atol=3e-8)
        torch.testing.assert_close(result["dipole_density"], target_p, rtol=0, atol=3e-8)
        self.assertTrue(any(item.get("accepted_step", 1.) < 1. for item in result["history"]))
        self.assertGreater(result["iterations"], 1)
        self.assertEqual(float(model.strength.grad), 7.)
        self.assertTrue(model.training)
        failure = PolarizationSolver(moments, model).solve(
            data, numbers, step_size=100., max_backtracks=1)
        self.assertFalse(failure["converged"])
        self.assertEqual(failure["status"], "line_search_failed")

    def test_grid_model_energy_integration(self):
        from equicdft import GridCACEModel, GridData, CartesianAFeatures, CartesianBFeatures, LocalReadout
        a = CartesianAFeatures(mean_density=0.5, dipole_density_scale=0.2,
                              include_polarization=True, cutoff_grid=1, max_power=0)
        b = CartesianBFeatures(0, 2, include_polarization=True)
        model = GridCACEModel(a_features=a, b_features=b,
                             readout=[LocalReadout(n_features=b.n_features + 1, hidden_sizes=(3,))],
                             grid_spacing=0.5, boltzmann_constant=1.)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.mul_(0.05)
        model.eval()
        data = GridData.from_dict({"grid_size": [3, 3, 3], "temperature": 1.4}, grid_info=model.grid_info)
        rho = 0.4 + 0.1*torch.rand(27, 1)
        polar = 0.02*torch.randn(27, 1, 3)
        solver = PolarizationSolver(0.8, model)
        target = solver.evaluate(dict(data, rho=rho, dipole_density=polar,
                                      V_ext=torch.zeros_like(rho), E_ext=torch.zeros_like(polar)))
        data["V_ext"] = -target["density_residual"]/data["beta"]
        data["E_ext"] = target["polarization_residual"]/data["beta"]
        with torch.no_grad():
            result = solver.solve(data, rho.sum(0)*0.5**3)
        self.assertTrue(result["converged"], str(result["maximum_residual"]))
        torch.testing.assert_close(result["rho"], rho, atol=1e-7, rtol=0)
        torch.testing.assert_close(result["dipole_density"], polar, atol=1e-7, rtol=0)
        self.assertFalse(model.training)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        with self.assertRaisesRegex(ValueError, "beta must agree"):
            solver.solve(dict(data, beta=data["beta"]*2), rho.sum(0)*0.5**3)


if __name__ == "__main__":
    unittest.main()
