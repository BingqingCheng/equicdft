"""Independent thermodynamic and canonical equilibrium tests for fixed dipoles."""

import itertools
import math
import unittest

import numpy as np
from tests.ideal_helpers import with_ideal
import torch

from equicdft.solver import (
    AndersonOptions,
    Constraints,
    ConvergenceOptions,
    EulerOptions,
    Initialization,
    MinimizeOptions,
)
from equicdft import FixedDipoleIdeal, GridSolver
from equicdft.functional.polarization_ideal import (
    inverse_langevin,
    langevin,
    log_sinhc,
)


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
        ideal = FixedDipoleIdeal([0.7, 1.9], thermal_wavelength=[1.2, 0.8])
        for volume in (0.125, 8.0):
            rho = (0.5 + torch.rand(3, 2)).requires_grad_()
            polar = (0.03*torch.randn(3, 2, 3)).requires_grad_()
            result = ideal.evaluate(dict(rho=rho, dipole_density=polar), volume)
            dr, dp = torch.autograd.grad(result["beta_F_id"], (rho, polar), create_graph=True)
            torch.testing.assert_close(dr/volume, result["ideal_density_derivative"], atol=2e-12, rtol=2e-11)
            torch.testing.assert_close(dp/volume, result["ideal_polarization_derivative"], atol=2e-12, rtol=2e-11)
            self.assertTrue(torch.autograd.gradcheck(lambda r, p: ideal.evaluate(dict(rho=r, dipole_density=p), volume)["beta_F_id"], (rho, polar), eps=1e-6, atol=2e-8, rtol=2e-6))
            mixed_r = torch.autograd.grad(dr[0, 0], polar, retain_graph=True)[0][0, 0, 1]
            mixed_p = torch.autograd.grad(dp[0, 0, 1], rho)[0][0, 0]
            torch.testing.assert_close(mixed_r, mixed_p, atol=1e-12, rtol=1e-10)

    def test_zero_polarization_hessian_and_scalar_limit(self):
        rho = torch.tensor([[0.4]], requires_grad=True)
        polar = torch.zeros(1, 1, 3, requires_grad=True)
        ideal = FixedDipoleIdeal(1.7, thermal_wavelength=1.3)
        result = ideal.evaluate(dict(rho=rho, dipole_density=polar), 8.)
        expected = 8*rho.sum()*(torch.log(rho[0, 0]*1.3**3) - 1)
        torch.testing.assert_close(result["beta_F_id"], expected)
        hessian = torch.autograd.functional.hessian(lambda p: ideal.evaluate(dict(rho=rho, dipole_density=p), 8.)["beta_F_id"], polar).reshape(3, 3)
        torch.testing.assert_close(hessian, 8*3/(1.7**2*0.4)*torch.eye(3), atol=1e-12, rtol=1e-12)
        self.assertTrue(torch.autograd.gradgradcheck(lambda r, p: ideal.evaluate(dict(rho=r, dipole_density=p), 8.)["beta_F_id"], (rho, polar), atol=2e-7))

    def test_near_saturation_and_inverse_derivative(self):
        p = torch.tensor([0., 1e-8, 1e-3, 0.2, 0.8, 0.99, 0.9999], requires_grad=True)
        xi = inverse_langevin(p)
        torch.testing.assert_close(langevin(xi), p, rtol=2e-12, atol=2e-14)
        self.assertTrue(torch.autograd.gradcheck(inverse_langevin, (p[:-1],), eps=1e-7, atol=1e-5, rtol=1e-5))
        polar = torch.zeros(1, 1, 3)
        polar[..., 0] = 0.9999
        self.assertTrue(torch.isfinite(FixedDipoleIdeal(1.).evaluate(dict(rho=torch.ones(1, 1), dipole_density=polar), 1.)["beta_F_id"]))

    def test_batch_and_rotation(self):
        ideal = FixedDipoleIdeal([1., 2.])
        rho = 0.5 + torch.rand(2, 5, 2)
        polar = 0.05*torch.randn(2, 5, 2, 3)
        rotation, _ = torch.linalg.qr(torch.randn(3, 3))
        original = ideal.evaluate(dict(rho=rho, dipole_density=polar), torch.tensor([0.125, 8.]))
        rotated = ideal.evaluate(dict(rho=rho, dipole_density=polar @ rotation.T), torch.tensor([0.125, 8.]))
        torch.testing.assert_close(original["beta_F_id"], rotated["beta_F_id"], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(original["ideal_polarization_derivative"] @ rotation.T, rotated["ideal_polarization_derivative"], rtol=1e-11, atol=1e-12)
        for batch, volume in enumerate((0.125, 8.)):
            torch.testing.assert_close(original["beta_F_id"][batch], ideal.evaluate(dict(rho=rho[batch], dipole_density=polar[batch]), volume)["beta_F_id"])

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
                ideal.local(rho, polar)
        with self.assertRaises(ValueError):
            ideal.local(torch.zeros_like(rho), torch.zeros(2, 1, 3))
        with self.assertRaises(ValueError):
            FixedDipoleIdeal(-1.)
        with self.assertRaisesRegex(ValueError, "must be real"):
            FixedDipoleIdeal(torch.tensor(1.7 + 0.2j))
        with self.assertRaisesRegex(ValueError, "must be real"):
            FixedDipoleIdeal(1.7, thermal_wavelength=torch.tensor(1.0 + 0.2j))


class TestGridSolverPolarization(unittest.TestCase):
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
        data = dict(data, thermal_wavelength=torch.tensor([0.8, 1.3]))
        solver = GridSolver(with_ideal(None, FixedDipoleIdeal(moments)))
        kwargs = {}
        if perturbed:
            rho = torch.exp(0.4*torch.randn_like(data["V_ext"]))
            rho *= torch.as_tensor(numbers)/(data["grid_spacing"].prod()*rho.sum(0))
            polar = 0.1*torch.randn(*rho.shape, 3)
            polar /= 1 + polar.norm(dim=-1, keepdim=True)
            kwargs = {"initial": Initialization(
                rho=rho, polarization=polar*(rho*torch.as_tensor(moments))[..., None],
            )}
        result = solver.solve(
            data,
            **kwargs,
            convergence=ConvergenceOptions(max_iter=350, tolerance_residual=1e-7),
            constraints=Constraints(particle_numbers=numbers),
        )
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
        solver = GridSolver(with_ideal(None, FixedDipoleIdeal([0.7, 1.9])))
        first = solver.evaluate(dict(data, rho=rho, dipole_density=polar))
        shift = torch.tensor([2., -3.])
        second = solver.evaluate(dict(data, V_ext=data["V_ext"]+shift, rho=rho, dipole_density=polar))
        torch.testing.assert_close(first["density_residual"], second["density_residual"], atol=1e-12, rtol=0)
        torch.testing.assert_close(second["beta_mu"]-first["beta_mu"], data["beta"]*shift)
        result = solver.solve(
            data,
            convergence=ConvergenceOptions(max_iter=1, tolerance_residual=1e-12),
            minimize=MinimizeOptions(step_size=0.1),
            constraints=Constraints(particle_numbers=[3., 4.]),
        )
        self.assertFalse(result["converged"])
        self.assertEqual(result["status"], "max_iter")

    def test_alignment_coordinate_chain_rule(self):
        data = self.case()
        moments = torch.tensor([0.7, 1.9])
        volume = data["grid_spacing"].prod()
        rho = (0.3 + 0.2 * torch.rand_like(data["V_ext"])).requires_grad_()
        alignment = (0.15 * torch.randn(*rho.shape, 3)).requires_grad_()
        polarization = (rho * moments)[..., None] * alignment
        ideal = FixedDipoleIdeal(moments)
        intrinsic = ideal.evaluate(
            {"rho": rho, "dipole_density": polarization}, volume
        )
        objective = intrinsic["beta_F_id"] + data["beta"] * volume * (
            (rho * data["V_ext"]).sum()
            - (polarization * data["E_ext"]).sum()
        )
        grad_rho, grad_alignment = torch.autograd.grad(
            objective, (rho, alignment)
        )

        evaluated = GridSolver(with_ideal(None, ideal)).evaluate(
            dict(data, rho=rho, dipole_density=polarization)
        )
        expected_rho = evaluated["beta_mu_local"] + (
            moments[..., None]
            * alignment.detach()
            * evaluated["polarization_residual"]
        ).sum(dim=-1)
        expected_alignment = (
            rho.detach() * moments
        )[..., None] * evaluated["polarization_residual"]
        torch.testing.assert_close(
            grad_rho / volume, expected_rho, atol=2.0e-12, rtol=2.0e-11
        )
        torch.testing.assert_close(
            grad_alignment / volume,
            expected_alignment,
            atol=2.0e-12,
            rtol=2.0e-11,
        )

    def test_direct_alignment_minimizer(self):
        data = self.case()
        moments = torch.tensor([0.7, 1.9])
        numbers = torch.tensor([3.0, 4.0])
        volume = data["grid_spacing"].prod()
        rho = torch.exp(0.4 * torch.randn_like(data["V_ext"]))
        rho *= numbers / (volume * rho.sum(dim=0))
        alignment = 0.1 * torch.randn(*rho.shape, 3)
        alignment /= 1.0 + alignment.norm(dim=-1, keepdim=True)
        polarization = (rho * moments)[..., None] * alignment

        expected_rho, expected_p = exact_solution(data, numbers, moments)
        scale = expected_rho.mean(dim=0)
        result = GridSolver(
            with_ideal(None, FixedDipoleIdeal(moments))
        ).solve(
            data,
            method="minimize",
            minimize=MinimizeOptions(
                fixed_dipole_coordinate="alignment",
            ),
            convergence=ConvergenceOptions(
                max_iter=350, tolerance_residual=1.0e-7
            ),
            constraints=Constraints(particle_numbers=numbers),
            initial=Initialization(rho=rho, polarization=polarization),
        )

        self.assertTrue(result["converged"], str(result["maximum_residual"]))
        self.assertEqual(
            result["solver_fixed_dipole_coordinate"], "alignment"
        )
        self.assertLess(float(result["maximum_alignment"]), 1.0)
        self.assertTrue(np.isfinite(result["final_relative_alignment_change"]))
        torch.testing.assert_close(result["particle_numbers"], numbers)
        torch.testing.assert_close(
            result["rho"] / scale,
            expected_rho / scale,
            atol=3.0e-7,
            rtol=0,
        )
        torch.testing.assert_close(
            result["dipole_density"] / (scale * moments)[..., None],
            expected_p / (scale * moments)[..., None],
            atol=3.0e-7,
            rtol=0,
        )
        energies = np.asarray(result["objective_history"])
        self.assertLessEqual(
            np.diff(energies).max(initial=0.0),
            1.0e-11 * max(1.0, abs(energies[0])),
        )

    def test_direct_alignment_minimizer_enforces_fraction_cap(self):
        data = {
            "V_ext": torch.zeros(4, 1),
            "E_ext": torch.tensor([[[10.0, 0.0, 0.0]]] * 4),
            "beta": torch.tensor(1.0),
            "grid_spacing": torch.ones(3),
        }
        result = GridSolver(
            with_ideal(None, FixedDipoleIdeal(0.8))
        ).solve(
            data,
            method="minimize",
            minimize=MinimizeOptions(fixed_dipole_coordinate="alignment"),
            convergence=ConvergenceOptions(
                max_iter=500, tolerance_residual=1.0e-8
            ),
            constraints=Constraints(
                particle_numbers=[2.0],
                maximum_polarization_fraction=0.2,
            ),
        )
        self.assertTrue(result["converged"], str(result["maximum_residual"]))
        self.assertAlmostEqual(
            float(result["maximum_alignment"]), 0.2, places=10
        )
        self.assertLess(float(result["max_polarization_residual"]), 1.0e-8)

    def test_direct_alignment_minimizer_allows_cap_beyond_safety_margin(self):
        cap = 1.0 - 1.0e-10
        data = {
            "V_ext": torch.zeros(4, 1),
            "E_ext": torch.tensor([[[1.0e10, 0.0, 0.0]]] * 4),
            "beta": torch.tensor(1.0),
            "grid_spacing": torch.ones(3),
        }
        result = GridSolver(
            with_ideal(None, FixedDipoleIdeal(1.0))
        ).solve(
            data,
            method="minimize",
            minimize=MinimizeOptions(fixed_dipole_coordinate="alignment"),
            convergence=ConvergenceOptions(
                max_iter=10, tolerance_residual=1.0e-8
            ),
            constraints=Constraints(
                particle_numbers=[2.0],
                maximum_polarization_fraction=cap,
            ),
        )

        self.assertTrue(result["converged"], str(result["maximum_residual"]))
        self.assertEqual(result["n_iter"], 1)
        self.assertAlmostEqual(
            float(result["maximum_alignment"]), cap, places=12
        )

    def test_direct_alignment_minimizer_guards_fixed_polarization_trial(self):
        rho = torch.full((4, 1), 0.5)
        polarization = torch.zeros(4, 1, 3)
        polarization[0, 0, 0] = 0.49
        data = {
            "V_ext": torch.tensor([[1.0e4], [0.0], [0.0], [0.0]]),
            "E_ext": torch.zeros(4, 1, 3),
            "beta": torch.tensor(1.0),
            "grid_spacing": torch.ones(3),
        }
        result = GridSolver(
            with_ideal(None, FixedDipoleIdeal(1.0))
        ).solve(
            data,
            method="minimize",
            minimize=MinimizeOptions(fixed_dipole_coordinate="alignment"),
            convergence=ConvergenceOptions(
                max_iter=500, tolerance_residual=1.0e-6
            ),
            constraints=Constraints(
                particle_numbers=[2.0], fixed_field="dipole_density"
            ),
            initial=Initialization(
                rho=rho, polarization=polarization
            ),
        )

        margin = 1.0 - max(
            1.0e-8, 64.0 * torch.finfo(rho.dtype).eps
        )
        self.assertTrue(result["converged"], str(result["maximum_residual"]))
        self.assertLess(float(result["maximum_alignment"]), margin)
        torch.testing.assert_close(
            result["dipole_density"], polarization
        )

    def test_direct_alignment_minimizer_never_lands_on_the_interior_margin(self):
        """A trial that would cross the margin is backtracked, not clipped.

        The margin is not a constraint: the inverse Langevin diverges there,
        so an iterate clipped onto it reports a residual of order 1/(1-|u|)
        and cannot move again, because every later step overshoots and is
        clipped back to the same point. One entry is enough to hold up the
        whole solve, so the margin has to stay unreachable.
        """

        torch.manual_seed(0)
        data = self.case(amplitude=5.0)
        moments = torch.tensor([0.7, 1.9])
        numbers = torch.tensor([3.0, 4.0])
        volume = data["grid_spacing"].prod()
        rho = torch.exp(0.4 * torch.randn_like(data["V_ext"]))
        rho *= numbers / (volume * rho.sum(dim=0))
        alignment = 0.1 * torch.randn(*rho.shape, 3)
        alignment /= 1.0 + alignment.norm(dim=-1, keepdim=True)

        result = GridSolver(
            with_ideal(None, FixedDipoleIdeal(moments))
        ).solve(
            data,
            method="minimize",
            minimize=MinimizeOptions(fixed_dipole_coordinate="alignment"),
            convergence=ConvergenceOptions(
                max_iter=20, tolerance_residual=1.0e-7
            ),
            constraints=Constraints(particle_numbers=numbers),
            initial=Initialization(
                rho=rho,
                polarization=(rho * moments)[..., None] * alignment,
            ),
        )

        margin = 1.0 - max(1.0e-8, 64.0 * torch.finfo(rho.dtype).eps)
        self.assertLess(float(result["maximum_alignment"]), margin)
        self.assertNotEqual(result["status"], "line_search_failed")
        # Twenty iterations do not converge this problem, but the alignment
        # already tracks the exact saturation rather than the margin.
        expected_rho, expected_p = exact_solution(data, numbers, moments)
        expected_alignment = float(
            (expected_p.norm(dim=-1) / (expected_rho * moments)).max()
        )
        self.assertAlmostEqual(
            float(result["maximum_alignment"]), expected_alignment, places=5
        )

    def test_direct_alignment_minimizer_keeps_safe_interior_margin(self):
        data = {
            "V_ext": torch.zeros(4, 1),
            "E_ext": torch.zeros(4, 1, 3),
            "beta": torch.tensor(1.0),
            "grid_spacing": torch.ones(3),
        }
        rho = torch.full((4, 1), 0.5)
        polarization = torch.zeros(4, 1, 3)
        polarization[..., 0] = (1.0 - 1.0e-8) * rho
        result = GridSolver(
            with_ideal(None, FixedDipoleIdeal(1.0))
        ).solve(
            data,
            method="minimize",
            minimize=MinimizeOptions(fixed_dipole_coordinate="alignment"),
            convergence=ConvergenceOptions(max_iter=2),
            constraints=Constraints(particle_numbers=[2.0]),
            initial=Initialization(rho=rho, polarization=polarization),
        )
        self.assertTrue(torch.isfinite(result["beta_A"]))
        self.assertLessEqual(
            float(result["maximum_alignment"]), 1.0 - 1.0e-8
        )
        # Both bounds above hold vacuously when the solver returns its own
        # input, which is what a line search with an absolute backtracking
        # floor does here: the alignment gradient is of order 1/(1-|u|).
        self.assertNotEqual(result["status"], "line_search_failed")
        self.assertGreaterEqual(int(result["n_iter"]), 1)

    def test_fixed_dipole_coordinate_validation(self):
        self.assertEqual(
            EulerOptions().fixed_dipole_coordinate, "conjugate_field"
        )
        with self.assertRaisesRegex(TypeError, "fixed_dipole_coordinate"):
            MinimizeOptions(fixed_dipole_coordinate=1)
        with self.assertRaisesRegex(ValueError, "fixed_dipole_coordinate"):
            MinimizeOptions(fixed_dipole_coordinate="polarization")
        with self.assertRaisesRegex(TypeError, "fixed_dipole_coordinate"):
            EulerOptions(fixed_dipole_coordinate=1)
        with self.assertRaisesRegex(ValueError, "fixed_dipole_coordinate"):
            EulerOptions(fixed_dipole_coordinate="field")

    def test_validation(self):
        solver = GridSolver(with_ideal(None, FixedDipoleIdeal([0.7, 1.9])))
        data = self.case()
        for changes in ({"beta": torch.tensor(-1.)}, {"E_ext": torch.zeros(24, 3)},
                        {"excluded_mask": torch.ones(24, dtype=torch.bool)},
                        {"grid_spacing": torch.tensor([1., 2., 1.])}):
            with self.assertRaises(ValueError):
                solver.solve(
                    dict(data, **changes),
                    constraints=Constraints(particle_numbers=[3., 4.]),
                )
        with self.assertRaises(ValueError):
            solver.solve(
                data,
                constraints=Constraints(particle_numbers=[3., 4.]),
                initial=Initialization(rho=torch.ones(24, 2)),
            )
        with self.assertRaisesRegex(ValueError, "requires particle_numbers"):
            solver.solve(data)
        accelerated = solver.solve(
            data,
            method="euler",
            euler=EulerOptions(anderson=AndersonOptions()),
            convergence=ConvergenceOptions(max_iter=1),
            constraints=Constraints(particle_numbers=[3., 4.]),
        )
        self.assertEqual(
            accelerated["solver_fixed_dipole_coordinate"], "conjugate_field"
        )

    def test_density_cap_is_enforced_by_both_solvers(self):
        data = {
            "V_ext": torch.tensor([[-10.0], [0.0], [0.0], [0.0]]),
            "E_ext": torch.tensor(
                [[[0.2, -0.1, 0.3]], [[-0.4, 0.2, 0.1]],
                 [[0.1, 0.5, -0.2]], [[-0.2, -0.3, 0.4]]]
            ),
            "beta": torch.tensor(1.0),
            "grid_spacing": torch.ones(3),
        }
        for method in ("minimize", "euler"):
            with self.subTest(method=method):
                result = GridSolver(with_ideal(None, FixedDipoleIdeal(0.8))).solve(
                    data,
                    method=method,
                    convergence=ConvergenceOptions(max_iter=500, tolerance_residual=1.0e-10),
                    euler=EulerOptions(tolerance_change=1.0e-12, adaptive_mixing=False, mixing=0.2) if method == 'euler' else None,
                    constraints=Constraints(particle_numbers=[2.0], maximum_density=0.6),
                )
                self.assertTrue(
                    result["converged"], str(result["maximum_residual"])
                )
                self.assertLessEqual(
                    result["rho"].max().item(), 0.6 + 1.0e-12
                )
                self.assertAlmostEqual(
                    result["rho"].sum().item(), 2.0, places=12
                )
                self.assertAlmostEqual(
                    result["rho"][0, 0].item(), 0.6, places=10
                )
                self.assertLess(float(result["maximum_alignment"]), 1.0)

    def test_density_cap_projects_rho_and_preserves_alignment(self):
        data = {
            "V_ext": torch.zeros(4, 1),
            "E_ext": torch.zeros(4, 1, 3),
            "beta": torch.tensor(1.0),
            "grid_spacing": torch.ones(3),
        }
        rho = torch.tensor([[0.8], [0.4], [0.4], [0.4]])
        polarization = torch.zeros(4, 1, 3)
        polarization[..., 0] = 0.2 * 0.8 * rho
        result = GridSolver(with_ideal(None, FixedDipoleIdeal(0.8))).solve(
            data,
            convergence=ConvergenceOptions(max_iter=1, tolerance_residual=100.0),
            constraints=Constraints(particle_numbers=[2.0], maximum_density=0.6),
            initial=Initialization(rho=rho, polarization=polarization),
        )
        self.assertEqual(result["n_iter"], 0)
        self.assertAlmostEqual(result["rho"].max().item(), 0.6, places=12)
        alignment = (
            result["dipole_density"][..., 0]
            / (0.8 * result["rho"])
        )
        torch.testing.assert_close(alignment, torch.full_like(alignment, 0.2))

    def test_density_floor_is_enforced_by_all_drivers(self):
        # A strongly repulsive voxel would empty without the floor; with it
        # the voxel rests on the floor, its outward residual is a satisfied
        # KKT condition, and the rest of the fluid is in equilibrium.
        data = {
            "V_ext": torch.tensor([[12.0], [0.0], [0.0], [0.0]]),
            "E_ext": torch.tensor(
                [[[0.2, -0.1, 0.3]], [[-0.4, 0.2, 0.1]],
                 [[0.1, 0.5, -0.2]], [[-0.2, -0.3, 0.4]]]
            ),
            "beta": torch.tensor(1.0),
            "grid_spacing": torch.ones(3),
        }
        drivers = [
            ("minimize", dict(minimize=MinimizeOptions())),
            ("minimize", dict(minimize=MinimizeOptions(fixed_dipole_coordinate="alignment"))),
            ("euler", dict(euler=EulerOptions(tolerance_change=1.0e-12, adaptive_mixing=False, mixing=0.2))),
            ("euler", dict(euler=EulerOptions(tolerance_change=1.0e-12, adaptive_mixing=False, mixing=0.2, fixed_dipole_coordinate="alignment"))),
        ]
        for method, options in drivers:
            with self.subTest(method=method, **{k: v.fixed_dipole_coordinate for k, v in options.items()}):
                result = GridSolver(with_ideal(None, FixedDipoleIdeal(0.8))).solve(
                    data,
                    method=method,
                    convergence=ConvergenceOptions(max_iter=500, tolerance_residual=1.0e-8),
                    constraints=Constraints(particle_numbers=[2.0], minimum_density=0.05),
                    **options,
                )
                self.assertTrue(result["converged"], str(result["maximum_residual"]))
                self.assertGreaterEqual(result["rho"].min().item(), 0.05 - 1.0e-12)
                self.assertAlmostEqual(result["rho"][0, 0].item(), 0.05, places=7)
                self.assertAlmostEqual(result["rho"].sum().item(), 2.0, places=12)
                # The floor voxel wants to drain (positive unconstrained
                # residual) and that wish is excluded from the KKT residual.
                unconstrained = (
                    result["beta_mu_local"][0, 0] - result["beta_mu"][0]
                ).item()
                self.assertGreater(unconstrained, 1.0)
                self.assertAlmostEqual(result["density_residual"][0, 0].item(), 0.0, places=8)
                torch.testing.assert_close(
                    result["solver_minimum_density"], torch.tensor([0.05])
                )
                # Away from the floor the ideal fluid is exactly Boltzmann
                # distributed over the free voxels, orientational partition
                # function included.
                weights = torch.exp(log_sinhc(0.8 * data["E_ext"][1:, 0].norm(dim=-1)))
                free = result["rho"][1:, 0]
                torch.testing.assert_close(
                    free / free.sum(), weights / weights.sum(), rtol=0, atol=1.0e-8
                )

    def test_density_floor_projects_the_start_and_reports_it(self):
        data = {
            "V_ext": torch.zeros(4, 1),
            "E_ext": torch.zeros(4, 1, 3),
            "beta": torch.tensor(1.0),
            "grid_spacing": torch.ones(3),
        }
        rho = torch.tensor([[0.01], [0.79], [0.6], [0.6]])
        polarization = torch.zeros(4, 1, 3)
        polarization[..., 0] = 0.2 * 0.8 * rho
        result = GridSolver(with_ideal(None, FixedDipoleIdeal(0.8))).solve(
            data,
            convergence=ConvergenceOptions(max_iter=1, tolerance_residual=100.0),
            constraints=Constraints(particle_numbers=[2.0], minimum_density=0.1),
            initial=Initialization(rho=rho, polarization=polarization),
        )
        self.assertEqual(result["n_iter"], 0)
        self.assertAlmostEqual(result["rho"].min().item(), 0.1, places=12)
        self.assertAlmostEqual(result["rho"].sum().item(), 2.0, places=12)
        alignment = result["dipole_density"][..., 0] / (0.8 * result["rho"])
        torch.testing.assert_close(alignment, torch.full_like(alignment, 0.2))

    def test_density_floor_validation(self):
        data = {
            "V_ext": torch.zeros(4, 1),
            "E_ext": torch.zeros(4, 1, 3),
            "beta": torch.tensor(1.0),
            "grid_spacing": torch.ones(3),
        }
        solver = GridSolver(with_ideal(None, FixedDipoleIdeal(0.8)))
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            solver.solve(data, constraints=Constraints(particle_numbers=[2.0], minimum_density=0.0))
        with self.assertRaisesRegex(ValueError, "below maximum_density"):
            solver.solve(data, constraints=Constraints(
                particle_numbers=[2.0], minimum_density=0.6, maximum_density=0.6))
        with self.assertRaisesRegex(ValueError, "infeasible under minimum_density"):
            solver.solve(data, constraints=Constraints(particle_numbers=[2.0], minimum_density=0.5))
        with self.assertRaisesRegex(ValueError, "below minimum_density"):
            solver.solve(
                data,
                method="minimize",
                constraints=Constraints(particle_numbers=[2.0], minimum_density=0.1, fixed_field="rho"),
                initial=Initialization(
                    rho=torch.tensor([[0.05], [0.65], [0.65], [0.65]]),
                    polarization=torch.zeros(4, 1, 3),
                ),
            )
        from equicdft import IdealGas
        scalar = GridSolver(with_ideal(None, IdealGas()))
        with self.assertRaisesRegex(ValueError, "minimum_density requires"):
            scalar.solve(
                {"V_ext": torch.zeros(4, 1), "mu": torch.zeros(1), "beta": torch.tensor(1.0),
                 "grid_spacing": torch.ones(3)},
                constraints=Constraints(minimum_density=0.1),
            )

    def test_polarization_fraction_cap_is_enforced_with_kkt_residual(self):
        data = {
            "V_ext": torch.zeros(4, 1),
            "E_ext": torch.tensor(
                [[[10.0, 0.0, 0.0]]] * 4,
                dtype=torch.float64,
            ),
            "beta": torch.tensor(1.0, dtype=torch.float64),
            "grid_spacing": torch.ones(3, dtype=torch.float64),
        }
        for method in ("minimize", "euler"):
            with self.subTest(method=method):
                result = GridSolver(with_ideal(None, FixedDipoleIdeal(0.8))).solve(
                    data,
                    method=method,
                    convergence=ConvergenceOptions(max_iter=500, tolerance_residual=1.0e-8),
                    euler=EulerOptions(tolerance_change=1.0e-12, adaptive_mixing=False, mixing=1.0, maximum_mixing=1.0) if method == 'euler' else None,
                    constraints=Constraints(particle_numbers=[2.0], maximum_polarization_fraction=0.2),
                )
                self.assertTrue(
                    result["converged"], str(result["maximum_residual"])
                )
                self.assertLessEqual(
                    float(result["maximum_alignment"]), 0.2 + 1.0e-12
                )
                self.assertAlmostEqual(
                    float(result["maximum_alignment"]), 0.2, places=10
                )
                self.assertGreater(
                    float(
                        result[
                            "unconstrained_scaled_polarization_residual"
                        ].abs().max()
                    ),
                    1.0,
                )
                self.assertLess(
                    float(result["max_polarization_residual"]), 1.0e-8
                )

    def test_invalid_polarization_fraction_cap_is_rejected(self):
        solver = GridSolver(with_ideal(None, FixedDipoleIdeal([0.7, 1.9])))
        for value in (0.0, 1.0, [0.2, 1.0]):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "lie in \\(0, 1\\)"):
                    solver.solve(
                        self.case(),
                        constraints=Constraints(particle_numbers=[3.0, 4.0], maximum_polarization_fraction=value),
                    )

    def test_infeasible_density_cap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "infeasible"):
            GridSolver(with_ideal(None, FixedDipoleIdeal([0.7, 1.9]))).solve(
                self.case(),
                constraints=Constraints(particle_numbers=[3.0, 4.0], maximum_density=[0.9, 1.0]),
            )

    def test_initial_polarization_without_initial_density(self):
        data = self.case()
        numbers = torch.tensor([3.0, 4.0])
        volume = data["grid_spacing"].prod()
        rho = torch.ones_like(data["V_ext"]) * numbers / (24 * volume)
        direction = torch.tensor([1.0, 0.0, 0.0])
        initial_p = (
            0.05
            * rho[..., None]
            * torch.tensor([0.7, 1.9])[..., None]
            * direction
        )
        result = GridSolver(with_ideal(None, FixedDipoleIdeal([0.7, 1.9]))).solve(
            data,
            convergence=ConvergenceOptions(tolerance_residual=100.0),
            constraints=Constraints(particle_numbers=numbers),
            initial=Initialization(polarization=initial_p),
        )
        self.assertEqual(result["n_iter"], 0)
        torch.testing.assert_close(result["dipole_density"], initial_p)

    def test_excess_energy_and_line_search(self):
        class Quadratic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.strength = torch.nn.Parameter(torch.tensor(12.))

            def forward(self, data, **kwargs):
                energy = 0.5*self.strength*data["grid_spacing"].prod()*(
                    data["rho"].square().sum() + data["dipole_density"].square().sum())
                out = self.ideal.evaluate(data, data["grid_spacing"].prod())
                out.update(beta_F_exc=energy, beta_F=out["beta_F_id"]+energy)
                if kwargs.get("compute_c1", True):
                    out["c1"] = -self.strength * data["rho"]
                    out["density_derivative"] = out["ideal_density_derivative"]-out["c1"]
                if kwargs.get("compute_polarization_derivative", False):
                    out["polarization_derivative_excess"] = self.strength * data["dipole_density"]
                    out["polarization_derivative"] = out["ideal_polarization_derivative"]+out["polarization_derivative_excess"]
                return out

        data = self.case()
        model = Quadratic()
        model.strength.grad = torch.tensor(7.)
        target_rho = 0.4 + 0.1*torch.rand_like(data["V_ext"])
        target_p = 0.04*torch.randn(*target_rho.shape, 3)
        moments = [0.7, 1.9]
        ideal = FixedDipoleIdeal(moments).local(target_rho, target_p)
        data["V_ext"] = -(ideal["ideal_density_derivative"] + model.strength.detach()*target_rho)/data["beta"]
        data["E_ext"] = (ideal["ideal_polarization_derivative"] + model.strength.detach()*target_p)/data["beta"]
        numbers = target_rho.sum(0)*data["grid_spacing"].prod()
        result = GridSolver(with_ideal(model, FixedDipoleIdeal(moments))).solve(
            data,
            convergence=ConvergenceOptions(tolerance_residual=1.0e-7),
            constraints=Constraints(particle_numbers=numbers),
        )
        self.assertTrue(result["converged"], str(result["maximum_residual"]))
        torch.testing.assert_close(result["rho"], target_rho, rtol=0, atol=3e-8)
        torch.testing.assert_close(result["dipole_density"], target_p, rtol=0, atol=3e-8)
        self.assertTrue(any(item.get("accepted_step", 1.) < 1. for item in result["history"]))
        self.assertGreater(result["n_iter"], 1)
        self.assertEqual(float(model.strength.grad), 7.)
        self.assertTrue(model.training)
        failure = GridSolver(with_ideal(model, FixedDipoleIdeal(moments))).solve(
            data,
            minimize=MinimizeOptions(step_size=100., minimum_step_size=100.),
            constraints=Constraints(particle_numbers=numbers),
        )
        self.assertFalse(failure["converged"])
        self.assertEqual(failure["status"], "line_search_failed")

    def test_fixed_field_minimization_is_conditional(self):
        data = self.case()
        moments = torch.tensor([0.7, 1.9])
        target_rho = 0.35 + 0.12 * torch.rand_like(data["V_ext"])
        target_p = 0.06 * torch.randn(*target_rho.shape, 3)
        target_p *= (
            0.5 * target_rho * moments
            / target_p.norm(dim=-1).clamp_min(1.0e-12)
        )[..., None]
        ideal = FixedDipoleIdeal(moments).local(target_rho, target_p)
        numbers = target_rho.sum(0) * data["grid_spacing"].prod()
        solver = GridSolver(with_ideal(None, FixedDipoleIdeal(moments)))

        fixed_p_data = dict(
            data,
            V_ext=-ideal["ideal_density_derivative"] / data["beta"],
            E_ext=torch.full_like(data["E_ext"], 30.0),
        )
        initial_rho = target_rho.roll(1, dims=0)
        fixed_p = solver.solve(
            fixed_p_data,
            method="minimize",
            convergence=ConvergenceOptions(max_iter=500, tolerance_residual=1.0e-8),
            constraints=Constraints(particle_numbers=numbers, fixed_field="dipole_density"),
            initial=Initialization(rho=initial_rho, polarization=target_p),
        )
        self.assertTrue(fixed_p["converged"], str(fixed_p["maximum_residual"]))
        torch.testing.assert_close(fixed_p["rho"], target_rho, atol=2.0e-8, rtol=0)
        torch.testing.assert_close(fixed_p["dipole_density"], target_p)
        self.assertLess(float(fixed_p["maximum_residual"]), 1.0e-8)
        self.assertGreater(float(fixed_p["full_maximum_residual"]), 1.0)
        self.assertEqual(fixed_p["solver_fixed_field"], "dipole_density")
        self.assertEqual(fixed_p["final_relative_polarization_change"], 0.0)

        fixed_rho_data = dict(
            data,
            V_ext=30.0 * data["V_ext"],
            E_ext=ideal["ideal_polarization_derivative"] / data["beta"],
        )
        fixed_rho = solver.solve(
            fixed_rho_data,
            method="minimize",
            convergence=ConvergenceOptions(max_iter=500, tolerance_residual=1.0e-8),
            constraints=Constraints(particle_numbers=numbers, fixed_field="rho"),
            initial=Initialization(rho=target_rho, polarization=torch.zeros_like(target_p)),
        )
        self.assertTrue(
            fixed_rho["converged"], str(fixed_rho["maximum_residual"])
        )
        torch.testing.assert_close(fixed_rho["rho"], target_rho)
        torch.testing.assert_close(
            fixed_rho["dipole_density"], target_p, atol=2.0e-8, rtol=0
        )
        self.assertLess(float(fixed_rho["maximum_residual"]), 1.0e-8)
        self.assertGreater(float(fixed_rho["full_maximum_residual"]), 1.0)
        self.assertEqual(fixed_rho["solver_fixed_field"], "rho")
        self.assertEqual(fixed_rho["final_relative_density_change"], 0.0)

    def test_fixed_polarization_enforces_density_floor(self):
        data = {
            "V_ext": torch.tensor([[10.0], [0.0], [0.0], [0.0]]),
            "E_ext": torch.zeros(4, 1, 3),
            "beta": torch.tensor(1.0),
            "grid_spacing": torch.ones(3),
        }
        rho = torch.full((4, 1), 0.5)
        polarization = torch.zeros(4, 1, 3)
        polarization[0, 0, 0] = 0.3
        result = GridSolver(with_ideal(None, FixedDipoleIdeal(1.0))).solve(
            data,
            method="minimize",
            convergence=ConvergenceOptions(max_iter=500, tolerance_residual=1.0e-9),
            constraints=Constraints(particle_numbers=[2.0], fixed_field="dipole_density", maximum_polarization_fraction=0.75),
            initial=Initialization(rho=rho, polarization=polarization),
        )
        self.assertTrue(result["converged"], str(result["maximum_residual"]))
        self.assertAlmostEqual(float(result["rho"][0]), 0.4, places=9)
        torch.testing.assert_close(result["dipole_density"], polarization)
        self.assertLess(float(result["maximum_residual"]), 1.0e-9)

    def test_fixed_field_validation(self):
        solver = GridSolver(with_ideal(None, FixedDipoleIdeal(0.8)))
        data = self.case()
        rho = torch.ones_like(data["V_ext"])
        rho *= torch.tensor([3.0, 4.0]) / (
            data["grid_spacing"].prod() * rho.sum(0)
        )
        polarization = torch.zeros(*rho.shape, 3)
        with self.assertRaisesRegex(ValueError, "fixed_field must"):
            solver.solve(
                data,
                constraints=Constraints(particle_numbers=[3.0, 4.0], fixed_field="temperature"),
                initial=Initialization(rho=rho, polarization=polarization),
            )
        with self.assertRaisesRegex(ValueError, "only with method='minimize'"):
            solver.solve(
                data,
                method="euler",
                constraints=Constraints(particle_numbers=[3.0, 4.0], fixed_field="rho"),
                initial=Initialization(rho=rho, polarization=polarization),
            )
        with self.assertRaisesRegex(ValueError, "requires an initial polarization"):
            solver.solve(
                data,
                constraints=Constraints(particle_numbers=[3.0, 4.0], fixed_field="dipole_density"),
                initial=Initialization(rho=rho),
            )

    def test_coupled_euler_expression(self):
        data = self.case()
        moments = [0.7, 1.9]
        numbers = [3.0, 4.0]
        result = GridSolver(with_ideal(None, FixedDipoleIdeal(moments))).solve(
            data,
            method="euler",
            convergence=ConvergenceOptions(max_iter=2, tolerance_residual=1.0e-10),
            euler=EulerOptions(mixing=1.0, adaptive_mixing=False, maximum_mixing=1.0),
            constraints=Constraints(particle_numbers=numbers),
        )
        expected_rho, expected_p = exact_solution(data, numbers, moments)
        self.assertTrue(result["converged"])
        self.assertEqual(result["n_iter"], 1)
        torch.testing.assert_close(result["rho"], expected_rho)
        torch.testing.assert_close(result["dipole_density"], expected_p)

    def test_alignment_coordinate_euler_mixes_density_and_alignment(self):
        data = self.case()
        moments = torch.tensor([0.7, 1.9])
        numbers = torch.tensor([3.0, 4.0])
        volume = data["grid_spacing"].prod()
        mixing = 0.2
        rho = torch.exp(0.4 * torch.randn_like(data["V_ext"]))
        rho *= numbers / (volume * rho.sum(dim=0))
        alignment = 0.1 * torch.randn(*rho.shape, 3)
        alignment /= 1.0 + alignment.norm(dim=-1, keepdim=True)
        polarization = (rho * moments)[..., None] * alignment

        target_rho, target_p = exact_solution(data, numbers, moments)
        target_u = target_p / (target_rho * moments)[..., None]
        expected_rho = (1.0 - mixing) * rho + mixing * target_rho
        expected_u = (1.0 - mixing) * alignment + mixing * target_u
        expected_p = (expected_rho * moments)[..., None] * expected_u

        result = GridSolver(
            with_ideal(None, FixedDipoleIdeal(moments))
        ).solve(
            data,
            method="euler",
            convergence=ConvergenceOptions(
                max_iter=1, tolerance_residual=1.0e-14
            ),
            euler=EulerOptions(
                mixing=mixing,
                adaptive_mixing=False,
                fixed_dipole_coordinate="alignment",
            ),
            constraints=Constraints(particle_numbers=numbers),
            initial=Initialization(rho=rho, polarization=polarization),
        )

        self.assertEqual(
            result["solver_fixed_dipole_coordinate"], "alignment"
        )
        self.assertTrue(np.isfinite(result["final_relative_alignment_change"]))
        torch.testing.assert_close(result["rho"], expected_rho)
        torch.testing.assert_close(result["dipole_density"], expected_p)
        actual_u = result["dipole_density"] / (
            result["rho"] * moments
        )[..., None]
        torch.testing.assert_close(actual_u, expected_u)

    def test_alignment_coordinate_euler_enforces_fraction_cap(self):
        data = {
            "V_ext": torch.zeros(4, 1),
            "E_ext": torch.tensor([[[10.0, 0.0, 0.0]]] * 4),
            "beta": torch.tensor(1.0),
            "grid_spacing": torch.ones(3),
        }
        result = GridSolver(
            with_ideal(None, FixedDipoleIdeal(0.8))
        ).solve(
            data,
            method="euler",
            convergence=ConvergenceOptions(
                max_iter=3, tolerance_residual=1.0e-8
            ),
            euler=EulerOptions(
                mixing=1.0,
                adaptive_mixing=False,
                maximum_mixing=1.0,
                fixed_dipole_coordinate="alignment",
            ),
            constraints=Constraints(
                particle_numbers=[2.0],
                maximum_polarization_fraction=0.2,
            ),
        )

        self.assertTrue(result["converged"], str(result["maximum_residual"]))
        self.assertAlmostEqual(
            float(result["maximum_alignment"]), 0.2, places=10
        )
        self.assertLess(float(result["max_polarization_residual"]), 1.0e-8)

    @staticmethod
    def conjugate(**changes):
        options = dict(
            mixing=0.5, adaptive_mixing=False, tolerance_change=1.0e-13,
            fixed_dipole_coordinate="conjugate_field",
        )
        options.update(changes)
        return EulerOptions(**options)

    def test_conjugate_field_euler_matches_exact_solution(self):
        # The ideal fluid's Euler map is exact in one step, so unit mixing
        # converges immediately; Anderson and the adaptive schedule must
        # reproduce the same closed-form solution from a perturbed start.
        data = self.case()
        numbers, moments = [3.0, 4.0], [0.7, 1.9]
        expected_rho, expected_p = exact_solution(data, numbers, moments)
        solver = GridSolver(with_ideal(None, FixedDipoleIdeal(moments)))
        rho = torch.exp(0.4 * torch.randn_like(data["V_ext"]))
        rho *= torch.as_tensor(numbers) / (data["grid_spacing"].prod() * rho.sum(0))
        polar = 0.1 * torch.randn(*rho.shape, 3)
        polar /= 1 + polar.norm(dim=-1, keepdim=True)
        perturbed = Initialization(
            rho=rho, polarization=polar * (rho * torch.as_tensor(moments))[..., None],
        )
        variants = [
            ("unit mixing", self.conjugate(mixing=1.0), Initialization(), 1),
            ("anderson", self.conjugate(anderson=AndersonOptions(history=6)), perturbed, 12),
            (
                "adaptive anderson",
                self.conjugate(
                    mixing=0.2, adaptive_mixing=True, minimum_mixing=0.05,
                    maximum_mixing=1.0, anderson=AndersonOptions(history=6),
                ),
                perturbed, 12,
            ),
        ]
        for label, euler, initial, budget in variants:
            with self.subTest(label):
                result = solver.solve(
                    data, method="euler", euler=euler, initial=initial,
                    convergence=ConvergenceOptions(max_iter=60, tolerance_residual=1.0e-9),
                    constraints=Constraints(particle_numbers=numbers),
                )
                self.assertTrue(result["converged"], str(result["maximum_residual"]))
                self.assertLessEqual(result["n_iter"], budget)
                self.assertEqual(result["solver_fixed_dipole_coordinate"], "conjugate_field")
                scale = expected_rho.mean(0)
                torch.testing.assert_close(result["rho"] / scale, expected_rho / scale, rtol=0, atol=1e-7)
                torch.testing.assert_close(
                    result["dipole_density"] / scale[..., None],
                    expected_p / scale[..., None], rtol=0, atol=1e-7,
                )
                torch.testing.assert_close(
                    result["particle_numbers"], torch.as_tensor(numbers), rtol=2e-14, atol=0,
                )
                self.assertEqual(result["solver_anderson"], euler.anderson is not None)
                energies = np.array([item["beta_A"] for item in result["history"]])
                self.assertLessEqual(np.diff(energies).max(initial=0), 1e-9 * abs(energies[0]))
                if euler.adaptive_mixing:
                    self.assertGreater(result["final_mixing"], euler.mixing)

    def test_conjugate_field_euler_respects_the_density_floor(self):
        data = self.case()
        data["V_ext"][5, 0] = 9.0
        numbers, moments = [3.0, 4.0], [0.7, 1.9]
        result = GridSolver(with_ideal(None, FixedDipoleIdeal(moments))).solve(
            data, method="euler",
            euler=self.conjugate(anderson=AndersonOptions(history=6)),
            convergence=ConvergenceOptions(max_iter=80, tolerance_residual=1.0e-9),
            constraints=Constraints(particle_numbers=numbers, minimum_density=0.05),
        )
        self.assertTrue(result["converged"], str(result["maximum_residual"]))
        self.assertGreaterEqual(float(result["rho"].min()), 0.05 - 1e-12)
        self.assertAlmostEqual(float(result["rho"][5, 0]), 0.05, places=9)
        self.assertGreater(float(result["beta_mu_local"][5, 0] - result["beta_mu"][0]), 1.0)
        self.assertAlmostEqual(float(result["density_residual"][5, 0]), 0.0, places=9)
        torch.testing.assert_close(result["particle_numbers"], torch.as_tensor(numbers), rtol=2e-14, atol=0)

    def test_conjugate_field_anderson_accelerates_a_stiff_excess(self):
        class Quadratic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.strength = torch.nn.Parameter(torch.tensor(12.))

            def forward(self, data, **kwargs):
                energy = 0.5*self.strength*data["grid_spacing"].prod()*(
                    data["rho"].square().sum() + data["dipole_density"].square().sum())
                out = self.ideal.evaluate(data, data["grid_spacing"].prod())
                out.update(beta_F_exc=energy, beta_F=out["beta_F_id"]+energy)
                if kwargs.get("compute_c1", True):
                    out["c1"] = -self.strength * data["rho"]
                    out["density_derivative"] = out["ideal_density_derivative"]-out["c1"]
                if kwargs.get("compute_polarization_derivative", False):
                    out["polarization_derivative_excess"] = self.strength * data["dipole_density"]
                    out["polarization_derivative"] = out["ideal_polarization_derivative"]+out["polarization_derivative_excess"]
                return out

        # A local quadratic excess gives the Euler map the Jacobian
        # -strength*rho, so plain mixing is stable only below
        # 2/(1+strength*rho) ~ 0.3 and converges slowly at 0.15. Anderson
        # keeps the same fixed point and needs a fraction of the steps; the
        # free-energy safeguard turns an unstable mixing of 0.5 into a
        # convergent schedule instead of a divergence.
        data = self.case()
        target_rho = 0.4 + 0.1*torch.rand_like(data["V_ext"])
        target_p = 0.04*torch.randn(*target_rho.shape, 3)
        moments = torch.tensor([0.7, 1.9])
        ideal = FixedDipoleIdeal(moments).local(target_rho, target_p)
        data["V_ext"] = -(ideal["ideal_density_derivative"] + 12.*target_rho)/data["beta"]
        data["E_ext"] = (ideal["ideal_polarization_derivative"] + 12.*target_p)/data["beta"]
        numbers = target_rho.sum(0)*data["grid_spacing"].prod()
        solver = GridSolver(with_ideal(Quadratic(), FixedDipoleIdeal(moments)))

        def solve(euler):
            return solver.solve(
                data, method="euler", euler=euler,
                convergence=ConvergenceOptions(max_iter=400, tolerance_residual=1.0e-8),
                constraints=Constraints(particle_numbers=numbers),
            )

        plain = solve(self.conjugate(mixing=0.15))
        accelerated = solve(self.conjugate(mixing=0.15, anderson=AndersonOptions(history=10)))
        adaptive = solve(self.conjugate(
            mixing=0.5, adaptive_mixing=True, minimum_mixing=0.01, maximum_mixing=1.0,
            anderson=AndersonOptions(history=10),
        ))
        for label, result in (("plain", plain), ("anderson", accelerated), ("adaptive", adaptive)):
            with self.subTest(label):
                self.assertTrue(result["converged"], str(result["maximum_residual"]))
                torch.testing.assert_close(result["rho"], target_rho, rtol=0, atol=3e-8)
                torch.testing.assert_close(result["dipole_density"], target_p, rtol=0, atol=3e-8)
        self.assertGreater(plain["n_iter"], 3 * accelerated["n_iter"])
        self.assertGreater(plain["n_iter"], accelerated["n_iter"] + adaptive["n_iter"])
        self.assertGreater(adaptive["mixing_backtracks"], 0)
        self.assertLess(min(item["mixing"] for item in adaptive["history"] if "mixing" in item), 0.5)

    def test_euler_density_target_honours_an_active_polarization_cap(self):
        # Two voxels of an ideal dipolar fluid, one in a strong field whose
        # alignment the cap holds at q. The constrained equilibrium keeps
        # the work of the field beyond the cap in the density weight:
        # rho_2/rho_1 = exp(q |h| - q h_c + log sinhc(h_c)), h_c = L^-1(q).
        cap = 0.8
        data = {
            "V_ext": torch.zeros(2, 1),
            "E_ext": torch.tensor([[[0.0, 0.0, 0.0]], [[10.0, 0.0, 0.0]]]),
            "beta": torch.tensor(1.0),
            "grid_spacing": torch.ones(3),
        }
        h_cap = float(inverse_langevin(torch.tensor(cap)))
        weight = math.exp(cap * 10.0 - cap * h_cap + float(log_sinhc(torch.tensor(h_cap))))
        expected = torch.tensor([[2.0 / (1.0 + weight)], [2.0 * weight / (1.0 + weight)]])
        solver = GridSolver(with_ideal(None, FixedDipoleIdeal(1.0)))
        runs = {
            "minimize": dict(method="minimize"),
            "euler polarization": dict(method="euler", euler=EulerOptions(
                tolerance_change=1.0e-13, adaptive_mixing=False, mixing=0.5)),
            "euler alignment": dict(method="euler", euler=EulerOptions(
                tolerance_change=1.0e-13, adaptive_mixing=False, mixing=0.5,
                fixed_dipole_coordinate="alignment")),
            "euler conjugate": dict(method="euler", euler=self.conjugate(mixing=1.0)),
        }
        for label, kwargs in runs.items():
            with self.subTest(label):
                result = solver.solve(
                    data, **kwargs,
                    convergence=ConvergenceOptions(max_iter=400, tolerance_residual=1.0e-10),
                    constraints=Constraints(particle_numbers=[2.0], maximum_polarization_fraction=cap),
                )
                self.assertTrue(result["converged"], str(result["maximum_residual"]))
                torch.testing.assert_close(result["rho"], expected, rtol=0, atol=1.0e-8)
                self.assertAlmostEqual(float(result["maximum_alignment"]), cap, places=8)

    def test_conjugate_field_uphill_step_stops_instead_of_being_accepted(self):
        class Quadratic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.strength = torch.nn.Parameter(torch.tensor(12.))

            def forward(self, data, **kwargs):
                energy = 0.5*self.strength*data["grid_spacing"].prod()*(
                    data["rho"].square().sum() + data["dipole_density"].square().sum())
                out = self.ideal.evaluate(data, data["grid_spacing"].prod())
                out.update(beta_F_exc=energy, beta_F=out["beta_F_id"]+energy)
                if kwargs.get("compute_c1", True):
                    out["c1"] = -self.strength * data["rho"]
                    out["density_derivative"] = out["ideal_density_derivative"]-out["c1"]
                if kwargs.get("compute_polarization_derivative", False):
                    out["polarization_derivative_excess"] = self.strength * data["dipole_density"]
                    out["polarization_derivative"] = out["ideal_polarization_derivative"]+out["polarization_derivative_excess"]
                return out

        # A fixed mixing far above the stability limit raises the free
        # energy on the first step; the driver must keep its start and say so.
        data = self.case()
        numbers = torch.tensor([3.0, 4.0])
        solver = GridSolver(with_ideal(Quadratic(), FixedDipoleIdeal([0.7, 1.9])))
        result = solver.solve(
            data, method="euler", euler=self.conjugate(mixing=1.0),
            convergence=ConvergenceOptions(max_iter=50, tolerance_residual=1.0e-8),
            constraints=Constraints(particle_numbers=numbers),
        )
        self.assertEqual(result["status"], "line_search_failed")
        self.assertEqual(result["n_iter"], 0)
        self.assertEqual(result["line_search_failures"], 1)
        energies = [item["beta_A"] for item in result["history"]]
        self.assertEqual(len(energies), 1)

    def test_conjugate_field_reaches_strong_alignment_and_deep_potentials(self):
        # Exact ideal equilibria with |u| = 0.995 and a 60 kT potential step
        # must be reachable: nothing but representability limits the
        # coordinates when no physical constraint is set.
        numbers, moments = [3.0, 4.0], [0.7, 1.9]
        data = self.case()
        data["E_ext"] = torch.zeros_like(data["E_ext"])
        data["E_ext"][:, 0, 0] = float(inverse_langevin(torch.tensor(0.995))) / (0.7 * 1.4)
        data["V_ext"] = torch.zeros_like(data["V_ext"])
        data["V_ext"][:12, 1] = 60.0 / 1.4
        expected_rho, expected_p = exact_solution(data, numbers, moments)
        result = GridSolver(with_ideal(None, FixedDipoleIdeal(moments))).solve(
            data, method="euler",
            euler=self.conjugate(mixing=1.0, max_conjugate_field_change=1000.0),
            convergence=ConvergenceOptions(max_iter=60, tolerance_residual=1.0e-9),
            constraints=Constraints(particle_numbers=numbers),
        )
        self.assertTrue(result["converged"], str(result["maximum_residual"]))
        self.assertAlmostEqual(float(result["maximum_alignment"]), 0.995, places=9)
        scale = expected_rho.mean(0)
        torch.testing.assert_close(result["rho"] / scale, expected_rho / scale, rtol=0, atol=1e-7)
        torch.testing.assert_close(
            result["dipole_density"] / scale[..., None], expected_p / scale[..., None], rtol=0, atol=1e-7,
        )

    def test_conjugate_field_deep_potential_in_both_dtypes(self):
        # The dilute voxel's exact density is about 1.75e-26, well inside
        # float32's normal range. A numerical guard must not raise it.
        for dtype, tolerance in ((torch.float32, 1.0e-4), (torch.float64, 1.0e-10)):
            with self.subTest(dtype=dtype):
                data = {
                    "V_ext": torch.tensor([[0.0], [60.0]], dtype=dtype),
                    "E_ext": torch.zeros(2, 1, 3, dtype=dtype),
                    "beta": torch.tensor(1.0, dtype=dtype),
                    "grid_spacing": torch.ones(3, dtype=dtype),
                }
                result = GridSolver(with_ideal(None, FixedDipoleIdeal(1.0))).solve(
                    data, method="euler",
                    # Resolve changes in the dilute voxel rather than
                    # stopping on a tiny absolute density change.
                    euler=self.conjugate(mixing=1.0, tolerance_change=1.0e-16),
                    convergence=ConvergenceOptions(
                        max_iter=100, tolerance_residual=tolerance,
                    ),
                    constraints=Constraints(particle_numbers=[2.0]),
                )
                expected = 2.0 * torch.softmax(-data["V_ext"].double(), dim=0)
                self.assertTrue(result["converged"], str(result["maximum_residual"]))
                self.assertLessEqual(float(result["maximum_residual"]), tolerance)
                # No absolute tolerance: otherwise zero passes for the
                # dilute voxel and the regression misses a wrong tail.
                torch.testing.assert_close(
                    result["rho"], expected.to(dtype), rtol=tolerance, atol=0.0,
                )
                torch.testing.assert_close(
                    result["particle_numbers"],
                    torch.tensor([2.0], dtype=dtype), rtol=tolerance, atol=0.0,
                )

    @staticmethod
    def quadratic_excess(strength=12.0):
        class Quadratic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.strength = torch.nn.Parameter(torch.tensor(strength))

            def forward(self, data, **kwargs):
                energy = 0.5*self.strength*data["grid_spacing"].prod()*(
                    data["rho"].square().sum() + data["dipole_density"].square().sum())
                out = self.ideal.evaluate(data, data["grid_spacing"].prod())
                out.update(beta_F_exc=energy, beta_F=out["beta_F_id"]+energy)
                if kwargs.get("compute_c1", True):
                    out["c1"] = -self.strength * data["rho"]
                    out["density_derivative"] = out["ideal_density_derivative"]-out["c1"]
                if kwargs.get("compute_polarization_derivative", False):
                    out["polarization_derivative_excess"] = self.strength * data["dipole_density"]
                    out["polarization_derivative"] = out["ideal_polarization_derivative"]+out["polarization_derivative_excess"]
                return out

        return Quadratic()

    def test_lbfgs_minimizer_matches_exact_solution(self):
        data = self.case()
        numbers, moments = [3.0, 4.0], [0.7, 1.9]
        expected_rho, expected_p = exact_solution(data, numbers, moments)
        solver = GridSolver(with_ideal(None, FixedDipoleIdeal(moments)))
        rho = torch.exp(0.4 * torch.randn_like(data["V_ext"]))
        rho *= torch.as_tensor(numbers) / (data["grid_spacing"].prod() * rho.sum(0))
        polar = 0.1 * torch.randn(*rho.shape, 3)
        polar /= 1 + polar.norm(dim=-1, keepdim=True)
        perturbed = Initialization(
            rho=rho, polarization=polar * (rho * torch.as_tensor(moments))[..., None],
        )
        for window in (1, 5):
            with self.subTest(window=window):
                result = solver.solve(
                    data, method="minimize", initial=perturbed,
                    minimize=MinimizeOptions(lbfgs_history=10, nonmonotone_window=window),
                    convergence=ConvergenceOptions(max_iter=100, tolerance_residual=1.0e-9),
                    constraints=Constraints(particle_numbers=numbers),
                )
                self.assertTrue(result["converged"], str(result["maximum_residual"]))
                self.assertLess(result["n_iter"], 40)
                self.assertEqual(result["solver_lbfgs_history"], 10)
                self.assertGreater(result["n_energy_evaluations"], 0)
                scale = expected_rho.mean(0)
                torch.testing.assert_close(result["rho"] / scale, expected_rho / scale, rtol=0, atol=1e-7)
                torch.testing.assert_close(
                    result["dipole_density"] / scale[..., None],
                    expected_p / scale[..., None], rtol=0, atol=1e-7,
                )
                torch.testing.assert_close(
                    result["particle_numbers"], torch.as_tensor(numbers), rtol=2e-14, atol=0,
                )
                energies = np.array([item["beta_A"] for item in result["history"]])
                if window == 1:
                    self.assertLessEqual(np.diff(energies).max(initial=0), 1e-9 * abs(energies[0]))

    def test_lbfgs_minimizer_needs_fewer_evaluations_on_a_stiff_excess(self):
        # The local quadratic excess makes the mirror step's stability limit
        # 2/(1+strength*rho) ~ 0.3, so mirror descent backtracks on every
        # iteration; curvature pairs restore unit steps. Trials cost an
        # energy-only evaluation; only accepted points are differentiated.
        data = self.case()
        target_rho = 0.4 + 0.1*torch.rand_like(data["V_ext"])
        target_p = 0.04*torch.randn(*target_rho.shape, 3)
        moments = torch.tensor([0.7, 1.9])
        ideal = FixedDipoleIdeal(moments).local(target_rho, target_p)
        data["V_ext"] = -(ideal["ideal_density_derivative"] + 12.*target_rho)/data["beta"]
        data["E_ext"] = (ideal["ideal_polarization_derivative"] + 12.*target_p)/data["beta"]
        numbers = target_rho.sum(0)*data["grid_spacing"].prod()
        solver = GridSolver(with_ideal(self.quadratic_excess(), FixedDipoleIdeal(moments)))

        def solve(options):
            return solver.solve(
                data, method="minimize", minimize=options,
                convergence=ConvergenceOptions(max_iter=400, tolerance_residual=1.0e-8),
                constraints=Constraints(particle_numbers=numbers),
            )

        mirror = solve(MinimizeOptions())
        lbfgs = solve(MinimizeOptions(lbfgs_history=10))
        nonmonotone = solve(MinimizeOptions(lbfgs_history=10, nonmonotone_window=5))
        for label, result in (("mirror", mirror), ("lbfgs", lbfgs), ("nonmonotone", nonmonotone)):
            with self.subTest(label):
                self.assertTrue(result["converged"], str(result["maximum_residual"]))
                torch.testing.assert_close(result["rho"], target_rho, rtol=0, atol=3e-8)
                torch.testing.assert_close(result["dipole_density"], target_p, rtol=0, atol=3e-8)
        self.assertGreater(mirror["n_evaluations"], 3 * lbfgs["n_evaluations"])
        self.assertGreater(mirror["n_evaluations"], 3 * nonmonotone["n_evaluations"])
        self.assertGreater(mirror["n_evaluations"], lbfgs["n_evaluations"] + lbfgs["n_energy_evaluations"])
        self.assertEqual(lbfgs["lbfgs_pairs"], min(10, lbfgs["n_iter"]))

    def test_lbfgs_minimizer_handles_bounds_as_an_active_set(self):
        # A floor holding an emptied voxel, and a polarization cap holding a
        # strongly driven one: both are KKT points, and the free voxels must
        # be exactly Boltzmann distributed among themselves.
        data = {
            "V_ext": torch.tensor([[12.0], [0.0], [0.0], [0.0]]),
            "E_ext": torch.tensor(
                [[[0.2, -0.1, 0.3]], [[-0.4, 0.2, 0.1]],
                 [[0.1, 0.5, -0.2]], [[-0.2, -0.3, 0.4]]]
            ),
            "beta": torch.tensor(1.0),
            "grid_spacing": torch.ones(3),
        }
        result = GridSolver(with_ideal(None, FixedDipoleIdeal(0.8))).solve(
            data, method="minimize", minimize=MinimizeOptions(lbfgs_history=10),
            convergence=ConvergenceOptions(max_iter=200, tolerance_residual=1.0e-9),
            constraints=Constraints(particle_numbers=[2.0], minimum_density=0.05),
        )
        self.assertTrue(result["converged"], str(result["maximum_residual"]))
        # Active-set membership uses the KKT tolerance of the residual
        # (1e-6 absolute here), so the voxel rests within it of the floor.
        self.assertGreaterEqual(result["rho"][0, 0].item(), 0.05 - 1e-12)
        self.assertLessEqual(result["rho"][0, 0].item(), 0.05 + 2e-6)
        self.assertGreater((result["beta_mu_local"][0, 0] - result["beta_mu"][0]).item(), 1.0)
        weights = torch.exp(log_sinhc(0.8 * data["E_ext"][1:, 0].norm(dim=-1)))
        free = result["rho"][1:, 0]
        torch.testing.assert_close(free / free.sum(), weights / weights.sum(), rtol=0, atol=1e-8)

        cap = 0.8
        capped = {
            "V_ext": torch.zeros(2, 1),
            "E_ext": torch.tensor([[[0.0, 0.0, 0.0]], [[10.0, 0.0, 0.0]]]),
            "beta": torch.tensor(1.0),
            "grid_spacing": torch.ones(3),
        }
        h_cap = float(inverse_langevin(torch.tensor(cap)))
        weight = math.exp(cap * 10.0 - cap * h_cap + float(log_sinhc(torch.tensor(h_cap))))
        expected = torch.tensor([[2.0 / (1.0 + weight)], [2.0 * weight / (1.0 + weight)]])
        result = GridSolver(with_ideal(None, FixedDipoleIdeal(1.0))).solve(
            capped, method="minimize", minimize=MinimizeOptions(lbfgs_history=10),
            convergence=ConvergenceOptions(max_iter=200, tolerance_residual=1.0e-10),
            constraints=Constraints(particle_numbers=[2.0], maximum_polarization_fraction=cap),
        )
        self.assertTrue(result["converged"], str(result["maximum_residual"]))
        torch.testing.assert_close(result["rho"], expected, rtol=0, atol=1.0e-8)
        self.assertAlmostEqual(float(result["maximum_alignment"]), cap, places=8)

    def test_fully_bound_ideal_equilibrium_is_converged(self):
        data = {
            "V_ext": torch.tensor([[0.0], [10.0], [-3.0], [-20.0]]),
            "E_ext": torch.zeros(4, 1, 3),
            "beta": torch.tensor(1.0),
            "grid_spacing": torch.ones(3),
        }
        # Independent bounded Boltzmann solution: mu=-2.7 lies inside
        # [max(local at cap), min(local at floor)] and gives N=2 exactly.
        expected = torch.exp(-2.7 - data["V_ext"]).clamp(0.1, 0.9)
        drivers = (
            ("euler", {}),
            ("minimize", {}),
            ("minimize", {"minimize": MinimizeOptions(lbfgs_history=5)}),
        )
        for method, options in drivers:
            with self.subTest(method=method, options=options):
                result = GridSolver(with_ideal(None, FixedDipoleIdeal(1.0))).solve(
                    data, method=method, **options,
                    constraints=Constraints(
                        particle_numbers=[2.0], minimum_density=0.1,
                        maximum_density=0.9,
                    ),
                    initial=Initialization(rho=expected),
                    convergence=ConvergenceOptions(
                        max_iter=10, tolerance_residual=1.0e-10,
                    ),
                )
                self.assertTrue(result["converged"], result["status"])
                self.assertEqual(result["n_iter"], 0)
                self.assertEqual(float(result["maximum_residual"]), 0.0)
                torch.testing.assert_close(result["rho"], expected)
                torch.testing.assert_close(
                    result["dipole_density"], torch.zeros(4, 1, 3),
                )

    def test_lbfgs_reaches_fully_bound_equilibrium(self):
        # These starts previously reached the right solution but reported
        # a nonzero residual and stalled because the multiplier was wrong.
        for seed in (0, 7, 8):
            with self.subTest(seed=seed):
                potential = 3.0 * torch.randn(6, 1, generator=torch.Generator().manual_seed(seed))
                expected = torch.full((6, 1), 0.1)
                expected[potential[:, 0].argsort()[:3]] = 0.9
                local = expected.log() + potential
                mu_lower = local[expected == 0.9].max()
                mu_upper = local[expected == 0.1].min()
                self.assertLessEqual(float(mu_lower), float(mu_upper))
                # Verify the analytic bounded Boltzmann certificate.
                torch.testing.assert_close(
                    torch.exp(0.5 * (mu_lower + mu_upper) - potential).clamp(0.1, 0.9),
                    expected,
                )
                result = GridSolver(with_ideal(None, FixedDipoleIdeal(1.0))).solve(
                    dict(V_ext=potential, E_ext=torch.zeros(6, 1, 3),
                         beta=torch.tensor(1.0), grid_spacing=torch.ones(3)),
                    method="minimize", minimize=MinimizeOptions(lbfgs_history=5),
                    constraints=Constraints(
                        particle_numbers=[3.0], minimum_density=0.1, maximum_density=0.9,
                    ),
                    initial=Initialization(rho=torch.tensor([[0.1], [0.9]]).repeat(3, 1)),
                    convergence=ConvergenceOptions(max_iter=100, tolerance_residual=1.0e-6),
                )
                self.assertTrue(result["converged"], result["status"])
                torch.testing.assert_close(result["rho"], expected, rtol=0, atol=1.0e-10)

    def two_voxel_exchange(self, shift, minimize):
        # Both voxels start on a bound, so no interior voxel fixes the
        # chemical potential; the solve must still swap the mass.
        data = {
            "V_ext": torch.tensor([[-4.0], [4.0]], dtype=torch.float64) + shift,
            "E_ext": torch.zeros(2, 1, 3, dtype=torch.float64),
            "beta": torch.tensor(1.0, dtype=torch.float64),
            "grid_spacing": torch.ones(3, dtype=torch.float64),
        }
        return GridSolver(with_ideal(None, FixedDipoleIdeal(1.0))).solve(
            data, method="minimize", minimize=minimize,
            convergence=ConvergenceOptions(max_iter=100, tolerance_residual=1.0e-10),
            constraints=Constraints(particle_numbers=[1.0], minimum_density=0.1, maximum_density=0.9),
            initial=Initialization(rho=torch.tensor([[0.1], [0.9]], dtype=torch.float64)),
        )

    def test_lbfgs_bound_release_is_gauge_invariant(self):
        # A constant shift of the potential must not change a canonical
        # solve, in particular not which pinned voxels are released.
        expected = torch.tensor([[0.9], [0.1]], dtype=torch.float64)
        for shift in (0.0, 10.0, -10.0, 1000.0):
            with self.subTest(shift=shift):
                result = self.two_voxel_exchange(shift, MinimizeOptions(lbfgs_history=10))
                self.assertTrue(result["converged"], result["status"])
                torch.testing.assert_close(result["rho"], expected, rtol=0, atol=1.0e-9)

    def test_lbfgs_line_search_starts_from_step_size(self):
        # step_size scales the quasi-Newton direction: a step that reaches
        # the exchange in one trial when the independent log-change limit is
        # inactive, and one too short to move the state.
        result = self.two_voxel_exchange(
            0.0, MinimizeOptions(
                lbfgs_history=10, step_size=2.0, minimum_step_size=1.5,
                max_log_density_change=4.0,
            ),
        )
        self.assertTrue(result["converged"], result["status"])
        self.assertEqual(result["n_iter"], 1)
        self.assertEqual(result["n_energy_evaluations"], 1)
        self.assertEqual(result["history"][0]["accepted_step"], 2.0)
        result = self.two_voxel_exchange(
            0.0, MinimizeOptions(lbfgs_history=10, step_size=1.0e-6, minimum_step_size=1.0e-12),
        )
        self.assertEqual(result["history"][0]["accepted_step"], 1.0e-6)
        self.assertLess(abs(result["rho"][0, 0].item() - 0.1), 1.0e-2)

    def test_lbfgs_step_size_cannot_bypass_log_density_change_limit(self):
        data = {
            "V_ext": torch.tensor([[10.0], [-10.0]], dtype=torch.float64),
            "E_ext": torch.zeros(2, 1, 3, dtype=torch.float64),
            "beta": torch.tensor(1.0, dtype=torch.float64),
            "grid_spacing": torch.ones(3, dtype=torch.float64),
        }
        initial = torch.ones(2, 1, dtype=torch.float64)
        for step_size in (5.0, 1000.0):
            with self.subTest(step_size=step_size):
                result = GridSolver(
                    with_ideal(None, FixedDipoleIdeal(1.0))
                ).solve(
                    data,
                    method="minimize",
                    minimize=MinimizeOptions(
                        lbfgs_history=5, step_size=step_size,
                        max_log_density_change=0.1,
                    ),
                    constraints=Constraints(particle_numbers=[2.0]),
                    initial=Initialization(
                        rho=initial, polarization=torch.zeros(2, 1, 3),
                    ),
                    convergence=ConvergenceOptions(
                        max_iter=1, tolerance_residual=1.0e-15,
                    ),
                )
                # Canonical normalization adds a common log-density shift;
                # for the symmetric trial it is below 0.006.
                change = torch.log(result["rho"] / initial).abs().max()
                self.assertLessEqual(float(change), 0.106)
                self.assertTrue(torch.isfinite(result["rho"]).all())
                self.assertTrue(torch.all(result["rho"] > 0.0))

    def test_lbfgs_minimizer_validation(self):
        with self.assertRaisesRegex(ValueError, "lbfgs_history"):
            MinimizeOptions(lbfgs_history=-1)
        with self.assertRaisesRegex(ValueError, "nonmonotone_window"):
            MinimizeOptions(nonmonotone_window=0)
        with self.assertRaisesRegex(ValueError, "conjugate_field"):
            MinimizeOptions(lbfgs_history=5, fixed_dipole_coordinate="alignment")
        data = self.case()
        solver = GridSolver(with_ideal(None, FixedDipoleIdeal([0.7, 1.9])))
        with self.assertRaisesRegex(ValueError, "fixed_field"):
            solver.solve(
                data, method="minimize", minimize=MinimizeOptions(lbfgs_history=5),
                constraints=Constraints(particle_numbers=[3.0, 4.0], fixed_field="rho"),
                initial=Initialization(rho=torch.full((24, 2), 0.5) * torch.tensor([3.0, 4.0]) / (0.125 * 12.0)),
            )
        # A density-only model takes the option too: it selects the scalar
        # projected L-BFGS driver (tests/solver/test_solver.py covers it).
        from equicdft import IdealGas
        result = GridSolver(with_ideal(None, IdealGas())).solve(
            {"V_ext": torch.zeros(4, 1), "mu": torch.zeros(1), "beta": torch.tensor(1.0),
             "grid_spacing": torch.ones(3)},
            method="minimize", minimize=MinimizeOptions(lbfgs_history=5),
        )
        self.assertEqual(result["solver_lbfgs_history"], 5)

    def test_conjugate_field_validation(self):
        with self.assertRaisesRegex(ValueError, "max_conjugate_field_change"):
            self.conjugate(max_conjugate_field_change=0.0)
        with self.assertRaisesRegex(ValueError, "anderson_safeguard"):
            AndersonOptions(safeguard=1.0)
        data = self.case()
        solver = GridSolver(with_ideal(None, FixedDipoleIdeal([0.7, 1.9])))
        constraints = Constraints(particle_numbers=[3.0, 4.0])
        with self.assertRaisesRegex(ValueError, "conjugate_field"):
            solver.solve(
                data, method="euler", constraints=constraints,
                euler=EulerOptions(fixed_dipole_coordinate="alignment", anderson=AndersonOptions()),
            )
        # The shared EulerOptions default is accepted by scalar models, where
        # the fixed-dipole coordinate is irrelevant and never reported.
        from equicdft import IdealGas
        scalar = GridSolver(with_ideal(None, IdealGas())).solve(
            {"V_ext": torch.zeros(4, 1), "mu": torch.zeros(1),
             "beta": torch.tensor(1.0), "grid_spacing": torch.ones(3)},
            method="euler", euler=EulerOptions(),
            convergence=ConvergenceOptions(max_iter=1),
        )
        self.assertNotIn("solver_fixed_dipole_coordinate", scalar)

    def test_excluded_voxels(self):
        data = self.case()
        data["excluded_mask"] = torch.zeros(24, dtype=torch.bool)
        data["excluded_mask"][[0, 7, 19]] = True
        result = GridSolver(with_ideal(None, FixedDipoleIdeal([0.7, 1.9]))).solve(
            data,
            convergence=ConvergenceOptions(tolerance_residual=1.0e-9),
            constraints=Constraints(particle_numbers=[3.0, 4.0]),
        )
        excluded = data["excluded_mask"]
        self.assertTrue(result["converged"])
        self.assertTrue(torch.equal(result["rho"][excluded], torch.zeros(3, 2)))
        self.assertTrue(
            torch.equal(
                result["dipole_density"][excluded], torch.zeros(3, 2, 3)
            )
        )
        torch.testing.assert_close(
            result["particle_numbers"], torch.tensor([3.0, 4.0])
        )

    def test_solver_covariance_under_all_48_cubic_actions(self):
        size = 3
        positions = np.indices((size,) * 3).reshape(3, -1).T
        data = {
            "V_ext": torch.randn(size**3, 1),
            "E_ext": torch.randn(size**3, 1, 3),
            "beta": torch.tensor(0.8),
            "grid_spacing": torch.ones(3),
        }
        solver = GridSolver(with_ideal(None, FixedDipoleIdeal(1.2)))
        reference = solver.solve(
            data,
            convergence=ConvergenceOptions(tolerance_residual=1.0e-10),
            constraints=Constraints(particle_numbers=[5.0]),
        )
        actions = [
            np.eye(3, dtype=int)[list(permutation)]
            * np.asarray(signs)[:, None]
            for permutation in itertools.permutations(range(3))
            for signs in itertools.product((1, -1), repeat=3)
        ]
        self.assertEqual(len({tuple(a.flat) for a in actions}), 48)
        for action in actions:
            centers = ((2 * positions + 1) @ action.T) % (2 * size)
            destination = torch.tensor(
                np.ravel_multi_index(
                    ((centers - 1) // 2).T, (size, size, size)
                )
            )
            matrix = torch.tensor(action, dtype=torch.float64)
            transformed = dict(data)
            transformed["V_ext"] = torch.empty_like(data["V_ext"])
            transformed["V_ext"][destination] = data["V_ext"]
            transformed["E_ext"] = torch.empty_like(data["E_ext"])
            transformed["E_ext"][destination] = data["E_ext"] @ matrix.T
            result = solver.solve(
                transformed,
                convergence=ConvergenceOptions(tolerance_residual=1.0e-10),
                constraints=Constraints(particle_numbers=[5.0]),
            )
            torch.testing.assert_close(
                result["rho"][destination], reference["rho"]
            )
            torch.testing.assert_close(
                result["dipole_density"][destination],
                reference["dipole_density"] @ matrix.T,
            )

    def test_grid_model_energy_integration(self):
        from equicdft import GridCACEModel, GridData, CartesianAFeatures, CartesianBFeatures, LocalReadout
        a = CartesianAFeatures(mean_density=0.5,
                              include_polarization=True, cutoff_grid=1, max_power=0)
        b = CartesianBFeatures(0, 2, include_polarization=True)
        model = GridCACEModel(a_features=a, b_features=b,
                             readout=[LocalReadout(n_features=b.n_features + 1, hidden_sizes=(3,))],
                             polarization_scale=.2, grid_spacing=0.5, boltzmann_constant=1.)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.mul_(0.05)
        model.eval()
        data = GridData.from_dict({"grid_size": [3, 3, 3], "temperature": 1.4}, grid_info=model.grid_info)
        rho = 0.4 + 0.1*torch.rand(27, 1)
        polar = 0.02*torch.randn(27, 1, 3)
        solver = GridSolver(with_ideal(model, FixedDipoleIdeal(0.8)))
        target = solver.evaluate(dict(data, rho=rho, dipole_density=polar,
                                      V_ext=torch.zeros_like(rho), E_ext=torch.zeros_like(polar)))
        data["V_ext"] = -target["density_residual"]/data["beta"]
        data["E_ext"] = target["polarization_residual"]/data["beta"]
        for method in ("minimize", "euler"):
            with self.subTest(method=method), torch.no_grad():
                result = solver.solve(
                    data,
                    method=method,
                    convergence=ConvergenceOptions(max_iter=400, tolerance_residual=1.0e-7),
                    euler=EulerOptions(tolerance_change=1.0e-10) if method == 'euler' else None,
                    constraints=Constraints(particle_numbers=rho.sum(0)*0.5**3),
                )
                self.assertTrue(
                    result["converged"], str(result["maximum_residual"])
                )
                torch.testing.assert_close(
                    result["rho"], rho, atol=1e-7, rtol=0
                )
                torch.testing.assert_close(
                    result["dipole_density"], polar, atol=1e-7, rtol=0
                )
        self.assertFalse(model.training)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        with self.assertRaisesRegex(ValueError, "beta must agree"):
            solver.solve(
                dict(data, beta=data["beta"]*2),
                constraints=Constraints(particle_numbers=rho.sum(0)*0.5**3),
            )


if __name__ == "__main__":
    unittest.main()
