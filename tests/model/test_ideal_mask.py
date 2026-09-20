"""Masked ideal evaluation without changing excess inputs or solver physics."""

import copy
import unittest

import torch

from equicdft import FixedDipoleIdeal, GridCACEModel, GridSolver, IdealGas
from equicdft.solver import Constraints, ConvergenceOptions, Initialization
from tests.representation import test_polarization_symmetry as symmetry


class TestIdealMask(unittest.TestCase):
    def setUp(self):
        self.dtype, self.rng = torch.get_default_dtype(), torch.get_rng_state()
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(73)

    def tearDown(self):
        torch.set_default_dtype(self.dtype)
        torch.set_rng_state(self.rng)

    @staticmethod
    def model(polarized=True, rho_min=1e-4):
        return GridCACEModel(
            ideal=FixedDipoleIdeal([.7, 1.9]) if polarized else IdealGas(),
            n_types=2, grid_spacing=1., boltzmann_constant=1., rho_min=rho_min,
            compute_polarization_derivative=polarized,
        )

    @staticmethod
    def data(polarized=True):
        rho = torch.full((8, 2), .3)
        data = dict(rho=rho, temperature=torch.tensor(1.), beta=torch.tensor(1.),
                    grid_size=torch.tensor([2, 2, 2]), grid_spacing=torch.ones(3),
                    V_ext=torch.linspace(-.1, .1, 16).reshape_as(rho))
        if polarized:
            data.update(dipole_density=torch.full((8, 2, 3), .01),
                        E_ext=torch.full((8, 2, 3), .02))
        return data

    def test_masked_invalid_dipoles_energy_gradients_and_hessian_are_zero(self):
        ideal = FixedDipoleIdeal([.7, 1.9])
        data = self.data()
        mask = torch.ones_like(data["rho"], dtype=torch.bool)
        mask[0] = False
        data["dipole_density"][0] = 10.
        rho, polar = data["rho"].requires_grad_(), data["dipole_density"].requires_grad_()
        output = ideal.evaluate(data, .125, mask=mask)
        interior = ideal.local(rho[1:], polar[1:])
        expected_energy = .125 * (rho[1:] * interior["energy_per_particle"]).sum()
        torch.testing.assert_close(output["beta_F_id"], expected_energy)
        torch.testing.assert_close(output["ideal_derivative_mask"], mask)
        dr, dp = torch.autograd.grad(output["beta_F_id"], (rho, polar), create_graph=True)
        for derivative, analytic in ((dr, output["ideal_density_derivative"]),
                                     (dp, output["ideal_polarization_derivative"])):
            torch.testing.assert_close(derivative / .125, analytic)
            self.assertEqual(derivative[0].abs().max().item(), 0.)
        for derivative in (dr, dp):
            second = torch.autograd.grad(derivative.sum(), (rho, polar), retain_graph=True)
            for value in second:
                self.assertTrue(torch.isfinite(value).all())
                self.assertEqual(value[0].abs().max().item(), 0.)
        self.assertTrue(torch.equal(polar[0], torch.full_like(polar[0], 10.)))

    def test_unmasked_invalid_dipoles_and_vacuum_remain_strict(self):
        ideal, data = FixedDipoleIdeal([.7, 1.9]), self.data()
        data["dipole_density"][0] = 10.
        for mask in (None, torch.ones_like(data["rho"], dtype=torch.bool)):
            with self.assertRaisesRegex(ValueError, "fixed dipoles"):
                ideal.evaluate(data, 1., mask=mask)
        data["rho"][0] = 0.
        with self.assertRaisesRegex(ValueError, "vacuum"):
            ideal.evaluate(data, 1.)
        masked = ideal.evaluate(data, 1., mask=data["rho"] > 0)
        self.assertEqual(masked["ideal_polarization_derivative"][0].abs().max().item(), 0.)

    def test_model_cutoff_valid_and_excluded_masks_are_combined_per_species(self):
        model, data = self.model(), self.data()
        data["rho"][0, 0] = 1e-4  # strict rho > rho_min, independently per species
        data["rho"][1, 1] = 1e-5
        data["rho"][3] = 0.
        data["valid"] = torch.ones(8, dtype=torch.bool)
        data["valid"][2] = False
        data["excluded_mask"] = torch.arange(8) == 3
        expected = data["rho"] > 1e-4
        expected[[2, 3]] = False
        data["dipole_density"][~expected] = 10.
        batch = {key: torch.stack((value, value)) for key, value in data.items()}
        output = model(batch)
        torch.testing.assert_close(model.ideal_mask(batch), expected.expand(2, -1, -1))
        torch.testing.assert_close(output["ideal_derivative_mask"], expected.expand(2, -1, -1))
        for key in ("ideal_density_derivative", "ideal_polarization_derivative"):
            self.assertEqual(output[key][:, ~expected].abs().max().item(), 0.)
        data["dipole_density"][4] = 10.
        with self.assertRaisesRegex(ValueError, "fixed dipoles"):
            model(data)

    def test_scalar_mask_and_empty_selection(self):
        for polarized in (False, True):
            data, model = self.data(polarized), self.model(polarized)
            data["rho"][0] = 1e-5
            if polarized:
                data["dipole_density"][0] = 10.
            output = model(data)
            self.assertEqual(output["ideal_density_derivative"][0].abs().max().item(), 0.)
            model.rho_min = 100.
            empty = model(data)
            self.assertEqual(empty["beta_F_id"].item(), 0.)
            self.assertEqual(empty["ideal_density_derivative"].abs().max().item(), 0.)
            self.assertFalse(empty["ideal_derivative_mask"].any())
            with self.assertRaisesRegex(ValueError, "positive weight"):
                model(data, compute_local_mu=True)

    def test_explicit_weights_freeze_selection_without_weighting_energy(self):
        model, data = self.model(), self.data()
        data["rho"][0] = 1e-5
        data["dipole_density"][0] = 1e-7
        weights = torch.ones_like(data["rho"])
        weights[1] = 0.
        data["dipole_density"][1] = 10.
        data["beta_mu_weights"] = weights
        output = model(data, compute_local_mu=True)
        self.assertTrue(output["ideal_derivative_mask"][0].all())
        self.assertFalse(output["ideal_derivative_mask"][1].any())
        data["beta_mu_weights"] = 2.3 * weights
        scaled = model(data, compute_local_mu=True)
        for key in ("beta_F_id", "ideal_density_derivative", "ideal_polarization_derivative"):
            torch.testing.assert_close(output[key], scaled[key], atol=0, rtol=0)

    def test_excess_features_lda_coulomb_and_gaussian_keep_original_fields(self):
        model = symmetry.TestFullPolarizationSymmetry.model("fft", .4).train()
        model.rho_min = 1e-4
        data = symmetry.TestFullPolarizationSymmetry.data()
        data["dipole_density"] *= .1
        data["rho"][0] = 1e-5
        data["dipole_density"][0] = 1.
        original = copy.deepcopy(data)
        excess = model(copy.deepcopy(data), compute_ideal=False)
        output = model(data)
        for key in ("beta_F_exc", "c1", "polarization_derivative_excess"):
            torch.testing.assert_close(output[key], excess[key], atol=0, rtol=0)
        for key, value in original.items():
            torch.testing.assert_close(data[key], value, atol=0, rtol=0)
        gradients = torch.autograd.grad(output["beta_F"], (data["rho"], data["dipole_density"]))
        for gradient, key in zip(gradients, ("density_derivative", "polarization_derivative")):
            torch.testing.assert_close(gradient / model.voxel_volume, output[key])
        with self.assertRaisesRegex(ValueError, "fixed dipoles"):
            model(copy.deepcopy(original), apply_ideal_mask=False)

    def test_solver_evaluation_and_updates_ignore_fitting_masks(self):
        for polarized in (False, True):
            reference_model = self.model(polarized, rho_min=0.)
            masked_model = self.model(polarized, rho_min=100.)
            data = self.data(polarized)
            masked_data = dict(data, valid=torch.zeros(8, dtype=torch.bool),
                               beta_mu_weights=torch.zeros_like(data["rho"]))
            self.assertEqual(masked_model(copy.deepcopy(masked_data))["beta_F_id"].item(), 0.)
            expected = reference_model(copy.deepcopy(data))
            full = masked_model(copy.deepcopy(masked_data), apply_ideal_mask=False)
            for key in expected:
                torch.testing.assert_close(full[key], expected[key], atol=0, rtol=0)
            reference, masked = GridSolver(reference_model), GridSolver(masked_model)
            for derivatives in (False, True):
                actual = masked.evaluate(masked_data, compute_derivatives=derivatives)
                baseline = reference.evaluate(data, compute_derivatives=derivatives)
                for key in baseline:
                    torch.testing.assert_close(actual[key], baseline[key], atol=0, rtol=0)
            for method in ("euler", "minimize"):
                options = dict(method=method, constraints=Constraints(particle_numbers=[2.4, 2.4]),
                               convergence=ConvergenceOptions(max_iter=3),
                               initial=Initialization(rho=data["rho"], polarization=data.get("dipole_density")))
                actual, baseline = masked.solve(masked_data, **options), reference.solve(data, **options)
                for key in ("rho", "beta_F", "density_derivative"):
                    torch.testing.assert_close(actual[key], baseline[key], atol=0, rtol=0)
                if polarized:
                    torch.testing.assert_close(actual["dipole_density"], baseline["dipole_density"], atol=0, rtol=0)
                    bad = copy.deepcopy(masked_data)
                    bad["dipole_density"][0] = 10.
                    with self.assertRaisesRegex(ValueError, "fixed dipoles"):
                        masked.evaluate(bad)

    def test_mask_validation_and_global_input_validation(self):
        ideal, data = FixedDipoleIdeal([.7, 1.9]), self.data()
        for mask in (torch.ones(8, dtype=torch.bool), torch.ones_like(data["rho"]), True):
            with self.assertRaisesRegex(ValueError, "ideal mask"):
                ideal.evaluate(data, 1., mask=mask)
        for key in ("valid", "excluded_mask"):
            with self.assertRaisesRegex(ValueError, key):
                self.model()(dict(data, **{key: torch.ones(8)}))
        with self.assertRaisesRegex((ValueError, TypeError), "apply_ideal_mask"):
            self.model()(data, apply_ideal_mask="yes")
        # Masking does not legalize nonfinite fields or nonzero hard-wall rho.
        data = self.data()
        mask = torch.zeros_like(data["rho"], dtype=torch.bool)
        data["dipole_density"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            ideal.evaluate(data, 1., mask=mask)
        data["dipole_density"][0] = 0.
        data["excluded_mask"] = torch.arange(8) == 0
        with self.assertRaisesRegex(ValueError, "zero at excluded"):
            ideal.evaluate(data, 1., mask=mask)


if __name__ == "__main__":
    unittest.main()
