"""Physical chemical-potential outputs, masks, gradients and serialization."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from equicdft import (
    FixedDipoleIdeal, GridCACEModel, GridSolver, IdealGas, LDAReadout,
    Loss, Metrics, NoiseOptions, TensorLoss, Trainer, load_model, save_model,
)
from equicdft.serialization import build_model, model_payload
from equicdft.response.curvature import _energy_only_model


class TestChemicalPotential(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.get_default_dtype()
        self.rng = torch.get_rng_state()
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(109)

    def tearDown(self):
        torch.set_default_dtype(self.dtype)
        torch.set_rng_state(self.rng)

    def model(self, polarized=False, **options):
        ideal = (FixedDipoleIdeal([.7, 1.2], thermal_wavelength=[1.1, 1.3])
                 if polarized else IdealGas(thermal_wavelength=[1.1, 1.3]))
        settings = dict(compute_c1=False, compute_local_mu=True, rho_min=.15)
        settings.update(options)
        return GridCACEModel(
            None, None, [LDAReadout(n_types=2, hidden_sizes=(4,), zero_init=False,
                                   include_polarization=polarized)],
            ideal=ideal, n_types=2, polarization_scale=.3 if polarized else None, grid_spacing=.5, mean_density=.4,
            boltzmann_constant=1., **settings,
        )

    def data(self, polarized=False):
        rho = .2 + .3 * torch.rand(2, 5, 2)
        rho[:, 1, 0] = .1
        data = dict(rho=rho, V_ext=torch.randn_like(rho),
                    temperature=torch.tensor([1., 2.]), grid_spacing=torch.full((2, 3), .5))
        if polarized:
            data["dipole_density"] = .05 * rho[..., None] * torch.randn(2, 5, 2, 3)
        return data

    @staticmethod
    def loss():
        return Loss([TensorLoss(
            "beta_mu_local", "beta_mu_local", "beta_mu_average",
            weights_key="voxel_weights",
        )])

    def test_total_derivative_and_separate_field_species_means(self):
        for polarized in (False, True):
            with self.subTest(polarized=polarized):
                model, data = self.model(polarized), self.data(polarized)
                output = model(data)
                weights = (data["rho"] > .15).to(data["rho"])
                expected = output["density_derivative"] + data["V_ext"] / data["temperature"][:, None, None]
                expected = expected.masked_fill(weights == 0, 0.)
                torch.testing.assert_close(output["beta_mu_local"], expected)
                torch.testing.assert_close(output["beta_mu_average"],
                    (weights * expected).sum(-2) / weights.sum(-2))
                self.assertEqual(output["beta_mu_average"].shape, (2, 2))
                self.assertNotIn("local_chemical_potential", output)
                self.assertNotIn("average_chemical_potential", output)
                self.assertNotIn("chemical_potential_weights", output)
                self.assertIn("c1", output)  # requested automatically despite compute_c1=False
                self.assertFalse(output["voxel_weights"].requires_grad)

    def test_mask_exclusions_and_explicit_weights(self):
        model, data = self.model(), self.data()
        data["excluded_mask"] = torch.zeros(2, 5, dtype=torch.bool)
        data["excluded_mask"][:, 0] = True
        data["rho"][:, 0] = 0
        data["V_ext"][:, 0] = float("inf")
        data["valid"] = ~data["excluded_mask"]
        data["valid"][:, 2] = False
        weights = torch.ones_like(data["rho"])
        weights[:, 3] = .3
        data["voxel_weights"] = weights.requires_grad_()
        output = model(data)
        expected = weights.detach().clone()
        expected[:, [0, 2]] = 0
        torch.testing.assert_close(output["voxel_weights"], expected)
        self.assertTrue(torch.isfinite(output["beta_mu_local"]).all())
        self.assertTrue((output["beta_mu_local"][:, [0, 2]] == 0).all())
        self.loss()(output, data)["total"].backward()
        self.assertIsNone(weights.grad)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_errors_are_explicit_and_diagnostics_can_be_disabled(self):
        model, data = self.model(), self.data()
        missing = dict(data)
        missing.pop("V_ext")
        with self.assertRaisesRegex(ValueError, "requires V_ext"):
            model(missing)
        self.assertNotIn("beta_mu_local", model(missing, compute_local_mu=False))
        with self.assertRaisesRegex(ValueError, "total derivative"):
            model(data, compute_ideal=False)
        for value in (-1., float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "rho_min"):
                self.model(rho_min=value)
        for bad in (torch.ones(2), torch.full_like(data["rho"], -1.),
                    torch.full_like(data["rho"], float("nan"))):
            with self.assertRaisesRegex(ValueError, "voxel_weights"):
                model(dict(data, voxel_weights=bad))
        weights = torch.ones_like(data["rho"])
        weights[0, :, 1] = 0
        with self.assertRaisesRegex(ValueError, "every field/species"):
            model(dict(data, voxel_weights=weights))
        with self.assertRaisesRegex(ValueError, "V_ext must be finite"):
            model(dict(data, V_ext=torch.full_like(data["rho"], float("nan"))))
        vacuum = copy.deepcopy(data)
        vacuum["rho"].detach_()
        vacuum["rho"][:, 0] = 0
        with self.assertRaisesRegex(ValueError, "vacuum"):
            model(dict(vacuum, voxel_weights=torch.ones_like(data["rho"])))

    def test_external_shift_changes_only_diagnostics_and_not_the_loss(self):
        model, data = self.model(True), self.data(True)
        first = model(data)
        offset = torch.tensor([[1., 2.], [3., 4.]])
        shifted = dict(data, V_ext=data["V_ext"] + data["temperature"][:, None, None] * offset[:, None, :])
        second = model(shifted)
        for key in ("beta_F", "beta_F_exc", "beta_F_id", "density_derivative", "c1"):
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
        torch.testing.assert_close(second["beta_mu_average"],
                                   first["beta_mu_average"] + offset)
        torch.testing.assert_close(self.loss()(first, data)["total"],
                                   self.loss()(second, shifted)["total"])

    def test_loss_and_all_parameter_gradients_match_centered_derivative(self):
        for polarized in (False, True):
            model, data = self.model(polarized), self.data(polarized)
            output = model(data)
            actual = self.loss()(output, data)["total"]
            data["target"] = -data["V_ext"] / data["temperature"][:, None, None]
            expected = TensorLoss("rho", "density_derivative", "target", center_axis=-2,
                                  weights_key="voxel_weights")(output, data)
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
            parameters = tuple(model.parameters())
            for left, right in zip(torch.autograd.grad(actual, parameters, retain_graph=True),
                                   torch.autograd.grad(expected, parameters)):
                torch.testing.assert_close(left, right, rtol=1e-11, atol=1e-12)

    def test_sem_recomputes_selection_mu_and_average(self):
        for polarized in (False, True):
            model, batch = self.model(polarized), self.data(polarized)
            batch["rho"][:, 3, 0] = .2
            batch["rho_std"] = torch.zeros_like(batch["rho"])
            batch["rho_std"][:, 1, 0] = .2  # below cutoff -> above cutoff
            batch["rho_std"][:, 3, 0] = .1  # above cutoff -> below cutoff
            if polarized:
                batch["dipole_density"].zero_()
                batch["dipole_density_std"] = torch.zeros_like(batch["dipole_density"])
            original = model.voxel_weights(batch)
            trainer = Trainer(model, self.loss(), device="cpu")
            def draws(value):
                return torch.tensor([1., -1., 1., -1.]).reshape_as(value) if polarized else torch.where(
                    torch.arange(5)[None, :, None] == 3, -torch.ones_like(value), torch.ones_like(value))
            with patch("torch.randn_like", side_effect=draws):
                noisy, output, losses, _ = trainer._forward_loss(
                    batch, trainer.loss, noise=NoiseOptions(density=True, density_floor=.01))
            realized = model.voxel_weights(noisy)
            torch.testing.assert_close(output["voxel_weights"], realized)
            self.assertFalse(torch.equal(realized, original))
            self.assertTrue((noisy["rho"][:, 1, 0] > .15).all())
            self.assertTrue((noisy["rho"][:, 3, 0] < .15).all())
            expected = (output["density_derivative"] + noisy["V_ext"] / noisy["temperature"][:, None, None])
            torch.testing.assert_close(output["beta_mu_average"],
                                       (expected * realized).sum(-2) / realized.sum(-2))
            self.assertNotIn("voxel_weights", batch)
            noisy["target"] = -noisy["V_ext"] / noisy["temperature"][:, None, None]
            centered = TensorLoss("rho", "density_derivative", "target", center_axis=-2,
                                  weights_key="voxel_weights")(output, noisy)
            torch.testing.assert_close(losses["total"], centered, rtol=1e-12, atol=1e-12)
            parameters = tuple(model.parameters())
            for left, right in zip(torch.autograd.grad(losses["total"], parameters, retain_graph=True),
                                   torch.autograd.grad(centered, parameters)):
                torch.testing.assert_close(left, right, rtol=1e-11, atol=1e-12)

    def test_serialization_and_old_current_configs(self):
        model, data = self.model(True).eval(), self.data(True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"model.pt"
            save_model(model, path)
            restored = load_model(path)
        self.assertTrue(restored.compute_local_mu)
        self.assertEqual(restored.rho_min, .15)
        for key, value in model(copy.deepcopy(data)).items():
            torch.testing.assert_close(restored(copy.deepcopy(data))[key], value, rtol=0, atol=0)
        payload = model_payload(model)
        payload["config"].pop("compute_local_mu")
        payload["config"].pop("rho_min")
        old = build_model(payload)
        self.assertFalse(old.compute_local_mu)
        self.assertEqual(old.rho_min, 0.)

    def test_energy_and_solver_paths_ignore_fitting_diagnostics(self):
        for polarized in (False, True):
            model = self.model(polarized, rho_min=100.).eval()
            batch = self.data(polarized)
            data = {key: value[0] for key, value in batch.items()}
            data["beta"] = 1 / data["temperature"]
            if polarized:
                data["E_ext"] = torch.zeros_like(data["dipole_density"])
            fieldless = dict(data)
            fieldless.pop("V_ext")
            energy = _energy_only_model(model, fieldless)
            self.assertIn("beta_F_exc", energy)
            self.assertNotIn("beta_mu_local", energy)
            for derivatives in (False, True):
                output = GridSolver(model).evaluate(data, compute_derivatives=derivatives)
                self.assertNotIn("voxel_weights", output)
                self.assertNotIn("beta_mu_average", output)
                if derivatives:
                    torch.testing.assert_close(output["beta_mu_local"],
                                               output["density_derivative"] + data["beta"] * data["V_ext"])
                self.assertEqual("density_derivative" in output, derivatives)

    def test_metrics_compare_mu_with_its_average(self):
        model, data = self.model(), self.data()
        output = model(data)
        metric = Metrics("beta_mu_average", prediction_key="beta_mu_local",
                         selection_mask_key="voxel_weights")
        actual = metric(output, data)
        expected = self.loss()(output, data)["total"].sqrt()
        torch.testing.assert_close(actual["rmse"], expected)


if __name__ == "__main__":
    unittest.main()
