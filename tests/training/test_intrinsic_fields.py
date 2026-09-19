"""Canonical field fitting using repository losses and model-owned ideals."""

import unittest
import torch
from equicdft import TensorLoss, Metrics, FixedDipoleIdeal, NoiseOptions


class TestIntrinsicFieldLoss(unittest.TestCase):
    def test_noisy_total_and_explicit_excess_objectives_have_equal_gradients(self):
        """Same noisy fields, original masks, canonical gauge and moment scaling."""
        from tests.model_fixtures import example_model, grid_data
        from tests.representation.test_polarization_symmetry import TestFullPolarizationSymmetry

        with torch.random.fork_rng():
            torch.manual_seed(91)
            for polarized in (False, True):
                for density_noise in ((True,) if not polarized else (False, True)):
                    with self.subTest(polarized=polarized, density_noise=density_noise):
                        if polarized:
                            model = TestFullPolarizationSymmetry.model("fft", 0.).double().train()
                            model.ideal = FixedDipoleIdeal(2., thermal_wavelength=1.7).double()
                            batch = TestFullPolarizationSymmetry.data()
                        else:
                            model = example_model().double().train()
                            batch = grid_data(shape=(4, 4, 4))
                        batch = {key: value.double() if value.is_floating_point() else value
                                 for key, value in batch.items()}
                        rho = batch["rho"]
                        batch["rho_std"] = .03 * rho
                        batch["V_ext"] = torch.randn_like(rho)
                        batch["beta"] = rho.new_tensor(1. / float(batch["temperature"]))
                        batch["rho_target"] = -batch["beta"] * batch["V_ext"]
                        # Freeze the original-data mask before drawing noise.
                        batch["weights"] = (rho > rho.median()).to(rho)
                        if polarized:
                            batch["dipole_density_std"] = torch.full_like(batch["dipole_density"], .01)
                            batch["P_target"] = batch["beta"] * torch.randn_like(batch["dipole_density"])
                            batch["P_weights"] = batch["weights"][..., None].expand_as(batch["P_target"])
                        noisy = NoiseOptions(density=density_noise, polarization=polarized).apply(
                            batch, ideal=model.ideal,
                        )
                        outputs = model(noisy)
                        # This is the old target formula, evaluated explicitly on the
                        # very same perturbed fields, not supported by augmentation.
                        noisy["old_rho_target"] = outputs["ideal_density_derivative"].detach() - noisy["rho_target"]
                        density_options = dict(center_axis=-2, weights_key="weights", residual_scale=1.3)
                        new = TensorLoss("rho", "density_derivative", "rho_target", **density_options)(outputs, noisy)
                        old = TensorLoss("rho", "c1", "old_rho_target", **density_options)(outputs, noisy)
                        if polarized:
                            noisy["old_P_target"] = noisy["P_target"] - outputs["ideal_polarization_derivative"].detach()
                            vector_options = dict(weights_key="P_weights",
                                                  residual_scale=model.ideal.dipole_magnitude.reshape(-1, 1))
                            new = new + .7 * TensorLoss("P", "polarization_derivative", "P_target", **vector_options)(outputs, noisy)
                            old = old + .7 * TensorLoss("P", "polarization_derivative_excess", "old_P_target", **vector_options)(outputs, noisy)
                        torch.testing.assert_close(new, old, rtol=1e-12, atol=1e-12)
                        parameters = tuple(p for p in model.parameters() if p.requires_grad)
                        new_grad = torch.autograd.grad(new, parameters, retain_graph=True)
                        old_grad = torch.autograd.grad(old, parameters)
                        for actual, expected in zip(new_grad, old_grad):
                            torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)

    def test_model_derivative_loss_matches_old_local_mu_loss_and_gradients(self):
        from tests.model_fixtures import example_model, grid_data
        model = example_model().double()
        batch = {key: value.double() if value.is_floating_point() else value
                 for key, value in grid_data().items()}
        batch["target"] = -batch["beta"] * batch["V_ext"]
        batch["weights"] = (batch["rho"] > .4).to(batch["rho"])
        outputs = model(batch)
        actual = TensorLoss("mu", "density_derivative", "target",
                            center_axis=-2, weights_key="weights")(outputs, batch)
        local = (outputs["ideal_density_derivative"] + batch["beta"] * batch["V_ext"]
                 - outputs["c1"])
        weights = batch["weights"]
        mean = (weights * local).sum(-2, keepdim=True) / weights.sum(-2, keepdim=True)
        expected = (weights * (local - mean).square()).sum() / weights.sum()
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-13)
        parameters = tuple(model.parameters())
        actual_grad = torch.autograd.grad(actual, parameters, retain_graph=True)
        expected_grad = torch.autograd.grad(expected, parameters)
        for actual_value, expected_value in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual_value, expected_value, rtol=1e-11, atol=1e-12)

    def test_masked_gauge_and_scaling(self):
        prediction = torch.tensor([[[1.], [3.], [50.]], [[2.], [6.], [-30.]]],
                                  dtype=torch.float64, requires_grad=True)
        target = torch.tensor([[[0.], [1.], [0.]], [[1.], [3.], [0.]]], dtype=torch.float64)
        weights = torch.tensor([[[1.], [1.], [0.]], [[1.], [1.], [0.]]], dtype=torch.float64)
        loss = TensorLoss("rho", "prediction", "target", weights_key="weights",
                          center_axis=-2, residual_scale=2., weight=3.)
        batch = dict(target=target, weights=weights)
        actual = loss.evaluate(dict(prediction=prediction), batch)["loss"]
        residual = prediction-target
        residual = residual-(residual*weights).sum(-2, keepdim=True)/weights.sum(-2, keepdim=True)
        expected = 3*((2*residual).square()*weights).sum()/weights.sum()
        torch.testing.assert_close(actual, expected)
        shifted = prediction + torch.tensor([[[5.]], [[-7.]]])
        torch.testing.assert_close(loss.evaluate(dict(prediction=shifted), batch)["loss"], actual)
        actual.backward()
        self.assertEqual(prediction.grad[:, 2].abs().max(), 0)
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_metrics_use_same_canonical_projection(self):
        target = torch.tensor([[1.], [2.], [4.]])
        predicted = target + 17.
        metrics = Metrics("target", prediction_key="prediction", center_axis=-2)
        values = metrics(dict(prediction=predicted), dict(target=target))
        self.assertLess(values["rmse"], 1e-6)
        self.assertAlmostEqual(float(values["r2"]), 1.)

    def test_vector_scale_is_inside_square(self):
        p = torch.ones(2, 2, 3)
        target = torch.zeros_like(p)
        moments = torch.tensor([[0.5], [2.]])
        loss = TensorLoss("P", "p", "target", residual_scale=moments)
        torch.testing.assert_close(loss(dict(p=p), dict(target=target)),
                                   (p*moments).square().mean())

    def test_noise_requires_one_model_reference(self):
        rho = torch.full((4, 1), .4, dtype=torch.float64)
        data = dict(rho=rho, dipole_density=torch.zeros(4, 1, 3, dtype=torch.float64),
                    dipole_density_std=torch.full((4, 1, 3), .001, dtype=torch.float64))
        ideal = FixedDipoleIdeal(.7)
        noisy = NoiseOptions(polarization=True).apply(data, ideal=ideal)
        self.assertFalse(torch.equal(noisy["dipole_density"], data["dipole_density"]))
        self.assertTrue((noisy["dipole_density"].norm(dim=-1) < .7*rho).all())
        with self.assertRaisesRegex(ValueError, "FixedDipoleIdeal"):
            NoiseOptions(polarization=True).apply(data)
        with self.assertRaisesRegex(TypeError, "dipole_magnitude"):
            NoiseOptions(polarization=True, dipole_magnitude=1.)



if __name__ == "__main__":
    unittest.main()
