import unittest

import numpy as np
import torch
from torch import nn

from cace_grid import (
    CartesianAFeatures,
    CartesianBFeatures,
    GridCACEModel,
    LocalFreeEnergyReadout,
    get_neighbor_indices,
)


class _DensityFeatures(nn.Module):
    """Expose rho directly for an analytic model-integration test."""

    def forward(self, data):
        return data["rho"]


class _IdentityModule(nn.Module):
    def forward(self, values):
        return values


class TestGridCACEModel(unittest.TestCase):
    def _make_data(self):
        shape = (3, 3, 3)
        grid_positions = np.indices(shape, dtype=int).reshape(3, -1).T
        neighbor_indices, _ = get_neighbor_indices(
            grid_positions,
            cutoff_grid=1,
        )
        return {
            "rho": torch.linspace(0.2, 0.8, steps=27).reshape(27, 1),
            "local_density_index": torch.tensor(
                neighbor_indices,
                dtype=torch.long,
            ),
            "grid_spacing": torch.tensor([0.5, 0.5, 0.5]),
        }

    def _make_model(self, compute_c1=True):
        a_features = CartesianAFeatures(
            mean_density=0.5,
            cutoff_grid=1,
            max_power=2,
            n_radial_channels=1,
        )
        b_features = CartesianBFeatures(max_power=2, max_product_order=3)
        readout = LocalFreeEnergyReadout(
            n_types=1,
            hidden_sizes=(8,),
        )
        return GridCACEModel(
            a_features,
            b_features,
            readout,
            compute_c1=compute_c1,
        )

    def test_collects_outputs_and_initializes_density_gradient(self):
        model = self._make_model()
        data = self._make_data()

        outputs = model(data)

        self.assertEqual(list(outputs), model.model_outputs)
        self.assertTrue(data["rho"].requires_grad)
        self.assertEqual(
            outputs["beta_free_energy_per_particle"].shape,
            (27, 1),
        )
        self.assertEqual(outputs["beta_free_energy_density"].shape, (27,))
        self.assertEqual(outputs["beta_F_exc"].shape, ())
        self.assertEqual(outputs["c1"].shape, (27, 1))

    def test_model_applies_c1_sign_and_cell_volume(self):
        # With per-particle free energy equal to rho,
        # beta_F_exc = Delta V * sum_g rho_g^2. Its discrete derivative is
        # 2*Delta V*rho, while the continuum c1 must be -2*rho.
        model = GridCACEModel(
            a_features=_DensityFeatures(),
            b_features=_IdentityModule(),
            readout=_IdentityModule(),
            compute_c1=True,
        )
        data = self._make_data()
        data = {
            key: torch.stack((value, value.clone()))
            for key, value in data.items()
        }

        outputs = model(data)

        self.assertEqual(outputs["beta_F_exc"].shape, (2,))
        self.assertTrue(torch.allclose(outputs["c1"], -2.0 * data["rho"]))

    def test_empty_density_has_zero_integrated_free_energy(self):
        model = self._make_model()
        data = self._make_data()
        data["rho"].zero_()

        outputs = model(data)

        self.assertTrue(
            torch.all(torch.isfinite(outputs["beta_free_energy_per_particle"]))
        )
        self.assertTrue(
            torch.equal(
                outputs["beta_free_energy_density"],
                torch.zeros_like(outputs["beta_free_energy_density"]),
            )
        )
        self.assertEqual(outputs["beta_F_exc"].item(), 0.0)

    def test_training_c1_loss_reaches_readout_parameters(self):
        model = self._make_model()
        model.train()

        outputs = model(self._make_data())
        outputs["c1"].square().mean().backward()

        gradients = [parameter.grad for parameter in model.readout.parameters()]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(
            all(torch.all(torch.isfinite(gradient)) for gradient in gradients)
        )

    def test_evaluation_avoids_higher_order_graph(self):
        model = self._make_model()
        model.eval()

        with torch.no_grad():
            outputs = model(self._make_data())

        self.assertFalse(outputs["c1"].requires_grad)

    def test_c1_is_optional(self):
        model = self._make_model(compute_c1=False)
        model.eval()
        data = self._make_data()

        with torch.no_grad():
            outputs = model(data)

        self.assertEqual(
            list(outputs),
            [
                "beta_free_energy_per_particle",
                "beta_free_energy_density",
                "beta_F_exc",
            ],
        )
        self.assertEqual(model.required_derivatives, [])
        self.assertFalse(data["rho"].requires_grad)
        self.assertFalse(outputs["beta_F_exc"].requires_grad)

    def test_compute_c1_must_be_boolean(self):
        with self.assertRaises(TypeError):
            GridCACEModel(
                CartesianAFeatures(
                    mean_density=0.5,
                    cutoff_grid=1,
                    max_power=2,
                    n_radial_channels=1,
                ),
                CartesianBFeatures(max_power=2, max_product_order=3),
                LocalFreeEnergyReadout(n_types=1),
                compute_c1=1,
            )

    def test_multicomponent_latent_channels_feed_physical_readout(self):
        data = self._make_data()
        data["rho"] = torch.cat(
            (data["rho"], 0.5 * data["rho"]),
            dim=-1,
        )
        a_features = CartesianAFeatures(
            mean_density=0.5,
            cutoff_grid=1,
            max_power=2,
            n_radial_channels=1,
            n_types=2,
            n_channels=3,
        )
        b_features = CartesianBFeatures(max_power=2, max_product_order=3)
        readout = LocalFreeEnergyReadout(
            n_types=2,
            hidden_sizes=(8,),
        )
        model = GridCACEModel(a_features, b_features, readout)

        outputs = model(data)

        self.assertEqual(
            outputs["beta_free_energy_per_particle"].shape,
            (27, 2),
        )
        self.assertEqual(outputs["c1"].shape, (27, 2))
        outputs["c1"].square().mean().backward()
        mixing_gradient = model.a_features.channel_mixing.weight.grad
        self.assertIsNotNone(mixing_gradient)
        self.assertTrue(torch.all(torch.isfinite(mixing_gradient)))


if __name__ == "__main__":
    unittest.main()
