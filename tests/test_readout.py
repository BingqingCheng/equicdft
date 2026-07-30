import unittest

import numpy as np
import torch

from cace_grid import (
    CartesianAFeatures,
    CartesianBFeatures,
    LocalFreeEnergyReadout,
    compute_c1,
    compute_rms_feature_scale,
    get_neighbor_indices,
)


class TestLocalFreeEnergyReadout(unittest.TestCase):
    def _full_pipeline(self, rho):
        shape = (4, 4, 4)
        grid_positions = np.indices(shape, dtype=int).reshape(3, -1).T
        neighbor_indices, _ = get_neighbor_indices(
            grid_positions,
            cutoff_grid=1,
        )
        data = {
            "rho": rho,
            "local_density_index": torch.tensor(
                neighbor_indices,
                dtype=torch.long,
            ),
            "grid_spacing": torch.tensor(
                [0.5, 0.5, 0.5],
                dtype=rho.dtype,
            ),
        }

        A = CartesianAFeatures(
            cutoff_grid=1,
            max_power=2,
            n_alphas=1,
        )(data)
        B_module = CartesianBFeatures(max_power=2, max_nu=3)
        B = B_module(A)
        feature_scale = compute_rms_feature_scale(B)
        readout = LocalFreeEnergyReadout(
            n_features=B.shape[-3] * B.shape[-2] * B.shape[-1],
            n_types=1,
            hidden_sizes=(16, 8),
            feature_scale=feature_scale,
        )
        return data, A, B, readout, readout(B, data)

    def test_vacuum_anchor(self):
        rho = torch.zeros(
            64,
            1,
            dtype=torch.get_default_dtype(),
            requires_grad=True,
        )
        data, _, _, _, output = self._full_pipeline(rho)

        self.assertTrue(
            torch.equal(
                output["beta_free_energy_per_particle"],
                torch.zeros_like(output["beta_free_energy_per_particle"]),
            )
        )
        self.assertTrue(
            torch.equal(
                output["beta_free_energy_density"],
                torch.zeros_like(output["beta_free_energy_density"]),
            )
        )
        self.assertEqual(output["beta_F_exc"].item(), 0.0)

        # rho * anchored a_exc has zero first derivative at vacuum, not merely
        # zero total free energy.
        c1 = compute_c1(
            output["beta_F_exc"],
            data["rho"],
            data["grid_spacing"],
        )
        self.assertTrue(torch.equal(c1, torch.zeros_like(c1)))

    def test_batched_temperature_and_shapes(self):
        B = torch.randn(2, 5, 1, 23, 1)
        rho = torch.rand(2, 5, 1)
        data = {
            "rho": rho,
            "temperature": torch.tensor([1.5, 2.0], dtype=rho.dtype),
            "grid_spacing": torch.tensor(
                [[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]],
                dtype=rho.dtype,
            ),
        }
        readout = LocalFreeEnergyReadout(
            n_features=23,
            n_types=1,
            include_temperature=True,
        )
        output = readout(B, data)

        self.assertEqual(
            output["beta_free_energy_per_particle"].shape,
            (2, 5, 1),
        )
        self.assertEqual(
            output["beta_free_energy_density"].shape,
            (2, 5),
        )
        self.assertEqual(output["beta_F_exc"].shape, (2,))

    def test_fixed_rms_feature_scaling(self):
        B_training = torch.arange(1.0, 25.0).reshape(2, 3, 1, 2, 2)
        B_flat = B_training.flatten(start_dim=-3)
        expected_scale = torch.sqrt(torch.mean(B_flat.square(), dim=(0, 1)))
        scale = compute_rms_feature_scale(B_training)
        self.assertTrue(torch.allclose(scale, expected_scale))
        self.assertFalse(scale.requires_grad)

        readout = LocalFreeEnergyReadout(
            n_features=2,
            n_types=1,
            hidden_sizes=(2,),
            feature_scale=torch.tensor([2.0, 4.0]),
        )
        with torch.no_grad():
            for parameter in readout.nonlinear.parameters():
                parameter.zero_()
            readout.linear.weight.copy_(torch.tensor([[1.0, 1.0, 0.0]]))

        B = torch.tensor([[[[2.0], [4.0]]]])
        data = {
            "rho": torch.ones(1, 1),
            "grid_spacing": torch.ones(3),
        }
        output = readout(B, data)
        self.assertEqual(
            output["beta_free_energy_per_particle"].item(),
            2.0,
        )
        self.assertEqual(output["beta_F_exc"].item(), 2.0)

    def test_full_autograd_and_finite_difference(self):
        torch.manual_seed(7)
        rho = torch.linspace(
            0.2,
            0.8,
            steps=64,
            dtype=torch.get_default_dtype(),
        ).reshape(64, 1)
        rho.requires_grad_(True)
        data, A, B, readout, output = self._full_pipeline(rho)

        self.assertEqual(A.shape, (64, 1, 10, 1))
        self.assertEqual(B.shape, (64, 1, 23, 1))
        self.assertEqual(
            output["beta_free_energy_per_particle"].shape,
            (64, 1),
        )
        self.assertEqual(output["beta_free_energy_density"].shape, (64,))
        self.assertEqual(output["beta_F_exc"].shape, ())

        c1 = compute_c1(
            output["beta_F_exc"],
            rho,
            data["grid_spacing"],
            create_graph=True,
        )
        self.assertEqual(c1.shape, rho.shape)
        self.assertTrue(torch.all(torch.isfinite(c1)))

        # Compare the continuum-normalized c1 with a numerical derivative of
        # the complete rho -> A -> B -> F_exc pipeline.
        selected_index = 9
        # A 1e-2 central step is in the converged regime for the default
        # float32 test dtype; much smaller steps suffer subtractive cancellation.
        epsilon = 1.0e-2

        def evaluate(perturbed_rho):
            perturbed_data = dict(data)
            perturbed_data["rho"] = perturbed_rho
            perturbed_A = CartesianAFeatures(
                cutoff_grid=1,
                max_power=2,
                n_alphas=1,
            )(perturbed_data)
            perturbed_B = CartesianBFeatures(
                max_power=2,
                max_nu=3,
            )(perturbed_A)
            return readout(perturbed_B, perturbed_data)["beta_F_exc"]

        rho_plus = rho.detach().clone()
        rho_minus = rho.detach().clone()
        rho_plus[selected_index, 0] += epsilon
        rho_minus[selected_index, 0] -= epsilon
        finite_difference = (
            evaluate(rho_plus) - evaluate(rho_minus)
        ) / (2.0 * epsilon)
        cell_volume = torch.prod(data["grid_spacing"])
        expected_derivative = -cell_volume * c1[selected_index, 0].detach()
        self.assertTrue(
            torch.allclose(
                finite_difference,
                expected_derivative,
                atol=5.0e-6,
                rtol=5.0e-4,
            )
        )

        # A c1 loss requires second derivatives and must reach every readout
        # parameter through torch.autograd.grad(create_graph=True).
        c1.square().mean().backward()
        gradients = [
            parameter.grad
            for parameter in readout.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(
            all(torch.all(torch.isfinite(gradient)) for gradient in gradients)
        )
        self.assertGreater(
            sum(torch.sum(torch.abs(gradient)).item() for gradient in gradients),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
