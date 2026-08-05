import unittest

import numpy as np
import torch

from equicdft import (
    BulkReadout,
    CartesianAFeatures,
    CartesianBFeatures,
    LocalReadout,
)
from equicdft.derivatives import compute_grid_derivative
from equicdft.stencil import get_neighbor_indices


class TestBulkReadout(unittest.TestCase):
    def test_zero_initialization_gives_zero_per_particle_energy(self):
        readout = BulkReadout(n_types=2, hidden_sizes=(4,))
        state = torch.randn(3, 3)

        output = readout(state)

        self.assertEqual(output.shape, (3, 2))
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_linear_state_mapping(self):
        readout = BulkReadout(
            n_types=1,
            hidden_sizes=(),
            zero_init=False,
        )
        with torch.no_grad():
            readout.mlp[-1].weight.copy_(torch.tensor([[2.0, -1.0]]))
            readout.mlp[-1].bias.fill_(0.5)

        output = readout(torch.tensor([[1.5, 0.25], [0.5, 2.0]]))

        self.assertTrue(
            torch.equal(output, torch.tensor([[3.25], [-0.5]]))
        )

    def test_rejects_incompatible_state_width(self):
        readout = BulkReadout(n_types=2)

        with self.assertRaisesRegex(ValueError, "state_features"):
            readout(torch.ones(4, 2))


class TestLocalReadout(unittest.TestCase):
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
            mean_density=0.5,
            cutoff_grid=1,
            max_power=2,
            n_radial_channels=1,
        )(data)
        B_module = CartesianBFeatures(max_power=2, max_product_order=3)
        B = B_module(A)
        readout = LocalReadout(
            n_types=1,
            hidden_sizes=(16, 8),
        )
        local_features = B.flatten(start_dim=-3)
        return data, A, B, readout, readout(local_features)

    def test_batched_shapes(self):
        local_features = torch.randn(2, 5, 23)
        readout = LocalReadout(
            n_types=1,
        )
        output = readout(local_features)

        self.assertEqual(output.shape, (2, 5, 1))
        self.assertEqual(readout.mlp[0].in_features, 23)

    def test_readout_uses_local_features_directly(self):
        readout = LocalReadout(
            n_types=1,
            hidden_sizes=(),
            n_features=2,
        )
        local_features = torch.tensor([[2.0, 4.0]])
        with torch.no_grad():
            readout.mlp[0].weight.copy_(torch.tensor([[1.0, 1.0]]))
            readout.mlp[0].bias.zero_()

        output = readout(local_features)
        self.assertEqual(output.item(), 6.0)

    def test_full_autograd_and_finite_difference(self):
        torch.manual_seed(7)
        rho = torch.linspace(
            0.2,
            0.8,
            steps=64,
            dtype=torch.get_default_dtype(),
        ).reshape(64, 1)
        rho.requires_grad_(True)
        data, A, B, readout, beta_free_energy_per_particle = (
            self._full_pipeline(rho)
        )
        beta_free_energy_density = torch.sum(
            rho * beta_free_energy_per_particle,
            dim=-1,
        )
        beta_F_exc = torch.prod(data["grid_spacing"]) * torch.sum(
            beta_free_energy_density
        )

        self.assertEqual(A.shape, (64, 1, 10, 1))
        self.assertEqual(B.shape, (64, 1, 23, 1))
        self.assertEqual(
            beta_free_energy_per_particle.shape,
            (64, 1),
        )
        self.assertEqual(beta_free_energy_density.shape, (64,))
        self.assertEqual(beta_F_exc.shape, ())

        beta_F_exc_derivative = compute_grid_derivative(
            beta_F_exc,
            rho,
            create_graph=True,
        )
        self.assertEqual(beta_F_exc_derivative.shape, rho.shape)
        self.assertTrue(torch.all(torch.isfinite(beta_F_exc_derivative)))

        # Compare the raw grid derivative with a numerical derivative of the
        # complete rho -> A -> B -> F_exc pipeline.
        selected_index = 9
        # A 1e-2 central step is in the converged regime for the default
        # float32 test dtype; much smaller steps suffer subtractive cancellation.
        epsilon = 1.0e-2

        def evaluate(perturbed_rho):
            perturbed_data = dict(data)
            perturbed_data["rho"] = perturbed_rho
            perturbed_A = CartesianAFeatures(
                mean_density=0.5,
                cutoff_grid=1,
                max_power=2,
                n_radial_channels=1,
            )(perturbed_data)
            perturbed_B = CartesianBFeatures(
                max_power=2,
                max_product_order=3,
            )(perturbed_A)
            local_features = perturbed_B.flatten(start_dim=-3)
            per_particle = readout(local_features)
            density = torch.sum(perturbed_rho * per_particle, dim=-1)
            return torch.prod(perturbed_data["grid_spacing"]) * torch.sum(
                density
            )

        rho_plus = rho.detach().clone()
        rho_minus = rho.detach().clone()
        rho_plus[selected_index, 0] += epsilon
        rho_minus[selected_index, 0] -= epsilon
        finite_difference = (
            evaluate(rho_plus) - evaluate(rho_minus)
        ) / (2.0 * epsilon)
        self.assertTrue(
            torch.allclose(
                finite_difference,
                beta_F_exc_derivative[selected_index, 0].detach(),
                atol=5.0e-6,
                rtol=5.0e-4,
            )
        )

        # A loss on a response requires second derivatives and must reach every
        # readout parameter through torch.autograd.grad(create_graph=True).
        beta_F_exc_derivative.square().mean().backward()
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
