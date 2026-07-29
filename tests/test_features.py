import math
import unittest

import torch
from torch import nn

from cace_grid import CartesianAFeatures


class TestCartesianAFeatures(unittest.TestCase):
    def test_fixed_monomials_and_density_contraction(self):
        module = CartesianAFeatures(
            cutoff_grid=1,
            max_power=2,
            n_alphas=4,
            trainable_alphas=False,
        )

        expected_powers = torch.tensor(
            [
                (0, 0, 0),
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
                (2, 0, 0),
                (1, 1, 0),
                (1, 0, 1),
                (0, 2, 0),
                (0, 1, 1),
                (0, 0, 2),
            ],
            dtype=torch.long,
        )
        self.assertTrue(torch.equal(module.powers, expected_powers))
        self.assertEqual(module.monomial_values.shape, (7, 10))
        self.assertTrue(
            torch.allclose(
                module.alphas,
                torch.tensor(
                    [0.25, 0.5, 1.0, 2.0],
                    dtype=module.alphas.dtype,
                ),
            )
        )
        self.assertNotIsInstance(module.log_alphas, nn.Parameter)
        self.assertTrue(
            torch.equal(
                module.monomial_values[0],
                torch.tensor(
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    dtype=module.monomial_values.dtype,
                ),
            )
        )

        local_density = torch.arange(
            1,
            8,
            dtype=module.monomial_values.dtype,
        ).reshape(1, 7, 1)
        features = module(
            {
                "local_density": local_density,
                "grid_spacing": torch.tensor(
                    [0.5, 0.5, 0.5],
                    dtype=local_density.dtype,
                ),
            }
        )

        self.assertEqual(features.shape, (1, 4, 10, 1))
        for radial_index, alpha in enumerate(module.alphas.tolist()):
            radial_weight = math.exp(-alpha)
            expected_scalar = 0.125 * (1.0 + 27.0 * radial_weight)
            expected_vector = -0.125 * radial_weight
            self.assertAlmostEqual(
                features[0, radial_index, 0, 0].item(),
                expected_scalar,
                places=6,
            )
            self.assertAlmostEqual(
                features[0, radial_index, 1, 0].item(),
                expected_vector,
                places=6,
            )
            self.assertAlmostEqual(
                features[0, radial_index, 2, 0].item(),
                expected_vector,
                places=6,
            )
            self.assertAlmostEqual(
                features[0, radial_index, 3, 0].item(),
                expected_vector,
                places=6,
            )

    def test_batched_data(self):
        module = CartesianAFeatures(
            cutoff_grid=0,
            max_power=0,
            n_alphas=3,
        )
        local_density = torch.tensor(
            [[[[2.0]]], [[[3.0]]]],
            dtype=module.monomial_values.dtype,
        )
        features = module(
            {
                "local_density": local_density,
                "grid_spacing": torch.tensor(
                    [[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]],
                    dtype=local_density.dtype,
                ),
            }
        )

        self.assertEqual(features.shape, (2, 1, 3, 1, 1))
        self.assertTrue(
            torch.allclose(
                features[:, 0, :, 0, 0],
                torch.tensor(
                    [[0.25, 0.25, 0.25], [3.0, 3.0, 3.0]],
                    dtype=features.dtype,
                ),
            )
        )

    def test_trainable_alphas_receive_gradients(self):
        module = CartesianAFeatures(
            cutoff_grid=1,
            max_power=1,
            n_alphas=2,
            trainable_alphas=True,
        )
        self.assertIsInstance(module.log_alphas, nn.Parameter)

        local_density = torch.arange(
            1,
            8,
            dtype=module.monomial_values.dtype,
        ).reshape(1, 7, 1)
        features = module(
            {
                "local_density": local_density,
                "grid_spacing": torch.ones(3, dtype=local_density.dtype),
            }
        )
        features.sum().backward()

        self.assertIsNotNone(module.log_alphas.grad)
        self.assertTrue(torch.all(torch.isfinite(module.log_alphas.grad)))


if __name__ == "__main__":
    unittest.main()
