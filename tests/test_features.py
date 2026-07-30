import math
import unittest

import torch
from torch import nn

from cace_grid import AChannelMixing, CartesianAFeatures, CartesianBFeatures


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

        rho = torch.arange(
            1,
            8,
            dtype=module.monomial_values.dtype,
        ).reshape(7, 1)
        local_density_index = torch.arange(7).repeat(7, 1)
        features = module(
            {
                "rho": rho,
                "local_density_index": local_density_index,
                "grid_spacing": torch.tensor(
                    [0.5, 0.5, 0.5],
                    dtype=rho.dtype,
                ),
            }
        )

        self.assertEqual(features.shape, (7, 4, 10, 1))
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
        rho = torch.tensor(
            [[[2.0]], [[3.0]]],
            dtype=module.monomial_values.dtype,
        )
        local_density_index = torch.zeros((2, 1, 1), dtype=torch.long)
        features = module(
            {
                "rho": rho,
                "local_density_index": local_density_index,
                "grid_spacing": torch.tensor(
                    [[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]],
                    dtype=rho.dtype,
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

        rho = torch.arange(
            1,
            8,
            dtype=module.monomial_values.dtype,
        ).reshape(7, 1)
        local_density_index = torch.arange(7).repeat(7, 1)
        features = module(
            {
                "rho": rho,
                "local_density_index": local_density_index,
                "grid_spacing": torch.ones(3, dtype=rho.dtype),
            }
        )
        features.sum().backward()

        self.assertIsNotNone(module.log_alphas.grad)
        self.assertTrue(torch.all(torch.isfinite(module.log_alphas.grad)))


class TestAChannelMixing(unittest.TestCase):
    def test_known_linear_map_and_shape(self):
        module = AChannelMixing(n_types=2, n_channels=3)
        with torch.no_grad():
            module.weight.copy_(
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [2.0, -1.0]],
                    dtype=module.weight.dtype,
                )
            )

        A = torch.tensor(
            [[[[1.0, 3.0], [2.0, 5.0]]]],
            dtype=module.weight.dtype,
        )
        mixed = module(A)

        self.assertEqual(mixed.shape, (1, 1, 2, 3))
        self.assertTrue(
            torch.allclose(
                mixed,
                torch.tensor(
                    [[[[1.0, 3.0, -1.0], [2.0, 5.0, -1.0]]]],
                    dtype=mixed.dtype,
                ),
            )
        )

    def test_mixing_commutes_with_cubic_symmetry(self):
        mixer = AChannelMixing(n_types=2, n_channels=4)
        symmetrizer = CartesianBFeatures(max_power=2, max_nu=3)
        A = torch.randn(2, 3, 10, 2, dtype=mixer.weight.dtype)

        mixed = mixer(A)
        reference = symmetrizer(mixed)
        for index_map, sign_map in zip(
            symmetrizer.component_indices,
            symmetrizer.component_signs,
        ):
            sign_shape = [1] * A.ndim
            sign_shape[-2] = sign_map.shape[0]
            transformed = (
                A.index_select(-2, index_map) * sign_map.view(sign_shape)
            )

            # Mixing physical channels commutes with the signed permutation of
            # Cartesian components, and the subsequent B features are invariant.
            transformed_mixed = mixer(transformed)
            expected_mixed = (
                mixed.index_select(-2, index_map) * sign_map.view(sign_shape)
            )
            self.assertTrue(
                torch.allclose(transformed_mixed, expected_mixed, atol=1.0e-6)
            )
            self.assertTrue(
                torch.allclose(
                    symmetrizer(transformed_mixed),
                    reference,
                    atol=1.0e-6,
                )
            )

    def test_inputs_and_weights_receive_gradients(self):
        module = AChannelMixing(n_types=2, n_channels=3)
        A = torch.randn(
            2,
            4,
            10,
            2,
            dtype=module.weight.dtype,
            requires_grad=True,
        )
        module(A).square().sum().backward()

        self.assertIsNotNone(A.grad)
        self.assertIsNotNone(module.weight.grad)
        self.assertTrue(torch.all(torch.isfinite(A.grad)))
        self.assertTrue(torch.all(torch.isfinite(module.weight.grad)))


if __name__ == "__main__":
    unittest.main()
