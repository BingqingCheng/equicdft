import math
import unittest

import torch
from torch import nn

from equicdft import CartesianAFeatures, CartesianBFeatures


class TestCartesianAFeatures(unittest.TestCase):
    def test_fixed_monomials_and_density_contraction(self):
        module = CartesianAFeatures(
            mean_density=2.0,
            cutoff_grid=1,
            max_power=2,
            n_radial_channels=4,
            trainable_radial_exponents=False,
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
                module.radial_exponents,
                torch.tensor(
                    [0.5, 1.0, 2.0, 4.0],
                    dtype=module.radial_exponents.dtype,
                ),
            )
        )
        self.assertNotIsInstance(module.log_radial_exponents, nn.Parameter)
        self.assertEqual(module.mean_density.item(), 2.0)
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
            }
        )

        self.assertEqual(features.shape, (7, 4, 10, 1))
        for radial_index, alpha in enumerate(
            module.radial_exponents.tolist()
        ):
            radial_weight = math.exp(-alpha)
            expected_scalar = (1.0 + 27.0 * radial_weight) / 2.0
            expected_vector = -radial_weight / 2.0
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
            mean_density=2.0,
            cutoff_grid=0,
            max_power=0,
            n_radial_channels=3,
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
            }
        )

        self.assertEqual(features.shape, (2, 1, 3, 1, 1))
        self.assertTrue(
            torch.allclose(
                features[:, 0, :, 0, 0],
                torch.tensor(
                    [[1.0, 1.0, 1.0], [1.5, 1.5, 1.5]],
                    dtype=features.dtype,
                ),
            )
        )

    def test_raw_gaussians_and_raw_integer_monomials(self):
        module = CartesianAFeatures(
            mean_density=2.0,
            cutoff_grid=2,
            max_power=2,
            n_radial_channels=2,
        )

        position_lookup = {
            tuple(position): index
            for index, position in enumerate(
                module.local_density_positions.tolist()
            )
        }
        plus_two_z = position_lookup[(0, 0, 2)]
        self.assertEqual(module.monomial_values[plus_two_z, 3].item(), 2.0)
        self.assertEqual(module.monomial_values[plus_two_z, 9].item(), 4.0)

        n_neighbors = module.local_density_positions.shape[0]
        rho = torch.full(
            (n_neighbors, 1),
            2.0,
            dtype=module.monomial_values.dtype,
        )
        local_density_index = torch.arange(n_neighbors).repeat(
            n_neighbors,
            1,
        )
        features = module(
            {
                "rho": rho,
                "local_density_index": local_density_index,
            }
        )
        # For a uniform field equal to mean_density, A_000 is the unnormalized
        # discrete mass of its Gaussian channel.
        expected_mass = torch.exp(
            -module.squared_distances[:, None]
            * module.radial_exponents[None, :]
        ).sum(dim=0)
        self.assertTrue(
            torch.allclose(
                features[:, :, 0, 0],
                expected_mass.expand(n_neighbors, -1),
            )
        )

    def test_mean_density_must_be_positive_scalar(self):
        with self.assertRaises(ValueError):
            CartesianAFeatures(max_power=0, mean_density=0.0)
        with self.assertRaises(ValueError):
            CartesianAFeatures(max_power=0, mean_density=[1.0, 2.0])

    def test_trainable_radial_exponents_receive_gradients(self):
        module = CartesianAFeatures(
            mean_density=1.0,
            cutoff_grid=1,
            max_power=1,
            n_radial_channels=2,
            radial_exponents=(0.05, 0.2),
            trainable_radial_exponents=True,
        )
        self.assertIsInstance(module.log_radial_exponents, nn.Parameter)
        self.assertTrue(
            torch.allclose(
                module.radial_exponents.detach(),
                torch.tensor(
                    [0.05, 0.2], dtype=module.radial_exponents.dtype
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
            }
        )
        features.sum().backward()

        self.assertIsNotNone(module.log_radial_exponents.grad)
        self.assertTrue(
            torch.all(torch.isfinite(module.log_radial_exponents.grad))
        )

    def test_explicit_radial_exponents_are_validated(self):
        with self.assertRaises(ValueError):
            CartesianAFeatures(
                max_power=0,
                mean_density=1.0,
                n_radial_channels=2,
                radial_exponents=(0.1,),
            )
        with self.assertRaises(ValueError):
            CartesianAFeatures(
                max_power=0,
                mean_density=1.0,
                n_radial_channels=2,
                radial_exponents=(0.1, 0.0),
            )
class TestCartesianAFeatureChannelMixing(unittest.TestCase):
    def test_known_linear_map_and_shape(self):
        module = CartesianAFeatures(
            mean_density=1.0,
            cutoff_grid=0,
            max_power=0,
            n_radial_channels=1,
            n_types=2,
            n_channels=3,
        )
        with torch.no_grad():
            module.channel_mixing.weight.copy_(
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [2.0, -1.0]],
                    dtype=module.monomial_values.dtype,
                )
            )

        rho = torch.tensor(
            [[1.0, 3.0]],
            dtype=module.monomial_values.dtype,
        )
        mixed = module(
            {
                "rho": rho,
                "local_density_index": torch.zeros(
                    (1, 1),
                    dtype=torch.long,
                ),
            }
        )

        self.assertEqual(mixed.shape, (1, 1, 1, 3))
        self.assertTrue(
            torch.allclose(
                mixed,
                torch.tensor(
                    [[[[1.0, 3.0, -1.0]]]],
                    dtype=mixed.dtype,
                ),
            )
        )
        self.assertEqual(module.n_output_channels, 3)

    def test_mixing_commutes_with_cubic_symmetry(self):
        module = CartesianAFeatures(
            mean_density=1.0,
            cutoff_grid=0,
            max_power=2,
            n_radial_channels=1,
            n_types=2,
            n_channels=4,
        )
        symmetrizer = CartesianBFeatures(
            max_power=2,
            max_product_order=3,
        )
        A = torch.randn(
            2,
            3,
            10,
            2,
            dtype=module.monomial_values.dtype,
        )

        mixed = module.channel_mixing(A)
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
            transformed_mixed = module.channel_mixing(transformed)
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
        module = CartesianAFeatures(
            mean_density=1.0,
            cutoff_grid=0,
            max_power=2,
            n_radial_channels=1,
            n_types=2,
            n_channels=3,
        )
        A = torch.randn(
            2,
            4,
            10,
            2,
            dtype=module.monomial_values.dtype,
            requires_grad=True,
        )
        module.channel_mixing(A).square().sum().backward()

        self.assertIsNotNone(A.grad)
        self.assertIsNotNone(module.channel_mixing.weight.grad)
        self.assertTrue(torch.all(torch.isfinite(A.grad)))
        self.assertTrue(
            torch.all(torch.isfinite(module.channel_mixing.weight.grad))
        )

    def test_one_component_disables_channel_mixing(self):
        module = CartesianAFeatures(
            mean_density=1.0,
            cutoff_grid=0,
            max_power=0,
            n_types=1,
        )
        self.assertIsNone(module.channel_mixing)
        self.assertEqual(module.n_output_channels, 1)

        with self.assertRaises(ValueError):
            CartesianAFeatures(
                mean_density=1.0,
                cutoff_grid=0,
                max_power=0,
                n_types=1,
                n_channels=2,
            )


if __name__ == "__main__":
    unittest.main()
