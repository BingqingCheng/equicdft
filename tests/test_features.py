import math
import unittest

import numpy as np
import torch
from torch import nn

from equicdft import (
    CartesianAFeatures,
    CartesianBFeatures,
    get_neighbor_indices,
)


def _gather_reference(module, rho, grid_size):
    """Evaluate the former gather/einsum definition for regression tests."""

    positions = np.indices(grid_size, dtype=np.int64).reshape(3, -1).T
    neighbor_indices, _ = get_neighbor_indices(
        positions,
        cutoff_grid=module.cutoff_grid,
    )
    local_density = rho[torch.as_tensor(neighbor_indices)]
    radial_values = torch.exp(
        -module.squared_distances[:, None]
        * module.radial_exponents[None, :]
    )
    basis_values = (
        radial_values[:, :, None] * module.monomial_values[:, None, :]
    )
    return torch.einsum(
        "gjt,jnk->gnkt",
        local_density / module.mean_density,
        basis_values,
    )


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

        grid_size = (3, 3, 3)
        rho = torch.arange(
            1,
            math.prod(grid_size) + 1,
            dtype=module.monomial_values.dtype,
        ).reshape(-1, 1)
        features = module(
            {
                "rho": rho,
                "grid_size": torch.tensor(grid_size),
            }
        )

        reference = _gather_reference(module, rho, grid_size)
        self.assertEqual(features.shape, (27, 4, 10, 1))
        self.assertTrue(torch.allclose(features, reference, atol=1.0e-6))

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
        features = module(
            {
                "rho": rho,
                "grid_size": torch.ones((2, 3), dtype=torch.long),
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

        grid_size = (3, 4, 5)
        n_grid = math.prod(grid_size)
        rho = torch.full(
            (n_grid, 1),
            2.0,
            dtype=module.monomial_values.dtype,
        )
        features = module(
            {
                "rho": rho,
                "grid_size": torch.tensor(grid_size),
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
                expected_mass.expand(n_grid, -1),
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
            trainable_radial_exponents=True,
        )
        self.assertIsInstance(module.log_radial_exponents, nn.Parameter)

        grid_size = (3, 3, 3)
        rho = torch.arange(
            1,
            math.prod(grid_size) + 1,
            dtype=module.monomial_values.dtype,
        ).reshape(-1, 1)
        features = module(
            {
                "rho": rho,
                "grid_size": torch.tensor(grid_size),
            }
        )
        features.sum().backward()

        self.assertIsNotNone(module.log_radial_exponents.grad)
        self.assertTrue(
            torch.all(torch.isfinite(module.log_radial_exponents.grad))
        )

    def test_fixed_kernel_is_cached_but_trainable_kernel_is_live(self):
        fixed = CartesianAFeatures(
            mean_density=1.0,
            cutoff_grid=2,
            max_power=2,
            n_radial_channels=2,
        )
        self.assertIs(
            fixed._convolution_kernel(),
            fixed.fixed_convolution_kernel,
        )

        trainable = CartesianAFeatures(
            mean_density=1.0,
            cutoff_grid=2,
            max_power=2,
            n_radial_channels=2,
            trainable_radial_exponents=True,
        )
        before = trainable._convolution_kernel().detach().clone()
        with torch.no_grad():
            trainable.log_radial_exponents.add_(0.2)
        after = trainable._convolution_kernel().detach()
        self.assertIsNone(trainable.fixed_convolution_kernel)
        self.assertFalse(torch.equal(before, after))

    def test_matches_gather_values_and_density_gradients(self):
        cases = (
            ((4, 5, 6), 2),
            # cutoff > a box dimension exercises repeated periodic images.
            ((2, 3, 2), 3),
        )
        for grid_size, cutoff in cases:
            with self.subTest(grid_size=grid_size, cutoff=cutoff):
                torch.manual_seed(7)
                module = CartesianAFeatures(
                    mean_density=0.73,
                    cutoff_grid=cutoff,
                    max_power=3,
                    n_radial_channels=3,
                    n_types=2,
                ).double()
                rho = torch.randn(
                    math.prod(grid_size),
                    2,
                    dtype=torch.float64,
                    requires_grad=True,
                )
                actual = module(
                    {
                        "rho": rho,
                        "grid_size": torch.tensor(grid_size),
                    }
                )
                reference = _gather_reference(module, rho, grid_size)
                self.assertTrue(
                    torch.allclose(actual, reference, atol=1.0e-12, rtol=1.0e-12)
                )

                weights = torch.randn_like(actual)
                actual_gradient = torch.autograd.grad(
                    torch.sum(actual * weights),
                    rho,
                    retain_graph=True,
                )[0]
                reference_gradient = torch.autograd.grad(
                    torch.sum(reference * weights),
                    rho,
                )[0]
                self.assertTrue(
                    torch.allclose(
                        actual_gradient,
                        reference_gradient,
                        atol=1.0e-11,
                        rtol=1.0e-11,
                    )
                )

    def test_periodic_boundary_preserves_odd_monomial_sign(self):
        grid_size = (5, 4, 3)
        module = CartesianAFeatures(
            mean_density=1.0,
            cutoff_grid=1,
            max_power=1,
            n_radial_channels=1,
        )
        rho = torch.zeros(math.prod(grid_size), 1)
        plus_x = np.ravel_multi_index((1, 0, 0), grid_size, order="C")
        minus_x = np.ravel_multi_index((4, 0, 0), grid_size, order="C")

        rho[plus_x] = 1.0
        plus_feature = module(
            {"rho": rho, "grid_size": torch.tensor(grid_size)}
        )[0, 0, 1, 0]
        rho.zero_()
        rho[minus_x] = 1.0
        minus_feature = module(
            {"rho": rho, "grid_size": torch.tensor(grid_size)}
        )[0, 0, 1, 0]

        expected = math.exp(-module.radial_exponents[0].item())
        self.assertAlmostEqual(plus_feature.item(), expected, places=6)
        self.assertAlmostEqual(minus_feature.item(), -expected, places=6)

    def test_rejects_mixed_grid_sizes_inside_one_dense_batch(self):
        module = CartesianAFeatures(
            mean_density=1.0,
            cutoff_grid=0,
            max_power=0,
        )
        with self.assertRaisesRegex(ValueError, "share grid_size"):
            module(
                {
                    "rho": torch.ones(2, 8, 1),
                    "grid_size": torch.tensor([[2, 2, 2], [1, 2, 4]]),
                }
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
                "grid_size": torch.ones(3, dtype=torch.long),
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
