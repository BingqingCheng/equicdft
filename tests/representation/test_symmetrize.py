import unittest

import torch

from equicdft import CartesianBFeatures


class TestCartesianBFeatures(unittest.TestCase):
    def test_feature_counts_and_shape(self):
        module = CartesianBFeatures(max_power=2, max_product_order=3)
        A = torch.randn(2, 3, 4, 10, 2)
        B = module(A)

        self.assertEqual(module.n_features_by_order, (2, 6, 15))
        self.assertEqual(module.n_features, 23)
        self.assertEqual(B.shape, (2, 3, 4, 23, 2))
        self.assertTrue(
            torch.equal(
                module.correlation_orders,
                torch.tensor([1] * 2 + [2] * 6 + [3] * 15),
            )
        )

    def test_first_order_scalar_features(self):
        module = CartesianBFeatures(max_power=2, max_product_order=1)
        A = torch.arange(10, dtype=torch.get_default_dtype()).reshape(
            1, 1, 1, 10, 1
        )
        B = module(A)

        # The first invariant is A_000. The second is the cubic average of
        # A_200, A_020, and A_002, whose indices are 4, 7, and 9.
        self.assertEqual(B.shape, (1, 1, 1, 2, 1))
        self.assertAlmostEqual(
            B[0, 0, 0, 0, 0].item(),
            A[0, 0, 0, 0, 0].item(),
            places=6,
        )
        self.assertAlmostEqual(
            B[0, 0, 0, 1, 0].item(),
            (
                (
                    A[0, 0, 0, 4, 0]
                    + A[0, 0, 0, 7, 0]
                    + A[0, 0, 0, 9, 0]
                )
                / 3
            ).item(),
            places=6,
        )

    def test_invariant_under_all_signed_axis_permutations(self):
        module = CartesianBFeatures(max_power=2, max_product_order=3)
        A = torch.randn(2, 3, 10, 2)
        reference = module(A)

        for index_map, sign_map in zip(
            module.component_indices,
            module.component_signs,
        ):
            sign_shape = [1] * A.ndim
            sign_shape[-2] = sign_map.shape[0]
            transformed = A.index_select(-2, index_map) * sign_map.view(sign_shape)
            self.assertTrue(
                torch.allclose(module(transformed), reference, atol=1.0e-6)
            )

    def test_gradients_and_maximum_order(self):
        module = CartesianBFeatures(max_power=2, max_product_order=3)
        A = torch.randn(1, 2, 10, 1, requires_grad=True)
        loss = module(A).square().sum()
        loss.backward()

        self.assertIsNotNone(A.grad)
        self.assertTrue(torch.all(torch.isfinite(A.grad)))
        with self.assertRaises(ValueError):
            CartesianBFeatures(max_power=2, max_product_order=4)

    def test_pre_invariant_radial_mixing_creates_cross_terms(self):
        module = CartesianBFeatures(max_power=0, max_product_order=2)
        primitive_A = torch.tensor([[[[2.0]], [[3.0]]]])
        mixed_A = primitive_A.sum(dim=-3, keepdim=True)

        primitive_B = module(primitive_A)
        mixed_B = module(mixed_A)

        self.assertTrue(
            torch.equal(primitive_B.flatten(), torch.tensor([2.0, 4.0, 3.0, 9.0]))
        )
        self.assertTrue(
            torch.equal(mixed_B.flatten(), torch.tensor([5.0, 25.0]))
        )
        self.assertEqual(mixed_B.flatten()[1].item(), 4.0 + 12.0 + 9.0)


if __name__ == "__main__":
    unittest.main()
