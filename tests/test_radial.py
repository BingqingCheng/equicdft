import math
import unittest

import torch

from equicdft._radial import (
    _RadialTransform,
    bessel_radial_values,
    gaussian_radial_values,
    whiten_radial_cartesian_basis,
)
from equicdft.features import _make_powers
from equicdft.stencil import make_stencil


class TestRadialValues(unittest.TestCase):
    def test_gaussian_formula_is_unchanged(self):
        squared_distances = torch.tensor([0.0] + [1.0] * 6)
        exponents = torch.tensor([0.5])
        centers = torch.zeros(1)
        mask = torch.ones(7, dtype=torch.bool)

        actual = gaussian_radial_values(
            squared_distances,
            exponents,
            centers,
            mask,
        )
        raw = torch.exp(-squared_distances[:, None] * exponents[None, :])
        expected = raw / raw.sum(dim=0, keepdim=True)

        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual.sum().item(), 1.0)

    def test_bessel_values_have_finite_center_and_zero_boundary(self):
        squared_distances = torch.arange(5, dtype=torch.float64)
        actual = bessel_radial_values(
            squared_distances,
            n_radial_functions=2,
            cutoff_grid=2,
            neighbor_mask=torch.ones(5, dtype=torch.bool),
        )
        expected = torch.tensor(
            [
                (math.pi / 2.0, math.pi),
                (1.0, 0.0),
                (0.562640058572400, -0.681582017381037),
                (0.235891598125591, -0.430607939476443),
                (0.0, 0.0),
            ],
            dtype=torch.float64,
        )

        self.assertTrue(
            torch.allclose(actual, expected, rtol=0.0, atol=1.0e-14)
        )
        self.assertTrue(torch.equal(actual[-1], torch.zeros(2)))

        masked = bessel_radial_values(
            squared_distances,
            n_radial_functions=2,
            cutoff_grid=2,
            neighbor_mask=torch.tensor([False, True, True, True, True]),
        )
        self.assertTrue(torch.equal(masked[0], torch.zeros(2)))


class TestRadialConditioning(unittest.TestCase):
    @staticmethod
    def _basis_inputs(cutoff_grid, n_radial_functions, max_power):
        positions = torch.from_numpy(make_stencil(cutoff_grid)).to(
            dtype=torch.float64
        )
        squared_distances = positions.square().sum(dim=1)
        powers = _make_powers(max_power)
        monomials = torch.ones(
            positions.shape[0],
            powers.shape[0],
            dtype=torch.float64,
        )
        for axis in range(3):
            monomials = monomials * positions[:, axis, None].pow(
                powers[None, :, axis]
            )
        support = (squared_distances > 0) & (
            squared_distances < cutoff_grid**2
        )
        radial = bessel_radial_values(
            squared_distances,
            n_radial_functions,
            cutoff_grid,
            support,
        )
        return radial, monomials, powers, support

    def test_degree_blocks_are_discretely_whitened(self):
        radial, monomials, powers, support = self._basis_inputs(2, 2, 3)
        basis, eigenvalues = whiten_radial_cartesian_basis(
            radial,
            monomials,
            powers,
            support,
        )

        n_active = int(support.sum())
        degrees = powers.sum(dim=-1)
        for degree in range(4):
            component_mask = degrees == degree
            samples = basis[support][:, :, component_mask]
            samples = samples.permute(0, 2, 1).reshape(-1, 2)
            gram = samples.transpose(0, 1).matmul(samples)
            gram = gram / float(component_mask.sum())
            expected = torch.eye(2, dtype=gram.dtype) / float(n_active)
            self.assertTrue(torch.allclose(gram, expected, atol=1.0e-12))

        self.assertEqual(eigenvalues.shape, (4, 2))
        self.assertTrue(torch.all(eigenvalues > 0.0).item())
        expected_eigenvalues = torch.tensor(
            [
                [3.00707839982836, 14.2948805321535],
                [1.20665721918500, 8.97083018508661],
                [0.65984982727415, 6.95540181314234],
                [0.424172705517784, 5.85373747658564],
            ],
            dtype=torch.float64,
        )
        self.assertTrue(
            torch.allclose(
                eigenvalues,
                expected_eigenvalues,
                rtol=0.0,
                atol=1.0e-12,
            )
        )

    def test_conditioning_zeroes_rows_outside_the_mask(self):
        radial = torch.tensor([[99.0], [1.0], [2.0]], dtype=torch.float64)
        monomials = torch.ones(3, 1, dtype=torch.float64)
        powers = torch.zeros(1, 3, dtype=torch.long)
        support = torch.tensor([False, True, True])

        basis, _ = whiten_radial_cartesian_basis(
            radial,
            monomials,
            powers,
            support,
        )

        self.assertTrue(torch.equal(basis[0], torch.zeros_like(basis[0])))

    def test_rank_deficient_basis_is_rejected(self):
        radial, monomials, powers, support = self._basis_inputs(2, 4, 0)
        with self.assertRaisesRegex(ValueError, "rank deficient"):
            whiten_radial_cartesian_basis(
                radial,
                monomials,
                powers,
                support,
            )


class TestRadialTransform(unittest.TestCase):
    def test_rectangular_identity_selects_leading_channels(self):
        transform = _RadialTransform(
            max_power=3,
            n_radial_functions=6,
            n_radial_channels=4,
        )
        basis = torch.randn(7, 6, 20)
        powers = _make_powers(3)

        actual = transform(basis, powers)

        self.assertTrue(torch.equal(actual, basis[:, :4]))
        self.assertEqual(sum(p.numel() for p in transform.parameters()), 96)
        self.assertEqual(tuple(transform.weight.shape), (4, 6, 4))

    def test_one_transform_is_shared_by_each_degree(self):
        transform = _RadialTransform(1, 2, 2)
        with torch.no_grad():
            transform.weight[1] = torch.tensor(
                [[0.0, 1.0], [1.0, 0.0]]
            )
        basis = torch.tensor(
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [10.0, 20.0, 30.0, 40.0],
                ]
            ]
        )

        actual = transform(basis, _make_powers(1))

        self.assertTrue(torch.equal(actual[0, :, 0], torch.tensor([1.0, 10.0])))
        self.assertTrue(
            torch.equal(
                actual[0, :, 1:],
                torch.tensor([[20.0, 30.0, 40.0], [2.0, 3.0, 4.0]]),
            )
        )

    def test_transform_receives_gradients(self):
        transform = _RadialTransform(2, 3, 2)
        basis = torch.randn(5, 3, 10)

        transform(basis, _make_powers(2)).square().sum().backward()

        self.assertIsNotNone(transform.weight.grad)
        self.assertTrue(torch.all(torch.isfinite(transform.weight.grad)))
        for degree in range(3):
            self.assertGreater(transform.weight.grad[degree].abs().sum(), 0.0)

    def test_expanding_transform_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            _RadialTransform(1, 2, 3)


if __name__ == "__main__":
    unittest.main()
