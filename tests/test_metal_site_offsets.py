"""Roundoff-bounded offset grouping, independent of electrode physics."""

import unittest

import torch

from equicdft._fourier_sites import FourierSites


SHAPE = (4, 3, 2)
SPACING = torch.tensor([.5, .75, 1.25], dtype=torch.float64)
TOLERANCE = 32 * torch.finfo(torch.float64).eps * max(SHAPE)


def _sampling(scaled_positions, dtype=torch.float64):
    positions = scaled_positions * SPACING
    return FourierSites(positions, SHAPE, SPACING, dtype, torch.device("cpu"))


class TestMetalSiteOffsets(unittest.TestCase):
    def test_roundoff_offsets_coalesce_without_rounding_the_representative(self):
        fraction = torch.tensor([.25, .375, .125], dtype=torch.float64)
        scaled = torch.tensor([[0., 0., 0.], [1., 1., 0.]], dtype=torch.float64) + fraction
        scaled[1, 0] += TOLERANCE / 4
        positions = scaled * SPACING
        before = positions.clone()
        sampling = FourierSites(positions, SHAPE, SPACING, torch.float64,
                                    torch.device("cpu"))

        self.assertEqual(len(sampling.groups), 1)
        self.assertEqual(sampling.groups[0][0].tolist(), [0, 1])
        self.assertEqual(sampling.offset_tolerance, TOLERANCE)
        self.assertGreater(sampling.max_offset_error, 0.)
        self.assertLessEqual(sampling.max_offset_error, TOLERANCE)
        torch.testing.assert_close(sampling.groups[0][3], fraction * SPACING,
                                   atol=0., rtol=0.)
        torch.testing.assert_close(positions, before, atol=0., rtol=0.)

        # A constant Fourier kernel has G(0)=1/voxel_volume, including the
        # self term. Common fractional offsets must not change this value.
        matrix = sampling.matrix(torch.ones(SHAPE, dtype=torch.float64))
        expected = torch.full((2,), 1. / float(SPACING.prod()), dtype=torch.float64)
        torch.testing.assert_close(matrix.diag(), expected, atol=2e-14, rtol=0.)
        torch.testing.assert_close(matrix, matrix.T, atol=0., rtol=0.)

    def test_distinct_offsets_do_not_merge_when_density_uses_float32(self):
        scaled = torch.tensor([[.25, .375, .125], [1.25, 1.375, .125]],
                              dtype=torch.float64)
        scaled[1, 0] += 4 * TOLERANCE
        for dtype in (torch.float32, torch.float64):
            with self.subTest(density_dtype=dtype):
                sampling = _sampling(scaled, dtype)
                self.assertEqual(sampling.offset_tolerance, TOLERANCE)
                self.assertEqual(len(sampling.groups), 2)
                self.assertEqual(sampling.max_offset_error, 0.)
                self.assertEqual([group[0].tolist() for group in sampling.groups], [[0], [1]])
                self.assertTrue(all(group[3].dtype == torch.float64 for group in sampling.groups))
                matrix = sampling.matrix(torch.ones(SHAPE, dtype=dtype))
                self.assertEqual(matrix.dtype, dtype)
                self.assertTrue(bool(torch.isfinite(matrix).all()))
                torch.testing.assert_close(matrix, matrix.T, atol=0., rtol=0.)

    def test_near_integer_carry_wraps_periodic_indices_but_retains_small_offset(self):
        epsilon = TOLERANCE / 4
        scaled = torch.tensor([SHAPE, (1, 1, 1)], dtype=torch.float64) - epsilon
        sampling = _sampling(scaled)
        self.assertEqual(len(sampling.groups), 1)
        _, indices, flat, offset = sampling.groups[0]
        self.assertEqual(indices.tolist(), [[0, 0, 0], [1, 1, 1]])
        self.assertEqual(flat.tolist(), [0, 9])
        self.assertTrue(bool((offset < 0).all()))
        self.assertLessEqual(float((offset / SPACING).abs().max()), TOLERANCE)
        self.assertLessEqual(sampling.max_offset_error, TOLERANCE)

        # Express exactly the same sites on a different periodic branch.
        translated = scaled - torch.tensor(SHAPE, dtype=torch.float64)
        other = _sampling(translated)
        kernel = torch.ones(SHAPE, dtype=torch.float64)
        charge = torch.linspace(-.2, .3, 24, dtype=torch.float64)
        torch.testing.assert_close(sampling.matrix(kernel), other.matrix(kernel),
                                   atol=2e-13, rtol=0.)
        torch.testing.assert_close(sampling.potential(charge, kernel),
                                   other.potential(charge, kernel), atol=2e-13, rtol=0.)


if __name__ == "__main__":
    unittest.main()
