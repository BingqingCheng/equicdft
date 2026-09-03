import math
import unittest

import torch

from equicdft import LongRangeReadout, PairwiseReadout, ReciprocalFeatures
from equicdft._component_pairs import symmetric_component_pairs


class TestSymmetricComponentPairs(unittest.TestCase):
    def test_three_component_order_is_shared(self):
        expected = (
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 1),
            (1, 2),
            (2, 2),
        )
        pairwise = PairwiseReadout(
            cutoff_grid=2,
            n_types=3,
            hidden_sizes=(),
        )
        reciprocal = ReciprocalFeatures(
            radial_exponents=(0.5,),
            n_types=3,
        )
        long_range = LongRangeReadout(
            n_kernels=1,
            n_types=3,
            charges=(2.0, -3.0, 5.0),
            coulomb_amplitude=2.0,
        )

        self.assertEqual(symmetric_component_pairs(3), expected)
        self.assertEqual(pairwise.type_pairs, expected)
        self.assertEqual(pairwise.n_type_pairs, len(expected))
        self.assertEqual(reciprocal.n_type_pairs, len(expected))
        self.assertEqual(long_range.n_type_pairs, len(expected))
        self.assertTrue(
            torch.equal(
                long_range.pair_charge_products,
                torch.tensor([4.0, -6.0, 10.0, 9.0, -15.0, 25.0]),
            )
        )

    def test_reciprocal_order_preserves_off_diagonal_multiplicity(self):
        dtype = torch.float64
        shape = (4, 1, 1)
        coordinate = torch.arange(shape[0], dtype=dtype)
        wave = torch.cos(2.0 * math.pi * coordinate / shape[0])
        amplitudes = torch.tensor([0.1, 0.3, 0.7], dtype=dtype)
        rho = 0.8 + wave[:, None] * amplitudes[None, :]
        module = ReciprocalFeatures(
            radial_exponents=(0.5,),
            n_types=3,
        ).to(dtype=dtype)

        features = module(
            rho,
            grid_size=torch.tensor(shape),
            grid_spacing=torch.ones(3, dtype=dtype),
        )[0]

        relative_pair_power = torch.tensor(
            [0.01, 0.06, 0.14, 0.09, 0.42, 0.49],
            dtype=dtype,
        )
        scale = features[0] / relative_pair_power[0]
        self.assertTrue(
            torch.allclose(
                features,
                scale * relative_pair_power,
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        )


if __name__ == "__main__":
    unittest.main()
