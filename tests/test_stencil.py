import unittest

import numpy as np

from cace_grid import coarsen_grid, get_local_density


class TestGetLocalDensity(unittest.TestCase):
    def test_center_is_first_and_cutoff_two_has_257_positions(self):
        shape = (10, 10, 10)
        positions = np.indices(shape, dtype=int).reshape(3, -1).T
        rho = np.arange(len(positions), dtype=float).reshape(-1, 1)

        local_density, local_positions = get_local_density(
            rho=rho,
            grid_positions=positions,
            grid_spacing=(0.5, 0.5, 0.5),
            cutoff=2.0,
        )

        self.assertEqual(local_density.shape, (1000, 257, 1))
        self.assertEqual(local_positions.shape, (257, 3))
        self.assertTrue(np.array_equal(local_positions[0], (0, 0, 0)))
        self.assertTrue(np.array_equal(local_density[:, 0, :], rho))

    def test_coarsen_grid_averages_all_fields(self):
        shape = (4, 4, 4)
        positions = np.indices(shape, dtype=int).reshape(3, -1).T
        first = np.arange(len(positions), dtype=float)
        values = np.column_stack((first, -first))

        coarse, coarse_positions, spacing = coarsen_grid(
            values=values,
            grid_positions=positions,
            grid_spacing=0.25,
            target_grid_spacing=0.5,
        )

        expected = first.reshape(shape).reshape(2, 2, 2, 2, 2, 2).mean(
            axis=(1, 3, 5)
        )
        self.assertEqual(coarse.shape, (8, 2))
        self.assertTrue(np.allclose(coarse[:, 0], expected.reshape(-1)))
        self.assertTrue(np.allclose(coarse[:, 1], -expected.reshape(-1)))
        self.assertTrue(np.array_equal(coarse_positions[1], (0, 0, 1)))
        self.assertTrue(np.allclose(spacing, (0.5, 0.5, 0.5)))


if __name__ == "__main__":
    unittest.main()
