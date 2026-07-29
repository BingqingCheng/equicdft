import unittest

import numpy as np

from cace_grid import get_local_density


class TestGetLocalDensity(unittest.TestCase):
    def test_center_is_first_and_cutoff_two_has_257_positions(self):
        shape = (10, 10, 10)
        positions = np.indices(shape, dtype=int).reshape(3, -1).T
        rho = np.arange(len(positions), dtype=float)

        local_density, local_positions = get_local_density(
            rho=rho,
            grid_positions=positions,
            grid_spacing=(0.5, 0.5, 0.5),
            cutoff=2.0,
        )

        self.assertEqual(local_density.shape, (1000, 257))
        self.assertEqual(local_positions.shape, (257, 3))
        self.assertTrue(np.array_equal(local_positions[0], (0, 0, 0)))
        self.assertTrue(np.array_equal(local_density[:, 0], rho))


if __name__ == "__main__":
    unittest.main()
