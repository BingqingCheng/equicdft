import unittest

import numpy as np

from equicdft import coarsen_grid, get_neighbor_indices, make_stencil


class TestGridStencil(unittest.TestCase):
    def test_default_stencil_has_canonical_123_positions(self):
        shape = (10, 10, 10)
        positions = np.indices(shape, dtype=int).reshape(3, -1).T
        rho = np.arange(len(positions), dtype=float).reshape(-1, 1)

        neighbor_indices, local_positions = get_neighbor_indices(
            grid_positions=positions,
        )
        local_density = rho[neighbor_indices]

        self.assertEqual(local_density.shape, (1000, 123, 1))
        self.assertEqual(local_positions.shape, (123, 3))
        self.assertTrue(np.array_equal(local_positions[0], (0, 0, 0)))
        self.assertTrue(np.array_equal(local_density[:, 0, :], rho))

        expected_first_shells = np.asarray(
            [
                (0, 0, 0),
                (0, 0, 1),
                (0, 0, -1),
                (0, 1, 0),
                (0, -1, 0),
                (1, 0, 0),
                (-1, 0, 0),
                (0, 1, 1),
                (0, 1, -1),
                (0, -1, 1),
                (0, -1, -1),
                (1, 0, 1),
                (1, 0, -1),
                (-1, 0, 1),
                (-1, 0, -1),
                (1, 1, 0),
                (1, -1, 0),
                (-1, 1, 0),
                (-1, -1, 0),
                (1, 1, 1),
                (1, 1, -1),
                (1, -1, 1),
                (1, -1, -1),
                (-1, 1, 1),
                (-1, 1, -1),
                (-1, -1, 1),
                (-1, -1, -1),
            ],
            dtype=np.int64,
        )
        self.assertTrue(
            np.array_equal(
                local_positions[: len(expected_first_shells)],
                expected_first_shells,
            )
        )

    def test_stencil_cutoff_is_inclusive(self):
        positions = make_stencil(cutoff_grid=3)
        squared_distances = np.sum(positions**2, axis=1)

        self.assertTrue(np.all(squared_distances <= 9))
        self.assertTrue(np.any(squared_distances == 9))
        self.assertIn((0, 0, 3), map(tuple, positions))

    def test_cutoff_can_include_multiple_periodic_images(self):
        shape = (4, 4, 4)
        grid_positions = np.indices(shape, dtype=int).reshape(3, -1).T

        neighbor_indices, local_positions = get_neighbor_indices(
            grid_positions=grid_positions,
            cutoff_grid=3,
        )

        position_to_column = {
            tuple(position): column
            for column, position in enumerate(local_positions)
        }
        positive_image = position_to_column[(0, 0, 1)]
        negative_image = position_to_column[(0, 0, -3)]

        # Both offsets wrap to the same stored voxel around the origin, but
        # remain distinct environment entries with distinct relative vectors.
        self.assertEqual(
            neighbor_indices[0, positive_image],
            neighbor_indices[0, negative_image],
        )
        self.assertNotEqual(positive_image, negative_image)

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
