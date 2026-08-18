import unittest

import numpy as np
import torch

from equicdft._grid import (
    grid_spacing_tensor,
    require_matching_grid_spacing,
    voxel_volume,
)


class TestGridGeometry(unittest.TestCase):
    def test_scalar_spacing_expands_to_cubic_voxels(self):
        spacing = grid_spacing_tensor(0.5)

        self.assertTrue(torch.equal(spacing, torch.full((3,), 0.5)))
        self.assertEqual(voxel_volume(spacing).item(), 0.125)

    def test_rectangular_and_batched_voxel_volumes(self):
        spacing = torch.tensor(
            [[0.5, 0.75, 1.0], [0.25, 0.5, 2.0]],
            dtype=torch.float64,
        )

        volumes = voxel_volume(spacing)

        self.assertEqual(volumes.dtype, torch.float64)
        self.assertTrue(
            torch.equal(
                volumes,
                torch.tensor([0.375, 0.25], dtype=torch.float64),
            )
        )

    def test_spacing_accepts_numpy_values_and_requested_dtype(self):
        spacing = grid_spacing_tensor(
            np.array([0.25, 0.5, 1.0]),
            dtype=torch.float64,
        )

        self.assertEqual(spacing.dtype, torch.float64)
        self.assertTrue(
            torch.equal(
                spacing,
                torch.tensor([0.25, 0.5, 1.0], dtype=torch.float64),
            )
        )

    def test_invalid_spacing_is_rejected(self):
        for spacing in (
            [0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, float("inf"), 0.5],
        ):
            with self.subTest(spacing=spacing):
                with self.assertRaises(ValueError):
                    grid_spacing_tensor(spacing)

        with self.assertRaises(TypeError):
            voxel_volume([0.5, 0.5, 0.5])
        with self.assertRaises(ValueError):
            voxel_volume(torch.ones(2))

    def test_matching_spacing_supports_batches_and_numeric_dtypes(self):
        reference = torch.tensor([0.5, 0.5, 0.5])
        require_matching_grid_spacing(
            torch.full((4, 3), 0.5, dtype=torch.float64),
            reference,
        )

        with self.assertRaisesRegex(ValueError, "does not match"):
            require_matching_grid_spacing(
                torch.zeros(3, dtype=torch.long),
                reference,
            )


if __name__ == "__main__":
    unittest.main()
