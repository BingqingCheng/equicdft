import unittest

import numpy as np
import torch

from equicdft._grid import (
    gather_neighbors,
    grid_spacing_tensor,
    periodic_stencil_convolution,
    require_matching_grid_spacing,
    voxel_volume,
)
from equicdft.stencil import get_neighbor_indices


class TestGridGeometry(unittest.TestCase):
    def test_periodic_convolution_matches_shuffled_neighbor_gather(self):
        torch.manual_seed(17)
        shape = (3, 4, 5)
        canonical_positions = np.indices(shape, dtype=int).reshape(3, -1).T
        generator = np.random.default_rng(17)
        positions = []
        neighbor_indices = []
        for _ in range(2):
            field_positions = canonical_positions[
                generator.permutation(len(canonical_positions))
            ]
            neighbor_index, stencil_positions = get_neighbor_indices(
                field_positions,
                cutoff_grid=2,
            )
            positions.append(field_positions)
            neighbor_indices.append(neighbor_index)
        values = torch.randn(
            2,
            len(canonical_positions),
            2,
            3,
            dtype=torch.float64,
        )
        basis = torch.randn(
            len(stencil_positions),
            2,
            4,
            dtype=torch.float64,
        )
        indices = torch.tensor(np.stack(neighbor_indices))

        expected = torch.einsum(
            "bgjnc,jnk->bgnkc",
            gather_neighbors(values, indices),
            basis,
        )
        for backend in ("conv3d", "fft"):
            with self.subTest(backend=backend):
                actual = periodic_stencil_convolution(
                    values,
                    basis,
                    torch.tensor(np.stack(positions)),
                    torch.tensor(shape).expand(2, -1),
                    torch.tensor(stencil_positions),
                    backend=backend,
                )
                self.assertTrue(
                    torch.allclose(
                        actual,
                        expected,
                        rtol=1.0e-12,
                        atol=1.0e-12,
                    )
                )

    def test_fft_preserves_offset_direction_duplicates_and_aliases(self):
        shape = (2, 3, 4)
        positions = torch.tensor(
            np.indices(shape, dtype=int).reshape(3, -1).T
        )
        offsets = torch.tensor(
            [
                [1, 0, 0],
                [-1, 0, 0],
                [3, 0, 0],
                [1, 0, 0],
                [0, -1, 1],
            ]
        )
        values = torch.arange(
            np.prod(shape),
            dtype=torch.float64,
        ).reshape(-1, 1, 1)
        basis = torch.arange(1, 6, dtype=torch.float64).reshape(-1, 1, 1)
        neighbor_positions = (
            positions[:, None] + offsets[None]
        ).remainder(torch.tensor(shape))
        neighbor_index = (
            (neighbor_positions[..., 0] * shape[1]
             + neighbor_positions[..., 1])
            * shape[2]
            + neighbor_positions[..., 2]
        )
        expected = torch.einsum(
            "gjnc,jnk->gnkc",
            values[neighbor_index],
            basis,
        )

        actual = periodic_stencil_convolution(
            values,
            basis,
            positions,
            torch.tensor(shape),
            offsets,
            backend="fft",
        )

        self.assertTrue(
            torch.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)
        )

    def test_fft_stencil_has_first_and_second_derivatives(self):
        torch.manual_seed(23)
        shape = (3, 4, 5)
        positions = torch.tensor(
            np.indices(shape, dtype=int).reshape(3, -1).T
        )
        offsets = torch.tensor(
            [[0, 0, 0], [1, -1, 0], [-2, 1, 1], [1, -1, 0]]
        )
        values = torch.randn(
            np.prod(shape),
            1,
            2,
            dtype=torch.float64,
            requires_grad=True,
        )
        basis = torch.randn(
            len(offsets),
            1,
            2,
            dtype=torch.float64,
            requires_grad=True,
        )

        def apply_fft(current_values, current_basis):
            return periodic_stencil_convolution(
                current_values,
                current_basis,
                positions,
                torch.tensor(shape),
                offsets,
                backend="fft",
            )

        self.assertTrue(
            torch.autograd.gradcheck(
                apply_fft,
                (values, basis),
                fast_mode=True,
            )
        )
        self.assertTrue(
            torch.autograd.gradgradcheck(
                apply_fft,
                (values, basis),
                fast_mode=True,
            )
        )

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
