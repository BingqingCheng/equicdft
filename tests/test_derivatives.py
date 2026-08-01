import unittest

import torch

from equicdft import compute_grid_derivative


class TestGridDerivatives(unittest.TestCase):
    def test_compute_grid_derivative_has_no_physical_rescaling(self):
        grid_field = torch.tensor(
            [[0.2], [0.5], [0.8]],
            requires_grad=True,
        )
        output = 0.125 * torch.sum(grid_field.square())

        derivative = compute_grid_derivative(output, grid_field)

        self.assertTrue(
            torch.allclose(derivative, 0.25 * grid_field)
        )


if __name__ == "__main__":
    unittest.main()
