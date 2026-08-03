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

    def test_unused_grid_field_can_return_differentiable_zero(self):
        grid_field = torch.ones(3, requires_grad=True)
        parameter = torch.tensor(2.0, requires_grad=True)

        derivative = compute_grid_derivative(
            parameter.square(),
            grid_field,
            create_graph=True,
            allow_unused=True,
        )
        derivative.sum().backward()

        self.assertTrue(torch.equal(derivative, torch.zeros_like(grid_field)))
        self.assertEqual(parameter.grad.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
