import unittest

import torch

from cace_grid import compute_c1


class TestFunctionalDerivatives(unittest.TestCase):
    def test_compute_c1(self):
        rho = torch.tensor(
            [[0.2], [0.5], [0.8]],
            requires_grad=True,
        )
        grid_spacing = torch.tensor([0.5, 0.5, 0.5])
        beta_F_exc = torch.prod(grid_spacing) * torch.sum(rho.square())

        c1 = compute_c1(beta_F_exc, rho, grid_spacing)

        self.assertTrue(torch.allclose(c1, -2.0 * rho))


if __name__ == "__main__":
    unittest.main()
