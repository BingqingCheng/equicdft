"""Direct regression tests for constrained canonical stationarity."""

import unittest

import torch

from equicdft.solver.numerics import _canonical_density_residual


class TestCanonicalDensityResidual(unittest.TestCase):
    def test_all_bound_equilibrium_and_gauge_shifts(self):
        for dtype in (torch.float32, torch.float64):
            rho = torch.tensor([[0.1], [0.1], [0.9], [0.9]], dtype=dtype)
            potential = rho.new_tensor([[0.0], [10.0], [-3.0], [-20.0]])
            for shift in (0.0, -10.0, 10.0, 1000.0):
                with self.subTest(dtype=dtype, shift=shift):
                    local = rho.log() + potential + shift
                    residual, mu, maximum, rms = _canonical_density_residual(
                        rho, local, 0.0,
                        maximum_density=rho.new_tensor([0.9]),
                        minimum_density=rho.new_tensor(0.1),
                    )
                    self.assertGreaterEqual(mu.item(), local[2:].max().item())
                    self.assertLessEqual(mu.item(), local[:2].min().item())
                    torch.testing.assert_close(residual, torch.zeros_like(rho))
                    self.assertEqual(maximum, 0.0)
                    self.assertEqual(rms, 0.0)

    def test_empty_kkt_interval_is_not_equilibrium(self):
        rho = torch.tensor([[0.1], [0.9]], dtype=torch.float64)
        local = rho.new_tensor([[-4.0], [2.0]])
        residual, mu, maximum, rms = _canonical_density_residual(
            rho, local, 0.0, maximum_density=rho.new_tensor([0.9]),
            minimum_density=rho.new_tensor(0.1),
        )
        torch.testing.assert_close(mu, rho.new_tensor([-1.0]))
        torch.testing.assert_close(residual, rho.new_tensor([[-3.0], [3.0]]))
        self.assertEqual(maximum, 3.0)
        self.assertEqual(rms, 3.0)

    def test_only_one_bound_type_is_occupied(self):
        for bound in ("floor", "cap"):
            with self.subTest(bound=bound):
                rho = torch.full((3, 1), 0.1 if bound == "floor" else 0.9,
                                 dtype=torch.float64)
                local = rho.new_tensor([[-7.0], [2.0], [10.0]])
                kwargs = (
                    {"minimum_density": rho.new_tensor(0.1)}
                    if bound == "floor"
                    else {"maximum_density": rho.new_tensor([0.9])}
                )
                residual, mu, maximum, rms = _canonical_density_residual(
                    rho, local, 0.0, **kwargs,
                )
                self.assertEqual(mu.item(), -7.0 if bound == "floor" else 10.0)
                torch.testing.assert_close(residual, torch.zeros_like(rho))
                self.assertEqual(maximum, 0.0)
                self.assertEqual(rms, 0.0)

    def test_mixed_species_and_inaccessible_voxels(self):
        # Species 0 is fully bound; species 1 has two interior voxels.
        # The final voxel is excluded despite its adversarial rho/local values.
        rho = torch.tensor([[0.1, 0.1], [0.9, 0.3], [0.1, 0.6], [0.9, 0.9]],
                           dtype=torch.float64)
        local = rho.new_tensor([[2.0, 20.0], [-2.0, 2.0], [10.0, 5.0],
                                [1.0e6, -1.0e6]])
        residual, mu, maximum, rms = _canonical_density_residual(
            rho, local, 0.0, maximum_density=rho.new_tensor([0.9, 0.9]),
            minimum_density=rho.new_tensor(0.1),
            accessible_mask=torch.tensor([True, True, True, False]),
        )
        torch.testing.assert_close(mu, rho.new_tensor([0.0, 4.0]))
        torch.testing.assert_close(
            residual, rho.new_tensor([[0.0, 0.0], [0.0, -2.0],
                                     [0.0, 1.0], [0.0, 0.0]]),
        )
        self.assertAlmostEqual(maximum, 2.0)
        self.assertAlmostEqual(rms, (1.8 / 2.1) ** 0.5)


if __name__ == "__main__":
    unittest.main()
