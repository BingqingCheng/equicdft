import unittest

import torch

from equicdft.energy import (
    density_weighted_integral,
    ideal_free_energy,
    log_dimensionless_density,
)


class TestDensityWeightedIntegral(unittest.TestCase):
    def test_local_multicomponent_integral(self):
        rho = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.float64,
        )
        per_particle = torch.tensor(
            [[0.5, -1.0], [2.0, 0.25]],
            dtype=torch.float64,
        )

        energy = density_weighted_integral(
            rho,
            per_particle,
            torch.tensor(0.125),
        )

        self.assertEqual(energy.dtype, torch.float64)
        self.assertAlmostEqual(energy.item(), 0.6875)

    def test_spatially_constant_values_and_batched_volumes(self):
        rho = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[2.0, 1.0], [4.0, 3.0]],
            ]
        )
        per_particle = torch.tensor(
            [[[2.0, -1.0]], [[0.5, 1.0]]]
        )

        energy = density_weighted_integral(
            rho,
            per_particle,
            torch.tensor([0.5, 0.25]),
        )

        self.assertTrue(torch.equal(energy, torch.tensor([1.0, 1.75])))

    def test_autograd_reaches_density_and_per_particle_values(self):
        rho = torch.tensor(
            [[1.0], [2.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        per_particle = torch.tensor(
            [[3.0], [4.0]],
            dtype=torch.float64,
            requires_grad=True,
        )

        energy = density_weighted_integral(
            rho,
            per_particle,
            torch.tensor(0.5),
        )
        density_gradient, value_gradient = torch.autograd.grad(
            energy,
            (rho, per_particle),
        )

        self.assertTrue(
            torch.equal(
                density_gradient,
                torch.tensor([[1.5], [2.0]], dtype=torch.float64),
            )
        )
        self.assertTrue(
            torch.equal(
                value_gradient,
                torch.tensor([[0.5], [1.0]], dtype=torch.float64),
            )
        )

    def test_zero_density_is_exactly_zero(self):
        rho = torch.zeros(2, 3, 1)
        per_particle = torch.randn_like(rho)

        energy = density_weighted_integral(rho, per_particle, 0.125)

        self.assertTrue(torch.equal(energy, torch.zeros(2)))

    def test_invalid_shapes_and_volume_are_rejected(self):
        rho = torch.ones(2, 3, 1)
        with self.assertRaisesRegex(ValueError, "rank"):
            density_weighted_integral(rho, torch.ones(3, 1), 1.0)
        with self.assertRaisesRegex(ValueError, "one or n_grid"):
            density_weighted_integral(rho, torch.ones(2, 2, 1), 1.0)
        with self.assertRaisesRegex(ValueError, "leading field shape"):
            density_weighted_integral(
                rho,
                torch.ones_like(rho),
                torch.ones(3),
            )
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            density_weighted_integral(rho, torch.ones_like(rho), 0.0)


class TestIdealGasThermodynamics(unittest.TestCase):
    def test_log_density_and_vacuum_convention(self):
        rho = torch.tensor([[0.0, 0.5], [2.0, 1.0]], dtype=torch.float64)
        wavelength = torch.tensor([2.0, 3.0], dtype=torch.float64)

        value = log_dimensionless_density(rho, wavelength)

        self.assertTrue(torch.all(torch.isfinite(value)))
        self.assertEqual(value[1, 0], rho.new_tensor(16.0).log())
        self.assertEqual(value[1, 1], rho.new_tensor(27.0).log())

    def test_positive_subnormal_density_is_floored(self):
        tiny = torch.finfo(torch.float64).tiny
        rho = torch.tensor([[0.5 * tiny]], dtype=torch.float64)
        wavelength = torch.ones(1, dtype=torch.float64)

        self.assertEqual(
            log_dimensionless_density(rho, wavelength).item(),
            rho.new_tensor(tiny).log().item(),
        )

    def test_thermal_wavelength_contributes_expected_linear_term(self):
        rho = torch.tensor([[0.5, 1.0], [1.5, 2.0]], dtype=torch.float64)
        wavelength = torch.tensor([2.0, 3.0], dtype=torch.float64)
        volume = torch.tensor(0.25, dtype=torch.float64)

        full = ideal_free_energy(rho, wavelength, volume)
        unit_wavelength = ideal_free_energy(
            rho,
            torch.ones_like(wavelength),
            volume,
        )
        expected_offset = volume * torch.sum(
            rho * (3.0 * torch.log(wavelength))[None, :]
        )

        self.assertTrue(
            torch.allclose(full - unit_wavelength, expected_offset)
        )

    def test_vacuum_has_exactly_zero_ideal_free_energy(self):
        rho = torch.zeros((2, 3, 2), dtype=torch.float64)
        wavelength = torch.tensor([2.0, 3.0], dtype=torch.float64)

        value = ideal_free_energy(rho, wavelength, 0.125)

        self.assertTrue(torch.equal(value, torch.zeros(2, dtype=rho.dtype)))


if __name__ == "__main__":
    unittest.main()
