import unittest

import torch

from equicdft import GGAReadout, LDAReadout
from equicdft.semilocal import periodic_gradient_energy_density


class TestLDAReadout(unittest.TestCase):
    def test_linear_pointwise_state_mapping(self):
        readout = LDAReadout(
            mean_density=2.0,
            n_types=2,
            hidden_sizes=(),
        )
        with torch.no_grad():
            readout.mlp[-1].weight.copy_(
                torch.tensor([[2.0, -1.0, 0.5]])
            )
            readout.mlp[-1].bias.fill_(0.25)
        local_state = torch.tensor(
            [
                [[1.0, 0.5, 2.0], [0.0, 3.0, 1.0]],
                [[2.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
            ]
        )

        output = readout(local_state)

        expected = torch.tensor([[[2.75], [-2.25]], [[3.25], [1.75]]])
        self.assertEqual(output.shape, (2, 2, 1))
        self.assertTrue(torch.equal(output, expected))
        self.assertEqual(readout.mean_density.item(), 2.0)

    def test_zero_initialization_gives_zero_energy_per_particle(self):
        readout = LDAReadout(
            mean_density=0.5,
            n_types=1,
            hidden_sizes=(4,),
            zero_init=True,
        )

        output = readout(torch.randn(3, 7, 2))

        self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_configuration_and_input_width_are_validated(self):
        for mean_density in (0.0, -1.0, float("nan"), [1.0, 2.0]):
            with self.subTest(mean_density=mean_density):
                with self.assertRaisesRegex(ValueError, "mean_density"):
                    LDAReadout(mean_density=mean_density)
        for n_types in (0, -1, 1.5, True):
            with self.subTest(n_types=n_types):
                with self.assertRaises((TypeError, ValueError)):
                    LDAReadout(mean_density=1.0, n_types=n_types)

        readout = LDAReadout(mean_density=1.0, n_types=2)
        with self.assertRaisesRegex(ValueError, "local_state"):
            readout(torch.ones(4, 2))


class TestGGAReadout(unittest.TestCase):
    def test_initial_coefficient_is_small_positive_constant(self):
        readout = GGAReadout(
            hidden_sizes=(5,),
            n_features=3,
            minimum_coefficient=0.002,
            initial_coefficient=0.02,
        )
        features = torch.randn(2, 7, 3, requires_grad=True)

        coefficient = readout(features)

        self.assertEqual(coefficient.shape, (2, 7, 1))
        self.assertTrue(
            torch.allclose(coefficient, torch.full_like(coefficient, 0.02))
        )
        coefficient.square().sum().backward()
        self.assertIsNotNone(features.grad)
        self.assertIsNotNone(readout.mlp[-1].weight.grad)
        self.assertIsNotNone(readout.mlp[-1].bias.grad)

    def test_coefficient_cannot_cross_configured_lower_bound(self):
        readout = GGAReadout(
            hidden_sizes=(),
            n_features=2,
            minimum_coefficient=0.1,
            initial_coefficient=0.2,
        )
        with torch.no_grad():
            readout.mlp[-1].weight.fill_(-100.0)
            readout.mlp[-1].bias.fill_(-100.0)

        coefficient = readout(torch.ones(4, 2))

        self.assertTrue(torch.all(coefficient >= 0.1))

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "n_features"):
            GGAReadout(hidden_sizes=())
        with self.assertRaisesRegex(ValueError, "initial_coefficient"):
            GGAReadout(
                n_features=2,
                minimum_coefficient=0.1,
                initial_coefficient=0.1,
            )


class TestPeriodicGradientEnergyDensity(unittest.TestCase):
    def test_homogeneous_field_has_exactly_zero_energy(self):
        rho = torch.full((2, 24, 3), 0.7)
        coefficient = torch.rand(2, 24, 1) + 0.1

        energy_density = periodic_gradient_energy_density(
            rho,
            coefficient,
            grid_size=torch.tensor([[4, 3, 2], [4, 3, 2]]),
            grid_spacing=torch.tensor([0.5, 0.7, 1.0]),
        )

        self.assertTrue(torch.equal(energy_density, torch.zeros_like(energy_density)))

    def test_checkerboard_energy_and_derivative_are_analytic(self):
        rho = torch.tensor([[0.0], [1.0], [0.0], [1.0]], requires_grad=True)
        coefficient = torch.full((4, 1), 0.2)

        energy_density = periodic_gradient_energy_density(
            rho,
            coefficient,
            grid_size=torch.tensor([4, 1, 1]),
            grid_spacing=torch.ones(3),
        )
        energy = energy_density.sum()
        derivative = torch.autograd.grad(energy, rho)[0]

        self.assertAlmostEqual(energy.item(), 0.4, places=6)
        expected_derivative = torch.tensor([[-0.4], [0.4], [-0.4], [0.4]])
        self.assertTrue(torch.allclose(derivative, expected_derivative))

    def test_periodic_translation_does_not_change_total_energy(self):
        rho = torch.rand(3, 3, 2, 1)
        coefficient = torch.rand(3, 3, 2, 1) + 0.1

        energy = periodic_gradient_energy_density(
            rho.reshape(18, 1),
            coefficient.reshape(18, 1),
            grid_size=torch.tensor([3, 3, 2]),
            grid_spacing=torch.ones(3),
        ).sum()
        translated_energy = periodic_gradient_energy_density(
            torch.roll(rho, shifts=1, dims=0).reshape(18, 1),
            torch.roll(coefficient, shifts=1, dims=0).reshape(18, 1),
            grid_size=torch.tensor([3, 3, 2]),
            grid_spacing=torch.ones(3),
        ).sum()

        self.assertTrue(torch.allclose(energy, translated_energy))

    def test_grid_and_coefficient_shapes_are_validated(self):
        rho = torch.ones(4, 1)
        with self.assertRaisesRegex(ValueError, "coefficient"):
            periodic_gradient_energy_density(
                rho,
                torch.ones(4),
                grid_size=torch.tensor([4, 1, 1]),
                grid_spacing=torch.ones(3),
            )
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            periodic_gradient_energy_density(
                rho,
                torch.ones(4, 1),
                grid_size=torch.tensor([2, 1, 1]),
                grid_spacing=torch.ones(3),
            )


if __name__ == "__main__":
    unittest.main()
