import math
import unittest

import torch

from equicdft import LongRangeReadout, ReciprocalFeatures


class TestReciprocalFeatures(unittest.TestCase):
    def _field(self, shape=(4, 4, 4)):
        coordinates = torch.arange(math.prod(shape)).reshape(*shape)
        rho = 0.7 + 0.1 * torch.sin(coordinates.to(torch.float32))
        return rho.reshape(-1, 1)

    def test_homogeneous_density_has_zero_features(self):
        rho = torch.full((4 * 4 * 4, 1), 0.7)
        for kernel in (
            "gaussian",
            "screened_inverse_laplacian",
            "coulomb",
        ):
            with self.subTest(kernel=kernel):
                features = ReciprocalFeatures(
                    radial_exponents=(0.5, 1.0),
                    screening=0.25,
                    kernel=kernel,
                )(
                    rho,
                    grid_size=torch.tensor([4, 4, 4]),
                    grid_spacing=torch.tensor([0.5, 0.5, 0.5]),
                )
                self.assertTrue(torch.equal(features, torch.zeros_like(features)))

    def test_kernel_flag_selects_expected_radial_function(self):
        shape = (4, 4, 4)
        spacing = torch.ones(3)
        alpha = 0.5
        k = 2.0 * math.pi / shape[0]
        gaussian = math.exp(-alpha * k**2)
        expected = {
            "gaussian": gaussian,
            "screened_inverse_laplacian": gaussian / (k**2 + 0.25**2),
            "coulomb": 4.0 * math.pi * gaussian / k**2,
        }

        for kernel, expected_value in expected.items():
            with self.subTest(kernel=kernel):
                module = ReciprocalFeatures(
                    radial_exponents=(alpha,),
                    screening=0.25,
                    kernel=kernel,
                )
                values = module._kernel_values(
                    grid_size=shape,
                    grid_spacing=spacing,
                    device=spacing.device,
                    dtype=spacing.dtype,
                )

                self.assertEqual(values[0, 0, 0, 0].item(), 0.0)
                self.assertAlmostEqual(
                    values[0, 1, 0, 0].item(),
                    expected_value,
                    places=6,
                )

    def test_unknown_kernel_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "kernel must be one of"):
            ReciprocalFeatures(kernel="unknown")

    def test_features_are_translation_invariant(self):
        shape = (4, 4, 4)
        rho = self._field(shape)
        translated = torch.roll(
            rho.reshape(*shape, 1),
            shifts=(1, -2, 1),
            dims=(0, 1, 2),
        ).reshape(-1, 1)
        module = ReciprocalFeatures(radial_exponents=(0.25, 1.0))

        first = module(rho, torch.tensor(shape), torch.ones(3))
        second = module(translated, torch.tensor(shape), torch.ones(3))

        self.assertTrue(torch.allclose(first, second, atol=1.0e-6))

    def test_tiling_scales_features_extensively(self):
        shape = (4, 4, 4)
        rho_grid = self._field(shape).reshape(*shape, 1)
        tiled = rho_grid.repeat(2, 1, 1, 1)
        module = ReciprocalFeatures(radial_exponents=(0.5, 1.0))

        original_features = module(
            rho_grid.reshape(-1, 1),
            torch.tensor(shape),
            torch.ones(3),
        )
        tiled_features = module(
            tiled.reshape(-1, 1),
            torch.tensor([8, 4, 4]),
            torch.ones(3),
        )

        self.assertTrue(
            torch.allclose(tiled_features, 2.0 * original_features, atol=1.0e-5)
        )

    def test_single_cosine_has_continuum_fft_normalization(self):
        shape = (4, 4, 4)
        spacing = torch.tensor([0.5, 0.5, 0.5])
        amplitude = 0.2
        x = torch.arange(shape[0], dtype=torch.float32)
        cosine = amplitude * torch.cos(2.0 * math.pi * x / shape[0])
        rho = (0.7 + cosine[:, None, None]).expand(*shape).reshape(-1, 1)
        alpha = 0.5
        features = ReciprocalFeatures(radial_exponents=(alpha,))(
            rho,
            torch.tensor(shape),
            spacing,
        )

        volume = math.prod(shape) * torch.prod(spacing).item()
        wavevector = 2.0 * math.pi / (shape[0] * spacing[0].item())
        expected = (
            amplitude**2
            * volume
            * math.exp(-alpha * wavevector**2)
            / 4.0
        )
        self.assertAlmostEqual(features.item(), expected, places=6)

    def test_features_remain_differentiable_with_respect_to_density(self):
        rho = self._field().requires_grad_(True)
        features = ReciprocalFeatures(radial_exponents=(0.5, 1.0))(
            rho,
            torch.tensor([4, 4, 4]),
            torch.ones(3),
        )

        features.sum().backward()

        self.assertIsNotNone(rho.grad)
        self.assertTrue(torch.all(torch.isfinite(rho.grad)))


class TestLongRangeReadout(unittest.TestCase):
    def test_zero_initialization_gives_zero_energy(self):
        readout = LongRangeReadout(n_kernels=3, hidden_sizes=(4,))
        reciprocal = torch.randn(2, 3, 1)
        state = torch.randn(2, 2)

        energy = readout(reciprocal, state)

        self.assertTrue(torch.equal(energy, torch.zeros_like(energy)))

    def test_linear_contraction_preserves_leading_batch_shape(self):
        readout = LongRangeReadout(
            n_kernels=2,
            hidden_sizes=(),
            zero_init=False,
        )
        with torch.no_grad():
            readout.mlp[-1].weight.zero_()
            readout.mlp[-1].bias.copy_(torch.tensor([2.0, -1.0]))
        reciprocal = torch.tensor(
            [[[3.0], [4.0]], [[1.0], [5.0]]]
        )
        state = torch.ones(2, 2)

        energy = readout(reciprocal, state)

        self.assertTrue(torch.equal(energy, torch.tensor([2.0, -3.0])))

    def test_fixed_charge_factorized_coefficients(self):
        readout = LongRangeReadout(
            n_kernels=1,
            n_types=2,
            charges=(1.0, -1.0),
            coulomb_amplitude=3.0,
        )
        state = torch.ones(2, 3)

        coefficients = readout.coefficients(state)

        expected = torch.tensor([3.0, -3.0, 3.0]).expand(2, 1, 3)
        self.assertTrue(torch.equal(coefficients, expected))
        self.assertIsNone(readout.mlp)
        self.assertEqual(
            sum(parameter.numel() for parameter in readout.parameters()),
            0,
        )

    def test_charge_factorized_mode_learns_one_shared_amplitude(self):
        readout = LongRangeReadout(
            n_kernels=1,
            n_types=2,
            hidden_sizes=(),
            zero_init=False,
            charges=(2.0, -1.0),
        )
        with torch.no_grad():
            readout.mlp[-1].weight.zero_()
            readout.mlp[-1].bias.fill_(1.5)
        state = torch.ones(2, 3)

        coefficients = readout.coefficients(state)

        expected = torch.tensor([6.0, -3.0, 1.5]).expand(2, 1, 3)
        self.assertTrue(torch.equal(coefficients, expected))
        coefficients.sum().backward()
        self.assertIsNotNone(readout.mlp[-1].bias.grad)

    def test_charge_factorized_arguments_are_validated(self):
        with self.assertRaisesRegex(ValueError, "requires charges"):
            LongRangeReadout(n_kernels=1, coulomb_amplitude=1.0)
        with self.assertRaisesRegex(ValueError, "one value per type"):
            LongRangeReadout(n_kernels=1, n_types=2, charges=(1.0,))
        with self.assertRaisesRegex(ValueError, "requires n_kernels=1"):
            LongRangeReadout(
                n_kernels=2,
                n_types=2,
                charges=(1.0, -1.0),
            )


if __name__ == "__main__":
    unittest.main()
