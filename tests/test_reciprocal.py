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
        for kernel in ("gaussian", "screened_inverse_laplacian"):
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


if __name__ == "__main__":
    unittest.main()
