"""Nonnegative periodic averaging of grid fields."""

import itertools
import math
import unittest

import numpy as np
import torch

from equicdft import regularized_reduced_polarization
from equicdft.geometry.spectral import nonnegative_average


def step_moments(spacing, sigma):
    """Documented steps: whole steps of s_max^2 = 1 / (2 sum h^-2), then the rest."""

    widest = 0.5 / sum(h ** -2 for h in spacing)
    steps = max(1, math.ceil(sigma ** 2 / widest - 1e-12))
    return [widest] * (steps - 1) + [min(widest, sigma ** 2 - (steps - 1) * widest)]


def step_transfer(k, spacing, moment):
    """t(k) = 1 - 2 s^2 sum_i sin^2(k_i h_i / 2) / h_i^2 of one step."""

    return 1 - 2 * moment * sum(np.sin(ki * h / 2) ** 2 / h ** 2 for ki, h in zip(k, spacing))


def transfer(k, spacing, sigma):
    return np.prod([step_transfer(k, spacing, moment) for moment in step_moments(spacing, sigma)], axis=0)


def reference_average(field, shape, spacing, sigma):
    """Independent evaluation of the average as a Fourier multiplier."""

    axes = [2 * np.pi * np.fft.fftfreq(n, d=h) for n, h in zip(shape, spacing)]
    window = transfer(np.meshgrid(*axes, indexing="ij"), spacing, sigma)[None, ..., None]
    values = field.reshape(-1, *shape, field.shape[-1])
    spectrum = np.fft.fftn(values, axes=(1, 2, 3))
    return np.fft.ifftn(window * spectrum, axes=(1, 2, 3)).real.reshape(field.shape)


def zone_gaussian(field, shape, spacing, sigma):
    """The zone-restricted Gaussian exp(-sigma^2 k^2 / 2), which this average replaces."""

    axes = [2 * np.pi * np.fft.fftfreq(n, d=spacing) for n in shape]
    squared = sum(np.meshgrid(*[axis ** 2 for axis in axes], indexing="ij"))
    values = field.reshape(*shape, field.shape[-1])
    spectrum = np.fft.fftn(values, axes=(0, 1, 2))
    window = np.exp(-0.5 * sigma ** 2 * squared)[..., None]
    return np.fft.ifftn(window * spectrum, axes=(0, 1, 2)).real.reshape(field.shape)


def plane_wave_factor(index, n, h, sigma):
    """Eigenvalue of the average on cos(k . r), measured on an n^3 grid."""

    positions = np.indices((n,) * 3).reshape(3, -1).T * h
    k = 2 * np.pi * np.array(index) / (n * h)
    wave = torch.tensor(np.cos(positions @ k))[:, None]
    averaged = nonnegative_average(wave, (n, n, n), h, sigma)
    factor = float((averaged * wave).sum() / (wave * wave).sum())
    torch.testing.assert_close(averaged, factor * wave, rtol=0, atol=1e-13)
    return factor


class TestNonnegativeAverage(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(7)

    def tearDown(self):
        torch.set_default_dtype(self.dtype)

    def test_matches_its_transfer_function_on_odd_and_even_boxes(self):
        # sigma^2 / s_max^2 = 3.6: three whole steps and a remainder.
        shape, spacing, sigma = (5, 4, 6), (1.0, 1.3, 0.9), 0.8
        self.assertEqual(len(step_moments(spacing, sigma)), 4)
        field = torch.randn(2, 3, 5 * 4 * 6, 2)
        averaged = nonnegative_average(field, shape, spacing, sigma)
        self.assertEqual(averaged.shape, field.shape)
        np.testing.assert_allclose(
            averaged.numpy(), reference_average(field.numpy(), shape, spacing, sigma),
            rtol=0, atol=1e-13,
        )
        # Unbatched fields, per-field grid sizes and tensor geometry agree.
        single = nonnegative_average(field[0, 0], torch.tensor(shape), torch.tensor(spacing), sigma)
        torch.testing.assert_close(single, averaged[0, 0], rtol=0, atol=1e-14)
        rows = nonnegative_average(field[0], torch.tensor([shape] * 3), spacing, sigma)
        torch.testing.assert_close(rows, averaged[0], rtol=0, atol=1e-14)

    def test_kernel_is_nonnegative_normalized_symmetric_and_compact(self):
        n, h = 16, 2.0
        shape = (n, n, n)
        offsets = np.indices(shape)
        offsets = np.where(offsets > n // 2, offsets - n, offsets)       # minimum image
        reach = np.abs(offsets).sum(axis=0)
        delta = torch.zeros(n ** 3, 1)
        delta[0] = 1.0
        widest = h / math.sqrt(6)
        for sigma, steps in ((0.7, 1), (widest, 1), (1.0001 * widest, 2), (1.2, 3), (2.0, 6)):
            with self.subTest(sigma=sigma):
                kernel = nonnegative_average(delta, shape, h, sigma).reshape(shape)
                self.assertGreaterEqual(float(kernel.min()), 0.0)
                self.assertAlmostEqual(float(kernel.sum()), 1.0, places=14)
                # One voxel per step, exact zeros beyond.
                self.assertTrue(bool((kernel[torch.tensor(reach > steps)] == 0).all()))
                self.assertGreater(float(kernel[steps, 0, 0]), 0.0)
                # Second moment sigma^2 along every axis, no cross moment.
                weights = kernel.numpy()
                for axis in range(3):
                    self.assertAlmostEqual(
                        float((weights * (h * offsets[axis]) ** 2).sum()), sigma ** 2, places=12,
                    )
                self.assertAlmostEqual(float((weights * offsets[0] * offsets[1]).sum()), 0.0, places=14)
                # Inversion and the cubic group map the kernel onto itself.
                mirrored = torch.roll(torch.flip(kernel, dims=(0, 1, 2)), shifts=(1, 1, 1), dims=(0, 1, 2))
                torch.testing.assert_close(mirrored, kernel, rtol=0, atol=1e-16)
                for order in itertools.permutations(range(3)):
                    torch.testing.assert_close(kernel.permute(*order), kernel, rtol=0, atol=1e-16)
        # The documented one-step weights, 1 - 6a at the centre and
        # a = sigma^2 / (2 h^2) on the faces: one half and 1/12 at s_max.
        for sigma, face in ((0.7, 0.49 / 8), (widest, 1 / 12)):
            kernel = nonnegative_average(delta, shape, h, sigma).reshape(shape)
            self.assertAlmostEqual(float(kernel[0, 0, 0]), 1 - 6 * face, places=15)
            self.assertAlmostEqual(float(kernel[1, 0, 0]), face, places=15)
            self.assertEqual(float(kernel[1, 1, 0]), 0.0)
        # A constant field is unchanged and <a, G b> = <G a, b>.
        constant = torch.full((n ** 3, 2), 0.3)
        torch.testing.assert_close(nonnegative_average(constant, shape, h, 1.2), constant)
        a, b = torch.randn(n ** 3, 3), torch.randn(n ** 3, 3)
        self.assertAlmostEqual(
            float((a * nonnegative_average(b, shape, h, 1.2)).sum()),
            float((nonnegative_average(a, shape, h, 1.2) * b).sum()), places=12,
        )

    def test_plane_waves_are_eigenvectors_with_the_documented_transfer(self):
        n, h = 8, 2.0
        indices = ((1, 0, 0), (4, 0, 0), (4, 4, 0), (4, 4, 4), (3, -2, 1))
        for sigma in (0.7, h / math.sqrt(6), 1.2):
            for index in indices:
                k = 2 * np.pi * np.array(index) / (n * h)
                self.assertAlmostEqual(
                    plane_wave_factor(index, n, h, sigma), float(transfer(k, (h,) * 3, sigma)), places=13,
                )
        # At s_max = h / sqrt(6): 2/3 at the axis Nyquist, 1/3 on the [110]
        # edge and zero at the corner, the bounds 1 - (2, 4, 6) sigma^2 / h^2
        # of every nonnegative kernel with this second moment.
        for index, factor in (((4, 0, 0), 2 / 3), ((4, 4, 0), 1 / 3), ((4, 4, 4), 0.0)):
            self.assertAlmostEqual(plane_wave_factor(index, n, h, h / math.sqrt(6)), factor, places=15)

    def test_transfer_decreases_continuously_with_sigma(self):
        # Whole steps of s_max and a remainder: no jump where a step is added,
        # and the corner stays at zero once sigma >= s_max.
        n, h = 8, 2.0
        widest = h / math.sqrt(6)
        sigmas = sorted(set(np.linspace(0.2, 2.5, 24)) | {
            widest * (1 - 1e-7), widest, widest * (1 + 1e-7),
            math.sqrt(2) * widest * (1 - 1e-7), math.sqrt(2) * widest * (1 + 1e-7),
        })
        for index in ((1, 0, 0), (2, 1, 0), (4, 0, 0), (4, 4, 0), (4, 4, 4)):
            factors = [plane_wave_factor(index, n, h, sigma) for sigma in sigmas]
            self.assertTrue(all(0 <= value <= 1 + 1e-15 for value in factors))
            self.assertTrue(all(later <= earlier + 1e-15 for earlier, later in zip(factors, factors[1:])))
            for scale in (1.0, math.sqrt(2)):
                below, above = (plane_wave_factor(index, n, h, scale * widest * (1 + sign * 1e-7))
                                for sign in (-1, 1))
                self.assertLess(abs(above - below), 1e-6)
        for sigma in sigmas:
            if sigma >= widest:
                self.assertAlmostEqual(plane_wave_factor((4, 4, 4), n, h, sigma), 0.0, places=15)

    def test_admissible_fields_stay_admissible(self):
        # rho >= 0 and |P| <= m rho per species and voxel, with exact zeros,
        # six decades of density and saturated alignments.
        shape, spacing = (5, 6, 7), (0.9, 1.2, 1.0)
        moment = torch.tensor([0.48937, 0.3])
        n_grid = math.prod(shape)
        for sigma in (0.3, 0.8, 1.5):
            with self.subTest(sigma=sigma):
                rho = 10.0 ** (-6 + 5 * torch.rand(2, n_grid, 2))
                rho = torch.where(torch.rand_like(rho) < 0.3, torch.zeros_like(rho), rho)
                direction = torch.nn.functional.normalize(torch.randn(2, n_grid, 2, 3), dim=-1)
                alignment = torch.where(torch.rand_like(rho) < 0.3, torch.ones_like(rho), torch.rand_like(rho))
                dipole = (alignment * moment * rho)[..., None] * direction
                averaged = nonnegative_average(
                    torch.cat((rho, dipole.flatten(-2)), dim=-1), shape, spacing, sigma,
                )
                rho_bar, dipole_bar = averaged[..., :2], averaged[..., 2:].unflatten(-1, (2, 3))
                self.assertGreaterEqual(float(rho_bar.min()), 0.0)
                excess = dipole_bar.norm(dim=-1) - moment * rho_bar
                self.assertLessEqual(float(excess.max()), 1e-14 * float(rho.max()))
                inputs = regularized_reduced_polarization(rho_bar, dipole_bar, moment, 1e-5)
                self.assertLessEqual(float(inputs.norm(dim=-1).max()), 1 + 1e-12)

    def test_dense_dilute_interface_keeps_alignment_physical(self):
        # 0.09 and 0.001 A^-3 halves of a 16-voxel, 2 A periodic box with
        # alignment 0.8 along +x and -x; m = 0.48937 e A, floor 1e-5 A^-3.
        m, floor, h, shape = 0.48937, 1e-5, 2.0, (16, 3, 3)
        x = np.indices(shape)[0].reshape(-1)
        rho = np.where(x < 8, 0.09, 0.001)[:, None]
        dipole = np.zeros((x.size, 1, 3))
        dipole[:, 0, 0] = 0.8 * m * rho[:, 0] * np.where(x < 8, 1.0, -1.0)
        fields = np.concatenate((rho, dipole[:, 0]), axis=-1)
        probe = np.flatnonzero(x == 9)[0]                 # two voxels into the dilute half

        def alignment(averaged):
            averaged = torch.as_tensor(averaged)
            return regularized_reduced_polarization(
                averaged[:, :1], averaged[:, None, 1:], m, floor,
            ).norm(dim=-1)[:, 0]

        # The counterexample: the zone-restricted Gaussian (sigma = 0.6 A)
        # leaves 1.05e-4 of the dilute density, nearly doubles its polarization
        # and reports alignment 14.4.
        gaussian = zone_gaussian(fields, shape, h, 0.6)
        self.assertAlmostEqual(gaussian[probe, 0], 1.053e-4, delta=1e-7)
        self.assertAlmostEqual(abs(gaussian[probe, 1]), 7.496e-4, delta=1e-7)
        self.assertGreater(float(alignment(gaussian)[probe]), 14.0)

        for sigma in (0.6, h / math.sqrt(6)):
            with self.subTest(sigma=sigma):
                averaged = nonnegative_average(torch.tensor(fields), shape, h, sigma)
                # Densities stay inside the input range, alignments inside 0.8.
                self.assertGreaterEqual(float(averaged[:, 0].min()), 0.001 * (1 - 1e-14))
                self.assertLessEqual(float(averaged[:, 0].max()), 0.09 * (1 + 1e-14))
                self.assertLessEqual(float(alignment(averaged).max()), 0.8 * (1 + 1e-12))
                # One step reaches one voxel: the probe sees only dilute fluid.
                self.assertAlmostEqual(float(averaged[probe, 0]), 0.001, delta=1e-17)
                self.assertAlmostEqual(float(alignment(averaged)[probe]), 0.8, delta=1e-4)

    def test_arbitrary_row_order_follows_grid_positions(self):
        shape, spacing, sigma = (4, 5, 3), (1.1, 0.8, 1.4), 0.6
        positions = torch.tensor(np.indices(shape).reshape(3, -1).T)
        field = torch.randn(2, 60, 4)
        canonical = nonnegative_average(field, shape, spacing, sigma, grid_positions=positions)
        torch.testing.assert_close(canonical, nonnegative_average(field, shape, spacing, sigma), rtol=0, atol=0)
        # One permutation for both fields, then a different one per field.
        order = torch.randperm(60)
        permuted = nonnegative_average(field[:, order], shape, spacing, sigma, grid_positions=positions[order])
        torch.testing.assert_close(permuted, canonical[:, order], rtol=0, atol=1e-14)
        orders = torch.stack((torch.randperm(60), torch.randperm(60)))
        per_field = nonnegative_average(
            torch.stack([field[f, orders[f]] for f in range(2)]), shape, spacing, sigma,
            grid_positions=torch.stack([positions[orders[f]] for f in range(2)]),
        )
        torch.testing.assert_close(per_field, torch.stack([canonical[f, orders[f]] for f in range(2)]),
                                   rtol=0, atol=1e-14)

    def test_differentiable_and_validated(self):
        field = torch.randn(27, 2, requires_grad=True)
        weights = torch.randn(27, 2)
        gradient, = torch.autograd.grad(
            (weights * nonnegative_average(field, (3, 3, 3), 1.0, 0.5)).sum(), field,
        )
        # The average is its own adjoint, so the gradient is the averaged weight.
        torch.testing.assert_close(gradient, nonnegative_average(weights, (3, 3, 3), 1.0, 0.5))
        self.assertTrue(torch.autograd.gradcheck(
            lambda x: nonnegative_average(x, (3, 2, 2), (1.0, 0.7, 1.2), 0.9),
            torch.randn(12, 1, requires_grad=True),
        ))
        for sigma in (0.0, -0.5, float("inf"), float("nan")):
            with self.assertRaisesRegex(ValueError, "sigma"):
                nonnegative_average(field, (3, 3, 3), 1.0, sigma)
        for sigma in (True, "0.5", None):
            with self.assertRaises(TypeError):
                nonnegative_average(field, (3, 3, 3), 1.0, sigma)
        with self.assertRaisesRegex(ValueError, "grid_size"):
            nonnegative_average(field, (3, 3, 2), 1.0, 0.5)
        with self.assertRaises(TypeError):
            nonnegative_average(torch.ones(27, 1, dtype=torch.long), (3, 3, 3), 1.0, 0.5)
        with self.assertRaisesRegex(ValueError, "shape"):
            nonnegative_average(torch.ones(27), (3, 3, 3), 1.0, 0.5)
        with self.assertRaisesRegex(ValueError, "complete regular grid"):
            positions = torch.tensor(np.indices((3, 3, 3)).reshape(3, -1).T)
            positions[1] = positions[0]
            nonnegative_average(field.detach(), (3, 3, 3), 1.0, 0.5, grid_positions=positions)


if __name__ == "__main__":
    unittest.main()
