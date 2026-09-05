"""Grid-domain and alias checks for Fourier stability and response probes."""

import itertools
import math
import unittest

import torch
from torch import nn

from equicdft import FourierResponse, FourierStabilityLoss
from equicdft._fourier import (
    canonical_grid_mode,
    canonical_mode_triplets,
    feasible_modes,
)


class _QuadraticModel(nn.Module):
    """A local excess functional whose bulk curvature is 1 + a * rho."""

    def __init__(self):
        super().__init__()
        self.coefficient = nn.Parameter(torch.tensor(-4.0, dtype=torch.float64))
        self.last_rho = None

    def forward(self, batch, compute_c1=False):
        rho = batch["rho"]
        self.last_rho = rho.detach()
        return {
            "beta_F_exc": (
                0.5 * self.coefficient * batch["grid_spacing"].prod(dim=-1)
                * rho.square().sum(dim=(-2, -1))
            )
        }


def _batch(size=(16, 16, 16), spacing=(0.5, 0.5, 0.5)):
    positions = torch.tensor(list(itertools.product(*(range(n) for n in size))))
    return {
        "rho": torch.full((1, positions.shape[0], 1), 0.5, dtype=torch.float64),
        "temperature": torch.ones(1, dtype=torch.float64),
        "grid_spacing": torch.tensor([spacing], dtype=torch.float64),
        "grid_size": torch.tensor([size]),
        "grid_positions": positions[None],
    }


class TestFourierModeDomains(unittest.TestCase):
    def test_cube_accepts_corner_outside_default_sphere(self):
        mode = ((6, 6, -5),)
        geometry = ((16, 16, 16), (0.5, 0.5, 0.5))
        with self.assertRaisesRegex(ValueError, "Nyquist sphere"):
            canonical_mode_triplets(mode, *geometry)
        selected = canonical_mode_triplets(mode, *geometry, mode_domain="cube")
        self.assertEqual(selected.tolist(), list(map(list, mode)))

    def test_cube_contains_each_nonzero_discrete_real_mode_once(self):
        for size in ((16, 16, 16), (5, 7, 3), (4, 5, 6)):
            with self.subTest(size=size):
                modes = feasible_modes(size, (0.5, 1.0, 2.0), mode_domain="cube")
                # A real wave is identified by the unordered pair of FFT
                # residues {n, -n}; this checks completeness independently
                # of the canonical representative chosen by the library.
                pairs = {
                    min(
                        tuple(n % axis for n, axis in zip(mode, size)),
                        tuple(-n % axis for n, axis in zip(mode, size)),
                    )
                    for mode in modes
                }
                self_conjugate = math.prod(2 if n % 2 == 0 else 1 for n in size)
                expected = (math.prod(size) + self_conjugate) // 2 - 1
                self.assertEqual(len(modes), expected)
                self.assertEqual(len(pairs), expected)
                self.assertNotIn((0, 0, 0), pairs)

    def test_nyquist_and_global_sign_aliases_are_canonicalized(self):
        size = (16, 16, 16)
        for mode in ((8, -1, -8), (-1, 8, 0), (-8, -8, 0), (6, 6, -5)):
            with self.subTest(mode=mode):
                canonical = canonical_grid_mode(mode, size)
                opposite = canonical_grid_mode(tuple(-n for n in mode), size)
                self.assertEqual(canonical, opposite)
                self.assertEqual(canonical_grid_mode(canonical, size), canonical)
        self.assertEqual(canonical_grid_mode((8, -1, -8), size), (8, 1, 8))
        with self.assertRaisesRegex(ValueError, "equivalent"):
            canonical_mode_triplets(
                ((8, -1, -8), (-8, 1, 8)),
                size,
                (0.5, 0.5, 0.5),
                mode_domain="cube",
            )

    def test_cube_still_rejects_zero_and_out_of_band_aliases(self):
        for size, mode in (
            ((16, 16, 16), (9, 0, 0)),
            ((16, 16, 16), (0, -9, 0)),
            ((16, 16, 16), (16, 0, 0)),
            ((5, 7, 3), (3, 0, 0)),
        ):
            with self.subTest(size=size, mode=mode):
                with self.assertRaisesRegex(ValueError, "Nyquist cube"):
                    canonical_mode_triplets(
                        (mode,), size, (1.0, 1.0, 1.0), mode_domain="cube",
                    )
        with self.assertRaisesRegex(ValueError, "zero mode"):
            canonical_mode_triplets(
                ((0, 0, 0),), (16, 16, 16), (1.0, 1.0, 1.0), mode_domain="cube",
            )

    def test_anisotropic_cube_uses_componentwise_not_isotropic_limit(self):
        size, spacing = (8, 6, 5), (0.5, 1.0, 2.0)
        cube = feasible_modes(size, spacing, mode_domain="cube")
        sphere = feasible_modes(size, spacing)
        self.assertIn((4, 3, 2), cube)
        self.assertNotIn((4, 3, 2), sphere)
        self.assertTrue(set(sphere) < set(cube))
        self.assertEqual(cube, feasible_modes(size, (1.0, 1.0, 1.0), "cube"))
        self.assertEqual(sphere, feasible_modes(size, spacing, "sphere"))

    def test_random_cube_sampling_and_physical_wavevector_filter(self):
        batch = _batch(size=(8, 8, 8), spacing=(1.0, 1.0, 1.0))
        candidates = feasible_modes((8, 8, 8), (1.0, 1.0, 1.0), "cube")
        term = FourierStabilityLoss(
            random_modes_per_field=len(candidates), mode_domain="cube",
        )
        selected = term._select_modes(batch, batch["rho"])[0]
        self.assertEqual(sorted(map(tuple, selected.tolist())), candidates)
        # This entire band is outside the isotropic sphere |k| <= pi.
        term = FourierStabilityLoss(
            random_modes_per_field=4,
            mode_domain="cube",
            wavevector_range=(4.0, 4.5),
        )
        selected = term._select_modes(batch, batch["rho"])
        magnitudes = (2.0 * torch.pi * selected.to(torch.float64) / 8).norm(dim=-1)
        self.assertTrue(torch.all((magnitudes >= 4.0) & (magnitudes <= 4.5)))

    def test_cube_response_scalar_and_matrix_have_expected_normalization(self):
        batch = _batch()
        model = _QuadraticModel()
        modes = torch.tensor([[[6, 6, -5]]])
        response = FourierResponse(relative_amplitude=2.0**-8, mode_domain="cube")
        curvature, valid = response(
            model, batch, modes, directions=torch.ones(1, 1, dtype=torch.float64),
        )
        matrix, active = response.matrix(model, batch, modes)
        self.assertEqual(curvature.shape, (1, 1, 2, 1))
        self.assertEqual(matrix.shape, (1, 1, 2, 1, 1))
        self.assertTrue(torch.all(valid))
        self.assertTrue(torch.all(active))
        torch.testing.assert_close(curvature, -torch.ones_like(curvature), atol=1e-4, rtol=0)
        torch.testing.assert_close(matrix[..., 0], curvature, atol=1e-8, rtol=0)
        with self.assertRaisesRegex(ValueError, "Nyquist sphere"):
            FourierResponse().matrix(model, batch, modes)

    def test_cube_stability_backward_and_fixed_particle_number(self):
        batch = _batch()
        batch["rho"][:, ::2] *= 0.8
        model = _QuadraticModel()
        term = FourierStabilityLoss(modes=((6, 6, -5),), mode_domain="cube")
        value = term(model(batch), batch, model=model)
        value.backward()
        self.assertGreater(value.item(), 0.0)
        self.assertTrue(torch.isfinite(model.coefficient.grad))
        self.assertNotEqual(model.coefficient.grad.item(), 0.0)
        self.assertGreaterEqual(model.last_rho.min().item(), 0.0)
        reference = batch["rho"].sum(dim=-2)[:, None]
        torch.testing.assert_close(
            model.last_rho.sum(dim=-2), reference.expand(1, 4, 1), atol=1e-10, rtol=0,
        )

    def test_self_conjugate_corner_keeps_cosine_and_masks_sine(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                batch = {
                    key: value.to(dtype) if value.is_floating_point() else value
                    for key, value in _batch().items()
                }
                model = _QuadraticModel().to(dtype)
                curvature, valid = FourierResponse(mode_domain="cube")(
                    model, batch, torch.tensor([[[8, 8, 8]]]),
                    directions=torch.ones(1, 1, dtype=dtype),
                )
                self.assertEqual(valid.flatten().tolist(), [True, False])
                self.assertTrue(torch.all(torch.isfinite(curvature)))

    def test_mode_domain_is_validated(self):
        for mode_domain in (None, "full", "Cube"):
            with self.subTest(mode_domain=mode_domain):
                with self.assertRaisesRegex(ValueError, "mode_domain"):
                    FourierResponse(mode_domain=mode_domain)
                with self.assertRaisesRegex(ValueError, "mode_domain"):
                    FourierStabilityLoss(
                        random_modes_per_field=1, mode_domain=mode_domain,
                    )


if __name__ == "__main__":
    unittest.main()
