"""A density-scaled polarization kernel, and a stability probe that sees dense layers.

A negative transverse kernel coefficient that makes bulk water at rho_ref
correctly soft (4 pi beta C/(eps - 1) < K_ideal) must not destabilize a denser
contact layer. With the kernel acting on P itself its contribution to the local
stiffness is independent of density, while the ideal restoring force falls as
rho_ref/rho, so a layer at 3 rho_ref goes unstable. Acting on
Q = P (rho_ref/rho)^(1/2) the kernel scales like the ideal term and the layer
stays stable. A box-wide probe wave averages over the box and misses the
unstable layer; the density-weighted probe finds it.
"""

import math
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from equicdft import (
    FixedDipoleIdeal, FourierStabilityLoss, GridCACEModel, GridData,
    LongRangeReadout, ReciprocalFeatures,
)
from equicdft._config import build
from equicdft.response.curvature import polarization_fourier_curvature

MOMENT = 0.48937
RHO = 1092.0 / 32.0 ** 3
K_IDEAL = 3.0 / (RHO * MOMENT ** 2)
COULOMB_AMPLITUDE = 18.74 * K_IDEAL / (4.0 * math.pi)
SPACING = 2.0
GRID = 16
LENGTH = GRID * SPACING
SOFTENING = -0.8   # transverse kernel coefficient in ideal units: bulk T ~ 1 - 0.8 = 0.2


def kernel_readout(exponent):
    return LongRangeReadout(
        n_types=1, hidden_sizes=(), zero_init=True,
        features=ReciprocalFeatures(
            radial_exponents=(2.0,), kernel="gaussian", n_types=1, variable="dipole_density",
            density_exponent=exponent,
            reference_density=RHO if exponent else None,
            density_floor=1.0e-4 if exponent else 0.0,
        ),
        coefficient_scale=(K_IDEAL,), initial_coefficients=(K_IDEAL * SOFTENING,),
    )


def model(exponent):
    coulomb = LongRangeReadout(
        n_kernels=1, n_types=1, charges=(0.0,), include_polarization=True,
        coulomb_amplitude=COULOMB_AMPLITUDE,
        features=ReciprocalFeatures(radial_exponents=(0.0,), kernel="coulomb", n_types=1),
    )
    return GridCACEModel(
        readout=[coulomb, kernel_readout(exponent)], ideal=FixedDipoleIdeal(MOMENT),
        polarization_scale=0.0025, mean_density=RHO, grid_spacing=SPACING,
        mean_temperature=1.0, boltzmann_constant=1.0, compute_c1=False,
        compute_polarization_derivative=False, rho_min=1.0e-4,
    ).to(dtype=torch.float64).eval()


def positions():
    return torch.tensor(np.indices((GRID,) * 3).reshape(3, -1).T, dtype=torch.float64) * SPACING


def frame(fitted, rho, polarization=None):
    data = dict(GridData.from_dict(
        {"grid_size": [GRID] * 3, "grid_spacing": fitted.grid_info["grid_spacing"],
         "temperature": 1.0, "n_types": 1},
        grid_info=fitted.grid_info, boltzmann_constant=1.0, include_local_density_index=False,
    ))
    data["rho"] = rho[:, None].clone()
    data["dipole_density"] = (
        torch.zeros((GRID ** 3, 1, 3), dtype=torch.float64) if polarization is None
        else polarization[:, None, :]
    )
    return data


def layered_density(peak):
    """Mean rho_ref, one Gaussian layer of the given peak/rho_ref at x = 16 A."""

    x = positions()[:, 0]
    distance = torch.minimum((x - 16.0).abs(), LENGTH - (x - 16.0).abs())
    rho = RHO * (1.0 + (peak - 1.0) * torch.exp(-0.5 * (distance / 2.0) ** 2))
    return rho * (RHO / rho.mean())


def energy(fitted, rho, polarization=None):
    batch = next(iter(DataLoader([frame(fitted, rho, polarization)], batch_size=1)))
    with torch.no_grad():
        return fitted(batch)["beta_F"].item()


def transverse_curvature(fitted, rho, profile, amplitude=1.0e-4):
    """Curvature of P_y = A profile(x) (divergence-free), per ideal stiffness at rho_ref."""

    probe = amplitude * profile[:, None] * torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)[None]
    second = energy(fitted, rho, probe) + energy(fitted, rho, -probe) - 2.0 * energy(fitted, rho)
    return second / (amplitude ** 2 * (profile ** 2).sum().item() * SPACING ** 3) / K_IDEAL


def local_profile(centre, width=3.0):
    x = positions()[:, 0]
    distance = torch.minimum((x - centre).abs(), LENGTH - (x - centre).abs())
    return torch.exp(-0.5 * (distance / width) ** 2)


class TestDensityScaledKernel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)

    @classmethod
    def tearDownClass(cls):
        torch.set_default_dtype(cls._dtype)

    def test_features_match_at_the_reference_density_and_scale_with_it(self):
        plain = ReciprocalFeatures(radial_exponents=(1.0, 4.0), variable="dipole_density", include_divergence=True)
        scaled = ReciprocalFeatures(
            radial_exponents=(1.0, 4.0), variable="dipole_density", include_divergence=True,
            density_exponent=0.5, reference_density=RHO, density_floor=1.0e-8,
        )
        torch.manual_seed(2)
        polarization = 1.0e-3 * torch.randn(1, GRID ** 3, 1, 3)
        grid = torch.tensor([GRID] * 3); spacing = torch.tensor([SPACING] * 3)
        at_reference = torch.full((1, GRID ** 3, 1), RHO)
        torch.testing.assert_close(
            scaled(at_reference, grid, spacing, dipole_density=polarization),
            plain(at_reference, grid, spacing, dipole_density=polarization),
        )
        # Uniformly twice as dense: every feature is quadratic in Q = P/sqrt(2).
        torch.testing.assert_close(
            scaled(2.0 * at_reference, grid, spacing, dipole_density=polarization),
            0.5 * plain(at_reference, grid, spacing, dipole_density=polarization),
        )
        # The weight is smooth in rho, so the features feed the density derivative.
        rho = (RHO * (1.0 + 0.2 * torch.rand(1, GRID ** 3, 1))).requires_grad_(True)
        features = scaled(rho, grid, spacing, dipole_density=polarization)
        gradient, = torch.autograd.grad(features.sum(), rho)
        self.assertTrue(bool(torch.isfinite(gradient).all()) and gradient.abs().max() > 0)

    def test_kernel_stiffness_follows_the_ideal_density_scaling(self):
        profile = torch.cos(2 * math.pi / LENGTH * positions()[:, 0])
        kernel = SOFTENING * math.exp(-2.0 * (2 * math.pi / LENGTH) ** 2)
        for exponent, expected in ((0.0, lambda f: 1.0 / f + kernel),
                                   (0.5, lambda f: (1.0 + kernel) / f)):
            fitted = model(exponent)
            for fraction in (0.5, 1.0, 2.0):
                with self.subTest(exponent=exponent, density=fraction):
                    rho = torch.full((GRID ** 3,), fraction * RHO)
                    self.assertAlmostEqual(
                        transverse_curvature(fitted, rho, profile), expected(fraction), places=3,
                    )

    def test_dense_layer_is_stable_only_with_the_density_scaled_kernel(self):
        rho = layered_density(3.0)
        in_layer, in_bulk = local_profile(16.0, 2.0), local_profile(0.0)
        box_wave = torch.cos(2 * math.pi / LENGTH * positions()[:, 0])
        density_independent, density_scaled = model(0.0), model(0.5)
        # Both are equally soft in the bulk and pass a box-wide test ...
        for fitted in (density_independent, density_scaled):
            self.assertGreater(transverse_curvature(fitted, rho, in_bulk), 0.2)
            self.assertGreater(transverse_curvature(fitted, rho, box_wave), 0.2)
        # ... but only the density-independent kernel makes the layer unstable.
        self.assertLess(transverse_curvature(density_independent, rho, in_layer), -0.1)
        self.assertGreater(transverse_curvature(density_scaled, rho, in_layer), 0.1)

    def test_density_weighted_probe_finds_the_unstable_layer(self):
        rho = layered_density(3.0)
        curvatures, penalties = {}, {}
        for exponent in (0.0, 0.5):
            fitted = model(exponent)
            batch = next(iter(DataLoader([frame(fitted, rho)], batch_size=1)))
            with torch.no_grad():
                outputs = fitted(batch)
            for power in (0.0, 2.0):
                # Both phases of the box-wide transverse wave along x.
                curvature, valid = polarization_fourier_curvature(
                    fitted, outputs, batch, batch["rho"], torch.tensor([[[1, 0, 0]]]), 1.0e-3,
                    torch.tensor([[0.0, 1.0, 0.0]]), density_power=power,
                )
                self.assertTrue(bool(valid.all()))
                curvatures[exponent, power] = curvature.min().item()
                term = FourierStabilityLoss(
                    wavevector_indices=((1, 0, 0),), variable="dipole_density", relative_amplitude=1.0e-3,
                    include_zero_wavevector=False, polarization_direction="transverse",
                    minimum_curvature=0.0, probe_density_power=power,
                )
                term.train()
                penalties[exponent, power] = term(outputs, batch, model=fitted).item()
        # The plain probe (already weighted by rho through the alignment
        # displacement) barely registers the unstable layer; weighting by
        # (rho/max rho)^2 concentrates the wave in it.
        self.assertGreater(curvatures[0.0, 0.0], -0.05)
        self.assertLess(curvatures[0.0, 2.0], -0.15)
        self.assertGreater(penalties[0.0, 2.0], 20.0 * penalties[0.0, 0.0])
        # The density-scaled kernel is stable under both probes.
        self.assertGreater(curvatures[0.5, 0.0], 0.2)
        self.assertGreater(curvatures[0.5, 2.0], 0.2)
        self.assertEqual(penalties[0.5, 0.0], 0.0)
        self.assertEqual(penalties[0.5, 2.0], 0.0)
        with self.assertRaisesRegex(ValueError, "probe_density_power"):
            FourierStabilityLoss(wavevector_indices=((1, 0, 0),), variable="dipole_density", probe_density_power=-1.0)
        with self.assertRaisesRegex(ValueError, "probe_density_power"):
            FourierStabilityLoss(wavevector_indices=((1, 0, 0),), probe_density_power=2.0)

    def test_configuration_and_validation(self):
        features = ReciprocalFeatures(
            radial_exponents=(1.0,), variable="dipole_density",
            density_exponent=0.5, reference_density=RHO, density_floor=1.0e-3,
        )
        rebuilt = build(features.to_config())
        self.assertEqual(rebuilt.density_exponent, 0.5)
        self.assertEqual(rebuilt.density_floor, 1.0e-3)
        torch.testing.assert_close(rebuilt.reference_density, features.reference_density)
        self.assertNotIn("reference_density", features.state_dict())
        legacy = build({key: value for key, value in features.to_config().items()
                        if key not in ("density_exponent", "reference_density", "density_floor")})
        self.assertEqual(legacy.density_exponent, 0.0)
        for kwargs, message in (
            (dict(density_exponent=0.5, reference_density=RHO), "positive density_floor"),
            (dict(density_exponent=0.5, density_floor=1e-3), "reference_density"),
            (dict(density_exponent=-0.5), "nonnegative"),
            (dict(reference_density=RHO), "requires density_exponent"),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                ReciprocalFeatures(radial_exponents=(1.0,), variable="dipole_density", **kwargs)
        with self.assertRaisesRegex(ValueError, "direct polarization"):
            ReciprocalFeatures(radial_exponents=(1.0,), density_exponent=0.5, reference_density=RHO, density_floor=1e-3)


if __name__ == "__main__":
    unittest.main()
