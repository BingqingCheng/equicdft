"""A polarization kernel whose cooperative excess follows the local alignment.

The bulk linear response of water (S(k), g_K) needs a transverse excess of
about -0.7 ideal units: the H-bonded, unpolarized liquid turns as a network.
A kernel quadratic in P keeps that excess at every alignment. The uniform-field
SPC/E responses show it growing with alignment like the Langevin ideal free
energy itself up to u ~ 0.5 and fading mildly beyond: ``alignment_scaling="ideal"``
applies the kernels to Q = c(u) P with c^2 = 2 f_id(u)/(3 u^2), and
``saturation="gaussian"`` multiplies by s(u), s^2 = exp(-a u^2), with a learned
a > 0. Both leave the second variation at P = 0 unchanged; the contributions at
alignment u follow closed forms checked here.
"""

import math
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from equicdft import (
    FixedDipoleIdeal, GridCACEModel, GridData, LongRangeReadout, ReciprocalFeatures,
)
from equicdft._config import build
from equicdft.functional.reciprocal import _ideal_alignment_weight_squared
from equicdft.response.bulk import bulk_response_symbol

MOMENT = 0.48937
RHO = 1092.0 / 32.0 ** 3
K_IDEAL = 3.0 / (RHO * MOMENT ** 2)
COULOMB_AMPLITUDE = 18.74 * K_IDEAL / (4.0 * math.pi)
SPACING = 2.0
GRID = 16
LENGTH = GRID * SPACING
ALPHA = 2.0
SOFTENING = -0.7     # vector coefficient in ideal units


def kernel_features(exponent, ideal=False, **kwargs):
    options = {}
    if exponent is not None:
        options.update(saturation="gaussian", saturation_initial=exponent)
    if ideal:
        options.update(alignment_scaling="ideal")
    if options:
        options.update(dipole_magnitude=MOMENT)
    return ReciprocalFeatures(
        radial_exponents=(ALPHA,), kernel="gaussian", n_types=1, variable="dipole_density",
        density_floor=1.0e-8, **options, **kwargs,
    )


def kernel_readout(exponent, ideal=False):
    return LongRangeReadout(
        n_types=1, hidden_sizes=(), zero_init=True, features=kernel_features(exponent, ideal),
        coefficient_scale=(K_IDEAL,), initial_coefficients=(K_IDEAL * SOFTENING,),
    )


def model(exponent, with_kernel=True, ideal=False, derivative=False):
    coulomb = LongRangeReadout(
        n_kernels=1, n_types=1, charges=(0.0,), include_polarization=True,
        coulomb_amplitude=COULOMB_AMPLITUDE,
        features=ReciprocalFeatures(radial_exponents=(0.0,), kernel="coulomb", n_types=1),
    )
    readouts = [coulomb] + ([kernel_readout(exponent, ideal)] if with_kernel else [])
    return GridCACEModel(
        readout=readouts, ideal=FixedDipoleIdeal(MOMENT), polarization_scale=0.0025,
        mean_density=RHO, grid_spacing=SPACING, mean_temperature=1.0, boltzmann_constant=1.0,
        compute_c1=False, compute_polarization_derivative=derivative, rho_min=1.0e-4,
    ).to(dtype=torch.float64).eval()


def positions():
    return torch.tensor(np.indices((GRID,) * 3).reshape(3, -1).T, dtype=torch.float64) * SPACING


def frame(fitted, rho, polarization):
    data = dict(GridData.from_dict(
        {"grid_size": [GRID] * 3, "grid_spacing": fitted.grid_info["grid_spacing"],
         "temperature": 1.0, "n_types": 1},
        grid_info=fitted.grid_info, boltzmann_constant=1.0, include_local_density_index=False,
    ))
    data["rho"] = rho[:, None].clone()
    data["dipole_density"] = polarization[:, None, :].clone()
    return data


def energy(fitted, rho, polarization):
    batch = next(iter(DataLoader([frame(fitted, rho, polarization)], batch_size=1)))
    with torch.no_grad():
        return fitted(batch)["beta_F"].item()


def curvature(fitted, rho, base, direction, amplitude=2.0e-5):
    """Second derivative along ``direction * cos(2 pi x / L)``, per K_ideal at rho_ref."""

    profile = torch.cos(2 * math.pi / LENGTH * positions()[:, 0])
    probe = amplitude * profile[:, None] * direction[None]
    second = energy(fitted, rho, base + probe) + energy(fitted, rho, base - probe) - 2 * energy(fitted, rho, base)
    return second / (amplitude ** 2 * (profile ** 2).sum().item() * SPACING ** 3) / K_IDEAL


def uniform_state(alignment):
    rho = torch.full((GRID ** 3,), RHO)
    base = alignment * MOMENT * RHO * torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)[None].repeat(GRID ** 3, 1)
    return rho, base


def kernel_curvature(exponent, alignment, direction, ideal=False):
    """Kernel contribution alone: model with the kernel minus the same model without."""

    rho, base = uniform_state(alignment)
    return (curvature(model(exponent, ideal=ideal), rho, base, direction)
            - curvature(model(exponent, with_kernel=False), rho, base, direction))


def weight_derivatives(u, exponent=None, ideal=False):
    """``c(u) s(u)`` and its first two u-derivatives, by autograd."""

    def weight(value):
        square = value * value
        total = torch.ones_like(square)
        if ideal:
            total = total * _ideal_alignment_weight_squared(square)
        if exponent is not None:
            total = total * torch.exp(-exponent * square)
        return total.sqrt()

    value = torch.tensor(u, dtype=torch.float64, requires_grad=True)
    w = weight(value)
    (first,) = torch.autograd.grad(w, value, create_graph=True)
    (second,) = torch.autograd.grad(first, value)
    return float(w.detach()), float(first.detach()), float(second.detach())


def closed_forms(u, a_k, a_zero, exponent=None, ideal=False):
    """Kernel transverse and longitudinal stiffness at uniform alignment u."""

    c, c1, c2 = weight_derivatives(u, exponent, ideal)
    transverse = a_k * c * c + a_zero * u * c * c1
    longitudinal = a_k * (c + u * c1) ** 2 + a_zero * (2 * u * c * c1 + u * u * c * c2)
    return transverse, longitudinal


class TestSaturatingKernel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)

    @classmethod
    def tearDownClass(cls):
        torch.set_default_dtype(cls._dtype)

    def test_bulk_linear_response_is_unchanged_for_any_exponent(self):
        indices = torch.tensor([[[1, 0, 0], [1, 1, 0], [2, 1, 1]]])
        rho = torch.full((GRID ** 3,), RHO)
        zero = torch.zeros((GRID ** 3, 3), dtype=torch.float64)
        reference = None
        for exponent, ideal in ((None, False), (0.5, False), (3.0, False), (None, True), (1.0, True)):
            fitted = model(exponent, ideal=ideal)
            batch = next(iter(DataLoader([frame(fitted, rho, zero)], batch_size=1)))
            symbol = bulk_response_symbol(fitted, batch, indices).detach()
            if reference is None:
                reference = symbol
            else:
                torch.testing.assert_close(symbol, reference, rtol=0, atol=1e-10)
        # The softening is present and transverse at P = 0: K_T = 1 + A(k)
        # (channel order rho, P_x, P_y, P_z; k along x, so P_y is transverse).
        k = 2 * math.pi / LENGTH
        self.assertAlmostEqual(
            float(reference[0, 0, 2, 2].real), 1.0 + SOFTENING * math.exp(-ALPHA * k ** 2), places=6,
        )

    def test_uniform_alignment_follows_the_closed_forms(self):
        k = 2 * math.pi / LENGTH
        a_k = SOFTENING * math.exp(-ALPHA * k ** 2)   # A(k); A(0) = SOFTENING
        transverse = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        longitudinal = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        for exponent in (0.7, 2.0):
            for u in (0.3, 0.6):
                s2 = math.exp(-exponent * u * u)
                expected_t = s2 * (a_k - exponent * u * u * SOFTENING)
                expected_l = s2 * (a_k * (1 - exponent * u * u) ** 2
                                   - SOFTENING * (3 * exponent * u * u - exponent ** 2 * u ** 4))
                # The explicit forms agree with the general c(u) s(u) expressions.
                general = closed_forms(u, a_k, SOFTENING, exponent)
                self.assertAlmostEqual(general[0], expected_t, places=10)
                self.assertAlmostEqual(general[1], expected_l, places=10)
                with self.subTest(exponent=exponent, alignment=u):
                    self.assertAlmostEqual(kernel_curvature(exponent, u, transverse), expected_t, places=4)
                    self.assertAlmostEqual(kernel_curvature(exponent, u, longitudinal), expected_l, places=4)
        # Ideal alignment scaling, alone and with a fading factor.
        for exponent in (None, 0.5):
            for u in (0.3, 0.7):
                expected_t, expected_l = closed_forms(u, a_k, SOFTENING, exponent, ideal=True)
                with self.subTest(exponent=exponent, alignment=u, scaling="ideal"):
                    self.assertAlmostEqual(kernel_curvature(exponent, u, transverse, ideal=True), expected_t, places=4)
                    self.assertAlmostEqual(kernel_curvature(exponent, u, longitudinal, ideal=True), expected_l, places=4)
        # Without saturation the kernel contributes A(k) at every alignment.
        self.assertAlmostEqual(kernel_curvature(None, 0.6, transverse), a_k, places=4)
        self.assertAlmostEqual(kernel_curvature(None, 0.6, longitudinal), a_k, places=4)
        # At u = 0.6 and a = 2 the softening has faded: the aligned liquid is stiffer.
        self.assertGreater(kernel_curvature(2.0, 0.6, transverse), kernel_curvature(None, 0.6, transverse) + 0.4)

    def test_ideal_scaling_makes_the_kernel_track_the_ideal_term_at_k_zero(self):
        # Uniform alignment: the kernel's k = 0 polarization derivative is A times
        # the ideal's, L^-1(u)/m, at every u, so the orientational free energy is
        # (1 + A) times the ideal one (the Kirkwood super-dipole picture).
        from equicdft.functional.polarization_ideal import inverse_langevin
        for u in (0.05, 0.3, 0.6, 0.78):
            rho, base = uniform_state(u)
            derivatives = []
            for with_kernel in (True, False):
                fitted = model(None, with_kernel=with_kernel, ideal=True, derivative=True)
                batch = next(iter(DataLoader([frame(fitted, rho, base)], batch_size=1)))
                with torch.no_grad():
                    derivatives.append(fitted(batch)["polarization_derivative"][0, :, 0, 0])
            kernel_part = (derivatives[0] - derivatives[1]).mean().item()
            expected = SOFTENING * float(inverse_langevin(torch.tensor(u))) / MOMENT
            with self.subTest(alignment=u):
                self.assertAlmostEqual(kernel_part / expected, 1.0, places=5)
        # The quadratic kernel instead keeps A * 3u/m: at u = 0.78 that is 1.95x too weak.
        rho, base = uniform_state(0.78)
        plain = []
        for with_kernel in (True, False):
            fitted = model(None, with_kernel=with_kernel, derivative=True)
            batch = next(iter(DataLoader([frame(fitted, rho, base)], batch_size=1)))
            with torch.no_grad():
                plain.append(fitted(batch)["polarization_derivative"][0, :, 0, 0])
        self.assertAlmostEqual((plain[0] - plain[1]).mean().item() / (SOFTENING * 3 * 0.78 / MOMENT), 1.0, places=5)

    def test_small_exponent_recovers_the_plain_kernel_and_gradients_flow(self):
        torch.manual_seed(3)
        plain = kernel_features(None)
        saturated = kernel_features(1.0e-6)
        grid, spacing = torch.tensor([GRID] * 3), torch.tensor([SPACING] * 3)
        rho = RHO * (1.0 + 0.2 * torch.rand(1, GRID ** 3, 1))
        polarization = 0.5 * MOMENT * rho[..., None] * torch.randn(1, GRID ** 3, 1, 3) / math.sqrt(3)
        torch.testing.assert_close(
            saturated(rho, grid, spacing, dipole_density=polarization),
            plain(rho, grid, spacing, dipole_density=polarization), rtol=1e-5, atol=0,
        )
        strong = kernel_features(2.0)
        rho = rho.clone().requires_grad_(True)
        features = strong(rho, grid, spacing, dipole_density=polarization)
        self.assertLess(float(features.sum().detach()), float(plain(rho, grid, spacing, dipole_density=polarization).sum().detach()))
        gradient_rho, gradient_a = torch.autograd.grad(features.sum(), (rho, strong.saturation_parameter))
        self.assertTrue(bool(torch.isfinite(gradient_rho).all()) and gradient_rho.abs().max() > 0)
        self.assertNotEqual(float(gradient_a), 0.0)
        # The exponent stays positive however far the raw parameter is pushed.
        with torch.no_grad():
            strong.saturation_parameter.fill_(-40.0)
        self.assertGreater(float(strong.saturation_exponent.detach()), 0.0)

    def test_configuration_state_and_validation(self):
        features = kernel_features(2.0)
        rebuilt = build(features.to_config())
        self.assertEqual(rebuilt.saturation, "gaussian")
        self.assertEqual(rebuilt.saturation_initial, 2.0)
        torch.testing.assert_close(rebuilt.dipole_magnitude, features.dipole_magnitude)
        self.assertNotIn("dipole_magnitude", features.state_dict())
        self.assertIn("saturation_parameter", features.state_dict())
        # A fitted exponent travels in the state dict, not the config.
        with torch.no_grad():
            features.saturation_parameter.fill_(1.5)
        rebuilt.load_state_dict(features.state_dict())
        torch.testing.assert_close(rebuilt.saturation_exponent, features.saturation_exponent)
        plain = kernel_features(None)
        self.assertNotIn("saturation_parameter", plain.state_dict())
        self.assertIsNone(plain.saturation_exponent)
        legacy = build({key: value for key, value in plain.to_config().items()
                        if key not in ("alignment_scaling", "saturation", "saturation_initial", "dipole_magnitude")})
        self.assertIsNone(legacy.saturation)
        self.assertIsNone(legacy.alignment_scaling)
        ideal = build(kernel_features(None, ideal=True).to_config())
        self.assertEqual(ideal.alignment_scaling, "ideal")
        self.assertNotIn("saturation_parameter", ideal.state_dict())
        for kwargs, message in (
            (dict(saturation="gaussian", dipole_magnitude=MOMENT, density_floor=0.0), "positive density_floor"),
            (dict(saturation="gaussian", density_floor=1e-3), "dipole_magnitude"),
            (dict(alignment_scaling="ideal", density_floor=1e-3), "dipole_magnitude"),
            (dict(saturation="gaussian", density_floor=1e-3, dipole_magnitude=MOMENT, saturation_initial=0.0), "positive"),
            (dict(saturation="langevin", density_floor=1e-3, dipole_magnitude=MOMENT), "saturation must be"),
            (dict(alignment_scaling="langevin", density_floor=1e-3, dipole_magnitude=MOMENT), "alignment_scaling must be"),
            (dict(dipole_magnitude=MOMENT), "requires alignment_scaling or saturation"),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                ReciprocalFeatures(radial_exponents=(1.0,), variable="dipole_density", **kwargs)
        with self.assertRaisesRegex(ValueError, "direct polarization"):
            ReciprocalFeatures(radial_exponents=(1.0,), saturation="gaussian", density_floor=1e-3, dipole_magnitude=MOMENT)


if __name__ == "__main__":
    unittest.main()
