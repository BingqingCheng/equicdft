"""A longitudinal/transverse-resolved dielectric kernel from the existing pieces.

Ideal dipoles + the exact nonzero-mode dipolar Coulomb term + a
``LongRangeReadout`` over direct-polarization Gaussian features with the
divergence block give the tensor kernel ``A(k) I + B(k) k k``. With
``coefficient_scale`` set to the ideal alignment stiffness and
``initial_coefficients`` chosen from a dielectric constant and a screening
function, the bulk polarization stiffness is anisotropic and non-monotonic in
``k`` — the shape a dipolar liquid needs and no local term can express — and
the transverse/longitudinal stability probes read it back.
"""

import math
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from equicdft import (
    FixedDipoleIdeal,
    FourierStabilityLoss,
    GridCACEModel,
    GridData,
    LongRangeReadout,
    ReciprocalFeatures,
)
from equicdft._config import build
from equicdft.response.curvature import polarization_fourier_curvature

# Water-like reduced units: beta = 1, moment m, density rho, Coulomb amplitude
# beta*C chosen so that 4 pi beta C / K_ideal = 18.74 as for SPC/E at 298 K.
MOMENT = 0.48937
RHO = 1092.0 / 32.0 ** 3
K_IDEAL = 3.0 / (RHO * MOMENT ** 2)
COULOMB_AMPLITUDE = 18.74 * K_IDEAL / (4.0 * math.pi)
SPACING = 2.0
ALPHA = (1.0, 2.5, 8.0)  # squared lengths of the Gaussian kernels

# Reference coefficients in ideal-stiffness units, the least-squares
# projection on these exponents of a water-like target: a transverse excess
# of -0.74 (eps = 71) up to k ~ 0.5 decaying to zero by k ~ 1.2, and a
# longitudinal excess that follows the measured screening at small k and
# dips to about -2.6 near k = 1 (overscreening). Vector channels first.
VECTOR = (-0.258, -0.739, 0.227)
DIVERGENCE = (-8.073, 6.586, 16.429)


def learned_excess(k):
    """Return the intended (transverse, longitudinal) learned excess at ``k`` in ideal units."""

    transverse = sum(a * math.exp(-alpha * k ** 2) for a, alpha in zip(VECTOR, ALPHA))
    longitudinal = transverse + sum(
        d * k ** 2 * math.exp(-alpha * k ** 2) for d, alpha in zip(DIVERGENCE, ALPHA)
    )
    return transverse, longitudinal


def dielectric_readout():
    return LongRangeReadout(
        n_types=1,
        hidden_sizes=(16, 8),
        zero_init=True,
        features=ReciprocalFeatures(
            radial_exponents=ALPHA,
            kernel="gaussian",
            n_types=1,
            variable="dipole_density",
            include_divergence=True,
        ),
        coefficient_scale=(K_IDEAL,) * 6,
        initial_coefficients=tuple(K_IDEAL * value for value in VECTOR + DIVERGENCE),
    )


def coulomb_readout():
    return LongRangeReadout(
        n_kernels=1,
        n_types=1,
        charges=(0.0,),
        include_polarization=True,
        coulomb_amplitude=COULOMB_AMPLITUDE,
        features=ReciprocalFeatures(radial_exponents=(0.0,), kernel="coulomb", n_types=1),
    )


def model(readouts):
    return GridCACEModel(
        readout=readouts,
        ideal=FixedDipoleIdeal(MOMENT),
        polarization_scale=0.0025,
        mean_density=RHO,
        grid_spacing=SPACING,
        mean_temperature=1.0,
        boltzmann_constant=1.0,
        compute_c1=False,
        compute_polarization_derivative=False,
        rho_min=1.0e-4,
    ).to(dtype=torch.float64).eval()


def bulk_frame(fitted, grid, polarization=None):
    frame = dict(GridData.from_dict(
        {"grid_size": [grid, grid, grid], "grid_spacing": fitted.grid_info["grid_spacing"],
         "temperature": 1.0, "n_types": 1},
        grid_info=fitted.grid_info, boltzmann_constant=1.0, include_local_density_index=False,
    ))
    frame["rho"] = torch.full((grid ** 3, 1), RHO, dtype=torch.float64)
    frame["dipole_density"] = (
        torch.zeros((grid ** 3, 1, 3), dtype=torch.float64) if polarization is None
        else polarization[:, None, :]
    )
    return frame


def energy(fitted, grid, polarization=None):
    batch = next(iter(DataLoader([bulk_frame(fitted, grid, polarization)], batch_size=1)))
    with torch.no_grad():
        return fitted(batch, compute_c1=False, compute_polarization_derivative=False)["beta_F"].item()


def stiffness(fitted, grid, n_wave, direction, amplitude=1.0e-4):
    """Curvature of a single cosine wave in units of the ideal stiffness."""

    length = grid * SPACING
    positions = torch.tensor(np.indices((grid, grid, grid)).reshape(3, -1).T, dtype=torch.float64) * SPACING
    wave = torch.cos(2 * math.pi * n_wave / length * positions[:, 0])
    probe = amplitude * wave[:, None] * torch.tensor(direction, dtype=torch.float64)[None]
    reference = energy(fitted, grid)
    second = energy(fitted, grid, probe) + energy(fitted, grid, -probe) - 2 * reference
    return 2 * second / (amplitude ** 2 * length ** 3) / K_IDEAL


class TestDielectricKernel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)

    @classmethod
    def tearDownClass(cls):
        torch.set_default_dtype(cls._default_dtype)

    def test_reference_coefficients_give_the_intended_anisotropic_stiffness(self):
        fitted = model([coulomb_readout(), dielectric_readout()])
        for grid, n_wave in ((32, 1), (16, 1), (16, 2), (16, 4), (16, 6)):
            k = 2 * math.pi * n_wave / (grid * SPACING)
            transverse, longitudinal = learned_excess(k)
            with self.subTest(k=round(k, 3)):
                self.assertAlmostEqual(stiffness(fitted, grid, n_wave, (0.0, 1.0, 0.0)), 1.0 + transverse, places=2)
                self.assertAlmostEqual(
                    stiffness(fitted, grid, n_wave, (1.0, 0.0, 0.0)), 1.0 + longitudinal + 18.74, places=2,
                )
        # Anisotropic and non-monotonic: the longitudinal stiffness dips below
        # the bare Coulomb value (overscreening) while the transverse one rises.
        self.assertLess(stiffness(fitted, 16, 5, (1.0, 0.0, 0.0)), 1.0 + 18.74)
        self.assertLess(stiffness(fitted, 16, 1, (0.0, 1.0, 0.0)), stiffness(fitted, 16, 6, (0.0, 1.0, 0.0)))

    def test_coefficient_scale_and_initial_coefficients_act_as_documented(self):
        readout = dielectric_readout().to(dtype=torch.float64)
        state = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
        expected = torch.tensor(VECTOR + DIVERGENCE, dtype=torch.float64) * K_IDEAL
        torch.testing.assert_close(readout.coefficients(state).reshape(-1), expected)
        with torch.no_grad():
            readout.mlp[-1].bias.fill_(0.5)
        torch.testing.assert_close(readout.coefficients(state).reshape(-1), expected + 0.5 * K_IDEAL)
        rebuilt = build(readout.to_config())
        torch.testing.assert_close(rebuilt.coefficient_scale, readout.coefficient_scale)
        torch.testing.assert_close(rebuilt.initial_coefficients, readout.initial_coefficients)
        self.assertNotIn("coefficient_scale", readout.state_dict())
        self.assertNotIn("initial_coefficients", readout.state_dict())
        legacy = build({key: value for key, value in readout.to_config().items()
                        if key not in ("coefficient_scale", "initial_coefficients")})
        self.assertIsNone(legacy.coefficient_scale)
        self.assertIsNone(legacy.initial_coefficients)
        with self.assertRaisesRegex(ValueError, "coefficient_scale"):
            LongRangeReadout(features=readout.features, coefficient_scale=(1.0,) * 5)
        with self.assertRaisesRegex(ValueError, "coefficient_scale"):
            LongRangeReadout(features=readout.features, coefficient_scale=0.0)
        with self.assertRaisesRegex(ValueError, "initial_coefficients"):
            LongRangeReadout(features=readout.features, initial_coefficients=(1.0,) * 4)
        with self.assertRaisesRegex(ValueError, "free kernel-coefficient mode"):
            LongRangeReadout(n_kernels=1, n_types=1, charges=(1.0,), coefficient_scale=2.0)

    def test_stability_probes_resolve_the_transverse_and_longitudinal_stiffness(self):
        fitted = model([coulomb_readout(), dielectric_readout()])
        grid = 16
        batch = next(iter(DataLoader([bulk_frame(fitted, grid)], batch_size=1)))
        outputs = fitted(batch, compute_c1=False, compute_polarization_derivative=False)
        k = 2 * math.pi * 2 / (grid * SPACING)
        transverse, longitudinal = learned_excess(k)
        # The probe's normalized curvature is the total stiffness in ideal units.
        indices = torch.tensor([[[2, 0, 0]]])
        for direction, expected in ((torch.tensor([[0.0, 1.0, 0.0]]), 1.0 + transverse),
                                    (torch.tensor([[1.0, 0.0, 0.0]]), 1.0 + longitudinal + 18.74)):
            curvature, valid = polarization_fourier_curvature(
                fitted, outputs, batch, batch["rho"], indices, 1.0e-3, direction.to(torch.float64),
            )
            self.assertTrue(valid[0, 0, 0].item())
            self.assertAlmostEqual(curvature[0, 0, 0].item(), expected, places=2)
        # Per-wave directions: transverse to (2,0,0) and (0,3,0) at once, and
        # the summed pattern of transverse waves is still transverse.
        indices = torch.tensor([[[2, 0, 0], [0, 3, 0]]])
        directions = torch.tensor([[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]], dtype=torch.float64)
        curvature, valid = polarization_fourier_curvature(
            fitted, outputs, batch, batch["rho"], indices, 1.0e-3, directions,
        )
        self.assertTrue(torch.all(valid[:, :, 0]).item())
        k3 = 2 * math.pi * 3 / (grid * SPACING)
        self.assertAlmostEqual(curvature[0, 0, 0].item(), 1.0 + transverse, places=2)
        self.assertAlmostEqual(curvature[0, 1, 0].item(), 1.0 + learned_excess(k3)[0], places=2)
        summed, valid = polarization_fourier_curvature(
            fitted, outputs, batch, batch["rho"], indices, 1.0e-3, directions,
            wavevector_phases=torch.zeros((1, 2), dtype=torch.float64),
        )
        self.assertTrue(valid[0, 0, 0].item())
        self.assertLess(summed[0, 0, 0].item(), 1.5)  # no longitudinal admixture
        # The hinge at a floor of 0.8: the transverse stiffness at this
        # wavevector is 0.60 and is penalized, the longitudinal 19 is not.
        penalties = {}
        for direction in ("transverse", "longitudinal"):
            term = FourierStabilityLoss(
                wavevector_indices=((2, 0, 0),), variable="dipole_density", relative_amplitude=1.0e-3,
                include_zero_wavevector=False, polarization_direction=direction, minimum_curvature=0.8,
            )
            term.train()
            penalties[direction] = term(outputs, batch, model=fitted).item()
        self.assertGreater(penalties["transverse"], 0.0)
        self.assertEqual(penalties["longitudinal"], 0.0)
        # Random wavevectors, the uniform wave included, independent and
        # superposed: every direction mode evaluates to a finite penalty.
        torch.manual_seed(3)
        for treatment in ("independent", "superposition"):
            for direction in ("random", "transverse", "longitudinal"):
                term = FourierStabilityLoss(
                    random_wavevectors_per_field=3, wavevector_treatment=treatment,
                    variable="dipole_density", relative_amplitude=1.0e-3,
                    polarization_direction=direction, minimum_curvature=0.8,
                )
                term.train()
                value = term(outputs, batch, model=fitted).item()
                self.assertTrue(math.isfinite(value) and value >= 0.0, (treatment, direction, value))
        with self.assertRaisesRegex(ValueError, "polarization_direction"):
            FourierStabilityLoss(wavevector_indices=((1, 0, 0),), variable="dipole_density",
                                 polarization_direction="sideways")
        with self.assertRaisesRegex(ValueError, "polarization_direction"):
            FourierStabilityLoss(wavevector_indices=((1, 0, 0),), polarization_direction="transverse")


if __name__ == "__main__":
    unittest.main()
