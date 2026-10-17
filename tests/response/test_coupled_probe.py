"""Joint density-polarization Fourier stability probe.

In a polarized state a functional can be stable against density-only waves
and against polarization-only waves and still be unstable against a wave
that moves both together, when the density-polarization coupling is large.
The coupled probe evaluates the 2 x 2 (per species) or 2n x 2n Hessian of
those channels in the ideal metric and penalizes its lowest eigenvalue.
"""

import math
import unittest

import numpy as np
import torch
from torch import nn

from equicdft import FixedDipoleIdeal, FourierStabilityLoss
from equicdft.response.curvature import (
    coupled_fourier_curvature_matrix,
    fourier_curvature_matrix,
    polarization_fourier_curvature,
)

GRID = 8
MOMENT = 0.5
WAVES = torch.tensor([[[1, 0, 0], [0, 0, 2]]])


class LocalPolar(nn.Module):
    """``beta f_exc = A rho^3 + B rho |P|^2 / m^2`` per voxel and species: a local cubic."""

    def __init__(self, a, b, n_types=1):
        super().__init__()
        self.a, self.b = a, b
        self.ideal = FixedDipoleIdeal([MOMENT] * n_types)

    def forward(self, batch, **kwargs):
        rho, pol = batch["rho"], batch["dipole_density"]
        volume = torch.prod(batch["grid_spacing"].to(rho), dim=-1)
        local = self.a * rho**3 + self.b * rho * pol.square().sum(-1) / MOMENT**2
        return {"beta_F_exc": volume * local.sum(dim=(-2, -1))}


def positions():
    return torch.tensor(np.indices((GRID,) * 3).reshape(3, -1).T)


def batch(rho, pol):
    return {
        "rho": rho[None], "dipole_density": pol[None], "grid_positions": positions()[None],
        "grid_size": torch.tensor([[GRID] * 3]), "grid_spacing": torch.ones(1, 3),
        "temperature": torch.ones(1),
    }


def uniform(rho0, alignment, n_types=1):
    rho = torch.full((GRID ** 3, n_types), rho0)
    pol = torch.zeros((GRID ** 3, n_types, 3))
    pol[..., 2] = alignment * MOMENT * rho0
    return rho, pol


def langevin_entropy(p_squared):
    """Exact per-particle orientational entropy of fixed dipoles, s(p)."""

    from equicdft.functional.polarization_ideal import inverse_langevin, log_sinhc
    p = p_squared.sqrt()
    xi = inverse_langevin(p)
    return xi * p - log_sinhc(xi)


def local_matrix(model, rho0, alignment, direction, exact=False):
    """Normalized local Hessian of excess + reference at a uniform state.

    For a uniform state and a local functional the probe matrix is the local
    Hessian in the coordinates delta rho = rho0 x w, delta P = m rho0 y w n,
    divided by the ideal norms rho0 and 3 rho0.
    """

    x = torch.tensor([rho0, 0.0, 0.0, alignment * MOMENT * rho0], requires_grad=True)

    def density(v):
        r, p = v[0], v[1:]
        excess = model.a * r**3 + model.b * r * p.square().sum() / MOMENT**2
        if exact:
            orientational = r * langevin_entropy(p.square().sum() / (MOMENT * r) ** 2)
        else:
            orientational = 1.5 * p.square().sum() / (MOMENT**2 * r)
        return excess + r * (torch.log(r) - 1) + orientational

    hessian = torch.autograd.functional.hessian(density, x)
    n = torch.tensor(direction, dtype=x.dtype)
    jacobian = torch.zeros(4, 2, dtype=x.dtype)
    jacobian[0, 0] = rho0
    jacobian[1:, 1] = MOMENT * rho0 * n
    quadratic = jacobian.T @ hessian @ jacobian
    norm = torch.tensor([rho0, 3 * rho0], dtype=x.dtype)
    return quadratic / torch.sqrt(norm[:, None] * norm[None, :])


class TestCoupledProbe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)

    @classmethod
    def tearDownClass(cls):
        torch.set_default_dtype(cls._dtype)

    def coupled(self, model, rho, pol, direction=(0.0, 0.0, 1.0), amplitude=0.05, **kwargs):
        kwargs.setdefault("alignment_reference", "quadratic")
        data = batch(rho, pol)
        return coupled_fourier_curvature_matrix(
            model, model(data), data, data["rho"], WAVES, amplitude,
            torch.tensor([direction]), **kwargs,
        )

    def test_uniform_state_matches_the_local_hessian(self):
        # Excess and reference are cubic in (rho, P) along each probe, so the
        # central differences are exact and the probe equals the analytic
        # local Hessian for every wave, phase, alignment and direction.
        model = LocalPolar(a=0.3, b=-0.4)
        for alignment in (0.0, 0.3, 0.6):
            for direction in ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)):
                with self.subTest(alignment=alignment, direction=direction):
                    rho, pol = uniform(0.8, alignment)
                    matrix, active = self.coupled(model, rho, pol, direction)
                    self.assertTrue(bool(active.all()))
                    expected = local_matrix(model, 0.8, alignment, direction)
                    torch.testing.assert_close(matrix[0], expected.expand_as(matrix[0]), rtol=1e-8, atol=1e-9)

    def test_exact_alignment_reference_matches_the_langevin_hessian(self):
        model = LocalPolar(a=0.3, b=-0.4)
        # The reference is the analytic base-state Hessian, so the result does
        # not depend on the probe amplitude (the excess is cubic: exact too).
        for alignment in (0.005, 0.3, 0.6, 0.85, 0.97):
            for amplitude in (1.0e-3, 0.05):
                with self.subTest(alignment=alignment, amplitude=amplitude):
                    rho, pol = uniform(0.8, alignment)
                    matrix, active = self.coupled(
                        model, rho, pol, amplitude=amplitude, alignment_reference="exact",
                        parallel_to_polarization=True,
                    )
                    self.assertTrue(bool(active.all()))
                    expected = local_matrix(model, 0.8, alignment, (0.0, 0.0, 1.0), exact=True)
                    torch.testing.assert_close(matrix[0], expected.expand_as(matrix[0]), rtol=1e-7, atol=1e-8)
        # Strongly aligned bulk: the weak-alignment form underestimates the
        # orientational stiffness, the exact entropy does not.
        rho, pol = uniform(0.8, 0.85)
        quadratic, _ = self.coupled(LocalPolar(0.0, 0.0), rho, pol, parallel_to_polarization=True)
        exact, _ = self.coupled(LocalPolar(0.0, 0.0), rho, pol, parallel_to_polarization=True,
                                alignment_reference="exact", amplitude=1.0e-3)
        self.assertGreater(exact[0, 0, 0, 1, 1].item(), 2.0 * quadratic[0, 0, 0, 1, 1].item())
        # Voxels at or beyond saturation use capped Langevin coefficients instead of failing.
        rho, pol = uniform(0.8, 0.5)
        pol[:5, 0, 2] = 1.2 * MOMENT * 0.8
        matrix, active = self.coupled(LocalPolar(0.3, -0.4), rho, pol, parallel_to_polarization=True,
                                      alignment_reference="exact")
        self.assertTrue(bool(torch.isfinite(matrix).all()) and bool(active.all()))

    def test_strongly_aligned_ideal_gas_is_never_unstable(self):
        # Regression: a physically valid field with alignment 0.96-0.98, zero
        # excess and amplitude 0.05. Switching the entropy regime per row made
        # the assembled matrix indefinite (lowest eigenvalue -2.1, loss 1.1).
        x = positions()[:, 0].double()
        y = positions()[:, 1].double()
        rho = (0.8 + 0.2 * torch.cos(2 * math.pi * x / GRID))[:, None]
        alignment = 0.97 + 0.01 * torch.sin(2 * math.pi * y / GRID)
        pol = torch.zeros(rho.shape + (3,))
        pol[:, 0, 2] = alignment * MOMENT * rho[:, 0]
        ideal = LocalPolar(a=0.0, b=0.0)
        data = batch(rho, pol)
        for parallel in (True, False):
            for amplitude in (1.0e-3, 0.05):
                with self.subTest(parallel=parallel, amplitude=amplitude):
                    matrix, active = coupled_fourier_curvature_matrix(
                        ideal, ideal(data), data, data["rho"], WAVES, amplitude,
                        torch.tensor([[0.3, 0.5, 0.8]]), parallel_to_polarization=parallel,
                    )
                    self.assertTrue(bool(active.all()))
                    self.assertGreater(torch.linalg.eigvalsh(matrix).min().item(), 0.0)
        for direction in ("parallel", "random", "transverse"):
            term = FourierStabilityLoss(
                wavevector_indices=((1, 0, 0), (0, 1, 0), (0, 0, 2)), relative_amplitude=0.05,
                minimum_curvature=0.0, variable="coupled", polarization_direction=direction,
            )
            term.train()
            self.assertEqual(term(ideal(data), data, model=ideal).item(), 0.0)

    def test_ideal_gas_is_the_identity(self):
        # Unpolarized ideal gas in an inhomogeneous two-species field: identity,
        # including every cross-species block, with and without weighting.
        x = positions()[:, 0].double()
        profile = 1.0 + 0.6 * torch.cos(2 * math.pi * x / GRID)
        rho = torch.stack((0.7 * profile, 0.3 * profile.flip(0)), dim=-1)
        pol = torch.zeros(rho.shape + (3,))
        model = LocalPolar(a=0.0, b=0.0, n_types=2)
        for power in (0.0, 2.0):
            with self.subTest(power=power):
                matrix, active = self.coupled(model, rho, pol, (0.0, 1.0, 0.0), density_power=power)
                self.assertTrue(bool(active.all()))
                torch.testing.assert_close(matrix[0], torch.eye(4).expand_as(matrix[0]), rtol=0, atol=1e-6)

    def test_channels_reduce_to_the_existing_probes(self):
        model = LocalPolar(a=0.3, b=-0.4)
        x = positions()[:, 0].double()
        rho = (0.8 + 0.3 * torch.cos(2 * math.pi * x / GRID))[:, None]
        pol = torch.zeros(rho.shape + (3,))
        data = batch(rho, pol)
        outputs = model(data)
        matrix, _ = coupled_fourier_curvature_matrix(
            model, outputs, data, data["rho"], WAVES, 1.0e-3, torch.tensor([[0.0, 1.0, 0.0]]),
        )
        # Unpolarized base: the density block is the ideal-metric density
        # matrix (whose ideal term is a finite difference, here exact to
        # O(amplitude^2); the coupled probe's reference is analytic).
        density, _ = fourier_curvature_matrix(model, outputs, data, data["rho"], WAVES, 1.0e-3)
        torch.testing.assert_close(matrix[0, ..., 0, 0], density[0, ..., 0, 0], rtol=1e-6, atol=1e-8)
        matrix, _ = coupled_fourier_curvature_matrix(
            model, outputs, data, data["rho"], WAVES, 0.05, torch.tensor([[0.0, 1.0, 0.0]]),
            alignment_reference="quadratic",
        )
        # The polarization channel is the polarization probe along the same direction.
        polar, _ = polarization_fourier_curvature(
            model, outputs, data, data["rho"], WAVES, 0.05, torch.tensor([[0.0, 1.0, 0.0]]),
        )
        torch.testing.assert_close(matrix[0, ..., 1, 1], polar[0], rtol=1e-8, atol=1e-10)

    def test_coupled_mode_is_found_when_each_channel_alone_is_stable(self):
        # Strong negative rho |P|^2 coupling at alignment 0.5: both diagonal
        # entries positive, joint eigenvalue negative.
        model = LocalPolar(a=0.2, b=-1.5)
        rho, pol = uniform(0.8, 0.5)
        expected = local_matrix(model, 0.8, 0.5, (0.0, 0.0, 1.0))
        self.assertGreater(expected[0, 0].item(), 0.0)
        self.assertGreater(expected[1, 1].item(), 0.0)
        self.assertLess(torch.linalg.eigvalsh(expected)[0].item(), -0.3)
        data = batch(rho, pol)
        penalty = {}
        for variable, kwargs in (
            ("rho", {}),
            ("dipole_density", {"include_zero_wavevector": False}),
            ("coupled", {"polarization_direction": "parallel"}),
            ("coupled", {"polarization_direction": "transverse"}),
        ):
            term = FourierStabilityLoss(
                wavevector_indices=((1, 0, 0), (0, 2, 0)), relative_amplitude=0.05,
                minimum_curvature=0.0, variable=variable, **kwargs,
            )
            term.train()
            penalty[variable, kwargs.get("polarization_direction")] = term(model(data), data, model=model).item()
        self.assertEqual(penalty["rho", None], 0.0)
        self.assertEqual(penalty["dipole_density", None], 0.0)
        self.assertGreater(penalty["coupled", "parallel"], 0.01)
        # A transverse direction is perpendicular to the base P here only if
        # it happens to be along x or y; the parallel direction always couples.
        self.assertLessEqual(penalty["coupled", "transverse"], penalty["coupled", "parallel"] + 1e-12)

    def test_species_blocks_and_absent_species(self):
        model = LocalPolar(a=0.2, b=-1.5, n_types=2)
        rho, pol = uniform(0.8, 0.5, n_types=2)
        rho[:, 1] = 0.0
        pol[:, 1] = 0.0
        for dtype in (torch.float64, torch.float32):
            for couple in (False, True):
                with self.subTest(dtype=dtype, couple=couple):
                    data = {k: v.to(dtype) if v.is_floating_point() else v for k, v in batch(rho, pol).items()}
                    matrix, active = coupled_fourier_curvature_matrix(
                        model.to(dtype), model(data), data, data["rho"], WAVES, 0.05,
                        torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype), couple_species=couple,
                        parallel_to_polarization=True,
                    )
                    self.assertTrue(bool(torch.isfinite(matrix).all()))
                    # Channels: rho_0, rho_1, P_0, P_1; species 1 absent.
                    self.assertTrue(bool(active[..., [0, 2]].all()))
                    self.assertFalse(bool(active[..., [1, 3]].any()))
                    self.assertEqual(matrix[..., 1, :].abs().max().item(), 0.0)
        model = model.to(torch.float64)
        # With two present species, independent blocks keep cross-species entries zero.
        rho, pol = uniform(0.8, 0.5, n_types=2)
        blocks, _ = self.coupled(model, rho, pol, couple_species=False, parallel_to_polarization=True)
        full, _ = self.coupled(model, rho, pol, couple_species=True, parallel_to_polarization=True)
        self.assertEqual(blocks[..., 0, 1].abs().max().item(), 0.0)
        self.assertEqual(blocks[..., 0, 3].abs().max().item(), 0.0)
        torch.testing.assert_close(blocks[..., 0, 2], full[..., 0, 2])
        for treatment in ("independent", "full_matrix"):
            term = FourierStabilityLoss(
                wavevector_indices=((1, 0, 0),), relative_amplitude=0.05, minimum_curvature=0.0,
                variable="coupled", polarization_direction="parallel", component_treatment=treatment,
            )
            term.train()
            data = batch(rho, pol)
            self.assertGreater(term(model(data), data, model=model).item(), 0.0)

    def test_exact_coefficients_are_continuous_up_to_saturation(self):
        # Regression: the exact coefficients used to drop from ~1e6 to the
        # weak-alignment 3 at alignment 0.999. They must be nondecreasing,
        # continuous, follow s'' ~ 1/(1-p)^2 and s'/p ~ 1/(1-p) near
        # saturation, and be held (not weakened) at and beyond the cap.
        from equicdft.response.curvature import _orientational_hessian
        for dtype in (torch.float64, torch.float32):
            with self.subTest(dtype=dtype):
                eps = torch.finfo(dtype).eps
                cap = 1.0 - math.sqrt(eps)
                grid = [0.5, 0.9, 0.99, 0.998, 0.998999, 0.999, 0.9995, 0.9999]
                grid = sorted(set(v for v in grid if v < cap) | {cap, 1.0, 1.2})
                p = torch.tensor(grid, dtype=dtype)
                rho = torch.ones(len(grid), 1, dtype=dtype)
                pol = torch.zeros(len(grid), 1, 3, dtype=dtype)
                pol[:, 0, 2] = p * MOMENT
                along, across, _, _ = _orientational_hessian(
                    rho, pol, torch.tensor([MOMENT], dtype=dtype), exact=True,
                )
                along, across = along[:, 0] * MOMENT**2, across[:, 0] * MOMENT**2   # s'' and s'/p
                self.assertTrue(bool(torch.isfinite(along).all() and torch.isfinite(across).all()))
                self.assertTrue(bool((along[1:] >= along[:-1] * (1 - 1e-6)).all()), along)
                self.assertTrue(bool((across[1:] >= across[:-1] * (1 - 1e-6)).all()), across)
                below = [i for i, v in enumerate(grid) if 0.99 <= v < cap]
                for i in below:
                    one_minus = 1.0 - grid[i]
                    self.assertAlmostEqual(along[i].item() * one_minus**2, 1.0, delta=0.05)
                    self.assertAlmostEqual(across[i].item() * one_minus, 1.0, delta=0.05)
                at_cap = grid.index(cap)
                torch.testing.assert_close(along[at_cap:], along[at_cap].expand(len(grid) - at_cap))
                torch.testing.assert_close(across[at_cap:], across[at_cap].expand(len(grid) - at_cap))
                if dtype == torch.float64:
                    i, j = grid.index(0.998999), grid.index(0.999)
                    self.assertLess(abs(along[j].item() / along[i].item() - 1.0), 0.01)

    def test_polarization_without_density_is_rejected(self):
        rho, pol = uniform(0.8, 0.5)
        rho[3, 0] = 0.0
        with self.assertRaisesRegex(ValueError, "zero where rho is zero"):
            self.coupled(LocalPolar(0.3, -0.4), rho, pol)

    def test_validation_and_repr(self):
        with self.assertRaisesRegex(ValueError, "zero wavevector"):
            FourierStabilityLoss(wavevector_indices=((1, 0, 0),), variable="coupled", include_zero_wavevector=True)
        with self.assertRaisesRegex(ValueError, "component_treatment"):
            FourierStabilityLoss(wavevector_indices=((1, 0, 0),), variable="coupled", component_treatment="charge", charges=(1.0,))
        with self.assertRaisesRegex(ValueError, "parallel"):
            FourierStabilityLoss(wavevector_indices=((1, 0, 0),), variable="dipole_density", polarization_direction="parallel")
        term = FourierStabilityLoss(wavevector_indices=((1, 0, 0),), variable="coupled", probe_density_power=2.0)
        self.assertEqual(term.ideal_reference, "translational_ideal_gas_plus_langevin_alignment")
        quadratic = FourierStabilityLoss(wavevector_indices=((1, 0, 0),), variable="coupled", alignment_reference="quadratic")
        self.assertEqual(quadratic.ideal_reference, "translational_ideal_gas_plus_quadratic_alignment")
        with self.assertRaisesRegex(ValueError, "alignment_reference"):
            FourierStabilityLoss(wavevector_indices=((1, 0, 0),), variable="rho", alignment_reference="exact")
        with self.assertRaisesRegex(ValueError, "alignment_reference"):
            FourierStabilityLoss(wavevector_indices=((1, 0, 0),), variable="coupled", alignment_reference="langevin")
        self.assertIn("variable='coupled'", repr(term))
        default = FourierStabilityLoss(wavevector_indices=((1, 0, 0),))
        self.assertEqual(default.variable, "rho")


if __name__ == "__main__":
    unittest.main()
