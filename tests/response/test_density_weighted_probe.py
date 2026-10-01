"""Density-weighted density stability probes.

A box-wide density wave averages the local compressibility over the field, so
a functional that is unstable inside a dense layer can pass a curvature
floor. Weighting the wave by ``(rho / max rho)^p`` concentrates it in the
layer while keeping every particle number exact, and the curvature is then
measured in the ideal-gas metric of the weighted displacement.
"""

import unittest

import numpy as np
import torch
from torch import nn

from equicdft import FourierStabilityLoss
from equicdft.response.modes import component_fourier_directions
from equicdft.response.response import FourierResponse

GRID = 16
CUBIC = -0.05   # local ideal-unit curvature 1 + 6 c rho^2: 0.70 at rho = 1, -0.875 at 2.5


class CubicLocal(nn.Module):
    """``beta F_exc = DeltaV * sum c * rho_total^3``: a purely local functional."""

    def __init__(self, coefficient):
        super().__init__()
        self.coefficient = coefficient

    def forward(self, batch, **kwargs):
        rho = batch["rho"]
        volume = torch.prod(batch["grid_spacing"].to(rho), dim=-1)
        return {"beta_F_exc": volume * (self.coefficient * rho.sum(-1) ** 3).sum(-1)}


def positions():
    return torch.tensor(np.indices((GRID,) * 3).reshape(3, -1).T)


def batch(rho):
    return {
        "rho": rho[None], "grid_positions": positions()[None],
        "grid_size": torch.tensor([[GRID] * 3]), "grid_spacing": torch.ones(1, 3),
        "temperature": torch.ones(1),
    }


def layered(peak, n_types=1):
    """Unit density with one Gaussian layer (width 1.5) of the given peak at x = 8."""

    x = positions()[:, 0].double()
    distance = torch.minimum((x - 8).abs(), GRID - (x - 8).abs())
    total = 1.0 + (peak - 1.0) * torch.exp(-0.5 * (distance / 1.5) ** 2)
    if n_types == 1:
        return total[:, None]
    # Two components with different profiles: 70/30 in the layer, 40/60 outside.
    fraction = 0.4 + 0.3 * torch.exp(-0.5 * (distance / 1.5) ** 2)
    return torch.stack((fraction * total, (1 - fraction) * total), dim=-1)


WAVES = torch.tensor([[[1, 0, 0], [2, 0, 0]]])


class TestDensityWeightedProbe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)

    @classmethod
    def tearDownClass(cls):
        torch.set_default_dtype(cls._dtype)

    def setUp(self):
        self.response = FourierResponse(relative_amplitude=1.0e-3)

    def curvature(self, model, rho, power, **kwargs):
        if power is None:
            values, valid = self.response(model, batch(rho), WAVES, torch.ones(1, rho.shape[-1]), **kwargs)
        else:
            values, valid = self.response(
                model, batch(rho), WAVES, torch.ones(1, rho.shape[-1]), density_power=power, **kwargs,
            )
        self.assertTrue(bool(valid.all()))
        return values

    def test_default_is_the_original_probe(self):
        rho, model = layered(2.5), CubicLocal(CUBIC)
        torch.testing.assert_close(self.curvature(model, rho, 0.0), self.curvature(model, rho, None), rtol=0, atol=0)
        matrix, _ = self.response.matrix(model, batch(rho), WAVES)
        weighted_zero, _ = self.response.matrix(model, batch(rho), WAVES, density_power=0.0)
        torch.testing.assert_close(weighted_zero, matrix, rtol=0, atol=0)
        default = FourierStabilityLoss(wavevector_indices=((1, 0, 0),))
        self.assertEqual(default.probe_density_power, 0.0)

    def test_weighted_directions_conserve_every_particle_number(self):
        rho = layered(2.5, n_types=2)
        data = batch(rho)
        for power in (0.0, 1.0, 2.0, 5.0):
            with self.subTest(power=power):
                directions, valid = component_fourier_directions(data, data["rho"], WAVES, density_power=power)
                self.assertTrue(bool(valid.all()))
                # directions: [field, probe, grid, type]; sum over the grid per component.
                scale = directions.abs().sum().item()
                torch.testing.assert_close(
                    directions.sum(dim=-2), torch.zeros_like(directions.sum(dim=-2)), rtol=0, atol=1e-14 * scale,
                )
                relative = (directions / data["rho"][:, None]).abs().amax(dim=-2)
                torch.testing.assert_close(relative, torch.ones_like(relative))
        # Larger powers move the displacement into the layer.
        x = positions()[:, 0].double()
        inside = (torch.minimum((x - 8).abs(), GRID - (x - 8).abs()) <= 2.0)
        share = []
        for power in (0.0, 2.0, 5.0):
            directions, _ = component_fourier_directions(data, data["rho"], WAVES, density_power=power)
            squared = directions.square().sum(dim=-1)
            share.append((squared[..., inside].sum() / squared.sum()).item())
        self.assertLess(share[0], share[1])
        self.assertLess(share[1], share[2])

    def test_weighted_curvature_is_in_ideal_units_everywhere(self):
        ideal = CubicLocal(0.0)
        rho = layered(2.5)
        # Ideal gas in an inhomogeneous field: exactly one for p > 0. The
        # default mean-density normalization is only exact when homogeneous.
        torch.testing.assert_close(self.curvature(ideal, rho, 2.0), torch.ones(1, 2, 2, 1), rtol=0, atol=1e-6)
        plain = self.curvature(ideal, rho, 0.0)
        self.assertGreater((plain - 1).abs().max().item(), 0.05)
        # Homogeneous field: the weight is one and both normalizations agree.
        uniform = torch.full((GRID ** 3, 1), 1.3)
        model = CubicLocal(CUBIC)
        torch.testing.assert_close(self.curvature(model, uniform, 2.0), self.curvature(model, uniform, 0.0))
        expected = 1 + 6 * CUBIC * 1.3 ** 2
        torch.testing.assert_close(self.curvature(model, uniform, 2.0), torch.full((1, 2, 2, 1), expected), rtol=0, atol=1e-6)
        # The full-matrix path already used the ideal metric; one component agrees.
        weighted = self.curvature(model, rho, 2.0)
        matrix, active = self.response.matrix(model, batch(rho), WAVES, density_power=2.0)
        self.assertTrue(bool(active.all()))
        torch.testing.assert_close(matrix[..., 0, 0], weighted[..., 0], rtol=1e-8, atol=1e-10)

    def test_weighted_probe_finds_the_unstable_dense_layer(self):
        rho, model = layered(2.5), CubicLocal(CUBIC)
        plain, weighted = self.curvature(model, rho, 0.0), self.curvature(model, rho, 2.0)
        # The layer itself is unstable (1 + 6 c 2.5^2 = -0.875); the box-wide
        # wave passes a zero floor, the weighted one does not.
        self.assertGreater(plain.min().item(), 0.15)
        self.assertLess(weighted.min().item(), -0.2)
        penalties = {}
        for power in (0.0, 2.0):
            for treatment in ("independent", "superposition"):
                term = FourierStabilityLoss(
                    wavevector_indices=((1, 0, 0), (2, 0, 0)), relative_amplitude=1.0e-3,
                    minimum_curvature=0.0, probe_density_power=power, wavevector_treatment=treatment,
                )
                term.train()
                data = batch(rho)
                penalties[power, treatment] = term(model(data), data, model=model).item()
        self.assertEqual(penalties[0.0, "independent"], 0.0)
        self.assertGreater(penalties[2.0, "independent"], 0.01)
        self.assertTrue(np.isfinite(penalties[2.0, "superposition"]))
        # The full-matrix treatment sees the same instability.
        matrix_penalty = {}
        for power in (0.0, 2.0):
            term = FourierStabilityLoss(
                wavevector_indices=((1, 0, 0), (2, 0, 0)), relative_amplitude=1.0e-3, minimum_curvature=0.0,
                component_treatment="full_matrix", probe_density_power=power,
            )
            term.train()
            data = batch(rho)
            matrix_penalty[power] = term(model(data), data, model=model).item()
        self.assertEqual(matrix_penalty[0.0], 0.0)
        self.assertGreater(matrix_penalty[2.0], 0.01)
        self.assertIn("probe_density_power=2.0", repr(term))

    @staticmethod
    def cast(data, dtype):
        return {key: value.to(dtype) if value.is_floating_point() else value for key, value in data.items()}

    def stability(self, data, power, **kwargs):
        # The cubic excess has an exact central second difference, so a
        # larger amplitude keeps float32 roundoff well below the signal.
        term = FourierStabilityLoss(
            wavevector_indices=((1, 0, 0), (2, 0, 0)), relative_amplitude=0.05,
            minimum_curvature=0.0, probe_density_power=power, **kwargs,
        )
        term.train()
        model = CubicLocal(CUBIC)
        return term(model(data), data, model=model)

    def test_globally_absent_species_is_inactive_not_nan(self):
        # One present species (with the unstable layer) and one absent species.
        present = layered(2.5)
        mixture = torch.cat((present, torch.zeros_like(present)), dim=-1)
        for dtype in (torch.float64, torch.float32):
            with self.subTest(dtype=dtype):
                data = self.cast(batch(mixture), dtype)
                alone = self.cast(batch(present), dtype)
                penalty = self.stability(data, 2.0)
                self.assertTrue(bool(torch.isfinite(penalty)))
                # The absent species' directions are invalid, so the penalty
                # is the present species' alone.
                rtol = 1e-8 if dtype == torch.float64 else 1e-2
                torch.testing.assert_close(penalty, self.stability(alone, 2.0), rtol=rtol, atol=0.0)
                self.assertGreater(penalty.item(), 0.01)
                curvature, valid = FourierResponse(relative_amplitude=0.05)(
                    CubicLocal(CUBIC), data, WAVES, torch.eye(2, dtype=dtype), density_power=2.0,
                )
                self.assertTrue(bool(valid[..., 0].all()) and not bool(valid[..., 1].any()))
                self.assertTrue(bool(torch.isfinite(curvature).all()))
                for treatment in ("total_density", "full_matrix"):
                    self.assertTrue(bool(torch.isfinite(self.stability(data, 2.0, component_treatment=treatment))))

    def test_dilute_species_keeps_its_particle_number(self):
        # Species b is present (grid total above the 1e-12 presence threshold)
        # but its weighted mass sum(rho_b g) lies below 1e-12.
        main = layered(2.5)
        dilute = 3.0e-16 * (2.0 - torch.cos(2 * torch.pi * positions()[:, 1].double() / GRID))[:, None]
        mixture = torch.cat((main, dilute), dim=-1)
        self.assertGreater(dilute.sum().item(), 1.0e-12)
        for dtype in (torch.float64, torch.float32):
            with self.subTest(dtype=dtype):
                data = self.cast(batch(mixture), dtype)
                rho = data["rho"]
                weight = (rho.sum(-1) / rho.sum(-1).max()) ** 2
                self.assertLess((rho[..., 1] * weight).sum().item(), 1.0e-12)
                directions, valid = component_fourier_directions(data, rho, WAVES, density_power=2.0)
                self.assertTrue(bool(valid.all()))
                total = directions.sum(dim=-2).abs()
                size = directions.abs().sum(dim=-2)
                tolerance = 1.0e-12 if dtype == torch.float64 else 2.0e-5
                self.assertLess((total / size).max().item(), tolerance)
                relative = (directions / rho[:, None]).abs().amax(dim=-2)
                torch.testing.assert_close(relative, torch.ones_like(relative), rtol=1e-5, atol=1e-5)

    def test_species_whose_weight_underflows_is_inactive(self):
        # Species b lives only where the total density is 1e-3 of its
        # maximum; with p = 20 its weight is 1e-60: a normal float64, zero in
        # float32. Float64 keeps it active and conserved, float32 drops it.
        x = positions()[:, 0].double()
        far = torch.minimum((x - 8).abs(), GRID - (x - 8).abs()) >= 6
        main = torch.where(far, torch.zeros_like(x), torch.ones_like(x))[:, None]
        sparse = torch.where(far, torch.full_like(x, 1.0e-3), torch.zeros_like(x))[:, None]
        mixture = torch.cat((main, sparse), dim=-1)
        for dtype, active in ((torch.float64, True), (torch.float32, False)):
            with self.subTest(dtype=dtype):
                data = self.cast(batch(mixture), dtype)
                directions, valid = component_fourier_directions(data, data["rho"], WAVES, density_power=20.0)
                self.assertTrue(bool(valid[..., 0].all()))
                self.assertEqual(bool(valid[..., 1].all()), active)
                self.assertTrue(bool(torch.isfinite(directions).all()))
                if not active:
                    self.assertEqual(directions[..., 1].abs().max().item(), 0.0)
                self.assertTrue(bool(torch.isfinite(self.stability(data, 20.0))))

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "probe_density_power"):
            FourierStabilityLoss(wavevector_indices=((1, 0, 0),), probe_density_power=-1.0)
        rho, model = layered(2.0), CubicLocal(CUBIC)
        with self.assertRaisesRegex(ValueError, "density_power"):
            self.response(model, batch(rho), WAVES, torch.ones(1, 1), density_power=-0.5)
        with self.assertRaisesRegex(ValueError, "density_power"):
            self.response.matrix(model, batch(rho), WAVES, density_power=float("nan"))


if __name__ == "__main__":
    unittest.main()
