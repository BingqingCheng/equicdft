"""Checks for shared liquid Coulomb state and explicit-site kernels."""

import itertools
import math
import unittest
from unittest.mock import patch

import torch

from equicdft._fourier_sites import FourierSites
from equicdft.metalwall import MetalWall
from metal_helpers import _run, liquid_coulomb, metal_sites
from test_metalwall import _dense_kernels, _field, _site_rows


def _grid_potential(charge, kernel, spacing, shape):
    spectrum = torch.fft.fftn(charge.reshape(shape)) * kernel
    return torch.fft.ifftn(spectrum).real.reshape(-1) / spacing.prod()


class TestMetalWallKernel(unittest.TestCase):
    def test_liquid_coulomb_evaluates_potential_at_requested_sites(self):
        data = _field()
        readout = liquid_coulomb(exponent=.2).double()
        context = dict(
            rho=data["rho"],
            grid_size=data["grid_size"],
            grid_spacing=data["grid_spacing"],
            state_features=torch.cat((
                torch.ones(1, dtype=data["rho"].dtype),
                data["rho"].mean(0),
            )),
        )
        result = readout.coulomb_at(
            context, data["metal_positions"], target_sigma=.35,
        )
        q_liquid = (
            data["grid_spacing"].prod()
            * (data["rho"][:, 0] - data["rho"][:, 1])
        )
        liquid_metal, _ = _dense_kernels(data, .35, 1.7)
        expected = liquid_metal[_site_rows(data)] @ q_liquid
        torch.testing.assert_close(
            result.potential, expected, atol=2e-12, rtol=2e-12,
        )
        torch.testing.assert_close(result.energy, readout.energy(context))
        torch.testing.assert_close(
            result.amplitude, result.amplitude.new_tensor(1.7),
        )
        torch.testing.assert_close(result.total_charge, q_liquid.sum())

    def test_explicit_site_matrix_matches_grid_unit_probes(self):
        for shape in ((5, 4, 3), (4, 3, 2), (1, 4, 5)):
            for dtype in (torch.float32, torch.float64):
                with self.subTest(shape=shape, dtype=dtype):
                    n_grid = math.prod(shape)
                    selected = torch.unique(torch.cat((
                        torch.arange(0, n_grid, 3), torch.tensor([n_grid - 1]),
                    )))
                    coordinates = torch.tensor(
                        list(itertools.product(*(range(n) for n in shape))),
                        dtype=dtype,
                    )
                    spacing = torch.tensor([.7, 1.1, 1.3], dtype=dtype)
                    sites = FourierSites(
                        coordinates[selected] * spacing,
                        shape, spacing, dtype, torch.device("cpu"),
                    )
                    rho = torch.zeros((n_grid, 2), dtype=dtype)
                    state = liquid_coulomb().features._coulomb_evaluation(
                        rho, torch.tensor(shape), spacing,
                        torch.tensor([1., -1.], dtype=dtype),
                    )
                    k2 = _wavevector_squared(shape, spacing, dtype)
                    kernel = _full_coulomb_kernel(k2) * torch.exp(-.35**2 * k2)
                    actual = sites.matrix(kernel)
                    probes = torch.eye(n_grid, dtype=dtype)[selected]
                    expected = torch.stack([
                        _grid_potential(probe, kernel, spacing, shape)[selected]
                        for probe in probes
                    ], dim=1)
                    tolerance = 3e-6 if dtype == torch.float32 else 2e-12
                    torch.testing.assert_close(
                        actual, expected, atol=tolerance, rtol=tolerance,
                    )
                    torch.testing.assert_close(
                        actual, actual.T, atol=tolerance, rtol=tolerance,
                    )

    def test_full_spectrum_contains_the_long_range_complement(self):
        data = _field()
        readout = liquid_coulomb(exponent=.2).double()
        rho = data["rho"]
        state = readout.features._coulomb_evaluation(
            rho, data["grid_size"], data["grid_spacing"], readout.charges,
        )
        q = data["grid_spacing"].prod() * (rho[:, 0] - rho[:, 1])
        charge_spectrum = torch.fft.fftn(
            q.reshape(tuple(data["grid_size"].tolist()))
        )
        full_kernel = _full_coulomb_kernel(
            _wavevector_squared(
                tuple(data["grid_size"].tolist()), data["grid_spacing"], rho.dtype,
            )
        )
        torch.testing.assert_close(
            state.full_potential_spectrum,
            charge_spectrum * full_kernel,
            atol=2e-15,
            rtol=2e-15,
        )
        self.assertGreater(
            float((state.full_potential_spectrum
                   - state.long_range_potential_spectrum).abs().max()),
            0.0,
        )

    def test_wall_reuses_one_liquid_fft_and_amplitude_evaluation(self):
        data = _field()
        wall = MetalWall(
            liquid_coulomb=liquid_coulomb(),
            metal_sites=metal_sites(data),
            metal_sigma=.35,
        ).double()
        _run(wall, data)
        with patch.object(torch.fft, "fftn", wraps=torch.fft.fftn) as fft:
            with patch.object(
                wall.liquid_coulomb,
                "amplitude",
                wraps=wall.liquid_coulomb.amplitude,
            ) as amplitude:
                result = _run(wall, data)
        self.assertEqual(fft.call_count, 1)
        self.assertEqual(amplitude.call_count, 1)
        torch.testing.assert_close(
            fft.call_args[0][0].reshape(-1),
            data["grid_spacing"].prod()
            * (data["rho"][:, 0] - data["rho"][:, 1]),
        )


def _wavevector_squared(shape, spacing, dtype):
    """Independent wave-number construction for the Gaussian test filter."""
    axes = [
        2 * math.pi * torch.fft.fftfreq(n, d=float(step), dtype=dtype)
        for n, step in zip(shape, spacing)
    ]
    return sum(value.square() for value in torch.meshgrid(*axes, indexing="ij"))


def _full_coulomb_kernel(k2):
    safe = torch.where(k2 > 0, k2, torch.ones_like(k2))
    return torch.where(k2 > 0, 4 * math.pi / safe, torch.zeros_like(k2))


if __name__ == "__main__":
    unittest.main()
