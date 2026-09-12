"""Periodic lookup equivalence to the former full-grid unit-probe operator."""
import math
import unittest
from unittest.mock import patch

import torch
from metal_helpers import _run, metal_sites

from equicdft import MetalWall
from test_metalwall import _field, _site_rows
from equicdft._metal_sites import ExplicitSiteGrid


def _potential(charge, kernel, spacing, shape):
    """Potential conjugate to integrated grid charges, not charge density."""

    values = charge.reshape(*charge.shape[:-1], *shape)
    transformed = torch.fft.fftn(values, dim=(-3, -2, -1))
    result = torch.fft.ifftn(
        transformed * kernel, dim=(-3, -2, -1),
    ).real / spacing.prod()
    return result.reshape_as(charge)


class _ProbeMetalWall(MetalWall):
    """Test oracle: retain the original probe assembly and FFT metal energy."""

    def _geometry(self, shape, spacing, positions, groups, ids, rho):
        kernels = self._kernels(shape, spacing, rho)
        sampling = ExplicitSiteGrid(positions, shape, spacing, rho.dtype, rho.device)
        coordinates = (positions / spacing).round().long()
        index = (coordinates[:, 0]*shape[1] + coordinates[:, 1])*shape[2] + coordinates[:, 2]
        probes = rho.new_zeros((index.numel(), math.prod(shape)))
        probes[torch.arange(index.numel()), index] = 1.
        A = _potential(probes, kernels[1], spacing, shape)[:, index].T
        C = (groups[:, None] == ids[None, :]).to(A)
        system = self._factor_system(A, C)
        self._cache = (kernels, sampling, system)
        return self._cache

    def _evaluate_field(self, shape, **field):
        out = super()._evaluate_field(shape, **field)
        kernels, _, _ = self._cache
        spacing = field["spacing"]
        site_positions = self.metal_positions.to(device=spacing.device)
        coordinates = (site_positions / spacing).round().long()
        index = (coordinates[:, 0]*shape[1] + coordinates[:, 1])*shape[2] + coordinates[:, 2]
        grid_charge = torch.zeros_like(out["q_liquid"]).index_copy(0, index, out["metal_site_q"])
        out['coulomb_metal_energy'] = .5 * torch.sum(
            grid_charge * _potential(grid_charge, kernels[1], spacing, shape))
        out['electrode_coulomb_energy'] = (out['coulomb_cross_energy'] + out['coulomb_metal_energy'])
        return out


def _wall(cls=MetalWall, dtype=torch.float64, **options):
    kwargs = dict(metal_sites=metal_sites(_field()), liquid_charges=[1., -1.],
                  metal_sigma=.35, liquid_sigma=.12,
                  coulomb_amplitude=1.7, boundary='periodic')
    kwargs.update(options)
    return cls(**kwargs).to(dtype=dtype)


class TestMetalWallKernel(unittest.TestCase):
    def test_matrix_matches_unit_probes_for_periodic_grids(self):
        for shape in ((5, 4, 3), (4, 3, 2), (1, 4, 5)):
            for dtype in (torch.float32, torch.float64):
                for width in (.19, .6):
                    with self.subTest(shape=shape, dtype=dtype, width=width):
                        n = math.prod(shape)
                        index = torch.unique(torch.cat((torch.arange(0, n, 3), torch.tensor([n-1]))))
                        groups = torch.where(torch.arange(len(index)) % 2 == 0, 7, 2)
                        ids = torch.tensor([7, 2])  # Deliberately unsorted labels.
                        spacing = torch.tensor([.7, 1.1, 1.3], dtype=dtype)
                        rho = torch.zeros(n, 2, dtype=dtype)
                        coordinates = torch.stack((index // (shape[1]*shape[2]),
                                                   (index // shape[2]) % shape[1], index % shape[2]), dim=1)
                        positions = coordinates.double() * spacing.double()
                        wall = _wall(dtype=dtype, metal_sigma=width)
                        with patch('torch.fft.ifftn', wraps=torch.fft.ifftn) as inverse:
                            kernels, _, (A, C, _, _) = wall._geometry(shape, spacing, positions, groups, ids, rho)
                        self.assertEqual(inverse.call_count, 1)
                        self.assertEqual(tuple(inverse.call_args[0][0].shape), shape)
                        expected = _wall(_ProbeMetalWall, dtype=dtype, metal_sigma=width)._geometry(
                            shape, spacing, positions, groups, ids, rho)[2]
                        tolerance = 3e-6 if dtype == torch.float32 else 2e-12
                        torch.testing.assert_close(A, expected[0], atol=tolerance, rtol=tolerance)
                        torch.testing.assert_close(A, A.T, atol=tolerance, rtol=tolerance)
                        torch.testing.assert_close(C, expected[1], atol=0, rtol=0)

    def test_outputs_and_derivatives_match_probe_method(self):
        for dtype in (torch.float32, torch.float64):
            for liquid_sigma in (0., .12):
                with self.subTest(dtype=dtype, liquid_sigma=liquid_sigma):
                    data = _field(shape=(5, 3, 2))
                    data = {k: v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v
                            for k, v in data.items()}
                    # Three species, including a neutral component; fixed group totals.
                    neutral = .3 * (~data['excluded_mask']).to(dtype)
                    data['rho'] = torch.cat((data['rho'], neutral[:, None]), dim=-1).requires_grad_()
                    data['metal_external_field'] = torch.tensor([.03, -.02, .01], dtype=dtype)
                    data['metal_field_origin'] = -data['grid_spacing']/2
                    actual = _run(_wall(dtype=dtype, liquid_charges=[1., -1., 0.], liquid_sigma=liquid_sigma), data)
                    expected = _run(_wall(_ProbeMetalWall, dtype, liquid_charges=[1., -1., 0.], liquid_sigma=liquid_sigma), data)
                    tolerance = 2e-5 if dtype == torch.float32 else 2e-11
                    self.assertEqual(actual.keys(), expected.keys())
                    for key in actual:
                        torch.testing.assert_close(actual[key], expected[key], atol=tolerance, rtol=tolerance)
                    ga = torch.autograd.grad(actual['electrode_coulomb_energy'] + actual['metal_external_energy'],
                                             data['rho'], create_graph=True)[0]
                    ge = torch.autograd.grad(expected['electrode_coulomb_energy'] + expected['metal_external_energy'],
                                             data['rho'], create_graph=True)[0]
                    direction = torch.sin(torch.arange(ga.numel(), dtype=dtype)).reshape_as(ga)
                    ha = torch.autograd.grad((ga * direction).sum(), data['rho'])[0]
                    he = torch.autograd.grad((ge * direction).sum(), data['rho'])[0]
                    torch.testing.assert_close(ga, ge, atol=tolerance, rtol=tolerance)
                    torch.testing.assert_close(ha, he, atol=tolerance, rtol=tolerance)

    def test_cached_evaluation_does_not_fft_the_metal_charges(self):
        data = _field()
        wall = _wall()
        _run(wall, data)
        with patch.object(torch.fft, 'fftn', wraps=torch.fft.fftn) as fft:
            out = _run(wall, data)
        self.assertEqual(fft.call_count, 1)  # Only the liquid source is transformed.
        torch.testing.assert_close(fft.call_args[0][0].reshape(-1), out['q_liquid'])
        q = out['metal_site_q']
        torch.testing.assert_close(out['coulomb_metal_energy'], .5 * q @ wall._cache[2][0] @ q)


if __name__ == '__main__':
    unittest.main()
