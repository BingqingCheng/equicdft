"""Periodic lookup equivalence to the former full-grid unit-probe operator."""
import math
import unittest
from unittest.mock import patch

import torch

from equicdft import MetalWall
from equicdft.metalwall import _potential
from test_metalwall import _field


class _ProbeMetalWall(MetalWall):
    """Test oracle: retain the original probe assembly and FFT metal energy."""

    def _compute_S_matrix(self, index, mask, ids, kernel, spacing, shape):
        probes = kernel.new_zeros((index.numel(), math.prod(shape)))
        probes[torch.arange(index.numel()), index] = 1.
        A = _potential(probes, kernel, spacing, shape)[:, index].T
        C = (mask[index, None] == ids[None, :]).to(A)
        kkt = torch.cat((torch.cat((A, C), dim=1),
                         torch.cat((C.T, A.new_zeros(len(ids), len(ids))), dim=1)))
        factor, pivots = torch.linalg.lu_factor(kkt)
        return A, C, factor, pivots

    def _evaluate_field(self, shape, density, mask, ids, totals, spacing, field, origin):
        out = super()._evaluate_field(shape, density, mask, ids, totals, spacing, field, origin)
        kernels, _, _ = self._cache
        out['coulomb_metal_energy'] = .5 * torch.sum(
            out['metal_q'] * _potential(out['metal_q'], kernels[2], spacing, shape))
        out['coulomb_energy'] = (out['coulomb_liquid_energy']
                                + out['coulomb_cross_energy'] + out['coulomb_metal_energy'])
        return out


def _wall(cls=MetalWall, dtype=torch.float64, **options):
    kwargs = dict(charges=[1., -1.], sigma=.35, liquid_sigma=.12,
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
                        mask = torch.full((n,), -1, dtype=torch.long)
                        mask[index] = torch.where(torch.arange(len(index)) % 2 == 0, 7, 2)
                        ids = torch.tensor([7, 2])  # Deliberately unsorted labels.
                        spacing = torch.tensor([.7, 1.1, 1.3], dtype=dtype)
                        rho = torch.zeros(n, 2, dtype=dtype)
                        wall = _wall(dtype=dtype, sigma=width)
                        with patch('torch.fft.ifftn', wraps=torch.fft.ifftn) as inverse:
                            kernels, selected, (A, C, _, _) = wall._geometry(shape, spacing, mask, ids, rho)
                        self.assertEqual(inverse.call_count, 1)
                        self.assertEqual(tuple(inverse.call_args[0][0].shape), shape)
                        expected = _wall(_ProbeMetalWall, dtype=dtype, sigma=width)._compute_S_matrix(
                            selected, mask, ids, kernels[2], spacing, shape)
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
                    neutral = .3 * (data['metal_mask'] < 0).to(dtype)
                    data['rho'] = torch.cat((data['rho'], neutral[:, None]), dim=-1).requires_grad_()
                    data['metal_external_field'] = torch.tensor([.03, -.02, .01], dtype=dtype)
                    data['metal_field_origin'] = -data['grid_spacing']/2
                    actual = _wall(dtype=dtype, charges=[1., -1., 0.], liquid_sigma=liquid_sigma)(data)
                    expected = _wall(_ProbeMetalWall, dtype, charges=[1., -1., 0.], liquid_sigma=liquid_sigma)(data)
                    tolerance = 2e-5 if dtype == torch.float32 else 2e-11
                    self.assertEqual(actual.keys(), expected.keys())
                    for key in actual:
                        torch.testing.assert_close(actual[key], expected[key], atol=tolerance, rtol=tolerance)
                    ga = torch.autograd.grad(actual['coulomb_energy'] + actual['metal_external_energy'],
                                             data['rho'], create_graph=True)[0]
                    ge = torch.autograd.grad(expected['coulomb_energy'] + expected['metal_external_energy'],
                                             data['rho'], create_graph=True)[0]
                    direction = torch.sin(torch.arange(ga.numel(), dtype=dtype)).reshape_as(ga)
                    ha = torch.autograd.grad((ga * direction).sum(), data['rho'])[0]
                    he = torch.autograd.grad((ge * direction).sum(), data['rho'])[0]
                    torch.testing.assert_close(ga, ge, atol=tolerance, rtol=tolerance)
                    torch.testing.assert_close(ha, he, atol=tolerance, rtol=tolerance)

    def test_cached_evaluation_does_not_fft_the_metal_charges(self):
        data = _field()
        wall = _wall()
        wall(data)
        with patch('equicdft.metalwall._potential', wraps=_potential) as potential:
            out = wall(data)
        self.assertEqual(potential.call_count, 2)  # Liquid->metal and liquid->liquid only.
        q = out['metal_q'][data['metal_mask'] >= 0]
        torch.testing.assert_close(out['coulomb_metal_energy'], .5 * q @ wall._cache[2][0] @ q)


if __name__ == '__main__':
    unittest.main()
