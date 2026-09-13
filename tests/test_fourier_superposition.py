import unittest
from unittest.mock import patch

import torch
from torch import nn

from equicdft import FourierResponse, FourierStabilityLoss
from equicdft._fourier import component_fourier_directions
from test_fourier_amplitudes import batch
import test_loss as fixtures


class SpatialQuadratic(nn.Module):
    """Known spatial Hessian, including off-diagonal Fourier blocks."""
    def __init__(self, hessian):
        super().__init__()
        self.hessian = nn.Parameter(hessian.clone())

    def forward(self, data, **kwargs):
        rho = data['rho'][..., 0]
        return {'beta_F_exc': .5 * torch.einsum('...i,ij,...j->...', rho, self.hessian, rho)}


class TestFourierSuperposition(unittest.TestCase):
    def test_sum_before_projection_counts_masks_and_bound(self):
        data = batch()
        rho = data['rho']
        rho[:, 0] = 0
        rho[:, 1:] *= torch.linspace(.2, 1.5, 7)[None, :, None]
        rho[:, :, 1] *= torch.linspace(1.5, .4, 8)
        modes = torch.tensor([[[1, 0, 0], [2, 0, 0]]] * 2)
        phases = torch.tensor([[.3, 1.2], [.7, 2.3]], dtype=rho.dtype)
        directions, active = component_fourier_directions(data, rho, modes, phases)
        angles = 2 * torch.pi * torch.einsum(
            'fgd,fmd->fmg', data['grid_positions'].to(rho) / data['grid_size'][:, None],
            modes.to(rho))
        waves = torch.cos(angles + phases[..., None]).sum(1)
        centered = waves[..., None] - (rho * waves[..., None]).sum(1)[:, None] / rho.sum(1)[:, None]
        expected = rho * centered / centered.abs().amax(1, keepdim=True)
        torch.testing.assert_close(directions[:, 0], expected)
        torch.testing.assert_close(directions.sum(-2), torch.zeros_like(directions.sum(-2)), atol=1e-14, rtol=0)
        self.assertTrue(active.all())
        for sign in (-1, 1):
            perturbed = rho[:, None] + sign * .1 * directions
            self.assertTrue((perturbed >= 0).all())
            self.assertTrue((perturbed[:, :, 0] == 0).all())
            torch.testing.assert_close(perturbed.sum(-2), rho.sum(-2)[:, None])
        self.assertTrue((directions.abs() <= rho[:, None] + 1e-14).all())

    def test_sampling_one_amplitude_and_shared_pattern_for_full_matrix(self):
        data = batch(8)
        model = fixtures._CoupledQuadraticExcessModel([[-4., 3.], [3., -4.]])
        for count in (1, 2, 3, (1, 3)):
            term = FourierStabilityLoss(random_modes_per_field=count, mode_domain='cube',
                                        mixture_mode='full_matrix', relative_amplitude=(.01, .1))
            with patch.object(term.response, 'matrix', wraps=term.response.matrix) as call:
                torch.manual_seed(7)
                loss = term(model(data), data, model=model)
                loss.backward()
                args = call.call_args.kwargs
                amplitude = args['relative_amplitude']
                self.assertEqual(amplitude.shape, (8, 1))
                self.assertTrue(((amplitude >= .01) & (amplitude < .1)).all())
                if args['modes'].shape[1] == 1:
                    self.assertIsNone(args['mode_phases'])
                else:
                    self.assertEqual(args['mode_phases'].shape, args['modes'].shape[:2])
                    value, active = term.response.matrix(**args)
                    self.assertEqual(value.shape, (8, 1, 1, 2, 2))
                    self.assertTrue(active.all())
                    self.assertTrue(torch.isfinite(model.matrix.grad).all())

    def test_composite_detects_cross_wavevector_curvature(self):
        data = batch(1)
        data['rho'] = data['rho'][..., :1]
        x = torch.arange(8, dtype=torch.float64)
        a = torch.cos(2 * torch.pi * x / 8); a /= a.norm()
        b = torch.cos(4 * torch.pi * x / 8); b /= b.norm()
        hessian = -4 * (a[:, None] * b + b[:, None] * a)
        model = SpatialQuadratic(hessian)
        ideal = SpatialQuadratic(torch.zeros_like(hessian))
        modes = torch.tensor([[[1, 0, 0], [2, 0, 0]]])
        response = FourierResponse(relative_amplitude=.01)
        single, active = response.matrix(model, data, modes)
        base, _ = response.matrix(ideal, data, modes)
        torch.testing.assert_close(single - base, torch.zeros_like(single), atol=1e-9, rtol=0)
        phases = torch.zeros((1, 2), dtype=torch.float64)
        composite, _ = response.matrix(model, data, modes, mode_phases=phases)
        base, _ = response.matrix(ideal, data, modes, mode_phases=phases)
        d, _ = component_fourier_directions(data, data['rho'], modes, phases)
        d = d[0, 0, :, 0]
        expected = (d @ hessian @ d) / (d.square() / data['rho'][0, :, 0]).sum()
        torch.testing.assert_close((composite - base).squeeze(), expected, atol=1e-9, rtol=0)
        self.assertLess(float(composite.squeeze()), 0.)
        self.assertTrue((single[active[..., None]] > 0).all())

    def test_matrix_and_directional_normalization_and_chunked_gradients(self):
        data = batch()
        modes = torch.tensor([[[1, 0, 0], [2, 0, 0]]] * 2)
        phases = torch.tensor([[.2, .7], [.3, 1.1]], dtype=torch.float64)
        amplitudes = torch.tensor([[.03], [.08]], dtype=torch.float64)
        model = fixtures._CoupledQuadraticExcessModel([[.2, .3], [.3, -.4]])
        ideal = fixtures._CoupledQuadraticExcessModel(torch.zeros(2, 2))
        results, gradients = [], []
        for chunk in (None, 2):
            response = FourierResponse(perturbations_per_forward=chunk)
            value, _ = response.matrix(model, data, modes, mode_phases=phases, relative_amplitude=amplitudes)
            base, _ = response.matrix(ideal, data, modes, mode_phases=phases, relative_amplitude=amplitudes)
            torch.testing.assert_close(value - base, (.5 * model.matrix).expand_as(value), atol=1e-9, rtol=0)
            projected, _ = response(model, data, modes, torch.eye(2),
                                    mode_phases=phases, relative_amplitude=amplitudes)
            torch.testing.assert_close(projected, torch.diagonal(value, dim1=-2, dim2=-1), atol=1e-9, rtol=0)
            results.append(value.detach())
            gradients.append(torch.autograd.grad(value.square().sum(), model.matrix)[0])
        torch.testing.assert_close(results[0], results[1])
        torch.testing.assert_close(gradients[0], gradients[1])

    def test_single_mode_keeps_cosine_sine_and_rng(self):
        data = batch()
        model = fixtures._CoupledQuadraticExcessModel([[-4., 3.], [3., -4.]])
        term = FourierStabilityLoss(random_modes_per_field=1, mode_domain='cube', mixture_mode='full_matrix')
        torch.manual_seed(19)
        modes = term._select_modes(data, data['rho'])
        rng = torch.get_rng_state().clone()
        matrix, active = term.response.matrix(model, data, modes)
        expected = term._matrix_loss(matrix, active)
        torch.manual_seed(19)
        actual = term(model(data), data, model=model)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(matrix.shape[2], 2)

    def test_phase_validation_and_nyquist(self):
        data = batch(1)
        modes = torch.tensor([[[4, 0, 0]]])
        for invalid in (torch.zeros(2), torch.ones(1, 1, dtype=torch.bool),
                        torch.full((1, 1), float('nan')), torch.ones(1, 1, dtype=torch.complex64)):
            with self.assertRaisesRegex(ValueError, 'mode_phases'):
                component_fourier_directions(data, data['rho'], modes, invalid)
        d, valid = component_fourier_directions(data, data['rho'], modes,
                                                torch.tensor([[.3]], dtype=torch.float64))
        expected = data['rho'][:, None] * torch.tensor([1., -1.] * 4)[None, None, :, None]
        torch.testing.assert_close(d, expected)
        self.assertTrue(valid.all())
