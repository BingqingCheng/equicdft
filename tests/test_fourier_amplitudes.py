import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from equicdft import FourierResponse, FourierStabilityLoss, Loss, Trainer
import test_loss as fixtures


def batch(n_fields=2):
    data = fixtures.TestFourierStabilityLoss._batch(n_fields=n_fields)
    data['rho'] = data['rho'].expand(-1, -1, 2).clone()
    return data


class TestFourierAmplitudes(unittest.TestCase):
    def test_interval_validation(self):
        for interval in ((), (.1,), (.1, .2, .3), (0, .1), (.1, 1), (.2, .1),
                         (float('nan'), .2), (.1, float('inf')), (True, .2)):
            with self.subTest(interval=interval), self.assertRaises((ValueError, TypeError)):
                FourierStabilityLoss(modes=((1, 0, 0),), relative_amplitude=interval)

    def test_sampling_per_field_mode_and_eval_does_not_sample(self):
        data = batch(64)
        model = fixtures._CoupledQuadraticExcessModel([[0., 3.], [3., 0.]])
        term = FourierStabilityLoss(modes=((1, 0, 0), (2, 0, 0)),
                                    mixture_mode='full_matrix', relative_amplitude=(.02, .1))
        with patch.object(term.response, 'matrix', wraps=term.response.matrix) as response:
            torch.manual_seed(10)
            term(model(data), data, model=model)
            first = response.call_args.kwargs['relative_amplitude'].clone()
            term(model(data), data, model=model)
            second = response.call_args.kwargs['relative_amplitude']
            torch.manual_seed(10)
            term(model(data), data, model=model)
            replay = response.call_args.kwargs['relative_amplitude']
            self.assertEqual(first.shape, (64, 2))
            self.assertTrue(torch.all((first >= .02) & (first < .1)))
            self.assertFalse(torch.equal(first, second))
            self.assertTrue(torch.equal(first, replay))
            self.assertGreater(torch.unique(first).numel(), 120)
            self.assertAlmostEqual(float(first.mean()), .06, delta=.008)
            rng = torch.get_rng_state().clone()
            term.eval()
            self.assertEqual(float(term(model(data), data, model=model)), 0.)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_fixed_and_degenerate_intervals_preserve_values_gradients_rng(self):
        data = batch()
        for mode in ('independent', 'total_density', 'charge', 'full_matrix'):
            kwargs = {'charges': (1., -1.)} if mode == 'charge' else {}
            values, gradients = [], []
            for amplitude in (.05, (.05, .05), [.05, .05]):
                model = fixtures._CoupledQuadraticExcessModel([[-4., 3.], [3., -4.]])
                term = FourierStabilityLoss(modes=((1, 0, 0),), mixture_mode=mode,
                                            relative_amplitude=amplitude, **kwargs)
                rng = torch.get_rng_state().clone()
                value = term(model(data), data, model=model)
                value.backward()
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                values.append(value.detach()); gradients.append(model.matrix.grad.clone())
            for value, grad in zip(values[1:], gradients[1:]):
                self.assertTrue(torch.equal(value, values[0]))
                self.assertTrue(torch.equal(grad, gradients[0]))

    def test_per_mode_matrices_and_projections_match_scalar_calls(self):
        data = batch()
        modes = torch.tensor([[[1, 0, 0], [2, 0, 0]]] * 2)
        amplitudes = torch.tensor([[.02, .04], [.07, .1]], dtype=torch.float64)
        model = fixtures._CoupledQuadraticExcessModel([[.2, .3], [.3, -.4]])
        response = FourierResponse()
        directions = torch.tensor([[1., 0.], [0., 1.], [1., -1.]])
        matrices, _ = response.matrix(model, data, modes, relative_amplitude=amplitudes)
        projected, _ = response(model, data, modes, directions, relative_amplitude=amplitudes)
        for f in range(2):
            one = {k: v[f:f+1] for k, v in data.items()}
            for m in range(2):
                selected = modes[f:f+1, m:m+1]
                scalar = FourierResponse(relative_amplitude=float(amplitudes[f, m]))
                expected, _ = scalar.matrix(model, one, selected)
                torch.testing.assert_close(matrices[f, m], expected[0, 0], atol=1e-10, rtol=1e-10)
                expected, _ = scalar(model, one, selected, directions)
                torch.testing.assert_close(projected[f, m], expected[0, 0], atol=1e-10, rtol=1e-10)
        self.assertEqual(response.relative_amplitude, .01)

    def test_quadratic_hessian_and_chunked_gradients(self):
        data = batch()
        modes = torch.tensor([[[1, 0, 0], [2, 0, 0]]] * 2)
        amplitudes = torch.tensor([[.01, .02], [.03, .04]], dtype=torch.float64)
        model = fixtures._CoupledQuadraticExcessModel([[.2, .3], [.3, -.4]])
        # Subtract finite-amplitude ideal response to isolate the known
        # quadratic excess Hessian exactly rather than approximating ideal F.
        ideal = fixtures._CoupledQuadraticExcessModel(torch.zeros(2, 2))
        results, gradients = [], []
        for chunk in (None, 2):
            response = FourierResponse(perturbations_per_forward=chunk)
            value, _ = response.matrix(model, data, modes, relative_amplitude=amplitudes)
            ideal_value, _ = response.matrix(ideal, data, modes, relative_amplitude=amplitudes)
            expected = .5 * model.matrix
            torch.testing.assert_close(value - ideal_value, expected.expand_as(value), atol=1e-9, rtol=1e-9)
            results.append(value.detach())
            gradients.append(torch.autograd.grad(value.square().sum(), model.matrix)[0])
        torch.testing.assert_close(results[0], results[1], atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(gradients[0], gradients[1], atol=1e-9, rtol=1e-9)

    def test_masks_counts_and_fractional_bounds(self):
        data = batch(1)
        data['rho'][:, 0, :] = 0.
        data['excluded_mask'] = torch.zeros((1, 8), dtype=torch.bool)
        data['excluded_mask'][:, 0] = True
        data['rho'][:, 1:, :] *= torch.linspace(.2, 1., 7)[None, :, None]
        model = fixtures._CoupledQuadraticExcessModel([[0., 3.], [3., 0.]])
        original = data['rho'].clone()
        seen = []
        forward = model.forward
        def record(fields, **kwargs):
            seen.append(fields['rho'].detach().clone())
            return forward(fields, **kwargs)
        with patch.object(model, 'forward', side_effect=record):
            term = FourierStabilityLoss(modes=((1, 0, 0), (2, 0, 0)), mixture_mode='full_matrix',
                                        relative_amplitude=(.1, .3), perturbations_per_forward=2)
            term(model(data), data, model=model)
        for rho in seen:
            rho = rho.reshape(-1, original.shape[-2], original.shape[-1])
            self.assertTrue(torch.all(rho >= 0))
            self.assertTrue(torch.all(rho[:, 0] == 0))
            torch.testing.assert_close(rho.sum(-2), original.sum(-2).expand(rho.shape[0], -1))
            self.assertTrue(torch.all((rho - original).abs() <= .3 * original + 1e-12))
        self.assertTrue(torch.equal(data['rho'], original))

    def test_override_validation(self):
        data = batch()
        model = fixtures._CoupledQuadraticExcessModel([[0., 3.], [3., 0.]])
        modes = torch.tensor([[[1, 0, 0], [2, 0, 0]]] * 2)
        for amplitude in (torch.ones(2), torch.zeros(2, 2), torch.ones(2, 2),
                          torch.full((2, 2), float('nan')), torch.ones(2, 2, dtype=torch.bool),
                          torch.ones(2, 2, dtype=torch.complex64), -1.):
            with self.subTest(amplitude=amplitude), self.assertRaises((ValueError, TypeError)):
                FourierResponse().matrix(model, data, modes, relative_amplitude=amplitude)

    def test_trainer_checkpoint_replays_random_modes_and_amplitudes(self):
        data = batch(2)
        dataset = [{k: v[i] for k, v in data.items()} for i in range(2)]
        def setup(path=None):
            model = fixtures._CoupledQuadraticExcessModel([[-4., 3.], [3., -4.]])
            term = FourierStabilityLoss(random_modes_per_field=2, mode_domain='cube',
                                        mixture_mode='full_matrix', relative_amplitude=(.02, .1))
            trainer = Trainer(model, Loss([term]), optimizer_args={'lr': .001},
                              device='cpu', checkpoint_dir=path)
            loader = DataLoader(dataset, batch_size=1, shuffle=True,
                                generator=torch.Generator().manual_seed(10))
            return trainer, loader
        torch.manual_seed(17)
        full, loader = setup()
        full.fit(loader, loader, epochs=3, verbose=False)
        with tempfile.TemporaryDirectory() as tmp:
            torch.manual_seed(17)
            first, loader = setup(tmp)
            first.fit(loader, loader, epochs=1, verbose=False)
            resumed, loader = setup()
            resumed.load_checkpoint(Path(tmp) / 'last.pt', train_loader=loader)
            resumed.fit(loader, loader, epochs=2, verbose=False)
            self.assertTrue(torch.equal(full.model.matrix, resumed.model.matrix))
            self.assertEqual(full.history, resumed.history)


if __name__ == '__main__':
    unittest.main()
