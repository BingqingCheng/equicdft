import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from equicdft import FourierStabilityLoss, Loss, Trainer
from test_fourier_amplitudes import batch
import test_loss as fixtures


class TestFourierModeCounts(unittest.TestCase):
    def test_validation(self):
        for count in ((), (1,), (1, 2, 3), (0, 2), (-1, 2), (3, 2),
                      (True, 2), (1, 2.5), (1, float('inf')), True, 1.5):
            with self.subTest(count=count), self.assertRaises((ValueError, TypeError)):
                FourierStabilityLoss(random_modes_per_field=count)
        with self.assertRaisesRegex(ValueError, 'must be zero'):
            FourierStabilityLoss(modes=((1, 0, 0),), random_modes_per_field=(1, 2))

    def test_inclusive_draw_shared_count_independent_modes_and_replay(self):
        data = batch(8)
        term = FourierStabilityLoss(random_modes_per_field=(1, 3), mode_domain='cube')
        def sample():
            return [term._select_modes(data, data['rho']) for _ in range(90)]
        torch.manual_seed(17)
        first = sample()
        torch.manual_seed(17)
        replay = sample()
        counts = [m.shape[1] for m in first]
        self.assertEqual(set(counts), {1, 2, 3})
        for n in (1, 2, 3):
            self.assertGreater(counts.count(n), 15)
        for a, b in zip(first, replay):
            self.assertTrue(torch.equal(a, b))
            self.assertEqual(a.shape, (8, a.shape[1], 3))
            self.assertTrue(torch.any(a != a[:1]))
            self.assertTrue(torch.all(torch.any(a != 0, dim=-1)))
            for modes in a:
                self.assertEqual(torch.unique(modes, dim=0).shape[0], a.shape[1])

    def test_fixed_count_equal_endpoints_values_gradients_rng(self):
        data = batch()
        for mixture in ('independent', 'charge', 'total_density', 'full_matrix'):
            results = []
            for count in (2, (2, 2), [2, 2]):
                model = fixtures._CoupledQuadraticExcessModel([[-4., 3.], [3., -4.]])
                kwargs = {'charges': (1., -1.)} if mixture == 'charge' else {}
                term = FourierStabilityLoss(random_modes_per_field=count, mode_domain='cube',
                                            mixture_mode=mixture, relative_amplitude=(.01, .1), **kwargs)
                torch.manual_seed(9)
                value = term(model(data), data, model=model)
                value.backward()
                results.append((value.detach(), model.matrix.grad, torch.get_rng_state()))
            for result in results[1:]:
                for a, b in zip(results[0], result):
                    self.assertTrue(torch.equal(a, b))

    def test_upper_endpoint_validated_even_when_small_count_drawn(self):
        data = batch()
        term = FourierStabilityLoss(random_modes_per_field=(1, 100), mode_domain='cube')
        with patch('torch.randint', return_value=torch.tensor(1)):
            with self.assertRaisesRegex(ValueError, 'exceeds the feasible'):
                term._select_modes(data, data['rho'])
        term = FourierStabilityLoss(random_modes_per_field=(1, 2),
                                    wavevector_range=(.7, .8), mode_domain='cube')
        with self.assertRaisesRegex(ValueError, 'wavevector_range'):
            term._select_modes(data, data['rho'])

    def test_eval_no_draw_and_loss_is_mean_not_sum(self):
        data = batch()
        model = fixtures._CoupledQuadraticExcessModel([[-4., 3.], [3., -4.]])
        term = FourierStabilityLoss(random_modes_per_field=(1, 3), mode_domain='cube',
                                    mixture_mode='full_matrix', relative_amplitude=(.01, .1))
        def matrix(**kwargs):
            shape = kwargs['modes'].shape[:2]
            value = -2 * torch.eye(2).expand(*shape, 2, 2)
            return value, torch.ones(*shape, 2, dtype=torch.bool)
        with patch.object(term.response, 'matrix', side_effect=matrix):
            for n in (1, 2, 3):
                with patch('torch.randint', return_value=torch.tensor(n)):
                    self.assertEqual(float(term(model(data), data, model=model)), 4.)
        term.eval()
        rng = torch.get_rng_state().clone()
        self.assertEqual(float(term(model(data), data, model=model)), 0.)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_checkpoint_replays_counts_modes_amplitudes(self):
        data = batch()
        dataset = [{k: v[i] for k, v in data.items()} for i in range(2)]
        def setup(path=None):
            model = fixtures._CoupledQuadraticExcessModel([[-4., 3.], [3., -4.]])
            term = FourierStabilityLoss(random_modes_per_field=(1, 3), mode_domain='cube',
                                        mixture_mode='full_matrix', relative_amplitude=(.01, .1))
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
