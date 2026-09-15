import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from ase import Atoms
from ase.io import write
from torch import nn
from torch.utils.data import DataLoader

from equicdft import (
    CartesianAFeatures, CartesianBFeatures, GridCACEModel, GridData,
    LDAReadout, LocalReadout, Loss, TensorLoss, Trainer, TrainingStream,
)
from equicdft._density_noise import add_density_noise


class RecordingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))
        self.seen = []

    def forward(self, batch):
        self.seen.append((self.training, batch['rho'].detach().clone()))
        return {'prediction': self.weight * batch['rho']}


def loss():
    return Loss([TensorLoss('density', 'prediction', 'target', loss_fn=nn.MSELoss())])


def loader():
    return DataLoader([
        {'rho': torch.ones(4, 3) * value,
         'rho_std': torch.ones(4, 3) * 0.1,
         'target': torch.ones(4, 3)} for value in (1., 2.)
    ], batch_size=1, shuffle=True, generator=torch.Generator().manual_seed(2))


def trainer(**kwargs):
    return Trainer(RecordingModel(), loss(), optimizer_cls=torch.optim.SGD,
                   optimizer_args={'lr': 0.01}, device='cpu', **kwargs)


class TestDensityNoise(unittest.TestCase):
    def test_literal_sem_floor_mask_and_immutable_input(self):
        rho = torch.tensor([[[0.0, 0.0], [0.1, 0.4], [0.2, 0.8]]])
        sem = torch.tensor([[[0., 0.], [0.2, 0.3], [0., 0.1]]])
        batch = {'rho': rho, 'rho_std': sem,
                 'excluded_mask': torch.tensor([[True, False, False]])}
        before = rho.clone()
        with patch('torch.randn_like', return_value=torch.full_like(rho, -1.)):
            out = add_density_noise(batch, 0.05)
        expected = torch.tensor([[[0., 0.], [0.05, 0.1], [0.2, 0.7]]])
        torch.testing.assert_close(out['rho'], expected)
        self.assertTrue(torch.equal(rho, before))
        self.assertNotEqual(out['rho'].data_ptr(), rho.data_ptr())
        self.assertIs(out['rho_std'], sem)
        self.assertFalse(torch.equal(out['rho'].sum(dim=1), rho.sum(dim=1)))

    def test_proposal_moments_fresh_draws_and_zero_sigma(self):
        torch.manual_seed(18)
        rho = torch.full((1, 100000, 2), 10.)
        sem = torch.tensor([0.1, 0.4]).expand_as(rho)
        batch = {'rho': rho, 'rho_std': sem}
        a = add_density_noise(batch, 0.)['rho'] - rho
        b = add_density_noise(batch, 0.)['rho'] - rho
        torch.testing.assert_close(a.mean(dim=1), torch.zeros(1, 2), atol=.004, rtol=0)
        torch.testing.assert_close(a.std(dim=1), torch.tensor([[.1, .4]]), atol=.004, rtol=0)
        self.assertFalse(torch.equal(a, b))
        batch['rho_std'] = torch.zeros_like(rho)
        self.assertTrue(torch.equal(add_density_noise(batch, 0.)['rho'], rho))

    def test_derived_logarithmic_targets_use_noisy_rho(self):
        rho = torch.ones(2, 4, 3)
        batch = {'rho': rho, 'rho_std': rho * .1,
                 'thermal_wavelength': torch.full((2, 3), 2.),
                 'beta': torch.tensor([2., 3.]), 'V_ext': rho * .4,
                 'beta_mu': torch.ones(2, 3) * .2,
                 'c1': rho.clone(), 'c1_plus_beta_mu': rho.clone()}
        with patch('torch.randn_like', return_value=torch.ones_like(rho)):
            out = add_density_noise(batch, 0.)
        expected = torch.log(out['rho'] * 8.) + batch['beta'][:, None, None] * .4
        torch.testing.assert_close(out['c1_plus_beta_mu'], expected)
        torch.testing.assert_close(out['c1'], expected - .2)
        with patch('torch.randn_like', return_value=-100 * rho):
            out = add_density_noise(batch, 0.)
        self.assertTrue(torch.isfinite(out['c1']).all())

    def test_invalid_noise(self):
        rho = torch.ones(1, 2, 3)
        with self.assertRaisesRegex(KeyError, 'requires rho_std'):
            add_density_noise({'rho': rho}, 0.)
        for std in (torch.ones(2), -rho, rho * float('nan'), rho * float('inf')):
            with self.assertRaises(ValueError):
                add_density_noise({'rho': rho, 'rho_std': std}, 0.)
        for value in (-1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                trainer(density_noise_floor=value)

    def test_train_only_and_clean_explicit_training_evaluation(self):
        t = trainer(density_noise=True)
        train, valid = loader(), loader()
        with patch('torch.randn_like', side_effect=lambda x: torch.ones_like(x)):
            t.fit(train, valid, epochs=1, verbose=False)
        for is_training, rho in t.model.seen:
            expected = rho.round() + (.1 if is_training else 0.)
            torch.testing.assert_close(rho, expected)
        t.model.seen.clear()
        t.evaluate_streams([TrainingStream('field', train, valid, t.loss,
                                           density_noise=True)], subset='train')
        for is_training, rho in t.model.seen:
            self.assertFalse(is_training)
            torch.testing.assert_close(rho, rho.round())
        for frame in train.dataset:
            torch.testing.assert_close(frame['rho'], frame['rho'].round())

    def test_disabled_never_samples_and_joint_stream_opt_in(self):
        t = trainer()
        with patch('torch.randn_like', side_effect=AssertionError('unexpected noise')):
            t.fit(loader(), loader(), epochs=1, verbose=False)
        t = trainer()
        streams = [TrainingStream('field', loader(), loader(), loss(), density_noise=True),
                   TrainingStream('response', loader(), loader(), loss())]
        with patch('torch.randn_like', side_effect=lambda x: torch.ones_like(x)) as draws:
            t.fit_streams(streams, epochs=1, verbose=False)
        self.assertEqual(draws.call_count, 2)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'joint.pt'
            t.save_checkpoint(checkpoint, t.history[-1])
            restored = trainer()
            restored_streams = [TrainingStream(s.name, loader(), loader(), loss()) for s in streams]
            restored.load_stream_checkpoint(checkpoint, restored_streams)
            self.assertTrue(restored_streams[0].density_noise)
            self.assertFalse(restored_streams[1].density_noise)

    def test_checkpoint_resume_is_exact_and_legacy_disables_noise(self):
        torch.manual_seed(7)
        full = trainer(density_noise=True)
        full.fit(loader(), loader(), epochs=3, verbose=False)
        with tempfile.TemporaryDirectory() as tmp:
            torch.manual_seed(7)
            first = trainer(density_noise=True, checkpoint_dir=tmp)
            first.fit(loader(), loader(), epochs=1, verbose=False)
            resumed = trainer()
            train, valid = loader(), loader()
            resumed.load_checkpoint(Path(tmp) / 'last.pt', train)
            self.assertTrue(resumed.density_noise)
            resumed.fit(train, valid, epochs=2, verbose=False)
            self.assertTrue(torch.equal(full.model.weight, resumed.model.weight))
            self.assertEqual(full.history, resumed.history)
            checkpoint = torch.load(Path(tmp) / 'last.pt', weights_only=False)
            for key in ('density_noise', 'density_noise_floor', 'stream_density_noise'):
                checkpoint.pop(key)
            legacy = Path(tmp) / 'legacy.pt'
            torch.save(checkpoint, legacy)
            resumed.load_checkpoint(legacy, train)
            self.assertFalse(resumed.density_noise)

    def test_real_functional_gradients_and_lazy_initialization_are_safe(self):
        atoms = TestDensityUncertaintyData.frame(n_types=2)
        atoms.arrays['V_ext'] = .2 * np.sin(atoms.arrays['density'])
        fields = TestDensityUncertaintyData().read(atoms)
        model = GridCACEModel(
            a_features=CartesianAFeatures(
                mean_density=1., cutoff_grid=1, max_power=1, n_types=2,
                radial_basis='gaussian', radial_exponents=(.2,),
                convolution_backend='fft'),
            b_features=CartesianBFeatures(max_power=1, max_product_order=1),
            readout=[LDAReadout(mean_density=1., n_types=2, hidden_sizes=(4,)),
                     LocalReadout(n_types=2, hidden_sizes=(4,))],
            grid_spacing=.5, mean_temperature=1., boltzmann_constant=1.,
            thermal_wavelength=1., free_energy_mode='beta', compute_local_mu=True,
        )
        seen = []
        hook = model.register_forward_pre_hook(
            lambda module, args: seen.append((module.training, args[0]['rho'].detach().clone()))
        )
        t = Trainer(model, Loss([TensorLoss(
            'mu', 'local_chemical_potential', 'average_chemical_potential',
            weights_key='chemical_potential_weights')]),
            density_noise=True, optimizer_args={'lr': 1.e-4}, device='cpu')
        clean = fields[0]['rho'].clone()
        draws = torch.zeros_like(clean).unsqueeze(0)
        draws[:, :5] = -1000  # Force zeros at formerly positive cells.
        with patch('torch.randn_like', return_value=draws):
            t.fit(DataLoader(fields), DataLoader(fields), epochs=1, verbose=False)
        hook.remove()
        self.assertGreater(len(seen), 2)  # lazy initialization, train, validation
        for is_training, rho in seen:
            if not is_training:
                torch.testing.assert_close(rho[0], clean)
        self.assertTrue(torch.equal(fields[0]['rho'], clean))
        self.assertTrue(np.isfinite(t.history[0]['train_losses']['total']))
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertTrue(any(torch.any(g != 0) for g in grads))


class TestDensityUncertaintyData(unittest.TestCase):
    @staticmethod
    def frame(n_types=3, sem=True):
        q = np.indices((4, 4, 4)).reshape(3, -1).T
        order = np.random.default_rng(5).permutation(64)
        atoms = Atoms('X64', positions=q[order], cell=[4, 4, 4], pbc=True)
        rho = np.arange(64 * n_types).reshape(64, n_types) * .01
        atoms.arrays['density'] = rho[order]
        if sem:
            atoms.arrays['density_sem'] = (rho * .03)[order]
        atoms.info.update(grid_size=[4, 4, 4], grid_spacing=[.5, .5, .5],
                          grid_indexing='zero_based', T=1.)
        return atoms

    def read(self, frames, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'data.extxyz'
            write(path, frames, format='extxyz')
            return GridData.from_xyz(path, cutoff_grid=1, boltzmann_constant=1.,
                                     data_key={'rho_std': 'density_sem'}, **kwargs)

    def test_sem_mapping_order_multispecies_and_optional(self):
        for n in (1, 2, 3):
            frame = self.read(self.frame(n))[0]
            self.assertEqual(frame['rho_std'].shape, (64, n))
            torch.testing.assert_close(frame['rho_std'], frame['rho'] * .03)
        self.assertNotIn('rho_std', self.read(self.frame(sem=False))[0])

    def test_invalid_data_and_mixed_missing(self):
        for bad in (-1., float('nan'), float('inf')):
            atoms = self.frame()
            atoms.arrays['density_sem'][0, 0] = bad
            with self.assertRaises(ValueError):
                self.read(atoms)
        atoms = self.frame()
        atoms.arrays['density_sem'] = np.ones((64, 2))
        with self.assertRaises(ValueError):
            self.read(atoms)
        with self.assertRaisesRegex(ValueError, 'every frame or none'):
            self.read([self.frame(), self.frame(sem=False)])

    def test_masks_and_no_uncertainty_coarsening(self):
        atoms = self.frame()
        atoms.arrays['excluded_mask'] = np.zeros(64, dtype=bool)
        atoms.arrays['excluded_mask'][0] = True
        atoms.arrays['density'][0] = 0.
        with self.assertRaisesRegex(ValueError, 'zero at excluded'):
            self.read(atoms)
        atoms.arrays['density_sem'][0] = 0.
        data = self.read(atoms)[0]
        self.assertTrue(torch.all(data['rho_std'][data['excluded_mask']] == 0))
        self.read(self.frame(), target_grid_spacing=.5)
        with self.assertRaisesRegex(ValueError, 'covariance'):
            self.read(self.frame(), target_grid_spacing=1.)


if __name__ == '__main__':
    unittest.main()
