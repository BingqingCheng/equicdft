"""Local energy selection, unmasked neighbor information, and full LR."""

import copy
import unittest
from unittest.mock import patch

import torch

from equicdft import (
    CACEBranch, FixedDipoleIdeal, FourierResponse, GridCACEModel, LDAReadout,
    LocalReadout, Loss, NoiseOptions, TensorLoss, Trainer,
)
from equicdft.response.curvature import polarization_fourier_curvature
from tests.model import test_cace_branches as branches
from tests.representation import test_polarization_symmetry as symmetry


class TestVoxelMask(unittest.TestCase):
    def setUp(self):
        self.dtype, self.rng = torch.get_default_dtype(), torch.get_rng_state()
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(313)

    def tearDown(self):
        torch.set_default_dtype(self.dtype)
        torch.set_rng_state(self.rng)

    @staticmethod
    def data():
        data = symmetry.TestFullPolarizationSymmetry.data()
        data['dipole_density'] *= .1
        data['voxel_weights'] = torch.ones_like(data['rho']) * .3
        data['voxel_weights'][::3] = 0.
        return data

    def test_local_sums_and_full_lr_with_both_cace_forms_and_messages(self):
        for independent in (False, True):
            for messages in (0, 1):
                with self.subTest(independent=independent, messages=messages):
                    model = symmetry.TestFullPolarizationSymmetry.model(
                        'fft', .4, messages, independent).train()
                    data = self.data()
                    terms, hooks = [], []
                    for item in model.readout:
                        local = item.readout if isinstance(item, CACEBranch) else item
                        if isinstance(local, (LDAReadout, LocalReadout)):
                            hooks.append(local.register_forward_hook(
                                lambda module, args, value: terms.append(value)))
                    try:
                        out = model(data, use_voxel_mask=True)
                    finally:
                        for hook in hooks:
                            hook.remove()
                    mask = data['voxel_weights'] > 0
                    expected_local = model.voxel_volume * sum(
                        (data['rho'] * value * mask).sum() for value in terms)
                    lr = copy.deepcopy(model)
                    lr.readout = torch.nn.ModuleList([
                        item for item in lr.readout
                        if not isinstance(item, (CACEBranch, LocalReadout, LDAReadout))])
                    lr_full = lr(copy.deepcopy(data), compute_ideal=False)
                    lr_masked = lr(copy.deepcopy(data), compute_ideal=False, use_voxel_mask=True)
                    for key in ('beta_F_exc', 'c1', 'polarization_derivative_excess'):
                        torch.testing.assert_close(lr_full[key], lr_masked[key], atol=0, rtol=0)
                    torch.testing.assert_close(out['beta_F_exc'], expected_local + lr_full['beta_F_exc'])
                    expected_ideal = model.ideal.evaluate(data, model.voxel_volume, mask=mask)
                    torch.testing.assert_close(out['beta_F_id'], expected_ideal['beta_F_id'])
                    derivatives = torch.autograd.grad(out['beta_F'], (data['rho'], data['dipole_density']))
                    for value, key in zip(derivatives, ('density_derivative', 'polarization_derivative')):
                        torch.testing.assert_close(value / model.voxel_volume, out[key])
                    # Weight magnitudes score losses; they never rescale energy.
                    scaled = dict(copy.deepcopy(data), voxel_weights=data['voxel_weights'] * 7)
                    scaled_out = model(scaled, use_voxel_mask=True)
                    for key in ('beta_F', 'density_derivative', 'polarization_derivative'):
                        torch.testing.assert_close(out[key], scaled_out[key], atol=0, rtol=0)

    def test_masked_centers_still_influence_neighbor_features_and_gradients(self):
        for backend in ('gather', 'fft'):
            branch = branches.TestCACEBranches.branch(True, backend, messages=1)
            model = branches.TestCACEBranches.model([branch]).train()
            data = self.data()
            before = copy.deepcopy(data)
            out = model(data, compute_ideal=False, use_voxel_mask=True)
            omitted = data['voxel_weights'] == 0
            for key in ('rho', 'dipole_density'):
                torch.testing.assert_close(data[key], before[key], atol=0, rtol=0)
            for key in ('density_derivative_excess', 'polarization_derivative_excess'):
                self.assertGreater(out[key][omitted].abs().max().item(), 1e-9)
            changed = copy.deepcopy(before)
            changed['rho'][omitted] *= .5
            changed['dipole_density'][omitted] *= .5
            other = model(changed, compute_ideal=False, use_voxel_mask=True)
            self.assertGreater((out['beta_F_exc'] - other['beta_F_exc']).abs().item(), 1e-9)
            (out['c1'].square().sum() + out['polarization_derivative_excess'].square().sum()).backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_species_specific_mask_and_empty_local_energy(self):
        model = GridCACEModel(
            readout=[branches.TestCACEBranches.branch(True, n_types=2),
                     LDAReadout(n_types=2, include_polarization=True, zero_init=False)],
            ideal=FixedDipoleIdeal([.7, 1.2]), n_types=2, grid_spacing=.7,
            mean_density=.5, polarization_scale=.3, compute_polarization_derivative=True,
        ).double()
        data = self.data()
        data['rho'] = data['rho'].expand(-1, 2).clone()
        data['dipole_density'] = data['dipole_density'].expand(-1, 2, -1).clone()
        data['voxel_weights'] = torch.zeros_like(data['rho'])
        out = model(data, use_voxel_mask=True)
        for key in ('beta_F', 'density_derivative', 'polarization_derivative'):
            self.assertEqual(out[key].abs().max().item(), 0.)
        data['voxel_weights'][:, 1] = .2
        out = model(data, use_voxel_mask=True)
        self.assertFalse(out['local_energy_mask'][:, 0].any())
        self.assertTrue(out['local_energy_mask'][:, 1].all())
        self.assertEqual(out['ideal_density_derivative'][:, 0].abs().max().item(), 0.)
        self.assertGreater(out['density_derivative_excess'][:, 0].abs().max().item(), 0.)

    def test_trainer_freezes_weights_before_sem_and_supports_full_override(self):
        model = GridCACEModel(readout=[LDAReadout(zero_init=False)], mean_density=.5,
                              grid_spacing=1., rho_min=.15, compute_local_mu=False).double()
        data = dict(rho=torch.tensor([[[.1], [.2]]]), temperature=torch.ones(1),
                    rho_std=torch.full((1, 2, 1), .1), target=torch.zeros(1))
        loss = Loss([TensorLoss('energy', 'beta_F', 'target')])
        trainer = Trainer(model, loss, device='cpu')
        with patch('torch.randn_like', return_value=torch.tensor([[[1.], [-1.]]])):
            noisy, out, _, _ = trainer._forward_loss(data, loss, noise=NoiseOptions(density=True))
        torch.testing.assert_close(noisy['rho'], torch.tensor([[[.2], [.1]]]))
        torch.testing.assert_close(out['local_energy_mask'], data['rho'] > .15)
        self.assertNotIn('voxel_weights', data)
        _, validation, _, _ = trainer._forward_loss(data, loss)
        torch.testing.assert_close(validation['local_energy_mask'], out['local_energy_mask'])
        _, full, _, _ = trainer._forward_loss(data, loss, model_kwargs={'use_voxel_mask': False})
        self.assertNotIn('local_energy_mask', full)
        self.assertTrue(full['ideal_derivative_mask'].all())

    def test_fourier_uses_full_central_energy_even_after_a_masked_forward(self):
        model = symmetry.TestFullPolarizationSymmetry.model('fft', .4).train()
        data = self.data()
        data['grid_spacing'] = torch.full((3,), .7)
        batch = {key: value.unsqueeze(0) for key, value in data.items()}
        full = model(batch)
        masked = model(batch, use_voxel_mask=True)
        self.assertGreater((full['beta_F_exc'] - masked['beta_F_exc']).abs().item(), 1e-8)
        modes = torch.tensor([[[1, 0, 0]]])
        response = FourierResponse(relative_amplitude=.02)
        rho_full = response.matrix(model, batch, modes, outputs=full)
        rho_masked = response.matrix(model, batch, modes, outputs=masked)
        def p_response(outputs):
            return polarization_fourier_curvature(
                model, outputs, batch, batch['rho'], modes, relative_amplitude=.02,
                polarization_directions=torch.tensor([[1., 0., 0.]]))
        for actual, expected in zip((*rho_masked, *p_response(masked)), (*rho_full, *p_response(full))):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for value in (rho_masked[0], p_response(masked)[0]):
            gradients = torch.autograd.grad(value.sum(), tuple(model.parameters()),
                                            allow_unused=True, retain_graph=True)
            self.assertTrue(all(g is None or torch.isfinite(g).all() for g in gradients))


if __name__ == '__main__':
    unittest.main()
