"""Opt-in deterministic face quadrature for strongly depleted fluid regions."""
import math
import unittest
import torch
from torch import nn
from equicdft.evolution import (EulerIntegrator, SemiImplicitIntegrator, ConstantMobility,
                                DensityMobility, LocalMobility, AdaptiveDynamics, StepControl)
from tests.evolution.test_integrator import field, QuadraticModel


class IdealModel(nn.Module):
    def forward(self, data, compute_c1=False, **kwargs):
        rho = data['rho']
        safe = torch.where(rho > 0, rho, torch.ones_like(rho))
        result = {'beta_F': (rho*(safe.log()-1)).sum()*data['grid_spacing'].prod()}
        if compute_c1:
            result['density_derivative'] = safe.log()
        return result


class TestHarmonicFaces(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def make(self, model, mobility, dt=.001):
        return EulerIntegrator(model, mobility, dt, face_average='harmonic-density')

    def test_constant_mobility_is_unchanged_and_option_validated(self):
        data = field()
        model, mobility = QuadraticModel([[1., .2], [.2, 2.]]), ConstantMobility([[1., -.1], [-.1, 1.]])
        for cls in (EulerIntegrator, SemiImplicitIntegrator):
            a = cls(model, mobility, .001).step(data)
            b = cls(model, mobility, .001, face_average='harmonic-density').step(data)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, 'face_average'):
                cls(model, mobility, .001, face_average='invalid')

    def test_extreme_core_relative_rate_remains_finite_and_evolves(self):
        data = field(shape=(16, 2, 2), spacing=(1., 1., 1.))
        v = torch.where(data['grid_positions'][:, 0] < 8, 110., 0.).double()[:, None]
        rho = torch.exp(-v).expand(-1, 2).clone()
        data.update(rho=rho, V_ext=(v+1e-5*torch.sin(data['grid_positions'][:, :1].double()))
                    .expand_as(rho)/data['beta'])
        mobility = LocalMobility(lambda r, t: r.sqrt()[..., :, None]*r.new_tensor([[1., .3], [.3, .8]])*r.sqrt()[..., None, :])
        old = EulerIntegrator(IdealModel(), mobility, .001)
        new = self.make(IdealModel(), mobility)
        old_bound = .05/float((old.rhs(data).abs()/rho).max())
        new_bound = .05/float((new.rhs(data).abs()/rho).max())
        self.assertLess(old_bound, 1e-35)
        self.assertGreater(new_bound, .001)
        result = AdaptiveDynamics(new, StepControl(dt_min=1e-10, dt_max=.001)).run(data, .005)
        self.assertTrue(result['metadata']['completed'])
        self.assertEqual(result['metadata']['face_average'], 'harmonic-density')
        self.assertTrue((result['rho'] > 0).all())
        torch.testing.assert_close(result['rho'][-1].sum(0), rho.sum(0), rtol=1e-14, atol=1e-14)
        self.assertLessEqual(float(result['beta_A'][-1]), float(result['beta_A'][0])+1e-10)
        self.assertLess(float((result['rho'][-1]-rho).abs().max()), 1e-5)
        # No artificial floor: 110 kBT depletion remains represented.
        self.assertLess(float(result['rho'][-1].min()), 1e-45)

    def test_matrix_conservation_and_dissipation_three_species(self):
        data = field(shape=(7, 3, 2), species=3)
        gen = torch.Generator().manual_seed(918)
        data['rho'] = torch.exp(-40*torch.rand(data['rho'].shape, generator=gen, dtype=torch.float64))
        data['V_ext'] = torch.randn(data['rho'].shape, generator=gen, dtype=torch.float64)
        a = torch.tensor([[1., -.3, .2], [.2, 1., .1], [.1, .2, 1.]], dtype=torch.float64)
        m = a@a.T
        mobility = LocalMobility(lambda r, t: r.sqrt()[..., :, None]*m*r.sqrt()[..., None, :])
        integrator = self.make(IdealModel(), mobility)
        rate = integrator.rhs(data)
        mu = data['rho'].log()+data['beta']*data['V_ext']
        self.assertLess(float((mu*rate).sum()), 0.)
        torch.testing.assert_close(rate.sum(0), torch.zeros(3, dtype=torch.float64), atol=1e-12, rtol=0)
        perm = torch.randperm(len(rate), generator=gen)
        shuffled = dict(data, rho=data['rho'][perm], grid_positions=data['grid_positions'][perm], V_ext=data['V_ext'][perm])
        torch.testing.assert_close(integrator.rhs(shuffled), rate[perm], rtol=1e-13, atol=1e-13)

    def test_mask_and_excluded_faces_preserve_constraints(self):
        data = field(shape=(8, 2, 2))
        data['excluded_mask'] = data['grid_positions'][:, 0] == 0
        data['mask'] = data['grid_positions'][:, 0] == 4
        data['rho'][data['excluded_mask']] = 0.
        integ = self.make(IdealModel(), DensityMobility(.1))
        rate = integ.rhs(data)
        self.assertTrue((rate[data['excluded_mask'] | data['mask']] == 0).all())
        new = integ.step(data)
        torch.testing.assert_close(new.sum(0), data['rho'].sum(0), rtol=1e-14, atol=1e-14)
        self.assertTrue(torch.equal(new[data['mask']], data['rho'][data['mask']]))

    def test_second_order_spatial_consistency_for_smooth_diffusion(self):
        errors = []
        for n in (32, 64, 128):
            data = field(shape=(n, 1, 1), species=1, spacing=(2*math.pi/n, 1., 1.))
            x = data['grid_positions'][:, :1].double()*2*math.pi/n
            data['rho'] = 1.+.2*torch.cos(x)
            rate = self.make(IdealModel(), DensityMobility(.7)).rhs(data)
            errors.append(float((rate+.14*torch.cos(x)).square().mean().sqrt()))
        self.assertGreater(errors[0]/errors[1], 3.8)
        self.assertGreater(errors[1]/errors[2], 3.8)


if __name__ == '__main__':
    unittest.main()
