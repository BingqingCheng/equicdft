import unittest
import torch
from equicdft.evolution import (FluxLimitedSemiImplicitIntegrator as Limited,
    SemiImplicitIntegrator, FluxLimitedEulerIntegrator, ConstantMobility,
    AdaptiveDynamics, StepControl)
from tests.evolution.test_integrator import field, QuadraticModel


class TestLimitedImplicit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_inactive_matches_existing_masked_and_periodic(self):
        for masked in (False, True):
            data = field(species=3)
            h = torch.tensor([[2., .2, -.1], [.2, 1., .3], [-.1, .3, 2.]], dtype=torch.float64)
            if masked:
                data['excluded_mask'] = data['grid_positions'][:, 0] == 0
                data['rho'][data['excluded_mask']] = 0
            opts = dict(reference_density=[1.]*3, reference_curvature=h)
            model, mob = QuadraticModel(h), ConstantMobility(.1)
            integ = Limited(model, mob, .01, density_floor=1e-6, **opts)
            actual = integ.step(data)
            expected = SemiImplicitIntegrator(model, mob, .01, **opts).step(data)
            torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
            self.assertEqual(integ.last_limiter['limited_faces'], 0)

    def test_large_step_floor_and_disconnected_region_conservation(self):
        data = field(species=3)
        x = data['grid_positions'][:, 0]
        data['excluded_mask'] = x == 0
        data['mask'] = x == 4
        data['rho'][::2] = 1e-6
        data['rho'][data['excluded_mask']] = 0
        data['V_ext'] = 100*torch.cos(x[:, None].double()).expand_as(data['rho'])
        integ = Limited(QuadraticModel(torch.eye(3)), ConstantMobility(.1), 10.,
                        reference_density=[1.]*3, reference_curvature=torch.eye(3), density_floor=1e-6)
        result = integ.step(data)
        self.assertTrue((result[~data['excluded_mask']] >= 1e-6).all())
        self.assertTrue(torch.equal(result[data['mask']], data['rho'][data['mask']]))
        self.assertTrue((result[data['excluded_mask']] == 0).all())
        for region in integ._grid.free_regions():
            torch.testing.assert_close(result[region].sum(0), data['rho'][region].sum(0), rtol=1e-12, atol=1e-12)
        self.assertGreater(integ.last_limiter['limited_faces'], 0)

    def test_cached_rate_identity_and_permutation(self):
        data = field()
        opts = dict(reference_density=[1., 1.], reference_curvature=torch.eye(2), density_floor=1e-6)
        integ = Limited(QuadraticModel(torch.eye(2)), ConstantMobility(.1), .01, **opts)
        rate = integ.rhs(data)
        out = integ.advance(data, rate)
        integ.rhs(dict(data, rho=out))
        torch.testing.assert_close(integ.advance(data, rate), out, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'unmodified RHS'):
            integ.advance(dict(data, rho=out), rate)
        rate.add_(0)
        with self.assertRaisesRegex(ValueError, 'unmodified RHS'):
            integ.advance(data, rate)
        perm = torch.arange(len(out)-1, -1, -1)
        shuffled = {k: v[perm] if k in ('rho', 'grid_positions') else v for k, v in data.items()}
        torch.testing.assert_close(integ.step(shuffled), out[perm], rtol=0, atol=0)

    def test_control_uses_explicit_constrained_rate_not_smoothed_rate(self):
        data = field()
        data['rho'][::2] = 1e-6
        model, mob = QuadraticModel(torch.eye(2)), ConstantMobility(.1)
        integ = Limited(model, mob, 1., reference_density=[1., 1.],
                        reference_curvature=100*torch.eye(2), density_floor=1e-6)
        explicit = FluxLimitedEulerIntegrator(model, mob, 1., density_floor=1e-6)
        rate = integ.rhs(data)
        actual = integ.control_rate(data, rate, 1.)
        expected = explicit.control_rate(data, explicit.rhs(data), 1.)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertGreater(float((actual-(integ.advance(data, rate)-data['rho'])).abs().max()), 1e-4)

    def test_no_explicit_stability_cap_in_adaptive_driver(self):
        data = field()
        data['rho'].fill_(1.)
        integ = Limited(QuadraticModel(torch.eye(2)), ConstantMobility(.1), .02,
                        reference_density=[1., 1.], reference_curvature=torch.eye(2), density_floor=1e-6)
        integ.stability_limit = lambda *a: (_ for _ in ()).throw(AssertionError('explicit cap called'))
        result = AdaptiveDynamics(integ, StepControl(dt_min=1e-6, dt_max=.02)).run(data, .04)
        self.assertEqual(result['metadata']['dt_cap'], .02)
        torch.testing.assert_close(result['rho'][-1], data['rho'])

    def test_temporal_convergence_and_large_step_linear_stability(self):
        data = field(shape=(8, 2, 2), spacing=(1., 1., 1.))
        data['rho'] = 1 + .01*torch.cos(torch.pi*data['grid_positions'][:, :1].double()).expand_as(data['rho'])
        model, mob = QuadraticModel(torch.eye(2)), ConstantMobility(1.)
        errors = []
        for dt in (.1, .05, .025):
            integ = Limited(model, mob, dt, reference_density=[1., 1.],
                            reference_curvature=torch.eye(2), density_floor=1e-6)
            state = dict(data)
            for _ in range(round(.2/dt)):
                state = dict(state, rho=integ.step(state))
            expected = 1+(data['rho']-1)*torch.exp(-.8/data['beta'])
            errors.append(float((state['rho']-expected).abs().max()))
        self.assertLess(errors[1], errors[0]*.6)
        self.assertLess(errors[2], errors[1]*.6)
        integ.dt = 2.
        out = integ.step(data)
        self.assertLess(float((out-1).abs().max()), float((data['rho']-1).abs().max()))
        self.assertEqual(integ.last_limiter['limited_faces'], 0)

    def test_invalid_parameters(self):
        for name, values in [('density_floor', [-1., float('nan'), True]),
                             ('outflow_fraction', [0., 1., float('nan'), True])]:
            for value in values:
                with self.assertRaises(ValueError):
                    Limited(QuadraticModel(torch.eye(2)), ConstantMobility(1.), .01, **{name: value})
