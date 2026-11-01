import math
import unittest
import torch
from equicdft.evolution import (FluxLimitedEulerIntegrator, EulerIntegrator, ConstantMobility,
                               LocalMobility, AdaptiveDynamics, StepControl, RampedField)
from tests.evolution.test_field import Switch
from tests.evolution.test_integrator import field, QuadraticModel
from tests.evolution.test_harmonic_faces import IdealModel


class TestFluxLimitedEuler(unittest.TestCase):
    def test_floor_conserves_each_species_without_clipping(self):
        data = field(species=3)
        floor = 1e-6
        data['rho'][::2] = floor
        data['V_ext'] = 100*torch.cos(data['grid_positions'][:, :1].double()).expand_as(data['rho'])
        integ = FluxLimitedEulerIntegrator(QuadraticModel(torch.eye(3)), ConstantMobility(.1),
                                          10., density_floor=floor)
        out = integ.step(data)
        self.assertTrue((out >= floor).all())
        torch.testing.assert_close(out.sum(0), data['rho'].sum(0), rtol=1e-13, atol=1e-13)

    def test_controller_uses_constrained_flux_at_floor(self):
        data = field()
        data['rho'].fill_(1e-8)
        data['V_ext'] = torch.cos(data['grid_positions'][:, :1].double()).expand_as(data['rho'])
        integ = FluxLimitedEulerIntegrator(QuadraticModel(torch.eye(2)), ConstantMobility(.1),
                                          .001, density_floor=1e-8)
        rate = integ.rhs(data)
        self.assertGreater(float(rate.abs().max()), .01)
        self.assertEqual(float(integ.control_rate(data, rate, .001).abs().max()), 0.)
        result = AdaptiveDynamics(integ, StepControl(dt_min=1e-6, dt_max=.002)).run(data, .005)
        self.assertTrue(result['metadata']['completed'])
        self.assertLess(result['metadata']['accepted'], 10)
        self.assertTrue(torch.equal(result['rho'][-1], data['rho']))

    def test_invalid_density_floors_rejected(self):
        for value in (-1., float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                FluxLimitedEulerIntegrator(QuadraticModel(torch.eye(2)), ConstantMobility(.1),
                                           .001, density_floor=value)

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_inactive_limiter_is_euler_and_cached_rates_survive_other_rhs(self):
        data = field()
        model, mobility = QuadraticModel([[1., .2], [.2, 1.]]), ConstantMobility(.1)
        integrator = FluxLimitedEulerIntegrator(model, mobility, .001)
        rate = integrator.rhs(data)
        first = integrator.advance(data, rate)
        integrator.rhs(dict(data, rho=first))
        again = integrator.advance(data, rate)
        expected = EulerIntegrator(model, mobility, .001).step(data)
        torch.testing.assert_close(first, expected, rtol=0, atol=0)
        torch.testing.assert_close(again, expected, rtol=0, atol=0)
        self.assertEqual(integrator.last_limiter['limited_faces'], 0)
        with self.assertRaisesRegex(ValueError, 'unmodified RHS'):
            integrator.advance(dict(data, rho=first), rate)
        rate.add_(0.)
        with self.assertRaisesRegex(ValueError, 'unmodified RHS'):
            integrator.advance(data, rate)

    def test_large_step_positive_conservative_and_descent_direction(self):
        data = field(species=3)
        generator = torch.Generator().manual_seed(881)
        data['rho'] = torch.exp(-110*torch.rand(data['rho'].shape, generator=generator, dtype=torch.float64))
        data['V_ext'] = torch.randn(data['rho'].shape, generator=generator, dtype=torch.float64)
        m = torch.tensor([[2., -.2, .1], [-.2, 1., .3], [.1, .3, 1.]], dtype=torch.float64)
        integrator = FluxLimitedEulerIntegrator(IdealModel(), ConstantMobility(m), 100.)
        result = integrator.step(data)
        self.assertTrue((result >= .099999*data['rho']).all())
        torch.testing.assert_close(result.sum(0), data['rho'].sum(0), rtol=1e-13, atol=1e-13)
        direction = result-data['rho']
        mu = data['rho'].log()+data['beta']*data['V_ext']
        self.assertLessEqual(float((mu*direction).sum()), 1e-14)
        self.assertGreater(integrator.last_limiter['limited_faces'], 0)

    def test_small_composition_channel_with_mixed_error_norm(self):
        data = field(shape=(12, 2, 2), spacing=(1., 1., 1.))
        # Both species are deep inside a core, with a severe composition ratio.
        # A small canonical-gauge variation leaves large *relative* rates in
        # the minority species but a negligible absolute particle transfer.
        data['rho'][:] = data['rho'].new_tensor([1e-48, 1e-20])
        data['V_ext'] = -data['rho'].log()/data['beta']
        data['V_ext'] += .01*torch.cos(data['grid_positions'][:, :1].double())
        mobility = LocalMobility(lambda r, t: r.sqrt()[..., :, None]*r.new_tensor([[1., .3], [.3, 1.]])*r.sqrt()[..., None, :])
        integ = FluxLimitedEulerIntegrator(IdealModel(), mobility, .001, face_average='harmonic-density')
        rate = integ.rhs(data)
        self.assertGreater(float((rate.abs()/data['rho']).max()), 1e8)
        control = StepControl(dt_min=1e-10, dt_max=.001, density_absolute_tolerance=1e-14)
        result = AdaptiveDynamics(integ, control).run(data, .005)
        self.assertTrue(result['metadata']['completed'])
        self.assertTrue((result['rho'] > 0).all())
        torch.testing.assert_close(result['rho'][-1].sum(0), data['rho'].sum(0), rtol=1e-13, atol=0)
        self.assertLess(float(result['rho'][-1][:, 0].max()), 1e-30)

    def test_frozen_excluded_and_permuted_rows(self):
        data = field()
        data['excluded_mask'] = data['grid_positions'][:, 0] == 0
        data['mask'] = data['grid_positions'][:, 0] == 4
        data['rho'][data['excluded_mask']] = 0.
        integ = FluxLimitedEulerIntegrator(QuadraticModel([[1., .2], [.2, 1.]]), ConstantMobility(.1), 100.)
        expected = integ.step(data)
        torch.testing.assert_close(expected.sum(0), data['rho'].sum(0), rtol=1e-14, atol=1e-14)
        self.assertTrue(torch.equal(expected[data['mask']], data['rho'][data['mask']]))
        self.assertTrue((expected[data['excluded_mask']] == 0).all())
        perm = torch.arange(len(expected)-1, -1, -1)
        shuffled = {k: (v[perm] if k in ('rho', 'grid_positions', 'excluded_mask', 'mask') else v)
                    for k, v in data.items()}
        torch.testing.assert_close(integ.step(shuffled), expected[perm], rtol=0, atol=0)

    def test_repeated_depletion_does_not_underflow_or_create_mass(self):
        for dtype in (torch.float32, torch.float64):
            data = field(shape=(4, 2, 2), spacing=(1., 1., 1.))
            data = {k: v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v
                    for k, v in data.items()}
            tiny = torch.finfo(dtype).tiny
            data['rho'][:] = tiny*100
            # Constant positive x-directed flux drains x=0; others receive.
            integ = FluxLimitedEulerIntegrator(QuadraticModel([[1., 0.], [0., 1.]]).to(dtype), ConstantMobility(1.), 1.)
            initial = data['rho'].sum(0)
            for _ in range(400):
                rate = integ.rhs(data)
                flux = torch.zeros((4, 2, 2, 2), dtype=dtype)
                flux[0] = 1.
                rate._face_fluxes = (flux, torch.zeros_like(flux), torch.zeros_like(flux))
                data['rho'] = integ.advance(data, rate)
                self.assertTrue((data['rho'] > 0).all())
            self.assertTrue((data['rho'] >= tiny).all())
            torch.testing.assert_close(data['rho'].sum(0)/initial, torch.ones_like(initial),
                                       rtol=20*torch.finfo(dtype).eps, atol=0)

    def test_absolute_error_scale_not_a_floor_and_validation(self):
        data = field()
        integ = FluxLimitedEulerIntegrator(QuadraticModel([[1., 0.], [0., 1.]]), ConstantMobility(0.), .01)
        rho = torch.tensor([[1e-49, 2.]], dtype=torch.float64)
        control = StepControl(dt_min=1e-10, dt_max=.01, error_tolerance=1e-3, density_absolute_tolerance=1e-14)
        driver = AdaptiveDynamics(integ, control)
        torch.testing.assert_close(control.error_tolerance*driver._density_scale(rho), 1e-3*rho+1e-14)
        self.assertEqual(float(rho[0, 0]), 1e-49)
        result = driver.run(data, .02)
        torch.testing.assert_close(result['rho'][-1], data['rho'], rtol=0, atol=0)
        for value in (-1., math.inf, math.nan, True):
            with self.assertRaises(ValueError):
                StepControl(dt_min=1e-10, dt_max=.01, density_absolute_tolerance=value)
        for value in (0., 1., -1., math.nan, True):
            with self.assertRaises(ValueError):
                FluxLimitedEulerIntegrator(integ.model, integ.mobility, .01, outflow_fraction=value)

    def _compare_driven_euler(self, switching):
        data = field()
        potential = .1*torch.cos(data['grid_positions'][:, :1].double()).expand_as(data['rho'])
        drive = Switch(potential, .015) if switching else RampedField(torch.zeros_like(potential), potential, .03)
        control = StepControl(dt_min=1e-10, dt_max=.01)
        results = []
        for cls in (EulerIntegrator, FluxLimitedEulerIntegrator):
            integ = cls(QuadraticModel([[1., .2], [.2, 1.]]), ConstantMobility(.1), .005)
            result = AdaptiveDynamics(integ, control, drive).run(data, .04)
            self.assertTrue(result['metadata']['completed'])
            self.assertGreater(result['metadata']['accepted'], 2)
            results.append(result)
        for key in ('time', 'rho', 'V_ext', 'beta_A', 'beta_work'):
            torch.testing.assert_close(results[0][key], results[1][key], rtol=0, atol=0)
        if switching:
            index = results[1]['time'].tolist().index(.015)
            torch.testing.assert_close(results[1]['V_ext'][index], potential, rtol=0, atol=0)

    def test_moving_field_equal_endpoint_copies_preserve_rate_ownership(self):
        self._compare_driven_euler(False)

    def test_field_jump_refreshes_rate_and_preserves_right_limit(self):
        self._compare_driven_euler(True)


if __name__ == '__main__':
    unittest.main()
