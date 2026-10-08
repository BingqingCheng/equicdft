import unittest
from unittest import mock

import torch

from equicdft.evolution import (AdaptiveDynamics, ConstantMobility, Dynamics,
                               DensityMobility, EulerIntegrator, LocalMobility, StepControl,
                               StochasticEulerIntegrator)
from tests.evolution.test_integrator import QuadraticModel, field


class TestStochastic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.data = field(shape=(5, 2, 2))
        self.data['rho'].fill_(20)
        self.model = QuadraticModel(torch.eye(2, dtype=torch.float64))
        self.mobility = ConstantMobility(torch.tensor([[.3, -.1], [-.1, .2]], dtype=torch.float64))

    def make(self, seed=1, dt=.001, mobility=None):
        return StochasticEulerIntegrator(self.model, self.mobility if mobility is None else mobility,
                                         dt, seed=seed)

    def test_conservation_energy_can_rise_and_metadata(self):
        traj = Dynamics(self.make()).run(self.data, 10, save_every=3)
        torch.testing.assert_close(traj['particle_numbers'],
                                   traj['particle_numbers'][0].expand_as(traj['particle_numbers']),
                                   rtol=0, atol=1e-12)
        self.assertGreater(float(traj['beta_A'][1]), float(traj['beta_A'][0]))
        self.assertFalse(traj['metadata']['deterministic'])
        self.assertEqual(len(traj['rng_state']), len(traj['rho']))

    def test_restart_bitwise(self):
        full = Dynamics(self.make()).run(self.data, 8)
        first = Dynamics(self.make()).run(self.data, 3)
        state = dict(self.data, rho=first['rho'][-1].clone())
        resumed = self.make(seed=999)
        resumed.set_rng_state(state, first['rng_state'][-1])
        second = Dynamics(resumed).run(state, 5)
        self.assertTrue(torch.equal(full['rho'][3:], second['rho']))
        self.assertTrue(torch.equal(full['rng_state'][-1], second['rng_state'][-1]))
        self.assertFalse(torch.equal(full['rho'][1], self.make(seed=2).step(self.data)))

    def test_mask_and_permuted_rows(self):
        self.data['excluded_mask'] = torch.zeros(len(self.data['rho']), dtype=torch.bool)
        self.data['excluded_mask'][:4] = True
        self.data['rho'][:4] = 0
        order = torch.arange(len(self.data['rho'])-1, -1, -1)
        other = dict(self.data)
        for key in ('rho', 'grid_positions', 'excluded_mask'):
            other[key] = other[key][order]
        result = self.make().step(self.data)
        permuted = self.make().step(other)
        torch.testing.assert_close(result[order], permuted, rtol=0, atol=0)
        self.assertTrue(torch.equal(result[:4], torch.zeros_like(result[:4])))
        torch.testing.assert_close(result.sum(0), self.data['rho'].sum(0), rtol=0, atol=1e-12)

    def test_zero_mobility_and_null_species_mode(self):
        self.assertTrue(torch.equal(self.make(mobility=ConstantMobility(0.)).step(self.data), self.data['rho']))
        result = self.make(mobility=ConstantMobility(torch.ones(2, 2))).step(self.data)
        torch.testing.assert_close(result[:, 0]-result[:, 1], torch.zeros(len(result), dtype=result.dtype),
                                   rtol=0, atol=1e-14)

    def test_unsupported_controls(self):
        with self.assertRaises(TypeError):
            self.make(mobility=LocalMobility(lambda rho, T: rho))
        for seed in (True, -1, 1.5):
            with self.assertRaises(ValueError): self.make(seed=seed)
        with self.assertRaisesRegex(ValueError, 'deterministic-only'):
            AdaptiveDynamics(self.make(), StepControl(dt_min=1e-6, dt_max=.01))
        with self.assertRaises(ValueError): self.make().advance(self.data, torch.zeros_like(self.data['rho']))

    def test_failure_preserves_rng_and_last_valid_state(self):
        self.data['rho'].fill_(1e-12)
        integrator = self.make(dt=1.)
        initial_rng = integrator.get_rng_state(self.data)
        with self.assertRaises(ValueError) as caught:
            Dynamics(integrator).run(self.data, 2)
        traj = caught.exception.trajectory
        self.assertFalse(traj['metadata']['completed'])
        self.assertEqual(traj['metadata']['steps_completed'], 0)
        self.assertTrue(torch.equal(traj['rng_state'][-1], initial_rng))
        self.assertTrue(torch.equal(traj['rho'][-1], self.data['rho']))

    def test_flux_covariance_and_volume_scaling(self):
        # At one voxel Var(delta rho)=4 dt M/(beta dV) sum_i 1/h_i^2.
        integrator = self.make()
        grid = integrator._grid_for(self.data)
        samples = torch.stack([integrator._noise_increment(self.data, grid)[0] for _ in range(6000)])
        covariance = samples.T @ samples / len(samples)
        expected = 4 * integrator.dt / (grid.beta * grid.volume) * (1/grid.spacing.square()).sum() * self.mobility.value
        torch.testing.assert_close(covariance, expected, rtol=.08, atol=.001)
        doubled = dict(self.data, grid_spacing=self.data['grid_spacing']*2)
        a, b = self.make(), self.make()
        x = a._noise_increment(self.data, a._grid_for(self.data))
        y = b._noise_increment(doubled, b._grid_for(doubled))
        torch.testing.assert_close(y, x/2**2.5, rtol=1e-13, atol=1e-14)

    def test_stationary_quadratic_fluctuations(self):
        # Many independent spatial modes; scalar H_beta=1, M=1/beta=1.
        data = field(shape=(4, 4, 4), species=1, spacing=(1., 1., 1.))
        data['rho'].fill_(20)
        data['beta'] = 1.
        model = QuadraticModel(torch.ones(1, 1, dtype=torch.float64))
        integrator = StochasticEulerIntegrator(model, ConstantMobility(1.), .02, seed=31)
        power = torch.zeros(4, 4, 4, dtype=torch.float64)
        for step in range(6500):
            data['rho'] = integrator.step(data)
            if step >= 1500:
                mode = torch.fft.fftn(data['rho'].reshape(4, 4, 4)-20)/64
                power += mode.abs().square()/5000
        q = 4*torch.sin(torch.pi*torch.arange(4, dtype=torch.float64)/4).square()
        q2 = q[:, None, None]+q[None, :, None]+q[None, None, :]
        # Explicit Euler has a known O(dt) equilibrium bias, not exact sampling.
        expected = 1/64/(1-.02*q2/2)
        ratio = power[q2>0]/expected[q2>0]
        self.assertLess(abs(float(ratio.mean())-1), .05)
        self.assertLess(float((ratio-1).abs().max()), .25)

    def test_density_mobility_law_and_validation(self):
        diffusion = torch.tensor([.1, .2], dtype=torch.float64)
        mobility = DensityMobility(diffusion)
        expected = torch.diag_embed(self.data['rho'] * diffusion * self.data['beta'])
        torch.testing.assert_close(mobility(self.data), expected)
        for bad in (-1., float('nan'), [], [[.1, 0.], [0., .2]]):
            with self.assertRaises(ValueError): DensityMobility(bad)
        with self.assertRaises(ValueError): DensityMobility([1., 2., 3.])(self.data)
        with self.assertRaises(ValueError): mobility(dict(self.data, beta=0.))
        with self.assertRaises(ValueError): mobility(dict(self.data, rho=-self.data['rho']))
        a = EulerIntegrator(self.model, mobility, .001).rhs(self.data)
        b = EulerIntegrator(self.model, LocalMobility(lambda rho, T: rho*diffusion*self.data['beta']), .001).rhs(self.data)
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_density_restart_mask_and_null_species(self):
        self.data['excluded_mask'] = torch.zeros(len(self.data['rho']), dtype=torch.bool)
        self.data['excluded_mask'][0] = True
        self.data['rho'][0] = 0
        self.data['rho'][1:, 0] += torch.arange(len(self.data['rho'])-1)*.1
        mobility = DensityMobility(torch.tensor([.1, 0.], dtype=torch.float64))
        full = Dynamics(self.make(mobility=mobility)).run(self.data, 8)
        first = Dynamics(self.make(mobility=mobility)).run(self.data, 3)
        state = dict(self.data, rho=first['rho'][-1])
        resumed = self.make(seed=99, mobility=mobility)
        resumed.set_rng_state(state, first['rng_state'][-1])
        second = Dynamics(resumed).run(state, 5)
        self.assertTrue(torch.equal(full['rho'][3:], second['rho']))
        self.assertTrue(torch.equal(full['rho'][:, 0], torch.zeros_like(full['rho'][:, 0])))
        self.assertTrue(torch.equal(full['rho'][:, :, 1], self.data['rho'][:, 1].expand(9, -1)))
        torch.testing.assert_close(full['particle_numbers'], full['particle_numbers'][0].expand_as(full['particle_numbers']),
                                   rtol=0, atol=1e-12)
        self.assertEqual(full['metadata']['mobility']['class'], 'DensityMobility')

    def test_density_conditional_covariance_and_temperature(self):
        data = field(shape=(5, 1, 1), species=2, spacing=(.4, .7, 1.1))
        data['rho'] = torch.arange(1, 11, dtype=torch.float64).reshape(5, 2)
        mobility = DensityMobility(torch.tensor([.1, .2], dtype=torch.float64))
        integrator = self.make(mobility=mobility)
        grid = integrator._grid_for(data)
        samples = torch.stack([integrator._noise_increment(data, grid)[0] for _ in range(6000)])
        cell = mobility(data).diagonal(dim1=-2, dim2=-1)
        adjacent_faces = (2*cell[0]+cell[1]+cell[-1])/2
        expected = 2*integrator.dt/(grid.beta*grid.volume*grid.spacing[0]**2)*adjacent_faces
        torch.testing.assert_close(samples.square().mean(0), expected, rtol=.08, atol=0)
        self.assertLess(abs(float((samples[:, 0]*samples[:, 1]).mean())), .05*float(expected.prod().sqrt()))
        # Fixed D: beta cancels from thermal noise, since M=beta D rho.
        other = dict(data, beta=data['beta']*2)
        a, b = self.make(mobility=mobility), self.make(mobility=mobility)
        torch.testing.assert_close(a._noise_increment(data, a._grid_for(data)),
                                   b._noise_increment(other, b._grid_for(other)), rtol=1e-13, atol=1e-14)

    def test_density_face_operator_has_zero_ito_drift(self):
        # K=G^T M_face G/(beta*dV), d rho=-K grad(beta A)dt+sqrt(2K)dW.
        # Check sum_j d K_ij / d rho_j by autograd, also with blocked faces.
        rho = torch.arange(1, 11, dtype=torch.float64).reshape(5, 2)
        eye = torch.eye(5, dtype=torch.float64)
        gradient = (torch.roll(eye, -1, 0)-eye)/.4
        diffusion = torch.tensor([.1, .3], dtype=torch.float64)
        for mask in (torch.ones(5), torch.tensor([0., 1., 1., 1., 0.])):
            def operator(x, power=1):
                cells = .4*diffusion*x.pow(power)
                face = (cells+torch.roll(cells, -1, 0))/2*mask[:, None]
                return (torch.einsum('fg,fa,fh,ab->gahb', gradient, face, gradient,
                                    torch.eye(2, dtype=torch.float64))/(.4*.4*.7*1.1)).reshape(10, 10)
            jac = torch.autograd.functional.jacobian(operator, rho).reshape(10, 10, 10)
            drift = jac.diagonal(dim1=1, dim2=2).sum(-1)
            torch.testing.assert_close(drift, torch.zeros_like(drift), rtol=0, atol=1e-13)
            nonlinear = torch.autograd.functional.jacobian(lambda x: operator(x, 2), rho).reshape(10, 10, 10)
            self.assertGreater(float(nonlinear.diagonal(dim1=1, dim2=2).sum(-1).abs().max()), 1.)

    def test_density_stationary_quadratic_fluctuations(self):
        data = field(shape=(4, 4, 4), species=1, spacing=(1., 1., 1.))
        data['rho'].fill_(20)
        data['beta'] = 1.
        model = QuadraticModel(torch.ones(1, 1, dtype=torch.float64))
        # Mean mobility one; density dependence is evaluated anew every step.
        integrator = StochasticEulerIntegrator(model, DensityMobility(.05), .005, seed=73)
        variance = 0.
        for step in range(12000):
            data['rho'] = integrator.step(data)
            if step >= 2000:
                variance += float((data['rho']-20).square().mean())/10000
        self.assertLess(abs(variance/(63/64)-1), .06)

    def test_floor_inactive_is_bitwise_unchanged(self):
        a = self.make()
        b = StochasticEulerIntegrator(self.model, self.mobility, .001, seed=1, rho_min=1e-4)
        self.assertTrue(torch.equal(a.step(self.data), b.step(self.data)))
        self.assertTrue(torch.equal(a.get_rng_state(self.data), b.get_rng_state(self.data)))
        self.assertEqual(int(b.last_floor_correction['floor_clamped_voxels'].sum()), 0)

    def test_floor_preserves_species_mass_mask_and_unaffected_species(self):
        data = dict(self.data, rho=self.data['rho'].clone().requires_grad_())
        mask = torch.zeros(len(data['rho']), dtype=torch.bool)
        mask[0] = True
        data['excluded_mask'] = mask
        with torch.no_grad():
            data['rho'][0] = 0
            data['rho'][1:, 0] = .2
        integrator = StochasticEulerIntegrator(self.model, ConstantMobility(0.), .01,
                                             seed=2, rho_min=1e-4)
        noise = torch.zeros_like(data['rho'])
        noise[1, 0], noise[2, 0] = -.3, .3
        with mock.patch.object(integrator, '_noise_increment', return_value=noise):
            result = integrator.step(data)
        self.assertGreaterEqual(float(result[~mask].min()), 1e-4)
        self.assertTrue(torch.equal(result[mask], data['rho'][mask]))
        self.assertTrue(torch.equal(result[:, 1], data['rho'][:, 1]))
        torch.testing.assert_close(result.sum(0), data['rho'].sum(0), rtol=0, atol=1e-12)
        self.assertFalse(result.requires_grad)
        self.assertEqual(integrator.last_floor_correction['floor_clamped_voxels'].tolist(), [1, 0])
        expected_added = .1001 * data['grid_spacing'].prod()
        torch.testing.assert_close(integrator.last_floor_correction['floor_added_particles'][0], expected_added)
        torch.testing.assert_close(integrator.last_floor_correction['floor_l1_particles'][0], 2*expected_added)

    def test_floor_at_minimum_total_mass(self):
        self.data['rho'].fill_(1e-4)
        integrator = StochasticEulerIntegrator(self.model, self.mobility, .01, seed=2, rho_min=1e-4)
        result = integrator.step(self.data)
        self.assertTrue(torch.equal(result, self.data['rho']))
        self.assertGreater(int(integrator.last_floor_correction['floor_clamped_voxels'].sum()), 0)

    def test_floor_validation_and_nonfinite_failure_rng(self):
        for bad in (0., -1., float('nan'), float('inf'), True, [1e-4, 1e-4]):
            with self.assertRaises(ValueError):
                StochasticEulerIntegrator(self.model, self.mobility, .01, seed=1, rho_min=bad)
        integrator = StochasticEulerIntegrator(self.model, self.mobility, .01, seed=1, rho_min=1e-4)
        self.data['rho'].fill_(1e-5)
        with self.assertRaisesRegex(ValueError, 'at least rho_min'):
            integrator.step(self.data)
        self.data['rho'].fill_(1.)
        rng = integrator.get_rng_state(self.data)
        draw = integrator._noise_increment
        def nonfinite(data, grid):
            noise = draw(data, grid)
            noise[0, 0] = float('nan')
            return noise
        with mock.patch.object(integrator, '_noise_increment', side_effect=nonfinite):
            with self.assertRaisesRegex(ValueError, 'cannot repair') as caught:
                Dynamics(integrator).run(self.data, 2)
        traj = caught.exception.trajectory
        self.assertEqual(traj['metadata']['steps_completed'], 0)
        self.assertTrue(torch.equal(rng, traj['rng_state'][-1]))
        self.assertTrue(torch.equal(self.data['rho'], traj['rho'][-1]))
        self.assertEqual(int(traj['floor_clamped_voxels'][-1].sum()), 0)

    def test_floor_recording_totals_and_restart(self):
        self.data['rho'].fill_(.1)
        def make(seed=11):
            return StochasticEulerIntegrator(self.model, self.mobility, .01, seed=seed, rho_min=1e-4)
        integrator = make()
        full = Dynamics(integrator).run(self.data, 7, save_every=3)
        every = Dynamics(make()).run(self.data, 7)
        self.assertEqual(full['metadata']['rho_min'], 1e-4)
        self.assertTrue(torch.all(full['rho'] >= 1e-4))
        for key in ('floor_clamped_voxels', 'floor_added_particles',
                    'floor_l1_particles', 'floor_max_density_change'):
            self.assertTrue(torch.equal(full[key], every[key][[0, 3, 6, 7]]))
        self.assertGreater(int(full['floor_clamped_voxels'][-1].sum()), 0)
        first = Dynamics(make()).run(self.data, 3)
        state = dict(self.data, rho=first['rho'][-1])
        resumed = make(seed=99)
        resumed.set_rng_state(state, first['rng_state'][-1])
        second = Dynamics(resumed).run(state, 4)
        self.assertTrue(torch.equal(every['rho'][3:], second['rho']))
        for key in ('floor_clamped_voxels', 'floor_added_particles', 'floor_l1_particles'):
            torch.testing.assert_close(first[key][-1] + second[key][-1], every[key][-1], rtol=0, atol=1e-14)
        torch.testing.assert_close(full['particle_numbers'], full['particle_numbers'][0].expand_as(full['particle_numbers']),
                                   rtol=0, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
