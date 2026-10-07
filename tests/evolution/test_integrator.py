import math
import unittest

import numpy as np
import torch
from torch import nn

from equicdft import GridCACEModel, GridSolver
from equicdft.evolution import ConstantMobility, Dynamics, LocalMobility, EulerIntegrator, SemiImplicitIntegrator
from equicdft.solver.options import Constraints, ConvergenceOptions
from tests.model_fixtures import example_model, grid_data, long_range_model


class QuadraticModel(nn.Module):
    """beta F = dV/2 sum rho^T H_beta rho; optionally shift the AB coupling."""

    def __init__(self, curvature, shifted=False):
        super().__init__()
        self.register_buffer("curvature", torch.as_tensor(curvature, dtype=torch.float64))
        self.shifted = shifted

    def forward(self, data, compute_c1=False, **kwargs):
        rho = data["rho"]
        derivative = rho @ self.curvature.T
        energy = (rho * derivative).sum() / 2
        if self.shifted:
            shape = tuple(data["grid_size"].tolist())
            dense = rho.reshape(shape + (2,))
            neighbor_b = torch.roll(dense[..., 1], -1, 0)
            energy += (dense[..., 0] * neighbor_b).sum() * 0.4
            derivative = derivative + torch.stack((neighbor_b, torch.roll(dense[..., 0], 1, 0)), -1).reshape_as(rho) * 0.4
        volume = data["grid_spacing"].prod()
        result = {"beta_F": volume * energy}
        if compute_c1:
            result["density_derivative"] = derivative
        return result


def field(shape=(8, 3, 2), species=2, spacing=(0.4, 0.7, 1.1)):
    positions = torch.cartesian_prod(*(torch.arange(size) for size in shape))
    mode = torch.cos(2 * math.pi * positions[:, 0].double() / shape[0])
    rho = torch.full((len(positions), species), 2.0, dtype=torch.float64)
    rho += 0.04 * mode[:, None] * torch.arange(1, species + 1)
    return {
        "rho": rho, "grid_size": torch.tensor(shape), "grid_positions": positions,
        "grid_spacing": torch.tensor(spacing, dtype=torch.float64),
        "temperature": torch.tensor(2.5, dtype=torch.float64),
        "beta": torch.tensor(0.4, dtype=torch.float64),
    }


class TestIntegrator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.data = field()
        self.h = torch.tensor([[2.0, 0.5], [0.5, 1.0]], dtype=torch.float64)
        self.m = torch.tensor([[1.0, 0.3], [0.3, 2.0]], dtype=torch.float64)
        self.model = QuadraticModel(self.h)
        self.mobility = ConstantMobility(self.m)

    def q2(self):
        return 4 * math.sin(math.pi / 8) ** 2 / 0.4 ** 2

    def test_fourier_relaxation_and_matrix_order(self):
        data = self.data
        fluctuation = data["rho"] - data["rho"].mean(0)
        rate_matrix = self.q2() * self.m @ self.h / data["beta"]
        dt = 0.01
        for integrator in (EulerIntegrator(self.model, self.mobility, dt),
                           SemiImplicitIntegrator(self.model, self.mobility, dt)):
            with self.subTest(method=type(integrator).__name__):
                if isinstance(integrator, SemiImplicitIntegrator):
                    expected = torch.linalg.solve(torch.eye(2, dtype=torch.float64) + dt * rate_matrix, fluctuation.T).T
                else:
                    expected = fluctuation - dt * fluctuation @ rate_matrix.T
                updated = integrator.step(data)
                torch.testing.assert_close(updated, data["rho"].mean(0) + expected, rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(updated.sum(0), data["rho"].sum(0), rtol=1e-14, atol=1e-12)
        self.assertFalse(data["rho"].requires_grad)

    def test_constant_species_potentials_are_gauge(self):
        changed = dict(self.data, V_ext=torch.tensor([8., -12.], dtype=torch.float64).expand_as(self.data["rho"]))
        for cls in (EulerIntegrator, SemiImplicitIntegrator):
            torch.testing.assert_close(cls(self.model, self.mobility, .001).step(self.data),
                                       cls(self.model, self.mobility, .001).step(changed), rtol=1e-13, atol=1e-13)
        differentiable_field = dict(self.data, V_ext=torch.zeros_like(self.data["rho"], requires_grad=True))
        self.assertFalse(EulerIntegrator(self.model, self.mobility, .001).rhs(differentiable_field).requires_grad)

    def test_uniform_field_and_rank_deficient_mobility(self):
        uniform = dict(self.data, rho=torch.full_like(self.data["rho"], 1.7))
        for cls in (EulerIntegrator, SemiImplicitIntegrator):
            torch.testing.assert_close(cls(self.model, self.mobility, .2).step(uniform), uniform["rho"])
            integrator = cls(self.model, ConstantMobility([0., 1.]), .001)
            updated = integrator.step(self.data)
            torch.testing.assert_close(updated[:, 0], self.data["rho"][:, 0], rtol=1e-14, atol=1e-14)
            exchange_only = cls(self.model, ConstantMobility([[1., -1.], [-1., 1.]]), .001)
            torch.testing.assert_close(exchange_only.step(self.data).sum(-1), self.data["rho"].sum(-1), rtol=1e-14, atol=1e-14)

    def test_mobility_module_is_evaluated_deterministically(self):
        class WithDropout(nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout = nn.Dropout(.5)

            def forward(self, rho, temperature):
                return self.dropout(rho) / temperature

        function = WithDropout()
        mobility = LocalMobility(function)
        for cls in (EulerIntegrator, SemiImplicitIntegrator):
            integrator = cls(self.model, mobility, .001)
            first = integrator.step(self.data)
            torch.testing.assert_close(first, integrator.step(self.data), rtol=0, atol=0)
            self.assertTrue(function.training)
            self.assertTrue(function.dropout.training)

    def test_permuted_rows(self):
        generator = torch.Generator().manual_seed(13)
        order = torch.randperm(len(self.data["rho"]), generator=generator)
        permuted = dict(self.data, rho=self.data["rho"][order], grid_positions=self.data["grid_positions"][order])
        for cls in (EulerIntegrator, SemiImplicitIntegrator):
            expected = cls(self.model, self.mobility, .001).step(self.data)[order]
            torch.testing.assert_close(cls(self.model, self.mobility, .001).step(permuted), expected)

    def test_permuted_rows_with_excluded_mask(self):
        data = field(shape=(8, 2, 2))
        x = data["grid_positions"][:, 0]
        excluded = (x == 0) | (x == 4)
        data["excluded_mask"] = excluded
        data["rho"][excluded] = 0
        order = torch.randperm(len(data["rho"]), generator=torch.Generator().manual_seed(3))
        permuted = dict(data, rho=data["rho"][order], grid_positions=data["grid_positions"][order],
                        excluded_mask=excluded[order])
        expected = EulerIntegrator(self.model, self.mobility, .001).step(data)[order]
        torch.testing.assert_close(EulerIntegrator(self.model, self.mobility, .001).step(permuted), expected, rtol=0, atol=0)

    def test_relaxation_rates_and_stability_limit(self):
        integrator = EulerIntegrator(self.model, self.mobility, .001)
        rates = integrator.relaxation_rates(self.data)
        self.assertEqual(tuple(rates.shape), (8, 3, 2, 2))
        torch.testing.assert_close(rates[0, 0, 0], torch.zeros(2, dtype=torch.float64), rtol=0, atol=1e-12)
        coupling = torch.linalg.eigvals(self.m @ self.h).real.sort().values / self.data["beta"]
        q2 = torch.zeros(8, 3, 2, dtype=torch.float64)
        for axis, (size, spacing) in enumerate(zip((8, 3, 2), (0.4, 0.7, 1.1))):
            modes = 4 * torch.sin(math.pi * torch.arange(size, dtype=torch.float64) / size) ** 2 / spacing ** 2
            q2 += modes.reshape([size if i == axis else 1 for i in range(3)])
        torch.testing.assert_close(rates.sort(-1).values, q2[..., None] * coupling, rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(integrator.stability_limit(self.data), float(2 / (q2.max() * coupling.max())), places=12)
        with self.assertRaisesRegex(ValueError, "fully accessible"):
            integrator.relaxation_rates(dict(self.data, excluded_mask=torch.zeros(48, dtype=torch.bool).index_fill(0, torch.tensor([0]), True),
                                             rho=self.data["rho"].index_fill(0, torch.tensor([0]), 0.)))

    def test_reference_density_change_is_rejected(self):
        integrator = SemiImplicitIntegrator(self.model, self.mobility, .001)
        updated = integrator.step(self.data)
        integrator.step(dict(self.data, rho=updated))
        with self.assertRaisesRegex(ValueError, "reference density"):
            integrator.step(dict(self.data, rho=3 * self.data["rho"]))

    def test_changed_timestep_rebuilds_only_propagator(self):
        integrator = SemiImplicitIntegrator(self.model, self.mobility, .001)
        state = dict(self.data, rho=integrator.step(self.data))
        operator = integrator._operator
        propagator = integrator._propagator_for(.001)
        integrator.step(state)
        self.assertIs(integrator._propagator_for(.001), propagator)
        for dt in (.02, .0005):
            integrator.dt = dt
            fresh = SemiImplicitIntegrator(self.model, self.mobility, dt)
            torch.testing.assert_close(integrator.step(state), fresh.step(state), rtol=1e-13, atol=1e-13)
            self.assertIs(integrator._operator, operator)
            self.assertIsNot(integrator._propagator_for(dt), propagator)
            propagator = integrator._propagator_for(dt)
        # The two most recent step sizes stay cached, so a step and its half alternate for free.
        self.assertEqual(set(integrator._propagators), {.02, .0005})
        half = integrator._propagator_for(.01)
        self.assertEqual(set(integrator._propagators), {.0005, .01})
        self.assertIs(integrator._propagator_for(.01), half)
        torch.testing.assert_close(half, 1 / (1 + .01 * operator) if operator.shape[-1] == 1 else
                                   torch.linalg.inv(torch.eye(2, dtype=operator.dtype) + .01 * operator), rtol=0, atol=0)

    def test_timestep_assignment_rejects_invalid_values(self):
        for cls in (EulerIntegrator, SemiImplicitIntegrator):
            integrator = cls(self.model, self.mobility, .001)
            for dt in (0., -1., float('nan'), float('inf'), True, [1., 2.], 1j):
                with self.subTest(method=cls.__name__, dt=dt), self.assertRaises(ValueError):
                    integrator.dt = dt
                self.assertEqual(integrator.dt, .001)

    def test_long_time_limit_is_boltzmann_for_ideal_gas(self):
        data = {key: value.double() if value.is_floating_point() else value for key, value in grid_data(shape=(4, 4, 4)).items()}
        data["rho"] = torch.full_like(data["rho"], .5)
        data["V_ext"] = (.6 * torch.cos(2 * math.pi * data["grid_positions"][:, 0].double() / 4))[:, None]
        model = GridCACEModel(a_features=None, b_features=None, readout=[], n_types=1, grid_spacing=1., boltzmann_constant=1.).double()
        boltzmann = torch.exp(-data["beta"] * data["V_ext"])
        boltzmann *= data["rho"].sum() / boltzmann.sum()
        for cls, dt in ((EulerIntegrator, .1), (SemiImplicitIntegrator, .25)):
            trajectory = Dynamics(cls(model, ConstantMobility(1.), dt)).run(data, n_steps=300, save_every=300)
            torch.testing.assert_close(trajectory["rho"][-1], boltzmann, rtol=0, atol=1e-8)

    def test_long_time_limit_matches_equilibrium_solver(self):
        torch.manual_seed(0)
        model = example_model().double().eval()
        data = {key: value.double() if value.is_floating_point() else value for key, value in grid_data(shape=(4, 4, 4)).items()}
        data["rho"] = torch.full_like(data["rho"], .5)
        data["V_ext"] = (.3 * torch.cos(2 * math.pi * data["grid_positions"][:, 0].double() / 4))[:, None]
        solution = GridSolver(model).solve(dict(data), constraints=Constraints(particle_numbers=[float(data["rho"].sum())]),
                                           convergence=ConvergenceOptions(max_iter=3000, tolerance_residual=1e-10))
        integrator = EulerIntegrator(model, ConstantMobility(1.), .01)
        rates = integrator.relaxation_rates(data)
        self.assertGreater(float(rates.flatten()[1:].min()), 0)
        for cls, dt in ((EulerIntegrator, .8 * integrator.stability_limit(data)), (SemiImplicitIntegrator, .25)):
            trajectory = Dynamics(cls(model, ConstantMobility(1.), dt)).run(data, n_steps=300, save_every=30)
            torch.testing.assert_close(trajectory["rho"][-1], solution["rho"], rtol=0, atol=1e-5)
            self.assertTrue(torch.all(torch.diff(trajectory["beta_A"]) <= 1e-10))

    def test_geometry_cache_tracks_replaced_and_mutated_inputs(self):
        integrator = EulerIntegrator(self.model, self.mobility, .001)
        first = integrator.step(self.data)
        potential = torch.zeros_like(self.data["rho"])
        data = dict(self.data, V_ext=potential)
        torch.testing.assert_close(integrator.step(data), first, rtol=0, atol=0)
        potential[:, 0] = torch.linspace(0., 1., len(potential))      # in-place change must be seen
        changed = integrator.step(data)
        self.assertFalse(torch.allclose(changed, first))
        torch.testing.assert_close(changed, EulerIntegrator(self.model, self.mobility, .001).step(data), rtol=0, atol=0)
        walled = dict(self.data, excluded_mask=torch.zeros(48, dtype=torch.bool).index_fill(0, torch.tensor([5]), True),
                      rho=self.data["rho"].index_fill(0, torch.tensor([5]), 0.))
        torch.testing.assert_close(integrator.step(walled), EulerIntegrator(self.model, self.mobility, .001).step(walled), rtol=0, atol=0)
        torch.testing.assert_close(integrator.step(self.data), first, rtol=0, atol=0)

    def test_geometry_cache_rebuilds_for_mutable_non_tensor_inputs(self):
        for convert in (lambda value: value.tolist(), lambda value: value.numpy().copy()):
            data = dict(self.data, V_ext=convert(torch.zeros_like(self.data['rho'])))
            integrator = EulerIntegrator(self.model, self.mobility, .001)
            initial = integrator.rhs(data)
            data['V_ext'][0][0] = 1.
            changed = integrator.rhs(data)
            self.assertFalse(torch.allclose(changed, initial))
            fresh = EulerIntegrator(self.model, self.mobility, .001)
            torch.testing.assert_close(changed, fresh.rhs(data), rtol=0, atol=0)
            # Replacing arrays must not trigger ambiguous NumPy comparisons.
            data['V_ext'] = convert(torch.ones_like(self.data['rho']))
            torch.testing.assert_close(integrator.rhs(data), fresh.rhs(data), rtol=0, atol=0)
        data = dict(self.data, grid_spacing=np.array([.4, .7, 1.1]))
        integrator = EulerIntegrator(self.model, self.mobility, .001)
        integrator.rhs(data)
        data['grid_spacing'][0] = .8
        torch.testing.assert_close(integrator.rhs(data),
                                   EulerIntegrator(self.model, self.mobility, .001).rhs(data), rtol=0, atol=0)

    def test_changed_geometry_rechecks_previously_returned_density(self):
        integrator = EulerIntegrator(self.model, self.mobility, .001)
        state = dict(self.data, rho=integrator.step(self.data))
        state['excluded_mask'] = torch.zeros(48, dtype=torch.bool)
        state['excluded_mask'][0] = True
        with self.assertRaisesRegex(ValueError, 'Excluded voxels'):
            integrator.rhs(state)

    def test_own_output_is_trusted_until_modified(self):
        integrator = EulerIntegrator(self.model, self.mobility, .001)
        state = dict(self.data, rho=integrator.step(self.data))
        integrator.step(state)
        state["rho"][3, 1] = -1.0
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            integrator.step(state)

    def test_last_beta_A_matches_direct_evaluation(self):
        for cls in (EulerIntegrator, SemiImplicitIntegrator):
            integrator = cls(self.model, self.mobility, .001)
            data = dict(self.data, V_ext=torch.linspace(-1., 1., 96, dtype=torch.float64).reshape(48, 2))
            expected = integrator.beta_A(data)
            integrator.step(data)
            torch.testing.assert_close(integrator.last_beta_A, expected, rtol=1e-13, atol=1e-10)

    def test_excluded_faces_and_compartment_conservation(self):
        data = field(shape=(8, 2, 2))
        x = data["grid_positions"][:, 0]
        excluded = (x == 0) | (x == 4)
        data["excluded_mask"] = excluded
        data["rho"][excluded] = 0
        updated = EulerIntegrator(self.model, self.mobility, .001).step(data)
        self.assertTrue(torch.equal(updated[excluded], data["rho"][excluded]))
        for accessible in ((x > 0) & (x < 4), (x > 4)):
            torch.testing.assert_close(updated[accessible].sum(0), data["rho"][accessible].sum(0), rtol=1e-14, atol=1e-13)
        with self.assertRaisesRegex(ValueError, "fully accessible"):
            SemiImplicitIntegrator(self.model, self.mobility, .001).step(data)

    def test_local_matrix_flux_dissipates_energy(self):
        def local(rho, temperature):
            return self.m.expand(rho.shape + (2,)) * (rho.sum(-1) / temperature)[..., None, None]
        integrator = EulerIntegrator(self.model, LocalMobility(local), .001)
        rate = integrator.rhs(self.data)
        mu = self.data["rho"] @ self.h.T
        power = (mu * rate).sum() * self.data["grid_spacing"].prod()
        self.assertLess(float(power), 0)
        torch.testing.assert_close(rate.sum(0), torch.zeros(2, dtype=torch.float64), atol=1e-12, rtol=0)
        self.assertLess(integrator.beta_A(dict(self.data, rho=integrator.step(self.data))), integrator.beta_A(self.data))

    def test_negative_curvature_is_not_removed(self):
        data = field(species=1)
        model = QuadraticModel([[-1.]])
        dt = .001
        updated = SemiImplicitIntegrator(model, ConstantMobility(1.), dt).step(data)
        fluctuation = data["rho"] - data["rho"].mean()
        expected = data["rho"].mean() + fluctuation * (1 + dt * self.q2() / data["beta"])
        torch.testing.assert_close(updated, expected, rtol=1e-13, atol=1e-13)

    def test_rejects_nonpositive_trial_without_clipping(self):
        original = self.data["rho"].clone()
        with self.assertRaisesRegex(ValueError, "reduce the timestep"):
            EulerIntegrator(self.model, self.mobility, 100.).step(self.data)
        self.assertTrue(torch.equal(self.data["rho"], original))
        self.assertFalse(self.data["rho"].requires_grad)

    def test_implicit_stiff_mode_allows_timestep_euler_rejects(self):
        with self.assertRaisesRegex(ValueError, "reduce the timestep"):
            EulerIntegrator(self.model, self.mobility, 10.).step(self.data)
        integrator = SemiImplicitIntegrator(self.model, self.mobility, 10.)
        updated = integrator.step(self.data)
        self.assertTrue(torch.all(updated > 0))
        self.assertLess(integrator.beta_A(dict(self.data, rho=updated)), integrator.beta_A(self.data))
        self.assertLess(float((updated - updated.mean(0)).norm()),
                        float((self.data["rho"] - self.data["rho"].mean(0)).norm()))

    def test_both_methods_first_order_timestep_convergence(self):
        data = field(species=1)
        model = QuadraticModel([[1.]])
        rate = self.q2() / data["beta"]
        duration = .08
        exact = data["rho"].mean() + (data["rho"] - data["rho"].mean()) * math.exp(-float(rate) * duration)
        for cls in (EulerIntegrator, SemiImplicitIntegrator):
            errors = []
            for steps in (40, 80):
                state = dict(data)
                integrator = cls(model, ConstantMobility(1.), duration / steps)
                for _ in range(steps):
                    state["rho"] = integrator.step(state)
                errors.append(float((state["rho"] - exact).norm()))
            self.assertGreater(errors[0] / errors[1], 1.9)
            self.assertLess(errors[0] / errors[1], 2.1)

    def test_supplied_curvature_and_geometry_guard(self):
        automatic = SemiImplicitIntegrator(self.model, self.mobility, .001).step(self.data)
        explicit = SemiImplicitIntegrator(self.model, self.mobility, .001, reference_curvature=self.h)
        torch.testing.assert_close(explicit.step(self.data), automatic, rtol=1e-13, atol=1e-13)
        with self.assertRaisesRegex(ValueError, "temperature changed"):
            explicit.step(dict(self.data, beta=torch.tensor(.5)))
        with self.assertRaisesRegex(ValueError, "Hermitian"):
            SemiImplicitIntegrator(self.model, self.mobility, .001, reference_curvature=[[1., 1.], [0., 1.]]).step(self.data)
        bad = self.h.to(torch.complex128).expand((8, 3, 2, 2, 2)).clone()
        bad[1, ..., 0, 1] += .2j
        bad[1, ..., 1, 0] -= .2j
        with self.assertRaisesRegex(ValueError, "real fields"):
            SemiImplicitIntegrator(self.model, self.mobility, .001, reference_curvature=bad).step(self.data)

    def test_shifted_cross_species_hessian_orientation(self):
        model = QuadraticModel([[2., 0.], [0., 1.]], shifted=True)
        integrator = SemiImplicitIntegrator(model, self.mobility, .001)
        updated = integrator.step(self.data)
        h = torch.diag(torch.tensor([2., 1.], dtype=torch.complex128)).expand((8, 3, 2, 2, 2)).clone()
        phase = torch.exp(2j * math.pi * torch.arange(8, dtype=torch.float64) / 8)
        h[..., 0, 1] = .4 * phase[:, None, None]
        h[..., 1, 0] = .4 * phase.conj()[:, None, None]
        explicit = SemiImplicitIntegrator(model, self.mobility, .001, reference_curvature=h)
        torch.testing.assert_close(explicit.step(self.data), updated, rtol=1e-13, atol=1e-13)

    def test_real_ideal_gas_model_and_einstein_mobility(self):
        data = field(species=1)
        data["rho"] = 1 + (data["rho"] - data["rho"].mean()) * 1e-4
        model = GridCACEModel(a_features=None, b_features=None, readout=[], n_types=1, grid_spacing=data["grid_spacing"], boltzmann_constant=1.).double()
        diffusion = .3
        mobility = LocalMobility(lambda rho, temperature: diffusion * rho / temperature)
        integrator = EulerIntegrator(model, mobility, .001)
        expected = -diffusion * self.q2() * (data["rho"] - data["rho"].mean())
        torch.testing.assert_close(integrator.rhs(data), expected, rtol=1e-5, atol=1e-13)
        implicit = SemiImplicitIntegrator(model, mobility, .001)
        self.assertTrue(torch.isfinite(implicit.step(data)).all())

    def test_actual_cace_and_long_range_models(self):
        for model, data in ((example_model().double(), grid_data(shape=(4, 4, 4))),
                            (long_range_model(charges=[1., -1.], coulomb_amplitude=1.).double(),
                             grid_data(shape=(4, 4, 4), n_types=2, grid_spacing=.5))):
            data = {key: value.double() if value.is_floating_point() else value for key, value in data.items()}
            model.eval()
            integrator = SemiImplicitIntegrator(model, ConstantMobility(.001), .0001)
            updated = integrator.step(data)
            torch.testing.assert_close(updated.sum(0), data["rho"].sum(0), rtol=1e-13, atol=1e-12)
            self.assertFalse(data["rho"].requires_grad)


if __name__ == "__main__":
    unittest.main()
