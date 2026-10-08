"""Frozen voxels share one no-flux constraint across evolution methods."""
import math
import unittest

import torch

from equicdft.evolution import (
    AdaptiveDynamics, ConstantMobility, ConvergenceError, DensityMobility, Dynamics,
    EulerIntegrator, RampedField, SemiImplicitIntegrator, StepControl,
    StochasticEulerIntegrator,
)
from equicdft.evolution import integrator as integrator_module
from tests.evolution.test_dynamics import QuadraticModel, field_data
from tests.evolution.test_integrator import QuadraticModel as LinearModel, field as linear_field


class NonlocalQuadraticModel(QuadraticModel):
    def forward(self, data, **kwargs):
        out = super().forward(data, **kwargs)
        rho = data["rho"].reshape(tuple(data["grid_size"].tolist()) + (-1,))
        derivative = torch.zeros_like(rho)
        for axis, h in enumerate(data["grid_spacing"]):
            difference = (torch.roll(rho, -1, axis)-rho)/h
            out["beta_F"] = out["beta_F"] + .15*data["grid_spacing"].prod()*difference.square().sum()
            derivative += .3*(2*rho-torch.roll(rho, -1, axis)-torch.roll(rho, 1, axis))/h.square()
        out["density_derivative"] = out["density_derivative"] + derivative.reshape_as(data["rho"])
        return out


class TestEvolutionMask(unittest.TestCase):
    def setUp(self):
        self.data = field_data()
        self.mask = self.data["grid_positions"][:, 0] % 2 == 0

    def integrator(self, cls, mask=None, dt=.01, mobility=None):
        kwargs = dict(mask=mask)
        if cls is StochasticEulerIntegrator:
            kwargs.update(seed=17)
        return cls(QuadraticModel(), mobility or ConstantMobility([[.4, -.1], [-.1, .25]]), dt, **kwargs)

    def assert_constraints(self, rho, initial=None, mask=None):
        initial = self.data["rho"] if initial is None else initial
        mask = self.mask if mask is None else mask
        self.assertTrue(torch.equal(rho[mask], initial[mask]))
        torch.testing.assert_close(rho.sum(0), initial.sum(0), rtol=0., atol=3e-12)

    def test_all_integrators_freeze_and_conserve_each_disconnected_region(self):
        for cls in (EulerIntegrator, SemiImplicitIntegrator, StochasticEulerIntegrator):
            with self.subTest(cls=cls):
                integ = self.integrator(cls, self.mask, dt=1e-4)
                state = dict(self.data)
                for _ in range(5):
                    state["rho"] = integ.step(state)
                    self.assert_constraints(state["rho"])
                    for x in (1, 3):
                        cells = self.data["grid_positions"][:, 0] == x
                        torch.testing.assert_close(state["rho"][cells].sum(0), self.data["rho"][cells].sum(0),
                                                   rtol=0., atol=2e-12)
                self.assertGreater(float((state["rho"][~self.mask]-self.data["rho"][~self.mask]).detach().abs().max()), 0.)

    def test_false_mask_is_bitwise_same_as_existing_path_including_rng(self):
        mask = torch.zeros_like(self.mask)
        for cls in (EulerIntegrator, SemiImplicitIntegrator, StochasticEulerIntegrator):
            with self.subTest(cls=cls):
                plain, masked = self.integrator(cls, dt=1e-4), self.integrator(cls, mask, dt=1e-4)
                a, b = plain.step(self.data), masked.step(self.data)
                self.assertTrue(torch.equal(a, b))
                if cls is StochasticEulerIntegrator:
                    self.assertTrue(torch.equal(plain.get_rng_state(self.data), masked.get_rng_state(self.data)))

    def test_all_frozen_is_identity_including_stochastic_floor(self):
        mask = torch.ones_like(self.mask)
        for cls in (EulerIntegrator, SemiImplicitIntegrator, StochasticEulerIntegrator):
            integ = self.integrator(cls, mask)
            if cls is StochasticEulerIntegrator:
                integ.rho_min = 2.  # Frozen values need not satisfy an evolving-cell floor.
            self.assertTrue(torch.equal(integ.step(self.data), self.data["rho"]))
            if cls is StochasticEulerIntegrator:
                self.assertTrue(all(torch.all(v == 0) for v in integ.last_floor_correction.values()))

    def test_driver_masks_fields_and_metadata(self):
        final = self.data["V_ext"]
        field = RampedField(torch.zeros_like(final), final, duration=.025)
        for cls in (EulerIntegrator, SemiImplicitIntegrator, StochasticEulerIntegrator):
            for source in ("integrator", "data", "driver"):
                with self.subTest(cls=cls, source=source):
                    integ = self.integrator(cls, self.mask if source == "integrator" else None, dt=1e-4)
                    state = dict(self.data, mask=self.mask) if source == "data" else self.data
                    result = Dynamics(integ, field, mask=self.mask if source == "driver" else None).run(state, 3)
                    self.assertTrue(torch.equal(result["metadata"]["mask"], self.mask))
                    for rho in result["rho"]:
                        self.assert_constraints(rho)
        for cls in (EulerIntegrator, SemiImplicitIntegrator):
            result = AdaptiveDynamics(self.integrator(cls), StepControl(1e-6, .02), field,
                                      mask=self.mask).run(self.data, .04)
            for rho in result["rho"]:
                self.assert_constraints(rho)
            self.assertLessEqual(float(torch.diff(result["beta_A"]-result["beta_work"]).max()), 1e-10)

    def test_driver_overrides_data_overrides_integrator_without_mutating_input(self):
        initial = self.data["rho"].clone()
        false = torch.zeros_like(self.mask)
        integ = self.integrator(EulerIntegrator, torch.ones_like(self.mask))
        state = dict(self.data, mask=false)
        result = Dynamics(integ, mask=self.mask).run(state, 2)
        self.assert_constraints(result["rho"][-1])
        self.assertTrue(torch.equal(state["mask"], false))
        self.assertTrue(torch.equal(state["rho"], initial))

    def test_mask_mutation_invalidates_geometry_cache(self):
        mask = torch.zeros_like(self.mask)
        integ = self.integrator(EulerIntegrator, mask)
        old = integ.rhs(self.data)
        mask.copy_(self.mask)
        new = integ.rhs(self.data)
        self.assertTrue(torch.all(new[self.mask] == 0))
        self.assertGreater(float(old[self.mask].abs().max()), 0.)
        self.assert_constraints(integ.step(self.data))

    def test_permuted_rows_and_excluded_voxels(self):
        state = dict(self.data, mask=self.mask, excluded_mask=torch.zeros_like(self.mask))
        state["rho"] = state["rho"].clone()
        state["excluded_mask"][0] = True
        state["rho"][0] = 0.
        permutation = torch.arange(len(self.mask)-1, -1, -1)
        permuted = dict(state)
        for name in ("rho", "mask", "excluded_mask", "grid_positions", "grid_center", "V_ext"):
            permuted[name] = state[name][permutation]
        plain = self.integrator(EulerIntegrator).step(state)
        actual = self.integrator(EulerIntegrator).step(permuted)
        self.assertTrue(torch.equal(actual, plain[permutation]))
        self.assert_constraints(plain, state["rho"])
        self.assertTrue(torch.all(plain[0] == 0))

    def test_masked_implicit_matches_dense_linear_reference(self):
        integ = self.integrator(SemiImplicitIntegrator, self.mask, dt=.3)
        state = dict(self.data, V_ext=torch.zeros_like(self.data["rho"]))
        rhs = integ.rhs(state)
        columns = []
        for j in range(state["rho"].numel()):
            perturbed = state["rho"].clone()
            perturbed.flatten()[j] += .1
            columns.append(((integ.rhs(dict(state, rho=perturbed))-rhs)/.1).flatten())
        operator = torch.stack(columns, dim=1)
        # Frozen rows are identity; their increments are zero, including any
        # coupling of their fixed densities into free-cell chemical potentials.
        identity = torch.eye(len(operator), dtype=operator.dtype)
        delta = torch.linalg.solve(identity-.3*operator, .3*rhs.flatten()).reshape_as(rhs)
        expected = state["rho"]+delta
        actual = integ.step(state)
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
        self.assert_constraints(actual)

    def test_floor_preserves_frozen_values_and_disconnected_region_masses(self):
        integ = StochasticEulerIntegrator(QuadraticModel(), DensityMobility(.1), .01,
                                          seed=17, mask=self.mask, rho_min=.2)
        grid = integ._grid_for(self.data)
        trial = self.data["rho"].clone()
        for rows in grid.free_regions():
            trial[rows[0]] -= 2.
            trial[rows[1]] += 2.
        result, stats = integ._apply_density_floor(trial, grid)
        self.assert_constraints(result)
        self.assertTrue(torch.all(result[grid.free] >= .2))
        self.assertTrue(torch.all(stats["floor_added_particles"] > 0))
        for rows in grid.free_regions():
            torch.testing.assert_close(result[rows].sum(0), self.data["rho"][rows].sum(0), rtol=0., atol=2e-12)

    def test_invalid_masks_are_rejected(self):
        for mask in (self.mask.float(), self.mask[:, None], self.mask[:-1], torch.tensor(True)):
            with self.subTest(shape=mask.shape), self.assertRaisesRegex(ValueError, "boolean.*n_grid"):
                self.integrator(EulerIntegrator, mask).step(self.data)
        # A frozen voxel must hold a positive density; an empty one is an excluded voxel.
        empty = dict(self.data, rho=self.data["rho"].detach().clone())
        empty["rho"][torch.nonzero(self.mask)[0, 0]] = 0.
        with self.assertRaisesRegex(ValueError, "Frozen voxels keep finite positive densities"):
            self.integrator(EulerIntegrator, self.mask).step(empty)

    def test_floor_conserves_compartments_behind_excluded_walls(self):
        # Excluded walls split the grid like frozen ones; the repair must not move mass across them.
        data = linear_field(shape=(8, 2, 2))
        x = data["grid_positions"][:, 0]
        excluded = (x == 0) | (x == 4)
        data["excluded_mask"] = excluded
        data["rho"][excluded] = 0
        integ = StochasticEulerIntegrator(LinearModel([[2., .5], [.5, 1.]]), DensityMobility(.1), .01, seed=3, rho_min=.2)
        grid = integ._grid_for(data)
        self.assertEqual(len(grid.free_regions()), 2)
        trial = data["rho"].clone()
        rows = torch.nonzero((x > 0) & (x < 4)).flatten()
        trial[rows[0], 0] -= 1.95
        trial[rows[1], 0] += 1.95
        repaired, stats = integ._apply_density_floor(trial, grid)
        self.assertTrue(torch.all(repaired[excluded] == 0))
        self.assertTrue(torch.all(repaired[~excluded] >= .2))
        self.assertGreater(float(stats["floor_added_particles"][0]), 0.)
        for compartment in ((x > 0) & (x < 4), x > 4):
            torch.testing.assert_close(repaired[compartment].sum(0), data["rho"][compartment].sum(0), rtol=0., atol=2e-12)

    def test_reference_density_excludes_frozen_voxels(self):
        # A dense frozen slab must not shift the homogeneous reference of the stabilizer and the cap.
        data = linear_field(shape=(8, 3, 2))
        frozen = data["grid_positions"][:, 0] < 4
        data["mask"] = frozen
        data["rho"] = data["rho"].clone()
        data["rho"][frozen] = 9.
        integ = EulerIntegrator(LinearModel([[2., .5], [.5, 1.]]), ConstantMobility([[1., .3], [.3, 2.]]), 1e-3)
        fluid = data["rho"][~frozen].mean(0)
        self.assertAlmostEqual(integ.stability_limit(data), integ.stability_limit(data, reference_density=fluid), places=12)
        implicit = SemiImplicitIntegrator(LinearModel([[2., .5], [.5, 1.]]), ConstantMobility([[1., .3], [.3, 2.]]), .1)
        implicit.step(data)
        torch.testing.assert_close(implicit._reference, fluid, rtol=0, atol=0)
        # All frozen: nothing evolves, and the reference falls back to the accessible mean.
        still = dict(data, mask=torch.ones_like(frozen))
        self.assertTrue(torch.equal(SemiImplicitIntegrator(LinearModel([[2., .5], [.5, 1.]]), ConstantMobility([[1., .3], [.3, 2.]]), .1).step(still), still["rho"]))

    def test_masked_implicit_solve_converges_at_stiff_steps(self):
        # 16^3 two-species linear model: dt * lambda_max reaches 2e4; the preconditioned solve
        # stays within one restart cycle and the increment vanishes on frozen rows.
        shape = (16, 16, 16)
        positions = torch.cartesian_prod(*(torch.arange(size) for size in shape))
        x = positions[:, 0].double()
        rho = torch.full((len(positions), 2), 2.0, dtype=torch.float64)
        rho += .3 * torch.cos(2 * math.pi * x / 16)[:, None] * torch.arange(1, 3)
        rho += .1 * torch.cos(2 * math.pi * positions[:, 1].double() / 16)[:, None]
        data = {"rho": rho, "grid_size": torch.tensor(shape), "grid_positions": positions,
                "grid_spacing": torch.tensor([.5, .5, .5], dtype=torch.float64),
                "temperature": torch.tensor(2.5, dtype=torch.float64), "beta": torch.tensor(.4, dtype=torch.float64),
                "mask": x < 2}
        applications = []
        original = integrator_module._gmres

        def counting(action, rhs, precondition):
            calls = [0]

            def counted(vector):
                calls[0] += 1
                return action(vector)
            try:
                return original(counted, rhs, precondition)
            finally:
                applications.append(calls[0])
        integrator_module._gmres = counting
        try:
            for dt in (.1, 50.):
                integ = SemiImplicitIntegrator(LinearModel([[2., .5], [.5, 1.]]), ConstantMobility([[1., .3], [.3, 2.]]), dt, mask=data["mask"])
                updated = integ.step(data)
                self.assertTrue(torch.equal(updated[data["mask"]], data["rho"][data["mask"]]))
                torch.testing.assert_close(updated.sum(0), data["rho"].sum(0), rtol=0, atol=1e-10)
                self.assertLessEqual(float(integ.beta_A(dict(data, rho=updated))), float(integ.beta_A(data)))
        finally:
            integrator_module._gmres = original
        self.assertEqual(len(applications), 2)
        self.assertTrue(all(count <= 40 for count in applications), applications)

    def test_unconverged_solve_is_an_adaptive_rejection(self):
        attempts = []
        original = integrator_module._gmres

        def failing_once(action, rhs, precondition):
            attempts.append(1)
            if len(attempts) == 1:
                raise ConvergenceError("Masked implicit solve did not converge; reduce the timestep.")
            return original(action, rhs, precondition)
        integrator_module._gmres = failing_once
        try:
            result = AdaptiveDynamics(self.integrator(SemiImplicitIntegrator, self.mask), StepControl(1e-6, .02)).run(self.data, .04)
        finally:
            integrator_module._gmres = original
        self.assertIs(result["metadata"]["completed"], True)
        self.assertEqual(result["metadata"]["rejected_solve"], 1)
        self.assertGreater(len(attempts), 1)
        for rho in result["rho"]:
            self.assert_constraints(rho)

    def test_masked_implicit_float32_matches_float64(self):
        data = {key: (value.float() if torch.is_tensor(value) and value.is_floating_point() else value)
                for key, value in linear_field().items()}
        data["mask"] = data["grid_positions"][:, 0] % 4 == 0
        double = {key: (value.double() if torch.is_tensor(value) and value.is_floating_point() else value)
                  for key, value in data.items()}
        h = [[2., .5], [.5, 1.]]
        m = [[1., .3], [.3, 2.]]
        for dt in (1e-4, .01, .1):
            with self.subTest(dt=dt):
                single = SemiImplicitIntegrator(LinearModel(h).float(), ConstantMobility(m), dt)
                expected = SemiImplicitIntegrator(LinearModel(h), ConstantMobility(m), dt).step(double)
                actual = single.step(data)
                self.assertTrue(torch.equal(actual[data["mask"]], data["rho"][data["mask"]]))
                torch.testing.assert_close(actual.double(), expected, rtol=2e-6, atol=2e-6)
                for rows in single._grid.free_regions():
                    torch.testing.assert_close(actual[rows].double().sum(0), data["rho"][rows].double().sum(0),
                                               rtol=0, atol=1e-5)

    def test_preconditioner_projection_preserves_disconnected_regions(self):
        data = linear_field(shape=(8, 3, 2))
        data["mask"] = data["grid_positions"][:, 0] % 4 == 0
        grid = self.integrator(EulerIntegrator)._grid_for(data)
        vector = torch.randn(data["rho"].shape, generator=torch.Generator().manual_seed(23), dtype=torch.float64)
        original = vector.clone()
        projected = grid.project_free_increment(vector)
        self.assertTrue(torch.equal(vector, original))
        self.assertTrue(torch.all(projected[data["mask"]] == 0))
        for rows in grid.free_regions():
            torch.testing.assert_close(projected[rows].sum(0), projected.new_zeros(2), rtol=0, atol=3e-15)

    def test_stiff_implicit_conserves_each_region_over_multiple_steps(self):
        data = linear_field(shape=(16, 16, 16), spacing=(.5, .5, .5))
        data["rho"] = 2. + .1 * torch.randn(data["rho"].shape,
                                            generator=torch.Generator().manual_seed(17), dtype=torch.float64)
        data["mask"] = data["grid_positions"][:, 0] % 4 == 0
        integrator = SemiImplicitIntegrator(LinearModel([[2., .5], [.5, 1.]]),
                                          ConstantMobility([[1., .3], [.3, 2.]]), 50.)
        state = dict(data)
        for _ in range(5):
            state["rho"] = integrator.step(state)
            self.assertTrue(torch.equal(state["rho"][data["mask"]], data["rho"][data["mask"]]))
            for rows in integrator._grid.free_regions():
                torch.testing.assert_close((state["rho"][rows] - data["rho"][rows]).sum(0),
                                           data["rho"].new_zeros(2), rtol=0, atol=1e-10)

    def test_rng_restart_accepts_a_stacked_state_view(self):
        integrator = self.integrator(StochasticEulerIntegrator, self.mask, dt=1e-4)
        first = integrator.get_rng_state(self.data)
        integrator.step(self.data)
        second = integrator.get_rng_state(self.data)
        view = torch.stack((first, second))[1]
        self.assertGreater(view.storage_offset(), 0)
        integrator.set_rng_state(self.data, view)
        expected = self.integrator(StochasticEulerIntegrator, self.mask, dt=1e-4)
        expected.set_rng_state(self.data, view.clone())
        self.assertTrue(torch.equal(integrator.get_rng_state(self.data), view))
        self.assertTrue(torch.equal(integrator.step(self.data), expected.step(self.data)))

    def test_nonlocal_implicit_curvature_and_permuted_masked_rows(self):
        state = dict(self.data, mask=self.mask)
        mobility = ConstantMobility([[.4, -.1], [-.1, .25]])
        integ = SemiImplicitIntegrator(NonlocalQuadraticModel(), mobility, .2)
        result = integ.step(state)
        self.assert_constraints(result)
        self.assertLess(float(integ.beta_A(dict(state, rho=result))), float(integ.beta_A(state)))
        self.assertTrue(torch.isfinite(result).all())
        # Arbitrary input row ordering uses the same shared topology and solve.
        permutation = torch.arange(len(self.mask)-1, -1, -1)
        permuted = dict(state)
        for name in ("rho", "mask", "grid_positions", "grid_center", "V_ext"):
            permuted[name] = state[name][permutation]
        a = self.integrator(SemiImplicitIntegrator).step(state)
        b = self.integrator(SemiImplicitIntegrator).step(permuted)
        torch.testing.assert_close(b, a[permutation], rtol=1e-12, atol=1e-12)

    def test_restart_reuses_frozen_mask_and_absolute_field_time(self):
        field = RampedField(torch.zeros_like(self.data["V_ext"]), self.data["V_ext"], .025)
        integ = self.integrator(EulerIntegrator, dt=.001)
        first = Dynamics(integ, field, mask=self.mask).run(self.data, 3)
        resumed = dict(self.data, rho=first["rho"][-1], mask=first["metadata"]["mask"])
        second = Dynamics(self.integrator(EulerIntegrator, dt=.001), field).run(resumed, 3, start_time=.003)
        full = Dynamics(self.integrator(EulerIntegrator, dt=.001), field, mask=self.mask).run(self.data, 6)
        self.assertTrue(torch.equal(second["rho"][-1], full["rho"][-1]))
        self.assert_constraints(second["rho"][-1])
        torch.testing.assert_close(second["beta_work"][-1]+first["beta_work"][-1], full["beta_work"][-1])


if __name__ == "__main__":
    unittest.main()
