import math
import unittest

import torch

from equicdft.evolution import (
    AdaptiveDynamics, ConstantMobility, Dynamics, EulerIntegrator, ExternalField,
    RampedField, SemiImplicitIntegrator, StepControl, StochasticEulerIntegrator,
)
from tests.evolution.test_dynamics import QuadraticModel, field_data


class Switch(ExternalField):
    def __init__(self, value, time):
        self.value = value
        self.breakpoints = (time,)

    def __call__(self, time, data):
        return self.value if time >= self.breakpoints[0] else torch.zeros_like(self.value)


class TestExternalField(unittest.TestCase):
    def setUp(self):
        self.data = field_data()
        self.zero = torch.zeros_like(self.data["rho"])

    def integrator(self, cls=EulerIntegrator, dt=.01, mobility=.1):
        kwargs = {"seed": 17} if cls is StochasticEulerIntegrator else {}
        return cls(QuadraticModel(), ConstantMobility(mobility), dt, **kwargs)

    def control(self, **kwargs):
        return StepControl(dt_min=1e-8, dt_max=.04, **kwargs)

    def test_ramp_units_endpoints_and_input_ownership(self):
        initial, final = self.zero.clone(), torch.ones_like(self.zero)
        field = RampedField(initial, final, 2., start_time=1.)
        initial.add_(10)
        final.add_(10)
        for t, value in [(0., 0.), (1., 0.), (2., .5), (3., 1.), (5., 1.)]:
            torch.testing.assert_close(field(t, self.data), torch.full_like(self.zero, value))
        self.assertEqual(field.breakpoints, (1., 3.))

    def test_static_equivalence_all_integrators(self):
        for cls in (EulerIntegrator, SemiImplicitIntegrator, StochasticEulerIntegrator):
            field = RampedField(self.data["V_ext"], self.data["V_ext"], 1.)
            reference = Dynamics(self.integrator(cls)).run(self.data, 4)
            dynamic = Dynamics(self.integrator(cls), field).run(self.data, 4)
            torch.testing.assert_close(reference["rho"], dynamic["rho"], rtol=1e-13, atol=1e-13)
            torch.testing.assert_close(reference["beta_A"], dynamic["beta_A"])
            self.assertTrue(torch.all(dynamic["beta_work"] == 0))

    def test_fixed_ramp_matches_manual_euler_and_records_actual_fields(self):
        field = RampedField(self.zero, self.data["V_ext"], .03)
        original = self.data["V_ext"].clone()
        result = Dynamics(self.integrator(), field).run(self.data, 5, save_every=2)
        state = dict(self.data)
        manual = self.integrator()
        for step in range(5):
            state["V_ext"] = field(step * .01, state)
            state["rho"] = manual.step(state)
        torch.testing.assert_close(result["rho"][-1], state["rho"], rtol=1e-13, atol=1e-13)
        for time, rho, potential, energy in zip(result["time"], result["rho"], result["V_ext"], result["beta_A"]):
            torch.testing.assert_close(potential, field(float(time), self.data))
            torch.testing.assert_close(energy, manual.beta_A(dict(self.data, rho=rho, V_ext=potential)))
        torch.testing.assert_close(self.data["V_ext"], original)
        torch.testing.assert_close(result["particle_numbers"], result["particle_numbers"][0].expand_as(result["particle_numbers"]))

    def test_uniform_time_gauge_changes_only_energy(self):
        for adaptive in (False, True):
            baseline = RampedField(self.zero, self.zero, 1.)
            shifted = RampedField(self.zero, torch.tensor([2., -1.]).to(self.zero).expand_as(self.zero), 1.)
            def run(field):
                integ = self.integrator()
                if adaptive:
                    return AdaptiveDynamics(integ, self.control(), field).run(self.data, .1)
                return Dynamics(integ, field).run(self.data, 10)
            a, b = run(baseline), run(shifted)
            torch.testing.assert_close(a["time"], b["time"])
            torch.testing.assert_close(a["rho"], b["rho"], rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(b["beta_A"] - a["beta_A"], b["beta_work"], rtol=1e-12, atol=1e-12)

    def test_fixed_switch_splits_step_and_applies_jump_work_at_fixed_density(self):
        field = Switch(self.data["V_ext"], .015)
        integ = self.integrator(dt=.02)
        result = Dynamics(integ, field).run(self.data, 1)
        state = dict(self.data, V_ext=self.zero)
        first = self.integrator(dt=.015).step(state)
        expected_work = float(self.data["beta"] * self.data["grid_spacing"].prod() * (first * field.value).sum())
        final = self.integrator(dt=.005).step(dict(state, rho=first, V_ext=field.value))
        torch.testing.assert_close(result["rho"][-1], final)
        self.assertAlmostEqual(float(result["beta_work"][-1]), expected_work, places=12)
        self.assertEqual(integ.dt, .02)

    def test_adaptive_switch_is_recorded_right_continuously(self):
        field = Switch(torch.ones_like(self.zero), .015)
        result = AdaptiveDynamics(self.integrator(mobility=0.), self.control(), field).run(self.data, .04)
        self.assertIn(.015, result["time"].tolist())
        at_switch = result["time"].tolist().index(.015)
        torch.testing.assert_close(result["V_ext"][at_switch], field.value)
        torch.testing.assert_close(result["beta_A"] - result["beta_A"][0], result["beta_work"])
        torch.testing.assert_close(result["rho"], result["rho"][0].expand_as(result["rho"]))

    def test_stochastic_absolute_time_restart(self):
        field = RampedField(self.zero, self.data["V_ext"], .05)
        full = Dynamics(self.integrator(StochasticEulerIntegrator), field).run(self.data, 6)
        first = Dynamics(self.integrator(StochasticEulerIntegrator), field).run(self.data, 3)
        state = dict(self.data, rho=first["rho"][-1])
        integ = self.integrator(StochasticEulerIntegrator)
        integ.set_rng_state(state, first["rng_state"][-1])
        second = Dynamics(integ, field).run(state, 3, start_time=.03)
        torch.testing.assert_close(second["rho"], full["rho"][3:], rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(second["beta_work"] + first["beta_work"][-1], full["beta_work"][3:])
        torch.testing.assert_close(second["rng_state"][-1], full["rng_state"][-1])

    def test_stochastic_switch_matches_split_reference(self):
        field = Switch(self.data["V_ext"], .015)
        integ = self.integrator(StochasticEulerIntegrator, dt=.02)
        actual = Dynamics(integ, field).run(self.data, 1)
        reference = self.integrator(StochasticEulerIntegrator, dt=.015)
        state = dict(self.data, V_ext=self.zero)
        state["rho"] = reference.step(state)
        state["V_ext"] = field.value
        reference.dt = .02 - .015
        expected = reference.step(state)
        torch.testing.assert_close(actual["rho"][-1], expected, rtol=0, atol=0)
        torch.testing.assert_close(actual["rng_state"][-1], reference.get_rng_state(state))

    def test_adaptive_rejections_do_not_accumulate_work(self):
        field = RampedField(self.zero, self.data["V_ext"] * 100, .1)
        dynamics = AdaptiveDynamics(self.integrator(dt=.04), self.control(error_tolerance=1e-7), field)
        result = dynamics.run(self.data, .1)
        self.assertGreater(result["metadata"]["rejected_accuracy"] + result["metadata"]["rejected_energy"], 0)
        # All saved stages use their accepted time; failed trial fields never leak.
        for t, potential in zip(result["time"], result["V_ext"]):
            torch.testing.assert_close(potential, field(float(t), self.data))
        corrected = result["beta_A"] - result["beta_work"]
        self.assertTrue(torch.all(torch.diff(corrected) <= 1e-10))
        fine = Dynamics(self.integrator(dt=1e-4), field).run(self.data, 1000, save_every=1000)
        torch.testing.assert_close(result["beta_work"][-1], fine["beta_work"][-1], rtol=2e-3, atol=2e-4)

    def test_adaptive_restart_and_semiimplicit_ramp(self):
        field = RampedField(self.zero, self.data["V_ext"], .1)
        control = StepControl(dt_min=.01, dt_max=.01, error_tolerance=.1)
        def run(data, end, start=0):
            return AdaptiveDynamics(self.integrator(SemiImplicitIntegrator), control, field).run(
                data, end, start_time=start)
        full = run(self.data, .08)
        first = run(self.data, .04)
        second = run(dict(self.data, rho=first["rho"][-1]), .08, .04)
        torch.testing.assert_close(second["rho"][-1], full["rho"][-1], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(second["beta_work"][-1] + first["beta_work"][-1], full["beta_work"][-1])

    def test_work_uses_voxel_volume_and_beta(self):
        data = dict(self.data, grid_spacing=self.data["grid_spacing"] * 2, beta=torch.tensor(.25))
        field = RampedField(self.zero, torch.ones_like(self.zero), .1)
        result = Dynamics(self.integrator(mobility=0.), field).run(data, 10)
        expected = data["beta"] * data["grid_spacing"].prod() * data["rho"].sum()
        torch.testing.assert_close(result["beta_work"][-1], expected)

    def test_scalar_spacing_has_cubic_work_volume(self):
        from equicdft.evolution.field import _work
        a = dict(self.data, grid_spacing=.5, V_ext=self.zero)
        b = dict(a, V_ext=torch.ones_like(self.zero))
        torch.testing.assert_close(_work(a, b), .5**3 * self.data["beta"] * self.data["rho"].sum())

    def test_adaptive_ramp_converges_to_driven_linear_mode(self):
        # d a/dt = -lambda a - M q^2 t for V=t cos(kx), a(0)=0.
        data = dict(self.data, rho=torch.ones_like(self.zero))
        shape = torch.cos(2 * math.pi * data["grid_positions"][:, 0].double() / 4)
        field = RampedField(self.zero, shape[:, None].expand_as(self.zero), 1.)
        model = QuadraticModel()
        model.curvature.copy_(torch.eye(2, dtype=torch.float64))
        q2 = 2 / float(data["grid_spacing"][0])**2
        lam = .1 * q2 / float(data["beta"])
        t = .3
        amplitude = -.1 * q2 * (t / lam + math.expm1(-lam*t) / lam**2)
        expected = 1 + amplitude * shape[:, None].expand_as(self.zero)
        errors = []
        for tolerance in (1e-3, 1e-5):
            integ = EulerIntegrator(model, ConstantMobility(.1), .04)
            result = AdaptiveDynamics(integ, self.control(error_tolerance=tolerance), field).run(data, t)
            errors.append(float((result["rho"][-1] - expected).abs().max()))
            corrected = result["beta_A"] - result["beta_work"]
            self.assertTrue(torch.all(torch.diff(corrected) <= 1e-10))
        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[1], 3e-4)

    def test_field_replacement_reuses_geometry_not_stale_potential(self):
        integ = self.integrator()
        before = integ.rhs(dict(self.data, V_ext=self.zero))
        grid = integ._grid_for(dict(self.data, V_ext=self.zero))
        updated = dict(self.data, V_ext=self.data["V_ext"] * 10)
        after = integ.rhs(updated)
        self.assertIs(grid.geometry, integ._grid_for(updated).geometry)
        self.assertFalse(torch.equal(before, after))

    def test_masks_and_species_selectivity(self):
        data = dict(self.data, rho=torch.ones_like(self.zero))
        data["excluded_mask"] = torch.zeros(len(self.zero), dtype=torch.bool)
        data["excluded_mask"][0] = True
        data["rho"][0] = 0
        final = self.zero.clone()
        final[:, 0] = self.data["V_ext"][:, 0]
        result = Dynamics(self.integrator(), RampedField(self.zero, final, .02)).run(data, 5)
        self.assertTrue(torch.all(result["rho"][:, 0] == 0))
        torch.testing.assert_close(result["particle_numbers"], result["particle_numbers"][0].expand_as(result["particle_numbers"]))

    def test_invalid_field_preserves_last_state_and_rng(self):
        class Invalid(ExternalField):
            def __call__(self, time, data):
                return torch.zeros_like(data["rho"]) if time < .025 else data["rho"][:, 0]
        integ = self.integrator(StochasticEulerIntegrator)
        with self.assertRaisesRegex(ValueError, "ExternalField") as caught:
            Dynamics(integ, Invalid()).run(self.data, 5)
        trajectory = caught.exception.trajectory
        self.assertEqual(float(trajectory["time"][-1]), .02)
        torch.testing.assert_close(trajectory["rng_state"][-1], integ.get_rng_state(self.data))
        self.assertEqual(integ.dt, .01)

    def test_validation(self):
        for duration in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                RampedField(self.zero, self.zero, duration)
        with self.assertRaises(TypeError):
            Dynamics(self.integrator(), field=lambda t, d: self.zero)
        for t in (-1, float("inf")):
            with self.assertRaises(ValueError):
                Dynamics(self.integrator()).run(self.data, 1, start_time=t)
        with self.assertRaises(ValueError):
            AdaptiveDynamics(self.integrator(), self.control()).run(self.data, 1., start_time=2.)

    def test_invalid_shapes_dtype_and_breakpoints(self):
        class Bad(ExternalField):
            def __call__(self, time, data):
                return self.value
        bad = Bad()
        for value in (self.zero[:, 0], self.zero.float(), self.zero + float("nan")):
            bad.value = value
            with self.assertRaises(ValueError):
                Dynamics(self.integrator(), bad).run(self.data, 1)
        for points in ((.2, .1), (.1, .1), (-.1,), (float("inf"),)):
            bad.breakpoints = points
            with self.assertRaises(ValueError):
                Dynamics(self.integrator(), bad)


if __name__ == "__main__":
    unittest.main()
