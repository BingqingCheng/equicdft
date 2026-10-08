import math
import unittest

import torch

from equicdft.evolution import (
    AdaptiveDynamics, ConstantMobility, EulerIntegrator, LocalMobility, SemiImplicitIntegrator, StepControl,
)
from tests.evolution.test_integrator import QuadraticModel, field


def modal_rates(h, m, beta, shape=(8, 3, 2), spacing=(0.4, 0.7, 1.1)):
    """Exact relaxation rates q_h^2 eig(M H)/beta of the linear model, per mode."""
    q2 = torch.zeros(shape, dtype=torch.float64)
    for axis, (size, step) in enumerate(zip(shape, spacing)):
        modes = 4 * torch.sin(math.pi * torch.arange(size, dtype=torch.float64) / size) ** 2 / step ** 2
        q2 += modes.reshape([size if i == axis else 1 for i in range(3)])
    return q2, torch.linalg.eigvals(m @ h).real / beta


class OffsetModel(QuadraticModel):
    """The same functional with a constant added to its free energy."""

    def __init__(self, curvature, offset):
        super().__init__(curvature)
        self.offset = offset

    def forward(self, data, compute_c1=False, **kwargs):
        result = super().forward(data, compute_c1=compute_c1, **kwargs)
        result["beta_F"] = result["beta_F"] + self.offset
        return result


class TestAdvance(unittest.TestCase):
    def setUp(self):
        self.data = field()
        self.h = torch.tensor([[2.0, 0.5], [0.5, 1.0]], dtype=torch.float64)
        self.m = torch.tensor([[1.0, 0.3], [0.3, 2.0]], dtype=torch.float64)
        self.model = QuadraticModel(self.h)
        self.mobility = ConstantMobility(self.m)

    def test_advance_reproduces_step_without_reevaluating_the_model(self):
        for cls in (EulerIntegrator, SemiImplicitIntegrator):
            integrator = cls(self.model, self.mobility, .01)
            rate = integrator.rhs(self.data)
            expected = cls(self.model, self.mobility, .01).step(self.data)
            torch.testing.assert_close(integrator.advance(self.data, rate), expected, rtol=0, atol=0)
            smaller = cls(self.model, self.mobility, .0025).step(self.data)
            torch.testing.assert_close(integrator.advance(self.data, rate, dt=.0025), smaller, rtol=1e-14, atol=1e-14)
            self.assertEqual(integrator.dt, .01)
        # An explicit step far beyond its stability bound is rejected; the semi-implicit step of a
        # linear model is backward Euler and lands on the uniform equilibrium instead.
        euler = EulerIntegrator(self.model, self.mobility, .01)
        with self.assertRaisesRegex(ValueError, "reduce the timestep"):
            euler.advance(self.data, euler.rhs(self.data), dt=1e4)
        implicit = SemiImplicitIntegrator(self.model, self.mobility, .01)
        huge = implicit.advance(self.data, implicit.rhs(self.data), dt=1e6)
        torch.testing.assert_close(huge, self.data["rho"].mean(0).expand_as(huge), rtol=0, atol=1e-5)


class TestAdaptiveDynamics(unittest.TestCase):
    def setUp(self):
        self.data = field()
        self.h = torch.tensor([[2.0, 0.5], [0.5, 1.0]], dtype=torch.float64)
        self.m = torch.tensor([[1.0, 0.3], [0.3, 2.0]], dtype=torch.float64)
        self.model = QuadraticModel(self.h)
        self.mobility = ConstantMobility(self.m)
        self.q2, self.coupling = modal_rates(self.h, self.m, float(self.data["beta"]))
        # The initial field excites the lowest x mode only; this is its stiffer relaxation rate.
        self.rate = float((self.q2[1, 0, 0] * self.coupling).max())

    def exact(self, t, rho=None):
        """The linear model relaxes each Fourier mode of the fluctuation independently."""
        rho = self.data["rho"] if rho is None else rho
        mean = rho.mean(0)
        fluctuation = (rho - mean).reshape(8, 3, 2, 2)
        generator = -self.q2[..., None, None] * (self.m @ self.h) / float(self.data["beta"])
        propagated = torch.matrix_exp(generator * t).to(torch.complex128) @ torch.fft.fftn(fluctuation.to(torch.complex128), dim=(0, 1, 2)).unsqueeze(-1)
        return mean + torch.fft.ifftn(propagated.squeeze(-1), dim=(0, 1, 2)).real.reshape(48, 2)

    def implicit(self, stabilizer=1., dt=1e-3):
        """Semi-implicit integrator whose stabilizer is ``stabilizer`` times the true curvature."""
        return SemiImplicitIntegrator(self.model, self.mobility, dt, reference_curvature=stabilizer * self.h)

    @staticmethod
    def cost_bounds(metadata):
        """Model evaluations: one to start, two per accepted step, at most two per rejection."""
        accepted, accuracy, energy, positivity, solve = (metadata[key] for key in
                                                         ("accepted", "rejected_accuracy", "rejected_energy",
                                                          "rejected_positivity", "rejected_solve"))
        return 1 + 2 * accepted, 1 + 2 * (accepted + accuracy + energy) + positivity + solve

    def test_control_validation(self):
        StepControl(dt_min=1e-3, dt_max=1.)
        for kwargs in ({"dt_min": 2., "dt_max": 1.}, {"dt_min": 0., "dt_max": 1.}, {"dt_min": 1e-3, "dt_max": 1., "growth": .9},
                       {"dt_min": 1e-3, "dt_max": 1., "shrink": 1.}, {"dt_min": 1e-3, "dt_max": 1., "error_tolerance": -1.},
                       {"dt_min": 1e-3, "dt_max": 1., "max_relative_change": 0.},
                       {"dt_min": 1e-3, "dt_max": 1., "energy_increase": -1e-3}, {"dt_min": 1e-3, "dt_max": 1., "energy_roundoff": math.inf},
                       {"dt_min": 1e-3, "dt_max": 1., "energy_increase": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                StepControl(**kwargs)
        with self.assertRaises(TypeError):
            AdaptiveDynamics(EulerIntegrator(self.model, self.mobility, .01), control={"dt_min": 1e-3})

    def test_run_arguments_are_validated_before_any_evaluation(self):
        integrator = EulerIntegrator(self.model, self.mobility, .01)
        dynamics = AdaptiveDynamics(integrator, StepControl(dt_min=1e-4, dt_max=1.))
        for kwargs in ({"t_end": 0.}, {"t_end": 1., "save_every": 0}, {"t_end": 1., "save_every": 1.5}, {"t_end": 1., "save_every": True},
                       {"t_end": 1., "max_steps": 0}, {"t_end": 1., "max_steps": -3}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError) as caught:
                dynamics.run(self.data, **kwargs)
            self.assertFalse(hasattr(caught.exception, "trajectory"))
        self.assertIsNone(integrator.last_beta_A)

    def test_error_estimate_tracks_the_exact_one_step_error(self):
        """The estimate is never far below the error of the accepted half-step result, in any stiffness regime.

        Step doubling alone misses the slowed kinetics of a mode whose stabilizer is far stiffer than
        its physics (both step sizes freeze the mode alike); the half-step defects catch it.
        """
        control = StepControl(dt_min=1e-9, dt_max=1e3, max_relative_change=1e6, error_tolerance=1e9, energy_increase=1e9)
        accessible = torch.ones(48, dtype=torch.bool)
        cases = [("matched", 1., .1), ("matched", 1., 2.), ("matched", 1., 75.), ("stiff stabilizer", 100., .5),
                 ("stiff stabilizer", 100., 2.), ("stiff stabilizer", 20., .2), ("weak stabilizer", .1, 1.), ("euler", None, .1), ("euler", None, 1.)]
        for label, stabilizer, scaled_dt in cases:
            dt = scaled_dt / self.rate
            integrator = (EulerIntegrator(self.model, self.mobility, dt) if stabilizer is None else self.implicit(stabilizer, dt))
            dynamics = AdaptiveDynamics(integrator, control)
            state = dict(self.data, rho=self.data["rho"].clone())
            rate = integrator.rhs(state)
            counts = {"model_evaluations": 0}
            trial, _, _, estimate = dynamics._trial(state, rate, dt, accessible, integrator.last_beta_A, counts)
            target = self.exact(dt)
            half_error = float(((trial["rho"] - target).abs() / trial["rho"]).max())
            full = integrator.advance(state, rate, dt)
            full_error = float(((full - target).abs() / full).max())
            with self.subTest(label=label, stabilizer=stabilizer, scaled_dt=scaled_dt):
                self.assertEqual(counts["model_evaluations"], 2)
                self.assertGreaterEqual(estimate, .7 * half_error)
                self.assertLessEqual(estimate, 4 * full_error)
                if label == "stiff stabilizer" and scaled_dt >= .5:
                    doubling_only = float(((full - trial["rho"]).abs() / trial["rho"]).max())
                    self.assertLess(doubling_only, .1 * half_error)

    def test_converges_with_growing_steps_and_tolerance_controls_error(self):
        t_end = 2.0
        exact = self.exact(t_end)
        errors = {}
        for label, make in (("euler", lambda: EulerIntegrator(self.model, self.mobility, 1e-3)), ("matched", self.implicit),
                            ("stiff stabilizer", lambda: self.implicit(50.))):
            for tolerance in (1e-2, 1e-3, 1e-4):
                control = StepControl(dt_min=1e-6, dt_max=5., max_relative_change=.5, error_tolerance=tolerance)
                trajectory = AdaptiveDynamics(make(), control).run(self.data, t_end=t_end, save_every=1)
                metadata = trajectory["metadata"]
                with self.subTest(label=label, tolerance=tolerance):
                    self.assertIs(metadata["completed"], True)
                    self.assertAlmostEqual(float(trajectory["time"][-1]), t_end, places=12)
                    self.assertEqual(metadata["rejected_positivity"] + metadata["rejected_energy"], 0)
                    low, high = self.cost_bounds(metadata)
                    self.assertTrue(low <= metadata["model_evaluations"] <= high)
                    if metadata["rejected_accuracy"] == 0:
                        self.assertEqual(metadata["model_evaluations"], low)
                    self.assertEqual(trajectory["dt"].shape, trajectory["time"].shape)
                    self.assertEqual(trajectory["error"].shape, trajectory["time"].shape)
                    self.assertTrue(torch.isnan(trajectory["dt"][0]) and torch.isnan(trajectory["error"][0]))
                    self.assertTrue(torch.all(trajectory["error"][1:] <= tolerance))
                    self.assertTrue(torch.all(torch.diff(trajectory["beta_A"]) <= 1e-9))
                    self.assertGreater(float(trajectory["dt"][1:].max()), 3 * float(trajectory["dt"][1]))
                    torch.testing.assert_close(trajectory["particle_numbers"][-1], trajectory["particle_numbers"][0], rtol=0, atol=1e-12)
                errors[label, tolerance] = float((trajectory["rho"][-1] - exact).abs().max())
            self.assertLess(errors[label, 1e-4], 1e-3)
        # Over-stabilized, the semi-implicit step slows the kinetics; the controller holds that error to the tolerance.
        self.assertGreater(errors["stiff stabilizer", 1e-2], 3e-3)
        self.assertLess(errors["stiff stabilizer", 1e-3], errors["stiff stabilizer", 1e-2] / 3)
        self.assertLess(errors["stiff stabilizer", 1e-4], errors["stiff stabilizer", 1e-3] / 3)
        # Euler never exceeds its linear stability bound; the semi-implicit step is not bound by it.
        limit = float(2 / (self.q2.max() * self.coupling.max()))
        euler = AdaptiveDynamics(EulerIntegrator(self.model, self.mobility, 1e-3), StepControl(dt_min=1e-4, dt_max=5., error_tolerance=1.)).run(self.data, t_end=t_end, save_every=1)
        self.assertLessEqual(float(euler["dt"][1:].max()), .9 * limit + 1e-12)
        self.assertAlmostEqual(euler["metadata"]["dt_cap"], .9 * limit, places=12)

    def test_positivity_rejection_shrinks_and_recovers(self):
        # The first rate is evaluated with a mobility 1e4 times too large, so the first trial
        # step overshoots into negative densities and must be halved repeatedly from the same rate.
        calls = []

        def spiky(rho, temperature):
            calls.append(1)
            return (1e4 if len(calls) == 1 else 1.) * torch.ones_like(rho)

        control = StepControl(dt_min=1e-9, dt_max=10., max_relative_change=50., error_tolerance=1e3, growth=4.)
        trajectory = AdaptiveDynamics(SemiImplicitIntegrator(self.model, LocalMobility(spiky), 1e-2), control).run(self.data, t_end=.5, save_every=1)
        metadata = trajectory["metadata"]
        self.assertGreater(metadata["rejected_positivity"], 5)
        low, high = self.cost_bounds(metadata)
        self.assertTrue(low <= metadata["model_evaluations"] <= high)
        self.assertIs(metadata["completed"], True)
        self.assertTrue(torch.all(trajectory["min_density"] > 0))
        self.assertLess(float(trajectory["dt"][1]), 1e-4)
        self.assertGreater(float(trajectory["dt"][-1]), .1)

    def test_energy_rejection_with_a_weak_stabilizer(self):
        control = StepControl(dt_min=1e-4, dt_max=10., max_relative_change=50., error_tolerance=1e3, growth=4.)
        trajectory = AdaptiveDynamics(self.implicit(.02, 1.), control).run(self.data, t_end=.5)
        self.assertGreater(trajectory["metadata"]["rejected_energy"], 0)
        self.assertTrue(torch.all(torch.diff(trajectory["beta_A"]) <= 1e-9))
        self.assertIs(trajectory["metadata"]["completed"], True)
        # An explicit increment tolerance accepts what the default rejects.
        lenient = AdaptiveDynamics(self.implicit(.02, 1.), StepControl(dt_min=1e-4, dt_max=10., max_relative_change=50., error_tolerance=1e3,
                                                                       growth=4., energy_increase=1e3)).run(self.data, t_end=.5)
        self.assertEqual(lenient["metadata"]["rejected_energy"], 0)
        self.assertFalse(torch.all(torch.diff(lenient["beta_A"]) <= 1e-9))

    def test_energy_acceptance_is_independent_of_the_free_energy_zero(self):
        control = StepControl(dt_min=1e-4, dt_max=10., max_relative_change=50., error_tolerance=1e3, growth=4., energy_roundoff=0.)
        runs = []
        for offset in (0., 1e4):
            model = OffsetModel(self.h, offset)
            weak = SemiImplicitIntegrator(model, self.mobility, 1., reference_curvature=.02 * self.h)
            runs.append(AdaptiveDynamics(weak, control).run(self.data, t_end=.5))
        reference, shifted = runs
        self.assertGreater(reference["metadata"]["rejected_energy"], 0)
        for key in ("accepted", "rejected_energy", "rejected_accuracy", "rejected_positivity", "rejected_solve", "model_evaluations"):
            self.assertEqual(shifted["metadata"][key], reference["metadata"][key], key)
        torch.testing.assert_close(shifted["rho"], reference["rho"], rtol=0, atol=1e-12)
        torch.testing.assert_close(shifted["beta_A"] - 1e4, reference["beta_A"], rtol=0, atol=1e-8)

    def test_second_half_energy_increase_is_rejected_even_if_the_whole_step_descends(self):
        data = field(shape=(8, 1, 1), species=1, spacing=(1., 1., 1.))
        data["beta"] = data["temperature"] = torch.tensor(1., dtype=torch.float64)
        x = data["grid_positions"][:, 0].double()
        data["rho"] = (2. + .0004 * torch.cos(2 * math.pi * x / 8)
                       + .00005 * torch.cos(math.pi * x))[:, None]
        curvature = torch.tensor([[1.]], dtype=torch.float64)
        integrator = SemiImplicitIntegrator(QuadraticModel(curvature), ConstantMobility(1.), 1.5,
                                           reference_curvature=1e-4 * curvature)
        start_energy = integrator.beta_A(data)
        midpoint = dict(data, rho=integrator.advance(data, integrator.rhs(data), .75))
        midpoint_energy = integrator.beta_A(midpoint)
        end = dict(data, rho=integrator.advance(midpoint, integrator.rhs(midpoint), .75))
        end_energy = integrator.beta_A(end)
        self.assertLess(float(end_energy), float(start_energy))
        self.assertGreater(float(end_energy - midpoint_energy), 1e-12 * abs(float(midpoint_energy)))
        with self.assertRaisesRegex(ValueError, "free energy") as caught:
            AdaptiveDynamics(integrator, StepControl(1.5, 1.5)).run(data, t_end=1.5)
        self.assertEqual(caught.exception.trajectory["metadata"]["accepted"], 0)
        self.assertEqual(caught.exception.trajectory["metadata"]["rejected_energy"], 1)
        # The same half-step increase can be explicitly allowed, without changing accuracy controls.
        lenient = StepControl(1.5, 1.5, energy_increase=1e-7)
        trajectory = AdaptiveDynamics(integrator, lenient).run(data, t_end=1.5)
        self.assertIs(trajectory["metadata"]["completed"], True)
        self.assertEqual(trajectory["metadata"]["accepted"], 1)

    def test_accepted_step_safety_factor_does_not_stop_an_admissible_dt_min(self):
        integrator = self.implicit(dt=.001)
        dynamics = AdaptiveDynamics(integrator, StepControl(1e-8, 1., error_tolerance=1.))
        rate = integrator.rhs(self.data)
        _, _, _, error = dynamics._trial(self.data, rate, .001, torch.ones(48, dtype=torch.bool),
                                         integrator.last_beta_A, {"model_evaluations": 0})
        tolerance = 1.05 * error
        self.assertLess(.9 * math.sqrt(tolerance / error), 1.)
        trajectory = AdaptiveDynamics(self.implicit(dt=.001), StepControl(.001, .001, error_tolerance=tolerance)).run(
            self.data, t_end=.002)
        self.assertIs(trajectory["metadata"]["completed"], True)
        self.assertEqual(trajectory["metadata"]["accepted"], 2)
        self.assertTrue(torch.all(trajectory["error"][1:] <= tolerance))
        torch.testing.assert_close(trajectory["dt"][1:], torch.tensor([.001, .001], dtype=torch.float64))

    def test_accuracy_rejection_at_dt_min_still_stops(self):
        control = StepControl(.001, .001, error_tolerance=1e-8)
        with self.assertRaisesRegex(ValueError, "accuracy.*dt_min") as caught:
            AdaptiveDynamics(self.implicit(dt=.001), control).run(self.data, t_end=.002)
        self.assertEqual(caught.exception.trajectory["metadata"]["accepted"], 0)
        self.assertEqual(caught.exception.trajectory["metadata"]["rejected_accuracy"], 1)

    def test_step_bounds_below_dt_min_stop_with_the_last_accepted_state(self):
        integrator = self.implicit()
        control = StepControl(dt_min=1e-2, dt_max=1., max_relative_change=1e-6)
        with self.assertRaisesRegex(ValueError, "below dt_min") as caught:
            AdaptiveDynamics(integrator, control).run(self.data, t_end=1.)
        partial = caught.exception.trajectory
        self.assertIs(partial["metadata"]["completed"], False)
        self.assertEqual(partial["metadata"]["accepted"], 0)
        self.assertEqual(partial["rho"].shape[0], 1)
        torch.testing.assert_close(partial["rho"][0], self.data["rho"])
        # Only the final step may be shorter than dt_min, to land exactly on t_end.
        fixed = StepControl(dt_min=.1, dt_max=.1, max_relative_change=1e6, error_tolerance=1e6, energy_increase=1e6)
        trajectory = AdaptiveDynamics(self.implicit(), fixed).run(self.data, t_end=.25)
        self.assertIs(trajectory["metadata"]["completed"], True)
        torch.testing.assert_close(trajectory["time"], torch.tensor([0., .1, .2, .25], dtype=torch.float64), rtol=0, atol=1e-12)

    def test_failure_at_dt_min_attaches_trajectory(self):
        exploding = LocalMobility(lambda rho, temperature: 1e8 * torch.ones_like(rho))
        control = StepControl(dt_min=1., dt_max=1., max_relative_change=1e6, error_tolerance=1e6)
        with self.assertRaisesRegex(ValueError, "dt_min") as caught:
            AdaptiveDynamics(EulerIntegrator(self.model, exploding, 1.), control).run(self.data, t_end=3.)
        partial = caught.exception.trajectory
        self.assertIs(partial["metadata"]["completed"], False)
        self.assertEqual(partial["rho"].shape[0], 1)
        torch.testing.assert_close(partial["rho"][0], self.data["rho"])

    def test_adaptive_euler_supports_excluded_voxels(self):
        data = field(shape=(8, 2, 2))
        x = data["grid_positions"][:, 0]
        excluded = (x == 0) | (x == 4)
        data["excluded_mask"] = excluded
        data["rho"][excluded] = 0
        integrator = EulerIntegrator(self.model, self.mobility, 1e-3)
        control = StepControl(dt_min=1e-6, dt_max=1., error_tolerance=1e-3)
        trajectory = AdaptiveDynamics(integrator, control).run(data, t_end=.3, save_every=1)
        metadata = trajectory["metadata"]
        self.assertIs(metadata["completed"], True)
        self.assertGreater(metadata["accepted"], 3)
        self.assertTrue(torch.all(trajectory["rho"][:, excluded] == 0))
        self.assertTrue(torch.all(trajectory["min_density"] > 0))
        for compartment in ((x > 0) & (x < 4), (x > 4)):
            torch.testing.assert_close(trajectory["rho"][:, compartment].sum(1), trajectory["rho"][:1, compartment].sum(1).expand(len(trajectory["time"]), 2),
                                       rtol=1e-13, atol=1e-12)
        # The explicit cap is the periodic bound at the mean accessible density, which the walls cannot raise.
        periodic = {key: value for key, value in data.items() if key != "excluded_mask"}
        periodic["rho"] = data["rho"][~excluded].mean(0).expand_as(data["rho"]).clone()
        self.assertAlmostEqual(metadata["dt_cap"], .9 * EulerIntegrator(self.model, self.mobility, 1e-3).stability_limit(periodic), places=12)
        self.assertLessEqual(float(trajectory["dt"][1:].max()), metadata["dt_cap"] + 1e-12)
        with self.assertRaisesRegex(ValueError, "fully accessible"):
            AdaptiveDynamics(SemiImplicitIntegrator(self.model, self.mobility, 1e-3), control).run(data, t_end=.3)


if __name__ == "__main__":
    unittest.main()
