"""Shared reporting preserves driver decisions and physical diagnostics."""

import unittest
from unittest.mock import patch

import torch

from equicdft.solver import Constraints, ConvergenceOptions, EulerOptions
from equicdft import FixedDipoleIdeal, GridSolver
from equicdft.solver.reporting import finalize_result
from tests.ideal_helpers import IdealOnly


class TestSolveReporting(unittest.TestCase):
    def state(self, **extra):
        return dict(n_iter=2, n_evaluations=4, objective_history=[-1., -2., -3.],
                    line_search_failures=0, final_relative_density_change=.02, **extra)

    def finalize(self, state, converged=False, result=None):
        return finalize_result({} if result is None else result, state,
                               method="euler", beta_multiplier=1.2, converged=converged)

    def test_status_precedence_and_explicit_driver_status(self):
        cases = [
            (False, {}, "max_iter"),
            (False, {"stalled": True}, "stalled"),
            (False, {"line_search_failures": 1, "stalled": True}, "line_search_failed"),
            (True, {"line_search_failures": 1, "stalled": True}, "converged"),
            (False, {"status": "max_iter", "stalled": True}, "max_iter"),
            (False, {"status": "line_search_failed"}, "line_search_failed"),
        ]
        for converged, updates, expected in cases:
            with self.subTest(converged=converged, updates=updates):
                state = self.state()
                state.update(updates)
                result = self.finalize(state, converged)
                self.assertEqual(result["status"], expected)
                self.assertEqual(result["converged"], converged)

    def test_common_counters_defaults_and_optional_history(self):
        for extras in ({}, dict(history=[{"beta_A": -1.}],
                                final_relative_polarization_change=.01,
                                mixing_backtracks=3, final_mixing=.2,
                                solver_anderson=True, anderson_attempts=4,
                                anderson_accepted=2, anderson_rejected=1, anderson_resets=1)):
            state = self.state(**extras)
            result = self.finalize(state)
            for key in state:
                self.assertEqual(result[key], state[key])
            self.assertEqual(result["solver_method"], "euler")
            self.assertEqual(result["solver_beta_multiplier"], 1.2)
            self.assertEqual(result["mixing_backtracks"], extras.get("mixing_backtracks", 0))
            self.assertEqual(result["solver_anderson"], extras.get("solver_anderson", False))
            self.assertEqual("history" in result, "history" in extras)
            self.assertEqual("final_relative_polarization_change" in result,
                             "final_relative_polarization_change" in extras)

    def test_finalization_does_not_touch_physical_outputs_or_driver_state(self):
        physical = {key: torch.tensor(float(index)) for index, key in enumerate((
            "rho", "dipole_density", "beta_A", "density_residual",
            "polarization_kkt_residual", "maximum_residual", "full_converged",
        ))}
        result = dict(physical)
        state = self.state()
        before = dict(state)
        self.assertIs(self.finalize(state, result=result), result)
        for key, value in physical.items():
            self.assertIs(result[key], value)
        self.assertEqual(state, before)

    def case(self, polarized):
        rho = torch.full((4, 1), .2, dtype=torch.float64)
        data = dict(rho=rho, V_ext=rho.new_tensor([[-.2], [.1], [.3], [-.1]]),
                    beta=rho.new_tensor(1.), grid_spacing=rho.new_ones(3))
        if polarized:
            data.update(dipole_density=rho.new_zeros(4, 1, 3),
                        E_ext=rho.new_full((4, 1, 3), .1))
        model = IdealOnly(FixedDipoleIdeal(.7) if polarized else None)
        return GridSolver(model), data

    def test_all_four_drivers_use_shared_finalization(self):
        for polarized in (False, True):
            module = "dipoles" if polarized else "solver"
            for method in ("euler", "minimize"):
                with self.subTest(polarized=polarized, method=method):
                    solver, data = self.case(polarized)
                    with patch("equicdft.solver." + module + ".finalize_result",
                               wraps=finalize_result) as finalize:
                        result = solver.solve(
                            data,
                            method=method,
                            convergence=ConvergenceOptions(max_iter=2),
                            constraints=Constraints(particle_numbers=[.8]),
                        )
                    finalize.assert_called_once()
                    self.assertEqual(result["solver_method"], method)
                    self.assertEqual(result["n_iter"], len(result["objective_history"]) - 1)
                    self.assertGreaterEqual(result["n_evaluations"], result["n_iter"] + 1)
                    self.assertNotIn("iterations", result)
                    if polarized:
                        self.assertEqual(result["objective_history"],
                                         [entry["beta_A"] for entry in result["history"]])
                        self.assertIn("final_relative_polarization_change", result)
                    else:
                        self.assertNotIn("history", result)

    def test_model_training_mode_is_restored_on_evaluation_failure(self):
        for polarized in (False, True):
            for training in (False, True):
                solver, data = self.case(polarized)
                solver.model.train(training)

                def fail(*args, **kwargs):
                    self.assertFalse(solver.model.training)
                    raise RuntimeError("evaluation failed")

                with patch.object(solver, "evaluate", side_effect=fail):
                    with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                        solver.solve(
                            data,
                            convergence=ConvergenceOptions(max_iter=2),
                            constraints=Constraints(particle_numbers=[.8]),
                        )
                self.assertEqual(solver.model.training, training)

    def test_scalar_and_polarized_euler_distinguish_stall_from_budget(self):
        for polarized in (False, True):
            for budget, expected in ((1, "max_iter"), (10, "stalled")):
                with self.subTest(polarized=polarized, budget=budget):
                    solver, data = self.case(polarized)
                    result = solver.solve(
                        data, method="euler",
                        constraints=Constraints(particle_numbers=[.8]),
                        convergence=ConvergenceOptions(max_iter=budget, tolerance_residual=1e-14),
                        euler=EulerOptions(tolerance_change=1.),
                    )
                    self.assertFalse(result["converged"])
                    self.assertEqual(result["n_iter"], 1)
                    self.assertEqual(result["status"], expected)
