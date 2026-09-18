"""Configuration, checkpoint conversion and solve parity for coupled models."""

import copy
import tempfile
import unittest
from pathlib import Path

import torch

from equicdft import GridSolver, MetalWall, load_model, save_model
from equicdft.legacy import convert_legacy_model, legacy_model_to_current, verify_equivalent
from tests import test_polarization_symmetry as fixtures


class TestPolarizationSerialization(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.get_default_dtype()
        self.rng = torch.random.get_rng_state()
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(71)

    def tearDown(self):
        torch.set_default_dtype(self.dtype)
        torch.random.set_rng_state(self.rng)

    def case(self, metal=False, messages=1, backend="fft"):
        model = fixtures.TestFullPolarizationSymmetry.model(backend, 0. if metal else .7, messages)
        data = fixtures.TestFullPolarizationSymmetry.data()
        if metal:
            model.readout[2] = MetalWall(
                model.readout[2],
                dict(metal_positions=[[.1, .2, .3], [1.6, 1.7, 2.1]],
                     metal_site_groups=[4, 4], metal_group_ids=[4],
                     metal_total_charge=[0.], metal_charge_units="e"),
                metal_sigma=.35, external_field=(0., 0., .03),
            )
        return model, data

    def assert_outputs(self, expected, actual):
        self.assertEqual(expected.keys(), actual.keys())
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)

    def test_round_trip_complete_polarization_models(self):
        for metal in (False, True):
            for backend in ("gather", "fft"):
                for messages in (0, 1):
                    with self.subTest(metal=metal, backend=backend, messages=messages):
                        model, data = self.case(metal, messages, backend)
                        expected = model(dict(data))
                        config = model.to_config()
                        self.assertTrue(config["a_features"]["include_polarization"])
                        self.assertTrue(config["readout"][3]["features"]["include_divergence"])
                        with tempfile.TemporaryDirectory() as directory:
                            path = Path(directory) / "model.pt"
                            save_model(model, path)
                            restored = load_model(path)
                        self.assertEqual(config, restored.to_config())
                        if metal:
                            self.assertIsNone(restored.readout[2]._cache)
                        self.assert_outputs(expected, restored(dict(data)))
                        verify_equivalent(model, restored, data)

    def test_float32_metal_buffers_restore_saved_dtype(self):
        model, data = self.case(metal=True)
        model = model.float()
        data = {k: v.float() if v.is_floating_point() else v for k, v in data.items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_model(model, path)
            restored = load_model(path)
        for tensor in list(restored.parameters()) + list(restored.buffers()):
            if tensor.is_floating_point():
                self.assertEqual(tensor.dtype, torch.float32)
        self.assert_outputs(model(dict(data)), restored(dict(data)))

    def test_legacy_conversion_preserves_joint_model(self):
        for metal in (False, True):
            with self.subTest(metal=metal):
                model, data = self.case(metal=metal)
                with tempfile.TemporaryDirectory() as directory:
                    source = Path(directory) / "old.pt"
                    destination = Path(directory) / "converted.pt"
                    torch.save(model, source)
                    original = source.read_bytes()
                    convert_legacy_model(source, destination, verification_data=data)
                    self.assertEqual(source.read_bytes(), original)
                    restored = load_model(destination)
                self.assert_outputs(model(dict(data)), restored(dict(data)))

    def test_verification_checks_polarization_and_requires_metal_geometry(self):
        model, data = self.case(metal=True)
        with self.assertRaisesRegex(ValueError, "verification_data"):
            legacy_model_to_current(model)
        altered = copy.deepcopy(model)
        # A derivative-only defect must not escape energy/density checking.
        def wrong_derivative(module, args, output):
            output["polarization_derivative"] = output["polarization_derivative"] + .01
            return output
        handle = altered.register_forward_hook(wrong_derivative)
        with self.assertRaisesRegex(ValueError, "polarization_derivative"):
            verify_equivalent(model, altered, data)
        handle.remove()

    def test_serialized_model_preserves_coupled_and_conditional_solves(self):
        model, data = self.case(metal=True, messages=0)
        data = dict(data, V_ext=torch.zeros_like(data["rho"]),
                    E_ext=torch.zeros_like(data["dipole_density"]),
                    beta=1 / data["temperature"], grid_spacing=torch.full((3,), .7))
        model.requires_grad_(False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_model(model, path)
            restored = load_model(path).requires_grad_(False)
        options = dict(initial_rho=data["rho"], initial_polarization=data["dipole_density"],
                       particle_numbers=data["rho"].sum(0) * .7**3,
                       maximum_density=1., max_iter=3, tolerance_residual=1.e-10)
        for method, fixed in (("euler", None), ("minimize", None),
                              ("minimize", "rho"), ("minimize", "dipole_density")):
            with self.subTest(method=method, fixed=fixed):
                results = [GridSolver(m, dipole_magnitude=1.).solve(
                    data, method=method, fixed_field=fixed, **options,
                ) for m in (model, restored)]
                for key in ("rho", "dipole_density", "beta_A"):
                    torch.testing.assert_close(results[0][key], results[1][key], atol=0, rtol=0)
                electrodes = [m(dict(data, rho=r["rho"], dipole_density=r["dipole_density"]))
                              for m, r in zip((model, restored), results)]
                torch.testing.assert_close(electrodes[0]["metal_site_q"],
                                           electrodes[1]["metal_site_q"], atol=0, rtol=0)
                for key in ("n_iter", "n_evaluations", "status", "objective_history"):
                    self.assertEqual(results[0][key], results[1][key])


if __name__ == "__main__":
    unittest.main()
