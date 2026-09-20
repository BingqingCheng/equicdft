"""Model-owned normalization, physical derivatives, and historical conversion."""

import copy
import tempfile
import unittest
from pathlib import Path

import torch

from equicdft import (CartesianAFeatures, CartesianBFeatures, FixedDipoleIdeal,
                     GridCACEModel, LDAReadout, LocalReadout, load_model, save_model)
from equicdft.conversion import build_converted_model, load_converted_model
from equicdft.serialization import model_payload
from tests.representation.test_polarization_features import periodic_field


class TestPolarizationScale(unittest.TestCase):
    def setUp(self):
        self.dtype, self.rng = torch.get_default_dtype(), torch.get_rng_state()
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(471)

    def tearDown(self):
        torch.set_default_dtype(self.dtype)
        torch.set_rng_state(self.rng)

    def model(self, scale=.3):
        return GridCACEModel(
            CartesianAFeatures(1, .7, cutoff_grid=1, include_polarization=True),
            CartesianBFeatures(1, 2, include_polarization=True),
            [LocalReadout(hidden_sizes=(4,)), LDAReadout(include_polarization=True, hidden_sizes=(4,))],
            polarization_scale=scale, ideal=FixedDipoleIdeal(1.),
            grid_spacing=.5, boltzmann_constant=1., compute_polarization_derivative=True,
        ).eval()

    def data(self):
        data, _ = periodic_field()
        data["dipole_density"] *= .01
        return data

    def test_single_fixed_buffer_and_serialization(self):
        model, data = self.model(), self.data()
        self.assertEqual([k for k in model.state_dict() if "polarization_scale" in k], ["polarization_scale"])
        self.assertFalse(any("dipole_density_scale" in k for k in model.state_dict()))
        self.assertNotIn("polarization_scale", dict(model.named_parameters()))
        self.assertFalse(model.polarization_scale.requires_grad)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"model.pt"
            save_model(model, path)
            restored = load_model(path)
        for key, expected in model(copy.deepcopy(data)).items():
            torch.testing.assert_close(restored(copy.deepcopy(data))[key], expected, rtol=0, atol=0)

    def test_scale_only_normalization_and_physical_chain_rule(self):
        model, data = self.model(), self.data()
        # Nonzero uniform polarization must remain visible to the learned model.
        data["dipole_density"][:] = torch.tensor([.03, .02, .01])
        before = model(copy.deepcopy(data), compute_ideal=False)
        changed_model, changed_data = copy.deepcopy(model), copy.deepcopy(data)
        changed_model.polarization_scale *= 2
        changed_data["dipole_density"] *= 2
        after = changed_model(changed_data, compute_ideal=False)
        for key in ("beta_F_exc", "c1"):
            torch.testing.assert_close(after[key], before[key], rtol=0, atol=0)
        torch.testing.assert_close(after["polarization_derivative_excess"] * 2,
                                   before["polarization_derivative_excess"], rtol=0, atol=0)
        zero = dict(data, dipole_density=torch.zeros_like(data["dipole_density"]))
        self.assertFalse(torch.allclose(model(zero)["beta_F_exc"], before["beta_F_exc"]))
        original_ideal = model(copy.deepcopy(data))["beta_F_id"]
        torch.testing.assert_close(changed_model(copy.deepcopy(data))["beta_F_id"], original_ideal, rtol=0, atol=0)

    def test_scale_is_required_only_for_normalized_learned_inputs(self):
        with self.assertRaisesRegex(ValueError, "polarization_scale"):
            self.model(None)
        # Analytic ideal-only polarization has no normalized learned inputs.
        model = GridCACEModel(None, None, [], ideal=FixedDipoleIdeal(1.), n_types=1, grid_spacing=1.)
        self.assertIsNone(model.polarization_scale)
        with self.assertRaisesRegex(ValueError, "polarization_scale"):
            CartesianAFeatures(1, .7, include_polarization=True)(self.data())

    def test_v4_and_whole_object_conversion_preserve_equal_and_different_scales(self):
        for lda_scale in (.3, .6):
            with self.subTest(lda_scale=lda_scale):
                model, data = self.model(), self.data()
                payload = model_payload(model)
                payload["format_version"] = 4
                payload["config"].pop("polarization_scale")
                payload["state_dict"].pop("polarization_scale")
                payload["config"]["a_features"]["dipole_density_scale"] = .3
                payload["state_dict"]["a_features.dipole_density_scale"] = torch.tensor(.3)
                lda = payload["config"]["readout"][1]
                lda.pop("include_polarization")
                lda["dipole_density_scale"] = lda_scale
                payload["state_dict"]["readout.1.dipole_density_scale"] = torch.tensor(lda_scale)
                original_weight = payload["state_dict"]["readout.1.mlp.0.weight"].clone()
                # Independent old energy: local branch uses .3, LDA uses its own scale.
                lda_model = GridCACEModel(None, None, [copy.deepcopy(model.readout[1])],
                                          polarization_scale=lda_scale, mean_density=.7,
                                          grid_spacing=.5, boltzmann_constant=1.,
                                          compute_polarization_derivative=True).eval()
                local_model = copy.deepcopy(model)
                local_model.readout = torch.nn.ModuleList([local_model.readout[0]])
                left = local_model(copy.deepcopy(data), compute_ideal=False)
                right = lda_model(copy.deepcopy(data), compute_ideal=False)
                converted = build_converted_model(payload).eval()
                whole = copy.deepcopy(model)
                del whole.polarization_scale, whole.readout[1].include_polarization
                whole.a_features.register_buffer("dipole_density_scale", torch.tensor(.3))
                whole.readout[1].register_buffer("dipole_density_scale", torch.tensor(lda_scale))
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory)/"old.pt"
                    torch.save(whole, path)
                    from_whole = load_converted_model(path)
                for candidate in (converted, from_whole):
                    output = candidate(copy.deepcopy(data), compute_ideal=False)
                    for key in ("beta_F_exc", "c1", "polarization_derivative_excess"):
                        torch.testing.assert_close(output[key], left[key]+right[key], rtol=1e-12, atol=1e-12)
                    self.assertAlmostEqual(candidate.polarization_scale.item(), .3)
                torch.testing.assert_close(payload["state_dict"]["readout.1.mlp.0.weight"], original_weight, rtol=0, atol=0)
                if lda_scale == .3:
                    torch.testing.assert_close(converted.readout[1].mlp[0].weight, original_weight, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
