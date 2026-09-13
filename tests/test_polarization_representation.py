"""The split A/B path preserves descriptors, derivatives and legacy models."""

import io
import itertools
import unittest

import torch

from equicdft import (
    BChiMessage, CartesianAFeatures, GridCACEModel, LocalReadout,
    PolarizationAFeatures, PolarizationBFeatures,
    PolarizationFeatures, PolarizationReadout,
)
from test_polarization_features import periodic_field
import test_polarization_model as fixtures


def clone(data):
    return {key: value.detach().clone() for key, value in data.items()}


class TestPolarizationRepresentation(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(37)

    def tearDown(self):
        torch.set_default_dtype(self.dtype)

    def pair(self, power=1, order=2, n_types=1, reversal=False):
        args = dict(mean_density=0.5, dipole_density_scale=0.3,
                    cutoff_grid=1, max_power=power, n_types=n_types,
                    radial_exponents=(0.125, 0.5))
        old = PolarizationFeatures(
            **args, max_product_order=order, dipole_reversal_symmetry=reversal
        )
        a = PolarizationAFeatures(**args)
        b = PolarizationBFeatures(a, order, reversal)
        return old, a, b

    def models(self, mode="beta"):
        old, a, b = self.pair()
        legacy_readout = PolarizationReadout(old, hidden_sizes=(5,))
        readout = LocalReadout(n_features=b.n_features + 1, hidden_sizes=(5,))
        readout.mlp.load_state_dict(legacy_readout.mlp.state_dict())
        args = dict(grid_spacing=0.5, mean_temperature=1.5,
                    free_energy_mode=mode, compute_c1=True,
                    compute_polarization_derivative=True)
        legacy = GridCACEModel(None, None, [legacy_readout], **args)
        split = GridCACEModel(a, b, [readout], **args)
        return legacy, split

    def test_all_descriptor_orders_species_radials_and_reversal(self):
        for power, order, n_types, reversal in itertools.product(
                (0, 1, 2), (1, 2, 3), (1, 2), (False, True)):
            with self.subTest(power=power, order=order, n_types=n_types, reversal=reversal):
                old, a, b = self.pair(power, order, n_types, reversal)
                data, _ = periodic_field(size=3, cutoff=1, n_types=n_types)
                self.assertEqual(b.feature_names, old.feature_names)
                self.assertEqual(b.n_features, old.n_features)
                self.assertEqual(list(b.parameters()), [])
                self.assertEqual(list(b.children()), [])
                self.assertIsInstance(a(data), dict)
                torch.testing.assert_close(b(a(data)), old(data), atol=0, rtol=0)
                batch = {key: value.unsqueeze(0).expand(2, *value.shape)
                         for key, value in data.items()}
                torch.testing.assert_close(b(a(batch)), old(batch), atol=0, rtol=0)

    def test_energy_both_field_derivatives_c2_and_parameter_gradients(self):
        for mode, batched in itertools.product(("beta", "physical"), (False, True)):
            with self.subTest(mode=mode, batched=batched):
                old, split = self.models(mode)
                data = fixtures.TestPolarizationModel.data(batched=batched)
                outputs = [model(clone(data), compute_c2=True) for model in (old, split)]
                for key in ("beta_F_exc", "c1", "c2", "polarization_derivative"):
                    torch.testing.assert_close(outputs[0][key], outputs[1][key], atol=0, rtol=0)
                for output in outputs:
                    (output["c1"].square().sum()
                     + output["polarization_derivative"].square().sum()).backward()
                for left, right in zip(old.parameters(), split.parameters()):
                    self.assertEqual(left.grad is None, right.grad is None)
                    if left.grad is not None:
                        torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)

    def test_metadata_from_features_without_polarization_readout(self):
        _, split = self.models()
        self.assertTrue(split.has_local_features)
        self.assertTrue(split.requires_dipole_density)
        self.assertTrue(split.requires_local_density_index)
        self.assertFalse(getattr(split.readout[0], "requires_dipole_density", False))
        self.assertEqual(split.grid_info["cutoff_grid"], 1)
        self.assertEqual(split.n_types, 1)
        with self.assertRaisesRegex(ValueError, "cubic voxels"):
            GridCACEModel(split.a_features, split.b_features, split.readout,
                          grid_spacing=(0.5, 0.5, 1.0))
        with self.assertRaisesRegex(ValueError, "compute_local_mu"):
            GridCACEModel(split.a_features, split.b_features, split.readout,
                          grid_spacing=0.5, compute_local_mu=True)
        data = fixtures.TestPolarizationModel.data()
        del data["dipole_density"]
        with self.assertRaises((ValueError, KeyError)):
            split(data)

    def test_reject_mismatched_layouts_and_unsupported_joint_messages(self):
        old, a, b = self.pair()
        readout = [LocalReadout(n_features=b.n_features + 1)]
        for wrong in (old, CartesianAFeatures(mean_density=0.5, cutoff_grid=1, max_power=1)):
            with self.assertRaisesRegex(TypeError, "requires PolarizationAFeatures"):
                GridCACEModel(wrong, b, readout, grid_spacing=0.5)
        for args in (dict(max_power=2), dict(n_types=2), dict(radial_exponents=(0.125,))):
            options = dict(mean_density=0.5, dipole_density_scale=0.3,
                           cutoff_grid=1, max_power=1, radial_exponents=(0.125, 0.5))
            options.update(args)
            with self.assertRaisesRegex(ValueError, "must match"):
                GridCACEModel(PolarizationAFeatures(**options), b, readout, grid_spacing=0.5)
        with self.assertRaisesRegex(ValueError, "legacy scalar-invariant message"):
            GridCACEModel(a, b, readout, grid_spacing=0.5,
                          message_layers=[BChiMessage(b.n_features, 1, 1)])

    def test_state_dict_mapping_and_full_object_roundtrip(self):
        old, split = self.models()
        mapped = {key.replace("readout.0.features.", "a_features."): value
                  for key, value in old.state_dict().items()}
        split.load_state_dict(mapped, strict=True)
        data = fixtures.TestPolarizationModel.data()
        expected = old(clone(data))
        for model in (old, split):
            stream = io.BytesIO()
            torch.save(model, stream)
            stream.seek(0)
            restored = torch.load(stream, weights_only=False)
            actual = restored(clone(data))
            for key in ("beta_F_exc", "c1", "polarization_derivative"):
                torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
