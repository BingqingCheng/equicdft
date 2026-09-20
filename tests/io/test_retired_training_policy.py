"""Old model files lose fitting policy, not their learned functional."""

import unittest
import torch
from equicdft import GridCACEModel
from equicdft.conversion.legacy import upgrade_legacy_model
from equicdft.serialization import model_payload
from equicdft.conversion import build_converted_model, migrate_model_payload
from tests.model_fixtures import example_model, grid_data


class TestRetiredTrainingPolicy(unittest.TestCase):
    def test_version_three_migration_discards_only_policy(self):
        model = example_model().eval()
        payload = model_payload(model)
        payload["format_version"] = 3
        payload["config"].update(compute_local_mu=True, rho_min=.123)
        restored = build_converted_model(payload).eval()
        self.assertIn("rho_min", payload["config"])
        migrated = migrate_model_payload(payload)
        self.assertEqual(migrated["config"]["rho_min"], 0.)
        self.assertFalse(migrated["config"]["compute_local_mu"])
        for key, expected in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], expected, rtol=0, atol=0)
        data = grid_data()
        expected, actual = model(dict(data)), restored(dict(data))
        self.assertEqual(set(expected), set(actual))
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_whole_object_upgrade_resets_historical_fitting_policy(self):
        model = example_model()
        model.compute_local_mu = True
        model.rho_min = .123
        upgraded = upgrade_legacy_model(model)
        self.assertFalse(upgraded.compute_local_mu)
        self.assertEqual(upgraded.rho_min, 0.)

    def test_new_constructor_accepts_explicit_diagnostics(self):
        model = GridCACEModel(None, None, [], n_types=1, grid_spacing=1.,
                              compute_local_mu=True, rho_min=.1)
        self.assertTrue(model.compute_local_mu)
        self.assertEqual(model.rho_min, .1)


if __name__ == "__main__":
    unittest.main()
