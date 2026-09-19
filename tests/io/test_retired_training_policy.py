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
        self.assertNotIn("rho_min", migrated["config"])
        self.assertNotIn("compute_local_mu", migrated["config"])
        for key, expected in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], expected, rtol=0, atol=0)
        data = grid_data()
        expected, actual = model(dict(data)), restored(dict(data))
        self.assertEqual(set(expected), set(actual))
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_whole_object_upgrade_removes_retired_attributes(self):
        model = example_model()
        model.compute_local_mu = True
        model.rho_min = .123
        upgraded = upgrade_legacy_model(model)
        self.assertFalse(hasattr(upgraded, "compute_local_mu"))
        self.assertFalse(hasattr(upgraded, "rho_min"))
        self.assertNotIn("rho_min", upgraded.to_config())

    def test_new_constructor_does_not_accept_training_policy(self):
        for policy in ({"compute_local_mu": True}, {"rho_min": .1}):
            with self.assertRaisesRegex(TypeError, "unexpected keyword"):
                GridCACEModel(None, None, [], n_types=1, grid_spacing=1., **policy)


if __name__ == "__main__":
    unittest.main()
