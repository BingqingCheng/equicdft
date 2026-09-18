import copy
import json
import unittest

import torch

from equicdft import (
    BChiMessage,
    BulkReadout,
    CartesianAFeatures,
    CartesianBFeatures,
    GGAReadout,
    GridCACEModel,
    LDAReadout,
    LocalReadout,
    LongRangeReadout,
    PairwiseReadout,
    ReciprocalFeatures,
)
from equicdft._config import (
    Configurable,
    build,
    plain_value,
    register,
    registered_types,
    resolve,
)
from tests.model_fixtures import (
    bessel_message_model,
    example_model,
    gaussian_message_model,
    grid_data,
    long_range_model,
)


class TestConfigRegistry(unittest.TestCase):
    def test_all_model_building_blocks_are_registered(self):
        registered = registered_types()
        for cls in (
            CartesianAFeatures,
            CartesianBFeatures,
            BChiMessage,
            LocalReadout,
            BulkReadout,
            LongRangeReadout,
            ReciprocalFeatures,
            LDAReadout,
            GGAReadout,
            PairwiseReadout,
            GridCACEModel,
        ):
            self.assertIs(registered[cls.__name__], cls)
            self.assertIs(resolve(cls.__name__), cls)

    def test_unknown_type_is_rejected(self):
        with self.assertRaisesRegex(KeyError, "unknown configurable type"):
            build({"type": "NoSuchModule"})
        with self.assertRaisesRegex(ValueError, "requires a 'type' entry"):
            build({"max_power": 1})

    def test_type_tag_must_match_class(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            CartesianBFeatures.from_config(
                {"type": "CartesianAFeatures", "max_power": 1}
            )

    def test_register_rejects_non_configurable_and_duplicates(self):
        with self.assertRaisesRegex(TypeError, "Configurable"):
            register(torch.nn.Linear)

        class CartesianBFeatures(torch.nn.Module, Configurable):  # noqa
            pass

        with self.assertRaisesRegex(ValueError, "already registered"):
            register(CartesianBFeatures)

    def test_plain_value_converts_tensors_and_rejects_objects(self):
        self.assertEqual(plain_value(torch.tensor(2.5)), 2.5)
        self.assertEqual(plain_value(torch.tensor([1, 2])), [1, 2])
        self.assertEqual(plain_value((1, (2, 3))), [1, [2, 3]])
        self.assertEqual(plain_value({"a": torch.ones(1)}), {"a": [1.0]})
        with self.assertRaisesRegex(TypeError, "JSON-compatible"):
            plain_value(object())


class TestConfigRoundTrip(unittest.TestCase):
    def assert_round_trip(self, model, data, keys=("beta_F_exc", "c1")):
        model.eval()
        config = model.to_config()
        # Configurations are plain data.
        self.assertEqual(json.loads(json.dumps(config)), config)

        rebuilt = build(config)
        self.assertIsInstance(rebuilt, type(model))
        self.assertEqual(
            {key: tuple(value.shape) for key, value in model.state_dict().items()},
            {key: tuple(value.shape) for key, value in rebuilt.state_dict().items()},
        )
        self.assertEqual(rebuilt.to_config(), config)

        rebuilt.load_state_dict(model.state_dict(), strict=True)
        rebuilt.eval()
        expected = model(copy.deepcopy(data))
        actual = rebuilt(copy.deepcopy(data))
        for key in keys:
            self.assertTrue(
                torch.equal(actual[key].detach(), expected[key].detach()),
                key,
            )
        return config

    def testexample_model(self):
        model = example_model()
        data = grid_data(cutoff_grid=1)
        # Materialize the lazy local readout before describing the model.
        model(copy.deepcopy(data))
        config = self.assert_round_trip(
            model,
            data,
            keys=("beta_F_exc", "c1", "local_chemical_potential"),
        )
        self.assertEqual(config["type"], "GridCACEModel")
        self.assertEqual(config["a_features"]["radial_basis"], "none")
        self.assertIsNone(config["a_features"]["radial_exponents"])
        self.assertIsNone(config["a_features"]["n_radial_channels"])
        self.assertEqual(
            [item["type"] for item in config["readout"]],
            ["LDAReadout", "LocalReadout"],
        )
        self.assertEqual(config["message_layers"], [])
        self.assertIsInstance(config["readout"][1]["n_features"], int)

    def test_lazy_readout_reports_unknown_width_before_first_forward(self):
        readout = LocalReadout(n_types=1, hidden_sizes=(4,))
        self.assertIsNone(readout.to_config()["n_features"])
        readout(torch.zeros(3, 7))
        self.assertEqual(readout.to_config()["n_features"], 7)
        rebuilt = LocalReadout.from_config(readout.to_config())
        rebuilt.load_state_dict(readout.state_dict(), strict=True)

    def test_gaussian_message_model_with_density_transform(self):
        torch.manual_seed(1)
        model = gaussian_message_model("gaussian")
        data = grid_data(cutoff_grid=1, n_types=2, grid_spacing=0.5)
        model(copy.deepcopy(data))
        config = self.assert_round_trip(
            model,
            data,
            keys=("beta_F_exc", "F_exc", "c1"),
        )
        features = config["a_features"]
        self.assertEqual(features["radial_basis"], "gaussian")
        self.assertEqual(len(features["radial_exponents"]), 2)
        self.assertEqual(features["radial_centers"], [0.0, 1.0])
        self.assertEqual(features["n_radial_channels"], 2)
        self.assertEqual(
            features["density_transform"],
            [[0.5, 0.5], [-0.5, 0.5]],
        )
        message = config["message_layers"][0]
        self.assertEqual(message["radial_basis"], "gaussian")
        self.assertIsNone(message["radial_centers"])
        self.assertTrue(message["trainable_radial_centers"])
        self.assertEqual(config["free_energy_mode"], "physical")

    def test_trained_values_are_restored_from_state_not_config(self):
        torch.manual_seed(2)
        model = gaussian_message_model("gaussian")
        data = grid_data(cutoff_grid=1, n_types=2, grid_spacing=0.5)
        model(copy.deepcopy(data))
        with torch.no_grad():
            model.a_features.log_radial_exponents.add_(0.3)
            model.message_layers[0].learned_radial_centers.fill_(-0.4)
        # Negative trained centers cannot be constructor inputs, so the
        # configuration must not carry them.
        self.assert_round_trip(model, data)

    def test_shared_message_and_fft_backend(self):
        torch.manual_seed(3)
        model = gaussian_message_model("shared", backend="fft")
        data = grid_data(cutoff_grid=1, n_types=2, grid_spacing=0.5)
        model(copy.deepcopy(data))
        config = self.assert_round_trip(model, data)
        self.assertEqual(config["a_features"]["convolution_backend"], "fft")
        self.assertEqual(config["message_layers"][0]["radial_basis"], "shared")
        self.assertEqual(
            config["message_layers"][0]["convolution_backend"],
            "fft",
        )
        self.assertFalse(build(config).requires_local_density_index)

    def test_bessel_message_model_rebinds_basis(self):
        torch.manual_seed(4)
        model = bessel_message_model()
        data = grid_data(shape=(7, 7, 7), cutoff_grid=2)
        model(copy.deepcopy(data))
        config = self.assert_round_trip(model, data)
        self.assertEqual(config["a_features"]["n_radial_functions"], 3)
        self.assertEqual(config["message_layers"][0]["n_radial_functions"], 3)
        self.assertEqual(config["readout"][1]["type"], "GGAReadout")
        self.assertIsInstance(config["readout"][1]["n_features"], int)

    def test_long_range_and_pairwise_readouts(self):
        torch.manual_seed(5)
        for charges, amplitude in (
            (None, None),
            ((1.0, -1.0), None),
            ((1.0, -1.0), 2.5),
        ):
            with self.subTest(charges=charges, amplitude=amplitude):
                model = long_range_model(charges, amplitude)
                data = grid_data(
                    shape=(6, 6, 6),
                    n_types=2,
                    grid_spacing=0.5,
                )
                config = self.assert_round_trip(model, data)
                self.assertIsNone(config["a_features"])
                long_range = config["readout"][3]
                self.assertEqual(long_range["type"], "LongRangeReadout")
                self.assertEqual(
                    long_range["features"]["type"],
                    "ReciprocalFeatures",
                )
                self.assertEqual(long_range["features"]["kernel"], "coulomb")
                self.assertEqual(
                    long_range["charges"],
                    None if charges is None else list(charges),
                )
                self.assertEqual(long_range["coulomb_amplitude"], amplitude)

    def test_configuration_can_be_edited_before_rebuilding(self):
        model = example_model()
        model(copy.deepcopy(grid_data()))
        config = model.to_config()
        config["compute_local_mu"] = False
        config["rho_min"] = 0.0
        rebuilt = GridCACEModel.from_config(config)
        self.assertFalse(rebuilt.compute_local_mu)
        self.assertEqual(rebuilt.rho_min, 0.0)
        rebuilt.load_state_dict(model.state_dict(), strict=True)


if __name__ == "__main__":
    unittest.main()
