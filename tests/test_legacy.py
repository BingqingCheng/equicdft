import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from equicdft import (
    CartesianAFeatures,
    GridCACEModel,
    LDAReadout,
    LocalReadout,
    load_model,
    read_model_config,
)
from equicdft import convert as convert_cli
from equicdft.legacy import (
    _CLASS_ALIASES,
    _MODULE_ALIASES,
    _RemappingUnpickler,
    convert_legacy_model,
    legacy_model_to_current,
    load_legacy_model,
    upgrade_legacy_model,
    verify_equivalent,
)

from tests.test_config import (
    _bessel_message_model,
    _example_model,
    _gaussian_message_model,
    _grid_data,
    _long_range_model,
)


REPOSITORY = Path(__file__).resolve().parents[1]
FROZEN_LEGACY_MODEL = (
    REPOSITORY / "examples" / "lj_paper_v1_regression" / "model.pt"
)


def _whole_object_file(model, directory):
    path = Path(directory) / "legacy.pt"
    torch.save(model, str(path))
    return path


def _pop_child(module, name):
    """Remove a submodule slot whether it holds a module or ``None``."""

    if name in module._modules:
        return module._modules.pop(name)
    return module.__dict__.pop(name)


def _strip_if(module, name, legacy_default):
    """Delete ``name`` when its value equals the historical default.

    An object saved before an attribute existed could only have behaved like
    that attribute's later default, so only such attributes are removable
    without changing the model that the file describes.
    """

    if module.__dict__.get(name) == legacy_default:
        del module.__dict__[name]


def _strip_to_legacy_layout(model):
    """Remove attributes that later package versions introduced.

    The result mimics ``torch.save(model)`` objects written before density
    transforms, radial transforms, message passing, execution backends, the
    free-energy mode flag, and the recorded MLP widths existed.
    """

    features = model.a_features
    _strip_if(features, "convolution_backend", "gather")
    _strip_if(features, "separate_center", False)
    _strip_if(features, "coordinate_scaling", "none")
    _strip_if(features, "trainable_density_transform", True)
    del features.n_radial_functions
    del features._buffers["neighbor_mask"]
    features._non_persistent_buffers_set.discard("neighbor_mask")
    centers = features._buffers.get("fixed_radial_centers")
    if centers is not None and not torch.any(centers != 0.0):
        # Zero-centered Gaussians predate the centers buffer.
        del features._buffers["fixed_radial_centers"]
        features._non_persistent_buffers_set.discard("fixed_radial_centers")
    if _pop_child(features, "radial_transform") is not None:
        raise AssertionError("strip only models without a radial transform")
    # density_transform was called channel_mixing, and older objects stored
    # its absence as a plain attribute.
    mixing = _pop_child(features, "density_transform")
    if mixing is None:
        features.__dict__["channel_mixing"] = None
    else:
        features._modules["channel_mixing"] = mixing
    if features.radial_basis == "none":
        # The undamped basis predates explicit radial attributes entirely.
        del features.radial_basis
        del features.trainable_radial_exponents
        del features.trainable_radial_centers
        del features._buffers["fixed_radial_exponents"]

    if len(model.message_layers) == 0:
        del model._modules["message_layers"]
    else:
        for message in model.message_layers:
            message.independent_radial_basis = (
                message.radial_basis != "shared"
            )
            del message.radial_basis
            _strip_if(message, "convolution_backend", "gather")
            del message.hidden_sizes
    _strip_if(model, "free_energy_mode", "beta")
    for readout in model.readout:
        del readout.hidden_sizes
        if "zero_init" in readout.__dict__:
            _strip_if(
                readout,
                "zero_init",
                type(readout).__init__.__defaults__ and
                __import__("inspect").signature(type(readout).__init__)
                .parameters["zero_init"].default,
            )
    return model


class TestRemappingUnpickler(unittest.TestCase):
    def test_module_and_class_aliases_resolve_to_current_classes(self):
        unpickler = _RemappingUnpickler(io.BytesIO(b""))
        _MODULE_ALIASES["equicdft.old_features"] = "equicdft.features"
        _CLASS_ALIASES[("equicdft.readout", "OldLocalReadout")] = (
            "equicdft.readout",
            "LocalReadout",
        )
        try:
            self.assertIs(
                unpickler.find_class(
                    "equicdft.old_features",
                    "CartesianAFeatures",
                ),
                CartesianAFeatures,
            )
            self.assertIs(
                unpickler.find_class("equicdft.readout", "OldLocalReadout"),
                LocalReadout,
            )
        finally:
            del _MODULE_ALIASES["equicdft.old_features"]
            del _CLASS_ALIASES[("equicdft.readout", "OldLocalReadout")]
        self.assertIs(
            unpickler.find_class("equicdft.model", "GridCACEModel"),
            GridCACEModel,
        )


class TestLegacyConversion(unittest.TestCase):
    def assert_converted_matches(self, model, data, legacy_file):
        expected = model(copy.deepcopy(data))
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "converted.pt"
            written = convert_legacy_model(legacy_file, destination)
            self.assertEqual(written, destination)
            restored = load_model(destination)
        self.assertEqual(restored.to_config(), model.to_config())
        actual = restored(copy.deepcopy(data))
        for key in ("beta_F_exc", "c1"):
            self.assertTrue(
                torch.equal(actual[key].detach(), expected[key].detach()),
                key,
            )
        return restored

    def test_current_whole_object_file_is_converted(self):
        torch.manual_seed(21)
        model = _example_model()
        data = _grid_data(cutoff_grid=1)
        model(copy.deepcopy(data))
        model.eval()
        with tempfile.TemporaryDirectory() as directory:
            legacy_file = _whole_object_file(model, directory)
            self.assert_converted_matches(model, data, legacy_file)

    def test_pre_transform_attribute_layout_is_upgraded(self):
        for separate_center in (True, False):
            with self.subTest(separate_center=separate_center):
                torch.manual_seed(22)
                model = _example_model(separate_center=separate_center)
                data = _grid_data(cutoff_grid=1)
                model(copy.deepcopy(data))
                model.eval()
                reference = copy.deepcopy(model)
                legacy = _strip_to_legacy_layout(model)
                self.assertNotIn("neighbor_mask", legacy.a_features._buffers)
                self.assertNotIn("message_layers", legacy._modules)
                self.assertNotIn("radial_basis", legacy.a_features.__dict__)
                # Only the historical default can be absent from a file.
                self.assertEqual(
                    "separate_center" in legacy.a_features.__dict__,
                    separate_center,
                )
                with tempfile.TemporaryDirectory() as directory:
                    legacy_file = _whole_object_file(legacy, directory)
                    restored = self.assert_converted_matches(
                        reference,
                        data,
                        legacy_file,
                    )
                self.assertEqual(
                    restored.a_features.separate_center,
                    separate_center,
                )

    def test_gaussian_layout_with_channel_mixing_module_is_upgraded(self):
        torch.manual_seed(23)
        model = _gaussian_message_model("gaussian", radial_transform=False)
        data = _grid_data(cutoff_grid=1, n_types=2, grid_spacing=0.5)
        model(copy.deepcopy(data))
        model.eval()
        reference = copy.deepcopy(model)
        legacy = _strip_to_legacy_layout(model)
        self.assertIn("channel_mixing", legacy.a_features._modules)
        self.assertNotIn("radial_basis", legacy.message_layers[0].__dict__)
        self.assertTrue(legacy.message_layers[0].independent_radial_basis)
        with tempfile.TemporaryDirectory() as directory:
            legacy_file = _whole_object_file(legacy, directory)
            restored = self.assert_converted_matches(
                reference,
                data,
                legacy_file,
            )
        self.assertEqual(
            restored.a_features.density_transform.weight.tolist(),
            reference.a_features.density_transform.weight.tolist(),
        )
        self.assertEqual(restored.message_layers[0].radial_basis, "gaussian")

    def test_bessel_and_long_range_models_convert(self):
        torch.manual_seed(24)
        cases = (
            (_bessel_message_model(), _grid_data(shape=(7, 7, 7), cutoff_grid=2)),
            (
                _long_range_model((1.0, -1.0), None),
                _grid_data(shape=(6, 6, 6), n_types=2, grid_spacing=0.5),
            ),
        )
        for model, data in cases:
            with self.subTest(model=type(model.readout[-1]).__name__):
                model(copy.deepcopy(data))
                model.eval()
                with tempfile.TemporaryDirectory() as directory:
                    legacy_file = _whole_object_file(model, directory)
                    self.assert_converted_matches(model, data, legacy_file)

    def test_upgrade_is_idempotent_on_current_objects(self):
        model = _example_model()
        model(copy.deepcopy(_grid_data()))
        before = {key: value.clone() for key, value in model.state_dict().items()}
        upgrade_legacy_model(model)
        after = model.state_dict()
        self.assertEqual(set(before), set(after))
        for key in before:
            self.assertTrue(torch.equal(before[key], after[key]))

    def test_conversion_rejects_changed_state(self):
        model = _example_model()
        model(copy.deepcopy(_grid_data()))
        model.eval()
        broken = copy.deepcopy(model)
        # Corrupt a deterministic buffer so it no longer matches the value
        # the constructor recomputes.
        broken.a_features._buffers["neighbor_mask"] = ~broken.a_features.neighbor_mask
        broken.a_features._non_persistent_buffers_set.discard("neighbor_mask")
        with self.assertRaisesRegex(ValueError, "no deterministic counterpart"):
            legacy_model_to_current(broken, verify=False)

    def test_verify_equivalent_detects_differences(self):
        torch.manual_seed(25)
        model = _example_model()
        model(copy.deepcopy(_grid_data()))
        model.eval()
        other = copy.deepcopy(model)
        with torch.no_grad():
            other.readout[0].mlp[0].weight.add_(0.5)
        verify_equivalent(model, model)
        with self.assertRaisesRegex(ValueError, "differs from the legacy"):
            verify_equivalent(model, other)

    def test_load_legacy_rejects_non_model_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            torch.save({"a": torch.ones(1)}, str(path))
            with self.assertRaisesRegex(TypeError, "pickled model object"):
                load_legacy_model(path)
            with self.assertRaises(FileNotFoundError):
                load_legacy_model(Path(directory) / "absent.pt")

    def test_destination_is_protected_unless_overwrite(self):
        model = _example_model()
        model(copy.deepcopy(_grid_data()))
        with tempfile.TemporaryDirectory() as directory:
            legacy_file = _whole_object_file(model, directory)
            destination = Path(directory) / "taken.pt"
            destination.write_bytes(b"occupied")
            with self.assertRaises(FileExistsError):
                convert_legacy_model(legacy_file, destination)
            convert_legacy_model(legacy_file, destination, overwrite=True)
            load_model(destination)

    def test_in_place_conversion_replaces_the_legacy_file(self):
        model = _example_model()
        model(copy.deepcopy(_grid_data()))
        with tempfile.TemporaryDirectory() as directory:
            legacy_file = _whole_object_file(model, directory)
            convert_legacy_model(legacy_file)
            self.assertEqual(read_model_config(legacy_file)["type"], "GridCACEModel")

    def test_command_line_conversion(self):
        model = _example_model()
        model(copy.deepcopy(_grid_data()))
        with tempfile.TemporaryDirectory() as directory:
            legacy_file = _whole_object_file(model, directory)
            destination = Path(directory) / "cli.pt"
            output = io.StringIO()
            from contextlib import redirect_stdout

            with redirect_stdout(output):
                status = convert_cli.main(
                    [str(legacy_file), str(destination), "--show-config"]
                )
            self.assertEqual(status, 0)
            self.assertIn("wrote", output.getvalue())
            printed = output.getvalue().split("\n", 1)[1]
            self.assertEqual(json.loads(printed)["type"], "GridCACEModel")
            load_model(destination)


@unittest.skipUnless(
    FROZEN_LEGACY_MODEL.is_file(),
    "frozen LJ-paper-v1 whole-object model is not present",
)
class TestFrozenProductionModel(unittest.TestCase):
    def test_published_whole_object_model_converts_and_verifies(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "lj_paper_v1.pt"
            try:
                convert_legacy_model(FROZEN_LEGACY_MODEL, destination)
            except ValueError as error:
                if "not an equicdft-model" in str(error) or "pickled" in str(error):
                    self.skipTest("fixture is no longer a whole-object file")
                raise
            model = load_model(destination)
        config = model.to_config()
        self.assertEqual(config["a_features"]["radial_basis"], "none")
        self.assertEqual(
            [item["type"] for item in config["readout"]],
            ["LDAReadout", "LocalReadout"],
        )
        self.assertEqual(config["free_energy_mode"], "beta")
        self.assertFalse(model.training)


if __name__ == "__main__":
    unittest.main()
