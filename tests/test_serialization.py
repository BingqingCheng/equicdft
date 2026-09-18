import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch

from equicdft import (
    MODEL_FORMAT,
    MODEL_FORMAT_VERSION,
    GridCACEModel,
    load_model,
    read_model_config,
    save_model,
)
from equicdft.serialization import (
    build_model,
    default_dtype,
    dtype_from_name,
    dtype_name,
    model_dtype,
    model_payload,
    read_model_payload,
    validate_model_payload,
)

from tests.test_config import (
    _bessel_message_model,
    _example_model,
    _gaussian_message_model,
    _grid_data,
)


def _materialized_example():
    torch.manual_seed(11)
    model = _example_model()
    data = _grid_data(cutoff_grid=1)
    model(copy.deepcopy(data))
    model.eval()
    return model, data


class TestModelPayload(unittest.TestCase):
    def test_payload_contains_only_plain_data_and_cpu_tensors(self):
        model, _ = _materialized_example()
        payload = model_payload(model)

        self.assertEqual(payload["format"], MODEL_FORMAT)
        self.assertEqual(payload["format_version"], MODEL_FORMAT_VERSION)
        self.assertEqual(payload["default_dtype"], "float32")
        self.assertEqual(payload["config"], model.to_config())
        self.assertEqual(
            set(payload["state_dict"]),
            set(model.state_dict()),
        )
        for value in payload["state_dict"].values():
            self.assertEqual(value.device.type, "cpu")
            self.assertFalse(value.requires_grad)
        json.dumps(payload["config"])

    def test_payload_rejects_modules_without_configuration(self):
        with self.assertRaisesRegex(TypeError, "does not provide to_config"):
            model_payload(torch.nn.Linear(2, 2))
        with self.assertRaisesRegex(TypeError, "torch.nn.Module"):
            model_payload(object())

    def test_dtype_names_round_trip(self):
        for dtype in (torch.float32, torch.float64):
            self.assertEqual(dtype_from_name(dtype_name(dtype)), dtype)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            dtype_from_name("int64")
        with self.assertRaisesRegex(ValueError, "unsupported"):
            dtype_name(torch.int64)

    def test_model_dtype_is_read_from_state_not_global_default(self):
        with default_dtype(torch.float64):
            model = _example_model()
            data = {
                key: value.double() if value.is_floating_point() else value
                for key, value in _grid_data(cutoff_grid=1).items()
            }
            model(data)
        self.assertEqual(model_dtype(model), torch.float64)
        self.assertEqual(model_payload(model)["default_dtype"], "float64")
        model.a_features.mean_density = model.a_features.mean_density.float()
        with self.assertRaisesRegex(ValueError, "mixes floating dtypes"):
            model_dtype(model)

    def test_lazy_models_must_see_a_batch_before_saving(self):
        model = _example_model()
        with self.assertRaisesRegex(ValueError, "uninitialized lazy"):
            model_payload(model)

    def test_default_dtype_context_restores_previous_value(self):
        previous = torch.get_default_dtype()
        with default_dtype(torch.float64):
            self.assertEqual(torch.get_default_dtype(), torch.float64)
        self.assertEqual(torch.get_default_dtype(), previous)


class TestSaveLoadModel(unittest.TestCase):
    def test_round_trip_preserves_outputs(self):
        model, data = _materialized_example()
        expected = model(copy.deepcopy(data))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            self.assertEqual(save_model(model, path), path)
            self.assertTrue(path.is_file())

            restored = load_model(path)
            self.assertIsInstance(restored, GridCACEModel)
            self.assertFalse(restored.training)
            self.assertEqual(restored.to_config(), model.to_config())
            self.assertEqual(read_model_config(path), model.to_config())

            actual = restored(copy.deepcopy(data))
            for key in ("beta_F_exc", "c1", "local_chemical_potential"):
                self.assertTrue(
                    torch.equal(actual[key].detach(), expected[key].detach()),
                    key,
                )
            trainable = load_model(path, eval_mode=False)
            self.assertTrue(trainable.training)

    def test_saved_file_loads_with_restricted_unpickler(self):
        model, _ = _materialized_example()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_model(model, path)
            payload = torch.load(str(path), weights_only=True)
        validate_model_payload(payload, path)
        self.assertEqual(payload["config"]["type"], "GridCACEModel")

    def test_map_location_moves_every_tensor(self):
        model, _ = _materialized_example()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_model(model, path)
            restored = load_model(path, map_location="cpu")
        for tensor in list(restored.parameters()) + list(restored.buffers()):
            self.assertEqual(tensor.device.type, "cpu")

    def test_float64_models_are_rebuilt_under_their_dtype(self):
        with default_dtype(torch.float64):
            torch.manual_seed(12)
            model = _bessel_message_model()
            data = _grid_data(shape=(7, 7, 7), cutoff_grid=2)
            model(copy.deepcopy(data))
            model.eval()
            expected = model(copy.deepcopy(data))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_model(model, path)
            self.assertEqual(read_model_payload(path)["default_dtype"], "float64")
            restored = load_model(path)
        self.assertEqual(torch.get_default_dtype(), torch.float32)
        self.assertEqual(restored.a_features.monomial_values.dtype, torch.float64)
        actual = restored(copy.deepcopy(data))
        self.assertTrue(torch.equal(actual["c1"].detach(), expected["c1"].detach()))

    def test_message_model_round_trip(self):
        torch.manual_seed(13)
        model = _gaussian_message_model("gaussian", backend="fft")
        data = _grid_data(cutoff_grid=1, n_types=2, grid_spacing=0.5)
        model(copy.deepcopy(data))
        model.eval()
        expected = model(copy.deepcopy(data))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_model(model, path)
            restored = load_model(path)
        self.assertEqual(restored.a_features.convolution_backend, "fft")
        self.assertFalse(restored.requires_local_density_index)
        actual = restored(copy.deepcopy(data))
        self.assertTrue(
            torch.equal(actual["beta_F_exc"].detach(), expected["beta_F_exc"].detach())
        )

    def test_missing_file_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_model(Path(directory) / "absent.pt")

    def test_whole_object_pickle_is_rejected_with_conversion_hint(self):
        model, _ = _materialized_example()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save(model, str(path))
            with self.assertRaisesRegex(ValueError, "equicdft.convert"):
                load_model(path)

    def test_foreign_dictionaries_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            torch.save({"weight": torch.ones(2)}, str(path))
            with self.assertRaisesRegex(ValueError, "not an equicdft-model file"):
                load_model(path)

    def test_newer_format_versions_are_rejected(self):
        model, _ = _materialized_example()
        payload = model_payload(model)
        payload["format_version"] = MODEL_FORMAT_VERSION + 1
        with self.assertRaisesRegex(ValueError, "supports versions up to"):
            build_model(payload)
        payload["format_version"] = "1"
        with self.assertRaisesRegex(ValueError, "integer format_version"):
            build_model(payload)

    def test_incomplete_payloads_are_rejected(self):
        model, _ = _materialized_example()
        for key in ("config", "state_dict", "default_dtype"):
            payload = model_payload(model)
            del payload[key]
            with self.assertRaisesRegex(ValueError, key):
                build_model(payload)

    def test_state_is_loaded_strictly(self):
        model, _ = _materialized_example()
        payload = model_payload(model)
        del payload["state_dict"]["readout.0.mlp.0.weight"]
        with self.assertRaises(RuntimeError):
            build_model(payload)

    def test_save_is_atomic(self):
        model, _ = _materialized_example()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "model.pt"
            save_model(model, path)
            self.assertEqual(sorted(p.name for p in path.parent.iterdir()), ["model.pt"])

    def test_payload_survives_in_memory_round_trip(self):
        model, data = _materialized_example()
        buffer = io.BytesIO()
        torch.save(model_payload(model), buffer)
        buffer.seek(0)
        payload = torch.load(buffer, weights_only=True)
        restored = build_model(payload).eval()
        self.assertTrue(
            torch.equal(
                restored(copy.deepcopy(data))["c1"].detach(),
                model(copy.deepcopy(data))["c1"].detach(),
            )
        )


if __name__ == "__main__":
    unittest.main()
