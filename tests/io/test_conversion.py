"""Explicit model conversion is installed, source-preserving and opt-in."""

import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from equicdft import FixedDipoleIdeal, GridCACEModel, LDAReadout, load_model, save_model
from equicdft.conversion import convert_model, load_converted_model
from equicdft.serialization import model_payload
from tests.model import test_ideal_reference as fixtures


class TestExplicitConversion(unittest.TestCase):
    def test_cli_converts_versioned_file_outside_repository(self):
        fixture = fixtures.TestModelIdealReference()
        model = fixture.model(False).eval()
        data = fixture.data(False)
        payload = model_payload(model)
        payload["format_version"] = 3
        payload["config"].update(compute_local_mu=True, rho_min=.001)
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory)/"old.pt", Path(directory)/"new.pt"
            torch.save(payload, source)
            original = source.read_bytes()
            with self.assertRaisesRegex(ValueError, "convert explicitly"):
                load_model(source)
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]/"src")
            result = subprocess.run(
                [sys.executable, "-m", "equicdft.conversion", str(source), str(target)],
                cwd=directory, env=environment, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(source.read_bytes(), original)
            expected = model(copy.deepcopy(data))
            actual = load_model(target)(copy.deepcopy(data))
            for key in ("beta_F", "c1", "density_derivative"):
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_old_polarized_file_requires_metadata_then_loads_normally(self):
        fixture = fixtures.TestModelIdealReference()
        model = GridCACEModel(
            None, None, [LDAReadout(n_types=2, include_polarization=True)],
            ideal=FixedDipoleIdeal([.7, 1.9]), n_types=2,
            polarization_scale=.3, grid_spacing=1., boltzmann_constant=1., mean_density=.3,
            compute_polarization_derivative=True,
        ).eval()
        data = fixture.data(True)
        payload = model_payload(model)
        payload["format_version"] = 2
        payload["config"].pop("ideal")
        payload["config"]["thermal_wavelength"] = 1.
        payload["state_dict"] = {
            k: v for k, v in payload["state_dict"].items() if not k.startswith("ideal.")
        }
        payload["state_dict"]["thermal_wavelength"] = torch.ones(2)
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory)/"old.pt", Path(directory)/"new.pt"
            torch.save(payload, source)
            original = source.read_bytes()
            with self.assertRaisesRegex(ValueError, "molecular dipole magnitude"):
                convert_model(source, target)
            self.assertFalse(target.exists())
            ideal = copy.deepcopy(model.ideal)
            self.assertIsInstance(ideal, FixedDipoleIdeal)
            convert_model(source, target, ideal=ideal, verification_data=data)
            self.assertEqual(source.read_bytes(), original)
            actual = load_model(target)(copy.deepcopy(data))
            expected = model(copy.deepcopy(data))
            for key in ("beta_F", "density_derivative", "polarization_derivative"):
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_existing_output_and_source_aliases_are_protected(self):
        model = fixtures.TestModelIdealReference().model(False)
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory)/"model.pt", Path(directory)/"target.pt"
            save_model(model, source)
            save_model(model, target)
            original = source.read_bytes()
            with self.assertRaises(FileExistsError):
                convert_model(source, target)
            alias = Path(directory)/"alias.pt"
            alias.symlink_to(source)
            hardlink = Path(directory)/"hardlink.pt"
            os.link(source, hardlink)
            for path in (source, alias, hardlink):
                with self.assertRaisesRegex(ValueError, "differ from the source"):
                    convert_model(source, path, overwrite=True)
            convert_model(source, target, overwrite=True)
            self.assertEqual(source.read_bytes(), original)

    def test_unsupported_versions_and_checkpoint_are_not_models(self):
        payload = model_payload(fixtures.TestModelIdealReference().model(False))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/"invalid.pt"
            for version in (0, 999, True):
                invalid = dict(payload, format_version=version)
                torch.save(invalid, source)
                with self.assertRaises(ValueError):
                    load_converted_model(source)
            torch.save({"checkpoint_version": 3, "model_state_dict": {}}, source)
            with self.assertRaisesRegex(ValueError, "not an equicdft-model"):
                load_converted_model(source)


if __name__ == "__main__":
    unittest.main()
