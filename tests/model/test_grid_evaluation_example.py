"""Small end-to-end checks for the public grid-transfer evaluation example."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from ase import Atoms
from ase.io import write

from examples.grid_transfer.evaluate import main
from equicdft import GridData, save_model
from equicdft.serialization import default_dtype
from tests.model.test_regrid import _model


class TestGridEvaluationExample(unittest.TestCase):
    def test_native_and_fine_grid_for_two_species_and_two_radial_channels(self):
        old_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, old_threads)
        with tempfile.TemporaryDirectory() as directory, default_dtype(torch.float64):
            directory = Path(directory)
            model_path = directory / "model.pt"
            model = _model().requires_grad_(False)
            save_model(model, model_path)
            for spacing, size in ((0.5, 4), (0.25, 8)):
                positions = np.indices((size,) * 3).reshape(3, -1).T
                density = np.column_stack((
                    0.2 + 0.01 * np.sin(2 * np.pi * positions[:, 0] / size),
                    0.3 + 0.02 * np.cos(2 * np.pi * positions[:, 1] / size),
                ))
                atoms = Atoms("H" * len(positions), positions=positions)
                atoms.info.update(T=1.2, grid_spacing=spacing, grid_size=[size] * 3)
                atoms.new_array("density", density)
                field_path = directory / ("field-{}.extxyz".format(size))
                write(field_path, atoms, format="extxyz")
                for policy in ("strict", "reference"):
                    with self.subTest(spacing=spacing, policy=policy):
                        output_path = directory / ("{}-{}.pt".format(size, policy))
                        argv = ["evaluate.py", str(model_path), str(field_path),
                                "--spacing", str(spacing), "--center-normalization", policy,
                                "--output", str(output_path)]
                        with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                            main()
                        saved = torch.load(output_path, weights_only=True)
                        transferred = model.regrid(spacing, center_normalization=policy,
                                                   convolution_backend="fft")
                        field = GridData.from_xyz(field_path, index="0", grid_info=transferred.grid_info,
                                                  include_local_density_index=False)[0]
                        expected = transferred(field, compute_c1=True, compute_c2=False,
                                               compute_local_mu=False, compute_ideal=False)
                        for key in ("beta_F_exc", "c1"):
                            torch.testing.assert_close(saved[key], expected[key], rtol=0, atol=0)
                        self.assertEqual(tuple(saved["c1"].shape), (size ** 3, 2))
                        self.assertEqual(saved["center_normalization"], policy)
                        before = output_path.read_bytes()
                        with patch("sys.argv", argv), self.assertRaises(FileExistsError):
                            main()
                        self.assertEqual(output_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
