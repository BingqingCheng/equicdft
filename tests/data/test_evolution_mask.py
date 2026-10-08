"""Frozen voxel masks survive data loading without changing density values."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from ase.io import read, write

from equicdft import GridData
from tests.data.test_data import _write_grid


class TestEvolutionMaskData(unittest.TestCase):
    def test_from_dict_binary_mask_and_all_frozen(self):
        values = dict(grid_size=(2, 2, 2), grid_spacing=.5, n_types=1,
                      temperature=1., rho=np.ones(8), mask=np.ones(8, dtype=int))
        data = GridData.from_dict(values, cutoff_grid=0)
        self.assertTrue(torch.all(data["mask"]))
        self.assertTrue(torch.all(data["rho"] == 1.))
        self.assertFalse(torch.any(data["excluded_mask"]))
        del values["mask"]
        self.assertNotIn("mask", GridData.from_dict(values, cutoff_grid=0))

    def test_extxyz_rows_custom_key_and_density_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"field.xyz"
            _write_grid(path)
            original = GridData.from_xyz(path, cutoff_grid=0)[0]
            atoms = read(path)
            atoms.arrays["frozen"] = (atoms.positions[:, 0] == 1).astype(int)
            write(path, atoms, format="extxyz")
            data = GridData.from_xyz(path, cutoff_grid=0, data_key={"mask": "frozen"})[0]
            self.assertTrue(torch.equal(data["mask"], data["grid_positions"][:, 0] == 1))
            self.assertTrue(torch.equal(data["rho"], original["rho"]))

    def test_coarsening_uniform_masks_and_rejecting_mixed_blocks(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"field.xyz"
            _write_grid(path)
            atoms = read(path)
            atoms.arrays["mask"] = atoms.positions[:, 0] < 2
            write(path, atoms, format="extxyz")
            data = GridData.from_xyz(path, cutoff_grid=0, target_grid_spacing=1.)[0]
            self.assertTrue(torch.equal(data["mask"], data["grid_positions"][:, 0] == 0))
            atoms.arrays["mask"] = atoms.positions[:, 0] == 0
            write(path, atoms, format="extxyz")
            with self.assertRaisesRegex(ValueError, "partially frozen"):
                GridData.from_xyz(path, cutoff_grid=0, target_grid_spacing=1.)

    def test_invalid_mask_values_and_shape(self):
        values = dict(grid_size=(2, 2, 2), grid_spacing=.5, n_types=1, temperature=1.)
        for mask in (np.full(8, .5), np.full(8, np.nan), np.ones(7)):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                GridData.from_dict(dict(values, mask=mask), cutoff_grid=0)


if __name__ == "__main__":
    unittest.main()
