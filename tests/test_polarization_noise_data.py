"""Uncertainty fields preserve units, species axes and canonical grid order."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io import write
from torch.utils.data import DataLoader

from equicdft import GridData


class TestPolarizationNoiseData(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary_directory.name) / "uncertainty.extxyz"
        self.old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        self.positions = np.indices((2, 2, 2)).reshape(3, -1).T
        self.order = np.random.default_rng(24).permutation(8)

    def tearDown(self):
        torch.set_default_dtype(self.old_dtype)
        self.temporary_directory.cleanup()

    def fields(self, n_types=1):
        # Exact binary fractions also survive EXTXYZ's eight decimal places,
        # so ordering and no-rescaling checks can require exact equality.
        polarization = np.arange(8 * n_types * 3).reshape(8, n_types, 3) / 32
        return {
            "grid_size": [2, 2, 2], "grid_spacing": 0.5,
            "T": 1.0, "n_types": n_types,
            "rho": np.ones((8, n_types)),
            "rho_std": np.arange(8 * n_types).reshape(8, n_types) / 128,
            "dipole_density": polarization - 0.25,
            "dipole_density_std": polarization / 8,
        }

    def atoms(self, values, keys=None):
        keys = dict(keys or {})
        atom = Atoms("X" * 8, positions=self.positions[self.order])
        atom.info.update(T=values["T"], grid_size=values["grid_size"],
                         grid_spacing=values["grid_spacing"])
        for canonical, default in (
            ("rho", "density"), ("rho_std", "density_std"),
            ("dipole_density", "dipole_density"),
            ("dipole_density_std", "dipole_density_std"),
        ):
            if canonical in values:
                atom.arrays[keys.get(canonical, default)] = np.asarray(
                    values[canonical]
                ).reshape(8, -1)[self.order]
        if "excluded_mask" in values:
            atom.arrays["excluded_mask"] = values["excluded_mask"][self.order]
        return atom

    def read(self, atoms, **kwargs):
        write(self.path, atoms, format="extxyz")
        return GridData.from_xyz(self.path, cutoff_grid=0, **kwargs)

    def test_xyz_custom_properties_order_units_and_multiple_species(self):
        keys = {"rho_std": "rho_sem", "dipole_density_std": "P_sem"}
        for n_types in (1, 2, 3):
            with self.subTest(n_types=n_types):
                expected = self.fields(n_types)
                frame = self.read(self.atoms(expected, keys), data_key=keys)[0]
                for name in ("rho", "rho_std", "dipole_density", "dipole_density_std"):
                    np.testing.assert_array_equal(frame[name], expected[name])
                batch = next(iter(DataLoader([frame, frame], batch_size=2)))
                self.assertEqual(batch["dipole_density_std"].shape,
                                 (2, 8, n_types, 3))
                self.assertEqual(batch["rho_std"].shape, (2, 8, n_types))

    def test_from_dict_accepts_both_uncertainties(self):
        for n_types in (1, 2, 3):
            values = self.fields(n_types)
            frame = GridData.from_dict(values, cutoff_grid=0)
            for name in ("rho_std", "dipole_density_std"):
                np.testing.assert_array_equal(frame[name], values[name])

    def test_physical_centers_and_sem_share_sorting_and_batching(self):
        values = self.fields(2)
        shifted, unshifted = self.atoms(values), self.atoms(values)
        centers = self.positions * .5 + np.array([.125, -.25, .5])
        shifted.arrays["grid_center"] = centers[self.order]
        frames = self.read([shifted, unshifted])
        batch = next(iter(DataLoader(frames, batch_size=2)))
        np.testing.assert_array_equal(batch["grid_center"][0], centers)
        np.testing.assert_array_equal(batch["grid_center"][1], self.positions * .5)
        for name in ("rho", "rho_std", "dipole_density", "dipole_density_std"):
            for index in range(2):
                np.testing.assert_array_equal(batch[name][index], values[name])

    def test_each_uncertainty_is_independently_optional(self):
        for omit in (("rho_std",), ("dipole_density_std",),
                     ("rho_std", "dipole_density_std")):
            values = self.fields()
            for name in omit:
                values.pop(name)
            frame = self.read(self.atoms(values))[0]
            for name in ("rho_std", "dipole_density_std"):
                self.assertEqual(name in frame, name in values)

    def test_uncertainty_requires_corresponding_field(self):
        for name, field in (("rho_std", "rho"),
                            ("dipole_density_std", "dipole_density")):
            values = self.fields()
            values.pop(field)
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, name + " requires " + field):
                    GridData.from_dict(values, cutoff_grid=0)
                if field == "dipole_density":
                    with self.assertRaisesRegex(ValueError, name + " requires " + field):
                        self.read(self.atoms(values))

    def test_uncertainties_reject_nonfinite_negative_and_incorrect_shapes(self):
        for name, shape in (("rho_std", (8, 1)),
                            ("dipole_density_std", (8, 1, 3))):
            bad_fields = [
                np.full(shape, value) for value in (-0.01, np.nan, np.inf)
            ]
            bad_fields.append(np.ones((8, 2)))
            for field in bad_fields:
                with self.subTest(name=name, shape=field.shape, value=field.flat[0]):
                    values = self.fields()
                    values[name] = field
                    with self.assertRaisesRegex(ValueError, name):
                        GridData.from_dict(values, cutoff_grid=0)
                    with self.assertRaisesRegex(ValueError, name):
                        self.read(self.atoms(values))
        values = self.fields()
        values["dipole_density_std"] = np.zeros((8, 3))
        with self.assertRaisesRegex(ValueError, "dipole_density_std must have shape"):
            GridData.from_dict(values, cutoff_grid=0)
        values["dipole_density_std"] = np.zeros((8, 1, 3), dtype=complex)
        with self.assertRaisesRegex(ValueError, "finite real"):
            GridData.from_dict(values, cutoff_grid=0)

    def test_uncertainty_is_zero_on_excluded_cells(self):
        values = self.fields(2)
        mask = np.array([True] + [False] * 7)
        values["excluded_mask"] = mask
        for name in ("rho", "rho_std", "dipole_density", "dipole_density_std"):
            values[name][mask] = 0.0
        frame = self.read(self.atoms(values))[0]
        self.assertTrue(torch.all(frame["rho_std"][0] == 0))
        self.assertTrue(torch.all(frame["dipole_density_std"][0] == 0))
        for name in ("rho_std", "dipole_density_std"):
            values[name][0] = 0.1
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, name + " must be zero"):
                    GridData.from_dict(values, cutoff_grid=0)
                with self.assertRaisesRegex(ValueError, name + " must be zero"):
                    self.read(self.atoms(values))
            values[name][0] = 0.0

    def test_coarsening_rejected_but_same_grid_allowed(self):
        for name in ("rho_std", "dipole_density_std"):
            values = self.fields()
            other = "rho_std" if name == "dipole_density_std" else "dipole_density_std"
            values.pop(other)
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, name + " cannot be coarsened"):
                    self.read(self.atoms(values), target_grid_spacing=1.0)
                frame = self.read(self.atoms(values), target_grid_spacing=0.5)[0]
                np.testing.assert_array_equal(frame[name], values[name])

    def test_mixed_frame_uncertainty_availability_rejected(self):
        for name in ("rho_std", "dipole_density_std"):
            full = self.fields()
            partial = dict(full)
            partial.pop(name)
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, name + " must be present"):
                    self.read([self.atoms(full), self.atoms(partial)])


if __name__ == "__main__":
    unittest.main()
