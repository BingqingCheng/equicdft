"""Explicit-site loading and metadata plumbing, independent of electrostatics."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io import write
from torch.utils.data import DataLoader

from equicdft import EnergyReadout, GridCACEModel, GridData, GridSolver, read_metal_sites
from equicdft._metal_data import normalize_metal_metadata


def _metadata():
    return {
        "metal_positions": [[0., 0., 0.216690617966578], [1., .5, 1.716690617966578]],
        "metal_site_groups": [9, 2],
        "metal_group_ids": [2, 9],
        "metal_total_charge": [-.3, .3],
        "metal_charge_units": "e",
    }


def _grid_values():
    return dict(grid_size=[4, 2, 2], grid_spacing=1., temperature=1.,
                n_types=2, **_metadata())


def _atoms():
    shape = (4, 2, 2)
    order = np.random.default_rng(81).permutation(16)
    positions = np.indices(shape).reshape(3, -1).T
    atoms = Atoms("X" * 16, positions=positions[order], cell=np.diag(shape), pbc=True)
    atoms.arrays["density"] = np.full((16, 2), .2)
    atoms.arrays["V_ext"] = np.zeros((16, 2))
    atoms.info.update(grid_size=np.array(shape), grid_spacing=1., T=1.)
    atoms.info.update({key: np.asarray(value) if isinstance(value, list) else value
                       for key, value in _metadata().items()})
    return atoms


class _MetadataReadout(EnergyReadout):
    n_types = 2

    def energy(self, context):
        self.seen = {key: context[key] for key in _metadata()}
        return context["rho"].square().sum(dim=(-2, -1)) * 0.


class TestMetalSitesData(unittest.TestCase):
    def _read(self, atoms, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grid.extxyz"
            write(path, atoms, format="extxyz")
            return GridData.from_xyz(path, cutoff_grid=0, boltzmann_constant=1., **kwargs)

    def test_from_dict_preserves_positions_precision_and_independent_exclusion(self):
        values = _grid_values()
        values["excluded_mask"] = [False, True] + [False] * 14
        frame = GridData.from_dict(values, cutoff_grid=0)
        self.assertEqual(frame["metal_positions"].dtype, torch.float64)
        self.assertEqual(frame["metal_positions"][0, 2].item(), .216690617966578)
        self.assertEqual(frame["metal_site_groups"].tolist(), [9, 2])
        self.assertEqual(frame["metal_group_ids"].tolist(), [2, 9])
        self.assertEqual(frame["excluded_mask"].sum().item(), 1)
        self.assertNotIn("metal_mask", frame)

    def test_extxyz_metadata_is_not_sorted_as_voxel_data(self):
        frame = self._read(_atoms())[0]
        torch.testing.assert_close(frame["metal_positions"],
                                   torch.tensor(_metadata()["metal_positions"], dtype=torch.float64))
        self.assertEqual(frame["metal_site_groups"].tolist(), [9, 2])
        self.assertFalse(frame["excluded_mask"].any())
        self.assertTrue(torch.all(frame["rho"] == .2))

    def test_xyz_key_aliases_for_site_metadata(self):
        atoms = _atoms()
        atoms.info["sites"] = atoms.info.pop("metal_positions")
        atoms.info["labels"] = atoms.info.pop("metal_site_groups")
        frame = self._read(atoms, data_key={"metal_positions": "sites", "metal_site_groups": "labels"})[0]
        self.assertEqual(frame["metal_site_groups"].tolist(), [9, 2])

    def test_batch_collation_and_shared_metadata(self):
        frames = self._read([_atoms(), _atoms()])
        batch = next(iter(DataLoader(frames, batch_size=2)))
        self.assertEqual(batch["metal_positions"].shape, (2, 2, 3))
        data = normalize_metal_metadata(None, [2, 9], [-.3, .3], "e", n_grid=16,
            batch_shape=(2,), dtype=torch.float64, metal_positions=frames[0]["metal_positions"],
            metal_site_groups=frames[0]["metal_site_groups"])
        self.assertEqual(data["metal_positions"].shape, (2, 2, 3))
        self.assertEqual(data["metal_site_groups"].tolist(), [[9, 2], [9, 2]])
        single = normalize_metal_metadata(None, [2, 9], [-.3, .3], "e", n_grid=16,
            dtype=torch.float32, metal_positions=_metadata()["metal_positions"],
            metal_site_groups=[9, 2])
        self.assertEqual(single["metal_positions"].dtype, torch.float64)
        self.assertEqual(single["metal_positions"][0, 2].item(), .216690617966578)
        self.assertEqual(single["metal_total_charge"].dtype, torch.float32)

    def test_batch_requires_same_site_count_and_representation(self):
        first, second = _atoms(), _atoms()
        second.info["metal_positions"] = np.vstack((second.info["metal_positions"], [2., .5, .75]))
        second.info["metal_site_groups"] = np.array([9, 2, 2])
        with self.assertRaisesRegex(ValueError, "same metal site count"):
            self._read([first, second])
        second = _atoms()
        del second.info["metal_positions"], second.info["metal_site_groups"]
        second.arrays["metal_mask"] = np.array([2, 9] + [-1] * 14)
        second.arrays["density"][:2] = 0.
        with self.assertRaisesRegex(ValueError, "share metal metadata keys"):
            self._read([first, second])

    def test_changed_grid_spacing_requires_explicit_new_frame(self):
        with self.assertRaisesRegex(ValueError, "coarsening explicit metal sites"):
            self._read(_atoms(), target_grid_spacing=2.)
        self.assertEqual(self._read(_atoms(), target_grid_spacing=1.)[0]["metal_positions"].shape, (2, 3))

    def test_all_negative_legacy_mask_can_coexist(self):
        values = _grid_values()
        values["metal_mask"] = [-1] * 16
        frame = GridData.from_dict(values, cutoff_grid=0)
        self.assertTrue(torch.all(frame["metal_mask"] == -1))
        values["metal_mask"][0] = 2
        with self.assertRaisesRegex(ValueError, "cannot coexist"):
            GridData.from_dict(values, cutoff_grid=0)

    def test_invalid_explicit_metadata_is_rejected(self):
        cases = [
            ({"metal_positions": None}, "required together"),
            ({"metal_site_groups": None}, "required together"),
            ({"metal_positions": []}, "shape"),
            ({"metal_positions": [[0., 0.], [1., 1.]]}, "shape"),
            ({"metal_positions": [[float("nan"), 0., 0.], [1., 1., 1.]]}, "finite"),
            ({"metal_positions": [[True] * 3] * 2}, "real finite"),
            ({"metal_positions": [[1j, 0., 0.], [1., 1., 1.]]}, "real finite"),
            ({"metal_site_groups": [9]}, "shape"),
            ({"metal_site_groups": [9, -1]}, "nonnegative"),
            ({"metal_site_groups": [9, 2.5]}, "integers"),
            ({"metal_group_ids": [9, 3]}, "exactly cover"),
            ({"metal_group_ids": [9, 9]}, "unique"),
            ({"metal_charge_units": None}, "requires explicit"),
            ({"metal_positions": torch.ones(2, 3, requires_grad=True)}, "fixed geometry"),
        ]
        for changes, message in cases:
            with self.subTest(changes=changes):
                values = _grid_values()
                values.update(changes)
                with self.assertRaisesRegex(ValueError, message):
                    GridData.from_dict(values, cutoff_grid=0)

    def test_reader_plain_xyz_requires_explicit_labels_and_constraints(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sites.xyz"
            write(path, Atoms("XX", positions=[[1., 2., 3.], [4., 5., 6.]]), format="xyz")
            with self.assertRaisesRegex(ValueError, "required together"):
                read_metal_sites(path)
            result = read_metal_sites(path, origin=(.125, .25, .5), site_groups=7,
                                     group_ids=7, total_charge=0., charge_units="e")
            self.assertEqual(result["metal_site_groups"].tolist(), [7, 7])
            torch.testing.assert_close(result["metal_positions"],
                torch.tensor([[.875, 1.75, 2.5], [3.875, 4.75, 5.5]], dtype=torch.float64))
            self.assertEqual(set(result), set(_metadata()))
            for origin in ((0., 0.), (0., float("inf"), 0.), (False, False, False), (0., 0., 1j)):
                with self.subTest(origin=origin), self.assertRaisesRegex(ValueError, "origin"):
                    read_metal_sites(path, origin=origin, site_groups=7,
                                     group_ids=7, total_charge=0., charge_units="e")

    def test_reader_extxyz_uses_metadata_without_inferred_charge(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sites.extxyz"
            atoms = Atoms("XX", positions=_metadata()["metal_positions"])
            atoms.arrays["metal_site_groups"] = np.array([9, 2])
            atoms.info.update(metal_group_ids=np.array([2, 9]), metal_total_charge=np.array([-.3, .3]), metal_charge_units="e")
            write(path, atoms, format="extxyz")
            result = read_metal_sites(path)
            self.assertEqual(result["metal_group_ids"].tolist(), [2, 9])
            self.assertEqual(result["metal_site_groups"].tolist(), [9, 2])
            with self.assertRaisesRegex(ValueError, "one selected XYZ frame"):
                read_metal_sites(path, index=":")
            with self.assertRaisesRegex(ValueError, "must be 'e'"):
                read_metal_sites(path, charge_units="C")

    def test_model_and_solver_forward_and_retain_site_metadata(self):
        readout = _MetadataReadout()
        model = GridCACEModel(a_features=None, b_features=None, readout=[readout],
                              grid_spacing=1., mean_temperature=1., boltzmann_constant=1.)
        data = GridData.from_dict(_grid_values(), cutoff_grid=0, boltzmann_constant=1.)
        data["rho"] = torch.full((16, 2), .2)
        data["V_ext"] = torch.zeros_like(data["rho"])
        result = GridSolver(model).evaluate(data)
        for key in _metadata():
            if torch.is_tensor(data[key]):
                torch.testing.assert_close(readout.seen[key], data[key])
                torch.testing.assert_close(result[key], data[key])
            else:
                self.assertEqual(readout.seen[key], data[key])
                self.assertEqual(result[key], data[key])
        self.assertFalse(result["excluded_mask"].any())
        self.assertTrue(torch.all(result["rho"] == .2))


if __name__ == "__main__":
    unittest.main()
