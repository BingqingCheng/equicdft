"""Electrode-file loading and validation, independent of liquid GridData."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io import write

from equicdft import GridData, read_metal_sites
from equicdft._metal_data import normalize_metal_sites


def _metadata():
    return {
        "metal_positions": [[0., 0., 0.216690617966578], [1., .5, 1.716690617966578]],
        "metal_site_groups": [9, 2],
        "metal_group_ids": [2, 9],
        "metal_total_charge": [-.3, .3],
        "metal_charge_units": "e",
    }


def _grid_values():
    return dict(
        grid_size=[4, 2, 2], grid_spacing=1., temperature=1., n_types=2,
    )


def _atoms():
    shape = (4, 2, 2)
    order = np.random.default_rng(81).permutation(16)
    positions = np.indices(shape).reshape(3, -1).T
    atoms = Atoms("X" * 16, positions=positions[order], cell=np.diag(shape), pbc=True)
    atoms.arrays["density"] = np.full((16, 2), .2)
    atoms.arrays["V_ext"] = np.zeros((16, 2))
    atoms.info.update(grid_size=np.array(shape), grid_spacing=1., T=1.)
    return atoms


class TestMetalSitesData(unittest.TestCase):
    def test_plain_xyz_infers_one_group_from_scalar_total_charge(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sites.xyz"
            write(path, Atoms("XX", positions=[[1., 2., 3.], [4., 5., 6.]]))
            with self.assertRaises((TypeError, ValueError)):
                read_metal_sites(path)
            result = read_metal_sites(
                path, origin=(.125, .25, .5), total_charge=0.,
            )
        self.assertEqual(result["metal_site_groups"].tolist(), [0, 0])
        self.assertEqual(result["metal_group_ids"].tolist(), [0])
        self.assertEqual(result["metal_total_charge"].tolist(), [0.])
        self.assertEqual(result["metal_charge_units"], "e")
        torch.testing.assert_close(
            result["metal_positions"],
            torch.tensor([[.875, 1.75, 2.5], [3.875, 4.75, 5.5]], dtype=torch.float64),
        )

    def test_group_ids_are_inferred_from_site_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sites.xyz"
            write(path, Atoms("XXX", positions=np.zeros((3, 3))))
            result = read_metal_sites(
                path,
                site_groups=[7, 2, 7],
                total_charge=[-.3, .3],
            )
        self.assertEqual(result["metal_site_groups"].tolist(), [7, 2, 7])
        self.assertEqual(result["metal_group_ids"].tolist(), [2, 7])
        self.assertEqual(result["metal_total_charge"].tolist(), [-.3, .3])

    def test_multiple_groups_require_site_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sites.xyz"
            write(path, Atoms("XX", positions=np.zeros((2, 3))))
            with self.assertRaisesRegex(ValueError, "site_groups"):
                read_metal_sites(path, total_charge=[-.3, .3])

    def test_extxyz_uses_declared_metadata_without_reordering(self):
        atoms = Atoms("XX", positions=_metadata()["metal_positions"])
        atoms.arrays["metal_site_groups"] = np.array([9, 2])
        atoms.info.update(
            metal_group_ids=np.array([2, 9]),
            metal_total_charge=np.array([-.3, .3]),
            metal_charge_units="e",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sites.extxyz"
            write(path, atoms, format="extxyz")
            result = read_metal_sites(path)
            with self.assertRaisesRegex(ValueError, "one selected XYZ frame"):
                read_metal_sites(path, index=":")
        self.assertEqual(result["metal_group_ids"].tolist(), [2, 9])
        self.assertEqual(result["metal_site_groups"].tolist(), [9, 2])
        self.assertEqual(result["metal_positions"].dtype, torch.float64)
        self.assertAlmostEqual(result["metal_positions"][0, 2].item(), .216690617966578, places=8)

    def test_invalid_metadata_is_rejected(self):
        cases = [
            ({"metal_positions": []}, "shape"),
            ({"metal_positions": [[float("nan"), 0., 0.], [1., 1., 1.]]}, "finite"),
            ({"metal_site_groups": [9]}, "shape"),
            ({"metal_site_groups": [9, -1]}, "nonnegative"),
            ({"metal_site_groups": [9, 2.5]}, "integers"),
            ({"metal_group_ids": [9, 3]}, "exactly cover"),
            ({"metal_group_ids": [9, 9]}, "unique"),
            ({"metal_charge_units": None}, "require"),
            ({"metal_positions": torch.ones(2, 3, requires_grad=True)}, "fixed geometry"),
        ]
        for changes, message in cases:
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                values = _metadata()
                values.update(changes)
                normalize_metal_sites(
                    values["metal_group_ids"], values["metal_total_charge"],
                    values["metal_charge_units"],
                    metal_positions=values["metal_positions"],
                    metal_site_groups=values["metal_site_groups"],
                )

    def test_grid_data_rejects_electrode_ownership(self):
        values = dict(_grid_values(), **_metadata())
        with self.assertRaisesRegex(KeyError, "metal_positions"):
            GridData.from_dict(values, cutoff_grid=0)


if __name__ == "__main__":
    unittest.main()
