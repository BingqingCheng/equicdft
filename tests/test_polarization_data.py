"""Optional polar-vector grid fields retain their signs, ordering and units."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io import write
from torch.utils.data import DataLoader

from equicdft import GridData


class TestPolarizationData(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary_directory.name) / "polar.extxyz"
        self.shape = (4, 2, 2)
        self.positions = np.indices(self.shape).reshape(3, -1).T
        self.n_grid = len(self.positions)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write_frame(
        self, dipoles, n_types=1, key="dipole_density", excluded=None,
        include_rho=True,
    ):
        order = np.random.default_rng(13).permutation(self.n_grid)
        atoms = Atoms("X" * self.n_grid, positions=self.positions[order])
        density = np.full((self.n_grid, n_types), 0.5)
        if excluded is not None:
            density[excluded] = 0.0
            atoms.arrays["excluded_mask"] = np.asarray(excluded)[order]
        if include_rho:
            atoms.arrays["density"] = density[order]
        atoms.arrays["V_ext"] = np.zeros_like(density)[order]
        if dipoles is not None:
            atoms.arrays[key] = np.asarray(dipoles).reshape(self.n_grid, -1)[order]
        atoms.info.update(T=1.5, mu=-1.0, grid_spacing=0.5,
                          grid_size=np.asarray(self.shape))
        write(self.path, atoms, format="extxyz")

    def from_dict(self, dipoles, **kwargs):
        values = {
            "grid_size": list(self.shape),
            "grid_spacing": 0.5,
            "n_types": 1,
            "T": 1.5,
            "dipole_density": dipoles,
        }
        values.update(kwargs)
        return GridData.from_dict(values, cutoff_grid=1, boltzmann_constant=1)

    def test_xyz_custom_key_reorders_vectors_and_batches(self):
        dipoles = (np.arange(self.n_grid * 6).reshape(self.n_grid, 2, 3) - 30) / 8
        self.write_frame(dipoles, n_types=2, key="polarization_field")
        data = GridData.from_xyz(
            self.path, cutoff_grid=1,
            data_key={"dipole_density": "polarization_field"},
        )[0]
        self.assertEqual(data["dipole_density"].shape, (self.n_grid, 2, 3))
        np.testing.assert_allclose(data["dipole_density"], dipoles)
        np.testing.assert_array_equal(data["grid_positions"], self.positions)
        self.assertEqual(data["n_types"].item(), 2)
        batch = next(iter(DataLoader([data, data], batch_size=2)))
        self.assertEqual(batch["dipole_density"].shape, (2, self.n_grid, 2, 3))

    def test_xyz_single_type_and_type_count_from_external_field(self):
        dipoles = np.full((self.n_grid, 3), -0.25)
        self.write_frame(dipoles, include_rho=False)
        data = GridData.from_xyz(self.path, cutoff_grid=1)[0]
        self.assertNotIn("rho", data)
        self.assertEqual(data["dipole_density"].shape, (self.n_grid, 1, 3))
        self.assertTrue(torch.all(data["dipole_density"] == -0.25))

    def test_xyz_custom_key_can_be_an_ase_reserved_property(self):
        dipoles = np.full((self.n_grid, 3), -0.5)
        self.write_frame(dipoles, key="polarization")
        data = GridData.from_xyz(
            self.path, data_key={"dipole_density": "polarization"},
        )[0]
        np.testing.assert_allclose(data["dipole_density"][:, 0], dipoles)

    def test_componentwise_coarsening_preserves_integrated_vector(self):
        dipoles = (np.arange(self.n_grid * 6).reshape(self.n_grid, 2, 3) - 23) / 8
        self.write_frame(dipoles, n_types=2)
        data = GridData.from_xyz(
            self.path, cutoff_grid=1, target_grid_spacing=1.0,
        )[0]
        expected = dipoles.reshape(2, 2, 1, 2, 1, 2, 2, 3).mean(axis=(1, 3, 5))
        np.testing.assert_allclose(data["dipole_density"], expected.reshape(2, 2, 3))
        np.testing.assert_allclose(
            data["dipole_density"].sum(dim=0), dipoles.sum(axis=0) * 0.5**3,
        )

    def test_uniform_excluded_blocks_coarsen_and_remain_zero(self):
        excluded = self.positions[:, 0] < 2
        dipoles = np.ones((self.n_grid, 1, 3))
        dipoles[excluded] = 0.0
        self.write_frame(dipoles, excluded=excluded)
        data = GridData.from_xyz(
            self.path, cutoff_grid=1, target_grid_spacing=1.0,
        )[0]
        self.assertEqual(data["excluded_mask"].tolist(), [True, False])
        self.assertTrue(torch.all(data["dipole_density"][0] == 0))

    def test_excluded_dipoles_are_rejected_before_they_can_average_to_zero(self):
        excluded = self.positions[:, 0] < 2
        dipoles = np.zeros((self.n_grid, 1, 3))
        dipoles[0, 0, 0] = 1.0
        dipoles[1, 0, 0] = -1.0
        self.write_frame(dipoles, excluded=excluded)
        for target in (None, 1.0):
            with self.subTest(target=target):
                with self.assertRaisesRegex(ValueError, "dipole_density must be zero"):
                    GridData.from_xyz(self.path, target_grid_spacing=target)

    def test_xyz_rejects_wrong_type_count_or_nonfinite_components(self):
        for dipoles in (
            np.ones((self.n_grid, 2)),
            np.ones((self.n_grid, 6)),
            np.full((self.n_grid, 3), np.nan),
            np.full((self.n_grid, 3), np.inf),
        ):
            with self.subTest(shape=dipoles.shape, first=dipoles[0, 0]):
                self.write_frame(dipoles)
                with self.assertRaisesRegex(ValueError, "dipole_density"):
                    GridData.from_xyz(self.path)

    def test_from_dict_preserves_live_tensor_graph(self):
        dipoles = torch.randn(self.n_grid, 1, 3, dtype=torch.float64,
                              requires_grad=True)
        data = self.from_dict(dipoles)
        derivative = torch.autograd.grad(data["dipole_density"].sum(), dipoles)[0]
        torch.testing.assert_close(derivative, torch.ones_like(dipoles))

    def test_from_dict_accepts_signed_multitype_vectors(self):
        dipoles = -np.ones((self.n_grid, 2, 3))
        data = self.from_dict(dipoles, n_types=2)
        self.assertEqual(data["dipole_density"].shape, (self.n_grid, 2, 3))
        self.assertTrue(torch.all(data["dipole_density"] == -1.0))

    def test_from_dict_requires_canonical_shape_and_finite_real_values(self):
        for dipoles in (
            np.ones((self.n_grid, 3)),
            np.ones((self.n_grid, 2, 3)),
            np.ones((self.n_grid, 1, 2)),
            np.full((self.n_grid, 1, 3), np.inf),
            np.ones((self.n_grid, 1, 3), dtype=complex),
        ):
            with self.subTest(shape=dipoles.shape):
                with self.assertRaisesRegex(ValueError, "dipole_density"):
                    self.from_dict(dipoles)

    def test_from_dict_rejects_nonzero_excluded_dipoles(self):
        excluded = np.zeros(self.n_grid, dtype=bool)
        excluded[0] = True
        dipoles = torch.zeros(self.n_grid, 1, 3)
        dipoles[0, 0, 0] = 1e-15
        with self.assertRaisesRegex(ValueError, "dipole_density must be zero"):
            self.from_dict(dipoles, excluded_mask=excluded)
        dipoles[0] = 0
        data = self.from_dict(dipoles, excluded_mask=excluded)
        self.assertTrue(torch.all(data["dipole_density"][0] == 0))

    def test_absent_field_stays_absent_and_scalar_targets_are_unchanged(self):
        self.write_frame(None)
        data = GridData.from_xyz(self.path, boltzmann_constant=1)[0]
        self.assertNotIn("dipole_density", data)
        torch.testing.assert_close(data["rho"], torch.full((self.n_grid, 1), 0.5))
        expected = np.log(0.5) + 1.0 / 1.5
        torch.testing.assert_close(data["c1"], torch.full_like(data["c1"], expected))
        self.assertNotIn("dipole_density", self.from_dict(None))

    def test_polarized_frames_do_not_get_scalar_ideal_c1_targets(self):
        self.write_frame(np.zeros((self.n_grid, 3)))
        data = GridData.from_xyz(self.path, boltzmann_constant=1)[0]
        self.assertNotIn("c1", data)
        self.assertNotIn("c1_plus_beta_mu", data)
        torch.testing.assert_close(data["beta_mu"], torch.tensor([-1 / 1.5]))
        self.assertAlmostEqual(data["beta"].item(), 1 / 1.5, places=6)


if __name__ == "__main__":
    unittest.main()
