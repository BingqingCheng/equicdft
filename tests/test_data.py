import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io import write
from torch.utils.data import DataLoader

from cace_grid import CartesianAFeatures, GridData, default_data_key


def _write_grid(path, include_mu=True):
    shape = (4, 4, 4)
    spacing = 0.5
    grid_positions = np.indices(shape, dtype=int).reshape(3, -1).T
    values = np.arange(len(grid_positions), dtype=float)
    order = np.random.default_rng(7).permutation(len(grid_positions))

    atoms = Atoms(
        symbols=["X"] * len(grid_positions),
        positions=grid_positions[order],
        cell=np.diag(shape),
        pbc=True,
    )
    atoms.arrays["density"] = (values + 0.25)[order]
    atoms.arrays["V_ext"] = (-values)[order]
    atoms.info["grid_size"] = np.asarray(shape)
    atoms.info["grid_spacing"] = np.repeat(spacing, 3)
    atoms.info["grid_indexing"] = "zero_based"
    atoms.info["T"] = 1.5
    if include_mu:
        atoms.info["mu"] = -1.0
    write(path, atoms, format="extxyz")


def _write_grid_with_custom_keys(path):
    shape = (4, 4, 4)
    spacing = 0.5
    grid_positions = np.indices(shape, dtype=int).reshape(3, -1).T
    values = np.arange(len(grid_positions), dtype=float)

    atoms = Atoms(
        symbols=["X"] * len(grid_positions),
        positions=grid_positions,
        cell=np.diag(shape),
        pbc=True,
    )
    atoms.arrays["grid_coordinates"] = grid_positions
    atoms.arrays["number_density"] = values + 0.25
    atoms.arrays["external_field"] = -values
    atoms.info["shape"] = np.asarray(shape)
    atoms.info["spacing"] = np.repeat(spacing, 3)
    atoms.info["indexing"] = "zero_based"
    atoms.info["temperature_value"] = 1.5
    atoms.info["chemical_potential"] = -1.0
    write(path, atoms, format="extxyz")


def _write_multitype_grid(path):
    shape = (4, 4, 4)
    spacing = 0.5
    grid_positions = np.indices(shape, dtype=int).reshape(3, -1).T
    values = np.arange(len(grid_positions), dtype=float)

    atoms = Atoms(
        symbols=["X"] * len(grid_positions),
        positions=grid_positions,
        cell=np.diag(shape),
        pbc=True,
    )
    atoms.arrays["density"] = np.column_stack(
        (values + 0.25, values + 100.25)
    )
    atoms.arrays["V_ext"] = np.column_stack((-values, -values - 10.0))
    atoms.info["grid_size"] = np.asarray(shape)
    atoms.info["grid_spacing"] = np.repeat(spacing, 3)
    atoms.info["grid_indexing"] = "zero_based"
    atoms.info["T"] = 1.5
    atoms.info["mu"] = -1.0
    write(path, atoms, format="extxyz")


class TestGridData(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary_directory.name) / "grid.extxyz"
        _write_grid(self.path)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_grid_data_dictionary(self):
        data = GridData.from_xyz(self.path, cutoff_grid=1)[0]

        self.assertEqual(
            set(data),
            {
                "temperature",
                "beta",
                "mu",
                "n_types",
                "thermal_wavelength",
                "grid_spacing",
                "index",
                "grid_positions",
                "V_ext",
                "rho",
                "c1_plus_beta_mu",
                "c1",
                "local_density_index",
                "local_density_positions",
            },
        )
        self.assertEqual(data["rho"].shape, (64, 1))
        self.assertEqual(data["V_ext"].shape, (64, 1))
        self.assertEqual(data["c1_plus_beta_mu"].shape, (64, 1))
        self.assertEqual(data["c1"].shape, (64, 1))
        self.assertEqual(data["n_types"].item(), 1)
        self.assertEqual(data["mu"].shape, (1,))
        self.assertEqual(data["thermal_wavelength"].shape, (1,))
        expected_beta = 1.0 / (8.617333262e-5 * 1.5)
        self.assertTrue(
            np.isclose(data["beta"].item(), expected_beta, rtol=1.0e-6)
        )
        self.assertTrue(
            np.isclose(
                data["c1_plus_beta_mu"][0, 0].item(),
                np.log(0.25),
                rtol=1.0e-6,
            )
        )
        self.assertTrue(
            np.isclose(
                data["c1"][0, 0].item(),
                np.log(0.25) + expected_beta,
                rtol=1.0e-6,
            )
        )
        self.assertEqual(data["local_density_index"].shape, (64, 7))
        self.assertEqual(data["local_density_positions"].shape, (7, 3))
        self.assertTrue(
            torch.equal(
                data["local_density_positions"][0], torch.tensor([0, 0, 0])
            )
        )
        self.assertTrue(
            torch.equal(data["local_density_index"][:, 0], data["index"])
        )
        self.assertTrue(
            torch.equal(
                data["grid_positions"][1], torch.tensor([0, 0, 1])
            )
        )

    def test_mu_is_optional(self):
        path = Path(self.temporary_directory.name) / "without-mu.extxyz"
        _write_grid(path, include_mu=False)
        data = GridData.from_xyz(path, cutoff_grid=1)[0]

        self.assertNotIn("mu", data)
        self.assertNotIn("c1", data)
        self.assertIn("c1_plus_beta_mu", data)
        batch = next(iter(DataLoader([data, data], batch_size=2)))
        self.assertNotIn("mu", batch)
        self.assertNotIn("c1", batch)

    def test_periodic_local_density_and_default_batching(self):
        data = GridData.from_xyz(self.path, cutoff_grid=1)[0]
        offset_lookup = {
            tuple(position.tolist()): index
            for index, position in enumerate(data["local_density_positions"])
        }
        minus_z = offset_lookup[(0, 0, -1)]

        minus_z_index = data["local_density_index"][0, minus_z]
        self.assertEqual(data["rho"][minus_z_index, 0].item(), 3.25)

        batch = next(iter(DataLoader([data, data], batch_size=2)))
        self.assertEqual(batch["rho"].shape, (2, 64, 1))
        self.assertEqual(batch["local_density_index"].shape, (2, 64, 7))

        # The representation gathers neighborhoods from this exact batched
        # rho tensor, so its complete overlapping dependence is differentiable.
        batch["rho"].requires_grad_(True)
        features = CartesianAFeatures(
            mean_density=1.0,
            cutoff_grid=1,
            max_power=0,
            n_radial_channels=1,
        )(batch)
        # Differentiate only the scalar environment centered at grid point 0.
        # Its gradient must reach all seven stencil densities and no others.
        selected_environment = features[:, 0, 0, 0, 0].sum()
        gradient = torch.autograd.grad(
            selected_environment,
            batch["rho"],
        )[0]
        self.assertEqual(gradient.shape, batch["rho"].shape)
        self.assertTrue(torch.all(torch.isfinite(gradient)))
        self.assertTrue(
            torch.equal(
                torch.count_nonzero(gradient[..., 0], dim=-1),
                torch.tensor([7, 7]),
            )
        )
        neighbor_indices = batch["local_density_index"][:, 0, :]
        gathered_gradient = torch.gather(
            gradient[..., 0],
            dim=1,
            index=neighbor_indices,
        )
        self.assertTrue(torch.all(gathered_gradient != 0.0))

    def test_custom_data_key_mapping(self):
        path = Path(self.temporary_directory.name) / "custom.extxyz"
        _write_grid_with_custom_keys(path)
        data = GridData.from_xyz(
            path,
            cutoff_grid=1,
            data_key={
                "temperature": "temperature_value",
                "mu": "chemical_potential",
                "grid_spacing": "spacing",
                "grid_size": "shape",
                "grid_indexing": "indexing",
                "grid_positions": "grid_coordinates",
                "V_ext": "external_field",
                "rho": "number_density",
            },
        )[0]

        self.assertEqual(data["temperature"].item(), 1.5)
        self.assertEqual(data["mu"].item(), -1.0)
        self.assertTrue(
            torch.equal(data["local_density_index"][:, 0], data["index"])
        )
        self.assertEqual(default_data_key["rho"], "density")

    def test_optional_local_average(self):
        data = GridData.from_xyz(
            self.path,
            cutoff_grid=0,
            target_grid_spacing=1.0,
        )[0]

        source = (np.arange(64, dtype=float) + 0.25).reshape(4, 4, 4)
        expected = source.reshape(2, 2, 2, 2, 2, 2).mean(axis=(1, 3, 5))
        self.assertEqual(data["rho"].shape, (8, 1))
        self.assertEqual(data["n_types"].item(), 1)
        self.assertEqual(data["local_density_index"].shape, (8, 1))
        self.assertTrue(
            torch.allclose(
                data["rho"],
                torch.tensor(expected.reshape(-1, 1), dtype=data["rho"].dtype),
            )
        )
        self.assertTrue(
            torch.equal(
                data["grid_positions"][1], torch.tensor([0, 0, 1])
            )
        )
        self.assertTrue(
            torch.allclose(
                data["grid_spacing"],
                torch.tensor([1.0, 1.0, 1.0], dtype=data["grid_spacing"].dtype),
            )
        )
        expected_c1 = (
            torch.log(data["rho"])
            + data["beta"]
            * (data["V_ext"] - data["mu"].unsqueeze(0))
        )
        self.assertTrue(torch.allclose(data["c1"], expected_c1))

    def test_multitype_fields(self):
        path = Path(self.temporary_directory.name) / "multitype.extxyz"
        _write_multitype_grid(path)
        data = GridData.from_xyz(
            path,
            cutoff_grid=1,
            boltzmann_constant=1.0,
            thermal_wavelength=[1.0, 2.0],
        )[0]

        self.assertEqual(data["n_types"].item(), 2)
        self.assertEqual(data["rho"].shape, (64, 2))
        self.assertEqual(data["V_ext"].shape, (64, 2))
        self.assertEqual(data["mu"].shape, (2,))
        self.assertTrue(
            torch.equal(
                data["thermal_wavelength"],
                torch.tensor([1.0, 2.0], dtype=data["rho"].dtype),
            )
        )
        self.assertAlmostEqual(data["beta"].item(), 1.0 / 1.5)
        expected_c1_plus_beta_mu = (
            torch.log(
                data["rho"]
                * data["thermal_wavelength"].unsqueeze(0) ** 3
            )
            + data["beta"] * data["V_ext"]
        )
        self.assertTrue(
            torch.allclose(
                data["c1_plus_beta_mu"],
                expected_c1_plus_beta_mu,
                atol=5.0e-6,
                rtol=1.0e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                data["c1"],
                expected_c1_plus_beta_mu
                - data["beta"] * data["mu"].unsqueeze(0),
                atol=5.0e-6,
                rtol=1.0e-6,
            )
        )
        self.assertEqual(data["local_density_index"].shape, (64, 7))
        self.assertTrue(
            torch.equal(data["local_density_index"][:, 0], data["index"])
        )

        coarse = GridData.from_xyz(
            path,
            cutoff_grid=0,
            target_grid_spacing=1.0,
            boltzmann_constant=1.0,
            thermal_wavelength=[1.0, 2.0],
        )[0]
        self.assertEqual(coarse["n_types"].item(), 2)
        self.assertEqual(coarse["rho"].shape, (8, 2))
        self.assertEqual(coarse["V_ext"].shape, (8, 2))
        self.assertEqual(coarse["local_density_index"].shape, (8, 1))

    def test_thermodynamic_constants_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "boltzmann_constant"):
            GridData.from_xyz(
                self.path,
                boltzmann_constant=0.0,
            )
        with self.assertRaisesRegex(ValueError, "thermal_wavelength"):
            GridData.from_xyz(
                self.path,
                thermal_wavelength=0.0,
            )


if __name__ == "__main__":
    unittest.main()
