import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io import write
from torch.utils.data import DataLoader

from equicdft import CartesianAFeatures, GridData, default_data_key


def _write_grid(
    path,
    include_mu=True,
    include_rho=True,
    include_temperature=True,
    include_v_ext=True,
    density_offset=0.25,
    mu_value=-1.0,
):
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
    if include_rho:
        atoms.arrays["density"] = (values + density_offset)[order]
    if include_v_ext:
        atoms.arrays["V_ext"] = (-values)[order]
    atoms.info["grid_size"] = np.asarray(shape)
    atoms.info["grid_spacing"] = np.repeat(spacing, 3)
    atoms.info["grid_indexing"] = "zero_based"
    if include_temperature:
        atoms.info["T"] = 1.5
    if include_mu:
        atoms.info["mu"] = mu_value
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

    @staticmethod
    def _grid_info():
        return {
            "cutoff_grid": 1,
            "grid_spacing": [0.5, 0.5, 0.5],
            "n_types": 1,
            "boltzmann_constant": 1.0,
            "thermal_wavelength": [1.0],
        }

    def test_grid_info_configures_xyz_and_validates_geometry(self):
        data = GridData.from_xyz(
            self.path,
            grid_info=self._grid_info(),
        )[0]

        self.assertAlmostEqual(data["beta"].item(), 1.0 / 1.5)
        self.assertEqual(data["local_density_index"].shape, (64, 7))

        mismatched = self._grid_info()
        mismatched["grid_spacing"] = [1.0, 1.0, 1.0]
        with self.assertRaisesRegex(ValueError, "grid_spacing"):
            GridData.from_xyz(self.path, grid_info=mismatched)

    def test_from_xyz_accepts_ordered_path_sequence(self):
        second_path = Path(self.temporary_directory.name) / "grid-second.extxyz"
        _write_grid(second_path, density_offset=100.25)

        for paths in ([second_path, self.path], (second_path, self.path)):
            with self.subTest(container=type(paths).__name__):
                data = GridData.from_xyz(paths, cutoff_grid=1)

                self.assertEqual(len(data), 2)
                self.assertAlmostEqual(data[0]["rho"][0, 0].item(), 100.25)
                self.assertAlmostEqual(data[1]["rho"][0, 0].item(), 0.25)

    def test_from_xyz_rejects_empty_path_sequence(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            GridData.from_xyz([], cutoff_grid=1)

    def test_from_xyz_rejects_non_path_sequence_item(self):
        with self.assertRaisesRegex(TypeError, "every item"):
            GridData.from_xyz([self.path, 3], cutoff_grid=1)

    def test_grid_info_supplies_from_dict_model_metadata(self):
        data = GridData.from_dict(
            {
                "grid_size": [3, 4, 5],
                "temperature": 1.5,
            },
            grid_info=self._grid_info(),
        )

        self.assertEqual(data["n_types"].item(), 1)
        self.assertTrue(
            torch.equal(data["grid_size"], torch.tensor([3, 4, 5]))
        )
        self.assertTrue(
            torch.equal(
                data["grid_spacing"],
                torch.tensor([0.5, 0.5, 0.5]),
            )
        )
        self.assertAlmostEqual(data["beta"].item(), 1.0 / 1.5)
        self.assertEqual(data["local_density_index"].shape, (60, 7))

        conflicting_values = {
            "grid_size": [3, 4, 5],
            "grid_spacing": 1.0,
            "temperature": 1.5,
        }
        with self.assertRaisesRegex(ValueError, "does not match"):
            GridData.from_dict(
                conflicting_values,
                grid_info=self._grid_info(),
            )

    def test_from_dict_builds_grid_without_fields(self):
        data = GridData.from_dict(
            {
                "grid_size": [3, 4, 5],
                "n_types": 1,
                "grid_spacing": 0.5,
                "T": 1.5,
            },
            cutoff_grid=1,
            boltzmann_constant=1.0,
        )

        self.assertNotIn("rho", data)
        self.assertNotIn("V_ext", data)
        self.assertEqual(data["n_types"].item(), 1)
        self.assertTrue(
            torch.equal(data["grid_size"], torch.tensor([3, 4, 5]))
        )
        self.assertEqual(data["grid_positions"].shape, (60, 3))
        self.assertTrue(
            torch.equal(
                data["grid_positions"][1],
                torch.tensor([0, 0, 1]),
            )
        )
        self.assertTrue(
            torch.equal(
                data["grid_spacing"],
                torch.tensor([0.5, 0.5, 0.5]),
            )
        )
        self.assertAlmostEqual(data["beta"].item(), 1.0 / 1.5)
        self.assertEqual(data["local_density_index"].shape, (60, 7))

    def test_from_dict_builds_multicomponent_metadata(self):
        data = GridData.from_dict(
            {
                "grid_size": [2, 2, 2],
                "grid_spacing": [0.5, 0.6, 0.7],
                "temperature": 2.0,
                "n_types": 2,
            },
            cutoff_grid=0,
            boltzmann_constant=1.0,
            thermal_wavelength=[1.0, 2.0],
        )

        self.assertEqual(data["n_types"].item(), 2)
        self.assertTrue(
            torch.equal(data["grid_size"], torch.tensor([2, 2, 2]))
        )
        self.assertEqual(data["thermal_wavelength"].shape, (2,))
        self.assertNotIn("rho", data)
        self.assertNotIn("V_ext", data)

        data["V_ext"] = torch.zeros((8, 2))
        self.assertTrue(
            torch.equal(
                data["V_ext"],
                torch.zeros((8, 2)),
            )
        )

    def test_from_dict_requires_grid_geometry_and_temperature(self):
        with self.assertRaisesRegex(ValueError, "grid_size"):
            GridData.from_dict(
                {
                    "n_types": 1,
                    "grid_spacing": 0.5,
                    "temperature": 1.5,
                }
            )
        with self.assertRaisesRegex(ValueError, "temperature"):
            GridData.from_dict(
                {
                    "grid_size": [2, 2, 2],
                    "n_types": 1,
                    "grid_spacing": 0.5,
                }
            )
        with self.assertRaisesRegex(ValueError, "n_types"):
            GridData.from_dict(
                {
                    "grid_size": [2, 2, 2],
                    "grid_spacing": 0.5,
                    "temperature": 1.5,
                }
            )

    def test_grid_data_dictionary(self):
        data = GridData.from_xyz(self.path, cutoff_grid=1)[0]

        self.assertEqual(
            set(data),
            {
                "temperature",
                "beta",
                "mu",
                "beta_mu",
                "n_types",
                "grid_size",
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
        self.assertEqual(data["beta_mu"].shape, (1,))
        self.assertEqual(data["thermal_wavelength"].shape, (1,))
        expected_beta = 1.0 / (8.617333262e-5 * 1.5)
        self.assertTrue(
            np.isclose(data["beta"].item(), expected_beta, rtol=1.0e-6)
        )
        self.assertTrue(
            torch.allclose(data["beta_mu"], data["beta"] * data["mu"])
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
        self.assertNotIn("beta_mu", data)
        self.assertNotIn("c1", data)
        self.assertIn("c1_plus_beta_mu", data)
        batch = next(iter(DataLoader([data, data], batch_size=2)))
        self.assertNotIn("mu", batch)
        self.assertNotIn("c1", batch)

    def test_mixed_mu_selection_collates_with_nan_sentinel(self):
        without_mu_path = (
            Path(self.temporary_directory.name) / "without-mu-mixed.extxyz"
        )
        _write_grid(without_mu_path, include_mu=False)

        data = GridData.from_xyz(
            [without_mu_path, self.path],
            cutoff_grid=1,
        )
        batch = next(iter(DataLoader(data, batch_size=2, shuffle=False)))

        self.assertTrue(torch.isnan(data[0]["mu"]).all())
        self.assertTrue(torch.isnan(data[0]["beta_mu"]).all())
        self.assertTrue(torch.isfinite(data[1]["beta_mu"]).all())
        self.assertEqual(batch["beta_mu"].shape, (2, 1))
        self.assertTrue(torch.isnan(batch["beta_mu"][0]).all())
        self.assertTrue(torch.isfinite(batch["beta_mu"][1]).all())
        self.assertNotIn("c1", data[0])
        self.assertNotIn("c1", data[1])

    def test_nan_mu_placeholder_is_treated_as_absent(self):
        path = Path(self.temporary_directory.name) / "nan-mu.extxyz"
        _write_grid(path, mu_value=float("nan"))
        data = GridData.from_xyz(path, cutoff_grid=1)[0]

        self.assertNotIn("mu", data)
        self.assertNotIn("beta_mu", data)
        self.assertNotIn("c1", data)
        self.assertIn("c1_plus_beta_mu", data)

    def test_zero_density_is_retained_for_masked_losses(self):
        path = Path(self.temporary_directory.name) / "zero-density.extxyz"
        _write_grid(path, density_offset=0.0)

        data = GridData.from_xyz(path, cutoff_grid=1)[0]

        self.assertEqual(torch.count_nonzero(data["rho"] == 0.0).item(), 1)
        self.assertNotIn("c1_plus_beta_mu", data)
        self.assertNotIn("c1", data)

    def test_negative_density_is_rejected(self):
        path = Path(self.temporary_directory.name) / "negative-density.extxyz"
        _write_grid(path, density_offset=-0.25)

        with self.assertRaisesRegex(ValueError, "nonnegative"):
            GridData.from_xyz(path, cutoff_grid=1)

    def test_density_only_inference_data(self):
        path = Path(self.temporary_directory.name) / "density-only.extxyz"
        _write_grid(
            path,
            include_mu=False,
            include_v_ext=False,
        )
        data = GridData.from_xyz(
            path,
            cutoff_grid=0,
            target_grid_spacing=1.0,
        )[0]

        self.assertEqual(
            set(data),
            {
                "temperature",
                "beta",
                "n_types",
                "grid_size",
                "grid_spacing",
                "index",
                "grid_positions",
                "rho",
                "local_density_index",
                "local_density_positions",
            },
        )
        self.assertEqual(data["rho"].shape, (8, 1))
        self.assertTrue(
            torch.equal(data["grid_size"], torch.tensor([2, 2, 2]))
        )
        self.assertEqual(data["local_density_index"].shape, (8, 1))

    def test_external_potential_only_inference_data(self):
        path = Path(self.temporary_directory.name) / "external-only.extxyz"
        _write_grid(path, include_rho=False)
        data = GridData.from_xyz(
            path,
            cutoff_grid=0,
            target_grid_spacing=1.0,
        )[0]

        self.assertNotIn("rho", data)
        self.assertNotIn("c1_plus_beta_mu", data)
        self.assertNotIn("c1", data)
        self.assertEqual(data["V_ext"].shape, (8, 1))
        self.assertEqual(data["n_types"].item(), 1)
        self.assertEqual(data["thermal_wavelength"].shape, (1,))
        self.assertEqual(data["local_density_index"].shape, (8, 1))

    def test_density_or_external_potential_is_required(self):
        path = Path(self.temporary_directory.name) / "geometry-only.extxyz"
        _write_grid(
            path,
            include_rho=False,
            include_v_ext=False,
        )

        with self.assertRaisesRegex(ValueError, "rho or V_ext"):
            GridData.from_xyz(path, cutoff_grid=1)

    def test_external_potential_is_optional(self):
        no_external_path = (
            Path(self.temporary_directory.name) / "without-external.extxyz"
        )
        _write_grid(no_external_path, include_v_ext=False)
        no_external = GridData.from_xyz(no_external_path, cutoff_grid=1)[0]

        self.assertIn("temperature", no_external)
        self.assertIn("beta", no_external)
        self.assertIn("mu", no_external)
        self.assertNotIn("V_ext", no_external)
        self.assertNotIn("thermal_wavelength", no_external)
        self.assertNotIn("c1_plus_beta_mu", no_external)
        self.assertNotIn("c1", no_external)

    def test_temperature_is_required(self):
        path = Path(self.temporary_directory.name) / "without-temperature.extxyz"
        _write_grid(path, include_temperature=False)

        with self.assertRaisesRegex(ValueError, "temperature"):
            GridData.from_xyz(path, cutoff_grid=1)

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
            torch.allclose(data["beta_mu"], data["beta"] * data["mu"])
        )
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

    def test_frames_with_the_same_grid_share_neighborhood_geometry(self):
        data = GridData.from_xyz([self.path, self.path], cutoff_grid=1)

        self.assertIs(
            data[0]["local_density_index"],
            data[1]["local_density_index"],
        )
        self.assertIs(
            data[0]["local_density_positions"],
            data[1]["local_density_positions"],
        )


if __name__ == "__main__":
    unittest.main()
