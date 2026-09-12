"""Physical voxel coordinates, loading, and electrode-frame consistency."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from metal_helpers import _run
from ase import Atoms
from ase.io import write
from torch.utils.data import DataLoader

from equicdft import GridData, GridSolver, read_metal_sites
from test_metal_sites_data import _atoms, _grid_values
from test_metalwall_sites import _data, _wall, _reference, _inverse_fixture


def _centers(data, origin):
    indices = np.indices(tuple(data["grid_size"].tolist())).reshape(3, -1).T
    return (torch.tensor(indices, dtype=torch.float64) * data["grid_spacing"].double()
            + torch.as_tensor(origin, dtype=torch.float64))


class TestGridCenter(unittest.TestCase):
    def _read(self, atoms, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "field.extxyz"
            write(path, atoms, format="extxyz")
            return GridData.from_xyz(path, cutoff_grid=0, boltzmann_constant=1., **kwargs)

    def test_xyz_centers_follow_voxel_sorting_aliases_and_one_based_indices(self):
        for one_based in (False, True):
            with self.subTest(one_based=one_based):
                atoms = _atoms()
                atoms.arrays["centers"] = atoms.positions + np.array([.125, .25, -.5])
                if one_based:
                    atoms.positions += 1
                    atoms.info["grid_indexing"] = "one_based"
                frame = self._read(atoms, data_key={"grid_center": "centers"})[0]
                torch.testing.assert_close(frame["grid_center"], _centers(frame, [.125, .25, -.5]))
                self.assertEqual(frame["grid_center"].dtype, torch.float64)
                self.assertEqual(frame["grid_positions"].dtype, torch.long)

    def test_from_dict_preserves_double_precision_and_legacy_missing_centers(self):
        values = _grid_values()
        origin = [0.1234567890123, -.25, .5]
        centers = np.indices((4, 2, 2)).reshape(3, -1).T + np.array(origin)
        frame = GridData.from_dict(dict(values, grid_center=centers.tolist()), cutoff_grid=0)
        self.assertEqual(frame["grid_center"][0, 0].item(), origin[0])
        self.assertEqual(frame["grid_center"].dtype, torch.float64)
        self.assertNotIn("grid_center", GridData.from_dict(values, cutoff_grid=0))

    def test_malformed_centers_rejected_before_loading_or_coarsening(self):
        valid = np.indices((4, 2, 2)).reshape(3, -1).T.astype(float) + .125
        distorted = valid.copy()
        distorted[1, 0] += .01
        rotated = valid[:, [2, 1, 0]]
        for centers in (valid[:-1], valid[:, :2], valid * np.nan,
                        np.ones_like(valid, dtype=bool), valid.astype(complex),
                        distorted, rotated):
            with self.subTest(shape=centers.shape):
                with self.assertRaisesRegex(ValueError, "grid_center"):
                    GridData.from_dict(dict(_grid_values(), grid_center=centers), cutoff_grid=0)
        atoms = _atoms()
        # An alternating distortion would average out during coarsening.
        atoms.arrays["grid_center"] = atoms.positions + .125
        atoms.arrays["grid_center"][:, 0] += .01 * (2 * atoms.positions[:, 2] - 1)
        for key in list(atoms.info):
            if key.startswith("metal_"):
                del atoms.info[key]
        with self.assertRaisesRegex(ValueError, "regular grid"):
            self._read(atoms, target_grid_spacing=2.)

    def test_coarsening_averages_physical_centers_with_voxel_fields(self):
        for field in (None, "density", "V_ext"):
            with self.subTest(single_field=field):
                atoms = _atoms()
                for key in list(atoms.info):
                    if key.startswith("metal_"):
                        del atoms.info[key]
                if field is not None:
                    del atoms.arrays["V_ext" if field == "density" else "density"]
                atoms.arrays["excluded_mask"] = np.zeros(len(atoms), dtype=bool)
                atoms.arrays["grid_center"] = atoms.positions + np.array([.125, .25, -.5])
                frame = self._read(atoms, target_grid_spacing=2.)[0]
                torch.testing.assert_close(frame["grid_center"], _centers(frame, [.625, .75, 0.]))
                self.assertEqual(frame["grid_center"].shape, (2, 3))

    def test_mixed_center_metadata_collates_using_zero_origin_for_missing(self):
        first, second = _atoms(), _atoms()
        first.arrays["grid_center"] = first.positions + .125
        frames = self._read([first, second])
        batch = next(iter(DataLoader(frames, batch_size=2)))
        torch.testing.assert_close(batch["grid_center"][0], _centers(frames[0], [.125]*3))
        torch.testing.assert_close(batch["grid_center"][1], _centers(frames[1], [0.]*3))

    def test_physical_frame_matches_manual_shift_and_independent_fourier_oracle(self):
        for biased in (False, True):
            for shape in ((3, 3, 3), (4, 2, 4)):
                with self.subTest(biased=biased, shape=shape):
                    legacy = _data(shape=shape)
                    if biased:
                        legacy["metal_external_field"] = torch.tensor([.12, -.05, .08], dtype=torch.float64)
                        legacy["metal_field_origin"] = torch.tensor([-.1, .2, .3], dtype=torch.float64)
                    origin = torch.tensor([.125, -.2375, .43], dtype=torch.float64)
                    physical = dict(legacy, grid_center=_centers(legacy, origin),
                                    metal_positions=legacy["metal_positions"] + origin)
                    if biased:
                        physical["metal_field_origin"] = legacy["metal_field_origin"] + origin
                    expected, actual = _run(_wall(), legacy), _run(_wall(), physical)
                    for key in expected:
                        if key != "metal_positions":
                            torch.testing.assert_close(actual[key], expected[key], atol=3e-11, rtol=3e-11)
                    reference = _reference(physical)
                    for key in reference.keys() - {"B", "S"}:
                        torch.testing.assert_close(actual[key], reference[key], atol=3e-11, rtol=3e-11)
                    torch.testing.assert_close(actual["metal_positions"], physical["metal_positions"])
                    derivatives = []
                    for data in (legacy, physical):
                        rho = data["rho"].clone().requires_grad_()
                        result = _run(_wall(), dict(data, rho=rho))
                        energy = result["electrode_coulomb_energy"] + result["metal_external_energy"]
                        gradient = torch.autograd.grad(energy, rho, create_graph=True)[0]
                        direction = torch.sin(torch.arange(rho.numel(), dtype=rho.dtype)).reshape_as(rho)
                        hv = torch.autograd.grad((gradient * direction).sum(), rho)[0]
                        derivatives.append((gradient, hv))
                    for actual_derivative, expected_derivative in zip(*derivatives):
                        torch.testing.assert_close(actual_derivative, expected_derivative, atol=3e-11, rtol=3e-11)

    def test_center_shift_changes_sampling_without_moving_electrodes(self):
        data = _data()
        wall = _wall()
        before = _run(wall, data)
        cache = wall._cache
        shifted = dict(data, grid_center=_centers(data, [.125, .2, .3]))
        after = _run(wall, shifted)
        self.assertIsNot(wall._cache, cache)
        self.assertGreater((after["metal_site_q"] - before["metal_site_q"]).abs().max().item(), 1e-6)
        torch.testing.assert_close(after["metal_site_q"], _reference(shifted)["metal_site_q"], atol=3e-11, rtol=3e-11)

    def test_physical_zero_field_origin_needs_no_manual_half_voxel_shift(self):
        physical = _data(shape=(4, 2, 4))
        origin = physical["grid_spacing"] / 2
        physical["grid_center"] = _centers(physical, origin)
        length_z = physical["grid_size"][2] * physical["grid_spacing"][2]
        field_z = -2. / (5.217262802666367 * length_z)
        physical["metal_external_field"] = torch.tensor([0., 0., field_z], dtype=torch.float64)
        # Default wrapping center is physical zero, not grid_center[0].
        expected = dict(physical, metal_positions=physical["metal_positions"] - origin,
                        metal_field_origin=-origin)
        del expected["grid_center"]
        actual, reference = _run(_wall(), physical), _run(_wall(), expected)
        for key in ("metal_site_q", "metal_potential", "electrode_coulomb_energy", "metal_external_energy"):
            torch.testing.assert_close(actual[key], reference[key], atol=3e-11, rtol=3e-11)

    def test_float32_density_keeps_double_precision_physical_centers(self):
        data = _data()
        data["grid_center"] = _centers(data, [.125, .2, -.3])
        expected = _run(_wall(), data)
        single = dict(data, rho=data["rho"].float(), grid_spacing=data["grid_spacing"].float())
        actual = _run(_wall().float(), single)
        torch.testing.assert_close(actual["metal_site_q"].double(), expected["metal_site_q"], atol=2e-6, rtol=2e-5)
        self.assertEqual(data["grid_center"].dtype, torch.float64)

    def test_shared_and_per_field_centers_match_single_evaluations(self):
        data = _data()
        centers = _centers(data, [.125, .2, -.3])
        for shared in (False, True):
            with self.subTest(shared=shared):
                batch_centers = centers if shared else torch.stack([centers, centers + .1]).reshape(1, 2, -1, 3)
                batch = dict(data, rho=data["rho"].expand(1, 2, -1, -1), grid_center=batch_centers)
                result = _run(_wall(), batch)
                for i in range(2):
                    single = dict(data, grid_center=centers if shared else batch_centers[0, i])
                    expected = _run(_wall(), single)
                    for key in expected:
                        torch.testing.assert_close(result[key][0, i], expected[key], atol=3e-11, rtol=3e-11)

    def test_direct_wall_rejects_bad_or_differentiable_centers(self):
        data = _data()
        valid = _centers(data, [.125]*3)
        distorted = valid.clone()
        distorted[1, 0] += .01
        for centers in (valid[:-1], valid[:, :2], valid.expand(2, -1, -1),
                        valid * float("nan"), distorted, valid.bool(), valid.to(torch.complex128),
                        valid.clone().requires_grad_()):
            with self.subTest(shape=centers.shape), self.assertRaisesRegex(ValueError, "grid_center"):
                _run(_wall(), dict(data, grid_center=centers))

    def test_model_solver_and_two_start_inverse_keep_physical_coordinates(self):
        model, legacy = _inverse_fixture()
        model.compute_local_mu = True
        legacy["mu"] = torch.zeros(2, dtype=torch.float64)
        origin = torch.tensor([.125, .25, -.5], dtype=torch.float64)
        physical = dict(legacy, grid_center=_centers(legacy, origin),
                        metal_positions=legacy["metal_positions"] + origin,
                        metal_field_origin=legacy["metal_field_origin"] + origin)
        solver = GridSolver(model)
        old, new = _run(solver.evaluate, legacy), _run(solver.evaluate, physical)
        for key in ("beta_F_exc", "c1", "metal_site_q", "euler_lagrange_residual"):
            torch.testing.assert_close(new[key], old[key], atol=3e-11, rtol=3e-11)
        torch.testing.assert_close(new["grid_center"], physical["grid_center"])
        torch.testing.assert_close(new["metal_positions"], physical["metal_positions"])
        target = physical.pop("rho")
        numbers = model.voxel_volume * target.sum(0)
        for initial in (None, target.flip(0) + .05):
            result = _run(solver.solve, physical, initial_rho=initial, particle_numbers=numbers,
                                  method="euler", max_iter=250, mixing=.2,
                                  tolerance_residual=1e-9, tolerance_change=1e-12)
            self.assertTrue(result["converged"])
            torch.testing.assert_close(result["rho"], target, atol=1e-9, rtol=0)
            torch.testing.assert_close(result["grid_center"], physical["grid_center"])
            torch.testing.assert_close(result["metal_positions"], physical["metal_positions"])

    def test_xyz_reader_keeps_physical_electrode_coordinates(self):
        atoms = _atoms()
        atoms.arrays["grid_center"] = atoms.positions + .125
        frame = self._read(atoms)[0]
        sites = np.array([[.25, .3, .4], [1.5, .7, 1.1]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metal.xyz"
            write(path, Atoms("XX", positions=sites), format="xyz")
            metal = read_metal_sites(
                path, site_groups=0, group_ids=[0],
                total_charge=[0.], charge_units="e",
            )
        torch.testing.assert_close(
            metal["metal_positions"], torch.tensor(sites, dtype=torch.float64),
        )
        physical = _run(_wall(sites=metal), dict(frame, rho=frame["rho"].double()))
        legacy = dict(frame, rho=frame["rho"].double())
        del legacy["grid_center"]
        shifted_metal = dict(metal, metal_positions=metal["metal_positions"] - .125)
        expected = _run(_wall(sites=shifted_metal), legacy)
        torch.testing.assert_close(physical["metal_site_q"], expected["metal_site_q"], atol=3e-11, rtol=3e-11)


if __name__ == "__main__":
    unittest.main()
