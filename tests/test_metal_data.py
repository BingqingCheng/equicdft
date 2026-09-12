import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io import write
from torch import nn
from torch.utils.data import DataLoader

from equicdft import GridData, GridSolver
from equicdft._metal_data import normalize_metal_metadata


def _metal_atoms(single_group=False):
    shape = (4, 2, 2)
    positions = np.indices(shape).reshape(3, -1).T
    mask = np.full(len(positions), -1, dtype=int)
    mask[[0, 1]] = [2, 2 if single_group else 9]
    excluded = np.zeros(len(positions), dtype=bool)
    excluded[2] = True
    excluded |= mask >= 0
    rho = np.full((len(positions), 2), 0.2)
    rho[(mask >= 0) | excluded] = 0.0
    order = np.random.default_rng(11).permutation(len(positions))
    atoms = Atoms("X" * len(positions), positions=positions[order],
                  cell=np.diag(shape), pbc=True)
    atoms.arrays.update({
        "density": rho[order], "V_ext": np.zeros_like(rho),
        "excluded_mask": excluded[order],
    })
    atoms.info.update({
        "grid_size": np.array(shape), "grid_spacing": 1.0, "T": 1.0,
        "metal_positions": positions[mask >= 0].astype(float),
        "metal_site_groups": mask[mask >= 0],
        "metal_group_ids": 2 if single_group else np.array([9, 2]),
        "metal_total_charge": 0.0 if single_group else np.array([-0.3, 0.3]),
        "metal_charge_units": "e",
    })
    return atoms, mask, excluded


class _IdealGas(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("zero", torch.tensor(0.0))

    def forward(self, data, compute_c1=True):
        rho = data["rho"]
        result = {"beta_F_exc": self.zero.to(rho) * rho.sum(dim=(-2, -1))}
        if compute_c1:
            result["c1"] = torch.zeros_like(rho)
        return result


class TestMetalData(unittest.TestCase):
    def _read(self, atoms, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metal.extxyz"
            write(path, atoms, format="extxyz")
            return GridData.from_xyz(path, cutoff_grid=0,
                                     boltzmann_constant=1.0, **kwargs)

    def test_extxyz_preserves_site_group_order_and_exclusions(self):
        atoms, mask, excluded = _metal_atoms()
        frame = self._read(atoms)[0]
        self.assertTrue(torch.equal(frame["metal_site_groups"], torch.tensor(mask[mask >= 0])))
        self.assertTrue(torch.equal(frame["excluded_mask"], torch.tensor(excluded)))
        self.assertEqual(frame["metal_group_ids"].tolist(), [9, 2])
        torch.testing.assert_close(frame["metal_total_charge"],
                                   torch.tensor([-0.3, 0.3]))
        self.assertEqual(frame["metal_charge_units"], "e")
        self.assertTrue(torch.all(frame["rho"][(mask >= 0) | excluded] == 0))
        self.assertTrue(torch.all(frame["c1_plus_beta_mu"][(mask >= 0) | excluded] == 0))

    def test_extxyz_field_metadata_and_zero_field_batch(self):
        atoms, mask, _ = _metal_atoms(single_group=True)
        no_field = atoms.copy()
        atoms.info["metal_external_field"] = np.array([0., 0., -0.2])
        atoms.info["metal_field_origin"] = np.array([0., 0., -0.5])
        frames = self._read([no_field, atoms])
        batch = next(iter(DataLoader(frames, batch_size=2)))
        torch.testing.assert_close(batch["metal_external_field"],
                                   torch.tensor([[0., 0., 0.], [0., 0., -0.2]]))
        torch.testing.assert_close(batch["metal_field_origin"][1], torch.tensor([0., 0., -0.5]))
        direct = GridData.from_dict({
            "grid_size": [4, 2, 2], "grid_spacing": 1., "temperature": 1.,
            "n_types": 2, "metal_positions": [[0.,0.,0.], [0.,0.,1.]],
            "metal_site_groups": [2, 2], "metal_group_ids": 2,
            "metal_total_charge": 0., "metal_charge_units": "e",
            "metal_external_field": [0., 0., -0.2],
        }, cutoff_grid=0)
        torch.testing.assert_close(direct["metal_external_field"], frames[1]["metal_external_field"])
        torch.testing.assert_close(direct["metal_field_origin"], torch.zeros(3))
        atoms.info["metal_external_field"] = 0.2
        with self.assertRaisesRegex(ValueError, "metal_external_field"):
            self._read(atoms)

    def test_single_group_scalar_info_and_from_dict(self):
        atoms, mask, _ = _metal_atoms(single_group=True)
        frame = self._read(atoms)[0]
        self.assertEqual(frame["metal_group_ids"].shape, (1,))
        self.assertEqual(frame["metal_total_charge"].shape, (1,))
        direct = GridData.from_dict({
            "grid_size": [4, 2, 2], "grid_spacing": 1.0,
            "temperature": 1.0, "n_types": 2,
            "metal_positions": [[0.,0.,0.], [0.,0.,1.]], "metal_site_groups": [2, 2],
            "metal_group_ids": 2, "metal_total_charge": 0.0,
            "metal_charge_units": "e",
        }, cutoff_grid=0)
        self.assertEqual(direct["metal_group_ids"].tolist(), [2])
        self.assertFalse(torch.any(direct["excluded_mask"]))

    def test_explicit_charge_metadata_is_required(self):
        for key in ("metal_group_ids", "metal_total_charge", "metal_charge_units"):
            with self.subTest(key=key):
                atoms, _, _ = _metal_atoms()
                del atoms.info[key]
                with self.assertRaisesRegex(ValueError, "requires|requires explicit|metal requires"):
                    self._read(atoms)

    def test_invalid_group_metadata_is_rejected(self):
        cases = [
            ({"metal_group_ids": [2, 2]}, "unique"),
            ({"metal_group_ids": [2, 8]}, "exactly cover"),
            ({"metal_group_ids": [-1, 9]}, "nonnegative"),
            ({"metal_group_ids": [2, 9, 12], "metal_total_charge": [0, 0, 0]}, "exactly cover"),
            ({"metal_total_charge": [0.0]}, "one value"),
            ({"metal_total_charge": [0.0, float("nan")]}, "finite"),
            ({"metal_charge_units": "C"}, "must be 'e'"),
            ({"metal_site_groups": [2.5, 9]}, "integers"),
            ({"metal_site_groups": [1.0e30, 9]}, "integers"),
        ]
        for changes, message in cases:
            with self.subTest(changes=changes):
                values = dict(metal_positions=[[0.,0.,0.],[1.,0.,0.]],
                              metal_site_groups=[2, 9], metal_group_ids=[9, 2],
                              metal_total_charge=[0.0, 0.0], metal_charge_units="e")
                values.update(changes)
                with self.assertRaisesRegex(ValueError, message):
                    normalize_metal_metadata(**values)

    def test_nonzero_fluid_density_on_exclusions_is_rejected(self):
        atoms, _, _ = _metal_atoms()
        atoms.arrays["density"][atoms.arrays["excluded_mask"]] = 0.01
        with self.assertRaisesRegex(ValueError, "rho must be zero"):
            self._read(atoms)

    def test_site_coordinate_coarsening_requires_explicit_target(self):
        atoms, mask, _ = _metal_atoms()
        with self.assertRaisesRegex(ValueError, "coarsening explicit metal sites"):
            self._read(atoms, target_grid_spacing=2.0)
        unchanged = self._read(atoms, target_grid_spacing=1.0)[0]
        self.assertTrue(torch.equal(unchanged["metal_site_groups"], torch.tensor(mask[mask >= 0])))

    def test_batching_stacks_metadata_and_rejects_different_group_counts(self):
        atoms, _, _ = _metal_atoms()
        frames = self._read([atoms, atoms.copy()])
        batch = next(iter(DataLoader(frames, batch_size=2)))
        self.assertEqual(batch["metal_positions"].shape, (2, 2, 3))
        self.assertEqual(batch["metal_total_charge"].shape, (2, 2))
        validated = normalize_metal_metadata(
            batch["metal_group_ids"], batch["metal_total_charge"], batch["metal_charge_units"],
            metal_positions=batch["metal_positions"], metal_site_groups=batch["metal_site_groups"],
        )
        self.assertEqual(validated["metal_group_ids"].shape, (2, 2))
        one_group, _, _ = _metal_atoms(single_group=True)
        with self.assertRaisesRegex(ValueError, "same metal group count"):
            self._read([atoms, one_group])

    def test_shared_metadata_broadcasts_without_reordering(self):
        metadata = normalize_metal_metadata(
            [9, 2], [-0.4, 0.4], "e", dtype=torch.float64, batch_shape=(2,),
            metal_positions=[[0.,0.,0.],[1.,0.,0.]], metal_site_groups=[2, 9],
        )
        self.assertEqual(metadata["metal_positions"].shape, (2, 2, 3))
        self.assertEqual(metadata["metal_group_ids"].tolist(), [[9, 2], [9, 2]])
        self.assertEqual(metadata["metal_total_charge"].dtype, torch.float64)
        self.assertEqual(metadata["metal_total_charge"][0, 0].item(), -0.4)

    def test_batched_solver_evaluation_accepts_shared_sites(self):
        atoms, metal, excluded = _metal_atoms()
        frames = self._read([atoms, atoms.copy()])
        batch = next(iter(DataLoader(frames, batch_size=2)))
        batch["metal_positions"] = frames[0]["metal_positions"]
        batch["metal_site_groups"] = frames[0]["metal_site_groups"]
        result = GridSolver(_IdealGas()).evaluate(batch)
        self.assertEqual(result["beta_F"].shape, (2,))
        self.assertTrue(torch.all(result["rho"][:, (metal >= 0) | excluded] == 0))

    def test_exclusions_cannot_cover_every_fluid_grid_point(self):
        with self.assertRaisesRegex(ValueError, "accessible"):
            GridData.from_dict({
                "grid_size": [2, 1, 1], "grid_spacing": 1.0,
                "temperature": 1.0, "n_types": 1,
                "metal_positions": [[0.,0.,0.]], "metal_site_groups": [3],
                "excluded_mask": [True, True],
                "metal_group_ids": 3, "metal_total_charge": 0.0,
                "metal_charge_units": "e",
            }, cutoff_grid=0)

    def test_solver_enforces_exclusion_without_rewriting_geometry(self):
        for method in ("euler", "minimize"):
            with self.subTest(method=method):
                atoms, metal, excluded = _metal_atoms()
                data = self._read(atoms)[0]
                target_N = torch.tensor([1.3, 2.6])
                result = GridSolver(_IdealGas()).solve(
                    data, method=method, particle_numbers=target_N,
                    max_iter=20, tolerance_residual=1.0e-6,
                )
                inaccessible = (metal >= 0) | excluded
                self.assertTrue(result["converged"])
                self.assertTrue(torch.all(result["rho"][inaccessible] == 0))
                self.assertTrue(torch.all(result["euler_lagrange_residual"][inaccessible] == 0))
                torch.testing.assert_close(result["rho"].sum(dim=0), target_N)
                self.assertTrue(torch.equal(result["excluded_mask"], data["excluded_mask"]))
                self.assertTrue(torch.equal(result["metal_positions"], data["metal_positions"]))
                data["rho"][metal >= 0] = 0.1
                with self.assertRaisesRegex(ValueError, "rho must be zero"):
                    GridSolver(_IdealGas()).evaluate(data)


if __name__ == "__main__":
    unittest.main()
